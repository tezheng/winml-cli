"""Attention block consuming AttentionSpec.

M1 supports the Qwen3 shape: STANDARD (GQA) with split QKV, optional QK-norm
PRE_ROPE PER_HEAD_DH, RoPE SPLIT_HALF, contiguous KV cache, causal mask.

B0.5 (Gemma 4) extensions:
- attention_k_eq_v: skip v_proj allocation; alias V := K (Gemma 4 12B+ global).
- qk_norm_fixed_scale: when set, effective_scale becomes 1.0 (the QKNorm absorbs
  the 1/sqrt(Dh) factor into its learned weight at load time).
- mask_kind=SWA: sliding-window attention mask with sliding_window=W.
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from api import kvcache as _kvcache, norm, ops, rope as _rope, specs, types


class Attention(nn.Module):
    def __init__(
        self,
        spec: specs.AttentionSpec,
        hidden_size: int,
        max_seq: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if spec.kind not in (types.AttentionKind.STANDARD,
                             types.AttentionKind.MLA,
                             types.AttentionKind.DSA,
                             types.AttentionKind.CSA_HCA):
            raise NotImplementedError(
                f"v7 P2: STANDARD, MLA, DSA, and CSA_HCA only, got {spec.kind}"
            )
        if spec.kind == types.AttentionKind.CSA_HCA:
            # v7 P2: DeepSeek-V4 CSA + HCA composition — SHAPE-ONLY.
            # We allocate compressor projections sized off the head_dim and
            # indexer dims so the module composes correctly, but forward
            # raises NotImplementedError until the overlap-state cache,
            # two-series window scheme, and Lightning Indexer scorer are
            # ported. Source:
            #   - HCA compressor: modeling_deepseek_v4.py:362-444
            #   - CSA compressor + Indexer: modeling_deepseek_v4.py:587-749
            #   - Attention dispatch: modeling_deepseek_v4.py:751-869
            if spec.csa is None and spec.hca is None:
                raise ValueError(
                    "CSA_HCA kind requires at least one of "
                    "AttentionSpec.csa / AttentionSpec.hca"
                )
            self._init_csa_hca(spec, hidden_size, dtype)
            self._is_csa_hca = True
            return
        if spec.kind == types.AttentionKind.DSA:
            # B5: DSA shape-only — allocate the indexer module and the
            # full MLA backbone, but raise on forward.
            if spec.indexer is None:
                raise ValueError("DSA kind requires AttentionSpec.indexer")
            # Re-use the MLA init for the MLA backbone (DSA = MLA + indexer).
            if spec.qkv_layout != types.QKVLayout.MLA_LATENT:
                raise ValueError(
                    f"DSA kind requires QKVLayout.MLA_LATENT, got {spec.qkv_layout}"
                )
            self._init_mla(spec, hidden_size, max_seq, dtype)
            # Allocate indexer Q/K projections — shape-only.
            self.indexer_q_proj = nn.Linear(
                hidden_size, spec.n_q_heads * spec.indexer.indexer_dim,
                bias=False, dtype=dtype,
            )
            self.indexer_k_proj = nn.Linear(
                hidden_size, spec.indexer.indexer_dim,
                bias=False, dtype=dtype,
            )
            self._is_dsa = True
            return
        if spec.kind == types.AttentionKind.MLA:
            # B2b: MLA path — MiniCPM-3 / DeepSeek-V2/V3 family. The MLA branch
            # uses MLA_LATENT qkv_layout and ignores attention_k_eq_v / qk_norm
            # / v_norm (those are STANDARD-only features). Source:
            # modeling_minicpm.py:331-385 (MiniCPMAttention init), 427-526
            # (forward).
            if spec.qkv_layout != types.QKVLayout.MLA_LATENT:
                raise ValueError(
                    f"MLA kind requires QKVLayout.MLA_LATENT, got {spec.qkv_layout}"
                )
            # B5: q_lora_rank can be None for DeepSeek-V2-Lite (16B-A2.4B)
            # which uses a DIRECT q_proj (no LoRA). The kv_lora path is
            # always present. Source:
            # modeling_deepseek_v2.py:310-315 (if q_lora_rank is None: ...
            # self.q_proj = nn.Linear(hidden_size, num_heads * qk_head_dim));
            # deepseek-ai/DeepSeek-V2-Lite/config.json: q_lora_rank=null.
            for fname in ("kv_lora_rank", "qk_nope_head_dim",
                          "qk_rope_head_dim", "v_head_dim"):
                if getattr(spec, fname) is None:
                    raise ValueError(f"MLA spec missing {fname}")
            if spec.qk_nope_head_dim + spec.qk_rope_head_dim != spec.head_dim:
                raise ValueError(
                    f"MLA: spec.head_dim ({spec.head_dim}) must equal "
                    f"qk_nope_head_dim ({spec.qk_nope_head_dim}) + "
                    f"qk_rope_head_dim ({spec.qk_rope_head_dim})"
                )
            if spec.attention_k_eq_v:
                raise NotImplementedError("MLA + attention_k_eq_v unsupported")
            # NOTE: spec.qk_norm is OVERLOADED for MLA. For STANDARD attention
            # it would mean a Qwen3-style PRE-RoPE QK-norm; for MLA it carries
            # the rms_norm_eps used by q_a_layernorm / kv_a_layernorm. The MLA
            # branch reads ONLY .eps (and .weight_mode) from it; if None we
            # default to 1e-5. spec.v_norm is not used by MLA.
            if spec.v_norm is not None:
                raise NotImplementedError("MLA + v_norm unsupported")
            self._init_mla(spec, hidden_size, max_seq, dtype)
            return
        if spec.qkv_layout not in (types.QKVLayout.SPLIT, types.QKVLayout.FUSED):
            raise NotImplementedError(
                f"B2a: SPLIT and FUSED QKV only, got {spec.qkv_layout}"
            )
        if spec.mask_kind not in (
            types.MaskKind.CAUSAL, types.MaskKind.SWA,
            types.MaskKind.BLOCK_BIDIRECTIONAL, types.MaskKind.SINK,
        ):
            raise NotImplementedError(
                f"v5-phase2: CAUSAL, SWA, BLOCK_BIDIRECTIONAL, and SINK "
                f"masks only, got {spec.mask_kind}"
            )
        # v5-phase2 V1: ALiBi and RoPE are mutually exclusive (a model
        # uses one OR the other for positional encoding). MPT uses ALiBi
        # only; Falcon-7B uses RoPE only. Source: MPT config has no
        # rope_theta, FalconConfig.rotary = not self.alibi.
        if spec.alibi is not None and spec.rope is not None:
            raise ValueError(
                "AttentionSpec: alibi and rope are mutually exclusive"
            )
        self.spec = spec
        self.hidden_size = hidden_size

        q_proj_out = spec.n_q_heads * spec.head_dim
        kv_proj_out = spec.n_kv_heads * spec.head_dim
        # B2a: FUSED QKV (Phi-3 family) — one big projection of size
        #   (Hq + 2*Hk) * Dh, sliced at forward time.
        # Source: modeling_phi3.py:222 (`op_size = num_attention_heads * head_dim
        # + 2 * (num_key_value_heads * head_dim)`) and 224 (`qkv_proj = Linear(
        # hidden_size, op_size, bias=False)`). FUSED is incompatible with
        # `attention_k_eq_v` and biases must be uniform (Phi-3 uses no bias).
        # v5-phase2 V4: CLA (Cross-Layer Attention). When
        # `kv_source_layer_offset is not None`, this layer borrows K/V
        # from the layer at index (L + offset) — i.e. its k_proj/v_proj
        # weights are intentionally NOT built. The caller must pass
        # `kv_shared` to forward.
        self._cla_offset = spec.kv_source_layer_offset
        if self._cla_offset is not None and spec.qkv_layout == types.QKVLayout.FUSED:
            raise NotImplementedError(
                "v5-phase2: CLA + FUSED QKV unsupported (CLA uses split q_proj)"
            )
        if spec.qkv_layout == types.QKVLayout.FUSED:
            if spec.attention_k_eq_v:
                raise NotImplementedError(
                    "FUSED QKV with attention_k_eq_v is not supported"
                )
            if not (spec.q_bias == spec.k_bias == spec.v_bias):
                raise NotImplementedError(
                    "FUSED QKV requires q/k/v_bias to all match (Phi-3: all False)"
                )
            fused_out = q_proj_out + 2 * kv_proj_out
            self.qkv_proj = nn.Linear(hidden_size, fused_out, bias=spec.q_bias, dtype=dtype)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = nn.Linear(hidden_size, q_proj_out, bias=spec.q_bias, dtype=dtype)
            if self._cla_offset is not None:
                # CLA borrower: K/V come from the source layer.
                self.k_proj = None
                self.v_proj = None
            else:
                self.k_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.k_bias, dtype=dtype)
                # v_proj is skipped when attention_k_eq_v=True (Gemma 4 12B+ global)
                if spec.attention_k_eq_v:
                    self.v_proj = None
                else:
                    self.v_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.v_bias, dtype=dtype)
        self.o_proj = nn.Linear(q_proj_out, hidden_size, bias=spec.o_bias, dtype=dtype)

        # v5-phase2 V3: BitNet attention sub-norm — RMSNorm on attn
        # output BEFORE o_proj. Weight shape [hidden_size].
        if spec.attn_sub_norm is not None:
            if spec.attn_sub_norm.kind != types.NormKind.RMS:
                raise ValueError("attn_sub_norm: RMS kind only")
            self.attn_sub_norm = norm.RMSNorm(spec.attn_sub_norm, hidden_size, dtype=dtype)
        else:
            self.attn_sub_norm = None

        # v5-phase2 V1: ALiBi slopes buffer. Per MPT, this is built at
        # module init from n_heads and alibi_bias_max. We use a buffer
        # so it travels with the module on `.to(device)` and is excluded
        # from state_dict (not persistent — derived purely from config).
        if spec.alibi is not None:
            if spec.alibi.n_heads != spec.n_q_heads:
                raise ValueError(
                    f"AliBiSpec.n_heads ({spec.alibi.n_heads}) must equal "
                    f"AttentionSpec.n_q_heads ({spec.n_q_heads})"
                )
            if spec.alibi.slopes is not None:
                slopes = torch.tensor(spec.alibi.slopes, dtype=dtype)
            else:
                slopes = ops.build_alibi_slopes(
                    spec.alibi.n_heads,
                    alibi_bias_max=spec.alibi.alibi_bias_max,
                    dtype=dtype,
                )
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        else:
            self.alibi_slopes = None

        # v5-phase2 V5: trained sinks parameter. One learnable scalar per
        # head, allocated when `spec.n_sink_tokens` is set. HF GPT-OSS
        # uses n_sink_tokens=1 effectively (a single `sinks` parameter
        # of shape [n_heads]). Source:
        # `transformers/models/gpt_oss/modeling_gpt_oss.py:309`.
        if spec.n_sink_tokens is not None:
            if spec.n_sink_tokens != 1:
                raise NotImplementedError(
                    f"v5-phase2: only n_sink_tokens=1 lands (HF GPT-OSS form), "
                    f"got {spec.n_sink_tokens}"
                )
            self.sinks = nn.Parameter(torch.empty(spec.n_q_heads, dtype=dtype))
        else:
            self.sinks = None

        # effective_scale: when qk_norm_fixed_scale is set, the norm absorbs
        # 1/sqrt(Dh) and the effective scale is 1.0 (Gemma 4). Otherwise honor
        # spec.attn_scale or default to 1/sqrt(Dh).
        if spec.qk_norm_fixed_scale is not None:
            self.effective_scale = 1.0
        elif spec.attn_scale is not None:
            self.effective_scale = spec.attn_scale
        else:
            self.effective_scale = spec.head_dim ** -0.5

        if spec.qk_norm is not None:
            if spec.qk_norm_phase != types.QKNormPhase.PRE_ROPE:
                raise NotImplementedError("M1: PRE_ROPE QK-norm only")
            self.q_norm = norm.QKNorm(
                spec.qk_norm, head_dim=spec.head_dim,
                shape=spec.qk_norm_shape,
                n_heads=spec.n_q_heads if spec.qk_norm_shape == types.QKNormShape.FULL_HDH else None,
                dtype=dtype,
            )
            self.k_norm = norm.QKNorm(
                spec.qk_norm, head_dim=spec.head_dim,
                shape=spec.qk_norm_shape,
                n_heads=spec.n_kv_heads if spec.qk_norm_shape == types.QKNormShape.FULL_HDH else None,
                dtype=dtype,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # B0.6: Gemma 4 v_norm. Unit RMSNorm on V with `with_scale=False` —
        # no learnable parameter, no state-dict entry. We use a register_buffer
        # so it travels with the module on `.to(device)` and is excluded from
        # `parameters()`.
        if spec.v_norm is not None:
            if spec.v_norm.kind != types.NormKind.RMS:
                raise ValueError("v_norm: RMS kind only")
            self._v_norm_eps = spec.v_norm.eps
            self._v_norm_mode = ("one_plus_w"
                                 if spec.v_norm.weight_mode == types.NormWeightMode.ONE_PLUS_W
                                 else "standard_w")
            if spec.v_norm_with_scale:
                # learnable weight goes in state_dict
                self.v_norm_weight = nn.Parameter(
                    torch.ones(spec.head_dim, dtype=dtype)
                )
                self._v_norm_with_scale = True
            else:
                # frozen ones, no state_dict entry — buffer (non-persistent).
                self.register_buffer(
                    "v_norm_weight",
                    torch.ones(spec.head_dim, dtype=dtype),
                    persistent=False,
                )
                self._v_norm_with_scale = False
        else:
            self._v_norm_eps = None
            self._v_norm_mode = None
            self._v_norm_with_scale = None

        # B1: SmolLM3 NoPE per-layer — spec.rope may be None on layers where the
        # model disables RoPE (NoPE layers). The forward path then skips
        # rope_apply entirely; Q/K propagate to SDPA without position rotation.
        # Source: `transformers/models/smollm3/modeling_smollm3.py:211`
        # (`self.use_rope = config.no_rope_layers[layer_idx]`) and 233-235
        # (the `if self.use_rope:` branch around apply_rotary_pos_emb).
        if spec.rope is None:
            self.rope = None
        else:
            self.rope = _rope.RoPE(spec.rope, head_dim=spec.head_dim,
                                   max_seq=max_seq, dtype=dtype)

    def _init_mla(
        self,
        spec: specs.AttentionSpec,
        hidden_size: int,
        max_seq: int,
        dtype: torch.dtype,
    ) -> None:
        """MLA init — MiniCPM-3 / DeepSeek-V2 family.

        Layers (source: modeling_minicpm.py:360-384):
        - q_a_proj: Linear(hidden, q_lora_rank, bias=attention_bias)
        - q_a_layernorm: RMSNorm(q_lora_rank)
        - q_b_proj: Linear(q_lora_rank, n_heads * qk_head_dim, bias=False)
        - kv_a_proj_with_mqa: Linear(hidden, kv_lora_rank + qk_rope_head_dim,
            bias=attention_bias)
        - kv_a_layernorm: RMSNorm(kv_lora_rank)
        - kv_b_proj: Linear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim),
            bias=False)
        - o_proj: Linear(n_heads * v_head_dim, hidden, bias=attention_bias)
        - rotary_emb: built at qk_rope_head_dim (NOT a partial within a larger
            head). Source: modeling_minicpm.py:392-395 (dim=qk_rope_head_dim).

        Softmax scale: qk_head_dim ** -0.5 — source: modeling_minicpm.py:387.

        attention_bias is encoded via spec.q_bias (we use q_bias as the
        single attention_bias flag for MLA; k/v_bias are ignored for MLA
        per the HF code which has only one `config.attention_bias`).
        """
        self.spec = spec
        self.hidden_size = hidden_size
        # Use spec.q_bias as the single MLA attention_bias flag (both
        # q_a_proj and kv_a_proj_with_mqa and o_proj take it; q_b_proj and
        # kv_b_proj NEVER have bias per modeling_minicpm.py:364-378).
        attn_bias = spec.q_bias
        H = spec.n_q_heads
        qk_h = spec.head_dim                  # qk_nope + qk_rope
        v_h = spec.v_head_dim
        # Q path: LoRA (MiniCPM-3, V2-7B/full, V3) OR direct q_proj (V2-Lite).
        if spec.qk_norm is not None:
            qa_norm_spec = spec.qk_norm
        else:
            qa_norm_spec = specs.NormSpec(
                kind=types.NormKind.RMS, eps=1e-5,
                weight_mode=types.NormWeightMode.STANDARD_W,
            )
        if spec.q_lora_rank is None:
            # V2-Lite direct q_proj — modeling_deepseek_v2.py:310-311.
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
            self.q_proj = nn.Linear(hidden_size, H * qk_h,
                                    bias=False, dtype=dtype)
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(hidden_size, spec.q_lora_rank,
                                      bias=attn_bias, dtype=dtype)
            self.q_a_layernorm = norm.RMSNorm(qa_norm_spec, spec.q_lora_rank, dtype=dtype)
            self.q_b_proj = nn.Linear(spec.q_lora_rank, H * qk_h,
                                      bias=False, dtype=dtype)
        # KV LoRA path
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, spec.kv_lora_rank + spec.qk_rope_head_dim,
            bias=attn_bias, dtype=dtype,
        )
        self.kv_a_layernorm = norm.RMSNorm(qa_norm_spec, spec.kv_lora_rank, dtype=dtype)
        self.kv_b_proj = nn.Linear(
            spec.kv_lora_rank, H * (spec.qk_nope_head_dim + v_h),
            bias=False, dtype=dtype,
        )
        # Output
        self.o_proj = nn.Linear(H * v_h, hidden_size,
                                bias=attn_bias, dtype=dtype)
        # Effective scale: 1 / sqrt(qk_head_dim).
        # Source: modeling_minicpm.py:387 — softmax_scale = q_head_dim ** -0.5.
        self.effective_scale = spec.head_dim ** -0.5
        # RoPE — built at qk_rope_head_dim, NOT a partial within a larger head.
        # MiniCPM uses MiniCPMRotaryEmbedding(dim=qk_rope_head_dim) — so cos/sin
        # span the rope sub-head dim directly, and rotate_half pairs i ↔ i+rope/2
        # across that subspace. We therefore build an api.rope.RoPE with
        # head_dim=qk_rope_head_dim and partial_rotary_factor=1.0 (full rotation
        # over the rope slice). The slicing q[..., qk_nope:] is the caller's
        # job. Source: modeling_minicpm.py:391-420 (rotary_emb dim arg
        # is qk_rope_head_dim).
        if spec.rope is None:
            self.rope = None
        else:
            self.rope = _rope.RoPE(
                spec.rope, head_dim=spec.qk_rope_head_dim,
                max_seq=max_seq, dtype=dtype,
            )
        # Mark the rest of standard attention as unused. NOTE: q_proj may be
        # set above (V2-Lite direct path) — do not overwrite it here.
        self.qkv_proj = None
        self.k_proj = None
        self.v_proj = None
        self.q_norm = None
        self.k_norm = None
        self._v_norm_eps = None
        self._v_norm_mode = None
        self._v_norm_with_scale = None
        # Tag so forward dispatches.
        self._is_mla = True

    def _init_csa_hca(
        self,
        spec: specs.AttentionSpec,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        """v7 P2: CSA + HCA shape-only init.

        Allocates compressor projections at the dims V4 uses:
        - HCA compressor (modeling_deepseek_v4.py:384-392): kv_proj /
          gate_proj are Linear(hidden, head_dim) and a position_bias
          Parameter of shape [compress_rate, head_dim], plus a kv_norm
          RMSNorm.
        - CSA compressor (modeling_deepseek_v4.py:610-619): kv_proj /
          gate_proj are Linear(hidden, 2*head_dim) — the Ca/Cb two-series
          layout — and position_bias is [compress_rate, 2*head_dim]. Has an
          embedded Indexer (modeling_deepseek_v4.py:493-506) whose own
          kv_proj / gate_proj are Linear(hidden, 2*indexer_head_dim), with a
          weights_proj for the scoring head and a q_b_proj from the MLA
          q_lora_rank to the indexer Q.

        Because V4 is built on the MLA backbone (q_lora_rank, kv_lora_rank,
        rope+nope split), the OUTER attention init mirrors the MLA path. For
        the v7 IR shape-only landing we DO NOT build the MLA backbone here —
        the spec is intended to compose with an outer MLA. We only build the
        compressor / indexer projections so the module attribute tree is
        present. Source: modeling_deepseek_v4.py:751-869 (DeepseekV4Attention).
        """
        self.spec = spec
        self.hidden_size = hidden_size
        Dh = spec.head_dim
        if spec.hca is not None:
            self.hca_kv_proj = nn.Linear(hidden_size, Dh, bias=False, dtype=dtype)
            self.hca_gate_proj = nn.Linear(hidden_size, Dh, bias=False, dtype=dtype)
            self.hca_position_bias = nn.Parameter(
                torch.empty(spec.hca.compress_rate, Dh, dtype=dtype),
            )
            # Compressor RMSNorm — eps matches the rms_norm_eps of any
            # outer qk_norm; if not provided default to 1e-5 per V4 config.
            eps = spec.qk_norm.eps if spec.qk_norm is not None else 1e-5
            from api import norm as _norm
            self.hca_kv_norm = _norm.RMSNorm(
                specs.NormSpec(
                    kind=types.NormKind.RMS, eps=eps,
                    weight_mode=types.NormWeightMode.STANDARD_W,
                ),
                Dh, dtype=dtype,
            )
        else:
            self.hca_kv_proj = None
            self.hca_gate_proj = None
            self.hca_position_bias = None
            self.hca_kv_norm = None
        if spec.csa is not None:
            csa = spec.csa
            self.csa_kv_proj = nn.Linear(hidden_size, 2 * Dh, bias=False, dtype=dtype)
            self.csa_gate_proj = nn.Linear(hidden_size, 2 * Dh, bias=False, dtype=dtype)
            self.csa_position_bias = nn.Parameter(
                torch.empty(csa.compress_rate, 2 * Dh, dtype=dtype),
            )
            eps = spec.qk_norm.eps if spec.qk_norm is not None else 1e-5
            from api import norm as _norm
            self.csa_kv_norm = _norm.RMSNorm(
                specs.NormSpec(
                    kind=types.NormKind.RMS, eps=eps,
                    weight_mode=types.NormWeightMode.STANDARD_W,
                ),
                Dh, dtype=dtype,
            )
            # Embedded Lightning Indexer.
            i_dim = csa.indexer_head_dim
            i_heads = csa.indexer_n_heads
            self.indexer_kv_proj = nn.Linear(hidden_size, 2 * i_dim,
                                             bias=False, dtype=dtype)
            self.indexer_gate_proj = nn.Linear(hidden_size, 2 * i_dim,
                                               bias=False, dtype=dtype)
            self.indexer_position_bias = nn.Parameter(
                torch.empty(csa.compress_rate, 2 * i_dim, dtype=dtype),
            )
            self.indexer_kv_norm = _norm.RMSNorm(
                specs.NormSpec(
                    kind=types.NormKind.RMS, eps=eps,
                    weight_mode=types.NormWeightMode.STANDARD_W,
                ),
                i_dim, dtype=dtype,
            )
            self.indexer_weights_proj = nn.Linear(hidden_size, i_heads,
                                                  bias=False, dtype=dtype)
        else:
            self.csa_kv_proj = None
            self.csa_gate_proj = None
            self.csa_position_bias = None
            self.csa_kv_norm = None
            self.indexer_kv_proj = None
            self.indexer_gate_proj = None
            self.indexer_position_bias = None
            self.indexer_kv_norm = None
            self.indexer_weights_proj = None
        # The MLA / STANDARD backbone is NOT built here (shape-only).
        # Mark unused attrs.
        self.qkv_proj = None
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.rope = None
        self.q_norm = None
        self.k_norm = None
        self._v_norm_eps = None
        self._v_norm_mode = None
        self._v_norm_with_scale = None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: _kvcache.ContiguousKVCache,
        start_pos: int,
        vision_token_count: Optional[int] = None,
        kv_shared: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        spec = self.spec
        if getattr(self, "_is_dsa", False):
            raise NotImplementedError("DSA forward implementation deferred")
        if getattr(self, "_is_csa_hca", False):
            # v7 P2: CSA+HCA shape-only landing. The compressor projections
            # are allocated; the forward (two-series overlap window scheme +
            # Lightning Indexer top-k gather + concat-onto-KV-axis) is
            # deferred. Source: modeling_deepseek_v4.py:797-869.
            raise NotImplementedError(
                "CSA+HCA forward implementation deferred"
            )
        if getattr(self, "_is_mla", False):
            return self._forward_mla(x, position_ids, cache, start_pos)
        B, S, _ = x.shape
        Hq, Hk, Dh = spec.n_q_heads, spec.n_kv_heads, spec.head_dim

        if spec.qkv_layout == types.QKVLayout.FUSED:
            # B2a: Phi-3 fused QKV slicing — Q first, then K, then V along the
            # output dim. Source: modeling_phi3.py:237-241.
            qkv = self.qkv_proj(x)                              # [B, S, (Hq+2*Hk)*Dh]
            q_end = Hq * Dh
            k_end = q_end + Hk * Dh
            q = qkv[..., :q_end].view(B, S, Hq, Dh)
            k = qkv[..., q_end:k_end].view(B, S, Hk, Dh)
            v = qkv[..., k_end:].view(B, S, Hk, Dh)
        else:
            q = self.q_proj(x).view(B, S, Hq, Dh)
            if self._cla_offset is not None:
                # v5-phase2 V4: CLA — K/V come from a preceding layer,
                # passed by the caller in `kv_shared` as the already-
                # transposed [B, Hk, S, Dh] tensors that this layer would
                # otherwise have written into its own cache.
                if kv_shared is None:
                    raise ValueError(
                        "CLA layer (kv_source_layer_offset set) requires "
                        "kv_shared=(k,v) at forward"
                    )
                # k, v are already [B, Hk, S, Dh] — skip the projection +
                # transpose path below by short-circuiting here.
                k_cla, v_cla = kv_shared
                # We still need q in the standard [B, Hq, S, Dh] layout.
                q = q.transpose(1, 2)
                # Apply RoPE to Q only — K was already RoPE'd by the
                # source layer when it computed its own K. For ALiBi the
                # rope=None path is taken below.
                if self.rope is not None:
                    # rope.forward expects (q_4d, k_4d) — give it a dummy k.
                    # But we need q rotated by position_ids. Build a fake k
                    # of matching shape (Hk=Hq for q-only rotation works
                    # since rope acts on the last two dims).
                    # Simpler: re-implement just q rotation here using the
                    # rope module's exposed cos/sin if present. Punt: in
                    # practice CLA models share K/V with RoPE already
                    # baked-in, so q must also be rotated. We rotate via
                    # a fake k=q approach.
                    q_4d = q.transpose(1, 2)  # back to [B, S, Hq, Dh]
                    q_rot, _ = self.rope(q_4d, q_4d, position_ids)
                    q = q_rot.transpose(1, 2)  # [B, Hq, S, Dh]
                attn_out = self._sdpa_path(q, k_cla, v_cla, start_pos, S,
                                           vision_token_count)
                # Sub-norm + o_proj
                if self.attn_sub_norm is not None:
                    attn_out = self.attn_sub_norm(attn_out)
                return self.o_proj(attn_out)
            k = self.k_proj(x).view(B, S, Hk, Dh)
            # K = V branch: alias the K tensor as V (after the same projection)
            if spec.attention_k_eq_v:
                v = k
            else:
                v = self.v_proj(x).view(B, S, Hk, Dh)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # B1: NoPE branch — SmolLM3 disables RoPE on a periodic subset of layers.
        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)

        # B0.6: v_norm applied to V before transpose+cache.write (per HF
        # `modeling_gemma4.py:1265`). When `attention_k_eq_v=True` (v aliased
        # to raw K projection), this is what HF does too — k_norm and v_norm
        # are applied to the SAME tensor identity (raw K projection output)
        # in separate paths.
        if self._v_norm_mode is not None:
            v = ops.rms_norm(v, self.v_norm_weight, self._v_norm_eps,
                             mode=self._v_norm_mode)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cache.write(k, v, start_pos=start_pos)
        k_full, v_full = cache.read(seq_len=start_pos + S)

        # v5-phase2: route through unified SDPA helper. It handles mask
        # construction (CAUSAL / SWA / BLOCK_BIDIRECTIONAL / SINK), ALiBi
        # bias build, and trained-sink logits.
        attn_out = self._sdpa_path(q, k_full, v_full, start_pos, S,
                                   vision_token_count)
        # v5-phase2 V3: BitNet attn_sub_norm — RMSNorm on the merged
        # (B, S, Hq*Dh) representation BEFORE o_proj. Source:
        # modeling_bitnet.py:215-217.
        if self.attn_sub_norm is not None:
            attn_out = self.attn_sub_norm(attn_out)
        return self.o_proj(attn_out)

    def _sdpa_path(
        self,
        q: torch.Tensor,       # [B, Hq, S_q, Dh]
        k_full: torch.Tensor,  # [B, Hk, T, Dh]
        v_full: torch.Tensor,  # [B, Hk, T, Dh]
        start_pos: int,
        S: int,
        vision_token_count: Optional[int],
    ) -> torch.Tensor:
        """Mask construction + SDPA + reshape, shared between the standard
        and CLA paths.

        Returns [B, S, Hq * Dh] (post-transpose, pre-o_proj).
        """
        spec = self.spec
        B = q.shape[0]
        Hq, Dh = spec.n_q_heads, spec.head_dim
        scale = self.effective_scale
        T = start_pos + S
        device = q.device
        if spec.block_bidirectional_mask and vision_token_count is not None:
            # B8: Visual Causal Flow mask. Reserve `V = vision_token_count`
            # tokens at the prefix as bidirectional; everything else is
            # standard causal. Source: original DeepSeek-OCR paper §3.2.
            # Mask is built over (S_q, T) where T = start_pos + S; the
            # bidirectional block sits at columns [0, V).
            V = vision_token_count
            if V < 0 or V > T:
                raise ValueError(f"vision_token_count {V} out of [0, {T}]")
            i_idx = torch.arange(S, device=device).unsqueeze(1)
            j_idx = torch.arange(T, device=device).unsqueeze(0)
            # vision keys (j < V) seen by every query; text keys (j >= V)
            # only by text queries via causal rule.
            q_pos = start_pos + i_idx
            is_vision_q = q_pos < V
            is_vision_k = j_idx < V
            keep_text_text = (~is_vision_k) & (~is_vision_q) & (j_idx <= q_pos)
            keep_vis_vis = is_vision_k & is_vision_q
            # Text-to-vision (text query attends backward to vision keys):
            keep_text_vis = is_vision_k & (~is_vision_q)
            keep = keep_vis_vis | keep_text_text | keep_text_vis
            attn_mask = torch.zeros(S, T, dtype=q.dtype, device=device)
            attn_mask = attn_mask.masked_fill(
                ~keep, float("-inf"),
            )
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif spec.mask_kind == types.MaskKind.SWA and spec.sliding_window is not None:
            # SWA mask: (i, j) is kept iff j <= start_pos+i (causal) AND
            # (start_pos+i) - j <= W (within window).
            W = spec.sliding_window
            q_pos = torch.arange(start_pos, start_pos + S, device=device).view(S, 1)
            k_pos = torch.arange(T, device=device).view(1, T)
            keep = (k_pos <= q_pos) & (q_pos - k_pos <= W)
            attn_mask = torch.where(keep,
                                    torch.zeros((), dtype=q.dtype, device=device),
                                    torch.full((), float("-inf"), dtype=q.dtype, device=device))
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        else:
            # Build an explicit additive causal mask. Query token i has absolute
            # position (start_pos + i) and may attend to key positions j <= start_pos + i.
            # `is_causal=True` on SDPA would mis-align here when start_pos > 0 because
            # it assumes the diagonal at the top-left of [S_q × S_k].
            i_idx = torch.arange(S, device=device).unsqueeze(1)        # [S, 1]
            j_idx = torch.arange(T, device=device).unsqueeze(0)        # [1, T]
            allowed = j_idx <= (start_pos + i_idx)                     # [S, T]
            attn_mask = torch.zeros(S, T, dtype=q.dtype, device=device)
            attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)            # [1, 1, S, T]
        # v5-phase2 V1: ALiBi position bias — build [1, Hq, S, T] additive
        # bias and pass through sdpa(attn_bias=...). Source:
        # modeling_mpt.py:121 (`attention_scores + position_bias`).
        attn_bias = None
        if self.alibi_slopes is not None:
            # Build the (S, T) offset matrix where offset[i, j] = (j -
            # (start_pos + i)) and broadcast across heads via slopes.
            device2 = q.device
            q_abs = torch.arange(start_pos, start_pos + S,
                                 device=device2, dtype=torch.long).view(S, 1)
            k_abs = torch.arange(T, device=device2, dtype=torch.long).view(1, T)
            offset = (k_abs - q_abs).to(q.dtype)              # [S, T]
            slopes_h = self.alibi_slopes.to(q.dtype).view(1, Hq, 1, 1)
            attn_bias = slopes_h * offset.view(1, 1, S, T)    # [1, Hq, S, T]

        # v5-phase2 V5: trained sinks — pass the per-head learnable
        # scalar to sdpa, which appends the sink logit column and drops
        # it after softmax. Source: modeling_gpt_oss.py:267-275.
        sinks = self.sinks if self.sinks is not None else None

        # B3 (Gemma 2): attn_logit_softcap — when set, sdpa uses a manual
        # matmul→softcap→softmax path (see api/ops.py::sdpa). The softcap is
        # applied AFTER the matmul*scale and BEFORE the mask add, per HF
        # `modeling_gemma2.py:212-217`.
        attn_out = ops.sdpa(q, k_full, v_full,
                            attn_mask=attn_mask, scale=scale,
                            logit_softcap=spec.logit_softcap,
                            attn_bias=attn_bias, sinks=sinks)

        attn_out = attn_out.transpose(1, 2).reshape(B, S, Hq * Dh)
        return attn_out

    def _forward_mla(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: _kvcache.ContiguousKVCache,
        start_pos: int,
    ) -> torch.Tensor:
        """MLA forward — MiniCPM-3 reference path.

        Mirrors modeling_minicpm.py:427-526. Stores DECOMPRESSED K (at
        qk_head_dim) and V (at v_head_dim) in the KV cache; production
        deployments absorbing kv_b_proj into o_proj are deferred.
        """
        spec = self.spec
        B, S, _ = x.shape
        H = spec.n_q_heads
        qk_nope = spec.qk_nope_head_dim
        qk_rope = spec.qk_rope_head_dim
        qk_h = spec.head_dim          # qk_nope + qk_rope
        v_h = spec.v_head_dim
        # Q: LoRA path (q_a_proj → q_a_layernorm → q_b_proj) when q_lora_rank
        # is set; direct q_proj otherwise (V2-Lite).
        # Source: modeling_minicpm.py:444-448 (LoRA) /
        #         modeling_deepseek_v2.py:349-352 (LoRA OR direct).
        if self.q_proj is not None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        # Reshape to [B, S, H, qk_h] then split nope/rope on last dim.
        q = q.view(B, S, H, qk_h)
        q_nope = q[..., :qk_nope]
        q_pe = q[..., qk_nope:]                  # [B, S, H, qk_rope]
        # KV LoRA path: x -> [B, S, kv_lora_rank + qk_rope].
        # split: [..., :kv_lora_rank] -> norm -> kv_b_proj -> [B, S, H*(qk_nope+v_h)]
        # Source: modeling_minicpm.py:450-463.
        compressed = self.kv_a_proj_with_mqa(x)
        c_kv, k_pe_raw = torch.split(
            compressed, [spec.kv_lora_rank, qk_rope], dim=-1,
        )
        # k_pe_raw is [B, S, qk_rope] — a SINGLE rope sub-head shared across all H.
        kv = self.kv_b_proj(self.kv_a_layernorm(c_kv))
        kv = kv.view(B, S, H, qk_nope + v_h)
        k_nope, v = torch.split(kv, [qk_nope, v_h], dim=-1)
        # RoPE: rotate q_pe and k_pe_raw. q_pe has shape [B, S, H, qk_rope].
        # k_pe_raw has shape [B, S, qk_rope]; we treat it as [B, S, 1, qk_rope]
        # for the rope op (single-head), then broadcast to all H after rotation.
        if self.rope is not None:
            k_pe = k_pe_raw.view(B, S, 1, qk_rope)
            q_pe_rot, k_pe_rot = self.rope(q_pe, k_pe, position_ids)
        else:
            q_pe_rot = q_pe
            k_pe_rot = k_pe_raw.view(B, S, 1, qk_rope)
        # Assemble q and k at qk_h. q stays per-head, k_pe broadcast to H heads.
        # Source: modeling_minicpm.py:477-483.
        q_full = torch.cat([q_nope, q_pe_rot], dim=-1)            # [B, S, H, qk_h]
        k_pe_b = k_pe_rot.expand(B, S, H, qk_rope)
        k_full = torch.cat([k_nope, k_pe_b], dim=-1)              # [B, S, H, qk_h]
        # Transpose to [B, H, S, *] for SDPA + cache.
        q_full = q_full.transpose(1, 2)                            # [B, H, S, qk_h]
        k_full = k_full.transpose(1, 2)
        v = v.transpose(1, 2)                                      # [B, H, S, v_h]
        # Write decompressed K (at qk_h) and V (at v_h) into the cache.
        cache.write(k_full, v, start_pos=start_pos)
        k_cached, v_cached = cache.read(seq_len=start_pos + S)
        # Causal mask. MLA uses standard causal mask (modeling_minicpm.py:485-505).
        T = start_pos + S
        device = q_full.device
        i_idx = torch.arange(S, device=device).unsqueeze(1)        # [S, 1]
        j_idx = torch.arange(T, device=device).unsqueeze(0)        # [1, T]
        allowed = j_idx <= (start_pos + i_idx)                     # [S, T]
        attn_mask = torch.zeros(S, T, dtype=q_full.dtype, device=device)
        attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)            # [1, 1, S, T]
        # SDPA: q and k at qk_h, v at v_h. n_heads matches between q/k/v so no GQA.
        attn_out = ops.sdpa(q_full, k_cached, v_cached,
                            attn_mask=attn_mask, scale=self.effective_scale)
        # attn_out: [B, H, S, v_h] → [B, S, H*v_h] → o_proj.
        attn_out = attn_out.transpose(1, 2).reshape(B, S, H * v_h)
        return self.o_proj(attn_out)
