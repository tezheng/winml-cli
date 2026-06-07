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
                             types.AttentionKind.DSA):
            raise NotImplementedError(
                f"B5: STANDARD, MLA, and DSA only, got {spec.kind}"
            )
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
        if spec.mask_kind not in (types.MaskKind.CAUSAL, types.MaskKind.SWA):
            raise NotImplementedError(
                f"B0.5: CAUSAL and SWA masks only, got {spec.mask_kind}"
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
            self.k_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.k_bias, dtype=dtype)
            # v_proj is skipped when attention_k_eq_v=True (Gemma 4 12B+ global)
            if spec.attention_k_eq_v:
                self.v_proj = None
            else:
                self.v_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.v_bias, dtype=dtype)
        self.o_proj = nn.Linear(q_proj_out, hidden_size, bias=spec.o_bias, dtype=dtype)

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

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: _kvcache.ContiguousKVCache,
        start_pos: int,
    ) -> torch.Tensor:
        spec = self.spec
        if getattr(self, "_is_dsa", False):
            raise NotImplementedError("DSA forward implementation deferred")
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

        scale = self.effective_scale
        T = start_pos + S
        device = q.device
        if spec.mask_kind == types.MaskKind.SWA and spec.sliding_window is not None:
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
        # B3 (Gemma 2): attn_logit_softcap — when set, sdpa uses a manual
        # matmul→softcap→softmax path (see api/ops.py::sdpa). The softcap is
        # applied AFTER the matmul*scale and BEFORE the mask add, per HF
        # `modeling_gemma2.py:212-217`.
        attn_out = ops.sdpa(q, k_full, v_full,
                            attn_mask=attn_mask, scale=scale,
                            logit_softcap=spec.logit_softcap)

        attn_out = attn_out.transpose(1, 2).reshape(B, S, Hq * Dh)
        return self.o_proj(attn_out)

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
