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
        if spec.kind != types.AttentionKind.STANDARD:
            raise NotImplementedError(f"M1: STANDARD only, got {spec.kind}")
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

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: _kvcache.ContiguousKVCache,
        start_pos: int,
    ) -> torch.Tensor:
        spec = self.spec
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
        attn_out = ops.sdpa(q, k_full, v_full,
                            attn_mask=attn_mask, scale=scale)

        attn_out = attn_out.transpose(1, 2).reshape(B, S, Hq * Dh)
        return self.o_proj(attn_out)
