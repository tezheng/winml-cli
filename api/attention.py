"""Attention block consuming AttentionSpec.

M1 supports the Qwen3 shape: STANDARD (GQA) with split QKV, optional QK-norm
PRE_ROPE PER_HEAD_DH, RoPE SPLIT_HALF, contiguous KV cache, causal mask.
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
        if spec.qkv_layout != types.QKVLayout.SPLIT:
            raise NotImplementedError(f"M1: SPLIT QKV only, got {spec.qkv_layout}")
        if spec.mask_kind != types.MaskKind.CAUSAL:
            raise NotImplementedError(f"M1: CAUSAL mask only, got {spec.mask_kind}")
        self.spec = spec
        self.hidden_size = hidden_size

        q_proj_out = spec.n_q_heads * spec.head_dim
        kv_proj_out = spec.n_kv_heads * spec.head_dim
        self.q_proj = nn.Linear(hidden_size, q_proj_out, bias=spec.q_bias, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.k_bias, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.v_bias, dtype=dtype)
        self.o_proj = nn.Linear(q_proj_out, hidden_size, bias=spec.o_bias, dtype=dtype)

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

        if spec.rope is None:
            raise NotImplementedError("M1: RoPE required")
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

        q = self.q_proj(x).view(B, S, Hq, Dh)
        k = self.k_proj(x).view(B, S, Hk, Dh)
        v = self.v_proj(x).view(B, S, Hk, Dh)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.rope(q, k, position_ids)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cache.write(k, v, start_pos=start_pos)
        k_full, v_full = cache.read(seq_len=start_pos + S)

        scale = spec.attn_scale if spec.attn_scale is not None else (Dh ** -0.5)
        # Build an explicit additive causal mask. Query token i has absolute
        # position (start_pos + i) and may attend to key positions j <= start_pos + i.
        # `is_causal=True` on SDPA would mis-align here when start_pos > 0 because
        # it assumes the diagonal at the top-left of [S_q × S_k].
        T = start_pos + S
        device = q.device
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
