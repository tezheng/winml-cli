"""GQA attention forward (no sinks, no v_norm, no sliding window) in PyTorch.

Source: transformers/models/llama/modeling_llama.py:LlamaAttention.forward
plus the standard GQA broadcast pattern (repeat_interleave on K/V).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from components import GQASpec, RoPESpec

from .rope import apply_rotary_pos_emb_single, build_rope_cos_sin


def gqa_forward(
    spec: GQASpec,
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    *,
    rope_spec: Optional[RoPESpec] = None,
    position_ids: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full GQA attention unit: Q/K/V projection, optional RoPE, SDPA, O projection.

    M0 only supports the plain GQA path — none of the optional fields
    (``sliding_window``, ``output_gate``, ``v_norm``, ``qk_norm``,
    ``qk_norm_fixed_scale``, ``kv_source_layer_offset``, ``logit_softcap``) are
    implemented; this asserts they're at default values.

    Args:
        spec:         GQASpec with ``num_heads``, ``num_kv_heads``, ``head_dim``.
        x:            Hidden state shaped ``[B, S, D_model]`` where
                      ``D_model = num_heads * head_dim``.
        w_q:          Q projection weight, shape ``[H_q * Dh, D_model]``.
        w_k:          K projection weight, shape ``[H_kv * Dh, D_model]``.
        w_v:          V projection weight, shape ``[H_kv * Dh, D_model]``.
        w_o:          Output projection weight, shape ``[D_model, H_q * Dh]``.
        rope_spec:    If provided, RoPE is applied to Q and K before SDPA.
        position_ids: Required if ``rope_spec`` is given; shape ``[B, S]``.
        attn_mask:    Optional additive attention mask passed to SDPA.

    Returns:
        Attention output shaped ``[B, S, D_model]``.
    """
    # ---- M0 unsupported-feature guards --------------------------------------
    if spec.sliding_window is not None:
        raise NotImplementedError("GQA sliding_window not supported in M0.")
    if spec.output_gate is not None:
        raise NotImplementedError("GQA output_gate not supported in M0.")
    if spec.v_norm is not None:
        raise NotImplementedError("GQA v_norm not supported in M0.")
    if spec.qk_norm is not None:
        raise NotImplementedError("GQA qk_norm not supported in M0.")
    if spec.qk_norm_fixed_scale is not None:
        raise NotImplementedError("GQA qk_norm_fixed_scale not supported in M0.")
    if spec.kv_source_layer_offset != 0:
        raise NotImplementedError("GQA kv_source_layer_offset not supported in M0.")
    if spec.logit_softcap != 0.0:
        raise NotImplementedError("GQA logit_softcap not supported in M0.")

    B, S, _D = x.shape
    H_q, H_kv, Dh = spec.num_heads, spec.num_kv_heads, spec.head_dim
    D_model = H_q * Dh

    # ---- Q/K/V projections + head reshape -----------------------------------
    # F.linear expects weight shape [out_features, in_features]; matches the
    # spec's "[H*Dh, D_model]" weight layout.
    q = F.linear(x, w_q).view(B, S, H_q, Dh).transpose(1, 2)   # [B, H_q, S, Dh]
    k = F.linear(x, w_k).view(B, S, H_kv, Dh).transpose(1, 2)  # [B, H_kv, S, Dh]
    v = F.linear(x, w_v).view(B, S, H_kv, Dh).transpose(1, 2)  # [B, H_kv, S, Dh]

    # ---- RoPE on Q and K ----------------------------------------------------
    if rope_spec is not None:
        if position_ids is None:
            raise ValueError("position_ids is required when rope_spec is provided.")
        cos, sin = build_rope_cos_sin(rope_spec, position_ids, Dh, dtype=q.dtype)
        q = apply_rotary_pos_emb_single(q, cos, sin)
        k = apply_rotary_pos_emb_single(k, cos, sin)

    # ---- GQA broadcast on K/V -----------------------------------------------
    if H_kv < H_q:
        if H_q % H_kv != 0:
            raise ValueError(f"num_heads ({H_q}) must be divisible by num_kv_heads ({H_kv}).")
        repeat = H_q // H_kv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # ---- SDPA ---------------------------------------------------------------
    is_causal = spec.causal and attn_mask is None
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=spec.scale,
    )  # [B, H_q, S, Dh]

    # ---- Concat heads + O projection ----------------------------------------
    out = out.transpose(1, 2).contiguous().view(B, S, D_model)
    return F.linear(out, w_o)


__all__ = ["gqa_forward"]
