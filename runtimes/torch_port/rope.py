"""Rotary positional embedding forward in PyTorch.

Source: transformers/models/llama/modeling_llama.py: rotate_half,
apply_rotary_pos_emb, LlamaRotaryEmbedding (the cos/sin table construction).

Only ``partial_rotary_factor == 1.0`` is implemented for M0 (full rotary).
"""
from __future__ import annotations

import torch

from components import RoPESpec


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Llama-style half-rotation: split last dim in two halves, swap & negate.

    Source: modeling_llama.py:rotate_half. Returns ``concat(-x[..., d/2:], x[..., :d/2])``.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cos_sin(
    spec: RoPESpec,
    position_ids: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin lookup tables for the given positions.

    Args:
        spec:         RoPESpec carrying ``theta``.
        position_ids: Integer tensor of shape ``[B, S]`` with absolute positions.
        head_dim:     Per-head dim ``Dh``; must be even.

    Returns:
        ``(cos, sin)`` each shaped ``[B, S, Dh]``, ready to broadcast against a
        ``[B, H, S, Dh]`` Q/K layout.
    """
    if spec.partial_rotary_factor != 1.0:
        raise NotImplementedError(
            "M0 only supports full rotary (partial_rotary_factor == 1.0); "
            f"got {spec.partial_rotary_factor}."
        )
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim; got {head_dim}.")

    device = position_ids.device
    # inv_freq: [Dh/2]
    inv_freq = 1.0 / (
        spec.theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    # position_ids: [B, S] -> [B, S, 1] * [1, 1, Dh/2] -> [B, S, Dh/2]
    pos = position_ids.to(torch.float32).unsqueeze(-1)
    freqs = pos * inv_freq.unsqueeze(0).unsqueeze(0)
    # Match Llama: emb = concat(freqs, freqs) along the last dim.
    emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, Dh]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a single Q or K tensor.

    Args:
        x:    Tensor shaped ``[B, H, S, Dh]``.
        cos:  Cos table shaped ``[B, S, Dh]`` (broadcasts over H).
        sin:  Sin table shaped ``[B, S, Dh]``.

    Returns:
        Rotated tensor of the same shape and dtype as ``x``.
    """
    # cos/sin: [B, S, Dh] -> [B, 1, S, Dh] to broadcast across heads.
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    return (x * cos_b) + (_rotate_half(x) * sin_b)


def rope_forward(
    spec: RoPESpec,
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        spec:         RoPESpec.
        q:            Query tensor shaped ``[B, H_q, S, Dh]``.
        k:            Key tensor shaped ``[B, H_kv, S, Dh]``.
        position_ids: Position indices shaped ``[B, S]``.

    Returns:
        ``(q_rot, k_rot)`` with the same shapes/dtypes as the inputs.
    """
    head_dim = q.shape[-1]
    cos, sin = build_rope_cos_sin(spec, position_ids, head_dim, dtype=q.dtype)
    return apply_rotary_pos_emb_single(q, cos, sin), apply_rotary_pos_emb_single(k, cos, sin)


__all__ = ["rope_forward", "apply_rotary_pos_emb_single", "build_rope_cos_sin"]
