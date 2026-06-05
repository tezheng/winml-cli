"""Logical op primitives — pure functions on tensors.

These are the IHV-consensus floor (see research/03-ihv-opsets.v2.md). Each op
is a thin wrapper around PyTorch tensor ops; backends (ONNX/QNN/OpenVINO) would
re-implement the same signatures.

M1 implements the Qwen3 subset: silu, add, mul, linear, rms_norm, embed,
lm_head, rope_apply, sdpa.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def add(x: torch.Tensor, residual: torch.Tensor,
        scale: Optional[float] = None) -> torch.Tensor:
    if scale is None:
        return x + residual
    return x + scale * residual


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def linear(x: torch.Tensor, weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Standard linear; quantization is handled in api.quant via wrapper modules."""
    return F.linear(x, weight, bias)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    mode: str = "standard_w",
) -> torch.Tensor:
    """RMSNorm with optional Gemma 1+w mode.

    Compute in fp32 for numerical stability, return in x's dtype.
    """
    if mode not in ("standard_w", "one_plus_w"):
        raise ValueError(f"unknown mode: {mode!r}")
    orig_dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    x_normed = x32 * torch.rsqrt(variance + eps)
    w = weight.float()
    if mode == "one_plus_w":
        w = 1.0 + w
    return (x_normed * w).to(orig_dtype)


def embed(
    ids: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    out = F.embedding(ids, weight)
    if scale is not None:
        out = out * scale
    return out


def lm_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    logits = F.linear(x, weight)
    if scale is not None:
        logits = logits * scale
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """For SPLIT_HALF RoPE basis: rotate by π/2 around the head_dim axis."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(
    q: torch.Tensor,           # [B, S, H, Dh]
    k: torch.Tensor,           # [B, S, Hk, Dh]
    cos: torch.Tensor,         # [S, Dh]
    sin: torch.Tensor,         # [S, Dh]
    basis: str = "split_half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding.

    For Qwen3 / Llama / most modern SLMs we use SPLIT_HALF basis (GPT-NeoX style).
    INTERLEAVED (GPT-J) basis is supported for future models.
    """
    if basis not in ("split_half", "interleaved"):
        raise ValueError(f"unknown basis: {basis!r}")
    if basis == "interleaved":
        raise NotImplementedError("INTERLEAVED basis lands in a later milestone")

    # Broadcast cos/sin from [S, Dh] to [1, S, 1, Dh]
    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    q_rot = q * cos_b + _rotate_half(q) * sin_b
    k_rot = k * cos_b + _rotate_half(k) * sin_b
    return q_rot, k_rot


def sdpa(
    q: torch.Tensor,                       # [B, Hq, S, Dh]
    k: torch.Tensor,                       # [B, Hk, S, Dh]
    v: torch.Tensor,                       # [B, Hk, S, Dh]
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Scaled dot-product attention with GQA support.

    Repeats K/V to match Q heads when n_q_heads > n_kv_heads.
    """
    Hq = q.shape[1]
    Hk = k.shape[1]
    if Hk != Hq:
        if Hq % Hk != 0:
            raise ValueError(f"n_q_heads ({Hq}) must be divisible by n_kv_heads ({Hk})")
        repeats = Hq // Hk
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )
