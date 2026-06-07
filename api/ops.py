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
    cos: torch.Tensor,         # [S, Dh] for SPLIT_HALF; [S, Dh/2] for INTERLEAVED
    sin: torch.Tensor,         # [S, Dh] for SPLIT_HALF; [S, Dh/2] for INTERLEAVED
    basis: str = "split_half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding.

    For Qwen3 / Llama / most modern SLMs we use SPLIT_HALF basis (GPT-NeoX style).

    B4: INTERLEAVED basis (Llama 4) — pairs (q[..., 2i], q[..., 2i+1]) are
    rotated by angle θ_i directly, mathematically equivalent to complex
    multiplication `(real + j*imag) * (cos(θ) + j*sin(θ))`. cos/sin tables
    have shape [S, Dh/2] (one entry per pair). Source: modeling_llama4.py
    L239 (freqs_cis = polar(1, freqs)) and L245-254 (view_as_complex →
    complex multiply → view_as_real → flatten).
    """
    if basis not in ("split_half", "interleaved"):
        raise ValueError(f"unknown basis: {basis!r}")
    if basis == "interleaved":
        # cos/sin must span Dh/2 — one entry per (even, odd) pair.
        Dh = q.shape[-1]
        if Dh % 2 != 0:
            raise ValueError(f"head_dim must be even, got {Dh}")
        if cos.shape[-1] != Dh // 2:
            raise ValueError(
                f"INTERLEAVED cos last dim must be Dh/2 ({Dh // 2}), "
                f"got {cos.shape[-1]}"
            )
        # Broadcast cos/sin from [S, Dh/2] to [1, S, 1, Dh/2].
        cos_b = cos.unsqueeze(0).unsqueeze(2)
        sin_b = sin.unsqueeze(0).unsqueeze(2)

        def _rot(x: torch.Tensor) -> torch.Tensor:
            # Reshape last dim from Dh to (Dh/2, 2). Channel ordering is
            # (real, imag) per pair.
            x_pairs = x.reshape(*x.shape[:-1], Dh // 2, 2)
            x_re = x_pairs[..., 0]
            x_im = x_pairs[..., 1]
            y_re = x_re * cos_b - x_im * sin_b
            y_im = x_re * sin_b + x_im * cos_b
            y = torch.stack([y_re, y_im], dim=-1)
            return y.reshape(*x.shape)

        return _rot(q), _rot(k)

    # Broadcast cos/sin from [S, Dh] to [1, S, 1, Dh]
    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    q_rot = q * cos_b + _rotate_half(q) * sin_b
    k_rot = k * cos_b + _rotate_half(k) * sin_b
    return q_rot, k_rot


def rope_apply_partial(
    q: torch.Tensor,           # [B, S, H, Dh]
    k: torch.Tensor,           # [B, S, Hk, Dh]
    cos: torch.Tensor,         # [S, Dh_rot] — cos/sin computed at the rotated dim only
    sin: torch.Tensor,         # [S, Dh_rot]
    partial_rotary_factor: float,
    basis: str = "split_half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to only the first `partial_rotary_factor * Dh` channels of each head.

    For Gemma 4 global layers: partial_rotary_factor=0.25, so the first 64 of 256
    head_dim channels rotate, the remaining 192 pass through unchanged.

    For MLA: q_rope is a separate sub-head dim already so the caller pre-slices;
    this helper handles the simpler "head is a single tensor with a rotated prefix"
    case used by Gemma 4 and Phi-3 partial-RoPE.
    """
    if not (0.0 < partial_rotary_factor <= 1.0):
        raise ValueError(f"partial_rotary_factor must be in (0, 1], got {partial_rotary_factor}")
    if partial_rotary_factor == 1.0:
        return rope_apply(q, k, cos, sin, basis=basis)

    Dh = q.shape[-1]
    Dh_rot = int(Dh * partial_rotary_factor)
    if Dh_rot % 2 != 0:
        raise ValueError(f"rotated head_dim ({Dh_rot}) must be even")
    if cos.shape[-1] != Dh_rot or sin.shape[-1] != Dh_rot:
        raise ValueError(f"cos/sin last dim must equal Dh_rot ({Dh_rot}), got {cos.shape[-1]}")

    q_rot_in, q_pass = q[..., :Dh_rot], q[..., Dh_rot:]
    k_rot_in, k_pass = k[..., :Dh_rot], k[..., Dh_rot:]
    q_rot, k_rot = rope_apply(q_rot_in, k_rot_in, cos, sin, basis=basis)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """Gemma's `gelu_pytorch_tanh` activation — `F.gelu(x, approximate='tanh')`."""
    return F.gelu(x, approximate="tanh")


def sdpa(
    q: torch.Tensor,                       # [B, Hq, S, Dh]
    k: torch.Tensor,                       # [B, Hk, S, Dh]
    v: torch.Tensor,                       # [B, Hk, S, Dh]
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    logit_softcap: Optional[float] = None,
) -> torch.Tensor:
    """Scaled dot-product attention with GQA support.

    Repeats K/V to match Q heads when n_q_heads > n_kv_heads.

    When ``logit_softcap`` is set (Gemma 2: 50.0), falls back to a manual
    matmul → scale → tanh-softcap → mask-add → softmax → matmul path because
    ``F.scaled_dot_product_attention`` does not expose pre-softmax post-scale
    transforms. Softcap order matches HF Gemma 2 eager attention:

        attn = (q @ k.T) * scale
        attn = tanh(attn / softcap) * softcap
        attn = attn + mask
        attn = softmax(attn, fp32-upcast).to(q.dtype)
        out = attn @ v

    Source: ``modeling_gemma2.py:212-225`` (eager_attention_forward).
    """
    Hq = q.shape[1]
    Hk = k.shape[1]
    if Hk != Hq:
        if Hq % Hk != 0:
            raise ValueError(f"n_q_heads ({Hq}) must be divisible by n_kv_heads ({Hk})")
        repeats = Hq // Hk
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    if logit_softcap is not None:
        # Manual eager attention to honor Gemma 2 softcap semantics.
        # The default scale (when None) is head_dim ** -0.5; mirror SDPA's behavior.
        if scale is None:
            scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.tanh(attn / logit_softcap) * logit_softcap
        if attn_mask is not None:
            attn = attn + attn_mask
        elif is_causal:
            # Fall back to building a causal mask. We don't expect is_causal=True
            # in the softcap path (callers supply explicit attn_mask), but support
            # it defensively.
            S_q, S_k = q.shape[-2], k.shape[-2]
            mask = torch.full((S_q, S_k), float("-inf"), dtype=q.dtype, device=q.device)
            mask = torch.triu(mask, diagonal=1 + (S_k - S_q))
            attn = attn + mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn, v)

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Standard LayerNorm. Compute in fp32, return in x's dtype."""
    orig = x.dtype
    return F.layer_norm(x.float(), x.shape[-1:], weight.float(),
                        bias.float(), eps).to(orig)


def softmax(x: torch.Tensor, dim: int = -1, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if dtype is None:
        return F.softmax(x, dim=dim)
    return F.softmax(x, dim=dim, dtype=dtype)


def top_k(x: torch.Tensor, k: int, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (values, indices). MoE router uses this."""
    return torch.topk(x, k, dim=dim)


def gather(x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, dim, index)


def scatter(x: torch.Tensor, dim: int, index: torch.Tensor,
            src: torch.Tensor) -> torch.Tensor:
    """Non-in-place scatter — returns a new tensor."""
    out = x.clone()
    return out.scatter(dim, index, src)


def conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
) -> torch.Tensor:
    """1D conv — Mamba uses this for its causal conv."""
    return F.conv1d(x, weight, bias=bias, stride=stride, padding=padding, groups=groups)


def selective_scan(*args, **kwargs):
    """Mamba selective-scan op. Lands fully in the M2 SSM batch."""
    raise NotImplementedError("selective_scan is reserved for the M2 SSM batch")
