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


def rope_apply_mrope(
    q: torch.Tensor,                         # [B, S, H, Dh]
    k: torch.Tensor,                         # [B, S, Hk, Dh]
    cos: torch.Tensor,                       # [3, B, S, Dh]
    sin: torch.Tensor,                       # [3, B, S, Dh]
    mrope_section: tuple[int, ...],          # e.g. (16, 24, 24) summing to Dh/2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Multimodal RoPE (Qwen2.5-VL).

    HF's `apply_multimodal_rotary_pos_emb` stitches 3 per-axis cos/sin tables
    into a SINGLE cos/sin tensor by taking, for each output channel band,
    the per-axis (T / H / W) table whose row index modulo 3 matches the
    band's position. The split semantic is "double" — `mrope_section * 2`
    because rotate_half pairs the two halves of head_dim.

    Verified against `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:596-606`:
        mrope_section = mrope_section * 2
        cos = cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, -1))], -1)
        sin = cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, -1))], -1)
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin

    The split-and-stitch yields cos/sin of shape [B, S, Dh] — i.e. the
    leading "3-axis" dim is consumed and the result is broadcastable to
    [B, H, S, Dh] for the rotate_half multiply.

    Args:
        q, k: queries/keys in [B, S, H, Dh] layout (NOT yet transposed).
        cos, sin: per-axis tables in [3, B, S, Dh] layout from the M-RoPE
            cache. The 3 axes are (temporal, height, width) per HF semantics.
        mrope_section: per-axis channel counts summing to Dh // 2.

    Returns:
        (q_rot, k_rot) at the input shapes.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be rank-4 [B,S,H,Dh], got {q.shape}")
    if cos.shape[0] != 3:
        raise ValueError(
            f"M-RoPE cos must have leading dim 3 (T/H/W), got {cos.shape}"
        )
    Dh = q.shape[-1]
    if sum(mrope_section) * 2 != Dh:
        raise ValueError(
            f"mrope_section sum*2={sum(mrope_section) * 2} must equal head_dim={Dh}"
        )
    sec = list(mrope_section) * 2                              # length 6 for 3 axes
    # cos/sin split along last dim, picking [i % 3] per chunk.
    cos_chunks = cos.split(sec, dim=-1)                        # 6 tensors [3, B, S, sec_i]
    sin_chunks = sin.split(sec, dim=-1)
    cos_stitched = torch.cat(
        [m[i % 3] for i, m in enumerate(cos_chunks)], dim=-1,
    )                                                          # [B, S, Dh]
    sin_stitched = torch.cat(
        [m[i % 3] for i, m in enumerate(sin_chunks)], dim=-1,
    )
    # Broadcast cos/sin from [B, S, Dh] to [B, S, 1, Dh] so it lines up with
    # q's [B, S, H, Dh] head axis.
    cos_b = cos_stitched.unsqueeze(2)
    sin_b = sin_stitched.unsqueeze(2)
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


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation — `(F.relu(x)) ** 2`.

    Used by BitNet b1.58 (config.hidden_act='relu2') and Nemotron 3.
    Source: `transformers/activations.py::ReLU2Activation` —
    `relu(x).pow(2)`.
    """
    return F.relu(x).pow(2)


def build_alibi_slopes(n_heads: int, alibi_bias_max: float = 8.0,
                       device=None, dtype=torch.float32) -> torch.Tensor:
    """Build MPT canonical ALiBi slopes.

    Mirrors `transformers/models/mpt/modeling_mpt.py:42-62`
    (`build_mpt_alibi_tensor`) — slope construction only, NOT the
    full bias tensor. Returns shape [n_heads], fp32 by default.

    For n_heads that is a power of 2:
        base[h] = (h+1) * (alibi_bias_max / n_heads)   for h=0..n-1
        slope[h] = 1.0 / (2 ** base[h])
                 = 2 ** (-(h+1) * alibi_bias_max / n_heads)

    For non-power-of-2 n_heads, MPT computes the next power-of-2
    slopes then interleaves odd-then-even and truncates to n_heads.
    """
    import math
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    n_heads_p2 = 2 ** math.ceil(math.log2(n_heads)) if n_heads > 1 else 1
    base = torch.arange(1, n_heads_p2 + 1, dtype=torch.int64, device=device).float()
    base = base * (alibi_bias_max / n_heads_p2)
    slopes = 1.0 / torch.pow(2.0, base)            # [n_heads_p2]
    if n_heads_p2 != n_heads:
        # MPT interleave + truncate: take odd-indexed half then even-indexed
        # half from the larger power-of-2 set. Source: modeling_mpt.py:58-59.
        slopes = torch.cat([slopes[1::2], slopes[::2]], dim=0)[:n_heads]
    return slopes.to(dtype)


def apply_alibi(
    scores: torch.Tensor,           # [B, H, S_q, T]
    slopes: torch.Tensor,           # [H], pre-computed via build_alibi_slopes
    start_pos: int = 0,
) -> torch.Tensor:
    """Add ALiBi position bias to attention scores (pre-softmax).

    The bias for query at absolute position i and key at position j is
        bias[h, i, j] = slope[h] * (j - i)

    Note j <= i in causal contexts, so (j - i) <= 0 and the bias is
    negative-or-zero. The causal mask is applied SEPARATELY by the
    caller (so this op is identical for non-causal use, e.g. encoder).

    Mirrors `transformers/models/mpt/modeling_mpt.py:42-62` slope build
    + 121 (`attention_scores = attention_scores + position_bias`). The
    MPT impl uses `alibi[..., 0:1, :T] * slope` to broadcast — here we
    compute the full (S_q, T) bias to support `start_pos > 0` (cached
    decoding).
    """
    B, H, S_q, T = scores.shape
    if slopes.shape != (H,):
        raise ValueError(
            f"apply_alibi: slopes shape {tuple(slopes.shape)} must be ({H},)"
        )
    device = scores.device
    # Absolute positions: query i has position start_pos + i; key j has
    # position j. Offset (j - i_abs) is <= 0 under causal masking.
    q_pos = torch.arange(start_pos, start_pos + S_q,
                         device=device, dtype=torch.long).view(S_q, 1)
    k_pos = torch.arange(T, device=device, dtype=torch.long).view(1, T)
    offset = (k_pos - q_pos).to(scores.dtype)               # [S_q, T]
    # slopes: [H] -> [1, H, 1, 1]; offset: [S_q, T] -> [1, 1, S_q, T]
    bias = slopes.view(1, H, 1, 1).to(scores.dtype) * offset.view(1, 1, S_q, T)
    return scores + bias


def sdpa(
    q: torch.Tensor,                       # [B, Hq, S, Dh]
    k: torch.Tensor,                       # [B, Hk, S, Dh]
    v: torch.Tensor,                       # [B, Hk, S, Dh]
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    logit_softcap: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
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

    # v5-phase2: ALiBi (`attn_bias`) and trained sinks (`sinks`) both
    # require a manual matmul path because F.scaled_dot_product_attention
    # has no pre-softmax additive bias hook beyond `attn_mask`. Both
    # paths route through the manual branch below (the logit_softcap
    # branch is the existing manual path).
    if attn_bias is not None or sinks is not None:
        if scale is None:
            scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_bias is not None:
            attn = attn + attn_bias
        if attn_mask is not None:
            attn = attn + attn_mask
        elif is_causal:
            S_q, S_k = q.shape[-2], k.shape[-2]
            mask = torch.full((S_q, S_k), float("-inf"), dtype=q.dtype, device=q.device)
            mask = torch.triu(mask, diagonal=1 + (S_k - S_q))
            attn = attn + mask
        if sinks is not None:
            # GPT-OSS trained sinks: append the per-head learned scalar
            # as one extra logit column, softmax over (T+1) slots, then
            # drop the sink column. Source:
            # `transformers/models/gpt_oss/modeling_gpt_oss.py:267-275`.
            if sinks.dim() != 1 or sinks.shape[0] != Hq:
                raise ValueError(
                    f"sinks shape {tuple(sinks.shape)} must be ({Hq},)"
                )
            B, H, S_q, T = attn.shape
            sink_broadcast = sinks.view(1, H, 1, 1).expand(B, H, S_q, 1).to(attn.dtype)
            combined = torch.cat([attn, sink_broadcast], dim=-1)
            # Numerically stable softmax (HF subtracts max as part of the
            # sink impl — keep that to mirror modeling_gpt_oss.py:273).
            combined = combined - combined.max(dim=-1, keepdim=True).values
            probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
            probs = probs[..., :-1]
            return torch.matmul(probs, v)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(attn, v)

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


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Pad along the seq_len axis (axis 1). Mirrors modeling_mamba2.py:40-48."""
    if pad_size == 0:
        return input_tensor
    if input_tensor.dim() == 4:
        # [B, S, H, D] -> [B, S+pad, H, D]
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0)
    elif input_tensor.dim() == 3:
        # [B, S, H] -> [B, S+pad, H]
        pad_shape = (0, 0, 0, pad_size, 0, 0)
    else:
        raise ValueError(f"unsupported rank: {input_tensor.dim()}")
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int,
                         chunk_size: int) -> torch.Tensor:
    """Pad + reshape into chunks. Mirrors modeling_mamba2.py:51-68."""
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)
    if input_tensor.dim() == 3:
        # [B, S, H] -> [B, -1, chunk_size, H]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    # rank 4: [B, S, H, D] -> [B, -1, chunk_size, H, D]
    return input_tensor.reshape(
        input_tensor.shape[0], -1, chunk_size,
        input_tensor.shape[2], input_tensor.shape[3]
    )


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """Numerically stable segment cumulative sum.

    Mirrors `modeling_mamba2.py:71-88` (segment_sum).

    Input shape: [..., chunk_size]. Output shape: [..., chunk_size, chunk_size].
    Strict lower-triangular below the diagonal, -inf above (inclusive diagonal
    kept in step 4 of the original).

    Step-by-step (HF impl):
    1. expand last dim by chunk_size repetitions
    2. mask above sub-diagonal (diagonal=-1) → zeros above the sub-diag
    3. cumsum along the row axis (dim=-2)
    4. mask above diagonal (diagonal=0) → -inf to make exp() vanish
    """
    chunk_size = input_tensor.size(-1)
    # 1. [..., chunk_size] -> [..., chunk_size, chunk_size]
    expanded = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. lower-tri mask strictly below diagonal (so diagonal=0)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=-1,
    )
    expanded = expanded.masked_fill(~mask, 0)
    # 3. cumsum along the inner row axis
    tensor_segsum = torch.cumsum(expanded, dim=-2)
    # 4. inclusive lower-tri mask: keep diagonal and below, set above to -inf
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=0,
    )
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def selective_scan(
    hidden_states: torch.Tensor,    # [B, S, num_heads, head_dim] — discretized x = x * dt
    A: torch.Tensor,                # [B, S, num_heads]           — discretized A = A_log_neg * dt
    B: torch.Tensor,                # [B, S, num_heads, d_state]  — B repeated to num_heads
    C: torch.Tensor,                # [B, S, num_heads, d_state]  — C repeated to num_heads
    chunk_size: int,
    initial_state: Optional[torch.Tensor] = None,   # [B, num_heads, head_dim, d_state]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mamba-2 SSD (Structured State-Space Duality) chunk-parallel selective scan.

    REFERENCE implementation in pure PyTorch. Prioritizes CORRECTNESS over speed.
    Faithfully mirrors `modeling_mamba2.py::Mamba2Mixer.torch_forward` lines
    503-577 (`begin ssd naive` ... end of else branch).

    Inputs (already discretized by the caller — caller must pre-multiply x*dt
    and A_log_neg*dt because the HF code does that on lines 517-518):
        hidden_states: [B, S, num_heads, head_dim] — x * dt, fp32
        A: [B, S, num_heads]                       — A_neg * dt, fp32
        B: [B, S, num_heads, d_state]              — already tiled num_heads
        C: [B, S, num_heads, d_state]              — already tiled num_heads
        chunk_size: int — must match the SSDSpec.chunk_size
        initial_state: optional [B, num_heads, head_dim, d_state] prior state
            (for sequential prefill of a longer sequence; HF model uses zeros
            for the very first chunk batch).

    Returns:
        y: [B, S, num_heads, head_dim] — the scan output BEFORE D-residual and
           BEFORE the gated norm. Already truncated to the unpadded S.
        ssm_state: [B, num_heads, head_dim, d_state] — the final recurrent state
           after processing all chunks (suitable for cache.update_recurrent_state).

    Note: the D-residual `D * hidden_states_unscaled` and the final norm
    (`norm(y, gate)`) are applied by the CALLER, not this op (HF does them
    in the mixer's forward, lines 514, 569, 579).

    Correctness key: the four conceptual steps of SSD (HF inline comments,
    lines 527-565):
    1. Y_diag = intra-chunk causal SSD via L = exp(segment_sum(A)) and G = C·B^T.
    2. states (per-chunk) = right-term decay B_decay = B * exp(A_cumsum_last - A_cumsum), summed over the chunk.
    3. inter-chunk recurrence via decay_chunk = exp(segment_sum(pad(A_cumsum_last, (1,0)))).
    4. Y_off = C @ states * exp(A_cumsum) — left-term decay.
    Final y = Y_diag + Y_off, truncated to seq_len.
    """
    if hidden_states.dim() != 4:
        raise ValueError(
            f"hidden_states must be [B, S, num_heads, head_dim], got rank {hidden_states.dim()}"
        )
    B_n, S, num_heads, head_dim = hidden_states.shape
    if A.shape != (B_n, S, num_heads):
        raise ValueError(
            f"A shape {tuple(A.shape)} must be [B, S, num_heads]=({B_n},{S},{num_heads})"
        )
    if B.shape[:3] != (B_n, S, num_heads) or C.shape != B.shape:
        raise ValueError(
            f"B/C shape {tuple(B.shape)}/{tuple(C.shape)} must be "
            f"[B, S, num_heads, d_state]"
        )
    d_state = B.shape[-1]

    # Pad sequence length up to a multiple of chunk_size (HF line 512).
    pad_size = (chunk_size - S % chunk_size) % chunk_size

    # 1. Rearrange into chunks (HF line 521).
    hs, A_c, B_c, C_c = [
        _reshape_into_chunks(t, pad_size, chunk_size)
        for t in (hidden_states, A, B, C)
    ]
    # After reshape: hs [B, n_chunks, chunk, num_heads, head_dim]
    #               A_c [B, n_chunks, chunk, num_heads]
    #               B_c [B, n_chunks, chunk, num_heads, d_state]

    # A_c -> [B, num_heads, n_chunks, chunk] (HF line 524)
    A_c = A_c.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_c, dim=-1)

    # --- Step 1: intra-chunk diagonal block ---
    # L = exp(segment_sum(A))  -> [B, num_heads, n_chunks, chunk, chunk]
    L = torch.exp(_segment_sum(A_c))

    # G = C · B^T contracted over d_state -> [B, n_chunks, chunk_l, chunk_s, num_heads]
    # G_intermediate: [B, n_chunks, l, s, num_heads, d_state]
    G_intermediate = C_c[:, :, :, None, :, :] * B_c[:, :, None, :, :, :]
    G = G_intermediate.sum(dim=-1)

    # M = G * L  (apply per-head decay)
    # L permuted to [B, n_chunks, chunk_l, chunk_s, num_heads]
    L_perm = L.permute(0, 2, 3, 4, 1)
    M_intermediate = G[..., None] * L_perm[..., None]
    M = M_intermediate.sum(dim=-1)   # [B, n_chunks, l, s, num_heads]

    # Y_diag = sum_s M[..., l, s, h] * hidden_states[..., s, h, d]
    # M: [B, n_chunks, l, s, num_heads] -> add last dim
    # hidden_states reshape: [B, n_chunks, s, num_heads, head_dim]
    Y_diag = (M[..., None] * hs[:, :, None]).sum(dim=3)
    # Y_diag: [B, n_chunks, chunk_l, num_heads, head_dim]

    # --- Step 2: per-chunk states (right-term B_decay) ---
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # decay_states: [B, num_heads, n_chunks, chunk]
    # B_c: [B, n_chunks, chunk, num_heads, d_state]
    # B_decay should match B_c: permute decay_states to [B, n_chunks, chunk, num_heads]
    B_decay = B_c * decay_states.permute(0, 2, 3, 1)[..., None]
    # states = sum_s (B_decay[..., d_state, None] * hs[..., None]) along chunk axis (dim=2)
    states = (B_decay[..., None, :] * hs[..., None]).sum(dim=2)
    # states: [B, n_chunks, num_heads, head_dim, d_state]

    # --- Step 3: inter-chunk recurrence ---
    if initial_state is not None:
        # initial_state: [B, num_heads, head_dim, d_state]
        # Prepend as the "zeroth" chunk state.
        previous_states = initial_state.unsqueeze(1)
    else:
        previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(
        _segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
    )
    # decay_chunk: [B, num_heads, n_chunks+1, n_chunks+1]
    decay_chunk = decay_chunk.transpose(1, 3)  # [B, n_chunks+1, n_chunks+1, num_heads]
    # new_states[k] = sum_j decay_chunk[k, j] * states[j]
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    # new_states: [B, n_chunks+1, num_heads, head_dim, d_state]
    states, ssm_state = new_states[:, :-1], new_states[:, -1]

    # --- Step 4: state -> output (left-term C_decay) ---
    state_decay_out = torch.exp(A_cumsum)  # [B, num_heads, n_chunks, chunk]
    # C_c: [B, n_chunks, chunk, num_heads, d_state]
    # states: [B, n_chunks, num_heads, head_dim, d_state]
    C_times_states = C_c[..., None, :] * states[:, :, None, ...]
    # C_times_states: [B, n_chunks, chunk, num_heads, head_dim, d_state]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]
    # Y_off: [B, n_chunks, chunk, num_heads, head_dim]

    # Combine and reshape.
    y = Y_diag + Y_off
    y = y.reshape(B_n, -1, num_heads, head_dim)
    if pad_size > 0:
        y = y[:, :S, :, :]
    return y, ssm_state
