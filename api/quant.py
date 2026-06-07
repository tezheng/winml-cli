"""Quantization: AWQ W4A16, GGUF Q4_K_M, FP8 E4M3 W8A8, MXFP4 round-trip.

Each scheme below is grounded in the upstream reference implementation and
exercises a distinct axis of `specs.QuantSpec`:

  - AWQ W4A16          : INT4 grouped, asymmetric, AWQ_INTERLEAVE packing.
  - GGUF Q4_K_M        : INT4 super-block (256), 8 sub-blocks of 32 elements,
                         6-bit per-sub-block scale+min, fp16 super-block d/dmin.
  - FP8 E4M3 W8A8      : per-tensor weight scale + per-token activation scale,
                         fp32 accumulator.
  - MXFP4              : 32-element block; UE8M0 shared exponent + E2M1 mantissas.

Source citations are inline at each function. See per-function docstring for
tolerance expectations.
"""
from __future__ import annotations
from typing import Tuple

import torch

from api import specs, types


# ============================================================================
# AWQ W4A16 grouped INT4
# ============================================================================
#
# Reference: AWQ uses the nibble permutation [0,2,4,6,1,3,5,7] when packing
# 8 INT4 values into one INT32. We replicate the storage layout so we can
# load real AWQ checkpoints (verified in tests/models/qwen3/test_quant_awq.py
# on Qwen3-0.6B q_proj).
#
# Layout (for W of shape [K, N], group_size G along K):
#     qweight  : int32[K // 8, N]              -- 8 nibbles packed per int32
#     qzeros   : int32[K // G, N // 8]         -- 8 zero-nibbles packed per int32
#     scales   : float16[K // G, N]            -- per-group scale

# AWQ nibble permutation when packing 8 INT4 values into one INT32 along axis K.
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _awq_shifts() -> torch.Tensor:
    """Bit shifts for the 8 nibbles in one INT32, in AWQ order."""
    return torch.tensor([4 * p for p in AWQ_ORDER], dtype=torch.int32)


def awq_pack(int_values: torch.Tensor) -> torch.Tensor:
    """Pack INT4 (0..15) values along axis 0 into INT32 with AWQ interleave.

    int_values: [K, N] int32 in range [0, 16)
    returns:    [K // 8, N] int32
    """
    if int_values.dtype != torch.int32:
        raise ValueError(f"expected int32, got {int_values.dtype}")
    K, N = int_values.shape
    if K % 8 != 0:
        raise ValueError(f"K ({K}) must be divisible by 8")
    iv = int_values & 0xF
    iv = iv.reshape(K // 8, 8, N)
    shifts = _awq_shifts().view(1, 8, 1)
    packed = (iv << shifts).sum(dim=1).to(torch.int32)
    return packed


def awq_unpack(packed: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Inverse of awq_pack."""
    if packed.shape != (K // 8, N):
        raise ValueError(f"shape mismatch: packed {packed.shape} vs K//8={K//8}, N={N}")
    shifts = _awq_shifts().view(1, 8, 1)
    out = (packed.unsqueeze(1) >> shifts) & 0xF
    return out.reshape(K, N).to(torch.int32)


def awq_quantize(
    w: torch.Tensor,
    spec: specs.QuantSpec,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize w using AWQ-style grouped INT4 with asymmetric zero-point.

    w: [K, N] in scale_dtype (fp16/bf16)
    returns (qweight, scales, qzeros)
    """
    if spec.qdtype != types.QDType.INT4:
        raise ValueError("AWQ path requires INT4")
    if spec.group_size is None or spec.group_size <= 0:
        raise ValueError("AWQ requires positive group_size")
    if not spec.has_zero_point:
        raise ValueError("AWQ uses asymmetric quant — has_zero_point must be True")
    K, N = w.shape
    G = spec.group_size
    if K % G != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({G})")

    w32 = w.float().reshape(K // G, G, N)
    w_min = w32.min(dim=1, keepdim=True).values
    w_max = w32.max(dim=1, keepdim=True).values
    qmax = 15.0
    scales = (w_max - w_min) / qmax
    scales = scales.clamp_min(1e-8)
    zero_floats = -w_min / scales
    zeros_int = zero_floats.round().clamp(0, qmax).to(torch.int32).squeeze(1)

    q_int = ((w32 / scales) + zero_floats).round().clamp(0, qmax).to(torch.int32)
    q_int = q_int.reshape(K, N)
    qweight = awq_pack(q_int)

    qzeros_n_packed = _pack_n_axis(zeros_int)
    scales = scales.squeeze(1).to(spec.scale_dtype)
    return qweight, scales, qzeros_n_packed


def _pack_n_axis(zeros_int: torch.Tensor) -> torch.Tensor:
    """Pack 8 INT4 values along N axis into one INT32, AWQ order."""
    n_groups, N = zeros_int.shape
    if N % 8 != 0:
        raise ValueError(f"N ({N}) must be divisible by 8 for AWQ zeros packing")
    iv = zeros_int & 0xF
    iv = iv.reshape(n_groups, N // 8, 8)
    shifts = _awq_shifts().view(1, 1, 8)
    return (iv << shifts).sum(dim=-1).to(torch.int32)


def _unpack_n_axis(qzeros_packed: torch.Tensor, N: int) -> torch.Tensor:
    n_groups = qzeros_packed.shape[0]
    if qzeros_packed.shape[1] != N // 8:
        raise ValueError(f"qzeros shape mismatch: got {qzeros_packed.shape}, expected (*, {N//8})")
    shifts = _awq_shifts().view(1, 1, 8)
    out = (qzeros_packed.unsqueeze(-1) >> shifts) & 0xF
    return out.reshape(n_groups, N).to(torch.int32)


def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    spec: specs.QuantSpec,
    K: int,
    N: int,
) -> torch.Tensor:
    """Reverse of awq_quantize. Returns the dequantized weight [K, N] in scale_dtype."""
    if spec.group_size is None or spec.group_size <= 0:
        raise ValueError("requires positive group_size")
    G = spec.group_size
    q_int = awq_unpack(qweight, K, N)
    zeros_int = _unpack_n_axis(qzeros, N)
    n_groups = K // G
    q_int_g = q_int.reshape(n_groups, G, N).float()
    zeros_g = zeros_int.reshape(n_groups, 1, N).float()
    scales_g = scales.reshape(n_groups, 1, N).float()
    w = (q_int_g - zeros_g) * scales_g
    return w.reshape(K, N).to(spec.scale_dtype)


# ============================================================================
# GGUF Q4_K_M  (k-quant 4-bit, "K"-form with mins)
# ============================================================================
#
# Source: ggml/src/ggml-common.h  (block_q4_K struct)
#         ggml/src/ggml-quants.c  (quantize_row_q4_K_ref / dequantize_row_q4_K
#                                  / get_scale_min_k4)
# https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
# https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c
#
# Super-block of QK_K = 256 weights.
#   block_q4_K {
#     ggml_half d;          // fp16 super-block scale  (max_scale / 63)
#     ggml_half dmin;       // fp16 super-block min    (max_min   / 63)
#     uint8_t scales[12];   // 8 packed (6-bit scale, 6-bit min) sub-block params
#     uint8_t qs[128];      // 256 nibbles, two-nibble-per-byte
#   } // 4 + 12 + 128 = 144 bytes per block
#
# Sub-block layout: 8 sub-blocks of 32 weights each.
# Per-sub-block dequant:  y = d*(sc * q) - dmin*(m)
#   where sc ∈ [0, 63]  6-bit scale
#         m  ∈ [0, 63]  6-bit min
#         q  ∈ [0, 15]  4-bit quant
#
# Bit-pack of `scales[12]` (j = 0..7):
#   if j < 4:
#       scales[j]     = sc_j        (lower 6 bits)
#       scales[j + 4] = m_j         (lower 6 bits)
#   else:
#       scales[j + 4]  = (sc_j & 0x0F) | ((m_j & 0x0F) << 4)
#       scales[j - 4] |= ((sc_j >> 4) << 6)    // high 2 bits of sc_j into bits 6-7
#       scales[j]     |= ((m_j  >> 4) << 6)    // high 2 bits of m_j  into bits 6-7
#
# qs packing: per 64 weights, 32 bytes
#   for l in 0..31: qs[l] = q[base + l] | (q[base + l + 32] << 4)
#
# Tolerance: this module uses the SIMPLE per-sub-block min/max scheme rather
# than the reference `make_qkx2_quants` weighted search. Observed median
# rel-err on Gaussian rows / real Qwen3 q_proj: ~9-13%. The reference search
# gets to ~3-5% but is materially more code; we accept the looser bound and
# document it at the test site.

GGUF_QK_K = 256
GGUF_K_SCALE_SIZE = 12
GGUF_BYTES_PER_BLOCK = 4 + GGUF_K_SCALE_SIZE + GGUF_QK_K // 2  # = 144


def _gguf_pack_scales_mins(sc: torch.Tensor, mn: torch.Tensor) -> torch.Tensor:
    """Pack 8 (6-bit scale, 6-bit min) pairs into 12 bytes per block.

    sc, mn: int tensors of shape [B, 8] with values in [0, 63].
    returns: uint8 tensor of shape [B, 12].
    """
    B = sc.shape[0]
    out = torch.zeros((B, 12), dtype=torch.uint8)
    sc8 = sc.to(torch.int32) & 0x3F
    mn8 = mn.to(torch.int32) & 0x3F
    # j < 4
    for j in range(4):
        out[:, j] = (sc8[:, j] & 0x3F).to(torch.uint8)
        out[:, j + 4] = (mn8[:, j] & 0x3F).to(torch.uint8)
    # j >= 4
    for j in range(4, 8):
        out[:, j + 4] = ((sc8[:, j] & 0x0F) | ((mn8[:, j] & 0x0F) << 4)).to(torch.uint8)
        out[:, j - 4] = (out[:, j - 4].to(torch.int32) | ((sc8[:, j] >> 4) << 6)).to(torch.uint8)
        out[:, j] = (out[:, j].to(torch.int32) | ((mn8[:, j] >> 4) << 6)).to(torch.uint8)
    return out


def _gguf_unpack_scales_mins(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `_gguf_pack_scales_mins`.

    packed: uint8 [B, 12]
    returns: (sc, mn) int32 [B, 8] each in [0, 63]
    """
    B = packed.shape[0]
    p = packed.to(torch.int32)
    sc = torch.zeros((B, 8), dtype=torch.int32)
    mn = torch.zeros((B, 8), dtype=torch.int32)
    # mirrors get_scale_min_k4
    for j in range(8):
        if j < 4:
            sc[:, j] = p[:, j] & 0x3F
            mn[:, j] = p[:, j + 4] & 0x3F
        else:
            sc[:, j] = (p[:, j + 4] & 0x0F) | (((p[:, j - 4] >> 6) & 0x03) << 4)
            mn[:, j] = ((p[:, j + 4] >> 4) & 0x0F) | (((p[:, j] >> 6) & 0x03) << 4)
    return sc, mn


def _gguf_pack_qs(quants: torch.Tensor) -> torch.Tensor:
    """Pack 256 4-bit quants per block into 128 bytes.

    Mirrors ggml: per 64-weight chunk (j step 64),
        qs[l] = q[j + l] | (q[j + l + 32] << 4)   for l in [0, 32).

    quants: int32 [B, 256] in [0, 15].
    returns: uint8 [B, 128].
    """
    B = quants.shape[0]
    q = (quants & 0xF).to(torch.int32)
    out = torch.zeros((B, 128), dtype=torch.uint8)
    # 4 chunks of 64 weights -> 32 bytes each.
    for chunk in range(4):
        base = chunk * 64
        dst_base = chunk * 32
        low = q[:, base : base + 32]
        high = q[:, base + 32 : base + 64]
        out[:, dst_base : dst_base + 32] = (low | (high << 4)).to(torch.uint8)
    return out


def _gguf_unpack_qs(qs: torch.Tensor) -> torch.Tensor:
    """Inverse of `_gguf_pack_qs`. Returns int32 [B, 256]."""
    B = qs.shape[0]
    out = torch.zeros((B, 256), dtype=torch.int32)
    p = qs.to(torch.int32)
    for chunk in range(4):
        base = chunk * 64
        dst_base = chunk * 32
        low = p[:, dst_base : dst_base + 32] & 0xF
        high = (p[:, dst_base : dst_base + 32] >> 4) & 0xF
        out[:, base : base + 32] = low
        out[:, base + 32 : base + 64] = high
    return out


def gguf_q4_k_quantize(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 1D row to GGUF Q4_K blocks.

    Mirrors quantize_row_q4_K_ref (ggml-quants.c). Uses the simplified
    per-sub-block min/max search rather than the make_qkx2_quants weighted
    search — this is sufficient for synthetic round-trip verification and
    matches the dequant inverse exactly for any sub-block scale derivation.

    w: float tensor of shape [N], with N % 256 == 0.
    returns: (d_fp16, dmin_fp16, scales_packed_u8, qs_u8)
      d_fp16        : fp16 [B]      super-block scale (max sub-scale / 63)
      dmin_fp16     : fp16 [B]      super-block dmin  (max sub-min   / 63)
      scales_packed : uint8 [B, 12] 8 packed (6-bit sc, 6-bit min) pairs
      qs            : uint8 [B, 128] 256 packed 4-bit quants per block
    """
    if w.dim() != 1:
        raise ValueError(f"expected 1D weight, got shape {tuple(w.shape)}")
    N = w.shape[0]
    if N % GGUF_QK_K != 0:
        raise ValueError(f"N ({N}) must be divisible by QK_K ({GGUF_QK_K})")
    B = N // GGUF_QK_K
    w32 = w.float().reshape(B, 8, 32)  # [B, sub_blocks, 32]

    # Per-sub-block scale / min. We use the simple min/max scheme:
    #   q = clip(round((x - mn) / sc), 0, 15)
    #   dequant: x' = sc * q + mn
    # which matches dequantize_row_q4_K: y = d*sc_q * q - dmin*m_q with
    # sc = d*sc_q and mn = -dmin*m_q (mn is the additive offset that recovers
    # the per-sub-block min after the 0-anchored quant).
    mins = w32.min(dim=-1).values  # [B, 8] -- value at q=0
    maxs = w32.max(dim=-1).values
    sc_per_sb = (maxs - mins) / 15.0
    sc_per_sb = sc_per_sb.clamp_min(1e-8)

    # mins are stored as POSITIVE offsets to subtract:
    #   y = d*sc_q * q - dmin*m_q  =>  the constant term -dmin*m_q must equal
    #   our additive `mins` (which can be negative). So m_q is positive and
    #   we set:   dmin*m_q = -mins  =>  if mins > 0, the formula needs dmin*m_q
    #   to be NEGATIVE which it can't be (both >= 0). To handle that, GGUF's
    #   make_qkx2_quants does a non-trivial search. For synthetic round-trip
    #   we use the equivalent re-parametrization:
    #     mn_pos = -mins   (so y = sc * q + mins = sc * q - mn_pos)
    # then dmin and m_q encode mn_pos.
    mn_per_sb_pos = (-mins).clamp_min(0.0)
    # Note: when mins > 0 (rare for zero-mean weight tensors), mn_per_sb_pos
    # is 0 and the recovered min from dequant is 0, which slightly biases the
    # quant grid. For typical zero-centered or negative-tilted weight rows
    # this is fine. We accept the bias and clamp during synthesis.

    # Super-block max-scale and max-min.
    max_scale = sc_per_sb.max(dim=-1, keepdim=True).values  # [B, 1]
    max_min = mn_per_sb_pos.max(dim=-1, keepdim=True).values  # [B, 1]
    inv_scale = torch.where(
        max_scale > 0, 63.0 / max_scale, torch.zeros_like(max_scale)
    )
    inv_min = torch.where(
        max_min > 0, 63.0 / max_min, torch.zeros_like(max_min)
    )

    sc_q = (sc_per_sb * inv_scale).round().clamp(0, 63).to(torch.int32)  # [B, 8]
    mn_q = (mn_per_sb_pos * inv_min).round().clamp(0, 63).to(torch.int32)

    # Per-sub-block reconstructed scale & offset, used for the inner quant.
    d_super = (max_scale / 63.0).squeeze(-1).to(torch.float16)            # [B]
    dmin_super = (max_min / 63.0).squeeze(-1).to(torch.float16)
    d_f32 = d_super.float().unsqueeze(-1)         # [B, 1]
    dmin_f32 = dmin_super.float().unsqueeze(-1)
    sc_recon = d_f32 * sc_q.float()                # [B, 8]
    mn_recon = dmin_f32 * mn_q.float()             # [B, 8]
    sc_recon_safe = sc_recon.clamp_min(1e-8)

    # quants:  q = round((x + mn_recon) / sc_recon)   clamped [0, 15]
    #   y_recovered = sc_recon * q - mn_recon
    q_int = ((w32 + mn_recon.unsqueeze(-1)) / sc_recon_safe.unsqueeze(-1)).round()
    q_int = q_int.clamp(0, 15).to(torch.int32)     # [B, 8, 32]
    q_flat = q_int.reshape(B, GGUF_QK_K)

    scales_packed = _gguf_pack_scales_mins(sc_q, mn_q)
    qs = _gguf_pack_qs(q_flat)
    return d_super, dmin_super, scales_packed, qs


def gguf_q4_k_dequantize(
    d: torch.Tensor,
    dmin: torch.Tensor,
    scales_packed: torch.Tensor,
    qs: torch.Tensor,
) -> torch.Tensor:
    """Dequantize Q4_K blocks back to fp32.

    Mirrors dequantize_row_q4_K (ggml-quants.c):

        for each sub-block j:
            (sc, m) = get_scale_min_k4(j, scales)
            d1 = d * sc
            m1 = dmin * m
            for l in 0..31:
                y = d1 * q[j*32+l] - m1

    returns: fp32 [B, 256]
    """
    if qs.shape[1] != GGUF_QK_K // 2:
        raise ValueError(f"qs shape {qs.shape} != (*, {GGUF_QK_K // 2})")
    B = qs.shape[0]
    sc_q, mn_q = _gguf_unpack_scales_mins(scales_packed)   # [B, 8] each
    q_flat = _gguf_unpack_qs(qs)                            # [B, 256]
    q = q_flat.reshape(B, 8, 32).float()
    d_f32 = d.float().reshape(B, 1, 1)
    dmin_f32 = dmin.float().reshape(B, 1, 1)
    sc_f32 = sc_q.float().reshape(B, 8, 1)
    mn_f32 = mn_q.float().reshape(B, 8, 1)
    y = d_f32 * sc_f32 * q - dmin_f32 * mn_f32
    return y.reshape(B, GGUF_QK_K)


# ============================================================================
# FP8 E4M3 W8A8
# ============================================================================
#
# Format:
#   - Weights: per-tensor symmetric scale, cast to torch.float8_e4m3fn.
#   - Activations: per-token symmetric scale (one scale per row of [tokens, hidden]),
#     also cast to torch.float8_e4m3fn.
#   - Compute: fp32 accumulator (matmul performed after dequantizing both sides).
#
# torch.float8_e4m3fn carries:
#   sign(1) + exp(4, bias=7) + mantissa(3); dynamic range ~ [-448, 448].
#
# This is the dominant H100/B200 inference format (vLLM / TensorRT-LLM / Triton
# fused kernels). For our purposes we expose a tensor-level round-trip without
# depending on a CUDA backend.
#
# torch.float8_e4m3fn is the IEEE-incompatible "fn" variant — no infinities,
# NaN encoded only as 0x7F/0xFF. The max finite is 448. See
# https://docs.pytorch.org/docs/stable/tensors.html#torch.float8_e4m3fn

FP8_E4M3_MAX = 448.0


def fp8_e4m3_quantize_per_tensor(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric FP8 E4M3 weight quantization.

    Returns (w_fp8, scale_fp32).
    The original is recovered by `scale * w_fp8.to(fp32)`.
    """
    amax = w.abs().max().float().clamp_min(1e-12)
    scale = amax / FP8_E4M3_MAX
    w_scaled = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = w_scaled.to(torch.float8_e4m3fn)
    return w_fp8, scale


def fp8_e4m3_quantize_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token (per-row, last-dim) symmetric FP8 E4M3 activation quantization.

    x: shape [..., hidden]. The scale is per-row over the LAST axis.
    Returns (x_fp8, scales_fp32) where scales has shape x.shape[:-1] + (1,).
    """
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12)
    scale = amax / FP8_E4M3_MAX
    x_scaled = (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale


def fp8_e4m3_dequantize(t_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an FP8 E4M3 tensor back to fp32 using the supplied scale."""
    return t_fp8.to(torch.float32) * scale.to(torch.float32)


def fp8_e4m3_matmul(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """W8A8 matmul with FP32 accumulator.

    x_fp8 : [..., K] FP8 E4M3 activations
    x_scale: per-token scales, shape x_fp8.shape[:-1] + (1,)
    w_fp8 : [K, N] FP8 E4M3 weights
    w_scale: per-tensor scale (scalar)
    """
    x_f32 = x_fp8.to(torch.float32)
    w_f32 = w_fp8.to(torch.float32)
    acc = x_f32 @ w_f32                       # fp32 accumulator
    return acc * x_scale.to(torch.float32) * w_scale.to(torch.float32)


# ============================================================================
# MXFP4  (OCP Microscaling FP4)
# ============================================================================
#
# Source: OCP "Microscaling (MX) Formats" specification v1.0 (Open Compute
# Project, 2023), §5.4 + §5.5; NVIDIA Blackwell PTX guide §14.7.
# Block size = 32 elements. Shared scale is UE8M0 (8-bit unsigned exponent,
# no mantissa, no sign, bias=127) — a pure power-of-two. Each element is
# E2M1 (1 sign + 2 exp + 1 mantissa, bias=1):
#
#   E2M1 representable magnitudes:
#     0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0   (with sign bit)
#     => 4-bit code: sign(1) | exp(2) | mantissa(1)
#
# Dequant: y = 2**(ue8m0_exp - 127) * e2m1_value
# Quant:   choose shared exponent so that abs(max_in_block) / 2**(e-127) lies
#          within the E2M1 max (= 6.0); round each element to the nearest
#          E2M1 code.

# E2M1 magnitudes (positive). Indexed by the 3-bit (exp, mantissa) field.
_MXFP4_E2M1_MAGS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
MXFP4_E2M1_MAX = 6.0
MXFP4_BLOCK = 32


def _mxfp4_encode_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Encode a fp32 tensor (already scaled into the E2M1 range) to 4-bit codes.

    Returns int8 [...] with values in [0, 16).
    """
    sign = (x < 0).to(torch.int32)
    mag = x.abs()
    # nearest-magnitude search over the 8 positive codes.
    mags = _MXFP4_E2M1_MAGS.view(*([1] * mag.dim()), 8)
    diffs = (mag.unsqueeze(-1) - mags).abs()
    code3 = diffs.argmin(dim=-1).to(torch.int32)   # 0..7
    code = (sign << 3) | code3
    return code.to(torch.int8)


def _mxfp4_decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Decode 4-bit E2M1 codes to fp32 magnitudes (signed)."""
    c = codes.to(torch.int32)
    mag = _MXFP4_E2M1_MAGS.to(c.device)[c & 0x7]
    sign = ((c >> 3) & 0x1).to(torch.float32) * -2.0 + 1.0   # 0 -> +1, 1 -> -1
    return mag * sign


def mxfp4_quantize(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 quantize a 1D row.

    w: fp32/fp16 tensor of shape [N], with N % 32 == 0.
    returns:
      codes  : int8 [B, 32]  E2M1 codes (low 4 bits used)
      exps   : uint8 [B]     UE8M0 shared exponent per block
    """
    if w.dim() != 1:
        raise ValueError(f"expected 1D weight, got shape {tuple(w.shape)}")
    if w.shape[0] % MXFP4_BLOCK != 0:
        raise ValueError(f"length {w.shape[0]} not divisible by {MXFP4_BLOCK}")
    B = w.shape[0] // MXFP4_BLOCK
    w32 = w.float().reshape(B, MXFP4_BLOCK)
    amax = w32.abs().amax(dim=-1).clamp_min(1e-30)
    # Choose smallest power-of-two scale that fits amax / 2^(e-127) <= 6
    #   2^(e-127) >= amax / 6
    #   e >= log2(amax / 6) + 127
    needed = torch.log2(amax / MXFP4_E2M1_MAX)
    exps = torch.ceil(needed).to(torch.int32) + 127
    exps = exps.clamp(0, 255).to(torch.uint8)
    scale = torch.pow(2.0, exps.to(torch.float32) - 127.0).unsqueeze(-1)
    w_scaled = w32 / scale
    codes = _mxfp4_encode_e2m1(w_scaled)
    return codes, exps


def mxfp4_dequantize(codes: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 (codes, exps) back to fp32 [N]."""
    B, blk = codes.shape
    if blk != MXFP4_BLOCK:
        raise ValueError(f"codes blocksize {blk} != {MXFP4_BLOCK}")
    e = exps.to(torch.float32) - 127.0
    scale = torch.pow(2.0, e).unsqueeze(-1)  # [B, 1]
    vals = _mxfp4_decode_e2m1(codes)         # [B, 32]
    return (vals * scale).reshape(B * MXFP4_BLOCK)


# ============================================================================
# Codebook-quantization stubs  (axis A24 — reserved for M3)
# ============================================================================
#
# The QuantSpec already carries a `codebook` field for codebook-style quants.
# Two production-relevant formats remain UNEXERCISED after B10:
#
#   - IQ2_M  (llama.cpp k-quant family; uses precomputed 2-bit codebooks
#             with per-sub-block scales).
#             Reference: ggml/src/ggml-quants.c::dequantize_row_iq2_m
#                        ggml-common.h::block_iq2_m
#                        https://github.com/ggml-org/llama.cpp
#
#   - AQLM   (Additive Quantization for Language Models, ICML 2024;
#             multi-codebook additive product quantization).
#             Reference: https://github.com/Vahe1994/AQLM
#                        modeling_aqlm.py / inference_kernels/cuda_kernel.cu
#
# These are deferred to a follow-up batch (M3). The skeleton below makes the
# intent explicit and ensures any caller that asks for IQ2_M/AQLM fails fast.


def iq2_m_dequantize(*_args, **_kwargs):
    """IQ2_M codebook dequant — reserved for M3 (axis A24)."""
    raise NotImplementedError(
        "IQ2_M codebook dequant is deferred to M3. Reference: "
        "ggml/src/ggml-quants.c::dequantize_row_iq2_m + ggml-common.h block_iq2_m."
    )


def aqlm_dequantize(*_args, **_kwargs):
    """AQLM additive-codebook dequant — reserved for M3 (axis A24)."""
    raise NotImplementedError(
        "AQLM additive-codebook dequant is deferred to M3. Reference: "
        "https://github.com/Vahe1994/AQLM — modeling_aqlm.py + inference_kernels/."
    )
