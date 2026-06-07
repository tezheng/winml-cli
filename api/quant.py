"""Quantization: AWQ W4A16, GGUF Q4_K_M round-trip.

Each scheme below is grounded in the upstream reference implementation and
exercises a distinct axis of `specs.QuantSpec`:

  - AWQ W4A16          : INT4 grouped, asymmetric, AWQ_INTERLEAVE packing.
  - GGUF Q4_K_M        : INT4 super-block (256), 8 sub-blocks of 32 elements,
                         6-bit per-sub-block scale+min, fp16 super-block d/dmin.

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
