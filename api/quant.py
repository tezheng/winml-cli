"""Quantization: AWQ W4A16 grouped INT4 round-trip.

Reference: AWQ uses the nibble permutation [0,2,4,6,1,3,5,7] when packing
8 INT4 values into one INT32. We replicate the storage layout so we can
load real AWQ checkpoints in Task 19.

Layout (for W of shape [K, N], group_size G along K):
    qweight  : int32[K // 8, N]              -- 8 nibbles packed per int32
    qzeros   : int32[K // G, N // 8]         -- 8 zero-nibbles packed per int32
    scales   : float16[K // G, N]            -- per-group scale
"""
from __future__ import annotations
from typing import Tuple

import torch

from api import specs, types


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
