"""RMSNorm forward (standard variant) in PyTorch.

Source: transformers/models/llama/modeling_llama.py:LlamaRMSNorm.forward
"""
from __future__ import annotations

import torch

from components import NormStandardSpec


def rms_norm_forward(
    spec: NormStandardSpec,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Standard RMSNorm: ``y = x * rsqrt(mean(x**2, axis) + eps) * weight``.

    The variance reduction runs in fp32 (per ``spec.accumulator_dtype``) and the
    output is cast back to the input dtype. M0 only supports F32 accumulator —
    the only value any current spec uses.

    Args:
        spec:    NormStandardSpec carrying ``eps`` and reduction ``axis``.
        x:       Activation tensor, shape ``[..., D]``.
        weight:  Per-channel scale, shape ``[D]``.

    Returns:
        Tensor with the same shape and dtype as ``x``.
    """
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(spec.axis, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + spec.eps)
    y = weight.to(torch.float32) * x_normed
    return y.to(input_dtype)


__all__ = ["rms_norm_forward"]
