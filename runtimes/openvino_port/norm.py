"""RMSNorm as an OpenVINO opset16 sub-graph.

Decomposed form (matches torch_port.rms_norm_forward and HF LlamaRMSNorm):

    x32      = cast(x, f32)
    variance = reduce_mean(x32 * x32, axis, keep_dims=True)
    rsqrt    = (variance + eps) ** -0.5
    y        = (x32 * rsqrt) * weight_f32
    return  cast(y, x.dtype)

The pattern that OpenVINO's ``RMSFusion`` pass recognises is the canonical
"square -> reduce_mean -> add eps -> sqrt -> divide -> multiply" or its
equivalent ``power(-0.5)`` form. We use ``power(-0.5)`` because it composes
into the same op the fusion pass folds into the dedicated GPU ``RMS`` kernel.
"""
from __future__ import annotations

import openvino as ov
from openvino import Type
from openvino import opset16 as opset

from components import NormStandardSpec


def build_rms_norm(
    spec: NormStandardSpec,
    x: ov.Output,
    weight: ov.Output,
) -> ov.Output:
    """Build the decomposed RMSNorm graph and return the output Node.

    Args:
        spec:    NormStandardSpec carrying ``eps`` and reduction ``axis``.
        x:       Input activation. Any float dtype; computation runs in fp32.
        weight:  Per-channel scale. Same outer dtype as ``x``.

    Returns:
        Output node with the same dtype as ``x``.
    """
    # Run all math in fp32 (per spec.accumulator_dtype = F32) and cast back.
    out_type = x.get_element_type()
    x32 = opset.convert(x, Type.f32)
    w32 = opset.convert(weight, Type.f32)

    sq = opset.multiply(x32, x32)
    axis = opset.constant([spec.axis], Type.i32)
    mean = opset.reduce_mean(sq, axis, keep_dims=True)
    eps = opset.constant(spec.eps, Type.f32)
    var_eps = opset.add(mean, eps)
    rsqrt = opset.power(var_eps, opset.constant(-0.5, Type.f32))
    normed = opset.multiply(x32, rsqrt)
    scaled = opset.multiply(normed, w32)
    return opset.convert(scaled, out_type)


__all__ = ["build_rms_norm"]
