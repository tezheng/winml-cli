"""RMSNorm ONNX emit.

Uses the default-domain ``SimplifiedLayerNormalization`` op (since_version=1).
Despite the operator-doc page placing it under com.microsoft, ORT 1.26 actually
registers it in the empty domain — placing it under ``com.microsoft`` triggers
"is not a registered function/op". The op's semantics match RMSNorm exactly:
``y = x * rsqrt(mean(x*x, axis) + epsilon) * scale``.
"""
from __future__ import annotations

from onnx import NodeProto, helper

from components import NormStandardSpec


def build_rms_norm_nodes(
    spec: NormStandardSpec,
    x_name: str,
    weight_name: str,
    out_name: str,
) -> list[NodeProto]:
    """Emit a single SimplifiedLayerNormalization node.

    Args:
        spec:        NormStandardSpec carrying ``eps`` and ``axis``.
        x_name:      Tensor name of the activation input.
        weight_name: Tensor name of the scale vector.
        out_name:    Tensor name to bind the normalized output to.

    Returns:
        A single-element list containing the emitted node.
    """
    node = helper.make_node(
        "SimplifiedLayerNormalization",
        inputs=[x_name, weight_name],
        outputs=[out_name],
        name=f"{out_name}/SimplifiedLayerNormalization",
        epsilon=float(spec.eps),
        axis=int(spec.axis),
    )
    return [node]


__all__ = ["build_rms_norm_nodes"]
