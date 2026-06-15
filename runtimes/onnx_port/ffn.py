"""SwiGLU FFN ONNX emit — standard ops.

There is no fused SwiGLU in the contrib domain (``com.microsoft.QuickGelu``
covers a GeLU flavor, not SiLU-gate). We use plain MatMul/Sigmoid/Mul nodes.

Matches torch_port.swiglu_forward:
    y = (silu(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T
"""
from __future__ import annotations

from onnx import NodeProto, helper

from components import SwiGLUSpec


def build_swiglu_nodes(
    spec: SwiGLUSpec,
    x_name: str,
    w_gate_name: str,
    w_up_name: str,
    w_down_name: str,
    out_name: str,
    *,
    name_scope: str,
) -> list[NodeProto]:
    """Emit Transpose+MatMul+Sigmoid+Mul nodes implementing SwiGLU.

    Args:
        spec:         SwiGLUSpec; ``fused_gate_up`` must be False for M0.
        x_name:       Input activation ``[B, S, D_model]``.
        w_gate_name:  Gate weight ``[I, D_model]`` (torch convention).
        w_up_name:    Up weight ``[I, D_model]``.
        w_down_name:  Down weight ``[D_model, I]``.
        out_name:     Output tensor name.
        name_scope:   Prefix for intermediate names.

    Returns:
        Ordered node list.
    """
    if spec.fused_gate_up:
        raise NotImplementedError("SwiGLU fused_gate_up not supported in M0.")
    p = name_scope
    return [
        helper.make_node("Transpose", [w_gate_name], [f"{p}/w_gate_T"], perm=[1, 0]),
        helper.make_node("Transpose", [w_up_name],   [f"{p}/w_up_T"],   perm=[1, 0]),
        helper.make_node("Transpose", [w_down_name], [f"{p}/w_down_T"], perm=[1, 0]),

        helper.make_node("MatMul", [x_name, f"{p}/w_gate_T"], [f"{p}/gate_lin"],
                         name=f"{p}/gate_matmul"),
        helper.make_node("MatMul", [x_name, f"{p}/w_up_T"],   [f"{p}/up_lin"],
                         name=f"{p}/up_matmul"),
        # SiLU = x * sigmoid(x)
        helper.make_node("Sigmoid", [f"{p}/gate_lin"], [f"{p}/gate_sigmoid"],
                         name=f"{p}/gate_sigmoid"),
        helper.make_node("Mul", [f"{p}/gate_lin", f"{p}/gate_sigmoid"], [f"{p}/silu_gate"],
                         name=f"{p}/silu"),
        helper.make_node("Mul", [f"{p}/silu_gate", f"{p}/up_lin"], [f"{p}/gated"],
                         name=f"{p}/gated_mul"),
        helper.make_node("MatMul", [f"{p}/gated", f"{p}/w_down_T"], [out_name],
                         name=f"{p}/down_matmul"),
    ]


__all__ = ["build_swiglu_nodes"]
