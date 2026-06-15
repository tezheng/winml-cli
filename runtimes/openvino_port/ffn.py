"""SwiGLU MLP as an OpenVINO opset16 sub-graph.

Mirror of ``runtimes.torch_port.ffn.swiglu_forward``:

    gate = silu(x @ W_gate.T)
    up   = x @ W_up.T
    out  = (gate * up) @ W_down.T

OpenVINO's GLU pattern matcher may fold the silu + multiply pair.
"""
from __future__ import annotations

import openvino as ov
from openvino import opset16 as opset

from components import SwiGLUSpec


def _matmul_xWT(x: ov.Output, w: ov.Output) -> ov.Output:
    return opset.matmul(x, w, transpose_a=False, transpose_b=True)


def _silu(x: ov.Output) -> ov.Output:
    """Standard SiLU = x * sigmoid(x). OV opset has a dedicated swish op
    but ``Sigmoid + Multiply`` matches torch's F.silu reference exactly and
    lets the GLU fusion recognise the canonical pattern."""
    return opset.multiply(x, opset.sigmoid(x))


def build_swiglu(
    spec: SwiGLUSpec,
    x: ov.Output,
    w_gate: ov.Output,
    w_up: ov.Output,
    w_down: ov.Output,
) -> ov.Output:
    if spec.fused_gate_up:
        raise NotImplementedError("SwiGLU fused_gate_up not supported in M0.")
    g = _matmul_xWT(x, w_gate)
    u = _matmul_xWT(x, w_up)
    gated = opset.multiply(_silu(g), u)
    return _matmul_xWT(gated, w_down)


__all__ = ["build_swiglu"]
