"""N-way elementwise add as an OpenVINO opset16 sub-graph."""
from __future__ import annotations

import openvino as ov
from openvino import opset16 as opset


def build_add(*nodes: ov.Output) -> ov.Output:
    if not nodes:
        raise ValueError("build_add requires at least one input")
    out = nodes[0]
    for n in nodes[1:]:
        out = opset.add(out, n)
    return out


__all__ = ["build_add"]
