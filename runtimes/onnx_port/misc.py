"""Add/Mul ONNX emit — graph-primitive forwards.

Standard-domain ``Add`` is binary; for N-way add we chain pairwise. Matches
torch_port.add_forward's left-to-right reduction.
"""
from __future__ import annotations

from onnx import NodeProto, helper


def build_add_node(
    input_names: list[str],
    out_name: str,
    *,
    name_scope: str,
) -> list[NodeProto]:
    """Emit a chain of binary Add nodes summing ``input_names`` in order.

    For 2 inputs that's a single Add. For 3+ inputs (parallel-residual blocks)
    we chain: ``((a + b) + c)``.

    Args:
        input_names: 1+ source tensor names.
        out_name:    Final sum tensor name.
        name_scope:  Prefix for intermediate names when N > 2.

    Returns:
        Node list.
    """
    if not input_names:
        raise ValueError("build_add_node requires at least one input.")
    if len(input_names) == 1:
        # Pure pass-through. Emit Identity to keep the tensor name in env.
        return [helper.make_node("Identity", [input_names[0]], [out_name],
                                 name=f"{name_scope}/Identity")]
    if len(input_names) == 2:
        return [helper.make_node("Add", list(input_names), [out_name],
                                 name=f"{name_scope}/Add")]
    # 3+ way: chain.
    p = name_scope
    nodes: list[NodeProto] = []
    cur = input_names[0]
    for i, src in enumerate(input_names[1:], start=1):
        is_last = i == len(input_names) - 1
        nxt = out_name if is_last else f"{p}/partial_{i}"
        nodes.append(helper.make_node("Add", [cur, src], [nxt], name=f"{p}/Add_{i}"))
        cur = nxt
    return nodes


__all__ = ["build_add_node"]
