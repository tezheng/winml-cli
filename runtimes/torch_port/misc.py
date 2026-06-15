"""Graph-primitive forwards: AddSpec.

These are not registry components — they're DAG primitives wired by BlockGraph.
"""
from __future__ import annotations

import torch


def add_forward(*tensors: torch.Tensor) -> torch.Tensor:
    """N-way elementwise add (used for residual paths).

    Args:
        *tensors: One or more tensors of identical shape.

    Returns:
        The elementwise sum.
    """
    if not tensors:
        raise ValueError("add_forward requires at least one tensor.")
    out = tensors[0]
    for t in tensors[1:]:
        out = out + t
    return out


__all__ = ["add_forward"]
