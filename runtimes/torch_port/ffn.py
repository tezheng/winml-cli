"""SwiGLU MLP forward in PyTorch.

Source: transformers/models/llama/modeling_llama.py:LlamaMLP.forward
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from components import SwiGLUSpec


def swiglu_forward(
    spec: SwiGLUSpec,
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU MLP: ``y = w_down(silu(w_gate(x)) * w_up(x))``.

    M0 only supports the non-fused form (``spec.fused_gate_up == False``).

    Args:
        spec:    SwiGLUSpec; must have ``fused_gate_up=False`` for M0.
        x:       Hidden state shaped ``[B, S, D_model]``.
        w_gate:  Gate weight shaped ``[I, D_model]`` where ``I = intermediate_size``.
        w_up:    Up weight shaped ``[I, D_model]``.
        w_down:  Down weight shaped ``[D_model, I]``.

    Returns:
        Tensor shaped ``[B, S, D_model]``.
    """
    if spec.fused_gate_up:
        raise NotImplementedError("SwiGLU fused_gate_up not supported in M0.")
    gate = F.silu(F.linear(x, w_gate))
    up = F.linear(x, w_up)
    return F.linear(gate * up, w_down)


__all__ = ["swiglu_forward"]
