"""Feed-forward / channel-mixer building blocks.

M1 supports SwiGLU only — the dominant SLM channel mixer (Llama, Qwen, Mistral).
GeGLU (Gemma), MoE, fused gate_up (Phi-3) land in later milestones.
"""
from __future__ import annotations

import torch
from torch import nn

from api import ops, specs, types


class FeedForward(nn.Module):
    def __init__(self, spec: specs.FFNSpec, hidden_size: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.gate_kind != types.GateKind.SWIGLU:
            raise NotImplementedError(f"M1: SWIGLU only, got {spec.gate_kind}")
        if spec.activation != types.Activation.SILU:
            raise NotImplementedError(f"M1: SILU only, got {spec.activation}")
        if spec.fused_gate_up:
            raise NotImplementedError("M1: split gate/up only")
        self.spec = spec
        self.hidden_size = hidden_size
        I = spec.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, I, bias=spec.gate_bias, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, I, bias=spec.up_bias, dtype=dtype)
        self.down_proj = nn.Linear(I, hidden_size, bias=spec.down_bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(ops.mul(ops.silu(gate), up))
