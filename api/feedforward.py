"""Feed-forward / channel-mixer building blocks.

M1 supports SwiGLU only — the dominant SLM channel mixer (Llama, Qwen, Mistral).
B0.5 adds GeGLU (Gemma) using the gelu_pytorch_tanh activation. MoE, fused
gate_up (Phi-3) land in later milestones.
"""
from __future__ import annotations

import torch
from torch import nn

from api import ops, specs, types


class FeedForward(nn.Module):
    def __init__(self, spec: specs.FFNSpec, hidden_size: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.gate_kind not in (types.GateKind.SWIGLU, types.GateKind.GEGLU):
            raise NotImplementedError(
                f"B0.5: SWIGLU and GEGLU only, got {spec.gate_kind}"
            )
        if spec.gate_kind == types.GateKind.SWIGLU and spec.activation != types.Activation.SILU:
            raise NotImplementedError(
                f"B0.5: SWIGLU requires SILU activation, got {spec.activation}"
            )
        if spec.gate_kind == types.GateKind.GEGLU and spec.activation != types.Activation.GELU:
            raise NotImplementedError(
                f"B0.5: GEGLU requires GELU activation, got {spec.activation}"
            )
        self.spec = spec
        self.hidden_size = hidden_size
        I = spec.intermediate_size
        if spec.fused_gate_up:
            # B2a: Phi-3 fused gate/up — one big projection of size 2*I,
            # chunked at forward time. Source: modeling_phi3.py:54 (
            # `gate_up_proj = Linear(hidden_size, 2 * intermediate_size, bias=False)`)
            # and 58-62 (`up_states.chunk(2, dim=-1)` → `up_states * act(gate)`).
            if spec.gate_bias != spec.up_bias:
                raise NotImplementedError(
                    "fused_gate_up requires gate_bias == up_bias (Phi-3: both False)"
                )
            self.gate_up_proj = nn.Linear(hidden_size, 2 * I, bias=spec.gate_bias, dtype=dtype)
            self.gate_proj = None
            self.up_proj = None
        else:
            self.gate_up_proj = None
            self.gate_proj = nn.Linear(hidden_size, I, bias=spec.gate_bias, dtype=dtype)
            self.up_proj = nn.Linear(hidden_size, I, bias=spec.up_bias, dtype=dtype)
        self.down_proj = nn.Linear(I, hidden_size, bias=spec.down_bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.fused_gate_up:
            # Phi-3 order: chunk(2, dim=-1) → (gate, up) — verify modeling_phi3.py:61.
            up_states = self.gate_up_proj(x)
            gate, up = up_states.chunk(2, dim=-1)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        if self.spec.gate_kind == types.GateKind.SWIGLU:
            act = ops.silu(gate)
        else:  # GEGLU — use gelu_pytorch_tanh per Gemma's signature
            act = ops.gelu_pytorch_tanh(gate)
        return self.down_proj(ops.mul(act, up))
