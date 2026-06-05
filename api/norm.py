"""Norm building blocks consuming NormSpec."""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from api import ops, specs, types


class RMSNorm(nn.Module):
    """Wraps api.ops.rms_norm. Weight initialized to ones."""

    def __init__(self, spec: specs.NormSpec, hidden_size: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.kind != types.NormKind.RMS:
            raise ValueError(f"RMSNorm requires kind=RMS, got {spec.kind}")
        self.spec = spec
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = ("one_plus_w" if self.spec.weight_mode == types.NormWeightMode.ONE_PLUS_W
                else "standard_w")
        return ops.rms_norm(x, self.weight, self.spec.eps, mode=mode)


class QKNorm(nn.Module):
    """Per-head QK normalization.

    Two shape modes:
    - PER_HEAD_DH: weight shape [head_dim], applied independently to each head's Dh
      (Qwen3, Gemma 3)
    - FULL_HDH: weight shape [n_heads * head_dim], applied to the flattened
      head-dim slice (OLMo 2)
    """

    def __init__(self, spec: specs.NormSpec, head_dim: int,
                 shape: types.QKNormShape, n_heads: Optional[int] = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.kind != types.NormKind.RMS:
            raise ValueError("QKNorm only supports RMS")
        self.spec = spec
        self.shape = shape
        self.head_dim = head_dim
        self.n_heads = n_heads
        if shape == types.QKNormShape.PER_HEAD_DH:
            weight_size = head_dim
        elif shape == types.QKNormShape.FULL_HDH:
            if n_heads is None:
                raise ValueError("FULL_HDH requires n_heads")
            weight_size = n_heads * head_dim
        else:
            raise ValueError(f"unsupported shape: {shape}")
        self.weight = nn.Parameter(torch.ones(weight_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = ("one_plus_w" if self.spec.weight_mode == types.NormWeightMode.ONE_PLUS_W
                else "standard_w")
        if self.shape == types.QKNormShape.PER_HEAD_DH:
            return ops.rms_norm(x, self.weight, self.spec.eps, mode=mode)
        B, S, H, Dh = x.shape
        x_flat = x.reshape(B, S, H * Dh)
        out = ops.rms_norm(x_flat, self.weight, self.spec.eps, mode=mode)
        return out.reshape(B, S, H, Dh)
