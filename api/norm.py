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


class LayerNorm(nn.Module):
    """Standard LayerNorm with optional bias.

    Source: `transformers/models/mpt/modeling_mpt.py:163-165` (LN with bias=None)
    and `transformers/models/falcon/modeling_falcon.py:574-578` (LN with bias).

    Behaviour delegates to `api.ops.layer_norm` (fp32 compute, dtype preserved).
    """

    def __init__(self, spec: specs.NormSpec, hidden_size: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.kind != types.NormKind.LAYER:
            raise ValueError(f"LayerNorm requires kind=LAYER, got {spec.kind}")
        self.spec = spec
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        if spec.has_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            # F.layer_norm with bias=None: avoid creating a zero bias buffer at
            # each call. ops.layer_norm requires a bias, so we pass the
            # equivalent zeros (no allocation hit at trace time).
            return ops.layer_norm(
                x, self.weight,
                torch.zeros(self.hidden_size, dtype=self.weight.dtype, device=self.weight.device),
                self.spec.eps,
            )
        return ops.layer_norm(x, self.weight, self.bias, self.spec.eps)


def build_norm(spec: specs.NormSpec, hidden_size: int,
               dtype: torch.dtype = torch.float32) -> nn.Module:
    """Dispatch by NormKind. Returns RMSNorm or LayerNorm.

    Source: api/specs.NormSpec field `kind`.
    """
    if spec.kind == types.NormKind.RMS:
        return RMSNorm(spec, hidden_size, dtype=dtype)
    if spec.kind == types.NormKind.LAYER:
        return LayerNorm(spec, hidden_size, dtype=dtype)
    raise NotImplementedError(f"unknown NormKind: {spec.kind}")


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
