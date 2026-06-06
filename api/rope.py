"""RoPE block — precomputed cos/sin tables consuming RoPESpec.

M1 implements SPLIT_HALF basis with optional Llama-3 smooth scaling.
Other RoPE variants (PI, NTK, YaRN, LongRoPE) land in later milestones.
"""
from __future__ import annotations
import math

import torch
from torch import nn

from api import ops, specs, types


def _llama3_scale_inv_freq(
    inv_freq: torch.Tensor,
    extra: specs.Llama3RoPEParams,
) -> torch.Tensor:
    """Apply Llama-3 smooth RoPE scaling.

    From Meta's reference: inv_freq is scaled per-frequency by a smooth function
    of wavelength.
    """
    low_freq_wavelen = extra.original_context_length / extra.low_freq_factor
    high_freq_wavelen = extra.original_context_length / extra.high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_scaled = torch.where(
        wavelen > low_freq_wavelen,
        inv_freq / extra.factor,
        inv_freq,
    )
    smooth_factor = (extra.original_context_length / wavelen - extra.low_freq_factor) / (
        extra.high_freq_factor - extra.low_freq_factor
    )
    smoothed = (1 - smooth_factor) * inv_freq / extra.factor + smooth_factor * inv_freq
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_scaled)


class RoPE(nn.Module):
    def __init__(self, spec: specs.RoPESpec, head_dim: int, max_seq: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.basis != types.RoPEBasis.SPLIT_HALF:
            raise NotImplementedError("M1 supports SPLIT_HALF basis only")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.spec = spec
        self.head_dim = head_dim
        self.head_dim_rot = int(head_dim * spec.partial_rotary_factor)
        if self.head_dim_rot % 2 != 0:
            raise ValueError(f"rotated head_dim ({self.head_dim_rot}) must be even")
        self.max_seq = max_seq

        inv_freq = 1.0 / (
            spec.base_theta ** (torch.arange(0, self.head_dim_rot, 2).float() / self.head_dim_rot)
        )
        if spec.scaling == types.RoPEScaling.LLAMA3:
            if spec.llama3_extra is None:
                raise ValueError("LLAMA3 scaling requires llama3_extra")
            inv_freq = _llama3_scale_inv_freq(inv_freq, spec.llama3_extra)
        elif spec.scaling != types.RoPEScaling.NONE:
            raise NotImplementedError(
                f"M1 supports NONE and LLAMA3 scaling only, got {spec.scaling}"
            )

        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() != 1:
            raise NotImplementedError("M1 supports 1D position_ids only")
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        if self.spec.partial_rotary_factor == 1.0:
            return ops.rope_apply(q, k, cos, sin, basis="split_half")
        return ops.rope_apply_partial(
            q, k, cos, sin,
            partial_rotary_factor=self.spec.partial_rotary_factor,
            basis="split_half",
        )
