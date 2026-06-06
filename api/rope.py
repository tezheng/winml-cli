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
    """Precomputed cos/sin RoPE cache.

    Geometry note (B0.6): For `partial_rotary_factor < 1` we use the HF Gemma 4
    "proportional" rope_type — `inv_freq` is built with the first
    ``rope_angles = int(pr * head_dim / 2)`` slots holding REAL frequencies
    computed at base ** (arange(0, 2 * rope_angles, 2) / head_dim) — note the
    denominator is the FULL head_dim, NOT head_dim_rot — and the remaining
    ``head_dim/2 - rope_angles`` slots zeroed. cos/sin are then built at the
    full head_dim and the regular full-rotation `ops.rope_apply` path applies
    them across all head_dim channels. Channels whose inv_freq is 0 have
    cos=1, sin=0 — geometrically the identity, but they STILL participate in
    `rotate_half`'s ``i ↔ i + Dh/2`` pairing, which is the geometric difference
    vs the simpler "rotate a prefix" path.

    Source: `transformers/modeling_rope_utils.py::_compute_proportional_rope_parameters`
    (inv_freq layout with zero padding) and `modeling_gemma4.py:787-806`
    (apply_rotary_pos_emb uses rotate_half on the full head_dim).
    """

    def __init__(self, spec: specs.RoPESpec, head_dim: int, max_seq: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.basis != types.RoPEBasis.SPLIT_HALF:
            raise NotImplementedError("M1 supports SPLIT_HALF basis only")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.spec = spec
        self.head_dim = head_dim
        # rope_angles == number of real (non-zero) frequencies. HF formula:
        #   rope_angles = int(partial_rotary_factor * head_dim // 2)
        rope_angles = int(spec.partial_rotary_factor * head_dim // 2)
        self.rope_angles = rope_angles
        # For backward-compat we also expose head_dim_rot = 2 * rope_angles
        # (the size that a "rotate-only-prefix" implementation would use).
        self.head_dim_rot = 2 * rope_angles
        self.max_seq = max_seq

        # Real frequencies at the first `rope_angles` slots — denominator is
        # full head_dim per HF proportional formula.
        inv_freq_rotated = 1.0 / (
            spec.base_theta ** (torch.arange(0, 2 * rope_angles, 2).float() / head_dim)
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                [inv_freq_rotated, torch.zeros(nope_angles)],
                dim=0,
            )
        else:
            inv_freq = inv_freq_rotated

        if spec.scaling == types.RoPEScaling.LLAMA3:
            if spec.llama3_extra is None:
                raise ValueError("LLAMA3 scaling requires llama3_extra")
            # Llama-3 scaling only applies to the real frequencies; zeros stay zeros.
            if nope_angles > 0:
                rot_scaled = _llama3_scale_inv_freq(inv_freq[:rope_angles], spec.llama3_extra)
                inv_freq = torch.cat([rot_scaled, inv_freq[rope_angles:]], dim=0)
            else:
                inv_freq = _llama3_scale_inv_freq(inv_freq, spec.llama3_extra)
        elif spec.scaling != types.RoPEScaling.NONE:
            raise NotImplementedError(
                f"M1 supports NONE and LLAMA3 scaling only, got {spec.scaling}"
            )

        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)            # [max_seq, head_dim/2]
        # cos/sin at FULL head_dim. Channels with zero inv_freq → cos=1, sin=0.
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
        # Always use the full-rotation path. When partial_rotary_factor < 1,
        # cos/sin have ones/zeros in the trailing channels — those positions
        # multiply-through identity but still participate in rotate_half's
        # i↔i+Dh/2 pairing per HF proportional RoPE.
        return ops.rope_apply(q, k, cos, sin, basis="split_half")
