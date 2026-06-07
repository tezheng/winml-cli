"""RoPE block — precomputed cos/sin tables consuming RoPESpec.

M1 implements SPLIT_HALF basis with optional Llama-3 smooth scaling. B2a adds
LongRoPE (Phi-3 / Phi-4) — TWO inv_freq tables (short_factor and long_factor)
with per-position dispatch at the boundary
``original_max_position_embeddings``.
"""
from __future__ import annotations
import math

import torch
from torch import nn

from api import ops, specs, types


def _yarn_inv_freq_and_scale(
    base: float,
    dim_rot: int,
    extra: specs.YarnRoPEParams,
) -> tuple[torch.Tensor, float]:
    """YARN scaling (DeepSeek-V2/V3 family).

    Mirrors transformers/modeling_rope_utils.py::_compute_yarn_parameters
    (L327-459). Returns (inv_freq, attention_factor).

    Note: when `extra.factor <= 1` the helper get_mscale returns 1.0 and we
    short-circuit the standard NTK-like extrapolation path back to the
    "no-scale" branch.
    """
    import math
    factor = extra.factor
    original = extra.original_max_position_embeddings
    beta_fast = extra.beta_fast
    beta_slow = extra.beta_slow
    truncate = extra.truncate

    def get_mscale(scale: float, mscale: float = 1.0) -> float:
        if scale <= 1.0:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    if extra.mscale > 0 and extra.mscale_all_dim > 0:
        attention_factor = float(
            get_mscale(factor, extra.mscale) / get_mscale(factor, extra.mscale_all_dim)
        )
    else:
        attention_factor = get_mscale(factor)

    def find_correction_dim(num_rotations: float, dim: int, b: float, max_pos: int) -> float:
        return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(b))

    def find_correction_range(low_rot: float, high_rot: float, dim: int, b: float,
                              max_pos: int, trunc: bool) -> tuple[float, float]:
        low = find_correction_dim(low_rot, dim, b, max_pos)
        high = find_correction_dim(high_rot, dim, b, max_pos)
        if trunc:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(low: float, high: float, half_dim: int) -> torch.Tensor:
        if low == high:
            high = high + 0.001
        linear_func = (torch.arange(half_dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear_func, 0.0, 1.0)

    pos_freqs = base ** (torch.arange(0, dim_rot, 2, dtype=torch.float32) / dim_rot)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = find_correction_range(beta_fast, beta_slow, dim_rot, base, original, truncate)
    ramp = linear_ramp_factor(low, high, dim_rot // 2)
    inv_freq_extrapolation_factor = 1.0 - ramp
    inv_freq = (
        inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


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
        if spec.basis not in (types.RoPEBasis.SPLIT_HALF, types.RoPEBasis.INTERLEAVED):
            raise NotImplementedError(
                f"SPLIT_HALF and INTERLEAVED basis only, got {spec.basis}"
            )
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        # B4: INTERLEAVED basis (Llama 4) is incompatible with the partial-rotary
        # and LongRoPE codepaths in the current implementation. Source:
        # modeling_llama4.py:200-241 — Llama 4's RoPE uses full head_dim,
        # default scaling, and complex-multiply pairing.
        if spec.basis == types.RoPEBasis.INTERLEAVED:
            if spec.partial_rotary_factor != 1.0:
                raise NotImplementedError(
                    "INTERLEAVED basis with partial_rotary_factor != 1.0 not supported"
                )
            # B5: YARN-scaled INTERLEAVED is needed for DeepSeek-V2-Lite
            # production RoPE. Source: modeling_deepseek_v2.py:271-284
            # (complex-multiply pairing on (re, im) channel pairs).
            if spec.scaling not in (types.RoPEScaling.NONE,
                                    types.RoPEScaling.LLAMA3,
                                    types.RoPEScaling.YARN):
                raise NotImplementedError(
                    f"INTERLEAVED basis supports NONE / LLAMA3 / YARN scaling "
                    f"only, got {spec.scaling}"
                )
        if spec.partial_rotary_kind not in ("prefix", "proportional"):
            raise ValueError(
                f"partial_rotary_kind must be 'prefix' or 'proportional', "
                f"got {spec.partial_rotary_kind}"
            )
        self.spec = spec
        self.head_dim = head_dim
        self._yarn_attention_factor: float | None = None
        # rope_angles == number of real (non-zero) frequencies. HF formula:
        #   rope_angles = int(partial_rotary_factor * head_dim // 2)
        rope_angles = int(spec.partial_rotary_factor * head_dim // 2)
        self.rope_angles = rope_angles
        self.head_dim_rot = 2 * rope_angles
        self.max_seq = max_seq
        self._partial_kind = spec.partial_rotary_kind

        # Inv-freq denominator depends on the partial-rotary semantic:
        # - "prefix" (Phi-3): inv_freq_shape = arange(0, dim_rot, 2) / dim_rot
        #   — denominator is dim_rot (the rotated subset only).
        # - "proportional" (Gemma 4): inv_freq_shape = arange(0, dim_rot, 2)
        #   / head_dim — denominator is FULL head_dim, with zero padding in
        #   the trailing channels.
        # Source: modeling_phi3.py:113-114 vs modeling_gemma4.py (proportional
        # path uses head_dim denominator).
        if self._partial_kind == "prefix":
            denom = max(self.head_dim_rot, 2)
        else:  # proportional
            denom = head_dim
        inv_freq_rotated = 1.0 / (
            spec.base_theta ** (torch.arange(0, 2 * rope_angles, 2).float() / denom)
        )
        nope_angles = head_dim // 2 - rope_angles
        if self._partial_kind == "proportional" and nope_angles > 0:
            # Gemma 4: pad inv_freq with zeros to head_dim/2 so cos/sin live
            # at full head_dim.
            inv_freq = torch.cat(
                [inv_freq_rotated, torch.zeros(nope_angles)],
                dim=0,
            )
        else:
            # Phi-3 prefix: inv_freq stays at rope_angles entries; cos/sin live
            # at head_dim_rot (= 2*rope_angles), and apply uses rope_apply_partial.
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
        elif spec.scaling == types.RoPEScaling.LONGROPE:
            # B2a: LongRoPE — TWO inv_freq tables, dispatched per position. The
            # HF implementation uses `1 / (ext_factor * base ** (2i / dim))`
            # where dim = head_dim * partial_rotary_factor and ext_factor is
            # either short_factor or long_factor (each of length dim/2).
            # cos/sin are scaled by attention_factor.
            # Source: modeling_rope_utils.py:_compute_longrope_parameters
            # lines 462-547; dynamic_rope_update.longrope_frequency_update L47-80.
            if spec.longrope_extra is None:
                raise ValueError("LONGROPE scaling requires longrope_extra")
            lre = spec.longrope_extra
            if len(lre.short_factor) != rope_angles:
                raise ValueError(
                    f"LongRoPE short_factor length {len(lre.short_factor)} "
                    f"!= rope_angles {rope_angles} (= dim/2)"
                )
            if len(lre.long_factor) != rope_angles:
                raise ValueError(
                    f"LongRoPE long_factor length {len(lre.long_factor)} "
                    f"!= rope_angles {rope_angles}"
                )
            short = torch.tensor(lre.short_factor, dtype=torch.float32)
            long_ = torch.tensor(lre.long_factor, dtype=torch.float32)
            # LongRoPE always uses dim_rot denominator (per modeling_rope_utils
            # _compute_longrope_parameters line 544 — `inv_freq_shape =
            # arange(0, dim, 2) / dim` where `dim = head_dim * partial_rotary_factor
            # = head_dim_rot`).
            dim_rot = max(2 * rope_angles, 2)
            inv_freq_shape = torch.arange(0, 2 * rope_angles, 2).float() / dim_rot
            inv_freq_short_rot = 1.0 / (short * spec.base_theta ** inv_freq_shape)
            inv_freq_long_rot = 1.0 / (long_ * spec.base_theta ** inv_freq_shape)
            if self._partial_kind == "proportional" and nope_angles > 0:
                zeros = torch.zeros(nope_angles)
                inv_freq_short = torch.cat([inv_freq_short_rot, zeros], dim=0)
                inv_freq_long = torch.cat([inv_freq_long_rot, zeros], dim=0)
            else:
                # "prefix" mode: keep tables at head_dim_rot only.
                inv_freq_short = inv_freq_short_rot
                inv_freq_long = inv_freq_long_rot
            t = torch.arange(max_seq).float()
            freqs_short = torch.outer(t, inv_freq_short)
            freqs_long = torch.outer(t, inv_freq_long)
            att = lre.attention_factor
            cos_short = (torch.cat([freqs_short.cos(), freqs_short.cos()], dim=-1) * att).to(dtype)
            sin_short = (torch.cat([freqs_short.sin(), freqs_short.sin()], dim=-1) * att).to(dtype)
            cos_long = (torch.cat([freqs_long.cos(), freqs_long.cos()], dim=-1) * att).to(dtype)
            sin_long = (torch.cat([freqs_long.sin(), freqs_long.sin()], dim=-1) * att).to(dtype)
            self._longrope_boundary = lre.original_max_position_embeddings
            self.register_buffer("cos_cached", cos_short, persistent=False)
            self.register_buffer("sin_cached", sin_short, persistent=False)
            self.register_buffer("cos_cached_long", cos_long, persistent=False)
            self.register_buffer("sin_cached_long", sin_long, persistent=False)
            return
        elif spec.scaling == types.RoPEScaling.YARN:
            # B5: YARN scaling — DeepSeek-V2-Lite / V3. Replaces inv_freq with
            # the YARN linear-ramp blend of extrapolation and interpolation,
            # then multiplies cos/sin by attention_factor (the mscale term).
            # Compatible with INTERLEAVED + SPLIT_HALF basis. Compatible with
            # partial_rotary_factor < 1.0 in "prefix" kind only — the
            # YARN dim_rot is head_dim_rot (per HF
            # `dim = head_dim * partial_rotary_factor`).
            # Source: modeling_rope_utils.py:_compute_yarn_parameters L327-459.
            if spec.yarn_extra is None:
                raise ValueError("YARN scaling requires yarn_extra")
            if self._partial_kind == "proportional" and nope_angles > 0:
                raise NotImplementedError(
                    "YARN + proportional partial-rotary not supported"
                )
            dim_rot = max(2 * rope_angles, 2)
            inv_freq_yarn, att_yarn = _yarn_inv_freq_and_scale(
                spec.base_theta, dim_rot, spec.yarn_extra,
            )
            inv_freq = inv_freq_yarn
            self._yarn_attention_factor = float(att_yarn)
        elif spec.scaling != types.RoPEScaling.NONE:
            raise NotImplementedError(
                f"B5 supports NONE, LLAMA3, LONGROPE, YARN scaling only, got {spec.scaling}"
            )

        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)            # [max_seq, *]
        # cos/sin shape depends on basis and partial_rotary_kind:
        # - SPLIT_HALF + "prefix": [max_seq, head_dim_rot] — cos/sin at rotated channels.
        # - SPLIT_HALF + "proportional": [max_seq, head_dim] — full-dim, with
        #   cos=1, sin=0 in the trailing zero-inv_freq channels.
        # - INTERLEAVED: [max_seq, head_dim/2] — one entry per (real, imag) pair.
        att_post_mul = getattr(self, "_yarn_attention_factor", None)
        if spec.basis == types.RoPEBasis.INTERLEAVED:
            cos = freqs.cos()                       # [max_seq, head_dim/2]
            sin = freqs.sin()
            if att_post_mul is not None:
                cos = cos * att_post_mul
                sin = sin * att_post_mul
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        else:
            cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
            sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
            if att_post_mul is not None:
                cos = cos * att_post_mul
                sin = sin * att_post_mul
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self._longrope_boundary = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # B8: M-RoPE branch — Qwen2.5-VL. position_ids has shape [3, B, S]
        # (axes: temporal / height / width). cos/sin tables are computed
        # on-the-fly from inv_freq @ position_ids and stitched per
        # mrope_section. The intra-head dispatch is in api/ops.py.
        if position_ids.dim() == 3:
            if self.spec.mrope_section is None:
                raise ValueError(
                    "3D position_ids require RoPESpec.mrope_section to be set"
                )
            if position_ids.shape[0] != 3:
                raise NotImplementedError(
                    f"M-RoPE expects 3 axes (T/H/W), got {position_ids.shape[0]}"
                )
            # Recover inv_freq from cos_cached (shape [max_seq, head_dim] for
            # SPLIT_HALF). For M-RoPE we must rebuild cos/sin against the
            # arbitrary position_ids tensor — we cannot precompute since the
            # 3-axis position_ids change per forward. Use the rope_angles
            # inv_freq we built in __init__ (note: M-RoPE Qwen2.5-VL uses
            # `partial_rotary_factor=1.0`, `scaling=NONE`, so the construction
            # mirrors compute_default_rope_parameters).
            # Source: modeling_qwen2_5_vl.py:485-545 (Qwen2_5_VLRotaryEmbedding).
            if self.spec.partial_rotary_factor != 1.0:
                raise NotImplementedError(
                    "M-RoPE + partial_rotary_factor < 1 not supported"
                )
            if self.spec.scaling != types.RoPEScaling.NONE:
                raise NotImplementedError(
                    "M-RoPE only supports RoPEScaling.NONE"
                )
            # Rebuild inv_freq from base_theta (same formula as __init__'s
            # construction for SPLIT_HALF partial=1.0).
            head_dim_rot = 2 * self.rope_angles                 # = head_dim
            inv_freq = 1.0 / (
                self.spec.base_theta ** (
                    torch.arange(0, head_dim_rot, 2, dtype=torch.float32,
                                 device=q.device) / head_dim_rot
                )
            )
            # Expand for 3-axis projection: [3, B, head_dim/2, 1].
            B_n = position_ids.shape[1]
            inv_freq_expanded = (
                inv_freq[None, None, :, None].expand(3, B_n, -1, 1)
            )
            # position_ids: [3, B, S] -> [3, B, 1, S] for matmul.
            pos_exp = position_ids[:, :, None, :].float()
            # freqs: [3, B, head_dim/2, S] -> transpose to [3, B, S, head_dim/2]
            freqs = (inv_freq_expanded @ pos_exp).transpose(2, 3)
            # emb = cat([freqs, freqs], -1) -> [3, B, S, head_dim]
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(q.dtype)
            sin = emb.sin().to(q.dtype)
            return ops.rope_apply_mrope(q, k, cos, sin, self.spec.mrope_section)
        if position_ids.dim() != 1:
            raise NotImplementedError("M1 supports 1D position_ids only")
        # B2a: LongRoPE dispatch — HF picks the SHORT or LONG inv_freq table
        # for the WHOLE forward pass based on `max(position_ids) + 1` vs
        # `original_max_position_embeddings`. Source:
        # modeling_rope_utils.py:longrope_frequency_update lines 47-80.
        if self._longrope_boundary is not None:
            seq_len_max = int(position_ids.max().item()) + 1
            if seq_len_max > self._longrope_boundary:
                cos = self.cos_cached_long[position_ids]
                sin = self.sin_cached_long[position_ids]
            else:
                cos = self.cos_cached[position_ids]
                sin = self.sin_cached[position_ids]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        # Dispatch by basis + partial_rotary_kind:
        # - INTERLEAVED (Llama 4) → cos/sin span head_dim/2; rope_apply
        #   reshapes head into (Dh/2, 2) pairs and complex-multiplies.
        # - SPLIT_HALF + "proportional" (Gemma 4) → cos/sin span the full
        #   head_dim; rotate_half pairs i↔i+head_dim/2 across the full head_dim,
        #   with identity multiply on the trailing zero-inv_freq channels.
        # - SPLIT_HALF + "prefix" (Phi-3 / Phi-4) → cos/sin span only
        #   head_dim_rot; rope_apply_partial slices q[..., :rot_dim] (rotated)
        #   and q[..., rot_dim:] (pass-through) and concatenates.
        if self.spec.basis == types.RoPEBasis.INTERLEAVED:
            return ops.rope_apply(q, k, cos, sin, basis="interleaved")
        if self._partial_kind == "prefix" and self.spec.partial_rotary_factor < 1.0:
            return ops.rope_apply_partial(
                q, k, cos, sin,
                partial_rotary_factor=self.spec.partial_rotary_factor,
                basis="split_half",
            )
        return ops.rope_apply(q, k, cos, sin, basis="split_half")
