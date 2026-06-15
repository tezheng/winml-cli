"""Positional Encoding component — RoPE family + NoPE.

Source: registry/variants.ts:479-486 (slugs); spec field design from
modeling_llama.py / modeling_qwen.py / modeling_phi.py rope_scaling configs.

Important rename: the registry's `RoPE-Llama3Scaling` slug is exposed here as
`RoPE-SmoothScaling` (architecture-agnostic name — same algorithm, no model
family in the symbol).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._types import (
    ApiField, ApiSchema, AttrMeta, ComponentMeta, Variant,
    _ComponentBase, derive_api_attributes,
)


# ---- Spec types -------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class RoPESpec(_ComponentBase):
    """Plain rotary position embeddings."""
    kind: Literal["RoPE"] = "RoPE"
    theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    partial_rotary_kind: Literal["prefix", "proportional"] = "prefix"

    _ATTR_META = {
        "theta":                 AttrMeta("f32",    "Base for inverse-frequency computation.", "10000.0"),
        "partial_rotary_factor": AttrMeta("f32",    "Fraction of head_dim rotated by RoPE; remainder passes through.", "1.0"),
        "partial_rotary_kind":   AttrMeta("string", "Layout of the rotated/un-rotated split.", '"prefix"'),
    }


@dataclass(frozen=True, kw_only=True)
class RoPESmoothScalingSpec(_ComponentBase):
    """Piecewise frequency-dependent RoPE scaling for long-context.
    Source: modeling_llama.py rope_scaling.rope_type='llama3' branch.
    Renamed from registry's `RoPE-Llama3Scaling` per Cardinal Rule."""
    kind: Literal["RoPE-SmoothScaling"] = "RoPE-SmoothScaling"
    theta: float = 10000.0
    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_context_length: int = 8192

    _ATTR_META = {
        "theta":                   AttrMeta("f32", "Base for inverse-frequency.", "10000.0"),
        "factor":                  AttrMeta("f32", "Overall scaling factor for low-frequency dims.", "8.0"),
        "low_freq_factor":         AttrMeta("f32", "Cutoff factor for the low-frequency interpolation region.", "1.0"),
        "high_freq_factor":        AttrMeta("f32", "Cutoff factor for the high-frequency pass-through region.", "4.0"),
        "original_context_length": AttrMeta("i64", "Context length the base model was trained on.", "8192"),
    }


@dataclass(frozen=True, kw_only=True)
class RoPEYaRNSpec(_ComponentBase):
    """YaRN scaling: NTK-by-parts interpolation + attention-temperature mscale.
    Source: modeling_deepseek.py / modeling_qwen.py rope_scaling.rope_type='yarn'."""
    kind: Literal["RoPE-YaRN"] = "RoPE-YaRN"
    theta: float = 10000.0
    factor: float = 40.0
    attn_factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0
    original_context_length: int = 4096

    _ATTR_META = {
        "theta":                   AttrMeta("f32", "Base for inverse-frequency.", "10000.0"),
        "factor":                  AttrMeta("f32", "Linear scaling factor (target / original context length).", "40.0"),
        "attn_factor":             AttrMeta("f32", "Multiplier on the attention-temperature mscale.", "1.0"),
        "beta_fast":                AttrMeta("f32", "High-frequency cutoff (no scaling above).", "32.0"),
        "beta_slow":               AttrMeta("f32", "Low-frequency cutoff (full scaling below).", "1.0"),
        "mscale":                  AttrMeta("f32", "Attention-temperature multiplier.", "1.0"),
        "mscale_all_dim":          AttrMeta("f32", "Attention-temperature multiplier applied to all dims.", "0.0"),
        "original_context_length": AttrMeta("i64", "Context length the base model was trained on.", "4096"),
    }


@dataclass(frozen=True, kw_only=True)
class RoPELongRoPESpec(_ComponentBase):
    """LongRoPE: per-frequency rescale tables for short and long contexts.
    Source: modeling_phi3.py rope_scaling.rope_type='longrope'."""
    kind: Literal["RoPE-LongRoPE"] = "RoPE-LongRoPE"
    theta: float = 10000.0
    factor: float = 1.0
    short_factor: tuple[float, ...] = ()
    long_factor: tuple[float, ...] = ()
    original_context_length: int = 4096

    _ATTR_META = {
        "theta":                   AttrMeta("f32",   "Base for inverse-frequency.", "10000.0"),
        "factor":                  AttrMeta("f32",   "Overall scale factor applied alongside per-freq tables.", "1.0"),
        "short_factor":            AttrMeta("f32[]", "Per-frequency rescale table used at <= original_context_length."),
        "long_factor":             AttrMeta("f32[]", "Per-frequency rescale table used beyond original_context_length."),
        "original_context_length": AttrMeta("i64",   "Context length the base model was trained on.", "4096"),
    }

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.short_factor, list) or isinstance(self.long_factor, list):
            raise TypeError("RoPELongRoPESpec factor tables must be tuples, not lists.")


@dataclass(frozen=True, kw_only=True)
class RoPELinearSpec(_ComponentBase):
    """Plain linear position-id scaling. Source: Gemma-3 global layers, NTK-by-1."""
    kind: Literal["RoPE-Linear"] = "RoPE-Linear"
    theta: float = 10000.0
    factor: float = 1.0

    _ATTR_META = {
        "theta":  AttrMeta("f32", "Base for inverse-frequency.", "10000.0"),
        "factor": AttrMeta("f32", "Linear scaling factor applied to position ids.", "1.0"),
    }


@dataclass(frozen=True, kw_only=True)
class MRoPEInterleavedSpec(_ComponentBase):
    """Multimodal RoPE with interleaved (T, H, W) section assignment.
    Source: modeling_qwen2_vl.py / modeling_qwen2_5_vl.py mrope_section."""
    kind: Literal["MRoPE-Interleaved"] = "MRoPE-Interleaved"
    theta: float = 10000.0
    sections: tuple[int, ...] = ()                  # e.g. (16, 24, 24) — sums to head_dim/2
    partial_rotary_factor: float = 1.0
    partial_rotary_kind: Literal["prefix", "proportional"] = "prefix"

    _ATTR_META = {
        "theta":                 AttrMeta("f32",    "Base for inverse-frequency.", "10000.0"),
        "sections":              AttrMeta("i64[]",  "Per-axis (T, H, W, ...) frequency slot counts. Must sum to rotated dim/2."),
        "partial_rotary_factor": AttrMeta("f32",    "Fraction of head_dim rotated.", "1.0"),
        "partial_rotary_kind":   AttrMeta("string", "Layout of the rotated/un-rotated split.", '"prefix"'),
    }

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.sections, list):
            raise TypeError("MRoPEInterleavedSpec.sections must be a tuple, not list.")


@dataclass(frozen=True, kw_only=True)
class NoPESpec(_ComponentBase):
    """Identity on Q/K — explicit `no positional encoding` (SmolLM3 NoPE-skip)."""
    kind: Literal["NoPE"] = "NoPE"

    _ATTR_META = {}


PositionalEncodingSpec = (
    RoPESpec | RoPESmoothScalingSpec | RoPEYaRNSpec | RoPELongRoPESpec
    | RoPELinearSpec | MRoPEInterleavedSpec | NoPESpec
)


# ---- Shared API field fragments ---------------------------------------------

_PE_INPUTS: tuple[ApiField, ...] = (
    ApiField("q", "tensor<f16>", "Query tensor to be rotated.",
             shape="[B, S, H_q, Dh]"),
    ApiField("k", "tensor<f16>", "Key tensor to be rotated.",
             shape="[B, S, H_kv, Dh]"),
    ApiField("position_ids", "tensor<i64>", "Per-token position indices.",
             shape="[B, S]"),
)

_PE_OUTPUTS: tuple[ApiField, ...] = (
    ApiField("q_out", "tensor<f16>", "Rotated query.",
             shape="[B, S, H_q, Dh]"),
    ApiField("k_out", "tensor<f16>", "Rotated key.",
             shape="[B, S, H_kv, Dh]"),
)

_NOPE_INPUTS: tuple[ApiField, ...] = (
    ApiField("q", "tensor<f16>", "Query tensor (passed through).",
             shape="[B, S, H_q, Dh]"),
    ApiField("k", "tensor<f16>", "Key tensor (passed through).",
             shape="[B, S, H_kv, Dh]"),
)

_MROPE_INPUTS: tuple[ApiField, ...] = (
    ApiField("q", "tensor<f16>", "Query tensor to be rotated.",
             shape="[B, S, H_q, Dh]"),
    ApiField("k", "tensor<f16>", "Key tensor to be rotated.",
             shape="[B, S, H_kv, Dh]"),
    ApiField("position_ids", "tensor<i64>", "Per-axis position indices (e.g. T/H/W).",
             shape="[A, B, S]"),
)


def _pe_variant(slug: str, name: str, desc: str, spec, *, inputs=_PE_INPUTS) -> Variant:
    return Variant(
        slug=slug, name=name, description=desc, spec=spec,
        api=ApiSchema(
            inputs=inputs,
            outputs=_PE_OUTPUTS,
            attributes=derive_api_attributes(type(spec)),
            weights=(),
        ),
    )


# ---- Variants ---------------------------------------------------------------

_ROPE = _pe_variant(
    "rope", "RoPE",
    "Rotary position embeddings — the field default.",
    RoPESpec(),
)

_ROPE_SMOOTH = _pe_variant(
    "rope-smooth-scaling", "RoPE-SmoothScaling",
    "Piecewise frequency-dependent scaling for long-context extension.",
    RoPESmoothScalingSpec(),
)

_ROPE_YARN = _pe_variant(
    "rope-yarn", "RoPE-YaRN",
    "YaRN scaling: NTK-by-parts + attention-temperature mscale.",
    RoPEYaRNSpec(),
)

_ROPE_LONGROPE = _pe_variant(
    "rope-longrope", "RoPE-LongRoPE",
    "LongRoPE: short_factor/long_factor per-frequency rescale tables.",
    RoPELongRoPESpec(),
)

_ROPE_LINEAR = _pe_variant(
    "rope-linear", "RoPE-Linear",
    "Plain linear position-id scaling.",
    RoPELinearSpec(),
)

_MROPE = _pe_variant(
    "mrope-interleaved", "MRoPE-Interleaved",
    "Multimodal RoPE with interleaved (T, H, W) section assignment.",
    MRoPEInterleavedSpec(),
    inputs=_MROPE_INPUTS,
)

_NOPE = Variant(
    slug="nope", name="NoPE",
    description="Identity transform on Q/K — explicit `no positional encoding`.",
    spec=NoPESpec(),
    api=ApiSchema(
        inputs=_NOPE_INPUTS,
        outputs=(
            ApiField("q_out", "tensor<f16>", "Query passed through.",
                     shape="[B, S, H_q, Dh]"),
            ApiField("k_out", "tensor<f16>", "Key passed through.",
                     shape="[B, S, H_kv, Dh]"),
        ),
        attributes=derive_api_attributes(NoPESpec),
        weights=(),
    ),
)


COMPONENT: ComponentMeta = ComponentMeta(
    slug="positional-encoding",
    name="PositionalEncoding",
    oneliner="Position information injection — RoPE family and NoPE.",
    status="live",
    badges=("core", "decoder-only"),
    variants=(_ROPE, _ROPE_SMOOTH, _ROPE_YARN, _ROPE_LONGROPE, _ROPE_LINEAR, _MROPE, _NOPE),
)


__all__ = [
    "RoPESpec", "RoPESmoothScalingSpec", "RoPEYaRNSpec", "RoPELongRoPESpec",
    "RoPELinearSpec", "MRoPEInterleavedSpec", "NoPESpec",
    "PositionalEncodingSpec", "COMPONENT",
]
