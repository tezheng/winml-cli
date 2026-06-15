"""Norm component — RMSNorm with optional zero-centered weight (Gemma form).

The registry collapses LayerNorm-vs-RMSNorm into a single component with an
attribute distinguishing standard vs zero-centered weight semantics.
Here we expose two typed variants because consumers downstream typically
dispatch on `kind` rather than read a bool — keeps pattern-matching tight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._types import (
    ApiField, ApiSchema, AttrMeta, ComponentMeta, DType, Variant,
    _ComponentBase, derive_api_attributes,
)


# ---- Spec types -------------------------------------------------------------

@dataclass(frozen=True)
class _NormBase(_ComponentBase):
    """Common fields for both Norm variants. Source: registry/variants.ts:454-457"""
    eps: float = 1e-6
    axis: int = -1
    accumulator_dtype: DType = DType.F32


@dataclass(frozen=True)
class NormStandardSpec(_NormBase):
    """RMSNorm with plain `w * x_normed` form."""
    kind: Literal["Standard"] = "Standard"

    _ATTR_META = {
        "eps":               AttrMeta("f32", "Numerical stability epsilon inside the sqrt.", "1e-6"),
        "axis":              AttrMeta("i64", "Reduction axis (typically the last).", "-1"),
        "accumulator_dtype": AttrMeta("string", "Accumulator dtype for the variance reduction.", '"f32"'),
    }


@dataclass(frozen=True)
class NormZeroCenteredSpec(_NormBase):
    """RMSNorm with `(1 + w) * x_normed` form. Source: Gemma family."""
    kind: Literal["ZeroCentered"] = "ZeroCentered"

    _ATTR_META = {
        "eps":               AttrMeta("f32", "Numerical stability epsilon inside the sqrt.", "1e-6"),
        "axis":              AttrMeta("i64", "Reduction axis (typically the last).", "-1"),
        "accumulator_dtype": AttrMeta("string", "Accumulator dtype for the variance reduction.", '"f32"'),
    }


NormSpec = NormStandardSpec | NormZeroCenteredSpec


# ---- ApiSchema (shared inputs / outputs / weights) --------------------------
# Source: registry/variants.ts:447-462

_NORM_API_INPUTS: tuple[ApiField, ...] = (
    ApiField(
        name="input", type="tensor<f16>", shape="[..., D]",
        role="Activation to normalize. Reduction along the configured axis.",
    ),
)

_NORM_API_OUTPUTS: tuple[ApiField, ...] = (
    ApiField(
        name="output", type="tensor<f16>", shape="[..., D]",
        role="Normalized and scaled activation.",
    ),
)

_NORM_API_WEIGHTS: tuple[ApiField, ...] = (
    ApiField(
        name="weight", type="tensor<f16>", shape="[D]",
        role="Per-channel scale. Interpretation depends on variant kind.",
        lora_attach=False, role_tag="norm.weight",
    ),
)


# ---- Variants ---------------------------------------------------------------

_STANDARD = Variant(
    slug="rms-norm",
    name="RMSNorm",
    description=(
        "Root-mean-square normalization with a learned per-channel scale. "
        "Effective scale is `weight`. The standard SLM choice since Llama-2."
    ),
    spec=NormStandardSpec(),
    api=ApiSchema(
        inputs=_NORM_API_INPUTS,
        outputs=_NORM_API_OUTPUTS,
        attributes=derive_api_attributes(NormStandardSpec),
        weights=_NORM_API_WEIGHTS,
    ),
)

_ZERO_CENTERED = Variant(
    slug="rms-norm-zero-centered",
    name="RMSNorm-ZeroCentered",
    description=(
        "RMSNorm with `(1 + weight) * x_normed` form (Gemma family). "
        "Initializes scale around 0 instead of 1 — cleaner optimization signal."
    ),
    spec=NormZeroCenteredSpec(),
    api=ApiSchema(
        inputs=_NORM_API_INPUTS,
        outputs=_NORM_API_OUTPUTS,
        attributes=derive_api_attributes(NormZeroCenteredSpec),
        weights=_NORM_API_WEIGHTS,
    ),
)


COMPONENT: ComponentMeta = ComponentMeta(
    slug="norm",
    name="Norm",
    oneliner="Root-mean-square normalization with a learned per-channel scale.",
    status="live",
    badges=("core", "decoder-only"),
    variants=(_STANDARD, _ZERO_CENTERED),
)


__all__ = [
    "NormStandardSpec", "NormZeroCenteredSpec", "NormSpec", "COMPONENT",
]
