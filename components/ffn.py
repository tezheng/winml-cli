"""FFN component — gated MLP variants.

Source: registry/variants.ts:488-491; field design from modeling_llama.py
(SwiGLU), modeling_gemma.py (GeGLU), and modeling_gpt_oss.py (Clamped-SwiGLU).
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
class SwiGLUSpec(_ComponentBase):
    """SiLU-gated MLP: y = down(silu(gate(x)) * up(x))."""
    kind: Literal["SwiGLU"] = "SwiGLU"
    intermediate_size: int
    fused_gate_up: bool = False                    # Phi-family fuses gate+up into one weight

    _ATTR_META = {
        "intermediate_size": AttrMeta("i64",  "Inner / hidden dim of the MLP."),
        "fused_gate_up":     AttrMeta("bool", "When true, gate and up projections share one fused weight.", "false"),
    }


@dataclass(frozen=True, kw_only=True)
class GeGLUSpec(_ComponentBase):
    """GeLU-gated MLP — Gemma family."""
    kind: Literal["GeGLU"] = "GeGLU"
    intermediate_size: int
    activation: Literal["gelu_tanh", "gelu_exact"] = "gelu_tanh"

    _ATTR_META = {
        "intermediate_size": AttrMeta("i64",    "Inner / hidden dim of the MLP."),
        "activation":        AttrMeta("string", "GeLU flavor (tanh-approx or exact).", '"gelu_tanh"'),
    }


@dataclass(frozen=True, kw_only=True)
class ClampedSwiGLUSpec(_ComponentBase):
    """Clamped SwiGLU — GPT-OSS asymmetric clamp + alpha-scaled sigmoid.
    Source: modeling_gpt_oss.py MLP forward; up_shift, alpha=1.702."""
    kind: Literal["Clamped-SwiGLU"] = "Clamped-SwiGLU"
    intermediate_size: int
    clamp: float = 7.0
    alpha: float = 1.702

    _ATTR_META = {
        "intermediate_size": AttrMeta("i64", "Inner / hidden dim of the MLP."),
        "clamp":             AttrMeta("f32", "Symmetric clamp magnitude on gate/up pre-activations.", "7.0"),
        "alpha":             AttrMeta("f32", "Sigmoid-gain coefficient on the gate (Swish-beta).", "1.702"),
    }


FFNSpec = SwiGLUSpec | GeGLUSpec | ClampedSwiGLUSpec


# ---- Shared API field fragments ---------------------------------------------

_FFN_INPUTS: tuple[ApiField, ...] = (
    ApiField("input", "tensor<f16>", "Hidden state.", shape="[B, S, D_model]"),
)

_FFN_OUTPUTS: tuple[ApiField, ...] = (
    ApiField("output", "tensor<f16>", "MLP output.", shape="[B, S, D_model]"),
)

_FFN_WEIGHTS_GATED: tuple[ApiField, ...] = (
    ApiField("W_gate", "tensor<f16>", "Gate projection weight.",
             shape="[I, D_model]", lora_attach=True, role_tag="mlp.gate"),
    ApiField("W_up",   "tensor<f16>", "Up projection weight.",
             shape="[I, D_model]", lora_attach=True, role_tag="mlp.up"),
    ApiField("W_down", "tensor<f16>", "Down projection weight.",
             shape="[D_model, I]", lora_attach=True, role_tag="mlp.down"),
)

_FFN_WEIGHTS_FUSED: tuple[ApiField, ...] = (
    ApiField("W_gate_up", "tensor<f16>", "Fused gate||up projection weight.",
             shape="[2*I, D_model]", lora_attach=True, role_tag="mlp.gate_up"),
    ApiField("W_down",    "tensor<f16>", "Down projection weight.",
             shape="[D_model, I]", lora_attach=True, role_tag="mlp.down"),
)


# ---- Variants ---------------------------------------------------------------

_SWIGLU = Variant(
    slug="swiglu",
    name="SwiGLU",
    description=(
        "SiLU-gated linear unit — the dominant SLM choice. `fused_gate_up=true` "
        "selects the Phi-family single-weight form."
    ),
    spec=SwiGLUSpec(intermediate_size=14336),
    api=ApiSchema(
        inputs=_FFN_INPUTS,
        outputs=_FFN_OUTPUTS,
        attributes=derive_api_attributes(SwiGLUSpec),
        weights=_FFN_WEIGHTS_GATED,
    ),
)

_GEGLU = Variant(
    slug="geglu",
    name="GeGLU",
    description="GeLU-gated linear unit — Gemma family flavor.",
    spec=GeGLUSpec(intermediate_size=16384),
    api=ApiSchema(
        inputs=_FFN_INPUTS,
        outputs=_FFN_OUTPUTS,
        attributes=derive_api_attributes(GeGLUSpec),
        weights=_FFN_WEIGHTS_GATED,
    ),
)

_CLAMPED_SWIGLU = Variant(
    slug="clamped-swiglu",
    name="Clamped-SwiGLU",
    description=(
        "GPT-OSS asymmetric-clamp SwiGLU: pre-activation clamp on gate/up, "
        "Swish-beta sigmoid gain (alpha)."
    ),
    spec=ClampedSwiGLUSpec(intermediate_size=2880),
    api=ApiSchema(
        inputs=_FFN_INPUTS,
        outputs=_FFN_OUTPUTS,
        attributes=derive_api_attributes(ClampedSwiGLUSpec),
        weights=_FFN_WEIGHTS_GATED,
    ),
)


COMPONENT: ComponentMeta = ComponentMeta(
    slug="ffn",
    name="FFN",
    oneliner="Position-wise gated MLP — SwiGLU, GeGLU, Clamped-SwiGLU.",
    status="live",
    badges=("core", "decoder-only"),
    variants=(_SWIGLU, _GEGLU, _CLAMPED_SWIGLU),
)


__all__ = [
    "SwiGLUSpec", "GeGLUSpec", "ClampedSwiGLUSpec", "FFNSpec", "COMPONENT",
]
