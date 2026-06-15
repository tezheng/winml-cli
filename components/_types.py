"""Core types for the LLM components package — pure data, no ML deps."""
from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from enum import Enum
from typing import Literal, Optional


class DType(str, Enum):
    """Element dtypes used by component specs. String values map directly to
    MLIR `!winml.dtype` and ONNX `attribute s: "..."`."""
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U4 = "u4"
    U8 = "u8"
    BOOL = "bool"


# ---- LoRA -------------------------------------------------------------------

@dataclass(frozen=True)
class LoRAAdapterSpec:
    """A single LoRA adapter attached to one weight slot of a component."""
    rank: int
    alpha: float
    target_slot: str                        # name from ApiSchema.weights[*].name
    dropout: float = 0.0
    init: Literal["zeros", "kaiming"] = "zeros"


# ---- Spec base --------------------------------------------------------------

@dataclass(frozen=True)
class Spec:
    """Marker base. All component-attribute specs subclass this.

    Subclasses MUST declare a `kind: Literal[...]` tag as a class-level
    discriminator and override defaults rather than add fields-without-defaults.
    """


@dataclass(frozen=True)
class _ComponentBase(Spec):
    """Mixin for any spec that represents a *component instance* (not a primitive).
    Provides the uniform LoRA attachment slot."""
    lora: tuple[LoRAAdapterSpec, ...] = ()

    def __post_init__(self) -> None:
        # Reject `list` here — frozen dataclasses with list fields would be
        # unhashable, breaking cross-layer sharing as cache keys.
        if isinstance(self.lora, list):
            raise TypeError(
                f"{type(self).__name__}.lora must be a tuple, not list "
                f"(frozen specs require hashable fields)."
            )


# ---- API schema -------------------------------------------------------------

@dataclass(frozen=True)
class ApiField:
    """One slot in a component's API surface."""
    name: str
    type: str                               # e.g. "tensor<f16>", "i64", "f32"
    role: str
    shape: Optional[str] = None             # e.g. "[B, S, D]"; None for scalars
    default: Optional[str] = None           # human-readable default (string)
    lora_attach: bool = False               # only meaningful on weight slots
    optional: bool = False                  # field may be omitted entirely
    role_tag: Optional[str] = None          # LoRA-targeting tag (e.g. "attn.q")


@dataclass(frozen=True)
class AttrMeta:
    """Per-field metadata for auto-deriving `ApiSchema.attributes`.

    Stored as a class attribute `_ATTR_META: dict[str, AttrMeta]` on each Spec.
    """
    type: str                               # "i64", "f32", "string", "bool", ...
    role: str
    default_doc: Optional[str] = None       # human-readable default (string)


@dataclass(frozen=True)
class ApiSchema:
    inputs: tuple[ApiField, ...]
    outputs: tuple[ApiField, ...]
    attributes: tuple[ApiField, ...]
    weights: tuple[ApiField, ...]


# ---- ComponentMeta ----------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    slug: str
    name: str
    description: str
    spec: Spec                              # typed spec for this variant
    api: ApiSchema


@dataclass(frozen=True)
class ComponentMeta:
    slug: str
    name: str
    oneliner: str
    status: Literal["live", "stub"]
    badges: tuple[str, ...]
    variants: tuple[Variant, ...]


# ---- Auto-derive ApiSchema.attributes ---------------------------------------

def derive_api_attributes(spec_cls: type) -> tuple[ApiField, ...]:
    """Build ApiField objects for the attribute list by walking dataclass fields
    and joining with the spec class's `_ATTR_META`.

    Skips:
      - the `kind` discriminator (variant tag is emitted separately when needed)
      - the `lora` field (composition concern, not an op attribute)

    Raises ValueError if a dataclass field is missing from `_ATTR_META` — keeps
    the two-source-of-truth bug from re-emerging.
    """
    if not hasattr(spec_cls, "_ATTR_META"):
        return ()
    meta: dict[str, AttrMeta] = spec_cls._ATTR_META  # type: ignore[attr-defined]
    out: list[ApiField] = []
    for f in dc_fields(spec_cls):
        if f.name in ("kind", "lora"):
            continue
        if f.name not in meta:
            raise ValueError(
                f"{spec_cls.__name__} field '{f.name}' missing from _ATTR_META; "
                f"add or skip it explicitly."
            )
        m = meta[f.name]
        out.append(ApiField(
            name=f.name,
            type=m.type,
            role=m.role,
            default=m.default_doc,
        ))
    return tuple(out)


__all__ = [
    "DType", "Spec", "_ComponentBase", "LoRAAdapterSpec",
    "ApiField", "AttrMeta", "ApiSchema", "ComponentMeta", "Variant",
    "derive_api_attributes",
]
