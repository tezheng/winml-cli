"""LLM components — pure data definitions for the API layer.

Zero ML dependencies. Only stdlib (dataclasses, enum, typing).
"""
from __future__ import annotations

from typing import Optional

from ._types import (
    ApiField, ApiSchema, AttrMeta, ComponentMeta, DType, LoRAAdapterSpec,
    Spec, Variant, derive_api_attributes,
)
from .attention import (
    AttentionSpec, COMPONENT as attention_component,
    GQASinksSpec, GQASpec, MHASpec, MLASpec,
)
from .block_graph import (
    BlockGraph, Node,
    parallel_block, post_norm_reordered_block, pre_norm_block, sandwich_block,
)
from .ffn import (
    ClampedSwiGLUSpec, COMPONENT as ffn_component, FFNSpec, GeGLUSpec, SwiGLUSpec,
)
from .misc import AddSpec, MulSpec
from .norm import (
    COMPONENT as norm_component, NormSpec, NormStandardSpec, NormZeroCenteredSpec,
)
from .positional_encoding import (
    COMPONENT as positional_encoding_component,
    MRoPEInterleavedSpec, NoPESpec, PositionalEncodingSpec,
    RoPELinearSpec, RoPELongRoPESpec, RoPESmoothScalingSpec, RoPESpec, RoPEYaRNSpec,
)


ALL_COMPONENTS: tuple[ComponentMeta, ...] = (
    norm_component,
    attention_component,
    positional_encoding_component,
    ffn_component,
)


def get_component(slug: str) -> Optional[ComponentMeta]:
    """Look up a ComponentMeta by its slug. Returns None if not found."""
    return next((c for c in ALL_COMPONENTS if c.slug == slug), None)


__all__ = [
    # base types
    "DType", "Spec", "LoRAAdapterSpec",
    "ApiField", "ApiSchema", "AttrMeta", "ComponentMeta", "Variant",
    "derive_api_attributes",
    # norm
    "NormStandardSpec", "NormZeroCenteredSpec", "NormSpec",
    # attention
    "MHASpec", "GQASpec", "MLASpec", "GQASinksSpec", "AttentionSpec",
    # positional encoding
    "RoPESpec", "RoPESmoothScalingSpec", "RoPEYaRNSpec", "RoPELongRoPESpec",
    "RoPELinearSpec", "MRoPEInterleavedSpec", "NoPESpec", "PositionalEncodingSpec",
    # ffn
    "SwiGLUSpec", "GeGLUSpec", "ClampedSwiGLUSpec", "FFNSpec",
    # graph primitives
    "AddSpec", "MulSpec",
    # block graph
    "Node", "BlockGraph",
    "pre_norm_block", "post_norm_reordered_block", "parallel_block", "sandwich_block",
    # component registry
    "ALL_COMPONENTS", "get_component",
]
