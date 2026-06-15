"""Attention component — discriminated union of MHA, GQA, MLA, GQA+Sinks.

Source: registry/variants.ts:28-399
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ._types import (
    ApiField, ApiSchema, AttrMeta, ComponentMeta, DType, Variant,
    _ComponentBase, derive_api_attributes,
)
from .norm import NormSpec


# ---- Spec types -------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class _AttentionBase(_ComponentBase):
    """Fields shared by every attention variant.

    `kw_only=True` lets subclasses introduce additional required fields after
    fields-with-defaults inherited from `_ComponentBase` without ordering pain.
    """
    num_heads: int
    head_dim: int
    causal: bool = True
    qk_norm: Optional[NormSpec] = None             # Qwen3 / Gemma-3/4 / OLMo-2 / many
    rope_applied_inside: bool = False              # M0: RoPE lives outside attention
    accumulator_dtype: DType = DType.F32
    logit_softcap: float = 0.0                     # Gemma-2 tanh softcap; 0 disables

    @property
    def scale(self) -> float:
        """Derived; not stored. Avoids cross-instance float drift."""
        return 1.0 / (self.head_dim ** 0.5)


@dataclass(frozen=True, kw_only=True)
class MHASpec(_AttentionBase):
    """Classical multi-head attention. num_kv_heads == num_heads.
    Source: registry/variants.ts:30-124"""
    kind: Literal["MHA"] = "MHA"

    _ATTR_META = {
        "num_heads":           AttrMeta("i64",    "Total attention heads (e.g. 32)."),
        "head_dim":            AttrMeta("i64",    "Per-head dim — first-class, NOT derived from d_model/num_heads."),
        "causal":              AttrMeta("bool",   "Apply causal masking.", "true"),
        "qk_norm":             AttrMeta("Norm?",  "Optional Q/K normalization before RoPE."),
        "rope_applied_inside": AttrMeta("bool",   "When true, the attention op applies RoPE internally.", "false"),
        "accumulator_dtype":   AttrMeta("string", "GEMM accumulator dtype.", '"f32"'),
        "logit_softcap":       AttrMeta("f32",    "Tanh softcap on logits (Gemma-2). 0.0 disables.", "0.0"),
    }


@dataclass(frozen=True, kw_only=True)
class GQASpec(_AttentionBase):
    """Grouped-Query Attention. K/V projections shrink by num_heads/num_kv_heads.
    Source: registry/variants.ts:125-220"""
    kind: Literal["GQA"] = "GQA"
    num_kv_heads: int = 8
    sliding_window: Optional[int] = None           # SWA (Mistral / Gemma / GPT-OSS alt)
    output_gate: Optional[Literal["sigmoid"]] = None  # Qwen3.5/3.6 output gate
    v_norm: Optional[NormSpec] = None              # Gemma-4 V-norm (no learned scale)
    qk_norm_fixed_scale: Optional[float] = None    # Gemma-4 absorbs 1/sqrt(Dh) into qk_norm
    kv_source_layer_offset: int = 0                # Gemma-4 cross-layer / Hunyuan CLA

    _ATTR_META = {
        "num_heads":              AttrMeta("i64",    "Query heads (e.g. 32)."),
        "head_dim":               AttrMeta("i64",    "Per-head dim — first-class, NOT derived."),
        "causal":                 AttrMeta("bool",   "Apply causal masking.", "true"),
        "qk_norm":                AttrMeta("Norm?",  "Optional Q/K normalization before RoPE."),
        "rope_applied_inside":    AttrMeta("bool",   "When true, the attention op applies RoPE internally.", "false"),
        "accumulator_dtype":      AttrMeta("string", "GEMM accumulator dtype.", '"f32"'),
        "logit_softcap":          AttrMeta("f32",    "Tanh softcap on logits (Gemma-2). 0.0 disables.", "0.0"),
        "num_kv_heads":           AttrMeta("i64",    "K/V heads, must divide num_heads (e.g. 8).", "8"),
        "sliding_window":         AttrMeta("i64?",   "Window size in tokens; None = full attention."),
        "output_gate":            AttrMeta("string?","Per-head output gating activation (Qwen3.5/3.6).", "null"),
        "v_norm":                 AttrMeta("Norm?",  "Optional V normalization (Gemma-4)."),
        "qk_norm_fixed_scale":    AttrMeta("f32?",   "Gemma-4: absorbs 1/sqrt(head_dim) into qk_norm scale."),
        "kv_source_layer_offset": AttrMeta("i64",    "Negative offset to source K/V from an earlier layer's cache. 0 = own cache.", "0"),
    }


@dataclass(frozen=True, kw_only=True)
class MLASpec(_AttentionBase):
    """Multi-head Latent Attention (DeepSeek). K/V reconstructed from a compressed latent.
    Source: registry/variants.ts:222-304"""
    kind: Literal["MLA"] = "MLA"
    num_kv_heads: int = 16                         # MLA: equals num_heads (no GQA grouping)
    latent_dim: int = 512                          # kv_lora_rank
    nope_head_dim: int = 128                       # qk_nope_head_dim, also v_head_dim
    rope_head_dim: int = 64                        # qk_rope_head_dim
    q_lora_rank: Optional[int] = None              # null in DeepSeek-V2-Lite

    @property
    def scale(self) -> float:
        """Override base: MLA uses 1/sqrt(qk_nope + qk_rope)."""
        return 1.0 / ((self.nope_head_dim + self.rope_head_dim) ** 0.5)

    _ATTR_META = {
        "num_heads":           AttrMeta("i64",    "Number of attention heads (MLA has no GQA grouping)."),
        "head_dim":            AttrMeta("i64",    "Total per-head Q/K dim = nope_head_dim + rope_head_dim."),
        "causal":              AttrMeta("bool",   "Apply causal masking.", "true"),
        "qk_norm":             AttrMeta("Norm?",  "Optional Q/K normalization before RoPE."),
        "rope_applied_inside": AttrMeta("bool",   "When true, the attention op applies RoPE internally.", "false"),
        "accumulator_dtype":   AttrMeta("string", "GEMM accumulator dtype.", '"f32"'),
        "logit_softcap":       AttrMeta("f32",    "Tanh softcap on logits. 0.0 disables.", "0.0"),
        "num_kv_heads":        AttrMeta("i64",    "Equals num_heads for MLA.", "16"),
        "latent_dim":          AttrMeta("i64",    "KV latent compression dim (kv_lora_rank, e.g. 512).", "512"),
        "nope_head_dim":       AttrMeta("i64",    "Non-positional part of each head's Q/K (also = v_head_dim).", "128"),
        "rope_head_dim":       AttrMeta("i64",    "RoPE-rotated part of each head's Q/K. Shared across heads on K side.", "64"),
        "q_lora_rank":         AttrMeta("i64?",   "Q low-rank bottleneck; None = no Q low-rank (DeepSeek-V2-Lite)."),
    }


@dataclass(frozen=True, kw_only=True)
class GQASinksSpec(_AttentionBase):
    """GQA with learned per-head additive bias in the softmax denominator (GPT-OSS).
    Source: registry/variants.ts:305-398"""
    kind: Literal["GQA+Sinks"] = "GQA+Sinks"
    num_kv_heads: int = 8
    softmax_scale_log_base: int = 2                # GPT-OSS: log2-base scaling
    sliding_window: Optional[int] = None           # GPT-OSS alternating SWA layers

    _ATTR_META = {
        "num_heads":              AttrMeta("i64",    "Query heads."),
        "head_dim":               AttrMeta("i64",    "Per-head dim — first-class, NOT derived."),
        "causal":                 AttrMeta("bool",   "Apply causal masking.", "true"),
        "qk_norm":                AttrMeta("Norm?",  "Optional Q/K normalization before RoPE."),
        "rope_applied_inside":    AttrMeta("bool",   "When true, the attention op applies RoPE internally.", "false"),
        "accumulator_dtype":      AttrMeta("string", "GEMM accumulator dtype.", '"f32"'),
        "logit_softcap":          AttrMeta("f32",    "Tanh softcap on logits. 0.0 disables.", "0.0"),
        "num_kv_heads":           AttrMeta("i64",    "K/V heads.", "8"),
        "softmax_scale_log_base": AttrMeta("i64",    "Base for sink-logit normalization (GPT-OSS: 2).", "2"),
        "sliding_window":         AttrMeta("i64?",   "Window size in tokens; None = full attention. GPT-OSS alternates per layer."),
    }


AttentionSpec = MHASpec | GQASpec | MLASpec | GQASinksSpec


# ---- Shared API field fragments ---------------------------------------------

_MASK_INPUT = ApiField(
    name="mask", type="tensor<f16>", shape="[B?, H?, S, K]",
    role="Optional attention mask. Dynamic dims accommodate sliding-window, ALiBi, static masks.",
    optional=True,
)


# ---- MHA Variant ------------------------------------------------------------
# Source: registry/variants.ts:30-124

_MHA_INPUTS: tuple[ApiField, ...] = (
    ApiField("query", "tensor<f16>", "Query projection output (post-RoPE if present).",
             shape="[B, S, H*Dh]"),
    ApiField("key",   "tensor<f16>", "Key projection output (post-RoPE if present).",
             shape="[B, S, H*Dh]"),
    ApiField("value", "tensor<f16>", "Value projection output.",
             shape="[B, S, H*Dh]"),
    _MASK_INPUT,
    ApiField("kv_cache_in", "tensor<f16>", "Concatenated K|V cache from previous steps.",
             shape="[2, B, H, S_cache, Dh]", optional=True),
)

_MHA_OUTPUTS: tuple[ApiField, ...] = (
    ApiField("output", "tensor<f16>", "Attention output, pre-output-projection.",
             shape="[B, S, H*Dh]"),
    ApiField("kv_cache_out", "tensor<f16>", "Updated cache including new K/V.",
             shape="[2, B, H, S_cache + S, Dh]"),
)

_MHA_WEIGHTS: tuple[ApiField, ...] = (
    ApiField("W_q", "tensor<f16>", "Query projection weight.",
             shape="[H*Dh, D_model]", lora_attach=True, role_tag="attn.q"),
    ApiField("W_k", "tensor<f16>", "Key projection weight.",
             shape="[H*Dh, D_model]", lora_attach=True, role_tag="attn.k"),
    ApiField("W_v", "tensor<f16>", "Value projection weight.",
             shape="[H*Dh, D_model]", lora_attach=True, role_tag="attn.v"),
    ApiField("W_o", "tensor<f16>", "Output projection weight.",
             shape="[D_model, H*Dh]", lora_attach=True, role_tag="attn.o"),
)

_MHA = Variant(
    slug="mha",
    name="MHA",
    description=(
        "Classical multi-head attention. Every head has its own K and V — most "
        "expressive, highest KV-cache cost. Survives in smaller SLMs predating GQA."
    ),
    spec=MHASpec(num_heads=32, head_dim=128),
    api=ApiSchema(
        inputs=_MHA_INPUTS,
        outputs=_MHA_OUTPUTS,
        attributes=derive_api_attributes(MHASpec),
        weights=_MHA_WEIGHTS,
    ),
)


# ---- GQA Variant ------------------------------------------------------------
# Source: registry/variants.ts:125-220

_GQA_INPUTS: tuple[ApiField, ...] = (
    ApiField("query", "tensor<f16>", "Query projection output.",
             shape="[B, S, H_q*Dh]"),
    ApiField("key",   "tensor<f16>", "Key projection output (fewer heads than Q).",
             shape="[B, S, H_kv*Dh]"),
    ApiField("value", "tensor<f16>", "Value projection output.",
             shape="[B, S, H_kv*Dh]"),
    _MASK_INPUT,
    ApiField("kv_cache_in", "tensor<f16>", "Cache sized to KV heads only.",
             shape="[2, B, H_kv, S_cache, Dh]", optional=True),
)

_GQA_OUTPUTS: tuple[ApiField, ...] = (
    ApiField("output", "tensor<f16>", "Attention output.",
             shape="[B, S, H_q*Dh]"),
    ApiField("kv_cache_out", "tensor<f16>", "Updated cache.",
             shape="[2, B, H_kv, S_cache + S, Dh]"),
)

_GQA_WEIGHTS: tuple[ApiField, ...] = (
    ApiField("W_q", "tensor<f16>", "Query projection.",
             shape="[H_q*Dh, D_model]", lora_attach=True, role_tag="attn.q"),
    ApiField("W_k", "tensor<f16>", "Key projection.",
             shape="[H_kv*Dh, D_model]", lora_attach=True, role_tag="attn.k"),
    ApiField("W_v", "tensor<f16>", "Value projection.",
             shape="[H_kv*Dh, D_model]", lora_attach=True, role_tag="attn.v"),
    ApiField("W_o", "tensor<f16>", "Output projection.",
             shape="[D_model, H_q*Dh]", lora_attach=True, role_tag="attn.o"),
)

_GQA = Variant(
    slug="gqa",
    name="GQA",
    description=(
        "Grouped-Query Attention. Multiple Q heads share each K/V head — shrinks "
        "KV cache by num_heads/num_kv_heads. The dominant SLM choice."
    ),
    spec=GQASpec(num_heads=32, head_dim=128, num_kv_heads=8),
    api=ApiSchema(
        inputs=_GQA_INPUTS,
        outputs=_GQA_OUTPUTS,
        attributes=derive_api_attributes(GQASpec),
        weights=_GQA_WEIGHTS,
    ),
)


# ---- MLA Variant ------------------------------------------------------------
# Source: registry/variants.ts:222-304

_MLA_INPUTS: tuple[ApiField, ...] = (
    ApiField("query", "tensor<f16>", "Q is internally split into nope + rope parts.",
             shape="[B, S, H*(nope_head_dim + rope_head_dim)]"),
    ApiField("key", "tensor<f16>", "K reconstructed from latent; rope part shared across heads.",
             shape="[B, S, H*(nope_head_dim + rope_head_dim)]"),
    ApiField("value", "tensor<f16>", "V reconstructed from latent.",
             shape="[B, S, H*nope_head_dim]"),
    ApiField("kv_cache_in", "tensor<f16>", "Latent cache plus shared RoPE'd K — small.",
             shape="[B, S_cache, latent_dim + rope_head_dim]", optional=True),
)

_MLA_OUTPUTS: tuple[ApiField, ...] = (
    ApiField("output", "tensor<f16>", "Attention output.",
             shape="[B, S, H*nope_head_dim]"),
    ApiField("kv_cache_out", "tensor<f16>", "Updated latent cache.",
             shape="[B, S_cache + S, latent_dim + rope_head_dim]"),
)

_MLA_WEIGHTS: tuple[ApiField, ...] = (
    ApiField("W_dq", "tensor<f16>", "Q down-projection to latent. Present only when q_lora_rank is set.",
             shape="[q_lora_rank, D_model]", lora_attach=True, role_tag="attn.q_down", optional=True),
    ApiField("W_uq", "tensor<f16>", "Q up-projection from latent. Present only when q_lora_rank is set.",
             shape="[H*(nope+rope), q_lora_rank]", lora_attach=True, role_tag="attn.q_up", optional=True),
    ApiField("W_q",  "tensor<f16>", "Full-rank Q projection (used when q_lora_rank is None — DeepSeek-V2-Lite).",
             shape="[H*(nope+rope), D_model]", lora_attach=True, role_tag="attn.q", optional=True),
    ApiField("W_dkv", "tensor<f16>", "KV down-projection: produces latent + shared RoPE-K.",
             shape="[latent_dim + rope_head_dim, D_model]", lora_attach=True, role_tag="attn.kv_down"),
    ApiField("W_ukv", "tensor<f16>", "KV up-projection from latent (produces K_nope and V).",
             shape="[H*(nope_head_dim + nope_head_dim), latent_dim]", lora_attach=True, role_tag="attn.kv_up"),
    ApiField("W_o", "tensor<f16>", "Output projection.",
             shape="[D_model, H*nope_head_dim]", lora_attach=True, role_tag="attn.o"),
)

_MLA = Variant(
    slug="mla",
    name="MLA",
    description=(
        "Multi-head Latent Attention (DeepSeek). K and V reconstructed at compute "
        "time from a compressed latent — KV cache stores latent + shared RoPE-K only."
    ),
    spec=MLASpec(num_heads=16, head_dim=192, num_kv_heads=16,
                 latent_dim=512, nope_head_dim=128, rope_head_dim=64),
    api=ApiSchema(
        inputs=_MLA_INPUTS,
        outputs=_MLA_OUTPUTS,
        attributes=derive_api_attributes(MLASpec),
        weights=_MLA_WEIGHTS,
    ),
)


# ---- GQA+Sinks Variant ------------------------------------------------------
# Source: registry/variants.ts:305-398

_GQA_SINKS_INPUTS: tuple[ApiField, ...] = (
    ApiField("query", "tensor<f16>", "Query projection output.",
             shape="[B, S, H_q*Dh]"),
    ApiField("key",   "tensor<f16>", "Key projection output.",
             shape="[B, S, H_kv*Dh]"),
    ApiField("value", "tensor<f16>", "Value projection output.",
             shape="[B, S, H_kv*Dh]"),
    ApiField("sinks", "tensor<f16>",
             "Per-head sink logits added into softmax denominator. Learned weight.",
             shape="[H_q]"),
    _MASK_INPUT,
    ApiField("kv_cache_in", "tensor<f16>", "Cache.",
             shape="[2, B, H_kv, S_cache, Dh]", optional=True),
)

_GQA_SINKS_OUTPUTS: tuple[ApiField, ...] = (
    ApiField("output", "tensor<f16>", "Attention output.",
             shape="[B, S, H_q*Dh]"),
    ApiField("kv_cache_out", "tensor<f16>", "Updated cache.",
             shape="[2, B, H_kv, S_cache + S, Dh]"),
)

_GQA_SINKS_WEIGHTS: tuple[ApiField, ...] = (
    ApiField("W_q", "tensor<f16>", "Query projection.",
             shape="[H_q*Dh, D_model]", lora_attach=True, role_tag="attn.q"),
    ApiField("W_k", "tensor<f16>", "Key projection.",
             shape="[H_kv*Dh, D_model]", lora_attach=True, role_tag="attn.k"),
    ApiField("W_v", "tensor<f16>", "Value projection.",
             shape="[H_kv*Dh, D_model]", lora_attach=True, role_tag="attn.v"),
    ApiField("W_o", "tensor<f16>", "Output projection.",
             shape="[D_model, H_q*Dh]", lora_attach=True, role_tag="attn.o"),
    ApiField("sink_logits", "tensor<f16>",
             "Per-head trainable sink logit added into softmax denominator.",
             shape="[H_q]", lora_attach=False, role_tag="attn.sinks"),
)

_GQA_SINKS = Variant(
    slug="gqa-sinks",
    name="GQA+Sinks",
    description=(
        "GQA with a learned per-head additive bias inside the softmax denominator "
        "(GPT-OSS-20B). Stabilises long-context generation at minimal parameter cost."
    ),
    spec=GQASinksSpec(num_heads=64, head_dim=64, num_kv_heads=8, sliding_window=128),
    api=ApiSchema(
        inputs=_GQA_SINKS_INPUTS,
        outputs=_GQA_SINKS_OUTPUTS,
        attributes=derive_api_attributes(GQASinksSpec),
        weights=_GQA_SINKS_WEIGHTS,
    ),
)


COMPONENT: ComponentMeta = ComponentMeta(
    slug="attention",
    name="Attention",
    oneliner="Scaled dot-product attention — discriminated union over MHA, GQA, MLA, GQA+Sinks.",
    status="live",
    badges=("core", "decoder-only"),
    variants=(_MHA, _GQA, _MLA, _GQA_SINKS),
)


__all__ = [
    "MHASpec", "GQASpec", "MLASpec", "GQASinksSpec", "AttentionSpec", "COMPONENT",
]
