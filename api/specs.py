"""Frozen dataclasses defining the parameter spaces for each layer component.

No behaviour — pure data. Behaviour lives in api/{norm,rope,attention,...}.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from api import types


@dataclass(frozen=True)
class NormSpec:
    kind: types.NormKind
    eps: float
    weight_mode: types.NormWeightMode = types.NormWeightMode.STANDARD_W


@dataclass(frozen=True)
class Llama3RoPEParams:
    """Llama-3 smooth-scaling RoPE parameters."""
    factor: float                       # e.g. 8.0
    low_freq_factor: float              # e.g. 1.0
    high_freq_factor: float             # e.g. 4.0
    original_context_length: int        # e.g. 8192


@dataclass(frozen=True)
class LongRoPEParams:
    """LongRoPE (Phi-3 / Phi-4) parameters.

    Two scale-factor vectors of length ``head_dim_rot/2`` and a switching point
    at ``original_max_position_embeddings``. inv_freq is computed per-position
    as ``1 / (ext_factor * base ** (2i/head_dim_rot))`` where ``ext_factor`` is
    ``short_factor`` for positions ≤ ``original_max_position_embeddings`` and
    ``long_factor`` otherwise. cos/sin are multiplied by ``attention_factor``.

    Source: `transformers/modeling_rope_utils.py::_compute_longrope_parameters`
    (lines 462-547).
    """
    short_factor: tuple[float, ...]
    long_factor: tuple[float, ...]
    original_max_position_embeddings: int
    attention_factor: float = 1.0


@dataclass(frozen=True)
class RoPESpec:
    base_theta: float
    basis: types.RoPEBasis
    scaling: types.RoPEScaling = types.RoPEScaling.NONE
    scale_factor: Optional[float] = None
    llama3_extra: Optional[Llama3RoPEParams] = None
    longrope_extra: Optional[LongRoPEParams] = None

    # B0.5: partial-rotary (Gemma 4 global = 0.25; Phi-3 legacy = 0.5; MLA = qk_rope/qk_head)
    partial_rotary_factor: float = 1.0


@dataclass(frozen=True)
class AttentionSpec:
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    kind: types.AttentionKind
    qkv_layout: types.QKVLayout
    mask_kind: types.MaskKind

    q_bias: bool = False
    k_bias: bool = False
    v_bias: bool = False
    o_bias: bool = False

    attn_scale: Optional[float] = None   # overrides 1/sqrt(head_dim) (Gemma)
    logit_softcap: Optional[float] = None
    sliding_window: Optional[int] = None

    qk_norm: Optional[NormSpec] = None
    qk_norm_phase: types.QKNormPhase = types.QKNormPhase.NONE
    qk_norm_shape: types.QKNormShape = types.QKNormShape.NONE

    rope: Optional[RoPESpec] = None

    # B0.5: Gemma 4 additions (all backward-compatible — defaults match v1 behavior)
    attention_k_eq_v: bool = False             # Gemma 4 12B/26B/31B global: K and V projections aliased
    qk_norm_fixed_scale: Optional[float] = None # Gemma 4 fixed-scale gain absorbing 1/sqrt(Dh); when set, attn_scale should be 1.0

    # B0.6: Gemma 4 v_norm. A per-head RMSNorm applied to V after the V
    # projection, before transpose+cache.write. On HF Gemma 4 it is
    # `Gemma4RMSNorm(head_dim, with_scale=False)`, i.e. a unit RMSNorm with
    # no learnable weight (verified at modeling_gemma4.py:1215, 1265).
    # When set with `with_scale=False`, the runtime instantiates a frozen
    # ones-vector that is NOT in the state dict.
    v_norm: Optional[NormSpec] = None
    v_norm_with_scale: bool = True             # False for Gemma 4 (unit RMSNorm)


@dataclass(frozen=True)
class FFNSpec:
    intermediate_size: int
    activation: types.Activation
    gate_kind: types.GateKind
    fused_gate_up: bool = False
    gate_bias: bool = False
    up_bias: bool = False
    down_bias: bool = False


@dataclass(frozen=True)
class QuantSpec:
    qdtype: types.QDType
    group_size: Optional[int]            # None=per-tensor, -1=per-channel, N=blockwise
    quant_axis: int
    scale_dtype: torch.dtype
    has_zero_point: bool
    packing: types.PackingLayout
    accumulator_dtype: torch.dtype
    role: types.QuantRole = types.QuantRole.WEIGHT
    codebook: Optional[object] = None    # CodebookSpec — defined post-M1


@dataclass(frozen=True)
class KVCacheSpec:
    layout: types.CacheLayout
    memory_layout: types.MemoryLayout
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    ownership: types.CacheOwnership = types.CacheOwnership.EXPLICIT_PASS
    block_size: Optional[int] = None
    k_quant: Optional[QuantSpec] = None
    v_quant: Optional[QuantSpec] = None

    # B0.5: cross-layer sharing
    share_scheme: types.ShareScheme = types.ShareScheme.NONE
    num_kv_shared_layers: int = 0


@dataclass(frozen=True)
class ConvSpec:
    """1D causal conv (Mamba)."""
    kernel_size: int
    bias: bool = True
    activation: Optional[types.Activation] = None


@dataclass(frozen=True)
class SSMSpec:
    """Mamba-1 SSM (state-space) parameters."""
    d_state: int
    d_conv: int
    d_inner: int
    expand_factor: int = 2
    dt_rank: int = -1                  # -1 means "auto" (= hidden // 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True


@dataclass(frozen=True)
class SSDSpec:
    """Mamba-2 SSD form. Composition over SSMSpec via base; not inheritance, to keep frozen semantics clean."""
    base: SSMSpec
    chunk_size: int = 256
    headdim: int = 64
    ngroups: int = 1


@dataclass(frozen=True)
class GroupRoutingSpec:
    """DeepSeek-V3 group-limited routing."""
    n_groups: int
    topk_per_group: int
    routed_expert_grouping: bool = True


@dataclass(frozen=True)
class MoESpec:
    """MoE channel mixer."""
    n_experts: int
    top_k: int
    n_shared_experts: int = 0
    router_kind: str = "softmax"      # "softmax" | "sigmoid_plus_bias"
    router_norm: bool = False
    score_correction_bias: bool = False
    group_routing: Optional[GroupRoutingSpec] = None
    routed_scaling_factor: float = 1.0
    expert_ffn: Optional[FFNSpec] = None


@dataclass(frozen=True)
class LayerScaleSpec:
    """OpenELM-style per-layer scaling - list of overrides indexed by layer."""
    num_q_heads_per_layer: tuple[int, ...]
    num_kv_heads_per_layer: tuple[int, ...]
    ffn_multipliers_per_layer: tuple[float, ...]


@dataclass(frozen=True)
class PLESpec:
    """Per-Layer Embedding spec (Gemma 4 E2B/E4B — axis A19).

    The PLE table has dimension `ple_dim` (typically 256), much smaller than the
    main residual hidden_size. Its output is normalized and injected at every
    decoder layer as a residual term scaled by `residual_scale` (Gemma 4 uses 1/sqrt(2)).

    On Gemma 4 the PLE row for token t is computed as
        ple_row = (token_identity_emb + context_aware_projection) * residual_scale
    and the projection is the layer-local linear that maps from the embedding
    to the hidden_size for that layer's residual injection. Behaviour lives in
    api/embedding.py — this spec is pure data.
    """
    ple_dim: int
    residual_scale: float
    injection_norm: "NormSpec"


@dataclass(frozen=True)
class DecoderBlockSpec:
    attn_norm_position: types.NormPosition
    ffn_norm_position: types.NormPosition
    token_mixer: AttentionSpec           # only AttentionSpec for M1 (Qwen3 dense)
    channel_mixer: FFNSpec               # only FFNSpec for M1
    pre_attn_norm: Optional[NormSpec] = None
    post_attn_norm: Optional[NormSpec] = None
    pre_ffn_norm: Optional[NormSpec] = None
    post_ffn_norm: Optional[NormSpec] = None
    residual_scale: Optional[float] = None
    embedding_scale: Optional[float] = None
    logits_scale: Optional[float] = None

    # B0.5
    per_layer_embedding: Optional[PLESpec] = None
    final_logit_softcap: Optional[float] = None    # Gemma 4 = 30.0 (model-level; carried here for assembly)
