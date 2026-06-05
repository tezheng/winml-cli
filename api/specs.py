"""Frozen dataclasses defining the parameter spaces for each layer component.

No behaviour — pure data. Behaviour lives in api/{norm,rope,attention,...}.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

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
class RoPESpec:
    base_theta: float
    basis: types.RoPEBasis
    scaling: types.RoPEScaling = types.RoPEScaling.NONE
    scale_factor: Optional[float] = None
    llama3_extra: Optional[Llama3RoPEParams] = None


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


@dataclass(frozen=True)
class DecoderBlockSpec:
    attn_norm_position: types.NormPosition
    ffn_norm_position: types.NormPosition
    token_mixer: AttentionSpec           # only AttentionSpec for M1 (Qwen3 dense)
    channel_mixer: FFNSpec               # only FFNSpec for M1
    input_norm: NormSpec
    pre_attn_norm: Optional[NormSpec] = None
    post_attn_norm: Optional[NormSpec] = None
    pre_ffn_norm: Optional[NormSpec] = None
    post_ffn_norm: Optional[NormSpec] = None
    residual_scale: Optional[float] = None
    embedding_scale: Optional[float] = None
    logits_scale: Optional[float] = None
