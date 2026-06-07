"""Frozen dataclasses defining the parameter spaces for each layer component.

No behaviour — pure data. Behaviour lives in api/{norm,rope,attention,...}.py.
"""
from __future__ import annotations
from dataclasses import dataclass
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
class YarnRoPEParams:
    """YARN RoPE parameters (DeepSeek-V2 / V3 family).

    YARN blends two inv_freq tables based on a per-channel linear ramp between
    `beta_fast` (extrapolation boundary) and `beta_slow` (interpolation boundary):

        pos_freqs = base ** (arange(0, dim_rot, 2) / dim_rot)
        inv_freq_extrap = 1 / pos_freqs
        inv_freq_interp = 1 / (factor * pos_freqs)
        low, high = find_correction_range(beta_fast, beta_slow, dim_rot,
                                          base, original_max_position_embeddings)
        ramp = clamp((arange(dim_rot/2) - low) / (high - low), 0, 1)
        inv_freq_extrap_factor = 1 - ramp
        inv_freq = inv_freq_interp * (1 - inv_freq_extrap_factor)
                 + inv_freq_extrap * inv_freq_extrap_factor

    cos/sin are then multiplied by an `attention_factor` derived from
    `factor`, `mscale`, `mscale_all_dim`:

        get_mscale(scale, mscale) = 1.0 if scale <= 1
                                    else 0.1 * mscale * log(scale) + 1.0
        attention_factor = get_mscale(factor, mscale)
                         / get_mscale(factor, mscale_all_dim)

    Source: `transformers/modeling_rope_utils.py::_compute_yarn_parameters`
    (lines 327-459).

    For DeepSeek-V2-Lite the canonical values are:
        factor=40, beta_fast=32, beta_slow=1, mscale=0.707, mscale_all_dim=0.707,
        original_max_position_embeddings=4096.
    """
    factor: float
    original_max_position_embeddings: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0       # 0 disables the "all_dim" denominator
    truncate: bool = True


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
    yarn_extra: Optional[YarnRoPEParams] = None

    # B0.5: partial-rotary (Gemma 4 global = 0.25; Phi-3 legacy = 0.5; MLA = qk_rope/qk_head)
    partial_rotary_factor: float = 1.0

    # B2a: two distinct partial-rotary semantics in the wild:
    #
    # - "prefix" (Phi-3 / Phi-4 / most models): cos/sin tables are built at
    #   `head_dim * partial_rotary_factor` channels. apply_rotary_pos_emb
    #   splits q into [..., :rot_dim] (rotated) and [..., rot_dim:] (pass-through),
    #   rotates the first prefix via rotate_half pairing INSIDE the prefix
    #   (i ↔ i + rot_dim/2), then concats. Source: modeling_phi3.py:199-204.
    #
    # - "proportional" (Gemma 4 global): cos/sin tables are built at FULL
    #   head_dim. inv_freq has real frequencies in the first
    #   `int(pr * head_dim / 2)` slots and zeros in the rest. rotate_half
    #   pairs across the FULL head_dim (i ↔ i + head_dim/2). Zero-inv_freq
    #   channels have cos=1, sin=0 so they pass through but STILL participate
    #   in the rotate_half pairing. Source: modeling_gemma4.py:787-806.
    #
    # Default is "prefix" — backward-compat means partial_rotary_factor=1.0
    # makes the two indistinguishable.
    partial_rotary_kind: str = "prefix"   # "prefix" | "proportional"


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

    # B2b: MLA-specific dims (MiniCPM-3 / DeepSeek-V2 / V3 family). When
    # `kind == AttentionKind.MLA`, the following five fields are REQUIRED.
    # For non-MLA they MUST be None and `head_dim` carries the normal Dh.
    #
    # Source (MiniCPM-3): modeling_minicpm.py:351-357
    #   q_lora_rank, qk_rope_head_dim, kv_lora_rank, v_head_dim = hidden_size // num_attention_heads,
    #   qk_nope_head_dim, q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    # Source (DeepSeek-V2): modeling_deepseek_v2.py:300-305 (same five fields).
    #
    # Note that for MLA, the per-head Q/K assembly dim is qk_nope_head_dim +
    # qk_rope_head_dim, while V's per-head dim is v_head_dim — usually NOT
    # equal. The attention scale is `(qk_nope_head_dim + qk_rope_head_dim) ** -0.5`
    # per modeling_deepseek_v2.py:335 / modeling_minicpm.py:387.
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None


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
    """MoE channel mixer.

    Two router families are supported:

    - "softmax" (DeepSeek-V2): hidden_states is cast to fp32, gate weight is
      cast to fp32, F.linear produces router_logits, softmax along expert dim
      yields probabilities, top-k is taken on those probabilities. Optional
      group-limited greedy routing (when `group_routing` is set) takes per-
      group max scores, picks top-k groups, masks out non-selected groups
      with 0, then top-k within. After top-k, weights are multiplied by
      `routed_scaling_factor`. `norm_topk_prob` (V2 typically False) renorms
      top-k weights to sum-1 BEFORE the routed_scaling_factor multiply.
      Source: `transformers/models/deepseek_v2/modeling_deepseek_v2.py:100-130`.

    - "sigmoid_plus_bias" (DeepSeek-V3): router_logits = F.linear(x_fp32,
      weight_fp32). `router_probs = sigmoid(router_logits)`. For top-k
      selection a SECOND tensor `probs_for_choice = router_probs +
      e_score_correction_bias` is built; group-limited routing operates on
      `probs_for_choice` (groups score = sum of the top-2 entries per group);
      group mask is applied; final top-k indices are picked from
      `probs_for_choice` but the actual WEIGHTS gathered are from the
      bias-free `router_probs`. Renormalized if `router_norm=True` (V3 sets
      this True), then multiplied by `routed_scaling_factor`.
      Source: `transformers/models/deepseek_v3/modeling_deepseek_v3.py:194-237`.

    For both, `n_shared_experts > 0` means a single dense FFN-shaped block of
    `intermediate_size = moe_intermediate_size * n_shared_experts` is run on
    the residual stream (the SAME `x` that flowed into the gate, NOT the
    routed output's input) and added to the routed output. Source: same
    files, line 122-130 (V2) / 239-247 (V3).
    """
    n_experts: int
    top_k: int
    n_shared_experts: int = 0
    router_kind: str = "softmax"      # "softmax" | "sigmoid_plus_bias"
    router_norm: bool = False         # V3 norm_topk_prob: renorm top-k weights to sum 1
    score_correction_bias: bool = False  # V3 e_score_correction_bias buffer
    group_routing: Optional[GroupRoutingSpec] = None
    routed_scaling_factor: float = 1.0
    expert_ffn: Optional[FFNSpec] = None


@dataclass(frozen=True)
class IndexerSpec:
    """DeepSeek-V3.2 DSA Lightning Indexer (shape-only).

    DSA (DeepSeek Sparse Attention) adds a lightweight 'indexer' head whose
    only job is to score every (query, key) pair cheaply via a separate
    indexer Q/K projection (smaller `indexer_dim` than the main head_dim).
    The indexer score tensor is top-k'd along the key axis; the main SDPA
    is then run on only the top-k keys per query.

    For B5 the indexer forward is NOT implemented — the spec is wired into
    the AttentionKind.DSA composition path and `Attention.forward` raises
    NotImplementedError. `warmup_tokens` is the number of training tokens
    used before activating DSA (informational; not used at inference).

    Source: DeepSeek-V3.2 release notes (no upstream HF model file as of
    transformers 4.50; the architectural family is reserved here).
    """
    indexer_dim: int
    top_k: int
    warmup_tokens: int = 0


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
    channel_mixer: Union[FFNSpec, "MoESpec"]   # FFNSpec (dense) | MoESpec (MoE)
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
