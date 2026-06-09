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

    # v6 A1/A2: LayerNorm bias. Only meaningful when `kind == LAYER`. MPT
    # sets `norm_1.bias = None` (no bias), Falcon-7B's `LayerNorm(.., eps=..)`
    # uses bias=True (default). RMSNorm has no bias concept — this field is
    # ignored for `kind == RMS`. Source:
    #   - modeling_mpt.py:163, 165 (`LayerNorm(hidden_size, eps=..); norm_1.bias = None`)
    #   - modeling_falcon.py:574, 578 (`LayerNorm(hidden_size, eps=...)` -> bias=True)
    has_bias: bool = True


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
class AliBiSpec:
    """ALiBi (Attention with Linear Biases) position encoding.

    The bias added to the (Q @ K.T) scores BEFORE softmax is

        bias[h, i, j] = -m_h * (i - j)              for j <= i (causal)

    where `m_h` are head-specific slopes forming a geometric sequence:

        for h = 1..n where n is a power of 2:
            m_h = 1 / 2 ** (h * alibi_bias_max / n)

    For non-power-of-2 `n_heads`, the MPT impl uses the larger
    power-of-2 slope set then reorders/keeps `n_heads` of them — verified
    against `transformers/models/mpt/modeling_mpt.py:42-62`
    (`build_mpt_alibi_tensor`).

    `slopes` is optional; if None the runtime computes the canonical
    MPT slopes at module-init time. When provided, it must be a tuple of
    length `n_heads`.

    Source: `transformers/models/mpt/modeling_mpt.py:42-62` (MPT slopes)
    and `transformers/models/falcon/modeling_falcon.py:168-193`
    (alternate Falcon/Bloom slopes — NOT used; Falcon-7B uses RoPE).
    """
    n_heads: int
    alibi_bias_max: float = 8.0
    slopes: Optional[tuple[float, ...]] = None


@dataclass(frozen=True)
class RoPESpec:
    base_theta: float
    basis: types.RoPEBasis
    scaling: types.RoPEScaling = types.RoPEScaling.NONE
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

    # B8: M-RoPE (multimodal rotary position embedding) — Qwen2.5-VL family.
    # When `mrope_section` is set, the runtime expects `position_ids` of
    # shape [3, B, S] (channels: temporal, height, width) and slices cos/sin
    # along the head_dim axis using `mrope_section` (cyclically `i % 3`).
    # The tuple lengths must sum to `head_dim // 2` (HF concatenates the
    # split twice before applying rotate_half — see source ref).
    # Source: `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:564-606`
    # (apply_multimodal_rotary_pos_emb).
    mrope_section: Optional[tuple[int, ...]] = None


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

    # B5: DSA (DeepSeek-V3.2) Lightning Indexer. Required when
    # kind == AttentionKind.DSA. Carries the indexer's projection
    # dimension and the per-query top-k.
    indexer: Optional["IndexerSpec"] = None

    # v5-phase2 V1: ALiBi position encoding (MPT / Baichuan / Bloom).
    # When set, ALiBi adds head-specific linear position biases to the
    # attention scores BEFORE softmax — no RoPE, no learned positions.
    # ALiBi is MUTUALLY EXCLUSIVE with `rope`: setting both is a config
    # error and raises in Attention.__init__. Source:
    # `transformers/models/mpt/modeling_mpt.py:42-62, 121` (build slopes
    # + add bias before softmax).
    alibi: Optional["AliBiSpec"] = None

    # v5-phase2 V3: BitNet b1.58 sub-norm. RMSNorm applied to the
    # attention output AFTER the per-head sdpa concat-reshape but BEFORE
    # o_proj. Source:
    # `transformers/models/bitnet/modeling_bitnet.py:177, 216` (
    # `self.attn_sub_norm = BitNetRMSNorm(config.hidden_size)`,
    # `attn_output = self.attn_sub_norm(attn_output)`).
    attn_sub_norm: Optional[NormSpec] = None

    # v5-phase2 V4: CLA (Cross-Layer Attention) per-layer KV pointer.
    # When set to a negative int (e.g. -1), the layer at index L uses
    # the K/V projections of layer L+offset (i.e. shares K/V with a
    # preceding layer). Even-indexed layers compute K/V; odd-indexed
    # layers borrow from their predecessor.
    #
    # Architectural expression at the model assembly level: factories
    # set `kv_source_layer_offset=None` on owning layers (and build
    # k_proj/v_proj) and `kv_source_layer_offset=-1` on borrowing
    # layers (where k_proj/v_proj are intentionally NOT built — the
    # caller passes K/V from the predecessor at forward time).
    #
    # Source: Hunyuan-Large `config.json` `use_cla=True`,
    # `cla_share_factor=2`. The HF v5.10.2 `hunyuan_v1_dense` /
    # `hunyuan_v1_moe` modeling files do NOT carry this spec hook —
    # CLA is exercised only by the Hunyuan-Large public release.
    kv_source_layer_offset: Optional[int] = None

    # v5-phase2 V5: trained attention sinks (GPT-OSS family). When set,
    # the attention forward appends `n_sink_tokens` learnable per-head
    # logits to the (B, Hq, S, T) scores BEFORE softmax, then drops
    # them post-softmax — so each head spends some softmax mass on a
    # "trained sink" slot. Source:
    # `transformers/models/gpt_oss/modeling_gpt_oss.py:309, 267-275`
    # (`self.sinks = nn.Parameter(torch.empty(num_attention_heads))` /
    # `combined_logits = cat([attn_weights, sinks], -1); probs = softmax(
    # combined_logits); scores = probs[..., :-1]`).
    #
    # The HF impl uses a SINGLE learnable slot per head (so
    # n_sink_tokens=1 is the canonical value); the IR exposes the count
    # for forward-compat with multi-sink-slot variants.
    n_sink_tokens: Optional[int] = None

    # B8: When True, the attention forward expects an optional
    # `vision_token_count` int. The keep mask becomes:
    #   - vision keys (j < V): bidirectional within the vision block
    #     (every vision query and every text query may attend to them).
    #     Vision queries (i < V) attend to ALL vision keys but NOT to
    #     any text key (since text keys come after vision in the seq).
    #   - text keys (j >= V): standard causal (j <= start_pos + i),
    #     attended only by text queries (i >= V).
    # When False (default), the standard causal-or-SWA path is used.
    # This is the "Visual Causal Flow" mask of the original DeepSeek-OCR
    # paper. NOTE: HF v5.10.2 deepseek_ocr2 implements STANDARD causal —
    # this flag is reserved for the v3-spec hook and exercised in B8 by
    # a shape-only test, not by the numerical gate.
    block_bidirectional_mask: bool = False


@dataclass(frozen=True)
class FFNSpec:
    intermediate_size: int
    activation: types.Activation
    gate_kind: types.GateKind
    fused_gate_up: bool = False
    gate_bias: bool = False
    up_bias: bool = False
    down_bias: bool = False

    # v5-phase2 V3: BitNet b1.58 sub-norm. RMSNorm applied to the gated
    # activation `act(gate(x)) * up(x)` BEFORE the down_proj.
    # Source: `transformers/models/bitnet/modeling_bitnet.py:74, 77`
    # (`self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size)` /
    # `self.down_proj(self.ffn_sub_norm(self.act_fn(self.gate_proj(x))
    # * self.up_proj(x)))`).
    ffn_sub_norm: Optional[NormSpec] = None


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


@dataclass(frozen=True)
class KVCacheSpec:
    layout: types.CacheLayout
    memory_layout: types.MemoryLayout
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    ownership: types.CacheOwnership = types.CacheOwnership.EXPLICIT_PASS


@dataclass(frozen=True)
class ConvSpec:
    """1D causal depthwise conv used by the Mamba family front-end.

    For Mamba-2 the conv kernel size is `conv_kernel` (default 4 per
    `Mamba2Config.conv_kernel`), padded to `kernel_size-1` on the left so
    the output spans the same seq_len AFTER `[..., :seq_len]` slicing.
    Implemented as a SINGLE depthwise Conv1d with groups equal to the
    channel count, NOT a multi-channel conv. Source:
    `modeling_mamba2.py:155-162` (Mamba2Mixer init).
    """
    kernel_size: int
    bias: bool = True
    activation: Optional[types.Activation] = None


@dataclass(frozen=True)
class SSMSpec:
    """Mamba-1 SSM (state-space) parameters.

    Used directly by Mamba-1 models (v6 B1 forward); embedded as
    `SSDSpec.base` for Mamba-2.

    For Mamba-2, several of these fields carry duplicate meaning at the
    SSDSpec layer (`d_inner`, `d_state`, `d_conv`); the SSDSpec is the
    authoritative source for those shapes in the SSD path, and the SSMSpec
    inside is a convenient carrier for the per-head time-step / activation
    defaults.

    Source: `modeling_mamba.py:58-120` (MambaMixer init — Mamba-1) and
    `modeling_mamba2.py:121-220` (Mamba2Mixer init — Mamba-2).
    """
    d_state: int                       # state_size in HF config (default 16 for Mamba-1, 128 for Mamba-2)
    d_conv: int                        # conv_kernel in HF config (default 4)
    d_inner: int                       # intermediate_size = hidden * expand
    expand_factor: int = 2
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    activation: types.Activation = types.Activation.SILU

    # v6 B1: SSMKind dispatcher. MAMBA2_SSD is implied when this SSMSpec is
    # carried inside an SSDSpec (B7 SSD path doesn't read this field).
    # MAMBA1 selects the selective-scan reference forward + Mamba-1 shapes
    # (per-channel A_log, learned dt_proj). Default is MAMBA2_SSD for
    # backward-compat with the B7 SSDSpec carriers.
    kind: types.SSMKind = types.SSMKind.MAMBA2_SSD

    # v6 B1: Mamba-1 dt-rank (delta time step). Required when kind=MAMBA1.
    # `dt_proj` is `Linear(dt_rank, d_inner, bias=True)`.
    # Source: modeling_mamba.py:73, 96.
    dt_rank: Optional[int] = None

    # v6 B1: Jamba adds intra-mixer norms to dt, B, C. Each is a RMSNorm
    # on the per-token vector (dim = dt_rank, d_state, d_state). When None,
    # no norms are inserted. Source: modeling_jamba.py:249-251, 324-326.
    dt_layernorm: Optional[NormSpec] = None
    b_layernorm: Optional[NormSpec] = None
    c_layernorm: Optional[NormSpec] = None


@dataclass(frozen=True)
class SSDSpec:
    """Mamba-2 SSD form parameters.

    Composes an `SSMSpec` (carrying the per-head defaults and activation)
    with the Mamba-2-specific multi-head + grouped-B/C shape parameters.

    Mamba-2 layout (source `modeling_mamba2.py:121-220`):
    - `num_heads` (`n_heads` here): number of SSM heads (default 128 for 2.7B).
    - `head_dim`: per-head dim (default 64). Invariant:
      `n_heads * head_dim == hidden_size * expand == d_inner`.
      Validated by `Mamba2Config.validate_architecture()`.
    - `n_groups`: how many groups of (B, C) share the recurrence (default 8).
      `num_heads` MUST be divisible by `n_groups`; B and C are tiled
      `num_heads // n_groups` times across the head axis at scan time
      (`modeling_mamba2.py:510-511`).
    - `chunk_size`: chunk length for SSD chunk-parallel scan (default 256).
      The naive PyTorch fallback (`torch_forward`, lines 503-577) pads the
      sequence to a multiple of `chunk_size`.
    - `time_step_limit`: (low, high) clamp applied AFTER softplus(dt+dt_bias)
      (line 506). Default (0.0, inf) — i.e. no upper clamp.
    - `layer_norm_epsilon`: shared with the GATED RMS norm before out_proj
      (line 179).
    - `use_bias` (carried via base.bias): bias of in_proj and out_proj.
    - `use_conv_bias` (carried via base.conv_bias): bias of conv1d.
    """
    base: SSMSpec
    chunk_size: int = 256
    headdim: int = 64
    ngroups: int = 1
    n_heads: int = 1
    time_step_limit_low: float = 0.0
    time_step_limit_high: float = float("inf")
    layer_norm_epsilon: float = 1e-5


@dataclass(frozen=True)
class GroupRoutingSpec:
    """DeepSeek-V3 group-limited routing."""
    n_groups: int
    topk_per_group: int


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
    router_kind: str = "softmax"      # "softmax" | "sigmoid_plus_bias" | "topk_then_softmax_with_bias"
    router_norm: bool = False         # V3 norm_topk_prob: renorm top-k weights to sum 1
    group_routing: Optional[GroupRoutingSpec] = None
    routed_scaling_factor: float = 1.0
    expert_ffn: Optional[FFNSpec] = None

    # v6 A3: GPT-OSS expert layout — clamped SwiGLU with biased linears.
    # When True, the expert forward uses the GPT-OSS clamped SwiGLU math:
    #   gate, up = gate_up[..., ::2], gate_up[..., 1::2]   # INTERLEAVED
    #   gate.clamp_(max=expert_clamp_limit)
    #   up.clamp_(min=-expert_clamp_limit, max=expert_clamp_limit)
    #   glu = gate * sigmoid(gate * expert_swiglu_alpha)
    #   gated = (up + 1) * glu
    # and the experts have biases on BOTH gate_up_proj and down_proj.
    # Source: `transformers/models/gpt_oss/modeling_gpt_oss.py:73-119`
    # (GptOssExperts with `has_bias=True`, `is_transposed=True`,
    # `is_concatenated=False`, `alpha=1.702`, `limit=7.0`).
    expert_kind: str = "swiglu"        # "swiglu" | "gpt_oss_clamped_swiglu"
    expert_bias: bool = False
    expert_swiglu_alpha: float = 1.702
    expert_clamp_limit: float = 7.0


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
    NotImplementedError.

    Source: DeepSeek-V3.2 release notes (no upstream HF model file as of
    transformers 4.50; the architectural family is reserved here).
    """
    indexer_dim: int
    top_k: int


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
    # B7: token_mixer broadened to a Union of AttentionSpec | SSDSpec.
    # SSMSpec (Mamba-1) is reserved but not exercised by the B7 landing.
    # The DecoderBlock dispatches on isinstance at build time.
    # Default is AttentionSpec for all pre-B7 model factories.
    # Source: v3 design spec §5.2.5 — `token_mixer: Union[AttentionSpec,
    # SSMSpec, SSDSpec, ...]`.
    token_mixer: Union["AttentionSpec", "SSMSpec", "SSDSpec"]
    channel_mixer: Union[FFNSpec, "MoESpec"]   # FFNSpec (dense) | MoESpec (MoE)
    pre_attn_norm: Optional[NormSpec] = None
    post_attn_norm: Optional[NormSpec] = None
    pre_ffn_norm: Optional[NormSpec] = None
    post_ffn_norm: Optional[NormSpec] = None
    residual_scale: Optional[float] = None

    # B0.5
    per_layer_embedding: Optional[PLESpec] = None

    # B7: when the token_mixer is an SSM/SSD spec, the FFN may be optional —
    # Mamba-2 has NO FFN sublayer. When `channel_mixer` is None, the block
    # skips the FFN sublayer entirely. For Mamba-2 we set this to None; the
    # block then assembles `residual + ssm(norm(x))` and returns.
    # Source: `modeling_mamba2.py:617-640` (Mamba2Block — only one sublayer).
    skip_ffn: bool = False

    # v5-phase2 V2: SEQUENTIAL (default) vs PARALLEL residual flow.
    # PARALLEL is Falcon-7B / Cohere-style — one shared norm input fed to
    # both attention and FFN, both sublayer outputs summed into a single
    # residual add. In PARALLEL mode `pre_ffn_norm` MUST be None and
    # `ffn_norm_position` MUST be PRE. Source: see types.BlockLayout
    # docstring.
    block_layout: types.BlockLayout = types.BlockLayout.SEQUENTIAL
