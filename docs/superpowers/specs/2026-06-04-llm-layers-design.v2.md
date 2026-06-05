# llm-layers — Design Spec (v2)

**Date:** 2026-06-04
**Status:** Draft v2, awaiting user review
**Owner:** zhengte@microsoft.com
**Supersedes:** `2026-06-04-llm-layers-design.md` (v1)
**Critique applied:** `research/issues/06-spec-critique.md` (17 v2-fixes), `research/issues/00-master-issues.md` (issues 6.1–6.8)

---

## 0. Survey-paper preamble

The decoder-only transformer has evolved along eleven named axes since GPT-3
(2020). Each axis carries a representative model and a published paper.
**Attention** went MHA → MQA (Shazeer 2019, reintroduced for Falcon-7B 2023)
→ GQA (Ainslie 2023, Llama-2/3) → MLA (DeepSeek-V2 2024) → Differential
(Microsoft 2024) → linear/state-space (Mamba 2023, Mamba-2/SSD 2024, RWKV-7
2025). **Positional encoding** went absolute → ALiBi (2021) → RoPE
(Su 2021) → Position Interpolation (Chen 2023) → NTK-aware (bloc97 2023) →
YaRN (Peng 2023) → LongRoPE (Microsoft 2024) → Llama-3 scaling (Meta 2024)
→ NoPE-alternating (Llama-4, SmolLM3 2025) → iRoPE (Llama-4). **Normalisation**
went LayerNorm → RMSNorm (Zhang & Sennrich 2019, mainstream 2023) → QK-Norm
(Henry 2020, reintroduced Olmo-2/Qwen3 2024–2025) → Gemma `1+w` (2024) →
sandwich/dual pre+post (Gemma 3, OLMo 2). **Activation** went GeLU → SwiGLU
(Shazeer 2020, Llama 2023) → GeGLU (Gemma 2024) → ReLU² (Primer 2021,
BitNet 2024). **Mixture-of-Experts** went vanilla top-k softmax (Switch
Transformer 2021) → Mixtral top-2 (2023) → DeepSeek shared+routed +
sigmoid + auxiliary-loss-free balancing (2024) → group-limited routing
(DeepSeek-V3 2024). **KV cache** went contiguous → PagedAttention (2023) →
PA v2 + vAttention (2024) → CLA pairwise sharing (2024) → YOCO two-stage
producer/consumer (Microsoft 2024) → MLA latent compression (DeepSeek 2024)
→ NSA three-tier (DeepSeek 2025). **Quantization** went RTN INT8 → GPTQ
(2022) → SmoothQuant (2022) → AWQ (2023) → bitsandbytes NF4 (2023) → HQQ
(2024) → FP8 W8A8 (Hopper 2023, mainstream 2024) → MXFP4/MXINT4 (OCP 2024)
→ NVFP4 (Blackwell 2024) → BitNet b1.58 ternary (2024) → AQLM/EXL2/QuIP#
variable-bit (2024) → QuaRot/SpinQuant rotation-then-quantize (2024).
**Hybrid architectures** added a parallel-block axis: Hymba (Mamba‖Attn in
the same block, NVIDIA late 2024), Jamba (interleaved per block, AI21 2024),
Zamba2 (interleaved with shared global attention, 2024), Samba (Phi-4-mini-flash,
2025).

This project takes that lineage as a corpus and asks: **what is the smallest
API that can re-assemble any production small LM from raw weights without
embedding family-specific behaviour?** A model that requires no new
primitive *fits*; a model that requires a new op or a new spec field forces
a deliberate, evidence-justified extension. The final API is the floor that
survives every model surveyed (target ~24 architecturally-distinct families,
all <8B parameters). The API is the IR. Quantization and KV cache are
first-class parameters, not bolted-on layers — both are spec dataclasses
that decoder blocks consume on equal footing with attention and FFN.

We do not pretend this is a finished operator standard. ONNX-runtime,
TRT-LLM, OpenVINO, MLX, llama.cpp, QAIRT, Core ML, MIGraphX, KleidiAI and
MindIE each ship their own decoder primitives at different granularities;
this API targets the **≥95 % consensus floor** documented in
`research/03-ihv-opsets.md` and explicitly names where it punts to building
blocks (MoE dispatch, SSM scan) rather than ops. The deliverable is the
union of: (a) an op floor, (b) a small set of spec dataclasses spanning
fourteen architecture axes, (c) a per-family rollout that turns every
spec field into a tested example. The output is a reusable decoder IR
grounded in three years of decoder evolution, not a leaderboard or a
runtime.

---

## 1. Goal

Produce a **minimal, evidence-grounded API set** capable of assembling any
mainstream small language model (SLM, <8B params) as a transformer decoder
graph — with **quantization** and **KV-cache** as first-class parameters.

The API has three contracts:

1. **Logical-op floor.** A set of ~16 functional primitives that every
   surveyed runtime (≥7/9 of the IHV opsets in `research/03`) either
   exposes directly or implements as a single fused kernel. Decoder blocks
   are expressed entirely as compositions of these ops.
2. **Spec dataclasses.** Twelve `@dataclass(frozen=True)` types that
   capture the parameter spaces along which architectures vary
   (attention, RoPE, norm, FFN, MoE, SSM, KV-cache, quantization, block
   assembly, plus three helper specs). Specs are pure data; behaviour
   lives in the building blocks that consume them.
3. **Per-family proofs.** For each architectural family (~24 total) a
   `models/<family>/` directory containing a layer document, a layer
   factory using only API primitives, and tests. Qwen3 is the seed
   (Milestone 1); the remaining 23 ship in batches B0.5 → B9 (Milestone 2
   onward).

The IR is intended to be *expressible* in PyTorch, GGUF and ONNX-style
runtimes. Design choices follow runtime/IHV consensus where it exists; we
implement only a PyTorch eager reference.

## 2. Non-goals

- **Training or fine-tuning code.** Random weights for shape tests; HF
  weights only for the Qwen3 numerical-equivalence check and for AWQ unpack
  exercises.
- **Full-model inference.** We implement one decoder block per family — not
  tokenisers, samplers, beam search, speculative decoding, sharding, or
  pipelines.
- **Performance kernels.** PyTorch eager only. The API must be
  *expressible* in IHV runtimes; we do not implement them.
- **Bit-exact equivalence across all models.** Phase B targets structural
  and shape equivalence for all families; numerical equivalence is only
  proven for Qwen3 in the M1 kickoff gate.
- **Multi-axis sequence positions (M-RoPE for VL).** `RoPESpec` carries the
  `mrope_section` field for future extension; the M1 ops assume sequence is
  dim-1 of `[B,S,D]`. Vision-language models are out of scope.
- **Encoder-decoder.** Florence-2 and similar architectures are out of
  scope; the API is decoder-only.
- **Distributed / tensor-parallel sharding.** Documented as a runtime
  concern.
- **LoRA mergeability into quantized weights.** Out of scope for M1;
  `QuantSpec` reserves a `merged_lora_rank` field for future extension.
- **Continuous-batching shape `[T,D]`.** The API operates on `[B,S,D]`;
  vLLM/SGLang continuous-batching squashes B×S into T at the runtime layer,
  and the API assumes a logical batch axis is recoverable.

## 3. Sub-projects and sequencing

| Phase | Output | Status |
|---|---|---|
| A. Model census + variant-axis catalog | `research/01-model-census.v2.md` — ~59 models, 26 archs, 15 axes | In v2 |
| A.1 Cross-source layer survey | `research/02-layer-sources.v2.md` — 30+ families, HF/vLLM/llama.cpp triple-source | In v2 |
| A.2 IHV op-set survey | `research/03-ihv-opsets.v2.md` — 13+ runtimes including ggml/Neuron/TPU/DirectML, op consensus floor | In v2 |
| A.3 Quantization survey | `research/04-quantization.v2.md` — 42 schemes, 22-axis parameter space | In v2 |
| A.4 KV-cache + attention survey | `research/05-kvcache-attention.v2.md` — 25 attn × 14 RoPE × 12 cache | In v2 |
| A.5 Evolution synthesis | `research/00-evolution.md` — timeline 2022→2026, lineage, abandoned designs | In v2 |
| B. Per-model layer reference impls | `models/<family>/{layer.md, layer.py, test_layer.py}` ×~24 | Plan output |
| C. Minimal-API synthesis | `api/` package (the IR) | Plan output |

Sequencing: A is done in v2 form. B and C **co-evolve** in execution — each
new model in B either confirms or extends C. The Qwen3 sample (Section 8)
is the seed that bootstraps both.

## 4. Project layout

```
C:\Users\zhengte\external\llm-layers\
├── api/                            # Phase C — the minimal API (the IR)
│   ├── __init__.py                 # public surface
│   ├── ops.py                      # 16 logical primitives (functional)
│   ├── specs.py                    # 12 spec dataclasses (see §5.2)
│   ├── attention.py                # Attention block (standard, MLA, linear, differential)
│   ├── ssm.py                      # SSM/SSD/conv1d/selective_scan building blocks
│   ├── feedforward.py              # FFN + MoE (parameterized)
│   ├── norm.py                     # RMSNorm + LayerNorm + ScaleNorm + QK-norm variants
│   ├── rope.py                     # RoPE variants (vanilla / Llama-3 / YaRN / LongRoPE / PI / DynamicNTK / NoPE / iRoPE / partial / M-RoPE)
│   ├── kvcache.py                  # KVCache type + layouts (contiguous / paged / ring / MLA-latent / SSM-state / YOCO-shared)
│   ├── quant.py                    # QuantSpec realization + dtype casts + packed-int helpers (AWQ/GPTQ/HQQ/GGUF/MXFP4/NF4/IQ/FP8)
│   ├── block.py                    # DecoderBlock assembly (token mixer + channel mixer + residual structure)
│   ├── weights.py                  # HF/GGUF → API tensor-slot mapping (see §9)
│   └── README.md                   # API quick reference
├── models/                         # Phase B — per-model layers
│   ├── qwen3/{layer.md, layer.py, weight_loader.py, test_layer.py}
│   ├── llama3/...
│   ├── llama2/...                  # B0.5 baseline
│   ├── mistral_v01/...             # B0.5 baseline
│   ├── gemma3/...
│   ├── phi3/...
│   ├── deepseek_v3/...             # MLA + MoE + group-routing
│   ├── minicpm3/...                # MLA at small scale
│   ├── mixtral/...                 # MoE classic
│   ├── jamba/...                   # SSM-hybrid (Mamba+Attn interleaved)
│   ├── hymba/...                   # SSM-hybrid (Mamba‖Attn parallel)
│   ├── rwkv7/...                   # WKV recurrence
│   ├── mamba2/...                  # SSD formulation
│   ├── openelm/...                 # per-layer head-count overrides
│   └── ... (~24 total — see §10)
├── research/                       # Already populated by parallel agents (v2)
│   ├── 00-evolution.md             # NEW: cross-axis evolution narrative
│   ├── 01-model-census.v2.md
│   ├── 02-layer-sources.v2.md
│   ├── 03-ihv-opsets.v2.md
│   ├── 04-quantization.v2.md
│   ├── 05-kvcache-attention.v2.md
│   └── issues/                     # critique inputs
├── docs/superpowers/specs/
│   ├── 2026-06-04-llm-layers-design.md      # v1 (archived)
│   └── 2026-06-04-llm-layers-design.v2.md   ← this file
├── plans/                          # Implementation plan (from writing-plans skill)
├── pyproject.toml                  # uv-managed venv
├── uv.lock
└── README.md
```

Python env: `uv venv && uv sync` (Python 3.11+). Runtime dependencies:
`torch`, `safetensors`, `numpy`. Test dependencies: `pytest`, `transformers`
(numerical-equivalence cross-checks only). Dev dependencies: `ruff`,
`mypy` (strict on `api/`, lenient on `models/`), `black`. See §16 for the
quality bar.

## 5. Phase C — Minimal API design framework

The API has four layers:

1. **Logical ops** (functional primitives) — IHV-consensus floor (~16 ops)
2. **Specs** (dataclasses) — twelve frozen dataclasses spanning the
   parameter spaces that vary across models
3. **Building blocks** (small classes) — Attention, SSM, FFN, MoE, RoPE,
   Norm, KVCache; consume specs
4. **DecoderBlock** — assembles a token mixer + channel mixer with residual
   structure

### 5.1 Logical ops — the IHV-consensus floor

This expands v1's nine ops to **sixteen** to cover MoE routing (B6 batch)
and SSM scans (B7 batch). Each op is a pure function in `api/ops.py`. The
consensus column refers to the 13-runtime survey in `research/03-ihv-opsets.v2.md`
(OpenVINO, QAIRT, FastFlowLM, MIGraphX, TensorRT-LLM, Core ML, MLX,
KleidiAI, MindIE, ggml, AWS Neuron, TPU XLA, DirectML).

| # | Op | Signature | IHV consensus | Used by |
|---|---|---|---|---|
| 1 | `rms_norm(x, weight, eps, mode)` | `x:[B,S,D] → [B,S,D]`; `mode ∈ {classic, gemma_1plus_w, scale_norm}` | 13/13 | all |
| 2 | `layer_norm(x, weight, bias, eps)` | `x:[B,S,D] → [B,S,D]` | 13/13 | OLMo 1, MPT, Falcon |
| 3 | `linear(x, weight, bias?, quant?)` | `x:[B,S,K] × w:[N,K] → [B,S,N]`; weight may be a `QuantizedWeight` carrying scales/zeros/codebook | 13/13 | all |
| 4 | `rope_apply(qk, freqs, kind, partial_dim?)` | `qk:[B,S,H,Dh] → [B,S,H,Dh]`; `kind ∈ {standard, partial, m_rope}` | 9/13 explicit, 4/13 fused into attention | all attention models |
| 5 | `sdpa(q, k, v, mask, scale, softcap?, sink?, sliding_window?)` | `[B,H,S,Dh] × [B,Hk,Sk,Dh] × [B,Hk,Sk,Dv] → [B,H,S,Dv]`; fused softmax inside | 13/13 fused | all attention models |
| 6 | `softmax(x, axis, dtype?)` | `[..., N, ...] → [..., N, ...]`; explicit, used for MoE router and any standalone softmax | 13/13 | MoE router, sampling adapters |
| 7 | `top_k(scores, k, axis, sorted)` | returns `(values, indices)` of top-k entries along `axis` | 11/13 explicit, 2/13 via custom plugin (TRT-LLM `topkLastDimPlugin`) | MoE, NSA |
| 8 | `gather(table, indices, axis)` | `table[..., N, ...] × indices[..., M, ...] → [..., M, ...]` | 13/13 | MoE token dispatch, codebook embedding |
| 9 | `scatter(input, indices, updates, axis, reduction)` | inverse of `gather`; `reduction ∈ {none, add}` | 12/13 | MoE token combine |
| 10 | `conv1d(x, weight, bias?, groups, padding, causal)` | `x:[B,S,D] → [B,S,D]`; supports depthwise (`groups=D`) and causal padding | 12/13 (TRT-LLM `mambaConv1dPlugin` named fused) | Mamba, Hymba, RecurrentGemma |
| 11 | `selective_scan(x, A_log, B, C, D, dt, initial_state?, chunk_size?)` | implements Mamba-1 selective scan and Mamba-2 SSD when `chunk_size` is set; cumulative recurrence | 7/13 explicit (TRT-LLM `selectiveScanPlugin`, MLX `mx.cumsum`-based, ggml custom), 6/13 require composition from `cumsum` + `linear` | Mamba 1/2, Jamba, Zamba2, Hymba |
| 12 | `activation(x, kind)` | `[..., D] → [..., D]`; `kind ∈ {silu, gelu, gelu_tanh, relu2, gegelu}` (one op per the IHV survey, dispatched by kind) | 13/13 | all FFN |
| 13 | `add(x, residual, scale?)` | `[B,S,D]`; optional residual multiplier (Granite μP, LayerScale, DeepNorm) | 13/13 | all residual paths |
| 14 | `mul(a, b)` | element-wise; covers SwiGLU `silu(gate)*up`, attention mask scaling, scalar broadcast | 13/13 | all |
| 15 | `embed(ids, weight, scale?, codebook?)` | `ids:[B,S] → [B,S,D]`; optional scalar (Gemma `sqrt(D)`, Granite `embedding_multiplier`); optional codebook for codebook-quantized embeddings | 13/13 (codebook variant: 8/13 via ONNX `GatherBlockQuantized` and equivalents) | embeddings |
| 16 | `lm_head(x, weight, scale?, softcap?)` | `[B,S,D] → [B,S,V]`; logits scaling (Granite) and softcap (Gemma 2); optional tied weight | 13/13 | final layer |

**Sub-primitives folded into the above** (kept callable but not counted as
floor ops because every runtime composes them implicitly):

- `split(x, sizes, axis)` and `concat(xs, axis)` — used by MLA partial-RoPE
  splitting and codebook embedding. PyTorch makes these `tensor.split` /
  `torch.cat`; IHV runtimes expose them as `Split`/`Concat` ops. We document
  them as host-side reshapes in the op trace but do not count them as
  separate floor ops.
- `cast(x, dtype)` — dtype conversion. Every runtime exposes it; we make it
  callable but it is implicit inside `linear` (accumulator → compute) and
  `kv read` (storage → compute). Op-trace documents must show casts.
- `layer_scale(x, scale)` — scalar multiply. Folded into `mul` and `add`;
  exposed as a named helper for clarity in op traces for Granite μP,
  Gemma embedding scale, and LayerScale residual multipliers.
- `causal_mask(seq_len, window?, sink?, position_ids?)` — mask builder
  helper. Exposed in `api/ops.py` for layer.py implementations but does
  not count as a runtime primitive — runtimes either build the mask in
  kernel or accept a pre-built tensor.

**WKV recurrence (RWKV-7)** is expressed via `selective_scan` with a
specific parameterisation; we do *not* add a separate `wkv_update` op,
following the IHV survey which shows ggml and MLX implement WKV via the
same scan primitive Mamba-2 uses. If RWKV-7 implementation reveals a
distinct primitive (RWKV time-decay update), we add `wkv_update` as op 17
during B7.

Implementation style: **pure functions** in `api/ops.py`. Signatures take
positional tensors and named scalars; specs are passed by value. PyTorch
reference implementation provided; ONNX/runtime backends would slot in
behind the same signatures.

### 5.2 Specs — the parameter spaces

Specs are **frozen dataclasses** (`@dataclass(frozen=True)`). They are the
source of truth for "what varies across models." Each spec carries only
data; behaviour lives in the building blocks that consume them.

There are twelve specs: `AttentionSpec`, `RoPESpec`, `NormSpec`,
`FFNSpec`, `MoESpec`, `GroupRoutingSpec`, `SSMSpec`, `SSDSpec`,
`ConvSpec`, `LayerScaleSpec`, `KVCacheSpec`, `QuantSpec`, plus
`DecoderBlockSpec` which composes them. Enum definitions are in §14
(Glossary).

#### 5.2.1 `AttentionSpec`

```python
@dataclass(frozen=True)
class AttentionSpec:
    # Projection topology
    n_q_heads: int                               # e.g., 16 for Qwen3-0.6B, 32 for Llama-3-8B
    n_kv_heads: int                              # GQA: n_q_heads // gqa_groups; MQA: 1; MHA: == n_q_heads
    head_dim: int                                # e.g., 128; for Qwen3-0.6B: 128 (≠ hidden/H=64)
    kind: AttentionKind                          # see §14: STANDARD, MLA, DIFFERENTIAL,
                                                  # LINEAR_RETENTION, LINEAR_DELTANET, LINEAR_GLA, NSA
    qkv_layout: QKVLayout                        # see §14: SPLIT, FUSED, MLA_LATENT

    # Biases (4 independent flags — Qwen2 has q/k/v bias but not o; Llama-3 has none)
    q_bias: bool = False
    k_bias: bool = False
    v_bias: bool = False
    o_bias: bool = False

    # MLA-specific (None unless kind==MLA) — DeepSeek-V2/V3, MiniCPM-3
    q_lora_rank: Optional[int] = None            # e.g., 1536 for DeepSeek-V3
    kv_lora_rank: Optional[int] = None           # e.g., 512
    qk_nope_head_dim: Optional[int] = None       # e.g., 128 — non-rotated channels
    qk_rope_head_dim: Optional[int] = None       # e.g., 64 — partial-RoPE channels
    v_head_dim: Optional[int] = None             # e.g., 128 — V dim may differ from Q/K nope dim (MLA asymmetry)

    # Scale and softcap
    attn_scale: Optional[float] = None           # overrides 1/sqrt(head_dim) (Gemma 2 uses query_pre_attn_scalar)
    attn_logit_softcap: Optional[float] = None   # per-attention softcap (Gemma 2: 50). Distinct from final-logit softcap on DecoderBlockSpec.

    # Mask
    mask_kind: MaskKind                          # see §14: CAUSAL, SWA, SWA_GLOBAL_ALT, SINK, FULL, BLOCK_SPARSE, DOCUMENT_CAUSAL, CUSTOM
    sliding_window: Optional[int] = None         # e.g., 4096 for Gemma 3 SWA layers
    n_sink_tokens: Optional[int] = None          # e.g., 4 for GPT-OSS sink-trained models
    block_sparse_spec: Optional['BlockSparseSpec'] = None  # Phi-3-small

    # QK normalization
    qk_norm: Optional[NormSpec] = None           # carries kind/eps/weight_mode/shape (shape moved into NormSpec, see §5.2.3)
    qk_norm_phase: QKNormPhase = QKNormPhase.NONE  # NONE, PRE_ROPE (Qwen3 — per modeling_qwen3.py), POST_ROPE (Gemma 3)

    # RoPE
    rope: Optional[RoPESpec] = None              # None for NoPE layers (SmolLM3 every 4th layer, Llama-4 iRoPE NoPE layers)
    rope_partial_dim: Optional[int] = None       # GPT-J style partial RoPE; for MLA, use qk_rope_head_dim instead

    # Cross-layer sharing (CLA / YOCO)
    shares_kv_with: Optional[int] = None         # layer index this layer reuses; pairwise (CLA)
    cache_role: CacheRole = CacheRole.PRODUCER_CONSUMER  # PRODUCER (YOCO encoder stage), CONSUMER (YOCO decoder stage), PRODUCER_CONSUMER (default), SHARED_GROUP_MEMBER (CLA pair)

    # Differential-transformer extras (skip if kind != DIFFERENTIAL)
    diff_lambda_init: Optional[float] = None     # initial λ for differential block; residual paths unchanged (the "two attentions" are internal to the building block)

    # Per-layer head-count overrides (OpenELM)
    head_count_override: Optional[Tuple[int, int, int]] = None  # (n_q, n_kv, head_dim) overrides for this specific layer
```

#### 5.2.2 `RoPESpec`

```python
@dataclass(frozen=True)
class RoPESpec:
    base_theta: float                            # e.g., 1_000_000.0 for Qwen3-0.6B, 500_000.0 for Llama-3-8B, 10_000.0 for Llama-2
    basis: RoPEBasis                             # INTERLEAVED (HF reference) vs SPLIT_HALF (llama.cpp permute) — silent-degradation risk if mismatched
    scaling: RoPEScaling                         # NONE, PI, NTK_STATIC, DYNAMIC_NTK, YARN, LLAMA3, LONGROPE, IROPE
    scale_factor: Optional[float] = None         # PI / NTK / YaRN / Llama3 factor
    yarn_extra: Optional[YarnParams] = None      # original_max_pos, attn_factor, beta_fast, beta_slow
    llama3_extra: Optional[Llama3RoPEParams] = None  # low_freq_factor, high_freq_factor, original_max_position
    longrope_extra: Optional[LongRoPEParams] = None  # Phi-3: short_factor[], long_factor[], original_max_pos
    mrope_section: Optional[Tuple[int, ...]] = None  # M-RoPE for VL — out of scope for v2, field reserved
    irope_layer_pattern: Optional[Tuple[bool, ...]] = None  # Llama-4: per-block (apply_rope?) pattern

    # NoPE handling: at the layer level, NoPE = AttentionSpec.rope is None.
    # RoPESpec itself has no NONE; if a layer skips rope, the spec is absent.
```

#### 5.2.3 `NormSpec`

```python
@dataclass(frozen=True)
class NormSpec:
    kind: NormKind                               # RMS, LAYER, SCALE_NORM, DEEP_NORM (DeepNorm is training-only but documented)
    eps: float                                   # typically 1e-5 or 1e-6
    weight_mode: WeightMode                      # NONE (unscaled), STANDARD_W (default RMS), ONE_PLUS_W (Gemma), LEARNED_PER_HEAD (QK-norm per-head)
    bias: bool = False                           # LayerNorm has bias; RMSNorm typically does not
    shape: NormShape = NormShape.FULL_HIDDEN      # FULL_HIDDEN (default), PER_HEAD_DH (Qwen3 QK-norm), FULL_HDH (OLMo 2 QK-norm)
    # Placement (PRE/POST/SANDWICH) is on DecoderBlockSpec, not here.
```

#### 5.2.4 `FFNSpec` and `MoESpec`

```python
@dataclass(frozen=True)
class FFNSpec:
    intermediate_size: int                       # e.g., 3072 for Qwen3-0.6B
    activation: Activation                       # SILU, GELU, GELU_TANH, GEGELU, RELU2
    gate_kind: GateKind                          # SWIGLU (silu(gate)*up), GEGLU (gelu(gate)*up), GELU_ONLY (no gate), RELU2_ONLY
    fused_gate_up: bool = False                  # Phi-3 fuses gate+up into one linear with 2x intermediate
    gate_bias: bool = False
    up_bias: bool = False
    down_bias: bool = False

@dataclass(frozen=True)
class GroupRoutingSpec:
    """DeepSeek-V3 node-limited routing — applied before final top-k expert selection."""
    n_groups: int                                # e.g., 8 — number of expert groups (often one per node)
    top_k_groups: int                            # e.g., 4 — keep top-k groups, mask experts in remaining groups
    group_score_kind: GroupScoreKind             # SUM_TOP_K_IN_GROUP (DeepSeek-V3 v1), MAX, MEAN
    group_score_top_k: Optional[int] = None      # the "k" inside SUM_TOP_K_IN_GROUP (typically top-2 within group)

@dataclass(frozen=True)
class MoESpec:
    n_routed_experts: int                        # e.g., 160 for DeepSeek-V3 — the routed pool
    top_k: int                                   # e.g., 6
    n_shared_experts: int = 0                    # always-on experts; summed with routed output (DeepSeek-V3: 2)
    router_kind: RouterKind                      # SOFTMAX (Mixtral), SIGMOID_PLUS_BIAS (DeepSeek-V3 auxiliary-loss-free), LINEAR_TOP_K_NO_NORM
    router_norm: bool = False                    # post-routing weight normalization
    score_correction_bias: bool = False          # DeepSeek-V3 expert-bias auxiliary-loss-free balancing
    group_routing: Optional[GroupRoutingSpec] = None
    routed_scaling_factor: float = 1.0           # multiplier on routed-output before combining with shared experts
    expert_ffn: FFNSpec                          # shape of one expert (assumed uniform across experts)
    dispatch_kind: DispatchKind = DispatchKind.DENSE_SCATTER  # DENSE_SCATTER (reference), GROUPED_GEMM (runtime), TOKEN_PERMUTE (Megablocks)
```

#### 5.2.5 `SSMSpec`, `SSDSpec`, `ConvSpec`

```python
@dataclass(frozen=True)
class SSMSpec:
    """Mamba-1 selective state-space (Gu & Dao 2023)."""
    d_state: int                                 # SSM state dim per channel; e.g., 16 (Mamba-1), 128 or 256 (Mamba-2)
    d_conv: int                                  # 1D causal conv kernel width; e.g., 4
    d_inner: int                                 # expanded inner dim = expand_factor × d_model
    expand_factor: int                           # typically 2 (Mamba), 1 (Jamba inner-block)
    dt_rank: Union[int, Literal["auto"]]         # delta projection rank; "auto" → ceil(d_model/16)
    dt_init: Literal["constant", "random"]       # initialization scheme for dt projection
    dt_scale: float                              # init scale for dt projection (e.g., 1.0)
    dt_min: float                                # softplus floor for dt (e.g., 0.001)
    dt_max: float                                # softplus ceiling for dt (e.g., 0.1)
    dt_init_floor: float                         # tiny epsilon for dt to avoid log(0); e.g., 1e-4
    conv_bias: bool                              # bias on the causal conv1d
    bias: bool                                   # bias on the input projection
    use_fast_path: bool                          # whether the runtime should attempt the fused-kernel path
    activation: Activation = Activation.SILU     # post-conv activation
    state_dtype: DType = DType.FP32              # SSM state typically FP32 for numerical stability

@dataclass(frozen=True)
class SSDSpec(SSMSpec):
    """Mamba-2 SSD formulation (Dao & Gu 2024) — adds chunked scan parameters."""
    chunk_size: int                              # typically 256 — block size for chunked parallel scan
    headdim: int                                 # typically 64 — SSD heads, distinct from attention heads
    ngroups: int                                 # 1 (Mamba-2 default) or 8 (state-grouped variants); shared B,C across group

@dataclass(frozen=True)
class ConvSpec:
    """1D causal convolution used by SSM front-end and by Hymba/Samba-style conv-attention."""
    kernel_size: int                             # 4 (Mamba), 3 (some hybrids)
    bias: bool
    activation: Optional[Activation] = None      # None = identity; Mamba uses SiLU after conv
    groups: Optional[int] = None                 # None = depthwise (groups = channels)
    causal: bool = True                          # left-padded causal conv
```

`TokenMixerKind` enum (referenced by `DecoderBlockSpec`):

```python
class TokenMixerKind(Enum):
    ATTENTION_STANDARD = auto()                  # MHA/GQA/MQA — parameterized by n_kv_heads
    ATTENTION_MLA = auto()                       # DeepSeek-V2/V3, MiniCPM-3
    ATTENTION_LINEAR = auto()                    # generic linear attention bucket
    ATTENTION_DELTANET = auto()                  # DeltaNet / Gated DeltaNet
    ATTENTION_NSA = auto()                       # NSA three-tier (DeepSeek 2025)
    SSM_MAMBA1 = auto()
    SSM_MAMBA2 = auto()                          # SSD formulation
    SSM_GRIFFIN = auto()                         # RecurrentGemma LRU-based
    SSM_RWKV = auto()                            # WKV recurrence (RWKV-6/7)
    HYBRID_PARALLEL = auto()                     # Hymba: Mamba ‖ Attn in same block, outputs summed
    HYBRID_ALTERNATING = auto()                  # per-layer dispatch (Jamba, Zamba2) — encoded as list[DecoderBlockSpec]
```

#### 5.2.6 `LayerScaleSpec`

```python
@dataclass(frozen=True)
class LayerScaleSpec:
    """Per-layer scalar multipliers used by Granite μP, DeepNorm, LayerScale variants."""
    residual_scale: Optional[float] = None       # e.g., 0.22 for Granite — multiplies residual contributions
    embedding_scale: Optional[float] = None      # e.g., 12.0 for Granite, sqrt(D) for Gemma
    logits_scale: Optional[float] = None         # e.g., 8.0 for Granite (divides logits)
    final_logit_softcap: Optional[float] = None  # e.g., 30.0 for Gemma 2 (Gemma 3 dropped this)
    attn_residual_scale: Optional[float] = None  # DeepNorm-style separate scale for attention residual
    ffn_residual_scale: Optional[float] = None   # DeepNorm-style separate scale for FFN residual
```

#### 5.2.7 `KVCacheSpec`

```python
@dataclass(frozen=True)
class KVCacheSpec:
    layout: CacheLayout                          # CONTIGUOUS, PAGED, RING, MLA_LATENT_PLUS_KROPE, SSM_STATE, SSD_STATE_PLUS_CONV, NSA_THREE_TIER, YOCO_SHARED
    memory_layout: MemoryLayout                  # HND ([H,N,D]), NHD ([N,H,D]), BHND, BNHD (vLLM), MLX_LIST
    block_size: Optional[int] = None             # for PAGED, RING, vAttention; e.g., 16 (vLLM default)
    k_dtype: DType                               # FP16, BF16, FP8_E4M3, FP8_E5M2, INT8
    v_dtype: DType
    k_quant: Optional[QuantSpec] = None          # KIVI: K per-channel
    v_quant: Optional[QuantSpec] = None          # KIVI: V per-token (asymmetric to K)
    dequant_path: DequantPath = DequantPath.IN_SDPA  # where quantized cache is dequantized: IN_SDPA, AT_READ, AT_WRITE
    ownership: CacheOwnership                    # EXPLICIT_PASS (functional, test-friendly), STATEFUL (handle-on-block, runtime-friendly)
    share_group: Optional[int] = None            # CLA pairwise sharing — layer index of the shared cache
    vm_mapped: bool = False                      # vAttention: CUDA VM-mapped backing memory (logically same as PAGED, semantically different allocator)
    tier_count: Optional[int] = None             # NSA three-tier: 3
    cache_role: CacheRole = CacheRole.PRODUCER_CONSUMER  # mirror of AttentionSpec.cache_role; YOCO encoder→PRODUCER, decoder→CONSUMER
    # For MLA: shape is [c_kv: kv_lora_rank, k_rope: qk_rope_head_dim] per token (TWO tensors, encoded as MLA_LATENT_PLUS_KROPE layout name to surface this)
    # For SSM: shape is [d_state, d_inner] per token PLUS [d_conv, d_inner] for the conv1d state — encoded as SSD_STATE_PLUS_CONV
```

#### 5.2.8 `QuantSpec` (revised — addresses critique items 4.1, 4.4, 4.5, 4.6, 4.7, 6.3)

```python
@dataclass(frozen=True)
class QuantSpec:
    # Minimal core (covers AWQ/GPTQ/SmoothQuant/FP8 W8A8/HQQ/GGUF)
    qdtype: QDType                               # INT2, INT3, INT4, INT5, INT6, INT8, FP8_E4M3, FP8_E5M2, FP4, NF4, MX_FP4, MX_INT4, TERNARY (BitNet)
    group_size: Optional[int]                    # None=per-tensor, -1=per-channel, N=blockwise (e.g., 32, 64, 128, 256)
    quant_axis: int                              # output axis for W (typically 0 for [N,K]); channel/token axis for A
    scale_dtype: DType                           # FP32, FP16, BF16, UE8M0 (MXFP4 power-of-two-only scales), FP8
    has_zero_point: bool
    packing: PackingLayout                       # NONE, NIBBLE_LSB, NIBBLE_MSB, AWQ_INTERLEAVE ([0,2,4,6,1,3,5,7]), GPTQ_INT32_PACK, HQQ_NIBBLE, GGUF_K_SUPERBLOCK, MX_BLOCK
    accumulator_dtype: DType                     # FP32, FP16, INT32

    # Extended
    scale_quant: Optional['QuantSpec'] = None    # recursive: double-quant / NVFP4 outer / k-quant super-scale
    codebook: Optional[CodebookSpec] = None      # NF4 (16 codepoints), IQ-quants (E8/NL lattice), AQLM (multi-codebook)
    scale_axis: Optional[int] = None             # k-quants: scales on different axis from values
    zero_dtype: Optional[DType] = None
    compute_dtype: Optional[DType] = None        # storage ≠ compute (AWQ stores INT4, computes FP16)
    block_layout: Optional[BlockLayout] = None   # OCP_MX (MXFP4/6/8), GGUF_K (256-element super-block), AWQ_INTERLEAVE_K, GPTQ_BLOCK
    accumulator_dtype: DType = DType.FP32

    # Pre-transforms — split from v1's single pre_transform per critique 4.4
    rotation_kind: RotationKind = RotationKind.NONE  # NONE, DIAGONAL_SMOOTHQUANT (per-channel α), DENSE_HADAMARD_QUAROT, DENSE_LEARNED_SPINQUANT, HADAMARD_RUNTIME (applied at inference)
    runtime_apply: RuntimeApply = RuntimeApply.NONE  # NONE, INPUT (apply to activations at input only), INPUT_OUTPUT_BOTH (also at o_proj input)
    smoothquant_alpha: Optional[float] = None    # diagonal-rotation strength; typically 0.5–0.8

    # Codebook combine — covers AQLM (additive multi-codebook), QuIP# (lattice+residual), SqueezeLLM (per-row)
    num_codebooks: int = 1                       # 1 = single (NF4), 2+ = AQLM additive
    combine_op: CombineOp = CombineOp.NONE       # NONE, ADD (AQLM), CONCAT, MULTI_LATTICE (QuIP#)

    # Variable-bitwidth (EXL2)
    variable_bitwidth: bool = False              # if True, per-row bitwidth, with codebook describing per-row schedule

    # Outlier offloading (LLM.int8)
    outlier_offload: OutlierOffload = OutlierOffload.NONE  # NONE, FP16_COLUMN_SPLIT (LLM.int8: outlier columns kept in FP16, rest in INT8), FP16_ROW_SPLIT

    # imatrix calibration (llama.cpp)
    imatrix_calibrated: bool = False             # whether the k-quant rounder was driven by per-column importance weighting

    # Role and dynamic scale source
    role: QuantRole = QuantRole.WEIGHT           # WEIGHT, ACTIVATION, KV_K, KV_V, ATTN_INTERNAL, EMBEDDING
    scale_source: ScaleSource = ScaleSource.STATIC  # STATIC, DYNAMIC_PER_TOKEN (FP8 W8A8 activation), DYNAMIC_PER_TENSOR

    # Reserved for future LoRA mergeability work
    merged_lora_rank: Optional[int] = None       # reserved; None for v2
```

**"Support five" baseline** (replacing v1's "support three"; addresses
critique items 4.7 and 6.3):

1. **GGUF Q4_K_M** — dominant on-device (llama.cpp, Ollama, LM Studio).
   Exercises `block_layout=GGUF_K_SUPERBLOCK`, recursive `scale_quant`
   (6-bit sub-scales), `imatrix_calibrated`. Uses `QDType.INT4` with
   256-element super-block of 8×32 sub-blocks, sub-block scale and min
   each stored as 6-bit, plus FP16 super-block scale. (Sub-block size
   is 32 weights × 8 sub-blocks per critique 4.2 fix; v1's "16×16"
   claim was wrong.)
2. **AWQ INT4 W4A16 grouped (g=128)** — dominant server-side 4-bit
   (vLLM, TRT-LLM, SGLang). Exercises `packing=AWQ_INTERLEAVE` with
   permutation `[0,2,4,6,1,3,5,7]`, per-channel + per-group scales,
   activation-aware calibration. NOT storage-equivalent to GPTQ (see
   below).
3. **FP8 E4M3 W8A8** — dominant compute-quantized (H100, MI300, Gaudi 3).
   Per-tensor W with `scale_source=STATIC`, per-token A with
   `scale_source=DYNAMIC_PER_TOKEN`. Exercises the activation-dynamic-scale
   path.
4. **+ IQ2_M (or AQLM)** — exercises the `codebook` axis. IQ2_M uses an
   E8 lattice codebook (256 codepoints); AQLM uses additive multi-codebook
   with `num_codebooks=2, combine_op=ADD`. Choose IQ2_M if implementation
   complexity is a concern; AQLM if exercising `num_codebooks>1` matters
   more. (Author recommendation: **IQ2_M** — simpler, more representative
   of on-device codebook quantization.)
5. **+ MXFP4 (OCP)** — exercises the `UE8M0` scale dtype and
   `block_layout=OCP_MX`. 32-element micro-blocks with shared power-of-two
   scale stored as 8-bit unsigned exponent (UE8M0). Becoming the
   compute-quant standard (Blackwell, Hopper-Next).

**Correction (critique 6.3): AWQ, GPTQ, HQQ are NOT storage-identical.**
v1 claimed equivalence; v2 corrects:

- **Storage layout differs.** AWQ uses interleaved 32-bit packed nibbles
  with permutation `[0,2,4,6,1,3,5,7]` for MMA-friendly kernel access.
  GPTQ packs eight INT4 values into INT32 sequentially (column-major
  pack-factor). HQQ uses a simple nibble pack matching its calibration-free
  rounding.
- **Quantization metadata differs.** AWQ ships per-channel + per-group
  scales derived from activation calibration. GPTQ ships per-group scales
  derived from layer-wise Hessian descent. HQQ ships scales from
  half-quadratic optimization (no calibration data).
- **Calibration procedure differs.** AWQ requires a calibration dataset
  to estimate per-channel activation magnitudes. GPTQ requires calibration
  to compute Hessians. HQQ is data-free.

They share only the *kernel-level shape* (W4A16 grouped INT4 matrix-multiply);
the dequantised weights are mathematically distinct because the scales
were derived by different procedures. M1's AWQ exercise establishes the
loader for AWQ-specific scale/zero/pack layout; GPTQ and HQQ are
similar but **not** drop-in.

These five baselines together exercise every QuantSpec axis: storage
layouts (AWQ_INTERLEAVE, GGUF_K, MX_BLOCK), packing (NIBBLE, AWQ_INTERLEAVE,
INT32_PACK, MX), recursive scales (GGUF), codebook (IQ2_M), scale dtypes
(FP16, FP32, UE8M0, FP8), dynamic activation scales (FP8 W8A8),
rotation (deferred — `rotation_kind=NONE` for all five; QuaRot/SpinQuant
exercised in a B-batch rather than M1).

### 5.3 Building blocks

```
Attention(spec: AttentionSpec, kv_cache: KVCache, layer_scale: Optional[LayerScaleSpec])
  # composes linear, split (for MLA), rope, sdpa, kv read/write, qk_norm
SSM(spec: SSMSpec | SSDSpec, conv_spec: ConvSpec, ssm_cache: KVCache)
  # composes linear, conv1d, selective_scan, gather (for routing if grouped)
FeedForward(spec: FFNSpec)
  # composes linear(s) + activation + mul (for gated variants)
MoE(spec: MoESpec)
  # composes linear (gate), softmax|sigmoid_plus_bias (router), top_k, gather (dispatch),
  # per-expert FeedForward, scatter (combine), residual add of shared experts
RMSNorm(spec: NormSpec, hidden_size)            # supports shape=FULL_HIDDEN, PER_HEAD_DH, FULL_HDH
LayerNorm(spec: NormSpec, hidden_size)
RoPE(spec: RoPESpec, max_seq_len)               # holds freq tables; applies via api.ops.rope_apply
KVCache(spec: KVCacheSpec, layer_idx, max_seq, n_heads, head_dim)
QuantizedWeight(spec: QuantSpec, shape, ...)    # owns packed storage + scales + zeros + codebook; produces dequant on read
```

The MoE building block hides the routing primitives (`softmax`, `top_k`,
`gather`, `scatter`) as composition; the op trace in `models/<family>/layer.md`
documents them explicitly so the doc is faithful.

The differential-attention building block keeps the "two attentions"
internal — the decoder block's residual stream remains single. If a
future model requires dual residual streams, we promote `DecoderBlockSpec`
to support `residual_streams: int` and defer the API change to a hypothetical
"Phase D" (see §11 risks).

The hybrid-parallel building block (`TokenMixerKind.HYBRID_PARALLEL`,
Hymba) consumes a *pair* of token-mixer specs and sums their outputs
before the residual add. The DecoderBlock dispatches on `TokenMixerKind`.

### 5.4 DecoderBlock assembly

```python
@dataclass(frozen=True)
class DecoderBlockSpec:
    # Residual structure (covers OLMo 2 post-norm, Gemma 2/3 dual pre+post)
    attn_norm_position: NormPosition             # PRE, POST, PRE_AND_POST (sandwich), PARALLEL_FFN (Falcon-7B / GPT-J historical)
    ffn_norm_position: NormPosition

    # Token mixer kind + spec(s)
    token_mixer_kind: TokenMixerKind
    token_mixer: Union[AttentionSpec, SSMSpec, SSDSpec, Tuple[AttentionSpec, SSMSpec]]
    # For HYBRID_PARALLEL (Hymba), token_mixer is a tuple (attn_spec, ssm_spec); their outputs are summed before the residual add.
    # For HYBRID_ALTERNATING (Jamba/Zamba2), each layer is its own DecoderBlockSpec with the appropriate spec; model carries list[DecoderBlockSpec].

    # Channel mixer
    channel_mixer: Union[FFNSpec, MoESpec]
    # For DeepSeek-V3-style dense+MoE interleaving, each layer carries either FFNSpec or MoESpec.
    # For MoE with shared experts, MoESpec.n_shared_experts > 0; building block sums shared + routed.

    # Norms (each Optional; required when the corresponding position is PRE/POST/PRE_AND_POST)
    input_norm: NormSpec                          # always present
    pre_attn_norm: Optional[NormSpec] = None      # used when attn_norm_position in {PRE, PRE_AND_POST}
    post_attn_norm: Optional[NormSpec] = None     # used when attn_norm_position in {POST, PRE_AND_POST}
    pre_ffn_norm: Optional[NormSpec] = None
    post_ffn_norm: Optional[NormSpec] = None

    # Per-block layer-scale parameters
    layer_scale: Optional[LayerScaleSpec] = None

    # Cache (optional — typically a list[KVCacheSpec] is held at the model level, indexed by layer)
    kv_cache: Optional[KVCacheSpec] = None
```

This handles:

- **OLMo 2 post-norm** → `attn_norm_position=POST`, no `pre_attn_norm`,
  `post_attn_norm` present
- **Gemma 3 sandwich** → both `attn_norm_position=PRE_AND_POST`,
  `pre_attn_norm` and `post_attn_norm` both present
- **Granite μP** → `layer_scale=LayerScaleSpec(residual_scale=0.22,
  embedding_scale=12, logits_scale=8)`
- **Per-layer heterogeneity** (Gemma 3 SWA alternation, Jamba SSM/attn
  mix, SmolLM3 NoPE every 4th layer, Llama-4 iRoPE) → model's
  `ModelConfig` carries a `list[DecoderBlockSpec]` of length `num_layers`
- **Hymba parallel** → `token_mixer_kind=HYBRID_PARALLEL`,
  `token_mixer=(AttentionSpec(...), SSMSpec(...))`
- **OpenELM per-layer heads** → each `DecoderBlockSpec` carries an
  `AttentionSpec` with `head_count_override` set
- **DeepSeek-V3 dense/MoE interleaving** → each layer's
  `channel_mixer` is either `FFNSpec` (dense layers, first 3) or
  `MoESpec` (rest)

## 6. Phase B — Per-model layer factory pattern

Each `models/<family>/` contains:

### `layer.md`

Required sections:

1. **Identity** — family name, variants and param counts, release date,
   source paper, source repo.
2. **Decoder block diagram** — ASCII or mermaid showing residual structure,
   sub-layers, normalisation placement.
3. **Tensor IO trace** — per-step shapes from input `x:[B,S,D]` to output
   `[B,S,D]`, every intermediate annotated with dtype and shape.
4. **Op trace** — sequence of `api.ops.*` calls (including casts and
   reshapes) implementing the block. MoE and SSM blocks expand their
   internal routing/scan ops explicitly so the doc is faithful per critique
   1.7 / 6.6.
5. **Spec instantiation** — the `DecoderBlockSpec` values for the canonical
   variant (e.g., Qwen3-0.6B).
6. **Quirks** — anything model-specific that is *not* a generic axis (e.g.,
   `1+w` RMSNorm baking foot-gun, residual scaling magnitude, llama.cpp
   Q/K permute, post-norm FP16 overflow risk).
7. **Weight-name mapping** — table mapping HF tensor names →
   API tensor slots (see §9).
8. **Source citations** — file:line into HF transformers, vLLM, llama.cpp
   (from `research/02-layer-sources.v2.md`).

### `layer.py`

```python
def build_decoder_layer(config: <Family>Config, layer_idx: int = 0) -> DecoderBlock:
    """Assemble one decoder block using api/ primitives only.

    Must NOT import from transformers.models.<family>. Must use only api/.
    """
```

`<Family>Config` is a dataclass containing hyperparameters. Size variants
become different configs. The factory translates config → `DecoderBlockSpec`
→ `api.block.DecoderBlock(spec)`.

### `weight_loader.py`

```python
def load_weights(checkpoint_path: str, layer: DecoderBlock, layer_idx: int,
                 quant: Optional[QuantSpec] = None) -> None:
    """Map HF / GGUF / AWQ tensor names → API tensor slots."""
```

See §9 for the mapping schema.

### `test_layer.py`

Required tests:

- **Shape**: instantiate with random weights at ≥2 size variants; forward
  pass; assert output shape and dtype.
- **Determinism**: same seed → same output (max-abs diff = 0 across runs).
- **KV cache write/read**: prefill T tokens, decode token T+1, assert cache
  state evolves correctly (shape, write position, retrievability).
- **Numerical equivalence** (Qwen3 only — kickoff gate): see §8.

## 7. Validation hierarchy

| Level | What it proves | Coverage |
|---|---|---|
| **(a) Structural** | Shape and dtype correct, residual structure correct | All ~24 families |
| **(b) Op-trace** | API ops match the "intended" graph (verified against HF source op-by-op) | ~5 representative families (Qwen3, Gemma 3, DeepSeek-V3, Mixtral, Jamba) |
| **(c) Numerical** | Real HF weights → API-assembled layer produces fp-equivalent output to HF reference | Qwen3 sample only (M1 kickoff gate) |
| **(d) Quantization round-trip** | AWQ unpack → dequant → matmul produces output equivalent to HF AWQ reference within quant tolerance | Qwen3 + one B-batch model per quant scheme |

Numerical equivalence is the strongest claim and most expensive; we prove
it once on Qwen3 to demonstrate framework soundness, then rely on (a) +
(b) at scale. Quant round-trip is a separate level because dequant
correctness is orthogonal to layer correctness.

## 8. Qwen3 M1 kickoff gate

**Definition of done for M1 — user reviews before per-model rollout (M2+).**

### 8.1 Artifacts

| Artifact | Contents |
|---|---|
| `api/ops.py` | The 16 logical ops, PyTorch reference impl. softmax/top_k/gather/scatter/conv1d/selective_scan present even though Qwen3 doesn't exercise the latter four — they ship in M1 so B6/B7 don't bottleneck. |
| `api/specs.py` | All 12 spec dataclasses, including `SSMSpec`, `SSDSpec`, `ConvSpec`, `GroupRoutingSpec`, `LayerScaleSpec` (referenced by future batches; Qwen3 uses only Attention/RoPE/Norm/FFN/KVCache/Quant) |
| `api/norm.py` | `RMSNorm`, supporting `STANDARD_W` and `ONE_PLUS_W`, `shape ∈ {FULL_HIDDEN, PER_HEAD_DH, FULL_HDH}` |
| `api/rope.py` | RoPE block, `LLAMA3` scaling variant (Qwen3 reuses Llama-3 RoPE structure with `base_theta=1_000_000`) |
| `api/attention.py` | Attention block supporting GQA + QK-norm (Qwen3 phase = **PRE_ROPE** per critique 5.1 fix) + RoPE + KV cache (contiguous layout) |
| `api/feedforward.py` | FeedForward block, SwiGLU activation |
| `api/kvcache.py` | KVCache with `CONTIGUOUS` layout, FP16/BF16 dtypes |
| `api/quant.py` | QuantSpec realization; AWQ W4A16 (g=128) packed-int weight load + dequant + matmul path (see §9.2) |
| `api/block.py` | DecoderBlock with `PRE` norm position |
| `api/weights.py` | HF tensor-name → API-slot mapping helper (see §9) |
| `models/qwen3/layer.md` | Full layer doc per §6 schema, citing `research/02-layer-sources.v2.md` Qwen3 section |
| `models/qwen3/layer.py` | `build_qwen3_decoder_layer(config, layer_idx)` for Qwen3-0.6B, 1.7B, 4B, 8B (all dense) |
| `models/qwen3/weight_loader.py` | `load_qwen3_weights(safetensors_path, layer, layer_idx, quant=None)` |
| `models/qwen3/test_layer.py` | Tests (a)–(g) below |

### 8.2 Test specification (tightened — addresses critique 6.4)

**Reference implementation:** HuggingFace
`transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer` running in
eager mode, FP32 compute, FP16 storage. Pinned to
`transformers==4.46.x` (concrete pin recorded in `pyproject.toml`).

**Fixed test input** (deterministic, no tokenizer involvement):

```python
input_ids = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)
# batch=1, seq_len=9 — captures both prefill (8 tokens) and a partial decode (the 9th)
position_ids = torch.arange(0, 9, dtype=torch.long).unsqueeze(0)
attention_mask = torch.tril(torch.ones(9, 9))  # full causal
layer_idx = 0
```

Hidden states for the reference forward come from the embedding output
of the unmodified HF model on `input_ids`; this avoids drifting on
embedding scale (Qwen3 does not scale embeddings, but the test fixture
makes this explicit by reading embeddings from the reference).

**(a) Shape test** — 3 size variants (0.6B, 1.7B, 4B), random weights,
forward pass, assert output shape `(1, 9, hidden_size)` and dtype `FP16`.

**(b) Determinism** — same seed → max-abs-diff = 0 across two runs.

**(c) KV-cache prefill+decode** — prefill 8 tokens, save cache state,
decode token 9; assert (i) cache buffer at position 8 contains the new
K/V, (ii) the decode-output for position 9 equals the equivalent slice
from a full 9-token prefill within `atol=1e-6` (computational
equivalence of two paths within the same impl).

**(d) Sub-op isolation ladder** (NEW per critique 4.6) — before
block-level numerical equivalence, each sub-op must match its HF
counterpart independently:

| Sub-op | API call | HF reference | Tolerance |
|---|---|---|---|
| RMSNorm | `api.norm.RMSNorm(spec)(x)` | `Qwen3RMSNorm(x)` | `atol=1e-5, rtol=1e-5` |
| RoPE | `api.ops.rope_apply(qk, freqs, ...)` | `apply_rotary_pos_emb(q, k, cos, sin)` | `atol=1e-5, rtol=1e-5` |
| QK-norm | `api.norm.RMSNorm(qk_norm_spec)(qk)` | Qwen3's per-head QK norm | `atol=1e-5, rtol=1e-5` |
| SDPA | `api.ops.sdpa(q, k, v, mask, scale)` | `torch.nn.functional.scaled_dot_product_attention` | `atol=1e-5, rtol=1e-5` |
| FFN | `api.feedforward.FeedForward(spec)(x)` | `Qwen3MLP(x)` | `atol=1e-5, rtol=1e-5` |

Each sub-op test loads the corresponding weight subset (`q_proj`, `k_proj`,
etc.) from the canonical Qwen3-0.6B HF checkpoint at the chosen pinned
revision (see §16 for fixture provenance).

**(e) Block-level numerical equivalence** — load Qwen3-0.6B layer-0 HF
weights into the API-assembled layer, forward both with the fixed input,
assert:

```python
torch.allclose(out_api, out_ref, atol=5e-4, rtol=5e-4)
```

This is tighter than v1's 1e-3 absolute. The norm is `torch.allclose`
(elementwise, max-abs + rel) per critique 3.E; not L2. The reference dtype
is FP32 compute / FP16 storage for both candidate and reference, eliminating
dtype-induced drift as a confound.

**(f) KV-cache numerical equivalence** — prefill 8 tokens (positions 0–7)
into both API and HF layers; save both caches; decode position 8 in both;
assert output for position 8 matches at `atol=5e-4`. This catches RoPE
basis bugs that only surface during incremental decode (the partial-RoPE
basis mismatch is a classic silent-degradation source).

**(g) Quantized numerical equivalence** — load weights from
`Qwen/Qwen3-0.6B-AWQ` (if available at M1 time; else our own AWQ
quantization of the FP16 checkpoint via `autoawq`) into the API-assembled
layer with `QuantSpec(qdtype=INT4, group_size=128, packing=AWQ_INTERLEAVE,
scale_dtype=FP16, has_zero_point=True, role=WEIGHT)`. Compare against the
HF AWQ reference forward at `atol=5e-2, rtol=5e-2` (looser because INT4
quant introduces non-trivial output error; the test's purpose is to
verify our unpack/dequant path matches the kernel-equivalent AWQ output).

### 8.3 Kickoff review questions

1. Is the API surface what you expected? (16 ops, 12 specs)
2. Is the spec dataclass pattern right (`AttentionSpec` etc.)?
3. Is `models/qwen3/layer.md` the right doc format (with the §6
   weight-name mapping table)?
4. Is `build_qwen3_decoder_layer` the right factory shape?
5. Does the AWQ exercise look right for quantization, including the
   interleave-unpack lowering (§9.2)?
6. Did the sub-op isolation ladder catch any drift before block-level?
7. Did the FP16 cache exercise prefill+decode equivalence?

Approval gates M2 — the parallel rollout across remaining models.

**Why Qwen3 is the right seed:** GQA + RoPE (Llama-3 scaling) + RMSNorm +
SwiGLU + QK-Norm. It is the Llama-family template plus one named extension
(QK-norm), so the API floor lands quickly and the QK-norm axis gets
exercised in the seed instead of being an afterthought. AWQ exercises
the quantization axis at M1 since AWQ Qwen3 checkpoints exist on HF Hub.
The post-2024 axes (MLA, MoE, SSM) are deliberately deferred to keep M1
focused.

## 9. Weight-loader story (NEW — addresses critique 4.1, 6.5, 7.2)

The numerical-equivalence test demands a bit-exact path from HF safetensors
(or AWQ packed-int safetensors, or GGUF k-quant files) into API tensor
slots. M1 ships this for Qwen3; later batches ship the same shape for
each family.

### 9.1 HF → API tensor-name mapping (Qwen3-0.6B canonical)

| HF tensor name | API tensor slot |
|---|---|
| `model.embed_tokens.weight` | `model.embedding.weight` |
| `model.layers.0.input_layernorm.weight` | `block_0.input_norm.weight` |
| `model.layers.0.self_attn.q_proj.weight` | `block_0.attention.q_proj.weight` |
| `model.layers.0.self_attn.k_proj.weight` | `block_0.attention.k_proj.weight` |
| `model.layers.0.self_attn.v_proj.weight` | `block_0.attention.v_proj.weight` |
| `model.layers.0.self_attn.o_proj.weight` | `block_0.attention.o_proj.weight` |
| `model.layers.0.self_attn.q_norm.weight` | `block_0.attention.qk_norm_q.weight` |
| `model.layers.0.self_attn.k_norm.weight` | `block_0.attention.qk_norm_k.weight` |
| `model.layers.0.post_attention_layernorm.weight` | `block_0.pre_ffn_norm.weight` |
| `model.layers.0.mlp.gate_proj.weight` | `block_0.ffn.gate_proj.weight` |
| `model.layers.0.mlp.up_proj.weight` | `block_0.ffn.up_proj.weight` |
| `model.layers.0.mlp.down_proj.weight` | `block_0.ffn.down_proj.weight` |
| `model.norm.weight` | `model.final_norm.weight` |
| `lm_head.weight` (or tied with embed) | `model.lm_head.weight` |

`api/weights.py` ships a generic dotted-name mapping function plus
per-family overrides; the mapping table for each family lives in
`models/<family>/layer.md` §7.

### 9.2 AWQ unpacking (the hardest M1 deliverable)

AWQ ships three tensors per linear layer at INT4 W4A16 g=128:

- `qweight: int32[K // 8, N]` — eight INT4 values packed per INT32 along K
  axis, with the **interleave permutation `[0, 2, 4, 6, 1, 3, 5, 7]`**
  (i.e., the i-th INT4 in the packed word is the `perm[i]`-th value in
  the original sequence). Shifts for extraction:
  `shift = [0, 16, 4, 20, 8, 24, 12, 28]`. Read as
  `(qweight >> shift) & 0xF`.
- `qzeros: int32[K // 128, N // 8]` — per-group zero-points, with the same
  interleave permutation packed across the N axis (8 INT4 zeros per
  INT32). One row per group of 128 along K.
- `scales: float16[K // 128, N]` — per-group FP16 scales.

Dequantization (`api/quant.py`):

```python
def awq_dequant(qweight_int32, qzeros_int32, scales_fp16, group_size=128):
    # 1. Extract 8 INT4 lanes per INT32 word with the interleave shifts.
    K_div_8, N = qweight_int32.shape
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int32)
    # unpacked: [K_div_8, N, 8] of nibbles
    unpacked = (qweight_int32.unsqueeze(-1) >> shifts) & 0xF
    # Reshape into [K, N] with K = K_div_8 * 8, undoing the interleave by
    # using shifts already in interleaved order (the shifts list IS the inverse
    # permutation — the unpacked dim-2 is already in [0,2,4,6,1,3,5,7] order
    # of the *original* K positions, so we either accept that order or reindex).
    qweight = unpacked.permute(0, 2, 1).reshape(-1, N)  # [K, N]
    # 2. Unpack zeros similarly along N axis: qzeros[K/group, N/8] → [K/group, N]
    qzeros_unpacked = (qzeros_int32.unsqueeze(-1) >> shifts) & 0xF  # [K/group, N/8, 8]
    zeros = qzeros_unpacked.permute(0, 1, 2).reshape(K_div_8 * 8 // group_size, -1)[:, :N]
    # 3. Dequant: w_fp = (qweight - zero) * scale, broadcasting zero/scale per group of K.
    group_idx = torch.arange(qweight.shape[0]) // group_size  # [K]
    w_dequant = (qweight.float() - zeros[group_idx].float()) * scales[group_idx].float()
    return w_dequant.to(torch.float16)
```

(Sketch — exact ordering of the reshape/permute is verified against
`autoawq.utils.packing` during M1; the test in §8.2(g) is the bit-exact
ground truth.)

The dequantised weight then enters `linear(x, w_dequant)` via the standard
FP16 GEMM path. A future runtime kernel would fuse unpack+dequant+GEMM;
the reference impl keeps them separate for clarity.

### 9.3 GGUF unpacking (deferred to a B-batch, sketched here)

GGUF tensor names use the `blk.<i>.<sublayer>.<weight>` convention:

| GGUF name | API slot |
|---|---|
| `blk.0.attn_norm.weight` | `block_0.input_norm.weight` |
| `blk.0.attn_q.weight` | `block_0.attention.q_proj.weight` |
| `blk.0.attn_k.weight` | `block_0.attention.k_proj.weight` |
| `blk.0.attn_v.weight` | `block_0.attention.v_proj.weight` |
| `blk.0.attn_output.weight` | `block_0.attention.o_proj.weight` |
| `blk.0.attn_q_norm.weight` | `block_0.attention.qk_norm_q.weight` |
| `blk.0.attn_k_norm.weight` | `block_0.attention.qk_norm_k.weight` |
| `blk.0.ffn_norm.weight` | `block_0.pre_ffn_norm.weight` |
| `blk.0.ffn_gate.weight` | `block_0.ffn.gate_proj.weight` |
| `blk.0.ffn_up.weight` | `block_0.ffn.up_proj.weight` |
| `blk.0.ffn_down.weight` | `block_0.ffn.down_proj.weight` |

Q4_K_M weights are 256-element super-blocks containing 8 × 32-weight
sub-blocks, each with a 6-bit scale and 6-bit min, plus a single FP16
super-block scale. The unpack reproduces the llama.cpp reference; the
RoPE basis must be the `SPLIT_HALF` value (llama.cpp permutes Q/K
weights to use a different RoPE basis from HF — silent degradation if
mismatched, per `RoPESpec.basis`).

GGUF support is **deferred to B-batch B8** rather than M1 to keep M1's
scope tight. The mapping table above is documented now so the work can
land cleanly later.

### 9.4 Loader contract

```python
def load_weights(
    checkpoint_path: str,
    layer: DecoderBlock,
    layer_idx: int,
    *,
    format: Literal["hf_fp16", "hf_awq", "gguf_q4km"] = "hf_fp16",
    quant: Optional[QuantSpec] = None,
) -> None:
    """Load weights for a single decoder block layer.

    For format=hf_awq: reads qweight/qzeros/scales per linear; populates
    QuantizedWeight slots on the layer. The QuantSpec must be passed so
    the loader knows the expected packing/group_size.

    For format=gguf_q4km: reads super-block storage; populates equivalent
    QuantizedWeight slots with QuantSpec(qdtype=INT4, group_size=32,
    packing=GGUF_K_SUPERBLOCK, scale_quant=QuantSpec(qdtype=INT6, ...)).
    """
```

The loader's unit of work is **one layer**. Model-level loading is a thin
loop over layers + embedding + final norm + lm_head. This makes per-layer
testing tractable.

## 10. Per-model rollout (M2+) — revised batches

After Qwen3 approval, the rollout proceeds in **architectural-diversity
order**, batched by similarity. Each new model produces a **diff against
the API**: either it fits (no API changes), or it forces a *named
extension* with justification.

| Batch | Models | API extensions expected |
|---|---|---|
| **B0.5: 2023 baselines** (NEW) | Llama 2 7B, Mistral 7B v0.1 | None — sanity that the API expresses the pre-Llama-3 template (RoPE θ=10k, no QK-norm). Validates the historical floor; only 2 models. |
| **B1: Llama-family dense (Llama-3 era)** | Llama 3.1 (1B/3B/8B), Mistral 7B v0.3, TinyLlama 1.1B | None — Llama-3 RoPE scaling, RMSNorm, SwiGLU. |
| **B2: Llama-family with quirks** | SmolLM 2/3 (SmolLM3 NoPE every-4th moved here per critique 5.2), Granite 3.x, Phi-3 mini, Phi-4 mini | LayerScaleSpec for Granite μP, fused gate-up for Phi-3, LongRoPE for Phi-3, per-layer `rope=None` for SmolLM3 NoPE |
| **B3: Post-norm and dual-norm** | OLMo 2, Gemma 2 | `NormPosition.POST`, `NormPosition.PRE_AND_POST`, sub-norms, BF16-residual requirement for OLMo 2 (per critique 3.A), `LayerScaleSpec.final_logit_softcap=30` for Gemma 2 |
| **B4: Per-layer heterogeneous** | Gemma 3 (SWA/global alternation, dual RoPE bases, attn-softcap), OpenELM (per-layer head counts) ← OpenELM moved here per critique findings | `model_config.layers: list[DecoderBlockSpec]`, dual RoPE per block, `attn_logit_softcap` on AttentionSpec, `head_count_override` on AttentionSpec |
| **B5: MLA family** (expanded) | DeepSeek-V2-Lite, DeepSeek-V3-Lite, **MiniCPM 3** (added per critique 5.3 — was missing from v1) | `AttentionKind.MLA`, `KVCacheSpec.MLA_LATENT_PLUS_KROPE`, MLA-specific dims (`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`), partial RoPE on q_rope only |
| **B6: MoE family** | Mixtral 8×7B, Qwen3-MoE, OLMoE, DeepSeek-V3-MoE | MoESpec variants — DeepSeek's `SIGMOID_PLUS_BIAS` router, shared experts, `GroupRoutingSpec`. Exercises `softmax`/`top_k`/`gather`/`scatter` ops. |
| **B7: SSM-hybrid** (expanded per critique 5.3 + 6.1) | Mamba-1 2.8B, **Mamba-2 2.7B (SSD)**, **Hymba 1.5B (parallel Mamba+Attn)**, Jamba-mini, Zamba2, **RWKV-7** | `TokenMixerKind.{SSM_MAMBA1, SSM_MAMBA2, HYBRID_PARALLEL, HYBRID_ALTERNATING, SSM_RWKV}`, `SSMSpec`/`SSDSpec`/`ConvSpec`, `KVCacheSpec.SSD_STATE_PLUS_CONV`, per-layer alternation. Exercises `conv1d`/`selective_scan` ops. |
| **B8: Quant variants on existing models** | FP8 W8A8 exercised on **Llama-3 8B**; GGUF Q4_K_M exercised on **Mistral 7B v0.3**; MXFP4 exercised on **Llama-3 8B**; IQ2_M exercised on **Qwen3-0.6B** | QuantSpec block_layout extensions, UE8M0 scale dtype, OCP_MX block layout, codebook spec for IQ2_M |
| **B9: Per-layer-heterogeneous deep-dive** (NEW) | OpenELM (per-layer head counts revisited end-to-end), iRoPE pattern (Llama-4 if available) | `head_count_override` end-to-end test, iRoPE per-layer NoPE/RoPE pattern as `irope_layer_pattern` on RoPESpec |

The rollout is finished when one model from each batch has (a)-level
validation and the API has settled (no extension diffs in the last 2–3
models). Target model count (locked, per critique 5.1): **24** — Qwen3
(1) + B0.5 (2) + B1 (4) + B2 (4) + B3 (2) + B4 (2) + B5 (3) + B6 (4) +
B7 (6) — though B8 and B9 reuse earlier models so don't add to the
distinct-family count.

## 11. Risks and mitigations (expanded)

| Risk | Mitigation |
|---|---|
| API mutates too much during rollout, breaks earlier models | Each batch ends with a regression run of all earlier models' shape tests; spec dataclass changes are additive (new Optional fields with `None` default) |
| MLA dimensions break the standard `(n_heads, head_dim)` assumption in shared code | MLA is a first-class `AttentionKind` from M1; shared code paths conditional on `kind`; `MLA_LATENT_PLUS_KROPE` cache layout names the two-tensor structure |
| MoE token dispatch differs across runtimes; the "expert ffn" abstraction may leak | Reference impl uses dense scatter for clarity (`DispatchKind.DENSE_SCATTER`); document that production runtimes use grouped GEMM or token-permute; the building block's *interface* is dispatch-kind-stable |
| GGUF `1+w` baked-into-weight foot-gun (Gemma 3) | `NormSpec.weight_mode` is explicit; layer.md documents which mode the canonical checkpoint uses |
| llama.cpp permutes Q/K weights to switch RoPE basis (silent degradation if missed) | `RoPESpec.basis` is explicit; `models/<family>/layer.md` documents which basis the canonical checkpoint uses; the M1 sub-op ladder catches this on Qwen3 |
| Numerical equivalence test for Qwen3 fails due to subtle initialization or RoPE basis bug | Sub-op isolation ladder (§8.2 d) catches each component before block-level; the `atol=1e-5` per-sub-op bar means a single divergent op stands out |
| KIVI-style asymmetric K/V quantization breaks our symmetric KVCache assumption | `KVCacheSpec` has independent `k_quant` and `v_quant`; verified |
| Phase B grows too slowly to be useful | Subagent-driven development for M2+: dispatch one agent per batch in parallel after M1 approval |
| Post-norm models (OLMo 2) overflow FP16 in the residual stream | `NormSpec` + `DecoderBlockSpec` documents BF16-only requirement for `POST` placement; tests run BF16 storage when `attn_norm_position=POST` |
| Differential transformer dual-stream surprise | Dual streams handled internally to the differential block; if a future model genuinely needs dual residuals, escalate to "Phase D" spec change |
| Hybrid-parallel block (Hymba) breaks the `token_mixer: Union[…]` assumption | Resolved by `token_mixer: Union[AttentionSpec, SSMSpec, SSDSpec, Tuple[AttentionSpec, SSMSpec]]` + `TokenMixerKind.HYBRID_PARALLEL` discriminator |
| AWQ unpack lowering wrong, M1 quant test fails silently | Bit-exact test against `autoawq` reference dequant in §8.2(g); the test is the contract |
| HuggingFace transformers API version drift breaks reference comparisons | Pin `transformers==4.46.x` in `pyproject.toml`; document in §16 (fixture provenance) |
| Continuous-batching `[T,D]` shape mismatch (vLLM, SGLang) | Spec explicitly assumes `[B,S,D]`; `[T,D]` is a runtime concern documented in §2 non-goals; layer.py is shape-agnostic where possible |
| YOCO two-stage cache lifecycle doesn't fit per-layer cache assumption | `cache_role` (PRODUCER/CONSUMER/PRODUCER_CONSUMER/SHARED_GROUP_MEMBER) on both AttentionSpec and KVCacheSpec; the cache *handle* is shared across a block-of-layers at the model assembly level, not per-layer |
| Mutually-exclusive `kind` discriminator fails if a hypothetical model is "MLA + Differential" | Stated as an explicit assumption; no surveyed SLM violates it; promotion to product-of-kinds enum deferred |
| Single-residual-stream assumption violated by Differential transformer | Differential keeps two attentions internal to the building block (single residual at block level); flagged in §11 if a model with genuine dual residuals appears |

## 12. Open questions deferred to writing-plans

1. **Base class for layer factories** — should `models/<family>/layer.py`
   share a `DecoderLayerBase`? Lean: stay standalone for the first few;
   extract a base only if duplication emerges naturally.
2. **Numerical-equivalence reference choice for non-seed models** — HF
   transformers (slow, full Python) or vLLM (faster, more deps)? Lean:
   HF transformers only on CPU with small head counts; this is (b)-tier
   validation.
3. **Speculative decoding compatibility** — out of scope; flag in README
   but don't design for it now.
4. **Distributed / TP sharding** — out of scope; flag.
5. **Adapter (LoRA) mergeability into quantized weights** —
   `QuantSpec.merged_lora_rank` reserved; not built in v2.
6. **ONNX export round-trip** — useful demonstration that the API is
   portable; not built in M1.
7. **Vision-language M-RoPE** — `RoPESpec.mrope_section` reserved; VL is
   out of scope.
8. **Sampling / logits processors** — out of scope; the API ends at
   `lm_head`.
9. **Speculative decoding cache structure (MTP)** — `KVCacheSpec` covers
   the storage shape but the speculation policy is out of scope.
10. **Encoder-decoder cross-attention** — out of scope; SLM corpus is
    decoder-only.

## 13. Evidence index

All design decisions cite one or more of the v2 reports (verbatim
sections):

- **research/00-evolution.md** — timeline 2022→2026, lineage, abandoned
  designs, per-axis evolution arc (referenced by §0 preamble and the
  rollout batch ordering)
- **research/01-model-census.v2.md** — model selection, axis enumeration
  (15 axes), 2023 baseline reintroduced, top-5 surprises fixed
- **research/02-layer-sources.v2.md** — universal subset, 30+ families,
  fixed MLA compression factor (~71×), fixed sandwich-norm terminology,
  added BitNet sub-norms / OpenELM per-layer width / parallel residual
  / continuous-batching divergence
- **research/03-ihv-opsets.v2.md** — 13-runtime synthesis (added ggml,
  Neuron, TPU XLA, DirectML), 16-logical-op floor, KV cache representation
  taxonomy, MoE/embedding/quant rows added
- **research/04-quantization.v2.md** — 42 schemes, 22-axis QuantSpec,
  "support 5" baseline (Q4_K_M, AWQ, FP8 W8A8, IQ2_M, MXFP4), fixed
  Q4_K sub-block size (32×8 not 16×16), fixed imatrix semantics
  (objective weighting, not weight pre-multiply), split `pre_transform`
  into `rotation_kind` + `runtime_apply`
- **research/05-kvcache-attention.v2.md** — 25 attention × 14 RoPE × 12
  cache variants, KVCacheSpec axes expanded (vAttention, YOCO, NSA,
  MTP), AttentionSpec axes expanded (cache_role, block_sparse,
  head_count_override), fixed Qwen3 QK-norm placement (**PRE_ROPE**, not
  POST_ROPE; v1 was wrong per critique 5.1), fixed Qwen3-0.6B
  `rope_theta=1_000_000` (v1 said 5M; HF config confirms 1M), fixed
  MiniCPM-3 → MLA (not CLA), added SSM `d_conv` and SSD axes
  (`chunk_size`, `headdim`, `ngroups`)

## 14. Glossary

### Enum values

`AttentionKind`: `STANDARD` (MHA/GQA/MQA — parameterized by n_kv_heads;
used by Llama family, Qwen family, Mistral, Gemma); `MLA` (Multi-head
Latent Attention with c_kv latent + partial-RoPE on k_rope; used by
DeepSeek-V2/V3, MiniCPM-3); `DIFFERENTIAL` (two attentions subtracted
with learned λ; Microsoft 2024); `LINEAR_RETENTION` (RetNet-style);
`LINEAR_DELTANET` (DeltaNet, Gated DeltaNet); `LINEAR_GLA` (Gated Linear
Attention); `NSA` (DeepSeek 2025 three-tier sparse attention).

`QKVLayout`: `SPLIT` (three independent linears, default); `FUSED`
(single linear projecting to 3D, sliced; used by Phi-3); `MLA_LATENT`
(low-rank c_q, c_kv projections; used by MLA).

`MaskKind`: `CAUSAL` (default); `SWA` (sliding-window; Mistral, Gemma 3
local layers); `SWA_GLOBAL_ALT` (alternating sliding/global per layer;
Gemma 3); `SINK` (attention sinks; GPT-OSS, StreamingLLM); `FULL`
(non-causal — not used by decoders, included for completeness);
`BLOCK_SPARSE` (Phi-3-small); `DOCUMENT_CAUSAL` (packed-sample
training); `CUSTOM`.

`QKNormPhase`: `NONE`; `PRE_ROPE` (Qwen3, per critique 5.1 fix — apply
QK-norm before RoPE); `POST_ROPE` (Gemma 3 — apply after RoPE).

`NormShape`: `FULL_HIDDEN` (norm over `[..., D]`); `PER_HEAD_DH` (per-head
RMSNorm with weight shape `[head_dim]` — Qwen3 QK-norm); `FULL_HDH`
(per-head RMSNorm with full `[H, head_dim]` weight — OLMo 2 QK-norm).

`NormKind`: `RMS` (default); `LAYER` (LayerNorm with bias); `SCALE_NORM`
(T5 / older research models); `DEEP_NORM` (post-norm variant — documented,
training-time only).

`WeightMode`: `NONE` (unscaled RMS); `STANDARD_W` (default RMS with
multiplicative gain); `ONE_PLUS_W` (Gemma's `(1+w) * x_normed` foot-gun);
`LEARNED_PER_HEAD` (QK-norm per-head learned scale).

`GateKind`: `SWIGLU` (silu(gate)*up — Llama family); `GEGLU`
(gelu(gate)*up — Gemma); `GELU_ONLY` (no gate, single up+down — Phi-2);
`RELU2_ONLY` (BitNet).

`Activation`: `SILU`, `GELU`, `GELU_TANH`, `RELU2`, `GEGELU`.

`RouterKind`: `SOFTMAX` (Mixtral); `SIGMOID_PLUS_BIAS` (DeepSeek-V3
auxiliary-loss-free); `LINEAR_TOP_K_NO_NORM` (rare).

`GroupScoreKind` (within `GroupRoutingSpec`): `SUM_TOP_K_IN_GROUP`
(DeepSeek-V3 — sum of top-K expert scores within group); `MAX`; `MEAN`.

`DispatchKind`: `DENSE_SCATTER` (reference); `GROUPED_GEMM` (production);
`TOKEN_PERMUTE` (Megablocks).

`CacheLayout`: `CONTIGUOUS` (allocate max seq); `PAGED` (block-allocated,
vLLM/PagedAttention v1/v2); `RING` (StreamingLLM); `MLA_LATENT_PLUS_KROPE`
(two tensors: c_kv + k_rope); `SSM_STATE` (Mamba-1 state); `SSD_STATE_PLUS_CONV`
(Mamba-2 state + conv1d state); `NSA_THREE_TIER` (compressed +
selected + sliding); `YOCO_SHARED` (single global cache for a block of
layers).

`MemoryLayout`: `HND` (`[H, N, D]`); `NHD` (`[N, H, D]`); `BHND`; `BNHD`
(`[N_blocks, block_size, H, D]` — vLLM); `MLX_LIST`.

`CacheOwnership`: `EXPLICIT_PASS` (functional — cache passed through);
`STATEFUL` (held on the block module).

`CacheRole`: `PRODUCER` (writes cache, doesn't read — YOCO encoder
stage); `CONSUMER` (reads cache, doesn't write — YOCO decoder stage);
`PRODUCER_CONSUMER` (default); `SHARED_GROUP_MEMBER` (CLA pair).

`QDType`: `INT2`, `INT3`, `INT4`, `INT5`, `INT6`, `INT8`, `FP8_E4M3`,
`FP8_E5M2`, `FP4`, `NF4`, `MX_FP4`, `MX_INT4`, `TERNARY` (BitNet
{-1,0,+1}).

`PackingLayout`: `NONE`; `NIBBLE_LSB` / `NIBBLE_MSB`; `AWQ_INTERLEAVE`
(`[0,2,4,6,1,3,5,7]`); `GPTQ_INT32_PACK` (column-major INT32 pack);
`HQQ_NIBBLE`; `GGUF_K_SUPERBLOCK`; `MX_BLOCK`.

`BlockLayout`: `OCP_MX` (32-element micro-blocks with UE8M0 scale);
`GGUF_K` (256-element super-block); `AWQ_INTERLEAVE_K` (g=128 along K);
`GPTQ_BLOCK`.

`RotationKind`: `NONE`; `DIAGONAL_SMOOTHQUANT` (per-channel α);
`DENSE_HADAMARD_QUAROT` (Hadamard via offline reparameterisation);
`DENSE_LEARNED_SPINQUANT` (learned rotation);
`HADAMARD_RUNTIME` (applied at inference each forward).

`RuntimeApply`: `NONE`; `INPUT` (apply at activation input only);
`INPUT_OUTPUT_BOTH` (apply at input AND o_proj input).

`CombineOp` (for codebooks): `NONE` (single codebook); `ADD` (AQLM
additive); `CONCAT`; `MULTI_LATTICE` (QuIP#).

`OutlierOffload`: `NONE`; `FP16_COLUMN_SPLIT` (LLM.int8 — outlier columns
kept in FP16); `FP16_ROW_SPLIT`.

`QuantRole`: `WEIGHT`, `ACTIVATION`, `KV_K`, `KV_V`, `ATTN_INTERNAL`,
`EMBEDDING`.

`ScaleSource`: `STATIC` (pre-computed, stored alongside weights);
`DYNAMIC_PER_TOKEN` (computed at inference per token — FP8 W8A8 A);
`DYNAMIC_PER_TENSOR` (computed at inference per tensor).

`DequantPath`: `IN_SDPA` (dequant fused into SDPA kernel); `AT_READ`
(dequant during cache read); `AT_WRITE` (cache stored in compute dtype).

`RoPEBasis`: `INTERLEAVED` (HF reference — paired channels alternating);
`SPLIT_HALF` (llama.cpp — first half and second half rotated as pairs;
requires Q/K weight permutation).

`RoPEScaling`: `NONE`; `PI` (Position Interpolation, Chen 2023);
`NTK_STATIC`; `DYNAMIC_NTK` (per-call recomputed by seq_len; Qwen2 7B);
`YARN` (Peng 2023); `LLAMA3` (Meta 2024); `LONGROPE` (Microsoft 2024,
Phi-3); `IROPE` (Llama-4 interleaved RoPE/NoPE).

`NormPosition`: `PRE` (default); `POST` (OLMo 2); `PRE_AND_POST`
(sandwich — Gemma 2/3); `PARALLEL_FFN` (Falcon-7B, GPT-J historical —
attn and FFN consume the same normed input).

`TokenMixerKind`: see §5.2.5 definition.

### Acronyms

- **ALiBi** — Attention with Linear Biases (Press et al. 2021)
- **AWQ** — Activation-aware Weight Quantization (Lin et al. 2023)
- **CLA** — Cross-Layer Attention (sharing KV across pairs of layers)
- **DeepNorm** — post-norm variant with scaled residual (Wang et al. 2022)
- **FA1/2/3** — FlashAttention algorithms (Dao 2022, 2023, 2024)
- **FP8** — 8-bit floating point (E4M3 or E5M2 mantissa/exponent split)
- **GQA** — Grouped-Query Attention (Ainslie et al. 2023)
- **GGUF** — GPT-Generated Unified Format (llama.cpp tensor file format)
- **HQQ** — Half-Quadratic Quantization (Badri 2024 — calibration-free)
- **iRoPE** — interleaved RoPE/NoPE (Llama 4 design)
- **KV cache** — cached Key and Value tensors of past tokens
- **LayerScale** — per-channel learned residual multiplier (Touvron et al. 2021)
- **LRU** — Linear Recurrent Unit (Orvieto et al. 2023; basis of Griffin)
- **MLA** — Multi-head Latent Attention (DeepSeek-V2, 2024)
- **MoE** — Mixture of Experts
- **MQA** — Multi-Query Attention (Shazeer 2019)
- **MTP** — Multi-Token Prediction (DeepSeek-V3 training objective)
- **MXFP4** — 4-bit microscaling FP via the OCP Micro-scaling spec
- **NF4** — Normal Float 4-bit (Dettmers 2023; bitsandbytes)
- **NoPE** — No Positional Encoding (Kazemnejad 2023)
- **NSA** — Native Sparse Attention (DeepSeek 2025)
- **NTK** — Neural Tangent Kernel-aware scaling (bloc97 2023)
- **PagedAttention / PA v1, v2** — vLLM page-allocated KV cache
- **PI** — Position Interpolation (Chen et al. 2023)
- **QK-Norm** — RMSNorm applied to Q and K before/after RoPE
- **QuaRot / SpinQuant** — rotation-before-quantize techniques (2024)
- **RMSNorm** — Root-Mean-Square LayerNorm (Zhang & Sennrich 2019)
- **RoPE** — Rotary Position Embedding (Su et al. 2021)
- **SDPA** — Scaled Dot-Product Attention (the post-softmax fused op)
- **SSD** — Structured State-Space Duality (Dao & Gu 2024; Mamba-2)
- **SSM** — State-Space Model (Gu et al. 2022; Mamba 2023)
- **SwiGLU** — Swish-Gated Linear Unit activation (Shazeer 2020)
- **vAttention** — virtual-memory-mapped KV cache (Microsoft Research)
- **WKV** — RWKV's recurrent attention update
- **YaRN** — Yet another RoPE extensioN method (Peng et al. 2023)
- **YOCO** — You Only Cache Once (Microsoft 2024)
- **μP** — Maximal Update Parameterisation (Yang et al. 2022)

## 15. Bibliography

Foundational decoder-only architecture:

- Vaswani et al. 2017. "Attention is all you need." NeurIPS.
- Radford et al. 2018, 2019. GPT-1 / GPT-2 papers (OpenAI).
- Brown et al. 2020. "Language models are few-shot learners." GPT-3, NeurIPS.

Normalisation, activation:

- Zhang & Sennrich 2019. "Root Mean Square Layer Normalization." NeurIPS.
- Shazeer 2020. "GLU Variants Improve Transformer." arXiv:2002.05202.
- Henry et al. 2020. "Query-Key Normalization for Transformers." EMNLP.
- Wang et al. 2022. "DeepNet: Scaling Transformers to 1000 Layers." arXiv.
- Yang et al. 2022. "Tensor Programs V: Tuning Large Networks via Zero-Shot
  Hyperparameter Transfer." NeurIPS (μP).

Positional encoding:

- Su et al. 2021. "RoFormer: Enhanced Transformer with Rotary Position
  Embedding." arXiv:2104.09864 (RoPE).
- Press et al. 2021. "Train Short, Test Long: ALiBi." ICLR.
- Chen et al. 2023. "Extending Context Window via Position Interpolation."
  arXiv:2306.15595 (PI).
- bloc97 2023. "NTK-aware RoPE scaling." Reddit /r/LocalLLaMA.
- Peng et al. 2023. "YaRN." arXiv:2309.00071.
- Kazemnejad et al. 2023. "The Impact of Positional Encoding on Length
  Generalisation in Transformers" (NoPE).
- Microsoft 2024. "LongRoPE." arXiv (Phi-3).
- Meta 2024. Llama 3 technical report (Llama-3 RoPE scaling).

Attention variants:

- Shazeer 2019. "Fast Transformer Decoding: One Write-Head is All You
  Need." arXiv (MQA).
- Ainslie et al. 2023. "GQA: Training Generalized Multi-Query Transformer."
  EMNLP.
- DeepSeek-AI 2024. "DeepSeek-V2." arXiv:2405.04434 (MLA).
- Microsoft 2024. "Differential Transformer." arXiv:2410.05258.
- Dao 2022. "FlashAttention." NeurIPS (FA1).
- Dao 2023. "FlashAttention-2." arXiv (FA2).
- Shah et al. 2024. "FlashAttention-3." arXiv (FA3).
- DeepSeek 2025. "Native Sparse Attention (NSA)." arXiv.

KV-cache variants:

- Kwon et al. 2023. "Efficient Memory Management for Large Language Model
  Serving with PagedAttention." SOSP (vLLM).
- Microsoft 2024. "vAttention." arXiv.
- Microsoft 2024. "YOCO: You Only Cache Once." arXiv:2405.05254.

State-space models:

- Gu et al. 2022. "S4: Efficiently Modeling Long Sequences with Structured
  State Spaces." ICLR.
- Gu & Dao 2023. "Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces." arXiv:2312.00752.
- Dao & Gu 2024. "Transformers are SSMs (Mamba-2)." arXiv:2405.21060.
- Peng et al. 2025. "RWKV-7." arXiv.
- Orvieto et al. 2023. "Resurrecting RNNs for Long Sequences." ICML
  (LRU — basis of Griffin / RecurrentGemma).
- NVIDIA 2024. "Hymba." arXiv (parallel SSM+attn).

Mixture of Experts:

- Shazeer et al. 2017. "Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer." ICLR.
- Fedus, Zoph & Shazeer 2021. "Switch Transformer." JMLR.
- Mistral AI 2023. "Mixtral of Experts." arXiv:2401.04088.
- DeepSeek-AI 2024. "DeepSeek-V3." arXiv (shared+routed experts,
  sigmoid-plus-bias routing, group-limited routing,
  auxiliary-loss-free balancing).

Quantization:

- Frantar et al. 2022. "GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers." arXiv (GPTQ).
- Xiao et al. 2022. "SmoothQuant." ICML.
- Lin et al. 2023. "AWQ: Activation-aware Weight Quantization." MLSys.
- Dettmers et al. 2023. "QLoRA / NF4." NeurIPS.
- Badri 2024. "HQQ: Half-Quadratic Quantization." Mobius Labs blog.
- Microsoft 2024. "BitNet b1.58." arXiv:2402.17764.
- Egiazarian et al. 2024. "AQLM." ICML.
- Open Compute Project 2024. "OCP Microscaling (MX) Specification." (MXFP4,
  MXINT4).
- NVIDIA 2024. "NVFP4." Blackwell whitepaper.
- Ashkboos et al. 2024. "QuaRot." arXiv.
- Liu et al. 2024. "SpinQuant." arXiv.
- Tseng et al. 2024. "QuIP#." ICML.

Model families surveyed:

- Meta 2023. "LLaMA / LLaMA 2." arXiv.
- Meta 2024. "Llama 3 / Llama 3.1." technical reports.
- Meta 2025. "Llama 4." (iRoPE design).
- Mistral AI 2023. "Mistral 7B." arXiv.
- Alibaba 2024. "Qwen2." arXiv.
- Alibaba 2025. "Qwen3." technical report.
- Google 2024. "Gemma 2." arXiv.
- Google 2024–2025. "Gemma 3." technical report.
- Microsoft 2024. "Phi-3." arXiv.
- Microsoft 2024. "Phi-4 / Phi-4-mini." technical reports.
- IBM 2024. "Granite 3.x." technical report.
- AllenAI 2024. "OLMo 2." arXiv.
- HuggingFace 2024–2025. "SmolLM, SmolLM 2/3." reports.
- DeepSeek-AI 2024. "DeepSeek-V2 / V3." arXiv.
- MiniCPM 2024. "MiniCPM 3." arXiv (MLA at small scale).
- AI21 2024. "Jamba." arXiv.
- Zyphra 2024. "Zamba2." arXiv.
- NVIDIA 2024. "Hymba." arXiv.
- BlinkDL et al. 2025. "RWKV-7." arXiv.

IHV / runtime references (version-pinned in research/03-ihv-opsets.v2.md):

- vLLM project; TensorRT-LLM (NVIDIA); OpenVINO (Intel); QAIRT
  (Qualcomm); FastFlowLM (AMD); MIGraphX (AMD); Core ML (Apple); MLX
  (Apple); KleidiAI (Arm); MindIE (Huawei); ggml/llama.cpp; AWS Neuron;
  Google TPU XLA/Pallas; DirectML (Microsoft).

## 16. Code style, fixtures, contribution, versioning, perf budgets

### 16.1 Code style and quality bar

- **Formatting:** `black --line-length 100` on all `api/` and `models/`
  Python files.
- **Linting:** `ruff` with the configuration in `pyproject.toml`. CI runs
  `ruff check . && ruff format --check .`.
- **Type checking:** `mypy --strict` on `api/`; `mypy` (non-strict, but
  `--warn-unused-ignores`) on `models/`. The api/ surface is fully typed;
  layer factories may use Any sparingly in HF-config conversions.
- **Testing:** `pytest --strict-markers --tb=short`. Each model's
  `test_layer.py` runs in CI; the Qwen3 numerical-equivalence test runs
  on the seed only (slow due to HF weight load).
- **Pre-commit:** `pre-commit` hook chains black + ruff + mypy
  (api only).

### 16.2 Test fixture provenance

- **Qwen3-0.6B weights:** `Qwen/Qwen3-0.6B` on HuggingFace Hub. License:
  **Apache 2.0** (Qwen3 series is Apache 2.0; earlier Qwen had Tongyi
  Qianwen). Pinned revision: the commit SHA recorded in
  `models/qwen3/fixtures.toml` at M1 freeze time. Downloaded once into the
  CI cache and reused; do *not* refetch on every run.
- **Qwen3-0.6B AWQ weights:** if `Qwen/Qwen3-0.6B-AWQ` exists at M1
  time, use it; else quantize the FP16 checkpoint with `autoawq` and
  pin the resulting safetensors. Pin both the input checkpoint SHA and
  the autoawq version.
- **HuggingFace transformers:** pin `transformers==4.46.x` in
  `pyproject.toml`. The numerical-equivalence test breaks silently if HF
  changes the reference layer; the pin guards against that.

### 16.3 Contribution process (adding a new model)

Five steps from upstream config to merged PR:

1. **Census check.** Confirm the model is in `research/01-model-census.v2.md`.
   If not, add a row first (PR sequence: census → layer survey → spec
   extension → model PR).
2. **Layer doc.** Write `models/<family>/layer.md` per the §6 schema,
   including the weight-name mapping table (§9).
3. **Spec instantiation.** Implement `models/<family>/layer.py` with a
   `build_decoder_layer(config, layer_idx)` factory. If the family forces
   an API extension, file the extension as a separate PR first and cite
   it in the model PR.
4. **Weight loader.** Implement `models/<family>/weight_loader.py` for
   the canonical HF checkpoint format (+ AWQ / GGUF / FP8 variants when
   they exist).
5. **Tests.** Add `models/<family>/test_layer.py` with shape, determinism,
   KV cache, and (where applicable) numerical-equivalence + quant
   round-trip tests. Tests must pass on CPU within the perf budget
   (§16.5).

### 16.4 API versioning

- The API follows semver. The first cut of `api/` after M1 approval is
  `v0.1.0`. The version is unstable through M9 (post-rollout); breaking
  changes are allowed, but each change carries a migration note in
  `CHANGELOG.md`.
- Spec dataclass extensions are *additive only* by convention: a new
  field must default to `None` (or a backward-compatible default). Field
  *removals* and *type changes* are breaking and require a new minor
  version.
- Stability commitment begins at `v1.0.0`, which we tag after the last
  B-batch lands and the API has had two consecutive batches with no
  changes.

### 16.5 Performance budgets (documented, not enforced)

These are guidelines to catch egregious regressions, not hard CI gates:

- **Qwen3-0.6B single decoder layer, FP16, seq_len=128, batch=1,
  CPU:** < 100 ms forward in PyTorch eager. The numerical-equivalence
  test at seq_len=9 runs comfortably under 1 s end-to-end.
- **AWQ unpack + GEMM for one linear (in_features=1024, out_features=1024,
  g=128), CPU FP16:** < 50 ms. (Unpack dominates; this is the unit cost
  for evaluating dequant correctness, not a production target.)
- **KV cache write/read for prefill 128 + decode 1, CONTIGUOUS layout,
  FP16:** < 20 ms.

If any of these exceed by >2×, file an issue; it usually indicates a
naive Python loop where a tensor op should be. The PyTorch eager
reference will not be production-fast; these budgets are an early-warning
system, not a competitive bar.

---

**Ready for review.** The next step after spec approval is the
`superpowers:writing-plans` skill, which produces a detailed
implementation plan with the Qwen3 sample as Milestone 1 (kickoff gate),
B0.5 as M2 (sanity batch), and B1–B9 as M3–M11. The plan covers task
ordering, per-batch parallelism via subagent-driven development, and
checkpoint gates.
