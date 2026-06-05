# llm-layers — Design Spec

**Date:** 2026-06-04
**Status:** Draft, awaiting user review
**Owner:** zhengte@microsoft.com

---

## 1. Goal

Produce a **minimal, evidence-grounded API set** capable of assembling any mainstream small language model (SLM, <8B params) as a transformer decoder graph — with **quantization** and **KV-cache** as first-class parameters, not afterthoughts.

The API is the IR. The deliverable proves the API by re-implementing one decoder block from each architecturally-distinct SLM family using only API primitives. A model that requires no new primitives "fits"; a model that requires a new primitive or new parameter axis forces a deliberate, evidence-justified API extension. Final "minimal" API = the floor that survives every model.

The same primitives are intended to express models in PyTorch, GGUF, and ONNX-style runtimes — so design choices follow runtime/IHV consensus where it exists (see `research/03-ihv-opsets.md`).

## 2. Non-goals

- Training or fine-tuning code. Random weights for shape tests; HF weights only for the Qwen3 numerical-equivalence check.
- Full-model inference pipeline. We implement **one decoder block per family**, not full forward passes, tokenizers, sampling, or sharding.
- Performance kernels. PyTorch eager only. The API must be *expressible* in IHV runtimes; we don't implement them.
- Bit-exact equivalence across all models. Phase B targets structural & shape equivalence; numerical equivalence is only proven for Qwen3 in the kickoff gate.

## 3. Sub-projects and sequencing

| Phase | Output | Status |
|---|---|---|
| **A. Model census + variant-axis catalog** | `research/01-model-census.md` — 45 models, 22 archs, 11 axes | ✅ Done |
| **A.1 Cross-source layer survey** | `research/02-layer-sources.md` — 10 families, HF/vLLM/llama.cpp triple-source | ✅ Done |
| **A.2 IHV op-set survey** | `research/03-ihv-opsets.md` — 9 runtimes, cross-vendor synthesis | ✅ Done |
| **A.3 Quantization survey** | `research/04-quantization.md` — 25 schemes, 15-axis parameter space | ✅ Done |
| **A.4 KV-cache + attention survey** | `research/05-kvcache-attention.md` — 13 attn × 8 RoPE × 6 cache | ✅ Done |
| **B. Per-model layer reference impls** | `models/<family>/{layer.md, layer.py, test_layer.py}` ×N | Plan output |
| **C. Minimal-API synthesis** | `api/` package (the IR) | Plan output |

Sequencing: A is done. B and C **co-evolve** in execution — each new model in B either confirms or extends C. The Qwen3 sample (Section 8) is the seed that bootstraps both.

## 4. Project layout

```
C:\Users\zhengte\external\llm-layers\
├── api/                            # Phase C — the minimal API (the IR)
│   ├── __init__.py                 # public surface
│   ├── ops.py                      # 9 logical primitives (functional)
│   ├── specs.py                    # AttentionSpec, RoPESpec, NormSpec, FFNSpec, MoESpec, QuantSpec, KVCacheSpec
│   ├── attention.py                # Attention block (parameterized)
│   ├── feedforward.py              # FFN + MoE (parameterized)
│   ├── norm.py                     # RMSNorm + LayerNorm + QK-norm variants
│   ├── rope.py                     # RoPE variants (vanilla / Llama-3 / YaRN / LongRoPE / partial / M-RoPE)
│   ├── kvcache.py                  # KVCache type + layouts (contiguous / paged / ring / MLA-latent / SSM-state)
│   ├── quant.py                    # QuantSpec realization + dtype casts + packed-int helpers
│   ├── block.py                    # DecoderBlock assembly (token mixer + channel mixer + residual structure)
│   └── README.md                   # API quick reference
├── models/                         # Phase B — per-model layers
│   ├── qwen3/
│   │   ├── layer.md                # explanation, IO tensors, op trace, source citations
│   │   ├── layer.py                # build_qwen3_decoder_layer(config) -> nn.Module
│   │   └── test_layer.py           # shape + dtype tests, optional numerical-equivalence test
│   ├── llama3/...
│   ├── gemma3/...
│   ├── phi3/...
│   ├── deepseek_v3/...            # MLA
│   ├── mixtral/... or qwen3_moe/  # MoE
│   ├── jamba/...                   # SSM-hybrid
│   └── ... (~25 total — see §9)
├── research/                       # Already populated by parallel agents
│   ├── 01-model-census.md
│   ├── 02-layer-sources.md
│   ├── 03-ihv-opsets.md
│   ├── 04-quantization.md
│   └── 05-kvcache-attention.md
├── docs/superpowers/specs/
│   └── 2026-06-04-llm-layers-design.md   ← this file
├── plans/                          # Implementation plan (from writing-plans skill)
├── pyproject.toml                  # uv-managed venv, deps: torch, safetensors, numpy, pytest
├── uv.lock
└── README.md
```

Python env: `uv venv && uv sync` (Python 3.11+). Dependencies: `torch`, `safetensors`, `numpy`, `pytest`, `transformers` (for Phase B numerical-equivalence cross-checks only — NOT for assembling layers).

## 5. Phase C — Minimal API design framework

The API has four layers:

1. **Logical ops** (functional primitives) — IHV-consensus floor
2. **Specs** (dataclasses) — the parameter spaces that vary across models
3. **Building blocks** (small classes) — attention, FFN, MoE, RoPE, norm, KV cache; consume specs
4. **DecoderBlock** — assembles a token mixer + channel mixer with residual structure

### 5.1 Logical ops — the IHV-consensus floor

Across the 9 IHV runtimes surveyed (OpenVINO, QAIRT, FastFlowLM, MIGraphX, TensorRT-LLM, Core ML, MLX, KleidiAI, MindIE) and ONNX contrib operators, this is the smallest functional set every runtime exposes (≥7/9 consensus):

| Op | Signature (shapes annotated) | IHV consensus |
|---|---|---|
| `rms_norm(x, weight, eps, mode)` | `x:[B,S,D] → [B,S,D]`; `mode ∈ {classic, gemma_1plus_w}` | 9/9 |
| `linear(x, weight, bias?, quant?)` | `x:[B,S,K] × w:[N,K] → [B,S,N]`; quant via `QuantSpec` | 9/9 |
| `rope_apply(qk, freqs, kind, partial_dim?)` | `qk:[B,S,H,Dh] → [B,S,H,Dh]` | 6/9 explicit, 3/9 fused |
| `sdpa(q, k, v, mask, scale, softcap?, sink?, sliding_window?)` | `[B,H,S,Dh] × [B,Hk,S,Dh] × [B,Hk,S,Dv] → [B,H,S,Dv]` | 9/9 fused (granularity varies) |
| `silu(x)` / `gelu(x)` / `relu2(x)` | element-wise | 9/9 |
| `add(x, residual, scale?)` | `[B,S,D]`; optional Granite-style residual multiplier | 9/9 |
| `mul(gate, up)` | `[B,S,F]` element-wise for SwiGLU | 9/9 |
| `embed(ids, weight, scale?)` | `ids:[B,S] → [B,S,D]`; optional Gemma `sqrt(D)` scale | 9/9 |
| `lm_head(x, weight, scale?, softcap?)` | `[B,S,D] → [B,S,V]`; logits scaling (Granite) and softcap (Gemma) | 9/9 |

Implementation style: **pure functions** in `api/ops.py`, no PyTorch-only assumptions in signatures (positional tensors, named scalars, specs by value). PyTorch reference implementation provided; ONNX/runtime backends would slot in behind the same signatures.

### 5.2 Specs — the parameter spaces

Specs are **frozen dataclasses** (`@dataclass(frozen=True)`). They are the source of truth for "what varies across models." Each spec carries only data; behavior lives in the building blocks that consume them.

#### 5.2.1 `AttentionSpec` — ~19 axes (research/05, §11)

```python
@dataclass(frozen=True)
class AttentionSpec:
    # Projection topology
    n_q_heads: int
    n_kv_heads: int                              # GQA: == n_q_heads // gqa_groups; MQA: 1
    head_dim: int                                # ≠ hidden/n_q_heads in MLA
    kind: AttentionKind                          # {STANDARD, MLA, DIFFERENTIAL, LINEAR}
    qkv_layout: QKVLayout                        # {SPLIT, FUSED, MLA_LATENT}

    # Biases (4 independent flags — Llama-3 has q_proj bias but not o_proj, etc.)
    q_bias: bool = False
    k_bias: bool = False
    v_bias: bool = False
    o_bias: bool = False

    # MLA-specific (None unless kind==MLA)
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # Scale and softcap
    attn_scale: Optional[float] = None           # overrides 1/sqrt(head_dim) (Gemma)
    logit_softcap: Optional[float] = None        # Gemma 2/3

    # Mask
    mask_kind: MaskKind                          # {CAUSAL, SWA, SWA_GLOBAL_ALT, SINK, FULL, CUSTOM}
    sliding_window: Optional[int] = None
    n_sink_tokens: Optional[int] = None

    # QK normalization
    qk_norm: Optional[NormSpec] = None
    qk_norm_phase: QKNormPhase = QKNormPhase.NONE  # {NONE, PRE_ROPE, POST_ROPE}
    qk_norm_shape: QKNormShape = QKNormShape.NONE  # {NONE, PER_HEAD_DH, FULL_HDH}  ← Qwen3 vs OLMo 2

    # RoPE
    rope: Optional[RoPESpec] = None
    rope_partial_dim: Optional[int] = None       # partial rotation (MLA, GPT-J)

    # Cross-layer sharing
    shares_kv_with: Optional[int] = None         # layer index this layer reuses

    # Differential-transformer extras (skip if kind != DIFFERENTIAL)
    diff_lambda_init: Optional[float] = None
```

#### 5.2.2 `RoPESpec`

```python
@dataclass(frozen=True)
class RoPESpec:
    base_theta: float                            # 10000 / 500000 / 1e6
    basis: RoPEBasis                             # {INTERLEAVED, SPLIT_HALF}  ← llama.cpp permute
    scaling: RoPEScaling                         # {NONE, NTK, YARN, LLAMA3, LONGROPE}
    scale_factor: Optional[float] = None
    yarn_extra: Optional[YarnParams] = None
    llama3_extra: Optional[Llama3RoPEParams] = None
    longrope_extra: Optional[LongRoPEParams] = None      # Phi-3: short/long factor vectors
    mrope_section: Optional[Tuple[int, ...]] = None      # M-RoPE for VL
```

#### 5.2.3 `NormSpec`

```python
@dataclass(frozen=True)
class NormSpec:
    kind: NormKind                               # {RMS, LAYER}
    eps: float
    weight_mode: WeightMode                      # {STANDARD_W, ONE_PLUS_W}  ← Gemma
    # placement is on DecoderBlockSpec, not here
```

#### 5.2.4 `FFNSpec` + `MoESpec`

```python
@dataclass(frozen=True)
class FFNSpec:
    intermediate_size: int
    activation: Activation                       # {SILU, GELU, GEGELU, RELU2}
    gate_kind: GateKind                          # {SWIGLU, GEGLU, GELU_ONLY, RELU2_ONLY}
    fused_gate_up: bool = False                  # Phi-3 fuses
    gate_bias: bool = False
    up_bias: bool = False
    down_bias: bool = False

@dataclass(frozen=True)
class MoESpec:
    n_experts: int
    top_k: int
    n_shared_experts: int = 0
    router_kind: RouterKind                      # {SOFTMAX, SIGMOID_PLUS_BIAS}  ← DeepSeek-V3
    router_norm: bool = False
    score_correction_bias: bool = False
    group_routing: Optional[GroupRoutingSpec] = None     # DeepSeek-V3
    routed_scaling_factor: float = 1.0
    expert_ffn: FFNSpec                          # shape of one expert
```

#### 5.2.5 `KVCacheSpec` — 8 axes (research/05, §10)

```python
@dataclass(frozen=True)
class KVCacheSpec:
    layout: CacheLayout                          # {CONTIGUOUS, PAGED, RING, MLA_LATENT, SSM_STATE}
    memory_layout: MemoryLayout                  # {HND, NHD}
    block_size: Optional[int] = None             # for PAGED, RING
    k_dtype: DType
    v_dtype: DType
    k_quant: Optional[QuantSpec] = None          # KIVI: K per-channel
    v_quant: Optional[QuantSpec] = None          # KIVI: V per-token (asymmetric to K!)
    ownership: CacheOwnership                    # {EXPLICIT_PASS, STATEFUL}
    share_group: Optional[int] = None            # CLA
    # For MLA: shape is [c_kv: kv_lora_rank, k_rope: qk_rope_head_dim] per token
    # For SSM: shape is [d_state, d_inner] per token
```

#### 5.2.6 `QuantSpec` — 7 minimal + 8 extended axes (research/04, §10)

```python
@dataclass(frozen=True)
class QuantSpec:
    # Minimal core (covers AWQ/GPTQ/SmoothQuant/FP8 W8A8)
    qdtype: QDType                               # {INT4, INT8, FP8_E4M3, FP8_E5M2, FP4, NF4, MX_FP4, ...}
    group_size: Optional[int]                    # None=per-tensor, -1=per-channel, N=blockwise
    quant_axis: int                              # output axis for W; channel/token axis for A
    scale_dtype: DType                           # {FP32, FP16, BF16, UE8M0, FP8}
    has_zero_point: bool
    packing: PackingLayout                       # {NONE, NIBBLE_LSB, NIBBLE_MSB, AWQ_INTERLEAVE, GGUF_K, ...}
    accumulator_dtype: DType                     # FP32 / FP16 / INT32

    # Extended (covers k-quants, NVFP4, NF4, double-quant)
    scale_quant: Optional['QuantSpec'] = None    # recursive: double-quant / NVFP4 outer / k-quant super-scale
    codebook: Optional[CodebookSpec] = None      # NF4 / IQ-quants
    scale_axis: Optional[int] = None             # k-quants: scales on different axis from values
    zero_dtype: Optional[DType] = None
    compute_dtype: Optional[DType] = None        # storage ≠ compute
    block_layout: Optional[BlockLayout] = None   # OCP MX / GGUF / AWQ-interleave specific
    pre_transform: Optional[PreTransform] = None # Hadamard (QuaRot), SmoothQuant per-channel
    role: QuantRole = QuantRole.WEIGHT           # {WEIGHT, ACTIVATION, KV_K, KV_V, ATTN_INTERNAL}
```

**"Support only 3" baseline** (from research/04, §11):
1. **GGUF Q4_K_M** — dominant on-device (llama.cpp, Ollama, LM Studio, MLX)
2. **AWQ INT4 W4A16 grouped (g=128)** — dominant server-side 4-bit (vLLM, TRT-LLM, SGLang)
3. **FP8 E4M3 W8A8** (per-tensor W, per-token A) — dominant compute-quantized (H100, MI300, Gaudi 3)

These three together exercise every QuantSpec axis. Quant rollout follows the model rollout: Qwen3 sample exercises AWQ; the model after Qwen3 exercises GGUF Q4_K_M; one of the Llama-3 / Mistral models exercises FP8.

### 5.3 Building blocks

```
Attention(spec: AttentionSpec)               # composes linear, rope, sdpa, kv read/write, norm
FeedForward(spec: FFNSpec)                   # composes 2-3 linears + activation + mul
MoE(spec: MoESpec)                           # router + dispatch + per-expert FeedForward + combine
RMSNorm(spec: NormSpec, hidden_size)
RoPE(spec: RoPESpec)                         # holds freq tables; applies via api.ops.rope_apply
KVCache(spec: KVCacheSpec, layer_idx, max_seq, ...)   # buffer ownership + read/write
```

### 5.4 DecoderBlock assembly

```python
@dataclass(frozen=True)
class DecoderBlockSpec:
    # Residual structure (covers OLMo 2 post-norm, Gemma dual pre+post)
    attn_norm_position: NormPosition             # {PRE, POST, PRE_AND_POST}
    ffn_norm_position: NormPosition
    residual_scale: Optional[float] = None       # Granite μP
    embedding_scale: Optional[float] = None
    logits_scale: Optional[float] = None

    # Sub-specs (one of attention/ssm; one of ffn/moe)
    token_mixer: Union[AttentionSpec, SSMSpec]
    channel_mixer: Union[FFNSpec, MoESpec]
    input_norm: NormSpec
    pre_attn_norm: Optional[NormSpec]            # used when position==PRE
    post_attn_norm: Optional[NormSpec]
    pre_ffn_norm: Optional[NormSpec]
    post_ffn_norm: Optional[NormSpec]
```

This handles:
- OLMo 2 post-norm only → set `attn_norm_position=POST`, no `pre_attn_norm`
- Gemma 3 sandwich → both pre+post norms around each sublayer
- Granite μP → `residual_scale=0.22`, `embedding_scale=12`, `logits_scale=8`
- Per-layer heterogeneity (Gemma 3 SWA alternation, Jamba SSM/attn mix) → a model's `ModelConfig` carries a `list[DecoderBlockSpec]` of length `num_layers` instead of one spec replicated `num_layers` times

## 6. Phase B — Per-model layer factory pattern

Each `models/<family>/` contains:

### `layer.md`

Required sections:
1. **Identity** — family name, variants & param counts, release date, source paper/repo
2. **Decoder block diagram** — ASCII or mermaid showing residual structure, sub-layers, normalization placement
3. **Tensor IO trace** — per-step shapes from input `x:[B,S,D]` to output `[B,S,D]`, every intermediate annotated
4. **Op trace** — sequence of `api.ops.*` calls that implements the block
5. **Spec instantiation** — the `DecoderBlockSpec` values for the canonical variant (e.g., Qwen3-1.7B)
6. **Quirks** — anything model-specific that isn't a generic axis (e.g., `1+w` RMSNorm baking foot-gun, residual scaling)
7. **Source citations** — file:line into HF transformers, vLLM, llama.cpp (from `research/02-layer-sources.md`)

### `layer.py`

```python
def build_decoder_layer(config: <Family>Config, layer_idx: int = 0) -> DecoderBlock:
    """Assemble one decoder block using api/ primitives only.

    Must NOT import from transformers.models.<family>. Must use only api/.
    """
```

Where `<Family>Config` is a dataclass containing the hyperparameters (size variants → different configs). The factory translates the config into `DecoderBlockSpec` and calls `api.block.DecoderBlock(spec)`.

### `test_layer.py`

Required tests:
- **Shape test**: instantiate with random weights at ≥2 size variants; forward pass; assert output shape & dtype
- **Determinism**: same seed → same output
- **KV cache write/read**: prefill one token, decode next, assert shape & cache state evolves correctly
- For Qwen3 only (kickoff gate): **numerical-equivalence test** — load real HF weights into the API-assembled layer, compare against HF reference layer forward output within fp tolerance

## 7. Validation hierarchy

| Level | What it proves | Coverage |
|---|---|---|
| **(a) Structural** | Shape & dtype correct, residual structure correct | All 25 models |
| **(b) Op-trace** | API ops match the "intended" graph (verified against HF source op-by-op) | ~5 representative families (Qwen3, Gemma3, DeepSeek-V3, Mixtral, Jamba) |
| **(c) Numerical** | Real HF weights → API-assembled layer produces fp-equivalent output to HF reference | Qwen3 sample only (kickoff gate) |

Numerical equivalence is the strongest claim and most expensive; we prove it once on the seed to demonstrate the framework is sound, then rely on (a) + (b) for scale.

## 8. Qwen3 sample — kickoff gate (M1)

**Definition of done for the kickoff slice — user reviews before per-model rollout (M2+):**

| Artifact | Contents |
|---|---|
| `api/ops.py` | The 9 logical ops, PyTorch reference impl |
| `api/specs.py` | All spec dataclasses (Attention, RoPE, Norm, FFN, MoE, KVCache, Quant, DecoderBlock) |
| `api/norm.py` | `RMSNorm` block, supporting `STANDARD_W` and `ONE_PLUS_W` modes, `qk_norm` variants `PER_HEAD_DH` and `FULL_HDH` |
| `api/rope.py` | RoPE block, `Llama3` scaling variant (Qwen3 reuses Llama-3 rope structure) |
| `api/attention.py` | Attention block supporting GQA + QK-norm + RoPE + KV cache (contiguous layout) |
| `api/feedforward.py` | FeedForward block, SwiGLU activation |
| `api/kvcache.py` | KVCache with CONTIGUOUS layout, FP16/BF16 dtypes |
| `api/quant.py` | QuantSpec realization, AWQ W4A16 (g=128) packed-int weight load + dequant + matmul path |
| `api/block.py` | DecoderBlock with PRE norm position |
| `models/qwen3/layer.md` | Full layer doc following §6 schema, citing `research/02-layer-sources.md` Qwen3 section |
| `models/qwen3/layer.py` | `build_qwen3_decoder_layer(config, layer_idx)` for Qwen3-0.6B / 1.7B / 4B / 8B (all dense) |
| `models/qwen3/test_layer.py` | (a) shape test 3 sizes, (b) determinism, (c) KV cache prefill+decode, (d) AWQ W4A16 instantiation, (e) **numerical equivalence vs `transformers.Qwen3DecoderLayer` on a known prompt within 1e-3** |

**The kickoff review is the gate that authorizes per-model rollout.** User reviews:
- Is the API surface what you expected?
- Is the spec dataclass pattern right (`AttentionSpec` etc.)?
- Is `models/qwen3/layer.md` the right doc format?
- Is `build_qwen3_decoder_layer` the right factory shape?
- Does the AWQ exercise look right for quantization?
- Did we get the numerical-equivalence check?

Approval gates M2 — the parallel rollout across remaining models.

**Why Qwen3 is the right seed:** GQA + RoPE (Llama-3 scaling) + RMSNorm + SwiGLU + QK-norm. It's the Llama-family template plus one named extension (QK-norm), so the API floor lands quickly and the QK-norm axis gets exercised in the seed instead of being an afterthought.

## 9. Per-model rollout (M2+)

After Qwen3 approval, the rollout proceeds in **architectural-diversity order**, batched by similarity. Each new model produces a **diff against the API**: either it fits (no API changes), or it forces a *named extension* with justification.

| Batch | Models | API extensions expected |
|---|---|---|
| **B1: Llama-family dense** | Llama 3.x (1B/3B/8B), Mistral 7B, SmolLM 2/3, TinyLlama | None — all should fit Qwen3 API minus QK-norm |
| **B2: Llama-family with quirks** | Granite 3.x, MiniCPM 3, Phi-3 mini, Phi-4 mini | Residual/embedding/logits scaling; fused-QKV / fused-gate-up; LongRoPE |
| **B3: Post-norm / dual-norm** | OLMo 2, Gemma 2 | NormPosition.POST, NormPosition.PRE_AND_POST |
| **B4: Per-layer heterogeneous** | Gemma 3 (SWA/global alternation, dual RoPE bases, softcap) | `layers: list[DecoderBlockSpec]`, dual RoPE per block, logit_softcap |
| **B5: MLA family** | DeepSeek-V2-Lite, DeepSeek-V3-Lite | AttentionKind.MLA, KVCacheSpec.MLA_LATENT, MLA-specific dims, partial RoPE on q_rope only |
| **B6: MoE family** | Mixtral, Qwen3-MoE, OLMoE, DeepSeek-V3-MoE | MoESpec variants — DeepSeek's sigmoid+bias router, shared experts, group routing |
| **B7: SSM-hybrid** | Mamba 2.8B, Jamba-mini, Zamba2, RWKV-7 | TokenMixerKind.SSM, SSMSpec, KVCacheSpec.SSM_STATE, per-layer alternation |
| **B8: Other tokenizer / quant variants** | (no new models) Exercise GGUF Q4_K_M and FP8 W8A8 quant schemes on existing models | QuantSpec block_layout extensions |

The rollout is finished when one model from each batch has (a)-level validation and the API has settled (no extension diffs in the last 2-3 models).

Target model count: **~25**. The list above is ~23; we add 1-2 more if a B-batch surfaces a new axis (e.g., a hybrid coding model with notable specialization).

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| API mutates too much during rollout, breaks earlier models | Each batch ends with a regression run of all earlier models' shape tests; spec dataclass changes are additive (new optional fields) |
| MLA dimensions break the standard `(n_heads, head_dim)` assumption in shared code | MLA is a first-class `AttentionKind` from M1; shared code paths conditional on `kind` |
| MoE token dispatch differs across runtimes; the "expert ffn" abstraction may leak | Reference impl uses dense scatter for clarity; document that production runtimes use grouped GEMM |
| GGUF `1+w` baked-into-weight foot-gun (Gemma 3) | NormSpec `weight_mode` is explicit; layer.md must document which mode the canonical checkpoint uses |
| llama.cpp permutes Q/K weights to switch RoPE basis (silent degradation if missed) | RoPESpec.basis is explicit; `models/<family>/layer.md` documents which basis the canonical checkpoint uses |
| Numerical equivalence test for Qwen3 fails due to subtle initialization or RoPE basis bug | Test in isolation: rms_norm, rope, sdpa, ffn each independently before block-level. If still failing, narrow to the offending sub-op with diff-against-HF |
| KIVI-style asymmetric K/V quantization breaks our symmetric KVCache assumption | KVCacheSpec already has independent `k_quant` and `v_quant`; verified in spec design |
| Phase B grows too slowly to be useful | Subagent-driven development for M2+: dispatch one agent per batch in parallel after kickoff approval |

## 11. Open questions deferred to writing-plans

1. **Base class for layer factories** — should `models/<family>/layer.py` share a `DecoderLayerBase` or stay standalone? Lean: stay standalone for the first few; extract a base only if duplication emerges naturally.
2. **Numerical-equivalence reference choice** — for non-seed models, compare against HF transformers (slow, full Python) or against vLLM (faster, but adds a dep). Lean: HF transformers only, on CPU with small head counts, since this is (b)-tier validation.
3. **Speculative decoding compatibility** — out of scope for this iteration; flag in README but don't design for it now.
4. **Distributed / TP sharding** — out of scope; flag.
5. **Adapter (LoRA) mergeability** — out of scope; flag.
6. **GGUF tensor naming map** — useful for future weight loaders; not built in M1.
7. **ONNX export round-trip** — useful demonstration that the API is portable but adds dependency; not built in M1.

## 12. Evidence index

All design decisions cite one or more of these reports (verbatim sections):

- **research/01-model-census.md** — model selection, axis enumeration, top-5 surprises
- **research/02-layer-sources.md** — universal subset, 20-axis variant list, HF/vLLM/llama.cpp divergences
- **research/03-ihv-opsets.md** — 9-runtime synthesis, the 9-logical-op floor, KV cache representation taxonomy
- **research/04-quantization.md** — 25 schemes, 15-axis QuantSpec, "support only 3" baseline
- **research/05-kvcache-attention.md** — 13 attention × 8 RoPE × 6 cache variants, KVCacheSpec 8 axes, AttentionSpec 19 axes, axis coupling constraints

---

**Ready for review.** The next step after spec approval is the `superpowers:writing-plans` skill to produce a detailed implementation plan with the Qwen3 sample as Milestone 1 (kickoff gate) and the per-model batches as Milestones 2-9.
