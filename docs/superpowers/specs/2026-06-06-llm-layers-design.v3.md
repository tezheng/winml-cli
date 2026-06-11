# llm-layers — Design Spec (v3)

> **Status: SUPERSEDED.** This spec was the pre-rollout design intent for M2 (B0.5→B10). The post-rollout reference is `docs/API-REFERENCE.md` and the cumulative scoreboard is `docs/PROJECT-SUMMARY.md`. See `docs/issues/00-v7-final-audit-synthesis.md` for the 2026-06-10 audit.

**Date:** 2026-06-06
**Status:** Draft v3, awaiting user review
**Owner:** zhengte@microsoft.com
**Supersedes:** `2026-06-04-llm-layers-design.v2.md` (v2 dated 2026-06-04). M1 implementation matches v3 §5.1 op floor. v3 reflects the v3 research outputs landed 2026-06-06.
**Critique applied:** v2's 17 fixes are preserved verbatim; v3 layers on (a) `research/01-model-census.v3.md` (148 rows, 19 axes, +52 families), (b) `research/02-layer-sources.v3.md` (48 family sections, 38 axes), (c) `research/04-quantization.v3.md` (48 schemes, 21 axes), (d) `research/05-kvcache-attention.v3.md` (38 attention / 18 RoPE / 18 cache).

---

## 0. Survey-paper preamble

The decoder-only transformer has continued to evolve along its established
eleven axes — but 2025-H2 and 2026-H1 added five new structural primitives
that v2 did not anticipate. These are folded into the v3 axis catalog.

**Attention** went MHA → MQA (Shazeer 2019, reintroduced for Falcon-7B
2023) → GQA (Ainslie 2023, Llama-2/3) → MLA (DeepSeek-V2 2024) →
Differential (Microsoft 2024) → linear/state-space (Mamba 2023, Mamba-2
SSD 2024, RWKV-7 2025) → trained attention sinks (GPT-OSS Aug 2025,
Mistral Small 3.1 Mar 2025) → **iRoPE / NoPE-per-layer at production scale
(Llama 4 Apr 2025)** → **Lightning Indexer DSA (DeepSeek-V3.2 Sep 2025,
learned per-query top-k scorer)** → **CSA + HCA dual sparse (DeepSeek-V4
Apr 2026)** → **Mamba-3 complex-valued state with MIMO decoding (Mar
2026)** → **MiniMax Sparse Attention (M3.0 Jun 2026)** → **K = V global
unification (Gemma 4 sizes ≥12B, Apr 2026)** → **Gated DeltaNet 3:1
hybrid (Qwen3-Next Aug 2025)** → **MiniMax Lightning Attention 7:1
(MiniMax-Text-01 Jan 2025)** → **Granite 4 9:1 Mamba-2:attention
sequential hybrid (Oct 2025)** → **NVIDIA Nemotron 3 Super/Ultra hybrid
Mamba-2 + Transformer + MoE (Mar–Jun 2026)** → **Visual Causal Flow
block-bidirectional mask (DeepSeek-OCR Oct 2025)**.

**Positional encoding** went absolute → ALiBi (2021) → RoPE (Su 2021) →
PI (Chen 2023) → NTK-aware (bloc97 2023) → YaRN (Peng 2023) → LongRoPE
(Microsoft 2024) → Llama-3 smooth-wavelength (Meta 2024) → NoPE-alternating
(SmolLM3 Jul 2025) → iRoPE per-layer apply-rope flag (Llama 4 Apr 2025)
→ **partial-rotary p-RoPE on global layers only (Gemma 4 Apr 2026,
`partial_rotary_factor=0.25`)** → **M-RoPE / 2D-RoPE / TMRoPE / V2PE
formalized (Qwen2.5-VL, Qwen2.5-Omni, Qwen3-VL, InternVL 3)** → per-layer
dual-θ (Gemma 3/4 — `10_000` local, `1_000_000` global) → **scaling-mode
scaled by model size (Gemma 4 PROPORTIONAL — the same NTK-scaling
recipe scales smaller for E2B than for 31B)**.

**Normalisation** went LayerNorm → RMSNorm (Zhang & Sennrich 2019,
mainstream 2023) → QK-Norm (Henry 2020, reintroduced Olmo-2/Qwen3
2024–2025) → Gemma `1+w` (2024) → sandwich / dual pre+post (Gemma 3,
OLMo 2) → **Gemma 4 QK-norm fixed scales (local 0.9916 / global 1.0228
replacing 1/sqrt(head_dim))** → **four-norm sandwich preserved in Gemma
4** (input, post-attn, pre-ffn, post-ffn).

**Activation** went GeLU → SwiGLU (Shazeer 2020, Llama 2023) → GeGLU
(Gemma 2024) → ReLU² (Primer 2021, BitNet 2024) → GeGLU preserved in
Gemma 4 (`gelu_pytorch_tanh`) with **double-wide MLP on Gemma 4 edge
sizes (E2B/E4B)**.

**Mixture-of-Experts** went vanilla top-k softmax (Switch 2021) →
Mixtral top-2 (2023) → DeepSeek shared+routed + sigmoid +
auxiliary-loss-free balancing (2024) → group-limited routing (DeepSeek-V3
2024) → **ultra-sparse (Qwen3-Next 512 routed + 1 shared, top-10; the
sparsest production MoE so far, Aug 2025)** → **Mixture-of-LoRAs
(Phi-4-multimodal, modality-routed LoRA — vision LoRA + audio LoRA on the
same backbone, Feb 2025)** → **Mixture-of-Transformers (HY-Embodied-0.5
MoT-2B, Apr 2026)** → **first Gemma MoE (Gemma 4 26B-A4B: 128+1 experts,
top-k=8, `moe_intermediate_size=704`, Apr 2026)**.

**KV cache** went contiguous → PagedAttention (2023) → PA v2 + vAttention
(2024) → CLA pairwise sharing (2024) → YOCO two-stage producer/consumer
(Microsoft 2024) → MLA latent compression (DeepSeek 2024) → NSA three-tier
(DeepSeek 2025) → **Apple AFM 2-block cross-block KV sharing (Block-1
62.5% layers full KV, Block-2 37.5% reuses Block-1 K/V, 2025)** →
**Lightning Indexer separate state (DeepSeek-V3.2)** → **Gemma 4
same-block KV sharing (`num_kv_shared_layers=20` of 35 on E2B —
over half of layers reuse a previously-computed K/V tensor, Apr 2026)** →
**Gemma 4 Per-Layer Embeddings paged-to-flash cache (a second
small-dim embedding table that injects residual at every decoder layer,
storage-paged to flash on mobile)** → **Mamba-3 complex-valued state
cache (Mar 2026)** → **CSA + HCA dual-compressed cache (10% of V3.2 KV at
1M, DeepSeek-V4 Apr 2026)**.

**Quantization** went RTN INT8 → GPTQ (2022) → SmoothQuant (2022) → AWQ
(2023) → bitsandbytes NF4 (2023) → HQQ (2024) → FP8 W8A8 (Hopper 2023,
mainstream 2024) → MXFP4/MXINT4 OCP (2024) → NVFP4 Blackwell (2024) →
BitNet b1.58 ternary (2024) → AQLM / EXL2 / QuIP# variable-bit (2024) →
QuaRot / SpinQuant rotation-then-quantize (2024) → **MXFP4-native
training (GPT-OSS 20B, 2025-Q3; the format the model was trained in, not
a PTQ artifact)** → **Falcon-Edge retrainable 1.58-bit BitNet (TII May
2025; the first BitNet family that supports continued pretraining rather
than train-from-scratch)** → **Gemma 4 mobile INT4 QAT (Google Jun 2026;
shipped via LiteRT / MediaPipe / Apple ANE / Hexagon target runtimes)**
→ **NVFP4 training-mode (NVIDIA Blackwell B200, promoted to a
first-class training format)** → **DeepSeek-V3 native FP8 training
(promoted from a v2 note to a full scheme)** → **Apple AFM 2-bpw QAT
with embed-INT4 / KV-INT8 split (promoted to a first-class scheme,
2025)**.

**Hybrid architectures** added a parallel-block axis (Hymba Mamba‖Attn,
NVIDIA late 2024; Jamba interleaved AI21 2024; Zamba2 interleaved-shared
2024; Samba Phi-4-mini-flash 2025) and gained eight distinct topologies
by 2026: Jamba 1:8 sequential periodic, Zamba2 periodic shared, Hymba
parallel branches, Falcon-H1 parallel-head, Phi-4-mini-flash Samba-derived,
Granite 4 9:1 sequential, Qwen3-Next 3:1 Gated DeltaNet, Nemotron 3
Super/Ultra Mamba-2+MoE.

**Vision–language fusion topology** crystallized into six distinct
shapes by 2026-Q2: concat-prefix-LM (LLaVA-1.x), concat-projector
(Qwen2.5-VL), cross-attention adapter (Llama 3.2 Vision), early-fusion
native (Chameleon, Llama 4), mixture-of-LoRAs (Phi-4-multimodal), and
split-encoder for understanding-vs-generation (Janus-Pro). An eighth axis
**vision-token compression budget** parametrizes tokens-per-page at
1024² resolution: from ~6000 (Qwen2-VL native) through ~640 (MiniCPM-V
resampler), ~256–324 (GOT-OCR2 high-compression, mPLUG-DocOwl2), to
~100 (DeepSeek-OCR 16× compressor) — with M-RoPE / TMRoPE / V2PE /
2D-RoPE companions per fusion class.

**OCR-LLM lineage** matured between 2024-Q3 and 2026-Q2 from one model
(Pix2Struct 2023) into a coherent 15+ model family: GOT-OCR 2.0 (Sep
2024, 0.58B), DeepSeek-OCR (Oct 2025, DeepSeek3B-MoE-A570M + DeepEncoder
SAM-B + CLIP-L serial + 16× compressor), DeepSeek-OCR-2 (Q2 2026),
MinerU2.5 (Sep 2025, 1.2B), olmOCR / olmOCR-2 (Feb / Oct 2025, Qwen2.5-VL-7B
base), MonkeyOCR-v1.5 (Nov 2025, 3B), Nanonets-OCR-s / OCR2-3B (May 2025
/ Dec 2025, 3B), Dolphin-v2 (Q4 2025, 3B), HunyuanOCR (Nov 2025, 1B),
PaddleOCR-VL / PaddleOCR-VL-1.5 (Oct 2025 / Jan 2026, 0.9B), RolmOCR
(Jul 2025), Surya OCR 2 (Oct 2025, 0.65B). The architectural primitive
forced by this lineage is **block-bidirectional masking inside each
visual page, causal across pages** — encoded as `MaskKind.VISUAL_CAUSAL_FLOW`
in v3 (DeepSeek-OCR's "Visual Causal Flow").

**Audio LM** crystallized as a thin family: Moshi 7B (Sep 2024,
Helium-1 backbone + Mimi codec + Temporal + Depth transformers), Helium-1
2B (Jan 2025), Voxtral TTS 4B (Mar 2026, 5-sec zero-shot voice cloning,
90ms latency), Qwen2-Audio 7B (Aug 2024), Qwen2.5-Omni 3B/7B (Mar/Apr
2025, Thinker–Talker dual decoder + TMRoPE 4D positional), IBM Granite
Speech 4.1 AR-2B / NAR-2B (Apr 2026 — one of the only non-autoregressive
speech heads at <8B).

**Five v3 framing shifts beyond v2.** v2's framing was "small generative
text-only LLM decoder". v3 broadens to "**small generative decoder LM**,
including (a) text-only, (b) VLM text-tower with vision adapter, (c)
audio-LM with audio frontend, (d) OCR-LLM with specialized vision
frontend, (e) omni-modal with multi-modality input + text output, (f)
quantization-native architectures." The five concrete shifts:

1. **Active-params ≤8B** broadened to "active ≤8B *or* architecturally
   first-of-kind". The Qwen3-Next 80B-A3B (3B active) is a 2026-Q3 row
   despite total ≥8B because it defines the ultra-sparse axis (512+1
   experts, top-10). Llama 4 Scout 17B-A is a row despite 17B active
   because it is the production validation of NoPE-per-layer (iRoPE)
   that downstream <8B models will inherit. The v3 census labels these
   "oos, axis-load-bearing" and tabulates them; the spec follows the
   census.
2. **Modality-broadening.** OCR-LLM, audio-LM, VLM text-towers now
   contribute decoder-block requirements. The downstream effect on the
   spec is bounded: the vision encoder and the audio codec stay outside
   the API; only the *text-tower decoder block* enters the API, and the
   only new axis the text-tower forces is `MaskKind.VISUAL_CAUSAL_FLOW`
   for DeepSeek-OCR and the 2D/3D RoPE field set for the M-RoPE family.
3. **Sparse attention learned-scorers.** DSA (Lightning Indexer) and
   CSA+HCA are not new ops — they compose `linear + softmax + top_k`
   inside the Attention building block — but they force new spec
   fields (`IndexerSpec`, `CSASpec`, `HCASpec`) and one new cache layout
   each.
4. **Cross-layer and cross-block KV sharing.** Three distinct primitives
   in 2025–2026: Apple AFM's 2-block cross-block split (Block-1 with
   62.5% of layers carries full KV, Block-2 with 37.5% reuses Block-1's
   K/V), Gemma 4's same-block cross-layer share (`num_kv_shared_layers=20`
   out of 35 on E2B), and the historical YOCO / CLA primitives. The
   spec subsumes all four under `ShareScheme` + `kv_source_layer`.
5. **Native quantization.** GPT-OSS 20B was *trained* in MXFP4 (not
   PTQ'd); BitNet b1.58 was trained ternary from scratch; Falcon-Edge
   is a BitNet that supports continued pretraining; NVFP4 is a Blackwell
   training format. `QuantSpec.training_native` and `QuantSpec.retrainable`
   are informational fields that document this; the inference loader
   still reads packed weights the same way.

This project takes the v3 census as a corpus and asks: **what is the
smallest API that can re-assemble any production small LM from raw
weights without embedding family-specific behaviour?** A model that
requires no new primitive *fits*; a model that requires a new op or a new
spec field forces a deliberate, evidence-justified extension. The final
API is the floor that survives every model surveyed (the v3 census names
~52 architecturally-distinct families across 148 rows; the v2 rollout
plan's 24-family target is preserved as the rollout milestone, but the
axis catalog is expanded to 19 axes to cover the 2026 wave). The API is
the IR. Quantization and KV cache are first-class parameters, not
bolted-on layers — both are spec dataclasses that decoder blocks consume
on equal footing with attention and FFN.

**v3 surface-area summary (vs v2):**

| Surface | v2 | v3 | Delta |
|---|---|---|---|
| Logical ops | 16 (target) / 9 (M1 partial) | **16 (M1 landed)** | landed |
| First-class spec dataclasses | 12 (target) | **14 (M1 landed)** + 4 new helpers (v3) = **≥18 total** | +6 |
| `AttentionSpec` fields | 18 | **30** | +12 (`attention_k_eq_v`, `global_head_dim`, `lightning_indexer`, `csa_spec`, `hca_spec`, `apply_rope_per_layer`, `block_bidirectional_mask`, `output_gate`, `attn_temperature`, `final_logit_softcap`, `qk_norm_fixed_scale`, `is_complex_state`, `n_mimo_outputs`, `final_logit_softcap` — net 13 listed minus 1 dup) |
| `RoPESpec` fields | 9 | **12** | +3 (`partial_rotary_factor`, `mrope_section` typed, `is_2d`) |
| `KVCacheSpec` fields | 14 | **19** | +5 (`share_scheme`, `num_kv_shared_layers`, `share_block_split`, `kv_source_layer`, `per_layer_embedding`) |
| `QuantSpec` fields | 22 | **25** | +3 (`training_native`, `retrainable`, `target_runtime`) |
| `AttentionKind` enum values | 7 | **12** | +5 (DSA, CSA_HCA, MSA, MAMBA3_COMPLEX, GATED_DELTANET) |
| `MaskKind` enum values | 8 | **11** | +3 (VISUAL_CAUSAL_FLOW, DSA_TOP_K, MSA) |
| `CacheLayout` enum values | 8 | **12** | +4 (LIGHTNING_INDEXER, COMPLEX_STATE, CSA_HCA_DUAL_COMPRESSED, AFM_TWO_BLOCK) |
| `RoPEScaling` enum values | 8 | **9** | +1 (PROPORTIONAL) |
| `QDType` enum values | 13 | **15** | +2 (INT2_QAT, NVFP4) |
| `PackingLayout` enum values | 7 | **10** | +3 (MARLIN_4BIT, MARLIN_24, OCP_MX_BLOCK32) |
| `QuantRole` enum values | 6 | **9** | +3 (MOE_EXPERT_WEIGHT, MOE_ROUTER_WEIGHT, LM_HEAD) |
| `TokenMixerKind` enum values | 10 | **17** | +7 (ATTENTION_MLA_DSA, ATTENTION_CSA_HCA, ATTENTION_MSA, ATTENTION_GATED_DELTANET, ATTENTION_LIGHTNING, SSM_MAMBA3, HYBRID_SEQUENTIAL_N_M) |
| Rollout batches | 9 (B0.5 → B9) | **11** (B0 done; B0.5 → B10) | +2 (B8 OCR-LLM, B9 audio LM) |
| Census model count | 80 | **148** | +68 |
| Census architectural families | ~30 | **~52** | +22 |
| Axis catalog | 15 | **19** | +4 (A16 VLM fusion, A17 vision-token compression, A18 cross-layer KV, A19 per-layer embeddings) |
| Quant schemes surveyed | 42 | **48** | +6 (MXFP4-native, Falcon-Edge retrainable, Gemma 4 mobile-INT4 QAT, NVFP4 training, DeepSeek-V3 native FP8, Apple AFM 2-bpw) |
| "Support N" quant baseline | 5 | **6** | +1 (Gemma 4 mobile-INT4 QAT) |

We do not pretend this is a finished operator standard. ONNX-Runtime,
TRT-LLM, OpenVINO, MLX, llama.cpp, QAIRT, Core ML, MIGraphX, KleidiAI,
LiteRT, MediaPipe and MindIE each ship their own decoder primitives at
different granularities; this API targets the **≥95% consensus floor**
documented in `research/03-ihv-opsets.md` and explicitly names where it
punts to building blocks (MoE dispatch, SSM scan, Lightning Indexer
top-k) rather than ops. The deliverable is the union of: (a) an op floor,
(b) a set of spec dataclasses spanning *nineteen* architecture axes (was
14 in v2), (c) a per-family rollout that turns every spec field into a
tested example. The output is a reusable decoder IR grounded in three
years of decoder evolution.

---

## 1. Goal

Produce a **minimal, evidence-grounded API set** capable of assembling any
mainstream small language model (SLM, <8B active params, plus a curated
set of architecturally-load-bearing 8–80B references) as a transformer
decoder graph — with **quantization** and **KV-cache** as first-class
parameters.

The API has three contracts:

1. **Logical-op floor.** A set of **16 functional primitives** (M1
   shipped 16 — see §5.1 below). Decoder blocks are expressed entirely as
   compositions of these ops. The v3 axis additions did not force new
   ops; every new attention variant (DSA, CSA+HCA, MSA, K=V, p-RoPE,
   Visual Causal Flow, Gated DeltaNet, Mamba-3 complex state) is
   absorbed as either (a) a new `kind` value on an existing op, or (b) a
   new spec field consumed by the existing building block. See §11 for
   the named exception (`indexer_score_topk` for DSA, which is *not* a
   new op but a new sub-routine inside the Attention building block).
2. **Spec dataclasses.** **Fourteen** `@dataclass(frozen=True)` types
   (was 12 in v2; the M1 fix-pass landed the 14 actually shipped: the v2
   §5.2 helpers `SSMSpec`, `SSDSpec`, `ConvSpec`, `GroupRoutingSpec`,
   `MoESpec`, `LayerScaleSpec` are promoted from helper-only to
   first-class), plus **four new helper specs** added in v3
   (`IndexerSpec`, `CSASpec`, `HCASpec`, `PLESpec`). Total **≥18**
   dataclasses. Specs are pure data; behaviour lives in the building
   blocks that consume them.
3. **Per-family proofs.** For each architectural family (~24 in the
   rollout target; the census names ~52) a `models/<family>/` directory
   containing a layer document, a layer factory using only API
   primitives, and tests. Qwen3 is the seed (Milestone 1 — **complete
   2026-06-05, passed at max_abs_diff 4.77e-7 over 74 tests**); the
   remaining 23 ship in batches B0.5 → B10 (§10).

The IR is intended to be *expressible* in PyTorch, GGUF and ONNX-style
runtimes. Design choices follow runtime/IHV consensus where it exists; we
implement only a PyTorch eager reference.

## 2. Non-goals

- **Training or fine-tuning code.** Random weights for shape tests; HF
  weights only for the Qwen3 numerical-equivalence check and for AWQ
  unpack exercises.
- **Full-model inference.** We implement one decoder block per family —
  not tokenisers, samplers, beam search, speculative decoding, sharding,
  pipelines.
- **Performance kernels.** PyTorch eager only.
- **Bit-exact equivalence across all models.** Phase B targets
  structural and shape equivalence for all families; numerical
  equivalence is proven for Qwen3 in the M1 kickoff gate (achieved:
  `max_abs_diff = 4.77e-7`).
- **Vision-language as a separate primitive.** v3 still treats VL as
  out-of-scope for the API surface, but `RoPESpec.mrope_section` and
  `RoPESpec.is_2d` are now first-class fields (v2 reserved
  `mrope_section` but did not type it); the VL fusion topology axis A16
  and the vision-token compression budget axis A17 are documented in
  §0 and §10 but not enforced by the API. A VLM model that wants to
  reuse the decoder block can do so by supplying a 2D-aware RoPE; the
  vision encoder lives outside.
- **Encoder-decoder.** Florence-2 etc. out of scope.
- **Distributed / tensor-parallel sharding.** Runtime concern.
- **LoRA mergeability into quantized weights.** Out of scope for M1;
  `QuantSpec` reserves a `merged_lora_rank` field.
- **Continuous-batching shape `[T,D]`.** The API operates on `[B,S,D]`;
  vLLM/SGLang continuous-batching squashes B×S into T at the runtime
  layer.

## 3. Sub-projects and sequencing

| Phase | Output | Status |
|---|---|---|
| A. Model census + variant-axis catalog | `research/01-model-census.v3.md` — 148 rows, ~52 archs, 19 axes | Done v3 |
| A.1 Cross-source layer survey | `research/02-layer-sources.v3.md` — 48 families, 38 axes, HF/vLLM/llama.cpp triple-source | Done v3 |
| A.2 IHV op-set survey | `research/03-ihv-opsets.v2.md` — 13+ runtimes, op consensus floor | v2 still current; v3 update planned but not required |
| A.3 Quantization survey | `research/04-quantization.v3.md` — 48 schemes, 21-axis parameter space | Done v3 |
| A.4 KV-cache + attention survey | `research/05-kvcache-attention.v3.md` — 38 attn × 18 RoPE × 18 cache | Done v3 |
| A.5 Evolution synthesis | `research/00-evolution.md` — timeline 2022→2026 | v2 still current |
| B. Per-model layer reference impls | `models/<family>/{layer.md, layer.py, weight_loader.py, test_layer.py}` ×~24 | **Qwen3 done (M1, 2026-06-05); B0.5 → B10 pending** |
| C. Minimal-API synthesis | `api/` package (the IR) | **M1 surface landed (2026-06-05): 16 ops, 14 specs, 74 tests passing** |

Sequencing: A is done in v3 form. B and C **co-evolve** in execution —
each new model in B either confirms or extends C. M1 (Qwen3 seed) is
complete; v3 reorders the M2+ rollout (§10) to put architecturally-novel
families up front because they force the new axes (Gemma 4 alone forces
four new spec fields).

## 4. Project layout

```
C:\Users\zhengte\external\llm-layers\
├── api/                            # Phase C — the minimal API (the IR)
│   ├── __init__.py                 # public surface
│   ├── ops.py                      # 16 logical primitives (functional) — M1 LANDED
│   ├── specs.py                    # 14 spec dataclasses (see §5.2) — M1 LANDED
│   ├── types.py                    # enum types (NormKind, RoPEBasis, ShareScheme, ...)
│   ├── attention.py                # Attention block (standard, MLA, linear, differential, DSA, CSA+HCA)
│   ├── ssm.py                      # SSM / SSD / Mamba-3 / conv1d / selective_scan building blocks
│   ├── feedforward.py              # FFN + MoE (parameterized, incl. Mixture-of-LoRAs hook)
│   ├── norm.py                     # RMSNorm + LayerNorm + ScaleNorm + QK-norm + fixed-scale QK-norm
│   ├── rope.py                     # RoPE variants (vanilla / Llama-3 / YaRN / LongRoPE / PI / DynamicNTK / NoPE / iRoPE / partial / M-RoPE / 2D-RoPE / p-RoPE)
│   ├── kvcache.py                  # KVCache type + layouts (contiguous / paged / ring / MLA-latent / SSM-state / YOCO-shared / Gemma4-shared / AFM-block-shared / Lightning-indexer / complex-state)
│   ├── quant.py                    # QuantSpec realization + dtype casts + packed-int helpers (AWQ/GPTQ/HQQ/GGUF/MXFP4/NF4/IQ/FP8/NVFP4/MXFP4-native/BitNet-retrainable)
│   ├── block.py                    # DecoderBlock assembly (token mixer + channel mixer + residual structure)
│   ├── weights.py                  # HF/GGUF → API tensor-slot mapping (see §9)
│   ├── ple.py                      # Per-Layer Embeddings (Gemma 4) — paged-to-flash second embedding table
│   └── README.md                   # API quick reference
├── models/                         # Phase B — per-model layers
│   ├── qwen3/{layer.md, layer.py, weight_loader.py, test_layer.py}  # DONE (M1)
│   ├── gemma4/...                  # B0.5 (PROMOTED — forces 4 new spec fields)
│   ├── llama3/...
│   ├── llama4/...                  # iRoPE per-layer NoPE
│   ├── mistral_v01/...
│   ├── ministral/...               # interleaved SWA
│   ├── smolllm3/...                # NoPE every-4th
│   ├── tinyllama/...
│   ├── granite3/...                # μP four-scalar
│   ├── granite4/...                # 9:1 Mamba-2:attn hybrid
│   ├── granite41/...               # dense 8B (retired hybrid)
│   ├── minicpm3/...                # MLA at small scale
│   ├── phi3/...                    # fused QKV + LongRoPE
│   ├── phi3_small/...              # BlockSparse mask
│   ├── phi4_mini/...
│   ├── olmo2/...                   # post-norm + QK-norm
│   ├── gemma2/...                  # sandwich + softcap
│   ├── gemma3/...                  # SWA/global alternation + dual RoPE
│   ├── deepseek_v3/...             # MLA + MoE + group-routing
│   ├── deepseek_v32/...            # DSA + Lightning Indexer
│   ├── deepseek_v4/...             # CSA + HCA
│   ├── deepseek_ocr/...            # Visual Causal Flow
│   ├── mixtral/...                 # MoE classic
│   ├── qwen3_moe/...
│   ├── qwen3_next/...              # Gated DeltaNet 3:1 + ultra-sparse MoE 512+1/top-10
│   ├── jamba/...                   # SSM-hybrid alternating
│   ├── hymba/...                   # SSM-hybrid parallel
│   ├── mamba2/...                  # SSD formulation
│   ├── mamba3/...                  # complex state + MIMO
│   ├── rwkv7/...                   # WKV
│   ├── nemotron3/...               # hybrid Mamba-2 + Transformer + MoE
│   ├── apple_afm/...               # cross-block KV sharing
│   ├── gpt_oss/...                 # MXFP4-native + trained sinks
│   ├── falcon_edge/...             # retrainable 1.58-bit BitNet
│   ├── got_ocr2/...                # smallest OCR-LLM
│   ├── qwen25_vl/...               # M-RoPE + concat-projector
│   ├── moshi/...                   # Helium-1 + dual-stream audio
│   └── voxtral/...                 # TTS
├── research/                       # Already populated (v3)
│   ├── 00-evolution.md
│   ├── 01-model-census.v3.md
│   ├── 02-layer-sources.v3.md
│   ├── 03-ihv-opsets.v2.md          # (v2 still current)
│   ├── 04-quantization.v3.md
│   ├── 05-kvcache-attention.v3.md
│   └── issues/                     # critique inputs
├── docs/superpowers/specs/
│   ├── 2026-06-04-llm-layers-design.md      # v1 (archived)
│   ├── 2026-06-04-llm-layers-design.v2.md   # v2 (archived)
│   └── 2026-06-06-llm-layers-design.v3.md   ← this file
├── plans/                          # Implementation plan (from writing-plans skill)
├── pyproject.toml                  # uv-managed venv
├── uv.lock
└── README.md
```

Python env: `uv venv && uv sync` (Python 3.11+). Runtime dependencies:
`torch`, `safetensors`, `numpy`. Test dependencies: `pytest`,
`transformers` (numerical-equivalence cross-checks only). Dev
dependencies: `ruff`, `mypy` (strict on `api/`, lenient on `models/`),
`black`. See §16 for the quality bar.

## 5. Phase C — Minimal API design framework

The API has four layers:

1. **Logical ops** (functional primitives) — IHV-consensus floor (**16
   ops, M1 LANDED**)
2. **Specs** (dataclasses) — **14 frozen dataclasses** plus **4 helper
   specs** spanning nineteen axes
3. **Building blocks** (small classes) — Attention, SSM, FFN, MoE, RoPE,
   Norm, KVCache, PLE; consume specs
4. **DecoderBlock** — assembles a token mixer + channel mixer with
   residual structure

### 5.1 Logical ops — the IHV-consensus floor (M1 LANDED)

M1 closed v2's "~16" target into **exactly 16** functional ops in
`api/ops.py`. (v2 §5.1 said "expands to sixteen" but the v1-to-M1
implementation initially shipped only 9 of them; the M1 fix-pass
2026-06-05 closed that gap.) Each op is a pure function. The consensus
column refers to the 13-runtime survey in `research/03-ihv-opsets.v2.md`
(OpenVINO, QAIRT, FastFlowLM, MIGraphX, TensorRT-LLM, Core ML, MLX,
KleidiAI, MindIE, ggml, AWS Neuron, TPU XLA, DirectML).

| # | Op | Signature | IHV consensus | Used by |
|---|---|---|---|---|
| 1 | `rms_norm(x, weight, eps, mode)` | `x:[B,S,D] → [B,S,D]`; `mode ∈ {standard_w, one_plus_w, scale_norm}` | 13/13 | all |
| 2 | `layer_norm(x, weight, bias, eps)` | `x:[B,S,D] → [B,S,D]` | 13/13 | OLMo 1, MPT, Falcon |
| 3 | `linear(x, weight, bias?)` | `x:[B,S,K] × w:[N,K] → [B,S,N]`; quantization handled in `api.quant` via wrapper module | 13/13 | all |
| 4 | `rope_apply(qk, freqs, kind, partial_dim?)` | `qk:[B,S,H,Dh] → [B,S,H,Dh]`; `kind ∈ {standard, partial, m_rope, 2d_rope}` | 9/13 explicit, 4/13 fused into attention | all attention models |
| 5 | `sdpa(q, k, v, mask, scale, softcap?, sink?, sliding_window?, temperature?)` | `[B,H,S,Dh] × [B,Hk,Sk,Dh] × [B,Hk,Sk,Dv] → [B,H,S,Dv]`; fused softmax inside | 13/13 fused | all attention models |
| 6 | `softmax(x, axis, dtype?)` | `[..., N, ...] → [..., N, ...]`; explicit, used for MoE router and standalone softmax | 13/13 | MoE router, Lightning Indexer scorer |
| 7 | `top_k(scores, k, axis, sorted)` | returns `(values, indices)` of top-k entries along `axis` | 11/13 explicit, 2/13 via custom plugin | MoE, NSA, **DSA Lightning Indexer**, CSA |
| 8 | `gather(table, indices, axis)` | `table[..., N, ...] × indices[..., M, ...] → [..., M, ...]` | 13/13 | MoE token dispatch, codebook embedding, PLE table lookup |
| 9 | `scatter(input, indices, updates, axis, reduction)` | inverse of `gather`; `reduction ∈ {none, add}` | 12/13 | MoE token combine |
| 10 | `conv1d(x, weight, bias?, groups, padding, causal)` | `x:[B,S,D] → [B,S,D]`; supports depthwise (`groups=D`) and causal padding | 12/13 (TRT-LLM `mambaConv1dPlugin` named fused) | Mamba 1/2/3, Hymba, RecurrentGemma |
| 11 | `selective_scan(x, A_log, B, C, D, dt, initial_state?, chunk_size?, is_complex?)` | implements Mamba-1 selective scan, Mamba-2 SSD (when `chunk_size` is set), **Mamba-3 complex-state scan (when `is_complex=True`)**; cumulative recurrence | 7/13 explicit, 6/13 via composition | Mamba 1/2/3, Jamba, Zamba2, Hymba |
| 12 | `activation(x, kind)` | `[..., D] → [..., D]`; `kind ∈ {silu, gelu, gelu_tanh, gelu_pytorch_tanh, relu2, gegelu, swish}` | 13/13 | all FFN |
| 13 | `add(x, residual, scale?)` | `[B,S,D]`; optional residual multiplier (Granite μP, LayerScale, DeepNorm) | 13/13 | all residual paths |
| 14 | `mul(a, b)` | element-wise; covers SwiGLU `silu(gate)*up`, attention mask scaling, scalar broadcast | 13/13 | all |
| 15 | `embed(ids, weight, scale?, codebook?)` | `ids:[B,S] → [B,S,D]`; optional scalar (Gemma `sqrt(D)`, Granite `embedding_multiplier`); optional codebook | 13/13 | embeddings, **PLE** |
| 16 | `lm_head(x, weight, scale?, softcap?)` | `[B,S,D] → [B,S,V]`; logits scaling (Granite) and softcap (Gemma 2 + Gemma 4 final-logit-softcap=30); optional tied weight | 13/13 | final layer |

**M1 verification:** all 16 ops shipped, `op-by-op` isolation tests pass
at `atol=1e-5` against HF references for Qwen3-0.6B.

**Sub-primitives folded into the above** (kept callable but not counted
as floor ops because every runtime composes them implicitly):

- `split` and `concat` — host-side reshapes
- `cast` — implicit inside `linear` and `kv read`
- `layer_scale` — folded into `mul` and `add`
- `causal_mask` — mask builder helper; runtimes either build the mask in
  kernel or accept a pre-built tensor
- `visual_causal_mask` — block-bidirectional mask builder for OCR-LLM
  (DeepSeek-OCR); same op as `causal_mask` parameterized by a `block_lengths`
  argument that names which positions are bidirectionally connected
- `indexer_score_topk` — Lightning Indexer top-k routine inside the
  Attention building block (DeepSeek-V3.2 DSA); not a new op floor entry,
  but the building block consumes `IndexerSpec` and composes `linear` +
  `softmax` + `top_k`

**WKV recurrence (RWKV-7)** is expressed via `selective_scan` with a
specific parameterisation; we do *not* add a separate `wkv_update` op.

**Mamba-3 complex-valued state** is expressed by the `is_complex` flag
on `selective_scan`; the building block allocates complex-valued storage
in `KVCacheSpec.layout = COMPLEX_STATE`. No new op.

**DSA / CSA + HCA** are expressed inside the Attention building block as
compositions of `linear` + `softmax` + `top_k` (for the indexer / scorer)
plus the regular `sdpa` (for the actual attention over the top-k
selected positions). No new op.

**K = V global-layer unification (Gemma 4 ≥12B)** is expressed by the
weight loader aliasing W_k and W_v to a single tensor; the spec field
`AttentionSpec.attention_k_eq_v = True` signals this. The forward path is
unchanged (it computes K and V independently, but reads the same
underlying weight tensor); no new op.

Implementation style: **pure functions** in `api/ops.py`. Signatures
take positional tensors and named scalars; specs are passed by value.

### 5.2 Specs — the parameter spaces

Specs are **frozen dataclasses** (`@dataclass(frozen=True)`). They are
the source of truth for "what varies across models." Each spec carries
only data; behaviour lives in the building blocks that consume them.

There are **fourteen first-class specs** (the M1 fix-pass landed these
14, up from v2's 12 by promoting `MoESpec`, `SSMSpec`, `SSDSpec`,
`ConvSpec`, `GroupRoutingSpec`, `LayerScaleSpec` from "helper" to
first-class — and consolidating v2's stand-alone `LayerScaleSpec` into
DecoderBlockSpec attributes when shared, with the dataclass kept for
OpenELM-style per-layer overrides):

```
NormSpec
RoPESpec
Llama3RoPEParams
AttentionSpec
FFNSpec
MoESpec
GroupRoutingSpec
SSMSpec
SSDSpec
ConvSpec
LayerScaleSpec
KVCacheSpec
QuantSpec
DecoderBlockSpec
```

Plus **four new helper specs added in v3**:

```
IndexerSpec      # DeepSeek-V3.2 DSA Lightning Indexer
CSASpec          # DeepSeek-V4 Compressed Sparse Attention
HCASpec          # DeepSeek-V4 Hierarchical Compressed Attention
PLESpec          # Gemma 4 Per-Layer Embeddings
```

**Total ≥18 spec dataclasses.** Enum definitions (NormKind, RoPEBasis,
RoPEScaling, ShareScheme, MaskKind, AttentionKind, QDType, PackingLayout,
QuantRole, BlockLayout, RotationKind, ...) are in `api/types.py` and
documented in §14.

#### 5.2.1 `AttentionSpec` (extended for v3)

```python
@dataclass(frozen=True)
class AttentionSpec:
    # Projection topology
    n_q_heads: int                               # e.g., 16 for Qwen3-0.6B
    n_kv_heads: int                              # GQA: n_q_heads // gqa_groups; MQA: 1; MHA: == n_q_heads
    head_dim: int                                # e.g., 128 for Qwen3, 256 for Gemma 4 local layers
    kind: AttentionKind                          # see §14: STANDARD, MLA, DIFFERENTIAL,
                                                  # LINEAR_RETENTION, LINEAR_DELTANET, LINEAR_GLA,
                                                  # NSA, DSA, CSA_HCA, MSA, MAMBA3_COMPLEX, GATED_DELTANET
    qkv_layout: QKVLayout                        # SPLIT, FUSED (Phi-3), MLA_LATENT

    # Biases (4 independent flags)
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
    attn_scale: Optional[float] = None
    attn_temperature: Optional[float] = None                      # NEW (v3) — Llama 4 attention temperature tuning
    attn_logit_softcap: Optional[float] = None                    # Gemma 2 attn-softcap; Gemma 3/4 set this to None
    final_logit_softcap: Optional[float] = None                   # NEW (v3) — Gemma 4 restored 30.0; was on LayerScaleSpec in v2

    # Mask
    mask_kind: MaskKind                          # CAUSAL, SWA, SWA_GLOBAL_ALT, SINK, FULL,
                                                  # BLOCK_SPARSE, DOCUMENT_CAUSAL, VISUAL_CAUSAL_FLOW (NEW v3),
                                                  # DSA_TOP_K (NEW v3), MSA (NEW v3), CUSTOM
    sliding_window: Optional[int] = None
    n_sink_tokens: Optional[int] = None                            # GPT-OSS trained sinks
    block_sparse_spec: Optional['BlockSparseSpec'] = None          # Phi-3-small
    block_bidirectional_mask: bool = False                         # NEW (v3) — DeepSeek-OCR Visual Causal Flow

    # QK normalization
    qk_norm: Optional[NormSpec] = None
    qk_norm_phase: QKNormPhase = QKNormPhase.NONE                  # NONE, PRE_ROPE (Qwen3), POST_ROPE (Gemma 3)
    qk_norm_fixed_scale: Optional[float] = None                    # NEW (v3) — Gemma 4 fixed-scale QK-norm
                                                                    # (local 0.9916 / global 1.0228 replacing 1/sqrt(head_dim))

    # RoPE
    rope: Optional[RoPESpec] = None                                # None = NoPE layer
    rope_partial_dim: Optional[int] = None                         # GPT-J style partial RoPE on full head_dim
    apply_rope_per_layer: Optional[Sequence[bool]] = None          # NEW (v3) — Llama 4 iRoPE / SmolLM3 NoPE alternation
                                                                    # (when set, overrides rope-None at per-layer level)

    # Cross-layer K/V sharing (CLA / YOCO / Gemma 4 / Apple AFM)
    shares_kv_with: Optional[int] = None                           # layer index this layer reuses; pairwise (CLA / Gemma 4 same-block)
    cache_role: CacheRole = CacheRole.PRODUCER_CONSUMER

    # Differential-transformer extras
    diff_lambda_init: Optional[float] = None

    # Per-layer head-count overrides (OpenELM)
    head_count_override: Optional[Tuple[int, int, int]] = None

    # NEW (v3) — Gemma 4 size-dependent K=V global-layer unification (≥12B)
    attention_k_eq_v: bool = False                                 # global-layer K and V share weight tensor; M1's
                                                                    # weight loader aliases the slot

    # NEW (v3) — Gemma 4 dual-head-dim (global ≠ local)
    global_head_dim: Optional[int] = None                          # If set, global-layer head_dim ≠ head_dim
                                                                    # (Gemma 4 E2B/E4B: global = 512, local = 256)

    # NEW (v3) — DeepSeek-V3.2 DSA Lightning Indexer
    lightning_indexer: Optional["IndexerSpec"] = None              # if set, Attention block routes via indexer top-k

    # NEW (v3) — DeepSeek-V4 CSA + HCA
    csa_spec: Optional["CSASpec"] = None
    hca_spec: Optional["HCASpec"] = None

    # NEW (v3) — Mamba-3 complex-state, MIMO decoding
    is_complex_state: bool = False                                 # complex SSM state — Mamba-3
    n_mimo_outputs: int = 1                                        # MIMO heads — Mamba-3

    # NEW (v3) — Gated DeltaNet output gate (Qwen3-Next, Gated DeltaNet generally)
    output_gate: bool = False
```

Extended `AttentionKind` enum: `STANDARD`, `MLA`, `DIFFERENTIAL`,
`LINEAR_RETENTION`, `LINEAR_DELTANET`, `LINEAR_GLA`, `NSA`, **`DSA`**,
**`CSA_HCA`**, **`MSA`**, **`MAMBA3_COMPLEX`**, **`GATED_DELTANET`**.

Extended `MaskKind` enum: `CAUSAL`, `SWA`, `SWA_GLOBAL_ALT`, `SINK`,
`FULL`, `BLOCK_SPARSE`, `DOCUMENT_CAUSAL`, **`VISUAL_CAUSAL_FLOW`**,
**`DSA_TOP_K`**, **`MSA`**, `CUSTOM`.

**Gemma 4 K=V note.** `attention_k_eq_v` applies only to global layers
on Gemma 4 sizes ≥12B (12B Unified, 26B-A4B, 31B). It is **explicitly
absent on E2B/E4B** (the smaller edge sizes still have independent K
and V projections; the same-block KV sharing is provided instead by
`shares_kv_with` and `num_kv_shared_layers` on the cache). The v3
research file `research/05-kvcache-attention.v3.md §3.31` calls this a
"size-dependent" axis and the v3 census confirms it via direct
`config.json` verification.

#### 5.2.2 `RoPESpec` (extended for v3)

```python
@dataclass(frozen=True)
class RoPESpec:
    base_theta: float                            # 1_000_000 for Qwen3-0.6B / Gemma 4 global,
                                                  # 10_000 for Gemma 4 local, 500_000 for Llama-3-8B
    basis: RoPEBasis                             # INTERLEAVED (HF reference) vs SPLIT_HALF (llama.cpp permute)
    scaling: RoPEScaling                         # NONE, PI, NTK_STATIC, DYNAMIC_NTK, YARN, LLAMA3, LONGROPE,
                                                  # IROPE, PROPORTIONAL (NEW v3 — Gemma 4 size-scaled scaling)
    scale_factor: Optional[float] = None
    yarn_extra: Optional[YarnParams] = None
    llama3_extra: Optional[Llama3RoPEParams] = None
    longrope_extra: Optional[LongRoPEParams] = None
    irope_layer_pattern: Optional[Tuple[bool, ...]] = None  # Llama 4: per-block apply-rope pattern
                                                            # (this is the model-level pattern; per-layer use of it
                                                            # is on AttentionSpec.apply_rope_per_layer)

    # NEW (v3) — Gemma 4 partial-rotary RoPE
    partial_rotary_factor: Optional[float] = None    # 0.25 on Gemma 4 global layers — only 25% of head_dim rotates

    # NEW (v3) — VL M-RoPE / TMRoPE / 2D-RoPE
    mrope_section: Optional[tuple[int, ...]] = None  # Qwen2.5-VL / Qwen3-VL — head_dim section split for T/H/W
    is_2d: bool = False                              # M-RoPE / 2D-RoPE / V2PE flag
```

Extended `RoPEScaling` enum: `NONE`, `PI`, `NTK_STATIC`, `DYNAMIC_NTK`,
`YARN`, `LLAMA3`, `LONGROPE`, `IROPE`, **`PROPORTIONAL`** (Gemma 4 — the
scaling factor is itself scaled by `(d_model / d_model_base)` so the same
NTK-style scaling recipe scales smaller for E2B than for 31B).

**NoPE handling.** A NoPE layer has `AttentionSpec.rope = None`. The
v3 addition `apply_rope_per_layer` on `AttentionSpec` is for *iRoPE-style
per-layer alternation* where the model has one `RoPESpec` shared by all
layers and a *boolean per-layer mask* deciding which layers apply it.

#### 5.2.3 `NormSpec` (unchanged from v2 but with extended `WeightMode`)

```python
@dataclass(frozen=True)
class NormSpec:
    kind: NormKind                               # RMS, LAYER, SCALE_NORM, DEEP_NORM
    eps: float
    weight_mode: WeightMode                      # NONE, STANDARD_W, ONE_PLUS_W, LEARNED_PER_HEAD,
                                                  # FIXED_SCALE (NEW v3 — Gemma 4 QK-norm fixed scales)
    bias: bool = False
    shape: NormShape = NormShape.FULL_HIDDEN     # FULL_HIDDEN, PER_HEAD_DH (Qwen3 QK-norm), FULL_HDH (OLMo 2 QK-norm)
```

The `FIXED_SCALE` weight mode means the norm has a non-learned scalar
that overrides the typical `1/sqrt(head_dim)` factor on QK-norm; the
exact scalar is stored on `AttentionSpec.qk_norm_fixed_scale` (since
it is a function of *which axis* of layer-type the spec belongs to —
0.9916 for Gemma 4 local, 1.0228 for Gemma 4 global).

#### 5.2.4 `FFNSpec` and `MoESpec` (extended for v3)

```python
@dataclass(frozen=True)
class FFNSpec:
    intermediate_size: int
    activation: Activation                       # SILU, GELU, GELU_TANH, GEGELU, RELU2,
                                                  # GELU_PYTORCH_TANH (NEW v3 — Gemma 4)
    gate_kind: GateKind                          # SWIGLU, GEGLU, GELU_ONLY, RELU2_ONLY
    fused_gate_up: bool = False                  # Phi-3
    gate_bias: bool = False
    up_bias: bool = False
    down_bias: bool = False
    double_wide_mlp: bool = False                # NEW (v3) — Gemma 4 edge sizes (E2B/E4B)

@dataclass(frozen=True)
class GroupRoutingSpec:
    n_groups: int
    top_k_groups: int
    group_score_kind: GroupScoreKind             # SUM_TOP_K_IN_GROUP (DeepSeek-V3), MAX, MEAN
    group_score_top_k: Optional[int] = None

@dataclass(frozen=True)
class MoESpec:
    n_routed_experts: int                        # e.g., 160 for DeepSeek-V3, 512 for Qwen3-Next, 128 for Gemma 4 26B-A4B
    top_k: int                                   # e.g., 6 for DeepSeek-V3, 10 for Qwen3-Next, 8 for Gemma 4 MoE
    n_shared_experts: int = 0                    # 1 for DeepSeek-V3 / Qwen3-Next / Gemma 4 MoE; 0 for Mixtral, Qwen3-MoE
    router_kind: RouterKind                      # SOFTMAX (Mixtral), SIGMOID_PLUS_BIAS (DeepSeek-V3 auxiliary-loss-free),
                                                  # LINEAR_TOP_K_NO_NORM
    router_norm: bool = False
    score_correction_bias: bool = False          # DeepSeek-V3 expert-bias
    group_routing: Optional[GroupRoutingSpec] = None
    routed_scaling_factor: float = 1.0
    expert_ffn: FFNSpec
    dispatch_kind: DispatchKind = DispatchKind.DENSE_SCATTER  # DENSE_SCATTER, GROUPED_GEMM, TOKEN_PERMUTE
    moe_intermediate_size: Optional[int] = None  # NEW (v3) — Gemma 4 26B-A4B uses 704 (much smaller than dense FFN)
                                                  # When set, overrides expert_ffn.intermediate_size for the routed pool
    is_ultra_sparse: bool = False                # NEW (v3) — Qwen3-Next ultra-sparse marker (informational)
```

#### 5.2.5 `SSMSpec`, `SSDSpec`, `ConvSpec` (extended for v3)

```python
@dataclass(frozen=True)
class SSMSpec:
    """Mamba-1 selective state-space."""
    d_state: int                                 # 16 (Mamba-1), 128 or 256 (Mamba-2)
    d_conv: int                                  # 4
    d_inner: int                                 # expanded inner dim = expand_factor × d_model
    expand_factor: int = 2
    dt_rank: Union[int, Literal["auto"]] = "auto"
    dt_init: Literal["constant", "random"] = "random"
    dt_scale: float = 1.0
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    activation: Activation = Activation.SILU
    state_dtype: DType = DType.FP32

@dataclass(frozen=True)
class SSDSpec:
    """Mamba-2 SSD formulation."""
    base: SSMSpec
    chunk_size: int = 256                        # block size for chunked parallel scan
    headdim: int = 64                            # SSD heads
    ngroups: int = 1                             # 1 default; 8 for state-grouped variants

@dataclass(frozen=True)
class Mamba3Spec:
    """NEW (v3) — Mamba-3 complex-state + MIMO. Inherits SSD base."""
    base: SSDSpec
    is_complex_state: bool = True                # complex-valued state-update rule
    n_mimo_outputs: int = 1                      # 2 or 4 for the multi-output decode path

@dataclass(frozen=True)
class ConvSpec:
    """1D causal conv used by SSM front-end."""
    kernel_size: int                             # 4 (Mamba), 3 (some hybrids)
    bias: bool
    activation: Optional[Activation] = None
    groups: Optional[int] = None                 # None = depthwise
    causal: bool = True
```

`TokenMixerKind` enum (referenced by `DecoderBlockSpec`):

```python
class TokenMixerKind(Enum):
    ATTENTION_STANDARD = auto()                  # MHA/GQA/MQA
    ATTENTION_MLA = auto()                       # DeepSeek-V2/V3, MiniCPM-3
    ATTENTION_MLA_DSA = auto()                   # NEW (v3) — DeepSeek-V3.2 MLA + Lightning Indexer DSA
    ATTENTION_CSA_HCA = auto()                   # NEW (v3) — DeepSeek-V4
    ATTENTION_MSA = auto()                       # NEW (v3) — MiniMax M3.0
    ATTENTION_LINEAR = auto()                    # generic linear bucket
    ATTENTION_DELTANET = auto()                  # DeltaNet
    ATTENTION_GATED_DELTANET = auto()            # NEW (v3) — Qwen3-Next 3:1 hybrid
    ATTENTION_LIGHTNING = auto()                 # NEW (v3) — MiniMax-Text-01 Lightning 7:1
    ATTENTION_NSA = auto()                       # DeepSeek 2025
    SSM_MAMBA1 = auto()
    SSM_MAMBA2 = auto()                          # SSD
    SSM_MAMBA3 = auto()                          # NEW (v3) — complex state + MIMO
    SSM_GRIFFIN = auto()
    SSM_RWKV = auto()                            # WKV recurrence
    HYBRID_PARALLEL = auto()                     # Hymba — Mamba ‖ Attn in same block
    HYBRID_ALTERNATING = auto()                  # per-layer dispatch (Jamba, Zamba2)
    HYBRID_SEQUENTIAL_N_M = auto()               # NEW (v3) — Granite 4 9:1, Qwen3-Next 3:1, MiniMax 7:1
```

#### 5.2.6 `LayerScaleSpec`

```python
@dataclass(frozen=True)
class LayerScaleSpec:
    """Per-layer scalar multipliers (Granite μP, DeepNorm, LayerScale, OpenELM)."""
    residual_scale: Optional[float] = None
    embedding_scale: Optional[float] = None
    logits_scale: Optional[float] = None
    final_logit_softcap: Optional[float] = None  # (v3 note: this is the same value as
                                                  # AttentionSpec.final_logit_softcap when set on
                                                  # Gemma 4; the loader writes both)
    attn_residual_scale: Optional[float] = None
    ffn_residual_scale: Optional[float] = None

    # NEW (v3) — OpenELM-style per-layer overrides (kept as a typed tuple)
    num_q_heads_per_layer: Optional[Tuple[int, ...]] = None
    num_kv_heads_per_layer: Optional[Tuple[int, ...]] = None
    ffn_multipliers_per_layer: Optional[Tuple[float, ...]] = None
```

#### 5.2.7 `KVCacheSpec` (extended for v3)

```python
@dataclass(frozen=True)
class KVCacheSpec:
    layout: CacheLayout                          # CONTIGUOUS, PAGED, RING, MLA_LATENT_PLUS_KROPE,
                                                  # SSM_STATE, SSD_STATE_PLUS_CONV, NSA_THREE_TIER,
                                                  # YOCO_SHARED,
                                                  # LIGHTNING_INDEXER (NEW v3 — DeepSeek-V3.2 separate indexer state),
                                                  # COMPLEX_STATE (NEW v3 — Mamba-3 complex-valued state),
                                                  # CSA_HCA_DUAL_COMPRESSED (NEW v3 — DeepSeek-V4),
                                                  # AFM_TWO_BLOCK (NEW v3 — Apple AFM cross-block)
    memory_layout: MemoryLayout                  # HND, NHD, BHND, BNHD (vLLM), MLX_LIST
    block_size: Optional[int] = None             # for PAGED, RING, vAttention; e.g., 16 (vLLM default)
    k_dtype: DType
    v_dtype: DType
    k_quant: Optional[QuantSpec] = None
    v_quant: Optional[QuantSpec] = None
    dequant_path: DequantPath = DequantPath.IN_SDPA
    ownership: CacheOwnership                    # EXPLICIT_PASS, STATEFUL
    share_group: Optional[int] = None            # CLA pairwise sharing
    vm_mapped: bool = False                      # vAttention
    tier_count: Optional[int] = None             # NSA: 3
    cache_role: CacheRole = CacheRole.PRODUCER_CONSUMER

    # NEW (v3) — cross-layer KV sharing scheme
    share_scheme: ShareScheme = ShareScheme.NONE                  # NONE / CLA / YOCO / GEMMA4 / APPLE_AFM
    num_kv_shared_layers: Optional[int] = None                    # Gemma 4 E2B: 20 of 35
    share_block_split: Optional[float] = None                     # Apple AFM block split (0.625 / 0.375)
    kv_source_layer: Optional[Sequence[Optional[int]]] = None     # per-layer source layer index (length = n_layers)

    # NEW (v3) — Gemma 4 Per-Layer Embeddings cache
    per_layer_embedding: Optional["PLESpec"] = None               # if set, layer reads its PLE residual from this table

    # For MLA: shape is [c_kv, k_rope] per token (TWO tensors)
    # For SSM: shape is [d_state, d_inner] per token PLUS [d_conv, d_inner] for conv1d state
    # For Mamba-3: same as SSD but complex storage in COMPLEX_STATE
    # For Gemma 4 with share_scheme=GEMMA4: only ~half of layers actually allocate K/V tensors;
    #   the rest read from the source layer specified by kv_source_layer[i]
    # For Apple AFM (share_scheme=APPLE_AFM): Block-1 layers allocate; Block-2 layers read from Block-1
    # For DSA (layout=LIGHTNING_INDEXER): the indexer state is a separate per-layer tensor sized by IndexerSpec
```

Add `ShareScheme` enum: `NONE`, `CLA`, `YOCO`, `GEMMA4` (same-block
cross-layer sharing), `APPLE_AFM` (cross-block sharing).

Extended `CacheLayout` enum: existing values + `LIGHTNING_INDEXER`,
`COMPLEX_STATE`, `CSA_HCA_DUAL_COMPRESSED`, `AFM_TWO_BLOCK`.

#### 5.2.8 `QuantSpec` (extended for v3)

```python
@dataclass(frozen=True)
class QuantSpec:
    # Minimal core
    qdtype: QDType                               # INT2, INT3, INT4, INT5, INT6, INT8, FP8_E4M3, FP8_E5M2,
                                                  # FP4, NF4, MX_FP4, MX_INT4, TERNARY (BitNet),
                                                  # INT2_QAT (NEW v3 — Apple AFM, Gemma 4 mobile QAT),
                                                  # NVFP4 (NEW v3 — Blackwell training-mode)
    group_size: Optional[int]                    # None=per-tensor, -1=per-channel, N=blockwise
    quant_axis: int
    scale_dtype: DType                           # FP32, FP16, BF16, UE8M0 (MXFP4), FP8
    has_zero_point: bool
    packing: PackingLayout                       # NONE, NIBBLE_LSB, NIBBLE_MSB, AWQ_INTERLEAVE,
                                                  # GPTQ_INT32_PACK, HQQ_NIBBLE, GGUF_K_SUPERBLOCK, MX_BLOCK,
                                                  # MARLIN_4BIT (NEW v3 — vLLM Marlin kernel),
                                                  # MARLIN_24 (NEW v3 — Marlin 2:4 sparse),
                                                  # OCP_MX_BLOCK32 (NEW v3 — OCP MXFP4 32-element block)
    accumulator_dtype: DType

    # Extended
    scale_quant: Optional['QuantSpec'] = None
    codebook: Optional[CodebookSpec] = None
    scale_axis: Optional[int] = None
    zero_dtype: Optional[DType] = None
    compute_dtype: Optional[DType] = None
    block_layout: Optional[BlockLayout] = None
    accumulator_dtype: DType = DType.FP32

    # Pre-transforms
    rotation_kind: RotationKind = RotationKind.NONE
    runtime_apply: RuntimeApply = RuntimeApply.NONE
    smoothquant_alpha: Optional[float] = None

    # Codebook combine
    num_codebooks: int = 1
    combine_op: CombineOp = CombineOp.NONE

    # Variable-bitwidth (EXL2)
    variable_bitwidth: bool = False

    # Outlier offloading (LLM.int8)
    outlier_offload: OutlierOffload = OutlierOffload.NONE

    # imatrix calibration (llama.cpp)
    imatrix_calibrated: bool = False

    # Role and dynamic scale source
    role: QuantRole = QuantRole.WEIGHT           # WEIGHT, ACTIVATION, KV_K, KV_V, ATTN_INTERNAL, EMBEDDING,
                                                  # MOE_EXPERT_WEIGHT (NEW v3),
                                                  # MOE_ROUTER_WEIGHT (NEW v3),
                                                  # LM_HEAD (NEW v3 — explicit role for embed-INT4 / KV-INT8 split,
                                                  #          per Apple AFM)
    scale_source: ScaleSource = ScaleSource.STATIC

    # NEW (v3) — distinguishes natively-trained quantization from PTQ
    training_native: bool = False                # GPT-OSS MXFP4-native, BitNet b1.58 trained, Falcon-Edge,
                                                  # NVFP4 training-mode, DeepSeek-V3 native FP8
    # NEW (v3) — Falcon-Edge retrainable distinction
    retrainable: bool = False                    # Falcon-Edge supports continued pretraining (BitNet b1.58 does not)
    # NEW (v3) — target on-device runtime
    target_runtime: Optional[TargetRuntime] = None  # litert / mediapipe / ane (Apple Neural Engine) / hexagon / cpu / gpu

    # Reserved
    merged_lora_rank: Optional[int] = None       # reserved for future LoRA-merged-quant work
```

Extended `QDType` enum: existing values + **`INT2_QAT`** (Apple AFM,
Gemma 4 mobile QAT), **`NVFP4`** (Blackwell training-mode).

Extended `PackingLayout` enum: existing values + **`MARLIN_4BIT`**,
**`MARLIN_24`**, **`OCP_MX_BLOCK32`**.

Extended `QuantRole` enum: `WEIGHT`, `ACTIVATION`, `KV_K`, `KV_V`,
`ATTN_INTERNAL`, `EMBEDDING`, **`MOE_EXPERT_WEIGHT`**,
**`MOE_ROUTER_WEIGHT`**, **`LM_HEAD`**.

**"Support six" baseline** (was "support five" in v2; v3 adds the
mobile-INT4 QAT recipe):

1. **GGUF Q4_K_M** — dominant on-device (llama.cpp, Ollama, LM Studio).
2. **AWQ INT4 W4A16 grouped (g=128)** — dominant server-side 4-bit
   (vLLM, TRT-LLM, SGLang).
3. **FP8 E4M3 W8A8** — dominant compute-quantized (H100, MI300, Gaudi 3).
4. **+ IQ2_M** (or AQLM) — exercises the `codebook` axis.
5. **+ MXFP4 (OCP)** — exercises the `UE8M0` scale dtype and
   `block_layout=OCP_MX`.
6. **+ Gemma 4 mobile-INT4 QAT** (NEW in v3) — exercises
   `qdtype=INT2_QAT` / `INT4` with `training_native=True` and
   `target_runtime ∈ {litert, mediapipe, ane, hexagon}`. This is the
   first baseline that exercises an *axis that the model knows it has*
   (it was QAT-trained for this format and ships with format-specific
   layout).

These six baselines together exercise every QuantSpec axis: storage
layouts (AWQ_INTERLEAVE, GGUF_K, MX_BLOCK, OCP_MX_BLOCK32, MARLIN_4BIT),
packing (NIBBLE, AWQ_INTERLEAVE, INT32_PACK, MX, Marlin), recursive
scales (GGUF), codebook (IQ2_M), scale dtypes (FP16, FP32, UE8M0, FP8),
dynamic activation scales (FP8 W8A8), rotation (deferred), training-mode
(MXFP4-native, BitNet, NVFP4-training, Gemma 4 QAT), and retrainable
(Falcon-Edge). The full v3 quant survey enumerates 48 schemes across 21
axes; the six baselines anchor the API to the most-deployed corners.

#### 5.2.9 v3 helper specs

```python
@dataclass(frozen=True)
class IndexerSpec:
    """DeepSeek-V3.2 DSA Lightning Indexer.
    Per-query token-relevance scorer driving top-k cache selection.
    The Attention building block reads positions via `top_k(indexer_score(...))`."""
    indexer_dim: int                             # e.g., 256
    top_k: int                                   # e.g., 256 — number of cache positions selected per query
    warmup_tokens: int                           # number of tokens until indexer is trusted (training-time)

@dataclass(frozen=True)
class CSASpec:
    """DeepSeek-V4 Compressed Sparse Attention.
    Compresses the global token context."""
    compression_ratio: int                       # e.g., 8, 16
    block_size: int                              # e.g., 512

@dataclass(frozen=True)
class HCASpec:
    """DeepSeek-V4 Hierarchical Compressed Attention.
    Compresses historical KV blocks with hierarchical pyramid."""
    n_hierarchy_levels: int                      # e.g., 3
    branching_factor: int                        # e.g., 4

@dataclass(frozen=True)
class PLESpec:
    """Gemma 4 Per-Layer Embeddings.
    A second small-dim embedding table whose output is injected as a
    residual at every decoder layer. Computed as
    `(token_identity + context_aware_projection) × 1/√2` from a small
    (256-dim) PLE table that can be paged to flash storage on mobile."""
    embedding_dim: int                           # e.g., 256 for Gemma 4 E2B (vs main residual 1536)
    n_layers: int                                # number of layers using PLE (≤ total layers)
    paged_to_storage: bool = True                # pageable to flash
    fanout_per_layer: bool = True                # per-layer projection of the PLE table to main residual dim
```

### 5.3 Building blocks

```
Attention(spec: AttentionSpec, kv_cache: KVCache, layer_scale: Optional[LayerScaleSpec])
  # composes linear, split (for MLA), rope, sdpa, kv read/write, qk_norm
  # NEW v3: dispatches on spec.lightning_indexer / csa_spec / hca_spec / is_complex_state /
  #         attention_k_eq_v / global_head_dim / block_bidirectional_mask
  # NEW v3: when apply_rope_per_layer is set, the block consults its layer_idx to decide
  #         whether to skip RoPE on this layer (iRoPE / SmolLM3 NoPE)
SSM(spec: SSMSpec | SSDSpec | Mamba3Spec, conv_spec: ConvSpec, ssm_cache: KVCache)
  # composes linear, conv1d, selective_scan
FeedForward(spec: FFNSpec)
  # composes linear(s) + activation + mul
  # NEW v3: dispatches on spec.double_wide_mlp (Gemma 4 edge sizes)
MoE(spec: MoESpec)
  # composes linear (gate), softmax|sigmoid_plus_bias (router), top_k, gather (dispatch),
  # per-expert FeedForward, scatter (combine), residual add of shared experts
  # NEW v3: spec.moe_intermediate_size overrides per-expert FFN dim (Gemma 4 26B-A4B = 704)
RMSNorm(spec: NormSpec, hidden_size)
  # NEW v3: supports weight_mode=FIXED_SCALE (Gemma 4 QK-norm)
LayerNorm(spec: NormSpec, hidden_size)
RoPE(spec: RoPESpec, max_seq_len)
  # NEW v3: supports partial_rotary_factor (Gemma 4 p-RoPE) and is_2d / mrope_section (VL)
KVCache(spec: KVCacheSpec, layer_idx, max_seq, n_heads, head_dim)
  # NEW v3: when share_scheme != NONE, the cache may share storage with another layer per kv_source_layer
PLE(spec: PLESpec, vocab_size, target_dim)
  # NEW v3 — Gemma 4 Per-Layer Embeddings
  # Second embedding table; per-layer projects from embedding_dim to target_dim and adds to residual stream
QuantizedWeight(spec: QuantSpec, shape, ...)
  # owns packed storage + scales + zeros + codebook
```

The MoE building block hides the routing primitives (`softmax`, `top_k`,
`gather`, `scatter`) as composition.

The Attention building block, when `spec.lightning_indexer` is set,
adds a sub-step: compute indexer scores via a small linear + softmax,
select top-k positions, and run `sdpa` over just those positions. This
is *inside* the building block, not a new op.

The hybrid-parallel building block (`TokenMixerKind.HYBRID_PARALLEL`,
Hymba) consumes a *pair* of token-mixer specs and sums their outputs
before the residual add. The hybrid-sequential-N:M building block
(`HYBRID_SEQUENTIAL_N_M`, Granite 4 / Qwen3-Next / MiniMax / Nemotron 3)
is per-layer dispatch encoded as a `list[DecoderBlockSpec]` with each
layer carrying its own token-mixer kind.

### 5.4 DecoderBlock assembly

```python
@dataclass(frozen=True)
class DecoderBlockSpec:
    # Residual structure
    attn_norm_position: NormPosition             # PRE, POST, PRE_AND_POST, PARALLEL_FFN
    ffn_norm_position: NormPosition

    # Token mixer
    token_mixer_kind: TokenMixerKind
    token_mixer: Union[AttentionSpec, SSMSpec, SSDSpec, Mamba3Spec,
                       Tuple[AttentionSpec, SSMSpec]]  # tuple for HYBRID_PARALLEL

    # Channel mixer
    channel_mixer: Union[FFNSpec, MoESpec]

    # Norms (each Optional)
    pre_attn_norm: Optional[NormSpec] = None
    post_attn_norm: Optional[NormSpec] = None
    pre_ffn_norm: Optional[NormSpec] = None
    post_ffn_norm: Optional[NormSpec] = None

    # Layer-scale parameters (folded into block attrs in M1 fix-pass)
    residual_scale: Optional[float] = None
    embedding_scale: Optional[float] = None
    logits_scale: Optional[float] = None
    layer_scale: Optional[LayerScaleSpec] = None  # for OpenELM-style per-layer overrides

    # NEW (v3) — Gemma 4 PLE injection
    ple_injection: Optional[PLESpec] = None       # if set, this layer reads + adds a PLE residual

    # Cache (typically a list at model level)
    kv_cache: Optional[KVCacheSpec] = None
```

This handles:

- **OLMo 2 post-norm** → `attn_norm_position=POST`
- **Gemma 3 / Gemma 4 four-norm sandwich** → both
  `attn_norm_position=PRE_AND_POST` and `ffn_norm_position=PRE_AND_POST`,
  all four sub-norms present
- **Granite μP** → `residual_scale=0.22, embedding_scale=12, logits_scale=8`
- **Per-layer heterogeneity** (Gemma 3 SWA alternation, Gemma 4 SWA 4:1
  E2B / 5:1 larger, Jamba SSM/attn mix, SmolLM3 NoPE every 4th, Llama-4
  iRoPE, Granite 4 9:1, Qwen3-Next 3:1, MiniMax 7:1) → model's
  `ModelConfig` carries a `list[DecoderBlockSpec]` of length `num_layers`
- **Hymba parallel** → `token_mixer_kind=HYBRID_PARALLEL`,
  `token_mixer=(AttentionSpec(...), SSMSpec(...))`
- **OpenELM per-layer heads** → `AttentionSpec.head_count_override` or
  the `LayerScaleSpec.num_q_heads_per_layer` tuple
- **DeepSeek-V3 dense/MoE interleaving** → each layer's `channel_mixer`
  is either `FFNSpec` (dense, first 3) or `MoESpec` (rest)
- **Gemma 4 PLE injection** → each layer's `ple_injection` is set; PLE
  building block reads from the cache and adds to residual
- **DeepSeek-V3.2 DSA** → `AttentionSpec.lightning_indexer` set
- **DeepSeek-V4 CSA + HCA** → both `csa_spec` and `hca_spec` set
- **Apple AFM 2-block** → `KVCacheSpec.share_scheme=APPLE_AFM`,
  `share_block_split=0.625`, per-layer `kv_source_layer` indices set
- **Gemma 4 same-block cross-layer KV sharing** →
  `KVCacheSpec.share_scheme=GEMMA4`, `num_kv_shared_layers=20`, per-layer
  `kv_source_layer`

### 5.5 Worked spec examples — v3 anchor models

To make the abstraction concrete, this section spells out the
`DecoderBlockSpec` instantiations for five anchor models from v3
batches: Qwen3-0.6B (B0, the done seed), Gemma 4 E2B (B0.5, the promoted
first novel-axis model), DeepSeek-V3.2 (B5 DSA), Granite 4 H-Micro (B7
9:1 hybrid), and GPT-OSS 20B (B10 MXFP4-native + trained sinks).

#### 5.5.1 Qwen3-0.6B (B0, done)

```python
spec_qwen3_0p6b = DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer_kind=TokenMixerKind.ATTENTION_STANDARD,
    token_mixer=AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        kind=AttentionKind.STANDARD,
        qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        qk_norm=NormSpec(kind=NormKind.RMS, eps=1e-6, weight_mode=WeightMode.STANDARD_W,
                         shape=NormShape.PER_HEAD_DH),
        qk_norm_phase=QKNormPhase.PRE_ROPE,
        rope=RoPESpec(
            base_theta=1_000_000.0,
            basis=RoPEBasis.SPLIT_HALF,
            scaling=RoPEScaling.LLAMA3,
            llama3_extra=Llama3RoPEParams(
                factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
                original_context_length=8192),
        ),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=3072,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    pre_attn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6, weight_mode=WeightMode.STANDARD_W),
    pre_ffn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6, weight_mode=WeightMode.STANDARD_W),
    kv_cache=KVCacheSpec(
        layout=CacheLayout.CONTIGUOUS,
        memory_layout=MemoryLayout.HND,
        k_dtype=DType.BF16, v_dtype=DType.BF16,
        ownership=CacheOwnership.EXPLICIT_PASS,
    ),
)
```

#### 5.5.2 Gemma 4 E2B (B0.5, promoted novel-axis exemplar)

```python
# Gemma 4 E2B has 35 layers; 4:1 pattern means every 5th layer is global,
# the last layer is global, and 20 of 35 layers reuse a previous layer's K/V.

local_rope = RoPESpec(
    base_theta=10_000.0,
    basis=RoPEBasis.INTERLEAVED,
    scaling=RoPEScaling.PROPORTIONAL,  # NEW v3 — size-scaled NTK
    scale_factor=2.0,  # scaled smaller for E2B than 31B per PROPORTIONAL
)

global_rope = RoPESpec(
    base_theta=1_000_000.0,
    basis=RoPEBasis.INTERLEAVED,
    scaling=RoPEScaling.PROPORTIONAL,
    scale_factor=2.0,
    partial_rotary_factor=0.25,  # NEW v3 — only 25% of head_dim rotates
)

ple_spec = PLESpec(
    embedding_dim=256,           # smaller than main residual 1536
    n_layers=35,
    paged_to_storage=True,       # paged to flash on mobile
    fanout_per_layer=True,
)

local_attn = AttentionSpec(
    n_q_heads=8, n_kv_heads=1, head_dim=256,
    kind=AttentionKind.STANDARD,
    qkv_layout=QKVLayout.SPLIT,
    mask_kind=MaskKind.SWA,
    sliding_window=512,
    qk_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                     weight_mode=WeightMode.FIXED_SCALE,  # NEW v3
                     shape=NormShape.PER_HEAD_DH),
    qk_norm_phase=QKNormPhase.PRE_ROPE,
    qk_norm_fixed_scale=0.9916,                            # NEW v3 — Gemma 4 local
    final_logit_softcap=30.0,                              # NEW v3 — restored from Gemma 2
    rope=local_rope,
)

global_attn = AttentionSpec(
    n_q_heads=8, n_kv_heads=1, head_dim=256,
    global_head_dim=512,                                   # NEW v3 — global ≠ local
    kind=AttentionKind.STANDARD,
    qkv_layout=QKVLayout.SPLIT,
    mask_kind=MaskKind.CAUSAL,
    qk_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                     weight_mode=WeightMode.FIXED_SCALE,
                     shape=NormShape.PER_HEAD_DH),
    qk_norm_phase=QKNormPhase.PRE_ROPE,
    qk_norm_fixed_scale=1.0228,                            # NEW v3 — Gemma 4 global
    final_logit_softcap=30.0,
    attention_k_eq_v=False,                                # E2B does NOT have K=V (size-dependent)
    rope=global_rope,
)

# kv_source_layer is [None, None, None, None, 0, None, None, None, None, 4, ...]
# meaning layers 5, 10, 15, ... reuse the K/V from layers 0, 4, 9, ...
# Exact pattern per `num_kv_shared_layers=20` allocation.
kv_layout = KVCacheSpec(
    layout=CacheLayout.CONTIGUOUS,
    memory_layout=MemoryLayout.HND,
    k_dtype=DType.BF16, v_dtype=DType.BF16,
    ownership=CacheOwnership.EXPLICIT_PASS,
    share_scheme=ShareScheme.GEMMA4,                       # NEW v3
    num_kv_shared_layers=20,                               # NEW v3
    per_layer_embedding=ple_spec,                          # NEW v3
)

# Each layer is a separate DecoderBlockSpec; the model's ModelConfig
# carries a list[DecoderBlockSpec] of length 35. Layers 4, 9, 14, 19, 24,
# 29, 34 use global_attn; the rest use local_attn. All layers carry
# ple_injection=ple_spec.

def block_at(layer_idx: int) -> DecoderBlockSpec:
    is_global = (layer_idx + 1) % 5 == 0
    return DecoderBlockSpec(
        attn_norm_position=NormPosition.PRE_AND_POST,       # four-norm sandwich
        ffn_norm_position=NormPosition.PRE_AND_POST,
        token_mixer_kind=TokenMixerKind.ATTENTION_STANDARD,
        token_mixer=global_attn if is_global else local_attn,
        channel_mixer=FFNSpec(
            intermediate_size=8192,
            activation=Activation.GELU_PYTORCH_TANH,         # NEW v3
            gate_kind=GateKind.GEGLU,
            double_wide_mlp=True,                            # NEW v3 — E2B edge size
        ),
        pre_attn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                               weight_mode=WeightMode.ONE_PLUS_W),
        post_attn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                                weight_mode=WeightMode.ONE_PLUS_W),
        pre_ffn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                              weight_mode=WeightMode.ONE_PLUS_W),
        post_ffn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                               weight_mode=WeightMode.ONE_PLUS_W),
        ple_injection=ple_spec,                              # NEW v3
        kv_cache=kv_layout,
    )
```

#### 5.5.3 DeepSeek-V3.2 DSA (B5)

```python
indexer = IndexerSpec(indexer_dim=256, top_k=256, warmup_tokens=2000)

dsa_attn = AttentionSpec(
    n_q_heads=128, n_kv_heads=128, head_dim=128,
    kind=AttentionKind.DSA,                                # NEW v3
    qkv_layout=QKVLayout.MLA_LATENT,
    q_lora_rank=1536, kv_lora_rank=512,
    qk_nope_head_dim=128, qk_rope_head_dim=64,
    v_head_dim=128,
    mask_kind=MaskKind.DSA_TOP_K,                          # NEW v3
    lightning_indexer=indexer,                             # NEW v3
    rope=RoPESpec(
        base_theta=10_000.0,
        basis=RoPEBasis.INTERLEAVED,
        scaling=RoPEScaling.YARN,
        yarn_extra=YarnParams(original_max_pos=4096, attn_factor=1.0,
                              beta_fast=32, beta_slow=1),
    ),
)

kv_layout = KVCacheSpec(
    layout=CacheLayout.LIGHTNING_INDEXER,                  # NEW v3
    memory_layout=MemoryLayout.HND,
    k_dtype=DType.BF16, v_dtype=DType.BF16,
    ownership=CacheOwnership.EXPLICIT_PASS,
)
```

#### 5.5.4 Granite 4 H-Micro (B7, 9:1 hybrid)

A 24-layer model. The first 9 layers in each "decade" are Mamba-2; the
10th is GQA. The model assembler emits a `list[DecoderBlockSpec]` of
length 24, alternating per the pattern:

```python
attn_spec = AttentionSpec(
    n_q_heads=24, n_kv_heads=8, head_dim=64,
    kind=AttentionKind.STANDARD,
    qkv_layout=QKVLayout.SPLIT,
    mask_kind=MaskKind.CAUSAL,
    rope=RoPESpec(base_theta=1e7, basis=RoPEBasis.INTERLEAVED,
                  scaling=RoPEScaling.NONE),
)

ssm_spec = SSDSpec(
    base=SSMSpec(
        d_state=128, d_conv=4, d_inner=3072,
        expand_factor=2, dt_rank="auto",
        dt_init="random", dt_scale=1.0,
        dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
        conv_bias=True, bias=False, use_fast_path=True,
    ),
    chunk_size=256, headdim=64, ngroups=1,
)

def block_at(layer_idx: int) -> DecoderBlockSpec:
    is_attn = (layer_idx % 10) == 9  # every 10th layer
    if is_attn:
        return DecoderBlockSpec(
            attn_norm_position=NormPosition.PRE,
            ffn_norm_position=NormPosition.PRE,
            token_mixer_kind=TokenMixerKind.ATTENTION_STANDARD,
            token_mixer=attn_spec,
            channel_mixer=FFNSpec(
                intermediate_size=8192,
                activation=Activation.SILU,
                gate_kind=GateKind.SWIGLU,
            ),
            pre_attn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                                   weight_mode=WeightMode.STANDARD_W),
            pre_ffn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                                  weight_mode=WeightMode.STANDARD_W),
            residual_scale=0.22,                            # Granite μP
            embedding_scale=12.0,
            logits_scale=8.0,
            kv_cache=KVCacheSpec(
                layout=CacheLayout.CONTIGUOUS,
                memory_layout=MemoryLayout.HND,
                k_dtype=DType.BF16, v_dtype=DType.BF16,
                ownership=CacheOwnership.EXPLICIT_PASS,
            ),
        )
    else:
        return DecoderBlockSpec(
            attn_norm_position=NormPosition.PRE,
            ffn_norm_position=NormPosition.PRE,
            token_mixer_kind=TokenMixerKind.SSM_MAMBA2,
            token_mixer=ssm_spec,
            channel_mixer=FFNSpec(
                intermediate_size=8192,
                activation=Activation.SILU,
                gate_kind=GateKind.SWIGLU,
            ),
            pre_attn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                                   weight_mode=WeightMode.STANDARD_W),
            pre_ffn_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                                  weight_mode=WeightMode.STANDARD_W),
            residual_scale=0.22,
            embedding_scale=12.0,
            logits_scale=8.0,
            kv_cache=KVCacheSpec(
                layout=CacheLayout.SSD_STATE_PLUS_CONV,
                memory_layout=MemoryLayout.HND,
                k_dtype=DType.BF16, v_dtype=DType.BF16,
                ownership=CacheOwnership.EXPLICIT_PASS,
            ),
        )
```

#### 5.5.5 GPT-OSS 20B MXFP4-native (B10)

```python
# Trained-native MXFP4 weights; trained attention sinks.
gpt_oss_attn = AttentionSpec(
    n_q_heads=16, n_kv_heads=2, head_dim=128,
    kind=AttentionKind.STANDARD,
    qkv_layout=QKVLayout.SPLIT,
    mask_kind=MaskKind.SINK,
    n_sink_tokens=4,                                       # trained sinks
    rope=RoPESpec(
        base_theta=1_000_000.0,
        basis=RoPEBasis.INTERLEAVED,
        scaling=RoPEScaling.LONGROPE,
        longrope_extra=LongRoPEParams(
            short_factor=[...], long_factor=[...],
            original_max_pos=8192,
        ),
    ),
)

# MoE FFN: 32 experts, top-4, all weights packed MXFP4
moe_expert_quant = QuantSpec(
    qdtype=QDType.MX_FP4,
    group_size=32,
    quant_axis=0,
    scale_dtype=DType.UE8M0,                               # MXFP4 power-of-two scales
    has_zero_point=False,
    packing=PackingLayout.OCP_MX_BLOCK32,                  # NEW v3
    block_layout=BlockLayout.OCP_MX,
    accumulator_dtype=DType.FP32,
    role=QuantRole.MOE_EXPERT_WEIGHT,                      # NEW v3
    training_native=True,                                  # NEW v3
)

moe_router_quant = QuantSpec(
    qdtype=QDType.FP16, group_size=None, quant_axis=0,
    scale_dtype=DType.FP16, has_zero_point=False,
    packing=PackingLayout.NONE, accumulator_dtype=DType.FP32,
    role=QuantRole.MOE_ROUTER_WEIGHT,                      # NEW v3
)

moe_spec = MoESpec(
    n_routed_experts=32, top_k=4, n_shared_experts=0,
    router_kind=RouterKind.SOFTMAX,
    router_norm=True,
    expert_ffn=FFNSpec(
        intermediate_size=2880,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
)

# DecoderBlockSpec composes attn + moe + the two QuantSpecs;
# the weight loader knows from QuantRole how to pack each weight slot.
```

These five examples cover ~80% of the v3 spec surface. The remaining 20%
(Mamba-3 complex state, DeepSeek-V4 CSA+HCA, Llama 4 iRoPE per-layer NoPE,
Apple AFM cross-block, Visual Causal Flow for DeepSeek-OCR, Falcon-Edge
retrainable BitNet) follow the same pattern with the corresponding new
spec fields documented in §5.2.

## 6. Phase B — Per-model layer factory pattern

Each `models/<family>/` contains:

### `layer.md`

Required sections (the §6 schema with §7 weight-name mapping now
**mandatory** since M1 added it for Qwen3):

1. **Identity** — family name, variants and param counts, release date,
   source paper, source repo.
2. **Decoder block diagram** — ASCII or mermaid showing residual
   structure, sub-layers, normalisation placement.
3. **Tensor IO trace** — per-step shapes from input `x:[B,S,D]` to output
   `[B,S,D]`, every intermediate annotated with dtype and shape.
4. **Op trace** — sequence of `api.ops.*` calls (including casts and
   reshapes) implementing the block. MoE, SSM, DSA blocks expand their
   internal routing / scan / indexer ops explicitly.
5. **Spec instantiation** — the `DecoderBlockSpec` values for the
   canonical variant.
6. **Quirks** — anything model-specific that is *not* a generic axis
   (e.g., `1+w` RMSNorm baking foot-gun, Gemma 4 fixed-scale QK-norm
   values, llama.cpp Q/K permute, post-norm FP16 overflow risk, Visual
   Causal Flow block boundaries).
7. **Weight-name mapping** — table mapping HF tensor names → API tensor
   slots (see §9). **MANDATORY (v3) — added to schema since M1 shipped
   it for Qwen3 and the audit confirmed it is the cheapest section that
   prevents the most loader bugs.**
8. **Source citations** — file:line into HF transformers, vLLM, llama.cpp
   (from `research/02-layer-sources.v3.md`).

### `layer.py`

```python
def build_decoder_layer(config: <Family>Config, layer_idx: int = 0) -> DecoderBlock:
    """Assemble one decoder block using api/ primitives only.

    Must NOT import from transformers.models.<family>. Must use only api/.
    """
```

### `weight_loader.py`

```python
def load_weights(checkpoint_path: str, layer: DecoderBlock, layer_idx: int,
                 quant: Optional[QuantSpec] = None) -> None:
    """Map HF / GGUF / AWQ tensor names → API tensor slots."""
```

See §9 for the mapping schema.

### `test_layer.py`

Required tests:

- **Shape**: instantiate with random weights at ≥2 size variants;
  forward pass; assert output shape and dtype.
- **Determinism**: same seed → same output (max-abs diff = 0).
- **KV cache write/read**: prefill T tokens, decode token T+1, assert
  cache state evolves correctly.
- **Numerical equivalence** (Qwen3 only — kickoff gate, **done**): see §8.

## 7. Validation hierarchy

| Level | What it proves | Coverage |
|---|---|---|
| **(a) Structural** | Shape and dtype correct, residual structure correct | All ~24 families |
| **(b) Op-trace** | API ops match the "intended" graph (verified against HF source op-by-op) | ~5 representative families (Qwen3, Gemma 4, DeepSeek-V3, Mixtral, Jamba) |
| **(c) Numerical** | Real HF weights → API-assembled layer produces fp-equivalent output to HF reference | Qwen3 sample only (M1 kickoff gate — **done**, max_abs_diff 4.77e-7) |
| **(d) Quantization round-trip** | AWQ unpack → dequant → matmul produces output equivalent to HF AWQ reference within quant tolerance | Qwen3 + one B-batch model per quant scheme |

## 8. Qwen3 M1 kickoff gate — **STATUS: COMPLETE (2026-06-05)**

**M1 result:** **passed at `max_abs_diff = 4.77e-7`** on the Qwen3-0.6B
decoder layer block-level numerical equivalence test (§8.2(e)). The M1
implementation landed across 32 commits between 2026-06-04 and
2026-06-05; the §5.2 spec dataclass count grew from a partial 8 → 14
during the fix-pass on 2026-06-05; the §5.1 op floor grew from 9 → 16
during the same fix-pass. **74 tests passing.** This section now
documents the achieved gate, not the planned one.

The artifact list, sub-op isolation ladder, and review questions are
preserved from v2 §8 below for the historical record.

### 8.1 Artifacts (landed)

| Artifact | Contents |
|---|---|
| `api/ops.py` | 16 logical ops (M1 fix-pass completed all of softmax/top_k/gather/scatter/conv1d/selective_scan even though Qwen3 doesn't exercise the latter four — they ship in M1 so B6/B7 don't bottleneck) |
| `api/specs.py` | 14 spec dataclasses including `MoESpec`, `SSMSpec`, `SSDSpec`, `ConvSpec`, `GroupRoutingSpec`, `LayerScaleSpec` (referenced by future batches; Qwen3 uses only Attention/RoPE/Norm/FFN/KVCache/Quant) |
| `api/norm.py` | `RMSNorm` + `QKNorm` modules |
| `api/rope.py` | RoPE block with `SPLIT_HALF` basis (M1 chose llama.cpp's basis for storage compat with GGUF; the HF basis is provable from the same op via the QK-permutation flag) and `LLAMA3` scaling |
| `api/attention.py` | Attention block supporting GQA + QK-norm `PRE_ROPE` + RoPE + KV cache (contiguous layout) |
| `api/feedforward.py` | FeedForward block, SwiGLU |
| `api/kvcache.py` | `ContiguousKVCache` with HND memory layout, FP16/BF16 dtypes, `EXPLICIT_PASS` ownership |
| `api/quant.py` | AWQ W4A16 (g=128) quant/dequant round-trip path |
| `api/block.py` | DecoderBlock with PRE-norm residual structure |
| `api/weights.py` | `from_hf_dict` reading `dtype` (transformers 5.x) with `torch_dtype` fallback |
| `models/qwen3/layer.md` | Full layer doc per §6 schema with weight-name mapping table |
| `models/qwen3/layer.py` | `build_qwen3_decoder_layer(config, layer_idx)` |
| `models/qwen3/weight_loader.py` | `load_qwen3_weights(state_dict, layer, layer_idx)` |
| `models/qwen3/test_layer.py` | Shape, determinism, KV cache prefill+decode, sub-op isolation ladder (`atol=1e-5`), block-level numerical equivalence (`atol=5e-4` — passed at `4.77e-7`), AWQ round-trip on `q_proj` |

### 8.2 Test specification — passed

**Reference implementation:** HuggingFace `transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer`
in eager mode, FP32 compute, FP16 storage. Pinned to
`transformers==4.46.x`.

All seven test categories (a)–(g) from v2 §8.2 passed. The sub-op
isolation ladder (d) was particularly load-bearing: it caught a RoPE
basis convention mismatch early before it could pollute the block-level
test.

### 8.3 Kickoff review questions — answered

1. Is the API surface what you expected? **16 ops, 14 specs (was "12
   specs" — promoted in M1 fix-pass). Yes.**
2. Is the spec dataclass pattern right? **Yes; v3 extends but does not
   restructure.**
3. Is `models/qwen3/layer.md` the right doc format (with §6
   weight-name mapping table)? **Yes; v3 promotes the table to
   mandatory.**
4. Is `build_qwen3_decoder_layer` the right factory shape? **Yes.**
5. Does the AWQ exercise look right? **Yes; round-trip passes
   bit-exactly against `autoawq` reference dequant.**
6. Did the sub-op isolation ladder catch any drift before block-level?
   **Yes — caught a RoPE basis mismatch and a `dtype` vs
   `torch_dtype` config-key drift in transformers 5.x.**
7. Did the FP16 cache exercise prefill+decode equivalence? **Yes.**

**M1 approval gates M2** — the parallel rollout across remaining models.

## 9. Weight-loader story

### 9.1 HF → API tensor-name mapping (Qwen3-0.6B canonical — landed M1)

(Preserved from v2 §9.1. The full table is in `models/qwen3/layer.md §7`.)

### 9.2 AWQ unpacking (landed M1)

(Preserved from v2 §9.2. The AWQ round-trip test on Qwen3-0.6B
`q_proj` passes bit-exactly against `autoawq`.)

### 9.3 GGUF unpacking (deferred to B8 — sketched in v2 §9.3)

### 9.4 Loader contract (landed M1)

```python
def load_weights(
    checkpoint_path: str,
    layer: DecoderBlock,
    layer_idx: int,
    *,
    format: Literal["hf_fp16", "hf_awq", "gguf_q4km"] = "hf_fp16",
    quant: Optional[QuantSpec] = None,
) -> None:
    ...
```

### 9.5 Gemma 4 weight mapping (NEW v3, for B0.5)

Gemma 4 introduces several novel weight-loader concerns:

| HF tensor name | API tensor slot |
|---|---|
| `model.embed_tokens.weight` | `model.embedding.weight` |
| `model.embed_tokens_per_layer.weight` | `model.ple.embedding.weight` (NEW — PLE table) |
| `model.layers.{i}.per_layer_input_gate.weight` | `block_{i}.ple.fanout_gate.weight` (NEW) |
| `model.layers.{i}.per_layer_projection.weight` | `block_{i}.ple.fanout_proj.weight` (NEW) |
| `model.layers.{i}.self_attn.q_proj.weight` | `block_{i}.attention.q_proj.weight` |
| `model.layers.{i}.self_attn.k_proj.weight` | `block_{i}.attention.k_proj.weight` |
| `model.layers.{i}.self_attn.v_proj.weight` | `block_{i}.attention.v_proj.weight` (on Gemma 4 ≥12B global layers: aliased to k_proj.weight if `attention_k_eq_v=True`) |
| `model.layers.{i}.self_attn.o_proj.weight` | `block_{i}.attention.o_proj.weight` |
| `model.layers.{i}.self_attn.q_norm.weight` | `block_{i}.attention.qk_norm_q.weight` (Gemma 4 fixed-scale: 0.9916 local / 1.0228 global) |
| `model.layers.{i}.self_attn.k_norm.weight` | `block_{i}.attention.qk_norm_k.weight` |
| `model.layers.{i}.{four-norm-sub-norm}.weight` | `block_{i}.{pre_attn / post_attn / pre_ffn / post_ffn}_norm.weight` |
| `model.layers.{i}.mlp.gate_proj.weight` | `block_{i}.ffn.gate_proj.weight` (double-wide on E2B/E4B) |

The K=V loader behaviour on Gemma 4 12B / 26B-A4B / 31B: when
`attention_k_eq_v=True`, the loader detects the absence of a separate
`v_proj.weight` in the safetensors index and aliases the `v_proj` slot to
the `k_proj` tensor. This is the simplest expression of K=V; the
attention forward path is unchanged.

## 10. Per-model rollout (M2+) — v3 revised batches

After M1 (Qwen3 — **done 2026-06-05**), the rollout proceeds in
**architectural-diversity-first** order. v3 reorders to put
architecturally-novel families up front since they force the new axes
(Gemma 4 alone forces four new spec fields and one new building block).

Each new model produces a **diff against the API**: either it fits (no
API changes), or it forces a *named extension* with justification.

| Batch | Models | API extensions expected |
|---|---|---|
| **B0** (done) | **Qwen3 0.6B** | M1 kickoff — 16 ops + 14 specs landed at `max_abs_diff = 4.77e-7` |
| **B0.5** (**PROMOTED for v3**) | **Gemma 4 E2B** | Forces FOUR new spec fields (`partial_rotary_factor=0.25`, `num_kv_shared_layers=20`, `attention_k_eq_v` flag added even though E2B is False, `PLESpec(embedding_dim=256, paged_to_storage=True)`) plus the `FIXED_SCALE` weight mode on QK-norm (local 0.9916 / global 1.0228), the `PROPORTIONAL` RoPE scaling enum value, the dual-θ RoPE (10000 local / 1000000 global), the dual head_dim (`head_dim=256` local / `global_head_dim=512` global), the `final_logit_softcap=30.0` field on AttentionSpec (restored from Gemma 2; Gemma 3 had dropped it), the four-norm sandwich (input + post-attn + pre-ffn + post-ffn), the `double_wide_mlp=True` on FFNSpec, the `GELU_PYTORCH_TANH` activation, and the new building block `PLE`. Locking these early avoids retrofit. One model only — E2B chosen because it exercises the maximum new-axis surface (the 12B+ variants exercise K=V which E2B does not, but every other novel axis is on E2B). |
| **B1: Llama-family dense (Llama-3 era)** | Llama 3.x dense (1B/3B/8B), Llama 3.1, Mistral 7B v0.3, SmolLM3, TinyLlama 1.1B | None — Llama-3 RoPE scaling, RMSNorm, SwiGLU. Validates the historical floor. |
| **B2: Llama-family with quirks** | Granite 3.x / 4.1 μP, MiniCPM 3 MLA, Phi-3 fused QKV, Phi-4-mini, Phi-3-small BlockSparse | LayerScaleSpec for Granite μP, fused gate-up for Phi-3, LongRoPE for Phi-3, BlockSparse for Phi-3-small |
| **B3: Post-norm / dual-norm** | OLMo 2, Gemma 2/3 | `NormPosition.POST`, `NormPosition.PRE_AND_POST`, sub-norms, BF16-residual requirement for OLMo 2, `final_logit_softcap=30` for Gemma 2 |
| **B4: Per-layer heterogeneous** | Llama 4 iRoPE (Scout 17B-A), Gemma 3 SWA alternation, Mistral Ministral interleaved SWA | `apply_rope_per_layer` on AttentionSpec, `irope_layer_pattern` on RoPESpec, per-layer `list[DecoderBlockSpec]` |
| **B5: MLA family** | DeepSeek-V2-Lite, V3-Lite, **V3.2 DSA**, **V4 CSA+HCA**, MiniCPM-3 | `AttentionKind.MLA`, `AttentionKind.DSA`, `AttentionKind.CSA_HCA`, `IndexerSpec`, `CSASpec`, `HCASpec`, `KVCacheSpec.LIGHTNING_INDEXER`, `KVCacheSpec.CSA_HCA_DUAL_COMPRESSED` |
| **B6: MoE family** | Mixtral 8×7B, Qwen3-MoE, **Qwen3-Next ultra-sparse 512+1/top-10**, OLMoE, DeepSeek-V3-MoE, Granite 4 MoE | MoESpec variants — DeepSeek's `SIGMOID_PLUS_BIAS` router, shared experts, `GroupRoutingSpec`, `moe_intermediate_size`, `is_ultra_sparse` marker, `AttentionKind.GATED_DELTANET` for Qwen3-Next |
| **B7: SSM-hybrid** | Mamba 2/3, Jamba, Zamba2, Hymba, **Phi-4-mini-flash (Samba)**, **Falcon-H1**, **Granite 4 H 9:1**, **Nemotron 3 Super/Ultra**, RecurrentGemma, RWKV-7, **MiniMax-Text-01 Lightning 7:1** | `TokenMixerKind.{SSM_MAMBA1, SSM_MAMBA2, SSM_MAMBA3, HYBRID_PARALLEL, HYBRID_ALTERNATING, HYBRID_SEQUENTIAL_N_M, SSM_RWKV, ATTENTION_LIGHTNING}`, `SSMSpec`, `SSDSpec`, `Mamba3Spec`, `is_complex_state`, `n_mimo_outputs`, `KVCacheSpec.SSD_STATE_PLUS_CONV`, `KVCacheSpec.COMPLEX_STATE` |
| **B8: OCR-LLM** (NEW in v3) | **DeepSeek-OCR (Visual Causal Flow)**, **GOT-OCR 2.0**, **Qwen2.5-VL family** | `MaskKind.VISUAL_CAUSAL_FLOW`, `block_bidirectional_mask`, M-RoPE / 2D-RoPE via `RoPESpec.is_2d` + `mrope_section`. (Vision encoder lives outside the API; the decoder block is what we exercise.) |
| **B9: Audio LM** (NEW in v3) | **Moshi 7B (Helium-1 + Mimi + Temporal + Depth transformers)**, **Voxtral 4B** | None expected on the decoder side; the Mimi audio codec and the Temporal/Depth transformer stack live outside the API. The single-decoder block of Helium-1 reuses the GQA/RMSNorm/RoPE template. |
| **B10: Quant exercises** | **MXFP4-native on GPT-OSS 20B**, **Falcon-Edge 1.58-bit retrainable**, **Gemma 4 mobile-INT4 QAT**, **NVFP4 mixed-precision** | QuantSpec extensions: `qdtype=INT2_QAT`, `qdtype=NVFP4`, `packing=OCP_MX_BLOCK32`, `packing=MARLIN_4BIT`, `packing=MARLIN_24`, `training_native=True`, `retrainable=True`, `target_runtime ∈ {litert, mediapipe, ane, hexagon}`, role=`MOE_EXPERT_WEIGHT` / `MOE_ROUTER_WEIGHT` / `LM_HEAD` |

**Total batches: 11** (B0 done; B0.5 → B10 pending). The rollout is
finished when one model from each batch has (a)-level validation and
the API has settled (no extension diffs in the last 2–3 models). Target
model count remains **~24** (locked in v2 §10), distributed across B0
(1) + B0.5 (1) + B1 (5) + B2 (5) + B3 (3) + B4 (3) + B5 (5) + B6 (6) +
B7 (11) + B8 (3) + B9 (2) + B10 (4 quant-on-existing). Note B10 reuses
earlier models so doesn't add to the distinct-family count; the
distinct-family target stays 24, the v3 census names 52 architectural
families overall but the rollout target is the floor that validates the
spec axes.

### 10.1 Batch dependencies

Some batches strictly depend on earlier ones. The order is not just
"sequential" — many batches can run in parallel after their prerequisite
landed. Subagent-driven development (v2 §11 risk mitigation) dispatches
one agent per batch.

- B0.5 (Gemma 4 E2B) depends on B0 (Qwen3) because the M1 op floor and
  spec dataclasses are the foundation. **B0.5 must land before any other
  batch starts**, because it locks the four novel spec fields that B5,
  B6, B7 will reuse (`partial_rotary_factor`, `num_kv_shared_layers`,
  `attention_k_eq_v`, `PLESpec`).
- B1, B2, B3 are independent and can run in parallel after B0.5 lands.
- B4 (per-layer heterogeneous) depends on B3 (post-norm / dual-norm)
  because Gemma 3 SWA alternation uses the dual-norm machinery, and
  Llama 4 iRoPE uses the `apply_rope_per_layer` field which is locked in
  B4 itself but the layer-list machinery is locked in B3.
- B5 (MLA family) depends on B0.5 because DSA Lightning Indexer +
  CSA+HCA reuse the same `IndexerSpec` / `CSASpec` / `HCASpec` axes.
- B6 (MoE family) can run in parallel with B5 because MoE specs are
  independent of MLA/DSA. Qwen3-Next's Gated DeltaNet 3:1 hybrid is the
  first cross-batch dependency: it needs both `ATTENTION_GATED_DELTANET`
  (added in B6) and `HYBRID_SEQUENTIAL_N_M` (added in B7). v3 resolves
  this by adding both kinds to the enum during B6; the Gated DeltaNet
  building block ships in B6, and Qwen3-Next's specific 3:1 model
  assembly waits for B7's hybrid building block.
- B7 (SSM hybrid) depends on B6 because Granite 4 H-Tiny and
  Nemotron 3 Super are MoE-on-hybrid; the MoE building block needs to
  exist first.
- B8 (OCR-LLM) and B9 (audio LM) are independent of B5/B6/B7 and can
  start as soon as B4 lands (they reuse `apply_rope_per_layer` for
  Qwen2.5-VL and the four-norm sandwich for Voxtral). Both depend on
  B4.
- B10 (quant exercises) requires the loader infrastructure from each
  target batch. MXFP4-native exercise on GPT-OSS 20B needs the MoE
  building block from B6. Gemma 4 mobile-INT4 QAT on Gemma 4 E2B
  needs B0.5 to have landed. Falcon-Edge retrainable on Falcon-Edge 1B/3B
  is independent (Falcon-Edge architecture is GQA + RMSNorm, no novel
  axis). NVFP4 mixed-precision on a Llama-3 variant needs B1.

The critical path through the rollout is therefore: **B0 → B0.5 → B3 → B4 →
{B5 ‖ B6 ‖ B8} → B7 → B10**. B1 and B2 can run in parallel with B3+; B9
in parallel with B8.

## 11. Risks and mitigations (expanded for v3)

| Risk | Mitigation |
|---|---|
| API mutates too much during rollout, breaks earlier models | Each batch ends with a regression run of all earlier models' shape tests; spec dataclass changes are additive (new Optional fields with `None` default) |
| MLA dimensions break the standard `(n_heads, head_dim)` assumption | MLA is a first-class `AttentionKind` from M1; shared code paths conditional on `kind` |
| MoE token dispatch differs across runtimes | Reference impl uses dense scatter; production uses grouped GEMM or token-permute; the building block's *interface* is dispatch-kind-stable |
| Gemma 4 `attention_k_eq_v` is size-dependent (E2B/E4B no, ≥12B yes) | The flag is on `AttentionSpec`; the loader detects absence of `v_proj.weight` and aliases. B0.5 uses E2B (which doesn't have K=V); the K=V exercise is rolled into B0.5+ where the 12B variant is added |
| Gemma 4 PLE building block doesn't compose with the `DecoderBlockSpec` residual model | `DecoderBlockSpec.ple_injection: Optional[PLESpec]` adds a separate residual add; the PLE building block reads the PLE table (a model-level resource) and projects to residual dim. Acts like a second embedding stream, fully decoupled from KV cache |
| Lightning Indexer (DSA) leaks state across building blocks | The indexer state is a separate per-layer tensor on `KVCacheSpec`; the Attention building block consumes it but does not expose it. No global indexer state — each layer has its own |
| CSA + HCA composition order is unclear | Specified: CSA runs first (compresses global context), HCA runs on the compressed historical KV; both are inside the Attention building block. If a future model genuinely needs the reverse order, escalate to a `CSA_HCA_ORDER` enum |
| Mamba-3 complex storage forces complex-tensor support across the API | Limited to `KVCacheSpec.layout = COMPLEX_STATE` and `selective_scan(is_complex=True)`. Complex arithmetic stays inside the SSM building block; the rest of the API sees real-valued tensors |
| Visual Causal Flow mask is "block-bidirectional within a visual page, causal across pages" — block lengths are dynamic | `MaskKind.VISUAL_CAUSAL_FLOW` requires a `block_lengths` argument at forward time; the spec just names the kind. Treated as a runtime concern (block lengths come from the vision adapter, which lives outside the API) |
| iRoPE per-layer pattern doesn't match `RoPESpec` having a `irope_layer_pattern` | The pattern lives on RoPESpec (model-level shared frequencies + per-layer mask); per-layer dispatch is done by `AttentionSpec.apply_rope_per_layer[layer_idx]` consulted at build time. Two equivalent sources of truth; the loader populates both |
| Gemma 4 cross-layer KV sharing breaks the "each layer owns its cache" assumption | `KVCacheSpec.kv_source_layer: Sequence[Optional[int]]` — when set, the model assembler builds caches lazily so that layers reading from source layers don't allocate. Building block consumes the same `KVCache` reference as the source layer |
| Apple AFM 2-block sharing is different from Gemma 4 (cross-block vs same-block) | Both are subsumed under `ShareScheme` enum; `share_block_split: Optional[float]` carries AFM's 0.625 / 0.375 split; the assembler handles both cases via `kv_source_layer` |
| Trained attention sinks (GPT-OSS, Mistral Small 3.1) require sink-token allocation in cache | `AttentionSpec.n_sink_tokens: Optional[int]` already in v2; the cache layout reserves first `n_sink_tokens` positions; SDPA receives the `sink` argument |
| Phi-4-multimodal mixture-of-LoRAs requires modality-routed adapter selection | Out of scope for the decoder block; the LoRA selection is a runtime concern (the loader can load a vision-LoRA-merged or audio-LoRA-merged weight set per inference call). The decoder API surface is unchanged |
| LayerScaleSpec is duplicated between DecoderBlockSpec attrs and LayerScaleSpec dataclass | M1 fix-pass settled this: simple scalars (residual_scale, embedding_scale, logits_scale) are direct attrs; `LayerScaleSpec` is reserved for OpenELM-style per-layer-tuple overrides |
| Op-floor count grew silently during M1 fix-pass (from "9 of planned 16" to "16 actually shipped") | M1 fix-pass closed the gap; v3 §5.1 now lists all 16 with verified IHV consensus columns |
| Native quantization (GPT-OSS MXFP4-native, BitNet, NVFP4-training) needs differentiable storage in QuantSpec | `QuantSpec.training_native` is informational only; the inference-time loader still reads packed weights. Training-time differentiability is out of scope (training is non-goal §2) |
| Falcon-Edge retrainable BitNet vs BitNet b1.58 train-from-scratch | `QuantSpec.retrainable` is informational; both load the same packed ternary weights; the distinction matters only for whether continued pretraining can extend the model |

## 12. Open questions deferred to writing-plans

1. **Base class for layer factories** — should `models/<family>/layer.py`
   share a `DecoderLayerBase`? Lean: stay standalone for the first few;
   extract a base only if duplication emerges.
2. **Numerical-equivalence reference choice for non-seed models** — HF
   transformers (slow, full Python) or vLLM (faster, more deps)? Lean: HF.
3. **Speculative decoding compatibility** — out of scope.
4. **Distributed / TP sharding** — out of scope.
5. **Adapter (LoRA) mergeability into quantized weights** —
   `QuantSpec.merged_lora_rank` reserved; not built.
6. **ONNX export round-trip** — useful demonstration; not built in M1.
7. **Vision-language M-RoPE / 2D-RoPE for VL decoders** — `RoPESpec.mrope_section`
   and `is_2d` are typed in v3 but the VL fusion topology is out of
   scope. Adding a Qwen2.5-VL exercise in B8 will be the first time we
   exercise 2D positions end-to-end.
8. **Sampling / logits processors** — out of scope.
9. **Speculative decoding cache structure (MTP)** — `KVCacheSpec`
   covers the storage shape but the speculation policy is out of scope.
10. **Encoder-decoder cross-attention** — out of scope.
11. **DSA Lightning Indexer training-time warmup** — `IndexerSpec.warmup_tokens`
   is informational; inference-time behaviour is "trust the indexer
   always". Whether this matches the V3.2 production behaviour is
   flagged for verification.
12. **CSA + HCA composition order** — currently CSA first, HCA second.
   If V4-Pro vs V4-Flash differ on this, escalate.
13. **Mamba-3 MIMO `n_mimo_outputs > 1` semantics** — informational
   only in v3; the SSM building block computes a single output stream
   by default. Whether MIMO produces multiple residual streams (which
   would require dual-stream DecoderBlock) is flagged for B7.
14. **Apple AFM `target_runtime=ANE`** — `target_runtime` is on QuantSpec
   for quantization variants; whether the surrounding ops also depend on
   target (e.g., ANE prefers BF16, Hexagon prefers INT8) is a runtime
   concern outside the spec.

## 13. Evidence index

All design decisions cite one or more of the v3 reports (verbatim
sections):

- **research/00-evolution.md** — timeline 2022→2026, lineage,
  per-axis evolution arc (referenced by §0 preamble and the rollout
  batch ordering). v2-era; still current.
- **research/01-model-census.v3.md** — 148 model rows, 19-axis catalog
  (was 15 in v2), ~52 architectural families, OCR-LLM lineage, Gemma 4
  five-size sweep, Apple AFM promoted to first-class, axes A16/A17/A18/A19
  added (VLM fusion, vision-token compression, cross-layer KV sharing,
  per-layer embeddings). Supersedes v2.
- **research/02-layer-sources.v3.md** — 48 family sections, 38 axes,
  6 new axes added (§3.33 cross-layer KV sharing, §3.34 PLE, §3.35
  partial-rotary RoPE, §3.36 vision-adapter topology, §3.37
  block-bidirectional mask, §3.38 MXFP4-native training), 10 abandoned
  designs. Supersedes v2.
- **research/03-ihv-opsets.v2.md** — 13-runtime synthesis, 16-logical-op
  floor, KV cache representation taxonomy, MoE/embedding/quant rows.
  v2-era; still current.
- **research/04-quantization.v3.md** — 48 schemes, 21-axis QuantSpec,
  "support 6" baseline (Q4_K_M, AWQ, FP8 W8A8, IQ2_M, MXFP4, Gemma 4
  mobile-INT4 QAT), Era 6 (MXFP4-native, Falcon-Edge retrainable) and
  Era 7 (Gemma 4 mobile QAT, NVFP4 training-mode, DeepSeek FP8
  maturation, Apple AFM 2-bpw). Supersedes v2.
- **research/05-kvcache-attention.v3.md** — 38 attention × 18 RoPE × 18
  cache variants, KVCacheSpec axes expanded (vAttention, YOCO, NSA,
  MTP, Lightning Indexer, Gemma 4 cross-layer, Apple AFM cross-block,
  Mamba-3 complex), AttentionSpec axes expanded (attention_k_eq_v,
  global_head_dim, lightning_indexer, csa_spec, hca_spec,
  apply_rope_per_layer, block_bidirectional_mask, output_gate,
  attn_temperature, final_logit_softcap, qk_norm_fixed_scale,
  is_complex_state, n_mimo_outputs), RoPESpec axes expanded
  (partial_rotary_factor, mrope_section, is_2d, PROPORTIONAL scaling),
  AttentionKind extended (DSA, CSA_HCA, MSA, MAMBA3_COMPLEX,
  GATED_DELTANET), MaskKind extended (VISUAL_CAUSAL_FLOW, DSA_TOP_K,
  MSA). Supersedes v2.

## 14. Glossary

### Enum values

`AttentionKind`: `STANDARD` (MHA/GQA/MQA); `MLA` (DeepSeek-V2/V3,
MiniCPM-3); `DIFFERENTIAL` (Microsoft 2024); `LINEAR_RETENTION` (RetNet);
`LINEAR_DELTANET` (DeltaNet); **`GATED_DELTANET`** (Qwen3-Next 3:1
hybrid); `LINEAR_GLA` (Gated Linear Attention); `NSA` (DeepSeek 2025);
**`DSA`** (DeepSeek-V3.2 Lightning Indexer); **`CSA_HCA`** (DeepSeek-V4);
**`MSA`** (MiniMax-M3.0); **`MAMBA3_COMPLEX`** (Mamba-3).

`QKVLayout`: `SPLIT`; `FUSED` (Phi-3); `MLA_LATENT`.

`MaskKind`: `CAUSAL`; `SWA`; `SWA_GLOBAL_ALT` (Gemma 3/4); `SINK`
(GPT-OSS, Mistral Small 3.1); `FULL`; `BLOCK_SPARSE` (Phi-3-small);
`DOCUMENT_CAUSAL`; **`VISUAL_CAUSAL_FLOW`** (DeepSeek-OCR);
**`DSA_TOP_K`** (DeepSeek-V3.2); **`MSA`** (MiniMax-M3.0); `CUSTOM`.

`QKNormPhase`: `NONE`; `PRE_ROPE` (Qwen3); `POST_ROPE` (Gemma 3).

`NormShape`: `FULL_HIDDEN`; `PER_HEAD_DH` (Qwen3 QK-norm); `FULL_HDH`
(OLMo 2 QK-norm).

`NormKind`: `RMS`; `LAYER`; `SCALE_NORM`; `DEEP_NORM`.

`WeightMode`: `NONE`; `STANDARD_W`; `ONE_PLUS_W` (Gemma); `LEARNED_PER_HEAD`
(QK-norm per-head); **`FIXED_SCALE`** (Gemma 4 QK-norm — value lives on
`AttentionSpec.qk_norm_fixed_scale`).

`GateKind`: `SWIGLU`; `GEGLU`; `GELU_ONLY`; `RELU2_ONLY`.

`Activation`: `SILU`, `GELU`, `GELU_TANH`, **`GELU_PYTORCH_TANH`**
(Gemma 4), `RELU2`, `GEGELU`, `SWISH`.

`RouterKind`: `SOFTMAX` (Mixtral); `SIGMOID_PLUS_BIAS` (DeepSeek-V3);
`LINEAR_TOP_K_NO_NORM`.

`GroupScoreKind`: `SUM_TOP_K_IN_GROUP` (DeepSeek-V3); `MAX`; `MEAN`.

`DispatchKind`: `DENSE_SCATTER`; `GROUPED_GEMM`; `TOKEN_PERMUTE`.

`CacheLayout`: `CONTIGUOUS`; `PAGED` (vLLM); `RING` (StreamingLLM);
`MLA_LATENT_PLUS_KROPE`; `SSM_STATE`; `SSD_STATE_PLUS_CONV`;
`NSA_THREE_TIER`; `YOCO_SHARED`; **`LIGHTNING_INDEXER`** (DeepSeek-V3.2);
**`COMPLEX_STATE`** (Mamba-3); **`CSA_HCA_DUAL_COMPRESSED`**
(DeepSeek-V4); **`AFM_TWO_BLOCK`** (Apple AFM).

`ShareScheme` (NEW v3): `NONE`; `CLA`; `YOCO`; `GEMMA4` (same-block
cross-layer sharing, `num_kv_shared_layers` of layers reuse K/V); `APPLE_AFM`
(cross-block, Block-1 carries 62.5% of layers with full KV, Block-2
carries 37.5% and reuses Block-1's K/V).

`MemoryLayout`: `HND`; `NHD`; `BHND`; `BNHD` (vLLM); `MLX_LIST`.

`CacheOwnership`: `EXPLICIT_PASS`; `STATEFUL`.

`CacheRole`: `PRODUCER` (YOCO encoder); `CONSUMER` (YOCO decoder);
`PRODUCER_CONSUMER` (default); `SHARED_GROUP_MEMBER` (CLA pair).

`QDType`: `INT2`, `INT3`, `INT4`, `INT5`, `INT6`, `INT8`, `FP8_E4M3`,
`FP8_E5M2`, `FP4`, `NF4`, `MX_FP4`, `MX_INT4`, `TERNARY` (BitNet),
**`INT2_QAT`** (Apple AFM, Gemma 4 mobile QAT), **`NVFP4`** (Blackwell
training-mode).

`PackingLayout`: `NONE`; `NIBBLE_LSB`; `NIBBLE_MSB`; `AWQ_INTERLEAVE`
(`[0,2,4,6,1,3,5,7]`); `GPTQ_INT32_PACK`; `HQQ_NIBBLE`;
`GGUF_K_SUPERBLOCK`; `MX_BLOCK`; **`MARLIN_4BIT`** (vLLM Marlin kernel);
**`MARLIN_24`** (Marlin 2:4 sparse); **`OCP_MX_BLOCK32`** (OCP MXFP4
32-element block).

`BlockLayout`: `OCP_MX` (32-element micro-blocks with UE8M0 scale);
`GGUF_K` (256-element super-block); `AWQ_INTERLEAVE_K` (g=128 along K);
`GPTQ_BLOCK`.

`RotationKind`: `NONE`; `DIAGONAL_SMOOTHQUANT`; `DENSE_HADAMARD_QUAROT`;
`DENSE_LEARNED_SPINQUANT`; `HADAMARD_RUNTIME`.

`RuntimeApply`: `NONE`; `INPUT`; `INPUT_OUTPUT_BOTH`.

`CombineOp`: `NONE`; `ADD` (AQLM); `CONCAT`; `MULTI_LATTICE` (QuIP#).

`OutlierOffload`: `NONE`; `FP16_COLUMN_SPLIT` (LLM.int8);
`FP16_ROW_SPLIT`.

`QuantRole`: `WEIGHT`, `ACTIVATION`, `KV_K`, `KV_V`, `ATTN_INTERNAL`,
`EMBEDDING`, **`MOE_EXPERT_WEIGHT`**, **`MOE_ROUTER_WEIGHT`**, **`LM_HEAD`**.

`ScaleSource`: `STATIC`; `DYNAMIC_PER_TOKEN`; `DYNAMIC_PER_TENSOR`.

`DequantPath`: `IN_SDPA`; `AT_READ`; `AT_WRITE`.

`RoPEBasis`: `INTERLEAVED` (HF); `SPLIT_HALF` (llama.cpp).

`RoPEScaling`: `NONE`; `PI`; `NTK_STATIC`; `DYNAMIC_NTK`; `YARN`;
`LLAMA3`; `LONGROPE`; `IROPE`; **`PROPORTIONAL`** (Gemma 4 size-scaled).

`NormPosition`: `PRE`; `POST` (OLMo 2); `PRE_AND_POST` (sandwich —
Gemma 2/3/4); `PARALLEL_FFN` (Falcon-7B, GPT-J).

`TargetRuntime` (NEW v3): `LITERT`; `MEDIAPIPE`; `ANE` (Apple Neural
Engine); `HEXAGON` (Qualcomm); `CPU`; `GPU`.

`TokenMixerKind`: see §5.2.5 definition.

### Acronyms

- **AFM** — Apple Foundation Model (3.18B, 2025)
- **ALiBi** — Attention with Linear Biases (Press et al. 2021)
- **AWQ** — Activation-aware Weight Quantization (Lin et al. 2023)
- **CLA** — Cross-Layer Attention (CLA pairwise KV sharing)
- **CSA** — Compressed Sparse Attention (DeepSeek-V4, 2026)
- **DeepNorm** — post-norm variant with scaled residual (Wang et al. 2022)
- **DSA** — DeepSeek Sparse Attention (V3.2, 2025; Lightning Indexer)
- **FA1/2/3** — FlashAttention algorithms
- **FP8** — 8-bit floating point (E4M3 or E5M2)
- **GDN** — Gated DeltaNet (Qwen3-Next 2025)
- **GQA** — Grouped-Query Attention
- **GGUF** — GPT-Generated Unified Format (llama.cpp)
- **HCA** — Hierarchical Compressed Attention (DeepSeek-V4, 2026)
- **HQQ** — Half-Quadratic Quantization
- **iRoPE** — interleaved RoPE/NoPE (Llama 4 design)
- **KV cache** — cached Key and Value tensors
- **LayerScale** — per-channel learned residual multiplier
- **LRU** — Linear Recurrent Unit
- **MLA** — Multi-head Latent Attention (DeepSeek-V2)
- **MoE** — Mixture of Experts
- **MoT** — Mixture of Transformers (HY-Embodied-0.5)
- **MQA** — Multi-Query Attention
- **MSA** — MiniMax Sparse Attention (M3.0, 2026)
- **MTP** — Multi-Token Prediction (DeepSeek-V3, Gemma 4 drafters)
- **MXFP4** — 4-bit microscaling FP (OCP)
- **NF4** — Normal Float 4-bit (bitsandbytes)
- **NoPE** — No Positional Encoding
- **NSA** — Native Sparse Attention (DeepSeek 2025)
- **NTK** — Neural Tangent Kernel-aware scaling
- **NVFP4** — NVIDIA 4-bit FP (Blackwell)
- **OCP** — Open Compute Project (Microscaling spec)
- **PagedAttention** — vLLM page-allocated KV cache
- **PI** — Position Interpolation
- **PLE** — Per-Layer Embeddings (Gemma 4)
- **p-RoPE** — partial-rotary RoPE (Gemma 4 global)
- **QK-Norm** — RMSNorm applied to Q and K
- **QuaRot / SpinQuant** — rotation-before-quantize
- **RMSNorm** — Root-Mean-Square LayerNorm
- **RoPE** — Rotary Position Embedding
- **SDPA** — Scaled Dot-Product Attention
- **SSD** — Structured State-Space Duality (Mamba-2)
- **SSM** — State-Space Model
- **SwiGLU** — Swish-Gated Linear Unit
- **TMRoPE** — Temporal-M-RoPE (Qwen2.5-Omni)
- **vAttention** — virtual-memory-mapped KV cache
- **V2PE** — Variable Visual Position Encoding (InternVL 3)
- **VCF** — Visual Causal Flow (DeepSeek-OCR)
- **WKV** — RWKV's recurrent attention update
- **YaRN** — Yet another RoPE extensioN method
- **YOCO** — You Only Cache Once (Microsoft 2024)
- **μP** — Maximal Update Parameterisation

## 15. Bibliography

Foundational decoder-only architecture: Vaswani et al. 2017; Radford et
al. 2018, 2019; Brown et al. 2020.

Normalisation, activation: Zhang & Sennrich 2019; Shazeer 2020; Henry
et al. 2020; Wang et al. 2022; Yang et al. 2022 (μP).

Positional encoding: Su et al. 2021 (RoPE); Press et al. 2021 (ALiBi);
Chen et al. 2023 (PI); bloc97 2023 (NTK); Peng et al. 2023 (YaRN);
Kazemnejad et al. 2023 (NoPE); Microsoft 2024 (LongRoPE); Meta 2024
(Llama-3 scaling). **2025+:** Llama 4 technical report 2025 (iRoPE);
Gemma 4 technical report 2026 (p-RoPE, dual-θ, proportional scaling);
Qwen2.5-VL paper 2025 (M-RoPE); Qwen2.5-Omni 2025 (TMRoPE); InternVL 3
2025 (V2PE).

Attention variants: Shazeer 2019 (MQA); Ainslie et al. 2023 (GQA);
DeepSeek-AI 2024 (V2/MLA); Microsoft 2024 (Differential); Dao 2022/2023;
Shah et al. 2024 (FA3); DeepSeek 2025 (NSA). **2025+:** DeepSeek-V3.2
(Sep 2025, DSA + Lightning Indexer, arxiv:2512.02556); DeepSeek-V4-Pro/Flash
(Apr 2026, CSA+HCA); MiniMax-Text-01 (Jan 2025, Lightning Attention 7:1);
MiniMax-M3.0 (Jun 2026, MSA); Llama 4 (Apr 2025, iRoPE); Mamba-3 (Mar
2026, ICLR, complex state + MIMO, arxiv:2603.15569); Qwen3-Next (Aug
2025, Gated DeltaNet 3:1).

KV-cache variants: Kwon et al. 2023 (vLLM); Microsoft 2024 (vAttention,
YOCO); Apple 2025 (AFM 2-block cross-block, arxiv:2507.13575); Gemma 4
2026 (cross-layer KV sharing, PLE).

State-space models: Gu et al. 2022 (S4); Gu & Dao 2023 (Mamba); Dao &
Gu 2024 (Mamba-2 SSD); Peng et al. 2025 (RWKV-7); Orvieto et al. 2023
(LRU/Griffin); NVIDIA 2024 (Hymba). **2025+:** Mamba-3 (Mar 2026);
Granite 4 (Oct 2025, 9:1 hybrid); Nemotron 3 Super/Ultra (Mar–Jun 2026,
hybrid Mamba-2 + Transformer + MoE).

Mixture of Experts: Shazeer et al. 2017; Fedus, Zoph & Shazeer 2021
(Switch); Mistral AI 2023 (Mixtral); DeepSeek-AI 2024 (V3). **2025+:**
Qwen3-Next (Aug 2025, 512+1/top-10 ultra-sparse); Gemma 4 26B-A4B (Apr
2026, first Gemma MoE with `moe_intermediate_size=704`); Phi-4-multimodal
(Feb 2025, Mixture-of-LoRAs); HY-Embodied (Apr 2026, MoT).

Quantization: Frantar et al. 2022 (GPTQ); Xiao et al. 2022 (SmoothQuant);
Lin et al. 2023 (AWQ); Dettmers et al. 2023 (NF4/QLoRA); Badri 2024
(HQQ); Microsoft 2024 (BitNet b1.58); Egiazarian et al. 2024 (AQLM); OCP
2024 (MXFP4); NVIDIA 2024 (NVFP4); Ashkboos et al. 2024 (QuaRot); Liu
et al. 2024 (SpinQuant); Tseng et al. 2024 (QuIP#). **2025+:** GPT-OSS
2025 (MXFP4-native training); TII 2025 (Falcon-Edge retrainable 1.58-bit);
NVIDIA 2025 (NVFP4 training-mode promoted); DeepSeek 2025 (V3 native
FP8); Apple 2025 (AFM 2-bpw QAT with embed-INT4 / KV-INT8 split); Google
2026 (Gemma 4 mobile-INT4 QAT for LiteRT / MediaPipe / ANE / Hexagon).

Model families surveyed (selected v3 additions, full list in research
files): Meta 2025 (Llama 4 Scout/Maverick — iRoPE, early-fusion);
Google 2026 (Gemma 4 E2B/E4B/12B/26B-A4B/31B — p-RoPE, K=V, PLE,
cross-layer KV, first Gemma MoE); Qwen Team 2025 (Qwen3-Next 80B-A3B —
Gated DeltaNet 3:1, ultra-sparse MoE); DeepSeek-AI 2025/2026 (V3.1 dense,
V3.2-Exp DSA, V4-Pro/Flash CSA+HCA); IBM 2025/2026 (Granite 4.0 9:1
hybrid, Granite 4.1 dense, Granite Vision/Speech 4.1); NVIDIA 2025/2026
(Nemotron 3 Super/Ultra hybrid, Llama-Nemotron-Nano); Apple 2025 (AFM
3.18B); Tencent 2025/2026 (Hunyuan sweep 0.5B/1.8B/4B/7B, HunyuanOCR,
HY-Embodied-0.5, Hy3 Preview); TII 2025 (Falcon-H1, Falcon-Edge 1.58-bit
retrainable); Mistral AI 2024/2025/2026 (Ministral 3B/8B, Mistral Small
3/3.1/3.2, Magistral, Mistral Small 4, Mistral Medium 3.5, Voxtral 4B
TTS); NX-AI 2025 (xLSTM 7B); State-Spaces 2026 (Mamba-3); MiniMax 2025/2026
(MiniMax-Text-01, MiniMax-M3.0); Zai 2026 (GLM-5.1); Cohere 2026
(Command A+); DeepSeek-OCR 2025 (Visual Causal Flow); GOT-OCR 2.0 2024
(StepFun, smallest OCR-LLM); LG 2024/2026 (EXAONE 3.5, 4.5); Liquid AI
2026 (LFM2.5-VL); openbmb 2024+ (MiniCPM V/o); OpenGVLab 2024+ (InternVL
2/3/3.5); allenai 2024+ (Molmo, OLMo 2, OLMoE, olmOCR/olmOCR-2); Kyutai
2024/2025 (Moshi, Helium-1); Alibaba 2024/2025 (Qwen 2-Audio,
Qwen2.5-Omni, Qwen3-VL).

IHV / runtime references (v2 list still current; v3 adds LiteRT,
MediaPipe, Apple ANE toolchain, Qualcomm Hexagon QAIRT extensions): vLLM
project; TensorRT-LLM (NVIDIA); OpenVINO (Intel); QAIRT (Qualcomm,
incl. Hexagon); FastFlowLM (AMD); MIGraphX (AMD); Core ML (Apple);
MLX (Apple); KleidiAI (Arm); MindIE (Huawei); ggml/llama.cpp; AWS
Neuron; Google TPU XLA/Pallas; DirectML (Microsoft); Google AI Edge /
LiteRT; Google MediaPipe; Apple ANE / Foundation Models toolchain;
T-MAC / bitnet.cpp; onebitllms (Falcon-Edge).

## 16. Code style, fixtures, contribution, versioning, perf budgets

### 16.1 Code style and quality bar

- **Formatting:** `black --line-length 100`.
- **Linting:** `ruff` with the configuration in `pyproject.toml`.
- **Type checking:** `mypy --strict` on `api/`; lenient on `models/`.
- **Testing:** `pytest --strict-markers --tb=short`. M1 lands at **74
  tests passing**.
- **Pre-commit:** `pre-commit` hook chains black + ruff + mypy (api only).

### 16.2 Test fixture provenance

- **Qwen3-0.6B weights:** `Qwen/Qwen3-0.6B` (Apache 2.0). Pinned
  revision in `models/qwen3/fixtures.toml`. **(M1 landed.)**
- **Qwen3-0.6B AWQ weights:** AWQ round-trip exercise on `q_proj`
  passed bit-exactly. **(M1 landed.)**
- **HuggingFace transformers:** pin `transformers==4.46.x`.
- **Gemma 4 E2B (B0.5):** `google/gemma-4-E2B-it` (Apache 2.0,
  changed from Gemma Terms of Use). Verify `partial_rotary_factor=0.25`,
  `num_kv_shared_layers=20`, `final_logit_softcapping=30.0`,
  `attn_logit_softcapping=null` (text module),
  `qk_norm_fixed_scale=(0.9916 local, 1.0228 global)`, PLE table
  presence (`model.embed_tokens_per_layer.weight`).
- **GPT-OSS 20B (B10):** `openai/gpt-oss-20b`. Verify MXFP4-native
  packed weights, trained sinks (`n_sink_tokens=4`).
- **Falcon-Edge (B10):** `tiiuae/Falcon-Edge-1B`. Verify retrainable
  1.58-bit BitNet packed weights.

### 16.3 Contribution process

Five steps from upstream config to merged PR:

1. **Census check.** Confirm in `research/01-model-census.v3.md`.
2. **Layer doc.** Write `models/<family>/layer.md` per the §6 schema,
   **including the weight-name mapping table (§7, mandatory in v3).**
3. **Spec instantiation.** Implement `models/<family>/layer.py`.
4. **Weight loader.** Implement `models/<family>/weight_loader.py`.
5. **Tests.** Add `models/<family>/test_layer.py`.

### 16.4 API versioning

- The API follows semver. First cut after M1: **`v0.1.0`** (landed
  2026-06-05). The version is unstable through M10 (post-B10 rollout);
  breaking changes are allowed with a `CHANGELOG.md` migration note.
- Spec dataclass extensions are *additive only* by convention.
- Stability commitment begins at `v1.0.0`, tagged after the last
  B-batch lands and the API has had two consecutive batches with no
  changes.

### 16.5 Performance budgets (documented, not enforced)

- **Qwen3-0.6B single decoder layer, FP16, seq_len=128, batch=1, CPU:**
  < 100 ms forward in PyTorch eager. **M1 measured well under budget.**
- **AWQ unpack + GEMM for one linear (in=1024, out=1024, g=128), CPU
  FP16:** < 50 ms. **M1 measured under budget.**
- **KV cache write/read for prefill 128 + decode 1, CONTIGUOUS layout,
  FP16:** < 20 ms. **M1 measured under budget.**

---

**Ready for review.** The next step after spec approval is the
`superpowers:writing-plans` skill, which produces a detailed
implementation plan covering B0.5 (Gemma 4 E2B, promoted) → B10 (quant
exercises). The plan covers task ordering, per-batch parallelism via
subagent-driven development, and checkpoint gates. **M1 is done; B0.5
is next.**
