# llm-layers — Master Issues List (Critique Synthesis)

**Date:** 2026-06-04
**Source:** 7 parallel critique agents (`01-census-critique.md` through `07-meta-critique.md`)

## Verdict

**Artifact set is ~60% of the survey-paper bar.** Strong as an IR-design memo; weak as a comprehensive evolution survey. Single biggest gap: **no evolution narrative** anywhere — all reports are snapshots, zero timelines, zero lineage diagrams, zero "what was tried and abandoned" sections.

## Systemic defects (all reports)

| # | Defect | Severity | Affects |
|---|---|---|---|
| S1 | No evolution narrative — timelines, lineage, abandoned designs missing | MUST-FIX | 01, 02, 03, 04, 05 + spec |
| S2 | No 2023 baseline year — Llama 2, Mistral 7B v0.1, MPT, Falcon, ChatGLM, DeepSeek LLM, OLMo 1, Phi-2, StableLM 3B all absent | MUST-FIX | 01, 02 |
| S3 | No glossary / bibliography / equations / figures | SHOULD-FIX | All |
| S4 | Citation quality uneven; broken links in 03; version pins missing | SHOULD-FIX | 03, 04 |
| S5 | Cross-doc consistency: 45 models in census vs 10 in layer survey; AttentionSpec/QuantSpec/RoPESpec axis counts inconsistent with reports | MUST-FIX | spec, 02, 04, 05 |

## Per-report issues (MUST-FIX only)

### 01-model-census.md

| # | Issue |
|---|---|
| 1.1 | 14+ mainstream models missing (Llama 2, Mistral v0.1/v0.2, MPT 7B, Falcon family, OpenELM, BitNet b1.58, Phi-2, Phi-4-mini-flash, Gemma 1, Qwen 1/1.5/2, DeepSeek-LLM/Math/Coder V1, ChatGLM/GLM-4, OLMo 1, StarCoder 1) |
| 1.2 | 4 axes missing (per-layer width/depth scaling for OpenELM, parallel vs sequential sublayer, activation as standalone axis incl. ReLU² for BitNet, RoPE rotation domain) |
| 1.3 | Gemma 3 softcap fact error — Gemma 3 dropped softcap (both `final_logit_softcapping` and `attn_logit_softcapping` are null); axis 4.7 wrongly lumps with Gemma 2 |
| 1.4 | TinyLlama internal contradiction: axis 4.1 lists MHA, section 2.1 lists GQA |
| 1.5 | Surprise #12 head-dim decoupling example wrong (Phi-3-mini and Phi-4-mini both satisfy hidden/H=head_dim; correct example is Qwen3-0.6B: 1024/16=64 vs head_dim=128) |

### 02-layer-sources.md

| # | Issue |
|---|---|
| 2.1 | ~30 architectural families missing: BitNet (ternary, dual sub-norms), OpenELM (per-layer width), Mamba-2 (SSD vs Mamba-1), RecurrentGemma (Griffin/Hawk), Hymba (parallel SSM+attn heads), RWKV-7, Zamba2, Falcon-7B (parallel attn+FFN, historical), Falcon-Mamba, Phi-4-mini-flash (Samba), Phi-3-small (blocksparse), MiniCPM 3 (MLA), Granite 3.3, ChatGLM, Cohere, OLMoE, GPT-OSS, Qwen3-Next, etc. |
| 2.2 | **DeepSeek-V3 MLA cache compression factor wrong**: report says ~10×, actual ~71× for 128-head V3 config (40960 / 576) |
| 2.3 | "llama.cpp #29402" citation typo (actually transformers#29402) |
| 2.4 | Gemma 3 "sandwich norm" terminology is structurally pre-norm + sublayer-output-norm, not original CogView sandwich |
| 2.5 | 20-axis taxonomy missing: BitNet sub-norms inside attn/MLP, OpenELM per-layer width, Zamba2 parameter sharing, Falcon/GPT-J parallel residual, embedding tying, NoPE alternation (SmolLM3), sink tokens (GPT-OSS) |
| 2.6 | vLLM continuous batching shape change `[B,S,D] → [T,D]` never disclosed; TP sharding/FA backend selection/ggml quant matmul/llama.cpp QK permute all absent from divergence list |

### 03-ihv-opsets.md

| # | Issue |
|---|---|
| 3.1 | 4 P0 runtimes missing: AWS Neuron/NKI, Google TPU XLA HLO/Pallas, DirectML/Windows ML, **ggml itself** (dominant on-device CPU runtime — has no row) |
| 3.2 | `QNN_OP_KV_CACHE` claim unverified — single most consequential proprietary-op claim, anchors "op-as-cache" category, cannot be confirmed from any public link |
| 3.3 | Synthesis table missing 6+ rows: MoE, embedding lookup, all-reduce/TP collective, sampling/logits, quant-dequant, mask construction |
| 3.4 | KV-cache and fusion-shape counts under-counted (6 KV / 5 fusion vs actual 9–10 / 8 once XLA threading, RadixAttention, vLLM-PA v1/v2, ggml pool, DirectML state, eager-graph, HLO-custom-call added) |
| 3.5 | Pervasive version drift: OpenVINO `paged_attention.hpp` URL 404s; 28-input list includes likely feature-branch fields; no QAIRT/MLX/coremltools/OpenVINO release tags pinned |
| 3.6 | 5 P1 runtimes missing (TVM/MLC-LLM, TFLite GenAI, OneDNN graph SDPA, MTIA, Modular MAX/Mojo) |

### 04-quantization.md

| # | Issue |
|---|---|
| 4.1 | 17 schemes missing (36% undercount): Marlin (format, not just kernel), EXL2, AQLM, QuIP#, SqueezeLLM, OmniQuant, BiLLM 1-bit, QuaRot/SpinQuant proper rotation, MXINT4, ZeroQuant, LUT-based KleidiAI, FP8 sub-variants (per-tensor W + per-token A vs per-channel), pre-k-quant GGUF Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q3_K_S/M/L sub-variants |
| 4.2 | **Q4_K sub-block size wrong**: report says 16×16, actual is 32 weights × 8 sub-blocks; 16-weight sub-block is Q2_K/Q3_K only |
| 4.3 | **imatrix semantics wrong**: report says weights are multiplied by sqrt(importance) before RTN; actually imatrix is a per-column *weighting of the MSE objective* inside the k-quant rounder |
| 4.4 | `pre_transform` axis collapses QuaRot/SpinQuant/SmoothQuant (mathematically distinct) into one enum |
| 4.5 | `codebook` can't express AQLM (multi-codebook additive), SqueezeLLM/EXL2 (per-row), QuIP# (lattice+residual) |
| 4.6 | No per-row variable-bitwidth (EXL2) or outlier-offloading (LLM.int8 fp16 column split) representation |
| 4.7 | "Support only 3" overclaim — actually exercises ~half of axes (no codebook, no rotation, no UE8M0, no variable-bit, no outlier-offloading) |

### 05-kvcache-attention.md

| # | Issue |
|---|---|
| 5.1 | **Qwen3 QK-norm placement backwards**: report says post-RoPE; `modeling_qwen3.py` applies QK-norm BEFORE RoPE. Gemma 3 is the post-RoPE QK-norm model — they're conflated |
| 5.2 | **Qwen3 rope_theta = 5M wrong**: HF config for Qwen3-0.6B reports 1M |
| 5.3 | **MiniCPM-3 mischaracterized as CLA**: actually MLA per MiniCPM-3 paper |
| 5.4 | "Sink is pure inference-time recipe" no longer true post-2025 (GPT-OSS, Mistral-Small-3.1 train with sinks) |
| 5.5 | 12 attention variants missing: FA1/2/3 algorithmic diffs, NSA (DeepSeek 2025), Mamba selective-scan internals, Mamba-2 SSD, PagedAttention v2/vAttention, DeltaNet/Gated DeltaNet, GLA, RingAttention, TreeAttention, RadixAttention internals |
| 5.6 | 6 RoPE variants missing: Dynamic NTK (ChatGLM), PI as standalone, DCA, MM-RoPE update, xPos/NoPE/Sandwich group |
| 5.7 | 6 cache layouts missing/mis-described: YOCO mischaracterized as sharing scheme (actually 2-stage architecture), CacheGen, KVQuant 2-bit, tree/beam cache, vAttention VM-mapping, NSA three-tier, MLA prefix cache, MTP cache |
| 5.8 | MLA `q_head_dim ≠ v_head_dim` asymmetry not captured by single `head_dim` axis |
| 5.9 | SSM cache state incomplete — missing `conv_state` (`d_conv` axis); Mamba-2 SSD axes (`chunk_size`, `headdim`, `ngroups`) absent |

### Design Spec (2026-06-04-llm-layers-design.md)

| # | Issue |
|---|---|
| 6.1 | **`SSMSpec` and `TokenMixerKind` referenced but never defined** (§5.4 line 289, §9 B7). Biggest single hole; blocks B7 entirely |
| 6.2 | **Op floor (9) insufficient for declared scope**. Cannot express MoE routing (`softmax`, `top_k`, `gather/scatter`) or Mamba (`conv1d`, `selective_scan`) |
| 6.3 | **False AWQ ≡ GPTQ ≡ HQQ storage-identical claim**. Storage layout differs (AWQ interleave vs GPTQ int32 pack); the "3 baselines exercise every QuantSpec axis" claim fails for `codebook`, `pre_transform`, `block_layout=OCP_MX` |
| 6.4 | **M1 numerical-equivalence test underspecified**: "within 1e-3 on a known prompt" doesn't define input shape, position_ids, tolerance norm, reference dtype, or sub-op ladder. 1e-3 too permissive |
| 6.5 | **Weight-loader story missing**. AWQ unpacking (interleave + qzeros + scales → dequant) is M1's hardest deliverable but gets one bullet. No HF-name → API-slot mapping function in M1 |
| 6.6 | Dataclass gaps (18): AttentionSpec missing CLA/YOCO `cache_role` + BLOCK_SPARSE mask + LINEAR-kind too narrow; RoPESpec missing PI/DynamicNTK/NoPE/ALiBi; NormSpec missing ScaleNorm/DeepNorm; MoESpec `GroupRoutingSpec` undefined; KVCacheSpec missing vAttention/YOCO |
| 6.7 | 8 missing sections: glossary, weight loader, perf budget, fixture provenance, code style, contribution process, API versioning, survey-paper preamble |
| 6.8 | M1 op coverage insufficient: needs Conv1d, SelectiveScan, top-k routing, gather/scatter, softmax (explicit) added to support B7 SSM-hybrid batch |

## Cross-report inconsistencies (meta-review §1)

| # | Issue |
|---|---|
| M1 | Census says 45 models / 22 archs; layer survey covers 10. Why such a gap? |
| M2 | AttentionSpec in spec covers ≤19 axes but research/05 v2 will surface more (per-layer apply_rope, MLA asymmetric head_dim, dynamic NTK as standalone) |
| M3 | QuantSpec covers fewer axes than the 25-scheme research will need (rotation, codebook combine_op, scale_source, variable-bit, outlier_offload) |
| M4 | Attention-variant counts mismatch: spec 4, census 8, kvcache report 13 — never reconciled |
| M5 | Model-name spelling inconsistent (Llama 3 vs Llama-3 vs Llama3; DeepSeek-V3 vs DeepSeek V3) |
| M6 | Spec §12 citations refer to sections that don't exist with stated numbers in research/05 |

## Recommended v2 strategy

Per user directive: **no in-place edits — produce v2 files**.

| New / v2 file | Source | Adds |
|---|---|---|
| `research/00-evolution.md` (**NEW**) | Synthesis from all reports + Meta § 2 | Timeline 2022→2026, lineage diagrams, abandoned-design section, per-axis evolution arc, consensus vs divergence table, future directions |
| `research/01-model-census.v2.md` | v1 + 01-census-critique | +14 missing models, +4 axes, fix 5 errors, add per-family evolution, add 2023 baseline column |
| `research/02-layer-sources.v2.md` | v1 + 02-critique | +20 families, fix MLA compression (~71×), fix sandwich-norm, add missing axes incl. BitNet sub-norms / OpenELM per-layer width / parallel residual, add per-family evolution narrative |
| `research/03-ihv-opsets.v2.md` | v1 + 03-critique | +4 P0 runtimes (Neuron/TPU/DirectML/ggml), +5 P1, verify or remove QNN_OP_KV_CACHE claim, fix synthesis table rows, version pin all references |
| `research/04-quantization.v2.md` | v1 + 04-critique | +17 schemes (Marlin/EXL2/AQLM/QuIP#/BitNet/MXINT4/...), fix Q4_K sub-block size, fix imatrix semantics, split `pre_transform` into rotation vs scale, expand "support 3" to 5, add evolution narrative |
| `research/05-kvcache-attention.v2.md` | v1 + 05-critique | Fix Qwen3 QK-norm ordering (PRE-RoPE), fix rope_theta, fix MiniCPM-3→MLA, fix MLA head-dim axis (q/k/v separate), add SSM `d_conv`/SSD axes, +12 attention + 6 RoPE + 6 cache variants, add evolution narrative |
| `docs/superpowers/specs/2026-06-04-llm-layers-design.v2.md` | v1 + 06-critique | Define SSMSpec + SSDSpec + ConvSpec + TokenMixerKind, expand op floor to ~16 (add conv1d, selective_scan, softmax, top_k, gather/scatter, layer_scale), fix AWQ claim, tighten M1 numerical tolerance, add weight-loader section, add glossary, fix 18 dataclass gaps, address 6 contradictions |
