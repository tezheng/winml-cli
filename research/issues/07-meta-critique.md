# 07 — Meta-Critique: Does the Artifact Set Meet the Survey-Paper Bar?

**Reviewer:** skeptical technical meta-reviewer
**Date:** 2026-06-04
**Scope:** Reports 01–05 + design spec `2026-06-04-llm-layers-design.md`
**Bar:** *a survey paper analyzing the evolution of LLM layers over the past 3 years*
**Verdict (early):** **No — not yet.** The artifact set is an excellent *engineering pre-design memo*. It is **not a survey paper.** It catalogs *what is*, but it does not narrate *how we got here*, *what was tried and abandoned*, or *where it is going*. The most fundamental requirement of a survey — an evolution narrative — is essentially absent. See §2 below.

What the set does well: technical depth, cross-source verification, evidence-grounded synthesis for the IR design task. What it fails at: timelining, framing, lineage, comparative tables that span the whole survey, and treating the reader as a researcher who needs to understand *change over time*.

---

## Section 1 — Cross-Doc Consistency Issues

A careful traceback surfaces **17 distinct inconsistencies** between the five research reports and the design spec. Sorted by severity:

### 1.1 Model count mismatch: census 45 vs layer-source 10 (severity: high)

- Report 01 claims **45 distinct shipped variants across 22 distinct architectures** (line 137 of `01-model-census.md`).
- Report 02 covers exactly **10 families** (Llama 3, Qwen 3, Gemma 3, Phi-3, Mistral, DeepSeek-V3, Mixtral/Qwen3-MoE, Granite, OLMo 2, Jamba).
- The spec (§3) calls 02 "10 families, HF/vLLM/llama.cpp triple-source".
- The gap of **35 unaudited variants** is never explained. Several Phase-B batches in the spec name models that have no layer-source coverage: **MiniCPM 3** (MLA + LongRoPE + μP), **DeepSeek-V2-Lite** (the only mainstream <8B MLA model), **Phi-4 mini** (GQA + LongRoPE + partial RoPE = 0.75 — a unique combination), **Granite 3.3**, **RWKV-6/-7**, **Zamba2**, **StarCoder 2**, **StableLM 2**, **InternLM 3**, **Yi 1.5**, **SmolLM2/3**, **OLMoE**, **Mamba 2.8B** itself (mentioned in census, listed in B7, never extracted in 02). The spec depends on these models for spec axes (MLA in B5; SSM in B7; Granite μP in B2) yet no layer-source for them has been produced.
- The MLA description in 02 is for **DeepSeek-V3**, but the SLM census's only MLA models are **DeepSeek-V2-Lite** and **MiniCPM-3**. V3 is 671B — *outside the SLM scope*. The layer-source report covers an out-of-scope model and skips the in-scope MLA examples.

### 1.2 AttentionSpec axis coverage vs research/05 (severity: medium-high)

Report 05 claims **19 attention-side axes** (11 in `AttentionSpec` + 8 in `RoPESpec`). Tracing each into the spec:

| Axis from research/05 | Present in spec? | Notes |
|---|---|---|
| `n_q_heads` | ✓ AttentionSpec | |
| `n_kv_heads` | ✓ | |
| `head_dim` | ✓ | |
| `kind` | ✓ but enum smaller — research/05 lists `{standard, mla, differential, retention, mamba, mamba2, linear}` (7), spec lists `{STANDARD, MLA, DIFFERENTIAL, LINEAR}` (4). Mamba/Mamba2/retention dropped. SSM has its own spec but `retention` is missing entirely. |
| `q_lora_rank` | ✓ | |
| `kv_lora_rank` | ✓ | |
| `qk_nope_head_dim` | ✓ | |
| `qk_rope_head_dim` | ✓ | |
| `v_dim` | ✓ as `v_head_dim` | |
| `n_q_streams` (differential) | **✗ MISSING** — spec only has `diff_lambda_init`. Differential transformer uses 2× Q projections; the spec has no field for the projection multiplier. |
| `lambda_init` | ✓ as `diff_lambda_init` | |
| `mask` / `mask_kind` | ✓ but values diverge: research/05 lists `{causal, sliding_window, sliding_with_sink, full, block_diag, custom}` (6), spec lists `{CAUSAL, SWA, SWA_GLOBAL_ALT, SINK, FULL, CUSTOM}` (6) — **`SWA_GLOBAL_ALT` invented in spec**, not in 05; **`block_diag` for packed seqs dropped**, even though research/05 §1.2 calls it out for vLLM continuous batching. |
| `window_size` | ✓ as `sliding_window` | |
| `n_sink_tokens` | ✓ | |
| `qk_norm` (placement) | ✓ as `qk_norm_phase` | |
| `qk_norm_axis` (per_head vs per_layer) | **✗ MISSING.** Spec has `qk_norm_shape ∈ {PER_HEAD_DH, FULL_HDH}` which conflates *shape* with *axis* (research/05 §1.7 explicitly distinguishes Qwen3 head-dim norm from OLMo full-channel norm as a *shape* difference, but research/05 §5.2 separately lists `qk_norm_axis ∈ {per_head, per_layer}` for the rare μP-style "per-layer scaling on Q only" case. The spec omits `per_layer`). |
| `attn_scale` | ✓ | |
| `logit_softcap` | ✓ | |
| `rope` | ✓ | |
| `shares_kv_with` | ✓ | |

**RoPESpec** (research/05 lists 8 fields):
| Axis | Present? | Notes |
|---|---|---|
| `basis` (interleaved vs split_half) | ✓ | |
| `base_theta` | ✓ | But: **research/05 §1.2 calls out Gemma 3 per-layer `rope_theta` (10K local, 1M global) as the cleanest counter-example to a single scalar per model**. The spec does not have a `base_theta_local` or per-layer rope-base override on `RoPESpec`. The spec assumes a single `base_theta`; for Gemma 3 it would need different `RoPESpec` instances per layer (workable but undocumented). |
| `d_rope` | ✓ as `rope_partial_dim` on AttentionSpec, not on RoPESpec — split between two specs |
| `position_axes` | ✓ as `mrope_section` | |
| `scaling` enum | ✓ — but the spec enum lists `{NONE, NTK, YARN, LLAMA3, LONGROPE}` (5), research/05 lists `{none, linear_pi, ntk, yarn, llama3, longrope}` (6); **`linear_pi` missing** from spec. Census uses `linear_pi` for Gemma 3 4B/12B/27B (factor 8) per `01-model-census.md` §4.2. |
| `scale_factor` | ✓ | |
| `attn_factor`, `beta_fast`, `beta_slow`, `original_max_pos` | ✓ via `YarnParams` | |
| `short_factor`, `long_factor`, `long_threshold` | ✓ via `LongRoPEParams` | |
| `low_freq_factor`, `high_freq_factor` | ✓ via `Llama3RoPEParams` | |
| `dynamic NTK` (InternLM) | **✗ MISSING.** Spec collapses NTK and dynamic NTK into one enum value. Research/01 §4.2 explicitly lists `Dynamic NTK` separately from `NTK`. InternLM 2.5/3 use dynamic NTK with runtime-computed `factor`. |
| `NoPE` (SmolLM3) | **✗ MISSING.** SmolLM3 has `no_rope_layers[]` mask per research/01 §2.6 — every 4th layer skips RoPE. The spec lacks a `rope=None` or per-layer NoPE concept; you would have to express this with two `DecoderBlockSpec` variants per layer-list slot. Workable but undocumented. |

**Conclusion:** the spec covers **≈16 of 19** attention axes claimed by research/05, with **3 missing** and **2 enum-value shrinkages**. For a survey, this gap is a citation-integrity issue. For an IR, it is an under-coverage bug.

### 1.3 QuantSpec vs the 25 schemes in research/04 (severity: medium)

Research/04 enumerates **25 schemes** (line 829: "Scheme count: 25 distinct schemes documented across five families"). The spec's `QuantSpec` claims to be derived from research/04 §10 (the spec says "research/04, §10" but research/04 actually distills the parameter space in §7, not §10 — *citation off by 3 sections*).

Trace of 25 schemes through QuantSpec:

| Family | Scheme | Covered by QuantSpec? |
|---|---|---|
| W-only INT/FP | AWQ | ✓ |
| | GPTQ (incl. `act_order=True`) | ✓ but no `pre_transform = act_order_permute` value listed |
| | HQQ | ✓ |
| | bitsandbytes NF4 | ✓ — `codebook` field present |
| | bitsandbytes FP4 (E2M1 non-OCP) | ✗ FP4 enum value listed as `FP4` but research/04 §1.4 distinguishes "bnb FP4" from OCP `MXFP4` and Blackwell `NVFP4` — three different FP4 formats. Spec enum has only `FP4`, `MX_FP4`, `NVFP4` — bnb's FP4 (which differs in codepoints and grouping) maps to neither cleanly. |
| | LLM.int8() | ✗ MISSING — no `outlier_decomposition` axis. Research/04 §1.4 calls out the outlier-decomposition as a unique runtime behavior. |
| | MXFP4 | ✓ |
| | MXFP6 (E3M2, E2M3) | ✗ MISSING — spec has no FP6 dtype |
| | MXFP8 | ✗ MISSING from QDType enum visible in spec excerpt |
| | MXINT8 | ✗ MISSING |
| | NVFP4 | ✓ |
| W+A | SmoothQuant W8A8 | ✓ (`pre_transform=channelwise_smooth`) |
| | INT8 W8A8 dynamic | ✓ |
| | INT8 W8A8 static | ✓ |
| | W4A4 (QuaRot, SpinQuant) | partial — `pre_transform=hadamard` exists in research/04 §7 but the spec excerpt does not show the `PreTransform` enum members |
| | FP8 E4M3 | ✓ |
| | FP8 E5M2 | ✓ |
| | FP8 E4M3FNUZ (AMD) | ✗ MISSING. Research/04 §5.3 spends paragraphs on the 4 distinct FP8 codes that are *not* byte-compatible across NVIDIA/AMD; QuantSpec enum collapses to 2. |
| | FP8 E5M2FNUZ (AMD) | ✗ MISSING |
| | W4A8 AWQ | ✓ |
| | W4A8 QServe | ✓ |
| GGUF | Q2_K | ? (spec's QDType enum shown is incomplete — research/04 §3.1 lists Q2_K through Q8_K; spec's enum sketch shows only `INT4, INT8, FP8_…, FP4, NF4, MX_FP4` — **GGUF k-quants are absent from the enum shown.** The spec text *names* GGUF Q4_K_M as "support only 3" but the QDType excerpt does not include `q*_k` variants.) |
| | Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | same — not in enum excerpt |
| | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_S, IQ3_M, IQ4_XS, IQ4_NL | i-quants are 11 distinct schemes — not in the QDType enum excerpt. Spec mentions `codebook` axis covers IQ; but `qdtype=iq_codebook` is one value vs 11 lattice variants. |
| KV cache | INT8 KV | ✓ via `kv_dtype` |
| | INT4 KV (KIVI, per-channel K, per-token V) | ✓ via independent k_quant / v_quant |
| | FP8 KV | ✓ |
| Attention internal | FA3 FP8 SDPA | ✗ MISSING. Research/04 §5.1 introduces `attn_qk_qdtype`, `attn_pv_qdtype`, `online_rescale=True` axes. The spec's QuantSpec has `role=ATTN_INTERNAL` but the data model does not carry the FA3-specific online-rescale flag, nor does AttentionSpec carry quant fields. |

**Estimated coverage: 16 of 25 schemes representable, 9 missing or under-modeled.** This is a meaningful gap and the spec does not flag it.

### 1.4 IHV op-sets to 9 logical ops: union, intersection, or vote? (severity: medium)

Research/03 §11 says "RMSNorm — explicit op in 7/9", "RoPE — 6/9 explicit", "SiLU — 8/9 explicit", "SDPA as a single fused op — 9/9". The spec §5.1 lists 9 logical ops and labels them "≥7/9 consensus".

But the count doesn't quite match research/03. Tracing:
- `rms_norm`: 7/9 explicit per research/03, 9/9 if you accept "composed" as having an op. Spec says "9/9". **Inconsistent** if you read research/03's table strictly; reasonable if you read the §11 prose which says "RMSNorm is so ubiquitous it must be a first-class API".
- `rope_apply`: 6/9 explicit per research/03, 9/9 in spec. The spec re-promotes a 6/9-vote op to "consensus floor".
- `silu`: 8/9 in research/03, 9/9 in spec.
- `embed(ids, weight, scale?)`: this op is listed as "9/9" in spec but research/03 does not explicitly survey `Gather`/`Embedding` ops — there is no row in the §10 cross-runtime synthesis table for it. The 9/9 claim is unsupported.
- `lm_head` is just `linear` — listing it as a separate logical op is convenient for the API but inflates the count. Research/03 lists it as `MatMul`/`LinearOperation` in the synthesis table — same as `linear`.

Bottom line: **the 9 logical ops are a weighted-vote selection, not a strict intersection or union, and the rationale ("consensus floor") is asserted rather than derived.** A survey would either (a) make the voting algorithm explicit ("an op is in the floor if ≥6/9 runtimes expose it natively") or (b) admit it is curated. Currently it does neither.

### 1.5 Model-name normalization (severity: low but pervasive)

Spelling drift across the five reports:
- "Llama 3" / "Llama-3" / "Llama3" — all three occur (e.g., `01` mostly uses "Llama 3.x", `02` uses "Llama 3", spec uses "Llama-3" and "Llama3").
- "DeepSeek-V2-Lite" / "DeepSeek V2 Lite" / "DeepSeek-V2 Lite" / "DeepSeek-Coder-V2-Lite" — the same model is rendered four ways across 01/02/05.
- "Phi-3 mini" / "Phi-3-mini" / "Phi3mini" / "Phi-3-mini-4k" — `01` is the most rigorous (suffixes like "mini-4k"); `02` collapses to "Phi-3"; `05` says "Phi-3-mini-128k".
- "Gemma 3" / "Gemma-3" — mostly "Gemma 3" but spec sometimes drops the space.
- "Qwen 3" / "Qwen3" / "Qwen-3" — `01` says "Qwen 3"; `02` says "Qwen3"; `05` says "Qwen3"; spec says "Qwen3".
- "DeepSeek-V3" appears in 02 — but it is **out of scope** (671B) for the <8B census. This is a scope leak.

A survey paper would have a normalized glossary or model-naming convention table up front.

### 1.6 Spec citations point to wrong sections (severity: low)

Spec §12 says:
- "AttentionSpec — ~19 axes (research/05, §11)" — research/05 §11 doesn't exist; the AttentionSpec table is in §5.2. Off by 6 sections.
- "KVCacheSpec — 8 axes (research/05, §10)" — also §5.1, off by 5 sections.
- "QuantSpec — 7 minimal + 8 extended axes (research/04, §10)" — research/04 §10 is *References*, not the QuantSpec distillation; the parameter space lives in §7. Off by 3 sections.

These are small but symptomatic — citations weren't validated.

### 1.7 Number of attention variants disagrees

- Research/05 §5.3 says "Attention variants documented: **13** kinds".
- Research/01 §4.1 lists 8 in its attention-type table (MHA, GQA, MQA, MLA, SWA, block-sparse, linear/SSM, hybrid).
- Spec's `AttentionKind` has 4 values (`STANDARD, MLA, DIFFERENTIAL, LINEAR`).

The same conceptual quantity ("number of attention variants") has three different counts (13, 8, 4). A survey paper must reconcile these.

### 1.8 Number of axes claimed for `AttentionSpec`

- Research/05 §5.3: "11 fields + nested RoPESpec of 8 fields = 19 axes".
- Spec §5.2.1 says "~19 axes" but the dataclass shown has ~22 fields counted directly (including MLA, biases, diff_lambda_init, sliding_window, sink, qk_norm trio, rope_partial_dim, shares_kv_with). Discrepancy of 3.

### 1.9 Coverage of cache layouts in spec

- Research/05 §3 enumerates 6 cache models: contiguous, paged, ring/SWA, per-layer-separate-vs-unified (essentially a layout flag, not a cache kind), stateful, hybrid SSM/attn.
- Research/05 §5.1 reduces to **storage ∈ {contiguous, paged, ring_swa}** (3 storage kinds).
- Spec's `CacheLayout` has 5 (`CONTIGUOUS, PAGED, RING, MLA_LATENT, SSM_STATE`).
- These three lists are pairwise non-isomorphic. MLA_LATENT and SSM_STATE in the spec are cache *kinds* not *storage kinds*, but stateful (a *ownership* concern) is separately given its own `CacheOwnership` enum. Stateful per Core ML is half a layout, half a lifecycle. **The taxonomy is not stable across docs.**

### 1.10 KIVI K/V independence

- Research/04 §4.4 and research/05 §3.6 both state KIVI quantizes K per-channel, V per-token.
- Spec correctly carries independent `k_quant` and `v_quant`. ✓ One of the cleanest cross-doc agreements.

### 1.11 Granite multiplier set

- Research/01 §4.7 lists 6 multipliers: `embedding_multiplier`, `attention_multiplier`, `residual_multiplier`, `logits_scaling`, `query_pre_attn_scalar`, `attn_logit_softcapping`.
- Spec §5.4 carries 3 (`residual_scale`, `embedding_scale`, `logits_scale`). `attention_multiplier` is implicit in `attn_scale` on AttentionSpec. `query_pre_attn_scalar` is implicit in `attn_scale`. `attn_logit_softcapping` is `logit_softcap`.

So spec covers 5 of 6 — but doesn't say so explicitly. **`final_logit_softcapping` (Gemma) and `attn_logit_softcapping` (Gemma) are different operations** (one on final LM logits, one inside attention); the spec collapses both under `logit_softcap`. A reader would need to dig into the field semantics to know.

### 1.12 Per-layer RoPE base for Gemma 3

Research/05 §1.2 and §2.1: "Gemma 3 has per-layer `rope_theta` (10K local / 1M global)".
Research/01 §2.3, §4.2, §5.3: same observation.
Spec §5.2.2 RoPESpec carries one `base_theta`. The spec text in §5.4 (DecoderBlockSpec) implies each layer has its own RoPESpec via the `token_mixer` field's nested `AttentionSpec.rope`, so **Gemma 3 lowers correctly** by giving each layer its own RoPESpec — but this is not documented as the intended pattern. A reader has to derive it.

### 1.13 Hybrid SSM block kind

- Research/05 §1.3 lists block-level types: `{attention, mamba, mamba2, linear, identity}` — 5 kinds.
- Research/01 §2.8 lists token-mixer kinds: `Selective SSM (S6), Mamba2, RWKV WKV6, Mamba1 + 1-in-8 attention` — different granularity.
- Spec §5.4 `token_mixer: Union[AttentionSpec, SSMSpec]` — only 2 kinds. RWKV is not surfaced. The spec doesn't define `SSMSpec` in the excerpt (referenced in §5.4 but not specified in §5.2). **Missing dataclass.**

### 1.14 Vision-language model treatment

Research/01 §2.9 covers VLM text towers. The Gemma 3, Phi-3.5-vision, and PaliGemma text towers are listed *separately* from the dense decoder versions, even though `01` says they are architecturally identical to the corresponding text-only models. This is a redundant but harmless redundancy. **However**, no other report acknowledges VLMs. Reports 02, 03, 04, 05 read as if all models are text-only. For a survey, a one-paragraph acknowledgment of "the text towers of all surveyed VLMs are GQA/RoPE decoders identical to their text siblings, modulo embedding-table expansion for image tokens" would be standard.

### 1.15 MTP heads / speculative decoding

Research/05 §4.3–4.4 discusses speculative decoding compatibility (Medusa, EAGLE-2, tree masks). DeepSeek-V3's **Multi-Token Prediction (MTP)** heads are not mentioned anywhere. The user's brief explicitly flags MTP as a likely gap. The spec also drops it.

### 1.16 Per-runtime quantization support cross-check

Research/03 §11 says "**W4A16 grouped INT4** — universal. All 9/9 runtimes". Research/04 §1.1 (AWQ) lists runtime support as: AutoAWQ, vLLM, TRT-LLM, HF Transformers, MLX, llama.cpp via GGUF re-quant — 6 of 9 IHV runtimes from research/03 (no QNN, no Core ML, no MIGraphX, no MindIE, no KleidiAI explicit). The "9/9" claim in research/03 §11 conflates "supports some W4A16" with "supports AWQ-specific". A survey needs to be precise about what is and isn't supported.

### 1.17 Tokenizer treatment

Research/01 §4.6 has a tokenizer/vocab section. No other report discusses tokenization. The spec lists tokenization as out of scope. This is reasonable for an IR, **not** reasonable for a survey paper — tokenization evolution (sentencepiece → tiktoken → o200k → custom BPE; 32k → 256k vocab growth) is a major axis of LLM evolution.

**Total inconsistency count: ≈17 distinct issues, varying severity.**

---

## Section 2 — Evolution Narrative Deficit (the survey-paper-bar problem)

This is the single biggest deficiency. **None of the five reports tells the story of evolution.** Each is a snapshot of *what is true today*.

### 2.1 No timeline anywhere

The user's brief explicitly asked for a timeline like "GPT-NeoX 2022 → Llama 1 2023 → … → Gemma 3 March 2025". Checking the reports:

- Report 01 has **release dates per row of the census table** (good) but no aggregated timeline figure, no per-axis evolution chart ("RoPE base θ over time: 10k → 500k → 1M → 5M").
- Report 02 has zero dates.
- Report 03 has zero dates apart from opset versions ("opset13 SDPA").
- Report 04 has paper dates inline (e.g., "AWQ — MLSys 2024") but no time-ordered story.
- Report 05 has paper dates inline.

A survey paper would have:
- A timeline figure showing major architecture decisions on a horizontal axis.
- A per-axis evolution table: for each axis (attention type, RoPE base, norm placement, FFN family, vocab size, …), list the year and the dominant value adopted.
- Identification of "inflection points" where the community shifted (e.g., July 2023: GQA replaces MHA in mainstream models; Sept 2023: SWA introduced by Mistral; April 2024: MLA by DeepSeek-V2; July 2024: Llama-3 RoPE scaling formalized).

### 2.2 No architecture-family lineage diagram

A survey paper on transformer architecture evolution should have a *lineage diagram* showing which models descend from which. For example:
- Llama 2 → Llama 3 → Llama 3.1 → Llama 3.2 → Llama 4
- Qwen 1 → Qwen 1.5 → Qwen 2 → Qwen 2.5 → Qwen 3 → Qwen 3 MoE
- GPT-NeoX → Phi-1 → Phi-2 → Phi-3 → Phi-3.5 → Phi-4 → Phi-4-mini
- Mistral 7B v0.1 (SWA) → Mistral 7B v0.2 (no SWA) → Mistral 7B v0.3 → Mistral-Nemo (SWA back)
- Gemma 1 → Gemma 2 (sandwich norm, softcap) → Gemma 3 (5:1 SWA:full, dual RoPE, no softcap)
- DeepSeek-V2 → V2-Lite → V3 (auxiliary-free routing, MTP) → R1 distills

The point of the diagram is to show that some lineages *converged* (everyone uses GQA + RoPE + SwiGLU + RMSNorm by 2024), and some *diverged* (Gemma 2's softcap was *abandoned* in Gemma 3; Mistral v0.1's SWA was *abandoned* in v0.2 and then *re-adopted* in Nemo).

None of the reports has this.

### 2.3 No "what was tried and abandoned" section

The user's brief explicitly lists candidates: parallel attn+FFN, pure SWA without alternation, pure MQA, ALiBi. Let me check each:

- **Parallel attention+FFN** (GPT-J, PaLM, Falcon): mentioned **nowhere** in the five reports. This is the most consequential abandoned architecture choice of 2023.
- **Pure SWA without alternation** (Mistral v0.1): mentioned in research/05 §1.2 only as "Mistral-7B-v0.1 (W=4096)" — not flagged as abandoned despite Mistral v0.2 dropping it.
- **Pure MQA**: mentioned in research/01 and 05 — but as a current state, not as the once-popular-now-mostly-abandoned choice (Falcon, StarCoder, PaLM). The fact that **MQA largely lost to GQA** is the key narrative point and is not made.
- **ALiBi** (Press et al. 2021, MPT-7B, BLOOM, Falcon-180B): research/05 §1.8 says "**Not used** by any current top-tier SLM (Llama-3, Qwen3, Phi, Gemma all use RoPE)". This sentence is the *only* abandoned-architecture statement across all five reports. Good but isolated.

Other abandoned/superseded choices not mentioned:
- **LayerNorm** is being replaced by **RMSNorm**. Phi-3-small and StarCoder 2 still ship LayerNorm; this is becoming rare. Not framed as a transition.
- **Pre-LN** vs **DeepNorm** vs **Post-LN** vs **sandwich norm**: research/02 covers OLMo 2 (post-norm) and Gemma 2/3 (sandwich) but doesn't survey the literature on Post-LN's training-stability advantage.
- **Absolute learned position embeddings** (BERT, GPT-2): completely absent.
- **NTK-aware** RoPE scaling (early 2023, Together's LLongMA, CodeLlama-16k): research/05 §2.2 says "almost entirely superseded by YaRN / Llama-3 scaling in 2024+" — good, but isolated.
- **Outlier-aware INT8** (LLM.int8 with row decomposition): research/04 §1.4 mentions but doesn't frame as superseded by SmoothQuant + FP8.
- **Q4_0, Q4_1, Q5_0** (legacy llama.cpp linear-quant): not mentioned; superseded by k-quants and i-quants. A survey on quantization evolution would say so.
- **Multi-query attention** (Shazeer 2019): once promising, mostly displaced by GQA. Only sub-2B models keep it.

### 2.4 No "what every modern SLM agrees on vs fights over" comparison

The closest is research/01 §1 (executive summary) which says "while SwiGLU, RMSNorm pre-norm, RoPE, and GQA dominate the dense decoder space, every single design slot has at least two competing variants in production". This is one paragraph. A survey would have this as a centerpiece table:

| Design slot | Consensus (post-2024) | Active variation |
|---|---|---|
| Attention | GQA | MHA (Phi-3 mini, OLMo 2), MQA (Gemma 3 1B, CodeGemma), MLA (DeepSeek, MiniCPM 3) |
| Positional | RoPE | YaRN, Llama-3, LongRoPE, dual-base per-layer (Gemma 3), partial, NoPE (SmolLM3) |
| Norm | RMSNorm | LayerNorm (Phi-3-small, StarCoder 2) |
| Placement | Pre-norm | Post-norm (OLMo 2), sandwich (Gemma 2/3) |
| FFN | SwiGLU | GeGLU (Gemma), fused gate-up (Phi-3), gegelu (Phi-3-small), gated GELU w/ bias (StarCoder 2) |
| KV cache | Per-layer separate, FP16 | Paged (vLLM), ring (llama.cpp SWA), stateful (Core ML), MLA-latent (DeepSeek) |
| Quant | W4A16 grouped int4 | Q4_K_M (GGUF), FP8, NVFP4/MXFP4 |
| Sliding window | (none in Llama family) | All layers (Mistral v0.1), alternating (Gemma 2), 5:1 (Gemma 3), block-sparse (Phi-3-small) |

This table would let a reader take in the entire survey in 30 seconds. It does not exist anywhere.

### 2.5 Conclusion of §2

The set has **zero evolution narrative.** It is a snapshot collection. Without a timeline, lineage, abandonment-discussion, and consensus-vs-divergence table, the artifact fails the survey-paper bar. This is fixable but is the single largest revision required.

---

## Section 3 — Coverage Completeness Gaps

### 3.1 Models / families missing

Beyond the layer-source/census gap (§1.1), entire model families have zero coverage:
- **Apple OpenELM** — appears once in research/05 as a "YOCO/CLA reference" — no architectural treatment.
- **Apple Intelligence on-device 3B** (Apple silicon flagship SLM) — completely absent. Census names "Apple" zero times.
- **Microsoft Phi-Silica** (the Copilot+ PC NPU model, ~3B, GQA, NPU-quantized) — research/05 mentions briefly under "some Phi-Silica" in MQA row; never analyzed.
- **Falcon 3 / Falcon Mamba** — research/05 §1.3 mentions Falcon-Mamba; no architectural treatment.
- **Hymba** (NVIDIA hybrid SSM, 1.5B) — absent.
- **Bamba** (IBM hybrid) — absent.
- **Llama 4 / Llama 4 Scout / Llama 4 Maverick** (if released ≤ June 2026) — absent. Census stops at Llama 3.2.
- **DeepSeek-V3-0324, DeepSeek-V3.1** (March 2025 update) — absent. Census stops at V2-Lite.
- **Qwen 2.5-Math / Qwen 2.5-VL** text towers — absent from VLM section (only LLaVA-Mistral, PaliGemma, Florence-2, Phi-3.5-vision are listed).
- **GLM-4 / GLM-Edge** — absent. The Chinese GLM family ships an SLM under 8B with custom RoPE.
- **Yi-Coder / Yi-1.5-VL** — absent.
- **InternLM-XComposer / -Math** — absent.
- **AI21 Jamba 1.5 Mini** — research/01 lists Jamba dev card only; 1.5 Mini production is different.
- **NVIDIA Nemotron-Mini-4B-Instruct** — absent.
- **Cohere Command-R7B / Command-A** — research/05 mentions Cohere Command-R7B for "1:3 SWA pattern" once; never analyzed.
- **HuggingFaceTB SmolVLM** — absent.

For a survey claiming "every shipped SLM", the list of missing-but-relevant families is uncomfortably long. **Estimated coverage of currently-released ≤8B models: ~60–70%.**

### 3.2 Hardware runtimes missing

Research/03 covers 9 runtimes. Missing from the IHV survey:
- **AMD ROCm via `vllm-rocm`** (separate from MIGraphX; uses the same ORT contrib path).
- **Intel oneDNN / XPU PyTorch** (separate from OpenVINO; the PyTorch-eager path on Arc / Lunar Lake).
- **Apache TVM / MLC LLM** (a major runtime for diverse hardware).
- **TFLite / LiteRT for LLM** (Google AI Edge for Android NPU).
- **Hailo / DeepX / Tenstorrent** — emerging NPUs with their own op sets; relevance for SLM deployment growing.
- **Hugging Face TGI** as a quant-shipping reference (research/04 mentions TGI in passing; research/03 doesn't list it).
- **vLLM itself** — research/03 lists 9 vendor runtimes but **vLLM (the most influential serving runtime) is not in the synthesis table**. It is mentioned everywhere else but never gets a §x entry. This is a major structural gap because vLLM's op selection drives every other runtime's contrib-op set.

### 3.3 Quantization schemes missing

Beyond §1.3 above, research/04 doesn't cover:
- **AQLM (Additive Quantization)** — codebook-based, 2-bit, used in some HF Transformers releases.
- **QuIP** / **QuIP#** — beyond i-quants context.
- **PV-Tuning / SqueezeLLM** — older but still in benchmarks.
- **Mixed-precision per-channel** as a first-class scheme (e.g., AutoQuantize, Higgs).
- **Per-token activation outlier-detection** (separate from LLM.int8) — used in some Atom variants.
- **Sparsity + quantization** combinations (Wanda, SparseGPT) — only mentioned once.

### 3.4 Cross-modal coverage

VLM text towers are covered only in 01 §2.9 (a table). No analysis of:
- Cross-attention placement in encoder-decoder VLMs.
- How vision-token routing interacts with KV cache (Qwen2-VL's M-RoPE).
- Whether the "VLM text tower is just X with adapters" claim holds (it does for Qwen2-VL, Gemma 3, PaliGemma; it doesn't for Florence-2 which is BART-style).

A survey would have a §X "vision-language and multimodal extensions" section.

### 3.5 Reasoning-specific architectures

The user's brief flags DeepSeek-R1-distill and MTP. The reports:
- Research/01 §2.11 lists 3 R1 distills (Qwen-1.5B, Qwen-7B, Llama-8B). Notes correctly that distillation doesn't change architecture.
- **MTP (Multi-Token Prediction)** heads: never mentioned. DeepSeek-V3 ships an MTP head and so does some Qwen3 variants; this is *the* training-time architectural innovation of 2024–2025 for reasoning models.
- **Speculative-decoding heads** (Medusa, EAGLE) — research/05 §4.3 covers compatibility but not architecture impact.

---

## Section 4 — Depth / Quality Mismatches

### 4.1 Word counts and depth

User's brief gives word counts: 01=6.4k, 02=8.9k, 03=6.2k, 04=5.3k, 05=6.2k.

- **Report 02 is the deepest** (8.9k) and the most evidence-rich (HF + vLLM + llama.cpp line references). It is the model of what the others should be.
- **Report 04 (quant) is the shallowest** at 5.3k. But research/04 covers 25 schemes — *more* schemes than any other report has variants. The result is each scheme gets ~210 words. Some schemes (AWQ, GGUF k-quants, NVFP4) are well-developed; others (HQQ, GPTQ act_order, IQ family) are sketched. Compared to research/05's 6.2k for ~19 attention axes (~325 words/axis) the per-item depth gap is real.

A balanced survey would have:
- Report 01: enumeration, fine at 6.4k.
- Report 02: depth, 8.9k is right.
- Report 03: synthesis, 6.2k feels under for 9 runtimes. The MindIE/CANN section is openly thin ("Public details are limited; most authoritative documentation is in Mandarin"); FastFlowLM is admittedly proprietary; QNN op enumeration is partial. **The report is honest about its incompleteness but does not flag this as an issue requiring follow-up.**
- Report 04: should be 8–10k to cover 25 schemes adequately. **Underweighted.**
- Report 05: 6.2k is reasonable for 13×8×6 = 624 logical variants surveyed by axis, but the depth on per-runtime *implementation* of each variant is light. The "Compatibility matrix" §5.4 is the gem of the report and could be the spine of a survey paper.

### 4.2 Source-citation depth

- Report 02 cites HF transformers `model_*.py` files with line numbers. Top-tier.
- Report 03 cites IHV documentation URLs and source files. Strong.
- Report 04 cites papers (arXiv IDs) and source files. Strong.
- Report 05 cites papers and code pointers in an appendix. Strong.
- Report 01 cites HuggingFace model cards (`raw/main/config.json` URLs) and a tech report per family. **Strong but uneven**: some claims (e.g., "Phi-3-small uses gegelu") are direct from config, others (e.g., "MQA is alive — at the smallest sizes") are conclusions without citation.

The biggest citation gap is **primary papers**:
- Research/01 §6 cites Llama 3 paper (arXiv:2407.21783), Qwen 3 tech report, Gemma 2/3 tech reports — but does *not* cite the papers for the model-specific innovations. E.g., the Granite μP technique cites no paper; the SmolLM3 NoPE pattern cites no paper. (Both papers exist.)
- Research/02 has zero paper citations. It is a code-survey, which is internally consistent but, for a survey paper, primary sources should be the first citation per innovation.

### 4.3 Equations / formulas

A survey paper on LLM evolution would have equations for:
- RoPE: the rotation formula.
- YaRN: the three-piece interpolation, the temperature term.
- Llama-3 RoPE scaling: the wavelen condition.
- LongRoPE: the per-dim factor application.
- MLA: the absorption trick (`(Q · W_uk^T) · c_kv^T`).
- Softcap: the `cap · tanh(logits / cap)` formula.
- SwiGLU: `down(silu(gate(x)) * up(x))`.
- GeGLU: same with gelu.
- RMSNorm: the formula.
- KIVI per-channel vs per-token: the math of why K wants per-channel.

Research/05 has **3 inline equations** (MLA absorption hint at §1.1, RoPE freq formula at §2.1, Llama-3 piecewise at §2.4). All others use only prose or pseudocode. **An academic survey would have 30+ equations.**

### 4.4 Figures / diagrams

Zero figures across all five reports. Tables yes, ASCII diagrams in research/03 §0 (the decoder block pseudo-code), no actual figures. For a survey paper this is a major lack. Needed:
- Decoder block architecture diagram per family (or one parameterized diagram with overlays).
- Timeline figure.
- Lineage diagram.
- KV cache layout diagrams (contiguous vs paged vs ring vs MLA-latent).
- Attention mask shape illustrations (full, SWA, sink, block-sparse, tree).
- Quantization layout illustrations (block layout, super-block, NVFP4 two-level scale).

---

## Section 5 — Narrative Flow Issues

### 5.1 No introduction across the artifact

Each report has its own intro. There is no *meta-introduction* across all five that gives the reader the big-picture evolution story. The spec §1 is the closest but is design-focused, not survey-focused.

### 5.2 No section conclusions

Most sections end mid-table or mid-list. Research/05 §11 has an "Implications for `llm-layers`" — design-focused, not survey-paper conclusions. Research/04 §11 says "If you support only three, support these three" — again design-focused.

A survey paper would conclude each section with:
- The 3–5 most important findings.
- Citations to the most influential papers in that subarea.
- Open research questions in that subarea.

Research/04 §9 has open questions (good). Research/05 has none. Research/01, 02, 03 have none.

### 5.3 No future-directions section

Standard in surveys. Where is the answer to "what should the field do next"? The spec answers "what should `llm-layers` do next" but that is internal. Open research questions belong in the *reports*, not the design spec.

### 5.4 No one-page summary table

The closest is research/05's compatibility matrix (§5.4). That covers attention × RoPE × cache but not norm placement, FFN family, MoE specifics, quant scheme support. A true one-page survey summary would be a single grid: rows = models, columns = (Attn, Pos, Norm, FFN, MoE, KV cache, Quant). Research/01 §2 comes close per-family but spans 6 sub-tables; a unified column layout across all families is missing.

---

## Section 6 — Citation Quality

### 6.1 Spot-check of 10 claims

| # | Claim | Source location | Citation present? | Primary or secondary? |
|---|---|---|---|---|
| 1 | "Qwen 3 uses QK-RMSNorm on `head_dim` only" | 01 §3, 05 §1.7 | ✓ HF source `modeling_qwen3.py:248` | Primary (code) |
| 2 | "OLMo 2 uses QK-RMSNorm on full head channels" | 01 §3, 05 §1.7 | ✓ HF source `modeling_olmo2.py:231-232` | Primary (code) |
| 3 | "Gemma 3 has per-layer rope_theta" | 01 §5, 05 §1.2 | ✓ HF source `configuration_gemma3.py:105-116`; tech report claim | Primary (code) |
| 4 | "DeepSeek-V2-Lite uses MLA with `kv_lora_rank=512`" | 01 §2.7, 05 §1.1 | ✓ DeepSeek-V2 paper arXiv:2405.04434 (research/05 §0) | Primary (paper) |
| 5 | "AWQ uses fp16 zero, GPTQ uses int zero" | 04 §1.1 vs §1.2 | ✓ AWQ MLSys 2024 / GPTQ ICLR 2023 | Primary |
| 6 | "FastFlowLM kernels are proprietary binaries (IRON / AIE-MLIR)" | 03 §3a | ✓ `https://fastflowlm.com/how-it-works/` | Secondary (vendor blog) |
| 7 | "Genie's segment-KV-cache SDPA kernel" | 03 §2 | ✓ QNN doc URL | Secondary (vendor doc) |
| 8 | "vLLM 0.6 added MLA paged support" | 05 §5.4 | ✗ No explicit citation; "vLLM (vLLM 0.6+)" only | Implied |
| 9 | "NVFP4 uses fp8 E4M3 inner scale + fp32 outer per-tensor scale" | 04 §1.6 | ✓ NVIDIA Blackwell whitepaper | Primary (whitepaper) |
| 10 | "Cohere Command-R7B has 1:3 SWA pattern" | 05 §1.2 | ✗ No citation | Implied |

**8/10 cited, 2/10 uncited**. Average is good. The uncited ones are claims that would be easy to back up. For a survey paper, every claim should have a citation.

### 6.2 Citation format consistency

- Research/04 cites arXiv IDs and paper titles inline. Consistent.
- Research/05 cites papers in §0 then refers by name. Consistent.
- Research/03 cites URLs (docs, repos). Consistent within report.
- Research/01 cites HF Hub URLs to config.json. Consistent within report.
- Research/02 cites file paths to local repos. Consistent within report.

**Across reports: no unified format.** A survey paper would use a single bibliography file with [Author Year] or [number] citations. Currently citations are scattered inline by URL or arXiv ID, no `References` section consolidates them. Each report has its own mini-bibliography (sometimes labeled `§ References`, sometimes `§ Source citations`, sometimes none).

### 6.3 Primary vs secondary sources

- Research/02, 04, 05 are mostly primary (papers, code).
- Research/01 is primary (configs, tech reports) for most claims; some prose conclusions ("MQA is alive — at the smallest sizes") are interpretive.
- Research/03 leans secondary for Huawei (CSDN, Zhihu, Tencent blogs cited because "primary doc is gated behind Huawei developer portal and is Mandarin"). This is honest but flagged as a known limitation that should be resolved before survey publication.

---

## Section 7 — Audience / Writing-Quality Issues

### 7.1 Technical-term definitions

For a survey-paper audience (researchers, but possibly cross-discipline):
- "MLA" is first defined in research/05 §1.1 with the full math. Research/01 §2.7 and §4.1 use "MLA" without redefinition. **Acceptable** because research/05 is meant to be read first per the design spec — but the *order* in which to read the reports is undocumented.
- "GQA" is defined in research/05 §1.1. Research/01 §4.1 uses it. Same as above.
- "MQA" — same.
- "SwiGLU" is defined nowhere explicitly. Just used.
- "RoPE" is defined in research/05 §2.1. Other reports assume the term.
- "RMSNorm" never formally defined (`x / RMS(x) * gamma`).
- "GeGLU", "gegelu", "FastGelu", "QuickGelu" — names used, never math.
- "imatrix" — defined inline in research/04 §3.
- "ScaledDotProductAttention" / "SDPA" — used universally, not defined.
- "Pre-norm" / "Post-norm" / "sandwich norm" / "DeepNorm" — used; not defined in one place.
- "NoPE" — used in research/01 §2.6; not defined ("no positional embedding").

A survey paper needs a glossary or first-use definitions throughout.

### 7.2 Audience tone

- Research/02 reads like an engineering memo with code excerpts. Excellent for an engineer.
- Research/03 reads like an op-survey. Excellent for an IR designer.
- Research/04 reads like a paper survey + code archaeology. Closest to "survey paper" register.
- Research/05 reads like a research review with API implications. Strong.
- Research/01 reads like a config-archaeology census. Useful but list-heavy.

**For a survey audience**, research/02 and 03 are tonally too engineering-focused. The reader is led directly into op names rather than the conceptual point. A survey would lead with "what changed across the last 3 years in attention" and then *cite* the runtime evidence.

---

## Section 8 — Missing Reports

The user's brief lists 5 candidates. Assessment:

| # | Proposed report | Should exist? | Why |
|---|---|---|---|
| 06 | Tokenization evolution (sentencepiece → tiktoken → o200k → custom BPE; vocab 32k → 256k) | **Yes — high priority** | Tokenization is a first-class axis of LLM evolution. Research/01 §4.6 sketches it but doesn't analyze evolution (32k Llama-1/2 SP → 128k Llama-3 LL3-tiktoken → 200k Phi-4 o200k → 256k Gemma-3 GP-v3 → 262k+ for image tokens). |
| 07 | Data / training-data trends | **No — out of scope** | Not architecture. Belongs in a separate "training data" survey. |
| 08 | RoPE-base scaling for long context (deep dive) | **Optional — useful** | Research/05 §2 covers RoPE variants in 6.2k of mixed-topic content. A standalone "long-context history" report (NTK → YaRN → Llama-3 → LongRoPE → dual-base) with empirical perplexity comparisons would be a strong survey chapter, but may overlap too much with research/05. |
| 09 | Layer-norm placement / training stability research (Pre-LN vs Post-LN vs DeepNorm vs sandwich) | **Yes — medium priority** | The reports treat norm placement as a per-model axis. A survey of *why* OLMo went post, *why* Gemma went sandwich, *why* DeepNorm exists, *why* Pre-LN won early but is being revisited — would be a strong chapter and is missing. |
| 10 | Open weights licensing / availability per family | **No — out of scope** | Important for the field but not architecture evolution. |

Additionally proposed by this reviewer:
| # | Report | Why |
|---|---|---|
| 11 | Speculative decoding & MTP-head architectures | The training-time architectural innovation of 2024–2025 most absent from the current set. |
| 12 | Sliding-window history (Longformer 2020 → Sparse Transformer 2019 → Mistral SWA 2023 → Gemma 2 alternation 2024 → Gemma 3 5:1 2025 → block-sparse 2024) | SWA is one of the most-evolved choices and is scattered across 01, 02, 05. |
| 13 | Vision-Language extensions to text towers (M-RoPE, image-token vocabulary growth, cross-attention placement) | VLMs are increasingly the "real" deployment target; currently a one-table afterthought. |
| 14 | MoE routing evolution (top-k softmax → top-k softmax with shared experts → sigmoid + bias / auxiliary-free; group routing; expert specialization) | MoE is covered fragmentarily across 01, 02, spec; deserves a focused report. |
| 15 | Training-time tricks (μP, scale_emb, scale_depth, residual_multiplier, attention_multiplier, QK-norm as stability device) | Currently scattered as "Granite/MiniCPM quirks". Was a major 2023–2025 axis. |

So 5 of the user-proposed reports yield 3 new reports, plus 5 reviewer-added, for a total of **~8 additional reports** to reach survey-paper coverage.

---

## Section 9 — Recommended Structural Fixes for v2

Ranked by impact:

1. **Add a `research/00-evolution.md` "introductory survey + timeline"** report. This is the missing spine. Contents:
   - Timeline figure with horizontal axis 2022→2026 and major architectural inflection points.
   - Architecture-family lineage diagram (Llama, Qwen, Phi, Gemma, DeepSeek, OLMo, Mistral, Granite).
   - Consensus-vs-divergence table (the §2.4 table above).
   - "What was tried and abandoned" section (parallel attn+FFN, MQA, ALiBi, pure SWA, Q4_0, LayerNorm, absolute positions, NTK-aware).
   - Per-axis evolution: how RoPE θ went 10K → 5M, how vocab went 32K → 262K, how head count converged on GQA.
   - Per-section conclusions and open research questions.
2. **Add `research/06-tokenization.md`**, `research/09-norm-placement.md`, `research/11-mtp-speculative.md`, `research/12-swa-history.md`, `research/14-moe-routing.md`, `research/15-mu-p-training-tricks.md`. (Six new focused reports.)
3. **Add `research/02-extension.md`** covering the 35 model variants in 01 that 02 doesn't cover — at least one variant per architecturally-distinct family (Phi-4-mini, MiniCPM 3, Granite 3.3, RWKV-7, Zamba2, SmolLM3, StarCoder 2). Re-establish that "10 families" claim with full triple-source HF/vLLM/llama.cpp coverage.
4. **Replace `research/03`'s missing vLLM section** by promoting vLLM to a top-level §10. Document its op set (PagedAttention, FusedMoE, MergedColumnParallelLinear, QKVParallelLinear, ScaledMM, Marlin GEMM, etc.) — currently vLLM is referenced by name in every other report and never given its own section.
5. **Reconcile attention-variant count and AttentionSpec axis count.** Either the spec is right (4 kinds + 22 fields) or research/05 is right (13 kinds, 19 axes). Pick one and update the other(s).
6. **Add missing AttentionSpec fields**: `n_q_streams` for differential, `qk_norm_axis` for per-layer μP, `linear_pi` for RoPE scaling enum, `dynamic_ntk` separate from `ntk`, NoPE handling per layer.
7. **Add missing QuantSpec dtypes**: GGUF k-quants (Q2_K..Q8_K), i-quants (IQ1..IQ4), MXFP6 (E3M2, E2M3), MXFP8, MXINT8, FP8_E4M3FNUZ, FP8_E5M2FNUZ, AQLM codebook indicator. Add `outlier_decomposition` axis for LLM.int8(). Add online-rescale flag for FA3 attention-internal quant.
8. **Add a unified `glossary.md`** with first-use definitions for MLA, GQA, MQA, MHA, SWA, RoPE, RMSNorm, LayerNorm, SwiGLU, GeGLU, gegelu, μP, NoPE, sink, MTP, KIVI, NF4, NVFP4, MXFP*, EAGLE, Medusa, RAdix attention, paged attention, chunked prefill.
9. **Add equations**. At minimum: RoPE, YaRN, MLA absorption, softcap, KIVI K/V asymmetry, RMSNorm vs LayerNorm. Inline in the relevant report.
10. **Add a unified `references.bib`** consolidating all paper citations. Convert inline arXiv mentions to `[Lin et al. 2024]` style. Use the same bibliography across all reports.
11. **Normalize model names**: pick "Llama 3" (space), "Qwen3" (no space) or vice versa, and apply globally. Document the convention.
12. **Fix citation off-by-N section numbers** in the spec §12 ("research/04 §10" → §7, etc.).
13. **Add VLM coverage** to reports 02–05: one paragraph per report on whether the analyzed axes apply to VLM text towers. Currently only 01 §2.9 mentions them.
14. **Add timeline annotations to research/02–05**. Each variant discussion should carry a release date and where it sits on the timeline. Currently 02–05 read as outside time.
15. **Add a one-page comparison table at the start of `research/00-evolution.md`** (the survey summary table sketched in §2.4 above).
16. **Add per-section conclusions** to each report. Each §N ends with 3–5 bullet "findings" and 1–2 "open questions".
17. **Add a "limitations and unresolved data" section per report.** Research/03 §12 hints at this (Huawei MindIE access limitations); make it standard.
18. **Add MTP and speculative-decoding architecture impact discussion** somewhere — proposed as report 11 above.

---

## Final Verdict

**Does the artifact set meet the survey-paper bar?**

**No.** It is approximately *60% of the way there*, with the missing 40% being precisely the parts that distinguish a survey from a design memo: evolution narrative (§2 is the biggest gap), figures/lineage diagrams, consolidated bibliography, glossary, future directions, and the missing reports proposed in §8.

**What it is excellent at**: depth-of-evidence (especially research/02's triple-source code-level analysis and research/04's scheme-by-scheme parameter mining), cross-source verification, identification of the architectural axes, and design-relevant synthesis. As input to an IR design, it is best-in-class. As a standalone survey paper, it needs the structural additions listed in §9.

**Net inconsistencies counted**: ≈17 (Section 1).
**Major coverage gaps counted**: ≈30+ items spread across missing models, runtimes, schemes, modalities (Section 3).
**Most damaging single absence**: the evolution narrative (Section 2).
**Easiest high-impact fixes**: model-name normalization (§1.5), citation off-by-N in spec §12 (§1.6), and adding `research/00-evolution.md` with timeline + lineage diagrams + abandonment-section (§9 item 1).

*End of meta-critique.*
