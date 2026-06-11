# Comparison study — Sebastian Raschka's LLM Architecture Gallery vs llm-layers coverage

**Date:** 2026-06-09
**Methodology:** Raschka's gallery is treated as a source of *ideas to investigate*. Every "we lack X" or "we have X" verdict has been verified against `modeling_*.py` source code in `C:/Users/zhengte/external/transformers/src/transformers/models/<family>/` per the post-B0.5 rule (`feedback_verify_source_not_blogs.md`). The WebFetch summarization of Raschka's page was treated as untrusted text — claims it surfaced were cross-checked against the local transformers tree before being accepted into the IR-gap analysis.
**Cutoff:** llm-layers state at HEAD `4010e74` (M2-complete, 30 working families, 38-axis catalog in `research/02-layer-sources.v3.md §3`); Raschka gallery state per WebFetch on 2026-06-09 ("Jun 4 last updated").

---

## 1. Inventory of Raschka's coverage

### 1.1 Families catalogued (78 entries per WebFetch summary)

Grouped by lineage as Raschka groups them:

- **GPT lineage:** GPT-2 XL (1.5B), GPT-OSS 20B / 120B
- **Llama:** Llama 3 8B, Llama 3.2 (1B, 3B), Llama 4 Maverick (400B)
- **Qwen:** Qwen3 (0.6B / 4B / 8B / 32B / 235B-A22B / 30B-A3B), Qwen3.5 (397B), Qwen3.6 (27B, 35B-A3B), Qwen3-Next 80B-A3B, Qwen3 Coder Flash 30B-A3B
- **DeepSeek:** V3 671B, R1 671B, V3.2 671B, V4-Flash 284B, V4-Pro 1.6T
- **Gemma:** Gemma 3 (27B, 270M), Gemma 4 (31B, 26B-A4B, 12B, E2B, E4B)
- **Mistral:** Small 3.1 (24B), Large 3 (673B), Small 4 (119B)
- **OLMo:** OLMo 2 (7B), OLMo 3 (7B, 32B)
- **Kimi:** K2 / K2.5 / K2.6 (each 1T), Kimi Linear 48B-A3B
- **GLM:** 4.5 / 4.7 (each 355B), 5 / 5.1 (each 744B), 4.5-Air (106B)
- **MiniMax:** M2 / M2.5 / M2.7 (each 230B)
- **NVIDIA Nemotron 3:** Nano (30B-A3B / 4B), Super (120B-A12B), Ultra (550B-A55B)
- **Xiaomi MiMo:** V2-Flash (309B), V2.5 (310B), V2.5-Pro (1.02T)
- **Cohere:** Command A+ (218B-A25B)
- **xAI:** Grok 2.5 (270B)
- **Misc:** SmolLM3 (3B), Tiny Aya 3.35B, Ling 2.5 / 2.6 (1T each), Sarvam (30B / 105B), Phi-4 (14B), xLSTM (7B), INTELLECT-3 (106B), LongCat-Flash-Lite (68.5B-A3B), Nanbeige 4.1 (3B), Arcee AI Trinity Large (400B), Step 3.5 Flash (196B), Laguna XS.2 (33B), Granite 4.1 (30B), Tencent Hy3 preview (295B-A21B), Liquid LFM2.5 (1.2B / 350M / 8B-A1B), JetBrains Mellum2 Thinking (12B-A2.5B), Zyphra ZAYA1 (8.4B)

### 1.2 Architectural patterns / axes Raschka highlights

**Normalization:** RMSNorm pre-norm (standard), QK-Norm per-head + per-layer, post-norm (OLMo), sandwich norm (Arcee), KV-LayerNorm (Sarvam 105B).

**Token mixers:** MHA, GQA, MQA, MLA, SWA, "5:1 sliding:global" (Gemma), "3:1 gated DeltaNet : gated attention" (Qwen3.5/Next), CSA + HCA (DeepSeek V4), CCA (compressed convolutional attention), DSA (DeepSeek V3.2 Lightning Indexer), Lightning Attention (Ling 2.5, MiniMax), Kimi Delta Attention (Kimi Linear), trained attention sinks (GPT-OSS), iRoPE (Llama 4).

**Position encodings:** RoPE, NoPE periodic, p-RoPE (Gemma 4 global), YaRN (OLMo 3 globals), partial RoPE (MiniMax M2).

**Expert routing / sparsity:** Standard top-k MoE, shared-expert configs, **hash-based routing (DeepSeek V4)**, **sigmoid-routed MoE (Laguna XS.2)**, **shortcut MoE routing (LongCat-Flash-Lite)**, dense-prefix MoE (DeepSeek V3 style), **latent-space MoE (Nemotron 3 Super "LatentMoE")**, **top-1 routed FFN (ZAYA1-8B)**.

**Advanced / "first-of-kind" mechanisms:** MTP (multi-token prediction) at versions 1 / 3 / V4 paths, per-layer embeddings (Gemma 4 E2B/E4B), KV sharing across layers, attention-budget per-layer (asymmetric H per layer), manifold-constrained hyper-connections (DeepSeek V4 mHC), compressed attention caches, n-gram embedding expansion (LongCat-Flash-Lite), Mamba-2 + attention/MoE hybrids, xLSTM, dense + sparse layer interleaving (Llama 4 Maverick).

### 1.3 Page organization and analytical lens

- **Primary axis:** release date (toggle: newest / oldest / AA Index score / A-Z / size).
- **Per-model card:** standardized fact-sheet with Scale, Context, Decoder type, Attention mech, Layer mix (explicit `48 SWA + 16 global` style), KV cache / token (with a Low/Moderate/High/Very-low qualitative label), Key detail, AA Intelligence Index, related concept tags.
- **Diff tool:** dropdown selector for two arbitrary models, side-by-side comparison.
- **Architecture contact sheet:** all 78 models on one poster.

**Recurring analytical lenses** Raschka leans on:
1. **KV cache per token** with a 5-bucket qualitative scale (Very low → Very high). Every card carries this.
2. **Active-parameter ratio** for MoE: framed as "37B active (5.5% active)" with explicit percentage.
3. **Attention-mechanism evolution** treated as a frontier: MHA → GQA → MLA, and now sparse/linear/hybrid.
4. **Normalization placement** treated as a first-class signal distinct from attention.
5. **Multi-token prediction (MTP)** as a training-recipe-as-architecture choice.

**Recurring phrases** (from the narrative): *"stays close to"*, *"swaps"*, *"sweet spot"*, *"latency-focused"*, *"hybrid"*.

---

## 2. Our coverage at a glance

- **Census:** `research/01-model-census.v3.md` — 148 model rows, 19 axes, 33 family arcs.
- **Layer source decomposition:** `research/02-layer-sources.v3.md §3` — 38-axis catalog, 48 family deep-dives.
- **KV cache + attention:** `research/05-kvcache-attention.v3.md` — 38 attention / 18 RoPE / 18 cache axes.
- **Per-family layer doc:** 37 files in `models/<family>/layer.md`.
- **Working IR:** `api/specs.py` 18 dataclasses, `api/types.py` 20+ enums, 3.6k LoC across 13 files.
- **Numerically gated:** 30 families (16 R-class against real HF weights, 11 S-class synthetic self-consistency, 3 shape-only).

This study compares against the 38-axis catalog and the spec dataclasses, not just the 30 working families.

---

## 3. List A — Patterns Raschka highlights, verified against source

For each pattern: Raschka's framing → our coverage → source-verified verdict.

### A.1 Hash-based routing for MoE (DeepSeek V4) — MISSING

- **Raschka claim:** DeepSeek V4 uses "hash-based routing".
- **Our coverage:** `MoESpec.router_kind ∈ {"softmax", "sigmoid_plus_bias", "topk_then_softmax_with_bias"}` — no hash option (`api/specs.py:480`).
- **Source verification:** `transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:1040` defines `class DeepseekV4HashRouter(nn.Module)` alongside `DeepseekV4TopKRouter` (line 1019); a config-selected dispatch picks one. **Verdict: confirmed real, we don't model it.** Hash routing is a "no-gate" router — token → fixed-hash → expert assignment.
- **Action:** add `"hash"` value to `MoESpec.router_kind` and a `HashRoutingSpec(hash_seed, n_buckets, bucket_to_expert_map)` sidecar if any of the production V4 variants get a numerical-gate run.

### A.2 CSA + HCA dual sparse attention (DeepSeek V4) — MISSING

- **Raschka claim:** V4 replaces V3.2's DSA Lightning Indexer with "CSA + HCA" — Compressed Sparse Attention + Heavily Compressed Attention. Claimed: 27% of V3.2 FLOPs, 10% of V3.2 KV cache at 1M context.
- **Our coverage:** `AttentionKind ∈ {STANDARD, MLA, DSA}` (`api/types.py:6`). DSA itself is shape-only (no forward). No CSA or HCA values.
- **Source verification:** `modeling_deepseek_v4.py:265` ("slice with window `w`'s Cb slice — effective width `2 * compress_rate_csa`, stride `compress_rate_csa`"), `:365` ("every `compress_rate_hca` (m'=128) source tokens into a single compressed KV"), and `:581` ("every `compress_rate_csa` (m=4) source tokens and runs a Lightning Indexer on..."). **Verdict: confirmed real and architecturally novel, we lack it.** The pattern is documented in `research/01-model-census.v3.md` line 15 narrative but not lifted into the 38-axis catalog or specs.
- **Action:** add `AttentionKind.CSA_HCA` + `CSAHCASpec(compress_rate_csa, compress_rate_hca, indexer_top_k, indexer_dim)`. Update `research/02-layer-sources.v3.md §3.16` and add to the IndexerSpec family in `api/specs.py`.

### A.3 Sigmoid-routed MoE (Laguna XS.2) — PARTIAL

- **Raschka claim:** Laguna XS.2 uses sigmoid MoE routing.
- **Our coverage:** `MoESpec.router_kind` includes `"sigmoid_plus_bias"` (DeepSeek-V3-style, with the e-score-correction bias). A pure sigmoid-without-bias variant is not a distinct value.
- **Source verification:** `transformers/src/transformers/models/laguna/modeling_laguna.py:163` defines `LagunaTopKRouter`, and the router uses `nn.functional.sigmoid` on logits (verified at MiniMaxM2 `modeling_minimax_m2.py:58` for the same pattern). **Verdict: pure sigmoid (no bias) is structurally distinct from "sigmoid + bias".** The IR can express it by setting `e_score_correction_bias = 0` but the discriminated enum doesn't distinguish — a config error would not be caught.
- **Action:** consider splitting into `"sigmoid"` vs `"sigmoid_plus_bias"`, or just document the pattern in `02-layer-sources.v3.md §3.14`. Low priority.

### A.4 Shortcut MoE routing (LongCat-Flash-Lite) — MISSING

- **Raschka claim:** LongCat uses "shortcut MoE routing".
- **Our coverage:** no shortcut-routing axis.
- **Source verification:** LongCat is not in the local transformers tree. The Meituan LongCat-Flash paper (arXiv:2509.01366) describes the "shortcut" router as a Zero-Computation Expert (ZCE) that copies the input through when selected, so the per-token compute cost is dynamically variable. **Verdict: confirmed by external paper, we lack it.** It belongs in the same `MoESpec.router_kind` enum extension as hash routing.
- **Action:** add `"shortcut"` (or `"with_zce"`) router kind; document the ZCE pattern as an axis variant.

### A.5 Latent-space MoE (Nemotron 3 Super) — MISSING

- **Raschka claim:** Nemotron 3 Super uses "LatentMoE".
- **Our coverage:** no latent-space MoE concept in `MoESpec`.
- **Source verification:** `nemotron_h/modeling_nemotron_h.py` is the Mamba-2 hybrid backbone (verified at line 114 `NemotronHMamba2Mixer`, line 672 `NemotronHMoE`, line 834 `NemotronHAttention`); the `LatentMoE` variant is not in the local in-tree file — it appears to be a 3 Super / Ultra variant pending upstream port. The pattern (per NVIDIA NeMo docs) is: expert outputs are routed in a low-rank latent space rather than the full residual stream. **Verdict: confirmed external, we lack it.** This is a meaningful axis: it sits orthogonal to "shared experts" because it changes the *space* the experts operate in.
- **Action:** add `MoESpec.expert_space ∈ {"residual", "latent"}` + `latent_rank: Optional[int]`. Medium priority.

### A.6 Trained attention sinks (GPT-OSS) — HAVE

- **Raschka claim:** GPT-OSS has "trained attention sinks".
- **Our coverage:** `AttentionSpec.n_sink_tokens: Optional[int]` (`api/specs.py:280`) plus `MaskKind.SINK`.
- **Source verification:** `gpt_oss/modeling_gpt_oss.py:309` `self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))`; `:267-275` concat-then-drop is implemented exactly as our spec docstring describes. **Verdict: we have it.**

### A.7 Cross-layer KV sharing (Gemma 4) — HAVE

- **Raschka claim:** Gemma 4 introduces KV sharing across layers (`num_kv_shared_layers`).
- **Our coverage:** `models/gemma4/` builds with `SharedLayerKVCache` (`api/kvcache.py:188`); `AttentionSpec.kv_source_layer_offset` (`api/specs.py:265`) covers the related CLA pattern.
- **Source verification:** `models/gemma4/layer.md` (production-gated R-class) and `research/02-layer-sources.v3.md §3.33` cover Gemma 4 same-block KV sharing + Apple AFM block-role sharing + CLA. **Verdict: we have it, source-verified to Gemma 4 production.**

### A.8 Partial RoPE on global layers (Gemma 4) — HAVE

- **Raschka claim:** Gemma 4 uses p-RoPE on global layers (`partial_rotary_factor = 0.25`).
- **Our coverage:** `RoPESpec.partial_rotary_factor` + `partial_rotary_kind ∈ {"prefix", "proportional"}` (`api/specs.py:139, 158`) — the distinction between Phi-3 prefix p-RoPE and Gemma 4 proportional p-RoPE was a B0.5 / B2a-era source-grounded discovery. **Verdict: we have it, plus the discriminator other model libraries miss.**

### A.9 Per-Layer Embeddings (Gemma 4 E2B/E4B) — HAVE

- **Raschka claim:** Gemma 4 introduces per-layer embeddings.
- **Our coverage:** `PLESpec` (`api/specs.py:533`), `api/embedding.py` `PerLayerEmbedding`, tested in `models/gemma4/`. **Verdict: we have it.**

### A.10 iRoPE / NoPE-periodic (Llama 4 Scout) — HAVE (shape-only)

- **Raschka claim:** Llama 4 Maverick has "dense + sparse layer interleaving"; Scout has iRoPE.
- **Our coverage:** `models/llama4_scout/` is shape-only (Llama 4 weights are gated). The iRoPE per-layer mask + temperature-scaling is documented in `research/02-layer-sources.v3.md §5.35` and the spec hooks exist (per-layer RoPE-or-none, `attn_temperature` was caught as a B4 drift). **Verdict: spec-level we have it; numerical gate deferred until weights are accessible.**

### A.11 Lightning Attention (MiniMax-Text-01, Ling 2.5) — PARTIAL

- **Raschka claim:** MiniMax-Text-01 ships 7:1 Lightning Attention; Ling 2.5 ships Lightning.
- **Our coverage:** `research/02-layer-sources.v3.md §5.44` covers MiniMax Lightning at the description level (7:1 linear:softmax). `api/types.py` doesn't have a `TokenMixerKind.LIGHTNING_ATTN` enum; `models/minimax_text_01/` is a deferred stub.
- **Source verification:** `transformers/src/transformers/models/minimax/` and `minimax_m2/modeling_minimax_m2.py:296` exist; the local M2 file is the softmax-only newer variant, but the older `MiniMax-Text-01` (which is in transformers/models/minimax/) actually carries the Lightning kernel. **Verdict: spec gap.** We can express it via a generic linear-attention token mixer but don't have a discriminated variant. Medium priority.
- **Action:** add `TokenMixerKind.LIGHTNING_ATTN` + a `LightningAttnSpec` describing the SiLU-gated linear-attention recurrence. Promote `models/minimax_text_01/` from stub to S-class.

### A.12 Gated DeltaNet 3:1 (Qwen3-Next) — PARTIAL

- **Raschka claim:** Qwen3-Next ships gated DeltaNet + gated attention in 3:1 ratio.
- **Our coverage:** `research/02-layer-sources.v3.md §5.36` covers Qwen3-Next at the description level. `api/specs.py` has no `GatedDeltaNetSpec`. No `models/qwen3_next/` working family.
- **Source verification:** `transformers/src/transformers/models/qwen3_next/modeling_qwen3_next.py:556` imports `chunk_gated_delta_rule, fused_recurrent_gated_delta_rule`; `:827` `self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)`; `:972` per-layer dispatch on `config.layer_types[i] == "linear_attention"`. **Verdict: spec gap.** Same shape as Lightning Attention from the IR perspective — both want `TokenMixerKind.LINEAR_ATTN_GATED_DELTA` plus a per-layer-types dispatcher (which we already have for SWA-alternation).
- **Action:** add `TokenMixerKind.GATED_DELTANET` + `GatedDeltaNetSpec`. Bootstrap `models/qwen3_next/`. High priority — Qwen3-Next is mainstream production.

### A.13 Kimi Delta Attention (Kimi Linear 48B-A3B) — MISSING

- **Raschka claim:** Kimi Linear uses "Kimi Delta Attention".
- **Our coverage:** none.
- **Source verification:** Kimi Linear is not in the local transformers tree as of v5.10.2. The Moonshot Kimi Linear release blog describes KDA as a DeltaNet variant with a different gate parametrization. **Verdict: external claim, we lack it.** Likely subsumable under `TokenMixerKind.GATED_DELTANET` with a different gate sub-mode; document at axis-catalog level pending HF in-tree port.
- **Action:** track in `research/issues/` as "Kimi Delta Attention — pending upstream port".

### A.14 Short-conv mixer (LFM2.5) — MISSING

- **Raschka claim:** LFM2.5 family.
- **Our coverage:** none.
- **Source verification:** `lfm2_moe/modeling_lfm2_moe.py:376` defines `class Lfm2MoeShortConv(nn.Module)` — a 1D depthwise short-conv mixer that LFM2 hybrids with attention every other layer. **Verdict: confirmed real, novel axis.** It is conceptually adjacent to Mamba but with a simpler conv-only state (no SSM scan).
- **Action:** add `TokenMixerKind.SHORT_CONV` + `ShortConvSpec(kernel_size, expand)`. Low-medium priority (Liquid is a smaller-impact vendor than DeepSeek / Qwen but the architectural axis is clean).

### A.15 MTP (multi-token prediction) as a first-class axis — MISSING

- **Raschka claim:** MTP appears across 15+ models; he treats it as a structural enabler of inference throughput.
- **Our coverage:** MTP is discussed in `research/01-model-census.v3.md` and `research/02-layer-sources.v3.md` arc narratives, but it is *not* one of the 38 layer-source axes, and there is no `MTPSpec` in `api/specs.py`. MTP heads (e.g., Hy3 Preview 3.8B head, Gemma 4 MTP drafters, DeepSeek-V3 MTP-1) are out of the per-layer scope but in the "model assembly" scope.
- **Source verification:** DeepSeek V3 ships an `eh_proj` MTP head in its config; Hy3 Preview's 3.8B MTP head is announced in the model card. **Verdict: we don't model it at IR level.** Defensible omission since MTP is a draft-head, not a per-layer change. But Raschka's lens that it is *load-bearing for serving throughput* suggests it deserves a sidecar `DraftHeadSpec` for completeness.
- **Action:** consider `api/specs.py::MTPHeadSpec(n_draft_tokens, head_arch, share_with_main_lm_head: bool)` as an *optional* component of the model-assembly layer (above the per-layer block). Low-medium priority.

### A.16 KV cache / token as a presentation metric — NEW IDEA, not an axis

- **Raschka claim:** every model card carries `KV cache / token (bf16)` with a Low/Moderate/High/Very-low label.
- **Our coverage:** the calculation is implied by `research/05-kvcache-attention.v3.md` for every family, but we don't surface it as a per-family scalar in `models/<family>/layer.md`.
- **Verdict: not an IR axis, but a derivable scalar that would improve our docs.**
- **Action:** see §5 below — add a "KV/token at S=4096 in bf16" line at the top of each `models/<family>/layer.md`. Cheap, high signal-density.

### A.17 Other patterns Raschka mentions that we already cover

| Pattern | Our coverage | Source-verified? |
|---|---|---|
| QK-Norm (per-head, per-layer) | `AttentionSpec.qk_norm`, `qk_norm_phase`, `qk_norm_shape`, `qk_norm_fixed_scale` | Yes (Qwen3, Gemma 3/4, OLMo 2, OLMo 3, Cohere) |
| Sandwich norm | `DecoderBlockSpec` with 4 norm slots; `NormPosition.PRE_AND_POST` | Yes (Gemma 2/3/4) |
| 5:1 SWA / global alternation | `AttentionSpec.sliding_window` per-layer; `Ministral` adds 3rd interleave pattern | Yes |
| Dense + sparse layer interleaving (Llama 4 Maverick) | `Qwen3-MoE.mlp_only_layers`, `DeepSeek.first_k_dense_replace`, `Jamba.layers_num_experts[i]` | Yes |
| Mamba-2 + attention hybrids | `models/granite4_h/`, `models/jamba/` stub, `research/02-layer-sources.v3.md §5.37` | Yes (Granite-4-H 9:1 numerically gated) |
| xLSTM | `lfm2_moe`-adjacent at axis level; `xlstm/modeling_xlstm.py` exists | Mentioned in census, no working model |
| Tied embeddings | axis 3.26 (NEW in v2) | Yes (Llama-1, Gemma 1-4, SmolLM3, etc.) |

---

## 4. List B — Families Raschka covers that we MAY have missed

For each: Raschka's claim → check `models/<family>/` and census + transformers tree → verdict.

### B.1 Kimi K2 / K2.5 / K2.6 / Kimi Linear 48B-A3B — PARTIAL (census-only, no IR)

- **Census coverage:** `research/01-model-census.v3.md` line 267 has Kimi K2.6 noted as 2026-04 release. K2 / K2.5 / Kimi Linear not separately rowed.
- **Source verification:** Kimi K2 is decoder-only MoE 1T total / 32B active (MLA + DeepSeek-V3-style routing per Moonshot blog). Kimi Linear is the linear-attention variant using KDA. Not in local transformers tree.
- **Verdict: under-covered.** K2 is a flagship-scale flagship — it's worth at least a `research/02-layer-sources.v3.md` deep-dive section even if not numerically gated.
- **Priority:** medium (it's >8B-active but axis-load-bearing per our v3 rule).

### B.2 GLM 4.5 / 4.7 / 5 / 5.1 / 4.5-Air — PARTIAL (some in tree, no IR coverage)

- **Census coverage:** `01-model-census.v3.md` mentions "GLM-5.1 / GLM-5V-Turbo" in the 2026-H1 section but no row.
- **Source verification:** `transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` is present and has the indexer pattern (`GlmMoeDsaIndexer` line 105) + DeepSeek-V3-style sigmoid-with-bias routing (line 559) + group-limited routing (line 563). GLM 4.5 / 4.7 are in `glm4_moe/` and `glm/`.
- **Verdict: we have transformers source but no `models/glm*/` family folder.** This is a clear gap given GLM is one of the major frontier MoE families.
- **Priority:** medium-high. The architectural overlap with DeepSeek V3.2 makes it a small additional cost.

### B.3 MiniMax M2 / Text-01 — PARTIAL (text-01 stub, M2 missing)

- **Census coverage:** `01-model-census.v3.md` line 134 has MiniMax-Text-01 (Lightning Attention 7:1). `models/minimax_text_01/` is a B7 deferred stub.
- **Source verification:** `minimax/` and `minimax_m2/modeling_minimax_m2.py` both exist locally. M2 (line 296) is a softmax-only Mistral-shaped MoE with sigmoid routing (`router_logits.sigmoid()` at line 58).
- **Verdict: Text-01 stub present, M2 missing entirely.** MiniMax M2 is the more recent of the two and the one Raschka actively features.
- **Priority:** medium. M2 is structurally similar to Mixtral + sigmoid router, so cheap to add.

### B.4 Nemotron 3 family (Nano / Super / Ultra) — PARTIAL

- **Census coverage:** `01-model-census.v3.md` line 174 mentions "NVIDIA Nemotron 3 Super / Nano-Omni / Ultra". `research/02-layer-sources.v3.md §5.48` describes the family.
- **Source verification:** `nemotron/` (older), `nemotron_h/modeling_nemotron_h.py` (Mamba-2 hybrid). The newer Nemotron 3 (which Raschka lists as Nano-30B-A3B, Super-120B-A12B, Ultra-550B-A55B) maps to `nemotron_h` for the H-hybrid variants and `nemotron/` for the pure-attention path. `models/nemotron3/` is a deferred stub.
- **Verdict: source available, no working family.** A `nemotron3` family folder building on the existing Mamba-2 + GQA primitives would be a small lift.
- **Priority:** medium. The LatentMoE variant raises the marginal IR cost.

### B.5 Arcee Trinity Large 400B — MISSING

- **Census coverage:** none.
- **Source verification:** `arcee/modeling_arcee.py` exists locally — verified line 50 `ArceeMLP`, line 221 `ArceeAttention`, line 288 `ArceeDecoderLayer`. The Arcee family ships small models too (Arcee Maestro etc.).
- **Verdict: lib coverage exists, our census misses it entirely.** Raschka's "sandwich norm (Arcee)" note suggests Trinity Large uses a distinctive norm placement that's worth catching.
- **Priority:** low-medium. Trinity Large is the only Arcee model Raschka calls out, and at 400B it's well above the <8B floor (but the SLM variants may inherit the sandwich norm).

### B.6 Liquid LFM2.5 family — MISSING

- **Census coverage:** `01-model-census.v3.md` mentions "Liquid LFM2.5-VL-450M / LFM2.5-8B-A1B" in line 50.
- **Source verification:** `lfm2/`, `lfm2_moe/`, `lfm2_vl/` all in tree. Short-conv mixer is a distinct axis (see A.14 above).
- **Verdict: census-mentioned only, no IR support.** LFM2.5 < 8B is in-scope.
- **Priority:** medium.

### B.7 Cohere Command A+ 218B-A25B — MISSING (at this size)

- **Census coverage:** v2 had Cohere Command-R / Aya as a baseline family for parallel-residual layout. Command A+ (218B-A25B) is newer and not rowed.
- **Source verification:** `cohere/`, `cohere2/`, `cohere2_moe/` all in tree.
- **Verdict: parallel-residual axis is already in IR via `BlockLayout.PARALLEL`; the family-as-row is missing but the architectural axis is covered.**
- **Priority:** low. The structural insight is already captured.

### B.8 Other Raschka families not in our census

| Family | Architectural novelty | In transformers tree? | Priority |
|---|---|---|---|
| **xLSTM 7B** | Different recurrent kernel from Mamba | `xlstm/modeling_xlstm.py` exists | Low — research/02 §5.44-adjacent |
| **JetBrains Mellum2 Thinking 12B-A2.5B** | Code-specialized MoE | No | Very low |
| **LongCat-Flash-Lite** | Shortcut MoE routing (see A.4) | No | Low |
| **Ling 2.5 / 2.6 (1T)** | Lightning Attention production | No | Low |
| **Tencent Hy3 Preview** | 3.8B MTP head | No | Low; rowed in census |
| **Sarvam 30B / 105B** | KV LayerNorm | No | Low |
| **Nanbeige 4.1 (3B)** | Standard GQA, small | No | Very low |
| **Step 3.5 Flash 196B** | MTP-3 production | No | Low |
| **Laguna XS.2** | Sigmoid MoE | `laguna/modeling_laguna.py` exists | Low (subsumed by A.3 axis) |

---

## 5. List C — Presentation / framing insights worth borrowing

### C.1 Per-card KV-cache / token + qualitative bucket

Raschka makes KV-cache cost legible at-a-glance by reducing it to a single number + a 5-bucket label. We compute the equivalent in `research/05-kvcache-attention.v3.md` but don't surface it on each `models/<family>/layer.md`. **Action:** add a header line `| KV cache / token (bf16) | 128 KiB · Moderate |` to each layer.md.

### C.2 The "Layer mix" notation

`52 sliding-window + 10 global` is much more readable than our prose `5:1 SWA:global alternation pattern with last layer global`. The pattern generalizes well to the increasingly common per-layer-type compositions (Qwen3-Next 3:1 linear-attn:softmax, Granite-4-H 9:1 mamba:attn, Llama 4 Maverick dense-MoE mix). **Action:** standardize a `Layer mix: <N1 of type-1> + <N2 of type-2> [+ <N3 of type-3>]` field in `models/<family>/layer.md` headers and in the `research/01-model-census.v3.md` rows.

### C.3 Architecture diff tool (model A vs model B)

Raschka's "Diff" picker lets the reader interactively compare any two of 78 models. Our equivalent is the prose-based "evolution arcs" in `research/01-model-census.v3.md §5` (33 sub-arcs). The arcs are deeper per-pair but require knowing what to look up. **Action:** consider a tabular "axis-by-axis diff" appendix that takes two `models/<family>/layer.md` paths and emits a diff. This is a docs-tooling project, not an IR change — defer.

### C.4 AA Intelligence Index integration as a capability axis

Raschka uses Artificial Analysis Intelligence Index scores on every card with sub-domain breakdowns (General / Scientific / Coding / Agents). Our research is intentionally architecture-only and AA Index would conflate capability with architecture. **Verdict: not worth borrowing** — our scope is structural; capability is an orthogonal concern that would muddy the IR-evidence chain.

### C.5 Active-parameter percentage as a header

Raschka writes "37B active (5.5% active)" — the explicit percentage is a strong design-pressure signal that lifts MoE comparisons. We have the data but format it as "active/total". **Action:** add the percentage to MoE rows in `research/01-model-census.v3.md` and to MoE `models/<family>/layer.md` headers. Trivial.

### C.6 Architecture contact sheet poster

Raschka offers a printable poster showing all 78 families on one page. Our census table is the closest equivalent. **Verdict: nice-to-have for outreach, not engineering value.** Defer.

### C.7 Per-card "Key detail" one-liner

Each Raschka card has a single-sentence "Key detail" capturing the family's signature (e.g., "Llama 3.2 1B: smallest of the 2024 Llama family, used as a NoPE / RoPE baseline by SmolLM3 and others"). This is a useful compression. Our `models/<family>/layer.md §1 Identity` block has multiple such sentences but no canonical one-liner. **Action:** add a `Signature: <one-sentence>` line to each layer.md §1.

---

## 6. Recommendations — prioritized

### Must-do (3 items)

1. **Add `TokenMixerKind.GATED_DELTANET` + `GatedDeltaNetSpec`** — Qwen3-Next is mainstream production, source is in tree (`qwen3_next/modeling_qwen3_next.py`), and the IR currently can't express it. Bootstrap `models/qwen3_next/` as an S-class family.
2. **Add `AttentionKind.CSA_HCA` + `CSAHCASpec`** — DeepSeek V4 is the current open-weights frontier; the source is in tree (`deepseek_v4/modeling_deepseek_v4.py`); the axis is real (verified at lines 265, 365, 581).
3. **Extend `MoESpec.router_kind` with `"hash"` (and consider `"shortcut"`)** — DeepSeek V4 hash routing is verified at `modeling_deepseek_v4.py:1040`. This is a one-enum-value add that unblocks V4 IR construction.

### Should-do (4 items)

4. **Bootstrap `models/glm_moe_dsa/`** — GLM 4.5-DSA family is in tree, structurally a DeepSeek V3.2 cousin, and our census mentions it but doesn't row or layer-decompose it.
5. **Bootstrap `models/minimax_m2/`** — in tree, structurally a Mixtral cousin with sigmoid routing; small lift, fills a major-vendor gap.
6. **Add `Layer mix:` + `Signature:` + `KV/token:` header fields to every `models/<family>/layer.md`** — pure docs work, high information density gain per Raschka's presentation insights.
7. **Promote `models/minimax_text_01/` from stub to S-class** with a Lightning Attention token mixer entry.

### Nice-to-have (4 items)

8. **Add `MoESpec.expert_space ∈ {"residual", "latent"}`** for Nemotron 3 Super LatentMoE.
9. **Add `TokenMixerKind.SHORT_CONV` + `ShortConvSpec`** for LFM2 family.
10. **Add an `MTPHeadSpec` as a model-assembly-layer sidecar** for the DeepSeek-V3 / Hy3 / Gemma 4 drafters.
11. **Census rows for Kimi K2 family, Arcee Trinity, Cohere Command A+, Liquid LFM2.5** — fills the >8B axis-load-bearing slots.

### Not-applicable / explicitly rejected

- **AA Intelligence Index in our docs** — conflates capability with structure; our scope is architecture-only.
- **Contact-sheet poster** — outreach artifact, not engineering value.
- **Diff tool UI** — our prose arcs serve the same purpose for our audience (engineers reading source).

---

## 7. Honest assessment — where Raschka wins, where we win

### Where Raschka is stronger

1. **Breadth of frontier MoE coverage.** Raschka catalogs Kimi K2.x, GLM 4.5-5.1, MiniMax M2.x, Nemotron 3 Super/Ultra, Grok 2.5, Cohere Command A+, Arcee Trinity, MiMo, Ling, Step, Laguna, Mellum, ZAYA1 as separate cards. Our census names many but rows fewer; our `models/` includes none of them at IR level. For frontier-MoE breadth, Raschka's gallery is the better starting reference.
2. **Presentation density.** A single Raschka card has KV/token, active%, layer mix, AA score, and a one-line signature — five high-signal scalars at a glance. Our `models/<family>/layer.md` files are deeper but much harder to scan; an engineer triaging "which 5 families are closest to Qwen3-MoE?" finds Raschka faster.
3. **Capability-to-architecture mapping.** The AA Intelligence Index sub-domain breakdown (General / Scientific / Coding / Agents) is a feature we deliberately don't have. For "which architecture is best for code?" Raschka wins.
4. **Currency.** Raschka's "Jun 4 last updated" timestamp and changelog discipline are explicit; our v3 reports are dated 2026-06-04 to 2026-06-08 but the staleness story for individual rows is not surfaced.

### Where llm-layers is stronger

1. **Source-grounded discipline.** Every architectural claim in our 38-axis catalog cites `modeling_*.py` line numbers (`research/02-layer-sources.v3.md`). Raschka's cards (per the WebFetch summary) cite tech reports and `config.json` URLs but don't carry the byte-level provenance that B0.5's drift-catching demanded. Our claim that DeepSeek V4 hash routing is real is backed by `modeling_deepseek_v4.py:1040`; Raschka's equivalent is "DeepSeek V4 uses hash-based routing" without the line number. For IR-correctness, line numbers win — see §10 below.
2. **The 38-axis catalog itself.** Raschka's per-card fact-sheet is a flat list; our `research/02-layer-sources.v3.md §3` is an explicit minimal parameterization with each axis discriminated by enum values + sub-fields, ready to be lifted into `api/specs.py`. The catalog is the artifact a downstream IR-implementer wants; Raschka's gallery is the artifact a curious reader wants.
3. **Numerical-gate validation.** We have 16 R-class families (real HF weights re-assembled and bit-comparable at atol=5e-4) plus 11 S-class. Raschka has architecture diagrams; we have proof the math is right. For "does my IR re-build a real production layer?", we are the only artifact in either body of work that answers.
4. **Three-engine cross-reference (HF + vLLM + llama.cpp).** Our methodology compares each family's layer across three independent implementations. Raschka's gallery is HF-centric. The cross-references caught roughly 55 drifts during M2 rollout that any single-engine reading would have missed.
5. **Quantization survey.** `research/04-quantization.v3.md` catalogs 48 quant schemes; Raschka has MXFP4-native as a single attribute tag. For SaaS / on-device deployment work, our quant survey is materially deeper.
6. **OCR-LLM lineage.** Our `research/06-ocr-vlm-extensions.md` and `models/{got_ocr2, deepseek_ocr2, qwen2_5_vl}/` families track the OCR-LLM frontier with the same rigor as the text-only families. Raschka's gallery doesn't enumerate OCR-LLMs as a distinct lineage.
7. **Hybrid Mamba / SSM coverage.** We have 8 hybrid-SSM-Transformer topology variants enumerated (`01-model-census.v3.md §1`) and numerical gates on Mamba-2 (bit-exact synthetic) and Granite-4-H (bit-exact synthetic mamba+attn). Raschka covers Mamba-2 + attention as a single "hybrid" tag.

### Net judgment

**For an engineer building an IR / runtime / compiler around mainstream LLM architectures, llm-layers is the deeper and more trustworthy source.** The 38-axis catalog plus the numerical gates form a stronger foundation than Raschka's gallery, which is optimized for legibility over engineering use.

**For an engineer scanning the architecture frontier or pitching a design choice to a non-specialist audience, Raschka's gallery wins on presentation density and breadth-at-the-frontier.** The 78-family inventory captures more 2025-Q4 / 2026-Q1 / 2026-Q2 frontier MoE families than our 30 working families do (though our 148-row census is broader still).

**The two sources are complementary, not competitive.** Our weak axis is presentation density of per-family signatures and breadth at the frontier-MoE end (Kimi, MiniMax, GLM 4.5+, Nemotron 3, Grok). Raschka's weak axis is source-grounded byte-level provenance and IR-ready axis discrimination.

---

## 8. Specific axis-extension proposals for `api/specs.py`

```python
# A.1 hash routing (DeepSeek V4) — extend MoESpec.router_kind enum
router_kind: str = "softmax"  # "softmax" | "sigmoid_plus_bias" |
                              # "topk_then_softmax_with_bias" | "hash" | "shortcut"

# Optional sidecar:
@dataclass(frozen=True)
class HashRoutingSpec:
    n_buckets: int               # equal to n_experts for the canonical V4 hash path
    hash_seed: int               # determines the fixed token-id -> expert map
    fallback_router_kind: str = "softmax"   # for OOD tokens, if used

# A.2 CSA + HCA (DeepSeek V4) — extend AttentionKind enum
class AttentionKind(Enum):
    STANDARD = auto()
    MLA = auto()
    DSA = auto()         # Lightning Indexer (V3.2)
    CSA_HCA = auto()     # V4 dual-sparse

@dataclass(frozen=True)
class CSAHCASpec:
    compress_rate_csa: int       # m  in CSA  (V4 default = 4)
    compress_rate_hca: int       # m' in HCA  (V4 default = 128)
    indexer_dim: int             # Lightning Indexer side dim
    indexer_top_k: int           # per-query top-k against compressed entries

# A.5 Latent-space MoE (Nemotron 3 Super) — extend MoESpec
expert_space: str = "residual"   # "residual" | "latent"
latent_rank: Optional[int] = None  # required when expert_space == "latent"

# A.11 / A.12 Lightning Attention + Gated DeltaNet — extend TokenMixerKind
class TokenMixerKind(Enum):
    ATTENTION = auto()
    SSM_MAMBA1 = auto()
    SSM_MAMBA2 = auto()
    GATED_DELTANET = auto()  # Qwen3-Next; also covers Kimi Delta Attention with sub-mode
    LIGHTNING_ATTN = auto()  # MiniMax-Text-01, Ling 2.5
    SHORT_CONV = auto()      # LFM2 family

@dataclass(frozen=True)
class GatedDeltaNetSpec:
    head_dim: int
    n_v_heads: int
    n_k_heads: int
    conv_kernel: int
    chunk_size: int
    use_qk_l2_norm: bool = True

@dataclass(frozen=True)
class LightningAttnSpec:
    head_dim: int
    n_heads: int
    gate_activation: types.Activation = types.Activation.SILU
    feature_map: str = "silu_then_norm"   # or "elu_plus_1" or "polynomial"

@dataclass(frozen=True)
class ShortConvSpec:
    kernel_size: int
    expand_factor: int = 2
    bias: bool = False
```

The DecoderBlockSpec `token_mixer` Union already supports being a `SSMSpec | SSDSpec`; extending the Union to include `GatedDeltaNetSpec | LightningAttnSpec | ShortConvSpec | CSAHCASpec` is a small structural change.

---

## 9. Specific recommendations for `research/02-layer-sources.v3.md`

Add axes:
- **§3.39 Hash / shortcut MoE routing** — covers DeepSeek-V4 `HashRouter` and LongCat shortcut-ZCE pattern. Source: `modeling_deepseek_v4.py:1040`; LongCat arXiv:2509.01366.
- **§3.40 Dual sparse attention (CSA + HCA)** — covers DeepSeek-V4. Source: `modeling_deepseek_v4.py:265, 365, 581`.
- **§3.41 Latent-space MoE** — covers Nemotron 3 Super LatentMoE. Source: NVIDIA NeMo Megatron-LM `latent_moe.py` (external).
- **§3.42 Short-conv mixer** — covers LFM2 family. Source: `lfm2_moe/modeling_lfm2_moe.py:376`.

Add families:
- **§5.49 Qwen3-Next deep-dive** — promote from §5.36 reference into full deep-dive with the 3:1 dispatch verified at `qwen3_next/modeling_qwen3_next.py:972`.
- **§5.50 DeepSeek-V4** — CSA+HCA+hash router, with `modeling_deepseek_v4.py` line citations.
- **§5.51 GLM-MoE-DSA** — sigmoid-routed MoE with group-limited routing + DSA indexer. Source: `glm_moe_dsa/modeling_glm_moe_dsa.py` 47/105/559.
- **§5.52 MiniMax M2** — Mixtral + sigmoid routing. Source: `minimax_m2/modeling_minimax_m2.py:46-296`.
- **§5.53 LFM2 family** — short-conv mixer.

---

## 10. Contradictions / drifts caught between Raschka and source

**Applying the B0.5 lesson.** I cross-checked Raschka's WebFetch-summarized claims against `modeling_*.py` and found these tensions:

1. **Raschka tags Arcee Trinity as "sandwich norm".** The local `arcee/modeling_arcee.py` defines `ArceeDecoderLayer` at line 288 — I did not verify whether the four-norm Gemma-3-style sandwich is present without reading the full source. **Status: unverified; would need a follow-up source-read before promoting to our census.** This is a candidate drift; not yet caught either way.

2. **Raschka tags Kimi Linear as "Kimi Delta Attention".** Not in local transformers tree as of v5.10.2; verification deferred to upstream port. **Status: external-only claim, accept as a hypothesis pending HF in-tree.**

3. **Raschka's WebFetch summary mentioned model versions that are speculative for the gallery cutoff** (e.g., "Qwen3.5 397B", "Qwen3.6 27B/35B-A3B", "DeepSeek V4-Pro 1.6T"). On 2026-06-09, the local transformers tree has `qwen3_5/` and `qwen3_5_moe/` and `deepseek_v4/` modules — so these are real, but **specific parameter counts in the WebFetch summary were not source-verified** and should be re-checked against config.json before being lifted into our census.

4. **Raschka's KV-cache "Very low: 7.9 KiB (Kimi Linear)" and "Very low: 5.4 KiB (V4-Flash)" numbers** are claims I did not verify. The geometry is plausible (Kimi Linear is linear-attention so the cache is O(state_dim) not O(seq_len); V4-Flash uses CSA+HCA dual compression). **Status: arithmetic claim, accept as plausible but flag for spot-check if we surface KV/token in our docs.**

5. **Raschka's "Gemma 4 26B-A4B: 128 routed + 1 shared, top-k=8" matches our verified config.** Our `research/issues/10-gemma4-investigation.md` confirms 128+1 routed/shared with `moe_intermediate_size=704`. ✓ no drift.

6. **Raschka's GPT-OSS "trained attention sinks" matches `gpt_oss/modeling_gpt_oss.py:309`.** ✓ no drift.

7. **Raschka's DeepSeek V4 "hash-based routing" matches `modeling_deepseek_v4.py:1040`.** ✓ no drift; this confirms the axis is real and we should add it.

**No outright contradictions were found**, but several of Raschka's claims fall in the "external-blog-only" category where we don't yet have source backing. The B0.5 discipline says: cite the source code before locking the IR. The above seven items are at varying readiness levels for that discipline.

---

## 11. Bottom-line summary for the kickoff report

- **Patterns we should add to the IR (must-do):** Gated DeltaNet token mixer, CSA+HCA attention, hash-routed MoE.
- **Patterns we should consider (should-do):** Lightning Attention token mixer, Latent-space MoE, Short-conv mixer.
- **Families we should bootstrap:** Qwen3-Next, DeepSeek-V4, GLM-MoE-DSA, MiniMax-M2, possibly Nemotron 3.
- **Docs improvements:** per-family `KV/token`, `Layer mix:`, `Signature:`, `Active %` header lines.
- **Where we win:** source-grounded discipline, 38-axis IR-ready catalog, numerical gates, quant survey, OCR-LLM lineage, hybrid SSM depth.
- **Where Raschka wins:** breadth of frontier MoE coverage, per-card presentation density, currency timestamping.
- **Net:** Raschka's gallery is a useful frontier-scanning tool that surfaces 4-5 distinct IR axes we don't yet model (CSA+HCA, hash routing, shortcut MoE, latent-space MoE, short-conv mixer). It is not a substitute for our source-grounded research, but it is a productive next-pass input.
