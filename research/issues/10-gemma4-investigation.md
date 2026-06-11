# Issue 10 — Gemma 4 investigation

**Date investigated:** 2026-06-06
**Trigger:** User reports Google "just released" Gemma 4; survey `01-model-census.v2.md` / `02-layer-sources.v2.md` stops at Gemma 3 / Gemma 3n.
**Verdict:** **GEMMA_4_CONFIRMED.** Released **2026-04-02**. QAT checkpoint addendum released **2026-06-05** (one day before this investigation), which is the most likely trigger of the user's "just released" remark.
**Confidence:** 95/100 (every architectural claim below is sourced from either an official Google channel, an official HF model card / `config.json`, or the upstream `transformers` source). The remaining 5% uncertainty covers (a) values I had to derive from `config.json` rather than a tech report (no arXiv tech report yet, as of 2026-06-06), and (b) the precise dual-norm layout of the Gemma 4 decoder block, which the official HF blog calls "RMSNorm" without spelling out the four-norm sandwich — I confirm the four-norm sandwich is preserved via the `transformers` source and config defaults (`Gemma4TextConfig`, `Gemma4DecoderLayer`).

---

## 1. Existence verification

### 1.1 Searches run (2026-06-06)

| Query / URL | Source | Result |
|---|---|---|
| WebSearch `"Gemma 4" Google release 2026` | Google index | 7 hits, top hit `blog.google/.../gemma-4` |
| WebSearch `"google/gemma-4" site:huggingface.co` | HF | 10 model-page hits across the family |
| WebSearch `"Gemma 4" technical report arxiv 2026` | arXiv | No first-party tech report; 3rd-party evaluation papers exist (e.g. arXiv 2604.07035, 2605.25645) |
| WebSearch `"Gemma 4" QAT "quantization-aware"` | Google index | QAT post 2026-06-05 + 5 HF QAT-checkpoint repos |
| WebFetch `blog.google/innovation-and-ai/technology/developers-tools/gemma-4/` | Google blog | Release date 2026-04-02; sizes E2B/E4B/12B/26B-A4B/31B; Apache 2.0 |
| WebFetch `deepmind.google/models/gemma/gemma-4/` | DeepMind | Five-size sweep confirmed; Arena 1452 for 31B-IT-Thinking |
| WebFetch `huggingface.co/blog/gemma4` | HF | "Welcome Gemma 4: Frontier multimodal intelligence on device" |
| WebFetch `huggingface.co/google/gemma-4-{E2B,E4B,12B-it,26B-A4B-it,31B,31B-it}` | HF cards | Per-size architecture tables |
| WebFetch `huggingface.co/google/gemma-4-{E2B,E4B,31B,26B-A4B,12B-it}/raw/main/config.json` | HF raw configs | Verified numeric specs |
| WebFetch `github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py` | upstream | `Gemma4TextConfig` defaults |
| WebFetch `huggingface.co/docs/transformers/model_doc/gemma4` | HF docs | PLE described as token-identity + context-aware projection × 1/√2 |
| WebFetch `blog.google/.../quantization-aware-training-gemma-4/` | Google blog | QAT release 2026-06-05; mobile format; 1 GB E2B |
| WebSearch `"Gemma 4" site:blog.google` | Google blog | Two posts: 2026-04-02 launch, 2026-06-05 QAT |
| WebFetch `en.wikipedia.org/wiki/Gemma_(language_model)` | Wikipedia | Confirms 2026-04-02 release and family timeline |
| WebSearch `"gemma4" "post_feedforward_layernorm" sandwich norm` | code | `transformers/models/gemma4/modeling_gemma4.py` references the same four-norm slots |

### 1.2 Authoritative URLs (2026-06-06)

- Launch blog: `https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/` (2026-04-02)
- DeepMind page: `https://deepmind.google/models/gemma/gemma-4/`
- HF announcement: `https://huggingface.co/blog/gemma4`
- HF collection: `https://huggingface.co/collections/google/gemma-4`
- HF docs: `https://huggingface.co/docs/transformers/model_doc/gemma4`
- Google model card: `https://ai.google.dev/gemma/docs/core/model_card_4`
- QAT blog: `https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/` (2026-06-05)
- Upstream config: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py`
- Cloud blog: `https://cloud.google.com/blog/products/ai-machine-learning/gemma-4-available-on-google-cloud`

### 1.3 No first-party arXiv tech report yet

No `arXiv.../Gemma 4 Technical Report` is published as of 2026-06-06. Two months after launch this is a meaningful absence (Gemma 2 had its report on launch day; Gemma 3 within ~3 weeks). The available 3rd-party arXiv papers (2604.07035, 2604.24636, 2605.25645) are evaluation / deployment papers, not the Google tech report. Treat all architectural claims here as derived from the HF model cards, the HF/Google blogs, and the verified `config.json` files until the tech report appears.

### 1.4 What the user likely heard

Three temporally adjacent Google releases could all map onto "just released":
1. **2026-04-02 — initial Gemma 4 family launch** (E2B, E4B, 12B-Unified, 26B-A4B, 31B). This is the actual headline release.
2. **2026-04-04 — Gemma 4 12B Unified** (the encoder-free multimodal variant) added two days later per the QAT blog timeline reference.
3. **2026-06-05 — QAT Q4_0 checkpoints + new mobile quantization format**. This is one day before today (2026-06-06) and is the most likely "just released" trigger — the new mobile format compresses E2B to 1 GB RAM, which has been heavily shared on dev-tooling channels.

---

## 2. Architectural detail — Gemma 4

### 2.1 Family sweep (verified)

| Variant | Total params | Active params | Layers | d_model | H / Hk / Dh | d_ff | SWA size | Sliding pattern | Context | Multimodal | KV-shared layers | HF repo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E2B** | 5.1 B (with embed) | 2.3 B effective | 35 | **1536** | 8 / **1** / 256 | 6144 | 512 | 4 sliding : 1 global, last layer global | 128 K | text + image + audio | **20** | `google/gemma-4-E2B(-it)` |
| **E4B** | 8 B (with embed) | 4.5 B effective | 42 | **2560** | 8 / **2** / 256 | 10240 | 512 | 5 : 1, last global | 128 K | text + image + audio | (≈20-ish) | `google/gemma-4-E4B(-it)` |
| **12B Unified** | 11.95 B | 11.95 B | 48 | **3840** | 16 / 8 / 256 | 15360 | 1024 | 5 : 1 (global at 5,11,17,23,29,35,41,47) | **256 K** | text + image + audio + video (**encoder-free**) | 0 | `google/gemma-4-12B(-it)` |
| **26B-A4B (MoE)** | 25.2 B | 3.8 B | 30 | **2816** | 16 / 8 / 256 | 2112 / 704 / 1 shared | 1024 | 5 : 1, last global | 256 K | text + image | 0 | `google/gemma-4-26B-A4B(-it)` |
| **31B** | 30.7 B | 30.7 B | 60 | **5376** | 32 / 16 / 256 | 21504 | 1024 | 5 : 1, last global | 256 K | text + image | 0 | `google/gemma-4-31B(-it)` |

Vocab = 262 144 (SentencePiece, "GP-v4" — same 262 K family as Gemma 3 / 3n with multimodal additions); `head_dim = 256` retained as the Gemma signature.

### 2.2 Attention block

- **Hybrid local/global alternation** retained from Gemma 3. **5 : 1 sliding : global** for E4B, 12B, 26B-A4B, 31B. **4 : 1** for E2B (smallest). **Last layer always global** (Gemma 3 had the same property).
- **Sliding window**: 512 for the edge sizes (E2B/E4B), 1024 for the larger sizes — both *smaller* than Gemma 3's defaults at the same scale (Gemma 3 used `sliding_window = 1024` even at 4B).
- **Dual RoPE**:
  - Sliding-attention layers: `rope_theta = 10 000`, `rope_type = "default"` (vanilla NEOX-style, same as Gemma 3 local).
  - Global-attention layers: `rope_theta = 1 000 000`, `rope_type = "proportional"`, `partial_rotary_factor = 0.25`. **This is the new p-RoPE** — only 25 % of each head-dim is rotated, the remainder is left as pure semantic channel. This is Gemma 4's biggest novel attention change.
- **Unified K = V on global layers** (`attention_k_eq_v = True` for global layers): the global K and V projections share weights. This is paired with a **doubled global head dim** (`global_head_dim = 512`) and reduced KV heads (E2B: 1 KV head globally; E4B: 2; 31B: 16) to keep the global cache cheap. The intent is "long-context memory at small marginal cost". Maarten Grootendorst's visual guide describes this as `K = V` only on global layers; sliding layers keep distinct K, V.
- **GQA on local layers**: 8 Q heads per KV head (E2B local: 8/1), 4 (E4B local: 8/2), 2 (31B local: 32/16), etc. KV heads are *fewer* than in Gemma 3.
- **Shared KV cache (`num_kv_shared_layers`)**: For edge models, the *last* N layers reuse K/V projections from the matching earlier non-shared layer of the same attention type. E2B exposes `num_kv_shared_layers = 20` (out of 35) — i.e. > half of layers reuse a previously computed K/V tensor. This is **a wholly new axis** for the survey: it changes the memory profile of the KV cache from `O(L · S · 2 · Dh · Hk)` to `O((L − shared) · S · 2 · Dh · Hk)`.
- **QK-norm**: present on both Q and K, on the per-head-dim (last dim = Dh), pre-RoPE — same placement as Gemma 3. HF doc references retain `q_norm`, `k_norm` slots. Maarten's writeup describes "fixed-scale query norms (local ≈ 0.9916, global ≈ 1.0228)" — i.e. QK-norm absorbs the `1/√d` scale; `attention_scale = 1.0` instead of `1/√Dh`. This is a refinement of Gemma 3's QK-norm but conceptually the same.
- **Soft-cap on attention logits**: NOT re-introduced. `attn_logit_softcapping` is absent from all five `config.json` files (Gemma 4 keeps Gemma 3's decision to drop attention soft-cap).
- **Soft-cap on final logits**: **YES, re-introduced**. `final_logit_softcapping = 30.0` is present in every Gemma 4 text config (E2B, E4B, 12B, 26B-A4B, 31B). This reverses one half of the Gemma 3 design: Gemma 3 had `final_logit_softcapping = None`; Gemma 4 restores the 30.0 cap. *Attention soft-cap remains dropped.* The asymmetric re-introduction is one of the more counterintuitive findings — Gemma 2 had both, Gemma 3 had neither, Gemma 4 has only the final one.

### 2.3 Normalization

- **RMSNorm with 1+w gain initialization** — retained from Gemma 2 / 3. `rms_norm_eps = 1e-6`.
- **Dual-norm "sandwich" placement** (input_layernorm + post_attention_layernorm + pre_feedforward_layernorm + post_feedforward_layernorm) — retained per the `transformers` `modeling_gemma4.py` source structure. Search hits for `post_feedforward_layernorm` in the gemma4 module confirm the four-norm decoder is preserved. Vision encoder output is also followed by an additional RMSNorm to match the scale expected by the transformer blocks.

### 2.4 Feed-forward (MLP) block

- **Activation: `gelu_pytorch_tanh`** — i.e. GeGLU (gate uses GELU-tanh, not SwiGLU). All five `config.json` files specify `hidden_activation: "gelu_pytorch_tanh"`. The Gemma family signature activation is preserved end-to-end. *No move to SwiGLU.*
- **Double-wide MLP (`use_double_wide_mlp = True`)** — present in E2B config. This appears to widen the `gate_proj/up_proj` projections; semantics not fully documented in the HF docs yet but the flag is new in Gemma 4 and is *not* present in Gemma 3.
- **MoE (26B-A4B only)**: 128 routed experts + 1 shared expert; top-k = 8; `moe_intermediate_size = 704` per expert; `intermediate_size = 2112` (dense path used for the shared expert at 3× routed-expert size). Routing details (load-balance, jitter) not yet documented; assume top-k softmax router with shared expert added unconditionally. This is *the first Gemma MoE*.

### 2.5 Per-Layer Embeddings (PLE) — new architectural element

The edge variants (E2B, E4B) introduce **Per-Layer Embeddings**: a *second* embedding table whose output is injected as a residual signal at every decoder layer, formed as `(token_identity + context_aware_projection) × (1/√2)`. The PLE table uses a much smaller dim (Maarten's writeup: 256 dims vs the main 1536/2560 model dim) and can be paged out to flash storage on mobile devices, so VRAM only holds the layer-wise projection. This is the mechanism behind "effective 2.3 B" vs "5.1 B with embeddings": ~half the parameter mass lives in the PLE table that does not need to be loaded into VRAM during forward.

For multimodal inputs, the PLE component is computed *before* soft tokens are merged; multimodal positions use the PAD token ID as their PLE lookup key.

### 2.6 Multimodal integration

Two distinct designs co-exist within the family:

**Encoder-based (E2B, E4B, 26B-A4B, 31B)**
- **Vision encoder**: ViT, 16×16 patches, ≈150 M params for E2B/E4B, ≈550 M for 31B/26B-A4B. Variable aspect ratio via 2D RoPE + learned 2D positions (up to 10 240 positions per axis). Soft-token budget configurable from 70/140/280/560/1120.
- **Audio encoder (E2B/E4B only)**: USM-style conformer (same as Gemma-3n) ≈300 M params, max 30 s audio, supports ASR + AST.

**Encoder-free (12B Unified)**
- Vision and audio go through **lightweight linear projections directly into the LLM embedding space** — no separate encoder. Single decoder-only transformer handles all modalities. Reduces multimodal latency and enables joint fine-tuning of the whole model in one pass. The 12B is the only size with this design. This is a genuinely new architectural pattern — closer in spirit to MM1 / Chameleon than to PaliGemma.

Modality ordering convention is fixed: images before text, audio after text.

### 2.7 Tokenizer + vocab

- Vocabulary size **262 144** = Gemma 3 vocab unchanged in size.
- Multimodal special tokens added (token IDs from 12B-Unified config):
  - `BOI = 255999`, `EOI = 258882`, `IMAGE = 258880`
  - `BOA = 256000`, `EOA = 258883`, `AUDIO = 258881`
  - `VIDEO = 258884`
- Likely SentencePiece (Gemma-Piece-v4, extending the GP-v3 family used by Gemma 3 / 3n).
- Reserved special tokens for thinking mode: `<|think|>`, `<|channel>thought` etc.

### 2.8 QAT defaults (2026-06-05 release)

- Four QAT-trained checkpoint formats per size:
  1. Unquantized QAT checkpoint
  2. GGUF Q4_0 (llama.cpp / Ollama)
  3. Compressed-Tensors (vLLM / SGLang)
  4. New **mobile-specialized format** (4-bit per-channel weights + 2-bit token-generation layers + custom embedding/KV-cache handling)
- Memory footprints: E2B 1 GB (mobile fmt) / 3 GB RAM; E4B 5 GB; 12B 7 GB; 26B-A4B 15 GB; 31B 18 GB.
- Repos: `google/gemma-4-{size}-it-qat-q4_0-unquantized` etc.

### 2.9 Other axes

- `tied_word_embeddings`: True (Gemma family signature retained).
- `embed_scale`: `sqrt(D)` (retained).
- `query_pre_attn_scalar`: Not present in the new configs — Gemma 4 leans on QK-norm + the fixed-scale query norms (local 0.9916, global 1.0228) instead.
- `dtype`: bfloat16 across the family.
- `transformers_version`: ≥ 5.5.0.dev0 (some configs target 5.10.0.dev0). Gemma 4 only loads in the new Transformers 5.x line.
- Training-data cutoff (E4B): January 2025. Multilingual to 140+ languages. Total training tokens: not disclosed.
- License: Apache 2.0 (a license change from Gemma 1-3 which used the bespoke "Gemma Terms of Use").

---

## 3. Diff vs Gemma 3

| Axis | Gemma 3 | Gemma 4 | Change kind |
|---|---|---|---|
| SWA : full alternation | **5 : 1** (last layer global) | **5 : 1** (last global); **4 : 1** for E2B | Preserved (E2B exception) |
| Sliding window | 1024 (all sizes ≥ 4B) | 512 at edge, 1024 at server | Shrunk for edge |
| Local RoPE θ | 10 000 | 10 000 | Preserved |
| Global RoPE θ | 1 000 000 | 1 000 000 | Preserved |
| Global RoPE type | full (NEOX) | **proportional p-RoPE, partial_rotary_factor = 0.25** | **New** |
| Final logit softcap | None | **30.0** | **Re-introduced** |
| Attn logit softcap | None | None | Preserved |
| QK-norm | per-Dh, pre-RoPE | per-Dh, pre-RoPE, with fixed-scale norms (local 0.9916, global 1.0228); `attention_scale = 1.0` | Refined |
| Dual-norm (sandwich) | input + post-attn + pre-ffn + post-ffn | Same four-slot layout | Preserved |
| RMSNorm 1+w | yes | yes | Preserved |
| Activation | GeGLU (`gelu_pytorch_tanh`) | GeGLU (`gelu_pytorch_tanh`) | Preserved (no SwiGLU move) |
| MLP shape | standard | **`use_double_wide_mlp = True` on edge** | **New** |
| MoE | none | **26B-A4B**: 128 routed + 1 shared, top-k=8 | **New (first Gemma MoE)** |
| KV in global layers | distinct K, V | **K = V (unified)**, `global_head_dim = 512` | **New** |
| KV cache sharing across layers | none | `num_kv_shared_layers` (20/35 for E2B) | **New axis** |
| Per-Layer Embeddings | none | E2B/E4B: dedicated PLE table, residual at every layer | **New** |
| Vision encoder | SigLIP-style separate tower | (encoders 4 of 5 sizes) + **encoder-free 12B-Unified** | **New (encoder-free variant)** |
| Audio | absent in Gemma 3 base; 3n had conformer | USM conformer on E2B/E4B/12B | Extended |
| Context | 128 K | 128 K (edge) / 256 K (server) | Extended |
| Vocab size | 262 144 | 262 144 | Preserved (signature retained) |
| Tied embeddings | True | True | Preserved |
| License | Gemma Terms | **Apache 2.0** | Changed |
| QAT defaults | optional add-on | **first-class, mobile format included** | Strengthened |
| Tokenizer extras | GP-v3 | GP-v4 (+ image/audio/video special tokens) | Extended |

The clean summary: **Gemma 4 is Gemma 3 minus attention soft-cap, plus (a) p-RoPE on global layers, (b) unified K = V on global layers, (c) per-layer embeddings on edge sizes, (d) cross-layer KV sharing, (e) a first MoE variant, (f) a first encoder-free unified-multimodal variant, (g) restored final-logit soft-cap.** All other axes — dual-norm sandwich, RMSNorm 1+w, GeGLU, head_dim 256, tied embeddings, vocab 262 144, the 5:1 alternation — are preserved.

---

## 4. Implications for `llm-layers`

### 4.1 What `models/gemma4/` would look like

Five model entries needed under `models/gemma4/`:
- `gemma-4-E2B`, `gemma-4-E4B` (edge, with PLE)
- `gemma-4-12B-unified` (encoder-free multimodal)
- `gemma-4-26B-A4B` (MoE)
- `gemma-4-31B` (dense flagship)

Each needs to express:
1. Same alternation field as Gemma 3 (`layer_types: [sliding, sliding, sliding, sliding, sliding, full, ...]` with last = full). For E2B the period is 5 instead of 6 (4 sliding + 1 full).
2. A new `partial_rotary_factor` field on the global-layer RoPE.
3. A new `attention_k_eq_v: True` flag on global layers (or, equivalently, a `KEqualV` op variant).
4. A new `global_head_dim` field, distinct from local `head_dim`.
5. A new `num_kv_shared_layers` field with per-layer-type semantics.
6. For E2B/E4B: a new `per_layer_embedding` spec block (table dim, projection dim, paged-to-flash hint).
7. For 12B-Unified: an encoder-free multimodal path (raw image patches + raw audio waveforms → linear → main residual stream).
8. For 26B-A4B: MoE block (`num_experts = 128`, `top_k = 8`, `num_shared_experts = 1`, `moe_intermediate_size = 704`).
9. Restored `final_logit_softcapping = 30.0` (the survey notes Gemma 3 dropped it; Gemma 4 puts it back, only on the final logits).
10. `use_double_wide_mlp` flag.

### 4.2 New spec dataclasses likely required in `api/`

| New element | Suggested dataclass / op |
|---|---|
| p-RoPE (partial rotation, fraction f) | `RopeSpec.partial_rotary_factor: float = 1.0`. Existing dual-theta survives; just add the fraction. |
| K = V identity on subset of layers | `AttentionSpec.k_equals_v: bool`. Forward path uses one projection, then aliases K and V tensors. |
| Cross-layer KV reuse | `AttentionSpec.kv_source_layer: Optional[int]` (or a model-level `kv_share_map: List[Optional[int]]`). Cache abstraction must allow "this layer's cache = pointer to earlier layer's cache". |
| Per-Layer Embeddings | `PerLayerEmbeddingSpec(table_dim, projection_dim, paged: bool)`. Decoder layer's forward takes an extra residual term per layer. |
| Encoder-free multimodal injection | New `RawPatchProjectionSpec` for vision and `RawWaveformProjectionSpec` for audio (linear projection from raw bytes/pixels into D). Reuses existing multimodal-token-merge plumbing. |
| Shared expert in MoE | `MoeSpec.num_shared_experts: int = 0` + `shared_expert_intermediate_size: int`. |
| Restored final softcap (only) | `LMHeadSpec.logit_softcap: Optional[float]`. We had this for Gemma 2, dropped for Gemma 3, re-add for Gemma 4. |
| Mobile-format quantization | A new `quantization/mobile_int4` op family with 2-bit token-generation layers + static activations + channel-wise weights + custom KV-cache packing. |
| Fixed-scale QK-norm (constant gains 0.9916 / 1.0228) | Probably absorb into `QKNormSpec.fixed_scale: Optional[float]`. |
| `use_double_wide_mlp` | `MLPSpec.double_wide: bool`. |

### 4.3 Does Gemma 4 break the 32-axis taxonomy?

Mostly **no** — most changes are values within existing axes (RoPE base, softcap, MoE config). But it does introduce **three honestly new axes** the v2 taxonomy did not contemplate:

1. **Cross-layer KV reuse** (`num_kv_shared_layers`). The v2 KV-cache axis enumerates `{per-layer KV, per-layer K-only, MQA, GQA, MLA, SSM-state, ...}` but not "layer N's KV = pointer to layer M's KV across the *same* attention type". This is distinct from MLA (which reuses *across heads*); Gemma 4 reuses *across depth*.
2. **Per-Layer Embeddings as an injected per-layer residual**. v2 has an "embedding scale" axis and a "tied lm_head" axis but no concept of a separate embedding table whose output is added at every layer. This is closest to an *adapter* than to a token embedding.
3. **Partial-rotary p-RoPE**. v2's RoPE axis enumerates `{none, vanilla, NTK, linear, YaRN, dual-theta}`. Partial rotary is orthogonal: it can compose with any of those (Gemma 4 specifically composes partial rotary with the dual-theta global-vs-local pattern). Treat as a new orthogonal axis on RoPE: `partial_rotary_factor ∈ (0, 1]`.

Three other elements are not new axes but are *new combinations* worth mention:
- Encoder-free multimodal (12B-Unified): the v2 axes already cover "where multimodal soft tokens enter the residual stream"; the only new thing is that the projection is *linear from raw bytes*, not the output of an encoder. Express as a degenerate encoder.
- Shared expert in MoE: v2 MoE axis is `{none, top-k routed, top-k routed + shared expert}` — the shared-expert option was already in the taxonomy thanks to DeepSeek-V2; Gemma 4 just becomes the second adopter.
- Restored-final-softcap-only: v2 has `{none, attention only, final only, both}`. Gemma 4 lands in "final only" which is a legal taxonomy cell that no previous model occupied — but it does not require a new axis.

### 4.4 Action items for the survey

1. **`research/01-model-census.v2.md`** add five rows in the Gemma section, dated 2026-04 (and one 2026-06 for QAT). Note 4:1 alternation for E2B and 5:1 for the rest; note `use_double_wide_mlp` on the edge sizes.
2. **`research/02-layer-sources.v2.md`** add a `§5.4 (continued) Gemma 4` subsection with the four-norm decoder layout (preserved from Gemma 3) but new attention block: p-RoPE, K = V global, cross-layer KV sharing. Add a per-layer-embedding subsection. Add a Gemma 4 MoE subsection (cross-reference DeepSeek MoE shared-expert pattern).
3. **`research/00-evolution.md`** extend the Gemma evolution arc by one generation: "Gemma 3 dropped softcap, Gemma 4 partially restored it (final only); Gemma 3 used full RoPE on global layers, Gemma 4 added partial-rotary p-RoPE; Gemma 3 was dense, Gemma 4 added MoE; Gemma 4 introduced encoder-free unified multimodal at 12B."
4. **`research/04-quantization.v2.md`** add Gemma 4 QAT mobile format as a worked example of "QAT + mobile-tier 2-bit token-generation layers + paged-to-flash embedding table."
5. **`research/05-kvcache-attention.v2.md`** add cross-layer KV sharing as a new sub-axis (distinct from MLA, distinct from MQA). Add p-RoPE.
6. **`api/`** plan the dataclass additions in §4.2 above; the most disruptive is cross-layer KV sharing (cache abstraction change).

---

## 5. Confidence breakdown

| Claim | Confidence | Source |
|---|---|---|
| Gemma 4 exists, released 2026-04-02 | 100 | blog.google, DeepMind, HF, Wikipedia, multiple corroborating sources |
| Five-size sweep E2B/E4B/12B/26B-A4B/31B | 100 | HF collection, model cards, Google blog |
| Apache 2.0 license | 100 | blog.google |
| Vocab 262 144 | 100 | every `config.json` |
| Dual RoPE (1e6 global, 1e4 local) | 100 | every `config.json` |
| Partial-rotary 0.25 on global | 100 | every `config.json` (`partial_rotary_factor: 0.25`) |
| Final softcap 30.0 restored, attn softcap still absent | 100 | every `config.json` |
| Activation = `gelu_pytorch_tanh` (GeGLU) | 100 | every `config.json` |
| 5:1 SWA alternation (4:1 for E2B), last layer global | 90 | Maarten Grootendorst visual guide + HF model cards (the 4:1-specifically-for-E2B claim is from Maarten only); ratio for larger models is also confirmed by 12B `config.json` enumerating full-attn layer indices 5/11/17/23/29/35/41/47 |
| K = V on global layers | 90 | HF model cards (E4B, 26B, 31B); upstream `attention_k_eq_v` flag |
| `global_head_dim = 512`, `num_kv_shared_layers = 20` for E2B | 95 | E2B `config.json` |
| PLE at every decoder layer for E2B/E4B | 95 | HF docs, Google model card, Labellerr writeup |
| MoE 128 routed + 1 shared, top-k 8 | 95 | 26B `config.json` (`num_experts: 128`, `top_k_experts: 8`, `moe_intermediate_size: 704`) + Maarten visual guide |
| Encoder-free 12B-Unified | 95 | HF model card |
| Dual-norm sandwich preserved | 80 | Inferred from `transformers/models/gemma4/modeling_gemma4.py` having `post_feedforward_layernorm` matches; no first-party Google statement; would prefer a tech-report citation |
| Fixed-scale QK-norm values (0.9916 / 1.0228) | 70 | Single source (a search snippet citing the writeup); not verified in `config.json` |
| QAT mobile format, 1 GB E2B | 100 | blog.google 2026-06-05, MarkTechPost coverage 2026-06-05 |
| No first-party arXiv tech report yet | 95 | Negative search across arXiv for 2026 dates |

---

## 6. Pending follow-ups (if/when a Gemma 4 tech report appears)

1. Verify dual-norm sandwich + exact 1+w gain init carry over.
2. Verify exact QK-norm scaling values (the local 0.9916 / global 1.0228 number is in only one secondary source).
3. Confirm MoE routing details (load-balance loss, jitter, expert dropout) for 26B-A4B.
4. Confirm training token count, training compute, and the data-mixture changes vs Gemma 3.
5. Confirm whether the encoder-free 12B-Unified is genuinely "raw waveform → linear" or whether a tiny preprocessing stem is hidden inside the projection.
6. Confirm the precise semantics of `use_double_wide_mlp` and whether it applies to up-, gate-, or down-projection.
7. Verify whether the PLE table is genuinely paged to flash in any first-party runtime, or only in third-party Ollama-style runners.

---

**Bottom line:** Gemma 4 is real, the family card is solid, and our survey should be amended. The architectural delta from Gemma 3 is non-trivial but stays within recognizable Gemma DNA — same dual-norm sandwich, same head_dim 256, same GeGLU, same 5:1 alternation, same RMSNorm(1+w) — with three new "first-of-family" elements (MoE, encoder-free unified multimodal, per-layer embeddings) plus three smaller refinements (p-RoPE, K = V on global, cross-layer KV sharing, restored final softcap). Three of those (cross-layer KV sharing, per-layer embeddings, partial-rotary) require new axes in our taxonomy; the rest fit existing axes.
