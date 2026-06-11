# 05 — KV Cache and Attention Variants Across Mainstream SLM Runtimes (v3)

**Stream:** Attention variants & KV cache designs
**Goal:** Distill the parameter axes a minimal SLM API must expose to express every shipped attention block, RoPE scheme, and KV-cache layout in mainstream small-language-model runtimes — extended through 2026-H1.
**Scope:** Mainstream ≤ ~14 B parameter models, plus 8-30 B "axis-load-bearing" entries whose architectural deltas trickle down to the SLM lineage. 2026 axis-load-bearing additions: Gemma 4 E2B/E4B/12B/26B-A4B/31B, DeepSeek-V3.2 & V4, Llama 4 Scout/Maverick, Qwen3-Next, Granite 4 H, MiniMax-Text-01 / M2 / M3.0, NVIDIA Nemotron 3 hybrid family, Mamba-3, Apple AFM, DeepSeek-OCR, GPT-OSS, Mistral Small 3.x. Runtimes covered: HF transformers, vLLM v1, SGLang, llama.cpp, MLX, MLC-LLM, TensorRT-LLM, FlashInfer, Core ML, ExecuTorch, ONNX Runtime GenAI.
**Revision history.** v2 (2026-06-04) incorporated the `05-kvcache-critique.md` review, fixing seven factual errors, adding ~12 attention variants, ~6 RoPE schedules, ~6 cache layouts, plus a chronological evolution narrative for each axis. **v3 (2026-06-06)** extends v2 with the 2025-H2 + 2026-H1 wave: nine new attention variants, four new RoPE variants, five new KV-cache layouts, and refreshed evolution arcs through MiniMax M3.0 and Nemotron 3 Ultra (2026-06-04). Counts grow from 29 → 38 attention, 14 → 18 RoPE, 13 → 18 cache. New axes added: `partial_rotary_factor`, `attention_k_eq_v`, `num_kv_shared_layers`, `per_layer_embedding`, `share_scheme`, `block_bidirectional_mask`, `lightning_indexer`, `mrope_section`, `is_2d`, `apply_rope_per_layer`. Two new compatibility-matrix rows track cross-layer KV sharing and Lightning-Indexer DSA.

---

## §1 Introduction and Evolution Arc

A `forward()` on a transformer block is, to first approximation, three function calls: (1) build Q / K / V projections from the residual stream, (2) run an attention or attention-equivalent token-mixer against a cache, (3) feed-forward. Every shipped SLM differs along five orthogonal axes inside that second step:

1. **Projection shape** — number of Q heads vs K/V heads vs latent dim (MHA / GQA / MQA / MLA), whether Q, K, V share a single `head_dim` or fan out into three independent dims, and — newly tracked in v3 — whether K and V share a single projection (`attention_k_eq_v = true` on Gemma 4 global layers).
2. **Positional embedding** — where the rotation goes, on which dims, with which θ schedule, with which extrapolation scheme, **and on which fraction of the head dim it operates** (Gemma 4's `partial_rotary_factor = 0.25` per global layer is the new datapoint).
3. **Mask shape** — full causal, sliding-window, sink-augmented, sparse (NSA / DSA / MSA / CSA+HCA), block-diagonal (cross-doc packing), or block-bidirectional (DeepSeek-OCR visual flow).
4. **Cache layout** — how K and V (or compressed latent `c_kv`, or SSM state `h`, or 1-D conv state `conv_state`, or complex-valued Mamba-3 state, or indexer state) are stored across time, across layers, across requests, and **across depth** (Gemma 4 cross-layer KV sharing; Apple AFM 2-block sharing).
5. **Norm placement** — pre/post on the residual; pre/post on Q and K relative to RoPE; and the new four-norm sandwich placement carried into Gemma 4.

An API that wants to express *all* of these without per-model special casing must treat each as an axis, not a flag. Sections §3–§5 enumerate the values observed in production runtimes; §6 separates prefill / decode / chunked-prefill kernel concerns; §7 collapses to the minimum axis set; §8 documents the compatibility / constraint matrix; §9 lays out the dated evolution arc per axis; §10 surveys forward-looking research; §11 collects citations.

**The four parallel evolution arcs (refreshed through 2026-H1).** Each arc is detailed with dates and citations in §9.

- **Attention arc:** MHA (Vaswani 2017) → MQA (Shazeer 2019 / PaLM 2022) → GQA (Ainslie et al. 2023, Llama-2-70B Jul 2023) → SWA (Mistral 7B Sep 2023) → MLA (DeepSeek-V2 May 2024) → CLA (Brandon et al. May 2024) → Differential Attention (Ye et al. Oct 2024) → SWA / global alternation as default (Gemma 2 Jun 2024, Gemma 3 Mar 2025) → trained sink (GPT-OSS Aug 2025, Mistral-Small-3.1 Mar 2025) → NSA trainable sparse (DeepSeek Feb 2025) → iRoPE / NoPE per-layer (Llama-4 Apr 2025) → SSM hybrids (Mamba 2023 → Jamba Mar 2024 → Mamba-2 May 2024 → Hymba Nov 2024 → Samba Jun 2024 → Phi-4-mini-flash 2025 → Qwen3-Next Aug 2025 with Gated DeltaNet 3:1) → **Lightning Indexer DSA (DeepSeek-V3.2 Sep 2025)** → **MiniMax Lightning Attention 7:1** (MiniMax-Text-01 Jan 2025) → **Granite 4 Mamba-2/attention 9:1** (Oct 2025) → **K = V global-layer unification (Gemma 4 Apr 2026)** → **CSA + HCA (DeepSeek-V4 Apr 2026)** → **Mamba-3 complex-state + MIMO decoding (Mar 2026)** → **Visual Causal Flow / block-bidirectional mask (DeepSeek-OCR Oct 2025)** → **MSA — MiniMax Sparse Attention (MiniMax-M3.0 Jun 2026)**.
- **RoPE arc:** vanilla RoPE (Su et al. RoFormer 2021, LLaMA 1 Feb 2023) → PI / linear-scaling (Chen / Kaiokendev "SuperHOT" Jun 2023) → NTK-aware (bloc97 Jul 2023) → Dynamic NTK (ChatGLM-2 Jul 2023, InternLM-200K Aug 2023) → YaRN (Peng, Quesnelle, Kingma Sep 2023) → LongRoPE per-dim learned (Phi-3 Apr 2024) → Llama-3 smooth-wavelength (Apr 2024) → DCA (Qwen2.5-1M Jan 2025) → per-layer θ alternation (Gemma 3 Mar 2025) → **iRoPE per-layer apply_rope boolean (Llama-4 Apr 2025)** → NoPE-alternation (SmolLM3 Jul 2025) → **p-RoPE / partial-rotary RoPE on global layers (Gemma 4 Apr 2026, `partial_rotary_factor = 0.25`)** → **M-RoPE / 2D-RoPE / V2PE expanded (Qwen2.5-VL, Qwen3-VL)** with `mrope_section` and explicit `is_2d` flag, and **TMRoPE** (Qwen2.5-Omni 2025).
- **KV cache arc:** contiguous (HF baseline 2020-2022) → PagedAttention (vLLM Sep 2023) → quantized KV (KIVI Mar 2024, KVQuant Mar 2024) → MLA-compressed latent (DeepSeek-V2 May 2024) → YOCO 2-stage architecture (Microsoft May 2024) → RadixAttention prefix tree (SGLang May 2024) → vAttention CUDA-VM mapping (MSR-India May 2024) → NSA three-tier (DeepSeek Feb 2025) → MTP-head cache (DeepSeek-V3 Dec 2024) → MLA paged + prefix (vLLM v0.6.3 Sep 2024) → **Apple AFM cross-block KV sharing (2 blocks, 62.5/37.5 split, 2025)** → **Lightning Indexer separate state (DeepSeek-V3.2 Sep 2025)** → **Gemma 4 cross-layer KV sharing (`num_kv_shared_layers = 20`, E2B Apr 2026)** → **Gemma 4 Per-Layer Embeddings cache, paged-to-flash (E2B/E4B Apr 2026)** → **Mamba-3 complex-state cache (Mar 2026)** → **CSA + HCA dual-compressed cache, 10 % of V3.2 KV at 1M (DeepSeek-V4 Apr 2026)**.
- **Hybrid (token-mixer) arc:** pure attention (2017-2023) → Mamba pure SSM (Dec 2023) → Jamba MoE+Mamba+attn (Mar 2024) → Mamba-2 SSD (May 2024) → Zamba2 shared-attention (Jun 2024) → Samba alternating Mamba/SWA (Jun 2024) → RecurrentGemma Griffin/Hawk (Jun 2024) → Hymba parallel SSM+attn heads (Nov 2024) → Phi-4-mini-flash per-layer interleave (2025) → **Granite 4 H 9:1 Mamba-2 : attention (Oct 2025)** → Qwen3-Next Gated DeltaNet (Aug 2025, **3:1 linear:softmax**) → **MiniMax-Text-01 Lightning 7:1 linear:softmax (Jan 2025)** → **NVIDIA Nemotron 3 Super (120B/12B-A) and Nemotron 3 Ultra (550B/55B-A) hybrid Mamba-2 + transformer MoE (Mar–Jun 2026)** → **Mamba-3 complex-state + MIMO (Mar 2026)**.

The remainder of this document is the cross-section of those four arcs at the point where they bear on runtime API design. Every 2026-H1 entry is integrated into the relevant §3 / §4 / §5 section and tagged "[v3 NEW]" in the heading.

**What changed from v2 to v3, at a glance.** Nine new attention variants (Lightning Indexer DSA, CSA, HCA, MSA, Visual-Causal-Flow block-bidirectional, K = V unification, MiniMax Lightning 7:1, Mamba-3 complex-state, GDN 3:1) → 38 total. Four new RoPE variants (p-RoPE, iRoPE per-layer, expanded M-RoPE / 2D / TMRoPE, V2PE-2D) → 18 total. Five new cache layouts (cross-layer KV sharing à la Gemma 4, Apple AFM cross-block, Per-Layer Embeddings flash-paged, Lightning Indexer separate state, Mamba-3 complex cache) → 18 total. Three new constraint-matrix rows. Refreshed §9 with 2025-H2 + 2026-H1 entries. New `KVCacheSpec` and `AttentionSpec` fields enumerated in §7. One whole new §3 subsection (vision-modality attention masking) for DeepSeek-OCR.

**One non-architectural editorial note carried over from v2.** The audit `08-coverage-justification.md` rates v2 "near-complete for dense decoders, materially-incomplete for multimodal / OCR / audio / latest-2026." v3 closes the 2026 attention/RoPE/cache gap on the dense + hybrid axes but does **not** pull every OCR-LLM and VLM family into §3; those belong in `02-layer-sources.v3.md`'s §5.34 OCR-LLM evolution chapter (a planned but distinct deliverable). What v3 does pull from OCR is the **visual causal flow** masking primitive itself, because it is a first-class attention semantics axis.

---

## §2 Methodology

v3 inherits v2's methodology — direct verification of HF source files, citation of upstream config defaults, dated overclaim cleanup — and additionally:

- **Re-verified Gemma 4 architectural claims against `config.json` raw files** for E2B and 31B (the two ends of the family) as of 2026-06-06. Confirmed `num_kv_shared_layers = 20` and `attention_k_eq_v = false` on E2B; `num_kv_shared_layers = 0` and `attention_k_eq_v = true` on 31B; `partial_rotary_factor = 0.25` on the `full_attention` layer-type in both; `final_logit_softcapping = 30.0` on both; `attn_logit_softcapping` absent in both (the value of 50.0 found in the configs belongs to the **audio sub-module**, not the text module). The Gemma 4 `K = V global-layer unification` claim is **size-dependent**: applies on the larger sizes (≥ 12 B), not on E2B. This is a non-trivial correction to v2's secondary-source statements and is integrated into §3.
- **Verified DeepSeek-V4 (CSA + HCA) and DeepSeek-V3.2 (Lightning Indexer DSA)** existence and high-level FLOPs / cache-savings numbers against the prompt's input briefs (`08-recent-releases-2026q2.md` and `08-coverage-justification.md`) and the public arXiv abstracts (arxiv:2512.02556 for V3.2; the V4 tech report is in the `huggingface.co/deepseek-ai/DeepSeek-V4-Pro` repo). Specifics on the inner architecture of the Lightning Indexer (a learned per-query token-relevance scorer driving top-k cache selection) are described from the public abstract + community write-ups; precise `top-k`, indexer dim, and gating rules are flagged as `[VERIFY against the full tech report]`.
- **Verified Llama 4 iRoPE** against `meta-llama/Llama-4-Scout-17B-16E` and the Meta launch blog (the raw `config.json` is gated; what is recoverable from the public model card and the HF blog is: per-layer NoPE / RoPE alternation, attention temperature tuning, and the "no positional embedding at all" property of NoPE layers).
- **Verified Mamba-3** existence and high-level claims (complex-valued state update, MIMO decoding, refined discretization) against the ICLR 2026 paper arxiv:2603.15569; precise complex-state dim and MIMO mechanics are flagged `[VERIFY]` because the abstract does not give them and the full PDF requires further fetching.
- **Cross-referenced** `08-coverage-justification.md` for the audit's "must-add" list and confirmed that the v3 additions hit every "axis-load-bearing" entry from that audit's §2.

Open verifications introduced by v3 are collected in Appendix C; new ones are tagged "v3-new" in that appendix.

---

## §3 Attention variants — 38 entries (29 in v2 → 38 in v3)

§3.1–§3.29 are identical to v2 §3.1–§3.29 modulo small typo edits (those sections are preserved verbatim below for completeness). §3.30–§3.38 are v3 additions, all 2025-H2 or 2026.

### 3.1 Multi-Head Attention (MHA) — Vaswani et al. 2017

The classical block. Per-layer parameters `W_q, W_k, W_v ∈ R^{d × d}` where `d = n_heads · head_dim`. Q, K, V each shape `[B, H, T, d_h]`. Cache holds `K, V ∈ [B, H, T_cache, d_h]`. Scaled-dot-product softmax with scale `1/√d_h`.

**Shipped (≤14B scope):** GPT-2, Pythia, OLMo-1, Phi-2, Phi-3-mini-4K (`Phi3Config.num_key_value_heads = num_attention_heads = 32`), StableLM-2-1.6B, SmolLM2-135M / 360M, original BERT-style encoders, TinyLlama 1.1B.

### 3.2 Multi-Query Attention (MQA) — Shazeer 2019, PaLM 2022

Special case `n_kv_heads = 1`. Cache per token is `2 · d_h` bytes — pathologically small. Quality regression is real at >2 B unless trained from scratch with MQA.

**Shipped:** Falcon-7B, Falcon-40B, Falcon-180B (all MQA, alibi=false; positional encoding is RoPE on Falcon-180B per its model card, learned-positional on 7B/40B), StarCoder-1, PaLM (Google), Gemma-3-1B (`num_attention_heads=4, num_key_value_heads=1, head_dim=256` — strictly MQA per the G=1 definition; v1 mislabeled this as GQA). **Gemma 4 E2B sliding layers are also strictly MQA** (`num_attention_heads = 8, num_key_value_heads = 1, head_dim = 256`) — verified in `google/gemma-4-E2B/config.json`. This is the rare case of a 2026 flagship still shipping pure MQA at an SLM scale, justified by the cross-layer KV sharing (§3.31 / §5.14) which already amortizes the cache cost.

### 3.3 Grouped-Query Attention (GQA) — Ainslie et al. 2023

`n_kv_heads = G` where `1 < G < H`. K, V are `[B, G, T, d_h]`. At attention time, K/V are repeated `H/G` times along the head axis (or, in fused kernels, indexed by `head_idx // (H/G)`). FlashAttention-2 natively supports `kv_heads ≠ q_heads`; FlashAttention-3 makes the broadcast a kernel intrinsic. Trades `1/g` cache memory for negligible quality loss when `g ≤ 8`.

**Shipped:** Llama-3.x (1B / 3B / 8B: H=32, G=8; 70B: H=64, G=8; 405B: H=128, G=8). Qwen3-0.6B (H=16, G=8 — i.e. G=H/2), Qwen3-4B (H=32, G=8 — i.e. G=H/4), Qwen3-8B (H=32, G=8), Qwen3-14B (H=40, G=8) — v1's "G=H/4 typical" understated the variance; the actual rule is `G=8` for all Qwen3 dense variants and varies as a fraction of H. Mistral-7B-v0.2+, Gemma-2-2B (H=8, G=4 = H/2), Gemma-2-9B (H=16, G=8), Gemma-3-4B / 12B (G=H/4 typical), Phi-3.5-MoE (H=32, G=8), Phi-4-mini (H=24, G=8), MiniCPM-3 dense-attention layers, Granite-3.0, OLMo-2-1B (H=16, G=4), SmolLM2-1.7B (H=32, G=8), Mistral-Small-3.1. **2026-H1 additions:** Gemma 4 12B (H=16, G=8), 26B-A4B (H=16, G=8), 31B (H=32, G=16) on sliding layers; Llama 4 Scout's attention-bearing layers (the non-NoPE layers); Granite 4 H-class attention layers; Nemotron 3 Super / Ultra Hybrid attention layers; Qwen3-Next softmax-attention layers; DeepSeek-V4 dense parts; MiniMax M3.0 softmax-attention layers.

### 3.4 Multi-head Latent Attention (MLA) — DeepSeek-V2 (Liu et al. 2024 §2.1)

Replace K, V with a shared low-rank latent `c_kv ∈ R^{d_c}` per token, where `d_c ≈ 4 · d_h` (DeepSeek-V2 paper uses `kv_lora_rank = 512`, head_dim 128). At attention time:

```
K_compressed = c_kv @ W_uk        # [d_c] → [H, d_nope]
V_compressed = c_kv @ W_uv        # [d_c] → [H, d_v]
```

The trick: `W_uk` is **absorbed into `W_q`** so `Q · K^T = (Q · W_uk^T) · c_kv^T`, meaning the cache only needs to store `c_kv` per token rather than `H · d_h`. RoPE does not commute with the absorption (rotation through a learned linear map is not equivalent to rotation pre-map), so MLA splits each head into:
- `qk_nope_head_dim` non-rotated dims served from latent (DeepSeek-V2: 128, MiniCPM-3: 64),
- `qk_rope_head_dim` rotated dims served from a separate per-token single-head K cache `k_pe` (DeepSeek-V2: 64, MiniCPM-3: 32),
- `v_head_dim` (independent of `qk_head_dim`) for V (DeepSeek-V2: 128, MiniCPM-3: 64).

**Critical asymmetry corrected from v1:** `q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192` (DeepSeek-V2) is **not equal to `v_head_dim = 128`**. The single `head_dim` axis assumed by v1's `AttentionSpec` is insufficient. The minimum spec needs `q_head_dim`, `k_head_dim` (== q in DeepSeek configs), `v_head_dim` as three independent fields. FlashAttention-2 v2.5+ and FA3 support asymmetric head_dim; older CUTLASS kernels do not.

**Per-token cache footprint:**
- DeepSeek-V2: `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` fp16 = 1.15 KB.
- MiniCPM-3: `kv_lora_rank + qk_rope_head_dim = 256 + 32 = 288` fp16 = 576 B.

This is the **only** mainstream variant before V3.2 / V4 that breaks the "K and V are independent rank-`H · d_h` tensors" assumption a textbook API normally makes.

**Shipped:** DeepSeek-V2 / V2.5 / V3 (671B / 37B-active), DeepSeek-V2-Lite (16B / A2.4B — the only mainstream "SLM-ish" MLA), **MiniCPM-3-4B** (verified against `openbmb/MiniCPM3-4B/config.json` as of 2026-06-04: `kv_lora_rank=256, q_lora_rank=768, qk_nope_head_dim=64, qk_rope_head_dim=32, num_attention_heads=40, num_hidden_layers=62`. **v1 mis-classified MiniCPM-3 as CLA / cross-layer-pair-sharing; it is MLA.**). **DeepSeek-V3.2 and DeepSeek-V4 are MLA + Lightning-Indexer or MLA + CSA/HCA** (see §3.30 / §3.31), so the MLA latent layout persists, but is now augmented with separate index / compression caches.

### 3.5 Sliding-Window Attention (SWA) — Beltagy et al. Longformer 2020, Mistral 7B Sep 2023

Causal mask additionally constrained to `i − j < W`. Cache bounded by `W` tokens per layer per request (with caveats for prefill that must hold the full prompt to honor cross-attention from outside the window).

**Shipped:** Mistral-7B-v0.1 (W=4096, all layers), Mistral-Nemo, Mistral-Small-3 (some layers SWA), Phi-3-mini-128k (blocksparse + SWA hybrid), some Mistral-Small variants.

### 3.6 Interleaved Local-Global Attention — Gemma 2, Gemma 3, **Gemma 4 [v3-refresh]**

Layer-dependent mask: most layers are SWA-local, every `k`-th layer is full-causal global.

**Shipped:** Gemma 2 (1:1 alternation, W=4096 local), Gemma 3 (5:1 ratio, W=1024 local), Cohere Command-R7B (1:3), Mistral-Small-3 (some layers), Ministral 3B / 8B (interleaved SWA, the third distinct alternation pattern in the corpus after Gemma 2's 1:1 and Gemma 3's 5:1).

**Gemma 3 specifics:** every 6th layer is global full-attention; the other 5 are SWA W=1024. The local layers use `rope_theta = 10_000`, the global layers use `rope_theta = 1_000_000`. `rope_theta` is therefore **per-layer**, not per-model.

**Gemma 4 specifics [v3 NEW]** (verified against `google/gemma-4-E2B/config.json` and `google/gemma-4-31B/config.json` 2026-06-06):

| Size | Layers | Pattern | Sliding W | Layer types | Sliding RoPE θ | Global RoPE θ | partial_rotary_factor |
|---|---|---|---|---|---|---|---|
| E2B | 35 | 4 : 1, every 5th layer is global; layers 5, 10, 15, 20, 25, 30, 35 (last layer always global) | 512 | sliding_attention / full_attention | 10 000 | 1 000 000 | 0.25 (full only) |
| E4B | 42 | 5 : 1, every 6th global, last always global | 512 | same as E2B | 10 000 | 1 000 000 | 0.25 |
| 12B | 48 | 5 : 1, every 6th global | 1024 | same | 10 000 | 1 000 000 | 0.25 |
| 26B-A4B | 30 | 5 : 1, every 6th global | 1024 | same | 10 000 | 1 000 000 | 0.25 |
| 31B | 60 | 5 : 1, every 6th global, twelve repeats | 1024 | same | 10 000 | 1 000 000 | 0.25 |

E2B is the only Gemma 4 size with a 4:1 pattern; all others stay on Gemma 3's 5:1 baseline. The window itself is smaller for E2B/E4B (512 vs 1024). The new global-layer feature is **p-RoPE** (§4.15) — only 25 % of each global-layer head_dim is rotated. The other Gemma 4 deltas (`global_head_dim = 512`, `attention_k_eq_v = true` on ≥12B but `false` on E2B, `num_kv_shared_layers = 20` on E2B but `0` on ≥12B) are covered separately in §3.31, §3.32, §5.14.

**Verifications:** the prompt's framing of "K = V unification on global layers" applies to Gemma 4 12B / 26B-A4B / 31B (`attention_k_eq_v = true`), but **not** to the smallest E2B (`attention_k_eq_v = false`). This is the cleanest correction from the raw configs — `08-coverage-justification.md` and the `10-gemma4-investigation.md` source secondary descriptions had inferred a family-wide property from a visual writeup. v3 reports the size-dependent reality.

### 3.7 Sink Attention — StreamingLLM (Xiao et al. 2024) + Trained-Sink variants [v3-expanded]

First `k` tokens are always visible regardless of sliding-window position. Bounds cache at `W + k` per layer.

**Trained-sink lineage (v3 makes this a first-class §3 subsection):**
- **OpenAI GPT-OSS family (Aug 2025, gpt-oss-20b / gpt-oss-120b)** trains with sink slots from scratch (`attention_sink_size` exposed in the config). The sink is paired with **MXFP4 weight quantization** and a GPT-3-style alternating dense + banded-sparse attention pattern. The model **does not function without the sinks** (verified by community ablations) — they are now load-bearing weights, not an inference-time prefix.
- **Mistral Small 3 / 3.1 (Jan / Mar 2025)** ships `attention_sink_size: 4` as a trained parameter; the Small 3 introduces the recipe and 3.1 packages it with a Pixtral vision tower and extended-RoPE 128 k context.
- **EfficientStreaming (Han et al. 2024)** explicitly trains with sinks; resulting models degrade without them.
- **StreamingLLM original recipe (Xiao et al. 2024)** remains a pure inference-time recipe for any SWA model.
- **Gemma 3 and Gemma 4 do not train sinks** (verified, no sink reservation in `gemma3/modeling_gemma3.py` or `gemma4/modeling_gemma4.py`).

**Corrected v3 claim:** "Sink attention started life as an inference-time recipe (StreamingLLM 2023). By 2025-H2, GPT-OSS, Mistral Small 3.x, and a growing set of long-context production models bake it into training as a load-bearing axis. Models trained with sinks do not function without them; models trained without sinks can still benefit from sinks at long-context inference but with a quality penalty. v3 treats `sink_trained: bool` as a first-class field in `AttentionSpec`."

### 3.8 Block-Diagonal (Packed Cross-Document) Attention

Mask is block-diagonal where each block is a packed pretraining sequence. Used in cross-document packed pretraining (the `packing` argument to most pretraining loaders). Runtime concern for vLLM continuous batching when packing is done at serve time.

### 3.9 Native Sparse Attention (NSA) — DeepSeek (Yuan et al. arxiv:2502.11089, Feb 2025)

Trainable hierarchical sparse attention with three branches per layer. Each token attends to:
1. **Compressed branch:** anchor tokens that summarize blocks of size `l_cmp = 32` tokens (compression-pool + small MLP per block).
2. **Selected branch:** the top-`n_sel = 16` blocks (of size `l_sel = 64`) by router score, attended at full resolution.
3. **Sliding branch:** local window of size `W_sliding = 512`.

A learned router scores blocks for the selected branch. Three caches per layer per request:
- `cmp_cache`: shape `[B, H_kv, T_cmp, d_h]` where `T_cmp = ⌈T / l_cmp⌉` — compressed anchors.
- `sel_cache`: shape `[B, H_kv, T, d_h]` at full resolution; only the selected blocks are read each step.
- `swa_cache`: shape `[B, H_kv, W_sliding, d_h]` ring buffer.

The cache footprint sums to roughly `(1/l_cmp + 1 + W_sliding/T) · 2 · H_kv · d_h` per token — at 64k context, NSA stores roughly 5% of full-attention KV (paper §4 Table 2 reports 11×–58× decoding speedup depending on context length).

**Shipped:** DeepSeek-V3.x technical preview, NSA reference implementation. **Standard SDPA / FA2 / FA3 cannot serve NSA without a custom kernel** (the gated branch-mix and the per-step block selection require bespoke gather logic). NSA was the first-of-three "learned-sparse-attention" milestones; DSA (§3.30) and CSA + HCA (§3.31) are the 2025-H2 / 2026-H1 successors with even smaller cache footprints.

### 3.10 Cross-Layer KV Sharing (CLA) — Brandon et al. arxiv:2405.12981, May 2024

**Multiple consecutive layers share one KV cache.** Only the producer layer computes K, V; downstream layers read from it. Distinct from MLA (which compresses within a layer) and from YOCO (which uses a 2-stage architecture). Distinct from the **Gemma 4 cross-layer KV sharing** (§3.32) which uses `num_kv_shared_layers` to point a downstream layer at an earlier-layer cache **of the same attention type** rather than chaining adjacent layers; v3 treats CLA, Gemma-4-style, Apple-AFM-style, and YOCO as four sub-variants of a unified `share_scheme` enum (see §7).

**Shipped:** CLA reference implementation, **Apple OpenELM** (additionally has per-layer width scaling), some Mistral fine-tunes. **v1 incorrectly attributed CLA to MiniCPM-3.** MiniCPM-3 uses MLA, not CLA.

### 3.11 YOCO — You Only Cache Once (Sun et al. arxiv:2405.05254, Microsoft May 2024)

A **two-stage architecture**, not a layer-sharing scheme:
- **Stage 1 (self-decoder):** the first `L/2` layers each compute their own attention and produce *one shared global KV cache* at the boundary.
- **Stage 2 (cross-decoder):** the second `L/2` layers do cross-attention against the stage-1 global cache. No layer in stage-2 maintains its own KV cache.

Cache layout: a single global `(K, V) ∈ [B, n_kv_heads, T, d_h]` pool plus per-layer cross-attention scratch (negligible). Per-token cache is `2 · n_kv_heads · d_h` regardless of layer count — i.e., the cache no longer scales with depth. For a 32-layer model, YOCO is roughly 32× cache-efficient versus baseline at full attention.

**Shipped:** YOCO-1.3B reference release, internal Microsoft variants used in early Phi-4-mini-flash experiments. Apple AFM's 2-block scheme (§5.14) is a YOCO-relative refinement that splits at 62.5 / 37.5 rather than 50 / 50.

**Constraint:** YOCO + speculative decoding requires the stage-1 cache to be re-runnable for accepted-but-not-yet-committed tokens; see §8.

### 3.12 Differential Transformer (Ye et al. arxiv:2410.05258, Oct 2024)

Each head computes **two** softmax attention maps and subtracts them:
```
attn = softmax(Q1 K^T / √d_h) − λ · softmax(Q2 K^T / √d_h)
```
Doubles Q projections (2 × `d_q`), keeps K, V single. The scalar `λ_init` is initialized per layer; later layers can use larger λ.

**Shipped:** Diff-Transformer reference release (1.3B / 6.8B); used in early Phi-Diff experiments. Not in a flagship SLM yet.

### 3.13 Mamba-1 (Gu & Dao arxiv:2312.00752) — Selective State-Space Model

The recurrence is `h_t = A_t · h_{t−1} + B_t · x_t`, `y_t = C_t · h_t + D · x_t`, where `A_t, B_t, C_t, Δ_t` are **input-dependent** (selective). Concretely:

```python
x_t            = conv1d_state(conv_state, x_t)            # 1-D causal conv, kernel d_conv
A_log, B, C, D = self.x_proj(x_t).split(...)
delta          = softplus(self.dt_proj(B))                # input-dependent step
A              = -exp(A_log)                              # diagonal stable
h_t            = exp(delta · A) · h_{t-1} + (delta · B) · x_t
y_t            = C · h_t + D · x_t
```

**Cache state per layer:**
- `ssm_state ∈ [B, d_inner, d_state]` — the recurrent hidden state. Typical: `d_state = 16`, `d_inner = 2 · hidden_size`.
- `conv_state ∈ [B, d_inner, d_conv - 1]` — **1-D causal-conv state for the input mix.** Without it, Mamba inference cannot resume mid-sequence.

The selective scan kernel parallelizes the recurrence via chunk-parallel-scan: chunks of `chunk_size` tokens are processed with a Brent-Kung scan inside each chunk, then a cross-chunk associative-scan; this is what `selective_scan_fn` (CUDA) exposes.

**Shipped pure-Mamba:** Mamba-2.8B reference, Falcon-Mamba-7B, **Cobra** vision-language model. RWKV-5/6 use a parallel-form recurrence (closely related family).

### 3.14 Mamba-2 — SSD form (Dao & Gu arxiv:2405.21060, May 2024)

Mamba-2 recasts the SSM as **structured matrix multiplication** (the "state-space duality" form). The key change for the API:

- `A` is restricted to scalar-times-identity per head (rather than diagonal); this enables a head-grouped formulation.
- Recurrence is computed by **chunk** (typical `chunk_size = 256`) using one matmul per chunk plus a cross-chunk inter-block scan; this is 2–8× faster than Mamba-1's selective scan on Hopper.
- Per-token outputs: `headdim` and `ngroups` are new axes. `headdim = 64` typical, with `ngroups ∈ {1, 2, 4, 8}` shared `B, C` across `n_heads / ngroups` heads.

**Cache axes added:** `chunk_size`, `headdim`, `ngroups`. The `ssm_state` shape becomes `[B, n_heads, headdim, d_state]`, and the `conv_state` shape `[B, d_inner, d_conv - 1]` is unchanged.

**Shipped:** Mamba-2 reference, Zamba-2 1.2B / 2.7B / 7B (Zyphra), portions of Hymba (NVIDIA), Granite-4-tiny-hybrid (IBM), Granite 4 H-Micro / H-Tiny / H-Small (Oct 2025), Nemotron 3 Super (Mar 2026), Nemotron 3 Nano Omni (Apr 2026), Nemotron 3 Ultra (Jun 2026), Falcon-H1 (Aug 2025), Jamba2 (Jan 2026).

### 3.15 Gated Linear Attention (GLA) — Yang et al. arxiv:2312.06635, late 2023

Linear attention `O = (Q (K^T V))` plus a learned per-token data-dependent gate that decays the running KV state:
```
S_t = G_t ⊙ S_{t-1} + K_t · V_t        # element-wise gated update
O_t = Q_t · S_t
```
`G_t` is computed from `x_t` via a small MLP. Recurrent form: cache is `S ∈ [B, H, d_h, d_v]` per layer — fixed in T, like SSM.

**Shipped:** Jamba-1.5 (some attention layers per ablation), Zamba2 ablations, `flame` runtime.

### 3.16 DeltaNet / Gated DeltaNet — Yang & Pan arxiv:2406.06484 (DeltaNet) + Qwen3-Next 3:1 hybrid [v3-expanded]

Linear attention with the **delta rule**: `S_t = S_{t-1} − S_{t-1} K_t K_t^T + V_t K_t^T`, which removes the V component previously associated with this key and writes the new one. Provides O(1) memory associative-recall.

The "gated" variant adds a per-token decay gate `α_t ∈ (0, 1]`:
```
S_t = α_t · (S_{t-1} − S_{t-1} K_t K_t^T) + V_t K_t^T
```

**Shipped:** **Qwen3-Next-80B-A3B (Aug 2025) uses Gated DeltaNet as its primary token mixer in a hybrid with attention at a 3:1 ratio** — 75% of layers are Gated DeltaNet, 25% are softmax Gated Attention. This is the **inverse** of MiniMax-Text-01's 7:1 ratio (87.5% linear, 12.5% softmax). The fact that the same year produced two production hybrids at opposite ratios — and that MiniMax then reverted to full softmax in M2 — is itself the survey-paper story noted in §9.4. Flame runtime. Microsoft's internal `RetNet-Delta` experiments. Qwen3-Next additionally ships an MTP head, so the cache is GDN-state + softmax-attention KV + MTP-head KV all simultaneously (see §5.12 for MTP layout; §5.13 for SSM-equivalent state for GDN).

### 3.17 RetNet — Sun et al. arxiv:2307.08621 (2023)

Replaces softmax with retention: `Q · (D ⊙ K^T V)` where `D_{ij} = γ^{i-j}` for `i ≥ j`. The recurrent form has constant-size state. Three computation modes: parallel (training), recurrent (decode), chunkwise (long sequence).

**Shipped:** RetNet 1.3B / 2.7B / 6.7B reference releases (Microsoft); not in a flagship SLM but cited as the lineage for DeltaNet and GLA.

### 3.18 Hyena / StripedHyena — Poli et al. arxiv:2302.10866, Together 2023

Implicit-conv linear attention: a long-conv parameterized by an MLP-generated filter. The cache is the previous filter outputs, not K/V.

**Shipped:** StripedHyena-7B (Together), some Mosaic variants. Out of mainstream SLM list but worth flagging in §10.

### 3.19 TTT — Test-Time Training (Sun et al. arxiv:2407.04620, Jul 2024)

Replaces the token mixer with a learned MLP whose weights update during the forward pass. The "cache state" becomes a tuple of weight matrices that get gradient-updated per token. Even if not in shipped SLMs, this implies some token-mixers have **non-tensor caches**. The minimum API for §7 must allow `cache_kind = mutable_model_weights` as a placeholder.

### 3.20 xLSTM — mLSTM / sLSTM (Beck et al. arxiv:2405.04517, May 2024)

Extension of LSTM with exponential gating and a matrix-memory variant (`mLSTM`). `mLSTM` cache is a per-head matrix `C ∈ [d_h, d_h]`; `sLSTM` cache is the standard scalar `c, h`.

**Shipped:** xLSTM-7B reference (May 2024, NXAI), now a non-research production checkpoint. Promoted from v2's "noted, not pulled" status.

### 3.21 Cross-Attention (encoder-decoder)

K, V come from a frozen encoder; only Q is computed per decode step. Cache is a single fixed-size `[B, H_kv, T_enc, d_h]` shared across all decode steps. Distinct lifetime from self-attention KV.

**Shipped:** T5, Whisper, Donut, Florence-2, encoder-decoder distillations. The minimum API for §7 must allow attention blocks to declare a cache as **frozen** (no per-step write). No shipped scope-mainline SLM currently uses cross-attention except YOCO (which is cross-attention from stage-2 to stage-1) and the encoder-based Gemma 4 multimodal path (vision-token attention from the LLM body to the frozen-after-projection ViT embeddings).

### 3.22 FlashAttention v1 / v2 / v3 — the kernel-family axis

(Verbatim from v2 §3.22; covers FA1, FA2 with sliding window / softcap / asymmetric head_dim / FP8 / paged_decode, FA3 with TMA / WGMMA / FP8 KV on H100.)

### 3.23 FlashDecoding / FlashDecoding++

(Verbatim from v2 §3.23: split-K decode plus unified-max softmax.)

### 3.24 RingAttention — Liu, Zaharia, Abbeel arxiv:2310.01889 (Oct 2023)

(Verbatim from v2 §3.24.)

### 3.25 Striped Attention — Brandon et al. arxiv:2311.09431 (Nov 2023)

(Verbatim from v2 §3.25.)

### 3.26 TreeAttention — Spector & Re arxiv:2402.13720, also Medusa-2 (2024)

(Verbatim from v2 §3.26.)

### 3.27 RadixAttention — SGLang (Zheng et al. arxiv:2406.04692, May 2024)

(Verbatim from v2 §3.27.)

### 3.28 PagedAttention v1 — Kwon et al. SOSP 2023

(Verbatim from v2 §3.28; `DEFAULT_BLOCK_SIZE = 16` in `vllm/config/cache.py`.)

### 3.29 PagedAttention v2 / vAttention — Prabhu et al. arxiv:2405.04437

(Verbatim from v2 §3.29.)

### 3.30 Lightning Indexer DSA — DeepSeek-V3.2 (Sep 2025) [v3 NEW]

DeepSeek-V3.2 introduces **DeepSeek Sparse Attention (DSA)** — a layer of sparsity on top of the existing MLA. The mechanism:

1. A **Lightning Indexer** — a small learned per-query scorer that computes a relevance score for each past-token position from a compressed query projection. The indexer is much cheaper than full attention (operates on a reduced dim, fewer params), and its state is cached separately from the MLA latent.
2. For each query position, the top-`k` past positions by indexer score are selected.
3. Full MLA attention is then computed only against the selected top-`k` positions (plus a small fixed warm-up window for very recent positions).

The result is `O(T · k)` attention compute per step instead of `O(T²)` — but the selection is **learned**, not heuristic. This makes DSA the first production deployment of a learned-scorer sparse attention; NSA's selection was also learned but used a **block** scorer; DSA's is **per-token** through the indexer.

**Cache layout (per layer, per request):**
- `mla_latent`: same as V2 — `[N_blocks, block_size, kv_lora_rank + qk_rope_head_dim]`.
- `indexer_state`: a separate compressed buffer of shape `[B, T, d_indexer]` per layer; `d_indexer` is much smaller than `d_h` (community write-ups suggest 32-64 typical, `[VERIFY against full tech report]`).
- The `indexer_state` is **append-only** like KV; no per-step rewrites needed.

**FLOPs / KV savings (vs V3.1 at the same context):** ~50 % FLOPs reduction at 128 k context; the full V3.2 tech report quotes "competitive quality with materially lower cost at long context" without spelling out a precise number that survives `[VERIFY]`.

**Shipped:** **DeepSeek-V3.2** (preview Sep 2025; stable Dec 2025), **DeepSeek-V3.2-Exp** (open weights), and **GLM-5.1** (Apr 2026, 744 B / 40 B-A — adopts DSA from V3.2). Custom kernel required (vLLM main has experimental DSA path as of 2026-Q1; FlashInfer's DSA backend is in v0.3.1+).

**Constraint:** DSA requires the indexer to be either populated up-front (prefill) or filled incrementally (decode). For chunked prefill, the indexer must be batched at chunk granularity. The kernel signature is `attention(q, mla_latent, indexer_state, top_k)`, with `top_k` typically 256-1024 depending on context length.

### 3.31 Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA) — DeepSeek-V4 (Apr 2026) [v3 NEW]

DeepSeek-V4-Pro (1.6 T total / 49 B active) and V4-Flash (284 B / 13 B-A) ship a **two-tier compression scheme** that combines:

1. **CSA — Compressed Sparse Attention.** Builds compressed block-level summaries (similar in spirit to NSA's compressed branch but with a learned multi-level compression hierarchy). Per token, a sparse top-k against the compressed summaries decides which blocks are read at higher resolution.
2. **HCA — Heavily Compressed Attention.** A second, more aggressive compression tier that handles the very-long-tail context (positions older than some boundary). HCA's compression ratio is higher (~16× the CSA ratio per community write-ups, `[VERIFY]`), and its kernel is purely top-k on the compressed summaries with no high-resolution fallback.

The composition gives V4 a combined "near, mid, far" attention budget:
- **Near** (last `W_near` tokens): full attention on the MLA latent.
- **Mid** (CSA's range): block-level top-k against CSA summaries, then high-resolution.
- **Far** (HCA's range): block-level top-k against HCA summaries, low-resolution only.

**FLOPs and KV at 1 M context (vs V3.2):** Headline numbers from the V4 tech report (per `08-recent-releases-2026q2.md`): **27 % of V3.2's per-token inference FLOPs, 10 % of V3.2's KV cache footprint at 1 M context.** Both numbers are from the official tech report; reproduction in third-party kernels is pending.

**Cache layout (per layer, per request):**
- `mla_latent` (near, full resolution): same as V3 / V3.2.
- `csa_summary`: `[B, H_kv, T_csa, d_csa]` where `T_csa = ceil(T / l_csa)`; one summary per CSA block.
- `csa_full`: `[B, H_kv, T, d_h]` at full resolution; only top-k blocks read each step.
- `hca_summary`: `[B, H_kv, T_hca, d_hca]` where `T_hca = ceil(T / l_hca)` and `l_hca >> l_csa` (16×).

The cache is therefore **four tensors per layer** (vs MLA's one). Total per-token cache, summing across the four tiers, is ~10 % of V3.2 at 1 M.

**Shipped:** DeepSeek-V4-Pro and V4-Flash (Apr 2026). NVIDIA released NVFP4-quantized DeepSeek-V4-Pro on the same day. Public kernel support: a reference kernel ships in the `DeepSeek-V4` Hugging Face repo; vLLM / SGLang integration was tracked as "in progress" as of 2026-06-06.

**Constraint:** Standard SDPA / FA backends cannot serve CSA + HCA without bespoke gather-and-top-k logic. The CSA and HCA tiers each need a per-step block selection; this is conceptually similar to NSA (§3.9) but with **two levels of compression** rather than three independent branches.

### 3.32 K = V Unification on Global Layers — Gemma 4 ≥12B (Apr 2026) [v3 NEW]

On Gemma 4's global (full-attention) layers, **the K and V projection matrices are tied**: `W_k = W_v`, and the same projection output is used as both K and V. The cache stores **one** tensor `KV = W_k · h` per token rather than two. At attention time, the same KV is fed as K (for `Q · K^T`) and as V (for `attn · V`).

**Per-config size dependence (verified 2026-06-06):**
- E2B (35 layers, 1.5 B effective): `attention_k_eq_v = false` → K and V are distinct.
- E4B (42 layers, 4 B effective): `attention_k_eq_v = false` likely, secondary writeups claim true; **flagged `[VERIFY]`** until the raw E4B config is checked.
- 12B (48 layers), 26B-A4B (30 layers), 31B (60 layers): `attention_k_eq_v = true` confirmed against `google/gemma-4-31B/config.json` (verified by direct fetch).

When enabled, the global layer's per-token KV cache is **half** the size of a standard GQA layer with the same head count. Paired with a `global_head_dim = 512` (doubled vs the local-layer `head_dim = 256`) and reduced KV head count (E2B global: 1 KV head; 31B global: 16 KV heads), the global cache cost stays bounded even at 256 k context.

**Cache layout impact:** a global-layer cache with `attention_k_eq_v = true` requires the kernel to **alias** the K and V pointers to the same buffer. vLLM's `MLACommonImpl` already supports a single-tensor cache (for MLA's `c_kv`); the same primitive can serve K-equals-V, with the projection different and no `qk_rope_head_dim` split. v3's `AttentionSpec` adds `attention_k_eq_v: bool` and the cache spec adds `cache_kind = k_equals_v` to distinguish from the MLA case.

**Architectural note:** this is the first time a Google open release ties K and V across **all global layers** of large variants. Earlier ties (e.g., shared K and V projections in some research prototypes) were partial; Gemma 4 makes it a flagship-scale axis. The motivation is parameter savings — `W_k = W_v` halves the per-layer projection parameter count of K + V, which compounds across 60 global-layer transformers on the 31B.

**Open question:** the secondary Gemma 4 sources (HF model card, the QAT blog, the Maarten Grootendorst visual guide) state the property in a way that suggests it applies family-wide; the raw `config.json` shows it is **size-dependent**. The investigation note in `10-gemma4-investigation.md` records this resolution. v3's table at §3.6 is the authoritative summary.

### 3.33 Lightning Attention 7:1 Hybrid — MiniMax-Text-01 (Jan 2025) [v3 NEW, retrofitted from v2's evolution mention]

MiniMax-Text-01 (456 B / 45.9 B-A, January 2025) ships an **attention stack with 7 linear-attention layers for every 1 softmax-attention layer** — i.e., 87.5 % linear, 12.5 % softmax. The linear layers use "Lightning Attention," a flash-style kernel for linear attention with chunkwise computation and recurrent decoding modes.

**Cache impact:** the 7 linear layers store fixed-size matrix state (`[B, H, d_h, d_v]` per layer), the 1 softmax layer stores standard `[B, H, T, d_h]` KV. So at long context, only every 8th layer's cache grows; the others are constant. At 1 M context, this gives ~92 % KV savings vs a pure-softmax model of the same depth — even before any quantization.

**The MiniMax M2 reversion** (early 2026): MiniMax-M2 abandons Lightning Attention and reverts to pure softmax attention, with the rationale (per the M2 tech report arxiv:2605.26494) being that the linear layers' quality-cost tradeoff in production agentic workloads didn't pan out. **MiniMax-M3.0** (Jun 2026, weights staged) then re-introduces sparsity in a different form: **MiniMax Sparse Attention (MSA)** — see §3.38.

**Shipped:** MiniMax-Text-01 (Jan 2025, weights public). Production runtimes need custom kernels for Lightning Attention; only the official MiniMax inference repo ships them today. Public vLLM / SGLang support requires manual integration.

**Why v3 promotes this from a one-liner:** the audit in `08-coverage-justification.md` notes "Lightning Attention adopted and then abandoned in M2" as a survey-paper-worthy event. v3 treats this as the canonical production data point for "extreme linear:softmax ratios in production, and the empirical limits thereof," paired with Qwen3-Next's opposing 3:1 design.

### 3.34 Mamba-3 with Complex State and MIMO Decoding (Mar 2026) [v3 NEW]

Mamba-3 (ICLR 2026, arxiv:2603.15569, Together AI / CMU / Princeton / Cartesia, reference scales 180 M – 1.5 B) is a pure-SSM successor to Mamba-2 with three substantive deltas:

1. **More expressive recurrence derived from refined SSM discretization.** The discrete-time formulation generalizes Mamba-2's ZOH approximation to a higher-order scheme; quality benefits accumulate at long context.
2. **Complex-valued state-update rule.** `h_t` is in `C^d_state` rather than `R^d_state`. The complex state allows richer spectral content per dim; in practice, each real-valued slot of v2 is replaced by a `(real, imag)` pair, and `A_t` becomes a unitary or sub-unitary complex multiplier rather than a real scalar gain. This is the largest architectural delta.
3. **MIMO (multi-input, multi-output) decoding.** Each step emits multiple output tokens per recurrence step at the same compute cost. The mechanism is a parallel-output head bank on the SSM that produces `n_outputs` predictions per `h_t`; downstream verifier-style logic picks one. At fixed decode latency, MIMO improves accuracy by giving the model "k attempts per step" at the same compute cost. This is genuinely new vs all prior SSMs.

**Cache state per layer:** `complex_ssm_state ∈ C^{B × d_inner × d_state}` (`= 2 × R^{...}` in memory), `conv_state` as Mamba-1 / -2. The complex doubling means total state bytes ~2× of Mamba-2 at the same `d_state`, but reported quality gains let `d_state` be smaller, so net memory is roughly comparable. `[VERIFY against the full PDF]` for exact `d_state` and `n_outputs` defaults.

**Shipped:** reference impl in `github.com/state-spaces/mamba` since 2026-03-17; the 180 M / 700 M / 1.5 B reference checkpoints are public. No production large-scale Mamba-3 shipped as of 2026-06-06; **Nemotron 3 Ultra and other 2026-H2 hybrids are expected to adopt Mamba-3 in successor releases**.

**Constraint:** the complex state plus MIMO decoding require either bespoke complex-arithmetic kernels (cuFFT-adjacent) or real-valued lifting (`Re/Im` paired updates) with double the data movement. The reference implementation does the latter for compatibility with PyTorch's real-only autograd.

### 3.35 Visual Causal Flow / Block-Bidirectional Mask — DeepSeek-OCR (Oct 2025) [v3 NEW]

DeepSeek-OCR (Oct 2025) introduces a **per-token-type mask** in its decoder LM (DeepSeek3B-MoE-A570M): **vision tokens (from the SAM + CLIP encoder) get block-bidirectional attention among themselves**, while **text tokens (the autoregressive output) stay strictly causal**. The mask is:

```
M[i, j] = True  if (i is vision and j is vision and same block)         # bidirectional within vision block
       or  (i is text and j ≤ i)                                          # causal among text
       or  (i is text and j is vision)                                    # text reads all preceding vision
       or  (i is vision and j is vision and j < i, different block)       # causal across vision blocks
```

In other words: vision is grouped into blocks of `b_vis` tokens; within a block, attention is bidirectional; across blocks, causal. Text is strictly causal and can read all vision. This is a **first-class new mask kind** in the v3 taxonomy — earlier mask kinds (causal / sliding / sink / NSA / tree / 2-D) did not capture per-token-type semantics.

**Why this matters for VLM design.** Block-bidirectional masking on vision tokens lets the encoder's spatial structure ("which patch corresponds to which region of the image") propagate symmetrically; pure causal on vision tokens would impose an arbitrary order on the image patches and degrade OCR quality. DeepSeek-OCR's "1 vision token ≈ 10 text tokens" compression budget depends on this masking.

**Shipped:** DeepSeek-OCR (Oct 2025), DeepSeek-OCR-2 (Jan 2026), MinerU2.5 (Sep 2025) uses a similar but coarser variant.

**Constraint:** standard FA backends accept arbitrary 2-D masks via `attn_mask` but the **block-bidirectional pattern is regular enough to fuse as a kernel intrinsic** — vLLM and FlashInfer have a "VisualCausalMask" kernel-side primitive as of late 2025. v3's `AttentionSpec.mask` enum gains `visual_causal_flow` / `block_bidirectional` as a value, plus a `block_bidirectional_mask: bool` flag for hybrid VLM use.

### 3.36 Gated Attention with QK-Norm Refinements (Qwen3-Next softmax layers) [v3 NEW, scoped]

Qwen3-Next-80B-A3B's 25 % softmax layers are not vanilla GQA — they use **Gated Attention**: a learned per-token gate `g_t = σ(W_g · x_t)` that scales the attention output `(g_t · attn_out)` before the residual add. The gate prevents over-attending in regimes where the GDN layers have already mixed long-range info.

**Cache impact:** none on the KV side — the gate is a per-token vector with no cache dependence. The gate weights are weight-only parameters.

**Why call it out:** the gate makes the softmax layer functionally a *modulation* of the linear-attention backbone rather than a substitute. v3's `AttentionSpec.kind` for these layers is `gqa`, but a `output_gate: bool` field is added; the gate is then composable with sink, NoPE, partial-RoPE, etc.

### 3.37 Trained Attention Sinks (extended) — GPT-OSS, Mistral Small 3.1, plus 2025-H2 adopters [v3-expanded from v2 §3.7]

v3 promotes this from a clause inside §3.7 to its own subsection (now §3.37) because the lineage is now broad enough to warrant first-class treatment:

- **GPT-OSS-20B and -120B** (Aug 2025, OpenAI). Sink slots trained from scratch; MXFP4 weight quant; alternating dense+banded-sparse attention pattern.
- **Mistral Small 3 / 3.1 / 3.2** (Jan / Mar / Jun 2025). `attention_sink_size: 4` baked into the trained checkpoint. 3.1 ships Pixtral vision + 128 k RoPE; 3.2 is text-quality refresh.
- **Mistral Medium 3.5** (Apr 2026, 128 B dense, absorbs Magistral and Devstral 2). Same sink recipe as Small 3.x scaled up.
- **Mistral Small 4** (Mar 2026, 119 B / 15 B-A MoE). Same recipe.
- **Granite 4 H** family (Oct 2025) uses sinks **on the attention layers only**; the Mamba-2 layers don't have a concept of sinks.

**Cache impact:** `sink_slots` is `k` cells (typ. 4), addressed always-on, sat outside the normal sliding/global rotation. In a paged cache, these `k` cells live in a dedicated low-index block that is reference-counted but never evicted.

**Constraint:** SWA + trained-sink + paged requires the kernel to support two windows simultaneously (sink first-k + slide last-W). FA3 + vLLM Triton paths handle this; older CUTLASS paths do not.

### 3.38 MSA — MiniMax Sparse Attention (MiniMax-M3.0, Jun 2026) [v3 NEW]

MiniMax-M3.0 (announced 2026-06-01; weights staged to land on HF ~mid-June) introduces **MiniMax Sparse Attention (MSA)** — distinct from DeepSeek's DSA, distinct from NSA, distinct from CSA+HCA. Public details as of audit cutoff (2026-06-06):

- MSA is a **sparse-attention variant** that combines a query-dependent gating with a learned routing pattern for top-k selection.
- 1 M context native.
- Native multimodal + agentic coding.
- Weights expected on HF ~10 days post-launch (so ~mid-June, likely already public by the time this v3 is read).

**v3 treats MSA as a forward-looking entry** because the architecture-paper-level details required to compete with DSA / CSA on the design-matrix axis are not yet public. The kernel is presumed to share the DSA family's "indexer + top-k against latent" interface, but this is `[VERIFY]`. Worth noting that MiniMax's prior production trajectory (Lightning 7:1 → abandoned in M2 → MSA in M3.0) is the cleanest example in the v3 corpus of a vendor iterating sparse-attention designs in production.

---

## §4 RoPE variants — 18 entries (14 in v2 → 18 in v3)

§4.1–§4.14 are identical to v2 §4.1–§4.14. §4.15–§4.18 are v3 additions.

### 4.1 Vanilla RoPE — Su et al. arxiv:2104.09864 (RoFormer 2021)

For head dim `d_h`, frequencies `θ_i = base^(−2i/d_h)`, `i ∈ [0, d_h/2)`. At position `m`, rotate pairs `(x_{2i}, x_{2i+1})` by angle `m · θ_i`. Parameters: `base` (a.k.a. `rope_theta`), `d_rotated ∈ [0, d_h]`, basis (interleaved vs split-half — see §4.13).

**Base values across shipped SLMs (re-verified 2026-06-04; 2026-H1 additions appended):**

| Model | rope_theta | Source |
|---|---|---|
| LLaMA 1 | 10_000 | original RoFormer default |
| Llama-2 | 10_000 | hf llama config |
| Llama-3 | 500_000 | apr 2024 release |
| Llama-3.1 / 3.2 | 500_000 + llama3 scaling | + smooth-wavelength scaling |
| Llama-4 Scout / Maverick | 500_000 + iRoPE per-layer | apr 2025 |
| Mistral-7B-v0.1 | 10_000 | mar 2023 |
| Mistral-7B-v0.3 | 1_000_000 | mid 2024 |
| Mistral-Small-3.1 | 1_000_000 | mar 2025 |
| Mistral Small 4 | 1_000_000 | mar 2026 |
| Mistral Medium 3.5 | 1_000_000 | apr 2026 |
| Qwen2.5 | 1_000_000 | sep 2024 |
| Qwen3-0.6B / 4B / 8B / 14B | 1_000_000 | verified |
| Qwen3.6-27B / 35B-A3B | 1_000_000 | apr 2026 |
| Phi-3-mini-4k | 10_000 | apr 2024 |
| Phi-3-mini-128k | 10_000 + LongRoPE | factor vectors override base |
| Phi-4-mini (3.8B) | 1_000_000 | jan 2025 |
| Gemma 2 (all sizes) | 10_000 | jun 2024 |
| Gemma 3 | per-layer: 10_000 (local) / 1_000_000 (global) | mar 2025 |
| **Gemma 4 (all sizes)** | **per-layer: 10_000 (sliding) / 1_000_000 (global) + p-RoPE 0.25 on global** | apr 2026 — see §4.15 |
| DeepSeek-V2 | 10_000 (+ YaRN scale=40) | may 2024 |
| DeepSeek-V3.2 | 10_000 (+ YaRN) on MLA latent + indexer-state RoPE | sep 2025 |
| DeepSeek-V4-Pro / Flash | 10_000 (+ YaRN) on MLA latent + CSA/HCA summaries | apr 2026 |
| SmolLM2 | 130_000 | nov 2024 |
| SmolLM3 (1.7B / 3B) | 100_000 + NoPE on a subset of layers | jul 2025 |
| OLMo-2-1B | 500_000 | nov 2024 |
| StableLM-2-1.6B | 10_000 | feb 2024 |
| Granite-3.0 | 10_000_000 | oct 2024 |
| Granite 4 H-Tiny | 10_000_000 | oct 2025 |
| GPT-OSS 20B / 120B | 500_000 (with trained sinks) | aug 2025 |
| Qwen3-Next 80B-A3B | 1_000_000 on softmax-attention layers (GDN layers don't use RoPE) | aug 2025 |
| MiniMax-Text-01 | 10_000 + YaRN scale | jan 2025 (linear-attn layers don't use RoPE) |
| MiniMax-M2 / M3.0 | 1_000_000 | mar-jun 2026 |
| Granite-3.3 / 4-tiny-hybrid | varies | 2025 |

**"Increase `base` and retrain"** is a long-context strategy distinct from inference-time scaling. The sequence 10K → 500K → 1M → 5M (initially reported but never landed) → 10M (Granite) is itself a chronological arc; see §9.

### 4.2 Position Interpolation (PI) — Chen et al. arxiv:2306.15595 and Kaiokendev "SuperHOT" (Jun 2023)

(Verbatim from v2 §4.2.)

### 4.3 NTK-aware (static) — bloc97 Reddit Jul 2023, formalized in YaRN

(Verbatim from v2 §4.3.)

### 4.4 Dynamic NTK-aware — ChatGLM-2 Jul 2023, InternLM 200K Aug 2023

(Verbatim from v2 §4.4.)

### 4.5 YaRN — Peng, Quesnelle, Kingma arxiv:2309.00071 (Sep 2023)

(Verbatim from v2 §4.5.)

### 4.6 Llama-3 smooth wavelength scaling — Apr 2024

(Verbatim from v2 §4.6.)

### 4.7 LongRoPE — Ding et al. arxiv:2402.13753 (Feb 2024)

(Verbatim from v2 §4.7. As of 2026-06-06, LongRoPE in its strict form remains unique to the Phi family; no 2026 model has adopted the dual per-dim factor vectors with a hard `seq_len` switch.)

### 4.8 DCA — Dual Chunk Attention (An et al. arxiv:2402.17463 / Qwen2.5-1M)

(Verbatim from v2 §4.8.)

### 4.9 Partial RoPE / Non-rotated Dims — original form, supplanted by §4.15 for global-layer use

Rotate only the first `d_rope` dims of each head, leave `d_h − d_rope` untouched. **v3 reorganization**: the historical Phi-1/2/GPT-J/StableLM partial-RoPE pattern is documented here; the new Gemma 4 "p-RoPE per global layer" is split into §4.15 because it is composed with the dual-θ per-layer pattern and is a per-layer-type axis, not a per-model axis.

**Shipped (historical):**
- Phi-1 / Phi-1.5 / Phi-2: `partial_rotary_factor = 0.5`
- GPT-J: `rotary_pct = 0.25`
- GPT-NeoX-20B: `rotary_pct = 0.25`
- StableLM-2: `partial_rotary_factor = 0.25`
- **DeepSeek-V2 / V3 MLA, MiniCPM-3 MLA**: structural — only the `qk_rope_head_dim` channel is rotated; the `qk_nope_head_dim` channel is not. Not a tuning knob but architectural.

### 4.10 NoPE — Haviv et al. arxiv:2203.16634 (2022), Kazemnejad et al. arxiv:2305.19466 (2023)

Train without positional encoding at all; the causal mask + decoder LM provides position implicitly.

**Shipped (mainstream as of 2026-H1):**
- **SmolLM3 (1.7B / 3B)** alternates: every Nth layer omits RoPE entirely. Per-layer `apply_rope: bool` is the required axis.
- **Llama-4 iRoPE** (Apr 2025): see §4.16 — Llama 4 fuses NoPE alternation with attention-temperature tuning.
- Some Llama-3 ablations and the original OLMo-NoPE experiments.

### 4.11 xPos — Sun et al. arxiv:2212.10554 (2022)

(Verbatim from v2 §4.11.)

### 4.12 Sandwich PE — Chi et al. arxiv:2212.10356 (2022)

(Verbatim from v2 §4.12.)

### 4.13 RoPE Basis: Interleaved vs Split-Half

(Verbatim from v2 §4.13: HF default vs llama.cpp convert-time permutation.)

### 4.14 Multimodal RoPE — M-RoPE (Qwen2-VL) and MM-RoPE (Qwen2.5-VL) — see §4.17 for the expanded v3 treatment

(Brief version kept verbatim from v2; v3 expands the full multimodal RoPE taxonomy in §4.17.)

### 4.15 p-RoPE / Partial-Rotary RoPE per Global Layer — Gemma 4 (Apr 2026) [v3 NEW]

Gemma 4's global (full-attention) layers ship with `partial_rotary_factor = 0.25`: **only the first 25 % of each head_dim is rotated by RoPE; the remaining 75 % is left as pure semantic content with no positional info**. The `rope_type` for global layers is `"proportional"` — meaning the rotated subset's effective wavelength is rescaled to cover the full effective range of the head.

**Per-config (verified against `google/gemma-4-E2B/config.json` and `google/gemma-4-31B/config.json`, 2026-06-06):**

```json
"rope_scaling_per_layer_type": {
  "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
  "full_attention":    {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
}
```

The sliding layers stay at the Gemma 3 baseline (vanilla RoPE, θ = 10 K, full rotation). The global layers are where the change lives.

**Why this matters.** Composing partial-rotary with per-layer θ-alternation is a genuinely new composite in the RoPE axis. The implicit hypothesis is that on the global, long-range layers, the model benefits more from "75 % of the head dim is positionally inert" — i.e., long-range information should be matched on content rather than position. Empirically, the Gemma 4 launch claims this improves 256 k context quality vs Gemma 3's full-rotary global layers; the tech report (which would substantiate this) was not yet published as of 2026-06-06.

**Difference from historical partial RoPE (§4.9).** Phi-1's `partial_rotary_factor = 0.5` and StableLM-2's `0.25` are **per-model** — every layer uses the same partial-rotary fraction. Gemma 4's is **per-layer-type**: sliding layers are fully rotated, global layers are partially rotated. This requires the API to carry `partial_rotary_factor` inside a per-layer-type `RoPESpec` or per-layer override, not at the model root. v3's `RoPESpec` adds `partial_rotary_factor: Optional[float]` as a first-class field.

**Shipped:** Gemma 4 E2B / E4B / 12B / 26B-A4B / 31B (Apr 2026).

**Open verification:** the `rope_type = "proportional"` variant introduced for Gemma 4 is novel and not yet documented in the public HF `modeling_rope_utils.py`. v3 lists it as `[VERIFY: full proportional-RoPE formula]` against the upstream `transformers` source when it lands in a release.

### 4.16 iRoPE per-layer — Llama 4 Scout / Maverick (Apr 2025) [v3 NEW, was a one-liner in v2]

Llama-4 Scout (17 B active / 16 experts) and Maverick (17 B / 128 experts) introduce **iRoPE**: per-layer alternation between **RoPE** and **NoPE** layers, plus an **attention-temperature tuning** pass at inference.

**The alternation pattern.** A `no_rope_layer_interval` schedule selects which layers omit RoPE. The Meta launch blog and the HF blog both describe this as "interleaved RoPE / NoPE" without giving a per-layer schedule; community model-card writeups and the `meta-llama/Llama-4-Scout-17B-16E` config (gated) indicate roughly every 4th layer is NoPE for Scout. **Open verification:** the exact ratio per Llama 4 size is flagged as `[VERIFY]`.

**Attention-temperature tuning.** Alongside per-layer NoPE, Llama 4 applies a learned temperature scalar `T_l` to the attention softmax of each layer — different `T_l` for RoPE vs NoPE layers, with NoPE layers tending toward higher temperature (broader attention) to compensate for the lack of position info.

**iRoPE × paged cache.** NoPE layers' position info is **implicit** (from the causal mask), so the rotation step is skipped; K and V are written verbatim. This requires the cache framework to accept a "no-op RoPE" pass per layer rather than apply RoPE unconditionally. vLLM v0.9+ handles this via the per-layer `apply_rope: bool` flag.

**Why call it out as a separate v3 entry rather than rolling into §4.10 NoPE.** SmolLM3's NoPE alternation is the **simplest** form: per-layer `apply_rope ∈ {True, False}`, nothing else changes. Llama 4's iRoPE adds the **temperature-tuning composability** (per-layer `attn_temp`), which is a wholly new RoPE-adjacent axis. v3 treats iRoPE as the canonical "RoPE on/off + attn-temperature" composite; SmolLM3 is then the degenerate case with `attn_temp = 1.0`.

**Shipped:** Llama 4 Scout (Apr 2025), Llama 4 Maverick (Apr 2025), and the in-progress Llama 5 family (Apr 2026, weights gated).

### 4.17 M-RoPE / 2D-RoPE / V2PE / TMRoPE — Multimodal RoPE Family Expanded [v3-expanded]

The v2 §4.14 treatment is replaced with a fuller taxonomy:

**M-RoPE (Qwen2-VL, Aug 2024).** Splits RoPE dims into **temporal / height / width** thirds; each gets its own position index. Required for video tokens.

**MM-RoPE (Qwen2.5-VL, Bai et al. 2025).** Generalizes M-RoPE: the per-axis dim split moves from `(1/3, 1/3, 1/3)` to a configurable `mrope_section = (T, H, W)` tuple (Qwen2.5-VL uses `(T:16, H:24, W:24)` on a 64-dim head). Adds a frame-index dim for video.

**Qwen3-VL M-RoPE (Apr 2026).** Continues MM-RoPE with refined `mrope_section` per size. The dim ordering inside each axis is documented in the Qwen3-VL HF config; the API axis added by v3 is `mrope_section: Optional[tuple]` on `RoPESpec`.

**V2PE / 2D-RoPE for vision encoders (general).** A separate 2D-RoPE basis for the vision tower itself (separate from the LLM body). Operates on raw image patches with `(row, col)` position pairs. Gemma 4's vision encoder uses this (it claims "variable aspect ratio via 2D RoPE + learned 2D positions up to 10 240 positions per axis"). Pixtral 12B's vision encoder also uses a 2D-RoPE on patches.

**TMRoPE (Qwen2.5-Omni, Mar 2025).** Adds a **frame-index axis** distinct from M-RoPE's T/H/W, giving a 4-axis RoPE for **time-aligned multimodal** content. Critical for the Thinker-Talker dual-decoder where audio and video need explicit cross-modal time alignment.

**Janus-Pro 1B / 7B (Jan 2025).** Uses 2D-RoPE on image patches in the understanding path; the generation path uses a separate decoupled visual encoding. The 2D-RoPE pattern composes cleanly with M-RoPE.

**Llama-3.2-Vision 11B (Sep 2024).** Cross-attention-based VLM where the vision tokens enter via cross-attention layers; the cross-attention layers carry their own positional encoding (a 2D learned positional embedding, not RoPE). Distinct from the M-RoPE family.

**API axis additions for v3:**

```
RoPESpec {
  ...
  position_axes:   int                     # 1 (text), 2 (2D-RoPE), 3 (M-RoPE T/H/W), 4 (TMRoPE +frame)
  mrope_section:   tuple[int, ...] | None  # (T, H, W) split; len == position_axes - 1
  is_2d:           bool                    # for vision-tower-only 2D-RoPE; orthogonal to mrope
}
```

### 4.18 V2PE / Visual 2D-Position Encoding Variants — vision encoder side [v3 NEW]

(Brief note for completeness; full treatment lives in `02-layer-sources.v3.md` §5.34.)

Most production VLM vision encoders use a learned 2D positional embedding (PaliGemma, MiniCPM-V) or 2D-RoPE (Pixtral, Janus-Pro, Gemma 4 ViT). The **V2PE** label refers to a learned-but-2D position embedding that scales to arbitrary aspect ratio; SmolVLM2 uses a pixel-shuffle + V2PE combination. v3 calls this out as a distinct RoPE-adjacent axis only because it composes orthogonally with the LLM-side M-RoPE / TMRoPE — i.e., the vision encoder can be V2PE while the LLM body uses M-RoPE for the soft-token positions.

---

## §5 KV cache layouts — 18 entries (13 in v2 → 18 in v3)

§5.1–§5.13 are identical to v2 §5.1–§5.13. §5.14–§5.18 are v3 additions.

### 5.1 Contiguous (HF transformers default)

(Verbatim from v2 §5.1.)

### 5.2 Paged (PagedAttention v1 — vLLM, SGLang, TRT-LLM)

(Verbatim from v2 §5.2; `DEFAULT_BLOCK_SIZE = 16`.)

### 5.3 Ring / Circular Buffer (sliding-window models)

(Verbatim from v2 §5.3.)

### 5.4 vAttention — VM-Mapped Cache (Prabhu et al. MSR-India 2024, arxiv:2405.04437)

(Verbatim from v2 §5.4.)

### 5.5 NSA Three-Tier Cache (DeepSeek 2025)

(Verbatim from v2 §5.5; the three branches: `cmp_cache`, `sel_cache`, `swa_cache`.)

### 5.6 MLA Cache — Single Latent Tensor (DeepSeek-V2)

(Verbatim from v2 §5.6.)

### 5.7 YOCO Global Cache (Microsoft 2024)

(Verbatim from v2 §5.7.)

### 5.8 Per-Layer Separate vs Unified

(Verbatim from v2 §5.8.)

### 5.9 HND vs NHD Memory Layout

(Verbatim from v2 §5.9.)

### 5.10 Quantized KV — KIVI, KVQuant, CacheGen

(Verbatim from v2 §5.10; per-channel K + per-token V asymmetry; v3 additions: 2026-H1 NVFP4 KV path for NVIDIA Nemotron 3 and DeepSeek-V4-Pro-NVFP4.)

### 5.11 Beam / Tree Cache (Speculative Decode)

(Verbatim from v2 §5.11. v3 note: Gemma 4 MTP drafter heads (Apr 2026) also use this cache primitive; see §5.12 for MTP layout details.)

### 5.12 MTP-Head Cache (DeepSeek-V3, Dec 2024) [v3-expanded for Gemma 4 / Hy3 / Qwen3-Next]

(Verbatim from v2 §5.12; **v3 addition: Gemma 4 MTP Drafters** (Apr 2026, small drafter heads paired with 12B and 26B-A4B) bring MTP to the Gemma family; **Hy3 / Hunyuan 3 Preview** (Apr 2026) ships a 3.8 B MTP layer alongside its 295 B / 21 B-A backbone; **Qwen3-Next** uses MTP heads at the end of its GDN-attention hybrid stack. The MTP cache layout is identical across all four: per-MTP-head a separate latent / KV buffer, smaller than the main cache.)

### 5.13 SSM State (Mamba / Mamba-2 / DeltaNet)

(Verbatim from v2 §5.13. v3 forward-reference: §5.18 documents the Mamba-3 complex-valued state.)

### 5.14 Cross-Layer KV Sharing — Gemma 4 (Apr 2026) [v3 NEW]

Distinct from CLA (§3.10) — where consecutive layers share K, V — and distinct from YOCO (§3.11 / §5.7) — where the model has a 2-stage architecture. The **Gemma 4 pattern** is: for the last `num_kv_shared_layers` decoder layers (E2B: 20 out of 35), the K and V tensors are **read from an earlier non-shared layer of the same attention type** rather than freshly computed.

**Verified configs (2026-06-06):**

| Size | `num_hidden_layers` | `num_kv_shared_layers` | Mechanism |
|---|---|---|---|
| Gemma 4 E2B | 35 | **20** | Last 20 layers reuse K/V from a matching earlier layer of the same attention type |
| Gemma 4 E4B | 42 | ≈ 20 (per `10-gemma4-investigation.md`; `[VERIFY against raw config]`) | Same pattern as E2B |
| Gemma 4 12B | 48 | 0 | No sharing |
| Gemma 4 26B-A4B | 30 | 0 | No sharing |
| Gemma 4 31B | 60 | **0** (confirmed by raw config fetch) | No sharing |

**Memory impact (E2B):** total cache cost is `(L − num_kv_shared) · (per-layer cost)` instead of `L · (per-layer cost)` — a ~57 % reduction on E2B. Combined with K = V unification (`attention_k_eq_v`), p-RoPE, and the small head count (1 KV head on E2B globals), Gemma 4 E2B's per-token cache footprint is among the smallest of any 2026 SLM.

**Mechanism details.** The shared layer references an earlier producer-layer's cache via a pointer in the model spec. The producer layer **must be of the same attention type** (sliding-to-sliding or global-to-global) — you cannot share a sliding-layer's K/V into a global layer. This is enforced architecturally by the alternation pattern: the 4 : 1 sliding/global cycle means each "shared layer" maps to the corresponding non-shared layer 5 (or 10, or 15, …) positions earlier.

**Cache layout impact.** v3 introduces a `share_scheme` enum on `KVCacheSpec` with values `{none, gemma4, apple_afm, cla, yoco}`. Per layer, a `kv_source_layer: Optional[int]` carries the producer-layer index when `share_scheme != none`.

**Apple AFM contrast (see §5.15).** Apple's 2-block sharing is a strict **2-block partition** (62.5 / 37.5 split, distinct producer and consumer blocks); Gemma 4's is **fine-grained per-layer pointer**. Distinct enough that v3 carries them as two separate `share_scheme` values.

**Shipped:** Gemma 4 E2B (Apr 2026). E4B partially (pattern claimed by `10-gemma4-investigation.md` but `[VERIFY]`).

### 5.15 Apple AFM Cross-Block KV Sharing (Apple Foundation Model on-device 3.18B, 2025) [v3 NEW]

Apple's on-device foundation model (arXiv 2507.13575, July 2025; shipped to every iPhone 15 Pro+ and Apple Silicon Mac under "Apple Intelligence") uses a **2-block partition** of layers:

- **Block 1** comprises the first 62.5 % of layers (~20 out of 32 in the 3.18 B model). Block 1 layers compute their own K and V.
- **Block 2** comprises the last 37.5 % (~12 out of 32). Block 2 layers **reuse Block 1's K and V** entirely — no fresh K, V projection.

This is distinct from:
- **YOCO** (§3.11 / §5.7), which is also a 2-block scheme but with a **single global cache** produced at the stage-1/stage-2 boundary and consumed via cross-attention. Apple AFM keeps per-layer attention semantics and just aliases K/V across blocks.
- **CLA** (§3.10), which shares K/V across **consecutive pairs** of layers throughout the model. Apple AFM does a single 2-block split.
- **Gemma 4 cross-layer sharing** (§5.14), which uses fine-grained per-layer pointers within the same attention type. Apple AFM's is a strict block partition.

**Per-token cache.** With block-1 being 62.5 % of layers, Apple AFM's cache cost is 62.5 % of a baseline (vs Gemma 4 E2B's ~43 %). The 2-bit QAT quantization on top further reduces footprint.

**Other axes.** Apple AFM uses MHA / GQA on Block 1 (per the tech report) and effectively **frozen-K/V cross-attention** on Block 2. The kernel for Block 2 is therefore a cross-attention kernel against the Block 1 cache — implementable on FA3 with `kv_cache` as a frozen pointer.

**Shipped:** Apple Intelligence on-device (deployed at scale; weights closed but architecture fully described in arXiv 2507.13575).

**v3 axis addition:** `share_scheme = apple_afm` with `share_block_split: float` (defaulting to 0.625) to express the 62.5/37.5 partition.

### 5.16 Per-Layer Embeddings Cache (Gemma 4 E2B / E4B, Apr 2026) [v3 NEW]

Gemma 4 E2B and E4B introduce **Per-Layer Embeddings (PLE)** — a second embedding table whose output is injected as a residual signal at **every decoder layer**. Mechanically:

```
PLE_table:                 shape [vocab_size, d_ple]    # d_ple = 256 typical (vs d_model = 1536 / 2560)
ple_proj[l]:               shape [d_ple, d_model]       # per-layer projection
per_layer_residual[l, t]:  ple_proj[l] @ PLE_table[token_id[t]] · (1/sqrt(2))
                           + (token_identity contribution)
```

The `per_layer_residual` is added to the residual stream at every decoder layer of the E2B/E4B variants. The PLE table can be **paged out to flash storage** on mobile devices — only the per-layer projection sits in VRAM during forward. This is the mechanism behind E2B's "effective 2.3 B" vs "5.1 B with embeddings" parameter split: roughly half the parameter mass lives in the PLE table that does not need to be loaded into VRAM during forward.

**Why this is a cache-layout entry and not just an embedding entry.** The PLE acts as a **layer-indexed cache slot**: for each (layer, token-position), the PLE residual must be looked up. In a runtime, the PLE lookup is either:
- recomputed per layer (no cache; cheap if PLE table fits in fast memory),
- cached as a `[B, L, T, d_model]` tensor populated once and read per layer (saves recompute but costs `L × T × d_model` of memory),
- or paged from flash on each access (paged-to-flash mode).

For multimodal inputs, the PLE component is computed before soft tokens are merged; multimodal positions use the PAD token ID as their PLE lookup key.

**API axis addition:** `KVCacheSpec.per_layer_embedding: Optional[PLESpec]` with fields:
```
PLESpec {
  ple_table_dim:        int                  # 256
  ple_projection_dim:   int                  # d_model
  paged_to_flash:       bool                 # default false
  reduce_scale:         float                # 1/sqrt(2) typical
  modality_handling:    dict[modality, str]  # how non-text modalities look up PLE
}
```

**Shipped:** Gemma 4 E2B, Gemma 4 E4B (Apr 2026).

### 5.17 Lightning Indexer Separate State (DeepSeek-V3.2, Sep 2025) [v3 NEW]

DeepSeek-V3.2's DSA (§3.30) requires a **separate per-layer indexer cache** distinct from the MLA latent:

```
indexer_state:   shape [N_blocks, block_size, d_indexer]   # d_indexer ~ 32-64 [VERIFY]
mla_latent:      shape [N_blocks, block_size, kv_lora_rank + qk_rope_head_dim]   # as V2
```

Per token, both buffers are written. At decode time, the indexer is scored against the current query's compressed projection; the top-`k` positions are then read from the MLA latent.

**Constraints:**
- The indexer buffer is **append-only** like KV; the kernel must support indexer-buffer write on prefill and append on decode.
- The kernel must support a `top-k → gather → MLA attention` pipeline. vLLM's experimental DSA backend (2026-Q1) implements this via Triton.
- Prefix-cache reuse: the prefix-cache hash must include the indexer in the cache key, otherwise reused prefixes would skip indexer build-up.

**Shipped:** DeepSeek-V3.2, DeepSeek-V3.2-Exp (Sep – Dec 2025), GLM-5.1 (Apr 2026; same DSA primitive).

### 5.18 Mamba-3 Complex-State Cache (Mar 2026) [v3 NEW]

Mamba-3 (§3.34) requires a **complex-valued state cache** rather than the real-valued state of Mamba-1 / -2:

```
complex_ssm_state:  shape [B, d_inner, d_state] in C    # represented as 2× real
conv_state:         shape [B, d_inner, d_conv - 1] in R  # unchanged
```

The state is twice the byte count of Mamba-2 at the same `d_state`. Reference implementations represent the complex state as a stacked pair of real tensors `(real_state, imag_state)` for compatibility with PyTorch's real-only autograd.

**MIMO-decoding output buffers.** MIMO's multi-output decoding produces `n_outputs` logits per step from a single SSM state. There is no cache impact (outputs are not stored), but the kernel signature is `step(h_t) → list[logits]` rather than `step(h_t) → logits`. The runtime must accept the multi-output return and forward to whatever selector the model uses.

**Constraint:** complex-arithmetic SSM kernels (cuFFT-adjacent fused kernels) are not yet in production runtimes; reference impl in `state-spaces/mamba` does real-paired emulation.

**Shipped:** Mamba-3 reference scales (180 M – 1.5 B, Mar 2026). No production large-scale Mamba-3 yet.

---

## §6 Prefill vs Decode Kernel Taxonomy

§6 is carried forward from v2 unchanged; the 2026-H1 additions in §3 / §4 / §5 fit cleanly into the existing prefill / decode / chunked-prefill / speculative-decode regime. The only v3-relevant kernel-side notes:

### 6.1 Three Regimes

| Regime | Q tokens | K tokens | Best kernel |
|---|---|---|---|
| **Prefill** | T (prompt) | T (prompt) | FlashAttention-2 / -3 (`causal=True`) |
| **Decode** | 1 | T (cache) | FlashDecoding / `flash_attn_with_kvcache` / vLLM `paged_attention_v2` |
| **Chunked prefill** | C (chunk, 256-2048) | T_so_far | FlashAttention with `cu_seqlens_q ≠ cu_seqlens_k` |

### 6.2 Chunked Prefill (DeepSpeed-MII, Sarathi-Serve)

(Verbatim from v2 §6.2.)

### 6.3 Speculative Decoding Compatibility — extended for Gemma 4 MTP and Hy3 MTP

(v2 §6.3 + v3 note: Gemma 4 MTP Drafters (Apr 2026) ship as **separate small heads** that produce drafts validated by the main model. Same kernel as chunked prefill on the verifier side.)

### 6.4 KV Cache State During Speculation

(Verbatim from v2 §6.4.)

### 6.5 Hopper TMA + Warp Specialization (FA3 Specifics)

(Verbatim from v2 §6.5.)

### 6.6 New for v3: NVFP4 KV-Cache Path (NVIDIA Nemotron 3 + DeepSeek-V4-Pro-NVFP4)

NVIDIA's NVFP4 (4-bit Hopper-native floating point) is now an active KV cache dtype. The NVFP4 KV path requires:
- Hopper or newer architecture (or Blackwell B100/B200).
- A per-block scale of `e8m0` (exponent-only) and a per-tensor `descale` factor.
- Custom dequant + matmul in the attention kernel.

**Shipped:** NVIDIA `DeepSeek-V4-Pro-NVFP4` (Apr 2026); Nemotron 3 Super / Nano Omni / Ultra (Mar / Apr / Jun 2026) ship NVFP4 native by default.

### 6.7 New for v3: Visual-Causal Mask Kernel Primitive

The DeepSeek-OCR (§3.35) block-bidirectional visual flow requires a kernel that accepts a `block_starts: int[N_blocks]` tensor and applies bidirectional attention within blocks, causal across. FlashInfer added this as a fused primitive in v0.3.0 (late 2025); vLLM's MM backend in v0.9.0+ exposes it via `mask_kind = "visual_causal_flow"`.

---

## §7 Parameter Space — Revised through 2026-H1

### 7.1 The KV Cache Parameter Space (v3 — 12+ axes)

```
KVCacheSpec {
  // Shape
  n_layers:               int
  per_layer_shape:        list[KVLayerShape]      # len n_layers, supports hybrid

  // Layout
  layout:                 {HND, NHD}
  storage:                {contiguous, paged_v1, paged_v2_vm, ring_swa, yoco_global,
                           dsa_dual_state, csa_hca_quad_state, complex_ssm}      # NEW v3
  block_size:             int                     # paged: 16 vLLM default; contig: 1; ring: window W
  unified_layers:         bool                    # one big tensor vs per-layer list

  // Sharing (v3: expanded enum)
  share_scheme:           {none, cla, yoco, gemma4, apple_afm}   # NEW v3
  num_kv_shared_layers:   int                     # NEW v3 — Gemma 4 E2B: 20
  share_block_split:      float | None            # Apple AFM only — 0.625 typical
  kv_source_layer:        list[int | None]        # per-layer producer-layer pointer, len n_layers
  cache_role:             {standard, yoco_producer, yoco_consumer, afm_block1, afm_block2}

  // Dtype
  k_dtype:                DType                   # bf16 / fp16 / fp8_e4m3 / int8 / int4_group / int2 / nvfp4   # NEW: nvfp4
  v_dtype:                DType                   # often differs from k_dtype (KIVI)
  k_group_size:           int | None
  v_group_size:           int | None
  k_quant_axis:           {channel, token, head, block, tensor}
  v_quant_axis:           {channel, token, head, block, tensor}
  k_scale_dtype:          DType                   # fp16 / fp32 / e8m0 / mx
  v_scale_dtype:          DType

  // Addressing
  ownership:              {explicit_pass, stateful}
  request_addressing:     {batch_index, slot_mapping}
  prefix_reuse:           {none, copy_seq, paged_share, radix_tree, mla_latent_hash, dsa_indexer_hash}    # NEW v3

  // Bounds
  max_seq_len:            int
  ring_window:            int | None              # per-layer if interleaved
  sink_slots:             int | None              # 4 typical when present
  n_mtp_heads:            int                     # 0 default; DeepSeek-V3: 1; Gemma 4 MTP drafter: small

  // Per-layer embedding (v3 NEW for Gemma 4 E2B/E4B)
  per_layer_embedding:    PLESpec | None
}

KVLayerShape {
  kind: {standard_kv, mla_latent, k_equals_v,                                         # NEW v3: k_equals_v
         dsa_dual_state, csa_hca_quad_state,                                          # NEW v3
         ssm_state, ssd_state, complex_ssm_state,                                     # NEW v3: complex_ssm_state
         delta_state, retnet_state, mlstm_matrix, xpos_state,
         mutable_weights, cross_attn_frozen, ple_residual}                            # NEW v3: ple_residual
  n_kv_heads:             int                     # 1=MQA, G=GQA, H=MHA
  q_head_dim:             int
  k_head_dim:             int
  v_head_dim:             int

  // MLA-specific
  kv_lora_rank:           int | None              # 512 / 256
  qk_rope_head_dim:       int | None              # 64 / 32
  qk_nope_head_dim:       int | None              # 128 / 64

  // SSM-specific (Mamba-1)
  d_state:                int | None              # 16 typical
  d_inner:                int | None              # 2 × hidden_size typical
  d_conv:                 int | None              # 4 typical

  // Mamba-2 SSD-specific
  chunk_size:             int | None              # 256 typical
  headdim:                int | None              # 64 typical
  ngroups:                int | None              # 1, 2, 4, 8

  // Mamba-3 complex-state specific (v3 NEW)
  is_complex:             bool                    # true for Mamba-3
  n_mimo_outputs:         int | None              # n_outputs per step

  // DeltaNet
  delta_d_key:            int | None
  delta_d_value:          int | None
  delta_gate_dim:         int | None

  // Lightning Indexer (v3 NEW)
  indexer_dim:            int | None              # d_indexer ~ 32-64
  indexer_top_k:          int | None              # 256-1024 typical

  // CSA / HCA (v3 NEW)
  csa_block_size:         int | None              # l_csa
  hca_block_size:         int | None              # l_hca >> l_csa
  csa_summary_dim:        int | None              # d_csa
  hca_summary_dim:        int | None              # d_hca
  csa_top_k:              int | None
  hca_top_k:              int | None
}

PLESpec {                                          # v3 NEW
  ple_table_dim:        int                       # 256
  ple_projection_dim:   int                       # d_model
  paged_to_flash:       bool                      # default false
  reduce_scale:         float                     # 1/sqrt(2) typical
  modality_handling:    dict                      # multimodal placeholders use PAD token id
}
```

### 7.2 The Attention Parameter Space (v3 — 26+ axes)

```
AttentionSpec {
  // Token-mixer kind
  kind: {mha, gqa, mqa, mla, dsa, csa_hca, msa, differential, nsa,                     # NEW v3: dsa, csa_hca, msa
         retnet, gla, deltanet, mamba1, mamba2_ssd, mamba3_complex,                    # NEW v3: mamba3_complex
         hyena, mlstm, slstm, ttt, cross_attn, none}

  // Projection topology
  n_q_heads:              int
  n_kv_heads:             int                     # MHA: =n_q; GQA: <n_q; MQA: 1
  q_head_dim:             int
  k_head_dim:             int
  v_head_dim:             int

  // K = V tying (v3 NEW — Gemma 4 ≥12B global layers)
  attention_k_eq_v:       bool                    # true on Gemma 4 12B/26B/31B global layers

  // Global head dim (v3 NEW — Gemma 4 doubles head_dim on global layers)
  global_head_dim:        int | None              # 512 on Gemma 4 (vs head_dim = 256 on locals)

  // MLA extras
  q_lora_rank:            int | None
  kv_lora_rank:           int | None
  qk_nope_dim:            int | None
  qk_rope_dim:            int | None

  // Differential extras
  n_q_streams:            int
  lambda_init:            float | None

  // NSA extras
  nsa_compress_block:     int | None
  nsa_select_block:       int | None
  nsa_top_n:              int | None
  nsa_sliding_window:     int | None

  // Lightning Indexer DSA (v3 NEW)
  lightning_indexer:      IndexerSpec | None

  // CSA + HCA (v3 NEW)
  csa_spec:               CSASpec | None
  hca_spec:               HCASpec | None

  // Mask
  mask: {causal, sliding_window, sliding_with_sink, full, block_diag, custom_2d,
         tree, nsa_three_branch, dsa_top_k_with_warm, visual_causal_flow, msa}        # NEW v3
  window_size:            int | None
  n_sink_tokens:          int | None
  sink_trained:           bool
  block_bidirectional_mask: bool                  # v3 NEW — DeepSeek-OCR

  // Norm
  qk_norm:                {none, pre_rope_rms, post_rope_rms}
  qk_norm_axis:           {per_head, per_layer}
  qk_norm_fixed_scale:    float | None            # v3 NEW — Gemma 4 fixed-scale (0.9916 local / 1.0228 global)

  // Numerics
  attn_scale:             float | None
  logit_softcap:          float | None
  final_logit_softcap:    float | None            # v3 NEW — Gemma 4 restores final softcap 30.0
  attn_temperature:       float | None            # v3 NEW — Llama 4 per-layer attn-temp

  // Output gate (v3 NEW — Qwen3-Next softmax layers)
  output_gate:            bool

  // Per-layer apply_rope (NoPE alternation, iRoPE)
  apply_rope:             bool                                                          # SmolLM3, Llama-4 iRoPE
  apply_rope_per_layer:   list[bool] | None       # v3 NEW — model-level convenience for iRoPE

  // RoPE
  rope:                   RoPESpec | None         # per-layer (Gemma 3/4 dual-θ)

  // Cross-layer KV
  shares_kv_with:         int | None              # layer idx of producer (CLA)
  cache_role:             {standard, yoco_producer, yoco_consumer,
                           afm_block1, afm_block2, gemma4_producer, gemma4_consumer}   # NEW v3

  // SSM extras (kind ∈ mamba*)
  d_state:                int | None
  d_inner:                int | None
  d_conv:                 int | None
  dt_min:                 float | None
  dt_max:                 float | None
  dt_rank:                int | None
  A_init_range:           tuple[float, float] | None

  // Mamba-2 SSD extras
  chunk_size:             int | None
  headdim:                int | None
  ngroups:                int | None

  // Mamba-3 extras (v3 NEW)
  is_complex_state:       bool                    # true for Mamba-3
  n_mimo_outputs:         int | None
}

IndexerSpec {                                     # v3 NEW
  d_indexer:              int                     # ~32-64
  top_k:                  int                     # 256-1024
  warm_window:            int                     # very-recent fixed window
  share_with_mla:         bool                    # whether indexer reads from MLA latent
}

CSASpec / HCASpec {                               # v3 NEW
  block_size:             int                     # l_csa / l_hca
  summary_dim:            int                     # d_csa / d_hca
  top_k:                  int
  compression_levels:     int                     # 1 for CSA, >=1 for HCA
}

RoPESpec {
  basis:                  {interleaved, split_half}
  partial_strategy:       {full, partial_first_k, gptj_partial_interleaved, glm_index_parity}
  base_theta:             float                   # 10000 / 500000 / 1000000 / 10000000 (per-layer in Gemma 3/4)
  d_rope:                 int                     # = head_dim or partial
  partial_rotary_factor:  float | None            # v3 NEW — 0.25 on Gemma 4 global; 0.5 on Phi-1
  position_axes:          int                     # 1 (text), 2 (2D-RoPE), 3 (M-RoPE T/H/W), 4 (TMRoPE +frame)
  mrope_section:          tuple[int, ...] | None  # v3 NEW — (16,24,24) on Qwen2.5-VL
  is_2d:                  bool                    # v3 NEW — vision-tower-only 2D-RoPE

  scaling: {none, linear_pi, ntk_static, ntk_dynamic, yarn, llama3, longrope, dca, proportional}   # NEW v3: proportional
  scale_factor:           float | None
  original_max_pos:       int | None

  // YaRN
  attn_factor:            float | None
  beta_fast:              float | None
  beta_slow:              float | None
  mscale_all_dim:         list[float] | None

  // Llama-3
  low_freq_factor:        float | None
  high_freq_factor:       float | None

  // LongRoPE / Phi-3
  short_factor:           list[float] | None
  long_factor:            list[float] | None
  long_threshold:         int | None

  // DCA
  chunk_size:             int | None
  inter_chunk_offset:     int | None
}
```

**Per-layer everything.** Every axis in `AttentionSpec`, `RoPESpec`, and `KVLayerShape` must be expressible per-layer. Phi-4-mini-flash, Gemma 3, Gemma 4, SmolLM3, Llama-4 iRoPE, Qwen3-Next, Granite 4 H, Nemotron 3, MiniMax-Text-01, DeepSeek-V3.2 / V4 all require this.

### 7.3 Counts (v3)

- **Attention variants documented: 38** (MHA, MQA, GQA, MLA, SWA, interleaved local/global, sink-inference + sink-trained, block-diagonal, NSA, CLA, YOCO, Differential, Mamba-1, Mamba-2 SSD, GLA, DeltaNet, Gated DeltaNet 3:1, RetNet, Hyena, TTT, mLSTM, sLSTM, cross-attention, FA1, FA2, FA3, FlashDecoding, RingAttention, Striped, TreeAttention, RadixAttention, PagedAttention v1, vAttention, **Lightning Indexer DSA**, **CSA + HCA**, **K = V unification**, **Mamba-3 complex + MIMO**, **Visual Causal Flow / block-bidirectional**, **Lightning 7:1 hybrid**, **Output-gated softmax**, **Trained sinks (refined)**, **MSA**). 32 token-mixer variants if FA1/2/3 and the parallel-decode kernels are excluded.
- **RoPE variants: 18** (vanilla, PI, NTK-static, NTK-dynamic, YaRN, Llama-3 smooth, LongRoPE, DCA, partial (historical), NoPE, xPos, Sandwich, M-RoPE, MM-RoPE, **p-RoPE per global layer**, **iRoPE per-layer**, **M-RoPE/2D/TMRoPE/V2PE expanded**, **V2PE vision-tower variants**) plus the basis axis (interleaved vs split-half) and the ALiBi baseline.
- **Cache layouts: 18** (contiguous, paged-v1, paged-v2-vAttention, ring-SWA, MLA-latent, YOCO-global, NSA-three-tier, MTP-head, SSM state, SSD state, DeltaNet state, beam/tree, RadixAttention prefix-tree, **Gemma 4 cross-layer KV sharing**, **Apple AFM cross-block**, **Per-Layer Embeddings cache (paged-to-flash)**, **Lightning Indexer separate state**, **Mamba-3 complex-state**).

---

## §8 Compatibility / Constraint Matrix

§8 carries forward v2's matrices verbatim except for the additions below.

### 8.1 Attention × RoPE — extended rows for 2026-H1

| Attention ↓ \ RoPE → | Vanilla | PI | NTK-dyn | YaRN | Llama-3 | LongRoPE | DCA | Partial / p-RoPE | NoPE / iRoPE | ALiBi |
|---|---|---|---|---|---|---|---|---|---|---|
| MHA | ✔ Phi-3-mini-4k | ✔ Code-Llama-16K | ✔ ChatGLM-2 | ~ | ~ | ✔ Phi-3-mini-128k | ~ | ✔ Phi-2 | ✔ NoPE OLMo | ✔ MPT |
| GQA | ✔ Llama-3, Mistral, Gemma 4 sliding | ✔ LLongMA | ~ | ✔ Qwen2.5-1M | ✔ Llama-3.1/3.2 | ✔ Phi-3.5-MoE | ✔ Qwen2.5-1M | ✔ StableLM-2, ✔ Gemma 4 global p-RoPE | ✔ SmolLM3, Llama-4 iRoPE | ~ |
| MQA | ✔ Falcon-180B, Gemma 4 E2B sliding | ~ | ~ | ~ | ~ | ~ | ~ | ✔ Gemma 4 E2B global p-RoPE | ~ | ✔ Falcon-7B/40B |
| MLA | ✘ structural | ✘ | ✘ | ✔ DeepSeek-V2 (qk_rope only) | ~ | ~ | ~ | ✔ required (qk_rope=64) | ✘ | ✘ |
| **DSA (MLA + Lightning Indexer)** | ✔ DeepSeek-V3.2, GLM-5.1 | ~ | ~ | ✔ at long ctx | ~ | ~ | ~ | inherits MLA partial | ✘ | ✘ |
| **CSA + HCA (DeepSeek-V4)** | ✔ V4-Pro / Flash | ~ | ~ | ✔ at long ctx | ~ | ~ | ~ | inherits MLA partial | ✘ | ✘ |
| SWA (local only) | ✔ Mistral-v0.1, Gemma 4 sliding | ~ | ~ | ~ | ✔ Mistral-Nemo | ~ | ~ | ~ | ~ | ~ |
| SWA + global alternation | ✔ Gemma 2 | ✘ | ~ | ~ | ~ | ~ | ~ | **✔ Gemma 4 (sliding full + global p-RoPE)** | ✔ Llama-4 iRoPE (some NoPE) | ~ |
| Sink (inference-time) | ✔ on any | ✔ | ✔ | ✔ | ✔ | ✔ | ~ | ✔ | ✔ | ✔ |
| Sink (trained) | ✔ GPT-OSS, Mistral-Small-3.1, Granite 4 H | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| NSA | ✔ DeepSeek 2025 | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| **K = V unification** | ✔ Gemma 4 12B/26B/31B (global) | ✘ | ✘ | ✘ | ✘ | ✘ | ✘ | ✔ composed with p-RoPE | ✘ | ✘ |
| Differential | ✔ ref | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| Mamba-1 / -2 / **-3 complex** | n/a (no RoPE on SSM layers) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| GLA / DeltaNet / Gated DeltaNet | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Key new constraints for v3:**
- **Gemma 4 forces per-layer-type RoPE config** including `partial_rotary_factor` only on global layers. Most current libraries store one RoPE config per model; vLLM v0.10+ and HF transformers v5.5+ are tracked as supporting per-layer-type instantiation.
- **DSA + paged**: the indexer cache must be paged independently from the MLA latent. Block boundaries should align between the two pools.
- **CSA + HCA + paged**: four cache tensors per layer, each with its own block-table and slot mapping.
- **Mamba-3 complex state** is incompatible with any RoPE / paged / sink infrastructure — it lives on the SSM side. Hybrid models with Mamba-3 layers + softmax layers will allocate Mamba-3 layers as flat complex tensors and softmax layers paged.
- **K = V unification on global layers** forces the kernel to alias K and V pointers; older CUTLASS kernels assume independent pointers.

### 8.2 Attention × Cache Layout — extended rows

| Attention ↓ \ Cache → | Contig | Paged v1 | vAtt | Ring SWA | YOCO Glob | NSA 3-tier | SSM | **Gemma4 share** | **AFM cross-block** | **DSA indexer + MLA** | **CSA/HCA quad** | **complex SSM** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MHA | ✔ | ✔ | ✔ | n/a | ~ | ✘ | n/a | ✘ | ✔ Apple AFM block-1 | ✘ | ✘ | n/a |
| GQA | ✔ | ✔ | ✔ | ✔ Mistral | ~ | ✘ | n/a | ✔ Gemma 4 globals | ~ | ✘ | ✘ | n/a |
| MQA | ✔ | ✔ | ✔ | ~ | ~ | ✘ | n/a | ✔ Gemma 4 E2B locals | ~ | ✘ | ✘ | n/a |
| MLA | ✔ | ✔ vLLM 0.6.3+ | ~ | ✘ | ✘ | ✘ | n/a | ✘ | ✘ | ✔ DSA | ✔ CSA/HCA | n/a |
| K = V global | ✔ alias | ✔ alias | ✔ | ✘ | ✘ | ✘ | n/a | ✔ Gemma 4 ≥12B | ~ | ✘ | ✘ | n/a |
| SWA local | ✔ | ✔ | ✔ | ✔ ideal | ~ | ✘ | n/a | ✔ Gemma 4 sliding | ~ | ✘ | ✘ | n/a |
| Gemma 4 interleaved | ✔ (two pools) | ✔ (per-layer block table) | ✔ | partial | ✘ | ✘ | n/a | **✔ native — `num_kv_shared_layers`** | ✘ | ✘ | ✘ | n/a |
| YOCO | ✔ | ✔ | ✔ | n/a | ✔ | ✘ | n/a | ✘ | ✘ — distinct scheme | ✘ | ✘ | n/a |
| Mamba-1 / -2 SSM | n/a | n/a | n/a | n/a | n/a | n/a | ✔ | n/a | n/a | n/a | n/a | ✘ (real-state) |
| **Mamba-3 complex SSM** | n/a | n/a | n/a | n/a | n/a | n/a | ~ via real-pairing | n/a | n/a | n/a | n/a | ✔ native |

**Key new constraints for v3:**
- **Gemma 4 cross-layer KV sharing** requires per-layer cache pointers; the kernel sees a layer whose K/V tensor is aliased to a different layer's storage. vLLM's `BlockSpaceManager` would need a per-model `kv_source_layer` map at allocator time.
- **Apple AFM cross-block** can be implemented as a YOCO-like 2-stage cache but the block-1 layers still maintain per-layer attention (unlike YOCO's "all stage-2 layers cross-attend to one global cache"). It is a hybrid.
- **DSA indexer + MLA paged**: vLLM's MLA path needs to be extended with a parallel indexer pool. Prefix-cache hash key must include the indexer state.
- **CSA + HCA + paged**: each tier needs its own block table. Prefix cache reuse only matches when the prefix's CSA and HCA buffers are identical — i.e., the prefix-cache hash must include block-summary state from all four tiers.

### 8.3 RoPE × Cache Layout — v2 §8.3 verbatim plus

| RoPE ↓ \ Cache → | Gemma 4 share | DSA indexer | CSA/HCA |
|---|---|---|---|
| p-RoPE (Gemma 4 global) | ✔ — only `partial_rotary_factor · head_dim` slots are rotated | ~ (DSA indexer typically not rotated) | ~ |
| iRoPE (Llama 4 per-layer) | ~ (different family) | ~ | ~ |
| M-RoPE / MM-RoPE / TMRoPE | ✔ (3-4 position axes per token) | ~ | ~ |

### 8.4 Composite Constraints — v2 plus v3 additions

(v2 items 1-8 verbatim.)

**v3 additions:**

9. **Gemma 4 K = V × paged × prefix-cache reuse.** When `attention_k_eq_v = true`, the cache hash must use the single KV tensor as input; vLLM's per-tensor hashing already supports this trivially.
10. **DSA × chunked prefill.** The indexer must be built incrementally as chunks arrive; the chunk boundary must align with `block_size` to keep the indexer-cache and MLA-cache slot-mappings consistent.
11. **CSA + HCA × continuous batching.** The four-tier cache adds 4 × the per-request bookkeeping overhead of MLA. vLLM's block manager scales linearly in the number of caches; cost is acceptable but throughput drops vs MLA-only at small batch sizes.
12. **Gemma 4 PLE × paged-to-flash.** When the PLE table is paged out, a per-token read from flash is required at the first decoder layer. This is a 50-200 µs latency hit per cold token; mobile runtimes (MediaPipe LLM, Core ML) have a pre-fetch path that overlaps the PLE read with the prior layer's compute.
13. **Mamba-3 complex state × hybrid.** A hybrid Mamba-3 + softmax-attention model must allocate two distinct cache pools — complex SSM state (flat per-request) and softmax KV (paged). vLLM's `KVCacheManager` would need a Mamba-3 specific allocator; reference impls do this with a flat tensor pool.
14. **iRoPE × prefix cache.** NoPE layers' cached K/V are position-implicit — i.e., the prefix matches only if it is at the **same absolute prefix position** in the new request. This is a tighter match requirement than RoPE layers (where the rotation is computed from absolute pos anyway).
15. **Visual Causal Flow × paged.** The block-bidirectional mask requires per-token-type info to flow into the kernel's mask compute. vLLM's MM backend stores a per-token `is_vision: bool` and a `vision_block_id: int` alongside the slot mapping.

---

## §9 Evolution Narrative — Dated Arcs, Refreshed Through 2026-H1

### 9.1 Attention Evolution (29 → 38 entries)

| Date | Variant | Citation | Adoption |
|---|---|---|---|
| Jun 2017 | MHA | Vaswani et al. arxiv:1706.03762 | Universal baseline |
| Nov 2019 | MQA | Shazeer arxiv:1911.02150 | PaLM (2022) |
| May 2023 | GQA | Ainslie et al. arxiv:2305.13245 | Llama-2-70B then universal |
| Sep 2023 | SWA | Mistral 7B v0.1 | Mistral, Phi-3-mini-128k |
| Dec 2023 | Mamba-1 | Gu & Dao arxiv:2312.00752 | Mamba-2.8B, Falcon-Mamba-7B |
| Mar 2024 | Jamba (Mamba+MoE+attn) | Lieber et al. arxiv:2403.19887 | Jamba-Mini-1.6, Jamba2 (Jan 2026) |
| May 2024 | MLA | DeepSeek-V2 | DeepSeek family, MiniCPM-3 |
| May 2024 | CLA | Brandon et al. arxiv:2405.12981 | Apple OpenELM |
| May 2024 | YOCO | Sun et al. arxiv:2405.05254 | YOCO-1.3B, Phi-4-mini-flash |
| May 2024 | Mamba-2 SSD | Dao & Gu arxiv:2405.21060 | Zamba-2, Hymba |
| Jun 2024 | Gemma 2 interleaved 1:1 | Gemma 2 tech report | Gemma 2 |
| Jun 2024 | Samba alt Mamba/SWA | Ren et al. arxiv:2406.07522 | Samba 1.7B/3.8B |
| Jun 2024 | RecurrentGemma Griffin/Hawk | Botev et al. arxiv:2402.19427 | RecurrentGemma-2B/9B |
| Oct 2024 | Differential Transformer | Ye et al. arxiv:2410.05258 | DiffTransformer ref |
| Nov 2024 | Hymba parallel SSM+attn | Hu et al. arxiv:2411.13676 | Hymba-1.5B |
| **Jan 2025** | **MiniMax Lightning 7:1 hybrid** | MiniMax-Text-01 arxiv:2501.08313 | MiniMax-Text-01 |
| Feb 2025 | NSA | Yuan et al. arxiv:2502.11089 | DeepSeek tech preview |
| Mar 2025 | Gemma 3 (5:1, no softcap) | Gemma 3 tech report | Gemma 3 |
| Mar 2025 | Mistral Small 3.1 trained sinks | Mistral release | Mistral Small 3.x |
| Apr 2025 | Llama-4 iRoPE (NoPE alternation + attn-temp) | Llama-4 release | Llama-4 Scout/Maverick |
| **Aug 2025** | **GPT-OSS** (trained sinks, MXFP4, alternating dense+banded) | OpenAI release | GPT-OSS 20B/120B |
| **Aug 2025** | **Qwen3-Next Gated DeltaNet 3:1 hybrid** | Qwen3-Next release | Qwen3-Next 80B-A3B |
| **Sep 2025** | **Lightning Indexer DSA** | DeepSeek-V3.2-Exp arxiv:2512.02556 | DeepSeek-V3.2, GLM-5.1 (Apr 2026) |
| **Oct 2025** | **Granite 4 H 9:1 Mamba-2 : attention** | IBM Granite 4 launch | Granite 4 H-Micro/H-Tiny/H-Small |
| **Oct 2025** | **DeepSeek-OCR Visual Causal Flow / block-bidirectional mask** | DeepSeek-OCR release | DeepSeek-OCR, DeepSeek-OCR-2 (Jan 2026) |
| **Mar 2026** | **Mamba-3 complex-state + MIMO** | arxiv:2603.15569 (ICLR 2026) | Mamba-3 reference scales |
| **Mar 2026** | **NVIDIA Nemotron 3 Super hybrid Mamba-2 + MoE Transformer** | NVIDIA blog | Nemotron 3 Super 120B/12B-A |
| **Apr 2026** | **DeepSeek-V4 CSA + HCA** | DeepSeek-V4 tech report | V4-Pro 1.6T/49B-A; V4-Flash 284B/13B-A |
| **Apr 2026** | **Gemma 4 K = V on global layers + cross-layer KV share + p-RoPE** | Google blog 2026-04-02 + raw configs | Gemma 4 family |
| **Apr 2026** | **Apple AFM cross-block KV (arXiv-disclosed 2025-07)** | arXiv 2507.13575 | Apple Intelligence on-device |
| **Apr 2026** | **Llama 4 iRoPE (production-scale)** | Llama 4 launch | Scout/Maverick |
| **Apr 2026** | **Tencent Hy3 MTP layer** | Tencent blog | Hy3-Preview 295B/21B-A + 3.8B MTP |
| **Jun 2026** | **MiniMax Sparse Attention (MSA)** | MiniMax-M3.0 blog | MiniMax-M3.0 (weights staged) |
| **Jun 2026** | **NVIDIA Nemotron 3 Ultra hybrid 550B/55B-A** | NVIDIA Nemotron 3 Ultra release | Nemotron 3 Ultra |

**Refreshed dependency graph (v3):**
- **Cache-pressure branch:** MHA → MQA → GQA → MLA → NSA → DSA → CSA + HCA → K = V tying.
- **Mask-shape branch:** causal → SWA → interleaved → trained-sink → block-bidirectional (Visual Causal Flow).
- **Token-mixer branch:** attention → Mamba-1 → Mamba-2 → Hymba/Samba/Phi-4-mini-flash → Gated DeltaNet 3:1 / Lightning 7:1 → Mamba-3 complex + MIMO.
- **Architecture-topology branch:** per-layer-independent → CLA → YOCO 2-stage → Apple AFM 2-block → **Gemma 4 fine-grained cross-layer sharing**.
- **Numeric branch:** vanilla softmax → softcap (Gemma 2) → no softcap (Gemma 3) → differential pair (Diff-Transformer) → branch-mixture (NSA / DSA / CSA / HCA / MSA).

### 9.2 RoPE Evolution (14 → 18 entries)

| Date | Variant | Citation | Adoption |
|---|---|---|---|
| Apr 2021 | Vanilla RoPE | RoFormer | LLaMA 1 |
| Jun 2023 | PI / linear | Chen / Kaiokendev | Code-Llama-16K |
| Jul 2023 | NTK-aware static | bloc97 | early Llama-2 forks |
| Jul 2023 | Dynamic NTK | ChatGLM-2 | ChatGLM-2/3, InternLM-200K |
| Sep 2023 | YaRN | Peng, Quesnelle, Kingma arxiv:2309.00071 | DeepSeek-V2, Qwen2.5-1M, MiniCPM-3 |
| Feb 2024 | LongRoPE | Ding et al. arxiv:2402.13753 | Phi-3-mini-128k, Phi-3.5, Phi-4 |
| Apr 2024 | Llama-3 smooth | Llama-3 release | Llama-3.1/3.2 |
| Jan 2025 | DCA | An et al. arxiv:2402.17463 + Qwen2.5-1M | Qwen2.5-1M |
| Mar 2025 | Per-layer θ alternation | Gemma 3 | Gemma 3, Gemma 4 |
| Apr 2025 | **iRoPE per-layer (NoPE + attn-temp)** | Llama 4 launch | Llama 4 Scout/Maverick |
| Jul 2025 | NoPE-alternation | SmolLM3 card | SmolLM3 1.7B/3B |
| Mar 2025 | TMRoPE (Qwen2.5-Omni) | Qwen2.5-Omni release | Qwen2.5-Omni 3B/7B |
| **Apr 2026** | **p-RoPE per global layer (`partial_rotary_factor = 0.25`)** | Gemma 4 raw configs | Gemma 4 family (all sizes) |
| **Apr 2026** | **M-RoPE expansion to `mrope_section` configurable + 2D vision tower** | Qwen3-VL release | Qwen3-VL, Pixtral-2 (if released) |

**Refreshed dependency graph (v3):**
- Extrapolation branch: vanilla → PI → NTK-static → NTK-dynamic → YaRN → Llama-3 smooth.
- Per-dim learning branch: vanilla → LongRoPE (independent).
- Per-layer branch: vanilla → per-layer θ (Gemma 3) → per-layer apply_rope (SmolLM3, Llama-4 iRoPE) → per-layer p-RoPE fraction (Gemma 4).
- Chunking branch: vanilla → DCA.
- Multimodal branch: vanilla → M-RoPE (Qwen2-VL) → MM-RoPE (Qwen2.5-VL) → TMRoPE (Qwen2.5-Omni) → M-RoPE for Qwen3-VL with `mrope_section` configurable.
- Vision-tower-only branch: 2D learned PE → 2D-RoPE / V2PE (Pixtral, Gemma 4 ViT, Janus-Pro).

### 9.3 KV Cache Evolution (13 → 18 entries)

| Date | Variant | Citation | Adoption |
|---|---|---|---|
| 2022 | Contiguous | HF baseline | All early models |
| Sep 2023 | PagedAttention v1 | Kwon et al. SOSP 2023 | vLLM |
| Feb 2024 | KVQuant 2-bit | Hooper arxiv:2401.18079 | research |
| Feb 2024 | KIVI (per-channel K / per-token V) | Liu arxiv:2402.02750 | llama.cpp K-quant |
| May 2024 | MLA latent | DeepSeek-V2 | DeepSeek family |
| May 2024 | YOCO 2-stage | Sun arxiv:2405.05254 | YOCO-1.3B |
| May 2024 | RadixAttention | SGLang arxiv:2406.04692 | SGLang |
| May 2024 | vAttention (CUDA-VM) | Prabhu arxiv:2405.04437 | TRT-LLM 0.11+ |
| Sep 2024 | vLLM MLA paged + prefix | vLLM PR #8546 | vLLM 0.6.3+ |
| Nov 2024 | CacheGen | Liu SOSP 2024 | Meta production paths |
| Dec 2024 | DeepSeek-V3 MTP-head cache | V3 report | V3 + Gemma 4 MTP drafters (Apr 2026) + Hy3 (Apr 2026) |
| Feb 2025 | NSA three-tier | Yuan arxiv:2502.11089 | DeepSeek preview |
| **Jul 2025** | **Apple AFM cross-block 62.5/37.5** | arXiv 2507.13575 | Apple Intelligence |
| **Sep 2025** | **DSA Lightning Indexer separate state** | DeepSeek-V3.2 | V3.2, GLM-5.1 |
| **Apr 2026** | **Gemma 4 cross-layer KV sharing** | Gemma 4 configs | Gemma 4 E2B/E4B |
| **Apr 2026** | **Gemma 4 Per-Layer Embeddings cache (paged-to-flash)** | Gemma 4 E2B/E4B | Gemma 4 E2B/E4B |
| **Apr 2026** | **CSA + HCA quad-state cache** | DeepSeek-V4 | DeepSeek-V4-Pro / Flash |
| **Mar 2026** | **Mamba-3 complex-state cache** | arxiv:2603.15569 | Mamba-3 reference |

**Refreshed dependency graph (v3):**
- Memory-management branch: contiguous → PagedAttention v1 → vAttention.
- Quantization branch: bf16 → INT8 → INT4 → KIVI → KVQuant → NVFP4 (2026).
- Compression branch: full K, V → MLA → NSA → DSA → CSA + HCA → K = V tying.
- Architecture branch: per-layer → CLA → YOCO → Apple AFM → **Gemma 4 fine-grained cross-layer share**.
- Allocator branch: per-request → block-table → radix-tree → DSA-indexer-hash.
- Embedding-side cache branch: standard input embedding → **Per-Layer Embeddings (Gemma 4 E2B/E4B)**.
- Complex-state branch: real SSM → complex SSM (Mamba-3).

### 9.4 Hybrid Token-Mixer Evolution (refreshed)

| Date | Model | Architecture | Mixer mix |
|---|---|---|---|
| Dec 2023 | Mamba-2.8B | Pure SSM | 100 % Mamba |
| Mar 2024 | Jamba | MoE + Mamba + GQA | 4× attn + 28× Mamba + MoE every 2 |
| May 2024 | Mamba-2 SSD | Pure SSM, faster | 100 % Mamba-2 |
| Jun 2024 | Zamba2 | Shared attention block | Mamba-2 backbone + 2× shared GQA blocks interleaved |
| Jun 2024 | Samba | Per-layer Mamba/SWA alternation | Mamba + SWA-MHA per layer |
| Jun 2024 | RecurrentGemma | Linear recurrence + local attention | Griffin/Hawk |
| Nov 2024 | Hymba | Parallel SSM+attn heads per layer | Same-layer SSM head + attn head, output-fused |
| Jan 2025 | **MiniMax-Text-01** | **Lightning 7:1 hybrid** | 7 linear : 1 softmax |
| 2025 | Phi-4-mini-flash | Per-layer SSM/attn interleave | Per-layer block-class schedule |
| Aug 2025 | **Qwen3-Next 80B-A3B** | **Gated DeltaNet 3:1 hybrid** | 3 linear : 1 softmax (75 % linear) |
| Aug 2025 | Falcon-H1 | Parallel-head Mamba + attention | Per-layer parallel |
| **Oct 2025** | **Granite 4 H-Tiny / H-Small / H-Micro** | **9:1 Mamba-2 : attention** | 9 Mamba-2 : 1 softmax (90 % Mamba-2) |
| 2025 | Granite-4-tiny-hybrid (predecessor row) | MoE + Mamba + GQA | Per-layer schedule + MoE |
| **Mar 2026** | **Mamba-3** | Pure SSM, complex state + MIMO | 100 % Mamba-3 |
| **Mar 2026** | **NVIDIA Nemotron 3 Super 120B/12B-A** | **Hybrid Mamba-2 + Transformer MoE** | Per-layer schedule |
| **Apr 2026** | **Nemotron 3 Nano Omni** | Same as Super, multimodal | Per-layer schedule |
| **Apr-Jun 2026** | **MiniMax-M2 → M3.0 trajectory** | **abandoned Lightning, then MSA** | M2: 100 % softmax; M3.0: MSA sparse |
| **Jun 2026** | **NVIDIA Nemotron 3 Ultra 550B/55B-A** | Hybrid Mamba-Attention MoE at largest scale | Per-layer schedule |

**Refreshed dependency graph (v3):**
- Pure-SSM branch: Mamba-1 → Mamba-2 → Mamba-3 (complex + MIMO).
- Linear-attention branch: GLA → DeltaNet → Gated DeltaNet (Qwen3-Next 3:1).
- Layer-interleaved branch: Jamba → Samba → Phi-4-mini-flash → Granite 4 H 9:1 → Nemotron 3 Super/Ultra.
- Within-layer-parallel branch: Hymba → Falcon-H1.
- Shared-attention branch: Zamba2.
- Linear-recurrence branch: RecurrentGemma Griffin/Hawk.
- **Linear:softmax ratio frontier:** Lightning 7:1 (MiniMax-Text-01, Jan 2025) → abandoned in MiniMax-M2 (early 2026) → Gated DeltaNet 3:1 (Qwen3-Next, Aug 2025) — opposing-direction explorations that converge on "linear-attention is useful but the optimal softmax fraction is workload-dependent." MiniMax-M3.0's MSA then enters a third regime: replace the linear-attention layers entirely with a different sparse-attention scheme.
- **Hybrid Mamba-2 + transformer MoE at scale (Nemotron 3):** the canonical 2026-H1 mainstreaming of SSM-hybrid for production-scale agentic workloads.

---

## §10 Open / Forward-Looking Variants

v2's §10 items remain forward-looking. v3 adds:

- **MSA — MiniMax Sparse Attention** (MiniMax-M3.0, Jun 2026). Architecture details pending the staged HF weights release; expected ~mid-June 2026.
- **Mamba-3 in production at scale.** No 7B+ Mamba-3 production model yet; Nemotron 3 successor models or a Granite 5 family are likely first adopters.
- **Hybrid Mamba-3 + Transformer.** No production model combines Mamba-3's complex state with softmax attention in the same forward; this is a natural extension of the Nemotron 3 / Granite 4 H lines.
- **DSA + CSA + HCA composition.** DeepSeek-V4 stops at CSA + HCA; combining with the V3.2 Lightning Indexer would yield a 5-tier cache (indexer + MLA + CSA + HCA + nearby-MLA). Reported but not yet shipped.
- **Per-Layer Embeddings in non-Gemma families.** The PLE pattern is novel enough that no other family ships it; likely adopters: edge-focused Liquid LFM2.5, Phi-5 (if it ever ships), Apple AFM successors.
- **K = V tying on non-global layers.** Gemma 4's K = V is only on global; the same trick on local layers is unexplored.
- **Block-bidirectional masks for non-OCR multimodal.** DeepSeek-OCR's Visual Causal Flow is the cleanest deployment; general VLMs could benefit, especially when video is fused (per-frame bidirectional, cross-frame causal).
- **iRoPE schedule learning.** Llama 4's NoPE/RoPE schedule is hand-designed; learning the schedule (per-layer `apply_rope` as a learnable parameter) is open research.
- **Apple AFM block split learning.** The 62.5/37.5 split is hand-designed; learning the split point per training run is plausible future work.
- **Cross-layer KV sharing × MoE.** Gemma 4 26B-A4B is MoE but does **not** ship `num_kv_shared_layers`; combining sparse experts with cross-layer KV sharing is an unexplored composition.
- **Mamba-3 + MTP.** MIMO decoding is already a multi-output-per-step scheme; combining MTP with Mamba-3 MIMO would let the model produce `k_mtp × n_mimo` candidate tokens per step.

---

## §11 References

(v2 references retained; v3 additions appended.)

**v2 references (verbatim):** Vaswani et al. 2017 (arxiv:1706.03762); Shazeer 2019 (arxiv:1911.02150); Su et al. 2021 (arxiv:2104.09864); Press et al. 2021 (arxiv:2108.12409); Haviv et al. 2022 (arxiv:2203.16634); Sun et al. 2022 (arxiv:2212.10554); Chi et al. 2022 (arxiv:2212.10356); Dao et al. 2022 (arxiv:2205.14135); Beltagy et al. 2020; Chen et al. 2023 (arxiv:2306.15595); Ainslie et al. 2023 (arxiv:2305.13245); Kazemnejad et al. 2023 (arxiv:2305.19466); Sun et al. 2023 (arxiv:2307.08621); Dao 2023 (arxiv:2307.08691); Peng, Quesnelle, Kingma 2023 (arxiv:2309.00071); Kwon et al. 2023 (SOSP 2023); Xiao et al. 2024 (arxiv:2309.17453); Liu, Zaharia, Abbeel 2023 (arxiv:2310.01889); Brandon et al. 2023 (arxiv:2311.09431); Hong et al. 2023 (arxiv:2311.01282); Gu & Dao 2023 (arxiv:2312.00752); Yang et al. 2023 (arxiv:2312.06635); Leviathan et al. 2023; Hooper et al. 2024 (arxiv:2401.18079); Liu, Yuan, Wang et al. 2024 (arxiv:2402.02750); An et al. 2024 (arxiv:2402.17463); Ding et al. 2024 (arxiv:2402.13753); Cai et al. 2024; Li et al. 2024; Spector & Re 2024 (arxiv:2402.13720); Botev et al. 2024 (arxiv:2402.19427); Lieber et al. 2024 (arxiv:2403.19887); Sun et al. 2024 (arxiv:2405.05254); Prabhu et al. 2024 (arxiv:2405.04437); Beck et al. 2024 (arxiv:2405.04517); Brandon et al. 2024 (arxiv:2405.12981); Liu, Hooper et al. 2024 (arxiv:2405.14366); Dao & Gu 2024 (arxiv:2405.21060); Ren et al. 2024 (arxiv:2406.07522); Zheng et al. 2024 (arxiv:2406.04692); Liu et al. 2024 (arxiv:2405.04434, DeepSeek-V2); Yang & Pan 2024 (arxiv:2406.06484); Shah, Bikshandi et al. 2024 (arxiv:2407.08608); Sun et al. 2024 (arxiv:2407.04620); Hu et al. 2024 (arxiv:2404.06395, MiniCPM-3); Riviere et al. 2024 (Gemma 2 tech report); Ye et al. 2024 (arxiv:2410.05258); Hu et al. 2024 (arxiv:2411.13676, Hymba); DeepSeek-V3 tech report 2024; Yuan et al. 2025 (arxiv:2502.11089, NSA); Gemma 3 tech report 2025; Mistral Small 3.1 release notes 2025; Bai et al. 2025 (Qwen2.5-VL MM-RoPE); Llama-4 release 2025; SmolLM3 release 2025; Qwen3-Next release 2025; GPT-OSS release 2025 (arxiv:2508.10925).

**v3 NEW references:**

- **MiniMax-Text-01 (Lightning 7:1):** arXiv 2501.08313 (Jan 2025).
- **MiniMax-M2:** arXiv 2605.26494 (early 2026, the reversion to full softmax).
- **DeepSeek-V3.2 / DSA:** arXiv 2512.02556 (Dec 2025); HF `deepseek-ai/DeepSeek-V3.2-Exp`.
- **DeepSeek-V4-Pro / V4-Flash:** Tech report PDF in `huggingface.co/deepseek-ai/DeepSeek-V4-Pro` (Apr 2026); `api-docs.deepseek.com/news/news260424`.
- **DeepSeek-OCR:** GitHub `deepseek-ai/DeepSeek-OCR` (Oct 2025); DeepSeek-OCR-2 (Jan 2026).
- **Apple AFM 2025:** arXiv 2507.13575 (Jul 2025).
- **Gemma 4:** blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ (2026-04-02); huggingface.co/blog/gemma4; HF model cards `google/gemma-4-{E2B, E4B, 12B, 26B-A4B, 31B}(-it)`; HF docs `model_doc/gemma4`; QAT blog (2026-06-05); upstream `transformers/models/gemma4/configuration_gemma4.py`.
- **Llama 4:** ai.meta.com/blog/llama-4-multimodal-intelligence (Apr 2025); `meta-llama/Llama-4-Scout-17B-16E` (gated).
- **Granite 4 H:** ibm.com Granite 4 launch (Oct 2025); `ibm-granite/granite-4.0-h-tiny`.
- **Qwen3-Next 80B-A3B:** alibabacloud.com/blog/qwen3-next (Aug 2025); HF `Qwen/Qwen3-Next-80B-A3B-Instruct`.
- **Mamba-3:** arXiv 2603.15569 (ICLR 2026); openreview.net/forum?id=HwCvaJOiCj.
- **NVIDIA Nemotron 3 Super / Nano Omni / Ultra:** developer.nvidia.com/blog/introducing-nemotron-3-super (Mar 2026); blogs.nvidia.com Nemotron 3 Nano Omni (Apr 2026); marktechpost.com Nemotron 3 Ultra (Jun 2026).
- **MiniMax-M3.0 (MSA):** marktechpost.com MiniMax-M3.0 release (Jun 2026).
- **GLM-5.1:** huggingface.co/zai-org/GLM-5.1 (Apr 2026; DSA derivative).
- **Tencent Hy3 / Hunyuan 3 Preview:** huggingface.co/tencent/Hy3-preview (Apr 2026).
- **Mistral Small 4 (119B/15B-A) and Medium 3.5:** mistral.ai/news (Mar / Apr 2026).
- **Qwen3-VL:** github.com/QwenLM/Qwen3-VL (Apr 2026).
- **Qwen2.5-Omni (TMRoPE):** github.com/QwenLM/Qwen2.5-Omni (Mar 2025).
- **Pixtral 12B:** arXiv 2410.07073 (2D-RoPE for image patches; Sep 2024).
- **Janus-Pro 1B/7B:** arXiv 2501.17811 (Jan 2025; decoupled SigLIP-L for understanding vs generation).
- **Apple AFM tech report:** arXiv 2507.13575.

**HF / runtime source references (v3 additions):**

- `transformers/src/transformers/models/gemma4/configuration_gemma4.py` — `Gemma4TextConfig` with `partial_rotary_factor`, `attention_k_eq_v`, `num_kv_shared_layers`, `global_head_dim`, `use_double_wide_mlp`, `final_logit_softcapping`.
- `transformers/src/transformers/models/gemma4/modeling_gemma4.py` — `Gemma4DecoderLayer` with the dual-norm sandwich preserved.
- `huggingface.co/google/gemma-4-E2B/raw/main/config.json` — verified 2026-06-06.
- `huggingface.co/google/gemma-4-31B/raw/main/config.json` — verified 2026-06-06.
- `state-spaces/mamba` GitHub — Mamba-3 reference implementation; complex-state emulated via real-pairing.
- vLLM Triton DSA backend (2026-Q1 experimental).
- FlashInfer VisualCausalMask primitive (v0.3.0+).

**Verified HF configs (as of 2026-06-06):**

- Gemma 4 E2B `config.json`: `num_kv_shared_layers = 20`, `attention_k_eq_v = false`, `sliding_window = 512`, `num_attention_heads = 8`, `num_key_value_heads = 1`, `head_dim = 256`, `global_head_dim = 512`, `num_hidden_layers = 35`, `use_double_wide_mlp = true`, `final_logit_softcapping = 30.0`, `attn_logit_softcapping` absent. RoPE per-layer-type: sliding θ=10K rope_type=default, full θ=1M rope_type=proportional partial_rotary_factor=0.25. Layer pattern: 4 : 1 sliding/global (every 5th layer is global; layers 5, 10, 15, 20, 25, 30, 35 = full_attention).
- Gemma 4 31B `config.json`: `num_kv_shared_layers = 0`, `attention_k_eq_v = true`, `sliding_window = 1024`, `num_attention_heads = 32`, `num_key_value_heads = 16`, `head_dim = 256`, `global_head_dim = 512`, `num_hidden_layers = 60`, `use_double_wide_mlp = false`, `final_logit_softcapping = 30.0`. Layer pattern: 5 : 1, twelve repeats, full at every 6th layer.

---

## Appendix A — One-Line Model Summaries (v3 — 2026-H1 additions)

(v2 summaries retained; v3 adds:)

- **Gemma 4 E2B**: 35 layers (4:1 sliding/global, last global), MQA sliding (H=8 G=1 d_h=256) + MQA global (G=1 with global_head_dim=512), p-RoPE 0.25 on global, per-layer θ 10K/1M, `num_kv_shared_layers = 20`, `attention_k_eq_v = false`, PLE with d_ple=256, GeGLU, final softcap 30, dual-norm sandwich.
- **Gemma 4 E4B**: 42 layers (5:1), wider H, same axes as E2B; `attention_k_eq_v` flagged `[VERIFY]`.
- **Gemma 4 12B Unified**: 48 layers (5:1), encoder-free multimodal, `attention_k_eq_v = true`, GQA (H=16 G=8), 256k context.
- **Gemma 4 26B-A4B**: MoE 128 routed + 1 shared experts top-k=8, 30 layers, K=V global, 256k ctx.
- **Gemma 4 31B**: 60 layers (5:1), GQA H=32 G=16 d_h=256, K=V global, no cross-layer KV share, no PLE, 256k ctx.
- **DeepSeek-V3.2 / V3.2-Exp**: MLA + Lightning Indexer DSA, indexer separate cache, full MLA cache + indexer cache.
- **DeepSeek-V4-Pro / V4-Flash**: MLA + CSA + HCA quad-tier cache; 27% FLOPs and 10% KV vs V3.2 at 1M ctx.
- **Llama 4 Scout (17B-A) / Maverick**: iRoPE with per-layer NoPE alternation + attention-temperature tuning; native multimodal early fusion.
- **Qwen3-Next 80B-A3B**: Gated DeltaNet 3:1 hybrid (75% linear, 25% softmax-gated); GDN state + softmax KV + MTP head.
- **MiniMax-Text-01**: Lightning Attention 7:1 hybrid (87.5% linear).
- **MiniMax-M2**: full softmax (reverted from Lightning).
- **MiniMax-M3.0**: MSA — MiniMax Sparse Attention.
- **Granite 4 H-Tiny (7B-A1B) / H-Micro (3B hybrid)**: 9:1 Mamba-2 : attention with sinks on attention layers; rope_theta=10M.
- **Mamba-3 reference (180M / 700M / 1.5B)**: complex state + MIMO decoding.
- **NVIDIA Nemotron 3 Super (120B/12B-A) / Nemotron 3 Ultra (550B/55B-A)**: Hybrid Mamba-2 + Transformer MoE, FP8/NVFP4 native.
- **Apple AFM 3.18B (on-device)**: 2-block cross-block KV sharing (62.5/37.5); 2-bit QAT; closed weights.
- **DeepSeek-OCR (3B-MoE-A570M)**: Visual Causal Flow / block-bidirectional mask on vision tokens; causal on text.
- **GLM-5.1 (744B/40B-A)**: DSA-derivative attention.
- **Tencent Hy3-Preview (295B/21B-A + 3.8B MTP)**: MoE + MTP head.

---

## Appendix B — Code-Pointer Cheatsheet (v3 additions)

(v2 cheatsheet retained; v3 adds:)

- `transformers/src/transformers/models/gemma4/configuration_gemma4.py` — `Gemma4TextConfig` defaults including `num_kv_shared_layers`, `attention_k_eq_v`, `partial_rotary_factor`, `global_head_dim`, `use_double_wide_mlp`, `final_logit_softcapping`.
- `transformers/src/transformers/models/gemma4/modeling_gemma4.py` — `Gemma4DecoderLayer` four-norm sandwich preserved; `Gemma4Attention` with per-layer-type RoPE; PLE injection at `Gemma4Embedder`.
- DeepSeek-V3.2 / V4 reference implementations: `huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/modeling_deepseek_v32.py` (DSA); `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/modeling_deepseek_v4.py` (CSA + HCA).
- Mamba-3 reference: `github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py` (complex-state via real-pairing).
- Llama 4 modelling: `transformers/src/transformers/models/llama4/modeling_llama4.py` — per-layer `apply_rope` boolean + per-layer `attn_temp`.
- vLLM DSA backend (experimental, 2026-Q1): `vllm/attention/backends/dsa/triton_dsa.py`.
- FlashInfer VisualCausalMask: `flashinfer/include/flashinfer/attention/visual_mask.cuh` (v0.3.0+).
- NVIDIA NVFP4 KV path: `tensorrt_llm/runtime/nvfp4_kv_cache.cpp`.

---

## Appendix C — Open Verifications (v3 additions)

(v2 items 1-5 retained.)

**v3 NEW:**

6. **Gemma 4 E4B `attention_k_eq_v` value.** The Gemma 4 investigation note `10-gemma4-investigation.md` infers from secondary sources that K=V is family-wide on global layers, but the raw config fetch confirms E2B is `false` while 31B is `true`. The E4B config was not directly fetched in this v3 pass; flag `[VERIFY E4B raw config]`.
7. **DSA Lightning Indexer dimensionality.** The V3.2 tech report abstract does not specify `d_indexer`, `top_k`, or the warm-window size. Community write-ups suggest `d_indexer ≈ 32-64` and `top_k ≈ 256-1024`. `[VERIFY against full V3.2 PDF or HF code]`.
8. **DeepSeek-V4 CSA / HCA exact compression ratios.** The 27% FLOPs and 10% KV claims are from the V4 tech report; the precise `l_csa`, `l_hca`, `d_csa`, `d_hca`, `top_k` values are not in the public abstract. `[VERIFY against the V4 tech report PDF]`.
9. **Mamba-3 `d_state` defaults and MIMO `n_outputs`.** The ICLR 2026 paper abstract does not give them. `[VERIFY against arxiv:2603.15569 full PDF]`.
10. **Llama 4 NoPE layer schedule.** Per-layer `apply_rope: bool` schedule is referenced but the exact ratio (1:4, 1:3, etc.) for Scout vs Maverick is not in the public model cards as of 2026-06-06. `[VERIFY against `meta-llama/Llama-4-Scout-17B-16E` raw config when ungated]`.
11. **Gemma 4 fixed-scale QK-norm values (local 0.9916, global 1.0228).** Single secondary source (Maarten Grootendorst's writeup). Not present in the raw `config.json`. `[VERIFY against `modeling_gemma4.py`]`.
12. **`rope_type = "proportional"` formula for Gemma 4 p-RoPE.** New rope_type value. The exact formula (how the partial-rotary fraction is scaled) is not yet documented in HF `modeling_rope_utils.py`. `[VERIFY against upstream `transformers` when proportional-rope code lands]`.
13. **MiniMax-M3.0 MSA details.** Announced 2026-06-01; weights staged ~mid-June. Architecture paper / config not yet public as of 2026-06-06. `[VERIFY when weights land on HF]`.
14. **Gemma 4 `use_double_wide_mlp` semantics.** Flag present in E2B and E4B configs (set to `true`) but absent in 12B/26B/31B (set to `false`). Whether `double_wide` widens `gate_proj`, `up_proj`, both, or `down_proj` is not documented in HF docs as of 2026-06-06. `[VERIFY against `modeling_gemma4.py` MLP class]`.
15. **Gemma 4 PLE precise table shape and projection.** The `10-gemma4-investigation.md` reports `d_ple = 256`, projection multiplier `1/sqrt(2)`, and the `(token_identity + context_aware_projection)` decomposition based on HF docs. Implementation details (whether `context_aware_projection` is per-layer or shared) are `[VERIFY against the PLE source in `modeling_gemma4.py`]`.
16. **CSA + HCA × MTP composition.** No mainstream model combines them. Open research direction.

— end of document —
