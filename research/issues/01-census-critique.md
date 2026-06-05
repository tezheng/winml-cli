# 01 — Critique of Model Census

*Reviewer: skeptical technical reviewer. Target: `research/01-model-census.md`. Date: 2026-06-04.*

The census is competent as a present-day snapshot. As a **survey of three years of SLM layer evolution**, however, it has structural defects: the historical axis is collapsed to a single column ("Released"), several first-tier mainstream models are absent, at least one architectural axis worth a dedicated section (per-layer width/depth scaling) is buried in a footnote, and a small number of header values do not match the cited `config.json`. The document is also strangely silent on three of the most-cited 2023 backbones (Llama 2 7B, Mistral 7B v0.1, MPT 7B) — the very models a "past 3 years" narrative is supposed to start from.

What follows is a section-by-section critique with evidence.

---

## Section 1. Missing Models

The census claims to cover "~45 mainstream small language models … shipped between 2023 and 2026." The set is heavily weighted to 2024–2025 dense decoders and underweights (a) the 2023 ancestors needed for an evolution story, (b) architecturally distinctive niche models, and (c) several 2025 releases that fit the "<8B mainstream" bar squarely.

| Model | Released | Why mainstream / distinct | Architectural distinction | Severity |
|---|---|---|---|---|
| **Llama 2 7B** | 2023-07 | The reference point for every "post-Llama consensus" claim in the executive summary. >70k downloads/week still. | MHA (no GQA), SiLU-SwiGLU, RoPE θ=10000, no SWA, vocab 32000 | **MUST-FIX**: cannot tell an evolution story without the baseline. |
| **Mistral 7B v0.1 / v0.2** | 2023-09 / 2024-03 | First widely-adopted GQA + SWA in a 7B; verified config has `sliding_window=4096`. Census only has v0.3 (SWA dropped). | GQA 32/8, SWA=4096 (v0.1), θ=10000; v0.2 removed SWA, kept GQA, raised θ to 1e6 | **MUST-FIX**: the SWA-introduction and SWA-abandonment story is *the* canonical example of a feature being adopted and dropped. |
| **MPT 7B** | 2023-05 | Mosaic's flagship 2023 7B; introduced ALiBi as a position scheme that the industry later abandoned in favor of RoPE-scaling. | ALiBi (not RoPE!), MHA, FlashAttention v1 baseline | **MUST-FIX**: the only ALiBi-class model the user would reach for; required to contrast against the RoPE-dominant present. |
| **Falcon 7B / Falcon Mamba 7B / Falcon 3 (1B/3B/7B/10B) / Falcon-H1** | 2023-05, 2024-08, 2024-12, 2025 | Falcon-Mamba was the first **pure-SSM** model trained competitive with Llama at 7B; Falcon-H1 is hybrid. Verified Falcon3-7B is Llama-arch with `head_dim=256`, `rope_theta=1000042`, `vocab=131072`, GQA 12/4. | Falcon3 has unusually large `head_dim=256` at 7B (similar to Gemma); Falcon Mamba is pure Mamba-1; Falcon-H1 is Mamba-2/attention hybrid. | **MUST-FIX**: omitting the Falcon family removes the only TII/MENA contribution and a unique head_dim=256 dense decoder. |
| **OpenELM (270M / 450M / 1.1B / 3B)** | 2024-04 | Apple's on-device family. Verified config: `num_query_heads` and `num_kv_heads` are **per-layer lists** (e.g. `[12,12,12,12,16,16,...,24,24]` for 3B), and `ffn_multipliers` is a **per-layer list** (`[0.5, 0.6, …, 4.0]`). This is the only published model that varies attention width and FFN width per layer. | "Decoupled per-layer scaling" — totally unique axis | **MUST-FIX**: without OpenELM, axis 4.10 ("per-layer heterogeneity") is missing its most extreme exemplar. |
| **RecurrentGemma 2B / 9B** | 2024-04 | Google's Griffin/Hawk backbone; mixes linear recurrence (LRU) with local attention. Verified parameters present (`attention_window_size`, `block_types`, `lru_width`). | Griffin/Hawk gated linear recurrence — a third hybrid family beyond Mamba and RWKV | **SHOULD-FIX**: census names only Mamba/Mamba2/RWKV/Zamba2/Jamba; Griffin/Hawk is omitted entirely. |
| **BitNet b1.58 2B-4T** | 2024-11 (paper) / 2025 (HF) | First production 1-bit-weight LM (`weight_bits=1`, ternary {-1,0,1}). Confirmed `BitNetForCausalLM`, `relu2` activation, `tie_word_embeddings=true`, 30 layers, 2560 hidden, 20/5 GQA. | 1.58-bit weights, ReLU² activation (not SwiGLU/GeGLU!) — orthogonal to all other axes | **MUST-FIX**: quantization-native architecture; the activation choice alone (ReLU²) is unique among modern decoders and breaks axis 4.4. |
| **Hymba 1.5B** | 2024-11 | NVIDIA's parallel Mamba+attention. Verified: `model_type=hymba`, 32 layers, 25/5 GQA, mamba_d_state=16, `attn_implementation=flex`. | **Parallel** (not sequential) Mamba+attention — every layer runs both branches and sums | **SHOULD-FIX**: distinct from Zamba2 (sequential pattern) and Jamba (periodic substitution). |
| **Phi-4-mini-flash / Phi-4-flash-reasoning** | 2025-07 | Microsoft's hybrid Mamba+attention 3.8B; confirmed `model_type=phi4flash`, `Phi4FlashForCausalLM`, hidden=2560, 32 layers, GQA 40/20, o200k vocab. | Hybrid Mamba+attention from Microsoft (SAMBA-derived) | **MUST-FIX**: a 2025 SLM <8B from a major vendor — directly in scope. |
| **Phi-3.5-MoE** | 2024-08 | 16-expert top-2 MoE at 3.8B active (6.6B total). Confirmed `PhiMoEForCausalLM`, 4096 hidden, 32 attn / 8 kv, 32 layers, 16 experts top-2. | Phi-family MoE — orthogonal to Qwen3-MoE / OLMoE / DeepSeek-V2-Lite-MoE | **SHOULD-FIX**: third MoE pattern (16 experts top-2, no shared) for triangulation. |
| **Phi-2 (2.7B)** | 2023-12 | Microsoft's transitional Phi; introduced "data-quality > scale" thesis that defines the whole Phi series. | MHA, partial rotary 32, GELU (no gate), parallel attention+FFN (GPT-J-style) | **MUST-FIX**: the only mainstream parallel-attention model in production; required for axis-completeness on "parallel vs serial sublayer". |
| **Phi-1 / Phi-1.5** | 2023-06 / 2023-09 | The progenitor; 1.3B. | MHA, classic LayerNorm, learned absolute pos? RoPE? worth tracing | SHOULD-FIX |
| **Gemma 1 (2B / 7B), Gemma 1.1** | 2024-02 / 2024-04 | The progenitor of the Gemma family. Census jumps straight to Gemma 2. | MQA in 2B, MHA in 7B, GeGLU, dual norm, **`head_dim=256`** (the Gemma signature); RoPE θ=10000 | **MUST-FIX**: cannot show Gemma 1→2→3 evolution without Gemma 1. |
| **Gemma 3n (E2B / E4B)** | 2025-05 | "Matformer" elastic-width Gemma — runtime per-layer width subsetting. | Matformer / per-layer activation slicing | **SHOULD-FIX**: a novel axis (elastic width) that no other model exposes. |
| **PaliGemma 1 / 2 (3B / 10B / 28B)** | 2024-05 / 2024-12 | Census mentions PaliGemma 3B but assumes Gemma 1 backbone; PaliGemma 2 is **Gemma 2** backbone. Should distinguish. | "Prefix-LM" mode (bidirectional attention over image+prefix tokens) — unique attention mask | **SHOULD-FIX**: the only PrefixLM mask in the census. |
| **CodeGemma 7B** | 2024-04 | Census has 7B row but cites no link, only "(gemma-1 family card)". | GQA (KV=16) on Gemma-1 backbone | SHOULD-FIX: dangling citation. |
| **Mistral 7B v0.1, v0.2** | 2023-09 / 2024-03 | See above — v0.1 has SWA=4096 (verified). Census omits both. | See above | MUST-FIX. |
| **Codestral Mamba 7B / Mathstral 7B** | 2024-07 | Mistral's Mamba-2 derived coder; only pure-Mamba 7B from Mistral. | Mamba-2 (not Mamba-1!) at 7B scale | SHOULD-FIX. |
| **Mistral Nemo 12B** | 2024-07 | Out of scope (>8B), but the **tekken tokenizer** (vocab=131072) it introduced trickled down to Mistral Small (24B) and arguably influenced the Tiktoken-style adoption in 7B-class models. | Tekken (Tiktoken-derived) 131k SP | Note in evolution section; can skip the row. |
| **Qwen 1 (7B), Qwen 1.5 (0.5B–7B), Qwen 2 (0.5B–7B)** | 2023-09 / 2024-02 / 2024-06 | Census starts at Qwen 2.5. Verified Qwen2-7B: GQA 28/4, head_dim=128 (from 3584/28), vocab 152064, θ=1e6. Qwen 2 was the first generation to fully unify on GQA + 1e6 theta. Qwen 1 had **QKV bias** (legacy from pre-GQA era) — the census notes this in axis 4.8 but never includes a Qwen 1 row to demonstrate. | Qwen 1 has QKV bias; Qwen 2 drops it; Qwen 3 adds QK-norm. Three-generation arc. | **MUST-FIX**: dropping the first two generations destroys the QKV-bias-removal narrative. |
| **Qwen2.5-Math-7B / Qwen2.5-Math-1.5B** | 2024-09 | Backbones of the R1-Distill-Qwen 1.5B/7B rows the census *does* include — but the backbones themselves are not listed. | Identical to Qwen2.5 base + math RLHF; relevant only for backbone provenance | SHOULD-FIX (citation-completeness). |
| **DeepSeek LLM 7B, DeepSeek-Math 7B, DeepSeek-Coder 6.7B base** | 2023-11 / 2024-02 / 2023-11 | DeepSeek's pre-MLA backbones. The MLA introduction at DeepSeek-V2 is meaningless to a reader who has never seen a pre-MLA DeepSeek config (vanilla GQA with their own SP tokenizer). | MHA → GQA → MLA progression | **MUST-FIX**: the MLA-introduction story (Section 5.4) has no contrast point. |
| **DeepSeek-V2-Lite-Chat / DeepSeek-V3.1** | 2024-05 / 2025 | V3.1 small variants if any; the census stops at V2-Lite. | DeepSeekMoE v2 vs v3 routing (aux-loss-free) | SHOULD-FIX. |
| **DeepSeek-V3 / R1 (full)** | 2024-12 / 2025-01 | Out of scope at full size (671B), but R1's aux-loss-free routing and 256-expert top-8 design is the architectural milestone of 2024–2025 MoE. Census mentions DeepSeek-V3 only in a footnote of 4.10. | Aux-loss-free routing, fp8 training | Mention in evolution. |
| **ChatGLM 6B / ChatGLM2 6B / ChatGLM3 6B / GLM-4-9B** | 2023-03 / 2023-06 / 2023-10 / 2024-06 | Tsinghua's GLM series — the first mainstream Chinese-language LM. ChatGLM2 introduced **multi-query attention** before Llama 3 did. | Prefix LM training; MQA in ChatGLM2; **`tie_word_embeddings` patterns differ** | **MUST-FIX**: omitting GLM removes the entire Chinese ecosystem's pre-Qwen-2.5 history. |
| **Baichuan 1 / 2 (7B)** | 2023-06 / 2023-09 | First Chinese decoder-only at 7B with ALiBi (Baichuan 1) → RoPE (Baichuan 2) transition. | ALiBi → RoPE transition within one vendor | SHOULD-FIX (a second ALiBi-removal data point alongside MPT). |
| **Yi-6B, Yi-9B** | 2023-11 / 2024-03 | 01-AI's predecessor to Yi-1.5. Census has Yi-1.5-6B only. | Vanilla Llama-arch, vocab=64000 | SHOULD-FIX. |
| **Yi-Coder 1.5B / 9B** | 2024-09 | 01-AI's coder. | Same arch as Yi-1.5 with code fine-tune | LOW. |
| **InternLM 1, InternLM 2 (1.8B / 7B / 20B)** | 2023-06 / 2024-01 | Predecessors to InternLM 2.5/3. Census skips them. | InternLM 1 used MHA + ALiBi (early); InternLM 2 moved to RoPE + GQA | SHOULD-FIX (another ALiBi-evolution datapoint). |
| **InternLM-XComposer, InternLM-Math 7B** | 2024 | Family multimodal/math variants. | LOW. |
| **BlueLM (Vivo), Hunyuan-Lite, SkyworkMath** | 2023–2024 | Chinese OEM models. | LOW. Census could add 1–2 to round out Chinese ecosystem coverage. |
| **MiniCPM 1 / 2 / 2.4B** | 2024-02 / 2024-04 | Predecessors to MiniCPM 3 (which the census has). | Without MiniCPM 1/2, μP-scalar narrative is undercut. | **SHOULD-FIX**. |
| **MiniCPM-o, MiniCPM-V** | 2024-12 | Multimodal MiniCPM. | LOW. |
| **StableLM 3B-4E1T, StableLM Zephyr 3B** | 2023-09 / 2023-10 | Stability AI's earlier StableLM. Census has StableLM 2 only. | StableLM 3B had partial rotary (25%) **before** StableLM 2; first to ship that idea at scale. | **SHOULD-FIX**: partial-RoPE narrative needs StableLM 3B as origin point. |
| **StableLM Zephyr 3B / StableCode 3B** | 2023-10 / 2024-02 | Coding variants. | LOW. |
| **OLMo 1B (2024-04) / OLMo 7B (2024-02)** | 2024-02 / 2024-04 | OLMo 1 (pre-norm RMS + SwiGLU, no QK-norm) is the contrast point for OLMo 2 (post-norm + full-channel QK-norm). | **OLMo 1 → OLMo 2 is the clearest pre→post norm flip in the corpus.** Census has OLMo 2 only. | **MUST-FIX**: cannot motivate "OLMo 2 is post-norm" without OLMo 1's pre-norm. |
| **OLMo 2 13B** | 2024-11 | Census has it (good). | — | — |
| **OLMo 2 32B** | 2025-03 | >8B, exclude. | — | — |
| **Tulu 3 8B** | 2024-11 | AI2's instruction-tuned Llama 3.1; arguably just a post-training. | LOW. |
| **SmolLM 135M / 360M / 1.7B v1** | 2024-07 | Census has v2 only; v1 had distinct vocab/tokenizer (49152 carried over but per-layer scaling differed). | LOW-MED: SmolLM v1 → v2 contrast is small. |
| **MobileLLM 125M / 350M / 600M / 1B / 1.5B** (Meta) | 2024-10 | Meta's deep-narrow on-device family; **shared embeddings + deep-and-thin geometry** ("deep>wide" hypothesis). | Per-architecture choice: deep+narrow, embedding tying, **block-wise weight sharing** (1B/1.5B). | **SHOULD-FIX**: weight-sharing is its own axis not covered. |
| **Llama-3 MobileLLM** (Meta) | 2024-12 | Follow-up. | LOW. |
| **Apple Foundation Model (~3B)** | 2024-09 | Apple's on-device model. Architecture only partially public, but published WWDC details indicate per-layer rank-1 LoRA adapters baked into the weights. | "Per-layer adapter" axis if architecturally distinct. | LOW (limited public info). |
| **Granite 3.0** | 2024-10 | Census has 3.1 and 3.3 — skips the first release. Granite 3.0 had the same μP scalars but smaller context (4096). | LOW-MED: would round out Granite generational story. |
| **Granite Code 3B / 8B / 20B (legacy 3.0)** | 2024-05 | IBM's coding variant on Granite. | LOW. |
| **Granite 4.0 (MoE)** | 2025 | If shipped. | LOW (recency unclear). |
| **StarCoder 1 (1B / 3B / 7B / 15B)** | 2023-05 | BigCode's predecessor to StarCoder 2; **the** 2023 coding baseline. **Multi-Query Attention** + LayerNorm + learned absolute pos! | MHA → MQA → GQA arc within BigCode | **MUST-FIX**: the MQA-at-scale exemplar of 2023. |
| **StarChat / OctoCoder / WizardCoder** | 2023 | Fine-tuned variants. | LOW. |
| **Cohere Aya 8B / Aya Expanse 8B** | 2024 | Cohere's multilingual model — based on Command R but at the <8B edge. | "Cohere arch" — uses **no QKV bias, mostly Llama-shaped** but with **`logit_scale=0.0625`** and tied embeddings even at 8B. | **SHOULD-FIX**: another logit-scale model alongside Granite/Gemma. |
| **Mosaic MPT 7B / DBRX (too big)** | 2023-05 | See above — MPT 7B critical. | — | — |
| **xLSTM 7B** | 2024-10 / 2025 | NXAI's extended LSTM; replaces all attention with sLSTM/mLSTM blocks. | New SSM-family member. | SHOULD-FIX: would complete the linear-recurrence taxonomy alongside Mamba/RWKV/Griffin. |
| **Samba (Microsoft)** | 2024-06 | Direct ancestor of Phi-4-mini-flash; hybrid Mamba+SWA. | First (Mamba-1 + SWA) hybrid from MSR. | LOW-MED (academic, but architecturally upstream of Phi-4-flash). |
| **RWKV-7 (Goose)** | 2025-01 | Census mentions in axis 4.1 but no model row. | New RWKV with delta-rule update. | SHOULD-FIX: census mentions axis but doesn't enumerate a v7 row. |
| **Zamba2 7B** | 2024-12 | Census has 2.7B only. Verified the 7B exists. | Same hybrid topology, scaled up. | LOW. |
| **MiniCPM-V variants, Idefics3, SmolVLM-Instruct** | 2024-08–2024-12 | SmolVLM uses Idefics3 (Llama-3.2-1B text tower, 32/32 MHA). | LOW (text tower is just SmolLM2-1.7B / Llama3.2-1B variants). |
| **Open-source Qwen2.5-VL text tower (3B / 7B)** | 2024-09 | Vision-language. | LOW. |
| **MiniMax-Text-01 (small variants)** | 2025-01 | Linear attention with mixed softmax at scale. | LOW (no <8B variant shipped publicly). |

### Severity tally (Section 1)
- **MUST-FIX** missing: **14** distinct models (Llama 2 7B, Mistral 7B v0.1/v0.2, MPT 7B, Falcon family, OpenELM, BitNet, Phi-4-mini-flash, Phi-2, Gemma 1, Qwen 1/1.5/2, DeepSeek LLM/Math/Coder-V1, ChatGLM/GLM-4, OLMo 1, StarCoder 1).
- **SHOULD-FIX** missing: ~15.
- **LOW** missing: ~15.

For an "evolution paper", the must-fix set means at minimum the **2023 baseline year is missing entirely** — there is not a single 2023 model in the census except Mamba 2.8B (Dec 2023) and TinyLlama (which the row mis-dates to 2024-01 but the v1.0 release was actually 2023-12-30; checkpoints were public from Q3 2023).

---

## Section 2. Evolution Narrative Gaps

The report's structure is taxonomic, not historical. The word "evolution" appears zero times in the body; "generation" is used loosely. A survey paper of layer evolution must contrast successive generations within each lineage. The following generation-to-generation transitions are **not narrated** despite being foundational to the "past 3 years" framing:

1. **Llama 1 (Feb 2023, MHA, RoPE θ=10000, vocab 32000 SP) → Llama 2 (July 2023, GQA at 70B only / MHA at 7B) → Llama 3 (April 2024, GQA at all sizes, vocab 128k Tiktoken, θ=500000) → Llama 3.1/3.2/3.3 (RoPE freq-band scaling, context 128k).**
   The census has only Llama 3.x. The single most cited family in the survey has *one* generation represented. The transitions worth narrating: (a) MHA → GQA at 7B happened at Llama 3, not Llama 2; (b) Tokenizer flip from SP to Tiktoken at Llama 3; (c) RoPE θ raised from 10000 → 500000 between Llama 2 and Llama 3; (d) Llama 3.1 introduced the four-parameter `rope_scaling` block which is now ubiquitous.

2. **Mistral 7B v0.1 (SWA=4096, GQA, θ=10000) → v0.2 (SWA dropped, θ=10000) → v0.3 (SWA dropped, θ=1e6, vocab 32768).**
   The census notes "SWA dropped in v0.2+" in a single parenthetical. The *why* (FlashAttention-2 made global attention cheap; SWA broke long-range retrieval in tests) is never said. The deprecation of SWA at 7B is then *reversed* by Gemma 2/3, which the census also doesn't tie back to Mistral's experience.

3. **Qwen 1 (Sept 2023, QKV bias, GQA, θ=10000) → Qwen 1.5 (Feb 2024) → Qwen 2 (June 2024, drops QKV bias, GQA, θ=1e6, vocab 151936) → Qwen 2.5 (Sept 2024) → Qwen 3 (April 2025, adds QK-RMSNorm per head_dim) → Qwen 3 MoE.**
   Five generations, the census shows two and a half. Critical transitions: removal of QKV bias (Qwen 1→2), introduction of QK-norm with head-dim shape (Qwen 2.5→3), MoE adoption (Qwen 3-Dense → Qwen3-MoE).

4. **Phi-1 (Jun 2023, code-only 1.3B) → Phi-1.5 (Sep 2023) → Phi-2 (Dec 2023, 2.7B, parallel attn/FFN) → Phi-3-mini-4k (Apr 2024) → Phi-3-mini-128k (Apr 2024, LongRoPE) → Phi-3-small (Apr 2024, gegelu+blocksparse+μP) → Phi-3.5-mini (Aug 2024) → Phi-3.5-MoE (Aug 2024) → Phi-4 (Dec 2024, 14B, out of scope) → Phi-4-mini (Feb 2025) → Phi-4-multimodal (Feb 2025) → Phi-4-mini-flash (Jul 2025, hybrid Mamba+attention).**
   The Phi series is the *single most architecturally heterogeneous* family in the corpus. The census narrates 5 of these 12 stops. Most jarring: Phi-2's parallel attention+FFN block (à la GPT-J) is not mentioned anywhere; Phi-3-mini's 4k vs 128k difference (vanilla RoPE vs LongRoPE) is glossed; Phi-3.5-MoE is missing; Phi-4-mini-flash (the only hybrid Mamba+attention from Microsoft) is missing.

5. **Gemma 1 (Feb 2024, 2B MQA + 7B MHA, GeGLU, dual norm, head_dim=256, θ=10000) → Gemma 2 (Jul 2024, GQA at all sizes, SWA every other layer, attn softcapping, final-logit softcapping) → Gemma 3 (Mar 2025, 5:1 SWA:full alternation, dual RoPE, drops softcapping, multimodal) → Gemma 3n (May 2025, matformer).**
   Census has Gemma 2 and 3. Gemma 1 (the inventor of the `head_dim=256` choice and GeGLU adoption!) and Gemma 3n (the only "elastic-width" model) are absent.

6. **DeepSeek LLM 7B (Nov 2023, vanilla Llama-arch, vocab 102400) → DeepSeek-Coder 6.7B (Nov 2023) → DeepSeek-Math 7B → DeepSeek-V2 / V2-Lite (May–Jun 2024, **MLA introduction**) → DeepSeek-V2.5 → DeepSeek-V3 (Dec 2024, aux-loss-free routing, fp8, 256 experts top-8) → R1 (Jan 2025) → R1-Distill (Jan 2025) → V3.1.**
   The MLA introduction needs a "before" (DeepSeek LLM 7B = MHA Llama-clone with their own vocab) and "after" (V2-Lite). Census shows only the after.

7. **OLMo 1 (Feb 2024, pre-norm RMS, SwiGLU, no QK-norm, MHA) → OLMo 2 (Nov 2024, post-norm RMS, SwiGLU, **full-channel QK-RMSNorm**, MHA).**
   This is the single clearest pre-norm→post-norm flip in the corpus. The census has OLMo 2; without OLMo 1 the flip cannot be seen.

8. **StableLM 3B-4E1T (Sep 2023, partial-RoPE 25%, MHA + QKV bias) → StableLM 2 1.6B (Jan 2024, partial-RoPE 25%, MHA + QKV bias) → StableLM 2 12B (Apr 2024, partial-RoPE 25%, GQA + QK-LayerNorm).**
   Partial-RoPE 0.25 is *the* StableLM signature; tracing it from StableLM 3B (2023) to StableLM 2 12B (2024) is missing.

9. **MoE evolution: Mixtral 8x7B (Dec 2023, 8 experts top-2, no shared) → DeepSeek-V2 (256 routed + 2 shared, top-6) → Qwen3-MoE (128 routed + 0 shared, top-8) → DeepSeek-V3 (256 + 1 shared, aux-loss-free).**
   The census has DeepSeek-V2-Lite, OLMoE, Qwen3-MoE. It's missing Mixtral (the 2023 baseline that everyone copied) and the V2→V3 transition. Mixtral 8x7B at 13B active is out of scope, but the "Mixtral cleaved active params open from total params" milestone is the start of the whole subfield.

10. **What was tried and abandoned.** A survey paper should report negative results. The census never says:
    - SWA was tried (Mistral v0.1), abandoned (v0.2/v0.3), re-adopted (Gemma 2/3, Phi-3-mini).
    - ALiBi was tried (MPT, Baichuan 1) and abandoned in favor of RoPE-scaling everywhere.
    - MQA was tried (Falcon 7B, ChatGLM2, Gemma 1 2B, CodeGemma 2B, PaliGemma) and largely replaced by GQA — but survives at <2B for on-device.
    - QKV biases (Qwen 1, StableLM) were largely dropped — except StarCoder 2 kept them.
    - Parallel attention+FFN (Phi-2, GPT-J inheritance) was abandoned at Phi-3.
    - μP was experimentally adopted at Phi-3-small and didn't propagate; rebranded as "multipliers" by Granite/MiniCPM.

These are the stories a survey paper exists to tell. They are absent.

### Severity (Section 2)
- **MUST-FIX**: 6 generation-arcs un-narrated (Llama, Mistral, Qwen, Phi, Gemma, OLMo).
- **SHOULD-FIX**: 4 (DeepSeek, StableLM, MoE arc, abandoned-feature retrospective).

---

## Section 3. Fact-Check Sample

Six rows verified against the cited `config.json` (one extra over the required five, for MLA confirmation). Discrepancies in **bold**.

### 3.1 Qwen3-1.7B (census section 2.2)

Census claim: `Attn = GQA + QK-norm`, `H/KV/hd = 16/8/128`, vocab `151936`, RoPE `vanilla (θ=1e6)`, ctx `40960`, tied yes.

Verified from <https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json>:
- `num_attention_heads = 16` ✓
- `num_key_value_heads = 8` ✓
- `head_dim = 128` ✓
- `hidden_size = 2048` (not in census row)
- `vocab_size = 151936` ✓
- `rope_theta = 1000000` ✓
- `intermediate_size = 6144` (not in census row)
- `max_position_embeddings = 40960` ✓
- `tie_word_embeddings = true` ✓
- `sliding_window = null` ✓ (census omits SWA column for Qwen 3, which is correct)
- `hidden_act = silu` ✓

**Verdict: clean. No fact errors.**
Note: `head_dim=128` while `hidden_size/num_attention_heads = 2048/16 = 128` — coincidence here, but Qwen3 explicitly sets `head_dim=128` in config and does *not* derive it. Worth flagging in the body that Qwen3 joins Gemma as a model that decouples head_dim from hidden_size/heads (the census buries this in Surprise #12 but doesn't list Qwen3 as an example — should add Qwen3-0.6B as the cleaner example: `H=16, hidden=1024 → hidden/H = 64`, but head_dim=128).

### 3.2 Qwen3-0.6B (verification of census Surprise #12)

Verified: `num_attention_heads=16, num_key_value_heads=8, head_dim=128, hidden_size=1024, num_hidden_layers=28`.

Census row says `16/8/128` ✓ but never points out that `hidden/H = 64 ≠ head_dim = 128` here. **The cleanest decoupling-of-head-dim example in the corpus is Qwen3-0.6B, and the census misses the opportunity.** Surprise #12 currently lists Gemma 2 2B, Phi-3-mini (which is *not* decoupled — 96 = 3072/32), and Phi-4-mini (also not decoupled — 128 = 3072/24 ⇒ that's 128, not decoupled). It should list **Qwen3 (all sizes from 0.6B to 8B), Gemma 2/3 (all sizes)**.

**Verdict: factually clean, but the architectural callout is mis-illustrated.**

### 3.3 Llama-3.1-8B (census section 2.1)

Census claim: `GQA`, `32/8/128`, vocab `128256`, RoPE `Llama-3 (factor 8)`, ctx `131072`, tied no.

Verified from <https://huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/config.json>:
- `num_attention_heads = 32` ✓
- `num_key_value_heads = 8` ✓
- `head_dim = 128` ✓
- `vocab_size = 128256` ✓
- `rope_theta = 500000.0` ✓
- `intermediate_size = 14336` (not in census row)
- `max_position_embeddings = 131072` ✓
- `tie_word_embeddings = false` ✓
- `rope_scaling = {factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192, rope_type="llama3"}` ✓

**Verdict: clean. No fact errors.**

### 3.4 Gemma-3-4B-pt (census section 2.3)

Census claim: `GQA, 5:1 SWA:full`, `8/4/256`, RoPE `global 1e6 + local 1e4 + linear factor 8`, vocab `262208`, ctx `131072 / 1024 alt`.

Verified from <https://huggingface.co/unsloth/gemma-3-4b-pt/raw/main/config.json> (text_config):
- `num_attention_heads = 8` ✓
- `num_key_value_heads = 4` ✓
- `head_dim = 256` ✓
- `hidden_size = 2560` (not in row)
- `vocab_size = 262208` ✓
- `rope_theta = 1000000.0` ✓
- `rope_local_base_freq = 10000.0` ✓
- `intermediate_size = 10240` (not in row)
- `max_position_embeddings = 131072` ✓
- `sliding_window = 1024` ✓
- `sliding_window_pattern = 6` ✓ (= 5 sliding then 1 full, matches the prose)
- `final_logit_softcapping = null` ⚠️ — the census prose (Section 5.3, Section 4.7) implies Gemma 3 has softcapping, but the verified config shows `null`. **Gemma 3 dropped softcapping**; Gemma 2 had it. Section 4.7 (axis row "`final_logit_softcapping`") lists "Gemma 2/3" together which is **incorrect — Gemma 3 has no logit softcap** (and no attn softcap either; `attn_logit_softcapping = null`). This is a **real fact error in axis 4.7**.
- `attn_logit_softcapping = null` ⚠️ — Same as above. Census 4.7 row "`attn_logit_softcapping` Gemma 2 (50.0)" is correct, but Surprise #3 implies Gemma 3 inherits Gemma 2 mechanisms wholesale; it does not.
- `query_pre_attn_scalar = 256` ✓
- `hidden_activation = gelu_pytorch_tanh` ✓

**Verdict: census row is clean for the 8/4/256 part, but axis 4.7 incorrectly lumps Gemma 3 with Gemma 2 for softcapping. MUST-FIX in axis 4.7. Also, the linear-factor-8 RoPE scaling claim does not appear in the top-level config snippet I retrieved; it's possible the original config has a `rope_scaling.factor=8.0` block — should re-verify.**

### 3.5 Phi-3-mini-4k-instruct (census section 2.4)

Census claim: `MHA`, `32/32/96`, RoPE `vanilla (θ=10000)`, RMS pre, SwiGLU fused, vocab `32064`, MS tok, untied, ctx `4096 / 2047`.

Verified from <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/main/config.json>:
- `num_attention_heads = 32` ✓
- `num_key_value_heads = 32` ✓
- `head_dim = 96` ✓ (= 3072/32)
- `vocab_size = 32064` ✓
- `rope_theta = 10000.0` ✓
- `intermediate_size = 8192` (not in row)
- `max_position_embeddings = 4096` ✓
- `tie_word_embeddings = false` ✓
- `sliding_window = 2047` ✓
- `rope_scaling = null` ✓
- `hidden_act = silu` ✓

**Verdict: clean. No fact errors.**

### 3.6 Mistral-7B-v0.3 (census section 2.5)

Census claim: `GQA (SWA dropped in v0.2+)`, `32/8/128`, RoPE `vanilla (θ=1e6)`, vocab `32768`, ctx `32768`, untied.

Verified from <https://huggingface.co/mistralai/Mistral-7B-v0.3/raw/main/config.json>:
- `num_attention_heads = 32` ✓
- `num_key_value_heads = 8` ✓
- `head_dim = 128` (computed, not in config) ✓
- `vocab_size = 32768` ✓
- `rope_theta = 1000000.0` ✓
- `intermediate_size = 14336` (not in row)
- `max_position_embeddings = 32768` ✓
- `tie_word_embeddings = false` ✓
- `sliding_window = null` ✓ (confirms v0.3 has no SWA)
- `hidden_act = silu` ✓

**Verdict: clean. No fact errors.**

### 3.7 DeepSeek-Coder-V2-Lite-Base (census sections 2.7 / 2.10)

Census claim: `MLA (kv_lora=512, qk_nope=128, qk_rope=64, v_head=128, no q_lora)`, `16/16 (latent)`, `64 routed + 2 shared, top-6`, `1 dense layer`, YaRN.

Verified from <https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/raw/main/config.json>:
- `num_attention_heads = 16` ✓
- `num_key_value_heads = 16` ✓
- `vocab_size = 102400` ✓
- `hidden_size = 2048`
- `rope_theta = 10000` ⚠️ — census section 2.7 doesn't quote θ for DeepSeek; but it's relevant because YaRN extends 4k → 163k.
- `intermediate_size = 10944`
- `max_position_embeddings = 163840` ✓
- `q_lora_rank = null` ✓ (census says "no q_lora")
- `kv_lora_rank = 512` ✓
- `qk_nope_head_dim = 128` ✓
- `qk_rope_head_dim = 64` ✓
- `v_head_dim = 128` ✓
- `n_routed_experts = 64` ✓
- `num_experts_per_tok = 6` ✓
- `n_shared_experts = 2` ✓
- `moe_intermediate_size = 1408` ✓ (census says "moe_ffn=1408")
- `first_k_dense_replace = 1` ✓
- `rope_scaling = {type: yarn, factor: 40, original_max_position_embeddings: 4096, beta_fast: 32, beta_slow: 1, mscale: 0.707, mscale_all_dim: 0.707}` ✓ (census has mscale=0.707, factor=40 — both check)

**Verdict: clean. No fact errors.**

### 3.8 OLMo-2-1124-7B (spot-check, not on the required-5 list)

Census claim: `MHA`, `32/32/128`, RoPE `vanilla (θ=5e5)`, vocab `100352`, untied.

Verified: `num_attention_heads = 32`, `num_key_value_heads = 32`, hidden=4096, `vocab_size = 100352`, `rope_theta = 500000`, `tie_word_embeddings = false`, `hidden_act = silu`, `max_position_embeddings = 4096`. **All ✓.**

### Fact-check tally

- **Numeric errors in the verified rows: 0**
- **Architectural-claim errors (Gemma 3 softcapping; Phi-3-mini partial-rotary in Surprise #12; TinyLlama listed as MHA in 4.1 row but as GQA in 2.1 row): 3**
- **Citation issues: 1 (CodeGemma 7B has no working link)**

The numeric rows are accurate; the prose claims about Gemma 3 inheriting Gemma 2's softcapping (axis 4.7) and about Phi-3-mini's head_dim decoupling (Surprise #12) are wrong.

---

## Section 4. Axis Completeness

The census names 11 axes. Working through each:

1. **Attention type** (4.1) — generally good, but the TinyLlama row inside the table labels it MHA with "(KV=4 actually GQA, see below)" — this is contradicted by the table at 2.1 which lists TinyLlama as GQA. **Internal inconsistency: must-fix.** Also: Mamba-2 not enumerated (only Mamba-1). Linear-attention sub-types (RWKV-6 vs RWKV-7 vs Mamba vs Mamba-2 vs Griffin vs xLSTM) are flattened into one row "Linear / SSM" — should be sub-divided because the cache shape differs (Mamba: `(d_state, expand·d)`; Mamba2: `(n_groups, d_state)`; RWKV-7: `(num_heads, head_size, head_size)`; Griffin: `(lru_width,)`).

2. **Positional encoding** (4.2) — missing variants:
   - **ALiBi** (MPT, Baichuan 1, parts of OPT family). Even if the census excludes pre-2024 ALiBi models, the *contrast* matters: the survey can't say "RoPE won" without an ALiBi row.
   - **Sinusoidal absolute** (Florence-2 listed but axis 4.2 calls it "Absolute learned" — different mechanism; verify which Florence-2 uses).
   - **CoPE / Context-position encoding** (Meta, 2024-05) — academic, may skip.
   - **3D RoPE / multimodal RoPE** (Qwen2.5-VL uses M-RoPE = 3-axis temporal/height/width). Once VLM text towers are in scope, M-RoPE belongs here.

3. **Normalization** (4.3) — missing:
   - **Sandwich norm** (Cogview / earlier Chinese models). Not in any 2024–2025 mainstream model, can skip.
   - **DeepNorm** (Microsoft, 2022). Skip.
   - **Granite explicitly carries** `embedding_multiplier` *and* a `pre_norm` — these interact. Census doesn't address the order.
   - **OLMo 2's `norm_first` / placement** is captured but the `RMSNorm without affine bias` toggle isn't an axis.

4. **FFN family** (4.4) — missing:
   - **ReLU²** (Primer, BitNet b1.58 4T: `hidden_act = "relu2"`). The BitNet b1.58 verified config shows `hidden_act = relu2`. This is **a third activation choice** (alongside SiLU/SwiGLU and GELU/GeGLU) and is omitted from axis 4.4. **MUST-FIX once BitNet is added.**
   - **SwishGLU vs SwiGLU** (Llama-2 vs Llama-3) — same.
   - **Reglu / Bilinear** (academic).
   - **No-FFN** (pure Mamba layers have an SSM that subsumes the FFN; this is mentioned in passing but not as an axis).

5. **MoE routing** (4.4 / 4.5) — missing as explicit axes:
   - **Aux-loss-free routing** (DeepSeek-V3): a per-expert bias is updated to balance load instead of an auxiliary loss. Not mentioned.
   - **Shared-experts always-on policy** (DeepSeek-V2/V3; OLMoE has 0): noted, but the **router activation** (sigmoid vs softmax) is also a per-model choice not enumerated.
   - **`norm_topk_prob`** is noted in DeepSeek/Qwen3 contrast but should be in the axis table.
   - **Expert-grouping** (DeepSeek's expert groups / Phi-3.5-MoE's `router_aux_loss_coef`) — missing.
   - **Top-k vs Switch (k=1)**: not enumerated.

6. **KV-cache shape** (4.5) — good but **incomplete for hybrid layouts**. Should distinguish Mamba-2 from Mamba-1 cache.

7. **Tokenizer/vocab** (4.6) — generally fine, but missing:
   - **Vocab-size→hidden-size product** as a cost-fraction analysis (e.g., for Llama 3.2 1B, `vocab*hidden = 128256*2048 = 262M`, which is 25% of total params at 1B). This kind of analysis is one the survey paper bar demands.
   - **Tekken / Mistral v3 tokenizer** (Mistral 7B v0.3 uses extended-32k = 32768; Nemo uses 131072 Tekken).
   - **Tokenizer normalization differences** (NFC vs NFKC) — these matter for cross-tokenizer evaluation comparability.

8. **Scalar multipliers / μP** (4.7) — covered well. Add Gemma 3 negative entry (it dropped softcapping) and Cohere/Aya `logit_scale=0.0625`.

9. **Bias presence** (4.8) — covered.

10. **Embedding tying** (4.9) — covered, but "the cutoff is ~3–4B" is hand-wavy; **Cohere Aya 8B has tied embeddings**, breaking the rule.

11. **Per-layer heterogeneity** (4.10) — incomplete:
    - **OpenELM** is the *canonical* per-layer heterogeneity model — `num_query_heads`, `num_kv_heads`, and `ffn_multipliers` are all per-layer lists. This is a fundamentally different axis than "alternating SWA/full" or "alternating dense/MoE". The census doesn't cover it because it doesn't include OpenELM. **MUST-FIX.**
    - **MobileLLM 1B/1.5B** uses block-wise weight sharing (the same Q/K/V/MLP weights are tied across N adjacent layers). This is a fourth heterogeneity dimension. **SHOULD-FIX.**
    - **Gemma 3n Matformer** — width is elastically subsetable at runtime. Fifth heterogeneity dimension.

### Missing axes (new axes the survey should add):

- **A. Per-layer width/depth scaling**. OpenELM, MobileLLM, Gemma 3n. Currently entangled with axis 4.10 but qualitatively different — these models vary the *shape* of each layer, not just the *type* of operator.
- **B. Position-extension method versus base-length-extension method.** Context-length extension is enumerated in 4.11 (six methods), but the axis is incomplete: missing **DCA (Dual Chunk Attention, ChatGLM3 32k), LongLora (low-rank fine-tune for context), PI (Position Interpolation, original Llama-2 long-context method), and the explicit "extend base θ only" approach (Qwen 2.5: just raise rope_theta to 1e7 without rope_scaling).**
- **C. Parallel vs sequential sublayer.** GPT-J / Phi-2 / RWKV channel-mix all run attention and FFN as parallel branches summed into the residual. Modern Llama-style is sequential. This is *not* listed as an axis. Required for Phi-2 / RWKV coverage.
- **D. Weight sharing across layers.** Universal Transformer style. MobileLLM, ALBERT-like sharing. Not in axis list.
- **E. Quantization-native architectures.** BitNet b1.58 has architectural choices (ReLU², `bitlinear` layers) baked into its training recipe. This is a real axis if any quantization-native model is included.
- **F. Activation function (standalone).** Currently entangled inside "FFN family" (4.4). Survey-quality treatment should pull activation out as its own axis: `{SiLU, GELU, GELU-tanh, gegelu, ReLU², sigmoid+ReLU² (RWKV), SwishGLU}`.
- **G. Attention scale**. The default `1/sqrt(head_dim)` is overridden by Gemma 2/3 (`query_pre_attn_scalar=256`), Granite (`attention_multiplier=0.015625`), Phi-3-small μP (`mup_attn_multiplier=1.0`). Survey-quality treatment: this is currently a sub-row of 4.7, but it determines whether a fused attention kernel can be reused across models.
- **H. RoPE rotation domain.** Full-rotary vs partial-rotary vs split (MLA's nope/rope split) vs nope-per-layer (SmolLM3). The current axis 4.2 conflates the scaling-method axis with the rotation-domain axis.

### Severity (Section 4)
- **MUST-FIX axes**: Per-layer width scaling (A), Parallel sublayer (C), Activation as standalone (F), RoPE rotation domain (H) — **4**.
- **SHOULD-FIX axes**: Position-extension completeness (B), Weight sharing (D), Quantization-native (E), Attention scale (G) — **4**.

**Total axis gaps: 8.**

---

## Section 5. Documentation Quality

For a survey-paper bar:

1. **Citations to tech reports / model cards are inconsistent.** Some Phi rows cite `arXiv:2404.14219`, Llama 3 cites `arXiv:2407.21783`, but Qwen, Gemma, OLMo, SmolLM, Granite, MiniCPM, InternLM, StableLM, RWKV, Mamba, Zamba, Jamba — none have paper citations. A real survey paper has a citation per row, not just a HuggingFace link. **MUST-FIX.**

2. **Several HuggingFace links cite `unsloth/` mirrors.** The census disclaims this in Section 6 ("For models where the original repo requires auth, an authoritative mirror (e.g. `unsloth/`, `deepseek-ai/`) was used"). For a peer-reviewed survey this is unprofessional — `unsloth` is a community quantization repo, not an authoritative source. Use the official Meta / Google / IBM / etc. repos with a note about auth, or use the `transformers` repo's `tests/fixtures/` configs. **SHOULD-FIX.**

3. **"Total cataloged: 45 distinct shipped variants across 22 distinct architectures"** — this number is unsourced and uncountable from the tables (the tables list ~50 rows depending on how multi-row entries like "Phi-3.5 vision" are counted; "22 distinct architectures" is not enumerated anywhere). **SHOULD-FIX.**

4. **Tokenizer details are imprecise.** Axis 4.6 says "Phi-3 / Phi-3.5 mini / Phi-3.5 vision" use MS-SP-32k. Phi-3.5-vision actually uses a derived Phi-3 tokenizer with image tokens prepended; the vocab is still 32064 + image-token offset. Phi-4-mini uses o200k (200064) — census gets this right. Qwen tokenizer is "BPE (Tiktoken-style, byte-fallback)" — actually it's a Tiktoken regex with `endoftext`/`im_start`/`im_end` reserved tokens, not classic GPT2 BPE. SmolLM is "HF-bpe 49k" — actually it's a StarCoder-2-derived 49152-vocab BPE; **the census separately calls it "SmolLM/StarCoder BPE" in the table** which is more accurate. **Inconsistency between axis-4.6 row labels and the per-family Tok column.** SHOULD-FIX.

5. **Relationship between models is underspecified.**
   - "Granite from Llama" is mentioned nowhere. Granite *is* a Llama-architecture model with μP scalars layered on top. The census should say so explicitly.
   - "R1-Distill-Qwen-7B = Qwen2.5-Math-7B backbone" is in the row, good — but Qwen2.5-Math-7B isn't itself in the census.
   - "PaliGemma 3B text tower = Gemma 1 2B" needs the Gemma 1 2B row to exist.
   - "DeepSeek-Coder-V2-Lite vs DeepSeek-V2-Lite" — are they the same architecture? (Yes, with code-only training data — should clarify.)

6. **No per-layer parameter breakdown is provided.** A survey paper of layer evolution should report, per model, the fraction of parameters in: embedding, attention QKV/output, FFN, LM head. For Llama 3.2 1B: vocab×hidden = 263M = ~25% of total; for Qwen 3 30B-A3B: routed-expert weights dominate; for Phi-3-small: μP scalars add zero params but change inference math. The axes are listed without quantitative weight. **MUST-FIX for survey quality.**

7. **No FLOP / inference-cost comparison.** Survey papers typically include a "compute per token" axis. For Mamba 2.8B, attention-free → linear-in-seq; for Gemma 3 (5:1 SWA), full-attn fraction = 1/6 → ~6× speedup at long context. The census says this in prose (Surprise #3) but doesn't quantify.

8. **The "Released" column for several models is inaccurate** in ways that matter for evolution:
   - TinyLlama listed as 2024-01 — v1.0 was Dec 30, 2023 (v0.1 release Sept 2023). Get the actual first-version date if telling an evolution story.
   - Mamba 2.8B listed as 2023-12 — the original paper is Dec 1, 2023 (correct), but the HF mirror was uploaded later.
   - Jamba "mini (early dev card) 2024-04" — Jamba was released March 2024; the dev card is a placeholder. **A dev card should not be the source.** **MUST-FIX**: use a real Jamba config (Jamba-v0.1 was the public 2024-03 release at 52B/12B active; the "tiny-dev" version is a developer fixture with 1 expert and is not representative).

9. **Source for Gemma 2 9B (row 2.3)** says "`(gemma-2 tech report)`" — no URL. Same for CodeGemma 7B. **SHOULD-FIX.**

10. **The cross-check claim** in Section 3 ("`C:\Users\zhengte\external\transformers\src\transformers\models\<name>\modeling_*.py`") is a local-filesystem path, not reproducible by readers. A published survey would cite git SHA + line numbers. **MUST-FIX for publication.**

### Severity (Section 5)
- **MUST-FIX**: items 1, 6, 8 (Jamba dev-card), 10 — **4**.
- **SHOULD-FIX**: items 2, 3, 4, 5, 7, 9 — **6**.

---

## Section 6. Recommended Fixes for v2

Concrete, prioritized actions.

### 6.1 Must-fix before claiming "survey paper depth"

1. **Add a 2023 baseline section.** Mandatory rows: Llama 2 7B, Mistral 7B v0.1, MPT 7B, Falcon 7B, StarCoder 1, Qwen 1 7B, ChatGLM2 6B, DeepSeek LLM 7B, OLMo 1 7B, Phi-1.5 / Phi-2, StableLM 3B-4E1T. Without this section the "past 3 years" claim is unfounded.

2. **Add per-generation contrast subsections** for Llama, Mistral, Qwen, Phi, Gemma, DeepSeek, OLMo, StableLM. Each subsection: 1 paragraph + a 2-column diff table ("`What changed`" / "`Why (paper-cited)`"). This is the survey content.

3. **Fix the Gemma 3 softcapping error in axis 4.7.** `final_logit_softcapping=null` and `attn_logit_softcapping=null` for Gemma 3 (verified). The axis row should show Gemma 2 only, with a footnote that Gemma 3 dropped softcapping (this is a *narrative* point — why was it dropped? `softpix` was found unstable for RL-tuned variants).

4. **Add OpenELM and elevate axis 4.10 ("per-layer heterogeneity") into two axes**: (a) per-layer *type* alternation (Gemma 3 SWA/full, Zamba2, Jamba, DeepSeek-V2-Lite dense/MoE), and (b) per-layer *width/depth scaling* (OpenELM Q/KV/FFN, MobileLLM weight sharing, Gemma 3n matformer).

5. **Add BitNet b1.58 and a "quantization-native architecture" axis.** ReLU² activation alone justifies a new row in axis 4.4.

6. **Add Phi-2 + a "parallel vs sequential sublayer" axis.** Currently no model represents the parallel-block lineage, even though RWKV's channel-mix is morally equivalent.

7. **Add Phi-4-mini-flash and Falcon-H1.** The "hybrid Mamba+attention" subspace currently has only Zamba2 and Jamba; Phi-4-mini-flash (Microsoft) and Falcon-H1 (TII) are the 2025-era exemplars and are missing.

8. **Replace `unsloth/` and `Jamba-tiny-dev` citations with authoritative sources.** Use `meta-llama/`, `google/`, `mistralai/`, `ai21labs/Jamba-v0.1` (or `Jamba-1.5-Mini` for the small variant).

9. **Add a paper citation per row** (arXiv ID + first-author + month). HuggingFace links are not survey citations.

10. **Add a parameter-fraction breakdown column** to each table: `embed%` / `attn%` / `ffn%` / `lm_head%`. This is one Python script away from existence.

11. **Resolve the TinyLlama MHA-vs-GQA contradiction** between section 2.1 (GQA 32/4) and axis 4.1 (listed under MHA "TinyLlama (KV=4 actually GQA, see below)"). TinyLlama is GQA. Move it.

12. **Add explicit local-file SHAs and git refs** for the `transformers` cross-check in Section 3, or remove the local-path citations.

### 6.2 Should-fix

13. Add RecurrentGemma (Griffin/Hawk), xLSTM, Mamba-2 (real config not just the family mention), Hymba — to round out the linear-recurrence taxonomy.
14. Add Cohere Aya 8B (tied-embeddings-at-8B counterexample).
15. Add Gemma 1 2B/7B and Gemma 3n.
16. Add Mixtral 8x7B (just for the "MoE arc origin point"; can be a footnote row noting `>8B active`).
17. Add Phi-3.5-MoE (4 routed/16 experts is a 3rd MoE pattern).
18. Add MiniCPM 1/2 (μP-scalar provenance).
19. Add ALiBi axis row even if no in-corpus model still uses it — the *abandonment* is the point.
20. Add an "abandoned features" subsection: SWA on/off/on, ALiBi → RoPE, MQA → GQA, QKV-bias → no-bias, parallel → sequential sublayer, μP → multipliers.
21. Add an evolution-graph figure (text DAG): "Llama-1 → Llama-2 → Llama-3" / "Mistral-v0.1 → v0.2 → v0.3" / "Gemma-1 → 2 → 3 → 3n" / etc., with annotations on the edges for what changed.
22. Subdivide axis 4.1's "Linear / SSM" row into Mamba-1, Mamba-2, RWKV-6, RWKV-7, Griffin/Hawk, xLSTM — each with its distinct cache shape.

### 6.3 Nice-to-have

23. Add a column for "activation precision in training" (fp16/bf16/fp8) — DeepSeek-V3's fp8 training is a 2024 milestone.
24. Add a column for "data tokens trained on" — Llama 3 herd's 15T, Qwen 2.5's 18T, Phi-3-mini's 3.3T are part of the architectural-decision context.
25. Add a section on "context-length extension at fine-tune time" (LongRoPE-tune vs YaRN-tune vs no-tune-just-scale).
26. Reference `transformers` `tests/fixtures/*` configs as a reproducible source rather than live HuggingFace fetches.

---

## Summary

The census is a solid present-day inventory at the *minimum operator API* level. As a survey of three years of evolution, it is a snapshot pretending to be a story: there is **no 2023 baseline year**, **no generation-to-generation diff narrative**, **no per-row paper citation**, and at least **3 prose-level factual errors** (Gemma 3 softcapping, TinyLlama MHA-vs-GQA, Surprise-#12 head-dim decoupling exemplar). At least **14 must-fix mainstream models** are absent, including the entire 2023 reference set (Llama 2, Mistral v0.1, MPT, Falcon, StarCoder 1, Qwen 1, ChatGLM, DeepSeek LLM, OLMo 1, Phi-1/2, StableLM 3B), the most architecturally distinctive 2024 release (OpenELM with per-layer scaling), and three 2025 hybrid/quantization-native releases (Phi-4-mini-flash, Falcon-H1, BitNet b1.58).

Fix the 2023 baseline and the generation-arc narrative and this becomes a survey. Without them, it's a config dump.
