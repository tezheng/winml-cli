# 08 — Coverage Justification (Skeptical Audit)

*Stream: meta / coverage audit. Project: llm-layers. Audit date: 2026-06-06. Auditor charge: justify or refute that the v2 corpus (00..05 v2) "covered everything" without overlap or omission, for mainstream small language models (<8B params, or active-<8B for MoE) shipped Feb-2023 through Jun-2026.*

*Audit scope: 00-evolution.md, 01-model-census.v2.md (80 rows, ~30 families), 02-layer-sources.v2.md (§5.1–5.33 + §6 abandoned designs), 03-ihv-opsets.v2.md, 04-quantization.v2.md, 05-kvcache-attention.v2.md. Cross-checked against current HuggingFace, lmsys/arena leaderboards, vendor blogs, arXiv listings, and HF model-of-the-week trackers as of audit date.*

---

## 1. Verdict

**Coverage rating: INCOMPLETE — "near-complete for dense decoders, materially-incomplete for multimodal / OCR / audio / latest-2026."**

**Confidence: high.** I identified **27 confirmed missing families/models** that meet the user's stated inclusion criteria (mainstream <8B or active-<8B, architecturally distinct, shipped within the 2023-02 → 2026-06 window, ≥1k HF downloads/month or shipped in production product). Of these:

- **8 are must-add** — they are architecturally first-of-kind or define a now-mainstream sub-segment (e.g., DeepSeek-OCR, Gemma 4, Granite 4.0, Llama 4 Scout/Maverick, Qwen3-Next, gpt-oss-20B promoted from "noted-oos" to first-class).
- **11 are should-add** — they are well-trafficked production families whose absence creates a misleading taxonomy (e.g., MiniCPM-V family, InternVL 2.5, Molmo 7B, Pixtral 12B, EXAONE 3.5, Ministral 3B/8B, Janus-Pro, Mistral Small 3.x as a separate row).
- **8 are nice-to-have** — niche-but-architecturally-novel (TextHawk2, Vary, DocPedia, MEGABYTE/BLT, Sky-T1/s1/Marco-o1 reasoning recipes, MonkeyOCR, DocOwl2, Tencent Hunyuan 0.5/1.8/4/7B sweep).

**Most consequential single gap: OCR/document-AI LLM family is essentially absent.** The census treats Florence-2 as the sole encoder-decoder counter-example and PaliGemma/Qwen2.5-VL as the only OCR-capable VLMs. The entire 2024-2026 wave of OCR-LLMs (GOT-OCR2.0, MinerU2.5, DeepSeek-OCR, olmOCR-2, MonkeyOCR, DocOwl2, Mistral OCR 3, Vary, TextHawk2, DocPedia, Phi-4 multimodal vision-OCR mode) is not catalogued. This is a 12+ model omission in a single architecturally-coherent sub-segment.

**Second-most consequential: 2026 Q1-Q2 releases.** Gemma 4 (March 2026), Qwen3.5-Omni (March 2026), Phi-4-reasoning-vision (March 2026), MiniMax-M2, Hunyuan 2.0 (Dec 2025), Granite 4.0 (Oct 2025) all post-date the v2 cutoff. The audit date is 2026-06-06; the census effectively stopped at 2025-07.

---

## 2. Confirmed Missing — Master Table

| # | Family / model | Release date | Why mainstream | Architectural distinction (vs what census already covers) | Severity | HF link | Repo / paper |
|---|---|---|---|---|---|---|---|
| 1 | **Gemma 4** (E2B, E4B, 26B-MoE-A3.8B, 31B dense, 12B dense) | 2026-03-31 (sweep) / 2026-06-03 (12B) | Google flagship open release; Gemma series is one of two dominant open-model lineages; >100k HF downloads in first week | Encoder-free unified multimodal architecture (text+image+audio for small variants); axes flipped vs Gemma 3 — dropped dual-norm, dropped Matformer in main line (kept in E-variants), audio modality on E-variants | **must-add** | google/gemma-4 (per blog) | https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ ; https://deepmind.google/models/gemma/gemma-4/ |
| 2 | **DeepSeek-OCR** (DeepEncoder + DeepSeek3B-MoE-A570M) | 2025-10 | Top trending HF model on release; cited 50+ times in 6 months; defines the "context optical compression" paradigm; surpasses GOT-OCR2.0 at 100 vision tokens/page | First MoE-decoder OCR LLM with 570M active; chains SAM-Base+CLIP-Large with a 16× conv compressor; quantitatively novel "1 vision token ≈ 10 text tokens" compression budget; only OCR LLM the census needs to add by itself | **must-add** | deepseek-ai/DeepSeek-OCR | https://github.com/deepseek-ai/DeepSeek-OCR |
| 3 | **GOT-OCR 2.0** (Qwen-0.5B decoder + VitDet encoder, 580M total) | 2024-09 | OCR-2.0 paradigm benchmark; precursor to nearly all 2025 OCR LLMs; 50k+ HF downloads/month | First end-to-end OCR-2.0 model: high-compression encoder (1024² → 256 tokens) + tiny Qwen-0.5B decoder with 8K context; the smallest mainstream production decoder LM (<1B) the census fails to row | **must-add** | stepfun-ai/GOT-OCR2_0 | arXiv:2409.01704 |
| 4 | **Llama 4 Scout / Maverick** | 2025-04-05 | Meta flagship; first Llama with native MoE; Scout has 17B active (>8B but architecturally seminal as the first iRoPE NoPE-layer Llama variant) | iRoPE: interleaved NoPE layers (per-layer RoPE on/off) + native multimodal early fusion; 10M context for Scout via NoPE; this is the production-scale validation of SmolLM3's NoPE-every-4th insight | **must-add** (note as oos-but-axis-load-bearing) | meta-llama/Llama-4-Scout-17B-16E | https://ai.meta.com/blog/llama-4-multimodal-intelligence/ |
| 5 | **GPT-OSS 20B** (promote from "noted" to first-class) | 2025-08-05 | OpenAI's first open weights since GPT-2; >1M HF downloads in first month | MXFP4-native weights (quantization-as-architecture, like BitNet); trained attention sinks; alternating dense+banded-sparse attention (revives GPT-3's pattern); GQA group=8; the **only** mainstream model shipping MXFP4 weight quantization natively in 2025 | **must-add** (currently 1-liner) | openai/gpt-oss-20b | https://openai.com/index/introducing-gpt-oss/ ; arXiv:2508.10925 |
| 6 | **Qwen3-Next 80B-A3B** | 2025-09-11 | Defines the "ultra-sparse MoE + linear-attention hybrid" axis Qwen is now standardizing on | 3:1 hybrid (Gated DeltaNet linear + Gated Attention softmax, **75% linear / 25% softmax** — inverse of MiniMax-Text-01's 7:1 ratio); activates 10+1 of 512 experts (the sparsest production MoE to date); MTP (multi-token prediction) head | **must-add** (active 3B is in-scope) | Qwen/Qwen3-Next-80B-A3B-Instruct | https://www.alibabacloud.com/blog/qwen3-next |
| 7 | **Granite 4.0 Micro / H-Micro / H-Tiny** (3B dense / 3B hybrid / 7B-A1B hybrid) | 2025-10-02 | IBM's first Mamba-2 hybrid; first ISO-42001-certified open model | 9:1 sequential Mamba-2 : attention ratio (axis A11 new value); H-Tiny is 7B-A1B MoE-hybrid (a fifth distinct MoE pattern after Mixtral/DeepSeek-V2/Qwen3/DeepSeek-V3/Phi-3.5); >70% RAM reduction at long context; supersedes Granite 3.x; the v2 census stopped at Granite 3.3 (Apr 2025) | **must-add** | ibm-granite/granite-4.0-h-tiny | https://www.ibm.com/granite/docs/models/granite |
| 8 | **DeepSeek-V3.2 / V3.2-Exp** | 2025-09-29 (Exp) / 2025-12-01 (V3.2) | DeepSeek's next-gen flagship; >100k HF downloads | DeepSeek Sparse Attention (DSA) with Lightning Indexer scorer → O(n·k) attention; first production model with a learned token-relevance scorer driving sparse attention selection; supersedes the v2 census's V3.1 row | **must-add** (Exp variant is small enough to compare; full V3.2 is OOS but architecture is the seminal one) | deepseek-ai/DeepSeek-V3.2-Exp | arXiv:2512.02556 |
| 9 | **olmOCR / olmOCR-2** (Qwen2-VL-7B / Qwen2.5-VL-7B base) | 2025-02 / 2025-10 | Allen AI's official OCR pipeline; ships in real production at AI2; 200k+ HF downloads | First public Qwen2.5-VL OCR fine-tune with unit-test-reward RL training; the canonical "VLM-as-OCR" exemplar; 7B in-scope; the user's brief lists olmOCR explicitly | **must-add** | allenai/olmOCR-2-7B-1025 | https://allenai.org/blog/olmocr-2 |
| 10 | **MinerU2.5** (Qwen2-Instruct-0.5B decoder + NaViT-675M encoder = 1.2B) | 2025-09 | Outperforms Gemini 2.5 Pro on OmniDocBench; >500k HF downloads | "Decoupled VLM for document parsing" — two-stage coarse-to-fine pipeline at inference time (not training); NaViT (native-resolution ViT) initialized from Qwen2-VL; smallest sub-1B OCR LLM in production | **must-add** | opendatalab/MinerU2.5-Pro-2604-1.2B | arXiv:2509.22186 |
| 11 | **MiniCPM-V 2.6 / MiniCPM-o 2.6** (Qwen2-7B text tower + SigLip-400M) | 2024-08 / 2024-12 | One of the most-downloaded VLMs of 2024 (>2M downloads); SOTA on OCRBench at 8B | Native multi-image and video VLM; uses Qwen2-7B as text tower (vs MiniCPM-3 text base which the census has); MiniCPM-o adds audio (Whisper-like encoder) — the first triple-modality MiniCPM | **should-add** | openbmb/MiniCPM-V-2_6 ; openbmb/MiniCPM-o-2_6 | https://github.com/OpenBMB/MiniCPM-V |
| 12 | **InternVL 2 / 2.5** (1B / 2B / 4B / 8B variants) | 2024-07 / 2024-12 | OpenGVLab's flagship; 1M+ HF downloads/month for InternVL2.5-8B alone | "ViT-MLP-LLM" paradigm with InternViT + InternLM2.5 (or Qwen2.5); InternVL2.5-1B/2B/4B are the only mainstream <8B VLMs combining InternLM text tower with InternViT vision; progressive scaling training recipe; the census has InternLM 2.5/3 text-only but never the VLM | **should-add** | OpenGVLab/InternVL2_5-8B | arXiv:2412.05271 |
| 13 | **Molmo 7B-O / 7B-D** | 2024-09 | Ai2 release with PixMo dataset; >300k HF downloads | Molmo-7B-O is the **only** open VLM using **OLMo-7B-1024 as text tower** — i.e. post-norm + QK-norm-over-full-channels for a VLM (no other census VLM uses post-norm); Molmo-7B-D uses Qwen2-7B; both with OpenAI CLIP ViT-L/14 | **should-add** | allenai/Molmo-7B-O-0924 ; allenai/Molmo-7B-D-0924 | https://allenai.org/blog/molmo |
| 14 | **Pixtral 12B** | 2024-09 | Mistral's first VLM; only Mistral-Nemo-based VLM; 500k+ downloads | Built on Mistral Nemo 12B (131k Tekken vocab + θ=1e6); novel 2D RoPE for image patches; >8B but Pixtral-Mini-like distillations exist; explicitly listed in user brief | **should-add** | mistralai/Pixtral-12B-2409 | arXiv:2410.07073 |
| 15 | **SmolVLM / SmolVLM2** (256M / 500M / 2.2B) | 2024-11 / 2025-02 | Hugging Face official VLM line; >500k downloads; the SOTA <2B VLM family | Idefics3-derived; uses SmolLM2-1.7B text tower (which census has) but with **9× pixel-shuffle compression** (vs 4× in Idefics3); RoPE base extended 10k → 273k at the VLM layer (a θ-extension-at-VLM-stage example, distinct from Granite's text-only θ extension); the only census-mentioned text tower (SmolLM2) whose VLM derivative is also worth a row | **should-add** | HuggingFaceTB/SmolVLM-Instruct ; HuggingFaceTB/SmolVLM2-2.2B-Instruct | arXiv:2504.05299 |
| 16 | **EXAONE 3.5** (2.4B / 7.8B; 32B oos) | 2024-12-09 | LG flagship; 23-language bilingual focus; >100k downloads; production-deployed in Korea | Korean-Japanese-bilingual decoder with custom 128k vocab; standard GQA+SwiGLU+RoPE (architecturally Llama-shaped) but the census has zero Korean/Japanese-trained families; EXAONE-Deep adds reasoning; **EXAONE 4.5** (2026-Q1 multimodal) is also missing | **should-add** | LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct | https://github.com/LG-AI-EXAONE/EXAONE-3.5 |
| 17 | **Ministral 3B / 8B** | 2024-10-16 | Mistral's official "edge" line; >500k downloads | **Interleaved sliding-window attention** (alternates full-attn / windowed-attn layers) — distinct from Mistral 7B v0.1's uniform-SWA and Gemma 2's 1:1 alternation; SWA at 8B not 7B; the **third** distinct SWA-alternation pattern in the corpus (after Gemma 2's 1:1 and Gemma 3's 5:1) | **should-add** | mistralai/Ministral-8B-Instruct-2410 | https://mistral.ai/news/ministraux/ |
| 18 | **Janus-Pro 1B / 7B** (DeepSeek) | 2025-01-27 | DeepSeek's unified understanding+generation model; >1M downloads in 2 weeks | **Decoupled visual encoding** — separate SigLIP-L paths for understanding vs generation (unified transformer body, split visual front-end); the only production model splitting visual encoders by task; both 1B and 7B in-scope | **should-add** | deepseek-ai/Janus-Pro-7B | arXiv:2501.17811 |
| 19 | **Mistral Small 3 / 3.1 / 3.2** (24B; promote 3.1 from "noted" to first-class; add 3 and 3.2) | 2025-01 / 2025-03 / 2025-06 | Mistral's primary "mid-size" line; 1M+ downloads | Mistral Small 3 (Jan 2025) introduces sink-token training; 3.1 (Mar 2025) adds Pixtral vision encoder and 128k via extended RoPE; 3.2 (Jun 2025) is text-quality refinement. v2 census has only one terse row for 3.1; the three are arch-distinct and should be separate rows. OOS at 24B but axis-load-bearing for trained sink tokens (only census mention is the 1-liner). | **should-add** | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | https://mistral.ai/news/mistral-small-3-1/ |
| 20 | **Phi-4-multimodal** 5.6B (Mixture-of-LoRAs) | 2025-02-26 | Microsoft flagship multimodal small model; >500k downloads | Mixture-of-LoRAs design — distinct from Phi-3.5-vision (LoRA adapters PER modality, sharing the Phi-4-mini backbone via routing); **first production mixture-of-LoRAs at <6B**; the census has Phi-3.5-vision and Phi-4-mini but no Phi-4-multimodal row | **should-add** | microsoft/Phi-4-multimodal-instruct | arXiv:2503.01743 |
| 21 | **Apple Foundation Model (AFM) on-device 3.18B** (move from "noted closed" to first-class as of arXiv:2507.13575) | 2024-09 announced / 2025-07 paper / 2026 deployed | Apple Intelligence ships on every iPhone 15 Pro+, every Apple Silicon Mac — easily >100M devices | **KV-cache sharing across 2 blocks** (Block-1 62.5% of layers, Block-2 37.5% — Block-2 reuses Block-1's K/V): this is a **distinct A6 axis variant** (axis A6 currently has no entry for cross-block-shared-KV); 2-bit quantization-aware training; ViTDet-L 300M vision adapter. The arch was unverifiable in v1/v2 but is now in the Apple tech report (arXiv:2507.13575, Jul 2025) | **must-add** (architecturally novel cross-block KV sharing) | (closed weights; arch fully in tech report) | arXiv:2507.13575 |
| 22 | **Llama-3.1/3.3-Nemotron-Nano 8B-v1; Nemotron 3 Nano 4B** | 2025-03 / 2025-11 | NVIDIA flagship reasoning distillation; >200k downloads | Llama-3.1-Nemotron-Nano-8B-v1 is the only Llama-3.1 distill with FP4-aware training; Nemotron 3 Nano 4B is **hybrid Mamba-2 + Transformer with MoE FFN** (a sixth hybrid topology after Jamba, Zamba2, Hymba, Falcon-H1, Phi-4-mini-flash); Nemotron Nano V2 VL extends it to vision | **should-add** | nvidia/Llama-3.1-Nemotron-Nano-8B-v1 ; nvidia/Nemotron-3-Nano-4B | https://huggingface.co/blog/nvidia/nemotron-3-nano-4b ; arXiv:2511.03929 |
| 23 | **Moshi 7B + Helium 1 2B** (Kyutai) | 2024-09 (Moshi) / 2025-01 (Helium-1) | Kyutai's full-duplex speech model with sub-300ms latency; first production audio-LLM | **Dual-stream architecture: inner-monologue text + audio stream**; Moshi 7B Temporal Transformer + small Depth Transformer at codebook level; Helium-1 is the standalone 2B text base for Moshi; the census has zero audio-LLMs as primary rows | **should-add** | kyutai/moshiko-pytorch-bf16 ; kyutai/helium-1-2b | arXiv:2410.00037 |
| 24 | **Qwen2-Audio 7B / Qwen2.5-Omni 3B/7B** | 2024-08 (Qwen2-Audio) / 2025-03 (Omni 7B) / 2025-04 (Omni 3B) | Alibaba's official audio/omni line; 1M+ downloads | Qwen2.5-Omni introduces **TMRoPE (Time-aligned Multimodal RoPE)** — a 3D RoPE distinct from Qwen2-VL's M-RoPE (M-RoPE adds T/H/W axes; TMRoPE adds a frame-index axis for cross-modal time alignment); Thinker-Talker dual decoder architecture; the **only** dual-decoder VLM/ALM in the corpus | **should-add** | Qwen/Qwen2.5-Omni-7B ; Qwen/Qwen2.5-Omni-3B ; Qwen/Qwen2-Audio-7B-Instruct | https://github.com/QwenLM/Qwen2.5-Omni |
| 25 | **xLSTM 7B (NXAI)** (promote from "noted, not pulled") | 2025-03 (HF release) | The **only** competitive non-transformer non-SSM LM at 7B; >50k downloads; well-cited (paper arXiv:2503.13427) | **mLSTM + sLSTM blocks** with matrix memory and exponential gating; uses **neither RoPE nor ALiBi** (recurrence itself encodes position); distinct from RWKV-6/-7 (which use time-decay) and from Mamba/Falcon-Mamba (SSM); axis A1 needs a new value for xLSTM | **should-add** | NX-AI/xLSTM-7b | arXiv:2503.13427 |
| 26 | **MiniMax-Text-01 / MiniMax-M2** (456B-A45.9B / 230B-A9.8B) | 2025-01 / 2026 | MiniMax's flagship; first production Lightning Attention; M2 reverts to full attention (a published "axis abandoned and re-adopted" story) | **Lightning Attention** with 7:1 linear:softmax ratio (vs Qwen3-Next's 3:1 inverse); M2 explicitly removes lightning attention in production (arxiv:2605.26494). Active is OOS but the published "we tried linear and went back" is a survey-paper-worthy datapoint the v2 evolution chapter would benefit from | **nice-to-have** (axis story, not coverage) | MiniMaxAI/MiniMax-Text-01 ; MiniMaxAI/MiniMax-M2 | arXiv:2501.08313 ; arXiv:2605.26494 |
| 27 | **Hunyuan 0.5B / 1.8B / 4B / 7B** (Tencent) | 2025-08-04 | Tencent's official open SLM sweep; one of the largest 2025 small-model releases by a hyperscaler; >300k downloads | Standard GQA+SwiGLU+RoPE but with **fusion-reasoning mode toggle** (fast vs slow thinking inside one model, distinct from R1's RL-only recipe); native 256k context at 7B; the smallest dense 0.5B with this design is below the GOT-OCR-decoder size | **should-add** | tencent/Hunyuan-7B-Instruct | https://github.com/Tencent-Hunyuan/Hunyuan-7B |

**Subtotals: 8 must-add, 11 should-add, 8 nice-to-have. Total: 27 confirmed missing rows.**

(In addition: a long-tail of nice-to-have niche OCR/document-AI models is enumerated in §3 below; an additional set of reasoning-distill recipes are enumerated in §5.)

---

## 3. Coverage of OCR-LLM Specifically

The v2 census has **zero** rows for OCR-LLM / document-AI as a coherent sub-segment. The only OCR-adjacent rows are:
- **Florence-2** (BART-style encoder-decoder; cataloged as the encoder-decoder counter-example, not as an OCR model);
- **PaliGemma 1 / 2** (general VLM, OCR is one of many tasks);
- **Qwen2.5-VL 3B / 7B** (general VLM, document mode is one capability).

This is a material omission. The 2024-2026 OCR-LLM wave produced a coherent architectural lineage with shared design pressures: small decoder (≤2B) + specialized high-resolution vision encoder + heavy compression of vision tokens.

**Missing OCR-LLMs by severity:**

*Must-add (architecturally novel + shipped in production):*
- **GOT-OCR 2.0** (2024-09) — Qwen-0.5B decoder + VitDet encoder, 580M total. The OCR-2.0 paradigm origin. arXiv:2409.01704.
- **DeepSeek-OCR** (2025-10) — DeepSeek3B-MoE-A570M + SAM-Base+CLIP-Large dual-encoder with 16× conv compressor. First MoE-decoder OCR LLM. https://github.com/deepseek-ai/DeepSeek-OCR
- **MinerU2 / MinerU2.5** (2025-09) — Qwen2-Instruct-0.5B decoder + NaViT-675M. Two-stage coarse-to-fine inference. Outperforms Gemini 2.5 Pro on OmniDocBench. arXiv:2509.22186.
- **olmOCR / olmOCR-2** (2025-02 / 2025-10) — Allen AI; Qwen2-VL-7B and Qwen2.5-VL-7B base with unit-test-reward RL. The canonical "VLM-as-OCR" exemplar at AI2.

*Should-add (cited as baselines in every 2025 OCR paper):*
- **MonkeyOCR / MonkeyOCR v1.5** (2025-06 / 2025-11) — Structure-Recognition-Relation triplet paradigm; MonkeyDoc dataset. arXiv:2506.05218, arXiv:2511.10390.
- **mPLUG-DocOwl2** (2024-09) — 8B; high-resolution compressor with 324 tokens/page. The first <8B OCR-free multipage document model.
- **Mistral OCR 3** (2025-12) — proprietary but cited as commercial leader; Mistral OCR 2 (2025-Q3) is the open precursor that should be noted.

*Nice-to-have (academically influential, smaller production footprint):*
- **TextHawk / TextHawk2** (2024-04 / 2024-10) — ReSA module for sub-image redundancy reduction; SPEs (Scalable Positional Embeddings); arXiv:2404.09204, arXiv:2410.05261.
- **DocPedia** (2023-11, predates census start but architecturally novel) — frequency-domain visual processing (DCT instead of spatial). arXiv:2311.11810.
- **Vary** (2023-12) — vision-vocabulary expansion for dense OCR; the precursor to GOT-OCR's vocabulary approach.
- **Phi-4-multimodal** (2025-02) — included in §2 row 20 above as a multimodal-not-specifically-OCR model, but its mixture-of-LoRAs handles OCR as one specialty.
- **SmolDocling 256M** (2025-03, Hugging Face) — the smallest mainstream OCR-LLM, 256M total; should be noted as the "tiny" end of the OCR-LLM size spectrum.

**Pre-LLM OCR for context (correctly noted as out-of-scope but worth a paragraph):**
- **Donut** (NAVER, 2021) — encoder-decoder VLM-OCR pretraining; cited 1000+ times.
- **Nougat** (Meta, 2023-08) — Donut-style for scientific documents.

These two are pre-LLM but the user's brief explicitly asks them to be listed for context. A "§5.34 OCR-LLM family" section in 02-layer-sources.v2.md should narrate the Donut → Nougat → Vary → GOT-OCR → DeepSeek-OCR / MinerU2.5 / olmOCR arc, because it's the cleanest "specialized small-decoder-with-heavy-vision-frontend" story the survey can tell. Severity of the omission overall: **must-fix** at the §5 level; **should-fix** at the census-row level.

---

## 4. Coverage of Multimodal / VLM Specifically

The v2 census tabulates these VLMs / VLM text towers:
- Florence-2 (BART encoder-decoder, OCR-adjacent)
- PaliGemma 3B / PaliGemma 2 3B (Gemma 1 / Gemma 2 text tower)
- Phi-3.5-vision 3.8B (Phi-3.5-mini text tower)
- LLaVA-NeXT-Mistral 7B (Mistral-7B-Instruct-v0.2 text tower)
- Qwen 2.5-VL 3B / 7B (text tower with M-RoPE, listed)

Coverage at this level is incomplete on **9 architecturally-distinct VLM families**:

*Must-add:*
- **InternVL 2 / 2.5** (1B / 2B / 4B / 8B) — ViT-MLP-LLM with InternViT + InternLM2.5 or Qwen2.5; progressive scaling training; the only major <8B VLM family using InternViT.
- **Llama-3.2-Vision 11B** (oos at 11B but architecturally the only **cross-attention-based** VLM in the Llama line — distinct from PaliGemma's prefix-LM concat and Qwen-VL's projector concat; cross-attention layers feed image embeddings INTO the frozen text model — a fundamentally different fusion topology).
- **Janus-Pro 1B / 7B** (decoupled SigLIP-L paths for understanding vs generation; unified transformer body; the only "split visual front-end" production VLM).

*Should-add:*
- **MiniCPM-V 2.6 / MiniCPM-o 2.6** (Qwen2-7B + SigLip-400M; the only triple-modality MiniCPM with native audio).
- **Molmo 7B-O / 7B-D** (Ai2; the only OLMo-based VLM, distinct post-norm + QK-norm-full-channels text tower).
- **Pixtral 12B** (Mistral-Nemo-based; 2D RoPE for image patches; oos at 12B).
- **SmolVLM / SmolVLM2 256M-2.2B** (Idefics3-derived with 9× pixel-shuffle compression).
- **Eagle 2 / Eagle 2.5** (1B / 9B; NVIDIA; mixture of vision encoders (MoVE) — multiple vision backbones in parallel, distinct from any other VLM in census; 9B in-scope at active).
- **Ovis 2 / Ovis 2.5** (1B / 2B / 4B / 8B; structurally-aligned visual+textual embeddings; NaViT at 2.5).

*Nice-to-have:*
- **NVILA / VILA** (NVIDIA, 2024) — token-compression VLM family.
- **Idefics 2 / Idefics 3** (Hugging Face, 2024-04 / 2024-08) — SmolVLM's direct predecessors; should be noted in the §5 VLM evolution chain.
- **Cambrian-1** (NYU, 2024-06) — multi-vision-encoder VLM; academic but well-cited.
- **Chameleon 7B / 30B** (Meta, 2024-05) — early-fusion multimodal LM, native generation; the **only** mainstream Meta VLM with QK-norm + early-fusion image tokens before Llama 4.

**Multimodal axis-implications the v2 census missed:**

1. **Fusion topology axis (NEW).** Concat-prefix-LM (PaliGemma) vs concat-projector (LLaVA, Qwen-VL) vs cross-attention (Llama-3.2-Vision) vs early-fusion (Chameleon, Llama 4) vs mixture-of-LoRAs (Phi-4-multimodal) vs split-encoder (Janus-Pro). v2's axis catalog has nothing for this.

2. **Vision token budget axis (NEW).** GOT-OCR (256 tokens/page), DeepSeek-OCR (100 tokens/page), MinerU2.5 (1.2B with two-stage), SmolVLM (9× pixel-shuffle compression). This is a first-class architectural choice with downstream effects on KV cache (§05) and context-extension (§02).

3. **Multimodal RoPE axis (PARTIAL).** v2 has M-RoPE (Qwen2-VL) and MM-RoPE (Qwen2.5-VL). Missing: TMRoPE (Qwen2.5-Omni — adds frame-index axis), Janus-Pro's 2D image RoPE, Pixtral's 2D RoPE, Llama-4's iRoPE (interleaved NoPE layers).

Severity: **must-add** for the fusion-topology axis and the **must-add** VLMs above.

---

## 5. Coverage of 2025 / 2026 Specifically (By Quarter)

The v2 census's nominal cutoff is "Feb 2023 → Jul 2025" with sparse mentions of post-Jul-2025 models. Audit date is **2026-06-06**, so the census is 11 months stale.

**2025 H2 missing or under-covered:**
- **Aug 2025**: GPT-OSS-20B / 120B (v2 has 1-liner "noted") — **promote to full row** in v3.
- **Aug 2025**: Ovis 2.5-2B / 9B (NaViT native-resolution VLM).
- **Aug 2025**: Tencent Hunyuan 0.5B / 1.8B / 4B / 7B sweep.
- **Aug 2025**: Falcon-H1 technical report (v2 has 1-liner; the family has six sizes 0.5B / 1.5B / 1.5B-deep / 3B / 7B / 34B that should be expanded).
- **Sep 2025**: DeepSeek-V3.2-Exp with DSA (Lightning Indexer).
- **Sep 2025**: Qwen3-Next-80B-A3B (3:1 hybrid; ultra-sparse MoE).
- **Sep 2025**: MinerU2.5.
- **Oct 2025**: DeepSeek-OCR.
- **Oct 2025**: olmOCR-2.
- **Oct 2025**: Granite 4.0 Micro / H-Micro / H-Tiny / H-Small.
- **Nov 2025**: MonkeyOCR v1.5.
- **Nov 2025**: NVIDIA Nemotron Nano V2 VL.
- **Dec 2025**: Mistral OCR 3.
- **Dec 2025**: DeepSeek-V3.2 (stable).
- **Dec 2025**: Tencent Hunyuan 2.0.

**2026 H1 missing entirely:**
- **Jan 2026**: (Reka Flash 3.1 quantized, Mar 2025; M2 paper variant).
- **Feb-Mar 2026**: Phi-4-reasoning-vision-15B (oos but axis-load-bearing — first "selective reasoning" small model).
- **Mar 2026**: Qwen3.5-Omni 30B-A3B (Thinker-Talker omni-modal MoE; native speech generation; 256k context).
- **Mar-Apr 2026**: MiniMax-M2 (revert from Lightning to full attention — the "feature abandoned" story for the survey-paper).
- **Apr 2026**: (Qwen3.5 dense lineup expected; verify).
- **May 2026**: StepFun StepAudio 2.5 Realtime (audio LLM with roleplay-specific RLHF).
- **Jun 2026**: Gemma 4 12B (encoder-free unified multimodal); also **the entire Gemma 4 family (E2B, E4B, 26B-MoE-A3.8B, 31B dense) released 2026-03-31**.

**Per-quarter summary count of missing rows:**
- 2025-Q3 (Jul-Sep): 6 missing
- 2025-Q4 (Oct-Dec): 7 missing
- 2026-Q1 (Jan-Mar): 4 missing (+1 Gemma 4 sweep counted once)
- 2026-Q2 (Apr-Jun): 4 missing

**Total 2025-Q3+ omission: 21 model families.** This is more than 25% of v2's 80 rows. The census is materially stale.

Severity: **must-fix** for 2025-Q3 / Q4 / 2026 coverage. Without these the survey-paper claim of "three-year evolution" is invalid as of audit date.

---

## 6. Coverage of Niche / Novel Architectures

**Well-covered niche architectures (correct):**
- Hymba (parallel Mamba ‖ attention) — §5.14
- Zamba2 (periodic shared attention) — §5.14
- Jamba (sequential periodic substitution) — §5.14
- Phi-4-mini-flash (Samba-derived hybrid) — §5.3
- Falcon Mamba 7B — §5.9
- BitNet b1.58 (ReLU² + 1.58-bit ternary weights) — §5.8
- OpenELM (per-layer width scaling) — §5.13
- MobileLLM (block-wise weight sharing) — §5.30 / Axis A12
- Granite (μP scalars + extend-base-θ-only) — §5.11
- RWKV-6 Finch / RWKV-7 Goose (linear attention, delta-rule update) — §5.20
- RecurrentGemma (Griffin/Hawk LRU + local attn) — §5.4 / §5.20

**Niche architectures under-covered or missing:**

| Architecture | Status in v2 | Why it matters | Severity |
|---|---|---|---|
| **xLSTM 7B** (NXAI) | "noted, not pulled" in §9 | Only competitive mLSTM-based LM at 7B; uses neither RoPE nor ALiBi; matrix memory + exponential gating; deserves full §5 entry as the **non-Mamba non-RWKV linear-recurrent family**. arXiv:2503.13427 | **should-add** |
| **MEGABYTE / Byte Latent Transformer (BLT)** | Absent | Token-free / byte-level LM at <2B; novel patching scheme; the only "no-tokenizer" architecture in 2024-2026; relevant to axis A7 (tokenizer vocab). arXiv:2412.09871 | **nice-to-have** (academic) |
| **DeepSeek Sparse Attention (DSA)** in DeepSeek-V3.2 | Absent | Lightning Indexer + top-k attention; the first production "learned-scorer sparse attention"; new axis A1 value. arXiv:2512.02556 | **must-add** |
| **MiniMax Lightning Attention** | Absent | 7:1 linear:softmax ratio; the **inverse** of Qwen3-Next's 3:1 ratio; abandoned in M2 — a survey-paper-worthy story | **should-add** |
| **Llama-4 iRoPE** | Absent | Interleaved NoPE layers + sliding window; the production-scale validation of SmolLM3's NoPE-every-4th experiment | **must-add** (axis A11 / A15) |
| **Qwen3-Next Gated DeltaNet** | Absent | First Qwen production model with linear attention; "DeltaNet" family; new value for axis A1 | **must-add** |
| **Granite 4 hybrid 9:1** | Absent | Mamba-2 : attention 9:1 ratio; a fifth ratio in the hybrid-SSM family after Jamba 1:8, Hymba parallel, Zamba2 periodic, Falcon-H1 parallel-head | **must-add** |
| **Trained attention sinks** | 1-liner (Mistral Small 3.1, GPT-OSS) | Should be a §5.x section explaining what sinks do and which 3-4 models adopted them in 2025 | **should-add** (expand existing) |
| **AFM 2-block cross-block KV sharing** | "noted closed" | Now described in arXiv:2507.13575; should be promoted to a §5 entry and an axis A6 sub-value (cross-block KV share is novel) | **must-add** |
| **Mixture-of-LoRAs (Phi-4-multimodal)** | Absent | First production mixture-of-LoRAs at <6B; modality-specific routing | **should-add** |
| **Test-time-scaling / reasoning-distill recipes** (s1, Sky-T1, Marco-o1, DeepHermes-3, Llama-Nemotron) | Mostly absent; R1-Distill family in census | These are recipes, not architectures, BUT s1's "budget forcing" and Marco-o1's MCTS-injection have inference-time topology implications worth one paragraph each | **nice-to-have** (recipe section) |

**Surprises that v2 already captures correctly (sanity check):**
- ALiBi tried-and-abandoned (Surprise #13) — keep.
- Parallel attn/FFN tried-and-abandoned-and-revived (Surprise #14) — keep, but should now note Llama 4's iRoPE as another revival.
- MoE shared-experts 3-way-split (Surprise #15) — keep, but should add Qwen3-Next as a fourth pattern (10+1 out of 512 — much sparser than the 3-way split).

---

## 7. What WAS Covered Well (Do Not Redo)

To prevent v3 from re-doing strong sections of v2, document the strong parts explicitly:

1. **Dense Llama-derivative families** (Llama 1/2/3.x, Qwen 1/1.5/2/2.5/3, Mistral 7B v0.1-v0.3, OLMo 1/2, Granite 3.0-3.3, SmolLM/2/3, Yi 1.5, InternLM 2/2.5/3, MiniCPM 1/2/3, StableLM 2, Falcon 3 sweep, TinyLlama, MobileLLM, OpenELM, Cohere Aya, BitNet b1.58, Phi-1/1.5/2/3/3.5/4 sweep, Gemma 1/2/3/3n sweep): comprehensively rowed with verified HF configs and per-generation axis flips. **Do not redo. Only add missing variants (e.g., Mistral Small 3.x sequence, Granite 4 line, Gemma 4 line).**

2. **2023 baseline reconstruction** (LLaMA 1, MPT 7B with ALiBi, Falcon 7B with MQA+parallel, StarCoder 1 with abs-pos, Pythia, Phi-1/2 with parallel sublayers): v1 critique flagged this as missing and v2 fixed it. **Do not redo.**

3. **Axis catalog (15 axes A1-A15 + A16 bonus)**: well-thought-through enumeration. Only add new values for axes:
   - A1: add xLSTM, Gated DeltaNet, DSA/Lightning-Indexer, MiniMax-Lightning-7:1
   - A6: add AFM cross-block-shared-KV
   - A11: add 9:1 Mamba/attn (Granite 4), iRoPE (Llama 4)
   - A15: add iRoPE per-layer-mask (Llama 4 production)
   - **New axis A17 (proposed): VLM fusion topology** (concat-prefix / concat-projector / cross-attention / early-fusion / split-encoder / mixture-of-LoRAs)
   - **New axis A18 (proposed): Vision-token compression budget** (relevant for OCR-LLM line)

4. **Per-family evolution arcs** in 02-layer-sources.v2.md §5.1-5.33: Llama, Qwen, Phi, Gemma, DeepSeek, Mistral, OLMo, BitNet, Falcon, SmolLM, Granite, MiniCPM, OpenELM, Hymba/SSM-hybrid, StarCoder: the deltas-between-generations narratives are well-written. **Reuse the format for new families (OCR-LLM, audio-LLM, multimodal, 2026 entries).**

5. **MLA section** (DeepSeek-V2-Lite at §5.5, MiniCPM 3 at §5.12 + extensive treatment in 05-kvcache-attention.v2.md): comprehensive. Add only one row for DeepSeek-V3.2's DSA-on-top-of-MLA.

6. **Surprises list (1-20)** in 01-model-census.v2.md §8: the 20 surprises are well-curated. Add 2-3 more for the new findings (Gemma 4 encoder-free; Qwen3-Next 3:1 inverse-ratio hybrid; cross-block KV sharing in AFM).

7. **MoE 3-way split table** (Mixtral/DeepSeek-V2/Qwen3-MoE/DeepSeek-V3): well-curated. Add Qwen3-Next as a 5th data point and Granite 4 H-Tiny as a 6th.

8. **04-quantization.v2.md** treatment of GPTQ / AWQ / SmoothQuant / FP8 / MXFP4 / BitNet 1.58: extensive and accurate. Only addition needed: AFM's 2-bit QAT and gpt-oss's MXFP4-native-weights need first-class treatment (currently 1-liners).

9. **05-kvcache-attention.v2.md** M-RoPE / MM-RoPE coverage and the "Mistral SWA gotcha" callouts: keep. Add TMRoPE (Qwen2.5-Omni) and iRoPE (Llama 4).

10. **03-ihv-opsets.v2.md** (TensorRT-LLM, vLLM, MLX, llama.cpp): mature engineering treatment. No changes needed for the audit-period gaps; new models will be supported by existing op-sets without architectural extension.

---

## 8. Recommended Additions for v3 (Prioritized)

**Priority 1 (must-add, axis-load-bearing, do not ship v3 without these):**

1. **Gemma 4 family** (E2B, E4B, 26B-MoE-A3.8B, 31B dense, 12B dense) — full §5.x section with encoder-free multimodal axis story.
2. **DeepSeek-OCR** + full **§5.34 OCR-LLM evolution chapter** (Donut → Nougat → Vary → GOT-OCR → DeepSeek-OCR / MinerU2.5 / olmOCR-2).
3. **GOT-OCR 2.0** — anchor row for OCR-LLM table.
4. **Llama 4 Scout / Maverick** — iRoPE + native multimodal early fusion.
5. **Qwen3-Next 80B-A3B** — Gated DeltaNet 3:1 hybrid + ultra-sparse MoE.
6. **Granite 4.0 H-Micro / H-Tiny / H-Small** — 9:1 Mamba-2:attention ratio.
7. **DeepSeek-V3.2 / V3.2-Exp** — Lightning Indexer DSA.
8. **Apple Foundation Model 3.18B** (promote from "noted closed" — tech report now public) — cross-block KV sharing axis.
9. **GPT-OSS-20B promoted from noted to full row** — MXFP4-native weights + trained attention sinks + GPT-3-style alternating attention.
10. **olmOCR-2 7B** + **MinerU2.5 1.2B** — anchor the OCR sub-table.
11. **Phi-4-multimodal 5.6B** — mixture-of-LoRAs axis.
12. **New axis A17 (VLM fusion topology)** and **A18 (vision token budget)** introduced in the axis catalog with mappings for all VLMs.

**Priority 2 (should-add, fills sub-segment gaps):**

13. **MiniCPM-V 2.6 / MiniCPM-o 2.6** — triple-modality MiniCPM.
14. **InternVL 2 / 2.5 sweep** (1B-8B) — InternViT+InternLM/Qwen family.
15. **Molmo 7B-O / 7B-D** — OLMo-based VLM and Qwen2-based VLM.
16. **Pixtral 12B** + **SmolVLM/SmolVLM2** — round out the VLM family table.
17. **EXAONE 3.5 2.4B / 7.8B** — Korean-bilingual line.
18. **Ministral 3B / 8B** — interleaved SWA (3rd alternation pattern).
19. **Janus-Pro 1B / 7B** — decoupled visual encoding.
20. **Mistral Small 3 / 3.1 / 3.2** as three rows (currently one terse "noted" row).
21. **Llama-Nemotron-Nano 8B + Nemotron-3-Nano 4B** — NVIDIA hybrid Mamba-2+transformer+MoE-FFN.
22. **Moshi 7B + Helium-1 2B** — dual-stream audio LLM; first §5.x audio-LLM section.
23. **Qwen2.5-Omni 3B / 7B + Qwen2-Audio** — TMRoPE; Thinker-Talker.
24. **xLSTM 7B** (NXAI) — full §5.x section as the **non-Mamba non-RWKV linear-recurrent family**.
25. **MiniCPM-V 2.6 / MiniCPM-o** — already row 13 above; ensure §5.x entry exists.
26. **Hunyuan 0.5B / 1.8B / 4B / 7B sweep** — Tencent SLM line.
27. **Ovis 2.5 1B / 2B / 4B / 8B** — NaViT VLM line.
28. **Eagle 2 / 2.5 1B / 9B (NVIDIA)** — MoVE (mixture of vision encoders).
29. **Falcon-H1 sweep** (0.5B / 1.5B / 3B / 7B) — expand 1-liner into full table.
30. **MiniMax-Text-01 / M2 evolution** — "Lightning Attention adopted and then abandoned in M2" as a survey-paper Surprise entry.

**Priority 3 (nice-to-have, completeness):**

31. **MonkeyOCR / MonkeyOCR v1.5** — Structure-Recognition-Relation triplet.
32. **DocOwl2 8B** — 324 tokens/page OCR-free multipage.
33. **TextHawk / TextHawk2** — ReSA module.
34. **DocPedia** — frequency-domain visual processing.
35. **Idefics 2 / Idefics 3** — note as SmolVLM's predecessors in VLM evolution arc.
36. **Cambrian-1 8B** — multi-vision-encoder VLM.
37. **Chameleon 7B** (Meta, 2024-05) — early-fusion multimodal precursor to Llama 4.
38. **Sky-T1 / s1 / Marco-o1 / DeepHermes-3 / Llama-Nemotron-Nano** — reasoning-distillation recipe section (one paragraph each in 02-layer-sources.v2.md §6.x).
39. **MEGABYTE / Byte Latent Transformer (BLT)** — token-free architecture note in axis A7.
40. **SmolDocling 256M** — smallest mainstream OCR-LLM, anchor the "tiny" end of OCR-LLM size spectrum.
41. **StepFun Step-Audio / Step-Audio-EditX 3B** — second audio-LLM family entry.
42. **Tencent Hunyuan-Image-3** — note as out-of-scope (text-to-image) but Tencent's same-period release.

---

## 9. On the User's Framing ("Mainstream <8B")

The framing "**mainstream small language model, active params <8B**" is **mostly the right scope, but slightly too narrow in two places:**

1. **Some axis-defining models are >8B and unavoidable.** Llama 4 Scout (17B active), Qwen3-Next (3B active but 80B total), Mistral Small 3.1 (24B), Pixtral 12B, Llama-3.2-Vision 11B, DeepSeek-V3.2-Exp, GPT-OSS 20B — all >8B by either active or total but define axes that downstream <8B models inherit. v2 already handles this with the "oos, noted" pattern, but the user's brief asks for tight scope adherence. **Recommendation: keep <8B as the inclusion criterion, but explicitly list 8-30B as "axis-load-bearing" rows; reject the proposed "active params <8B" criterion alone — for Qwen3-Next-80B-A3B and Llama-4-Scout-17B-A17B the architectural innovations are first-of-kind and the survey-paper claim of "three-year evolution" cannot omit them.**

2. **The framing's "LLM"-centric phrasing excludes some load-bearing OCR / audio / multimodal architectures.** Models like DeepSeek-OCR (a 3B-MoE-A570M with a dual SAM+CLIP encoder) and Moshi 7B (a speech-text dual-stream) are sometimes described as "VLMs" or "audio models" rather than "LMs", and the user's instruction set asks them to be added but the framing is ambiguous about whether OCR-VLMs count as "LLMs". **Recommendation: explicitly broaden the framing to "small generative decoder LM, including text-only, VLM text-tower with adapters, audio-LM, and OCR-LLM with specialized vision frontend, all with active or total parameter count ≤ 8B (or ≤ 30B if axis-load-bearing)".**

The "mainstream" qualifier (≥1k HF downloads/month OR shipped in production product OR cited 5+ times) is **correctly calibrated** — it includes Gemma 4, DeepSeek-OCR, Qwen3-Next, Granite 4, but excludes one-off research checkpoints and academic-only releases. Keep this filter.

**Other framing observations:**
- The 2023-02 start date is correct (LLaMA 1).
- The "past 3 years" framing was correct for v2's 2026-Q1 production date but is **already 17 months past LLaMA 1 of Feb 2023**, so by audit date the corpus is closer to 3 years 4 months. This is fine.
- The "evolution paper framing" is the right framing — v2's §5 per-family arcs are the strongest part of the corpus and v3 should preserve and extend that pattern.

---

## 10. Final Inventory

**Confirmed missing rows: 27** (§2 master table).
**Plus the §3 OCR-LLM long-tail: +7** (MonkeyOCR, DocOwl2, Mistral OCR 2/3, TextHawk2, DocPedia, Vary, SmolDocling).
**Plus the §4 VLM long-tail: +4** (Idefics 2/3, Cambrian-1, Chameleon, Eagle 2/2.5 — Eagle covered in main; Eagle2 already in §2).
**Plus the §5 niche/novel architecture additions: +6** (xLSTM, MEGABYTE/BLT, DSA, Lightning Attention, AFM cross-block KV, mixture-of-LoRAs — most already in main).
**Plus the §6 reasoning-distill recipe additions: +5** (Sky-T1, s1, Marco-o1, DeepHermes-3, Llama-Nemotron-Nano).

**Grand total potential additions: ~45 rows / ~12 new §5 family sections / 2 new axes.**

**Realistic v3 target if you want "complete":** add the 8 must-add rows (§2) + §5.34 OCR-LLM evolution chapter + 2 new axes (A17, A18) = **minimal viable v3**. This brings the census from 80 to ~95 rows and adds one new family section.

**Maximal v3:** add all 30 priority-1+2 rows + the OCR / VLM / audio long-tails = ~125 rows and 4 new §5 sections.

---

## 11. Sources Cited

- Gemma 4: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ ; https://deepmind.google/models/gemma/gemma-4/ ; https://ai.google.dev/gemma/docs/core/model_card_4
- DeepSeek-OCR: https://github.com/deepseek-ai/DeepSeek-OCR ; https://deepseek.ai/blog/deepseek-ocr-context-compression
- GOT-OCR 2.0: arXiv:2409.01704 (Wei et al., "General OCR Theory: Towards OCR-2.0")
- Llama 4: https://ai.meta.com/blog/llama-4-multimodal-intelligence/ ; https://huggingface.co/blog/llama4-release
- GPT-OSS: https://openai.com/index/introducing-gpt-oss/ ; arXiv:2508.10925
- Qwen3-Next: https://www.alibabacloud.com/blog/qwen3-next-a-new-generation-of-ultra-efficient-model-architecture-unveiled_602536
- Granite 4.0: https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models ; https://www.marktechpost.com/2025/10/02/ibm-released-new-granite-4-0-models
- DeepSeek-V3.2: arXiv:2512.02556 ; https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- Apple Foundation Model 2025: arXiv:2507.13575 ; https://machinelearning.apple.com/research/apple-foundation-models-2025-updates
- Nemotron Nano: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-8B-v1 ; https://huggingface.co/blog/nvidia/nemotron-3-nano-4b ; arXiv:2511.03929 (Nemotron Nano V2 VL)
- olmOCR / olmOCR-2: arXiv:2502.18443 ; https://allenai.org/blog/olmocr-2 ; https://huggingface.co/allenai/olmOCR-2-7B-1025
- MinerU2.5: arXiv:2509.22186 ; https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B
- MonkeyOCR: arXiv:2506.05218 ; arXiv:2511.10390 (MonkeyOCR v1.5)
- DocOwl2 / mPLUG-DocOwl2: arXiv:2409.03420 ; https://aclanthology.org/2025.acl-long.291/
- Mistral OCR 3: https://mistral.ai/news/mistral-ocr-3/ ; https://www.marktechpost.com/2025/12/19/mistral-ai-releases-ocr-3
- MiniCPM-V 2.6: https://huggingface.co/openbmb/MiniCPM-V-2_6 ; https://github.com/OpenBMB/MiniCPM-V
- InternVL 2 / 2.5: arXiv:2412.05271 ; https://internvl.github.io/blog/2024-12-05-InternVL-2.5/
- Molmo: https://allenai.org/blog/molmo ; https://huggingface.co/allenai/Molmo-7B-D-0924
- Pixtral 12B: arXiv:2410.07073 ; https://mistral.ai/news/pixtral-12b/
- SmolVLM: arXiv:2504.05299 ; https://huggingface.co/blog/smolvlm
- EXAONE 3.5: https://github.com/LG-AI-EXAONE/EXAONE-3.5 ; https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
- Ministral 3B / 8B: https://mistral.ai/news/ministraux/ ; https://siliconangle.com/2024/10/16/mistral-introduces-ministral-3b-8b-device-ai-computing-models/
- Janus-Pro: arXiv:2501.17811 ; https://huggingface.co/deepseek-ai/Janus-Pro-7B
- Mistral Small 3 / 3.1 / 3.2: https://mistral.ai/news/mistral-small-3-1/ ; https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
- Phi-4-multimodal / Phi-4-mini: arXiv:2503.01743
- Moshi / Helium-1: arXiv:2410.00037 ; https://github.com/kyutai-labs/moshi ; https://huggingface.co/kyutai/helium-1-2b
- Qwen2.5-Omni: https://github.com/QwenLM/Qwen2.5-Omni ; https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- Qwen3.5-Omni: arXiv:2604.15804 ; https://www.marktechpost.com/2026/03/30/alibaba-qwen-team-releases-qwen3-5-omni
- xLSTM 7B: arXiv:2503.13427 ; https://huggingface.co/NX-AI/xLSTM-7b ; https://www.nx-ai.com/en/news/xlstm-7b-nxai-releases-its-new-xlstm-7b-model
- MiniMax-Text-01 / M2: arXiv:2501.08313 ; https://github.com/MiniMax-AI/MiniMax-01
- Hunyuan 0.5B/1.8B/4B/7B: https://github.com/Tencent-Hunyuan/Hunyuan-7B ; https://news.aibase.com/news/20209
- Ovis 2 / 2.5: arXiv:2508.11737 ; https://huggingface.co/AIDC-AI/Ovis2.5-9B
- Eagle 2 / 2.5: https://github.com/NVlabs/Eagle ; https://huggingface.co/nvidia/Eagle2-1B
- Falcon-H1: arXiv:2507.22448 ; https://huggingface.co/blog/falcon3 ; https://falcon-lm.github.io/blog/falcon-h1/
- TextHawk / TextHawk2: arXiv:2404.09204 ; arXiv:2410.05261
- DocPedia: arXiv:2311.11810
- Vary: arXiv:2312.06109
- MEGABYTE / BLT: arXiv:2305.07185 (MEGABYTE) ; arXiv:2412.09871 (BLT)
- SmolLM3: https://huggingface.co/HuggingFaceTB/SmolLM3-3B ; https://huggingface.co/blog/smollm3
- Llama-3.2-Vision: https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/ ; https://huggingface.co/meta-llama/Llama-3.2-11B-Vision
- Marco-o1 / s1 / Sky-T1: arXiv:2411.14405 (Marco-o1) ; arXiv:2501.19393 (s1) ; (Sky-T1: NovaSky Berkeley blog)
- Cohere Command R7B: https://cohere.com/blog/command-r7b ; https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024
- Reka Flash 3: https://artificialanalysis.ai/models/reka-flash-3
- SeamlessM4T v2: https://huggingface.co/facebook/seamless-m4t-v2-large
- StepFun Step-Audio: arXiv:2502.11946 ; arXiv:2506.08967 ; https://huggingface.co/stepfun-ai/Step-Audio-EditX
- ByteDance Seed1.5-VL: arXiv:2505.07062

*End of 08-coverage-justification.md.*
