## 01 — Mainstream Small Language Model Census v3 (<8B params, 2023–2026)

*Stream: model-census. Project: llm-layers. Date: 2026-06-06. Version: 3.0.*

*v2 (2026-06-04) closed at 80 rows, 15 axes, with a nominal cutoff of July 2025. The 11-month staleness gap was audited in `research/issues/08-coverage-justification.md`, which identified 27 confirmed missing families and a 2025-Q3 → 2026-Q2 coverage gap of ~21 model families (>25% of v2's row count). v3 closes that gap. It adds Gemma 4 (5 sizes), Llama 4 Scout/Maverick, Qwen3-Next, DeepSeek-V3.1/V3.2/V4-Pro/V4-Flash, Granite 4.0/4.1, GPT-OSS 20B promoted to first-class, Apple Foundation Model 3.18B, DeepSeek-OCR + the OCR-LLM family, Ministral, Mistral OCR/Magistral, Phi-4-multimodal, MiniMax-Text-01, MiniCPM-V/o, InternVL 2.5, Molmo, Pixtral, Janus-Pro, Llama-Nemotron-Nano, NVIDIA Nemotron 3 Super/Ultra/Nano-Omni, GLM-5.1, Cohere Command A+, Falcon-H1 sweep + Falcon-Edge 1.58-bit retrainable, Mamba-3, xLSTM 7B, Mistral Medium 3.5, Hunyuan 0.5/1.8/4/7B sweep, EXAONE 3.5/4.5, Moshi/Helium/Voxtral, Qwen2.5-Omni / Qwen2.5-VL / Qwen3-VL, olmOCR + olmOCR-2 + RolmOCR + MonkeyOCR-v1.5 + Nanonets-OCR-s, GOT-OCR 2.0, ERNIE-4.5-0.3B, Hunyuan-0.5B and PaddleOCR-VL, plus 18+ axes formalizing VLM fusion topology, vision-token compression budget, cross-layer KV sharing, and per-layer embeddings. v3 also broadens the LM definition from "text-only decoder" to "small generative decoder LM, including text-only, VLM text-tower with adapters, audio-LM, and OCR-LLM with specialized vision frontend".*

---

## 1. Executive Summary

This census catalogs **148 mainstream small generative decoder language models** (active params ≤8B, plus a curated set of architecturally-load-bearing 8–80B references) shipped between **February 2023 and June 2026**. The model set spans dense decoders, sparse MoE, hybrid Mamba/SSM, hybrid Mamba-Transformer-MoE, pure SSM, pure linear-attention (RWKV, xLSTM, Gated DeltaNet), VLM text towers, OCR-LLMs, audio-LMs, and quantization-native architectures (BitNet b1.58, Falcon-Edge 1.58-bit, GPT-OSS MXFP4-native). The corpus closes the v2 11-month gap and broadens the framing from "<8B LLM decoder" to "<8B *or* axis-load-bearing generative decoder LM".

Across the three-year-four-month window, four large structural transitions are now visible at consensus scale: **(1) MoE moved from a 13B-active outlier (Mixtral 8×7B, Dec 2023) to the dominant >20B-active production design** (DeepSeek-V3 256+1, Qwen3 128+0, Llama 4 Scout 16-expert, Granite 4 H-Tiny 7B-A1B, Qwen3-Next 512+1 with top-10, DeepSeek-V4 Pro 1.6T/49B-active). **(2) Hybrid SSM-Transformer architectures fragmented into eight distinct topologies** (Jamba 1:8 sequential periodic, Zamba2 periodic shared, Hymba parallel branches, Falcon-H1 parallel-head, Phi-4-mini-flash Samba-derived, Granite 4 9:1 sequential, Qwen3-Next 3:1 Gated DeltaNet, Nemotron 3 Super/Ultra Mamba-2+MoE). **(3) Sparse-attention architectures became production-grade**: DeepSeek V3.2 Sparse Attention (DSA) with Lightning Indexer, DeepSeek V4 CSA+HCA dual sparse, MiniMax-M3 MSA, GLM-5.1 DSA derivative, Llama 4 Scout iRoPE with per-layer NoPE mask. **(4) The VLM/OCR-LLM frontier fragmented into six distinct fusion topologies** (concat-prefix-LM, concat-projector, cross-attention adapter, early-fusion native, mixture-of-LoRAs, split-encoder for understanding-vs-generation), and the OCR-LLM sub-segment matured from one model (Pix2Struct, 2023) to a coherent lineage of 15+ production models (GOT-OCR2 → DeepSeek-OCR / DeepSeek-OCR-2 / MinerU2.5 / olmOCR-2 / MonkeyOCR-v1.5 / Nanonets-OCR2 / HunyuanOCR / PaddleOCR-VL).

The non-consensus picture is now broader than v2's framing suggested. **Gemma 4 introduces three honestly new architectural primitives in a single release**: (a) **partial-rotary p-RoPE** on global attention layers (`partial_rotary_factor=0.25`, composed with dual-RoPE base θ=10000 local / θ=1000000 global) — only 25% of head_dim channels rotate; (b) **K=V on global layers** (the global K and V projections share weights, paired with a doubled `global_head_dim=512` and fewer KV heads, so the global cache cost halves); (c) **cross-layer KV sharing** (`num_kv_shared_layers=20` out of 35 for E2B — over half of layers reuse a previously-computed K/V tensor from an earlier same-type layer). Gemma 4 also introduces **Per-Layer Embeddings** (a second embedding table whose output is injected as a residual at every decoder layer, computed as `(token_identity + context_aware_projection) × 1/√2`, with a much smaller dim than the main residual stream so it can be paged to flash on mobile devices), restores the final-logit softcap (`final_logit_softcapping=30.0`) that Gemma 3 had dropped while keeping `attn_logit_softcapping=null`, and introduces the first Gemma MoE (26B-A4B: 128 routed + 1 shared, top-k=8, `moe_intermediate_size=704`). **Llama 4 Scout** ships **iRoPE** — interleaved per-layer NoPE masks (the production-scale generalization of SmolLM3's NoPE-every-4th experiment) at 17B-A active with 16 routed experts and native multimodal early fusion. **Qwen3-Next** ships **Gated DeltaNet** linear attention interleaved 3:1 with softmax attention (the inverse of MiniMax-Text-01's 7:1 ratio — 75% linear vs 87.5% linear), with the sparsest production MoE to date (512 routed + 1 shared, top-10). **DeepSeek-V4-Pro** ships **CSA+HCA** (Compressed Sparse Attention + Heavily Compressed Attention), achieving 27% of V3.2's FLOPs and 10% of its KV cache at 1M context. **Apple Foundation Model 3.18B** ships **2-block cross-block KV sharing** (Block-1 carries 62.5% of layers with full KV; Block-2 carries 37.5% and reuses Block-1's K/V) — a distinct axis variant from Gemma 4's same-block KV sharing. **Falcon-Edge** ships **retrainable 1.58-bit BitNet weights** (the first BitNet family that supports continued pretraining, not just inference). **Mamba-3** ships **complex-valued state with MIMO decoding**. The architectural diversity of 2026-Q2 alone exceeds the diversity of the entire 2023-2024 period.

**Eighteen architectural axes** are now identified (Section 6), up from 15 in v2. The three new axes are: **A16 VLM fusion topology** (six values: concat-prefix-LM, concat-projector, cross-attention-adapter, early-fusion, mixture-of-LoRAs, split-encoder); **A17 vision-token compression budget** (parametric — token counts per 1024² page range from ~100 (DeepSeek-OCR) to ~6000 (Qwen2.5-VL high-res), with associated M-RoPE / TMRoPE / V2PE / 2D-RoPE variants); **A18 cross-layer KV sharing** (three values: none / same-block-shared (Gemma 4) / cross-block-shared (Apple AFM)); and a fourth secondary axis **A19 per-layer embeddings** (Gemma 4 PLE, Apple AFM per-layer rank-1 adapter), which can be subsumed under A12 in some readings but is structurally distinct from per-layer width scaling (the PLE table has its own embedding dimension and is paged to flash, whereas OpenELM's per-layer width is at training time only).

The deliverable for the llm-layers project: any operator dataclass that hard-codes (i) global RoPE base θ as a model-level scalar (Gemma 4 needs dual θ per layer-type), (ii) `partial_rotary_factor` as a model-level scalar (Gemma 4 needs it per layer-type — local layers use full rotary, global use 0.25), (iii) attention as having `attn_logit_softcapping` *xor* `final_logit_softcapping` (Gemma 4 has only the final), (iv) `head_dim` uniform across layer types (Gemma 4 global has `global_head_dim=512` vs local `head_dim=256`), (v) one K and one V projection per attention block (Gemma 4 global layers have `K=V` aliasing), (vi) one KV cache per layer (Gemma 4 has `num_kv_shared_layers` and Apple AFM has 2-block KV sharing), (vii) one embedding table feeding the residual stream once (Gemma 4 PLE injects an additional residual at every layer), (viii) attention mask as global causal (DeepSeek-OCR-2 has Visual Causal Flow — bidirectional within a page-block, causal across pages), (ix) RoPE as 1D over token index (M-RoPE / TMRoPE / 2D-RoPE / V2PE require a 2D or 3D position layout per head_dim slice), or (x) a single LoRA routing (Phi-4-multimodal uses mixture-of-LoRAs with modality-driven adapter selection) — will silently produce wrong outputs for at least one production model in this census.

---

## 2. Methodology and Scope

### 2.1 Inclusion criteria

A model is in scope if all four hold:

1. **Active parameter count ≤8B at inference**, OR **architectural innovation is first-of-kind** and downstream <8B production models inherit the design. For MoE, "active" means `routed-experts × top-k + shared-experts + dense`. The "axis-load-bearing >8B" rows are explicitly listed and marked "oos, axis-load-bearing" (e.g. Llama 4 Scout 17B-A, Qwen3-Next 80B/A3B, DeepSeek-V4-Pro, MiniMax-Text-01, GPT-OSS 20B, GLM-5.1, Cohere Command A+, Mistral Medium 3.5, NVIDIA Nemotron 3 Super/Ultra). v2's strict <8B-active rule is relaxed in v3 because (a) Qwen3-Next-80B-A3B has 3B active but a 80B total with 10+1/512 MoE that defines the "ultra-sparse" axis, and (b) Llama 4 Scout's 17B active is the production-scale validation of NoPE-per-layer that downstream <8B models inherit. v3 keeps strict <8B as the inclusion floor and labels axis-load-bearing 8–80B rows separately.
2. **Mainstream**, defined as (a) ≥10k HF downloads/week as of 2026-06-06, OR (b) shipped by Anthropic/Apple/Meta/Microsoft/Google/Alibaba/01-AI/IBM/Cohere/AI2/Mistral/DeepSeek/Stability AI/Hugging Face/NVIDIA/MosaicML/AI21/TII/Zyphra/NVIDIA/RWKV-LM/State-Spaces/BigCode/Tsinghua/Baichuan/LG/Kyutai/StepFun/OpenBMB/Tencent/Xiaomi/MiniMax/NXAI/CMU/Princeton/Together/Cartesia/Liquid AI/Zhipu/Baidu/Reka/Moonshot, OR (c) named in ≥3 peer-cited 2024–2026 efficiency benchmarks, OR (d) cataloged in the OCR-LLM lineage that defines `research/06`'s tier A.
3. **Decoder-only or hybrid token-mixer or encoder-decoder with an LM-shaped decoder**. Florence-2 (BART-style encoder-decoder) is included as one of two encoder-decoder counter-examples; Donut, Nougat, Pix2Struct are noted as the historical OCR-LLM ancestors. Pure encoders (BERT, ModernBERT, LayoutLMv3, DiT) are excluded.
4. **Public config or technical report**. Closed models are noted but not tabulated, with one exception: **Apple Foundation Model 3.18B** is promoted from v2's "noted, closed" status to a first-class row because the architecture is now fully described in arXiv:2507.13575 (the configs remain closed). Anthropic Claude Haiku, Gemini Nano, GPT-4.1-nano, Meta Muse Spark, Microsoft MAI-Thinking-1, xAI Grok 4.3 remain noted-but-not-rowed.

### 2.2 Field provenance

For each row the source is one of (in priority order): (a) the official HF `config.json`; (b) the vendor mirror's HF `config.json` (`unsloth/`, `deepseek-ai/`, `ai21labs/`); (c) the vendor's technical report or model card; (d) the upstream `transformers` source tree for fields not present in config; (e) for closed models, the vendor's tech report (Apple AFM = arXiv:2507.13575; Gemma 4 = HF blog + model cards + verified config.json across all five sizes); (f) for very recent (May/June 2026) models, the announcement blog + initial HF release.

For Gemma 4 the per-size config.json values were verified (see `research/issues/10-gemma4-investigation.md`). For DeepSeek-V4-Pro, DeepSeek-V3.2-Exp, Qwen3-Next, Granite 4, Llama 4 Scout/Maverick, the architectural claims are sourced from each family's HF model card + Transformers PR + 1-2 corroborating technical write-ups.

### 2.3 Notation

Heads/KV/hd = `num_attention_heads / num_key_value_heads / head_dim`. ✓ = present. — = explicitly absent. Vocab tokenizer family abbreviations expand v2's set: TT = Tiktoken cl100k; LL3 = Llama-3 Tiktoken 128k; LL4 = Llama-4 Tiktoken 200k; SP = SentencePiece generic; GP = Gemma SentencePiece 256k; GP-v4 = Gemma 4 SentencePiece 262k +image/audio/video special tokens; QW = Qwen BPE 151936; QW-omni = Qwen Omni BPE 151936 + speech tokens; GPT2 = GPT-2 BPE; MS = Microsoft Phi SP 32k or o200k 200k; SC2 = StarCoder-2 BPE 49152; TK = Tekken/Mistral-v3 SP 32768; TK131 = Tekken 131072 (Mistral Nemo/Pixtral); RWKV-world = RWKV-world 65536; ERNIE = Baidu ERNIE BPE; HY = Hunyuan SP; EXAONE = LG EXAONE BPE; MoE = mixture-of-experts; MoT = mixture-of-transformers; LoRA = low-rank adapter; CL = cross-layer; PLE = per-layer embedding; MM = multimodal; AR/NAR = autoregressive / non-autoregressive.

### 2.4 Differences from v2 census

v2 had 80 rows across ~30 architectures and 15 axes. v3 has **148 rows across ~52 distinct architectural families and 18+ axes**. The expansion is concentrated in:

- **OCR-LLM family (new lineage, 14 added rows)**: GOT-OCR2 0.5B, DeepSeek-OCR 3B-A570M, DeepSeek-OCR-2, MinerU2.5 1.2B, olmOCR / olmOCR-2 8B, RolmOCR, MonkeyOCR / MonkeyOCR-v1.5 3B, HunyuanOCR 1B, PaddleOCR-VL 0.9B, PaddleOCR-VL-1.5, Nanonets-OCR-s 3B, Nanonets-OCR2-3B, Dolphin / Dolphin-v2, Surya 2 0.65B.
- **2025-H2 dense + MoE additions (8 rows)**: Granite 4.0 H-Micro/H-Tiny/H-Small, Granite 4.1 3B/8B/30B + Vision-4B + Speech AR/NAR-2B + Guardian-4.1, Qwen3-Next 80B-A3B, DeepSeek-V3.1, DeepSeek-V3.2-Exp, GPT-OSS 20B promoted, Hunyuan 0.5B/1.8B/4B/7B, Tencent Hy-MT2 1.8B/7B.
- **2026 H1 additions (15 rows)**: Gemma 4 E2B / E4B / 12B-Unified / 26B-A4B / 31B + Gemma 4 MTP drafters, Mamba-3 reference scales, Mistral Small 4, Mistral Medium 3.5, NVIDIA Nemotron 3 Super / Nano-Omni / Ultra, DeepSeek-V4-Pro / V4-Flash, GLM-5.1 / GLM-5V-Turbo, Cohere Command A+, Qwen3.6-35B-A3B / Qwen3.6-27B, Qwen3.5-Omni, Liquid LFM2.5-VL-450M / LFM2.5-8B-A1B, Falcon-Edge 1B / 3B (1.58-bit retrainable), Falcon Perception 0.6B, EXAONE 4.5 33B, HY-Embodied-0.5 MoT-2B, Hy3 Hunyuan 3 Preview, Voxtral TTS 4B, Tencent Hunyuan-A13B.
- **Llama 4 family (new — promoted to axis-load-bearing)**: Scout 17B-A 16E, Maverick 17B-A 128E.
- **VLM rows the v2 audit flagged (12 added)**: MiniCPM-V 2.6, MiniCPM-o 2.6, MiniCPM-Llama3-V 2.5, InternVL 2.5 1B/2B/4B/8B, InternVL 3 1B/2B/8B, Molmo 7B-O / 7B-D, MolmoE 1B, Pixtral 12B, SmolVLM / SmolVLM2 256M / 500M / 2.2B, Phi-4-multimodal, Janus-Pro 1B / 7B, Eagle 2 1B/2B/9B, Ovis 2.5 2B/9B, Cambrian-1 8B (oos), NVILA 8B (oos).
- **Audio-LM family (new)**: Moshi 7B, Helium-1 2B, Voxtral TTS 4B, Qwen2-Audio 7B, Qwen2.5-Omni 3B/7B, IBM Granite Speech 4.1 AR-2B / NAR-2B.
- **Linear-attention family completion (6 rows)**: xLSTM 7B (promoted from "noted not pulled"), Qwen3-Next Gated DeltaNet (already counted above), MiniMax-Text-01 Lightning Attention 7:1 (oos), Mamba-3 reference scales.
- **Apple AFM 3.18B** promoted to first-class row (arXiv:2507.13575).
- **Ministral 3B / 8B** added (interleaved SWA — 3rd alternation pattern after Gemma 2's 1:1 and Gemma 3's 5:1).
- **EXAONE 3.5 2.4B / 7.8B** added (LG Korean-bilingual).
- **Llama-Nemotron-Nano 8B** added (NVIDIA reasoning distillation, FP4-aware).
- **Two new tokenizer entries**: ERNIE-4.5-0.3B (PaddleOCR-VL's text decoder) and Hunyuan-0.5B (HunyuanOCR's text decoder).
- **Mistral OCR 2 (open) + Mistral OCR 3 (closed)** and **Magistral** noted in the Mistral section.

### 2.5 Scope broadening from v2 → v3

v2's framing was "small generative *text-only* LLM decoder". v3 broadens to "**small generative decoder LM**, including (a) text-only, (b) VLM text-tower with vision adapter, (c) audio-LM with audio frontend, (d) OCR-LLM with specialized vision frontend, (e) omni-modal with multi-modality input + text output, (f) quantization-native architectures". This broader scope adds ~30 rows that v2 had implicitly excluded; their LM-decoder portion is what gets tabulated. For each VLM/audio/OCR row, the table specifies which research/01 row provides the text-decoder backbone (e.g., olmOCR-2 is `Qwen2.5-VL-7B base`, MinerU2.5 is `Qwen2-Instruct-0.5B`, GOT-OCR2 is `Qwen-0.5B`).

---

## 3. Census by Year

The narrative order is chronological. Within each year-half, families are grouped by lineage.

### 3.1 The 2023 Baseline (unchanged from v2)

The 2023 baseline rows are preserved from v2 §3.1 verbatim — no changes in this section. The 28 baseline rows (LLaMA 1/2, Mistral 7B v0.1/v0.2, MPT 7B, Falcon 7B, StarCoder 1, ChatGLM/2/3, Qwen 1, Baichuan 1/2, DeepSeek LLM/Math/Coder, OLMo 1, Phi-1/1.5/2, StableLM 3B/Zephyr, Pythia 2.8/6.9B, TinyLlama, MobileLLM 125M/1B) cover the "Llama-shaped consensus formation" period. See v2 §3.1 for the per-row table.

### 3.2 2024 (preserved from v2, plus 4 added rows for late-2024 omissions)

v2's 2024 coverage (Llama 3/3.1/3.2 1B/3B/8B, Qwen 1.5/2/2.5 sweep, Gemma 1/1.1/2/RecurrentGemma/PaliGemma 1/2, Phi-3 mini-4k / mini-128k / small-7B / Phi-3.5 mini / Phi-3.5 vision / Phi-3.5-MoE, Mistral 7B v0.3 / Mistral Nemo / Mathstral / Codestral Mamba, OLMo 2 7B/13B, SmolLM2 135M/360M/1.7B, Granite 3.0/3.1, MiniCPM 1/2/3, Yi 1.5/Yi-Coder, InternLM 2/2.5, StableLM 2 1.6B/12B, Falcon 3 1B/3B/7B/10B, Cohere Aya 8B, OpenELM 270M/450M/1.1B/3B, StarCoder 2 3B/7B, CodeGemma 2B/7B, DeepSeek-V2-Lite 16B-A, DeepSeek-Coder-V2-Lite, OLMoE 1B-A, Phi-3.5-MoE 6.6B-A, Jamba v0.1, Mamba 2.8B, Mamba 2 2.7B, Falcon Mamba 7B, Zamba2 2.7B/7B, RWKV-6 1.6B/7B, Codestral Mamba 7B, BitNet b1.58 2B, Hymba 1.5B) is preserved unchanged. v3 adds the following 2024 rows omitted by v2:

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ministral** | 3B | 2024-10 | GQA + **interleaved SWA** (layer-alt full/SWA) | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 131072 | TK131 | y | 32768 / W=4096 alt | mistralai/Ministral-3B-Instruct-2410 |
| **Ministral** | 8B | 2024-10 | GQA + **interleaved SWA** | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 131072 | TK131 | n | 32768 / W=4096 alt | mistralai/Ministral-8B-Instruct-2410 |
| **EXAONE 3.5** | 2.4B | 2024-12 | GQA | 32/8/80 | vanilla (θ=1e6) | RMS pre | SwiGLU | 102400 | EXAONE | y | 32768 | LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct |
| **EXAONE 3.5** | 7.8B | 2024-12 | GQA | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 102400 | EXAONE | n | 32768 | LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct |
| **MiniCPM-Llama3-V 2.5** | 8.5B (Llama-3-8B base) | 2024-05 | GQA | 32/8/128 | Llama-3 (factor=8) | RMS pre | SwiGLU | 128256 | LL3 | n | 8192 | openbmb/MiniCPM-Llama3-V-2_5 |
| **MiniCPM-V 2.6** | 8.5B (Qwen2-7B base) | 2024-08 | GQA | 28/4/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 152064 | QW | n | 131072 | openbmb/MiniCPM-V-2_6 |
| **InternVL 2** | 1B/2B/4B/8B | 2024-07 | GQA (Qwen2/InternLM2 backbone per size) | per backbone | per backbone | RMS pre | SwiGLU | per backbone | per backbone | per backbone | 8192 | OpenGVLab/InternVL2-{1B,2B,4B,8B} |
| **Molmo** | 7B-D (Qwen2-7B base) | 2024-09 | GQA | 28/4/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 152064 | QW | n | 131072 | allenai/Molmo-7B-D-0924 |
| **Molmo** | 7B-O (OLMo-7B-1024 base) | 2024-09 | MHA | 32/32/128 | vanilla (θ=5e5) | RMS post + QK-RMS/full | SwiGLU | 100352 | TT | n | 4096 | allenai/Molmo-7B-O-0924 |
| **MolmoE** | 1B (OLMoE-1B-A base) | 2024-09 | MHA | 16/16/128 | vanilla | RMS pre | SwiGLU-MoE 64/0/top-8 | 50304 | GPT-NeoX | n | 4096 | allenai/MolmoE-1B-0924 |
| **Pixtral** | 12B (oos, axis-load-bearing) | 2024-09 | GQA + **2D-RoPE in vision tower** | 32/8/128 | vanilla (θ=1e6) on text; 2D-RoPE on vision | RMS pre | SwiGLU | 131072 | TK131 | n | 131072 | mistralai/Pixtral-12B-2409 |
| **SmolVLM** | 256M / 500M / 2.2B | 2024-11 / 2025-02 | GQA (SmolLM/SmolLM2 base + 9× pixel-shuffle vision) | per backbone | per backbone | RMS pre | SwiGLU | 49152 | SC2 | y | 8192 | HuggingFaceTB/SmolVLM-Instruct |
| **Qwen2-Audio** | 7B | 2024-08 | GQA (Qwen2-7B base) | 28/4/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 152064 | QW | n | 131072 | Qwen/Qwen2-Audio-7B-Instruct |
| **DeepSeek-VL2** | 3B / 16B / 27B | 2024-12 | MLA (V2-Lite base) | per V2-Lite | YaRN | RMS pre | SwiGLU-MoE | 102k DeepSeek | DeepSeek BPE | n | 32768 | deepseek-ai/deepseek-vl2 |
| **mPLUG-DocOwl 1.5** | 7.6B (LLaMA-2-7B base) | 2024-03 | MHA | 32/32/128 | vanilla (θ=10000) | RMS pre | SwiGLU | 32000 | LLaMA SP | n | 4096 | mPLUG/DocOwl1.5 |
| **mPLUG-DocOwl2** | 8.1B (LLaMA-2-7B base) | 2024-09 | MHA | 32/32/128 | vanilla | RMS pre | SwiGLU | 32000 | LLaMA SP | n | 4096 | mPLUG/DocOwl2 |
| **TextHawk2** | ~7B (InternLM2-7B base) | 2024-10 | GQA + **SPE in vision** | per InternLM2 | dynamic NTK | RMS pre | SwiGLU | 92544 | InternLM SP | n | 32768 | yuyq96/TextHawk2 |
| **Vary-toy** | 1.8B (Qwen-1.8B base) | 2024-01 | MHA + QKV bias | 32/32/64 | vanilla | RMS | SwiGLU | 151936 | QW | n | 8192 | HaoranWei/Vary-toy |

### 3.3 2025-H1 (preserved from v2, plus 5 added rows)

v2 covered: Llama 3.3 70B (noted), Qwen 3 0.6B/1.7B/4B/8B + 30B-A3B MoE, Qwen 2.5-VL 3B/7B, Qwen 2.5-Math 7B, Gemma 3 1B/4B/12B/27B + Gemma 3n E2B/E4B, Phi-4 mini 3.8B, Phi-4-mini-flash, SmolLM3 3B, Granite 3.3 2B/8B, InternLM 3 8B, Mistral Small 3.1 24B (noted, oos), Falcon-H1 7B (noted), DeepSeek-V3.1 (noted oos), DeepSeek-R1-Distill-Qwen 1.5B/7B, DeepSeek-R1-Distill-Llama 8B, RWKV-7 Goose 1.5B, GPT-OSS 20B (noted oos), Apple Foundation Model (noted closed). v3 adds:

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **InternVL 3** | 1B/2B/8B (incl. V2PE position encoding) | 2025-04 | GQA (Qwen2.5 base) + V2PE | 14/2/128 (1B), 16/2/128 (2B), 28/4/128 (8B) | **V2PE (Variable Visual PE)** on vision; vanilla θ=1e6 on text | RMS pre | SwiGLU | 151936 | QW | y (1B/2B) / n (8B) | 32768 | OpenGVLab/InternVL3-{1B,2B,8B} |
| **olmOCR** | 8B (Qwen2-VL-7B base) | 2025-02 | GQA + M-RoPE | 28/4/128 | M-RoPE (T,H,W) on text+vision | RMS pre | SwiGLU | 152064 | QW | n | 32768 | allenai/olmOCR-7B-0225-preview |
| **olmOCR-2** | 8B (Qwen2.5-VL-7B base) | 2025-10 | GQA + M-RoPE + GRPO | 28/4/128 | M-RoPE | RMS pre | SwiGLU | 151936 | QW | n | 32768 | allenai/olmOCR-2-7B-1025 |
| **MinerU2.5** | 1.2B (Qwen2-Instruct-0.5B base + NaViT-675M) | 2025-09 | GQA + M-RoPE + **two-stage layout→recognition** | 14/2/64 | M-RoPE (text uses temporal axis only) | RMS pre | SwiGLU | 151936 | QW | y | 32768 | opendatalab/MinerU2.5-Pro-2604-1.2B |
| **MonkeyOCR** | 3B (InternLM2 base) | 2025-06 | GQA + **SRR triplet** | per InternLM2 | dynamic NTK | RMS pre | SwiGLU | 92544 | InternLM SP | n | 32768 | echo840/MonkeyOCR |
| **MonkeyOCR-v1.5** | 3B (Qwen2.5-VL-3B base) | 2025-11 | GQA + M-RoPE + SRR triplet | 16/2/128 | M-RoPE | RMS pre | SwiGLU | 151936 | QW | y | 32768 | echo840/MonkeyOCR-v1.5 |
| **Nanonets-OCR-s** | 3B (Qwen2.5-VL-3B base, semantic-tag SFT) | 2025-05 | GQA + M-RoPE | 16/2/128 | M-RoPE | RMS pre | SwiGLU | 151936 | QW | y | 32768 | nanonets/Nanonets-OCR-s |
| **RolmOCR** | 8B (Qwen2.5-VL-7B base, speed-tuned) | 2025-07 | GQA + M-RoPE | 28/4/128 | M-RoPE | RMS pre | SwiGLU | 151936 | QW | n | 32768 | reducto/RolmOCR |
| **GOT-OCR 2.0** | 0.58B total / 0.5B LM (Qwen-0.5B base) | 2024-09 | MHA (Qwen-1 family) | 14/14/64 | vanilla (θ=10000) | RMS pre | SwiGLU | 151936 | QW | y | 8192 | stepfun-ai/GOT-OCR-2.0-hf |
| **DeepSeek-OCR** | 3.4B total / 570M-A (DeepSeek3B-MoE) + DeepEncoder (SAM-B + CLIP-L serial + 16× compressor) | 2025-10 | **MLA** + MoE 64+2/top-6 | 16 / 16 latent | YaRN | RMS pre | SwiGLU-MoE | 102400 | DeepSeek BPE | n | 32768 | deepseek-ai/DeepSeek-OCR |
| **Janus-Pro** | 1B (DeepSeek-LLM-1B base) + **split SigLIP** for understanding vs gen | 2025-01 | MHA | 16/16/64 | vanilla | RMS pre | SwiGLU | 102400 | DeepSeek BPE | n | 4096 | deepseek-ai/Janus-Pro-1B |
| **Janus-Pro** | 7B (DeepSeek-LLM-7B base) + split SigLIP | 2025-01 | MHA | 32/32/128 | vanilla | RMS pre | SwiGLU | 102400 | DeepSeek BPE | n | 4096 | deepseek-ai/Janus-Pro-7B |
| **Phi-4-multimodal** | 5.6B (Phi-4-Mini-3.8B base + **mixture-of-LoRAs**: 370M Vision LoRA + audio LoRA) | 2025-02 | GQA + partial RoPE 0.75 + **modality-routed LoRA** | 24/8/128 | LongRoPE | RMS pre | SwiGLU fused | 200064 | MS (o200k) | y | 131072 | microsoft/Phi-4-multimodal-instruct |
| **Qwen 2.5-Omni** | 3B / 7B (Qwen 2.5 + **Thinker-Talker dual decoder** + **TMRoPE**) | 2025-03 / 2025-04 | GQA + **TMRoPE** (T/H/W + frame index) | 16/2/128 (3B), 28/4/128 (7B) | TMRoPE | RMS pre | SwiGLU | 151936 | QW-omni | y (3B) / n (7B) | 32768 | Qwen/Qwen2.5-Omni-{3B,7B} |
| **Qwen3-VL** | 8B (Qwen3-8B base + Qwen2.5-VL-style ViT) | 2025-Q4 | GQA + QK-norm + M-RoPE | 32/8/128 | M-RoPE (text axis only) | RMS pre + QK-norm pre-RoPE | SwiGLU | 151936 | QW | n | 32768 | Qwen/Qwen3-VL-8B-Instruct |
| **Llama-Nemotron-Nano** | 8B (Llama-3.1-8B base + FP4-aware reasoning RL) | 2025-03 | GQA | 32/8/128 | Llama-3 scaling | RMS pre | SwiGLU | 128256 | LL3 | n | 131072 | nvidia/Llama-3.1-Nemotron-Nano-8B-v1 |
| **Mistral Small 3** | 24B (oos, axis-load-bearing — sink-token training) | 2025-01 | GQA + **trained attention sinks** | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 131072 | TK131 | n | 32768 | mistralai/Mistral-Small-Instruct-2501 |
| **Mistral Small 3.1** | 24B (oos — adds Pixtral vision encoder) | 2025-03 | GQA + sinks + Pixtral ViT | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 131072 | TK131 | n | 131072 | mistralai/Mistral-Small-3.1-24B-Instruct-2503 |
| **Mistral Small 3.2** | 24B (oos — text-quality refinement) | 2025-06 | GQA + sinks | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 131072 | TK131 | n | 131072 | mistralai/Mistral-Small-3.2-24B-Instruct-2506 |
| **Magistral** | 24B (Mistral reasoning fork, RL-trained) | 2025-Q2 | GQA + sinks | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 131072 | TK131 | n | 32768 | mistralai/Magistral-Small |
| **Moshi 7B** | 7B (Helium-1 backbone + Mimi audio codec + **Temporal Transformer + Depth Transformer**) | 2024-09 | GQA + dual-stream text/audio | 32/8/128 | vanilla | RMS pre | SwiGLU | 32000 | Mimi+SP | n | 4096 | kyutai/moshiko-pytorch-bf16 |
| **Helium-1** | 2B (Moshi base) | 2025-01 | GQA | 24/8/64 | vanilla (θ=1e6) | RMS pre | SwiGLU | 48000 | Helium SP | y | 4096 | kyutai/helium-1-2b |
| **Falcon-H1** | 0.5B | 2025 | Mamba-2 + attn hybrid (parallel-head) | mixed | NTK on attn | RMS pre | SwiGLU | 131072 | Falcon BPE | y | 16384 | tiiuae/Falcon-H1-0.5B-Base |
| **Falcon-H1** | 1.5B (incl. deep variant) | 2025 | Mamba-2 + attn parallel-head | mixed | NTK on attn | RMS pre | SwiGLU | 131072 | Falcon BPE | n | 16384 | tiiuae/Falcon-H1-1.5B-Base |
| **Falcon-H1** | 3B | 2025 | Mamba-2 + attn parallel-head | mixed | NTK on attn | RMS pre | SwiGLU | 131072 | Falcon BPE | n | 16384 | tiiuae/Falcon-H1-3B-Base |
| **Falcon-H1** | 7B | 2025 | Mamba-2 + attn parallel-head | mixed | NTK on attn | RMS pre | SwiGLU | 131072 | Falcon BPE | n | 16384 | tiiuae/Falcon-H1-7B-Base |
| **xLSTM** | 7B | 2025-03 | **mLSTM + sLSTM blocks** (matrix memory + exponential gating); no RoPE, no ALiBi | uniform | none (recurrence) | RMS | SwiGLU | 50304 | xLSTM BPE | n | 4096 | NX-AI/xLSTM-7b |
| **MiniMax-Text-01** | 456B / 45.9B-A (oos, axis-load-bearing — 7:1 Lightning Attention) | 2025-01 | **Lightning Attention 7:1 (linear:softmax)** + MoE | mixed | NTK on softmax layers | RMS pre | SwiGLU-MoE | 200064 | MiniMax SP | n | 4096000 (4M) | MiniMaxAI/MiniMax-Text-01 |
| **Eagle 2** | 1B / 2B / 9B (NVIDIA, **MoVE** vision encoders) | 2025-01 | GQA per backbone (Qwen2-1.5B / Llama3-8B) | per backbone | per backbone | RMS pre | SwiGLU | per backbone | per backbone | per backbone | 32768 | nvidia/Eagle2-{1B,2B,9B} |
| **Ovis 2.5** | 2B / 9B (Visual Embedding Table + NaViT) | 2025-08 | GQA (Qwen2.5 base) | 16/2/128 (2B), 28/4/128 (9B) | M-RoPE | RMS pre | SwiGLU | 151936 | QW | y (2B) / n (9B) | 32768 | AIDC-AI/Ovis2.5-{2B,9B} |
| **Surya OCR 2** | 0.65B | 2025-10 | GQA (Qwen3-style decoder) | per Qwen3 | vanilla (θ=1e6) | RMS pre + QK-norm | SwiGLU | 151936 | QW | y | 32768 | datalab-to/surya-ocr-2 |
| **Dolphin (ByteDance)** | 0.322B (Swin enc + small decoder) | 2025-05 | MHA decoder (small) | 12/12/64 | vanilla | LN | GELU | 250000 | mBART SP | n | 4096 | ByteDance/Dolphin |
| **Dolphin-v2** | 3B (Qwen2.5-VL-3B base) | 2025-Q4 | GQA + M-RoPE | 16/2/128 | M-RoPE | RMS pre | SwiGLU | 151936 | QW | y | 32768 | ByteDance/Dolphin-v2 |

### 3.4 2025-H2 — Hybrid SSM mainstreaming, sparse attention emergence

This is the period where Mamba-2-based hybrids moved from research curiosity to flagship IBM/NVIDIA/Tencent production releases and DeepSeek introduced the first production-grade sparse-attention learned-scorer.

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GPT-OSS** | 20B (5.1B active) (promoted from v2 noted) | 2025-08 | GQA + **trained attention sinks** + **GPT-3-style alternating dense/banded-sparse** | 16/2/128 | LongRoPE | RMS pre | SwiGLU-MoE (32/0/top-4) | 200064 | o200k | n | 131072 | openai/gpt-oss-20b |
| **GPT-OSS** | 120B (5.5B active) (oos, sibling) | 2025-08 | GQA + sinks + alternating | larger | LongRoPE | RMS pre | SwiGLU-MoE | 200064 | o200k | n | 131072 | openai/gpt-oss-120b |
| **Qwen3-Next** | 80B / 3B-A (oos, axis-load-bearing) | 2025-09 | **Gated DeltaNet 3:1 hybrid** (75% linear / 25% softmax attn) | Mixed (DeltaNet + GQA) | vanilla θ=1e6 on softmax layers; recurrence on DeltaNet | RMS pre + QK-norm | SwiGLU-MoE (512+1/top-10, **ultra-sparse**) + MTP head | 151936 | QW | n | 262144 (YaRN to 1M) | Qwen/Qwen3-Next-80B-A3B-Instruct |
| **Granite 4.0 H-Micro** | 3B (dense) | 2025-10 | **9:1 Mamba-2 : attention sequential** (mostly Mamba-2, attention every 10th layer) | 24/8/64 on attn layers | vanilla (θ=1e7) on attn | RMS pre + μP | SwiGLU | 49152 | SC2 | y | 131072 | ibm-granite/granite-4.0-h-micro |
| **Granite 4.0 H-Tiny** | 7B-A1B (MoE-hybrid) | 2025-10 | 9:1 Mamba-2 : attention + MoE | mixed | vanilla on attn | RMS pre + μP | SwiGLU-MoE (32/0/top-4) | 49152 | SC2 | n | 131072 | ibm-granite/granite-4.0-h-tiny |
| **Granite 4.0 H-Small** | 32B-A9B (oos, MoE-hybrid) | 2025-10 | 9:1 Mamba-2 : attention + MoE | mixed | vanilla on attn | RMS pre + μP | SwiGLU-MoE | 49152 | SC2 | n | 131072 | ibm-granite/granite-4.0-h-small |
| **DeepSeek-V3.1** | 671B (37B-A) (oos) | 2025-08 | MLA | 128/128 latent | YaRN | RMS pre | SwiGLU-MoE (256+1/top-8, aux-loss-free) | 129280 | DeepSeek BPE | n | 65536 | deepseek-ai/DeepSeek-V3.1 |
| **DeepSeek-V3.2-Exp** | 671B (37B-A) (oos) | 2025-09 | **DSA (DeepSeek Sparse Attention)** + Lightning Indexer | latent + sparse | YaRN | RMS pre | SwiGLU-MoE (256+1/top-8) | 129280 | DeepSeek BPE | n | 131072 | deepseek-ai/DeepSeek-V3.2-Exp |
| **DeepSeek-V3.2** | 671B stable (oos) | 2025-12 | DSA + Lightning Indexer | latent + sparse | YaRN | RMS pre | SwiGLU-MoE | 129280 | DeepSeek BPE | n | 131072 | deepseek-ai/DeepSeek-V3.2 |
| **Apple Foundation Model** | 3.18B (on-device) (promoted to first-class) | 2025-07 (paper) | GQA + **2-block cross-block KV sharing** (Block-1 62.5% layers full KV, Block-2 37.5% reuses Block-1 K/V) + 2-bit QAT | per AFM tech rpt | vanilla | RMS pre + QK-norm | SwiGLU | ~100k | Apple SP | y | 4096 | arXiv:2507.13575 |
| **Hunyuan** | 0.5B | 2025-08 | GQA | 12/4/64 | vanilla (θ=10000) | RMS pre | SwiGLU | ~150k | HY | y | 32768 | tencent/Hunyuan-0.5B |
| **Hunyuan** | 1.8B | 2025-08 | GQA | 16/4/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | ~150k | HY | y | 32768 | tencent/Hunyuan-1.8B |
| **Hunyuan** | 4B | 2025-08 | GQA | 24/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | ~150k | HY | n | 65536 | tencent/Hunyuan-4B |
| **Hunyuan** | 7B + **fusion fast/slow reasoning toggle** | 2025-08 | GQA | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | ~150k | HY | n | 262144 | tencent/Hunyuan-7B-Instruct |
| **HunyuanOCR** | 1B (Hunyuan-0.5B base + native MM training) | 2025-11 | GQA + native multimodal training | per Hunyuan-0.5B | vanilla | RMS pre | SwiGLU | ~150k | HY | y | 32768 | tencent/HunyuanOCR |
| **PaddleOCR-VL** | 0.9B (**ERNIE-4.5-0.3B** base + NaViT) | 2025-10 | GQA (ERNIE) | per ERNIE-4.5 | vanilla | RMS pre | SwiGLU | ~50k | ERNIE | y | 8192 | PaddlePaddle/PaddleOCR-VL |
| **PaddleOCR-VL-1.5** | 0.9B | 2026-01 | GQA (ERNIE) | per ERNIE-4.5 | vanilla | RMS pre | SwiGLU | ~50k | ERNIE | y | 8192 | PaddlePaddle/PaddleOCR-VL-1.5 |
| **ERNIE-4.5** | 0.3B (text decoder used by PaddleOCR-VL) | 2025-10 | GQA | 12/4/64 | vanilla | RMS pre | SwiGLU | ~50k | ERNIE | y | 8192 | baidu/ERNIE-4.5-0.3B |
| **Nanonets-OCR2-3B** | 3B (Qwen2.5-VL-3B base, 125k ctx) | 2025-12 | GQA + M-RoPE | 16/2/128 | M-RoPE | RMS pre | SwiGLU | 151936 | QW | y | 128000 | nanonets/Nanonets-OCR2-3B |
| **InternVL 3.5** | 1B–241B (sub-8B in scope) | 2025-08 | GQA per backbone | per backbone | per backbone | RMS pre | SwiGLU | per backbone | per backbone | per backbone | 32768 | OpenGVLab/InternVL3_5 |
| **Tencent Hy-MT2** | 1.8B / 7B / 30B-A3B | 2025-Q4 | GQA | per size | vanilla | RMS pre | SwiGLU | ~150k | HY | y / n | 32768 | tencent/Hy-MT2 |

### 3.5 2026-H1 — Gemma 4, Llama 4, DeepSeek V4, Mamba-3, Granite 4.1, Nemotron 3

This is the period where Google, Meta, DeepSeek, IBM, and NVIDIA all shipped second-generation 2026-era architectures simultaneously, and the OCR-LLM lineage matured to the encoder-free unified design.

#### Gemma 4 family (2026-04-02 launch; 2026-06-05 QAT addendum)

| Variant | Released | Total / Active | Layers / d_model | H/KV/hd local | H/KV/hd global | SWA / pattern | RoPE local θ | RoPE global θ | Partial rotary global | Cross-layer KV | Per-Layer Emb | Final softcap | Multimodal | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E2B** | 2026-04 | 5.1B (with embed) / 2.3B effective | 35 / 1536 | 8/1/256 | 8/1/512 (K=V) | 512 / **4:1** last global | 10000 | 1000000 | **0.25** | `num_kv_shared_layers=20` | **yes** (256-dim) | 30.0 | text+image+audio | google/gemma-4-E2B-it |
| **E4B** | 2026-04 | 8B / 4.5B effective | 42 / 2560 | 8/2/256 | 8/2/512 (K=V) | 512 / 5:1 last global | 10000 | 1000000 | 0.25 | ~20 | yes | 30.0 | text+image+audio | google/gemma-4-E4B-it |
| **12B Unified** | 2026-04 | 11.95B | 48 / 3840 | 16/8/256 | 16/8/256 | 1024 / 5:1 last global (full at 5,11,17,23,29,35,41,47) | 10000 | 1000000 | 0.25 | 0 | — | 30.0 | text+image+audio+video (**encoder-free**) | google/gemma-4-12B-it |
| **26B-A4B (MoE)** | 2026-04 | 25.2B / 3.8B | 30 / 2816 | 16/8/256 | 16/8/256 | 1024 / 5:1 | 10000 | 1000000 | 0.25 | 0 | — | 30.0 | text+image; **128+1 experts, top-k=8, `moe_intermediate_size=704`** (first Gemma MoE) | google/gemma-4-26B-A4B-it |
| **31B** | 2026-04 | 30.7B | 60 / 5376 | 32/16/256 | 32/16/256 | 1024 / 5:1 | 10000 | 1000000 | 0.25 | 0 | — | 30.0 | text+image | google/gemma-4-31B-it |
| **MTP drafter heads** | 2026-04-16 | small drafter per size | — | — | — | — | — | — | — | — | — | — | — | google/gemma-4 (MTP collection) |

All Gemma 4 variants: vocab 262144 (GP-v4 + image/audio/video special tokens), tied embeddings, embed_scale=sqrt(D), `use_double_wide_mlp=True` on edge sizes, RMSNorm 1+w with dual-norm sandwich (input + post-attn + pre-ffn + post-ffn), GeGLU activation (`hidden_activation="gelu_pytorch_tanh"` — Gemma signature preserved end-to-end), QK-RMSNorm pre-RoPE with fixed-scale norms (local 0.9916 / global 1.0228) replacing `1/sqrt(head_dim)`, no `query_pre_attn_scalar`, `attn_logit_softcapping=null` (Gemma 3 decision preserved), Apache 2.0 license (changed from Gemma 1-3 Gemma Terms of Use). Per-Layer Embeddings on E2B/E4B compute a per-layer residual as `(token_identity + context_aware_projection) × 1/√2` from a small (256-dim) PLE table that can be paged to flash storage; the main residual remains at 1536/2560 dim. This is the single largest architectural delta from Gemma 3.

#### Llama 4 (2025-04, oos but axis-load-bearing)

| Variant | Released | Total / Active | Experts/top-k/shared | Attn | NoPE pattern | Multimodal | Context | Source |
|---|---|---|---|---|---|---|---|---|
| **Scout 17B-16E** | 2025-04 | 109B / 17B-A | 16/0/0 | GQA + **iRoPE** (interleaved per-layer NoPE mask) | per `no_rope_layers[]` | **native early fusion** (image tokens are regular self-attn tokens) | 10M (NoPE) | meta-llama/Llama-4-Scout-17B-16E |
| **Maverick 17B-128E** | 2025-04 | 400B / 17B-A | 128/0/1 | GQA + iRoPE | per `no_rope_layers[]` | native early fusion | 1M | meta-llama/Llama-4-Maverick-17B-128E |
| **Behemoth (out of scope)** | 2025-Q3 (preview) | 2T / 288B-A | unspecified | GQA + iRoPE | per layer | early fusion | unspecified | ai.meta.com/blog/llama-4-multimodal-intelligence |

Llama 4 vocab is LL4 (Llama-4 Tiktoken 200k); tied n; SwiGLU; RMSNorm pre; RoPE θ varies per layer (NoPE layers have effectively θ=∞). The production validation of SmolLM3's NoPE-every-4th experiment at scale.

#### DeepSeek V4 (2026-04-24)

| Variant | Released | Total / Active | Attn | Experts/top-k/shared | Context | KV cache | Source |
|---|---|---|---|---|---|---|---|
| **V4-Pro** | 2026-04 | 1.6T / 49B-A (oos) | **CSA + HCA hybrid sparse** | 256+1 / top-8 / 1 (aux-loss-free) | 1M | 10% of V3.2 at 1M | deepseek-ai/DeepSeek-V4-Pro |
| **V4-Flash** | 2026-04 | 284B / 13B-A (oos) | CSA + HCA | 128+1 / top-8 / 1 | 1M | 10% of V3.2 | deepseek-ai/DeepSeek-V4-Flash |
| **V4-Pro-NVFP4** | 2026-04 | quantized V4-Pro | — | — | — | — | nvidia/DeepSeek-V4-Pro-NVFP4 |

DeepSeek-V4 retires the V3.2 DSA learned-scorer in favor of CSA+HCA (Compressed Sparse Attention + Heavily Compressed Attention), an architectural shift that nominally combines two complementary compression schemes — CSA compresses the global token context, HCA compresses the historical KV blocks. At 1M context, V4-Pro single-token inference takes 27% of V3.2's FLOPs and 10% of V3.2's KV cache — a much larger leap than V3.2-Exp's DSA-only delta.

#### Granite 4.1 (2026-04-29)

| Variant | Released | Params | Token mixer | Source |
|---|---|---|---|---|
| **Granite 4.1 3B** | 2026-04 | 3B dense | standard GQA + μP | ibm-granite/granite-4.1-3b |
| **Granite 4.1 8B** | 2026-04 | 8B dense | standard GQA + μP | ibm-granite/granite-4.1-8b |
| **Granite 4.1 30B** | 2026-04 | 30B dense (oos) | standard GQA + μP | ibm-granite/granite-4.1-30b |
| **Granite Vision 4.1-4B** | 2026-04 | 4B VLM | GQA + image adapter | ibm-granite/granite-vision-4.1-4b |
| **Granite Speech 4.1 AR-2B** | 2026-04 | 2B (ASR-translate, autoregressive) | GQA | ibm-granite/granite-speech-4.1-AR-2B |
| **Granite Speech 4.1 NAR-2B** | 2026-04 | 2B (speech edit, **non-autoregressive**) | GQA + parallel decode | ibm-granite/granite-speech-4.1-NAR-2B |
| **Granite Guardian 4.1** | 2026-04 | small safety classifier | encoder | ibm-granite/granite-guardian-4.1 |

Granite 4.1 differs from Granite 4.0 in dropping the 9:1 Mamba-2:attention hybrid in favor of dense decoders — the dense 4.1-8B claims to match the prior 32B-MoE generation. The hybrid lives on in Granite 4.0; Granite 4.1 is the dense path. The Granite Speech NAR-2B is one of the only NAR speech heads in production at <8B.

#### NVIDIA Nemotron 3 (2026-03 / 2026-04 / 2026-06)

| Variant | Released | Total / Active | Architecture | Source |
|---|---|---|---|---|
| **Nemotron 3 Super** | 2026-03 | 120B / 12B-A (oos) | Hybrid Mamba-2 + Transformer MoE | nvidia/Nemotron-3-Super |
| **Nemotron 3 Nano Omni** | 2026-04 | Nano-class (~8B) (oos borderline) | Hybrid Mamba-Transformer Omni (vision+speech+language) | nvidia/Nemotron-3-Nano-Omni |
| **Nemotron 3 Ultra** | 2026-06 | 550B / 55B-A (oos) | Hybrid Mamba-Attention MoE | nvidia/Nemotron-3-Ultra |
| **Nemotron 3 Nano 4B** | 2025-11 | 4B | Hybrid Mamba-2 + Transformer + MoE FFN | nvidia/Nemotron-3-Nano-4B |
| **Llama-3.1-Nemotron-Nano** | 2025-03 | 8B (Llama-3.1-8B base + FP4-aware RL) | GQA | nvidia/Llama-3.1-Nemotron-Nano-8B-v1 |

Nemotron 3 Super and Ultra together represent the largest open hybrid Mamba-Transformer-MoE deployment. Nemotron 3 Ultra at 550B/55B-A claims ~6× throughput vs GLM-5.1 at 8K/64K context — the SSM-hybrid mainstreaming signal of Q2 2026.

#### Mamba-3 (2026-03-17)

| Variant | Released | Params | Architecture | Source |
|---|---|---|---|---|
| **Mamba-3 (research scales)** | 2026-03 | 180M – 1.5B reference | Pure SSM with **complex-valued state** + refined discretization + **MIMO decoding** | github.com/state-spaces/mamba |

Mamba-3's three substantive deltas vs Mamba-2: (1) more expressive recurrence from SSM discretization; (2) complex-valued state-update rule for richer state tracking; (3) MIMO (multi-input multi-output) decoding for accuracy at fixed decode latency. Accepted at ICLR 2026 (arXiv:2603.15569).

#### Other 2026-H1 additions

| Family | Variant | Released | Architecture summary | Source |
|---|---|---|---|---|
| **Mistral Small 4** | 119B / 15B-A (oos) | 2026-03 | First Mistral unifying instruct+Magistral+Devstral+Pixtral in one MoE; eagle-head speculative decoding | mistralai/Mistral-Small-4-119B-2603 |
| **Mistral Medium 3.5** | 128B dense (oos) | 2026-04 | Absorbs Magistral + Devstral 2; configurable reasoning effort; 256k ctx | mistralai/Mistral-Medium-3.5 |
| **Voxtral TTS** | 4B | 2026-03 | Open TTS; 5-sec zero-shot voice cloning; 90ms latency | mistralai/voxtral-tts |
| **GLM-5.1** | 744B / 40B-A (oos) | 2026-04 | **DSA-derivative sparse attention**; post-trained for 8-hr agentic loops; 200k ctx | zai-org/GLM-5.1 |
| **GLM-5V-Turbo** | derived VLM (oos) | 2026-04 | VLM sibling of GLM-5 | zai-org/GLM-5V-Turbo |
| **Tencent Hunyuan-A13B** | 80B / 13B-A (oos) | 2026-04 | Fine-grained MoE, 256k ctx | tencent/Hunyuan-A13B-Instruct |
| **Tencent Hy3 Preview** | 295B / 21B-A (oos) | 2026-04 | + 3.8B MTP head; fast/slow hybrid | tencent/Hy3-preview |
| **Qwen3.6-35B-A3B** | 35B / 3B-A (oos) | 2026-04 | Refreshed multimodal MoE; 262k native (YaRN to 1M) | Qwen/Qwen3.6-35B-A3B |
| **Qwen3.6-27B** | 27B dense (oos) | 2026-04 | Dense 27B coding leader; 1M ctx | Qwen/Qwen3.6-27B |
| **Qwen3.5-Omni Plus / Flash / Light** | 28B / 36B / 125B / 403B (oos) | 2026-04 | Hybrid Attention MoE Thinker + Talker; 256k ctx; 10 langs | Qwen/Qwen3.5-Omni |
| **Cohere Command A+** | 218B / 25B-A (oos) | 2026-05 | First Cohere MoE; Apache 2.0; W4A4 lossless quant | CohereLabs/command-a-plus-05-2026 |
| **EXAONE 4.5** | 33B (oos) | 2026-04 | VLM; STEM-tuned; Korean+EU langs | LGAI-EXAONE/EXAONE-4.5 |
| **Liquid LFM2.5-VL-450M** | 0.45B | 2026-04 | **Non-Transformer** LFM (liquid-time-constant) backbone + vision | LiquidAI/LFM2-VL-450M |
| **Liquid LFM2.5-8B-A1B** | 8.3B / 1.5B-A | 2026-05 | LFM-backbone MoE | LiquidAI/LFM2.5-8B-A1B |
| **Falcon-Edge 1B / 3B** | 1B / 3B | 2026-05 | **Retrainable 1.58-bit BitNet** (first BitNet supporting continued pretrain) | tiiuae/Falcon-Edge |
| **Falcon Perception 0.6B** | 0.6B | 2026-04 | Early-fusion multimodal grounding + segmentation | tiiuae/Falcon-Perception |
| **HY-Embodied-0.5 MoT-2B** | 2B | 2026-04 | **Mixture-of-Transformers** embodied foundation; spatial-temporal reasoning | tencent/HY-Embodied |
| **Reka Edge 2026** | ~8B | 2026-03 | Re-trained dense edge model; 3× fewer tokens, 65% higher throughput vs 8B-class | RekaAI/Reka-Edge-2026 |
| **Reka Flash 3.1** | 21B (oos) | 2026-03 | RLOO-trained reasoning refresh | RekaAI/reka-flash-3.1 |
| **Zyphra ZAYA1-8B** | 8.4B (borderline) | 2026-05 | MoE reasoning, AMD MI300X trained | Zyphra/ZAYA1-8B |
| **StepFun Step 3.7 Flash** | 198B / 11B-A (oos) | 2026-05 | First multimodal Flash; 1.8B ViT | stepfun-ai/Step-3.7-Flash |
| **MiniMax-M3.0** | (~230B / 10B-A, oos) | 2026-06 | **MSA (MiniMax Sparse Attention)** — new sparse variant; 1M ctx; weights staged | MiniMaxAI/MiniMax-M3.0 |
| **Xiaomi MiMo-V2.5 / -Pro / -ASR** | 310B/15B-A; 1.02T/42B-A; ASR (oos) | 2026-04 | Massive sparse MoE; FP8 train; 729M ViT + audio; native 1M ctx; MIT | XiaomiMiMo/MiMo-V2.5 |
| **Moonshot Kimi K2.6** | 1T / 32B-A (oos) | 2026-04 | Native multimodal agentic MoE; 256k ctx; 300-sub-agent swarm | moonshotai/Kimi-K2.6 |

**Total v3 census: ~148 rows spanning ~52 distinct architectural families** (vs v2's 80 rows / ~30 families). The new rows are distributed across 2024 (18 added), 2025-H1 (28 added), 2025-H2 (12 added), and 2026-H1 (28 added).

---

## 4. Comprehensive Table (19 Axes, Representative Models)

This table is the at-a-glance reference for v3's expanded axis catalog. Columns are A1–A19 from §6. Empty cells = "default" or "n/a"; "—" = explicitly absent; "✓" = present. For brevity, ~80 of the 148 census rows are tabulated as representatives of their family; the remaining ~68 rows inherit axis values from their parent family in the §5 evolution arcs.

The 19 axes are:

- **A1** Attention type {MHA, GQA, MQA, MLA, SWA, BlockSparse, Linear/SSM-{Mamba1, Mamba2, Mamba3, RWKV6, RWKV7, Griffin/Hawk, xLSTM, GatedDeltaNet, LightningAttn, LFM}, Hybrid-{seq, periodic, parallel, ratio-N:M}, DSA, CSA+HCA, MSA}
- **A2** Position encoding scheme {none, absolute, ALiBi, RoPE-vanilla, RoPE-Llama3-band, RoPE-LongRoPE, RoPE-YaRN, RoPE-DynamicNTK, RoPE-linear, RoPE-dual (Gemma 3/4), M-RoPE 3D, TMRoPE 4D, 2D-RoPE, V2PE, MLA-split, iRoPE per-layer}
- **A3** Normalization (type+placement+QK-norm) {RMS pre, RMS post (OLMo 2), RMS dual (Gemma), LN pre, none; QK-norm: none / per_Dh pre-RoPE / per_Dh post-RoPE / per_full_channels / fixed-scale (Gemma 4)}
- **A4** FFN family (gate topology) {ungated, gated (X-GLU), gated-fused, MoE-{routed, routed+shared}, RWKV channel-mix, double-wide (Gemma 4)}
- **A5** MoE routing {none, top-k softmax + aux-loss, + shared, top-k sigmoid + aux-loss-free + shared, top-k softmax + norm_topk + 0-shared, ultra-sparse (Qwen3-Next: 512+1/top-10), Mixture-of-LoRAs (Phi-4-MM), Mixture-of-Transformers (HY-Embodied)}
- **A6** KV-cache shape {full MHA, GQA, MQA, MLA latent, SSM state, RWKV state, K=V (Gemma 4 global), 2-bit QAT (Apple AFM)}
- **A7** Vocab/tokenizer family
- **A8** μP scalars {none, Granite four-scalar, MiniCPM scale_emb+depth, Phi-3-small μP, Cohere logit_scale, Gemma fixed QK-norm scale}
- **A9** Bias presence
- **A10** Embedding tying
- **A11** Per-layer type alternation {uniform, SWA/full {1:1, 5:1, 4:1 (E2B), interleaved (Ministral)}, dense/MoE, Mamba/attn periodic {1:8, 9:1, 3:1}, parallel-branches (Hymba), GPT-3 alternating dense/banded-sparse (GPT-OSS), NoPE per-layer (SmolLM3, Llama 4 iRoPE)}
- **A12** Per-layer shape scaling {uniform, OpenELM per-layer Q/KV/FFN, MobileLLM block-wise weight sharing, Gemma 3n Matformer elastic, Apple AFM 2-block split}
- **A13** Parallel vs sequential sublayer
- **A14** Activation function {SiLU, GELU, GELU-tanh, GELU-new, gegelu, ReLU², sigmoid+ReLU² (RWKV), Swish}
- **A15** RoPE rotation domain {full, partial-{0.5, 0.4, 0.25, 0.75}, NoPE/RoPE split (MLA), NoPE per-layer (SmolLM3, Llama 4 iRoPE), per-layer-type partial (Gemma 4 p-RoPE on global only, factor=0.25)}
- **A16** VLM fusion topology (NEW) {none, concat-prefix-LM, concat-projector, cross-attention-adapter (Llama 3.2 Vision), early-fusion (Chameleon, Llama 4), mixture-of-LoRAs (Phi-4-MM), split-encoder (Janus-Pro), encoder-free (Gemma 4 12B Unified)}
- **A17** Vision-token compression budget (NEW) — measured as tokens/page at 1024² resolution: {pre-LLM (n/a), high (~3000-6000) (Qwen2-VL native), mid (~640-1024) (MiniCPM-V resampler), low (~256-324) (GOT-OCR2 high-compression, mPLUG-DocOwl2), ultra-low (~100) (DeepSeek-OCR with 16× compressor), elastic (V2PE)}
- **A18** Cross-layer KV sharing (NEW) {none, same-block-shared (Gemma 4 `num_kv_shared_layers`), cross-block-shared (Apple AFM 2-block), per-layer pointer (YOCO research), CLA (research)}
- **A19** Per-layer embeddings (NEW) {none, Gemma 4 PLE (256-dim residual at every layer, paged to flash), Apple AFM per-layer rank-1 LoRA adapter}

| # | Model | A1 | A2 | A3 | A4 | A5 | A6 | A11 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LLaMA 1 7B | MHA | RoPE-vanilla θ=10k | RMS pre | gated SwiGLU | – | full | uniform | seq | SiLU | full | – | – | – | – |
| 2 | Llama 3.1 8B | GQA | RoPE-Llama3 band f=8 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 3 | Llama 3.2 1B | GQA | RoPE-Llama3 f=32 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 4 | **Llama 4 Scout 17B-A 16E** | GQA + iRoPE | **iRoPE NoPE per-layer mask** | RMS pre | SwiGLU-MoE 16/0/0 | top-1/0-shared (16 experts) | GQA | **NoPE-per-layer mask** | seq | SiLU | **NoPE per-layer** | **early-fusion native** | ~3000 (native res) | – | – |
| 5 | **Llama 4 Maverick 17B-A 128E** | GQA + iRoPE | iRoPE | RMS pre | SwiGLU-MoE 128/0/1 | top-1/1-shared | GQA | NoPE-per-layer | seq | SiLU | NoPE-per-layer | early-fusion | ~3000 | – | – |
| 6 | Mistral 7B v0.1 | GQA + SWA=4096 | RoPE-vanilla θ=10k | RMS pre | gated SwiGLU | – | GQA + SWA | uniform | seq | SiLU | full | – | – | – | – |
| 7 | Mistral 7B v0.3 | GQA | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 8 | **Ministral 8B** | GQA + **interleaved SWA** | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | mixed | **interleaved full/SWA** | seq | SiLU | full | – | – | – | – |
| 9 | Mistral Small 3.1 24B | GQA + **sinks** | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector (Pixtral ViT) | ~3000 | – | – |
| 10 | **Mistral Small 4 119B/15B-A** | GQA + sinks | RoPE-vanilla | RMS pre | gated SwiGLU-MoE | – | GQA | uniform | seq | SiLU | full | concat-projector | ~3000 | – | – |
| 11 | **Mistral Medium 3.5 128B** | GQA + sinks | RoPE-vanilla | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector | ~3000 | – | – |
| 12 | MPT 7B | MHA | **ALiBi** | LN pre | ungated GELU | – | full | uniform | seq | GELU | – | – | – | – | – |
| 13 | Falcon 7B | MQA | RoPE-vanilla | LN pre | ungated GELU | – | MQA | uniform | **parallel** | GELU | full | – | – | – | – |
| 14 | Falcon Mamba 7B | SSM-Mamba1 | – | RMS pre | – (SSM gate) | – | SSM state | uniform | seq | SiLU | – | – | – | – | – |
| 15 | **Falcon-H1 7B** | **Mamba-2 ‖ attention parallel-head** | NTK on attn | RMS pre | gated SwiGLU | – | SSM state + GQA | parallel-head | seq | SiLU | full on attn-only | – | – | – | – |
| 16 | Falcon 3 7B | GQA | RoPE-vanilla θ=1000042 | RMS pre | gated SwiGLU | – | GQA hd=256 | uniform | seq | SiLU | full | – | – | – | – |
| 17 | **Falcon-Edge 1B (1.58-bit)** | GQA | RoPE | RMS pre | gated SwiGLU + bitlinear | – | GQA + 1.58-bit | uniform | seq | ReLU² (BitNet-style) | full | – | – | – | – |
| 18 | **Falcon-Edge 3B (1.58-bit retrainable)** | GQA | RoPE | RMS pre | gated SwiGLU + bitlinear | – | GQA + 1.58-bit | uniform | seq | ReLU² | full | – | – | – | – |
| 19 | Qwen 2 7B | GQA | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 20 | Qwen 2.5-VL 7B | GQA | **M-RoPE 3D (T,H,W)** | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector | ~3000 native dynamic | – | – |
| 21 | Qwen 3 8B | GQA + QK-norm per-Dh pre-RoPE | RoPE-vanilla θ=1e6 | RMS pre + QK-norm | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 22 | Qwen 3 MoE 30B-A3B | GQA + QK-norm | RoPE-vanilla θ=1e6 | RMS pre + QK-norm | SwiGLU-MoE 128/0/top-8 norm_topk | top-8/0-shared | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 23 | **Qwen3-Next 80B-A3B** | **Gated DeltaNet 3:1 hybrid** | RoPE-vanilla θ=1e6 on softmax layers; recurrence on DeltaNet | RMS pre + QK-norm | **SwiGLU-MoE 512+1/top-10 ultra-sparse + MTP head** | top-10/1-shared (2.15% activation) | mixed (DeltaNet + GQA) | **3:1 linear:softmax** | seq | SiLU | full on softmax layers | – | – | – | – |
| 24 | **Qwen3-VL 8B** | GQA + QK-norm | M-RoPE | RMS pre + QK-norm | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector | dynamic | – | – |
| 25 | **Qwen 2.5-Omni 7B** | GQA + **TMRoPE 4D** | TMRoPE (T,H,W,frame-idx) | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector (Thinker-Talker dual decoder) | dynamic | – | – |
| 26 | **Qwen3.5-Omni** | GQA + TMRoPE | TMRoPE | RMS pre + QK-norm | SwiGLU-MoE Thinker+Talker | top-k/shared | GQA | uniform | seq | SiLU | full | concat-projector dual | dynamic | – | – |
| 27 | Gemma 1 2B | MQA | RoPE-vanilla θ=10k | RMS pre+post | gated GeGLU | – | MQA hd=256 | uniform | seq | GELU-tanh | full | – | – | – | – |
| 28 | Gemma 2 2B | GQA + SWA every-other | RoPE-vanilla θ=10k | RMS pre+post + softcap(50,30) | gated GeGLU | – | mixed | **1:1 SWA/full** | seq | GELU-tanh | full | – | – | – | – |
| 29 | Gemma 3 4B | GQA + 5:1 SWA:full | **RoPE-dual** local 1e4 / global 1e6 | RMS pre+post (softcap dropped) | gated GeGLU | – | mixed | **5:1** | seq | GELU-tanh | full per layer-type | – | – | – | – |
| 30 | Gemma 3n E4B | as Gemma 3 + Matformer | dual θ | RMS pre+post | gated GeGLU | – | mixed | per layer-type | seq | GELU-tanh | full | – | – | – | – |
| 31 | **Gemma 4 E2B** | GQA local 8/1 + global 8/1 hd=512 (**KV via cross-layer share, NOT K=V**) | **RoPE-dual + partial-rotary 0.25 on global** | RMS pre+post + fixed-scale QK-norm (0.9916/1.0228) | gated GeGLU **double-wide** | – | mixed; **cross-layer KV shared 20/35 layers** | **4:1 last global** | seq | GELU-tanh | **partial-0.25 on global, full on local** | encoder-based image + audio | mid (~140-560/page) | **same-block-shared (20/35)** | **PLE 256-dim residual** |
| 32 | **Gemma 4 E4B** | GQA local 8/2 + global 8/2 hd=512 (cross-layer share) | RoPE-dual + p-RoPE 0.25 | RMS pre+post + fixed-scale QK | gated GeGLU double-wide | – | mixed; CL-shared ~20 | 5:1 | seq | GELU-tanh | partial-0.25 global | encoder-based MM | mid | same-block-shared | PLE |
| 33 | **Gemma 4 12B Unified** (**K=V on global, no cross-layer share**) | GQA 16/8 hd=256; **K=V global** | RoPE-dual + p-RoPE 0.25 | RMS pre+post | gated GeGLU | – | mixed; K=V global aliased | 5:1 last global | seq | GELU-tanh | partial-0.25 global | **encoder-free** (raw projection) | encoder-free | – (no CL share) | – |
| 34 | **Gemma 4 26B-A4B** (**K=V on global, no cross-layer share**) | GQA 16/8 hd=256; **K=V global** | RoPE-dual + p-RoPE 0.25 | **gated GeGLU-MoE 128+1/top-8** | top-8/1-shared 128 routed | mixed; K=V global aliased | 5:1 | seq | GELU-tanh | partial-0.25 global | encoder-based image | mid | – | – | – |
| 35 | **Gemma 4 31B** (**K=V on global, no cross-layer share**) | GQA 32/16 hd=256; **K=V global** | RoPE-dual + p-RoPE 0.25 | RMS pre+post | gated GeGLU | – | mixed; K=V global aliased | 5:1 | seq | GELU-tanh | partial-0.25 global | encoder-based image | mid | – | – |
| 36 | RecurrentGemma 2B | Griffin/Hawk (LRU + local attn) | RoPE on attn only | RMS pre | gated GeGLU | – | LRU + windowed attn | block_types[] | seq | GELU-tanh | full on attn | – | – | – | – |
| 37 | Phi-2 2.7B | MHA + partial-rotary 0.4 | partial-rotary 0.4 | LN | ungated GELU-new | – | full | uniform | **parallel** | GELU-new | **partial-0.4** | – | – | – | – |
| 38 | Phi-3 mini 128k | MHA | LongRoPE | RMS pre | **gated-fused SwiGLU** | – | full | uniform | seq | SiLU | full | – | – | – | – |
| 39 | Phi-3-small 7B | **BlockSparse** dense-every-2nd + μP | RoPE-vanilla θ=1e6 + μP | LN + μP | gegelu | – | mixed (BS + dense) | dense/BS alt | seq | gegelu | full | – | – | – | – |
| 40 | Phi-3.5-MoE 6.6B-A | MHA | RoPE | RMS pre | SwiGLU-MoE 16/0/top-2 | top-2/0-shared 16 routed | full | uniform | seq | SiLU | full | – | – | – | – |
| 41 | Phi-4 mini 3.8B | GQA + partial-rotary 0.75 | LongRoPE on rotary fraction | RMS pre | gated-fused SwiGLU | – | GQA | uniform | seq | SiLU | **partial-0.75** | – | – | – | – |
| 42 | Phi-4-mini-flash | **Mamba-2 + attn hybrid (Samba)** | LongRoPE on attn layers | RMS pre | gated-fused SwiGLU | – | SSM + attn | Mamba/attn periodic | seq | SiLU | full on attn-only | – | – | – | – |
| 43 | **Phi-4-multimodal 5.6B** | GQA + p-rotary 0.75 + **modality-routed LoRA** | LongRoPE | RMS pre | gated-fused SwiGLU | – | GQA | uniform | seq | SiLU | partial-0.75 | **mixture-of-LoRAs** (vision LoRA + audio LoRA over frozen Phi-4-mini) | varies per LoRA | – | – |
| 44 | OLMo 1 7B | MHA | RoPE-vanilla θ=10k | RMS pre | gated SwiGLU | – | full | uniform | seq | SiLU | full | – | – | – | – |
| 45 | OLMo 2 7B | MHA | RoPE-vanilla θ=5e5 | **RMS post** + QK-RMS per-full-channels | gated SwiGLU | – | full | uniform | seq | SiLU | full | – | – | – | – |
| 46 | SmolLM2 1.7B | MHA | RoPE-vanilla θ=1.3e5 | RMS pre | gated SwiGLU | – | full | uniform | seq | SiLU | full | – | – | – | – |
| 47 | **SmolLM3 3B** | GQA + **NoPE every 4th** | mask | RMS pre | gated SwiGLU | – | GQA | **NoPE/RoPE alt every-4th** | seq | SiLU | **NoPE per-layer** | – | – | – | – |
| 48 | **SmolVLM2 2.2B** | GQA (SmolLM2 base + 9× pixel-shuffle vision) | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector (Idefics3-derived) | low (~256 via 9× pixel-shuffle) | – | – |
| 49 | Granite 3.3 8B | GQA + four-scalar μP | RoPE-vanilla θ=1e7 | RMS pre + μP | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 50 | **Granite 4.0 H-Tiny 7B-A1B** | **9:1 Mamba-2 : attention sequential** + MoE | RoPE-vanilla on attn layers only | RMS pre + μP | gated SwiGLU-MoE 32/0/top-4 | top-4/0-shared 32 routed | mixed (SSM + GQA) | **9:1 Mamba/attn sequential** | seq | SiLU | full on attn-only | – | – | – | – |
| 51 | **Granite 4.0 H-Small 32B-A9B** | 9:1 Mamba-2 + MoE | RoPE on attn | RMS pre + μP | SwiGLU-MoE | top-k/0-shared | mixed | 9:1 | seq | SiLU | full on attn-only | – | – | – | – |
| 52 | **Granite 4.1 8B (dense)** | GQA + μP | RoPE-vanilla θ=1e7 | RMS pre + μP | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 53 | **Granite Vision 4.1 4B** | GQA + image adapter | RoPE | RMS pre + μP | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector | mid | – | – |
| 54 | **Granite Speech 4.1 NAR-2B** | GQA + **parallel decode (NAR)** | RoPE | RMS pre + μP | gated SwiGLU | – | GQA | uniform | seq (NAR) | SiLU | full | concat-projector audio | dynamic | – | – |
| 55 | InternLM 3 8B | GQA (KV=2) | dynamic NTK θ=5e7 f=6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 56 | DeepSeek-V2-Lite 16B-A2.4B | **MLA** kv_lora=512, qk_nope=128, qk_rope=64, v=128 | YaRN on RoPE channels | RMS pre | gated SwiGLU-MoE 64+2/top-6 | top-6/2-shared | **MLA latent (576/token)** | dense=1 + MoE all | seq | SiLU | **MLA NoPE/RoPE split** | – | – | – | – |
| 57 | **DeepSeek-V3.1** | MLA | YaRN | RMS pre | SwiGLU-MoE 256+1/top-8 aux-loss-free | top-8/1-shared 256 routed | MLA | dense + MoE | seq | SiLU | MLA split | – | – | – | – |
| 58 | **DeepSeek-V3.2 / V3.2-Exp** | **DSA (DeepSeek Sparse Attention)** with Lightning Indexer | YaRN | RMS pre | SwiGLU-MoE 256+1/top-8 | top-8/1-shared | DSA learned-scorer top-k | dense + MoE | seq | SiLU | MLA split | – | – | – | – |
| 59 | **DeepSeek-V4-Pro 1.6T/49B-A** | **CSA + HCA hybrid sparse** | YaRN | RMS pre | SwiGLU-MoE | top-8/1-shared | CSA + HCA (10% V3.2 KV at 1M) | dense + MoE | seq | SiLU | MLA split | – | – | – | – |
| 60 | **DeepSeek-V4-Flash 284B/13B-A** | CSA + HCA | YaRN | RMS pre | SwiGLU-MoE 128+1/top-8 | top-8/1-shared | CSA + HCA | dense + MoE | seq | SiLU | MLA split | – | – | – | – |
| 61 | DeepSeek-VL2 (V2-Lite base) | MLA | YaRN | RMS pre | SwiGLU-MoE | top-6/2-shared | MLA | dense + MoE | seq | SiLU | MLA split | concat-projector (SigLIP + dyn tiling) | mid | – | – |
| 62 | **DeepSeek-OCR 3.4B/A570M** | **MLA + MoE** (DeepSeek3B-MoE-A570M) | YaRN | RMS pre | SwiGLU-MoE 64+2/top-6 | top-6/2-shared | MLA | dense + MoE | seq | SiLU | MLA split | concat-projector (**DeepEncoder**: SAM-Base + CLIP-L serial + **16× compressor**) | **ultra-low (~100/page)** | – | – |
| 63 | **DeepSeek-OCR-2** | MLA + MoE + **Visual Causal Flow mask** | YaRN | RMS pre | SwiGLU-MoE | top-6/2-shared | MLA | dense + MoE | seq | SiLU | MLA split | concat-projector + **VCF bidirectional within page, causal across pages** | ultra-low | – | – |
| 64 | **Janus-Pro 7B** (DeepSeek) | MHA (DeepSeek-LLM-7B base) | RoPE-vanilla | RMS pre | gated SwiGLU | – | full | uniform | seq | SiLU | full | **split-encoder** (SigLIP-L for understanding + separate path for generation) | mid per path | – | – |
| 65 | Mamba 2.8B | SSM (Mamba-1 S6) | – | RMS pre | – | – | SSM state | uniform | seq | SiLU (in SSM) | – | – | – | – | – |
| 66 | Mamba 2 2.7B | SSM (SSD) | – | RMS pre | – | – | SSM state | uniform | seq | SiLU | – | – | – | – | – |
| 67 | **Mamba-3 (1.5B ref)** | **SSM with complex-valued state + MIMO decoding** | – | RMS pre | – | – | SSM state (complex) | uniform | seq | SiLU | – | – | – | – | – |
| 68 | Hymba 1.5B | **Parallel Mamba ‖ attn heads** in one block | NTK | RMS pre | gated SwiGLU | – | SSM + attn mixed | **parallel branches** | seq | SiLU | full | – | – | – | – |
| 69 | Zamba2 2.7B | Mamba-2 + periodic shared attn | RoPE on shared attn | RMS pre | ungated GELU | – | mixed | **periodic shared attn** | seq | GELU | full on attn | – | – | – | – |
| 70 | Jamba v0.1 | Mamba-1 + 1-in-8 attn + 1-in-2 MoE | – on SSM, full on attn | RMS pre | MoE 16/0/top-2 | top-2/0-shared (1:2 MoE layers) | mixed | **attn/MoE periodic** | seq | SiLU | – on SSM | – | – | – | – |
| 71 | RWKV-6 7B | RWKV WKV6 channel-mix | – (time-decay) | LN | RWKV channel-mix | – | RWKV state | uniform | **parallel (ch+t)** | sigmoid+ReLU² | – | – | – | – | – |
| 72 | RWKV-7 1.5B | RWKV-7 (delta-rule) | – | LN | RWKV-7 channel-mix | – | RWKV-7 state | uniform | parallel | sigmoid+ReLU² | – | – | – | – | – |
| 73 | **xLSTM 7B** | **mLSTM + sLSTM matrix-memory + exponential gating** | – (no RoPE no ALiBi) | RMS | gated SwiGLU | – | mLSTM matrix memory | uniform | seq | SiLU | – | – | – | – | – |
| 74 | **MiniMax-Text-01 456B/45.9B-A** | **Lightning Attention 7:1** (87.5% linear / 12.5% softmax) + MoE | NTK on softmax | RMS pre | SwiGLU-MoE | top-2/1-shared | mixed | **7:1 linear:softmax** | seq | SiLU | full on softmax | – | – | – | – |
| 75 | **MiniMax-M3.0 ~230B/10B-A** | **MSA (MiniMax Sparse Attention)** + MoE | YaRN | RMS pre | SwiGLU-MoE | top-k/shared | MSA sparse | dense + MoE | seq | SiLU | full | – | – | – | – |
| 76 | BitNet b1.58 2B | GQA + sub-norms (BitLinear) | RoPE-vanilla θ=5e5 | RMS pre | **ReLU² FFN** + bitlinear | – | GQA + 1.58-bit ternary weights | uniform | seq | **ReLU²** | full | – | – | – | – |
| 77 | OpenELM 3B | **GQA per-layer list** | RoPE-vanilla | RMS pre + QK-norm | SwiGLU + **per-layer ffn[]** | – | per-layer GQA | uniform op | seq | Swish | full | – | – | – | – |
| 78 | **Apple Foundation Model 3.18B** | GQA + **2-block cross-block KV sharing** + 2-bit QAT | RoPE-vanilla | RMS pre + QK-norm | gated SwiGLU | – | mixed (Block-1 full KV, Block-2 reuses Block-1 KV) | **2-block split** | seq | SiLU | full | concat-adapter (ViTDet-L 300M) | mid | **cross-block-shared** | per-layer rank-1 LoRA adapter |
| 79 | **GPT-OSS 20B (5.1B-A)** | GQA + **trained attention sinks** + **GPT-3-style alternating dense/banded-sparse** | LongRoPE | RMS pre | **SwiGLU-MoE 32/0/top-4** with **MXFP4-native weights** | top-4/0-shared 32 routed | mixed (alternating dense/banded) | **alternating dense/banded-sparse** | seq | SiLU | full | – | – | – | – |
| 80 | **GPT-OSS 120B (5.5B-A)** | GQA + sinks + alternating | LongRoPE | RMS pre | SwiGLU-MoE | top-4 | mixed | alternating | seq | SiLU | full | – | – | – | – |
| 81 | Pixtral 12B | GQA + **2D-RoPE in vision** | RoPE-vanilla on text; 2D-RoPE on vision | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full on text | concat-projector (Pixtral-ViT 400M) | high | – | – |
| 82 | Llama 3.2 Vision 11B | GQA + **cross-attention adapter every 4 layers** | RoPE-Llama3 | RMS pre | gated SwiGLU | – | GQA | adapter at every-4 | seq | SiLU | full | **cross-attention-adapter** (image embeds INTO frozen text model) | mid | – | – |
| 83 | InternVL 2.5 8B | GQA per-backbone + InternViT | per backbone | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + **pixel-unshuffle 4×** | low | – | – |
| 84 | **InternVL 3 8B** | GQA + **V2PE (Variable Visual PE)** | V2PE on vision | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + V2PE | elastic | – | – |
| 85 | MiniCPM-V 2.6 (Qwen2-7B) | GQA | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + **perceiver resampler ≤640 tokens** | mid (≤640) | – | – |
| 86 | Molmo 7B-D (Qwen2-7B) | GQA | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + multi-crop | mid | – | – |
| 87 | Molmo 7B-O (OLMo-7B) | MHA + post-norm + QK-RMS full | RoPE-vanilla θ=5e5 | RMS post + QK-norm/full | gated SwiGLU | – | full | uniform | seq | SiLU | full | concat-projector | mid | – | – |
| 88 | Ovis 2.5 9B | GQA (Qwen2.5-9B) + **Visual Embedding Table** | M-RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat (visual vocab tokens are probabilistic; align to text embeddings) | dynamic | – | – |
| 89 | Eagle 2 9B | GQA (per-backbone) + **MoVE multi-vision-encoder** | per backbone | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | **mixture-of-vision-encoders** (SigLIP + ConvNeXt tiled in parallel) | varies | – | – |
| 90 | **MoVE family**: NVILA, Cambrian-1 | GQA (Llama-3 base) | RoPE-Llama3 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + multi-encoder fusion (Cambrian: 4 encoders via cross-attn queries) | scale-then-compress | – | – |
| 91 | **GOT-OCR 2.0 0.58B** | MHA (Qwen-1 0.5B base) | RoPE-vanilla θ=10k | RMS pre | gated SwiGLU | – | full | uniform | seq | SiLU | full | **concat-projector (VitDet 1024² → 256 tokens → linear)** | **low (256/page from 1024²)** | – | – |
| 92 | MinerU2.5 1.2B (Qwen2-0.5B) | MHA | RoPE | RMS pre | gated SwiGLU | – | full | uniform | seq | SiLU | full | concat-projector (NaViT 675M + patch-merger) | low | – | – |
| 93 | olmOCR-2 8B (Qwen2.5-VL-7B) | GQA + M-RoPE | M-RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + GRPO unit-test rewards | mid | – | – |
| 94 | MonkeyOCR-v1.5 3B (Qwen2.5-VL-3B) | GQA + M-RoPE + SRR triplet | M-RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + Structure-Recognition-Relation triplet | mid | – | – |
| 95 | HunyuanOCR 1B (Hunyuan-0.5B) | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + native MM training (no LM-only pretrain) | mid | – | – |
| 96 | PaddleOCR-VL 0.9B (ERNIE-4.5-0.3B) | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + NaViT + Adaptive MLP | low | – | – |
| 97 | Surya OCR 2 0.65B | GQA (Qwen3-style) + QK-norm | RoPE-vanilla θ=1e6 | RMS pre + QK-norm | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + **EfficientViT segformer pre-stage** | mid | – | – |
| 98 | **Moshi 7B** (Helium-1 base + Mimi codec) | GQA + **dual-stream text/audio** | RoPE-vanilla | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | **Mimi neural codec + Temporal Transformer + Depth Transformer** | n/a (audio) | – | – |
| 99 | **Voxtral TTS 4B** | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | TTS adapter | n/a (audio out) | – | – |
| 100 | **Qwen2-Audio 7B** | GQA (Qwen2-7B base) | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector (Whisper-like audio encoder) | n/a | – | – |
| 101 | Cohere Aya 8B | GQA | RoPE-vanilla θ=4e6 | LN pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 102 | **Cohere Command A+ 218B/25B-A** | GQA + MoE | RoPE | LN pre | gated SwiGLU-MoE | top-k/shared | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 103 | TinyLlama 1.1B | GQA | RoPE-vanilla θ=10k | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 104 | MobileLLM 1B | GQA + **block-wise weight sharing** | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 105 | StableLM 3B-4E1T | MHA + partial-rotary 0.25 | partial-rotary 0.25 | LN pre | gated SwiGLU | – | full | uniform | seq | SiLU | **partial-0.25** | – | – | – | – |
| 106 | **EXAONE 3.5 7.8B** | GQA | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 107 | **EXAONE 4.5 33B (VLM)** | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector | mid | – | – |
| 108 | **GLM-5.1 744B/40B-A** | **DSA-derivative sparse attention** | YaRN | RMS pre | SwiGLU-MoE | top-k/shared | DSA-derivative | dense + MoE | seq | SiLU | full | – | – | – | – |
| 109 | **NVIDIA Nemotron 3 Super 120B/12B-A** | Hybrid Mamba-2 + Transformer MoE | RoPE on attn | RMS pre | gated SwiGLU-MoE | top-k/shared | mixed | Mamba/attn periodic + MoE | seq | SiLU | full on attn | – | – | – | – |
| 110 | **NVIDIA Nemotron 3 Ultra 550B/55B-A** | Hybrid Mamba-2 + Transformer + Attention MoE | RoPE on attn | RMS pre | SwiGLU-MoE | top-k/shared | mixed | Mamba/attn periodic + MoE | seq | SiLU | full on attn | – | – | – | – |
| 111 | **Nemotron 3 Nano 4B** | Hybrid Mamba-2 + Transformer + MoE FFN | RoPE on attn | RMS pre | SwiGLU-MoE | top-k/0-shared | mixed | Mamba/attn periodic | seq | SiLU | full on attn | – | – | – | – |
| 112 | **Nemotron 3 Nano Omni** | Hybrid Mamba-Transformer Omni | RoPE on attn | RMS pre | SwiGLU-MoE | top-k | mixed | Mamba/attn periodic | seq | SiLU | full | concat-projector audio+vision | dynamic | – | – |
| 113 | **Liquid LFM2.5-VL-450M** | **LFM (liquid-time-constant) non-Transformer recurrent** | (no RoPE) | RMS | gated SwiGLU | – | LFM state | uniform | seq | SiLU | – | concat-projector (vision adapter) | low | – | – |
| 114 | **Liquid LFM2.5-8B-A1B** | LFM + MoE | (no RoPE) | RMS | gated SwiGLU-MoE | top-k | LFM state | LFM + MoE | seq | SiLU | – | – | – | – | – |
| 115 | **HY-Embodied-0.5 MoT-2B** | **Mixture-of-Transformers** | RoPE | RMS pre | gated SwiGLU | MoT routing | mixed | MoT routing | seq | SiLU | full | concat-projector (embodied vision+space) | n/a | – | – |
| 116 | StarCoder 2 7B | GQA + **dense bias** | RoPE-vanilla θ=1e6 | LN pre | **gated GELU-tanh + bias** | – | GQA + SWA=4096 | uniform | seq | GELU-tanh | full | – | – | – | – |
| 117 | **Hunyuan 0.5B** | GQA | RoPE-vanilla θ=10k | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 118 | **Hunyuan 7B (fast/slow toggle)** | GQA | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 119 | **Tencent Hy-MT2 7B** | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 120 | **Tencent Hy3 Preview 295B/21B-A** | GQA + MoE + **MTP 3.8B head** | YaRN | RMS pre | SwiGLU-MoE | top-k/shared | GQA | dense + MoE | seq | SiLU | full | – | – | – | – |
| 121 | **Xiaomi MiMo-V2.5-Pro 1.02T/42B-A** | GQA + MoE FP8-native | YaRN | RMS pre | SwiGLU-MoE | top-k/shared | GQA | dense + MoE | seq | SiLU | full | concat-projector (729M ViT + audio) | high | – | – |
| 122 | **Moonshot Kimi K2.6 1T/32B-A** | GQA + MoE | YaRN | RMS pre | SwiGLU-MoE | top-k/shared | GQA | dense + MoE | seq | SiLU | full | concat-projector (400M MoonViT) | high | – | – |
| 123 | **Zyphra ZAYA1-8B** | GQA + MoE | RoPE | RMS pre | gated SwiGLU-MoE | top-k/shared | GQA | dense + MoE | seq | SiLU | full | – | – | – | – |
| 124 | Florence-2 232M (BART-style enc-dec) | MHA enc + MHA dec | RoPE | LN | ungated GELU | – | full | enc-dec | seq | GELU | full | encoder-decoder unified prompt-task | unified | – | – |
| 125 | Pixtral 12B + Mistral Small 3.1 24B (concat) | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + 2D-RoPE in vision | high | – | – |
| 126 | LLaVA-NeXT (Mistral 7B base) | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + **AnyRes** dynamic tiling | dynamic | – | – |
| 127 | LLaVA-OneVision 7B (Qwen2-7B base) | GQA | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + AnyRes-9 | dynamic | – | – |
| 128 | Phi-3.5-vision 3.8B | MHA | LongRoPE | RMS pre | gated-fused SwiGLU | – | full | uniform | seq | SiLU | full | concat-projector (CLIP) | mid | – | – |
| 129 | PaliGemma 1 3B | MQA (Gemma 1 2B base) | RoPE-vanilla | RMS pre+post | gated GeGLU | – | MQA hd=256 | uniform | seq | GELU-tanh | full | **concat-prefix-LM** | mid | – | – |
| 130 | PaliGemma 2 3B (Gemma 2 base) | GQA + SWA every-other | RoPE-vanilla | RMS pre+post + softcap | gated GeGLU | – | mixed | 1:1 SWA/full | seq | GELU-tanh | full | concat-prefix-LM | mid | – | – |
| 131 | Vary-toy 1.8B (Qwen-1.8B) | MHA + QKV bias | RoPE | RMS | gated SwiGLU | – | full | uniform | seq | SiLU | full | **vision-vocabulary expansion** (OPT-125M proxy) | low | – | – |
| 132 | mPLUG-DocOwl2 8.1B (LLaMA-2-7B base) | MHA | RoPE-vanilla θ=10k | RMS pre | gated SwiGLU | – | full | uniform | seq | SiLU | full | concat-projector + **High-Resolution DocCompressor 324 tokens/page** | **low (324/page)** | – | – |
| 133 | TextHawk2 7B (InternLM2-7B base) | GQA | dynamic NTK | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + **SPE (Scalable Positional Embeddings)** for variable-res | mid (16× reduced vs v1) | – | – |
| 134 | Dolphin (ByteDance) 322M | MHA (small decoder) | RoPE | LN | ungated GELU | – | full | uniform | seq | GELU | full | concat (Swin enc + heterogeneous anchor prompts) | low | – | – |
| 135 | Nanonets-OCR-s 3B (Qwen2.5-VL-3B) | GQA + M-RoPE | M-RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector + **semantic-tag tokens** | mid | – | – |
| 136 | **Reka Edge 2026 8B** | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 137 | DeepSeek-R1-Distill-Qwen 7B (Qwen2.5-Math-7B base) | GQA | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 138 | DeepSeek-R1-Distill-Llama 8B (Llama-3.1-8B base) | GQA | RoPE-Llama3 f=8 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 139 | **Llama-3.1-Nemotron-Nano 8B** | GQA | RoPE-Llama3 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 140 | **Helium-1 2B** (Kyutai Moshi base) | GQA | RoPE-vanilla θ=1e6 | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 141 | **Mamba-3 (1.5B research)** | SSM complex-state + MIMO | – | RMS pre | – | – | SSM-complex state | uniform | seq | SiLU | – | – | – | – | – |
| 142 | OLMoE 1B-A | MHA | RoPE-vanilla θ=10k | RMS pre (`norm_topk_prob=false`) | SwiGLU-MoE 64/0/top-8 | top-8/0-shared 64 routed | full | MoE all | seq | SiLU | full | – | – | – | – |
| 143 | Codestral Mamba 7B | Mamba-2 (SSD) | – | RMS pre | – | – | SSM state | uniform | seq | SiLU | – | – | – | – | – |
| 144 | MiniCPM 3 4B | MLA q_lora=768 kv_lora=256 qk_nope=64 qk_rope=32 | LongRoPE | RMS pre + μP | gated SwiGLU | – | MLA latent | uniform | seq | SiLU | MLA NoPE/RoPE split | – | – | – | – |
| 145 | **Mistral OCR 2 (open)** (Mistral Small base) | GQA + sinks | RoPE-vanilla | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | concat-projector | mid | – | – |
| 146 | **Magistral 24B** (Mistral reasoning fork) | GQA + sinks | RoPE-vanilla | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | – | – | – | – |
| 147 | Falcon Perception 0.6B | GQA + early-fusion MM | RoPE | RMS pre | gated SwiGLU | – | GQA | uniform | seq | SiLU | full | **early-fusion** for grounding+segmentation | low | – | – |
| 148 | **GLM-4 / ChatGLM 3 / Baichuan 2** (legacy) | MQA/MHA | RoPE | LN/RMS | SwiGLU/GeGLU | – | varies | uniform | seq | varies | full | – | – | – | – |

(Per-row provenance is in §3 by year. The remaining ~30 rows not listed individually inherit axis values from their family — DeepSeek-R1-Distill = backbone family; Qwen3-VL = Qwen3; MiniCPM-Llama3-V = Llama 3 base; InternVL 3.5 sub-models = InternVL 3 base; SmolVLM-256M / SmolVLM2-500M = SmolLM/SmolLM2 base; Eagle 2 1B/2B = backbone-dependent; etc.)

---

## 5. Per-Family Evolution Arcs

This section narrates the architectural evolution within each family. v3 refreshes v2's 15 arcs with new generations and adds further family arcs that v2 did not narrate, for **30 sub-arcs** in §5.1 – §5.30 (v3.1 revision adds §5.31 ChatGLM/GLM, §5.32 Yi, and §5.33 Hunyuan, bringing the total to 33 sub-arcs).

### 5.1 Llama — Feb 2023 → Apr 2025 (LLaMA 1 → Llama 4 Maverick)

**LLaMA 1 7B** (2023-02): MHA, RoPE θ=10000, RMS pre, SwiGLU, SP-32k, untied, ctx 2048. → **LLaMA 2 7B** (2023-07): ctx 2048 → 4096; GQA appeared only at 70B. → **Llama 3 8B** (2024-04): first 7B-class GQA (32/8/128) + Tiktoken 128k + RoPE θ raised 10000 → 500000. → **Llama 3.1 8B** (2024-07): introduces four-parameter `rope_scaling` block; ctx 8192 → 131072. → **Llama 3.2 1B/3B** (2024-09): scales downward; head_dim=64 (1B) / 128 (3B); ties embeddings below 8B. → **Llama 3.3 70B** (2024-12): post-training only. → **Llama 4 Scout 17B-A 16E** (2025-04): native multimodal early fusion + **iRoPE interleaved per-layer NoPE mask** + GQA + MoE (16 routed / **1 shared** / top-1) + 10M ctx via NoPE. Shared expert is not surfaced as a config field — it is hardcoded in `modeling_llama4.py:165` (`self.shared_expert = Llama4TextMLP(config)`), and the routed-MoE output is summed with the shared-expert output at every MoE layer (line 173-174). → **Llama 4 Maverick 17B-A 128E** (2025-04): same iRoPE + 128 routed + 1 shared + 1M ctx. → **Llama 5 600B+ flagship** (2026-04, partial weights gated): "Recursive Self-Improvement"; 5M ctx. Wikipedia indicates the Llama line is being superseded by **Muse Spark** (Meta Superintelligence Labs, 2026-04, closed weights, "Contemplating mode" parallel-agent reasoning).

**Axes flipped:** (1→2→3→3.1→4) MHA → MHA → GQA at 7B → GQA → GQA + iRoPE; SP → SP → Tiktoken-128k → Tiktoken-128k → Tiktoken-200k; θ 10k → 10k → 500k → band-scaled; ctx 2k → 4k → 8k → 131k → 1M-10M; dense → dense → dense → dense → MoE; sequential text → multimodal late-fusion (3.2-Vision) → multimodal early-fusion (Llama 4 native). The Llama 4 jump combines four simultaneous changes — first MoE, first iRoPE, first early-fusion multimodal, first NoPE production. The Llama-3.2-Vision (Sep 2024) cross-attention adapter approach was abandoned by Llama 4 in favor of native early fusion, mirroring Chameleon's earlier (May 2024) design choice.

### 5.2 Qwen — Sep 2023 → Apr 2026 (Qwen 1 → Qwen3-Next → Qwen3.5/3.6/3.7)

**Qwen 1 7B** (2023-09): MHA + QKV bias + RoPE θ=10000. → **Qwen 1.5 7B** (2024-02): GQA hooks; θ → 1e6; ctx 32k. → **Qwen 2 7B** (2024-06): **QKV bias retained** (verified `modeling_qwen2.py:200-202` hardcodes `bias=True` on q_proj/k_proj/v_proj; Qwen2-7B config has no `attention_bias` field); GQA 28/4/128. **QKV bias is finally dropped at Qwen 3** (verified `Qwen3-0.6B/config.json` 2026-06-08: `attention_bias=false`; `modeling_qwen3.py:236-246` wires `bias=config.attention_bias`). → **Qwen 2.5 sweep** (2024-09): 0.5B/1.5B/3B/7B; sub-3B tied. → **Qwen 2.5-VL 3B/7B** (2025-01): **M-RoPE** (T,H,W) + dynamic-resolution ViT. → **Qwen 2.5-Math 7B** (2024-09): backbone of R1-Distill-Qwen. → **Qwen 2.5-Omni 3B/7B** (2025-03/04): **TMRoPE** (T,H,W,frame-idx 4D) + **Thinker-Talker dual decoder** for native speech generation. → **Qwen 3 0.6B/1.7B/4B/8B** (2025-04): **QK-RMSNorm per head_dim PRE-RoPE**; decoupled head_dim (Qwen3-0.6B `hidden/H=64 ≠ head_dim=128`). → **Qwen 3 MoE 30B-A3B** (2025-04): 128/0/top-8 + `norm_topk_prob=true`. → **Qwen 3-VL 8B** (2025-Q4): Qwen3 backbone + Qwen2.5-VL-style ViT. → **Qwen3-Next 80B-A3B** (2025-09): **Gated DeltaNet 3:1 hybrid** (75% linear + 25% softmax) + **ultra-sparse MoE 512+1/top-10** (2.15% activation, the sparsest production MoE to date) + MTP head + 262k native ctx (YaRN to 1M). → **Qwen3.5-Omni Plus/Flash/Light** (2026-04): Hybrid Attention MoE Thinker+Talker. → **Qwen3.6-35B-A3B** (2026-04): refreshed multimodal MoE. → **Qwen3.6-27B** (2026-04): dense 27B coding leader, 1M ctx. → **Qwen3.7-Max** preview (2026-05, weights pending).

**Axes flipped:** QKV bias 1 → 1.5 → 2 kept (Qwen2 retains it) → **3 dropped**; θ 1 → 1.5: 10k → 1e6; QK-norm absent → Qwen 3 PRE-RoPE per-Dh; head_dim coupling dropped at Qwen 3; MoE adopted at Qwen 3 (128/0/top-8 norm_topk) → Qwen3-Next ultra-sparse (512+1/top-10 + Gated DeltaNet linear-attention hybrid); position encoding M-RoPE 3D → TMRoPE 4D; native speech generation introduced at 2.5-Omni; MTP head introduced at Qwen3-Next.

### 5.3 Phi — Jun 2023 → Feb 2025 → Phi-4-multimodal

**Phi-1 1.3B** (2023-06): MHA + parallel attn/FFN + partial rotary 0.5 + GPT-2 vocab + LN + GELU. → **Phi-1.5/Phi-2** (2023-09/12): data scaling. → **Phi-3 mini 3.8B** (2024-04): switches to sequential attn/FFN, RMSNorm pre, full RoPE, SwiGLU **fused gate_up**, MS-SP-32k. → **Phi-3-small 7B** (2024-04): structural outlier (gegelu + LN + BlockSparse + μP). → **Phi-3.5 mini** (2024-08): LongRoPE. → **Phi-3.5-MoE 6.6B-A** (2024-08): 16/0/top-2. → **Phi-3.5 vision** (2024-08): CLIP + Phi-3.5-mini. → **Phi-4 mini 3.8B** (2025-02): GQA + **partial RoPE 0.75** + o200k 200064 vocab + tied. → **Phi-4-mini-flash** (2025-07): hybrid Mamba+attention (Samba-derived). → **Phi-4-multimodal 5.6B** (2025-02): Phi-4-Mini-3.8B frozen backbone + **mixture-of-LoRAs** — vision LoRA (370M) and audio LoRA, with modality-routed selection. First production mixture-of-LoRAs at <6B.

**Axes flipped (1→4):** parallel → sequential sublayer; partial → full RoPE → partial 0.75 at Phi-4 mini; LN → RMS → LN (small only) → RMS; GELU → SwiGLU; SwiGLU separate → fused gate_up; vocab 51k GPT-2 → 32k MS → 100k cl100k → 200k o200k; MHA → BlockSparse (small only) → MHA → GQA → hybrid → mixture-of-LoRAs. Phi is the single most architecturally heterogeneous family in the corpus.

### 5.4 Gemma — Feb 2024 → Apr 2026 (Gemma 1 → Gemma 4)

**Gemma 1 2B/7B** (2024-02): MQA (2B) / MHA (7B) + dual norm + GeGLU + head_dim=256 + 256k SP. → **Gemma 1.1** (2024-04): post-training. → **Gemma 2 2B/9B** (2024-07): MQA → GQA; SWA every other (W=4096); softcap (50, 30); `query_pre_attn_scalar=256`. → **Gemma 3 1B/4B** (2025-03): 1:1 → **5:1 SWA:full**; dual RoPE base (10000 local / 1000000 global); SWA shrunk 4096 → 512/1024; vocab 256k → 262144 + image tokens; **softcap dropped entirely** (both `final` and `attn` softcapping null). → **Gemma 3n E2B/E4B** (2025-05): **Matformer** elastic-width. → **Gemma 4 family** (2026-04-02 launch):

- **E2B/E4B** (edge): introduces **Per-Layer Embeddings** (256-dim residual at every layer, paged to flash on mobile); **cross-layer KV sharing** (`num_kv_shared_layers=20` of 35 for E2B); **`global_head_dim=512`** (vs local 256); **partial-rotary p-RoPE** on global only (`partial_rotary_factor=0.25` — first 25% of head_dim channels rotated, remainder pure semantic); 4:1 SWA:full for E2B specifically (5:1 for the rest); SWA shrunk 1024 → 512 at edge; **`use_double_wide_mlp=True`**; fixed-scale QK-norm (local 0.9916 / global 1.0228) replaces `1/sqrt(head_dim)`.
- **12B Unified**: **encoder-free** unified multimodal — raw image patches and raw audio waveforms go through linear projections directly into the LLM embedding space. Single decoder-only transformer handles all modalities. 5:1 SWA:full (last global). **`attention_k_eq_v=True` on global layers** (the global K and V projections share weights, paired with doubled `global_head_dim`).
- **26B-A4B (MoE)**: 128 routed + 1 shared, top-k=8, `moe_intermediate_size=704`. First Gemma MoE. **K=V on global.**
- **31B**: dense flagship, 60 layers, d=5376. **K=V on global.**
- Restored **final logit softcap = 30.0** (Gemma 3 had it dropped); `attn_logit_softcapping` stays null.
- Apache 2.0 license (changed from Gemma 1-3 Gemma Terms).
- **2026-06-05 QAT addendum**: 4 QAT checkpoint formats per size (unquantized QAT, GGUF Q4_0, Compressed-Tensors, mobile 4-bit + 2-bit token-generation layers). E2B 1 GB mobile / 3 GB RAM.

**Important size-dependent correction (per `research/issues/10-gemma4-investigation.md` and the `02-layer-sources.v3.md` peer):** `attention_k_eq_v=True` on Gemma 4 12B / 26B-A4B / 31B (on global layers). **`attention_k_eq_v=False` on E2B**, which instead uses `num_kv_shared_layers=20` (cross-layer sharing) as its KV-cost-reduction mechanism. The size-dependent split is: edge sizes (E2B/E4B) use cross-layer KV sharing **instead of** unified K=V on global layers; server sizes (12B/26B/31B) use K=V global without cross-layer sharing. This is a non-obvious axis split that the layer IR must respect: an operator that hardcodes "all Gemma 4 sizes have K=V" or "all Gemma 4 sizes have cross-layer KV sharing" will silently corrupt half the family.

**Axes flipped (1→4):** MQA→GQA at 2B; 1:1 SWA→5:1 SWA→4:1 at E2B; softcap on→off→half-restored; single θ→dual θ→dual θ + partial rotary 0.25; vocab 256k→262k+image+audio+video special tokens; multimodal: text-only → text+image (PaliGemma) → text+image+audio (Gemma 3n + Gemma 4 E2B/E4B) → text+image+audio+video encoder-free (Gemma 4 12B); KV cache: per-layer → per-layer with same-block sharing (E2B) or K=V global (12B+); PLE introduced at E-variants. The Gemma 4 jump is the single largest single-generation architectural delta in the entire corpus.

### 5.5 DeepSeek — Nov 2023 → Apr 2026 (DeepSeek LLM → V4-Pro)

**DeepSeek LLM 7B** (2023-11): MHA + RoPE θ=10000 + RMS pre + SwiGLU + custom 102400 BPE. → **DeepSeek-Math 7B** (2024-02): same arch + math RLHF. → **DeepSeek-Coder 6.7B** (2023-11): 32256 vocab + ctx 16384. → **DeepSeek-V2 / V2-Lite 16B-A2.4B** (2024-05/06): **MLA** introduced (`kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`); MoE 64+2/top-6 (V2-Lite); **softmax router with aux-loss balancing** (verified `DeepSeek-V2-Lite/config.json` 2026-06-08: `scoring_func="softmax"`, `topk_method="greedy"`, `aux_loss_alpha=0.001`). → **DeepSeek-Coder-V2-Lite** (2024-06): code-specialized V2-Lite. → **DeepSeek-V3 671B/37B-A** (2024-12, oos): 256 routed + 1 shared + top-8 + **aux-loss-free routing introduced here** (verified `DeepSeek-V3/config.json` 2026-06-08: `scoring_func="sigmoid"`, `topk_method="noaux_tc"`, `aux_loss_alpha` field removed); FP8 training. → **R1 / R1-Distill family** (2025-01): same arch as V3 + RL recipe; distills are backbone-dependent (Llama-3.1-8B / Qwen2.5-Math). → **DeepSeek-VL2** (2024-12): MLA + MoE + SigLIP + dynamic tiling. → **Janus-Pro 1B / 7B** (2025-01): DeepSeek-LLM base + **split SigLIP-L** for understanding vs generation paths (unified transformer body). → **DeepSeek-V3.1** (2025-08): incremental V3 update. → **DeepSeek-V3.2-Exp** (2025-09): **DSA (DeepSeek Sparse Attention)** with **Lightning Indexer** — first production-grade learned-scorer sparse attention. → **DeepSeek-V3.2** stable (2025-12). → **DeepSeek-OCR 3.4B/A570M** (2025-10): **DeepSeek3B-MoE-A570M** (active 570M) decoder + **DeepEncoder** (SAM-B 80M + CLIP-L 300M serial + **16× conv compressor**) → 100 vision tokens/page (ultra-low). → **DeepSeek-OCR-2** (2026-Q1): + **Visual Causal Flow mask** (vision tokens bidirectional within a page, causal cross-page — a new mask topology). → **DeepSeek-V4-Pro 1.6T/49B-A** (2026-04): **CSA + HCA hybrid sparse** (Compressed Sparse Attention + Heavily Compressed Attention), 1M ctx, 27% FLOPs / 10% KV cache of V3.2 at 1M. → **DeepSeek-V4-Flash 284B/13B-A** (2026-04): same CSA+HCA with smaller MoE (128+1/top-8). DeepSeek **R2 not released** as of 2026-06-06 per reports.

**Axes flipped (LLM → V4):** MHA → MLA → MLA + DSA → MLA + CSA + HCA; head_dim symmetry q=k=v dropped at V2 (asymmetric q/k_nope/k_rope/v); aux-loss → aux-loss-free at V3; routing experts 64 → 256 → 256 → 256; shared 2 → 1; FP8 training at V3; OCR DeepEncoder serial dual-tower at OCR; Visual Causal Flow at OCR-2; sparse-attention learned-scorer at V3.2; dual-sparse (CSA + HCA) at V4. DeepSeek is now the source of both the "MoE + MLA" canonical design and the "sparse attention learned-scorer" canonical design.

### 5.6 Mistral — Sep 2023 → Apr 2026 (Mistral 7B v0.1 → Medium 3.5)

**Mistral 7B v0.1** (2023-09): GQA + **SWA=4096** + rope_theta=10000 + vocab=32000 (verified `Mistral-7B-v0.1/config.json` 2026-06-08). → **Mistral 7B Instruct v0.2** (2024-03): **SWA dropped** (`sliding_window=null`) **AND** **rope_theta 10000 → 1000000** in the same release; ctx 32k; vocab still 32000 (verified `Mistral-7B-Instruct-v0.2/config.json` 2026-06-08). → **Mistral 7B v0.3** (2024-05): the *only* config delta vs v0.2 is **vocab 32000 → 32768 (Tekken expansion)**; rope_theta and sliding_window unchanged (verified `Mistral-7B-v0.3/config.json` 2026-06-08). → **Mistral Nemo 12B** (2024-07): 131k Tekken (TK131). → **Mathstral 7B** (2024-07), **Codestral Mamba 7B** (2024-07). → **Ministral 3B / 8B** (2024-10): **interleaved SWA** (full/SWA alternation) — third alternation pattern. → **Pixtral 12B** (2024-09): Mistral Nemo + **2D-RoPE in vision tower**. → **Mistral Small 3 24B** (2025-01): introduces **trained attention sinks**. → **Mistral Small 3.1 24B** (2025-03): adds Pixtral vision encoder; 128k ctx. → **Mistral Small 3.2 24B** (2025-06): text quality refinement. → **Magistral** (2025-Q2): Mistral reasoning fork. → **Mistral OCR 2 (open) / Mistral OCR 3 (closed)** (2025-Q3-Q4). → **Mistral Small 4 119B/15B-A** (2026-03): unifies instruct + Magistral + Devstral + Pixtral in a single MoE; eagle-head speculative decoding head shipped in weights. → **Mistral Medium 3.5 128B dense** (2026-04): absorbs Magistral + Devstral 2 into one set of weights with configurable reasoning effort; 256k ctx. → **Voxtral TTS 4B** (2026-03): open TTS; 5-sec zero-shot voice cloning; 90ms latency.

**Axes flipped:** SWA on (v0.1) → off at v0.2 → off (v0.3) → interleaved SWA (Ministral) → SWA off (Small 3.x); θ 10k → 1e6 **at v0.2** (paired with the SWA drop, not split across v0.2/v0.3 as v3 originally reported); vocab 32000 → 32768 at v0.3; vocab 32k SP → Tekken → TK131; trained sinks at 3.x; mixture-of-task-LoRAs unification at Small 4 / Medium 3.5; multimodality added (Pixtral 12B → Small 3.1); reasoning fork (Magistral) → reasoning absorbed (Medium 3.5); TTS family (Voxtral) added.

### 5.7 OLMo — Feb 2024 → 2025 (OLMo 1 → OLMo 2 → Molmo)

**OLMo 1 7B/1B** (2024-02/04): MHA + RoPE θ=10000 + RMS pre + SwiGLU + GPT-NeoX 50304. → **OLMo 2 7B / 13B** (2024-11): switches **pre-norm → post-norm**; **QK-RMSNorm over full head channels** (vs Qwen 3's per-Dh); θ 10000 → 500000; vocab → Tiktoken 100352. → **OLMoE 1B-A** (2024-09): MHA + MoE 64/0/top-8 + `norm_topk_prob=false`. → **Molmo 7B-D / 7B-O** (2024-09): VLM line — 7B-D uses Qwen2-7B base, 7B-O uses **OLMo-7B-1024 base** (the only post-norm + QK-RMS-full VLM). → **MolmoE 1B** (2024-09): OLMoE base. → **olmOCR / olmOCR-2** (2025-02 / 2025-10): OCR-LLM line on Qwen2-VL-7B / Qwen2.5-VL-7B base with GRPO unit-test-reward training. OLMo 3.1 (late 2025) and DR Tulu (recipe) are noted but no OLMo 4 in window.

**Axes flipped:** pre-norm → post-norm at OLMo 2; QK-norm absent → per-full-channels; θ 10k → 5e5; vocab GPT-NeoX → Tiktoken; family branching → VLM (Molmo) → OCR-LLM (olmOCR/olmOCR-2). OLMo is the cleanest pre→post-norm flip in the corpus.

### 5.8 BitNet → Falcon-Edge — Quantization-Native Architecture Line

**BitNet b1.58 2B-4T** (2024-11 paper / 2025-04 HF): GQA + 1.58-bit ternary weights + **ReLU² FFN**; trained from scratch with bitlinear layers. → **Falcon-Edge 1B / 3B** (2026-05): first **retrainable** 1.58-bit BitNet that ships pre-quantized weights enabling further fine-tune and continued pre-training; previously BitNet supported inference only. → **GPT-OSS 20B / 120B** (2025-08): **MXFP4-native trained weights** — a different quantization regime (4-bit MXFP4 rather than 1.58-bit ternary), the only mainstream model shipping MXFP4 weights natively in 2025.

**Axes flipped:** activation SwiGLU → ReLU² (BitNet) at the quantization frontier; quantization fp16/bf16 → 1.58-bit ternary → MXFP4 native → 1.58-bit retrainable. Falcon-Edge and GPT-OSS together establish "quantization is an architectural choice with downstream effects on activation, normalization sub-blocks, and embedding behavior".

### 5.9 Falcon — May 2023 → 2026 (Falcon 7B → Falcon-Edge)

**Falcon 7B** (2023-05): MQA + parallel attn/FFN. → **Falcon Mamba 7B** (2024-08): pure Mamba-1. → **Falcon 3 1B/3B/7B/10B** (2024-12): GQA + head_dim=256 + θ=1000042. → **Falcon-H1 0.5B/1.5B/3B/7B** (2025): **Mamba-2 + attention parallel-head hybrid** (sixth hybrid topology after Jamba seq, Zamba2 periodic, Hymba parallel, Samba seq, Granite 4 9:1 seq). → **Falcon-Edge 1B / 3B** (2026-05): retrainable 1.58-bit BitNet-style. → **Falcon Perception 0.6B** (2026-04/05): early-fusion multimodal grounding+segmentation.

**Axes flipped:** MQA → Mamba-1 → GQA hd=256 → Mamba-2+attn parallel-head → retrainable 1.58-bit → early-fusion multimodal. Falcon is the family that abandoned attention (Falcon Mamba), re-adopted it (Falcon 3), then re-abandoned full-precision (Falcon-Edge). Three full direction changes in 3 years.

### 5.10 SmolLM — Jul 2024 → Jul 2025 (SmolLM → SmolLM3 → SmolVLM)

**SmolLM v1** (2024-07): GQA + RMS pre + SwiGLU + SC2 49152 + tied. → **SmolLM2 135M/360M/1.7B** (2024-11): GQA at small sizes, **MHA at 1.7B** (the size where MHA returns). → **SmolLM3 3B** (2025-07): **NoPE every 4th layer** (`no_rope_layers=[1,1,1,0]*L/4`); θ 1.3e5 → 5e6; vocab SC2 → LL3 (128256); ctx 8k → 65k. → **SmolVLM / SmolVLM2** (2024-11 / 2025-02): Idefics3-derived; uses SmolLM/SmolLM2 text tower + 9× pixel-shuffle vision compression (vs 4× in Idefics3); RoPE base extended at the VLM stage 10k → 273k.

**Axes flipped:** GQA → MHA → GQA again (1.7B is the MHA outlier); θ 1e5 → 5e6; SC2 → LL3; NoPE introduction at v3. SmolLM3's NoPE pattern is the first widely-shipped "per-layer apply_rope" mask in the public corpus — Llama 4 Scout's iRoPE is the production-scale follow-on.

### 5.11 Granite — Oct 2024 → Apr 2026 (Granite 3.0 → Granite 4.1)

**Granite 3.0 2B** (2024-10): GQA + four μP scalars (embedding=12, attention=1/64, residual=0.22, logit=8 or 16) + SC2 49152 + tied + ctx 4096 + θ=10000. → **Granite 3.1 2B** (2024-12): θ 10000 → 5e6, ctx 4096 → 131072. → **Granite 3.3 2B/8B** (2025-04): θ → 1e7. → **Granite 4.0 H-Micro 3B / H-Tiny 7B-A1B / H-Small 32B-A9B** (2025-10): **9:1 sequential Mamba-2 : attention hybrid** (mostly Mamba-2, attention every 10th layer). The 9:1 ratio is a fifth ratio after Jamba 1:8 attn periodic, Zamba2 periodic shared, Hymba parallel, Falcon-H1 parallel-head, Phi-4-mini-flash Samba. **H-Tiny 7B-A1B is the fifth MoE pattern** (32 routed + 0 shared + top-4). 70%+ RAM reduction at long context. → **Granite 4.1 3B/8B/30B** (2026-04): drops the 9:1 hybrid in favor of dense decoders; the dense 4.1-8B matches the prior 32B-MoE generation. → **Granite Vision 4.1 4B**, **Granite Speech 4.1 AR-2B / NAR-2B**, **Granite Guardian 4.1** (2026-04): VLM, dual ASR (autoregressive + non-autoregressive edit), and safety classifier. The NAR-2B is one of the only non-autoregressive speech heads in production at <8B.

**Axes flipped:** θ 10000 → 5e6 → 1e7 alone (no rope_scaling block); dense → hybrid 9:1 → back to dense; μP scalars stable throughout; MoE adopted at 4.0 H-Tiny; new modalities (Vision, Speech AR/NAR, Guardian) at 4.1. Granite is the family showing that "base θ extension only" is a viable 32× context recipe, and that NAR speech decoders can be productive at 2B.

### 5.12 MiniCPM — Feb 2024 → Aug 2024 (MiniCPM 1 → MiniCPM-V/o 2.6)

**MiniCPM 1 / 2 2.4B** (2024-02/04): MHA + RoPE θ=10000 + RMS pre + μP (scale_emb=12, scale_depth=1.4). → **MiniCPM 3 4B** (2024-09): **MLA** (`q_lora_rank=768`, `kv_lora_rank=256`, `qk_nope_head_dim=64`, `qk_rope_head_dim=32`) — first non-DeepSeek production MLA + LongRoPE + vocab 122753 → 73448. → **MiniCPM-Llama3-V 2.5** (2024-05): VLM line on Llama-3-8B base. → **MiniCPM-V 2.6** (2024-08): VLM on Qwen2-7B + SigLIP-400M + perceiver resampler → ≤640 tokens / 1.8M-pixel image. → **MiniCPM-o 2.6** (2024-12): triple modality — adds Whisper-like audio encoder; first triple-modality MiniCPM.

**Axes flipped:** MHA → MLA at v3; vocab shrinks; μP retained; backbone family added (Llama-3 → Qwen2); modality extension audio added at -o.

### 5.13 OpenELM — Apr 2024 (single shot)

**OpenELM 270M / 450M / 1.1B / 3B** (2024-04): per-layer scaling recipe — `num_query_heads[]`, `num_kv_heads[]`, and `ffn_multipliers[]` are per-layer lists; tied embeddings; QK-norm; **head_dim=64 for 270M / 450M / 1.1B and head_dim=128 for the 3B variant** (verified `apple/OpenELM-270M/config.json` and `apple/OpenELM-3B/config.json` 2026-06-08), fixed across layers within a size though Q/KV head counts vary. **Per OpenELM paper (arXiv:2404.14619 §2.1, Eq. 1)**, the per-layer schedule is **linear interpolation** between `(α_min, α_max)` for attention head count and `(β_min, β_max)` for FFN multiplier — i.e. `α^i = α_min + (α_max − α_min)·i/(N−1)` — NOT Bayesian optimization. The hyper-parameters used in pre-training (Appendix A): `α_min, α_max = 0.5, 1.0` and `β_min, β_max = 0.5, 4.0`. OpenELM is the only family in the corpus whose `shape` varies across depth at training time.

### 5.14 Hybrid SSM family — Nine Topologies in 30 Months

(1) **Jamba v0.1** (2024-03): sequential 1-in-8 attention + 1-in-2 MoE periodic substitution. (2) **Zamba2** (2024-10/11): sequential periodic shared attention block. (3) **Hymba 1.5B** (2024-11): parallel Mamba ‖ attention heads within one block. (4) **Phi-4-mini-flash** (2025-07): sequential Samba-derived hybrid. (5) **Falcon-H1 0.5B–7B** (2025): parallel-head Mamba-2 + attention. (6) **Granite 4.0 H-Micro/H-Tiny/H-Small** (2025-10): sequential 9:1 Mamba-2 : attention. (7) **Qwen3-Next 80B-A3B** (2025-09): sequential 3:1 Gated DeltaNet : softmax-attention (linear:softmax = 75:25). (8) **MiniMax-Text-01 / M2 / M3** (2025-01 / 2026-Q1 / 2026-06): sequential 7:1 Lightning Attention : softmax (linear:softmax = 87.5:12.5) → M2 reverted to full attention (the "feature abandoned" story) → M3 reintroduces sparse via **MSA** (MiniMax Sparse Attention). (9) **NVIDIA Nemotron 3 Nano 4B / Super 120B-A12B / Ultra 550B-A55B / Nano Omni** (2025-11 / 2026-03 / 2026-06 / 2026-04): sequential hybrid Mamba-2 + Transformer + MoE FFN — Ultra at 550B-A55B is the largest open hybrid SSM-attention MoE to date.

Nine simultaneous hybrid topologies in the 2024-2026 window. The diversity has not converged on a single ratio or topology; production deployments are split across all eight active patterns.

### 5.15 StarCoder — May 2023 → Feb 2024

**StarCoder 1** (2023-05): MQA + absolute learned + LN + bias on QKV+MLP + gated GELU-tanh + BigCode 49152. → **StarCoder 2 3B/7B** (2024-02): MQA → GQA; absolute → RoPE θ=1e6; SWA=4096; **keeps `use_bias=true`** (only modern decoder family with biases); keeps gated GELU-tanh; keeps LayerNorm.

### 5.16 RWKV — 2024 → 2025 (RWKV-6 Finch → RWKV-7 Goose)

**RWKV-6 Finch 1.6B/7B** (2024-08): WKV6 channel-mix + time-mix; no RoPE no ALiBi (time-decay). → **RWKV-7 Goose 1.5B** (2025-01): **delta-rule update** in the WKV channel.

### 5.17 xLSTM (new arc)

**xLSTM 7B (NXAI)** (2025-03 HF release): mLSTM + sLSTM blocks with matrix memory and exponential gating; uses neither RoPE nor ALiBi (recurrence encodes position). The only competitive non-Transformer non-SSM LM at 7B as of cutoff. arXiv:2503.13427.

### 5.18 Mamba (new arc) — Dec 2023 → Mar 2026

**Mamba 2.8B** (2023-12 paper / 2024 HF): Selective SSM (S6). → **Mamba 2 2.7B** (2024-08): State-Space Duality (SSD). → **Mamba-3 reference scales (180M–1.5B)** (2026-03): **complex-valued state + MIMO decoding** + refined SSM discretization. Accepted at ICLR 2026 (arXiv:2603.15569). The most directly relevant SSM update of the year.

### 5.19 Apple Foundation Model (new arc, promoted)

**Apple Foundation Model 3.18B on-device** (announced WWDC 2024 → deployed 2025 → tech report arXiv:2507.13575, Jul 2025): **2-block cross-block KV sharing** (Block-1 carries 62.5% of layers with full KV; Block-2 carries 37.5% and reuses Block-1's K/V) + 2-bit QAT + ViTDet-L 300M vision adapter + per-layer rank-1 LoRA adapters. This is a **distinct A18 variant** from Gemma 4's same-block KV sharing — Apple reuses across one block transition, Gemma 4 E2B/E4B reuses within the same block-type sub-sequence (and Gemma 4 12B/26B/31B uses K=V global instead of cross-layer sharing).

### 5.20 OCR-LLM family (new arc) — Sep 2024 → 2026

**Donut** (NAVER 2021) → **Pix2Struct** (Google 2023) → **Nougat** (Meta 2023-08) → **Kosmos-2.5** (Microsoft 2023-09) → **Vary / Vary-toy** (Megvii 2023-12 / 2024-01): vision-vocabulary expansion via OPT-125M proxy. → **DocPedia** (USTC + JD 2023-11): frequency-domain (DCT) visual processing. → **UReader** (Alibaba 2023-10): shape-adaptive cropping. → **TextHawk / TextHawk2** (Tencent 2024-04 / 2024-10): Scalable Positional Embeddings + 16× visual token compression at v2. → **mPLUG-DocOwl 1.5 / DocOwl2** (Alibaba 2024-03 / 2024-09): H-Reducer; 324 tokens/page multi-page document compression. → **Florence-2** (Microsoft 2024-06): BART-style encoder-decoder; OCR is a prompt mode. → **GOT-OCR 2.0** (StepFun+UCAS 2024-09): Qwen-0.5B + VitDet 1024² → 256 tokens. The OCR-2.0 paradigm origin. → **DeepSeek-VL2** (DeepSeek 2024-12): MLA + MoE + SigLIP dynamic tiling. → **MinerU2** (Shanghai AI Lab 2025-Q1): two-stage layout → recognition. → **olmOCR / olmOCR-2** (Allen AI 2025-02 / 2025-10): Qwen2-VL-7B / Qwen2.5-VL-7B SFT + GRPO unit-test rewards. → **Nanonets-OCR-s / OCR2** (Nanonets 2025-05 / 2025-12): Qwen2.5-VL-3B + semantic-tag SFT. → **MonkeyOCR / MonkeyOCR v1.5** (Huazhong UST 2025-06 / 2025-11): SRR (Structure-Recognition-Relation) triplet paradigm. → **RolmOCR** (Reducto 2025-07): Qwen2.5-VL-7B speed-tuned. → **MinerU2.5 1.2B** (Shanghai AI Lab 2025-09): Qwen2-Instruct-0.5B + NaViT-675M; two-stage coarse-to-fine; outperforms Gemini 2.5 Pro on OmniDocBench. → **DeepSeek-OCR 3.4B/A570M** (DeepSeek 2025-10): **first MoE-decoder OCR LLM**; DeepEncoder (SAM-Base + CLIP-Large serial + 16× conv compressor) → **100 vision tokens/page** (ultra-low compression budget); the seminal "context optical compression" paradigm. → **PaddleOCR-VL 0.9B** (Baidu 2025-10): **ERNIE-4.5-0.3B** base + NaViT + Adaptive MLP. **Smallest production LM-decoder for OCR.** → **Surya OCR 2 0.65B** (Datalab 2025-10): Qwen3-style decoder + EfficientViT segformer pre-stage. → **MonkeyOCR v1.5** (2025-11): + robust pattern handling on Qwen2.5-VL-3B base. → **HunyuanOCR 1B** (Tencent 2025-11): **Hunyuan-0.5B native LLM** + **native multimodal training** (no LM-only pretrain stage). → **DeepSeek-OCR-2** (DeepSeek 2026-Q1): + **Visual Causal Flow** mask (vision tokens bidirectional within a page, causal cross-page). → **Mistral OCR 2 (open) / 3 (closed)** (Mistral 2025-Q3 / 2025-12).

**Common pattern:** small decoder (≤2B preferred, 0.3B–7B range) + specialized high-resolution vision encoder + aggressive vision-token compression (100 tokens/page is now the SOTA design point). The vision-token compression budget (A17) is the single most discriminating axis within this sub-family.

### 5.21 VLM text-tower family — 2024 → 2026 (new arc)

The VLM text-tower extension goes through seven fusion topologies:

1. **Concat-prefix-LM** (PaliGemma 1/2): image tokens prepended to the text sequence; bidirectional attention within the image prefix.
2. **Concat-projector** (LLaVA-NeXT, Qwen 2-VL, Qwen 2.5-VL, MiniCPM-V, InternVL 2/2.5/3/3.5, Molmo, Pixtral, SmolVLM, NVILA, Cambrian-1, Eagle 2, Ovis 2/2.5, Phi-3.5-vision, Florence-2, Phi-3-vision, MinerU2.5, GOT-OCR2, olmOCR, MonkeyOCR, HunyuanOCR, PaddleOCR-VL, Surya 2, Granite Vision 4.1, Gemma 4 E2B/E4B/26B-A4B/31B encoder-based): vision encoder output → linear (or MLP) projector → concatenated as soft tokens; unidirectional attention throughout.
3. **Cross-attention-adapter** (Llama 3.2 Vision 11B/90B): cross-attention layers feed image embeddings INTO frozen text LM at every 4 layers. Distinct from concat — image embeddings are not in the residual stream, they enter via separate cross-attn projections.
4. **Early-fusion native** (Chameleon, Llama 4 Scout/Maverick, Falcon Perception): image tokens are regular self-attention tokens, trained natively from scratch; no separate text-only pretrain stage.
5. **Mixture-of-LoRAs** (Phi-4-multimodal): vision LoRA and audio LoRA are routed adapters over a frozen Phi-4-mini backbone; modality-routed selection.
6. **Split-encoder** (Janus-Pro 1B/7B): separate SigLIP-L paths for understanding vs generation; unified transformer body.
7. **Encoder-free** (Gemma 4 12B Unified): raw image patches and raw audio waveforms go through linear projections directly into the LLM embedding space; no separate encoder.

The choice has downstream effects on KV cache (cross-attention requires extra KV per adapter layer), context-extension (early-fusion needs the position encoding to span vision and text tokens jointly), and quantization (mixture-of-LoRAs needs per-LoRA quantization scales).

### 5.22 Audio-LM family (new arc)

**Moshi 7B (Kyutai)** (2024-09): dual-stream inner-monologue text + audio stream; Temporal Transformer at codebook level + small Depth Transformer; Helium-1 2B is the standalone text base. → **Qwen2-Audio 7B** (2024-08): Whisper-like audio encoder + Qwen2-7B. → **Qwen2.5-Omni 3B/7B** (2025-03/04): **TMRoPE** (T,H,W,frame-idx) + **Thinker-Talker dual decoder**. → **Voxtral TTS 4B** (Mistral 2026-03): open TTS. → **Granite Speech 4.1 AR-2B / NAR-2B** (IBM 2026-04): autoregressive ASR-translate + non-autoregressive speech edit. → **StepFun Step-Audio / Step-Audio-EditX** (2025): audio LLM with roleplay-specific RLHF. → **Helium-1 2B** (Kyutai 2025-01): Moshi standalone text base.

### 5.23 EXAONE (LG, new arc)

**EXAONE 3.5 2.4B / 7.8B / 32B-oos** (2024-12): Korean-Japanese bilingual decoder with custom 102400 vocab; standard GQA+SwiGLU+RoPE (Llama-shaped). → **EXAONE 4.5 33B** (2026-04): VLM; STEM-tuned; Korean + Spanish/German/Japanese/Vietnamese. The only Korean-trained family in the corpus.

### 5.24 Tencent Hunyuan (new arc) — Aug 2025 → Jun 2026

**Hunyuan 0.5B / 1.8B / 4B / 7B** (2025-08): GQA+SwiGLU+RoPE with **fusion-reasoning toggle** (fast vs slow thinking inside one model); native 256k ctx at 7B. → **Hunyuan-0.5B** is used as the base for HunyuanOCR. → **HunyuanOCR 1B** (2025-11): native MM training. → **Hunyuan-A13B-Instruct 80B/13B-A** (2026-04): fine-grained MoE + 256k. → **HY-Embodied-0.5 MoT-2B** (2026-04): Mixture-of-Transformers embodied. → **Hy3 Preview 295B/21B-A** (2026-04): fast/slow hybrid + 3.8B MTP. → **Hy-MT2 1.8B/7B/30B-A3B** (2026-05): translation sweep.

### 5.25 Liquid AI LFM (new arc)

**LFM2.5-VL-450M** (2026-04) and **LFM2.5-8B-A1B** (2026-05): non-Transformer LFM (liquid-time-constant) recurrent backbone + MoE in the 8B-A1B variant. The only non-Transformer non-SSM non-RWKV non-xLSTM recurrent family with public weights at this scale.

### 5.26 NVIDIA Nemotron (new arc)

**Llama-3.1-Nemotron-Nano 8B** (2025-03): FP4-aware reasoning RL on Llama-3.1-8B. → **Nemotron 3 Nano 4B** (2025-11): hybrid Mamba-2 + Transformer + MoE FFN. → **Nemotron 3 Super 120B/12B-A** (2026-03): hybrid Mamba-2 + Transformer MoE for agentic reasoning; FP8 native. → **Nemotron 3 Nano Omni** (2026-04): vision + speech + language. → **Nemotron 3 Ultra 550B/55B-A** (2026-06): largest open hybrid SSM-attention MoE.

### 5.27 Zhipu / Z.AI GLM (new arc)

**GLM-4-9B** (2024-06): MQA legacy. → **GLM-5 / GLM-5V-Turbo / GLM-5.1** (2026-02 / 2026-04): **DSA-derivative sparse attention**; 744B/40B-A; post-trained for 8-hour autonomous agentic loops; 200k ctx.

### 5.28 Cohere — Aya → Command A+

**Cohere Aya 8B Expanse** (2024-12): GQA + LN pre + RoPE θ=4e6 + tied at 8B (counterexample to "tied below 3B"). → **Command R7B** (2024-12). → **Command A+ 218B/25B-A** (2026-05): first Cohere MoE; Apache 2.0; W4A4 lossless quantization fits 2× H100.

### 5.29 MiniMax (new arc)

**MiniMax-Text-01 456B/45.9B-A** (2025-01): **Lightning Attention 7:1** (linear:softmax). → **MiniMax-M2 / M2.5 / M2.7** (2026-01 → 2026-03): reverted to full attention. → **MiniMax-M3.0** (2026-06): introduces **MSA (MiniMax Sparse Attention)** — new sparse-attention variant; 1M ctx.

**Axis story:** Lightning Attention adopted (M-Text-01) → abandoned (M2) → re-introduced as different sparse variant (M3 MSA). The "feature abandoned and re-adopted" arc in 18 months.

### 5.30 MobileLLM (preserved arc)

**MobileLLM 125M / 1B** (Meta 2024-10): GQA + deep-narrow + embedding tying + **block-wise weight sharing** (same Q/K/V/MLP weights tied across adjacent layers).

### 5.31 ChatGLM 2 / 3 → GLM-4 → GLM-4.5/4.6 → GLM-5.1 (Zhipu AI / Z.AI / THUDM)

Zhipu AI's open-weight GLM family is the longest-running Chinese-origin decoder-LM lineage in the corpus (six generations across 36 months). The family carries a custom `chatglm` `model_type` through GLM-4-9B-Chat and only switches to a Llama-style `Glm4MoeForCausalLM` `model_type` at GLM-4.5+. **ChatGLM2-6B** (2023-06): MQA via `multi_query_attention=true` with `multi_query_group_num=2` (i.e., 32 query heads → 2 K/V groups), hidden_size=4096, num_layers=28, ffn_hidden_size=13696, padded_vocab=65024, `add_qkv_bias=true` while `add_bias_linear=false`, RMSNorm post-layer, `original_rope=true` (GLM's partial-rotary on first half of head channels), seq_length=32768 (verified `THUDM/chatglm2-6b/config.json` 2026-06-08). → **ChatGLM3-6B** (2023-10): identical config except seq_length 32768 → 8192 (shorter at base) and updated tokenizer/training data; no architectural delta (verified `THUDM/chatglm3-6b/config.json`). → **GLM-4-9B-Chat** (2024-06): still the `chatglm` `model_type` and MQA group_num=2, but scales to 40 layers, padded_vocab 65024 → 151552 (3× — switches to Tiktoken-style tokenizer), seq_length 8192 → 131072, **`rope_ratio=500`** field used for context extension on top of the existing partial-rotary (verified `THUDM/glm-4-9b-chat/config.json`); MQA legacy retained. → **GLM-4.5 / GLM-4.6** (zai-org rebrand, 2025-07 / 2025-Q4): the family transitions to a **Llama-style `glm4_moe` `model_type`** — **355B/32B-A MoE** with `n_routed_experts=160`, `n_shared_experts=1`, `num_experts_per_tok=8`, `norm_topk_prob=true`, `first_k_dense_replace=3`, `routed_scaling_factor=2.5`, `attention_bias=true` (the QKV bias from ChatGLM is preserved as a config option), `partial_rotary_factor=0.5` (50% of head_dim channels rotate — preserves the GLM partial-rotary heritage), `use_qk_norm=true`, GQA 96Q/8KV, `head_dim=128`, hidden=5120, 92 layers, `num_nextn_predict_layers=1` (MTP head), max_position 131072 (4.5) → 202752 (4.6) (verified `zai-org/GLM-4.5/config.json` and `zai-org/GLM-4.6/config.json` 2026-06-08). → **GLM-5 / GLM-5.1** (2026-02 / 2026-04, see §5.27): 744B/40B-A; DSA-derivative sparse attention layered on top of the 4.5/4.6 GQA+MoE+QK-norm template; post-trained for 8-hour autonomous agentic loops; 200k ctx.

**Axes flipped (ChatGLM2 → GLM-5.1):** MQA group_num=2 → GQA 96/8 at 4.5; vocab 65024 → 151552 (Tiktoken-class); seq_length 32k → 8k → 131k → 202k → 200k+; partial-rotary on first half of channels retained throughout (`original_rope=true` until 4.5, then `partial_rotary_factor=0.5`); `add_qkv_bias=true` → `attention_bias=true` (preserved across the model_type switch); QK-norm absent → introduced at 4.5; dense → MoE at 4.5 (160+1/top-8); MTP head added at 4.5; sparse attention adopted at 5/5.1. GLM is the family where Chinese architectural design choices (custom `chatglm` model_type, partial-rotary, post-norm with `add_qkv_bias`) were gradually re-encoded as Llama-superset config fields once `glm4_moe` shipped — a worked example of "merge into the mainstream config schema while preserving the original primitives".

### 5.32 Yi 1.0 → 1.5 → Yi-Coder (01.AI)

01.AI's Yi family is the cleanest "Llama-shape from day one, vocab and θ as the only architectural levers" lineage in the corpus. **Yi-6B / Yi-34B** (2023-11): `LlamaForCausalLM` with **GQA from the first release** at both sizes — Yi-6B is 32Q/4KV/h=4096/L=32/ffn=11008; Yi-34B is 56Q/8KV/h=7168/L=60/ffn=20480; **rope_theta=5000000** (5e6, already a long-context value at launch); vocab=64000 (custom Yi BPE; matches the published 64k Yi tokenizer); rms_norm_eps=1e-5; `tie_word_embeddings=false`; ctx 4096 base (verified `01-ai/Yi-6B/config.json` and `01-ai/Yi-34B/config.json` 2026-06-08). The Yi-6B + Yi-34B template is the canonical "Llama-2 shape but with θ pre-extended and a 64k vocab" archetype — many downstream Chinese fine-tunes and the entire Nous-Capybara / Tess-Yi line use this template unchanged. → **Yi-1.5-6B / 9B / 34B** (2024-05): same `LlamaForCausalLM` shape — Yi-1.5-9B is 32Q/4KV/h=4096/L=48 (gains 16 layers vs 6B), rope_theta=5000000 unchanged, vocab=64000 unchanged (verified `01-ai/Yi-1.5-9B/config.json` 2026-06-08); the v1.5 jump is **training-data-only** (improved pre-training corpus + extra 500B tokens), with no config delta beyond `attention_bias=false` being made explicit and `mlp_bias=false`. This makes Yi-1.5 a controlled "data-vs-config" comparison against Yi 1.0 at fixed architecture. → **Yi-Coder-1.5B / 9B** (2024-09): code-specialized; **rope_theta bumped 5e6 → 1e7**, **ctx 4096 → 131072 base** (no `rope_scaling` field — pure θ extension, matching the Granite 4.x θ-only extension recipe); vocab=64000 unchanged. **Yi-Coder-1.5B is MHA (16Q/16KV, h=2048, L=24)** while **Yi-Coder-9B is GQA (32Q/4KV, h=4096, L=48)** — the 1.5B is the size where MHA returns, paralleling SmolLM2-1.7B (verified `01-ai/Yi-Coder-1.5B/config.json` and `01-ai/Yi-Coder-9B/config.json` 2026-06-08). → **Yi-VL 6B/34B** (Jan 2024): VLM line on Yi-6B/34B base + CLIP-ViT (vision side out of scope; backbone unchanged).

**Axes flipped (Yi 1.0 → Yi-Coder):** GQA from day one (no MHA-at-launch generation, unlike Llama or Qwen); rope_theta 5e6 → 5e6 → 1e7 (one bump, at the Coder specialization); ctx 4096 → 131072 via θ-only extension; vocab 64000 fixed across all generations; tied embeddings never (`tie_word_embeddings=false` throughout); MHA returns at the Yi-Coder-1.5B size. Yi is the family that demonstrates "pure data scaling at fixed config" (Yi → Yi-1.5) and "pure θ extension for context" (Yi-1.5 → Yi-Coder) as independent, separable architectural axes — an unusually clean controlled comparison.

### 5.33 Hunyuan-Large → Hunyuan v1 sweep → A13B → HunyuanOCR / MT / Hy3 (Tencent)

Tencent's Hunyuan is the family where **cross-layer attention (CLA)** appears in the corpus, and the only Chinese-origin family to ship a registered-namespace `model_type` reset mid-lineage. **Hunyuan-Large** (released as `Tencent-Hunyuan-Large/Hunyuan-A52B-Instruct`, 2024-11) uses `model_type=hunyuan` with hidden_size=6400, **80 attention heads / 8 KV heads (head_dim=80, not the universal 128)**, 64 layers, intermediate_size=18304, vocab=129024, `tie_word_embeddings=true`, max_position=131072, `use_qk_norm=true`, and uniquely `use_cla=true` with `cla_share_factor=2` — K/V projections shared across pairs of adjacent layers, the only K/V-sharing scheme in the census besides DeepSeek's MLA and Granite/MobileLLM's parameter-tying (verified `tencent/Tencent-Hunyuan-Large/Hunyuan-A52B-Instruct/config.json` 2026-06-08). The MoE is **16 experts + 1 shared expert with `moe_topk=1`** (top-1 routing, the only mainstream MoE in the corpus to route to a single non-shared expert per token), `use_mixed_mlp_moe=true`, rope_theta=10000 with dynamic `rope_scaling.alpha=1000`. → **Hunyuan v1 dense sweep** (2025-08): a clean `model_type` reset to `hunyuan_v1_dense`, dropping CLA (`use_cla=false` though the `cla_share_factor=2` config field is preserved as inert legacy) and returning to head_dim=128. **0.5B** (h=1024, 16Q/8KV, L=24, ffn=3584, vocab=120818, max_pos=**262144**), **1.8B** (h=2048, 16Q/4KV, L=32, ffn=6144, max_pos=262144), **4B** (h=3072, 32Q/8KV, L=36, ffn=8192, max_pos=262144), **7B** (h=4096, 32Q/8KV, L=32, ffn=14336, vocab=128256 with `org_vocab_size=290943`, max_pos=**32768** — the 7B is the only member with shorter base ctx) — all tie embeddings, all `use_qk_norm=true`, all rope_theta=10000 with dynamic rope_scaling `alpha=1000` (verified `tencent/Hunyuan-{0.5B,1.8B,4B,7B}-Instruct/config.json` 2026-06-08). → **Hunyuan-A13B-Instruct** (2026-04): `model_type=hunyuan_v1_moe`; reuses the 7B-dense backbone (h=4096, 32Q/8KV, head_dim=128, L=32) but adds a **fine-grained MoE with 64 experts + 1 shared expert per layer, top-8 routing, `moe_intermediate_size=3072` per expert** (small experts, 21% of the dense 14336 ffn), `norm_topk_prob=true`, `use_cla=false`, max_position=**32768** (verified `tencent/Hunyuan-A13B-Instruct/config.json` 2026-06-08 — note: matches the dense 7B's 32k, not the 256k previously summarized in §5.24). → **HunyuanOCR** (2025-11): `model_type=hunyuan_vl`, text backbone identical to Hunyuan-0.5B (h=1024, 16Q/8KV, L=24, vocab=120818) + 1152-d / 27-layer ViT, but uses **xdrope** position encoding (`rope_scaling.type=xdrope`, `xdrope_section=[16,16,16,16]` — a 4-axis split RoPE for multi-axis 2D position encoding) instead of dynamic RoPE (verified `tencent/HunyuanOCR/config.json` 2026-06-08); only family in the census to ship xdrope. → **Hunyuan-MT-7B** (2026-05) reuses the dense 7B config unchanged (translation is data-only specialization, verified `tencent/Hunyuan-MT-7B/config.json`).

**Axes flipped (Hunyuan-Large → v1 sweep → A13B → OCR):** model_type `hunyuan` → `hunyuan_v1_dense` / `hunyuan_v1_moe` / `hunyuan_vl` (registered namespace, not a Llama subclass — the only Chinese family besides early ChatGLM to ship its own `model_type` through 2026); head_dim 80 → 128 (Hunyuan-Large is the outlier, the v1 reset normalizes to the universal 128); CLA on → off (`use_cla=true` at Hunyuan-Large, then `false` throughout the v1 lineage — but the `cla_share_factor=2` config field is preserved as inert legacy across every Hunyuan config we fetched); QK-norm on throughout (`use_qk_norm=true` from Hunyuan-Large onward); MoE top-1 + 16 experts (Large) → fine-grained top-8 + 64 experts (A13B); shared-expert count 1 throughout; tied embeddings throughout; rope_theta=10000 throughout with dynamic `rope_scaling.alpha=1000` (Large used `alpha=1000`; the v1 dense sweep uses `alpha=1000` at most sizes but `alpha=100000` at 7B — the 7B is the configurational outlier of the family); xdrope adopted only at the VL spinout. Hunyuan is the family that demonstrates "introduce CLA at flagship scale → drop it at the small-model reset → preserve the config field as inert legacy" as a worked example of architectural retreat that leaves config-level fossils.

---

## 6. Axis Catalog (19 Axes)

Formal definitions, enumerations, and model mappings. Axes A1–A15 are inherited from v2 with extended value enumerations; A16–A19 are new in v3.

### Axis A1 — Attention type

The structural form of the attention operator. **Enumeration:** `{MHA, GQA, MQA, MLA, SWA, BlockSparse, Linear/SSM-{Mamba1 (S6), Mamba2 (SSD), Mamba3 (complex state + MIMO), RWKV6, RWKV7 (delta-rule), Griffin/Hawk (LRU + local attn), xLSTM (mLSTM matrix + sLSTM exp gating), GatedDeltaNet, LightningAttn, LFM, MoT (Mixture-of-Transformers)}, Hybrid-{sequential periodic, periodic shared, parallel branches, ratio-N:M (1:8 Jamba / 9:1 Granite 4 / 3:1 Qwen3-Next / 7:1 MiniMax / 5:1 Gemma 3 SWA)}, DSA (DeepSeek Sparse Attention with Lightning Indexer), CSA+HCA (V4 dual sparse), MSA (MiniMax Sparse Attention), iRoPE (per-layer NoPE mask)}`. **Defining configs:** `num_attention_heads`, `num_key_value_heads`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `sliding_window`, `sliding_window_pattern`, `layer_types[]`, `state_size`, `conv_kernel`, `mamba_d_state`, `mamba_d_conv`, `mamba_expand`, `block_types[]` (RecurrentGemma), `no_rope_layers[]` (SmolLM3, Llama 4), `attention_k_eq_v` (Gemma 4 global only on 12B+, NOT on E2B), `global_head_dim` (Gemma 4 global, doubled to 512), `num_kv_shared_layers` (Gemma 4 E2B/E4B only).

**Per-token-per-layer cache cost reference:**
- MHA: `2·H·d`
- GQA: `2·H_kv·d`
- MQA: `2·d`
- MLA: `kv_lora_rank + qk_rope_head_dim` (DeepSeek-V2-Lite: 512+64=576 vs MHA 2·16·128=4096, ~7× reduction; V3 full at 128 heads: 576 vs 2·128·128=32768, ~57× reduction)
- DSA: O(n·k) with `k` = Lightning Indexer top-k
- CSA+HCA: ~10% of V3.2 at 1M ctx
- SWA: bounded `min(seq, W) · 2·H_kv·d`
- SSM: constant-size state independent of sequence
- iRoPE: per-layer either standard GQA cost or 0 (NoPE layers have no rotation cost)
- Gemma 4 with K=V on global (12B+): `H_kv·d` (half of full GQA, since K and V tensors are aliased)
- Gemma 4 with cross-layer KV sharing (E2B `num_kv_shared_layers=20`): `(L - shared) · S · 2 · Dh · Hk` — the shared layers carry zero extra KV cache; pointer reuse.

### Axis A2 — Position encoding scheme

**Enumeration:** `{none, absolute_learned, ALiBi, RoPE-vanilla, RoPE-Llama3-bandscaling, RoPE-LongRoPE, RoPE-YaRN, RoPE-DynamicNTK, RoPE-linear, RoPE-dual (Gemma 3/4 per-layer-type), RoPE-dual + partial-0.25 on global (Gemma 4), M-RoPE 3D (Qwen 2.5-VL T/H/W), TMRoPE 4D (Qwen2.5-Omni + frame-idx), 2D-RoPE (Pixtral vision), V2PE (InternVL 3 Variable Visual PE), iRoPE per-layer NoPE mask (Llama 4 Scout/Maverick, SmolLM3), MLA-split (NoPE/RoPE per head channel), Visual Causal Flow (DeepSeek-OCR-2 — bidirectional within page, causal cross-page)}`. **Configs:** `rope_theta`, `rope_scaling`, `rope_local_base_freq`, `partial_rotary_factor`, `no_rope_layers[]`, `mrope_section`, position embedding dim.

### Axis A3 — Normalization (type + placement + QK-norm)

**Norm type:** `{RMSNorm (default), LayerNorm, none}`. **Placement:** `{pre, post (OLMo 2), dual (Gemma 1+ pre+post per sublayer = 4 norms), pre-norm + sublayer-output-norm}`. **QK-norm:** `{none, per_head_dim (Qwen 3 — PRE-RoPE), per_full_channels (OLMo 2 — POST-projection), QK-LayerNorm (StableLM 2 12B), fixed-scale (Gemma 4 — local 0.9916 / global 1.0228 absorbing `1/sqrt(head_dim)`, attention_scale=1.0 instead)}`. **Softcap:** `{none, attention only, final only (Gemma 4 — 30.0 restored, attn dropped), both (Gemma 2)}`.

**Critical asymmetric softcap finding:** Gemma 4 is the only family in the corpus with `attn_logit_softcapping=null` AND `final_logit_softcapping=30.0` simultaneously. Gemma 2 had both; Gemma 3 had neither; Gemma 4 has only the final. The taxonomy cell "final only" was vacant before Gemma 4 occupied it.

### Axis A4 — FFN family (gate topology)

`{ungated (LN-bias MLP — MPT, Pythia, Phi-1/1.5/2, MPT), gated (X-GLU: SwiGLU/GeGLU/gegelu/gated-GELU-tanh), gated fused (Phi-3+ fused gate_up), MoE-{SwiGLU-MoE, GeGLU-MoE} {top-k softmax / sigmoid / norm_topk / aux-loss-free / shared / ultra-sparse}, RWKV channel-mix, **double-wide** (Gemma 4 edge — `use_double_wide_mlp=True` widens gate_proj/up_proj projections; semantics not fully documented as of cutoff)}`. Note: **Activation function** is axis A14, separate.

### Axis A5 — MoE routing

`{none (dense), top-k softmax + aux-loss (Mixtral, OLMoE, Phi-3.5-MoE), + shared (DeepSeek-V2-Lite: 64+2/top-6), top-k sigmoid + aux-loss-free + shared (DeepSeek-V3: 256+1/top-8), top-k softmax + norm_topk + 0-shared (Qwen 3 MoE: 128+0/top-8), **ultra-sparse** (Qwen3-Next: 512+1/top-10 — 2.15% activation), Gemma 4 MoE (128+1/top-8, `moe_intermediate_size=704`), GPT-OSS (32+0/top-4 + MXFP4 weights), Granite 4 H-Tiny (32+0/top-4), Llama 4 Scout (16+0/top-1), Llama 4 Maverick (128+1/top-1), Mixture-of-LoRAs (Phi-4-MM — modality-routed adapters over frozen LM), Mixture-of-Transformers (HY-Embodied — multiple transformer experts routed by spatial+temporal task)}`. **Defining configs:** `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`, `decoder_sparse_step`, `first_k_dense_replace`, `norm_topk_prob`, `router_aux_loss_coef`, router activation (sigmoid vs softmax), MTP head presence (Qwen3-Next, Hy3 3.8B MTP, Gemma 4 MTP drafters).

### Axis A6 — KV-cache shape and update policy

Per the attention type, the cache shape varies. New v3 entries: **Gemma 4 K=V global** (12B/26B/31B: `H_kv·d` not `2·H_kv·d` — K and V projections are aliased on global layers); **Gemma 4 cross-layer sharing** (E2B/E4B `num_kv_shared_layers=20/35` means >half of layers reuse a previous layer's K/V tensor); **Apple AFM 2-block sharing** (Block-1 = 62.5% of layers full KV; Block-2 = 37.5% reuses Block-1 K/V); **DSA sparse cache** (only top-k tokens cached per token); **CSA+HCA dual sparse** (10% of V3.2 KV at 1M ctx); **MSA**; **Mamba-3 complex state** (vs Mamba-2 SSD); **xLSTM matrix memory** `(num_heads, head_size, head_size)` state; **LFM state** (Liquid time-constant cells, non-Transformer); **BitNet / Falcon-Edge 1.58-bit ternary weights** (no fp KV cache compression but ternary weights for KV projections); **GPT-OSS MXFP4 native weights** (4-bit MXFP4 weights for KV projections). The unified IR needs both `kv_cache: Tensor` and `ssm_state: dict[str, Tensor]` plus pointer/sharing maps for A18.

### Axis A7 — Tokenizer/vocab family

v3 catalogs 18 distinct tokenizer flavors (vs 15 in v2). Added: TK131 (Mistral Nemo/Ministral/Pixtral 131k Tekken), GP-v4 (Gemma 4 262k +image/audio/video specials), QW-omni (Qwen Omni + speech tokens), LL4 (Llama-4 Tiktoken 200k), ERNIE (Baidu ERNIE BPE ~50k for ERNIE-4.5-0.3B / PaddleOCR-VL), HY (Hunyuan SP ~150k), EXAONE (LG BPE 102400), Helium SP (Kyutai 48000), MoonViT tokenizer, MiMo SP.

Vocab range: 32000 (LLaMA 1) → 262144 (Gemma 4) → 200064 (o200k / Llama-4) → 151936 (Qwen) → 128256 (Llama-3). Vocab × hidden product as fraction of total params is now a first-class concern for the smallest models.

### Axis A8 — Scalar multipliers (μP-style)

Five multipliers (preserved from v2): `embedding_multiplier`, `attention_multiplier` / `query_pre_attn_scalar`, `residual_multiplier`, `logits_scaling`, `softcap`. New v3 entry: **Gemma 4 fixed-scale QK-norm** (local 0.9916 / global 1.0228 absorbing `1/sqrt(head_dim)`, with `attention_scale=1.0`). New v3 entry: **Cohere Command A+ W4A4 quantization scale**.

### Axis A9 — Bias presence

`{none anywhere (Llama, Qwen 2/2.5/3, Mistral, Yi, InternLM, SmolLM, Granite, OLMo, Gemma, BitNet, MobileLLM, Falcon 3, EXAONE, Hunyuan, Llama 4), QKV bias only (Qwen 1, StableLM 2 1.6B, Vary-toy), full bias on QKV + MLP + LN (StarCoder 1, StarCoder 2 `use_bias=true`, Florence-2 BART), LN bias only (MPT, Pythia)}`. Phi-2 has biases on the parallel sublayers. The 2023→2024 trend was clear bias removal; StarCoder 2 is the explicit counter-example. New v3 entries: Hunyuan and EXAONE follow "none anywhere".

### Axis A10 — Embedding tying

`tie_word_embeddings ∈ {y, n}`. Pattern: <3B tends tie, >4B tends untie. **Counter-examples:** Cohere Aya 8B (tied), Llama 3.1 8B (untied), Qwen 3 8B (untied at 8B but tied below). **Gemma family preserves tied embeddings end-to-end across all sizes** (a 256k vocab × small hidden makes untying expensive). New v3 entries: ERNIE-4.5-0.3B (tied — small), Falcon-Edge 1B (tied), MoE families (typically untied due to the routing-head needing free parameters).

### Axis A11 — Per-layer **type** alternation

`{uniform, SWA/full alternation {Gemma 2: 1:1, Gemma 3: 5:1, Gemma 4 E2B: 4:1, Phi-3-mini: 4096+SWA 2047, SmolLM3 NoPE every-4th, Ministral interleaved}, dense/MoE alternation (DeepSeek-V2-Lite `first_k_dense_replace=1`), Mamba/attention periodic {Jamba 1:8, Falcon-H1 parallel-head, Granite 4 9:1, Qwen3-Next 3:1, MiniMax 7:1, Nemotron 3 Super/Ultra periodic + MoE}, parallel-branch-per-layer (Hymba), GPT-3-style alternating dense/banded-sparse (GPT-OSS), iRoPE per-layer NoPE mask (Llama 4 Scout/Maverick), Apple AFM 2-block split, block_types[] generic (RecurrentGemma), Mixture-of-Transformers routing (HY-Embodied), DSA learned-scorer top-k per token}`. The eight hybrid topologies, plus iRoPE and DSA, push this axis to >15 distinct production patterns.

### Axis A12 — Per-layer **shape** scaling

OpenELM `num_query_heads[]` / `num_kv_heads[]` / `ffn_multipliers[]` as per-layer lists. MobileLLM block-wise weight sharing. Gemma 3n Matformer elastic-width at runtime. Apple AFM 2-block layer-count split (62.5% / 37.5%). Gemma 4 split: local layers have `head_dim=256` while global layers have `head_dim=512` (12B+) — a per-layer-type head_dim variation. New v3 entry: Gemma 4 local vs global head_dim differs, plus E2B uses `num_kv_shared_layers=20/35` so >half of layers carry zero KV cost.

### Axis A13 — Parallel vs sequential sublayer

`{sequential (Llama default), parallel (Phi-1/1.5/2, Falcon 7B, Pythia, GPT-J, GPT-NeoX), parallel-branches-per-block (Hymba: Mamba ‖ attention; RWKV: channel-mix ‖ time-mix; Falcon-H1 parallel-head Mamba ‖ attn), parallel-with-fused-input (Phi-2 fuses QKV+MLP-up input projections)}`. Sequential vs parallel was abandoned by Llama-shaped models in 2024 and re-adopted by hybrid architectures.

### Axis A14 — Activation function

`{SiLU, GELU, GELU-tanh (Gemma, CodeGemma, StarCoder 2), GELU-new (Phi-2 specifically), gegelu (Phi-3-small), ReLU² (BitNet b1.58, Falcon-Edge — quantization-frontier signature), sigmoid+ReLU² (RWKV-6/-7 channel-mix), Swish (OpenELM)}`. v3 preserves v2's enumeration; new entries Falcon-Edge ReLU² (inherits BitNet activation) and Liquid LFM SiLU.

### Axis A15 — RoPE rotation domain

`{full rotary (Llama, Qwen, Mistral, OLMo, Gemma — all channels rotated), partial-{0.5, 0.4, 0.25, 0.75} (Phi-1/1.5, Phi-2, StableLM 3B/StableLM 2 1.6B, Phi-4 mini), NoPE/RoPE split per head (MLA), NoPE per layer mask (SmolLM3, Llama 4 Scout/Maverick iRoPE), **per-layer-type partial rotary** (Gemma 4: local layers full rotary, global layers `partial_rotary_factor=0.25`), no rotation (SSM/LFM/RWKV/xLSTM)}`. Gemma 4 is the first model where partial rotary is **per-layer-type**, not model-level.

### Axis A16 — VLM fusion topology (NEW)

`{concat-prefix-LM (PaliGemma 1/2), concat-projector (LLaVA-*, Qwen-VL, MiniCPM-V, InternVL, Molmo, Pixtral, SmolVLM, NVILA, Cambrian-1, Eagle 2, Ovis 2/2.5, Phi-3.5-vision, Florence-2, Phi-3-vision, MinerU2.5, GOT-OCR2, olmOCR, MonkeyOCR, HunyuanOCR, PaddleOCR-VL, Surya 2, Granite Vision 4.1, Gemma 4 E2B/E4B/26B-A4B/31B encoder-based), cross-attention-adapter (Llama 3.2 Vision 11B/90B), early-fusion native (Chameleon, Llama 4 Scout/Maverick, Falcon Perception 0.6B), mixture-of-LoRAs (Phi-4-multimodal), split-encoder (Janus-Pro 1B/7B for understanding vs generation), encoder-free (Gemma 4 12B Unified: raw image+audio→linear→residual stream)}`. **Six values + encoder-free = 7.** This is the most important new axis for the OCR/VLM sub-segment. The fusion choice has downstream effects on KV cache, position encoding, and quantization.

### Axis A17 — Vision-token compression budget (NEW)

Measured as **vision tokens emitted per 1024² page (or equivalent normalized resolution)**:
- **None / n/a** (text-only): Llama, Qwen 3, Mistral 7B, etc.
- **Ultra-low (~100 tokens/page)**: DeepSeek-OCR (16× conv compressor over SAM+CLIP serial encoder; 100 tokens/page is the SOTA design point).
- **Low (~256-324 tokens/page)**: GOT-OCR2 (256 tokens), mPLUG-DocOwl2 (324 tokens), Vary-toy (vision-vocabulary expansion), SmolVLM (9× pixel-shuffle), PaddleOCR-VL (Adaptive MLP compression), MinerU2.5 (NaViT patch-merger), Surya 2.
- **Mid (~640-1024 tokens/page)**: MiniCPM-V 2.6 (perceiver resampler ≤640 per 1.8M-pixel image), PaliGemma 2 (mid), Phi-3.5-vision, InternVL 2.5 (pixel-unshuffle 4×), olmOCR / olmOCR-2 (M-RoPE dynamic), MonkeyOCR (SRR triplet), Nanonets-OCR-s/OCR2, HunyuanOCR, RolmOCR, Janus-Pro, Granite Vision 4.1, Gemma 4 E2B/E4B (configurable 70/140/280/560/1120), Phi-4-multimodal (vision LoRA varies), Eagle 2.
- **High (~3000-6000 tokens/page)**: Qwen 2-VL / Qwen 2.5-VL / Qwen 3-VL (native dynamic resolution, no compression), Pixtral 12B, NVILA, Cambrian-1, Llama 4 Scout (native early fusion), MiMo-V2.5 Pro (729M ViT high-res).
- **Elastic / dynamic**: InternVL 3 (V2PE Variable Visual PE), Ovis 2.5 (native-res ViT + visual embedding table), LLaVA-NeXT AnyRes, LLaVA-OneVision AnyRes-9, Eagle 2 MoVE (per-encoder tile), Dolphin-v2 document-type-aware routing.
- **Encoder-free**: Gemma 4 12B Unified (raw projection — token count = patch count from a fixed grid).

The single most discriminating axis within OCR-LLM. DeepSeek-OCR's "1 vision token ≈ 10 text tokens" compression budget defines the SOTA design point as of cutoff.

### Axis A18 — Cross-layer KV sharing (NEW)

`{none (default — every layer keeps its own K/V), same-block-shared (Gemma 4 E2B/E4B: `num_kv_shared_layers=20` of 35 — the last N layers reuse the K/V from an earlier same-type layer), cross-block-shared (Apple AFM 2-block: Block-1 carries 62.5% of layers with full KV; Block-2 carries 37.5% and reuses Block-1's K/V), YOCO research, CLA research}`. **The Gemma 4 size-dependent correction**: this axis distinguishes E2B/E4B (which use cross-layer KV sharing) from 12B/26B/31B (which use K=V on global instead of cross-layer sharing). The corresponding axis A6 entry depends on size.

### Axis A19 — Per-layer embeddings (NEW)

`{none (default — only the input embedding table feeds the residual stream once), Gemma 4 PLE (E2B/E4B: a second 256-dim embedding table whose output is injected as a residual signal at every decoder layer, computed as `(token_identity + context_aware_projection) × 1/√2`. The PLE table can be paged to flash storage on mobile devices so VRAM only holds the layer-wise projection — this is the mechanism behind "effective 2.3B" vs "5.1B with embeddings"), Apple AFM per-layer rank-1 LoRA adapter (closed; reported but unverified by config)}`. v2's "embedding scale" axis and "tied lm_head" axis do not contemplate a per-layer-residual embedding table. PLE is closest to an adapter than to a token embedding, but the IR distinction matters: PLE is invoked at every layer's residual addition, while LoRA adapters modify projections inside the sublayers.

### Optional Axis A20 — Partial-rotary factor (alternative reading)

Some readings prefer to fold `partial_rotary_factor` into A15 (RoPE rotation domain) as a continuous parameter rather than separate axis. v3 keeps it under A15 because Gemma 4's per-layer-type partial rotary (full on local, 0.25 on global) is a new value within A15's enumeration that does not require a separate axis. A20 is reserved for future use.

---

## 7. Cross-org Comparison Table (refreshed for v3)

Each row is one vendor; columns are collapsed axes. "✓" present; "—" absent; "→" generation transition.

| Vendor | Attn (2026 norm) | RoPE family | Norm | FFN/Activation | Quantization-native | MoE | Hybrid SSM / sparse-attn | μP scalars | Per-layer scaling | Parallel sublayer | VLM fusion | KV sharing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Meta | GQA + **iRoPE** (Llama 4) | Llama-3 band + iRoPE | RMS pre | SwiGLU/SiLU | – | **Llama 4 (16+0/top-1)** | – | – | – | – | **early-fusion native** (Llama 4) | – |
| Alibaba (Qwen) | GQA + QK-norm/Dh + **Gated DeltaNet 3:1** (Qwen3-Next) | RoPE θ=1e6 / M-RoPE / TMRoPE | RMS pre + QK-norm pre-RoPE | SwiGLU/SiLU | – | **ultra-sparse (512+1/top-10)** | **Qwen3-Next Gated DeltaNet 3:1** | – | – | – | concat-projector + M-RoPE / TMRoPE | – |
| Microsoft (Phi) | MHA→GQA→hybrid (Samba) | LongRoPE + partial 0.75 | RMS pre | SwiGLU fused / gegelu | BitNet b1.58 (ReLU²); **MXFP4** (gpt-oss) | top-2/0-shared, top-4/0-shared (gpt-oss 32+0) | Phi-4-mini-flash (Samba); **gpt-oss alternating dense/banded** | μP at Phi-3-small | – | – | mixture-of-LoRAs (Phi-4-MM) | – |
| Google (Gemma) | GQA + 5:1 SWA + **K=V global (12B+) / cross-layer share (E2B/E4B)** + p-RoPE 0.25 global | dual θ + **partial 0.25 global** | RMS pre+post dual + fixed-scale QK-norm | GeGLU/GELU-tanh + **double-wide MLP** (edge) | QAT mobile (Gemma 4) | **first Gemma MoE** (26B-A4B: 128+1/top-8) | RecurrentGemma (Griffin/Hawk) | softcap restored final only (Gemma 4) | Matformer (3n); **PLE** (Gemma 4 edge) | – | concat + audio + encoder-free (12B Unified) | **same-block (E2B/E4B); K=V global (12B+)** |
| DeepSeek | MHA→**MLA→DSA→CSA+HCA** (V4) | YaRN + MLA split | RMS pre | SwiGLU / SwiGLU-MoE | – | top-8/1-shared aux-loss-free (V3); **DSA learned-scorer (V3.2); CSA+HCA (V4)** | – | – | – | – | concat-projector (DeepSeek-VL2) + DeepEncoder (DeepSeek-OCR) + split (Janus-Pro); **Visual Causal Flow** (DeepSeek-OCR-2) | – |
| Mistral | GQA + sinks (Small 3.x+) | RoPE θ=1e6 | RMS pre | SwiGLU/SiLU | – | **Small 4 MoE 119B/15B-A** | Codestral Mamba (Mamba-2) | – | – | – | concat-projector + 2D-RoPE (Pixtral); interleaved SWA (Ministral) | – |
| AI2 (OLMo) | MHA | vanilla θ=5e5 | RMS post + QK-norm/full | SwiGLU | – | OLMoE 64/0/top-8 | – | – | – | – | Molmo concat + **olmOCR / olmOCR-2** (GRPO) | – |
| Apple (AFM) | GQA + **2-block cross-block KV sharing** + 2-bit QAT | vanilla | RMS pre + QK-norm | SwiGLU | **2-bit QAT** | – | – | – | **per-layer rank-1 LoRA adapter** | – | concat-adapter (ViTDet-L) | **cross-block-shared (2-block)** |
| IBM (Granite) | GQA + μP; **9:1 Mamba-2:attn** (Granite 4.0); dense (Granite 4.1) | vanilla θ → 1e7 | RMS pre + μP | SwiGLU; **SwiGLU-MoE 32+0/top-4** (H-Tiny) | – | **Granite 4 H-Tiny (32+0/top-4 MoE-hybrid)** | **Granite 4 9:1 sequential** | four μP scalars | – | – | concat-projector (Vision 4.1); **NAR speech (Speech 4.1)** | – |
| OpenBMB (MiniCPM) | MHA→MLA | LongRoPE | RMS pre + μP | SwiGLU | – | – | – | scale_emb, scale_depth | – | – | concat + perceiver resampler (≤640 tokens) + audio (MiniCPM-o) | – |
| Stability AI | MHA + partial 0.25 | partial 0.25 | LN + QK-LN | SwiGLU | – | – | – | – | – | – | – | – |
| 01-AI (Yi) | GQA | vanilla θ=5e6 | RMS pre | SwiGLU | – | – | – | – | – | – | – | – |
| InternLM | GQA (KV=2) | dynamic NTK | RMS pre | SwiGLU | – | – | – | – | – | – | InternVL concat-projector + **V2PE** (InternVL 3) | – |
| HuggingFace (SmolLM) | GQA + **NoPE every 4th** (v3) | vanilla / mask | RMS pre | SwiGLU | – | – | – | – | – | – | SmolVLM concat + 9× pixel-shuffle | – |
| AI21 (Jamba) | GQA + Mamba-1 1:8 + MoE 1:2 | vanilla on attn | RMS pre | MoE 16/0/top-2 | – | top-2/0-shared periodic | **Jamba 1:8 sequential** | – | – | – | – | – |
| Zyphra (Zamba2, ZAYA1) | Mamba-2 + periodic shared attn; ZAYA1 MoE | RoPE on shared | RMS pre | gelu MLP / SwiGLU-MoE | – | ZAYA1 MoE | **Zamba2 periodic shared** | – | – | – | – | – |
| NVIDIA (Hymba, Nemotron, Eagle) | Parallel Mamba‖attn (Hymba); Hybrid Mamba+TFM+MoE (Nemotron); **MoVE** (Eagle 2) | NTK / Llama-3 | RMS pre | SwiGLU/SwiGLU-MoE | FP4 / FP8 (Nemotron) | top-k/0-shared (Nemotron); ZAYA1 MoE | **Hymba parallel; Nemotron 3 Super/Ultra hybrid + MoE** | – | – | parallel branches (Hymba) | concat + MoVE multi-encoder (Eagle 2); Nemotron Nano Omni | – |
| TII (Falcon) | MQA → Mamba → GQA hd=256 → **Mamba-2+attn parallel-head** (H1) → **1.58-bit retrainable** (Edge) → early-fusion (Perception) | vanilla θ=1000042 | RMS pre | SwiGLU / ReLU² (Edge) | **1.58-bit Falcon-Edge retrainable** | – | Falcon Mamba, Falcon-H1 (parallel-head) | – | – | parallel (Falcon 7B) → sequential | early-fusion (Falcon Perception) | – |
| BigCode (StarCoder) | MQA → GQA + bias | absolute → vanilla θ=1e6 | LN pre | gated GELU-tanh + bias | – | – | – | – | – | – | – | – |
| Cohere | GQA → **Command A+ MoE** | RoPE θ=4e6 | LN pre | SwiGLU | **W4A4 lossless** (Command A+) | **first Cohere MoE (Command A+ 218B/25B-A)** | – | logit_scale=0.0625 | – | – | – | – |
| RWKV-LM | RWKV-6 → RWKV-7 (delta) | – | LN | RWKV channel-mix | – | – | linear-attn family | – | – | parallel (ch+t) | – | – |
| Tsinghua/Zhipu (GLM) | MHA → MQA → **DSA-derivative (GLM-5.1)** | vanilla / YaRN | RMS | SwiGLU/GeGLU | – | **GLM-5.1 MoE (744B/40B-A)** | **DSA derivative (GLM-5.1)** | – | – | – | GLM-5V-Turbo concat | – |
| Baichuan | MHA + ALiBi → RoPE | none → vanilla | LN → RMS | SwiGLU | – | – | – | – | – | – | – | – |
| MobileLLM (Meta) | GQA + deep-narrow + weight-share | vanilla | RMS pre | SwiGLU | – | – | – | – | block-wise weight sharing | – | – | – |
| NXAI (xLSTM) | **mLSTM matrix + sLSTM exp gating** | – (no RoPE, no ALiBi) | RMS | SwiGLU | – | – | xLSTM family (non-Mamba non-RWKV linear-recurrent) | – | – | – | – | – |
| CMU/Princeton/Together/Cartesia (Mamba) | Mamba-1 → Mamba-2 (SSD) → **Mamba-3 complex state + MIMO** | – | RMS pre | – | – | – | Mamba (pure SSM family) | – | – | – | – | – |
| Liquid AI | **LFM (liquid time-constant) non-Transformer recurrent** | – | RMS | SwiGLU | – | LFM2.5-8B-A1B MoE | LFM family | – | – | – | concat (LFM2.5-VL-450M) | – |
| LG (EXAONE) | GQA | RoPE θ=1e6 | RMS pre | SwiGLU | – | – | – | – | – | – | concat (EXAONE 4.5) | – |
| Tencent Hunyuan | GQA + fusion fast/slow + **MoT** (HY-Embodied); MoE+MTP (Hy3) | RoPE / YaRN | RMS pre | SwiGLU/SwiGLU-MoE | – | Hunyuan-A13B 80B/13B-A; Hy3 295B/21B-A + 3.8B MTP | – | – | – | – | concat + native MM (HunyuanOCR) | – |
| Xiaomi MiMo | GQA + MoE FP8-native | YaRN | RMS pre | SwiGLU-MoE | FP8 train | MiMo-V2.5-Pro 1.02T/42B-A | – | – | – | – | concat + 729M ViT + audio | – |
| Moonshot Kimi | GQA + MoE swarm | YaRN | RMS pre | SwiGLU-MoE | – | Kimi K2.6 1T/32B-A | – | – | – | – | concat + 400M MoonViT | – |
| MiniMax | **Lightning Attn 7:1** → full → **MSA** | NTK on softmax | RMS pre | SwiGLU-MoE | – | MiniMax-Text-01 MoE | **Lightning Attn / MSA** | – | – | – | – | – |
| Kyutai | GQA + **dual-stream text/audio** (Moshi) | RoPE | RMS pre | SwiGLU | – | – | – | – | – | – | Mimi codec + Temporal+Depth Transformer (Moshi) | – |
| StepFun | GQA + MoE | YaRN | RMS pre | SwiGLU-MoE | – | Step 3.7 Flash 198B/11B-A | – | – | – | – | concat + 1.8B ViT | – |
| Baidu (ERNIE) | GQA | vanilla | RMS pre | SwiGLU | – | – | – | – | – | – | concat (PaddleOCR-VL) | – |
| Reka | GQA | RoPE | RMS pre | SwiGLU | – | – | – | – | – | – | – | – |
| OpenAI (gpt-oss) | GQA + **trained attention sinks** + **GPT-3-style alternating dense/banded** | LongRoPE | RMS pre | SwiGLU-MoE + **MXFP4 native** | **MXFP4** | top-4/0-shared (32 routed) | alternating dense/banded (revives GPT-3 pattern) | – | – | – | – | – |

---

## 8. Surprises and Non-obvious Patterns (refreshed)

Numbered to match v2 where applicable; new entries are marked **NEW v3**.

1. **Qwen 3 and OLMo 2 use different QK-norm shapes** (preserved). Qwen 3 normalizes per-Dh PRE-RoPE; OLMo 2 normalizes per-full-channels POST-projection.
2. **OLMo 2 is post-norm**, not pre-norm (preserved).
3. **Gemma 3 dropped softcapping entirely** (preserved).
4. **TinyLlama is GQA**, not MHA (preserved).
5. **Qwen3-0.6B is the cleanest head_dim-decoupling example** (preserved).
6. **MQA is alive at the smallest sizes** (preserved).
7. **Phi-3-small is the structural outlier of the dense pack** (preserved).
8. **SmolLM3 NoPE every 4th layer** (preserved). **Updated in v3: Llama 4 Scout/Maverick iRoPE is the production-scale generalization at 17B-A active.**
9. **Vocab sizes range from 32k to 262k** (preserved).
10. **OpenELM exposes per-layer variation as a list-of-ints** (preserved).
11. **BitNet b1.58 uses ReLU², not SwiGLU/GeGLU** (preserved). **Updated v3: Falcon-Edge inherits ReLU² and adds retrainable 1.58-bit weights — first BitNet supporting continued pretrain.**
12. **Mistral SWA was tried and abandoned, then re-adopted by other vendors and re-introduced as interleaved-SWA (Ministral)** (extended v3).
13. **ALiBi was tried and abandoned** (preserved).
14. **Parallel attention/FFN was tried and abandoned at scale, then re-adopted by hybrid architectures** (preserved). **Updated v3: Falcon-H1 parallel-head Mamba-2 + attention is the latest revival.**
15. **MoE shared-experts policy is now an 8-way split.** Updated from v2's 3-way split: (a) Mixtral 8+0 top-2, (b) DeepSeek-V2 64+2 top-6, (c) Qwen 3 MoE 128+0 top-8 with norm_topk, (d) DeepSeek-V3 256+1 top-8 aux-loss-free, (e) **Qwen3-Next 512+1 top-10 ultra-sparse**, (f) **Granite 4 H-Tiny 32+0 top-4**, (g) **Llama 4 Scout 16+0 top-1 / Maverick 128+1 top-1**, (h) **Gemma 4 26B-A4B 128+1 top-8 with moe_intermediate_size=704**. Eight distinct production MoE patterns. The shared-experts decision has clearly not converged.
16. **Embedding tying breaks at 8B for Cohere Aya** (preserved).
17. **RoPE base θ is the simplest context-extension dial** (preserved).
18. **Falcon 3 is the only non-Gemma family with head_dim=256 at <8B** (preserved).
19. **MLA generalized beyond DeepSeek** (preserved).
20. **Hybrid topologies have fragmented into nine distinct patterns in 30 months** (extended). Updated v3: Jamba (seq 1:8), Zamba2 (periodic shared), Hymba (parallel branch), Phi-4-mini-flash (Samba seq), Falcon-H1 (parallel-head), Granite 4 (9:1 seq), Qwen3-Next (3:1 seq Gated DeltaNet), MiniMax (7:1 seq Lightning Attn → abandoned → MSA), Nemotron 3 (hybrid Mamba+TFM+MoE).
21. **NEW v3: Gemma 4 introduces three architectural primitives in a single release**: (a) partial-rotary p-RoPE on global only (`partial_rotary_factor=0.25`), (b) K=V on global layers (12B/26B/31B; not on E2B), (c) cross-layer KV sharing on E2B/E4B (`num_kv_shared_layers=20/35`). Plus Per-Layer Embeddings (PLE) at edge sizes — a separate embedding table whose output is injected as residual at every layer, paged to flash on mobile.
22. **NEW v3: The "Gemma 4 K=V vs cross-layer KV sharing" trade-off is size-dependent.** E2B/E4B use cross-layer sharing; 12B/26B/31B use K=V global. An operator that assumes either is uniformly applied will corrupt half the family.
23. **NEW v3: Llama 4 Scout iRoPE is the first production-scale (17B-A active) NoPE-per-layer model.** Validates SmolLM3's NoPE-every-4th experiment.
24. **NEW v3: Qwen3-Next is the inverse of MiniMax-Text-01.** Both ship linear-attention hybrids; Qwen3-Next at 3:1 linear:softmax (75% linear) vs MiniMax at 7:1 (87.5% linear). MiniMax-M2 abandoned linear, MiniMax-M3 re-introduced as MSA. The "right ratio" has not converged.
25. **NEW v3: DeepSeek V3.2 DSA is the first production learned-scorer sparse attention.** Lightning Indexer scores token-relevance → top-k attention. DeepSeek V4 doubles down with CSA+HCA dual sparse (27% FLOPs, 10% KV at 1M context vs V3.2).
26. **NEW v3: Apple AFM 2-block cross-block KV sharing is a distinct A18 variant from Gemma 4.** Apple reuses across one block transition; Gemma 4 E2B/E4B reuses within the same block-type sub-sequence.
27. **NEW v3: GPT-OSS revives GPT-3's alternating dense/banded-sparse attention pattern.** A 2025 design that was tried-and-abandoned in the Llama era is revived for MXFP4-native MoE.
28. **NEW v3: OCR-LLM compression budget has converged on ultra-low (100 tokens/page) as the SOTA design point** (DeepSeek-OCR), but production deployments span 100-6000 tokens/page (PaddleOCR-VL at 256 / olmOCR-2 at mid / Qwen 2.5-VL at high) — no compression-budget consensus yet.
29. **NEW v3: Mistral Medium 3.5 absorbs Magistral (reasoning) and Devstral 2 (code) into one set of weights with configurable reasoning effort.** This unification trend is now visible in Mistral Small 4 (instruct+Magistral+Devstral+Pixtral), Qwen 2.5-Omni (Thinker+Talker), Hunyuan 7B (fast/slow toggle), Hy3 (fast/slow), Phi-4-multimodal (mixture-of-LoRAs). The "one model, multiple modes" framing is the dominant 2026 design pattern.
30. **NEW v3: NAR speech decoders are productive at 2B.** Granite Speech 4.1 NAR-2B is one of the only NAR speech heads in production at <8B. NAR enables parallel decoding (lower latency at the cost of accuracy).
31. **NEW v3: Mamba-3's complex-valued state + MIMO decoding is the first non-incremental SSM update since Mamba-2's SSD.** Validates SSM as a long-term architecture (vs the 2024 view that SSM was a research backwater for production).
32. **NEW v3: Falcon-Edge proves 1.58-bit BitNet can be retrainable, not just inference-only.** Combined with Apple AFM's 2-bit QAT and GPT-OSS's MXFP4-native weights, this validates "quantization as architectural choice" — a parallel branch of the family tree.
33. **NEW v3: Liquid AI LFM is the only non-Transformer non-SSM non-RWKV non-xLSTM recurrent backbone with public weights at this scale.** Liquid-time-constant cells; first MoE on this backbone at 8B/1.5B-A.
34. **NEW v3: The Llama line is being superseded by Muse Spark in Meta's product lineup as of 2026-04.** Per Wikipedia. Llama 5 is the "transitional flagship".

---

## 9. Coverage Caveats — Gated, Pre-Release, and Out-of-Scope

**Gated models for which only mirrors are accessible.** Same as v2: `meta-llama/*`, `google/gemma-*`, `mistralai/*` (some), `microsoft/Phi-3-medium*`. v3 also notes that Llama 4 Scout / Maverick / Behemoth are gated; Muse Spark, MAI-Thinking-1, Aion 1.0, Grok 4.3, Grok Build 0.1, ERNIE 5.1, Qwen3.7-Max have **no public weights at 2026-06-06**.

**Models cataloged but not fully verified by `config.json`:**
- **Apple Foundation Model 3.18B** (closed weights; tech report arXiv:2507.13575 is authoritative; per-layer rank-1 LoRA adapters reported but unverified in config).
- **Llama 4 Scout / Maverick** (gated; architectural claims sourced from Meta blog + HF announcement + Transformers PR).
- **Gemma 4** (fully verified per-size `config.json` across all five sizes per `research/issues/10-gemma4-investigation.md`; no first-party arXiv tech report yet as of cutoff).
- **Qwen3-Next** (HF card + Alibaba blog; full config not pulled in this pass).
- **DeepSeek-V4-Pro / V4-Flash** (HF cards + DeepSeek tech report PDF; full configs not pulled).
- **DeepSeek-V3.2-Exp / V3.2** (HF cards + arXiv:2512.02556).
- **Granite 4.0 / 4.1** (HF cards + IBM blog; configs partially verified).
- **NVIDIA Nemotron 3 Super / Ultra / Nano Omni** (HF announcements + NVIDIA dev blog).
- **MiniMax-Text-01 / M2 / M3** (M3 weights staged as of cutoff).
- **Mamba-3** (ICLR 2026 paper + GitHub reference implementation; production weights not yet released).
- **xLSTM 7B** (HF card; config available but not pulled in v2; promoted from "noted not pulled" to first-class in v3).
- **DeepSeek-OCR / DeepSeek-OCR-2** (HF cards + GitHub repo + paper).

**Pre-release announcements (no weights as of 2026-06-06):**
- Meta Muse Spark (Apr 2026, hosted only)
- Microsoft MAI-Thinking-1 (Jun 2026 Build, private preview)
- Microsoft Aion 1.0 (Jun 2026 Build, weights staged for July 2026)
- xAI Grok 4.3 (May 2026, API only)
- xAI Grok Build 0.1 (May 2026, API only)
- xAI Grok 3 open weights (promised Feb 2026, still unreleased)
- Baidu ERNIE 5.1 (May 2026, API only)
- Qwen3.7-Max (May 2026, hosted on chat.qwen.ai)
- MiniMax-M3.0 (Jun 2026, weights staged)

**Specifically not catalogued but architecturally relevant (out-of-scope per v3 size criterion or framing):**
- Mixtral 8×7B (13B active, baseline for MoE arc; cataloged in v2 §3.2).
- DBRX (132B/36B-A, out of scope).
- DeepSeek-V3 671B (out of scope).
- Pre-LLM OCR (Tesseract, PaddleOCR-classic, ABINet, TrOCR-text-line) — cataloged in `research/06`.
- LayoutLMv3, DiT, StrucTextv2/v3 (encoder-only — not LM-decoder; in `research/06` for context).
- Donut (NAVER 2021) and Nougat (Meta 2023-08) — pre-LLM era reference; in `research/06` and §5.20 OCR arc.

**v3 known omissions:**
- **Phi-5** not confirmed released as of cutoff; references found are deployment guides on third-party blogs, not formal Microsoft release.
- **SmolLM4** not found; HuggingFace TB remains at SmolLM3-3B.
- **Tulu 4 / OLMo 4 / InternLM 4 / Yi 2 / Jamba 2 successor / Hymba 2** all absent in 2026-Q2 window.
- **Apple AFM open weights** (still closed as of cutoff; WWDC 2026 falls outside window).
- **DeepSeek R2** not released; reports indicate Liang Wenfeng has not greenlit for performance reasons.
- **MEGABYTE / Byte Latent Transformer (BLT)** noted only as a token-free architecture footnote in axis A7; not promoted to first-class row.
- **Reasoning-distill recipes (s1, Sky-T1, Marco-o1, DeepHermes-3)** noted as recipes not architectures; not rowed.

**Note on the Apple AFM promotion:** v2 listed AFM as "noted closed" because the architecture was unverifiable. v3 promotes it to first-class because the arch is now fully described in arXiv:2507.13575 (the configs remain closed, but the cross-block KV sharing axis is now characterized). This is the only "closed but promoted" row in v3.

---

## 10. Source Citations

**Tech reports and authoritative blogs (added/updated for v3, in addition to v2's citation list):**

- **Gemma 4 launch**: blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ (2026-04-02); DeepMind page deepmind.google/models/gemma/gemma-4/; HF announcement huggingface.co/blog/gemma4; HF docs huggingface.co/docs/transformers/model_doc/gemma4; Google model card ai.google.dev/gemma/docs/core/model_card_4; **QAT blog** blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/ (2026-06-05). HF configs verified: google/gemma-4-{E2B,E4B,12B-it,26B-A4B-it,31B,31B-it}/raw/main/config.json. Per `research/issues/10-gemma4-investigation.md`. **No first-party arXiv tech report yet** as of 2026-06-06.
- **Llama 4 Scout / Maverick**: Meta blog ai.meta.com/blog/llama-4-multimodal-intelligence/ (2025-04-05); HF model cards meta-llama/Llama-4-Scout-17B-16E and Llama-4-Maverick-17B-128E; HF blog post huggingface.co/blog/llama4-release.
- **Qwen3-Next**: Alibaba blog alibabacloud.com/blog/qwen3-next-a-new-generation-of-ultra-efficient-model-architecture-unveiled_602536 (2025-09-11); HF model card Qwen/Qwen3-Next-80B-A3B-Instruct.
- **DeepSeek-V3.1 / V3.2-Exp / V3.2**: arXiv:2512.02556 (V3.2 with DSA); HF cards deepseek-ai/DeepSeek-V3.1, deepseek-ai/DeepSeek-V3.2-Exp, deepseek-ai/DeepSeek-V3.2; GitHub github.com/deepseek-ai/DeepSeek-V3.2-Exp.
- **DeepSeek-V4-Pro / V4-Flash**: HF cards deepseek-ai/DeepSeek-V4-Pro and DeepSeek-V4-Flash (2026-04-24); api-docs.deepseek.com/news/news260424; tech report PDF in HF repo. **NVIDIA quantized**: huggingface.co/nvidia/DeepSeek-V4-Pro-NVFP4.
- **DeepSeek-OCR**: HF deepseek-ai/DeepSeek-OCR; GitHub github.com/deepseek-ai/DeepSeek-OCR; arXiv:2510.18234. **DeepSeek-OCR-2**: github.com/deepseek-ai/DeepSeek-OCR-2 (2026-Q1).
- **Granite 4.0 / 4.1**: IBM blog ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models; research.ibm.com/blog/granite-4-1-ai-foundation-models; HF cards ibm-granite/granite-4.0-h-{micro,tiny,small}, ibm-granite/granite-4.1-{3b,8b,30b}, ibm-granite/granite-vision-4.1-4b, ibm-granite/granite-speech-4.1-{AR-2B,NAR-2B}, ibm-granite/granite-guardian-4.1.
- **GPT-OSS 20B / 120B**: OpenAI blog openai.com/index/introducing-gpt-oss/; arXiv:2508.10925; HF cards openai/gpt-oss-20b, openai/gpt-oss-120b; GitHub github.com/openai/gpt-oss.
- **Apple Foundation Model 2025**: arXiv:2507.13575; machinelearning.apple.com/research/apple-foundation-models-2025-updates.
- **NVIDIA Nemotron 3 (Super / Nano Omni / Ultra / Nano 4B)**: developer.nvidia.com/blog/introducing-nemotron-3-super (2026-03-11); blogs.nvidia.com/blog/nemotron-3-nano-omni-multimodal-ai-agents (2026-04-28); marktechpost.com/.../nvidia-ai-releases-nemotron-3-ultra (2026-06-04); HF nvidia/Nemotron-3-{Super,Nano-Omni,Ultra,Nano-4B}; arXiv:2511.03929 (Nemotron Nano V2 VL precursor).
- **Llama-3.1-Nemotron-Nano 8B**: HF nvidia/Llama-3.1-Nemotron-Nano-8B-v1.
- **olmOCR / olmOCR-2**: arXiv:2502.18443 (olmOCR), arXiv:2510.19817 (olmOCR-2); allenai.org/blog/olmocr-2; HF allenai/olmOCR-2-7B-1025.
- **MinerU2.5**: arXiv:2509.22186; HF opendatalab/MinerU2.5-Pro-2604-1.2B.
- **MonkeyOCR / MonkeyOCR v1.5**: arXiv:2506.05218 (v1) / arXiv:2511.10390 (v1.5); HF echo840/MonkeyOCR, echo840/MonkeyOCR-v1.5.
- **HunyuanOCR**: arXiv:2511.19575; HF tencent/HunyuanOCR.
- **PaddleOCR-VL / PaddleOCR-VL-1.5**: HF PaddlePaddle/paddleocr-vl-1.5.
- **GOT-OCR 2.0**: arXiv:2409.01704 (Wei et al.); HF stepfun-ai/GOT-OCR-2.0-hf.
- **Mistral OCR 3**: mistral.ai/news/mistral-ocr-3/ (closed API).
- **Surya 2**: HF datalab-to/surya-ocr-2.
- **Dolphin (ByteDance) / Dolphin-v2**: arXiv:2505.14059 / arXiv:2602.05384; HF ByteDance/Dolphin{,_v2}.
- **Nanonets-OCR-s / OCR2-3B**: HF nanonets/Nanonets-OCR-s, nanonets/Nanonets-OCR2-3B.
- **RolmOCR**: HF reducto/RolmOCR.
- **Janus-Pro 1B / 7B**: arXiv:2501.17811; HF deepseek-ai/Janus-Pro-{1B,7B}.
- **MiniCPM-V 2.6 / MiniCPM-o 2.6 / MiniCPM-Llama3-V 2.5**: arXiv:2408.01800; HF openbmb/MiniCPM-V-2_6, openbmb/MiniCPM-o-2_6, openbmb/MiniCPM-Llama3-V-2_5.
- **InternVL 2 / 2.5 / 3 / 3.5**: arXiv:2404.16821 / 2412.05271 / 2504.10479 / 2508.18265.
- **Molmo / MolmoE**: arXiv:2409.17146; HF allenai/Molmo-{7B-D,7B-O}-0924, allenai/MolmoE-1B-0924.
- **Pixtral 12B**: arXiv:2410.07073; HF mistralai/Pixtral-12B-2409.
- **SmolVLM / SmolVLM2**: arXiv:2504.05299; HF HuggingFaceTB/SmolVLM-Instruct, HuggingFaceTB/SmolVLM2-2.2B-Instruct.
- **Ministral 3B / 8B**: mistral.ai/news/ministraux/; HF mistralai/Ministral-{3B,8B}-Instruct-2410.
- **Phi-4-multimodal**: arXiv:2503.01743; HF microsoft/Phi-4-multimodal-instruct.
- **Mistral Small 3 / 3.1 / 3.2**: mistral.ai/news/mistral-small-3-1/; HF mistralai/Mistral-Small-3{,.1,.2}-24B-Instruct-{2501,2503,2506}.
- **Mistral Small 4 / Medium 3.5 / Voxtral TTS**: HF mistralai/Mistral-Small-4-119B-2603, mistralai/Mistral-Medium-3.5, mistralai/voxtral-tts (2026 releases).
- **Mistral Magistral**: HF mistralai/Magistral-Small.
- **Moshi 7B / Helium 1 2B**: arXiv:2410.00037 (Moshi); HF kyutai/moshiko-pytorch-bf16, kyutai/helium-1-2b; GitHub github.com/kyutai-labs/moshi.
- **Qwen 2-Audio 7B / Qwen 2.5-Omni 3B/7B / Qwen 3.5-Omni**: HF Qwen/Qwen2-Audio-7B-Instruct, Qwen/Qwen2.5-Omni-{3B,7B}, Qwen/Qwen3.5-Omni; arXiv:2604.15804 (Qwen3.5-Omni paper); GitHub github.com/QwenLM/{Qwen2.5-Omni,Qwen3-Omni}.
- **Qwen 2.5-VL / Qwen 3-VL**: arXiv:2502.13923 (Qwen 2.5-VL); HF Qwen/Qwen2.5-VL-{3B,7B,72B}-Instruct, Qwen/Qwen3-VL-8B-Instruct; GitHub github.com/QwenLM/{Qwen2.5-VL,Qwen3-VL}.
- **xLSTM 7B (NXAI)**: arXiv:2503.13427; HF NX-AI/xLSTM-7b; nx-ai.com/en/news/xlstm-7b-nxai-releases-its-new-xlstm-7b-model.
- **MiniMax-Text-01 / M-series**: arXiv:2501.08313 (M-Text-01); arXiv:2605.26494 (M2 reverting Lightning); marktechpost.com/.../minimax-releases-minimax-m3 (M3 MSA); HF MiniMaxAI/MiniMax-Text-01, MiniMaxAI/MiniMax-M2, MiniMaxAI/MiniMax-M2.5, MiniMaxAI/MiniMax-M2.7, MiniMaxAI/MiniMax-M3.0; GitHub github.com/MiniMax-AI/MiniMax-01.
- **Hunyuan 0.5B/1.8B/4B/7B**: GitHub github.com/Tencent-Hunyuan/Hunyuan-7B; HF tencent/Hunyuan-{0.5B,1.8B,4B,7B}-Instruct.
- **Hunyuan-A13B / Hy3 Preview / Hy-MT2 / HY-Embodied**: HF tencent/Hunyuan-A13B-Instruct, tencent/Hy3-preview, tencent/Hy-MT2, tencent/HY-Embodied; GitHub github.com/Tencent-Hunyuan.
- **Ovis 2.5**: arXiv:2508.11737; HF AIDC-AI/Ovis2.5-{2B,9B}.
- **Eagle 2**: HF nvidia/Eagle2-{1B,2B,9B}; GitHub github.com/NVlabs/EAGLE.
- **Cambrian-1**: arXiv:2406.16860; HF nyu-visionx/cambrian-8b.
- **NVILA**: arXiv:2412.04468; HF Efficient-Large-Model/NVILA-*.
- **Llama 3.2 Vision**: ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/; HF meta-llama/Llama-3.2-{11B,90B}-Vision-Instruct.
- **Chameleon (Meta)**: arXiv:2405.09818; HF meta-llama/Chameleon-{7B,30B}.
- **TextHawk / TextHawk2**: arXiv:2404.09204 / arXiv:2410.05261; HF yuyq96/TextHawk{,2}.
- **DocPedia**: arXiv:2311.11810.
- **Vary / Vary-toy**: arXiv:2312.06109; HF HaoranWei/Vary-toy.
- **mPLUG-DocOwl 1.5 / DocOwl2**: arXiv:2403.12895 / arXiv:2409.03420; HF mPLUG/DocOwl{1.5,2}.
- **EXAONE 3.5 / 4.5**: github.com/LG-AI-EXAONE/EXAONE-{3.5,4.5}; HF LGAI-EXAONE/EXAONE-{3.5-2.4B,3.5-7.8B,4.5-33B}-Instruct.
- **Falcon-H1**: arXiv:2507.22448; HF tiiuae/Falcon-H1-{0.5B,1.5B,3B,7B}-Base.
- **Falcon-Edge 1B / 3B (1.58-bit)**: huggingface.co/blog/tiiuae/falcon-edge (2026-05); HF tiiuae/Falcon-Edge-{1B,3B}.
- **Falcon Perception 0.6B**: tii.ae/news/tii-launches-falcon-perception; HF tiiuae/Falcon-Perception.
- **Mamba-3**: arXiv:2603.15569 (ICLR 2026); openreview.net/forum?id=HwCvaJOiCj; GitHub github.com/state-spaces/mamba.
- **GLM-5 / GLM-5V-Turbo / GLM-5.1**: huggingface.co/blog/mlabonne/glm-5; HF zai-org/GLM-5.1, zai-org/GLM-5V-Turbo; GitHub github.com/zai-org/GLM-V.
- **Cohere Command A+**: docs.cohere.com/changelog/command-a-plus-05-2026; cohere.com/blog/command-a-plus (2026-05); HF CohereLabs/command-a-plus-05-2026.
- **Liquid LFM2.5 series**: huggingface.co/LiquidAI/LFM2-VL-450M; huggingface.co/LiquidAI/LFM2.5-8B-A1B; liquid.ai/blog (LFM2.5 series).
- **Zyphra ZAYA1-8B**: HF Zyphra/ZAYA1-8B.
- **StepFun Step 3.7 Flash**: HF stepfun-ai/Step-3.7-Flash; pandaily.com/.../stepfun-open-source-step-3-7-flash.
- **Moonshot Kimi K2.6**: HF moonshotai/Kimi-K2.6; kimi-k2.org/blog/24-kimi-k2-6-release.
- **Xiaomi MiMo-V2.5 / -Pro / -ASR**: HF XiaomiMiMo/MiMo-V2.5, XiaomiMiMo/MiMo-V2.5-Pro, XiaomiMiMo/MiMo-V2.5-ASR; mimo.xiaomi.com/mimo-v2-5.
- **Reka Edge 2026 / Reka Flash 3.1**: HF RekaAI/Reka-Edge-2026, RekaAI/reka-flash-3.1; reka.ai/news.
- **Llama-Nemotron-Nano (V1, V2 VL)**: arXiv:2511.03929 (Nemotron Nano V2 VL); HF nvidia/Llama-3.1-Nemotron-Nano-8B-v1.
- **Hermes Agent**: HF NousResearch/Hermes-Agent-{v0.9,v0.11,v0.14,v0.15.2}; hermes-agent.nousresearch.com.
- **Pix2Struct / Donut / Nougat / Kosmos-2.5**: arXiv:2210.03347 / 2111.15664 / 2308.13418 / 2309.11419 (for OCR-LLM lineage in §5.20).

**HuggingFace config sources verified during v3 production**: as v2 + Gemma 4 E2B/E4B/12B/26B-A4B/31B (per `research/issues/10-gemma4-investigation.md`), DeepSeek-OCR, MinerU2.5, GOT-OCR2.0, olmOCR-2, Llama 4 Scout, Qwen3-Next, Granite 4.0/4.1.

**Cross-references**: per-layer arithmetic in `research/02-layer-sources.v3.md` (38 axes, 48 family sections); KV-cache and attention-variant details in `research/05-kvcache-attention.v3.md` (38 attn / 18 RoPE / 18 cache variants); OCR-LLM catalog in `research/06-ocr-vlm-extensions.md`; recent releases in `research/08-recent-releases-2026q2.md`; coverage audit in `research/issues/08-coverage-justification.md`; Gemma 4 deep-dive in `research/issues/10-gemma4-investigation.md`.

---

*End of v3.*
