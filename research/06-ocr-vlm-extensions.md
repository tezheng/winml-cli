# 06 — OCR-LLM and Document-AI-LLM Catalog (2023–2026)

*Stream: vlm/ocr-extensions. Project: llm-layers. Date: 2026-06-06. Version: 1.0.*

*Companion to research/01-model-census (which catalogs <8B text decoders) and research/02-layer-sources (per-family implementation analysis). This file extends the survey into the OCR/document-AI sub-domain where vision encoders are bolted onto LM decoders; we are interested in (a) which LM decoders are reused as backbones, (b) which OCR/doc-AI families introduce architecturally-novel choices that the layer-IR must parametrize, and (c) which deserve a `models/<name>/` slot in this repo.*

---

## 1. Scope

### 1.1 What counts as an "OCR-LLM" for this catalog

An OCR-LLM (or document-AI-LLM) for the purposes of this survey is a model that:

1. **Has a transformer/LLM decoder** that auto-regressively emits text tokens, distinguishing it from pre-LLM OCR (Tesseract, PaddleOCR-classic, EasyOCR, ABINet, CRNN, TrOCR-text-line). The decoder must be at least loosely "Llama-shaped" (decoder-only, multi-head attention, FFN, autoregressive). Encoder-decoder Donut-style models *are included* as a historical bridge because they still use a transformer decoder.
2. **Has a vision pathway that injects pixels** (not OCR-text-as-input) into the decoder context. This excludes pure layout-text models like LayoutLM (which consume OCR-tokens, not pixels). LayoutLMv3 is included as a historical reference but flagged "not an LLM-decoder model".
3. **Is publicly shipped** with either an HF model card, a vendor tech report, or both. Closed OCR APIs (Mistral OCR 3 endpoint, Google Document AI, Azure Document Intelligence) are noted but excluded from the architectural axes work.
4. **Ships in 2023–2026** (Donut 2021 and Pix2Struct 2023 are included as the LM-decoder ancestors of the wave).

The catalog spans three loose tiers:

- **Tier A — Dedicated OCR/document models**: GOT-OCR2, DeepSeek-OCR, olmOCR, MinerU2.5, MonkeyOCR, HunyuanOCR, Nanonets-OCR, PaddleOCR-VL, Surya 2, Dolphin (ByteDance), Vary-toy. These are trained explicitly for OCR/parsing and ship a vision encoder + small LM decoder.
- **Tier B — General MLLMs with strong OCR**: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, MiniCPM-V 2.6, InternVL3/3.5, DeepSeek-VL2, Phi-3.5-Vision, Phi-4-multimodal, Llama 3.2 Vision, LLaVA-OneVision, Molmo/MolmoE, Cambrian-1, NVILA, Eagle 2, Pixtral 12B, Florence-2, BLIP-3 / xGen-MM, Ovis2/2.5.
- **Tier C — Historical bridge models**: Donut, Nougat, Pix2Struct, Kosmos-2.5, DocPedia, mPLUG-DocOwl 1.5 / DocOwl2, UReader, TextHawk/TextHawk2. These predate the consensus or use encoder-decoder shape.

### 1.2 Relation to research/01 and research/02

research/01 catalogs **text-only** <8B decoders (Qwen3, Llama 3.2 1B/3B, Phi-3 Mini, etc.). research/02 extracts the **per-layer source** of those decoders from HF/vLLM/llama.cpp. This catalog (research/06) extends the picture: **most "OCR-LLMs" in 2024–2026 are a known <8B text decoder from research/01 + a vision encoder + a connector**. The LM-portion is *exactly* a research/01 model in most cases — Qwen2.5-7B for olmOCR, Qwen2.5-VL-3B (Qwen2.5-3B + ViT) for Nanonets-OCR, Qwen2-0.5B for GOT-OCR2 and MinerU2.5, Phi-3-Mini for xGen-MM, Phi-4-Mini for Phi-4-multimodal, etc. The architectural axes of the LM-portion are therefore **already covered** by research/01's 15-axis taxonomy. The interesting question is: **does the vision-side bring novel LM-decoder modifications** (cross-attention, layout-aware position encoding, visual-causal masks, low-rank visual KV caches, 2D-RoPE, multi-resolution patch handling, optical compression) that the layer-IR must additionally parametrize?

The short answer (Section 4): **yes, but the deltas are concentrated in five axes** — (a) 2D-RoPE / M-RoPE / per-dim position encoding; (b) cross-attention adapter layers (Llama 3.2 Vision); (c) "visual causal flow" attention masking (DeepSeek-OCR-2); (d) MoE-on-active-decoder (DeepSeek-OCR's 3B/A570M); (e) mixture-of-LoRAs (Phi-4-multimodal). Most other extensions live in the *vision encoder* (NaViT, dynamic tiling, multi-tower MoVE) and are outside the LM-decoder scope of llm-layers.

---

## 2. Catalog table

Columns: `model | org | release | total params | LM-decoder | LM-portion in <8B scope? | LM base / arch | HF | repo | arxiv | novel arch element`.

| Model | Org | Released | Total | LM-decoder portion | LM <8B scope? | LM base / arch summary | HF | Repo | arxiv | Novel arch element |
|---|---|---|---|---|---|---|---|---|---|---|
| **Donut** | NAVER (CLOVA) | 2021-11 | 200–700M | BART-style decoder (~200M) | ✓ | Swin-T encoder + BART-style decoder, encoder-decoder | naver-clova-ix/donut-base | clovaai/donut | 2111.15664 | First OCR-free doc-VLM; encoder-decoder w/ Swin |
| **Pix2Struct** | Google | 2023-03 | 282M / 1.3B | T5-style encoder-decoder | ✓ | ViT image encoder + T5 decoder; variable resolution; rendered text-as-image | google/pix2struct-base | google-research/pix2struct | 2210.03347 | Screenshot pretraining; question-rendered-on-image |
| **Nougat** | Meta FAIR | 2023-08 | 350M / 1.3B | mBART decoder | ✓ | Swin encoder + mBART decoder; identical to Donut shape | facebook/nougat-base | facebookresearch/nougat | 2308.13418 | Academic-PDF specialist; LaTeX output |
| **Kosmos-2.5** | Microsoft | 2023-09 | 1.37B | Magneto decoder-only | ✓ | Pix2Struct-style ViT + Magneto/sub-LN decoder; produces markdown + bbox spans | microsoft/kosmos-2.5 | microsoft/unilm | 2309.11419 | Spatial-aware text blocks; bbox-as-tokens |
| **Vary / Vary-toy** | Megvii | 2023-12 / 2024-01 | 1.8B (Vary-toy) | Qwen-1.8B | ✓ | SAM-base vision-vocab + CLIP + Qwen-1.8B; dual-tower vision encoder | HaoranWei/Vary-toy | Ucas-HaoranWei/Vary | 2312.06109 | "Vision vocabulary" trained via OPT-125M autoregressive proxy |
| **DocPedia** | USTC + JD | 2023-11 | ~7B | Vicuna-7B | ✓ | DCT-frequency-domain vision input (no spatial vit); 2560×2560 res | not released | docpedia.github.io | 2311.11810 | **Frequency-domain vision tokens** (JPEG DCT coefficients, not spatial patches) |
| **UReader** | Alibaba | 2023-10 | ~7B | mPLUG-Owl (LLaMA-7B) | ✓ | Shape-adaptive cropping + low-res ViT/L-14 encoder × N sub-images | LukeForeverYoung/UReader | LukeForeverYoung/UReader | 2310.05126 | Shape-adaptive cropping (anyres ancestor) |
| **TextHawk** | Tencent | 2024-04 | ~7B | InternLM-7B | ✓ | Scalable Positional Embeddings (SPE) + Query Proposal Net + Multi-Level Cross-Attn (MLCA) | yuyq96/TextHawk | yuyq96/TextHawk | 2404.09204 | **Scalable Positional Embeddings (SPE)** for variable-resolution images |
| **TextHawk2** | Tencent | 2024-10 | ~7B | InternLM2-7B | ✓ | SPE + MLCA + 16× fewer image tokens than v1 | yuyq96/TextHawk2 | yuyq96/TextHawk2 | 2410.05261 | Aggressive visual token compression (16× reduction) |
| **mPLUG-DocOwl 1.5** | Alibaba | 2024-03 | 7.6B | LLaMA-2-7B (mPLUG-Owl2 base) | ✓ | ViT/L-14 (CLIP) + H-Reducer + LLaMA-7B w/ Modality Adaptive Module | mPLUG/DocOwl1.5 | X-PLUG/mPLUG-DocOwl | 2403.12895 | **H-Reducer** (1D-conv horizontal merging) preserves layout |
| **mPLUG-DocOwl2** | Alibaba | 2024-09 | 8.1B | LLaMA2-7B | ✓ | High-Resolution DocCompressor → 324 tokens per page → multi-page | mPLUG/DocOwl2 | X-PLUG/mPLUG-DocOwl | 2409.03420 | Multi-page document compression |
| **Florence-2** | Microsoft | 2024-06 | 232M / 771M | BART-style encoder-decoder (~80M / ~315M) | ✓ | DaViT vision encoder + BART decoder; *prompt-based unified task interface* | microsoft/Florence-2-base / -large | microsoft/Florence-2 | 2311.06242 | Unified prompt-task interface w/ encoder-decoder; OCR is a prompt mode |
| **GOT-OCR 2.0** | StepFun + UCAS | 2024-09 | 580M | Qwen-0.5B (~500M) | ✓ | VitDet 1024² → 256 tokens → 1024-d linear → Qwen-0.5B decoder | stepfun-ai/GOT-OCR-2.0-hf | Ucas-HaoranWei/GOT-OCR2.0 | 2409.01704 | **High-compression encoder** 1024² → 256 vis tokens; tiny decoder |
| **DeepSeek-VL2** | DeepSeek | 2024-12 | 3B / 16B / 27B | DeepSeek-V2-Lite (MLA) / V2 MoE; 1.0/2.8/4.5B active | ✓ (Tiny+Small) | SigLIP-SO400M + dynamic tiling + DeepSeek-MoE w/ **MLA** | deepseek-ai/deepseek-vl2 | deepseek-ai/DeepSeek-VL2 | 2412.10302 | **MLA + MoE at <5B-active w/ vision** |
| **DeepSeek-OCR** | DeepSeek | 2025-10 | 3.4B total | DeepSeek3B-MoE-A570M (active 570M) | ✓ | DeepEncoder (SAM-base 80M + CLIP-L 300M serial) + 3B MoE decoder (A570M active) | deepseek-ai/DeepSeek-OCR | deepseek-ai/DeepSeek-OCR | 2510.18234 | **Optical context compression**; serial SAM+CLIP encoder; MoE at decoder |
| **DeepSeek-OCR-2** | DeepSeek | 2026-Q1 | 3.4B+ | DeepSeek3B-MoE-A570M variant | ✓ | + **Visual Causal Flow** attention masking | deepseek-ai/DeepSeek-OCR-2 | deepseek-ai/DeepSeek-OCR | (TBD) | **Visual Causal Flow** mask (vision tokens bidirectional within page, causal cross-page) |
| **olmOCR (0225 / 0725)** | Allen AI | 2025-02 / 2025-07 | 8B | Qwen2-VL-7B base | ✗ (LM is 7.6B but VLM wrap is 8B+) | Qwen2-VL-7B SFT'd on olmOCR-mix | allenai/olmOCR-7B-0225-preview | allenai/olmocr | 2502.18443 | RL/GRPO post-training on doc-OCR rubrics |
| **olmOCR-2 (1025)** | Allen AI | 2025-10 | 8B | Qwen2.5-VL-7B base | ✗ (8B) | Qwen2.5-VL-7B SFT + GRPO on unit-test rewards | allenai/olmOCR-2-7B-1025 | allenai/olmocr | 2510.19817 | **Unit-test rewards for OCR** (math/table verification as RL signal) |
| **MinerU2.5** | Shanghai AI Lab | 2025-09 | 1.2B | Qwen2-Instruct-0.5B | ✓ | NaViT 675M (Qwen2-VL init) + patch-merger + Qwen2-Instruct-0.5B; **two-stage layout→recognition** | opendatalab/MinerU2.5 | opendatalab/MinerU | 2509.22186 | Decoupled coarse-to-fine pipeline; NaViT + 2D-RoPE |
| **MonkeyOCR** | Huazhong UST | 2025-06 | 3B (also 0.6/1.2B) | InternLM2 / Qwen2.5-VL | ✓ | Donut-style encoder-decoder; SRR (Structure-Recognition-Relation) triplet | echo840/MonkeyOCR | Yuliang-Liu/MonkeyOCR | 2506.05218 | SRR triplet paradigm; CPD parameter degradation |
| **MonkeyOCR v1.5** | Huazhong UST | 2025-11 | 3B | Qwen2.5-VL-3B | ✓ | + Robust pattern handling | echo840/MonkeyOCR-v1.5 | Yuliang-Liu/MonkeyOCR | 2511.10390 | Same arch; robustness improvements |
| **HunyuanOCR** | Tencent | 2025-11 | 1B | Hunyuan-0.5B native LLM | ✓ | 0.4B native-res ViT + learnable pooling MLP + 0.5B Hunyuan LLM | tencent/HunyuanOCR | Tencent-Hunyuan/HunyuanOCR | 2511.19575 | **Native multimodal** training (no LM-only pretrain stage) |
| **Mistral OCR 3** | Mistral AI | 2025-12 | (API-only) | (closed) | (unknown) | VLM trained for doc parsing; markdown output w/ HTML tables | (API) | (closed) | n/a | API-only; not architecturally analyzable |
| **Dolphin (ByteDance)** | ByteDance | 2025-05 (ACL) | 322M | Swin-encoder + small decoder | ✓ | Donut-shape; analyze-then-parse two-stage; heterogeneous anchor prompts | ByteDance/Dolphin | bytedance/Dolphin | 2505.14059 | Heterogeneous anchor prompts (layout-element tokens as conditioning) |
| **Dolphin-v2** | ByteDance | 2025-Q4 | 3B | Qwen2.5-VL-3B | ✓ | Qwen2.5-VL-3B + document-type-aware two-stage | ByteDance/Dolphin-v2 | bytedance/Dolphin | 2602.05384 | Document-type-aware routing (digital-born vs photographed) |
| **Nanonets-OCR-s** | Nanonets | 2025-05 | 3B | Qwen2.5-VL-3B-Instruct | ✓ | Qwen2.5-VL-3B fine-tuned for image→markdown w/ semantic tags | nanonets/Nanonets-OCR-s | (private) | n/a | Semantic-tag-aware fine-tune (latex/img-desc/checkbox tokens) |
| **Nanonets-OCR2-3B** | Nanonets | 2025-12 | 3B | Qwen2.5-VL-3B | ✓ | Same arch; 125K ctx; VQA support | nanonets/Nanonets-OCR2-3B | (private) | n/a | Extended context for long docs |
| **RolmOCR** | Reducto | 2025-07 | 8B | Qwen2.5-VL-7B | ✗ (8B) | Qwen2.5-VL-7B SFT; speed-optimized | reducto/RolmOCR | (private) | n/a | Speed-tuned variant of Qwen2.5-VL |
| **Surya OCR 2** | Datalab | 2025-10 | 650M | Qwen3-style decoder (~650M) | ✓ | Shared VLM for layout + OCR + tables; **EfficientViT segformer** for line-detection (separate) | datalab-to/surya-ocr-2 | datalab-to/surya | n/a | Single VLM emits layout-JSON or full-page-HTML; separate segformer pre-stage |
| **PaddleOCR-VL** | Baidu | 2025-10 | 0.9B | ERNIE-4.5-0.3B | ✓ | NaViT-style dynamic-res encoder + Adaptive MLP + ERNIE-4.5-0.3B | (private/Paddle) | PaddlePaddle/PaddleOCR | n/a | ERNIE-4.5-0.3B as smallest production LM-decoder |
| **PaddleOCR-VL-1.5** | Baidu | 2026-01 | 0.9B | ERNIE-4.5-0.3B | ✓ | Same arch; SOTA on OmniDocBench v1.5 (94.5) | PaddlePaddle/paddleocr-vl-1.5 | PaddlePaddle/PaddleOCR | n/a | Same |
| **Qwen2-VL** | Alibaba | 2024-08 | 2B / 7B / 72B | Qwen2-1.5B / 7B / 72B | ✓ (2B,7B) | ViT + **M-RoPE** + Qwen2 decoder; dynamic resolution; native aspect ratio | Qwen/Qwen2-VL-{2B,7B,72B}-Instruct | QwenLM/Qwen2-VL | 2409.12191 | **M-RoPE** (multimodal-RoPE: t/h/w dims); naive dyn-res ViT |
| **Qwen2.5-VL** | Alibaba | 2025-01 | 3B / 7B / 72B | Qwen2.5-3B / 7B / 72B | ✓ (3B,7B) | Redesigned ViT (window attn, SwiGLU, RMSNorm) + 2D-RoPE + Qwen2.5 decoder | Qwen/Qwen2.5-VL-{3B,7B,72B}-Instruct | QwenLM/Qwen2.5-VL | 2502.13923 | ViT uses SwiGLU+RMSNorm (LM-style ViT); window attn; M-RoPE retained |
| **Qwen3-VL** | Alibaba | 2025-Q4 | 8B / others | Qwen3-8B | ✗ (8B) | Qwen3 backbone + Qwen2.5-VL-style ViT; better low-light/blur OCR; 32 langs | Qwen/Qwen3-VL-8B-Instruct | QwenLM/Qwen3-VL | (TBD) | Qwen3-decoder w/ vision wrap |
| **MiniCPM-V 2.6** | OpenBMB / Tsinghua | 2024-08 | 8.5B | Qwen2-7B | ✗ (7.6B LM, total 8.5B) | SigLIP-400M + Qwen2-7B; perceiver-resampler → 640 tokens / 1.8M-pixel image | openbmb/MiniCPM-V-2_6 | OpenBMB/MiniCPM-V | 2408.01800 | Adaptive visual encoding; resampler compresses to ≤640 tokens |
| **MiniCPM-Llama3-V 2.5** | OpenBMB / Tsinghua | 2024-05 | 8.5B | Llama-3-8B | ✗ (8B) | SigLIP + Llama-3-8B; same perceiver resampler | openbmb/MiniCPM-Llama3-V-2_5 | OpenBMB/MiniCPM-V | 2408.01800 | Llama-3 backbone variant |
| **InternVL 2** | Shanghai AI Lab | 2024-07 | 1B–76B | InternLM/Qwen2 variants | ✓ (sub-8B) | InternViT + MLP + LLM; pixel-unshuffle 4× token reduction | OpenGVLab/InternVL2-{1B,2B,4B,8B} | OpenGVLab/InternVL | 2404.16821 | Pixel-unshuffle visual compression |
| **InternVL 3** | Shanghai AI Lab | 2025-04 | 1B / 2B / 8B / 14B / 38B / 78B | Qwen2.5 / InternLM3 | ✓ (1B,2B,8B-borderline) | InternViT + MLP + Qwen2.5/InternLM3; **V2PE** variable position encoding | OpenGVLab/InternVL3-{1B,2B,8B} | OpenGVLab/InternVL | 2504.10479 | **V2PE** (Variable Visual Position Encoding) for long multimodal ctx |
| **InternVL 3.5** | Shanghai AI Lab | 2025-08 | 1B–241B | Qwen2.5/InternLM3 variants | ✓ (sub-8B) | Same shape; improved training recipe | OpenGVLab/InternVL3_5-{...} | OpenGVLab/InternVL | 2508.18265 | Same; training-recipe scaling |
| **Phi-3-Vision** | Microsoft | 2024-05 | 4.2B | Phi-3-Mini-3.8B | ✓ | CLIP-ViT-L/14 + projector + Phi-3-Mini | microsoft/Phi-3-vision-128k-instruct | microsoft/Phi-3CookBook | 2404.14219 | Compact (4B) doc-capable VLM |
| **Phi-3.5-Vision** | Microsoft | 2024-08 | 4.2B | Phi-3.5-Mini-3.8B | ✓ | CLIP + Phi-3.5-Mini; multi-frame video; OCR tuned | microsoft/Phi-3.5-vision-instruct | (Phi cookbook) | 2404.14219 | Multi-image support; OCR fine-tune |
| **Phi-4-multimodal** | Microsoft | 2025-02 | 5.6B | Phi-4-Mini-3.8B + **mixture-of-LoRAs** | ✓ | Phi-4-Mini-3.8B (frozen) + 370M Vision-LoRA + audio LoRA; **modality-LoRA routing** | microsoft/Phi-4-multimodal-instruct | (Phi cookbook) | 2503.01743 | **Mixture-of-LoRAs** — vision/audio routed through frozen-LM LoRA stacks |
| **Llama 3.2 Vision** | Meta | 2024-09 | 11B / 90B | Llama-3.1-8B / 70B + 7.2B vision adapter | ✗ (11B total, 8B LM-portion is borderline) | ViT-H/14 850M + **cross-attention adapter every 4 LM layers** | meta-llama/Llama-3.2-11B-Vision-Instruct | meta-llama/llama-models | 2407.21783 | **Cross-attention adapter layers** (not concat-prefix) — distinct from Qwen/InternVL pattern |
| **LLaVA-NeXT (1.6)** | UW + ByteDance + others | 2024-01 | 7B / 13B / 34B | Vicuna-7B / Mistral-7B / Yi-34B | ✓ (7B) | CLIP ViT-L/14 + MLP + Vicuna/Mistral; **AnyRes** dynamic tiling | llava-hf/llava-v1.6-* | LLaVA-VL/LLaVA-NeXT | 2407.07895 | **AnyRes** dynamic tiling (1×1 to 2×2 grid of 336² crops) |
| **LLaVA-OneVision** | ByteDance + NTU | 2024-08 | 0.5B / 7B / 72B | Qwen2-0.5B / 7B / 72B | ✓ (0.5B,7B) | SigLIP-384 + MLP + Qwen2; AnyRes-9 (up to 9 patches) | lmms-lab/llava-onevision-qwen2-* | LLaVA-VL/LLaVA-NeXT | 2408.03326 | Unified single-image / multi-image / video |
| **Cambrian-1** | NYU | 2024-06 | 8B / 13B / 34B | Llama-3-8B / Vicuna-13B / Yi-34B | ✗ (8B) | **Spatial Vision Aggregator** + 4 vision encoders (SigLIP/CLIP/DINOv2/ConvNeXt) | nyu-visionx/cambrian-8b | cambrian-mllm/cambrian | 2406.16860 | **Multi-encoder Spatial Vision Aggregator** (4 encoders fused via cross-attn queries) |
| **NVILA** | NVIDIA | 2024-12 | 8B / 15B | Llama-3-8B | ✗ (8B) | "scale-then-compress" visual tokens; SigLIP-SO400M | Efficient-Large-Model/NVILA-* | NVlabs/VILA | 2412.04468 | Scale-then-compress visual token strategy |
| **Eagle 2** | NVIDIA | 2025-01 | 1B / 2B / 9B | Qwen2-1.5B / Llama3-8B etc. | ✓ (1B,2B,9B borderline) | **Mixture of Vision Encoders (MoVE)** — SigLIP + ConvNeXt tiled in parallel | nvidia/Eagle2-{1B,2B,9B} | NVlabs/EAGLE | 2501.14818 | **MoVE** — multi-encoder tiled fusion |
| **Molmo / MolmoE** | Allen AI | 2024-09 | 1B (MoE-1.5/7.2B) / 7B / 72B | OLMoE-1B-7B / OLMo / Qwen2 | ✓ (MolmoE-1B) | CLIP ViT-L/14 + multi-crop + MLP + OLMo/Qwen2; pointing tokens | allenai/Molmo-{7B-D, 7B-O, 72B}; allenai/MolmoE-1B-0924 | allenai/molmo | 2409.17146 | **Pointing tokens** (visual grounding via 2D coordinate emission) |
| **NVIDIA VILA / VILA-2** | NVIDIA | 2023-12 / 2024-07 | 3B / 8B / 13B / 40B | Llama-2 / Llama-3 | ✓ (3B,8B borderline) | CLIP-L + linear + LM; interleaved image-text pretrain | Efficient-Large-Model/VILA* | NVlabs/VILA | 2312.07533 | Interleaved-pretraining recipe |
| **InstructBLIP** | Salesforce | 2023-05 | 7B–13B | Vicuna / FlanT5 | ✓ (7B) | ViT-G + Q-Former + LLM | Salesforce/instructblip-* | salesforce/LAVIS | 2305.06500 | Q-Former (perceiver-like learned-queries) |
| **BLIP-3 / xGen-MM** | Salesforce | 2024-08 | 4–5B | Phi-3-Mini-3.8B | ✓ | ViT + **perceiver resampler** (replaces Q-Former) + Phi-3-Mini | Salesforce/xgen-mm-phi3-mini-instruct-r-v1 | salesforce/LAVIS xgen-mm | 2408.08872 | Perceiver-resampler replaces Q-Former |
| **Ovis 2** | Alibaba (AIDC) | 2025-01 | 1B/2B/4B/8B/16B/34B | Qwen2/Qwen2.5 variants | ✓ (1–8B) | **Visual Embedding Table** — vision-vocab probabilistic tokens align w/ text embeddings | AIDC-AI/Ovis2-{1B,2B,4B,8B} | AIDC-AI/Ovis | 2405.20797 | **Visual vocabulary table** — vision tokens are *probabilistic*, not features |
| **Ovis 2.5** | Alibaba | 2025-08 | 2B / 9B | Qwen2.5-2B / 9B | ✓ (2B) | Visual Embedding Table + native-resolution ViT + thinking mode | AIDC-AI/Ovis2.5-{2B,9B} | AIDC-AI/Ovis | 2508.11737 | Native-res visual perception + reflective reasoning |
| **Pixtral 12B** | Mistral AI | 2024-09 | 12B | Mistral Nemo 12B | ✗ (12B, out of size) | **Pixtral-ViT 400M** (variable image sizes, **2D-RoPE in vision**) + Mistral Nemo decoder | mistralai/Pixtral-12B-2409 | mistralai/pixtral | 2410.07073 | **2D-RoPE in vision encoder** (relative rotary, not learned-absolute) |
| **GLM-4V-9B** | Tsinghua/Zhipu | 2024-06 | 13B | GLM-4-9B | ✗ (9B) | EVA-CLIP + conv adapter + GLM-4-9B | THUDM/glm-4v-9b | THUDM/GLM-4 | n/a | GLM-4 backbone |
| **LayoutLMv3** | Microsoft | 2022 (still cited 2023–2024) | 133M / 368M | encoder-only (BERT-style) | ✓ but **not LLM-decoder** | linear patch embedding + 12L encoder; text+layout+image MLM | microsoft/layoutlmv3-base | microsoft/unilm | 2204.08387 | Encoder-only; text+image MLM; **not in scope** (no LM-decoder) |
| **DiT (Document Image Transformer)** | Microsoft | 2022 | 87M / 304M | encoder-only (ViT BEiT-style) | n/a | ViT encoder pretrained via masked-image modeling on doc images | microsoft/dit-base, microsoft/dit-large | microsoft/unilm | 2203.02378 | Encoder-only; **not in scope** |
| **StrucTexTv2** | Baidu | 2023 | 28M/87M | encoder-only | n/a | text+image masking, encoder-only | (not all on HF) | PaddlePaddle | 2303.00289 | Encoder-only; not in scope |
| **StrucTexTv3** | Baidu | 2024 | ~700M | encoder-only | n/a | doc-encoder | (limited) | PaddlePaddle | n/a | Encoder-only; not in scope |
| **LiLT** | SCUT-DLVCLab | 2022 | 30M | encoder-only | n/a | language-independent layout transformer | SCUT-DLVCLab/lilt-* | jpWang/LiLT | 2202.13669 | Encoder-only; layout-only |
| **LayoutMask** | Tencent | 2023 | encoder-only | n/a | encoder-only | n/a | n/a | 2305.18721 | Encoder-only |

---

## 3. Per-family deep-dives

Five families are architecturally distinct enough to warrant a closer look, plus three more notable for the LM-decoder being shared across the bulk of the ecosystem.

### 3.1 GOT-OCR 2.0 (StepFun, 2024-09)

GOT-OCR 2.0 is the architecturally-purest "small OCR-LLM" in the corpus and the ancestor of the 2025 lightweight-OCR wave. Total params 580M = VitDet vision encoder (~80M) + a 1024×1024 linear connector + **Qwen-0.5B** as the LM decoder. The Qwen-0.5B decoder is a vanilla Qwen-1 architecture: MHA at 24 layers × 14 heads × 64 head_dim (total hidden 896), RoPE θ=10000, RMSNorm pre, SwiGLU MLP, SentencePiece 151936 vocab, no QKV bias for Qwen-1 0.5B (Qwen-1 7B has QKV bias; 0.5B drops it). What makes GOT-OCR2 architecturally interesting is **not** the decoder (which is research/01 row 6 verbatim) but the **encoder→decoder bandwidth budget**: VitDet processes 1024² inputs into 256 visual tokens of dim 1024 — a 4096× spatial-compression rate that the decoder treats as a 256-token prefix. The 1024-dim connector is a single linear projection (no MLP, no resampler, no Q-Former, no cross-attention). Production verdict: deserves a `models/got-ocr2/` slot because (a) the decoder is the smallest production LM decoder in the OCR space, (b) the connector is the minimal possible adapter (just a `nn.Linear`), (c) this is the *referenced* lightweight-OCR shape that DeepSeek-OCR, MinerU2.5, HunyuanOCR all evolved from.

### 3.2 DeepSeek-OCR / DeepSeek-OCR-2 (DeepSeek, 2025-10 / 2026-Q1)

DeepSeek-OCR is the most architecturally-novel OCR-LLM of 2025. Three distinct innovations: (a) a serial two-tower vision encoder (DeepEncoder) — SAM-base 80M (windowed-attention, local detail) → CLIP-large 300M (global attention, semantic) — concatenated rather than parallel-fused, with a token-compression layer in between; (b) a **3B-total / 570M-active MoE decoder** based on DeepSeek-V3 / V2 lineage (so the decoder inherits **Multi-head Latent Attention (MLA)** — q_lora_rank ~512, kv_lora_rank ~64 — which is the same MLA shape used by DeepSeek-V2-Lite from research/01); (c) the **"contexts optical compression"** thesis: image tokens are explicitly framed as a more-efficient encoding of text contexts than text tokens, achieving 97% OCR accuracy at 10× compression and ~60% at 20×. DeepSeek-OCR-2 (2026) adds **Visual Causal Flow** — a mask schema where visual tokens are bidirectional within a page but causal across pages, which is a new attention-mask pattern that does *not* fit the standard `is_causal=True/False` HF API and would require an extension to the layer-IR's `attention_mask_kind` enum. For llm-layers: the decoder itself is a DeepSeek-V3-shape MoE + MLA that research/01 already catalogs at the V2-Lite row; the MoE config (small expert count, MLA at 570M active) is a research/01-extension worth a `models/deepseek-ocr/` slot to test the MLA layer-IR plumbing at a non-text use case.

### 3.3 MinerU2.5 (Shanghai AI Lab, 2025-09)

MinerU2.5 is the most LM-tiny model in the corpus: 1.2B total, with a 675M NaViT vision encoder (initialized from Qwen2-VL) and a 0.5B Qwen2-Instruct LM decoder. The decoder is **literally Qwen2-0.5B** from research/01, i.e., vanilla Qwen2: GQA (14 query / 2 KV / 64 head_dim, 24 layers), RoPE θ=1000000, RMSNorm pre, SwiGLU, tied embeddings (Qwen2-0.5B ties; Qwen2-7B does not). Architectural distinctness comes from the **decoupled coarse-to-fine pipeline**: stage 1 emits layout JSON on a low-resolution view of the whole page; stage 2 crops detected fragments and runs them at native resolution through the same model for recognition. The NaViT image encoder is the architecturally-novel part — it natively supports variable resolution / aspect ratio via 2D-RoPE on the vision side and dynamic patch packing. For llm-layers: MinerU2.5's LM decoder is identical to a research/01 row (Qwen2-0.5B-Instruct), so no new layer-IR work is needed. The model is interesting as a "lightweight OCR built on already-cataloged decoder" case study and could be referenced from `models/qwen2/` without its own slot.

### 3.4 HunyuanOCR (Tencent, 2025-11)

HunyuanOCR is the smallest "from-scratch native multimodal" OCR-LLM in the corpus: 1B total = 0.4B native-resolution ViT + a learnable pooling MLP + 0.5B Hunyuan LLM. The architecturally-distinctive choice is the **native multimodal training** — there is no separate LM-only pretraining stage; the 0.5B Hunyuan LM is trained from scratch with vision tokens interleaved. The 0.5B Hunyuan decoder is a Hunyuan-family architecture (GQA, RoPE, RMSNorm, SwiGLU — research/01 baseline shape), but its tokenizer and exact head-dim configuration are not as widely documented as Qwen's. Production claim: 1B params outperforming much larger OCR models on benchmarks; Tencent positions this as a "deployment-friendly" model. For llm-layers: if the Hunyuan 0.5B is structurally a Qwen2-clone (which appears likely from the technical report), a `models/hunyuan-ocr/` slot is duplicative; if it has Hunyuan-family quirks (Tencent's other models have hybrid SSM+attention variants), a slot becomes worthwhile.

### 3.5 Phi-4-multimodal (Microsoft, 2025-02)

Phi-4-multimodal is the only OCR-capable model in the corpus that uses a **mixture-of-LoRAs** approach. The 3.8B Phi-4-Mini LM is frozen; a 370M Vision LoRA stack is added on top of the LM layers; an audio LoRA is added in parallel; the modality-selection logic routes each input through the appropriate LoRA pathway. This is a fundamentally different architecture extension than the standard concat-prefix-and-finetune approach used by every other model in the catalog. From the layer-IR perspective, mixture-of-LoRAs requires (a) a `lora_routing` axis (which LoRA stack is active per token) and (b) the layer-IR's LoRA support — which currently is post-training PEFT — promoted to a first-class training-time mechanism. For llm-layers: the Phi-4-Mini LM portion is exactly research/01's Phi-4-mini row; the multimodal extension is the LoRA-routing axis. A `models/phi-4-multimodal/` slot would be valuable specifically as the canonical test case for the LoRA-routing-axis extension.

### 3.6 Llama 3.2 Vision (Meta, 2024-09)

Llama 3.2 Vision is the only model in the catalog (alongside Florence-2 in encoder-decoder form) that uses **cross-attention adapter layers** rather than concat-prefix. The Llama-3.1-8B / 70B LM decoder is unmodified; cross-attention adapter layers are inserted **after every 4th self-attention layer**, totaling 8 cross-attn layers in the 8B variant. The image encoder is a ViT-H/14 with 630M base + 220M of additional gated-self-attn layers (total 850M); the encoder emits a 7680-dim representation per patch (concatenated from layers 4/8/16/24/31 plus final), which the cross-attn adapter projects into the LM hidden size at attention-key/value time. Architectural significance: the cross-attn shape is a clean instance of a "perceiver-IO style" adapter that the layer-IR needs to express — `decoder_layer = self_attn → cross_attn(image_kv) → ffn`. This is a structurally-different `LMDecoderLayer` shape than the standard Qwen-VL concat-prefix pattern (where image tokens are just regular self-attn-able tokens). For llm-layers: deserves a `models/llama-3.2-vision/` slot specifically to exercise the cross-attention-adapter axis.

### 3.7 Pixtral 12B / Mistral OCR (Mistral AI, 2024-09 / 2025-12)

Pixtral 12B is out of size scope (12B), but worth noting because (a) its **Pixtral-ViT 400M** vision encoder is the first widely-deployed encoder to use **2D rotary position encoding** in the vision tower (most other ViT encoders use learned-absolute or 1D-RoPE), and (b) it variable-natively handles image sizes via the same 2D-RoPE relative encoding. Mistral OCR 3 (2025-12) is the production document-AI API endpoint; architecture details are not public but it is presumed to be a Pixtral / Mistral-Large lineage trained on doc parsing. The 2D-RoPE-in-vision pattern shows up in Qwen2-VL (M-RoPE), Qwen2.5-VL, MinerU2.5 (NaViT), PaddleOCR-VL, InternVL3 (V2PE) — it has clearly become a 2024–2025 consensus axis for the vision encoder, even though it lives outside the LM-decoder per se.

### 3.8 Qwen2-VL / Qwen2.5-VL / Qwen3-VL (Alibaba, 2024-08 / 2025-01 / 2025-Q4)

This is the de-facto dominant base for OCR-LLM fine-tunes in 2025–2026. The LM portion is verbatim Qwen2-1.5B / 7B (Qwen2-VL), Qwen2.5-3B / 7B (Qwen2.5-VL), Qwen3-8B (Qwen3-VL) — all already cataloged in research/01. The architecturally-novel parts live in the vision pathway: (a) **M-RoPE / Multimodal RoPE** — the position-encoding scheme uses separate temporal, height, width dimensions of RoPE, applied to both image and video tokens; the LM-decoder's RoPE is *also* M-RoPE so that text-position-id, image-h-position, image-w-position can be encoded into the same rotary; (b) **dynamic resolution / native aspect ratio** in the ViT — the encoder accepts variable-resolution images via padded patch packing; (c) Qwen2.5-VL's redesigned ViT is itself "LM-shaped" (uses SwiGLU, RMSNorm, window attention) — a notable architectural symmetry. For llm-layers: M-RoPE is a meaningful extension to the layer-IR's RoPE axis — the rotary now has a *channel-axis-aware* application (different head_dim slices receive different position frequencies). The LM-decoder of Qwen2.5-VL-3B is literally Qwen2.5-3B's decoder with M-RoPE substituted for vanilla RoPE; a `models/qwen2.5-vl-3b/` slot is the cleanest way to test M-RoPE in the IR.

### 3.9 olmOCR (Allen AI, 2025-02 / 2025-07 / 2025-10)

olmOCR is the highest-quality open-weight OCR-LLM in 2025–2026 by benchmark, but it is architecturally **just Qwen2-VL-7B / Qwen2.5-VL-7B** with SFT and GRPO post-training on a curated OCR dataset (olmOCR-mix-1025) and unit-test rewards (olmOCR-2). The LM-portion (Qwen2.5-7B) is exactly at the 7.6B mark — barely in scope. The training-data and RL-reward design is the contribution, not the architecture. For llm-layers: no new architectural axes; the model is already covered by qwen2.5-vl-7b's slot. olmOCR's broad adoption (HF top-trending) makes it the **single most cited "good OCR-LLM" of 2025** but it does not deserve its own `models/` slot.

### 3.10 PaddleOCR-VL (Baidu, 2025-10 / 2026-01)

PaddleOCR-VL is the smallest production OCR-LLM at 0.9B total: a NaViT-style dynamic-res vision encoder (~600M) + Adaptive MLP connector + **ERNIE-4.5-0.3B** LM decoder. ERNIE-4.5-0.3B is an entirely separate research/01-style architecture that is not yet in the census — it has its own tokenizer (ERNIE BPE), GQA shape, and SwiGLU. This is the only OCR-LLM in the corpus that uses a non-Qwen, non-Llama, non-Phi LM backbone at sub-1B scale. The model claims 94.5 on OmniDocBench v1.5 (SOTA). For llm-layers: ERNIE-4.5-0.3B is a candidate for a new research/01 row (the family is not currently cataloged); PaddleOCR-VL itself does not need a slot, but the ERNIE-4.5 0.3B decoder does.

---

## 4. Architectural axes specific to OCR/document LMs

The 32-axis taxonomy from research/02 covers the text-side decoder thoroughly. The OCR/document-AI extension introduces **seven new axes** that the layer-IR needs to parametrize, of which only **three** are strictly LM-decoder axes; the other four are vision-pathway concerns that affect the decoder context but live outside the decoder's per-layer forward.

### 4.1 LM-decoder axes (in scope for llm-layers)

**Axis 33 — Position-encoding dimension count.** Standard LLM RoPE is 1D (text token index). M-RoPE (Qwen2-VL onward), 2D-RoPE (Pixtral vision tower), and V2PE (InternVL3) extend RoPE to 2D or 3D: text tokens use only the temporal sub-channel; image tokens use temporal + height + width sub-channels of the same head_dim. The layer-IR must support a `rope_layout: enum[1d, 2d, 3d(t,h,w)]` field and a `rope_sub_channel_mask` that controls which head_dim slices rotate with which positional axis. Models: Qwen2-VL/2.5-VL/3-VL (M-RoPE), InternVL3 (V2PE), Pixtral (2D-RoPE in vision tower only). 

**Axis 34 — Cross-attention adapter layers.** Standard MLLM concat-prefix injects image tokens as a regular prefix of the decoder context; this requires no new axis. Llama 3.2 Vision uses **interleaved cross-attention adapter layers** that read image-KV from the encoder rather than from the LM's own self-attn cache. The layer-IR's `LMDecoderLayer` must express `cross_attn_after_every_N_layers: int` and a separate `image_kv_source`. Models: Llama 3.2 Vision, Florence-2 (encoder-decoder; cross-attn is its decoder shape). 

**Axis 35 — Visual causal mask schema.** Standard LLM attention is uniformly causal. DeepSeek-OCR-2's **Visual Causal Flow** is bidirectional within a page-block of visual tokens and causal across pages and into the text-decoding region. The layer-IR's `attention_mask_kind` enum currently has `causal | bidirectional | sliding_window`; this requires a new `block_bidirectional` value plus a `block_size` parameter. Models: DeepSeek-OCR-2 (the only confirmed user; LayoutLMv3 uses bidirectional but it is encoder-only).

### 4.2 Vision-pathway axes (out of strict LM-decoder scope but affect the decoder input)

**Axis V1 — Multi-resolution / native-resolution patches.** NaViT, dynamic tiling, AnyRes — the encoder receives variable-shape input and emits a variable number of vision tokens. This affects the decoder only via context length and through M-RoPE's coordinate encoding.

**Axis V2 — Multi-encoder fusion (MoVE).** Eagle 2 and Cambrian-1 use multiple vision encoders (SigLIP + ConvNeXt + DINOv2 + CLIP) and fuse them via SVA (Spatial Vision Aggregator) or tiled channel concatenation. The decoder sees the result via a standard MLP; no decoder-side change.

**Axis V3 — Optical context compression rate.** GOT-OCR (1024² → 256 tokens), DeepSeek-OCR (10× compression for 97% accuracy), TextHawk2 (16× compression vs v1). This affects the prefix length entering the decoder but does not modify the decoder forward.

**Axis V4 — Frequency-domain vs spatial vision tokens.** DocPedia is the sole outlier — it uses JPEG DCT coefficients as vision tokens instead of patch embeddings. The decoder sees them as ordinary tokens, but the encoder pathway is entirely different.

### 4.3 Mixture-of-LoRAs (Axis 36, edge-case)

Phi-4-multimodal's **mixture-of-LoRAs** routes vision and audio inputs through separate LoRA stacks layered onto the frozen LM. This is structurally **a per-layer LoRA-adapter routing decision based on input modality**, not a standard PEFT post-training adapter. The layer-IR needs an axis for "is this layer routed through modality-specific LoRAs at inference time", separate from the static LoRA-merge-back-into-weights approach. Models: Phi-4-multimodal (the only confirmed user; the broader idea is being adopted in 2026 by various adapter-routing approaches).

### 4.4 Summary: which axes need adding to research/02's 32-axis taxonomy

Three strictly-decoder axes: **33 rope_layout**, **34 cross_attn_adapter**, **35 block_bidirectional_mask**. One decoder-adjacent axis: **36 modality_lora_routing** (Phi-4-multimodal). The four vision-pathway axes (V1–V4) are documented but not strictly required for the LM-decoder layer-IR.

---

## 5. Recommendations for llm-layers

### 5.1 Models that deserve a `models/<name>/` slot

In priority order:

1. **`models/qwen2.5-vl-3b/`** — the canonical M-RoPE / dynamic-resolution VLM at 3B LM scale. Highest reuse in the ecosystem (Dolphin-v2, Nanonets-OCR-s, Nanonets-OCR2-3B, MonkeyOCR-v1.5 are all fine-tunes of this base). Exercises Axis 33 (rope_layout). LM-decoder is Qwen2.5-3B with M-RoPE substituted.
2. **`models/got-ocr2/`** — Qwen-0.5B + 1024-d linear connector. The architecturally-minimal OCR-LLM. Tests "what is the smallest possible Connector + smallest LM" baseline for the layer-IR. Exercises Axis V3 (compression rate).
3. **`models/deepseek-ocr/`** — DeepSeek3B-MoE-A570M with MLA. Tests MoE + MLA + vision-prefix at sub-1B-active scale. Exercises Axis 35 once DeepSeek-OCR-2 is widely deployed (Visual Causal Flow). Also tests how MLA from research/01's V2-Lite row generalizes to OCR.
4. **`models/llama-3.2-vision/`** — cross-attention adapter every 4 layers. The canonical test case for Axis 34 (cross_attn_adapter). LM-portion is Llama 3.1-8B (borderline; the 11B-total is what is interesting). Even at the size-scope edge, the *adapter-layer pattern* is what the layer-IR needs to express.
5. **`models/phi-4-multimodal/`** — mixture-of-LoRAs. The canonical test for Axis 36 (modality_lora_routing). LM-portion is Phi-4-Mini-3.8B from research/01.

Two more to consider as second-tier:

6. **`models/internvl3-2b/`** — Variable Visual Position Encoding (V2PE) and pixel-unshuffle compression. LM-portion is Qwen2.5 / InternLM3 sub-3B.
7. **`models/florence-2/`** — the only encoder-decoder in scope, single-prompt multi-task interface. ~80M LM-decoder.

### 5.2 Models to flag as "vision-encoder concern, LM is just Qwen/Llama/Phi"

These models have the *vision side* doing the work; the LM-decoder is a verbatim research/01 row. No new `models/<name>/` slot needed.

- olmOCR / olmOCR-2 → Qwen2-VL-7B / Qwen2.5-VL-7B (cover via `models/qwen2.5-vl-7b/` if added)
- Nanonets-OCR-s / Nanonets-OCR2-3B → Qwen2.5-VL-3B
- RolmOCR → Qwen2.5-VL-7B
- MonkeyOCR / MonkeyOCR-v1.5 → Qwen2.5-VL-3B (after v1.0 InternLM2)
- Dolphin / Dolphin-v2 → Qwen2.5-VL-3B
- MinerU2.5 → Qwen2-0.5B (cover via `models/qwen2/`)
- HunyuanOCR → Hunyuan-0.5B (research/01 should add a Hunyuan-family row first)
- PaddleOCR-VL → ERNIE-4.5-0.3B (research/01 should add an ERNIE-4.5 row first)
- LLaVA-OneVision-7B → Qwen2-7B
- BLIP-3 / xGen-MM → Phi-3-Mini-3.8B
- Pixtral 12B → Mistral Nemo 12B (size scope: out)
- Cambrian-1 → Llama-3-8B (size scope: borderline)
- NVILA → Llama-3-8B (size scope: borderline)
- Molmo-7B → OLMo / Qwen2-7B
- MolmoE-1B → OLMoE-1B-7B-0924 (already in research/01)

### 5.3 Models to leave out of llm-layers entirely

- LayoutLMv3, DiT, StrucTexT (v2/v3), LiLT, LayoutMask, Donut, Nougat, Kosmos-2.5, Pix2Struct — encoder-only or pre-LLM-decoder shape; included in the catalog as historical references but not architecturally relevant to the <8B-LM-decoder layer-IR.
- Mistral OCR 3 — API only, no public weights; not analyzable.
- DocPedia — encoder uses DCT coefficients; LM is Vicuna-7B (already covered); not deployed; cite as the lone Axis-V4 example.

---

## 6. Bibliography

- GOT-OCR2 paper: [arXiv:2409.01704](https://arxiv.org/abs/2409.01704) · [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) · [Ucas-HaoranWei/GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- DeepSeek-OCR paper: [arXiv:2510.18234](https://arxiv.org/abs/2510.18234) · [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) · [deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- olmOCR paper: [arXiv:2502.18443](https://arxiv.org/abs/2502.18443) · olmOCR-2: [arXiv:2510.19817](https://arxiv.org/abs/2510.19817) · [allenai/olmOCR-2-7B-1025](https://huggingface.co/allenai/olmOCR-2-7B-1025)
- MinerU2.5 paper: [arXiv:2509.22186](https://arxiv.org/abs/2509.22186) · [opendatalab/MinerU](https://github.com/opendatalab/MinerU)
- MonkeyOCR: [arXiv:2506.05218](https://arxiv.org/abs/2506.05218) · MonkeyOCR-v1.5: [arXiv:2511.10390](https://arxiv.org/abs/2511.10390) · [Yuliang-Liu/MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR)
- HunyuanOCR: [arXiv:2511.19575](https://arxiv.org/abs/2511.19575) · [Tencent-Hunyuan/HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR)
- Qwen2-VL: [arXiv:2409.12191](https://arxiv.org/abs/2409.12191) · Qwen2.5-VL: [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) · [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- DeepSeek-VL2: [arXiv:2412.10302](https://arxiv.org/abs/2412.10302) · [deepseek-ai/deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2)
- Llama 3.2 Vision: [Llama 3 paper arXiv:2407.21783](https://arxiv.org/abs/2407.21783) · [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- Phi-4-multimodal: [arXiv:2503.01743](https://arxiv.org/abs/2503.01743) · [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- Phi-3.5-Vision: [Phi-3 tech report arXiv:2404.14219](https://arxiv.org/abs/2404.14219) · [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- Florence-2: [arXiv:2311.06242](https://arxiv.org/abs/2311.06242) · [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)
- Pixtral 12B: [arXiv:2410.07073](https://arxiv.org/abs/2410.07073) · [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409)
- LLaVA-OneVision: [arXiv:2408.03326](https://arxiv.org/abs/2408.03326)
- Molmo / MolmoE: [arXiv:2409.17146](https://arxiv.org/abs/2409.17146) · [allenai/MolmoE-1B-0924](https://huggingface.co/allenai/MolmoE-1B-0924)
- Cambrian-1: [arXiv:2406.16860](https://arxiv.org/abs/2406.16860)
- NVILA: [arXiv:2412.04468](https://arxiv.org/abs/2412.04468) · Eagle 2: [arXiv:2501.14818](https://arxiv.org/abs/2501.14818)
- InternVL3: [arXiv:2504.10479](https://arxiv.org/abs/2504.10479) · InternVL3.5: [arXiv:2508.18265](https://arxiv.org/abs/2508.18265)
- MiniCPM-V: [arXiv:2408.01800](https://arxiv.org/abs/2408.01800)
- mPLUG-DocOwl 1.5: [arXiv:2403.12895](https://arxiv.org/abs/2403.12895) · DocOwl2: [arXiv:2409.03420](https://arxiv.org/abs/2409.03420)
- TextHawk: [arXiv:2404.09204](https://arxiv.org/abs/2404.09204) · TextHawk2: [arXiv:2410.05261](https://arxiv.org/abs/2410.05261)
- DocPedia: [arXiv:2311.11810](https://arxiv.org/abs/2311.11810)
- Vary / Vary-toy: [arXiv:2312.06109](https://arxiv.org/abs/2312.06109) / [arXiv:2401.12503](https://arxiv.org/abs/2401.12503)
- Donut: [arXiv:2111.15664](https://arxiv.org/abs/2111.15664) · Nougat: [arXiv:2308.13418](https://arxiv.org/abs/2308.13418) · Kosmos-2.5: [arXiv:2309.11419](https://arxiv.org/abs/2309.11419) · Pix2Struct: [arXiv:2210.03347](https://arxiv.org/abs/2210.03347)
- BLIP-3 / xGen-MM: [arXiv:2408.08872](https://arxiv.org/abs/2408.08872)
- Ovis2.5: [arXiv:2508.11737](https://arxiv.org/abs/2508.11737)
- Dolphin (ByteDance): [arXiv:2505.14059](https://arxiv.org/abs/2505.14059) · [bytedance/Dolphin](https://github.com/bytedance/Dolphin)
- Surya / Marker: [datalab-to/surya](https://github.com/datalab-to/surya) · [datalab-to/marker](https://github.com/datalab-to/marker)
- PaddleOCR-VL: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- LayoutLMv3: [arXiv:2204.08387](https://arxiv.org/abs/2204.08387)
- Nanonets-OCR: [nanonets/Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B)
- Mistral OCR 3 announcement (Dec 2025): [Mistral AI blog](https://mistral.ai/news/)
