# Recent Mainstream LM Releases — March 6, 2026 → June 6, 2026

*Quarterly census for the llm-layers project. Compiled 2026-06-06.*

---

## 1. Methodology, Scope, and Caveats

**Window.** This document covers releases that became publicly visible (HF model card or GitHub repo with downloadable artifacts, or formal vendor blog announcement) between **2026-03-06 and 2026-06-06**, inclusive. Where a family started earlier (e.g., Qwen3.5 in Feb 2026) I include only the sub-variants whose checkpoints went live inside the window; I cross-reference older anchors when needed for architectural context.

**Search strategy.**
1. WebSearch queries against each first-tier org name, joined with "release 2026", "April 2026", "May 2026", "huggingface". For Chinese labs I also searched the romanized + product names (e.g., "Qwen3.6", "Hy3", "MiMo-V2.5").
2. Roll-up sources: `llm-stats.com/ai-news`, `releasebot.io`, `huggingface.co/papers/trending`, `marktechpost.com`, Simon Willison's blog (he tags model launches reliably), VentureBeat coverage.
3. Cross-checked release dates against the relevant blog/post when an HF model card existed.
4. For multimodal/SSM/audio I went category-by-category rather than org-by-org.

**Caveats.**
- "Release" here means *weights or API generally available*, not "previewed at a keynote." A few flagship items in this window (Meta Llama 5, Microsoft MAI-Thinking-1) were announced but had no downloadable weights as of the cutoff; they appear in §6.
- A small number of secondary-source dates conflict by 1-2 days (e.g., Mistral Medium 3.5 listed as 2026-04-29 in one blog and 2026-04-30 in another). I take the earliest source-supported date and note both where it matters.
- Some Q1 2026 anchors (Qwen3.5 base, GLM-5, Falcon-H1R, Voxtral Transcribe 2, MiniMax-M2.5, DeepSeek-OCR-2, Liquid LFM2.5-base) are referenced for lineage; the *master table* lists only items whose primary release fell in the 03-06 to 06-06 window.
- "Architectural novelty" is judged conservatively: a new attention scheme, new SSM/hybrid stack, or new training-recipe class. Same-architecture scale-ups are flagged "scale-up / not novel."
- For models <8B I attempt to capture the LM-decoder parameter count (text backbone only, excluding ViT / audio encoder where stated). When the model is wholly multimodal-native with a fused decoder I list the total.

---

## 2. Master Table — Every Release 2026-03-06 → 2026-06-06

Legend: **Novel?** = Yes / No / Partial (refinement of an existing novel block). **LM<8B** = parameter count of the text decoder if it is strictly <8B (useful for the llm-layers in-scope filter). "n/a" = decoder is large / not applicable.

| # | Family / Variant | Date | Org | Sizes / Variants | LM<8B? | HF URL | GitHub | Paper / Blog | Architecture one-liner | Novel? |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Mamba-3** (research release + reference impl) | 2026-03-17 | CMU / Princeton / Together AI / Cartesia | 180M–1.5B reference scales | yes (all variants) | — (paper-only checkpoints at this date) | github.com/state-spaces/mamba | openreview.net/forum?id=HwCvaJOiCj ; ICLR 2026 paper [arxiv 2603.15569] | Pure SSM with complex-valued state, refined SSM discretization, and **MIMO decoding** | **Yes** |
| 2 | **Mistral Small 4 (Mistral-Small-4-119B-2603)** | 2026-03-16 | Mistral AI | 119B (15B active MoE) | no | huggingface.co/mistralai/Mistral-Small-4-119B-2603 | github.com/mistralai (org) | mistral.ai/news/mistral-small-4 | First Mistral model unifying instruct + Magistral reasoning + Devstral coding + Pixtral vision in a single MoE; eagle-head speculative decoding head shipped in weights | Partial (unified-task MoE) |
| 3 | **NVIDIA Nemotron 3 Super** | 2026-03-11 | NVIDIA | 120B total / 12B active (hybrid Mamba-Transformer MoE) | no | huggingface.co/nvidia (Nemotron-3 collection) | github.com/NVIDIA/Megatron-LM | developer.nvidia.com/blog/introducing-nemotron-3-super | Hybrid Mamba-2 + Transformer MoE for agentic reasoning, open-weights, FP8 native | **Yes** |
| 4 | **Mistral Voxtral TTS (4B)** | 2026-03-26 | Mistral AI | 4B TTS | yes (4B) | huggingface.co/mistralai (voxtral-tts collection) | github.com/mistralai/voxtral | mistral.ai/news/voxtral | 4B open-weights TTS with 5-sec zero-shot voice cloning, 90 ms latency | Partial |
| 5 | **Xiaomi MiMo-V2 (Pro / Omni / TTS)** | 2026-03-18 | Xiaomi | Pro: 1T total / 42B active MoE; Omni: omni-modal; TTS specialized | no | huggingface.co/XiaomiMiMo (collection appears later for V2.5 as MIT) | github.com/XiaomiMiMo | mimo.xiaomi.com/mimo-v2-pro | Three coordinated specialized models; Pro is huge MoE with 1M context; Omni unified text+vision+audio+video; TTS adds dialect/singing | Partial |
| 6 | **Reka Edge 2026** | 2026-03 | Reka AI | 8B-class | yes | huggingface.co/RekaAI | github.com/reka-ai | reka.ai (blog) | Re-trained dense edge model optimized for embodied/physical AI; 3× fewer tokens, 65 % higher throughput vs same-class 8B | No (recipe, not architecture) |
| 7 | **MiniMax-M2.7** | 2026-03-18 | MiniMax | ~230B total / 10B active MoE | no | huggingface.co/MiniMaxAI | github.com/MiniMax-AI | platform.minimax.io/docs/release-notes/models | Iteration of M2 coding/agentic MoE; minor architectural delta vs M2.5 | No |
| 8 | **Reka Flash 3.1** (refresh) | 2026-03 | Reka AI | 21B dense reasoning | no | huggingface.co/RekaAI/reka-flash-3.1 | github.com/reka-ai | reka.ai/news | RLOO-trained reasoning refresh, Apache 2.0 | No |
| 9 | **Gemma 4 (E2B, E4B, 12B, 26B-A4B, 31B)** | 2026-04-02 | Google DeepMind | E2B (~2B effective), E4B (~4B effective), 12B dense, 26B (A4B MoE), 31B dense | yes (E2B, E4B) | huggingface.co/collections/google/gemma-4 | github.com/google-deepmind/gemma | huggingface.co/blog/gemma4 ; blog.google/.../gemma-4 | Multimodal text/image/audio in, text out; mix of dense + MoE in same family; 256K context, 140+ langs; introduces "Elastic" sizes (E2B/E4B) shared activations | **Yes** (Elastic + audio-in for small variant) |
| 10 | **Gemma 4 MTP Drafters** | 2026-04-16 | Google DeepMind | small drafter heads paired w/ 12B and 26B | yes | huggingface.co/google (gemma-4 collection) | — | ai.google.dev/gemma/docs/releases | Multi-Token Prediction draft heads for speculative decoding | Partial |
| 11 | **GLM-5.1** | 2026-04-07 | Z.AI / Zhipu | 744B total / 40B active MoE (DSA attention) | no | huggingface.co/zai-org/GLM-5.1 | github.com/zai-org/GLM-V (for VLM siblings) | huggingface.co/blog/mlabonne/glm-5 (lineage) | DeepSeek-Sparse-Attention based MoE with 200K context, post-trained for 8-hour autonomous agentic loops | Partial (DSA derivative) |
| 12 | **GLM-5V-Turbo** | 2026-04-02 | Z.AI | VLM derived from GLM-5 | no | huggingface.co/zai-org (GLM-V) | github.com/zai-org/GLM-V | docs.z.ai/release-notes/new-released | Multimodal sibling to GLM-5/5.1 | No |
| 13 | **Tencent Hunyuan-A13B-Instruct** | 2026-04-07 | Tencent | 80B total / 13B active MoE | no | huggingface.co/tencent/Hunyuan-A13B-Instruct | github.com/Tencent-Hunyuan/Hunyuan-A13B | tencent (release) | Fine-grained MoE, native 256K context, hybrid fast/slow reasoning modes | Partial |
| 14 | **Qwen3.6-35B-A3B** | 2026-04-16 | Alibaba Qwen | 35B total / ~3B active MoE | no (active 3B but total 35B; LM decoder counts total) | huggingface.co/Qwen/Qwen3.6-35B-A3B | github.com/QwenLM/Qwen3.6 | qwen.ai/blog?id=qwen3.6-35b-a3b | Refreshed multimodal MoE, 262K native (YaRN to ~1M) | No (scale) |
| 15 | **Qwen3.6-27B** | 2026-04-22 | Alibaba Qwen | 27B dense | no | huggingface.co/Qwen/Qwen3.6-27B | github.com/QwenLM/Qwen3.6 | qwen.ai/blog?id=qwen3.6-27b | Dense 27B that beats 397B-A17B on coding bench; 1M context | No |
| 16 | **Qwen3.5-Omni (Plus / Flash / Light)** | 2026-04-22 (paper) | Alibaba Qwen | 403B, 125B, 36B, 28B | no | huggingface.co/collections/Qwen/qwen35 | github.com/QwenLM/Qwen3-Omni (predecessor) | arxiv.org/abs/2604.15804 | Hybrid Attention MoE Thinker + Talker, 256K ctx, 10 langs, native audio/video | Partial |
| 17 | **DeepSeek-V4-Pro & V4-Flash (Preview)** | 2026-04-24 | DeepSeek | V4-Pro: 1.6T total / 49B active; V4-Flash: 284B total / 13B active | no | huggingface.co/deepseek-ai/DeepSeek-V4-Pro , …-Flash | github.com/deepseek-ai | api-docs.deepseek.com/news/news260424 ; tech report PDF in HF repo | New hybrid attention combining Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA); 1M ctx; cuts FLOPs to 27 % and KV cache to 10 % of V3.2 at 1M | **Yes** |
| 17b | **DeepSeek-V4-Pro-NVFP4 (NVIDIA quantized)** | 2026-04-24 | NVIDIA × DeepSeek | V4-Pro quantized | no | huggingface.co/nvidia/DeepSeek-V4-Pro-NVFP4 | — | build.nvidia.com/.../deepseek-v4-pro | NVFP4 quantization release | No |
| 18 | **Tencent Hy3 (Hunyuan 3) Preview** | 2026-04-23 | Tencent | 295B total / 21B active MoE + 3.8B MTP layer | no | huggingface.co/tencent/Hy3-preview | github.com/Tencent-Hunyuan | huggingface.co/blog/imnotkitty/hy3-preview ; tencent.com/.../2202320 | First model from Tencent's fully rebuilt pre-train + RL infrastructure; fast-slow hybrid; 256K ctx; MTP head | Partial |
| 19 | **Hunyuan Video 1.5** | 2026-05-04 | Tencent | 8.3B text-to-video | yes (8.3B but video, not LM) | tencent (HF) | github.com/Tencent-Hunyuan | tencent video blog | T2V diffusion model, single-RTX-4090 capable | No (video, not LM) |
| 20 | **Hy-MT2 (1.8B / 7B / 30B-A3B)** | 2026-05-21 | Tencent | 1.8B dense, 7B dense, 30B-A3B MoE | yes (1.8B, 7B) | huggingface.co/tencent (Hy-MT2 collection) | github.com/Tencent-Hunyuan/Hy-MT2 | tencent / arXiv MT report | "Fast-thinking" multilingual translation across 33 languages | Partial |
| 21 | **Xiaomi MiMo-V2.5 / V2.5-Pro / V2.5-Base / V2.5-ASR** | 2026-04-22 | Xiaomi | V2.5: 310B-A15B MoE (text); V2.5-Pro: 1.02T-A42B MoE; V2.5-ASR speech | no | huggingface.co/XiaomiMiMo/MiMo-V2.5 , …-Pro , …-ASR | github.com/XiaomiMiMo | mimo.xiaomi.com/mimo-v2-5 ; platform.xiaomimimo.com/docs/en-US/news/v2.5-open-sourced | Massive sparse MoE with FP8 train; 729M-param vision tower + audio encoder; MIT licensed; native 1M ctx | Partial |
| 22 | **IBM Granite 4.1 (3B / 8B / 30B dense)** | 2026-04-29 | IBM | 3B, 8B, 30B dense | yes (3B), borderline (8B) | huggingface.co/ibm-granite/granite-4.1-3b , -4.1-8b , -4.1-30b | github.com/ibm-granite | research.ibm.com/blog/granite-4-1-ai-foundation-models | Dense decoder-only that matches prior 32B-MoE; tool-calling/instruction follow focus | No |
| 23 | **Granite Vision 4.1-4B** | 2026-04-29 | IBM | 4B VLM | yes (4B) | huggingface.co/ibm-granite/granite-vision-4.1-4b | github.com/ibm-granite/granite-vision-models | research.ibm.com/blog/granite-4-1-ai-foundation-models | Enterprise doc understanding VLM | No |
| 24 | **Granite Speech 4.1 2B (×2 variants)** | 2026-04-30 | IBM | 2B autoregressive ASR-translate + 2B non-autoregressive edit | yes | huggingface.co/ibm-granite (granite-speech-4.1) | github.com/ibm-granite | marktechpost.com/2026/04/30/... | Two paired ASR models, one AR, one NAR | Partial (NAR speech is unusual) |
| 25 | **Granite Guardian 4.1** | 2026-04-29 | IBM | small safety classifier | yes | huggingface.co/ibm-granite (guardian-4.1) | github.com/ibm-granite | research.ibm.com/blog/granite-4-1-ai-foundation-models | Harm-detection model | No |
| 26 | **Mistral Medium 3.5 (128B dense)** | 2026-04-29 | Mistral AI | 128B dense | no | huggingface.co/mistralai (Medium-3.5) | github.com/mistralai | mistral.ai (Medium 3.5 announce) | Dense 128B that absorbs Magistral (reasoning) + Devstral 2 (code) into one set of weights with configurable reasoning effort; 256K ctx; modified-MIT | Partial |
| 27 | **Meta Muse Spark (Muse-family; replaces Llama)** | 2026-04-08 | Meta Superintelligence Labs | Flagship + (unspecified) smaller variants; mostly via Meta AI app | unknown — weights gated | (none — see §6) | (none) | about.fb.com/news/2026/04/introducing-muse-spark-meta-superintelligence-labs ; ai.meta.com/blog/introducing-muse-spark-msl | Built natively multimodal w/ "Contemplating mode" parallel-agent reasoning. Not open weights at announce. | **Yes** (claimed) |
| 28 | **Meta Llama 5 (flagship 600B+)** | 2026-04-08 | Meta | 600B+ flagship; family sizes claimed but not all weights up | n/a | huggingface.co/meta-llama (Llama 5 collection — *partially live, gated*) | github.com/meta-llama | ai.meta.com/blog (LlamaCon) ; chroniclejournal/financialcontent coverage; Wikipedia | "Recursive Self-Improvement" capability, up to 5M-token context | Partial — note that Wikipedia indicates Llama line is being superseded by Muse Spark; treat Llama 5 as a transitional flagship |
| 29 | **EXAONE 4.5 (33B)** | 2026-04-09 | LG AI Research | 33B dense VLM | no | huggingface.co/LGAI-EXAONE (EXAONE-4.5 collection) | github.com/LG-AI-EXAONE/EXAONE-4.5 | prnewswire.com/.../lg-reveals-next-gen-multimodal-ai-exaone-4-5 | Proprietary vision encoder fused into LLM; STEM-tuned; Korean+Spanish/German/Japanese/Vietnamese | No |
| 30 | **Moonshot Kimi K2.6** | 2026-04-21 (GA from preview 2026-04-13) | Moonshot AI | 1T total / 32B active MoE + 400M MoonViT encoder | no | huggingface.co/moonshotai (K2.6) | github.com (Kimi org) | kimi-k2.org/blog/24-kimi-k2-6-release | Native multimodal agentic MoE with 256K ctx and 300-sub-agent swarm; Modified MIT | Partial (swarm architecture + scale) |
| 31 | **HY-Embodied-0.5 (MoT-2B)** | 2026-04-09 | Tencent | 2B (MoT — Mixture-of-Transformers / decoder backbone) | yes (2B) | huggingface.co/tencent (HY-Embodied) | github.com/Tencent-Hunyuan/HY-Embodied | tencent (HY-Embodied page) | Embodied-foundation model for spatial-temporal reasoning; 2B decoder | Partial |
| 32 | **Liquid LFM2.5-VL-450M** | 2026-04-11 | Liquid AI | 450M VLM | yes | huggingface.co/LiquidAI/LFM2-VL-450M (and LFM2.5 sibling) | github.com/LiquidAI | liquid.ai/blog (LFM2.5 series) | Compact VLM with bounding-box prediction, sub-250 ms edge inference; liquid-time-constant backbone | **Yes** (LFM family is a non-Transformer recurrent block) |
| 33 | **TII Falcon Perception (0.6B)** | 2026-04-03 (HF listing) — formally announced 2026-05-03 | TII Abu Dhabi | 0.6B early-fusion multimodal | yes (0.6B) | huggingface.co/tiiuae (Falcon-Perception) | github.com/falcon-llm | tii.ae/news/tii-launches-falcon-perception | Early-fusion transformer for open-vocabulary grounding + segmentation from natural language | Partial |
| 34 | **TII Falcon-Edge (1B / 3B base + instruct, 1.58-bit)** | 2026-05 | TII Abu Dhabi | 1B, 3B (BitNet-style 1.58-bit) | yes (1B, 3B) | huggingface.co/tiiuae (Falcon-Edge collection) | github.com/falcon-llm | huggingface.co/blog/tiiuae/falcon-edge | First open BitNet-style release that ships pre-quantized weights enabling further fine-tune / continue-pretrain | **Yes** |
| 35 | **xAI Grok 4.3** (API-only frontier) | 2026-05-04 | xAI | Multi-trillion-class; no weights | n/a | — (none) | — | x.ai/news | Cost-efficient frontier reasoning, 1M ctx, native video | Closed (announcement) |
| 36 | **xAI Grok Build 0.1** (coding-agent model, EA) | 2026-05-14 | xAI | unspecified | n/a | — | — | x.ai/news | Coding-agent specialization, 256K ctx, text+image | Closed (announcement) |
| 37 | **Zyphra ZAYA1-8B** | 2026-05-06 | Zyphra | 8.4B (MoE reasoning) | yes (on the line) | huggingface.co/Zyphra (ZAYA1-8B) | github.com/Zyphra | letsdatascience.com/.../zaya1-8b... | MoE reasoning model trained entirely on AMD MI300X; 89.6 HMMT '25 | Partial (training-stack novelty more than arch) |
| 38 | **Baidu ERNIE 5.1** (API-only) | 2026-05-08 | Baidu | MoE; ~1/3 params of ERNIE 5.0, half active | n/a | — (hosted) | — | ernie.baidu.com/blog/posts/ernie-5.1-0508-release ; cntechpost coverage | MoE-distilled successor of ERNIE 5.0; trained at 6 % cost; native full-modality | Partial |
| 39 | **Cohere Command A+** | 2026-05-20 | Cohere | 218B total / 25B active MoE (first Cohere MoE) | no | huggingface.co/CohereLabs (command-a-plus-05-2026) | github.com/cohere-ai | docs.cohere.com/changelog/command-a-plus-05-2026 ; cohere.com/blog/command-a-plus | First fully Apache-2.0 frontier model from Cohere; native citations; W4A4 lossless quant fits 2× H100 | Partial |
| 40 | **Qwen3.7-Max (preview)** | 2026-05-20 | Alibaba Qwen | hosted; multi-trillion class | n/a | (chat.qwen.ai / lmarena.ai) | github.com/QwenLM | qwen.ai/blog | 1M ctx reasoning-agent flagship; weights not yet public | Closed |
| 41 | **Liquid LFM2.5-8B-A1B** | 2026-05-28 | Liquid AI | 8.3B total / 1.5B active MoE on LFM backbone | yes (1.5B active LM only; 8.3B total) | huggingface.co/LiquidAI/LFM2.5-8B-A1B | github.com/LiquidAI | marktechpost.com/.../lfm2-5-8b-a1b | Liquid-architecture MoE for on-device tool-calling | **Yes** (LFM MoE) |
| 42 | **StepFun Step 3.7 Flash** | 2026-05-28 (OpenRouter list); 2026-05-29 (HF commit / blog) | StepFun | 198B total / ~11B active MoE; 196B LM + 1.8B ViT | no | huggingface.co/stepfun-ai/Step-3.7-Flash (+ FP8 / NVFP4 / GGUF) | github.com/stepfun-ai | pandaily.com/.../stepfun-open-source-step-3-7-flash | First multimodal Flash; dedicated ViT injected into language ctx; Apache-2.0 | Partial |
| 43 | **MiniMax-M3.0** | 2026-06-01 | MiniMax | MoE w/ new MSA (MiniMax Sparse Attention) | n/a (weights expected ~mid-June) | github.com/MiniMax-AI/MiniMax-M2 (org) — M3 weights staged | github.com/MiniMax-AI | marktechpost.com/.../minimax-releases-minimax-m3 | New MSA attention; 1M ctx; native multimodal + agentic coding; weights staged for HF release ~10 days post-launch | **Yes** (MSA) |
| 44 | **NVIDIA Nemotron 3 Nano Omni** | 2026-04-28 | NVIDIA | Compact omni-modal (vision + speech + language) | yes (Nano-class) | huggingface.co/nvidia (Nemotron-3 collection) | github.com/NVIDIA | blogs.nvidia.com/blog/nemotron-3-nano-omni-multimodal-ai-agents | Hybrid Mamba-Transformer Omni model, optimized for agents | Partial |
| 45 | **NVIDIA Nemotron 3 Ultra** | 2026-06-04 | NVIDIA | 550B total / 55B active MoE (Hybrid Mamba-Attention) | no | huggingface.co/nvidia (Nemotron-3-Ultra) | github.com/NVIDIA | marktechpost.com/.../nvidia-ai-releases-nemotron-3-ultra | Open 550B Mamba-Transformer MoE; up to 6× throughput vs GLM-5.1 at 8K/64K | **Yes** (hybrid SSM-Attn MoE at scale) |
| 46 | **MAI-Thinking-1 (Microsoft AI)** | 2026-06-02 | Microsoft AI | ~1T total / 35B active MoE | n/a (private preview, no open weights) | — | — | microsoft.ai/news/introducing-mai-thinking-1 ; Build 2026 keynote | Microsoft's first scratch-built reasoning model (no OpenAI distillation), 256K ctx | Closed (announcement) |
| 47 | **Microsoft Aion 1.0 (Instruct)** | 2026-06-02 (announced, weights staged July) | Microsoft AI | 14B-class on-device reasoner | yes (14B; staged) | (planned: huggingface.co/microsoft) | — | thurrott.com (Build 2026 coverage) | On-device 14B reasoner, weights committed for July 2026 HF release | Partial |
| 48 | **Hermes Agent (LM stack: v0.9, v0.11, v0.14, v0.15.2)** | 2026-04-13 / 2026-04-23 / 2026-05-16 / 2026-06-02 | Nous Research | Software / agent runtime (uses Hermes 4.3 weights from Aug 2025) | — | huggingface.co/NousResearch (Hermes 4.3) | github.com/NousResearch/hermes-agent | hermes-agent.nousresearch.com | Open agent harness updates around existing weights; not a new LM | No |

**Total releases captured (weights-available or formally announced) inside the window: 48 entries (45 distinct flagship items; entries 17b, 28, and 47 are partial/quantized/preview siblings).**

A few items present at the *very* edge of the window (Jan/Feb 2026 anchors) are NOT counted in this table even when I reference them: Qwen3.5 (2026-02-16), GLM-5 (2026-02-11), Falcon-H1R 7B (2026-01-05), Voxtral Transcribe 2 (2026-02-04), MiniMax-M2.5 (2026-02), Tiny Aya (2026-02-17), Liquid LFM2.5 base (2026-01-06), DeepSeek-OCR-2 (2026-01-27), Jamba2 (2026-01-08), Stable LM 2 12B refresh (2026-02), Granite 4.0 3B Vision (2026-04-01 — borderline; I include it as part of Granite 4.1 family below).

---

## 3. Per-Org Timelines (chronological, within window)

### Alibaba Qwen
- 2026-04-16 — Qwen3.6-35B-A3B (MoE 35B/3B-active, 1M ctx)
- 2026-04-22 — Qwen3.6-27B (dense 27B coding flagship, 1M ctx)
- 2026-04-22 — Qwen3.5-Omni Technical Report (sizes 28B / 36B / 125B / 403B)
- 2026-05-20 — Qwen3.7-Max preview (closed)

### DeepSeek
- 2026-04-24 — DeepSeek-V4-Pro (1.6T/49B-A) and V4-Flash (284B/13B-A), both with CSA+HCA hybrid sparse attention and 1M ctx

### Google DeepMind
- 2026-04-02 — Gemma 4 family (E2B, E4B, 12B, 26B-A4B, 31B) — multimodal, 256K ctx
- 2026-04-16 — Gemma 4 MTP drafter heads

### Meta / Meta Superintelligence Labs
- 2026-04-08 — Llama 5 announcement (600B+ flagship, 5M ctx) — partial weights
- 2026-04-08 — Muse Spark (Muse-family) — replaces Llama in product lineup; no weights at announce

### Mistral AI
- 2026-03-16 — Mistral Small 4 (119B/15B-A MoE, unified instruct + reasoning + multimodal + coding)
- 2026-03-26 — Voxtral TTS (4B open-weights TTS)
- 2026-04-29 — Mistral Medium 3.5 (128B dense; absorbs Magistral and Devstral 2)

### Microsoft
- 2026-06-02 (Build) — MAI-Thinking-1 (1T/35B-A MoE) — private preview, weights pending
- 2026-06-02 (Build) — Aion 1.0 on-device 14B reasoner, weights staged for July
- (No confirmed Phi-5 release found in this window; references found are forward-looking guides, not an actual HF/blog launch.)

### IBM
- 2026-04-29 — Granite 4.1 (3B / 8B / 30B dense) + Granite Vision 4.1-4B + Granite Guardian 4.1
- 2026-04-30 — Granite Speech 4.1 (2 × 2B variants, AR and NAR)

### Tencent Hunyuan
- 2026-04-07 — Hunyuan-A13B-Instruct (80B/13B-A, 256K ctx)
- 2026-04-09 — HY-Embodied-0.5 MoT-2B (embodied foundation)
- 2026-04-23 — Hy3 (Hunyuan 3) Preview (295B/21B-A MoE + 3.8B MTP)
- 2026-05-04 — Hunyuan Video 1.5 (8.3B T2V)
- 2026-05-21 — Hy-MT2 family (1.8B / 7B / 30B-A3B translation)

### Z.AI / Zhipu (GLM)
- 2026-04-02 — GLM-5V-Turbo (VLM sibling)
- 2026-04-07 — GLM-5.1 (744B/40B-A MoE with DSA, 8-hr agentic loops)

### Moonshot
- 2026-04-13 — Kimi K2.6 preview (1T/32B-A + 400M MoonViT)
- 2026-04-21 — Kimi K2.6 GA

### Xiaomi MiMo
- 2026-03-18 — MiMo-V2 (Pro / Omni / TTS)
- 2026-04-22 — MiMo-V2.5 (310B-A15B) and MiMo-V2.5-Pro (1.02T/42B-A); MIT licensed
- 2026-04-22 — MiMo-V2.5-ASR

### MiniMax
- 2026-03-18 — MiniMax-M2.7 (230B/10B-A)
- 2026-06-01 — MiniMax-M3.0 (MSA attention; weights staged)

### NVIDIA
- 2026-03-11 — Nemotron 3 Super (120B/12B-A hybrid Mamba-Transformer MoE)
- 2026-04-24 — DeepSeek-V4-Pro-NVFP4 (quantized release)
- 2026-04-28 — Nemotron 3 Nano Omni (vision+speech+language)
- 2026-06-04 — Nemotron 3 Ultra (550B/55B-A hybrid Mamba-Attention MoE)

### Cohere
- 2026-05-20 — Command A+ (218B/25B-A MoE, first Cohere MoE, Apache-2.0)

### Liquid AI
- 2026-04-11 — LFM2.5-VL-450M
- 2026-05-28 — LFM2.5-8B-A1B (on-device MoE on LFM backbone)

### TII (Abu Dhabi)
- 2026-04-03 / 2026-05-03 — Falcon Perception 0.6B (multimodal grounding)
- 2026-05 — Falcon-Edge 1B / 3B (1.58-bit BitNet-style, fine-tunable)

### LG AI Research
- 2026-04-09 — EXAONE 4.5 (33B dense VLM)

### xAI
- 2026-05-04 — Grok 4.3 (API)
- 2026-05-14 — Grok Build 0.1 (EA coding model)
- (No confirmed Grok 3 open-weights release inside window; community discussions indicate still pending.)

### Baidu
- 2026-05-08 — ERNIE 5.1 (API-only)

### Zyphra
- 2026-05-06 — ZAYA1-8B (8.4B MoE reasoning, AMD MI300X trained)

### StepFun
- 2026-05-28/29 — Step 3.7 Flash (198B/11B-A MoE w/ 1.8B ViT)

### Carnegie Mellon × Princeton × Together × Cartesia (academic / hybrid)
- 2026-03-17 — Mamba-3 (research release accepted at ICLR 2026)

### Nous Research
- Multiple Hermes-Agent harness releases (no new LM weights)

### Reka AI
- 2026-03 — Reka Edge 2026 rebuild + Flash 3.1 refresh

---

## 4. Architecturally Novel Entries (Spotlight)

This subset has materially new architectural blocks — not merely scale-ups or recipe tweaks.

### 4.1 Mamba-3 (2026-03-17)
- Three substantive deltas vs Mamba-2: (1) more expressive recurrence derived from SSM discretization; (2) complex-valued state-update rule for richer state tracking; (3) **MIMO** (multi-input, multi-output) decoding for accuracy at fixed decode latency.
- Why it matters for llm-layers: this is the most directly relevant SSM update of the year for our `models/mamba/` slot; the complex-valued state and MIMO are net-new ops.

### 4.2 DeepSeek-V4-Pro / V4-Flash (2026-04-24)
- New hybrid attention: **Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA)**.
- At 1M ctx, V4-Pro needs only 27 % of single-token inference FLOPs and 10 % of KV cache of V3.2 — a much bigger leap than V3.2's DSA alone.
- HF: huggingface.co/deepseek-ai/DeepSeek-V4-Pro

### 4.3 Gemma 4 Elastic (E2B / E4B) (2026-04-02)
- "Elastic" inference: the small E-variants share activations and weights with the larger ones via a parameter-routing scheme — effectively a structured-pruning training recipe baked into the weight format. Plus audio-in on the small models.
- HF: huggingface.co/google/gemma-4-E2B , huggingface.co/google/gemma-4-E4B

### 4.4 NVIDIA Nemotron 3 Super and Nemotron 3 Ultra (2026-03-11 and 2026-06-04)
- Hybrid Mamba-2 + Transformer MoE. Ultra at 550B/55B-A is the largest open hybrid SSM-attention MoE to date and reportedly ~6× throughput vs GLM-5.1 at 8K/64K context.
- Together with Mamba-3 this is the strongest signal of SSM-hybrid mainstreaming in 2026Q2.

### 4.5 TII Falcon-Edge (2026-05)
- First open BitNet-style 1.58-bit family that ships **pre-quantized** weights enabling further fine-tune / continued pre-training, not just inference. Important for low-bit research.

### 4.6 Liquid LFM2.5-8B-A1B and LFM2.5-VL-450M (2026-05-28 / 2026-04-11)
- Liquid uses a non-Transformer recurrent backbone (LFM / liquid-time-constant cells). The May release is the first MoE variant on this backbone with 8.3B total / 1.5B active.

### 4.7 MiniMax-M3.0 (2026-06-01)
- Introduces MiniMax Sparse Attention (MSA) — a new sparse-attention variant distinct from DeepSeek DSA. Weights staged.

### 4.8 IBM Granite Speech 4.1 NAR variant (2026-04-30)
- The non-autoregressive editing speech model is a relatively rare design — included as architectural novelty within the speech sub-domain.

### 4.9 Meta Muse Spark (2026-04-08)
- Meta says it is built natively multimodal with a "Contemplating mode" that orchestrates multiple agents reasoning in parallel. Architecturally interesting but **not open-weight** at announce, so we can't verify the claims independently.

### 4.10 Microsoft MAI-Thinking-1 (2026-06-02)
- Sparse MoE with ~35B active / ~1T total, claimed scratch-built without OpenAI distillation. Private preview.

---

## 5. In-Scope for llm-layers M2+ (LM decoder < 8B AND distinct enough)

The llm-layers project is looking for `<8B` LM-decoder, architecturally distinct candidates worth a dedicated `models/<name>/` slot. From the master table:

| Candidate | LM decoder | Why it deserves a slot |
|---|---|---|
| **Gemma 4 E2B** | ~2B (Elastic) | Net-new "Elastic" inference scheme, audio-in capability, MTP drafters |
| **Gemma 4 E4B** | ~4B (Elastic) | Same as above, larger |
| **Granite 4.1-3B** | 3B dense | Reference dense small-LM (enterprise tool-use focus); useful baseline |
| **Granite 4.1-8B** | 8B dense (borderline; counts as ≤8B) | Dense decoder-only that matches prior 32B-MoE; commonly cited |
| **Granite Vision 4.1-4B** | 4B (LM + ViT fused) | Document-VLM baseline |
| **Granite Speech 4.1 NAR-2B** | 2B (speech) | NAR speech head is rare |
| **HY-Embodied-0.5 MoT-2B** | 2B | First Mixture-of-Transformers embodied model in window |
| **Liquid LFM2.5-VL-450M** | 0.45B | Non-Transformer backbone (LFM); strong candidate for an "alternative architecture" slot |
| **Liquid LFM2.5-8B-A1B** | 1.5B active / 8.3B total | LFM-backbone MoE; novel |
| **TII Falcon Perception 0.6B** | 0.6B early-fusion | Multimodal grounding small model |
| **TII Falcon-Edge 1B / 3B** | 1B, 3B (1.58-bit) | Only open BitNet-style with retrainable quantized weights |
| **Mamba-3 reference scales (180M–1.5B)** | <1.5B | Net-new SSM block — the key small-model entry of the quarter |
| **Voxtral TTS 4B** | 4B (audio) | New audio LM family worth tracking in `models/audio/` |
| **Mistral Small 4 (15B active / 119B total)** | borderline (15B active) — likely out-of-scope strict |
| **Reka Edge 2026** | ~8B (borderline) | Strong physical-AI baseline |
| **Zyphra ZAYA1-8B** | 8.4B (borderline; technically slightly >8B) | AMD-trained reasoning MoE |
| **NVIDIA Nemotron 3 Nano Omni (Nano class)** | "Nano" size — exact LM decoder not yet published; expected <8B | Hybrid Mamba-Transformer Omni — strong M2 candidate |
| **Hy-MT2 1.8B / 7B** | 1.8B / 7B | Translation specialization; useful niche |

**Strong recommendations for new `models/<name>/` slots** (priority order):

1. **Mamba-3** — the SSM block of the year; mandatory.
2. **Gemma 4 Elastic (E2B/E4B)** — novel Elastic + audio-in.
3. **Liquid LFM2.5** family — only non-Transformer recurrent backbone with public weights at this scale.
4. **Falcon-Edge (1.58-bit)** — only retrainable BitNet at this scale.
5. **Granite 4.1-3B / -8B** — clean dense baselines, very portable, well-documented.
6. **DeepSeek V4-Flash** (despite being 284B / 13B-A, it is the canonical CSA+HCA reference; consider as a "kvcache/attention" reference even if outside size budget).

---

## 6. Notable Announcements Without Releases

These are confirmed announcements where **no public weights** existed at 2026-06-06.

| Entry | Status | Why flagged |
|---|---|---|
| **Meta Llama 5 flagship 600B+** (2026-04-08) | Announced; partial weights *gated* on llama.com / HF, hosted-only on Together / Groq | Mixed reports about whether top-end weights are fully downloadable |
| **Meta Muse Spark** (2026-04-08) | Product-only via Meta AI app and family of Meta apps | No weights, no HF card, no paper |
| **Microsoft MAI-Thinking-1** (2026-06-02) | Private preview in Microsoft Foundry | No weights, no HF card |
| **Microsoft Aion 1.0** (2026-06-02) | Announced, weights committed for July 2026 HF release | Not yet downloadable |
| **xAI Grok 4.3** (2026-05-04) | API-only | No weights |
| **xAI Grok Build 0.1** (2026-05-14) | Early-access API | No weights |
| **xAI Grok 3 open weights** | Promised ~Feb 2026 by Musk; still unreleased as of 2026-06-06 | Multiple sources flag continued delay |
| **Baidu ERNIE 5.1** (2026-05-08) | API-only via Qianfan | No weights |
| **Qwen3.7-Max preview** (2026-05-20) | Hosted on chat.qwen.ai / lmarena.ai | No weights yet |
| **MiniMax-M3.0** (2026-06-01) | Launched 2026-06-01; weights staged to land on HF ~10 days later (i.e., mid-June) | Weights pending at cutoff |

---

## 7. Gaps in Publicly-Available Info

Items where I could not verify with a credible source, or where info is partial:

- **Phi-5** — I could not confirm an actual Microsoft release in this window. Public references found are deployment guides on third-party blogs (Spheron) that note specs "based on pre-release announcements." Likely not yet shipped as of 2026-06-06; the MS Build 2026 announcement was for MAI-Thinking-1 and Aion 1.0, not Phi-5.
- **SmolLM4** — No SmolLM4 found. Latest from HuggingFace TB org remains SmolLM3-3B.
- **Tulu 4** — No Tulu 4 release in window. Allen AI continued with Olmo 3.1 (late 2025) and a "DR Tulu" deep-research recipe; no Tulu-4 product.
- **OLMo 4** — Not announced. Olmo 3.1 was a late-2025 release; the 2026Q2 window has no new OLMo flagship.
- **InternLM 4** — Could not find InternLM 4 release. InternLM3 remains current; Intern-S2-Preview (scientific) and InternVL3.5 (Aug 2025) are most recent.
- **Yi 2 (01.AI)** — Not found. 01.AI has continued with Yi Lightning / Wanzhi platform; no Yi 2 release in window.
- **Snowflake Arctic 2 / DBRX 2** — Not found. Both organizations have moved away from flagship-LLM releases in 2026Q2.
- **Apple AFM open weights** — WWDC 2026 (June 8-12) falls just outside the window cutoff; Apple Foundation Models framework continues but no new open weights have surfaced.
- **Qwen3-VL "Edit"** — Searches turned up no specific "Edit" variant; only ongoing Qwen3-VL FP8 work and the new Qwen3.5-Omni.
- **Jamba 2.0** (a true v2 successor) — Jamba**2** (Jan 2026 release with 3B + Mini sizes) is just *outside* the window. No further Jamba release in 2026Q2.
- **Hymba 2** — Not found in window.
- **Magistral / Devstral 2** — Both **retired** in the window (absorbed into Mistral Medium 3.5).
- **Pixtral 2** — Not released as a separate model; vision is now part of Mistral Small 4 and Mistral Medium 3.5.
- **Apollo 2 / Eagle 3 / NVILA-2** — No releases surfaced inside the window.
- **GLM-5.2** — Not found.
- **Hunyuan-Standard / Hunyuan-T1 open weights** — T1 remains API-only.
- **OpenAI GPT-OSS update** — No new gpt-oss release in window; gpt-oss-120b / gpt-oss-20b remain current (released Aug 2025), with renewed AWS Bedrock GovCloud availability noted in April 2026.
- **Moshi 2** — Not found; Kyutai has not released a Moshi 2 successor in window.
- **MinerU 3** — Only minor point releases (3.1.x → 3.2.0); no formal "MinerU 3" major.
- **olmOCR 2 successor** — olmOCR 2 (2025-10) remains current.
- **DeepSeek R2** — Not released; reports indicate Liang Wenfeng has not greenlit it for performance reasons.
- **Codestral / Qwen2.5-Coder / DeepSeek-Coder updates** — In this window, code-specialist functionality is being folded into general-purpose flagships (Mistral Medium 3.5 absorbs Devstral 2; Qwen3.6-27B is the coder leader; DeepSeek-V4 covers code). No standalone code model release in window.

---

## 8. Github Link Index (alphabetized)

- ACE Studio / StepFun ACE-Step 1.5 — github.com/ace-step/ACE-Step-1.5
- allenai / open-instruct — github.com/allenai/open-instruct
- DeepSeek-AI — github.com/deepseek-ai
- DeepSeek-OCR — github.com/deepseek-ai/DeepSeek-OCR
- DeepSeek-OCR-2 — github.com/deepseek-ai/DeepSeek-OCR-2
- DeepSeek-V3.2-Exp — github.com/deepseek-ai/DeepSeek-V3.2-Exp
- DeepSeek V4 (org) — github.com/deepseek-ai
- Falcon LLM (TII) — github.com/falcon-llm
- Google DeepMind Gemma — github.com/google-deepmind/gemma
- HuggingFace open-r1 — github.com/huggingface/open-r1
- HuggingFace SmolLM — github.com/huggingface/smollm
- HuggingFace TB — huggingface.co/HuggingFaceTB
- IBM Granite — github.com/ibm-granite
- IBM Granite Vision — github.com/ibm-granite/granite-vision-models
- InternLM — github.com/InternLM/InternLM
- InternVL — github.com/OpenGVLab/InternVL
- LG AI Research EXAONE 4.5 — github.com/LG-AI-EXAONE/EXAONE-4.5
- LiquidAI — github.com/LiquidAI
- Llama-Omni (ICTNLP) — github.com/ictnlp/LLaMA-Omni
- Meta Llama — github.com/meta-llama
- MiniMax-M2 (M3 staged) — github.com/MiniMax-AI/MiniMax-M2
- Mistral AI — github.com/mistralai
- Nous Research Hermes-Agent — github.com/NousResearch/hermes-agent
- NVIDIA Megatron Core — github.com/NVIDIA/Megatron-LM
- NVIDIA Nemotron — developer.nvidia.com/nemotron (model hub)
- OpenAI gpt-oss — github.com/openai/gpt-oss
- OpenGVLab InternVL3.5 — github.com/OpenGVLab/InternVL
- Qwen (QwenLM org) — github.com/QwenLM
- Qwen3-Coder — github.com/QwenLM/Qwen3-Coder
- Qwen3.6 — github.com/QwenLM/Qwen3.6
- Qwen3-Omni — github.com/QwenLM/Qwen3-Omni
- Qwen3-VL — github.com/QwenLM/Qwen3-VL
- Skywork — github.com/SkyworkAI
- Skywork-R1V — github.com/SkyworkAI/Skywork-R1V
- State-spaces Mamba (Mamba-3 ref impl) — github.com/state-spaces/mamba
- Tencent Hunyuan org — github.com/Tencent-Hunyuan
- Tencent Hunyuan-A13B — github.com/Tencent-Hunyuan/Hunyuan-A13B
- Tencent HY-Embodied — github.com/Tencent-Hunyuan/HY-Embodied
- Tencent Hy-MT2 — github.com/Tencent-Hunyuan/Hy-MT2
- Tencent Hunyuan-TurboS — github.com/Tencent-Hunyuan/Hunyuan-TurboS
- Tencent HunyuanImage-3.0 — github.com/Tencent-Hunyuan/HunyuanImage-3.0
- Tencent Hunyuan3D-2 — github.com/Tencent-Hunyuan/Hunyuan3D-2
- Tencent Hunyuan T1 — github.com/Tencent/llm.hunyuan.T1
- Xiaomi MiMo — github.com/XiaomiMiMo
- Xiaomi MiMo-V2-Flash — github.com/XiaomiMiMo/MiMo-V2-Flash
- xAI — github.com/xai-org
- Z.AI GLM-V — github.com/zai-org/GLM-V
- Zyphra — github.com/Zyphra

Items with **neither** a public HF card nor a public GitHub repo at 2026-06-06 (flagged in §6): Meta Muse Spark, Microsoft MAI-Thinking-1, Microsoft Aion 1.0 (planned July), xAI Grok 4.3 / Grok Build 0.1, Baidu ERNIE 5.1, Qwen3.7-Max (preview, weights pending).

---

## Closing notes for the project

For the llm-layers M2 milestone the most important architectural deltas of the quarter are:

1. **Mamba-3** (pure SSM with complex state and MIMO decoding).
2. **DeepSeek-V4 CSA+HCA** sparse attention.
3. **Gemma 4 Elastic** sizes (E2B / E4B) and the MTP drafter pattern.
4. **NVIDIA Nemotron-3 hybrid Mamba-Attention MoE** at 120B and 550B scales.
5. **Falcon-Edge 1.58-bit** retrainable weights — the first low-bit family that supports continued pre-training.
6. **Liquid LFM2.5** as the only non-Transformer recurrent backbone shipping at this scale.
7. **MiniMax MSA** (M3.0, weights staged).

The under-8B in-scope candidates fall cleanly into three groups: (a) the dense small baselines (Granite 4.1-3B, Gemma 4 E2B/E4B); (b) the alternative-architecture small ones (Mamba-3 reference scales, LFM2.5-VL-450M, Falcon-Edge 1B/3B); (c) the small-multimodal (Granite Vision 4.1-4B, HY-Embodied MoT-2B, Falcon Perception 0.6B).
