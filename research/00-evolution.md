# 00 — Evolution of the LLM Layer (2022 Nov → 2026 Q1)

*Stream: evolution-narrative. Project: llm-layers. Date: 2026-06-04.*

> The connecting tissue between this project's five snapshot reports
> (`01-model-census`, `02-layer-sources`, `03-ihv-opsets`,
> `04-quantization`, `05-kvcache-attention`) and the API design spec.
> Those documents describe what an SLM looks like *today*. This one
> describes how it got that way, what was tried and abandoned, and which
> axes are still actively contested. Read this first, then any of the
> snapshots; or read the snapshots first and use this as a debrief.

---

## 1. Introduction — Why an evolution survey?

A snapshot of mainstream small-language-model (SLM) architecture taken
in mid-2026 looks misleadingly uniform. Pick any model card from the
Llama-3, Qwen-3, Gemma-3, Mistral, Phi-4, or DeepSeek-R1-Distill
families and you will see the same six bullets: decoder-only,
pre-norm RMSNorm, RoPE, SwiGLU or GeGLU, GQA, BPE tokenizer of
100-200K. Read enough of them in a row and you start to believe the
transformer block stopped evolving in 2023.

It did not. The apparent uniformity is the surface of a three-year
optimization sweep that retired at least ten serious architectural
candidates and standardized survivors that did not exist in late 2022.

Consider what changed between three concrete reference points:

* **2022 November — GPT-3.5 / ChatGPT.** The publicly inferrable
  architecture: decoder-only, dense MHA, learned absolute position
  embeddings (or rotary in some forks), GELU FFN, LayerNorm, BPE
  tokenizer of ~50K. The largest open analog at the time was
  GPT-NeoX-20B; the same template was used by GPT-J, OPT, BLOOM,
  Pythia.
* **2023 February → September — the Llama era.** LLaMA-1 (Feb 2023)
  set the template that survived: RMSNorm, RoPE, SwiGLU, pre-norm.
  Mistral 7B (Sep 2023) demonstrated that GQA worked at small scale
  and added sliding-window attention. By the end of 2023 every new
  base model from Qwen, ChatGLM, Baichuan, Yi shipped a Llama-like
  block.
* **2026 Q1 — today.** The Llama block is still the backbone, but the
  details have shifted: vocabularies are 4× larger (128K-200K, all
  Tiktoken-style BPE); RoPE θ has gone from 10K to between 500K and
  5M depending on family; GQA is universal (kv_heads ∈ {2,4,7,8});
  SwiGLU is universal in non-Gemma families and GeGLU in Gemma;
  context windows are 32K-128K standard and 1M-10M for long-context
  forks; MoE has split the "frontier" track (DeepSeek-V3, Qwen3-MoE,
  Llama-4) from the SLM track; FP8 is the training default for new
  Hopper/Blackwell-class runs and INT4 quantized variants are
  routinely shipped alongside BF16.

The list of *abandoned* designs is just as telling. Parallel
attention/FFN (GPT-J, Falcon-7B, Pythia), pure MQA (Falcon, PaLM
draft), ALiBi (MPT, BLOOM, Falcon), absolute and learned positional
embeddings, fully dense post-norm transformers, INT8 W8A8 dynamic
quantization, NTK-aware-only RoPE scaling, vAttention process-VM
tricks, Q4_0/Q4_1 GGUF formats, and pure sliding-window-everywhere
attention are all either gone or kept only as legacy support.

This document is the spine of the `llm-layers` survey. It answers
three questions the snapshots cannot:

1. **Why** the current consensus exists. The fact that every dense
   SLM uses pre-norm RMSNorm is the conclusion of a search; this
   document gives the path.
2. **What** was tried and discarded. The API should not give first-class
   support to design choices whose only living implementations are
   legacy compatibility paths.
3. **Which** axes are *still* contested. The 2026 snapshot has
   genuine divergence on sliding-window attention, MoE ratios,
   QK-norm formulations, long-context recipes, and MLA adoption; the
   API must parametrize these.

How to read this survey:

* Start here for context, then read the design spec
  (`docs/superpowers/specs/2026-06-04-llm-layers-design.md`).
* For per-model facts and shape inventories, go to `01-model-census`.
* For per-layer source-code traces (HF / vLLM / llama.cpp), go to
  `02-layer-sources`.
* For runtime / IHV-side ops and op-fusion conventions, go to
  `03-ihv-opsets`.
* For quantization scheme details, go to `04-quantization`.
* For KV-cache and attention-variant details, go to
  `05-kvcache-attention`.

Each per-axis discussion in §4 cross-links to the relevant snapshot
section.

---

## 2. Timeline of Mainstream SLM Releases (2022 Nov → 2026 Q1)

The table below records *architecturally significant* releases. A
release is included if (a) it shipped a new design slot a later
mainstream family adopted, (b) it was widely served, or (c) it is a
reference point for an axis discussed in §4. "Param Counts" lists the
sizes shipped at release; later size additions are folded into the
parent row only if the architecture is unchanged. "Architecture
innovation" is restricted to what the block looked like — quality,
data, post-training, or licensing changes are noted separately.

Dates are calendar release of the base/pretrained weights or
technical report, whichever is earlier. Sources are in §10; for the
configs and shapes themselves, see `01-model-census` §2.

| Date | Family.Version | Active params | Org | Architectural innovation | Notes |
|---|---|---|---|---|---|
| 2022-11 | GPT-3.5 / ChatGPT | undisclosed (≈175B) | OpenAI | None new at the layer level (decoder, MHA, learned-abs PE, GELU, LayerNorm); the launch is product-level (RLHF + chat product) | Defines the "before" baseline. |
| 2023-02 | LLaMA 1 | 7 / 13 / 30 / 65 B | Meta | **The template that won.** Pre-norm RMSNorm + RoPE + SwiGLU + decoder-only + SentencePiece BPE 32K + tied embeddings | First widely-distributed open weights that combined all four; downstream "Llama-likes" trace from here. |
| 2023-04 | Pythia | 70M – 12B | EleutherAI | RoPE + parallel attn/FFN (GPT-NeoX heritage) | Documented training trajectory; parallel attn/FFN persisted from GPT-J. |
| 2023-04 | MPT 7B | 7B | MosaicML | ALiBi + FlashAttention + full attn | Pre-Llama-2 ALiBi shipment; demonstrated FlashAttention-1 in production. |
| 2023-05 | Falcon 7B / 40B | 7 / 40 B | TII | Pure **MQA** + parallel attn/FFN (Falcon-style) + ALiBi (40B has both ALiBi & RoPE forks) | The decisive MQA-at-scale shipment; later overturned by GQA. |
| 2023-07 | Llama 2 | 7 / 13 / 70 B | Meta | **GQA at 70B**, retained MHA at 7/13B; RLHF chat models | First publicly successful GQA shipment in a flagship model. |
| 2023-08 | Qwen 1 7B | 7B | Alibaba | RoPE + LayerNorm + QKV bias (legacy) + SwiGLU + Chinese-extended BPE | Established the Qwen family vocab convention. |
| 2023-09 | Mistral 7B v0.1 | 7B | Mistral | **GQA at 7B** (kv=8), **sliding-window attention** (W=4096), small SP-32K vocab | The template change that made GQA "default at SLM scale". |
| 2023-10 | ChatGLM2 / Baichuan-2 | 6 / 13 B | Tsinghua / Baichuan | RoPE-style + RMSNorm; ChatGLM uses prefix-bidirectional mask | Bridge between Chinese decoder family and Llama template. |
| 2023-11 | Qwen 1.5 | 0.5 – 72 B | Alibaba | Llama-style RMSNorm + RoPE; widened size ladder | Quality refresh; same block as Qwen 1. |
| 2023-12 | Mixtral 8x7B | 47B (12.9B active) | Mistral | **Sparse MoE at SLM scale** (8 experts, top-2, dense layers absent) | First widely-served open MoE; established `n_routed_experts` / `num_experts_per_tok` config keys. |
| 2023-12 | Phi-2 2.7B | 2.7B | Microsoft | MHA + partial RoPE (0.5) + LayerNorm; data-quality story | Small-data quality demonstration; LayerNorm holdover. |
| 2023-12 | Mamba 1 | 130M – 2.8B | Stanford / CMU | **Selective SSM (S6)**; replaces attention with state-space recurrence | First "pure attention-free" architecture to ship at >1B with competitive zero-shot. |
| 2024-01 | TinyLlama | 1.1B | community | GQA (kv=4) + RoPE; Llama-2 vocab and template at toy scale | Reference small-Llama. |
| 2024-02 | Gemma 1 | 2 / 7 B | Google | **GeGLU** (gelu-tanh + gate) replaces SwiGLU; head_dim=256; **dual norm** around each sublayer (sandwich-like) | First mainstream non-SiLU activation; Gemma-SP 256K vocab. |
| 2024-02 | OLMo 1 7B | 7B | AI2 | Fully open recipe (data + code + ckpts); Llama-style block | The first fully reproducible mainstream training run. |
| 2024-02 | StarCoder 2 | 3 / 7 / 15 B | BigCode | GQA + **biases on QKV and MLP** (legacy) + LayerNorm + gated GELU-tanh MLP + SWA | Code-domain holdout from the Llama template. |
| 2024-02 | BitNet b1.58 | up to 3B (paper) | Microsoft | **Ternary weights** (-1, 0, +1) trained from scratch; activations 8-bit | "The Era of 1-bit LLMs" — opens the 1.58-bit lineage. |
| 2024-03 | DBRX | 132B (36B active) | Databricks | **Fine-grained MoE** (16 experts, top-4) at frontier scale | Doubled the experts-per-token convention vs Mixtral. |
| 2024-04 | Llama 3 | 8 / 70 B | Meta | **128K tiktoken-style BPE vocab** (a 4× expansion over Llama-2); GQA at all sizes; RoPE θ=500K; otherwise the Llama block unchanged | First public 128K BPE; the new vocab is the headline architectural change. |
| 2024-04 | Mixtral 8x22B | 141B (39B active) | Mistral | MoE scaled up; 64K SWA | Sets Mixtral's larger template. |
| 2024-04 | Phi-3 mini / small / medium | 3.8 / 7 / 14 B | Microsoft | mini: **fused gate-up SwiGLU** + fused QKV; small: **block-sparse attention** (dense every 2nd layer) + cl100k tiktoken; medium: standard Phi-3 block | Microsoft's first widely-used SLM family; Phi-3-small is the structural outlier. |
| 2024-04 | Phi-3-mini 128K | 3.8B | Microsoft | **LongRoPE** (per-dim short/long factors) | First per-dim learned-factor RoPE rescaling. |
| 2024-04 | CodeGemma | 2 / 7 B | Google | Gemma 1 block specialized for code; 2B uses **MQA (kv=1)** | One of the last published MQA dense shipments. |
| 2024-05 | DeepSeek-V2 | 236B (21B active) | DeepSeek | **Multi-Head Latent Attention (MLA)** + alternating dense/sparse MoE + YaRN scaling | The largest single-axis innovation since 2023: replaces K and V with a shared low-rank latent, collapsing KV cache by ~50×. |
| 2024-05 | OpenELM | 270M / 450M / 1.1B / 3B | Apple | **Per-layer head and FFN width scaling**; not all layers equal | A "non-uniform depth" template; not widely adopted but persists in Apple's on-device line. |
| 2024-05 | Mistral 7B v0.3 | 7B | Mistral | Dropped SWA; widened SP-32K to 32768 with extended special tokens; RoPE θ raised to 1e6 | SWA-everywhere abandoned in dense Mistral. |
| 2024-06 | Qwen 2 | 0.5 – 72 B | Alibaba | Move to **Tiktoken-style BPE** (≈152K vocab); QKV bias retained | Vocab modernization; otherwise Llama-style. |
| 2024-06 | DeepSeek-Coder-V2-Lite | 16B (2.4B active) | DeepSeek | MLA in SLM-active-size MoE form (2 shared + 64 routed experts, first layer dense) | First "MLA SLM" — relevant to the census. |
| 2024-06 | Apple Foundation Model (on-device) | ≈3B | Apple | KV-cache sharing across layers + 2-bit QAT; decoder-only template otherwise | Defines the "production on-device" axis with cross-layer KV sharing. |
| 2024-07 | Llama 3.1 | 8 / 70 / 405 B | Meta | **Smooth piecewise RoPE scaling** ("llama3" scaling type) to 128K context | A new long-context recipe (parameters: `low_freq_factor`, `high_freq_factor`, `original_max_pos`, `factor=8`). |
| 2024-07 | Mistral Nemo 12B | 12B | Mistral × NVIDIA | Tekken tiktoken-style vocab; 128K context | Nvidia-codesigned training; first Mistral with a Tiktoken-class vocab. |
| 2024-07 | Gemma 2 | 2 / 9 / 27 B | Google | **Alternating local/global attention** (1:1 SWA:full per layer); **soft-cap on attention and final logits** (50, 30); sandwich norm (pre+post per sublayer); RMSNorm uses (1+w) gain | Most architecturally busy block of 2024. |
| 2024-07 | InternLM 2.5 | 7B | Shanghai AI Lab | Dynamic-NTK RoPE; ultra-long-context fork | Dynamic-NTK persists in InternLM through V3. |
| 2024-08 | Phi-3.5 mini / MoE / vision | 3.8 / 42 / 4.2 B | Microsoft | LongRoPE generalized across mini/MoE/vision | Vision tower bolt-on; same text block. |
| 2024-09 | Qwen 2.5 | 0.5 – 72 B | Alibaba | RoPE θ=1e6; Qwen-BPE 151936 cleaned up; GQA (kv ∈ {2,4}) | The Qwen "consensus" base used by R1-Distill. |
| 2024-09 | Llama 3.2 | 1 / 3 / 11 / 90 B | Meta | 1B/3B small-Llama with full 128K vocab; 11/90B = vision towers | First sub-3B model in the Llama mainline. |
| 2024-09 | OLMoE 1B-7B | 7B total, 1B active | AI2 | Fully open MoE recipe (64 experts, top-8, no shared) | Mirrors the Mixtral-style top-8 fine-grained MoE in the open. |
| 2024-09 | MiniCPM 3 4B | 4B | OpenBMB | **MLA in dense (non-MoE) SLM** + LongRoPE + μP-style scalars (`scale_emb`, `scale_depth`) + cross-layer KV sharing | MLA in a sub-5B dense model; rare. |
| 2024-10 | Granite 3.0 | 2 / 8 B | IBM | **μP scalar multipliers** (`embedding_multiplier`, `attention_multiplier`, `logits_scaling`, `residual_multiplier`) | Re-introduces μP into a shipped open model; the API must support these scalars. |
| 2024-10 | Mamba 2 | up to 2.7B | Albert Gu et al. | **SSD formulation**: SSM-as-structured-matrix; tied to attention via duality | Re-bases SSM theory; enables Hymba / Zamba2 / Falcon-Mamba families. |
| 2024-10 | Zamba 2 | 1.2 / 2.7 / 7 B | Zyphra | Mamba-2 layers with periodic shared global attention block | First "1 attention serves many SSM" pattern. |
| 2024-11 | OLMo 2 | 7 / 13 B | AI2 | **Post-norm (block-output) + QK-RMSNorm over full head channels** | The "post-norm minority" comeback; QK-norm with full-channel γ shape. |
| 2024-11 | Hymba 1.5B | 1.5B | NVIDIA | **Parallel Mamba + attention heads in the same hybrid module**; 90% SWA + 3 full + meta-tokens; KV cache shared across layer pairs | First "parallel hybrid heads" template. |
| 2024-11 | SmolLM2 | 135M / 360M / 1.7B | HuggingFace | Tiny-scale Llama-style; 1.7B uses MHA (not GQA); 49K BPE | Modern reference for sub-2B; tied embeddings on all sizes. |
| 2024-12 | Granite 3.1 / 3.2 | 2 / 8 B | IBM | Same Granite block; ctx 128K | Quality refresh. |
| 2024-12 | Phi-4 14B | 14B | Microsoft | Extended-training data quality; tiktoken o200k vocab (Phi-4-mini); architecturally close to Phi-3-medium | Data-centric upgrade; o200k vocab is new. |
| 2024-12 | DeepSeek-V3 | 671B (37B active) | DeepSeek | **MTP heads, FP8 training, sigmoid-routing + bias correction**, MoE with shared experts + routed experts | The current frontier reference; MTP and FP8 are first-class. |
| 2025-01 | DeepSeek-R1 + R1-Distill | up to 70B | DeepSeek | **R1-Zero / R1 reasoning recipe**; distilled into Qwen-2.5 and Llama-3.1 backbones — no architecture change in the distillates | Architecturally same backbones; reasoning trace post-train. |
| 2025-02 | BitNet b1.58 2B4T | 2B | Microsoft | First **native 1.58-bit weight** training at 2B / 4T tokens | Productionizes ternary weights end-to-end. |
| 2025-02 | NSA (paper, DeepSeek + PKU) | n/a | DeepSeek | **Native Sparse Attention**: trained-in sparse pattern, hardware-aligned | ACL 2025 Best Paper; defines a new attention design axis. |
| 2025-02 | Phi-4-mini 3.8B | 3.8B | Microsoft | Partial RoPE (0.75) + LongRoPE + GQA + o200k vocab + fused gate-up | Standard Phi-3-mini block updated. |
| 2025-03 | Gemma 3 1 / 4 / 12 / 27 B | 1 – 27 B | Google | **5:1 SWA:full alternation**, **dual RoPE θ per layer type** (10K local, 1M global), image input, **soft-cap removed**, QK-norm added | Removes the only Gemma-2 features that did not generalize; multimodal. |
| 2025-04 | Qwen 3 0.6 / 1.7 / 4 / 8 / 14 / 32 B + 30B-A3B MoE | 0.6 – 32 B (3B active MoE) | Alibaba | **QK-RMSNorm on head_dim**, **thinking/non-thinking dual mode**, RoPE θ=5M (small sizes), QKV bias removed | The Qwen-3 family establishes QK-norm-on-head_dim. |
| 2025-04 | Llama 4 Scout / Maverick / Behemoth | 17B active (×16 / ×128 experts) / unreleased | Meta | **MoE-only Llama**; **iRoPE** (interleaved NoPE layers + temperature scaling); native multimodal; 10M context | First Llama with MoE in the mainline; NoPE re-emerges via iRoPE. |
| 2025-05 | DeepSeek-V3.1 (interim refresh) | 671B (37B active) | DeepSeek | DeepSeek-V3 with extended training | Quality refresh; same block. |
| 2025-07 | SmolLM3 3B | 3B | HuggingFace | **NoPE every 4th layer**, RoPE θ=5e6, GQA, Llama-3 vocab (LL3 128K), embedding decoupling at 3B | Productionizes per-layer NoPE alternation at SLM scale. |
| 2025-07 | Phi-4-mini-flash 3.8B | 3.8B | Microsoft | **SambaY**: Mamba + SWA + 1 full-attention layer + cross-decoder with Gated Memory Units; differential attention | Microsoft's hybrid Mamba production SLM. |
| 2025-08 | GPT-OSS 20B / 120B | 20B / 117B (MoE) | OpenAI | **Trained-in attention sinks**; MXFP4 weights for MoE; SWA + sink + GQA | OpenAI's first open release; first widely-shipped "sinks trained-in". |
| 2025-09 | DeepSeek-V3.1 (refresh) / Qwen3-Next | various | DeepSeek / Alibaba | Hybrid linear/full attention, MoE with linear stack | Hybrid attention in MoE form. |
| 2025-10 | RWKV-7 / Mamba-Coder | up to 7B | community | RWKV-7 finalized linear-attn variant; Mamba-2 code specialization | Linear-attention production-ready. |
| 2025-11 | Mistral Small 3.1 | up to 24B | Mistral | **Sinks trained-in** + improved long-context | Confirms sinks as a cross-org trend in 2025. |
| 2026-Q1 | Apple Foundation Model on-device (2026 refresh) | ≈3B | Apple | Cross-layer KV sharing + 2-bit QAT, PT-MoE on the server tier | The current production on-device reference. |

The table is intentionally not exhaustive — it lists only releases
that ship a new design slot or anchor a discussion in §4. About 45
shipped variants total are catalogued in `01-model-census`.

A few cross-cutting observations from the timeline:

* **2023 was the consolidation year.** Every dense-decoder pre-norm
  RMSNorm + RoPE + SwiGLU choice from LLaMA-1 became standard by
  Mistral 7B (Sep 2023). The Falcon → Llama 2 → Mistral arc shows
  parallel-residual, MQA, and ALiBi being replaced almost
  simultaneously.
* **2024 was the long-context year.** RoPE θ values, scaling
  schedules (PI, NTK, YaRN, Llama-3, LongRoPE) and SWA alternations
  all moved between January and December. Every flagship model
  rebased its θ at least once.
* **2024 Q4 → 2025 was the efficiency year.** MLA (DeepSeek-V2/V3),
  MoE-with-shared-experts (DeepSeek), MTP heads, FP8 training, MXFP4
  weights, BitNet-1.58 — these are all "more useful work per unit of
  compute" innovations, not capability ones.
* **2025 was the hybrid year.** Hymba, SambaY/Phi-4-mini-flash,
  Zamba 2, Jamba, Falcon-H1 made the SSM+attention hybrid template
  shippable at sub-10B scale.
* **2026 Q1 is the "consensus + active divergence" year.** The
  Llama-style block is the floor; on top of that, organizations
  disagree on SWA presence, MoE on/off, QK-norm location, MLA vs
  GQA, and long-context recipe. The divergences are the focus of §6.

---

## 3. Lineage Diagrams

The genealogies below show base-checkpoint inheritance (solid
arrows), architectural-template inheritance (dashed-style notes), and
domain specialization (forks). Sizes that share architecture are
collapsed.

### 3.1 Llama family

```
GPT-NeoX (2022) ──── (Pre-LN + parallel attn/FFN; abandoned)
                      │
                      ▼
                 LLaMA 1  (2023-02; Pre-LN RMSNorm + RoPE + SwiGLU + serial)
                      │
                      ▼
                 Llama 2  (2023-07; GQA at 70B only; RLHF)
                      │
            ┌─────────┴────────────┐
            ▼                      ▼
       Llama 3   (2024-04)    CodeLlama (2023-08)   TinyLlama (2024-01)
       128k BPE, GQA          (Llama 2 base)        Llama 2 vocab, GQA
       all sizes
            │
            ▼
       Llama 3.1 (2024-07; smooth RoPE scaling → 128K ctx)
            │
            ▼
       Llama 3.2 (2024-09; 1B/3B added; 11/90B vision)
            │
            ▼
       Llama 4   (2025-04; MoE-only; iRoPE; multimodal)
       Scout 17B-16E / Maverick 17B-128E / Behemoth
            
External adoption of the Llama template:
   Llama 2/3   ──→ MiniCPM-Llama3, DeepSeek-R1-Distill-Llama,
                   Granite (vocab swap), LLaVA-NeXT-Llama (vision)
```

### 3.2 Mistral family

```
LLaMA 1 template
       │
       ▼
Mistral 7B v0.1 (2023-09)
GQA + SWA (W=4096) + SP-32K
       │
       ├──→ Mixtral 8x7B (2023-12; sparse MoE on Mistral block)
       │             │
       │             ▼
       │       Mixtral 8x22B (2024-04; scaled)
       │
       ├──→ Codestral (2024-05; code) / Mathstral (2024-07; math)
       │
       ▼
Mistral 7B v0.2/v0.3 (2024-03 / 05; SWA dropped; θ=1e6; SP-32768)
       │
       ▼
Mistral Nemo 12B (2024-07; Tekken tiktoken vocab)
       │
       ▼
Mistral Small 3 / 3.1 (2025; trained-in attention sinks)
```

### 3.3 Qwen family

```
Qwen 1 (2023-08; LN + QKV bias + SwiGLU; Qwen-BPE)
       │
       ▼
Qwen 1.5 (2023-11; RMSNorm; widened ladder)
       │
       ▼
Qwen 2 (2024-06; Tiktoken-style 152K vocab; QKV bias)
       │
       ▼
Qwen 2.5 (2024-09; θ=1e6; GQA kv ∈ {2,4}; standard Llama block)
       │
       ├──→ Qwen 2.5-Coder, -Math (2024-09 onward)
       │
       ▼
Qwen 3 (2025-04; QK-RMSNorm per-head_dim; thinking/non-thinking; QKV bias removed)
       │
       ├──→ Qwen 3 30B-A3B MoE (top-8/128, no shared)
       │
       ▼
Qwen 3-Next (2025 Q3; hybrid linear/full attention)
```

### 3.4 Phi family

```
Phi-1 (2023; small; data-centric)
       │
       ▼
Phi-1.5 / Phi-2 (2023-09 / 12; MHA + partial RoPE + LayerNorm)
       │
       ▼
Phi-3 mini (3.8B) ─ small (7B; block-sparse, cl100k) ─ medium (14B)  (2024-04)
       │           fused gate-up SwiGLU, fused QKV
       │
       ▼
Phi-3.5 mini / MoE / vision (2024-08; LongRoPE)
       │
       ▼
Phi-4 14B (2024-12; extended data) ── Phi-4-mini 3.8B (2025-02; partial RoPE 0.75; o200k)
                                                │
                                                ▼
                                         Phi-4-mini-flash (2025-07; SambaY hybrid + DiffAttn)
```

### 3.5 Gemma family

```
Gemma 1 (2024-02; GeGLU + head_dim=256 + SP-256K + dual norm)
       │
       ├──→ CodeGemma (2024-04; 2B uses MQA!)
       ├──→ RecurrentGemma (2024-04; Griffin SSM hybrid)
       ├──→ PaliGemma (2024-05; vision text tower)
       │
       ▼
Gemma 2 (2024-07; alternating SWA 1:1; soft-cap; sandwich norm; (1+w) RMSNorm)
       │
       ▼
Gemma 3 (2025-03; 5:1 SWA:full; dual RoPE θ; QK-norm; soft-cap REMOVED; multimodal)
       │
       └──→ Gemma 3n (2025; on-device PT-MoE variant)
```

### 3.6 DeepSeek family

```
DeepSeek LLM (2023-12; MHA + Llama template)
       │
       ▼
DeepSeek-V2 (2024-05; MLA + alternating dense/sparse MoE + YaRN)
       │
       ├──→ DeepSeek-Coder-V2-Lite (2024-06; 16B-A2.4B; MLA + MoE)
       │
       ▼
DeepSeek-V3 (2024-12; MTP heads + FP8 training + sigmoid routing + bias correction)
       │
       ▼
DeepSeek-R1 + R1-Zero (2025-01; reasoning recipe; backbone unchanged)
       │
       ├──→ R1-Distill-Qwen-1.5B / 7B  (Qwen 2.5 backbone)
       ├──→ R1-Distill-Llama-8B / 70B  (Llama 3.1 backbone)
       │
       ▼
DeepSeek-V3.1 (2025; NSA-influenced; sparse attn experiments)
```

### 3.7 SSM family

```
S4 (2021) ──→ S5 ──→ H3 ──→ Mamba 1 (2023-12; selective scan / S6)
                              │
                              ▼
                        Mamba 2 (2024-10; SSD; matrix-form duality)
                              │
                              ├──→ Hymba (NVIDIA, 2024-11; parallel attn+SSM)
                              ├──→ Zamba 2 (Zyphra, 2024-10; periodic shared attn)
                              ├──→ Falcon-Mamba (2024-08; pure Mamba 7B)
                              ├──→ Samba / Phi-4-mini-flash (2024-12 / 2025-07; Mamba+SWA)
                              └──→ Jamba (2024-04; Mamba + 1-in-8 attn + 1-in-2 MoE)
```

### 3.8 Falcon family

```
Falcon 7B / 40B (2023-05; MQA + parallel attn/FFN + ALiBi)
       │
       ▼
Falcon 180B (2023-09)
       │
       ├──→ Falcon-Mamba 7B (2024-08; pure Mamba)
       │
       ▼
Falcon 3 (2024-12; modernized: GQA + RoPE + Llama-template)
       │
       ▼
Falcon-H1 (2025; Mamba/attention hybrid)
```

### 3.9 BitNet family

```
BitNet 1.0 (2023-10; binary weights, no FP)
       │
       ▼
BitNet b1.58 (2024-02; ternary {-1, 0, +1}; 3B paper-scale; matches FP16 perplexity)
       │
       ▼
BitNet b1.58 2B4T (2025-04; first end-to-end 2B / 4T-token trained ternary model)
```

---

## 4. Per-Axis Evolution

This is the core of the survey. For each design axis we trace the
historical arc, name the inflection points, and identify the
currently active variants.

### 4.1 Attention head structure

**Arc:** MHA → MQA → GQA → MLA → SSM hybrids → NSA.

* **MHA (2017–2022).** `n_kv_heads = n_q_heads`. KV cache per token
  per layer is `2 · H · d_h`. The original Vaswani template.
* **MQA (2019 paper / PaLM 2022 / Falcon 7B 2023).** `n_kv_heads = 1`,
  collapsing the KV cache by a factor of `H`. Quality regression
  at ≥2B unless trained from scratch. Falcon 7B was the largest
  open MQA shipment.
* **GQA (Ainslie et al. 2023; Llama 2-70B 2023-07; Mistral 7B 2023-09).**
  `n_kv_heads ∈ {H/2, H/4, H/8}`. Trades a small KV cache fraction
  for negligible quality loss. Mistral 7B is the inflection: GQA at
  7B with kv=8 became the new default. Llama 3 (2024-04) shipped GQA
  at all sizes.
* **MLA (DeepSeek-V2, 2024-05).** Replaces K and V with a shared
  low-rank latent `c_kv ∈ R^{d_c}` (typically 512). Cache stores
  `c_kv + qk_rope_head_dim` per token (~576 dims). DeepSeek-V2's KV
  cache is ~57× smaller than the MHA equivalent. RoPE is incompatible
  with the latent-absorption trick, so each head is split into
  `nope` (128 dims, non-rotated, served from latent) and `rope`
  (64 dims, rotated, cached separately). Adopted by MiniCPM-3 4B
  (the only sub-5B dense MLA model).
* **SSM hybrids (Mamba 2023-12 → Jamba 2024-04 → Hymba 2024-11 → Samba/Phi-4-mini-flash 2024-12/2025-07).**
  Replace attention with a constant-state recurrent module on some
  layers; keep attention on others. Jamba: 1 attention every 8
  layers + 1 MoE every 2. Hymba: parallel heads. Phi-4-mini-flash:
  Mamba + SWA + 1 full + cross-decoder GMU.
* **NSA (DeepSeek 2025-02).** Trained-in sparse attention; three
  parallel branches (compressed, selected, sliding). Native sparsity
  rather than inference-time approximation. Influential but not yet
  productionized in a flagship dense SLM (DeepSeek-V3.1 incorporates
  ideas).

Comparison (per token, per layer, d_h = 128, H = 32):

| Variant | KV bytes (fp16) | Quality at SLM scale | FLOPs vs MHA |
|---|---|---|---|
| MHA | 8 192 (`2·H·d_h·2`) | reference | 1.0 |
| GQA (kv=8) | 2 048 | parity within noise | ≈0.78 at attention (FFN dominates total) |
| MQA (kv=1) | 256 | regression at >2B | ≈0.75 |
| MLA (d_c=512, rope=64) | 1 152 | ≥ parity (DeepSeek-V2 §3) | similar to GQA |
| Pure SSM (Mamba 2.8B) | constant ~16 KB total state per layer (no growth) | parity at <3B; mixed >3B | constant memory; lower than attention for long context |
| Hybrid Hymba | mixed: 3 full + many SWA + Mamba | competitive with Llama-3.2 1B | mixed |

(Detailed values per shipped variant in `01-model-census` §2 and
`05-kvcache-attention` §1.1.)

The API axis: `kv_heads` (integer), plus `attention.kind ∈ {mha, gqa,
mqa, mla, mamba, mamba2, rwkv, samba, differential, retention,
identity}` plus MLA-specific `q_lora_rank`, `kv_lora_rank`,
`qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`.

### 4.2 Positional encoding

**Arc:** absolute → ALiBi → RoPE → RoPE θ growth → PI → NTK → YaRN
→ Llama-3 smooth → LongRoPE → NoPE alternation. ALiBi mostly abandoned.

* **Absolute / learned (GPT-2, BLOOM, OPT).** Position-token embedding
  added at input. Hard to extrapolate; abandoned for decoders by
  ~2023.
* **ALiBi (Press et al. 2021; MPT 2023; BLOOM; Falcon 40B).** Adds
  `-m · (i - j)` to attention logits, with m per-head. No
  positional embedding to learn; extrapolates by construction. *Why
  abandoned*: empirically does not improve quality over RoPE at
  >8K context once RoPE scaling becomes available; and the
  per-head slope is one more axis runtimes have to plumb. No
  flagship SLM in 2025-26 uses ALiBi.
* **RoPE (Su et al. 2021; LLaMA 1 2023-02).** Rotate
  `(x_{2i}, x_{2i+1})` pairs by `m · θ_i`. Parameter: `θ_base`
  (rope_theta). LLaMA-1 used 10K; Llama-3 raised to 500K; Mistral
  v0.3 / Qwen 2.5 use 1M; Qwen 3 small uses 5M; Granite 3.3 uses
  1e7; SmolLM3 uses 5e6. **Why θ grew**: longer pretraining context
  needs lower-frequency rotations to disambiguate distant tokens.
  Increasing θ pushes the "wavelength too long to disambiguate"
  cutoff further into the dimension list.
* **Position Interpolation (PI; Chen et al. 2023).** Scale position
  index by `s = L_new / L_train` before applying RoPE. Cheap,
  effective for ~2× extrapolation. Used in Llama-2-32K, CodeLlama.
* **NTK-aware (bloc97 Reddit 2023; Together LLongMA).** Replace
  `θ` with `θ · s^(d_h/(d_h−2))`; scales high-frequency dims less,
  low-frequency dims more. Smoother than PI. *Why abandoned*: YaRN
  and Llama-3 smooth scaling produce better quality / context
  trade-off.
* **YaRN (Peng, Quesnelle, Kingma 2023; DeepSeek-V2 2024-05;
  Qwen2.5-1M).** Piecewise: extrapolate high-frequency, interpolate
  low-frequency, ramp in between; plus temperature `1/t = 0.1·ln(s)+1`
  on softmax. Parameters: `factor`, `original_max_position`,
  `attn_factor`, `beta_fast`, `beta_slow`.
* **Llama-3 smooth (Meta 2024-07).** Same shape as YaRN but
  parametrized as `low_freq_factor`, `high_freq_factor`,
  `factor=8`, and *no* softmax temperature term. Wavelength gate
  uses absolute wavelengths.
* **LongRoPE (Ding et al. 2024; Phi-3 long-ctx 2024-04).** Per-dim
  learned multipliers (`short_factor`, `long_factor` of length
  `d_h/2`); attention scale rebased as `sqrt(1 + log(scale)/log(orig))`.
  The most expressive RoPE variant; only Phi-3 / Phi-3.5 / Phi-4 /
  MiniCPM-3 use it.
* **NoPE alternation (SmolLM3 2025-07; Llama-4 iRoPE 2025-04).**
  Skip RoPE entirely on selected layers (every 4th for SmolLM3).
  The non-positional layers "see" relative ordering only through
  earlier-layer outputs.
* **Partial RoPE.** Apply RoPE to only `d_rope < d_h` dims of each
  head. Phi-1/1.5/2 (0.5); StableLM-2 (0.25); Phi-4-mini (0.75);
  GPT-NeoX (0.25). Persists primarily in Phi-derived models.
* **Per-layer θ (Gemma 3 2025-03).** Full-attention layers use
  `θ=1e6`, sliding layers use `θ=10000`. Two RoPE caches per model.
* **Multi-axis (M-RoPE) for VL models.** Qwen2-VL, Phi-3.5-Vision
  split RoPE dims into temporal/height/width axes. Not relevant for
  pure text SLMs.

**Why RoPE won.** It is multiplicative (commutes with linear maps
of the right form, enabling cache absorption in MLA), it extrapolates
gracefully with three independent scaling families (PI / NTK / YaRN
/ Llama-3 / LongRoPE), and runtimes can express it as a single op
(OpenVINO RoPEFusion, llama.cpp `ggml_rope_ext`).

The API axis: `rope.kind ∈ {none, vanilla, llama3, yarn, longrope,
dynamic_ntk, linear, dual_per_layer}`, `rope.theta`, `rope.partial`,
`rope.basis ∈ {interleaved, split_half}`, plus the per-variant
parameters.

### 4.3 Normalization

**Arc:** LayerNorm → RMSNorm → pre-norm dominance → QK-norm
revival → dual / sandwich norm → post-norm minority comeback → (1+w)
RMSNorm gain.

* **LayerNorm (GPT-2).** `mean, var` removal + learnable γ, β. Still
  used in Phi-3-small, StarCoder 2, RWKV. Slightly more expensive
  than RMSNorm; gradient-flow differences usually negligible at
  inference.
* **RMSNorm (Zhang & Sennrich 2019; T5; LLaMA 1).** Only the
  variance term: `x · rsqrt(mean(x²) + ε) · γ`. Half the FLOPs of
  LayerNorm; preserves stability at FP16. Universal in modern dense
  SLMs.
* **Pre-norm (Xiong et al. 2020; LLaMA 1).** Apply norm to the
  residual stream *before* the sublayer; add the sublayer output.
  Dominant in 2023-24.
* **Post-norm comeback (OLMo 2 2024-11).** Apply norm to the
  sublayer's *output* before adding to residual. OLMo 2 reports
  improved training stability at fp32; bf16 inference is
  identical. Minority position; ~2 published flagships use it.
* **Sandwich norm / dual norm (Gemma 2 2024-07; Gemma 3 2025-03;
  CodeGemma).** Four norms per layer: `input_layernorm` and
  `post_attention_layernorm` around the attention sublayer;
  `pre_feedforward_layernorm` and `post_feedforward_layernorm` around
  the FFN. The "post" norms run *inside* the residual branch (`y =
  res + norm(sublayer(norm(x)))`), not on the residual.
* **(1+w) RMSNorm (Gemma; the "gemma_1plus_w" mode).** Replace
  `output · γ` with `output · (1 + γ)`. Weights initialized to
  zero so initial gain is 1.0. Mathematically equivalent to RMSNorm
  with shifted weights, but the IR cannot silently transpose
  between modes without reweighting.
* **QK-norm.** Three locations × three shapes have shipped:
  * **OLMo 2 (2024-11): full-channel γ shape**. `q_norm =
    RMSNorm(num_attention_heads · head_dim)`; norm acts on the
    concatenated heads *before* the head-reshape. Applied before
    RoPE.
  * **Qwen 3 (2025-04): per-head_dim γ shape**. `q_norm =
    RMSNorm(head_dim)`; norm acts on each head individually after
    head-reshape, before RoPE. The natural per-head normalization.
  * **Gemma 3 (2025-03): per-head_dim γ shape, after RoPE**. Same
    shape as Qwen 3 but Gemma 3 applies after the rotation.
  These are mathematically distinct (the OLMo formulation ties γ
  across heads; the others do not; the placement-before-vs-after-RoPE
  changes the rotation-equivariance). The API must expose both
  shape and placement.
* **DeepNorm (Microsoft 2022).** Training-only stability hack
  (`α · LN(αx + sublayer(x))`); inference reduces to a re-weighted
  pre-norm. Not relevant to the API surface.

The API axes: `norm.type ∈ {rmsnorm, layernorm}`, `norm.mode ∈
{classic, gemma_1plus_w}`, `pre_norm_attn`, `post_norm_attn`,
`pre_norm_ffn`, `post_norm_ffn` (each a present/absent toggle, with
`gemma3` = all four), `qk_norm.kind ∈ {none, per_head_dim,
per_full_channels}`, `qk_norm.placement ∈ {pre_rope, post_rope}`.

### 4.4 FFN / Channel mixer

**Arc:** GeLU → SwiGLU → GeGLU → fused gate-up → SwiGLU-MoE → fine-grained
MoE → shared+routed experts → sigmoid routing.

* **GeLU MLP (GPT-2, BLOOM).** `down(gelu(up(x)))`. Two linears, no
  gate.
* **SwiGLU (Shazeer 2020; LLaMA 1 2023-02).** `down(silu(gate(x)) ·
  up(x))`. Three linears (`gate_proj`, `up_proj`, `down_proj`),
  hidden dim 8/3·D (LLaMA convention). Standard in every Llama,
  Qwen, Mistral, OLMo, SmolLM, Granite, Yi, InternLM, MiniCPM,
  StableLM, R1-distill.
* **GeGLU (Gemma 2024-02).** Same structure, GELU-tanh instead of
  SiLU. Slightly more compute for the activation; quality difference
  inside training noise.
* **Fused gate-up SwiGLU (Phi-3 2024-04).** `gate_up_proj: D → 2·Df`
  is one linear; chunk into gate/up at the activation. Same math;
  quantization-relevant because the gate and up halves quantize
  jointly. vLLM uses the same fusion for all Llama-style models
  (the weight loader merges).
* **GeGELU (Phi-3-small 2024-04).** Microsoft's paired-channel gated
  GELU; not the same as GeGLU. Distinct named activation; the only
  model that uses it is Phi-3-small.
* **Gated GELU-tanh with biases (StarCoder 2).** Maintains biases on
  all MLP linears; outlier among modern decoders.
* **SwiGLU-MoE (Mixtral 2023-12).** Each expert is a SwiGLU; router
  selects top-k experts. Mixtral: 8 experts, top-2. Becomes the
  template.
* **Fine-grained MoE (DBRX 2024-03; DeepSeek-V2 2024-05).** Many
  small experts. DBRX: 16 experts, top-4. DeepSeek-V2: 64 routed +
  2 shared, top-6. Qwen3-30B-A3B: 128 experts, top-8, no shared.
* **Shared + routed experts (DeepSeek-V2).** A small number of
  "always-on" shared experts plus routed experts. Stabilizes routing
  and reduces expert under-utilization.
* **Sigmoid routing + bias correction (DeepSeek-V3 2024-12).**
  Router uses sigmoid (not softmax) over expert logits; a small
  bias is added/subtracted from each expert's logit to push toward
  uniform load. Replaces the "auxiliary loss" load-balancing of
  Mixtral-era MoE.
* **`first_k_dense_replace` (DeepSeek-V2).** First N layers are
  dense (no MoE). DeepSeek-V2-Lite: N=1. Qwen3-30B-A3B: N=0
  (sparse from layer 0). Granite μP-scaled MoE: similar.
* **`norm_topk_prob`.** True (Qwen3-30B-A3B): softmax over the
  *chosen* top-k experts. False (OLMoE): raw router logits used as
  weights. Different inductive bias.

Granite's μP scalars (`embedding_multiplier`,
`attention_multiplier`, `logits_scaling`, `residual_multiplier`) and
MiniCPM-3's `scale_emb=12`, `scale_depth=1.4`, `dim_model_base=256`
multipliers attach to the channel-mixer path at specific points and
are *invisible* in a Llama-style API. The IR must expose them as
optional scalars or silently produce wrong outputs.

The API axes: `ffn.kind ∈ {swiglu, swiglu_fused, geglu, gegelu,
gated_gelu_tanh, gelu, silu, rwkv_channel_mix}`, `ffn.bias`,
`moe.{num_experts, num_experts_per_tok, num_shared_experts,
first_k_dense_replace, norm_topk_prob, router_activation}`, plus the
scalar multipliers `embedding_multiplier`, `residual_multiplier`,
`logits_scaling`, `attention_multiplier`, `query_pre_attn_scalar`,
`attn_logit_softcapping`, `final_logit_softcapping`.

### 4.5 Vocab and tokenizer

**Arc:** SP-BPE 32K → 50K-class BPE → Tiktoken cl100k → Llama-3
tiktoken 128K → Qwen 152K → o200k 200K.

* **SentencePiece BPE 32K (LLaMA 1 2023-02; TinyLlama; Mistral
  7B v0.1).** The original LLaMA convention; Llama-2 retains.
* **GPT-2 BPE 50K (BLOOM, Falcon, Pythia, OLMoE, Mamba).** Classic
  BPE with newline-separator tokens.
* **Qwen-BPE 151936 (Qwen 2 / 2.5 / 3 2024-06+).** Tiktoken-style
  byte-fallback BPE, extended for Chinese and code tokens.
* **Tiktoken cl100k (~100K) (Phi-3-small, OLMo 2, StableLM-2).**
  OpenAI's cl100k_base, reused.
* **Llama-3 tiktoken 128K (Llama-3 2024-04 → SmolLM3 2025-07 → many
  forks).** 128 256 tokens, BPE on tiktoken regex.
* **Gemma SP 256K-262K (Gemma 1/2/3).** Largest BPE vocabulary in
  any mainstream SLM; v3 expands to 262 144 with image tokens.
* **o200k 200K (Phi-4-mini 2025-02).** OpenAI's o200k_base; the
  largest non-Gemma tiktoken-class vocab.
* **Custom small (49K) (SmolLM2, Granite 3.x, StarCoder 2).**
  Compact BPE for code-friendly tokenization.

**Embedding tying.** Below ~3B, the embedding table and `lm_head`
typically share weights (`tie_word_embeddings=True`). At 7B+ they
are usually decoupled. SmolLM3 (3B, 2025-07) decoupled them at 3B
arguing for quality at the embed-decode head. This is now a per-size
decision; the IR must support both.

**Tokenizer special-token expansion.** Phi-3.5-vision, Gemma 3, and
the Llama-4 multimodal track add image / audio / control tokens
without changing the *embedding-table architecture* — only adding
rows. The IR has to permit cross-modal indices in the same lookup.

### 4.6 Quantization

**Arc:** Q4_0/Q4_1 → k-quants → AWQ/GPTQ → FP8 → MXFP4/NVFP4 → BitNet
ternary → 2-bit AQLM/IQ2.

* **Q4_0 / Q4_1 (llama.cpp 2023-03).** Block-wise int4 with one fp16
  scale (Q4_0) or scale+offset (Q4_1) per 32-element block. The
  first ggml formats. Replaced by k-quants in 2024.
* **k-quants Q4_K_M, Q5_K_M, Q3_K_L (llama.cpp 2024).** Per-block
  super-blocks: 256-element super-block with two-level scales (block
  scale + super-block delta). Better quality at the same bit-rate.
  Standard for llama.cpp GGUF in 2024–26.
* **AWQ (Lin et al. 2023; MLSys 2024).** Activation-aware int4
  W4A16 with fp16 scales and fp16 zero, group=128 along K, calibrated
  by activation magnitudes. Shipped in vLLM, TensorRT-LLM, AutoAWQ.
* **GPTQ (Frantar et al. 2023).** Hessian-aware int4 with optional
  `desc_act` reordering. Shipped in AutoGPTQ, vLLM, TensorRT-LLM.
* **FP8 (E4M3 / E5M2; H100 2022; production 2024).** Hardware-native
  in Hopper. DeepSeek-V3 (2024-12) was the first widely-served
  flagship trained end-to-end in FP8 (mixed precision).
* **MXFP4 / MXFP6 / MXFP8 (OCP Microscaling 2023-09; Blackwell native
  2024).** 32-element shared-exponent (UE8M0) blocks. GPT-OSS (2025-08)
  ships weights in MXFP4 for the MoE projections — the first widely
  served MXFP4 model.
* **NVFP4 (Blackwell 2025).** NVIDIA-specific 4-bit FP variant with
  hierarchical scales. Used by TensorRT-LLM on Blackwell.
* **BitNet b1.58 (Microsoft 2024-02 → 2025-04 2B model).** Ternary
  `{-1, 0, +1}` weights trained from scratch; activations int8.
  Productionized at 2B / 4T tokens in April 2025.
* **AQLM (Egiazarian et al. 2024) / IQ2 (llama.cpp 2024).** 2-bit
  weight quantization with codebooks. AQLM-2bit-128g and IQ2_XXS
  show production-quality for 70B-class at extreme compression.

**Abandoned:** Q4_0 / Q4_1 GGUF (replaced by k-quants for quality);
INT8 W8A8 dynamic (replaced by FP8 W8A8 on Hopper+); LLM.int8()
column-mixed (kept only in bitsandbytes for legacy 7B PTQ).

See `04-quantization` for the full 25-scheme taxonomy and the
15-axis parameter space.

### 4.7 KV cache

**Arc:** contiguous → PagedAttention → quantized KV → MLA-compressed
→ YOCO → vAttention → NSA tiered → MTP heads.

* **Contiguous (HF baseline, 2022).** Per layer, K and V are separate
  tensors of shape `[B, n_kv_heads, T_max, d_h]`, either
  preallocated or grown via `torch.cat`. Simple and easy to
  serialize; wastes memory at low batch occupancy.
* **PagedAttention (Kwon et al. 2023; vLLM).** KV cache stored as
  16- or 32-token "blocks" addressed via a page table. Block size
  is a kernel-template parameter (`block_size ∈ {16, 32, 64}`).
  Enables variable-length batching without preallocation. Standard
  in vLLM, SGLang, TensorRT-LLM, OpenVINO PagedAttention extension.
* **Quantized KV (KIVI 2024; OpenVINO `KV_CACHE_PRECISION=u8/u4`).**
  Store K and V in int8 or int4 with per-token or per-channel
  scales. Quality preserved at int8; some regression at int4.
* **MLA-compressed (DeepSeek-V2 2024-05).** Discussed in §4.1.
  Cache stores latent `c_kv` + rope channel only.
* **YOCO (You Only Cache Once; Sun et al. 2024).** Use the same KV
  cache across a contiguous group of layers; only the producer
  layer writes. Reduces cache by integer factor. Apple's
  on-device model uses cross-layer KV sharing.
* **Cross-Layer Attention / CLA (Brandon et al. 2024; MiniCPM-3).**
  Same idea: factor-2 sharing between layer pairs.
* **vAttention (2024).** Uses Linux process-VM tricks to give
  contiguous virtual address ranges to logically paged KV. Briefly
  shipped; superseded by PagedAttention v2 with native paging in
  TensorRT-LLM.
* **NSA tiered cache (DeepSeek 2025-02).** Native sparse attention
  with hardware-aligned block selection; each layer's effective
  cache is a fraction of the full T. Not yet in a shipped flagship.
* **MTP heads (DeepSeek-V3 2024-12).** Multi-token prediction during
  training; at inference, the auxiliary heads can run speculative
  decoding for ~1.8× throughput. The MTP heads are extra
  architecture, not cache changes, but they affect the inference
  layer.

**Abandoned:** vAttention's process-VM tricks (replaced by paged v2);
"grow via `torch.cat`" legacy DynamicCache (replaced by preallocated
StaticCache).

### 4.8 Training-time architecture innovations affecting inference

These are not block-level features but they alter what the inference
layer must look like:

* **Rotary scaling-aware training** (LongRoPE, YaRN). The model is
  trained with the specific scaling that will be used at
  inference; runtimes must serialize the exact scaling parameters.
* **Attention sinks trained in (GPT-OSS 2025-08; Mistral Small 3.1
  2025-11).** The model is trained to allocate attention to a
  small number of "sink" positions; at inference, the runtime
  must reserve those positions in the KV cache. OpenVINO's
  PagedAttentionExtension exposes a `sinks` input; vLLM exposes a
  `attention_sink_size` parameter.
* **MTP heads (DeepSeek-V3 2024-12).** Trained heads predict
  `t+1, t+2, ..., t+k`; at inference they can drive speculative
  decoding without a separate draft model.
* **R1-style reasoning traces.** The R1 / R1-Distill family is *not*
  a layer-architecture change; the distillates inherit Qwen-2.5 or
  Llama-3.1 blocks unchanged. The runtime sees the same forward
  pass; only the sampler and the chat template change.

---

## 5. Abandoned designs (the survey-paper graveyard)

Each entry identifies the design, names representative shipments,
and explains *why* it was abandoned.

* **Parallel attention/FFN (GPT-J 2022; Falcon 7B 2023; Pythia 2023).**
  Compute attention and FFN in parallel from the same residual
  input, then add both. Saves one layernorm and (in theory) helps
  parallelism. *Abandoned by Llama 2 onward.* Empirical regressions
  at >7B (Falcon's own Llama-2-comparison paper, EleutherAI's Pythia
  follow-up). Llama-3 and all current SLMs use serial
  attention-then-FFN.
* **Pure MQA (PaLM 2022; Falcon 7B/40B 2023; CodeGemma 2B 2024).**
  `n_kv_heads = 1`. *Abandoned because* the KV cache savings turn
  out to be unnecessary at the GQA-8 trade-off point. Mistral 7B's
  GQA-8 ships nearly the same KV savings (8× vs 32×) with
  consistently better quality. CodeGemma 2B is the only mainstream
  flagship of 2024 still on MQA; CodeGemma 7B and Gemma 2 onward
  use GQA.
* **Pure sliding-window everywhere (Mistral 7B v0.1, 2023-09).**
  Every layer applies SWA with W=4096. *Abandoned because* Gemma 3
  ablations show strict-better quality with 5:1 alternation
  (5 sliding + 1 full) at the same average compute. Mistral 7B
  v0.3 dropped SWA entirely; Gemma 3 standardized 5:1 alternation;
  GPT-OSS uses similar local/global alternation. Pure SWA persists
  only in Mistral v0.1 legacy weights.
* **ALiBi (MPT 2023; BLOOM 2022; Falcon 40B 2023; Baichuan-1
  2023).** *Abandoned because* RoPE + per-family scaling (PI /
  NTK / YaRN / Llama-3 smooth / LongRoPE) is more expressive and
  no model trained with ALiBi has matched RoPE-equivalents at
  >32K. ALiBi survives only in legacy MPT and BLOOM weights.
* **Absolute / learned positional embeddings (GPT-2, BLOOM, OPT, Phi-2's
  partial-RoPE pass-through).** *Abandoned* outside encoder-decoder
  niches (Florence-2 BART, T5). Pure absolute PE does not
  extrapolate; learned-relative PE was hard to scale.
* **NTK-aware-only RoPE (Together LLongMA 2023; CodeLlama 2023).**
  *Replaced by* YaRN (2023-09) and Llama-3 smooth scaling
  (2024-07), both of which produce better long-context
  perplexity at the same training cost.
* **Q4_0 / Q4_1 GGUF formats (llama.cpp 2023).** *Replaced by*
  k-quants (Q4_K_M, Q3_K_L). Same bit-rate, better perplexity.
  Q4_0 still ships as a fallback but is no longer the default for
  any modern model.
* **Pure post-LN for training stability (BERT, ViT, T5, OLMo 2 2024-11
  revival).** OLMo 2 reports better fp32 training stability with
  post-norm, but bf16 inference is unchanged and pre-norm models
  catch up with longer training. Post-LN is now a minority
  position; perhaps 2 published flagships in 2026.
* **INT8 W8A8 dynamic (transition 2023-24).** Per-token activation
  quant + per-channel weight quant. *Replaced by* FP8 W8A8 on
  Hopper (no calibration needed; better dynamic range).
  bitsandbytes' LLM.int8() persists in legacy 7B inference but is
  no longer the default.
* **vAttention's process-VM tricks (2024).** Used Linux VMA
  manipulation to give contiguous virtual address spaces to logical
  paged KV. *Superseded by* PagedAttention v2 (native page-table
  paging in TensorRT-LLM and vLLM); the process-VM approach added
  kernel-version dependency without quality benefit.
* **Pre-pretrained absolute / learned position
  embeddings as long-context recipe (Llama-2-Long 32K via PI
  fine-tune).** *Replaced by* the smooth schedules in Llama-3.1,
  which can be applied to the base model without fine-tuning.

A consistent theme: every abandoned design was replaced by a
*more parameterized* alternative whose extra parameters were freely
tunable. The graveyard is not a graveyard of bad ideas; it is a
graveyard of insufficiently-expressive predecessors.

---

## 6. Consensus vs Divergence (the 2026 snapshot)

### 6.1 Consensus 2026 (≥90% of mainstream SLMs)

The current "default" block, supported by every dense SLM shipped
since Llama 3 and consumed by every runtime surveyed in
`03-ihv-opsets`:

* **Decoder-only, causal, pre-norm** (with `post_attention_layernorm`
  and `post_feedforward_layernorm` as paired residual norms on the
  *next* sublayer's input — i.e. the "pre-norm" used in practice
  has two norms per sublayer block).
* **RMSNorm** (LayerNorm only in Phi-3-small, StarCoder 2, RWKV
  hold-outs).
* **RoPE** (with some `rope_scaling` for >32K context); θ ∈
  {500K, 1M, 5M, 1e7}.
* **SwiGLU or GeGLU** in FFN (SwiGLU dominates; GeGLU only in
  Gemma + CodeGemma + PaliGemma).
* **GQA** (kv_heads = q_heads // N, N ∈ {2, 4, 7, 8}). MHA persists
  only in Phi-3-mini/medium (≤14B sizes that pre-date the
  full-GQA Phi-4-mini), OLMo 2, SmolLM2-1.7B, StableLM 2-1.6B.
  Even those are converging.
* **Embedding** tied at ≤3B, decoupled at 7B+ (except SmolLM3 3B
  decoupling at 3B; Granite ties at 2B and 8B both).
* **Tiktoken-class BPE vocab** 128K-200K. Gemma stays on
  SentencePiece 256K-262K as the only flagship hold-out.
* **FP16/BF16 native; FP8 or INT4 quantized variants shipped.**
  Every flagship since Llama 3 ships both BF16 and an int4 GGUF /
  AWQ / GPTQ variant.

### 6.2 Divergence (what families fight over)

These axes have at least two production variants in active 2026 use:

* **Sliding window:**
  * *Yes, alternating:* Gemma 3 (5:1), Gemma 2 (1:1), GPT-OSS, some
    Mistral Small.
  * *Yes, per-layer alternation:* SmolLM3 (interleaved).
  * *No:* Llama 3.x, Qwen 3 ≤ 8B, Mistral 7B v0.3, Phi-4 14B.
  * The disagreement is *empirical*: Gemma 3's ablations show 5:1
    helps; Llama-3 / Qwen-3's data does not. No cross-org reproduction
    has been published.
* **MoE:**
  * *Always on (frontier):* DeepSeek-V3, Llama-4 (all sizes),
    Mixtral 8x22B, Qwen3-30B-A3B.
  * *Always off (SLM):* Llama 3.2, Qwen 3 ≤ 14B, Gemma 3, Phi-4.
  * *Niche on (sub-7B active):* OLMoE 1B-7B, DeepSeek-V2-Lite,
    Jamba.
* **QK-norm:**
  * *None:* Llama, Mistral, Phi (all variants except
    micromodels), Granite.
  * *Per-head_dim, before RoPE:* Qwen 3 family.
  * *Per-head_dim, after RoPE:* Gemma 3.
  * *Full-channel γ, before RoPE:* OLMo 2.
  * *Per-head_dim Q-only with QKV bias:* StableLM 2 12B.
  Three families, three formulations.
* **Soft-cap on logits:** Gemma 2 only (cap=50 attn, cap=30 LM head);
  Gemma 3 *removed* it. Effectively a graveyard entry already.
* **Hybrid SSM/attention:**
  * *Mainstream attention-only:* everything else.
  * *Hybrid niche:* Jamba (1-in-8 attn + Mamba + MoE),
    Phi-4-mini-flash (SambaY), Hymba (parallel heads), Zamba 2
    (periodic shared attn), Falcon-H1.
* **Long-context recipe:**
  * *Llama-3 smooth (factor=8):* Llama 3.x, R1-Distill-Llama,
    SmolLM3.
  * *YaRN:* DeepSeek-V2-Lite, Qwen2.5-1M.
  * *LongRoPE:* Phi-3.5 / Phi-4 / MiniCPM-3.
  * *Dynamic NTK:* InternLM 2.5 / 3.
  * *Linear scaling:* Gemma 3 (factor=8).
  * *Dual per-layer θ:* Gemma 3.
  Five active recipes; no single winner.
* **MLA:** DeepSeek family + MiniCPM-3 only. Adoption barrier:
  matrix-absorption trick requires retraining; no fine-tune
  conversion published from GQA to MLA.
* **Quantization default:**
  * *INT4 weight-only (AWQ/GPTQ/Q4_K_M):* Llama, Qwen, Mistral,
    Gemma, Phi.
  * *FP8 (W8A8):* DeepSeek-V3, frontier training defaults.
  * *MXFP4 (MoE projections):* GPT-OSS.
  * *Ternary (BitNet):* BitNet 2B4T.
* **NoPE alternation:** SmolLM3 + Llama-4 iRoPE adopt it; nobody
  else.
* **Cross-layer KV sharing:** Apple on-device, MiniCPM-3 (CLA),
  YOCO reference. No widespread adoption.
* **Trained-in attention sinks:** GPT-OSS (Aug 2025), Mistral Small
  3.1 (Nov 2025). Two families and counting.

---

## 7. Open research directions (forward-looking)

These directions are active as of 2026-Q1 but have not yet
homogenized into a "consensus 2026" line. The IR should reserve
extension points for each.

* **2-bit weight quantization productionization (AQLM, IQ2_M).**
  Codebook-based 2-bit weights with ≤1% perplexity regression at
  ≥7B. Production blockers are kernel availability on
  consumer GPUs and CPU dequant overhead. Likely consumer-mainline
  in 2026 H2.
* **Sub-linear-context attention (NSA, YOCO, vAttention v2).** Native
  sparse attention trained in; runtime kernel maturity is the
  blocker. DeepSeek's NSA paper is the canonical reference. Once
  in flagships, this collapses long-context cost by an order of
  magnitude.
* **SSM/attention hybrids reaching parity at <8B.** Hymba 1.5B,
  Phi-4-mini-flash 3.8B, Zamba 2 7B already match dense-attention
  baselines at training-compute parity. The 2026 question is
  whether a >7B hybrid beats Qwen 3 8B or Llama 3.2 7B in
  apples-to-apples training.
* **BitNet-style 1.58-bit weights at scale.** The 2B/4T BitNet
  showed perplexity parity; a 7B / 70B BitNet is the open
  question. If it works, runtime impact is enormous (no fp16
  weight unpacking).
* **Differential / multi-stream attention (Ye et al. 2024;
  Phi-4-mini-flash).** Subtracting two softmax maps reduces
  attention noise; production adoption is just starting.
* **MTP / speculative-decoding-co-designed architectures.** DeepSeek-V3
  showed MTP heads enable speculative decoding without a separate
  draft model. Likely standard in 2026 H2 large models.
* **Per-layer scaling (OpenELM).** Non-uniform depth / width
  schedules gain traction in on-device settings where compute
  budget is fixed. Apple's on-device line continues this.
* **iRoPE / NoPE alternation generalization.** Llama-4's iRoPE
  pattern (interleave NoPE layers with RoPE layers + temperature
  scaling) is the latest entrant in the long-context recipe
  family. Expect a "v0.2" with parameterized interleave ratio.
* **PT-MoE on-device (Apple Foundation Model 2026).**
  Parallel-Track MoE with track parallelism + local/global
  attention is the production frontier for on-device inference.
  Likely template for Android equivalents.

---

## 8. Cross-org comparison table

The 12 mainstream families today, summarized along the axes the
API must parametrize. (Per-size variation collapsed; see
`01-model-census` for the full enumeration.)

| Family (2026) | Attn kind | RoPE variant | Norm | FFN | MoE | Vocab (tok) | Quant default | KV cache | Long-context recipe |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3.x / 3.2 | GQA (kv=8) | Llama-3 smooth (θ=500K) | RMS pre | SwiGLU | no | 128K LL3 | bf16 + Q4_K_M | contiguous / paged | Llama-3 smooth, factor=8 |
| Llama 4 (Scout/Maverick) | GQA + iRoPE (NoPE alt) | RoPE + NoPE interleave | RMS pre | SwiGLU-MoE | yes (16 or 128 experts) | 200K LL4 | bf16 + MXFP4 | paged + sparse | iRoPE, 10M ctx |
| Qwen 3 (0.6–32B + 30B-A3B) | GQA + **QK-RMSNorm(head_dim)** | vanilla (θ=5M small; 1e6 8B+) | RMS pre + qk-norm | SwiGLU / SwiGLU-MoE | optional | 152K Qwen | bf16 + AWQ / GPTQ | contiguous / paged | YaRN (1M); RoPE base 5M |
| Mistral 7B v0.3 / Nemo / Small 3.1 | GQA | vanilla (θ=1e6) | RMS pre | SwiGLU | (no for dense; yes for Mixtral) | 32K SP / Tekken | bf16 + Q4_K_M | paged + sinks (3.1) | Llama-3-like; sinks |
| Phi-4 / Phi-4-mini / Phi-4-mini-flash | MHA (Phi-4) / GQA + partial RoPE 0.75 (mini) / SambaY hybrid (flash) | LongRoPE | RMS pre | SwiGLU fused gate-up | (no) | 200K o200k | bf16 + GPTQ | contiguous; Mamba state (flash) | LongRoPE per-dim |
| Gemma 3 (1/4/12/27B) | GQA + **5:1 SWA:full** + QK-norm | dual per-layer θ (10K local, 1M global) + linear scale | RMS pre+post (dual norm) | GeGLU | (no) | 262K Gemma SP | bf16 + Q4_K_M | paged per-layer-type | linear scaling factor=8 |
| DeepSeek-V3 + R1 | **MLA** + sigmoid-routing MoE | YaRN | RMS pre | SwiGLU-MoE (256 routed + 1 shared) | yes | 100K BPE | FP8 + MXFP4 (MoE) | MLA latent | YaRN factor=40 |
| DeepSeek-R1-Distill (-Qwen / -Llama) | GQA (inherits backbone) | inherits | inherits | inherits | inherits | inherits | bf16 + Q4_K_M | inherits | inherits |
| OLMo 2 | MHA + **post-norm + QK-norm full-channel** | vanilla (θ=5e5) | RMS post + qk-norm | SwiGLU | (no) | 100K cl100k | bf16 + GGUF | contiguous | none (4K ctx) |
| Granite 3.x | GQA + **μP multipliers** | vanilla (θ=5e6 → 1e7) | RMS pre | SwiGLU | (no for 2/8B) | 49K StarCoder | bf16 + Q4_K_M | contiguous | (none / 128K via large θ) |
| SmolLM3 3B | GQA + **NoPE every 4th layer** + SWA alternation | vanilla (θ=5e6); NoPE | RMS pre | SwiGLU | (no) | 128K LL3 | bf16 + Q4_K_M | paged per-layer-type | base θ growth + NoPE |
| GPT-OSS 20B / 120B | GQA + **SWA + trained-in attention sinks** | vanilla | RMS pre | SwiGLU-MoE (MXFP4 weights) | yes (20B and 120B both MoE) | OAI tiktoken | BF16 + MXFP4 | paged + sinks | (32K ctx) |
| Apple Foundation Model on-device | GQA + **cross-layer KV sharing** | vanilla | RMS pre | SwiGLU | (no for on-device; PT-MoE server) | Apple SP | 2-bit QAT | shared paged | (32K ctx) |

The same dataset, by axis dominance: GQA (10/12 families), RMS
pre-norm (10/12), SwiGLU (10/12), tiktoken-class vocab (9/12). The
non-conforming entries are exactly the families that drive most of
§4's "interesting" content.

---

## 9. Future-proofing the llm-layers IR

Mapping §4-§8 onto the IR design:

**Active axes (need first-class parameters):**

* `attention.kind` — must support `{mha, gqa, mqa, mla, mamba,
  mamba2, samba, rwkv, differential, retention, identity}`.
* `attention.kv_heads`, `q_heads`, `head_dim`, plus MLA's
  `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`,
  `qk_rope_head_dim`, `v_head_dim`.
* `attention.sliding_window` (int or None), `sliding_window_pattern`,
  `layer_types[]`.
* `attention.softcap` (float or None), `query_pre_attn_scalar`.
* `attention.attention_multiplier` (μP).
* `attention.sinks` (int — trained-in sink count).
* `qk_norm.kind ∈ {none, per_head_dim, per_full_channels}`,
  `qk_norm.placement ∈ {pre_rope, post_rope}`,
  `qk_norm.norm_type ∈ {rmsnorm, layernorm}`.
* `rope.kind`, `rope.theta`, `rope.partial`, `rope.basis`,
  `rope.scaling` (a sum-type covering llama3 / yarn / longrope /
  dynamic / linear).
* `rope.per_layer_theta` and `per_layer_kind` (Gemma 3).
* `norm.type`, `norm.mode (classic | gemma_1plus_w)`, four norm
  toggles per sublayer.
* `ffn.kind` and `ffn.bias`; MoE-block parameters
  `num_experts`, `num_experts_per_tok`, `num_shared_experts`,
  `first_k_dense_replace`, `norm_topk_prob`, `router_activation`.
* `quant` — first-class on every linear; supports all of int4-{AWQ,
  GPTQ, HQQ, k-quants}, FP8, MXFP4/8, NF4, ternary.
* `kv_cache` layout — contiguous / paged / MLA-latent / SSM-state /
  shared-across-layers (YOCO/CLA).
* `embedding_multiplier`, `residual_multiplier`, `logits_scaling`
  (μP scalars).

**Frozen axes (one canonical choice):**

* Decoder-only causal mask (no encoder/decoder.attention-cross-attention support
  needed for SLM scope).
* RMSNorm `eps` (float; convention varies but not a structural
  knob).
* Linear "bias" (boolean per sublayer; default off).

**Graveyard (explicitly unsupported):**

* Parallel attention/FFN — refuse to instantiate; force serial.
* ALiBi — supportable for legacy MPT compatibility but not a
  first-class option in `rope.kind`.
* Pure MQA at >2B — supported (no harm) but not a model default.
* INT8 W8A8 dynamic — not a `quant.kind` option; use FP8 instead.
* Process-VM-paged KV — out of scope; IR exposes only the cache
  layout, the runtime handles backing.
* Pure post-LN as the *only* norm strategy — supported via
  norm-toggle flags but tagged "minority position" in docs.

**Reserved extension points (for §7 directions):**

* `attention.kind = nsa` (Native Sparse Attention) with branches
  for `compressed / selected / sliding`.
* `attention.kind = differential` with `n_q_splits` and per-layer λ.
* `ffn.scalars.PT_MoE` for Apple's parallel-track MoE.
* `quant.kind = bitnet_b1_58` with proper {-1, 0, +1} weight
  dtype, 8-bit activations.
* `kv_cache.layout = shared_paged` for cross-layer KV sharing.
* `mtp_heads.k` for multi-token prediction head count (training-side;
  inference uses for spec decoding).

The principle: every axis in this taxonomy that is *currently
contested* (§6.2) must be exposed; every axis that is *currently
consensus* (§6.1) must have a canonical default but remain editable
for future drift; every axis in the *graveyard* (§5) must be either
unsupported or hidden behind explicit legacy flags.

---

## 10. References

### Models and reports

* GPT-3.5 / ChatGPT release announcement, OpenAI, 2022-11-30.
* LLaMA: Open and Efficient Foundation Language Models, Touvron et
  al., arXiv:2302.13971, 2023-02.
* Llama 2: Open Foundation and Fine-Tuned Chat Models, Touvron et
  al., arXiv:2307.09288, 2023-07-18.
* Mistral 7B, Jiang et al., arXiv:2310.06825, 2023-09-27.
  https://mistral.ai/news/announcing-mistral-7b/
* Mixtral of Experts, Jiang et al., arXiv:2401.04088, 2023-12.
* Llama 3 Herd of Models, Meta, 2024-04-18.
  https://ai.meta.com/blog/meta-llama-3/
* Llama 3.1 / 3.2 release notes, Meta, 2024-07 / 2024-09.
* Llama 4 release: Scout 17B-16E / Maverick 17B-128E,
  https://huggingface.co/blog/llama4-release, 2025-04-05.
* DeepSeek-V2: A Strong, Economical, and Efficient MoE Language
  Model, Liu et al., arXiv:2405.04434, 2024-05.
* DeepSeek-V3 Technical Report, arXiv:2412.19437, 2024-12-26.
* DeepSeek-R1: Incentivizing Reasoning Capability in LLMs,
  arXiv:2501.12948, 2025-01.
* Native Sparse Attention: Hardware-Aligned and Natively Trainable
  Sparse Attention, DeepSeek + PKU + UW, arXiv:2502.11089, 2025-02.
  ACL 2025 Best Paper.
* Gemma: Open Models Based on Gemini Research and Technology,
  Mesnard et al., 2024-02.
* Gemma 2 Technical Report, Riviere et al., 2024-07.
* Gemma 3 Technical Report, arXiv:2503.19786, 2025-03-12.
* Qwen Technical Report (Qwen 1), Bai et al., arXiv:2309.16609,
  2023-09.
* Qwen2 Technical Report, arXiv:2407.10671, 2024-06.
* Qwen2.5 Technical Report, arXiv:2412.15115, 2024-09.
* Qwen3 Technical Report, arXiv:2505.09388, 2025-05 (release
  2025-04-29). https://github.com/QwenLM/Qwen3
* Phi-3 Technical Report, Abdin et al., arXiv:2404.14219, 2024-04.
* Phi-3.5 model card (mini / MoE / vision), Microsoft 2024-08.
* Phi-4 Technical Report, Microsoft 2024-12.
  https://huggingface.co/microsoft/phi-4
* Phi-4-mini-flash-reasoning,
  https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning,
  2025-07.
* Samba: Simple Hybrid State Space Models for Efficient Unlimited
  Context Language Modeling, Ren et al., ICLR 2025.
  https://github.com/microsoft/Samba
* OLMo: Accelerating the Science of Language Models, AI2 / Groeneveld
  et al., arXiv:2402.00838, 2024-02.
* OLMo 2 (release blog), AI2, 2024-11-26.
  https://allenai.org/blog/olmo2
* OLMoE: Open Mixture-of-Experts Language Models, Muennighoff et al.,
  arXiv:2409.02060, 2024-09.
* SmolLM2, HuggingFace, 2024-11.
* SmolLM3, HuggingFace, 2025-07.
* Granite 3.0 / 3.1 / 3.3 Technical Report, IBM Research, 2024-10
  → 2025-04.
* MiniCPM-3 Technical Report (CLA cross-layer KV + MLA), OpenBMB,
  2024-09.
* StableLM 2: Stability AI, 2024-01 / 04.
* StarCoder 2 and The Stack v2, BigCode, arXiv:2402.19173, 2024-02.
* Falcon LLM: Almazrouei et al., 2023.
* Falcon 2 / Falcon 3 / Falcon-Mamba, TII, 2024.
* Mamba: Linear-Time Sequence Modeling with Selective State Spaces,
  Gu & Dao, arXiv:2312.00752, 2023-12.
* Mamba-2: Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality, Dao & Gu,
  arXiv:2405.21060, 2024-05.
* Jamba: A Hybrid Transformer-Mamba Language Model, Lieber et al.,
  arXiv:2403.19887, 2024-04.
* Zamba: A Compact 7B SSM Hybrid Model, Glorioso et al., 2024.
* Hymba: A Hybrid-head Architecture for Small Language Models,
  arXiv:2411.13676, 2024-11-22.
* RWKV-7 Technical Report, BlinkDL, 2025-10.
* GPT-OSS Model Card, OpenAI, 2025-08-05.
  https://huggingface.co/openai/gpt-oss-20b,
  https://openai.com/index/introducing-gpt-oss/
* BitNet: Scaling 1-bit Transformers, Wang et al., arXiv:2310.11453,
  2023-10.
* The Era of 1-bit LLMs: All LLMs are in 1.58 Bits, Ma et al.,
  arXiv:2402.17764, 2024-02-27.
* BitNet b1.58 2B4T Technical Report, arXiv:2504.12285, 2025-04.
* Apple Foundation Models, ML Research, 2024-06.
  https://machinelearning.apple.com/research/introducing-apple-foundation-models
* Apple Foundation Models Tech Report 2025, ML Research, 2025-08.
  https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025

### Architecture techniques

* RoFormer: Enhanced Transformer with Rotary Position Embedding,
  Su et al., arXiv:2104.09864, 2021.
* Train Short, Test Long: Attention with Linear Biases Enables Input
  Length Extrapolation, Press et al., arXiv:2108.12409, 2021.
* GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints, Ainslie et al., arXiv:2305.13245, 2023.
* Fast Transformer Decoding: One Write-Head is All You Need,
  Shazeer, arXiv:1911.02150, 2019 (MQA).
* RMSNorm: Root Mean Square Layer Normalization, Zhang & Sennrich,
  arXiv:1910.07467, 2019.
* GLU Variants Improve Transformer, Shazeer, arXiv:2002.05202,
  2020 (SwiGLU / GeGLU).
* On Layer Normalization in the Transformer Architecture, Xiong et
  al., 2020 (pre-LN vs post-LN).
* DeepNorm: Scaling Transformer to 1000 Layers, Wang et al.,
  arXiv:2203.00555, 2022.
* PaLM: Scaling Language Modeling with Pathways, Chowdhery et al.,
  2022 (parallel attn/FFN, MQA).
* Switch Transformers / GShard / Mixture-of-Experts Layer,
  Fedus et al., arXiv:2101.03961.
* StreamingLLM: Xiao et al., Efficient Streaming Language Models
  with Attention Sinks, arXiv:2309.17453, 2023-09.
* YaRN: Efficient Context Window Extension of LLMs, Peng et al.,
  arXiv:2309.00071, 2023.
* LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens,
  Ding et al., arXiv:2402.13753, 2024-02 (Phi-3 long).
* NTK-aware scaling (bloc97 Reddit post + Together LLongMA, 2023).
* Differential Transformer, Ye et al., arXiv:2410.05258, 2024.
* RetNet: Retentive Network, Sun et al., arXiv:2307.08621, 2023.
* Cross-Layer Attention (CLA), Brandon et al., arXiv:2405.12981,
  2024.
* YOCO: You Only Cache Once, Sun et al., arXiv:2405.05254, 2024.
* μP / Tensor Programs V, Yang & Hu, 2022 (Granite/MiniCPM scalars).

### Quantization / runtime / kernel

* GPTQ: Frantar et al., arXiv:2210.17323, 2023-ICLR.
* AWQ: Lin et al., arXiv:2306.00978, MLSys 2024.
* HQQ: Badri & Shaji, mobiusml/hqq, 2023.
* QLoRA / NF4: Dettmers et al., arXiv:2305.14314, 2023.
* LLM.int8(): Dettmers et al., arXiv:2208.07339, 2022.
* OCP Microscaling Formats v1.0, 2023-09.
* MX Data Formats for Deep Learning, Rouhani et al.,
  arXiv:2310.10537, 2023.
* AQLM: 2-bit weight compression with multi-codebook quantization,
  Egiazarian et al., arXiv:2401.06118, 2024.
* PagedAttention / vLLM: Kwon et al., arXiv:2309.06180, 2023.
* FlashAttention-2, FlashAttention-3: Dao et al., arXiv:2307.08691,
  arXiv:2407.08608.
* KIVI: Plug-and-Play 2-bit KV Cache Quantization, Liu et al.,
  arXiv:2402.02750, 2024.

### Runtime / IHV

* OpenVINO 2025.x operation sets,
  https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets.html
* OpenVINO PagedAttentionExtension,
  https://github.com/openvinotoolkit/openvino/blob/master/src/core/include/openvino/op/paged_attention.hpp
* Qualcomm QAIRT / QNN OpDefs (QnnOpDef.h),
  https://docs.qualcomm.com/doc/80-63442-10/topic/api-rst_program_listing_file_include_QNN_QnnOpDef_h.html
* vLLM model_executor,
  https://github.com/vllm-project/vllm/tree/main/vllm/model_executor
* llama.cpp `src/models/*.cpp`,
  https://github.com/ggml-org/llama.cpp
* TensorRT-LLM, NVIDIA,
  https://github.com/NVIDIA/TensorRT-LLM
* HuggingFace `transformers`, local checkout at
  `C:\Users\zhengte\external\transformers\`

### Companion reports in this project

* `research/01-model-census.md` — 45 models / 22 architectures / 11 axes.
* `research/02-layer-sources.md` — 10 families, HF / vLLM / llama.cpp.
* `research/03-ihv-opsets.md` — 9 runtimes synthesis.
* `research/04-quantization.md` — 25 schemes / 15 axes.
* `research/05-kvcache-attention.md` — 13 attn × 8 RoPE × 6 cache.
* `docs/superpowers/specs/2026-06-04-llm-layers-design.md` — design spec.

### Contradictions noted vs the existing v1 reports

While drafting this document I cross-checked against the five
companion reports. The following minor inconsistencies were found
and should be reconciled in a v2 sweep:

1. **Gemma 3 sliding window size.** `01-model-census` table (§2.3)
   records Gemma 3 1B SWA as `512` and Gemma 3 4B SWA as `1024`.
   `05-kvcache-attention` §1.2 records "W=1024 local" without
   per-size differentiation. Both can be true (1B uses 512, 4B
   uses 1024) but the survey text should pick one or split. This
   document reports the Gemma-3 family default as 1024 per the
   tech report; the 1B-specific 512 is noted in §2.
2. **Gemma 3 QK-norm placement.** `02-layer-sources` §3a says
   Gemma 3 applies q_norm / k_norm "before attention" without
   committing to before/after RoPE; `05-kvcache-attention` §1.7
   says "after RoPE" for Gemma 3 alongside Qwen 3. This document
   adopts the §1.7 reading (post-RoPE for Gemma 3) but the
   Gemma-3 HF source (`modeling_gemma3.py` lines 337-345) should
   be re-inspected to confirm — the order in `eager_attention_forward`
   appears to be `q_norm → RoPE` (i.e. pre-RoPE) in the
   reference impl. If pre-RoPE is correct, §6.2 row for QK-norm
   should be updated and `05-kvcache-attention` §1.7 fixed.
3. **GPT-OSS release date.** This survey records Aug 5, 2025
   per the OpenAI model card; this is consistent with all
   companion reports.
4. **Phi-3-mini partial RoPE factor.** `02-layer-sources` §4a
   mentions "partial_rotary_factor = 0.5"; `01-model-census` does
   not list partial RoPE for Phi-3-mini, only for Phi-4-mini
   (0.75). The Phi-3 family appears to have evolved this:
   Phi-1/1.5/2 use 0.5; Phi-3-mini-4k uses 1.0 (full RoPE) per
   its config; Phi-4-mini uses 0.75. The §2 entry here defers
   to that reading.

### Open citations / things not fully verified

* The 2026-Q1 Apple Foundation Model row in §2 reflects the most
  recent published update (Apple ML Research, Aug 2025 tech
  report); a 2026-Q1 refresh is implied by the rolling release
  cadence but has not been formally announced at the time of
  writing. Treat the row as a placeholder until WWDC 2026.
* The Qwen3-Next hybrid linear/full attention configuration is
  documented in the HF model card but the technical report has
  not been fully released; the §3.3 lineage tip and §6 divergence
  entry rely on the model card.
* The "DeepSeek-V3.1" entry (May/Sept 2025) is an interim refresh
  whose exact NSA adoption status is partially confirmed by the
  HF release notes; the precise sparse-attention configuration
  used in V3.1 production may differ from the NSA paper.
* The number of "active recipes" for long-context (5) in §6.2 is
  conservative; YaRN and Llama-3 smooth could be argued as a
  single family with two parameterizations.
