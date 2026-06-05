## 01 — Mainstream Small Language Model Census v2 (<8B params, 2023–2026)

*Stream: model-census. Project: llm-layers. Date: 2026-06-04. Version: 2.0.*

*v1 was a snapshot. v2 is a survey of three years of <8B-parameter LLM-layer evolution: 2023 baselines, generation-to-generation deltas inside each family, and the architectural axes the layer-IR must parametrize.*

---

## 1. Executive Summary

This census catalogs **80 mainstream small language models** (active params <8B, plus a handful of architecturally-load-bearing 8–16B references) shipped between **February 2023 and July 2025**, drawn from dense decoder, sparse MoE, hybrid SSM, pure SSM/linear-attention, vision-language text-tower, and code-specialized families. The goal is twofold: (a) surface the **architectural axes** that a minimal operator API (the llm-layers project) must parametrize to instantiate any of them, and (b) tell the **three-year evolution story** — what was adopted, what was abandoned, and what was re-adopted.

Three years is enough time for clear consensus and clear dissent. The consensus axes are easy to list: RoPE has out-competed ALiBi for position encoding everywhere except a handful of academic models; SwiGLU/GeGLU has displaced bare GELU/SiLU MLPs; RMSNorm has displaced LayerNorm in 90% of new architectures; GQA has displaced MHA below 14B and displaced MQA above 1B; rope_theta has migrated from 10000 to 500000–10000000 as context windows have grown; SentencePiece is being replaced by Tiktoken-regex BPE on the largest models. But every single one of these has at least two production counter-examples in the corpus.

The non-consensus picture is more interesting. **OLMo 2 ships post-norm RMSNorm while every other 2024 model ships pre-norm**, and OLMo 2 still measures competitively. **OpenELM** (Apple, 2024-04) varies the *number of attention heads, the number of KV heads, and the FFN-multiplier per-layer*, exposing per-layer `num_query_heads[]`, `num_kv_heads[]`, and `ffn_multipliers[]` lists — making it the only published model whose *shape* varies across depth. **BitNet b1.58** (Microsoft, 2024-11 / 2025-04) trains 1.58-bit ternary weights with a `relu2` activation, breaking the SwiGLU/GeGLU monoculture from the quantization side. **Gemma 3** dropped Gemma 2's logit softcapping entirely (both `final_logit_softcapping` and `attn_logit_softcapping` are null in the verified config), while keeping the dual norm and adding a 5:1 SWA:full alternation with two independent RoPE bases per layer-type. **DeepSeek-V2-Lite** is the first <8B production model with Multi-head Latent Attention (MLA), and **MiniCPM 3 4B** is the first non-DeepSeek production model with MLA. **Phi-4-mini-flash** (2025-07) ships a Samba-derived hybrid Mamba+attention. **Hymba** (NVIDIA, 2024-11) runs Mamba and attention as parallel heads inside one block. **SmolLM3** interleaves NoPE every 4th layer. **StableLM 3B-4E1T** (2023-09) pioneered 25% partial-rotary RoPE two years before SmolLM3 generalized the idea.

Fifteen architectural axes are identified (Section 6), up from 11 in v1. The four new axes are per-layer width/depth scaling (OpenELM), parallel vs sequential sublayer (Phi-2, Falcon-7B, GPT-J inheritance), activation function as standalone (BitNet's ReLU², Gemma's GELU-tanh, the SiLU/SwiGLU default), and RoPE rotation domain (full vs partial-RoPE vs MLA's NoPE/RoPE split vs NoPE-per-layer). Two more axes — embedding tying and position-extension method — are promoted from sub-rows in v1 to first-class axes (they exhibit too much variation to remain implicit).

The deliverable for the llm-layers project: any operator dataclass that hard-codes (i) `attn_scale = 1/sqrt(head_dim)`, (ii) `head_dim = hidden_size / num_attention_heads`, (iii) `apply_rope` as a global model-level flag, (iv) `tie_word_embeddings` as a dtype-only difference, (v) `num_kv_heads` as a model-level scalar, or (vi) `activation = SwiGLU` will silently produce wrong outputs for at least one production model in this census. The axes below enumerate the safe parametrization.

---

## 2. Methodology and Scope

### 2.1 Inclusion criteria

A model is in scope if all four hold:

1. **Active parameter count <8B** at inference. For MoE, "active" means routed-experts × top-k + shared-experts + dense. (Mixtral 8×7B at 13B active is excluded; DeepSeek-V2-Lite at 2.4B active is included.) Three exceptions are made for architecturally-load-bearing models (OLMo 2 13B, StableLM 2 12B, Qwen3-30B-A3B) — they are marked "noted, out of size scope".
2. **Mainstream**, defined as either (a) ≥10k HF downloads/week as of 2026-06-04, (b) shipped by Anthropic/Apple/Meta/Microsoft/Google/Alibaba/01-AI/IBM/Cohere/AI2/Mistral/DeepSeek/Stability AI/Hugging Face/NVIDIA/Apple/MosaicML/AI21/TII/Zyphra/Hymba/RWKV-LM/State-Spaces/BigCode/Tsinghua/Baichuan, or (c) named in at least three peer-cited 2024–2025 efficiency benchmarks.
3. **Decoder-only or hybrid token-mixer**. Florence-2 (encoder-decoder) is included as the single encoder-decoder counter-example. Pure encoders (BERT/ModernBERT) are excluded.
4. **Public config** (HF model card or vendor tech report). Closed models are noted but not tabulated: Apple Foundation Model ~3B (announced WWDC 2024, partial), Anthropic Claude Haiku (closed), Gemini Nano (closed), GPT-4.1-nano (closed). The single exception is "GPT-OSS 20B" which is mentioned but excluded as >8B.

### 2.2 Field provenance

For each row the source is one of (in priority order): (a) the official HF `config.json`, (b) the vendor mirror's HF `config.json` (e.g. `unsloth/`, `deepseek-ai/`, `ai21labs/`) where the original repo is gated, (c) the vendor's published technical report, (d) the `transformers` source tree (`src/transformers/models/<name>/modeling_*.py` and `configuration_*.py`) for fields not present in config (e.g. parallel attention/FFN in Phi-2 is a class-level choice in `PhiDecoderLayer`, not a config flag).

For verified rows, each fact is supported by an explicit URL or paper citation in §10. For gated-repo rows where only the unsloth mirror is accessible, this is noted with a † symbol and a tech-report fallback.

### 2.3 Notation

Heads/KV/hd = `num_attention_heads / num_key_value_heads / head_dim`. ✓ = present. Vocab tokenizer family abbreviations are the same as v1 (TT = Tiktoken cl100k/o200k; LL3 = Llama-3 Tiktoken 128k; SP = SentencePiece generic; GP = Gemma SentencePiece 256k; QW = Qwen BPE 151936; GPT2 = GPT-2 BPE; MS = Microsoft Phi SP 32k or o200k 200k; SC2 = StarCoder-2 BPE 49152), with one new family: **TK** = Tekken/Mistral-v3 SP 32768 (Mistral 7B v0.3, derived from tekken family).

### 2.4 Differences from v1 census

v1 had 45 rows across 22 architectures and 11 axes. v2 has **80 rows across ~30 architectures and 15 axes**. The expansion is concentrated in three places: **2023 baseline year** (15 added rows: LLaMA 1/2, Mistral 7B v0.1/v0.2, MPT 7B, Falcon 7B, StarCoder 1, Qwen 1, ChatGLM 2/3, GLM-4, Baichuan 1/2, DeepSeek LLM/Math/Coder v1, OLMo 1, Phi-1/1.5/2, StableLM 3B, Pythia 2.8B/6.9B, TinyLlama, MobileLLM); **architecturally-distinct 2024–2025 models** (12 added rows: OpenELM 270M/450M/1.1B/3B, BitNet b1.58, Phi-4-mini-flash, Phi-3.5-MoE, Gemma 1 2B/7B, RecurrentGemma 2B/9B, Falcon Mamba, Falcon 3, Falcon-H1, Hymba, Cohere Aya, MiniCPM 1/2, StableLM Zephyr 3B, MobileLLM, Mathstral, Codestral Mamba); and **per-family completeness** (8 added Qwen rows for 1→1.5→2 generations, 3 added Llama rows for Llama 1/2/3.3, 3 added DeepSeek rows).

---

## 3. Census by Year

The narrative order is chronological. Within each year, families are grouped by lineage (dense/MoE/hybrid SSM/code-specialized/multimodal text tower).

### 3.1 The 2023 Baseline

This is the year a "Llama-shaped" model emerged as the consensus design, and also the year of the most architectural divergence within that consensus. Llama 1 (Feb 2023) and Llama 2 (July 2023) both ship with **MHA** at 7B (GQA appears only at 70B in Llama 2); Mistral 7B v0.1 (Sep 2023) is the first widely-adopted **GQA-at-7B with sliding-window attention**; MPT 7B (May 2023) ships with **ALiBi** in lieu of RoPE, the high-water mark of the ALiBi tradition; Falcon 7B (May 2023) ships with **parallel attention/FFN** (`parallel_attn=true`, GPT-J-style) and a 71-head MQA; StarCoder 1 (May 2023) ships with classic MQA + learned absolute positional embeddings; Qwen 1 7B (Sep 2023) ships with GQA + **QKV bias** (a legacy that Qwen 2 will drop); ChatGLM2 6B (Jun 2023) introduces **MQA at 6B** (32 query / 2 group → kv_heads=2), the first Chinese decoder with grouped/multi-query attention; DeepSeek LLM 7B (Nov 2023) is a vanilla MHA Llama-clone with a custom 102400 vocab; OLMo 1 7B (Feb 2024 release but Q4 2023 lineage) is pre-norm RMS without QK-norm; Phi-1/1.5/2 (Jun–Dec 2023) cement Microsoft's "data-quality > scale" thesis with classic MHA, GPT-2-style learned-absolute or partial-rotary positional embeddings, and **parallel attention/FFN** in Phi-2 (a `PhiDecoderLayer`-level choice inherited from GPT-J via the original Phi-1 codebase); StableLM 3B-4E1T (Sep 2023) ships **25% partial-rotary RoPE**, the design StableLM 2 will inherit and SmolLM3 will generalize; Pythia 2.8B/6.9B (early 2023, EleutherAI's release of the GPT-NeoX scaling-laws suite) ships with classic MHA + parallel attention/FFN + GPT-NeoX BPE.

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA 1 | 7B | 2023-02 | MHA | 32/32/128 | vanilla (θ=10000) | RMS pre | SwiGLU | 32000 | SP (Llama-1) | n | 2048 | LLaMA paper (arXiv:2302.13971) |
| LLaMA 2 | 7B | 2023-07 | MHA | 32/32/128 | vanilla (θ=10000) | RMS pre | SwiGLU | 32000 | SP | n | 4096 | LLaMA 2 paper (arXiv:2307.09288) |
| Mistral | 7B v0.1 | 2023-09 | GQA + **SWA=4096** | 32/8/128 | vanilla (θ=10000) | RMS pre | SwiGLU | 32000 | SP | n | 8192 / 4096 | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) (verified: sliding_window=4096) |
| Mistral | 7B v0.2 | 2024-03 | GQA (SWA **dropped**) | 32/8/128 | vanilla (θ=10000) | RMS pre | SwiGLU | 32000 | SP | n | 32768 | Mistral release notes |
| MPT | 7B | 2023-05 | MHA + **ALiBi** | 32/32/128 | none | LN pre | GELU MLP | 50432 | GPT-NeoX | n | 2048 | [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) |
| Falcon | 7B | 2023-05 | **MQA** (71 query, 1 KV) + **parallel attn/FFN** | 71/1/64 | vanilla | LN pre | GELU MLP | 65024 | Falcon BPE | n | 2048 | [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) (verified: `parallel_attn=true`, `multi_query=true`) |
| StarCoder 1 | 7B | 2023-05 | **MQA** | 42/1/128 (per BigCode) | absolute learned | LN pre | GELU MLP + bias | 49152 | SC1 | n | 8192 | StarCoder paper (arXiv:2305.06161) |
| ChatGLM | 6B | 2023-03 | MHA + prefix-LM | 32/32/128 | RoPE 2D | LN | GeGLU | 130528 | ChatGLM SP | y | 2048 | THUDM/chatglm-6b |
| ChatGLM 2 | 6B | 2023-06 | **MQA** (32/2) | 32/2/128 | vanilla | RMS | SwiGLU | 65024 | ChatGLM SP | n | 32768 | [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) (verified: `multi_query_group_num=2`) |
| ChatGLM 3 | 6B | 2023-10 | MQA (32/2) | 32/2/128 | vanilla | RMS | SwiGLU | 65024 | ChatGLM SP | n | 32768 | THUDM/chatglm3-6b |
| Qwen 1 | 7B | 2023-09 | MHA + **QKV bias** | 32/32/128 | vanilla (θ=10000) | RMS pre | SwiGLU | 151936 | QW | n | 8192 | Qwen tech report (arXiv:2309.16609) |
| Baichuan 1 | 7B | 2023-06 | MHA + **ALiBi** | 32/32/128 | none | LN | SwiGLU | 64000 | Baichuan SP | n | 4096 | baichuan-inc/Baichuan-7B |
| Baichuan 2 | 7B | 2023-09 | MHA + **RoPE** (ALiBi→RoPE) | 32/32/128 | vanilla (θ=10000) | RMS | SwiGLU | 125696 | Baichuan SP | n | 4096 | baichuan-inc/Baichuan2-7B |
| DeepSeek LLM | 7B | 2023-11 | MHA | 32/32/128 | vanilla (θ=10000) | RMS pre | SwiGLU | 102400 | DeepSeek BPE | n | 4096 | [deepseek-ai/deepseek-llm-7b-base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base) (verified) |
| DeepSeek-Math | 7B | 2024-02 | MHA | 32/32/128 | vanilla | RMS pre | SwiGLU | 102400 | DeepSeek BPE | n | 4096 | deepseek-ai/deepseek-math-7b-base |
| DeepSeek-Coder | 6.7B | 2023-11 | MHA | 32/32/128 | vanilla | RMS pre | SwiGLU | 32256 | DeepSeek Coder SP | n | 16384 | deepseek-ai/deepseek-coder-6.7b-base |
| OLMo 1 | 7B | 2024-02 | MHA | 32/32/128 | vanilla (θ=10000) | RMS **pre** (no QK-norm) | SwiGLU | 50304 | GPT-NeoX | n | 2048 | [allenai/OLMo-7B-hf](https://huggingface.co/allenai/OLMo-7B-hf) (verified) |
| OLMo 1 | 1B | 2024-04 | MHA | 16/16/128 | vanilla | RMS pre | SwiGLU | 50304 | GPT-NeoX | n | 2048 | allenai/OLMo-1B-hf |
| Phi-1 | 1.3B | 2023-06 | MHA + **parallel attn/FFN** + partial RoPE | 32/32/64 | partial rotary 0.5 | LN | GELU (no gate) | 51200 | GPT-2 | n | 2048 | microsoft/phi-1 |
| Phi-1.5 | 1.3B | 2023-09 | MHA + parallel + partial rotary | 32/32/64 | partial rotary 0.5 | LN | GELU | 51200 | GPT-2 | n | 2048 | microsoft/phi-1_5 |
| Phi-2 | 2.7B | 2023-12 | MHA + **parallel attn/FFN** + partial rotary 0.4 | 32/32/80 | partial rotary 0.4 | LN | GELU (no gate) | 51200 | GPT-2 | n | 2048 | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) (verified `partial_rotary_factor=0.4`, hidden=2560/32=80) |
| StableLM 3B | 4E1T | 2023-09 | MHA + **partial rotary 0.25** + QKV bias | 32/32/80 | partial rotary | LN pre | SwiGLU | 50304 | GPT-NeoX | n | 4096 | [stabilityai/stablelm-3b-4e1t](https://huggingface.co/stabilityai/stablelm-3b-4e1t) (verified `partial_rotary_factor=0.25`, but `use_qkv_bias=false`) |
| StableLM Zephyr 3B | 3B | 2023-10 | MHA + partial rotary 0.25 | 32/32/80 | partial rotary | LN pre | SwiGLU | 50304 | GPT-NeoX | n | 4096 | stabilityai/stablelm-zephyr-3b |
| Pythia | 2.8B | 2023-04 | MHA + **parallel attn/FFN** | 32/32/80 | vanilla (θ=10000) | LN | GELU | 50304 | GPT-NeoX | n | 2048 | EleutherAI/pythia-2.8b |
| Pythia | 6.9B | 2023-04 | MHA + parallel | 32/32/128 | vanilla | LN | GELU | 50304 | GPT-NeoX | n | 2048 | EleutherAI/pythia-6.9b |
| TinyLlama | 1.1B | 2023-12 | **GQA** | 32/4/64 | vanilla (θ=10000) | RMS pre | SwiGLU | 32000 | SP (Llama-2) | n | 2048 | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (verified: `num_attention_heads=32, num_key_value_heads=4` — *GQA, NOT MHA*; v1 census internally contradicted itself; this row is the corrected fact) |
| MobileLLM | 125M | 2024-10 (paper Aug) | GQA + **deep-narrow** + embedding tying | 9/3/64 | vanilla | RMS pre | SwiGLU | 32000 | LLaMA SP | y | 2048 | facebook/MobileLLM-125M |
| MobileLLM | 1B | 2024-10 | GQA + deep-narrow + **block-wise weight sharing** | 9/3/64 (×54 layers) | vanilla | RMS | SwiGLU | 32000 | LLaMA SP | y | 2048 | facebook/MobileLLM-1B |

**Notable absent from this baseline year by intent:** the original RWKV-4 (which would have been the first linear-attention-style at scale) — RWKV-6 is the first row we cover for RWKV because it's the first widely-deployed Hugging Face-supported variant. The original Mamba paper landed Dec 2023; we tabulate it under §3.2 (2024 ssm boom) since the HF release was Q1 2024.

### 3.2 2024: GQA Consensus, MLA Introduction, SSM Boom

2024 is the year the "Llama-style 7B" became the universal floor. Every major dense decoder family converged on GQA + RMSNorm pre-norm + SwiGLU + RoPE, with two architecturally-distinct exits: (a) **MLA** (DeepSeek-V2-Lite, May 2024; MiniCPM 3, Sep 2024) cuts the KV-cache 10–70× via low-rank latent compression, and (b) **SSM** (Mamba 2.8B / Falcon Mamba 7B / Codestral Mamba / Zamba2 / RWKV-6 / Jamba) replaces attention entirely or partially. Concurrently, **OpenELM** (Apple, Apr 2024) introduces per-layer width variation — the most extreme departure from the assumption that all `L` layers are identically-shaped. **BitNet b1.58** (Microsoft/CAS paper Nov 2024) trains 1.58-bit ternary weights, introducing `relu2` activation. **Gemma 2** (Jul 2024) introduces dual norm + softcapping + 1:1 SWA:full alternation. **Phi-3.5-MoE** (Aug 2024) is the third distinct MoE pattern (16 experts, top-2, no shared) after Mixtral and DeepSeek-V2. **MobileLLM** (Meta, Oct 2024) demonstrates that depth-narrow + embedding tying + block-wise weight sharing reaches Llama-quality at sub-1B.

#### Dense decoders, 2024

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3 | 8B | 2024-04 | GQA | 32/8/128 | vanilla (θ=500000) | RMS pre | SwiGLU | 128256 | LL3 | n | 8192 | meta-llama/Meta-Llama-3-8B (Llama 3 paper, arXiv:2407.21783) |
| Llama 3.1 | 8B | 2024-07 | GQA | 32/8/128 | **Llama-3 freq-band scaling** (factor=8, low_freq=1, high_freq=4, orig=8192) | RMS pre | SwiGLU | 128256 | LL3 | n | 131072 | [unsloth/Meta-Llama-3.1-8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B) |
| Llama 3.2 | 1B | 2024-09 | GQA | 32/8/64 | Llama-3 (factor=32) | RMS pre | SwiGLU | 128256 | LL3 | **y** | 131072 | [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B) |
| Llama 3.2 | 3B | 2024-09 | GQA | 24/8/128 | Llama-3 (factor=32) | RMS pre | SwiGLU | 128256 | LL3 | **y** | 131072 | [unsloth/Llama-3.2-3B](https://huggingface.co/unsloth/Llama-3.2-3B) |
| Qwen 1.5 | 7B | 2024-02 | GQA + QKV bias | 32/32/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 151936 | QW | n | 32768 | Qwen/Qwen1.5-7B |
| Qwen 2 | 7B | 2024-06 | GQA (**bias dropped**) | 28/4/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 152064 | QW | n | 131072 | [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) (verified: hidden=3584, hd=128, kv=4) |
| Qwen 2.5 | 0.5B | 2024-09 | GQA | 14/2/64 | vanilla (θ=1e6) | RMS pre | SwiGLU | 151936 | QW | y | 32768 | Qwen/Qwen2.5-0.5B |
| Qwen 2.5 | 1.5B | 2024-09 | GQA | 12/2/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 151936 | QW | y | 131072 | Qwen/Qwen2.5-1.5B |
| Qwen 2.5 | 3B | 2024-09 | GQA | 16/2/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 151936 | QW | y | 32768 | Qwen/Qwen2.5-3B |
| Qwen 2.5 | 7B | 2024-09 | GQA | 28/4/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 152064 | QW | n | 131072 | Qwen/Qwen2.5-7B |
| Gemma 1 | 2B | 2024-02 | **MQA** + dual-norm | 8/1/256 | vanilla (θ=10000) | RMS **pre+post per sublayer** | **GeGLU** | 256000 | GP | y† | 8192 | google/gemma-2b |
| Gemma 1 | 7B | 2024-02 | MHA + dual-norm | 16/16/256 | vanilla | RMS pre+post | GeGLU | 256000 | GP | y† | 8192 | google/gemma-7b |
| Gemma 1.1 | 2B/7B | 2024-04 | as Gemma 1 | as 1 | vanilla | RMS pre+post | GeGLU | 256000 | GP | y† | 8192 | google/gemma-1.1-7b-it |
| Gemma 2 | 2B | 2024-07 | GQA + **SWA every other (W=4096)** | 8/4/256 | vanilla (θ=10000) | RMS pre+post + **softcap (50, 30)** | GeGLU | 256000 | GP | y† | 8192 / 4096 alt | unsloth/gemma-2-2b |
| Gemma 2 | 9B | 2024-07 | GQA + SWA every other | 16/8/256 | vanilla | RMS pre+post + softcap | GeGLU | 256000 | GP | y† | 8192 / 4096 alt | google/gemma-2-9b (tech report) |
| RecurrentGemma | 2B | 2024-04 | **Griffin/Hawk (LRU + local attn)** | per `block_types` | RoPE on attention blocks only | RMS pre | GeGLU | 256000 | GP | y | 8192 | [google/recurrentgemma-2b](https://huggingface.co/google/recurrentgemma-2b) (`block_types`, `attention_window_size`, `lru_width`) |
| RecurrentGemma | 9B | 2024-09 | Griffin/Hawk | per `block_types` | RoPE on attn-only | RMS pre | GeGLU | 256000 | GP | y | 8192 | google/recurrentgemma-9b-it |
| PaliGemma | 3B | 2024-05 | Gemma 1 2B text tower (MQA) | 8/1/256 | vanilla | RMS pre+post | GeGLU | 257152 | GP + img tokens | y | 8192 | google/paligemma-3b-pt-224 |
| PaliGemma 2 | 3B | 2024-12 | Gemma 2 2B text tower (GQA + SWA) | 8/4/256 | vanilla | RMS pre+post + softcap | GeGLU | 257152 | GP-v2 | y† | 8192 | google/paligemma2-3b-pt-224 |
| Phi-3 mini | 3.8B (4k) | 2024-04 | MHA | 32/32/96 | vanilla (θ=10000) | RMS pre | SwiGLU **fused gate_up** | 32064 | MS | n | 4096 / 2047 | microsoft/Phi-3-mini-4k-instruct |
| Phi-3 mini | 3.8B (128k) | 2024-04 | MHA | 32/32/96 | **LongRoPE** (per-dim factor[48]) | RMS pre | SwiGLU fused | 32064 | MS | n | 131072 | microsoft/Phi-3-mini-128k-instruct |
| Phi-3 small | 7B | 2024-04 | **BlockSparse** (dense every 2nd) | 32/8/128 | vanilla (θ=1e6) + μP | LN pre + μP scalars | **gegelu** | 100352 | TT (cl100k) | n | 8192 | microsoft/Phi-3-small-8k-instruct |
| Phi-3.5 mini | 3.8B | 2024-08 | MHA | 32/32/96 | LongRoPE | RMS pre | SwiGLU fused | 32064 | MS | n | 131072 / 262144 | microsoft/Phi-3.5-mini-instruct |
| Phi-3.5 vision | 3.8B text tower | 2024-08 | MHA | 32/32/96 | LongRoPE | RMS pre | SwiGLU fused | 32064 | MS + img tokens | n | 131072 | microsoft/Phi-3.5-vision-instruct |
| Mistral | 7B v0.3 | 2024-05 | GQA (no SWA) | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 32768 | TK | n | 32768 | mistralai/Mistral-7B-v0.3 |
| Mistral Nemo | 12B | 2024-07 | GQA | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 131072 | TK | n | 131072 | mistralai/Mistral-Nemo-Base-2407 (noted: >8B, but tekken tokenizer downstreams to 7B variants) |
| Mathstral | 7B | 2024-07 | GQA | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 32768 | TK | n | 32768 | mistralai/Mathstral-7B-v0.1 |
| OLMo 2 | 7B | 2024-11 | MHA | 32/32/128 | vanilla (θ=5e5) | **Post-norm + QK-RMSNorm over full head channels** | SwiGLU | 100352 | TT | n | 4096 | allenai/OLMo-2-1124-7B |
| OLMo 2 | 13B (oos, noted) | 2024-11 | MHA | 40/40/128 | vanilla (θ=5e5) | Post-norm + QK-norm | SwiGLU | 100352 | TT | n | 4096 | allenai/OLMo-2-1124-13B |
| SmolLM2 | 135M | 2024-11 | GQA | 9/3/64 | vanilla (θ=1e5) | RMS pre | SwiGLU | 49152 | SC2 | y | 8192 | HuggingFaceTB/SmolLM2-135M |
| SmolLM2 | 360M | 2024-11 | GQA | 15/5/64 | vanilla (θ=1e5) | RMS pre | SwiGLU | 49152 | SC2 | y | 8192 | HuggingFaceTB/SmolLM2-360M |
| SmolLM2 | 1.7B | 2024-11 | MHA | 32/32/64 | vanilla (θ=1.3e5) | RMS pre | SwiGLU | 49152 | SC2 | y | 8192 | HuggingFaceTB/SmolLM2-1.7B |
| Granite 3.0 | 2B | 2024-10 | GQA + μP scalars | 32/8/64 | vanilla (θ=10000) | RMS pre + μP | SwiGLU | 49152 | SC2 | y | 4096 | ibm-granite/granite-3.0-2b-base |
| Granite 3.1 | 2B | 2024-12 | GQA + μP scalars | 32/8/64 | vanilla (θ=5e6) | RMS pre + μP | SwiGLU | 49152 | SC2 | y | 131072 | ibm-granite/granite-3.1-2b-base |
| MiniCPM 1 | 2.4B | 2024-02 | MHA + μP (scale_emb/depth) | 36/36/64 | vanilla | RMS pre + μP | SwiGLU | 122753 | MiniCPM SP | y | 4096 | openbmb/MiniCPM-2B-sft-bf16 |
| MiniCPM 2 | 2.4B | 2024-04 | as MiniCPM 1, w/ "Dragonfly" merge | 36/36/64 | vanilla | RMS pre + μP | SwiGLU | 122753 | MiniCPM SP | y | 4096 | openbmb/MiniCPM-2B-128k |
| MiniCPM 3 | 4B | 2024-09 | **MLA** (q_lora=768, kv_lora=256, qk_nope=64, qk_rope=32) | 40/40 (latent) | LongRoPE | RMS pre + μP | SwiGLU | 73448 | MiniCPM SP | — | 32768 | openbmb/MiniCPM3-4B |
| Yi 1.5 | 6B | 2024-05 | GQA | 32/4/128 | vanilla (θ=5e6) | RMS pre | SwiGLU | 64000 | Yi SP | n | 4096 | 01-ai/Yi-1.5-6B |
| Yi-Coder | 1.5B | 2024-09 | GQA | 16/4/128 | vanilla (θ=5e6) | RMS pre | SwiGLU | 64000 | Yi SP | n | 4096 | 01-ai/Yi-Coder-1.5B |
| Yi-Coder | 9B (oos) | 2024-09 | GQA | 32/4/128 | vanilla | RMS pre | SwiGLU | 64000 | Yi SP | n | 4096 | 01-ai/Yi-Coder-9B |
| InternLM 2 | 7B | 2024-01 | GQA | 32/8/128 | dynamic NTK (factor=2) | RMS pre | SwiGLU | 92544 | InternLM SP | n | 32768 | internlm/internlm2-7b |
| InternLM 2.5 | 7B | 2024-07 | GQA | 32/8/128 | dynamic NTK (factor=2, θ=5e7) | RMS pre | SwiGLU | 92544 | InternLM SP | n | 262144 | internlm/internlm2_5-7b |
| StableLM 2 | 1.6B | 2024-01 | MHA + QKV bias + partial rotary 0.25 | 32/32/64 | partial rotary (θ=10000) | LN pre | SwiGLU | 100352 | TT | n | 4096 | stabilityai/stablelm-2-1_6b |
| StableLM 2 | 12B (oos) | 2024-04 | GQA + **QK-LayerNorm** + partial rotary 0.25 | 32/8/128 | partial rotary (θ=10000) | LN pre + QK-LN | SwiGLU | 100352 | TT | n | 4096 | stabilityai/stablelm-2-12b |
| Falcon 3 | 1B | 2024-12 | GQA | 8/4/256 | vanilla (θ=1000042) | RMS pre | SwiGLU | 131072 | Tekken-style | y | 32768 | tiiuae/Falcon3-1B-Base |
| Falcon 3 | 3B | 2024-12 | GQA | 10/4/256 | vanilla (θ=1000042) | RMS pre | SwiGLU | 131072 | Tekken-style | y | 32768 | tiiuae/Falcon3-3B-Base |
| Falcon 3 | 7B | 2024-12 | GQA | 12/4/256 | vanilla (θ=1000042) | RMS pre | SwiGLU | 131072 | Tekken-style | n | 32768 | [tiiuae/Falcon3-7B-Base](https://huggingface.co/tiiuae/Falcon3-7B-Base) (verified: head_dim=256, hidden=3072) |
| Falcon 3 | 10B (oos) | 2024-12 | GQA | 12/4/256 | vanilla (θ=1000042) | RMS pre | SwiGLU | 131072 | Tekken-style | n | 32768 | tiiuae/Falcon3-10B-Base |
| Cohere Aya 8B | Expanse 8B | 2024-12 | GQA + no QKV bias | 32/8/128 | vanilla (θ=4e6) | LN pre | SwiGLU | 256000 | Cohere SP | **y (tied at 8B!)** | 8192 | CohereForAI/aya-expanse-8b |
| OpenELM | 270M | 2024-04 | **GQA per-layer list** | per-layer `num_query_heads[]`, `num_kv_heads[]` | vanilla (θ=10000) | RMS pre + **QK-norm** | SwiGLU + **per-layer `ffn_multipliers[]`** | 32000 | LLaMA SP | y | 2048 | apple/OpenELM-270M |
| OpenELM | 450M | 2024-04 | GQA per-layer list | per-layer | vanilla | RMS pre + QK-norm | SwiGLU + per-layer ffn | 32000 | LLaMA SP | y | 2048 | apple/OpenELM-450M |
| OpenELM | 1.1B | 2024-04 | GQA per-layer list | per-layer | vanilla | RMS pre + QK-norm | SwiGLU + per-layer ffn | 32000 | LLaMA SP | y | 2048 | apple/OpenELM-1_1B |
| OpenELM | 3B | 2024-04 | **GQA per-layer list** [12,12,…,24,24] / [3,3,…,6,6] | per-layer (verified) | vanilla (θ=10000) | RMS pre + QK-norm | SwiGLU + per-layer ffn[0.5..4.0] | 32000 | LLaMA SP | y | 2048 | [apple/OpenELM-3B](https://huggingface.co/apple/OpenELM-3B) (verified) |
| StarCoder 2 | 3B | 2024-02 | GQA + **dense QKV biases** (`use_bias=true`) | 24/2/128 | vanilla (θ≈1e6) | LN pre | gated GELU-tanh + bias | 49152 | SC2 | — | 16384 / 4096 | bigcode/starcoder2-3b |
| StarCoder 2 | 7B | 2024-02 | GQA + bias | 36/4/128 | vanilla (θ=1e6) | LN pre | gated GELU-tanh + bias | 49152 | SC2 | — | 16384 / 4096 | bigcode/starcoder2-7b |
| CodeGemma | 2B | 2024-04 | **MQA** | 8/1/256 | vanilla (θ=10000) | RMS pre+post (Gemma 1 dual norm) | GeGLU | 256000 | GP | y | 8192 | unsloth/codegemma-2b |
| CodeGemma | 7B | 2024-04 | **MHA** (not GQA — v1 census error) | 16/16/256 | vanilla (θ=10000) | RMS pre+post (Gemma 1 dual norm) | GeGLU | 256000 | GP | n | 8192 | [unsloth/codegemma-7b](https://huggingface.co/unsloth/codegemma-7b) (verified: `num_attention_heads=16, num_key_value_heads=16`) |

#### MoE, 2024

| Family | Variant | Released | Attn | H/KV/hd | Experts | Top-k | Shared | Notes | Source |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 16B-A2.4B | 2024-05 | **MLA** (kv_lora=512, qk_nope=128, qk_rope=64, v_head=128, **no q_lora**) | 16/16 (latent) | 64 | 6 | **2** | first_k_dense_replace=1; YaRN | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) (verified) |
| DeepSeek-Coder-V2-Lite | 16B-A2.4B | 2024-06 | MLA | 16/16 (latent) | 64 | 6 | 2 | code-specialized DV2-Lite | deepseek-ai/DeepSeek-Coder-V2-Lite-Base |
| OLMoE | 1B-A1B (7B total) | 2024-09 | MHA | 16/16/128 | 64 | 8 | 0 | no shared, `norm_topk_prob=false` | allenai/OLMoE-1B-7B-0125 |
| Phi-3.5-MoE | 6.6B-A2B | 2024-08 | MHA | 32/8/128 | 16 | 2 | 0 | first non-DeepSeek non-Mixtral MoE in our scope | microsoft/Phi-3.5-MoE-instruct |
| Jamba (v0.1) | 52B-A12B (oos but noted) | 2024-03 | hybrid: 1-in-8 attention + 1-in-2 MoE | 32/8/128 | 16 | 2 | 0 | mamba_d_state=16, mamba_d_conv=4 | [ai21labs/Jamba-v0.1](https://huggingface.co/ai21labs/Jamba-v0.1) (verified — replaces Jamba-tiny-dev citation from v1) |

#### Hybrid SSM and pure SSM/linear, 2024

| Family | Variant | Released | Token mixer | Layout | RoPE? | Norm | FFN | Vocab | Source |
|---|---|---|---|---|---|---|---|---|---|
| Mamba | 2.8B | 2023-12 (paper) / 2024 (HF) | **Selective SSM (S6)** | 64 SSM blocks, no attention | none | RMS pre | none | 50280 | state-spaces/mamba-2.8b-hf |
| Mamba 2 | 2.7B (oos in HF, paper Aug 2024) | 2024 | **SSD (state-space duality)** | uniform | none | RMS pre | none | 50280 | state-spaces/mamba2-2.7b |
| Falcon Mamba | 7B | 2024-08 | Mamba-1 (selective scan) | 64 SSM blocks | none | RMS pre | none | 65024 | [tiiuae/falcon-mamba-7b](https://huggingface.co/tiiuae/falcon-mamba-7b) (verified: state_size=16, expand=16, conv_kernel=4) |
| Zamba2 | 2.7B | 2024-10 | Mamba2 + periodic shared attention | hybrid pattern (54 layers) | RoPE on shared attn | RMS pre | gelu MLP | 32000 | Zyphra/Zamba2-2.7B |
| Zamba2 | 7B | 2024-11 | Mamba2 + periodic shared attention | hybrid pattern | RoPE on shared attn | RMS pre | gelu MLP | 32000 | Zyphra/Zamba2-7B |
| RWKV-6 (Finch) | 1.6B | 2024-08 | **RWKV WKV6** linear attention + channel mix | uniform | none (time-decay) | LN | RWKV channel-mix | 65536 | RWKV/v6-Finch-1B6-HF |
| RWKV-6 (Finch) | 7B | 2024-08 | RWKV WKV6 | uniform | none | LN | RWKV channel-mix | 65536 | RWKV/v6-Finch-7B-HF |
| Codestral Mamba | 7B | 2024-07 | **Mamba-2** | 64 SSD blocks | none | RMS pre | none | 32768 | mistralai/Mamba-Codestral-7B-v0.1 |
| BitNet b1.58 | 2B-4T | 2024-11 paper / 2025-04 HF | MHA→GQA | 20/5/128 | vanilla (θ=5e5) | RMS pre + sub-norms | **ReLU² FFN** (not SwiGLU/GeGLU) | 128256 | [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) (verified: `hidden_act=relu2`, 1.58-bit ternary weights) |
| Hymba | 1.5B | 2024-11 | **Parallel Mamba + attention heads** in one block | 32 layers, sliding W=1024 + global at [0,15,31] | NTK | RMS | SwiGLU | 32001 | [nvidia/Hymba-1.5B-Base](https://huggingface.co/nvidia/Hymba-1.5B-Base) (verified: `model_type=hymba`, parallel) |

### 3.3 2025: Hybrid Mainstreaming, QK-Norm, NoPE, MLA Generalization

In 2025 the architectural diversity that bloomed in 2024 either consolidates (GQA + RMSNorm + SwiGLU + RoPE is now the assumed default) or jumps to the next axis (NoPE interleaving, QK-norm placement, MLA generalization to non-DeepSeek vendors, hybrid Mamba+attention from a major IHV).

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.3 | 70B (oos, noted) | 2024-12 | GQA | 64/8/128 | Llama-3 (factor=8) | RMS pre | SwiGLU | 128256 | LL3 | n | 131072 | meta-llama/Llama-3.3-70B-Instruct |
| Qwen 3 | 0.6B | 2025-04 | GQA + **QK-RMSNorm per head_dim**, pre-RoPE | 16/8/**128 (decoupled — hidden 1024 ⇒ hidden/H=64)** | vanilla (θ=1e6) | RMS pre + QK-norm | SwiGLU | 151936 | QW | y | 40960 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (verified: head_dim=128, hidden_size=1024, num_attention_heads=16 — the *cleanest decoupling example*; replaces v1's incorrect Phi-3-mini exemplar) |
| Qwen 3 | 1.7B | 2025-04 | GQA + QK-norm | 16/8/128 | vanilla (θ=1e6) | RMS pre + QK-norm | SwiGLU | 151936 | QW | y | 40960 | Qwen/Qwen3-1.7B |
| Qwen 3 | 4B | 2025-04 | GQA + QK-norm | 32/8/128 | vanilla (θ=1e6) | RMS pre + QK-norm | SwiGLU | 151936 | QW | y | 40960 | Qwen/Qwen3-4B |
| Qwen 3 | 8B | 2025-04 | GQA + QK-norm | 32/8/128 | vanilla (θ=1e6) | RMS pre + QK-norm | SwiGLU | 151936 | QW | n | 40960 | Qwen/Qwen3-8B |
| Qwen 3 MoE | 30B-A3B | 2025-04 | GQA + QK-norm | 32/4/128 | vanilla (θ=1e6) | RMS pre + QK-norm | SwiGLU-MoE (moe_ffn=768, norm_topk=true, dense_step=1) | 151936 | QW | n | 40960 | Qwen/Qwen3-30B-A3B |
| Qwen 2.5-VL | 3B / 7B text tower | 2025-01 | GQA + **M-RoPE (3D)** | 16/2/128 (3B), 28/4/128 (7B) | M-RoPE (T,H,W axes) | RMS pre | SwiGLU | 151936 | QW | y (3B) / n (7B) | 32768 | Qwen/Qwen2.5-VL-3B / 7B |
| Qwen 2.5-Math | 7B | 2024-09 | GQA | 28/4/128 | vanilla | RMS pre | SwiGLU | 152064 | QW | n | 4096 | Qwen/Qwen2.5-Math-7B (backbone of R1-Distill-Qwen-7B) |
| Gemma 3 | 1B | 2025-03 | GQA + **5:1 SWA:full**, dual RoPE | 4/1/256 | global θ=1e6 + **local θ=10000** for SWA layers | RMS pre+post per sublayer (**softcap dropped**) | GeGLU | 262144 | GP-v3 | n | 32768 / 512 alt | unsloth/gemma-3-1b-pt |
| Gemma 3 | 4B (text tower) | 2025-03 | GQA + 5:1 SWA:full | 8/4/256 | global 1e6 + local 1e4 + linear factor 8 | RMS pre+post (**`final_logit_softcapping=null`, `attn_logit_softcapping=null` — Gemma 3 dropped softcap; verified**) | GeGLU | 262208 | GP-v3 | — | 131072 / 1024 alt | [unsloth/gemma-3-4b-pt](https://huggingface.co/unsloth/gemma-3-4b-pt) |
| Gemma 3 | 12B / 27B (oos, noted) | 2025-03 | GQA + 5:1 SWA:full | larger | dual RoPE | RMS pre+post | GeGLU | 262208 | GP-v3 | n | 131072 | google/gemma-3-27b-pt |
| Gemma 3n | E2B / E4B (noted) | 2025-05 | **Matformer (elastic width)** | per-runtime-width | dual RoPE | RMS pre+post | GeGLU | 262208 | GP-v3 | n | 32768 | google/gemma-3n-E4B-it |
| Phi-4 mini | 3.8B | 2025-02 | GQA + **partial RoPE (0.75)** | 24/8/128 | LongRoPE on rotary fraction | RMS pre | SwiGLU fused | 200064 | MS (o200k) | y | 131072 / 262144 | microsoft/Phi-4-mini-instruct |
| Phi-4-mini-flash | 3.8B | 2025-07 | **Hybrid Mamba + attention (Samba-derived)** | 32 layers | LongRoPE on attention layers | RMS pre | SwiGLU fused | 200064 | MS (o200k) | y | 131072 | microsoft/Phi-4-mini-flash-reasoning |
| SmolLM3 | 3B | 2025-07 | GQA + **NoPE every 4th layer** | 16/4/128 | vanilla (θ=5e6); NoPE on selected layers | RMS pre | SwiGLU | 128256 | LL3 | y | 65536 | HuggingFaceTB/SmolLM3-3B |
| Granite 3.3 | 2B | 2025-04 | GQA + μP scalars | 32/8/64 | vanilla (θ=1e7) | RMS pre + μP | SwiGLU | 49152 | SC2 | y | 131072 | ibm-granite/granite-3.3-2b-base |
| Granite 3.3 | 8B | 2025-04 | GQA + μP scalars | 32/8/128 | vanilla (θ=1e7) | RMS pre + μP | SwiGLU | 49152 | SC2 | y | 131072 | ibm-granite/granite-3.3-8b-base |
| InternLM 3 | 8B | 2025-01 | GQA (KV=2!) | 32/2/128 | dynamic NTK (factor=6, θ=5e7) | RMS pre | SwiGLU | 128512 | InternLM SP | n | 32768 | internlm/internlm3-8b-instruct |
| Mistral Small 3.1 | 24B (oos, noted) | 2025-03 | GQA + **sink tokens** | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 131072 | TK | n | 32768 | mistralai/Mistral-Small-3.1-24B (noted: trains with attention sinks, a 2025 innovation) |
| Falcon-H1 | 7B | 2025 | Mamba-2 + attention hybrid | mixed | NTK on attn | RMS pre | SwiGLU | 131072 | Falcon BPE | n | 16384 | tiiuae/Falcon-H1-7B |
| DeepSeek-V3.1 | full (oos) | 2025 | MLA | 128/128 (latent) | YaRN | RMS pre | SwiGLU-MoE (256+1, top-8, aux-loss-free) | 129280 | DeepSeek BPE | n | 65536 | deepseek-ai/DeepSeek-V3.1 (noted only) |
| DeepSeek-R1-Distill-Qwen | 1.5B | 2025-01 | GQA | 12/2/128 | vanilla | RMS pre | SwiGLU | 151936 | QW | y | 131072 | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (backbone: Qwen2.5-Math-1.5B) |
| DeepSeek-R1-Distill-Qwen | 7B | 2025-01 | GQA | 28/4/128 | vanilla | RMS pre | SwiGLU | 152064 | QW | n | 131072 | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (backbone: Qwen2.5-Math-7B) |
| DeepSeek-R1-Distill-Llama | 8B | 2025-01 | GQA | 32/8/128 | Llama-3 (factor=8) | RMS pre | SwiGLU | 128256 | LL3 | n | 131072 | deepseek-ai/DeepSeek-R1-Distill-Llama-8B (backbone: Llama-3.1-8B) |
| RWKV-7 (Goose) | 1.5B | 2025-01 | **RWKV-7 (delta-rule update)** | uniform | none (decay) | LN | RWKV-7 channel-mix | 65536 | RWKV-world | n | 4096 | RWKV/rwkv-7-world-1B5-v3 |
| GPT-OSS | 20B (oos, noted) | 2025 | GQA + **trained sink tokens** | 16/2/128 | LongRoPE | RMS pre | SwiGLU-MoE | 200064 | o200k | n | 131072 | openai/gpt-oss-20b (noted: >8B, but the sink-token-from-training innovation is architecturally novel) |
| Apple Foundation Model | ~3B on-device | 2024-09 (announced) / 2025 (deployed) | partial (closed) | ~? | partial info | RMS | SwiGLU | ~? | Apple SP | y | 4096 | WWDC 2024 (noted: closed; per-layer rank-1 LoRA adapters reported, unverifiable) |

**Total in v2 census: 80 rows (vs 45 in v1), spanning ~30 distinct architectural families.**

---

## 4. Comprehensive Table (15 Axes, All Models)

This table is the at-a-glance reference. Columns are the 15 axes from §6. Empty cells = "default" or "n/a"; "-" = explicitly absent. For brevity, ~40 of the 80 models are tabulated here as representatives of their family; the per-family arcs in §5 list the rest.

| # | Model | A1: Attn | A2: PosEnc | A3: Norm | A4: FFN family | A5: MoE | A6: KV-cache | A7: Vocab/Tok | A8: μP/scalar | A9: Bias | A10: Tied | A11: Per-layer type | A12: Per-layer shape | A13: Parallel sublayer | A14: Activation | A15: RoPE rotation domain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LLaMA 1 7B | MHA | vanilla θ=10k | RMS pre | SwiGLU | – | full | 32k SP | – | none | n | uniform | uniform | sequential | SiLU | full rotary |
| 2 | LLaMA 2 7B | MHA | vanilla θ=10k | RMS pre | SwiGLU | – | full | 32k SP | – | none | n | uniform | uniform | sequential | SiLU | full |
| 3 | Llama 3 8B | GQA | vanilla θ=5e5 | RMS pre | SwiGLU | – | GQA | 128k LL3 | – | none | n | uniform | uniform | sequential | SiLU | full |
| 4 | Llama 3.1 8B | GQA | **Llama-3 scaling** | RMS pre | SwiGLU | – | GQA | 128k LL3 | – | none | n | uniform | uniform | sequential | SiLU | full |
| 5 | Llama 3.2 1B/3B | GQA | Llama-3 scaling | RMS pre | SwiGLU | – | GQA | 128k LL3 | – | none | **y** | uniform | uniform | sequential | SiLU | full |
| 6 | Mistral 7B v0.1 | GQA | vanilla θ=10k | RMS pre | SwiGLU | – | **SWA=4096** | 32k SP | – | none | n | uniform | uniform | sequential | SiLU | full |
| 7 | Mistral 7B v0.2 | GQA | vanilla θ=10k | RMS pre | SwiGLU | – | full (SWA off) | 32k SP | – | none | n | uniform | uniform | sequential | SiLU | full |
| 8 | Mistral 7B v0.3 | GQA | vanilla θ=1e6 | RMS pre | SwiGLU | – | full | 32k TK | – | none | n | uniform | uniform | sequential | SiLU | full |
| 9 | MPT 7B | MHA | **ALiBi** | LN pre | GELU MLP | – | full | 50k GPT-NeoX | – | none | n | uniform | uniform | sequential | GELU | – (no RoPE) |
| 10 | Falcon 7B | **MQA** | vanilla | LN pre | GELU MLP | – | MQA (1 KV) | 65k Falcon BPE | – | none | n | uniform | uniform | **parallel** | GELU | full |
| 11 | Falcon Mamba 7B | SSM (Mamba-1) | – | RMS pre | – (SSM gate) | – | SSM state | 65k Falcon BPE | – | none | n | uniform | uniform | sequential | SiLU (in SSM) | – |
| 12 | Falcon 3 7B | GQA | vanilla θ=1000042 | RMS pre | SwiGLU | – | GQA, hd=**256** | 131k Tekken-style | – | none | n | uniform | uniform | sequential | SiLU | full |
| 13 | Qwen 1 7B | MHA | vanilla θ=10k | RMS pre | SwiGLU | – | full | 151936 QW | – | **QKV bias** | n | uniform | uniform | sequential | SiLU | full |
| 14 | Qwen 2 7B | GQA | vanilla θ=1e6 | RMS pre | SwiGLU | – | GQA | 152064 QW | – | none (dropped) | n | uniform | uniform | sequential | SiLU | full |
| 15 | Qwen 2.5 7B | GQA | vanilla θ=1e6 | RMS pre | SwiGLU | – | GQA | 152064 QW | – | none | n | uniform | uniform | sequential | SiLU | full |
| 16 | Qwen 3 0.6B | GQA + **QK-RMSNorm/head_dim, pre-RoPE** | vanilla θ=1e6 | RMS pre + QK-norm | SwiGLU | – | GQA | 151936 QW | – | none | y | uniform | uniform | sequential | SiLU | full; head_dim **decoupled** (128 vs hidden/H=64) |
| 17 | Qwen 3 MoE 30B-A3B | GQA + QK-norm | vanilla θ=1e6 | RMS pre + QK-norm | SwiGLU-MoE | 128/0/top-8, dense=1 | GQA | 151936 QW | – | none | n | **MoE/all** | uniform | sequential | SiLU | full |
| 18 | Gemma 1 2B | **MQA** | vanilla θ=10k | RMS pre+post per sublayer | **GeGLU** | – | MQA, hd=**256** | 256k GP | – | none | y† | uniform | uniform | sequential | **GELU-tanh** | full |
| 19 | Gemma 2 2B | GQA + **SWA every other** | vanilla | RMS pre+post + **softcap (50,30)** | GeGLU | – | mixed | 256k GP | softcap, q_pre_attn=256 | none | y† | **SWA/full alt** | uniform | sequential | GELU-tanh | full |
| 20 | Gemma 3 1B/4B | GQA + **5:1 SWA:full**, dual RoPE | global θ=1e6 + local θ=10k | RMS pre+post (**softcap dropped**) | GeGLU | – | mixed | 262k GP-v3 | q_pre_attn=256 only | none | n | **5:1 alt + dual θ** | uniform | sequential | GELU-tanh | full per layer-type |
| 21 | Gemma 3n E2B/E4B | as Gemma 3 + **Matformer** | dual θ | RMS pre+post | GeGLU | – | mixed | 262k GP-v3 | – | none | n | per layer-type | **elastic per-runtime** | sequential | GELU-tanh | full |
| 22 | RecurrentGemma 2B | **Griffin/Hawk (LRU + local attn)** | RoPE on attn only | RMS pre | GeGLU | – | LRU state + windowed attn | 256k GP | – | none | y | **block_types[]** | uniform | sequential | GELU-tanh | full on attn-only |
| 23 | Phi-1.5 1.3B | MHA + partial rotary | partial rotary 0.5 θ=10k | LN | GELU MLP | – | full | 51k GPT-2 | – | bias | n | uniform | uniform | **parallel** | GELU | **partial 0.5** |
| 24 | Phi-2 2.7B | MHA + partial rotary | partial rotary 0.4 θ=10k | LN | GELU MLP | – | full | 51k GPT-2 | – | bias | n | uniform | uniform | **parallel** | GELU | **partial 0.4** |
| 25 | Phi-3 mini 4k | MHA | vanilla θ=10k | RMS pre | SwiGLU **fused gate_up** | – | full + SWA=2047 | 32k MS | – | none | n | uniform | uniform | sequential | SiLU | full |
| 26 | Phi-3 mini 128k | MHA | **LongRoPE** | RMS pre | SwiGLU fused | – | full | 32k MS | – | none | n | uniform | uniform | sequential | SiLU | full |
| 27 | Phi-3 small 7B | **BlockSparse** dense-every-2nd | vanilla + μP | LN + μP scalars | **gegelu** | – | mixed | 100k cl100k | μP attn/emb mult | none | n | **dense/blocksparse alt** | uniform | sequential | gegelu | full |
| 28 | Phi-3.5 mini | MHA | LongRoPE | RMS pre | SwiGLU fused | – | full + SWA | 32k MS | – | none | n | uniform | uniform | sequential | SiLU | full |
| 29 | Phi-3.5-MoE | MHA | vanilla | RMS pre | SwiGLU-MoE (16/0/top-2) | 16/0/top-2 | full | 32k MS | – | none | n | MoE all | uniform | sequential | SiLU | full |
| 30 | Phi-4 mini 3.8B | GQA + **partial RoPE 0.75** | LongRoPE | RMS pre | SwiGLU fused | – | GQA | 200k o200k | – | none | y | uniform | uniform | sequential | SiLU | **partial 0.75** |
| 31 | Phi-4-mini-flash | **Hybrid Mamba + attn** (Samba) | LongRoPE on attn | RMS pre | SwiGLU fused | – | SSM state + attn | 200k o200k | – | none | y | **Mamba/attn mix** | uniform | sequential | SiLU | full on attn-only |
| 32 | OLMo 1 7B | MHA | vanilla θ=10k | RMS **pre** (no QK-norm) | SwiGLU | – | full | 50k GPT-NeoX | – | none | n | uniform | uniform | sequential | SiLU | full |
| 33 | OLMo 2 7B | MHA | vanilla θ=5e5 | **Post-norm + QK-RMSNorm/full channels** | SwiGLU | – | full | 100k TT | – | none | n | uniform | uniform | sequential | SiLU | full |
| 34 | SmolLM2 135M | GQA | vanilla θ=1e5 | RMS pre | SwiGLU | – | GQA | 49k SC2 | – | none | y | uniform | uniform | sequential | SiLU | full |
| 35 | SmolLM3 3B | GQA + **NoPE every 4th** | vanilla θ=5e6; mask | RMS pre | SwiGLU | – | mixed | 128k LL3 | – | none | y | **NoPE/RoPE alt** | uniform | sequential | SiLU | **NoPE per layer** |
| 36 | Granite 3.3 2B | GQA + **μP scalars (attn/emb/res/logit)** | vanilla θ=1e7 | RMS pre + μP | SwiGLU | – | GQA | 49k SC2 | attn=1/64, emb=12, res=0.22, logit=8 | none | y | uniform | uniform | sequential | SiLU | full |
| 37 | MiniCPM 3 4B | **MLA** | LongRoPE | RMS pre + μP (scale_emb=12, scale_depth=1.4) | SwiGLU | – | **MLA latent** | 73k MiniCPM SP | μP | none | – | uniform | uniform | sequential | SiLU | **NoPE/RoPE split (q_lora+kv_lora)** |
| 38 | StableLM 3B-4E1T | MHA + partial rotary 0.25 | partial rotary | LN pre | SwiGLU | – | full | 50k GPT-NeoX | – | none | n | uniform | uniform | sequential | SiLU | **partial 0.25** |
| 39 | StableLM 2 1.6B | MHA + QKV bias + partial 0.25 | partial rotary | LN pre | SwiGLU | – | full | 100k TT | – | **QKV bias** | n | uniform | uniform | sequential | SiLU | partial 0.25 |
| 40 | DeepSeek LLM 7B | MHA | vanilla θ=10k | RMS pre | SwiGLU | – | full | 102k DeepSeek BPE | – | none | n | uniform | uniform | sequential | SiLU | full |
| 41 | DeepSeek-V2-Lite 2.4B-A | **MLA** | YaRN | RMS pre | SwiGLU-MoE 64+2/top-6 | 64/2/top-6 | **MLA (lora=512+64)** | 102k DeepSeek BPE | – | none | n | **dense=1 + MoE all** | uniform | sequential | SiLU | **NoPE/RoPE split** |
| 42 | OLMoE 1B-A | MHA | vanilla θ=10k | RMS pre (`norm_topk_prob=false`) | SwiGLU-MoE 64/0/top-8 | 64/0/top-8 | full | 50k GPT-NeoX | – | none | n | MoE all | uniform | sequential | SiLU | full |
| 43 | Mamba 2.8B | **Selective SSM** | – | RMS pre | – | – | **SSM state** | 50k GPT-NeoX | – | – | – | uniform | uniform | sequential | SiLU (in SSM gate) | – |
| 44 | Zamba2 2.7B | Mamba-2 + periodic shared attn | RoPE on shared | RMS pre | gelu MLP | – | mixed | 32k LLaMA SP | – | – | – | **block_type[]** | uniform | sequential | GELU | full on attn-only |
| 45 | Jamba v0.1 | Mamba-1 + 1-in-8 attn + 1-in-2 MoE | – on Mamba | RMS pre | MoE (16/0/top-2) | 16/0/top-2 (1:2 layers) | mixed | 65k RWKV | – | – | – | **attn/expert periodic** | uniform | sequential | SiLU | – on SSM, full on attn |
| 46 | RWKV-6 7B | RWKV WKV-6 channel-mix | – (time-decay) | LN | RWKV channel-mix | – | RWKV state | 65k RWKV-world | – | – | – | uniform | uniform | **parallel (channel + time)** | sigmoid+ReLU² | – |
| 47 | RWKV-7 1.5B | **RWKV-7 (delta-rule)** | – (time-decay) | LN | RWKV-7 channel-mix | – | RWKV-7 state | 65k RWKV-world | – | – | – | uniform | uniform | parallel | sigmoid+ReLU² | – |
| 48 | BitNet b1.58 2B | GQA + sub-norms | vanilla θ=5e5 | RMS pre | **ReLU² FFN** | – | GQA | 128k LL3 | 1.58-bit weights | none | y | uniform | uniform | sequential | **ReLU²** | full |
| 49 | OpenELM 3B | **GQA per-layer list** | vanilla θ=10k | RMS pre + QK-norm | SwiGLU + **per-layer ffn[]** | – | per-layer GQA | 32k LLaMA SP | – | none | y | uniform op | **per-layer Q/KV/FFN** | sequential | SiLU | full |
| 50 | Hymba 1.5B | **Parallel Mamba + attn heads** | NTK | RMS pre | SwiGLU | – | SSM + attn (mixed) | 32k LLaMA SP | – | none | y | **global[0,15,31] + SWA** | uniform | **parallel (Mamba ‖ attn)** | SiLU | full |
| 51 | CodeGemma 7B | **MHA** (corrected from v1) | vanilla θ=10k | RMS pre+post (Gemma 1 dual) | GeGLU | – | full, hd=**256** | 256k GP | – | none | n | uniform | uniform | sequential | GELU-tanh | full |
| 52 | StarCoder 2 7B | GQA + **dense bias** | vanilla θ=1e6 | LN pre | **gated GELU-tanh + bias** | – | GQA + SWA=4096 | 49k SC2 | – | **full bias** | – | uniform | uniform | sequential | GELU-tanh | full |
| 53 | Cohere Aya 8B | GQA | vanilla θ=4e6 | LN pre | SwiGLU | – | GQA | 256k Cohere SP | logit_scale=0.0625 | none | **y (counterexample at 8B!)** | uniform | uniform | sequential | SiLU | full |
| 54 | TinyLlama 1.1B | **GQA** (corrected) | vanilla θ=10k | RMS pre | SwiGLU | – | GQA | 32k LLaMA SP | – | none | n | uniform | uniform | sequential | SiLU | full |
| 55 | MobileLLM 1B | GQA + **deep-narrow + weight-share** | vanilla | RMS pre | SwiGLU | – | GQA | 32k LLaMA SP | – | none | y | uniform | uniform (but **block-wise shared**) | sequential | SiLU | full |

The remaining ~25 rows (R1-Distill family, Qwen 2.5-VL text towers, PaliGemma, Florence-2, LLaVA-NeXT-Mistral, Mathstral, Codestral Mamba, Phi-3.5 vision text tower, OLMo 2 13B, Yi 1.5/Yi-Coder, InternLM 2/2.5/3, StableLM Zephyr 3B, Pythia, Baichuan 1/2, ChatGLM/2/3, GLM-4, Mamba 2, Falcon-H1, Gemma 1 7B, Gemma 1.1, etc.) inherit their parent family's axis values; see §5 evolution arcs for explicit per-generation diffs.

---

## 5. Per-Family Evolution Arcs

Eight families chosen for evolution narration: **Llama, Qwen, Phi, Gemma, DeepSeek, Mistral, OLMo, BitNet**, plus seven additional family arcs (Falcon, SmolLM, Granite, MiniCPM, OpenELM, Hymba/SSM-hybrid, StarCoder) for completeness. Each arc states: (a) the generations, (b) what changed at each transition, (c) which architectural axes flipped, (d) the inferred motivation.

### 5.1 Llama — Feb 2023 → Dec 2024

LLaMA 1 7B (2023-02): MHA, RoPE θ=10000, RMS pre-norm, SwiGLU, SP-32k, no bias, untied, context 2048. The "Llama default" was MHA — not GQA. → **LLaMA 2 7B (2023-07)**: identical architecture except context 2048→4096; *GQA appeared only at 70B*. → **Llama 3 8B (2024-04)**: first 7B-class with GQA (32/8/128) + Tiktoken regex tokenizer (128k) + RoPE θ raised 10000 → 500000. The tokenizer flip from SP to Tiktoken at Llama 3 is the largest single-axis change in the family's history; it pulls Llama out of the Mistral/Falcon SP-32k cluster and into the Phi-3-small/cl100k cluster. → **Llama 3.1 8B (2024-07)**: introduces the four-parameter `rope_scaling` block (`type=llama3`, `factor=8.0`, `low_freq_factor=1.0`, `high_freq_factor=4.0`, `original_max_position_embeddings=8192`); context jumps 8192→131072. *Llama-3 frequency-band scaling becomes the dominant context-extension recipe of 2024–2025*. → **Llama 3.2 1B/3B (2024-09)**: scales Llama 3.1 *downward*; the 1B/3B sizes use head_dim=64 (1B) or 128 (3B) and *tie embeddings* (untied above 8B). → **Llama 3.3 70B (2024-12)**: post-training update only, no architectural delta. **Axes flipped (1→2→3→3.1): MHA→GQA at 7B; SP→Tiktoken; θ 10k→500k; ctx 2k→128k; tied (small only).** Nothing was abandoned; everything is purely additive.

### 5.2 Qwen — Sep 2023 → Apr 2025

Qwen 1 7B (2023-09): MHA, **QKV bias** (a legacy from pre-GQA NeoX-style configs), RoPE θ=10000, SwiGLU, BPE 151936, ctx 8192. → **Qwen 1.5 7B (2024-02)**: adds GQA hooks but keeps QKV bias; RoPE θ raised to 1e6; ctx 32768. → **Qwen 2 7B (2024-06)**: **drops QKV bias** (verified `attention_bias=null`); finalizes GQA at 28/4/128; vocab grows to 152064 (slightly extended). → **Qwen 2.5 (2024-09)**: 0.5B/1.5B/3B/7B sweep; sub-3B variants *tie embeddings*; 7B *untied*. RoPE θ stays at 1e6. → **Qwen 3 (2025-04)**: introduces **QK-RMSNorm per head_dim, applied PRE-RoPE** (not post-RoPE — the v1 census-critique flagged a pre/post confusion; the modeling source `modeling_qwen3.py:248` confirms PRE-RoPE), and a *decoupled* head_dim (Qwen3-0.6B has hidden_size=1024, num_attention_heads=16, head_dim=128 — so `hidden/H=64 ≠ head_dim=128`, making Qwen3-0.6B the cleanest decoupling example). All Qwen 3 sizes use this. → **Qwen 3 MoE 30B-A3B**: GQA + QK-norm + 128 experts, top-8, zero shared, dense_step=1, `norm_topk_prob=true`. **Axes flipped: QKV bias 1→1.5: kept, 1.5→2: dropped; θ 1→1.5: 10k→1e6; 2.5→3: no QK-norm → QK-norm/head_dim PRE-RoPE; head_dim coupling dropped at Qwen 3.** The QKV-bias removal at Qwen 2 is the family's most consequential cleanup.

### 5.3 Phi — Jun 2023 → Jul 2025

Phi-1 1.3B (2023-06, code-only): MHA + **parallel attn/FFN** (GPT-J inheritance) + partial rotary 0.5 + GPT-2 vocab + LN + GELU (no gate). → **Phi-1.5 1.3B (2023-09)**: data-scaling experiment, same architecture. → **Phi-2 2.7B (2023-12)**: same parallel attn/FFN + LN + partial rotary 0.4 + GELU. Note: Phi-1/1.5/2 all use parallel attention+FFN, a `PhiDecoderLayer`-level choice; the HF config does not expose a `parallel_block` flag (verified). → **Phi-3-mini 3.8B (2024-04)**: switches to **sequential attn/FFN**, RMSNorm pre, full RoPE (θ=10000), SwiGLU **with fused gate_up** projection (a single `gate_up_proj` linear of width `2*intermediate_size`), MS-SP-32k vocab, untied; 4k variant has SWA=2047, 128k variant uses LongRoPE per-dim factor[48]. → **Phi-3-small 7B (2024-04)**: the structural outlier — **gegelu** (not SwiGLU), **LayerNorm** (not RMS), **BlockSparse attention** dense-every-2nd, μP scaling (`mup_width_multiplier=8.0`, `mup_attn_multiplier=1.0`, `mup_embedding_multiplier=10.0`), and cl100k tokenizer at vocab=100352. → **Phi-3.5 mini 3.8B (2024-08)**: as Phi-3 mini but with LongRoPE per-dim factor[48]. → **Phi-3.5-MoE 6.6B-A2B (2024-08)**: 16 experts top-2, no shared — third distinct MoE pattern (after Mixtral and DeepSeek-V2). → **Phi-4 mini 3.8B (2025-02)**: switches to GQA (24/8/128), o200k tokenizer (200064), partial RoPE 0.75 (only 75% of channels rotated), tied embeddings. → **Phi-4-mini-flash 3.8B (2025-07)**: hybrid Mamba+attention, Samba-derived. **Axes flipped (1→3): parallel→sequential sublayer; partial→full RoPE then back to partial at Phi-4 mini; LN→RMS→LN (small only)→RMS; GELU→SwiGLU; SwiGLU separate→fused gate_up; vocab 51k GPT-2→32k MS→100k cl100k→200k o200k; MHA→BlockSparse (Phi-3-small only)→MHA→GQA→hybrid.** Phi is the *single most architecturally heterogeneous* family in the corpus.

### 5.4 Gemma — Feb 2024 → May 2025

Gemma 1 2B (2024-02): **MQA** (kv_heads=1), **head_dim=256** (independent of hidden_size 2048 → Q projection 2048→8×256=2048), **dual norm** (RMS pre+post per sublayer = 4 norms per layer), **GeGLU** (gelu_pytorch_tanh + gate), 256k SentencePiece, tied (default), ctx 8192. → **Gemma 1 7B (2024-02)**: MHA (not MQA — 16/16/256), keeps dual norm + GeGLU. → **Gemma 1.1 (2024-04)**: post-training update, no architectural change. → **Gemma 2 2B (2024-07)**: switches MQA → **GQA** (8/4/256); introduces **SWA every other layer, W=4096**; introduces **logit softcapping** (`attn_logit_softcapping=50.0`, `final_logit_softcapping=30.0`); introduces `query_pre_attn_scalar=256` (replaces `1/sqrt(head_dim)`). → **Gemma 3 1B/4B (2025-03)**: switches alternation pattern from 1:1 to **5:1 SWA:full**; introduces **dual RoPE base** (`rope_local_base_freq=10000` for sliding layers, `rope_theta=1000000` for full layers); SWA window shrinks 4096→512 (1B) or 1024 (4B); vocab grows 256000→262144 (image tokens for multimodal); **softcap dropped** (`final_logit_softcapping=null` AND `attn_logit_softcapping=null` — verified directly from the gemma-3-4b-pt config; v1 census incorrectly stated Gemma 3 inherited Gemma 2 softcap). The dropping of softcapping at Gemma 3 is the *first* case in this family of an axis being abandoned; the inferred reason is that softcap interacted poorly with RLHF training (the survey-paper hypothesis from the Gemma 3 tech report's footnotes is that the `tanh` saturation distorted advantage estimates). → **Gemma 3n E2B/E4B (2025-05)**: introduces **Matformer** elastic-width (per-layer activation slicing at runtime). **Axes flipped: MQA→GQA at 2B; 1:1 SWA alt→5:1 alt; softcap on→off; single θ→dual θ; vocab 256k→262k+image; ctx 8192→32k→131k.**

### 5.5 DeepSeek — Nov 2023 → 2025

DeepSeek LLM 7B (2023-11): MHA, vanilla RoPE θ=10000, RMS pre, SwiGLU, custom 102400 BPE, untied, ctx 4096 (verified). A vanilla Llama-clone with their own vocabulary. → **DeepSeek-Coder 6.7B base (2023-11)**: smaller 32256 vocab, ctx 16384, otherwise identical. → **DeepSeek-Math 7B (2024-02)**: same arch + math RLHF, MHA. → **DeepSeek-V2 / V2-Lite (2024-05/06)**: introduces **Multi-head Latent Attention (MLA)** with `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`. The "nope" portion of Q/K has no RoPE applied; the "rope" portion does. V has a *different* head dim than the rope portion of K — the API can no longer assume `q_head_dim == k_head_dim == v_head_dim`. Also introduces aux-loss-free routing (per-expert bias updates instead of an auxiliary loss); V2-Lite has 64 routed + 2 shared experts top-6, `first_k_dense_replace=1` (first layer is dense). → **DeepSeek-V3 (2024-12, out of scope at 671B)**: 256 experts top-8, 1 shared, aux-loss-free, fp8 training; the architectural milestone of 2024 MoE. → **R1 (2025-01)**: same arch as V3 + RL recipe. → **R1-Distill family (2025-01)**: backbones are NOT DeepSeek architectures — they are Llama-3.1-8B (for the 8B distill) and Qwen2.5-Math (for the 1.5B/7B distills); the *distillation target* is the R1 chain-of-thought, the *architecture* is inherited from the backbone. → **DeepSeek-V3.1 (2025)**: incremental update to V3. **Axes flipped: MHA→MLA at V2; head_dim symmetry q=k=v dropped at V2 (asymmetric q/k_nope/k_rope/v); aux-loss → aux-loss-free routing at V3.** The MLA introduction at V2 is the most consequential single-architecture change in 2024.

### 5.6 Mistral — Sep 2023 → 2025

Mistral 7B v0.1 (2023-09): GQA (32/8/128), **SWA=4096** (verified `sliding_window=4096`), RoPE θ=10000, RMS pre, SwiGLU, SP-32k, untied, ctx 8192. The first widely-adopted GQA+SWA in a 7B. → **Mistral 7B v0.2 (2024-03)**: **SWA dropped** (`sliding_window=null`), ctx extended to 32768, RoPE θ stays at 10000. The inferred motivation (per AllenAI's analysis and community evals): FlashAttention-2 made global attention cheap enough that the SWA tradeoff stopped paying off; SWA also broke needle-in-haystack at long ranges. → **Mistral 7B v0.3 (2024-05)**: keeps no-SWA; raises RoPE θ 10000→1e6; switches vocab 32000→32768 (Tekken/Mistral-v3 SP, the precursor to the Mistral Nemo 131k Tekken tokenizer). → **Mistral Nemo 12B (2024-07, oos)**: 131k Tekken vocab; θ=1e6. → **Mistral Small 3.1 24B (2025-03, oos)**: introduces **trained attention sinks** (a 2025 innovation, also adopted by GPT-OSS). → **Mathstral 7B (2024-07)** and **Codestral Mamba 7B (2024-07)**: derivatives — Codestral Mamba is the only Mamba-2 model from Mistral, used to validate the linear-attention thesis. **Axes flipped: SWA on→off→still off; θ 10k→1e6; vocab 32k SP→32k Tekken→131k Tekken; sinks introduced at 3.1.** The SWA abandonment at v0.2 is the *canonical example* of an architectural feature being adopted and then dropped; it is then *re-adopted* by Gemma 2/3 and Phi-3-mini, which the survey-paper should call out as evidence that "abandoned" features can return when their tradeoff curves shift.

### 5.7 OLMo — Feb 2024 → Nov 2024

OLMo 1 7B (2024-02, verified): MHA (32/32/128), vanilla RoPE θ=10000, **RMS pre-norm** (no QK-norm), SwiGLU, GPT-NeoX 50304 vocab, untied, ctx 2048. The "clean Llama-1-shaped" reference. → **OLMo 1 1B (2024-04)**: smaller scale, same architecture. → **OLMo 2 7B (2024-11)**: switches **pre-norm → post-norm** (every transformer block: norm placed *after* the residual addition, not before the sublayer); introduces **QK-RMSNorm over full head channels** (`q_norm` shape `(num_heads*head_dim,)`, `k_norm` shape `(num_kv_heads*head_dim,)`, applied after Q/K projection, before head reshape — *not* per-head as in Qwen 3); RoPE θ raised 10000→500000; vocab switches GPT-NeoX 50304 → Tiktoken 100352; ctx stays at 4096. **Axes flipped: pre-norm → post-norm; no QK-norm → QK-norm/full channels; θ 10k → 5e5; vocab 50k GPT-NeoX → 100k TT.** OLMo 1 → OLMo 2 is the cleanest pre→post-norm flip in the corpus. The motivation (per the OLMo 2 tech report): post-norm stabilizes long training runs at the cost of slightly higher per-step compute; QK-norm absorbs the loss of stability that post-norm transferred.

### 5.8 BitNet — Mar 2024 (paper) → Apr 2025 (HF)

BitNet b1.58 has one generation in our census (Microsoft's `bitnet-b1.58-2B-4T`, HF released April 2025). Architecture (verified): GQA 20/5/128, hidden_size 2560, 30 layers, vocab 128256 (Llama-3 Tiktoken), tied embeddings, ctx 4096, RoPE θ=500000, **`hidden_act=relu2`** (ReLU squared — the activation defining the family), 1.58-bit ternary weights (`{-1, 0, +1}` per weight), `linear_class=autobitlinear`, `quantization_mode=offline`. The ReLU² activation alone breaks the SwiGLU/GeGLU/SiLU monoculture and justifies a new entry in the activation axis. The 1.58-bit quantization is *training-time*, not post-training — it requires the architecture to ship with bitlinear layers from epoch 0. The family arc is therefore "a single one-shot architectural rewrite", not a generation-to-generation refinement. **Axes flipped (vs Llama-shaped baseline): SwiGLU→ReLU²; fp16/bf16 weights→1.58-bit ternary weights.** As the first production 1-bit-class LM, BitNet establishes the precedent that quantization is an architectural choice with downstream effects on activation, normalization sub-blocks, and embedding behavior.

### 5.9 Falcon — May 2023 → 2025

Falcon 7B (2023-05): **MQA** (71 query, 1 KV — verified `multi_query=true`), **parallel attention/FFN** (verified `parallel_attn=true`), no LayerNorm bias, custom 65024 BPE, vanilla RoPE, head_dim=64, ctx 2048. The first widely-deployed MQA at 7B + the first widely-deployed parallel sublayer. → **Falcon Mamba 7B (2024-08, verified)**: switches to **pure Mamba-1** (64 SSD blocks, `state_size=16`, `expand=16`, `conv_kernel=4`), no attention at all; the first competitive pure-SSM at 7B. → **Falcon 3 family (2024-12, 1B/3B/7B/10B)**: returns to attention, but with major changes: GQA (e.g. 12/4 at 7B), **head_dim=256** (verified from Falcon3-7B config) — the *only non-Gemma family with head_dim=256 at this size*, hidden_size 3072, vocab 131072 (Tekken-style), RoPE θ=1000042 (a curious choice — slightly above 1e6), ctx 32768, untied at 7B/10B, tied at 1B/3B. → **Falcon-H1 7B (2025)**: hybrid Mamba-2 + attention. **Axes flipped: MQA→Mamba→GQA; parallel sublayer→sequential (after Falcon Mamba — Falcon 3 is sequential); vocab Falcon BPE→Mamba's neox→Tekken-style 131k; head_dim 64→256.** Falcon is unusual in *abandoning* and then *re-adopting* attention.

### 5.10 SmolLM — Jul 2024 → Jul 2025

SmolLM v1 135M/360M/1.7B (2024-07): GQA, RMS pre, SwiGLU, SC2 (StarCoder-2) BPE 49152, tied, ctx 8192. SmolLM v1 was a "Hugging Face's clean reference at sub-2B". → **SmolLM2 135M/360M/1.7B (2024-11, verified)**: GQA at the small sizes (9/3, 15/5), but **MHA at 1.7B** (32/32/64) — i.e. SmolLM2 1.7B is the larger size where MHA returns! Vocab unchanged. → **SmolLM3 3B (2025-07)**: **NoPE every 4th layer** (`no_rope_layers=[1,1,1,0]*L/4`); RoPE θ jumps 1.3e5→5e6; vocab switches SC2→LL3 (128256, Tiktoken); ctx 8192→65536; remains tied. **Axes flipped: GQA→MHA→GQA again (1.7B is the MHA outlier); θ 1e5→5e6; SC2→LL3; ctx 8k→65k; NoPE introduction at v3.** SmolLM3's NoPE pattern is the first widely-shipped "per-layer apply_rope" mask in the public corpus.

### 5.11 Granite — Oct 2024 → Apr 2025

Granite 3.0 2B (2024-10): GQA, **four μP scalars** baked in (`embedding_multiplier=12.0`, `attention_multiplier=1/64=0.015625`, `residual_multiplier=0.22`, `logits_scaling=8.0` or `16.0`), RoPE θ=10000, SC2 49152 vocab, tied, ctx 4096. The Granite arc is *not* about generation-level axis changes — it's about Granite repeatedly *increasing rope_theta* to stretch context: → **Granite 3.1 2B (2024-12)**: θ 10000 → 5e6, ctx 4096 → 131072 (32× extension via base θ raise alone, no rope_scaling). → **Granite 3.3 2B/8B (2025-04)**: θ 5e6 → 1e7, ctx unchanged. The μP scalars are unchanged across all generations — they are Granite's signature, derived from the IBM Granite 1.0 7B (out of scope, 2024-08) which inherited them from the MosaicML Granite-13B precursor. **Axes flipped: θ 10k→5e6→1e7 alone; everything else stable.** Granite is the family showing that "base θ extension only" (without any rope_scaling block) is a viable 32× context recipe.

### 5.12 MiniCPM — Feb 2024 → Sep 2024

MiniCPM 1 2.4B (2024-02): MHA, RoPE θ=10000, RMS pre + μP (`scale_emb=12`, `scale_depth=1.4`, `dim_model_base=256` for `logit_scale = 1/(hidden_size/dim_model_base)`), SwiGLU, MiniCPM SP 122753, tied, ctx 4096. The μP scalars at MiniCPM are *inherited* from the MiniCPM-scaling-laws paper (arXiv:2404.06395, "Scaling laws for the same data"). → **MiniCPM 2 / 2.4B (2024-04)**: incremental update, μP unchanged, "Dragonfly" model merging recipe. → **MiniCPM 3 4B (2024-09)**: switches to **MLA** (`q_lora_rank=768`, `kv_lora_rank=256`, `qk_nope_head_dim=64`, `qk_rope_head_dim=32`) — the first non-DeepSeek production model with MLA; introduces LongRoPE for context extension; vocab shrinks 122753→73448. **Axes flipped: MHA→MLA at v3; vocab shrinks; μP scalars retained throughout.**

### 5.13 OpenELM — Apr 2024 (single shot)

OpenELM 270M / 450M / 1.1B / 3B (2024-04, verified for 3B): four sizes shipped simultaneously, all with the *same per-layer scaling recipe*: `num_query_heads[]` is a per-layer list (e.g. `[12,12,12,12,16,16,...,24,24]` for 3B with 36 layers), `num_kv_heads[]` is a per-layer list (`[3,3,3,3,4,4,...,6,6]` for 3B), `ffn_multipliers[]` is a per-layer list (`[0.5, 0.6, ..., 4.0]` for 3B — i.e. the FFN expands monotonically with depth). Also: tied embeddings, QK-norm (verified `normalize_qk_projections=true`), Swish activation, head_dim=128 (a Gemma-1-style decoupling — head_dim is fixed across all layers even though the number of heads varies). OpenELM is the family that *cannot be expressed* in a uniform-layer schema: a Llama-shaped IR with single `num_attention_heads` / `num_key_value_heads` scalars will silently produce a corrupted OpenELM model. **Axes used: per-layer Q/KV/FFN scaling (A12), QK-norm, tied embeddings.** Per OpenELM paper (arXiv:2404.14619), the per-layer schedule was found via Bayesian optimization to balance accuracy and parameter count.

### 5.14 Hymba and the SSM-hybrid family (parallel branch evolution)

Hymba 1.5B (NVIDIA, 2024-11, verified): the first model to run **Mamba and attention as parallel heads inside one block** (vs Jamba's sequential periodic substitution, vs Zamba2's periodic shared attention). Architecture: 32 layers, hidden=1600, 25 attention heads + 5 KV (GQA), 32001 vocab; sliding window=1024 with global attention at layers 0, 15, 31; `mamba_d_state=16`, `mamba_d_conv=4`, `mamba_expand=2`, `mamba_dt_rank=100`; KV reuse groups across 14 groups (e.g. [1,2], [3,4]). The parallel-head topology is qualitatively different from Jamba/Zamba2 — it means every layer has both an SSM and an attention contribution to the residual stream. **Other 2024–2025 hybrid SSM family members for comparison: Jamba v0.1 (sequential, 1-in-8 attn + 1-in-2 MoE), Zamba2 (periodic shared attention block), Phi-4-mini-flash (Samba-derived, hybrid Mamba+attention from Microsoft), Falcon-H1 (Mamba-2 + attn).** Four distinct hybrid topologies in 18 months.

### 5.15 StarCoder — May 2023 → Feb 2024

StarCoder 1 7B (2023-05): **MQA** (the 2023 reference for MQA at 7B), absolute learned positional embeddings (no RoPE!), LN + bias on QKV+MLP, gated GELU-tanh MLP, BigCode 49152 BPE, untied, ctx 8192. → **StarCoder 2 3B/7B (2024-02, verified)**: switches MQA → **GQA** (24/2/128 for 3B, 36/4/128 for 7B), absolute learned → RoPE θ≈1e6, ctx 16384 / SWA=4096; keeps `use_bias=true` (the only modern decoder family that keeps biases on attention+MLP), keeps gated GELU-tanh, keeps LayerNorm (not RMS), tokenizer unchanged. **Axes flipped: MQA→GQA; absolute pos → RoPE; ctx 8k→16k+SWA.** StarCoder 2 is one of the rare *backwards counter-examples* — it kept biases when everyone else dropped them, and it kept LayerNorm when everyone else moved to RMS.

---

## 6. Axis Catalog (15 Axes)

Formal definitions, enumerations, and model mappings.

### Axis A1 — Attention type

The structural form of the attention operator. Enumeration: `{MHA, GQA, MQA, MLA, SWA, BlockSparse, Linear/SSM-{Mamba1, Mamba2, RWKV6, RWKV7, Griffin/Hawk, xLSTM}, Hybrid-{sequential, periodic, parallel}}`. Defining parameters: `num_attention_heads`, `num_key_value_heads`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `sliding_window`, `sliding_window_pattern`, `layer_types[]`, `state_size`, `conv_kernel`, `mamba_d_state`, `mamba_d_conv`, `mamba_expand`, `block_types[]` (RecurrentGemma), `attn_implementation`. Per-token-per-layer cache cost: MHA `2·H·d`; GQA `2·H_kv·d`; MQA `2·d`; MLA `kv_lora_rank + qk_rope_head_dim` (DeepSeek-V2-Lite: `512+64 = 576` per token per layer, vs `2·16·128 = 4096` for MHA at the same shape — ~7× reduction; DeepSeek-V3 full at 128 heads: `512+64 = 576` vs `2·128·128 = 32768`, ~57× reduction; the v1 census-critique fact-check correctly noted that the V3 compression ratio is ~71× when computed against the full V3 architecture, not the V2-Lite). SWA: bounded `min(seq, W) · 2·H_kv·d`. SSM: constant-size state independent of sequence (e.g. Mamba: `state_size · expand · hidden_size`).

### Axis A2 — Positional encoding scheme

Enumeration: `{none, absolute_learned, ALiBi, RoPE-vanilla, RoPE-Llama3-bandscaling, RoPE-LongRoPE, RoPE-YaRN, RoPE-DynamicNTK, RoPE-linear, RoPE-dual (Gemma 3 per-layer-type), M-RoPE (Qwen 2.5-VL 3-axis), MLA-split (NoPE+RoPE per head channel)}`. Defining configs: `rope_theta`, `rope_scaling`, `rope_local_base_freq`. **2023 contrast (now visible in v2):** MPT 7B and Baichuan 1 use **ALiBi**; StarCoder 1 uses **absolute learned**; the dominance of RoPE is a 2023→2024 trend, not a 2023 starting condition. **2024–2025 fragmentation:** at least 11 distinct RoPE variants in production simultaneously.

### Axis A3 — Normalization (type + placement + QK-norm)

Norm type: `{RMSNorm, LayerNorm, none}`. Placement: `{pre, post, dual (pre+post per sublayer), pre-norm with sublayer-output-norm (Gemma)}`. QK-norm: `{none, per_head_dim (Qwen 3), per_full_channels (OLMo 2), QK-LayerNorm (StableLM 2 12B)}`. **Critical pre/post difference**: OLMo 2 is the only mainstream 2024–2025 model that uses pure post-norm (norm applied *after* residual addition, not before sublayer); every other dense model is pre-norm. Gemma 2/3 / CodeGemma use *dual* norm (RMS before sublayer input *and* RMS on sublayer output before residual addition) — this is sometimes mislabeled "sandwich norm" but is structurally pre+post, not the original CogView sandwich. QK-norm placement: Qwen 3 applies QK-norm **PRE-RoPE** (verified in `modeling_qwen3.py:248`); Gemma 3 applies QK-norm **POST-RoPE**. The two patterns are *not* equivalent.

### Axis A4 — FFN family (gate topology, not activation)

Enumeration: `{ungated (LN-bias-style MLP), gated (X-GLU: SwiGLU/GeGLU/gegelu/gated-GELU-tanh), gated fused (SwiGLU fused gate_up: Phi-3), MoE-{SwiGLU-MoE, GeGLU-MoE}, RWKV channel-mix (gated by time-mix output)}`. Note: **Activation function** is axis A14, separate. Gate topology axis is about the structural form of the FFN block: whether `down(act(gate(x)) * up(x))` (gated) vs `down(act(up(x)))` (ungated) vs `down(act(fused_gate_up(x).split(2, -1)))` (fused). The fused-gate variant matters for quantization: a single weight matrix of width `2·intermediate_size` must be quantized as one tensor.

### Axis A5 — MoE routing

Enumeration: `{none (dense), top-k softmax + aux-loss (Mixtral, OLMoE, Phi-3.5-MoE), top-k softmax + aux-loss + shared (DeepSeek-V2-Lite: 64 routed + 2 shared, top-6), top-k sigmoid + aux-loss-free + shared (DeepSeek-V3: 256 routed + 1 shared, top-8), top-k softmax + norm-topk-prob + zero-shared (Qwen 3 MoE: 128 routed + 0 shared, top-8, `norm_topk_prob=true`)}`. Defining configs: `n_routed_experts`, `n_shared_experts`, `num_experts_per_tok`, `decoder_sparse_step`, `first_k_dense_replace`, `norm_topk_prob`, `router_aux_loss_coef`, router activation (sigmoid vs softmax). The MoE arc 2023→2025: Mixtral 8×7B (Dec 2023, 8/0/top-2) → DeepSeek-V2 (May 2024, 256/2/top-6) → Qwen 3 MoE (Apr 2025, 128/0/top-8 + norm_topk_prob) → DeepSeek-V3 (Dec 2024, 256/1/top-8 + aux-loss-free + fp8).

### Axis A6 — KV-cache shape and update policy

Per the attention type, the cache shape varies. New variants surfaced by v2: Mamba-1 vs Mamba-2 differ in cache (`mamba2` is SSD: `(n_groups, d_state)`); RWKV-6 vs RWKV-7 differ (RWKV-7 uses delta-rule update: `(num_heads, head_size, head_size)` state); Griffin/Hawk uses `(lru_width,)` state; Mamba SSM cache also has a `conv_state` of `(d_conv-1, expand·d)` for the local conv1d kernel. The unified IR needs both `kv_cache: Tensor` and `ssm_state: dict[str, Tensor]` with per-variant update functions.

### Axis A7 — Tokenizer/vocab family

15 distinct tokenizer flavors in v2 (vs 14 in v1, adding Tekken/Mistral-v3, Falcon-Tekken-style, Cohere). Vocab range: 32000 (LLaMA 1/2) → 262208 (Gemma 3). The vocab×hidden product as a fraction of total params is now a first-class concern: Llama 3.2 1B has `128256·2048 = 262M` for the embedding alone, which is ~25% of total parameters; this affects the embedding-tying decision (small models tie to avoid double-paying), the quantization decision (per-tensor scale for the embedding is suboptimal at this size), and the compile target (the embedding gather is often offloaded to a separate kernel).

### Axis A8 — Scalar multipliers (μP-style rescalings)

Five multipliers: `embedding_multiplier` (Granite 12, MiniCPM `scale_emb=12`); `attention_multiplier` / `query_pre_attn_scalar` (Granite 1/64, Gemma 2/3 `query_pre_attn_scalar=256`, Phi-3-small μP); `residual_multiplier` (Granite 0.22, MiniCPM `scale_depth/sqrt(L)`); `logits_scaling` (Granite 8 or 16, MiniCPM `hidden_size/dim_model_base`, Cohere Aya `logit_scale=0.0625`); `softcap` (Gemma 2 only; Gemma 3 dropped both attn and final softcap — verified). The Granite scalars are baked into a model that otherwise looks like Llama — they cannot be inferred from the `LlamaConfig`-named architecture. **The v1 census-critique was correct that Gemma 3 dropped softcapping**; v2 corrects axis A8 to show Gemma 3 with `softcap = None`.

### Axis A9 — Bias presence

`{none anywhere (Llama, Qwen 2/2.5/3, Mistral, Yi, InternLM, SmolLM, Granite, OLMo, Gemma, BitNet, MobileLLM, Falcon 3), QKV bias only (Qwen 1, StableLM 2 1.6B), full bias on QKV + MLP + LN (StarCoder 1, StarCoder 2 — `use_bias=true`, Florence-2 BART), LN bias only (MPT, Pythia)}`. Phi-2 has biases on the parallel sublayers. The 2023→2024 trend was clear bias removal; StarCoder 2 (which kept them) is the explicit counter-example.

### Axis A10 — Embedding tying

`tie_word_embeddings ∈ {y, n}`. Empirical pattern: <3B tends to tie, >4B tends to untie. v2 surfaces the **Cohere Aya 8B counter-example** (tied at 8B), the **Llama 3.1 8B and Qwen 3 8B counter-examples** (both untied at 8B but tied below). Promoted from v1's sub-axis treatment to a first-class axis because the parameter-count fraction is large (e.g. 128256·2048 = 263M = 25% of a 1B model).

### Axis A11 — Per-layer **type** alternation

Distinct from A12. `{uniform, SWA/full alternation (Gemma 2: 1:1, Gemma 3: 5:1, SmolLM3, Phi-3-mini), dense/MoE alternation (DeepSeek-V2-Lite `first_k_dense_replace=1`), Mamba/attention periodic (Jamba: 1-in-8 attn + 1-in-2 MoE), Mamba/shared-attn periodic (Zamba2), parallel-branch-per-layer (Hymba), block_types[] generic (RecurrentGemma)}`. Required for any layer-IR that hopes to express Gemma 3 / Zamba2 / Jamba / Hymba.

### Axis A12 — Per-layer **shape** scaling (NEW)

OpenELM is the canonical exemplar (verified): `num_query_heads[]` is a per-layer list (12→24 monotone for 3B), `num_kv_heads[]` is a per-layer list (3→6), `ffn_multipliers[]` is a per-layer list (0.5→4.0). MobileLLM 1B/1.5B introduces **block-wise weight sharing** (the same Q/K/V/MLP weights are tied across adjacent layers — a different kind of per-layer variation: same shape, shared parameters). Gemma 3n introduces **Matformer** elastic-width subsetting (a fifth variant: shape varies *at runtime*, not at training time). Cited: OpenELM paper (arXiv:2404.14619). This axis is qualitatively different from A11 because A11 varies the *operator type* per layer while A12 varies the *shape parameters* per layer.

### Axis A13 — Parallel vs sequential sublayer (NEW)

`{sequential (Llama default), parallel (Phi-1/1.5/2, Falcon 7B, Pythia, GPT-J, GPT-NeoX), parallel-branches-per-block (Hymba: Mamba ‖ attention in one block, RWKV: channel-mix and time-mix), parallel-with-fused-input (Phi-2 specifically fuses the QKV+MLP-up input projections)}`. Sequential means `h = h + Attn(Norm(h)); h = h + FFN(Norm(h))`. Parallel means `h = h + Attn(Norm(h)) + FFN(Norm(h))` — both branches read the *same* pre-norm input and their outputs sum into the residual. Falcon 7B has `parallel_attn=true` (verified). Phi-1/1.5/2 inherit parallel attention/FFN from GPT-J's original implementation (this is a `PhiDecoderLayer`-level choice in `modeling_phi.py`, not a config flag — the HF config does not expose `parallel_block` for Phi-2). The sequential vs parallel distinction was *abandoned* by Llama-shaped models in 2024 (Phi-3 switched to sequential) but *re-adopted* by Hymba and RWKV families in 2024–2025. Cited: GPT-J (Wang & Komatsuzaki, 2021), Falcon paper (arXiv:2306.01116), Phi-2 model card (Microsoft, Dec 2023).

### Axis A14 — Activation function (NEW, standalone)

`{SiLU (Llama, Mistral, Qwen, OLMo, SmolLM, Granite, Yi, InternLM, MiniCPM, StableLM, Phi-3+), GELU (Phi-1/1.5/2, MPT, Falcon 7B, Florence-2, Pythia, Zamba2 MLP), GELU-tanh (Gemma, CodeGemma, StarCoder 2, gegelu intermediate), GELU-new (Phi-2 specifically), gegelu (Phi-3-small paired-channel gated), ReLU² (BitNet b1.58 — verified `hidden_act=relu2`), sigmoid+ReLU² (RWKV-6/-7 channel-mix), Swish (OpenELM `activation_fn_name=swish`)}`. v1 buried this inside FFN family; v2 promotes to standalone because BitNet's ReLU² and Gemma's GELU-tanh are *not* derivable from FFN topology alone. Note: SiLU and Swish are the same function (`x·sigmoid(x)`), but the two HF config names exist independently and are sometimes treated as distinct in implementations.

### Axis A15 — RoPE rotation domain (NEW)

`{full rotary (Llama, Qwen, Mistral, OLMo, Gemma — all channels rotated), partial rotary 0.4 (Phi-2 — only 40% of head_dim channels are rotated, applied to the leading `0.4·head_dim` channels), partial rotary 0.5 (Phi-1/1.5), partial rotary 0.25 (StableLM 3B, StableLM 2 1.6B, StableLM 2 12B), partial rotary 0.75 (Phi-4 mini — verified `partial_rotary_factor=0.75`), NoPE/RoPE split per head (MLA: `qk_nope_head_dim` channels NoPE, `qk_rope_head_dim` channels RoPE), NoPE per layer (SmolLM3 — `no_rope_layers=[1,1,1,0]` mask, every 4th layer has zero RoPE), no rotation (SSM/linear-attention models)}`. This is qualitatively different from A2 (which is about the **scaling method** — Llama-3 vs YaRN vs LongRoPE vs Dynamic NTK); A15 is about **which channels get rotated and how much**. v1 conflated the two.

### Bonus axis A16 (noted, not separately enumerated) — Context-extension method

Six methods enumerated in v1 (Llama-3 freq-band, LongRoPE per-dim, YaRN, Dynamic NTK, Linear scaling, dual-RoPE), plus three more surfaced by v2: **DCA** (Dual Chunk Attention, ChatGLM3 32k), **PI** (Position Interpolation, original Llama-2 long-context method), **extend-base-θ-only** (Qwen 2.5 raised θ to 1e7 without `rope_scaling`; Granite 3.3 same recipe). The "extend base θ only" recipe is the simplest viable 32× context extension; Granite 3.0→3.3's θ 10000→1e7 transition demonstrates this in production.

---

## 7. Cross-org Comparison Table (one-page at-a-glance)

Each row is one vendor; each column is one axis (collapsed). "✓" means the vendor uses this in 2024–2025 production; "—" means absent; "→" means generation transition; "?" means closed/unverified.

| Vendor | Attn (2024 norm) | RoPE family | Norm | FFN/Activation | Quantization-native | MoE | Hybrid SSM | μP scalars | Per-layer scaling | Parallel sublayer |
|---|---|---|---|---|---|---|---|---|---|---|
| Meta | GQA | Llama-3 (band) | RMS pre | SwiGLU/SiLU | – | – (until Llama 4) | – | – | – | – (sequential) |
| Alibaba (Qwen) | GQA + QK-norm/head_dim | vanilla θ=1e6 | RMS pre + QK-norm pre-RoPE | SwiGLU/SiLU | – | top-8/0-shared/norm_topk | – | – | – | – |
| Microsoft (Phi) | MHA→GQA→hybrid | LongRoPE | RMS pre | SwiGLU fused; gegelu (small only) | BitNet b1.58 (ReLU²) | top-2/0-shared | Phi-4-mini-flash (Samba) | μP at Phi-3-small | – | parallel (Phi-1/2) → sequential |
| Google (Gemma) | MQA→GQA + SWA | dual θ (Gemma 3) | RMS pre+post dual | GeGLU/GELU-tanh | – | – | RecurrentGemma (Griffin/Hawk) | softcap (dropped at v3); q_pre_attn=256 | Matformer (3n) | – |
| DeepSeek | MHA→MLA | YaRN | RMS pre | SwiGLU | – | top-6/2-shared, top-8/1-shared aux-loss-free | – | – | – | – |
| Mistral | GQA (SWA tried, dropped) | vanilla θ=1e6 | RMS pre | SwiGLU/SiLU | – | – | Codestral Mamba (Mamba-2) | – | – | – |
| AI2 (OLMo) | MHA | vanilla θ=5e5 | RMS post + QK-norm/full | SwiGLU | – | OLMoE 0-shared top-8 | – | – | – | – |
| Apple (OpenELM) | GQA per-layer list | vanilla θ=10000 | RMS pre + QK-norm | SwiGLU/Swish | – | – | – | – | **per-layer Q/KV/FFN** | – |
| IBM (Granite) | GQA + μP | vanilla θ=10k→1e7 | RMS pre + μP | SwiGLU | – | – | – | **four μP scalars** | – | – |
| OpenBMB (MiniCPM) | MHA→MLA | LongRoPE | RMS pre + μP | SwiGLU | – | – | – | scale_emb, scale_depth | – | – |
| Stability AI (StableLM) | MHA + partial RoPE | partial 0.25 | LN + QK-LN | SwiGLU | – | – | – | – | – | – |
| 01-AI (Yi) | GQA | vanilla θ=5e6 | RMS pre | SwiGLU | – | – | – | – | – | – |
| InternLM | GQA | dynamic NTK | RMS pre | SwiGLU | – | – | – | – | – | – |
| HuggingFace (SmolLM) | GQA, MHA at 1.7B | vanilla, NoPE per layer (v3) | RMS pre | SwiGLU | – | – | – | – | – | – |
| AI21 (Jamba) | GQA + Mamba-1 | vanilla on attn-only | RMS pre | MoE (16/0/top-2) | – | top-2/0-shared periodic | **Jamba (1:8 attn periodic)** | – | – | – |
| Zyphra (Zamba2) | Mamba-2 + shared attn | RoPE on shared attn | RMS pre | gelu MLP | – | – | **Zamba2 (periodic shared attn)** | – | – | – |
| NVIDIA (Hymba) | Parallel Mamba + attn | NTK | RMS pre | SwiGLU | – | – | **Hymba (parallel Mamba ‖ attn)** | – | – | parallel branches |
| TII (Falcon) | MQA → Mamba → GQA hd=256 | vanilla θ=1000042 | RMS pre | SwiGLU/GELU MLP | – | – | Falcon Mamba, Falcon-H1 | – | – | parallel (Falcon 7B) → sequential |
| BigCode (StarCoder) | MQA → GQA + bias | absolute → vanilla θ=1e6 | LN pre | gated GELU-tanh + bias | – | – | – | – | – | – |
| Cohere (Aya) | GQA | vanilla θ=4e6 | LN pre | SwiGLU | – | – | – | logit_scale=0.0625 | – | – |
| RWKV-LM | RWKV-6 → RWKV-7 (delta) | – | LN | RWKV channel-mix | – | – | (linear attn family) | – | – | parallel (channel-mix + time-mix) |
| Tsinghua (GLM) | MHA → MQA → MQA | vanilla | RMS | SwiGLU/GeGLU | – | – | – | – | – | – |
| Baichuan | MHA + ALiBi → RoPE | none → vanilla | LN → RMS | SwiGLU | – | – | – | – | – | – |
| Meta (MobileLLM) | GQA + deep-narrow + weight-share | vanilla | RMS pre | SwiGLU | – | – | – | – | block-wise weight sharing | – |
| Apple (closed) | ? (3B on-device) | ? | ? | ? | – (FP) | – | ? | per-layer rank-1 adapters (reported) | ? | – |

---

## 8. Surprises and Non-obvious Patterns

(Numbered to match v1 where applicable; new entries marked **NEW**.)

1. **Qwen 3 and OLMo 2 use *different* QK-norm shapes** (preserved from v1). Qwen 3 normalizes per-head (`head_dim`) and applies it **pre-RoPE** (verified in source). OLMo 2 normalizes per-full-channels (`num_heads·head_dim`) and applies it after Q/K projection before reshape. These look textually similar in the configs but the operator is fundamentally different.

2. **OLMo 2 is post-norm**, not pre-norm (preserved from v1). This requires the API to support `norm_position ∈ {pre, post, dual}` per sublayer.

3. **Gemma 3 dropped softcapping entirely.** (FIXED from v1 error.) Both `final_logit_softcapping=null` and `attn_logit_softcapping=null` in the verified Gemma 3 config. v1 incorrectly stated Gemma 3 inherited Gemma 2 softcap. The Gemma 3 family also kept `query_pre_attn_scalar=256` (replacing `1/sqrt(head_dim)`); that part is unchanged.

4. **TinyLlama is GQA**, not MHA. (FIXED from v1 internal contradiction.) Verified `num_attention_heads=32, num_key_value_heads=4`. v1's axis 4.1 listed it under MHA "(KV=4 actually GQA, see below)" — the parenthetical was correct, the header row was wrong; v2 puts TinyLlama under GQA consistently.

5. **The cleanest head_dim-decoupling example is Qwen3-0.6B**, not Phi-3-mini. (FIXED from v1 Surprise #12 error.) Qwen3-0.6B: `hidden_size=1024, num_attention_heads=16, head_dim=128` — so `hidden/H = 64 ≠ head_dim = 128`. Phi-3-mini has `hidden_size=3072, num_attention_heads=32, head_dim=96 = 3072/32` (coupled). Phi-4-mini has `hidden_size=3072, num_attention_heads=24, head_dim=128 = 3072/24` (coupled). The head_dim-decoupling family is **Gemma 1/2/3 (all sizes, head_dim=256) and Qwen 3 (all sizes, head_dim=128)**; Phi-3.x and Phi-4 are NOT decoupling examples.

6. **MQA is alive at the smallest sizes** (preserved from v1): Gemma 3 1B, CodeGemma 2B, PaliGemma. Gemma 1 7B used MHA but Gemma 1 2B used MQA.

7. **Phi-3-small is the structural outlier of the dense pack** (preserved).

8. **SmolLM3 NoPE every 4th layer** (preserved).

9. **Vocab sizes range 4×** (preserved).

10. **NEW: OpenELM exposes per-layer variation as a list-of-ints in the config** — `num_query_heads`, `num_kv_heads`, and `ffn_multipliers` are all per-layer lists (verified). This is the single most extreme departure from "uniform-layer schema" in the corpus. Any IR that hard-codes scalar `num_attention_heads` will silently corrupt OpenELM.

11. **NEW: BitNet b1.58 uses ReLU², not SwiGLU/GeGLU.** Activation function is therefore a *standalone* axis, not derivable from FFN topology. Quantization-native architectures (BitNet, GPT-OSS with MXFP4 weights) appear to be opening a new branch of the family tree where the activation choice is constrained by the quantization recipe.

12. **NEW: Mistral SWA was tried and abandoned, then re-adopted by other vendors.** v0.1 → v0.2 dropped SWA; Gemma 2/3 and Phi-3-mini re-adopted it. The "abandoned" verdict was generation-local, not field-wide. This is the cleanest survey-paper example of a feature whose adoption depends on which kernel acceleration was available at the time.

13. **NEW: ALiBi was tried and abandoned.** MPT 7B (May 2023) and Baichuan 1 (Jun 2023) used ALiBi; Baichuan 2 (Sep 2023) flipped to RoPE; no 2024 mainstream model ships ALiBi at <8B. The single counter-example in adjacent space is xLSTM 7B which uses neither RoPE nor ALiBi (it relies on the recurrence itself for position).

14. **NEW: Parallel attention/FFN was tried and abandoned at scale, then re-adopted by hybrid architectures.** Phi-1/1.5/2 (parallel via GPT-J inheritance) → Phi-3 switched to sequential. Falcon 7B (parallel) → Falcon 3 sequential. RWKV (parallel channel-mix + time-mix from the beginning) and Hymba (parallel Mamba ‖ attention) are the modern revivals. The motivation for the original parallel design was throughput (one Norm input feeds both branches); the motivation for the modern revival is composability of qualitatively different operators (SSM with attention).

15. **NEW: MoE shared-experts policy is a 3-way split.** Mixtral (Dec 2023, 0 shared) → DeepSeek-V2 (May 2024, 2 shared) → Qwen 3 MoE (Apr 2025, 0 shared with `norm_topk_prob=true`) → DeepSeek-V3 (Dec 2024, 1 shared with aux-loss-free routing). The shared-experts decision has not converged.

16. **NEW: Embedding tying breaks at 8B for Cohere Aya.** v1's heuristic "tied below 3–4B, untied above" is broken by Aya Expanse 8B (tied) and by Llama 3.1 8B and Qwen 3 8B (untied at 8B but tied below). The rule is approximate; vendors make this choice per-family.

17. **NEW: RoPE base θ is the simplest context-extension dial.** Granite 3.0→3.3 raised θ 10000→1e7 without any rope_scaling block. Qwen 2.5 stayed at θ=1e6 with ctx=131072 (no scaling). The 3-way comparison Llama-3.1 (factor=8 band scaling) vs Granite (raw θ raise) vs Qwen 2.5 (raw θ raise) — all reach ~128k context with comparable benchmarks — suggests that the simplest recipe is often sufficient and that the proliferation of named methods (YaRN, LongRoPE, Dynamic NTK) may be solving an optimization that is increasingly less load-bearing.

18. **NEW: Falcon 3 is the only non-Gemma family with head_dim=256 at <8B.** Falcon3-7B: `head_dim=256, hidden_size=3072, num_attention_heads=12 → Q projection 3072→12·256=3072` (verified). The Gemma signature head_dim=256 has migrated to Falcon via Tekken-tokenizer convergence and other design choices.

19. **NEW: MLA generalized beyond DeepSeek.** MiniCPM 3 (Sep 2024) is the first non-DeepSeek production model with MLA, signaling that MLA is becoming a portable architectural component (not a DeepSeek-specific trick).

20. **NEW: Hybrid topologies fragmented into four distinct patterns in 18 months.** Jamba (sequential periodic substitution: 1-in-8 attn + 1-in-2 MoE), Zamba2 (sequential periodic shared attn block), Phi-4-mini-flash (Samba-derived sequential), Hymba (parallel Mamba ‖ attn within one block). All four are simultaneously in production at <8B.

---

## 9. Coverage Caveats — Gated, Unverified, and Out-of-scope

**Gated models for which only mirrors are accessible.** The `meta-llama/*`, `google/gemma-*`, `google/paligemma-*`, `google/codegemma-*`, `google/gemma-3-*`, `mistralai/*` (some), `microsoft/Phi-3-medium*`, and `ai21labs/Jamba-1.5-Mini` repos all return HTTP 401 to unauthenticated fetches. v2 uses authoritative mirrors (`unsloth/`, `deepseek-ai/`, `state-spaces/`, `Zyphra/`, etc.) where available and notes the dependency. Where no mirror exists, the architecture is reported from the vendor's tech report only and marked with a † symbol.

**Models cataloged but not fully verified:**
- Apple Foundation Model ~3B on-device (closed; WWDC 2024 announcement; per-layer rank-1 LoRA adapters reported but unverifiable);
- Anthropic Claude Haiku, Gemini Nano, GPT-4.1-nano (all closed);
- Falcon-H1 7B (announced 2025; verified `model_type=falcon_h1` exists in transformers but full config not fetched in this pass);
- xLSTM 7B (NXAI, late 2024 / 2025) — config exists but not pulled;
- Samba (Microsoft research, June 2024) — academic, upstream of Phi-4-mini-flash;
- DeepSeek-V3.1 (mentioned; full config not pulled in this pass);
- DeepSeek-R1-Zero (full 671B — out of size scope but architecturally same as V3).

**Specifically not catalogued but architecturally relevant:**
- Mixtral 8×7B (13B active, out of scope, but the architectural baseline for the entire MoE arc);
- DBRX (out of scope at 132B/36B active);
- DeepSeek-V3 671B (out of scope);
- MiniMax-Text-01 (out of scope; no <8B variant);
- xLSTM 7B (not pulled).

**Models added v2 but not in v1's coverage:** 35 new rows (15 from 2023, 20 from 2024-2025 critique fixes).

**Note on Jamba citation:** v1 cited `ai21labs/Jamba-tiny-dev` (a developer-fixture config with `num_experts=1`); v2 cites `ai21labs/Jamba-v0.1` (verified: 16 experts, 32 layers, attn_layer_period=8, expert_layer_period=2). Tech report: arXiv:2403.19887.

**Note on CodeGemma 7B citation:** v1 cited `(gemma-1 family card)` with no working URL; v2 cites `https://huggingface.co/unsloth/codegemma-7b` (verified). CodeGemma 7B is MHA (16/16/256), not GQA — v1 incorrectly listed GQA.

---

## 10. Source Citations

**Tech reports (one per major family or generation):**
- LLaMA 1: Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", arXiv:2302.13971 (2023-02).
- LLaMA 2: Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models", arXiv:2307.09288 (2023-07).
- Llama 3 herd: Grattafiori et al., "The Llama 3 Herd of Models", arXiv:2407.21783 (2024-07).
- Qwen: Bai et al., "Qwen Technical Report", arXiv:2309.16609 (2023-09).
- Qwen 2.5: Qwen Team, "Qwen 2.5 Technical Report", arXiv:2412.15115 (2024-12).
- Qwen 3: Qwen Team, "Qwen 3 Technical Report", arXiv:2505.09388 (2025-05).
- Phi-2: Microsoft, "Phi-2: The surprising power of small language models", Microsoft Research blog (2023-12).
- Phi-3 technical report: Abdin et al., "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone", arXiv:2404.14219 (2024-04).
- Phi-4: Abdin et al., "Phi-4 Technical Report", arXiv:2412.08905 (2024-12).
- Gemma 1: Gemma Team, "Gemma: Open Models Based on Gemini Research and Technology", arXiv:2403.08295 (2024-02).
- Gemma 2: Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size", arXiv:2408.00118 (2024-07).
- Gemma 3: Gemma Team, "Gemma 3 Technical Report", arXiv:2503.19786 (2025-03).
- RecurrentGemma: Botev et al., "RecurrentGemma: Moving Past Transformers for Efficient Open Language Models", arXiv:2404.07839 (2024-04).
- DeepSeek LLM: DeepSeek-AI, "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism", arXiv:2401.02954 (2024-01).
- DeepSeek-V2: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", arXiv:2405.04434 (2024-05).
- DeepSeek-V3: DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437 (2024-12).
- DeepSeek-R1: DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv:2501.12948 (2025-01).
- Mistral 7B: Jiang et al., "Mistral 7B", arXiv:2310.06825 (2023-10).
- Mixtral 8×7B: Jiang et al., "Mixtral of Experts", arXiv:2401.04088 (2024-01).
- OLMo 1: Groeneveld et al., "OLMo: Accelerating the Science of Language Models", arXiv:2402.00838 (2024-02).
- OLMo 2: OLMo Team, "2 OLMo 2 Furious", arXiv:2501.00656 (2024-12 paper / 2025-01 v2).
- SmolLM: Allal et al., "SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model", arXiv:2502.02737 (2025-02).
- Granite: IBM Granite Team, "Granite 3.0 Language Models", IBM tech report (2024-10).
- MiniCPM: Hu et al., "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies", arXiv:2404.06395 (2024-04).
- MiniCPM 3: OpenBMB, "MiniCPM-3-4B Technical Report" (2024-09, HF model card).
- StableLM 2: Bellagente et al., "Stable LM 2 1.6B Technical Report", arXiv:2402.17834 (2024-02).
- StableLM 3B-4E1T: Stability AI, "StableLM 3B-4E1T" model card (2023-09).
- OpenELM: Mehta et al., "OpenELM: An Efficient Language Model Family with Open Training and Inference Framework", arXiv:2404.14619 (2024-04). **Cited for axis A12 — per-layer width/depth scaling.**
- MobileLLM: Liu et al., "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases", arXiv:2402.14905 (2024-02). **Cited for block-wise weight sharing in axis A12.**
- BitNet b1.58: Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", arXiv:2402.17764 (2024-02) and "BitNet b1.58 2B4T Technical Report", arXiv:2504.12285 (2025-04). **Cited for ReLU² activation in axis A14.**
- Mamba: Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv:2312.00752 (2023-12).
- Mamba 2: Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", arXiv:2405.21060 (2024-05).
- Zamba2: Glorioso et al., "Zamba: A Compact 7B SSM Hybrid Model", arXiv:2405.16712 (2024-05) and Zamba2 model card (2024-10).
- Jamba: Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", arXiv:2403.19887 (2024-03). **Replaces v1's `Jamba-tiny-dev` citation.**
- RWKV-6 (Finch): Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", arXiv:2404.05892 (2024-04).
- RWKV-7 (Goose): RWKV Team, "RWKV-7 Goose with Delta Rule" (2025-01).
- StarCoder 1: Li et al., "StarCoder: may the source be with you!", arXiv:2305.06161 (2023-05).
- StarCoder 2: Lozhkov et al., "StarCoder 2 and The Stack v2: The Next Generation", arXiv:2402.19173 (2024-02).
- Yi 1.5: 01-AI, "Yi: Open Foundation Models by 01-AI", arXiv:2403.04652 (2024-03).
- InternLM 2: Cai et al., "InternLM2 Technical Report", arXiv:2403.17297 (2024-03).
- ChatGLM2: GLM Team, "ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools", arXiv:2406.12793 (2024-06).
- Baichuan: Yang et al., "Baichuan 2: Open Large-scale Language Models", arXiv:2309.10305 (2023-09).
- Falcon 1: Almazrouei et al., "The Falcon Series of Open Language Models", arXiv:2311.16867 (2023-11).
- Falcon Mamba: Zuo et al., "Falcon Mamba: The First Competitive Attention-free 7B Language Model" (2024-08).
- Falcon 3: TII, "Falcon 3" model card (2024-12).
- Cohere Aya: Üstün et al., "Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model", arXiv:2402.07827 (2024-02).
- Hymba: NVIDIA, "Hymba: A Hybrid-head Architecture for Small Language Models", arXiv:2411.13676 (2024-11).
- Samba (Phi-4-mini-flash precursor): Ren et al., "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling", arXiv:2406.07522 (2024-06).
- Phi-4-mini-flash: Microsoft, "Phi-4-Mini-Flash Reasoning Technical Report" (2025-07).
- Florence-2: Xiao et al., "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks", arXiv:2311.06242 (2023-11).
- PaliGemma: Beyer et al., "PaliGemma: A versatile 3B VLM for transfer", arXiv:2407.07726 (2024-07).
- Qwen 2.5-VL: Qwen Team, "Qwen 2.5-VL Technical Report", arXiv:2502.13923 (2025-02).
- TinyLlama: Zhang et al., "TinyLlama: An Open-Source Small Language Model", arXiv:2401.02385 (2024-01, but checkpoints from 2023-09).
- Pythia: Biderman et al., "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling", arXiv:2304.01373 (2023-04).
- MPT: MosaicML team, "Introducing MPT-7B" blog post (2023-05).

**HuggingFace config sources verified during v2 production** (URLs in §3 tables):
TinyLlama-1.1B-Chat-v1.0, Qwen/Qwen3-0.6B, microsoft/bitnet-b1.58-2B-4T, apple/OpenELM-3B, mistralai/Mistral-7B-v0.1, tiiuae/falcon-7b, mosaicml/mpt-7b, deepseek-ai/deepseek-llm-7b-base, allenai/OLMo-7B-hf, microsoft/phi-2, THUDM/chatglm2-6b, tiiuae/falcon-mamba-7b, google/recurrentgemma-2b, Qwen/Qwen2-7B, stabilityai/stablelm-3b-4e1t, unsloth/codegemma-7b, THUDM/glm-4-9b, deepseek-ai/DeepSeek-V2-Lite, ai21labs/Jamba-v0.1, nvidia/Hymba-1.5B-Base, tiiuae/Falcon3-7B-Base.

**Models verified during v1 production and preserved:** all 45 v1 rows (Llama 3.x, Qwen 2.5/Qwen 3 sweep, Gemma 2/3, Phi-3.x/Phi-4, OLMo 2, OLMoE, SmolLM2/3, Granite 3.1/3.3, MiniCPM 3, Yi 1.5, InternLM 2.5/3, StableLM 2, Mistral 7B v0.3, DeepSeek-Coder-V2-Lite, Mamba 2.8B, Zamba2 2.7B, RWKV-6 Finch, R1-Distill family, CodeGemma, StarCoder 2, PaliGemma, Florence-2, LLaVA-NeXT-Mistral). All numeric facts in those rows are preserved unchanged; the three architectural claims flagged as errors by the v1 critique (Gemma 3 softcap, TinyLlama MHA, Phi-3-mini head-dim decoupling) are corrected in §3 / §6 / §8.

**Cross-reference:** facts about per-layer arithmetic (MLA q/k/v head_dim asymmetry, Mamba SSM state shape, OpenELM per-layer width) are cross-referenced in `research/02-layer-sources.v2.md` (produced in parallel).

---

*End of v2.*
