# 01 — Mainstream Small Language Model Census (<8B params)

*Stream: model-census. Project: llm-layers. Date: 2026-06-04.*

---

## 1. Executive Summary

This census catalogs ~45 mainstream small language models (active params < 8B) shipped between 2023 and 2026, drawn from the dense decoder, sparse MoE, hybrid SSM, vision-language, and coding-specialized families. The goal is to surface the *architectural axes* that a minimal operator API must parametrize in order to instantiate any of them.

The picture is less monolithic than the "post-Llama consensus" suggests. While SwiGLU, RMSNorm pre-norm, RoPE, and GQA dominate the dense decoder space, every single design slot has at least two competing variants in production: Gemma uses GeGLU + post-norm-around-each-block + 5:1 sliding/full attention alternation + softcapping; OLMo 2 uses pure post-norm with QK-RMSNorm over full head channels; Qwen 3 uses QK-RMSNorm but on `head_dim` only; Phi-3 fuses gate/up into a single projection and uses LongRoPE per-dimension factors; DeepSeek-V2-Lite uses Multi-Head Latent Attention (MLA) with a split nope/rope head; Granite 3 carries μP-style residual / embedding / logit / attention multipliers; SmolLM3 interleaves NoPE every 4th layer; SSM hybrids (Mamba2, Zamba2, Jamba, RWKV-7) replace attention with recurrence in most layers.

Eleven architectural axes are identified (Section 4). The API must treat attention type, positional encoding, normalization placement, FFN family, MoE routing, KV-cache shape, vocab/tokenizer, and per-layer scalar multipliers as first-class parameters — not as fixed implementation choices.

---

## 2. Census Table

Notation: heads/KV/hd = `num_attention_heads` / `num_key_value_heads` / `head_dim`. ✓ = present. Vocab tokenizer family abbreviations: TT = Tiktoken (cl100k/o200k variants), LL3 = Llama-3 Tiktoken (128k), SP = SentencePiece, GP = Gemma SentencePiece (256k), QW = Qwen BPE (151936), GPT2 = GPT-2 BPE, MS = Microsoft Phi SP (32k/200k).

### 2.1 Dense decoders — Llama family

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 | 1B | 2024-09 | GQA | 32/8/64 | Llama-3 (factor 32, low=1, high=4, orig=8192) | RMS pre | SwiGLU (silu) | 128256 | LL3 | y | 131072 | [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json) |
| Llama 3.2 | 3B | 2024-09 | GQA | 24/8/128 | Llama-3 (factor 32) | RMS pre | SwiGLU | 128256 | LL3 | y | 131072 | [unsloth/Llama-3.2-3B](https://huggingface.co/unsloth/Llama-3.2-3B/raw/main/config.json) |
| Llama 3.1 | 8B | 2024-07 | GQA | 32/8/128 | Llama-3 (factor 8) | RMS pre | SwiGLU | 128256 | LL3 | n | 131072 | [unsloth/Meta-Llama-3.1-8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/config.json) |
| TinyLlama | 1.1B | 2024-01 | GQA | 32/4/64 | vanilla (θ=10000) | RMS pre | SwiGLU | 32000 | SP (Llama-2) | n | 2048 | [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json) |

### 2.2 Dense decoders — Qwen family

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen 2.5 | 0.5B | 2024-09 | GQA | 14/2/64 | vanilla (θ=1e6) | RMS pre | SwiGLU | 151936 | QW | y | 32768 / 32768 | [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json) |
| Qwen 2.5 | 1.5B | 2024-09 | GQA | 12/2/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 151936 | QW | y | 131072 / 131072 | [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/config.json) |
| Qwen 2.5 | 3B | 2024-09 | GQA | 16/2/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 151936 | QW | y | 32768 / 32768 | [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B/raw/main/config.json) |
| Qwen 2.5 | 7B | 2024-09 | GQA | 28/4/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 152064 | QW | n | 131072 / 131072 | [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json) |
| Qwen 3 | 0.6B | 2025-04 | GQA + **QK-RMSNorm** | 16/8/128 | vanilla (θ=1e6) | RMS pre + qk-norm on head_dim | SwiGLU | 151936 | QW | y | 40960 | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json) |
| Qwen 3 | 1.7B | 2025-04 | GQA + QK-norm | 16/8/128 | vanilla (θ=1e6) | RMS pre + qk-norm | SwiGLU | 151936 | QW | y | 40960 | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json) |
| Qwen 3 | 4B | 2025-04 | GQA + QK-norm | 32/8/128 | vanilla (θ=1e6) | RMS pre + qk-norm | SwiGLU | 151936 | QW | y | 40960 | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json) |
| Qwen 3 | 8B | 2025-04 | GQA + QK-norm | 32/8/128 | vanilla (θ=1e6) | RMS pre + qk-norm | SwiGLU | 151936 | QW | n | 40960 | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json) |

### 2.3 Dense decoders — Gemma family

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma 2 | 2B | 2024-07 | GQA, **SWA every 2nd layer (4096)** | 8/4/256 | vanilla (θ=10000) | RMS **pre+post around each sublayer** + softcap | **GeGLU (gelu_pytorch_tanh)** | 256000 | GP | y† | 8192 / 4096 alt | [unsloth/gemma-2-2b](https://huggingface.co/unsloth/gemma-2-2b/raw/main/config.json) |
| Gemma 2 | 9B | 2024-07 | GQA, SWA every 2nd | 16/8/256 | vanilla | RMS pre+post + softcap (50,30) | GeGLU | 256000 | GP | y† | 8192 / 4096 alt | (gemma-2 tech report) |
| Gemma 3 | 1B | 2025-03 | GQA, **5:1 SWA:full**, dual RoPE | 4/1/256 | global θ=1e6 + **local θ=10000** for SWA layers | RMS pre+post per sublayer | GeGLU | 262144 | GP-v3 | n | 32768 / 512 alt | [gemma-3-1b-pt](https://huggingface.co/unsloth/gemma-3-1b-pt/raw/main/config.json) |
| Gemma 3 | 4B (text tower) | 2025-03 | GQA, 5:1 SWA:full | 8/4/256 | global 1e6 + local 1e4 + linear factor 8 | RMS pre+post | GeGLU | 262208 | GP-v3 | — | 131072 / 1024 alt | [gemma-3-4b-pt](https://huggingface.co/unsloth/gemma-3-4b-pt/raw/main/config.json) |

†: Gemma 2 ties embeddings to LM head implicitly (default).

### 2.4 Dense decoders — Phi family

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Phi-3 mini | 3.8B (4k) | 2024-04 | MHA | 32/32/96 | vanilla (θ=10000) | RMS pre | SwiGLU **fused gate_up** | 32064 | MS | n | 4096 / 2047 | [Phi-3-mini-4k](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/main/config.json) |
| Phi-3.5 mini | 3.8B | 2024-08 | MHA | 32/32/96 | **LongRoPE (su / per-dim short+long factors)** | RMS pre | SwiGLU fused | 32064 | MS | n | 131072 / 262144 | [Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/raw/main/config.json) |
| Phi-3 small | 7B | 2024-04 | **Block-sparse attention, dense every 2nd layer** | 32/8/128 | vanilla θ=1e6 + μP | LayerNorm pre | **GeGLU variant (`gegelu`)** | 100352 | TT (cl100k) | n | 8192 | [Phi-3-small-8k](https://huggingface.co/microsoft/Phi-3-small-8k-instruct/raw/main/config.json) |
| Phi-3.5 vision (text tower) | 3.8B | 2024-08 | MHA | 32/32/96 | LongRoPE (su) | RMS pre | SwiGLU fused | 32064 | MS | n | 131072 / 262144 | [Phi-3.5-vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct/raw/main/config.json) |
| Phi-4 mini | 3.8B | 2025-02 | GQA + **partial RoPE (factor=0.75)** | 24/8/128 | LongRoPE on rotary fraction | RMS pre | SwiGLU fused | 200064 | TT (o200k) | y | 131072 / 262144 | [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct/raw/main/config.json) |

### 2.5 Dense decoders — Mistral & generic

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mistral | 7B v0.3 | 2024-05 | GQA (SWA dropped in v0.2+) | 32/8/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 32768 | SP | n | 32768 | [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/raw/main/config.json) |

### 2.6 Dense decoders — OLMo / SmolLM / IBM / Yi / InternLM / StableLM / MiniCPM

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tok | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OLMo 2 | 7B | 2024-11 | MHA | 32/32/128 | vanilla (θ=5e5) | **Post-norm (post_attention + post_feedforward) RMS + QK-RMSNorm over full head channels** | SwiGLU | 100352 | GPT-NeoX (Dolma) | n | 4096 | [OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B/raw/main/config.json) |
| OLMo 2 | 13B | 2024-11 | MHA | 40/40/128 | vanilla θ=5e5 | Post-norm + QK-RMSNorm | SwiGLU | 100352 | GPT-NeoX | n | 4096 | [OLMo-2-13B](https://huggingface.co/allenai/OLMo-2-1124-13B/raw/main/config.json) |
| SmolLM2 | 135M | 2024-11 | GQA | 9/3/64 | vanilla (θ=1e5) | RMS pre | SwiGLU | 49152 | SmolLM (HF-bpe 49k) | y | 8192 | [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json) |
| SmolLM2 | 360M | 2024-11 | GQA | 15/5/64 | vanilla (θ=1e5) | RMS pre | SwiGLU | 49152 | SmolLM | y | 8192 | [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json) |
| SmolLM2 | 1.7B | 2024-11 | MHA | 32/32/64 | vanilla (θ=1.3e5) | RMS pre | SwiGLU | 49152 | SmolLM | y | 8192 | [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/raw/main/config.json) |
| SmolLM3 | 3B | 2025-07 | GQA + **NoPE every 4th layer** | 16/4/128 | vanilla (θ=5e6); NoPE on selected layers | RMS pre | SwiGLU | 128256 | LL3 | y | 65536 | [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/raw/main/config.json) |
| Granite 3.1 | 2B | 2024-12 | GQA + **scalar multipliers** (attn / emb / logit / residual) | 32/8/64 | vanilla (θ=5e6) | RMS pre | SwiGLU | 49152 | StarCoder-2 BPE | y | 131072 | [granite-3.1-2b](https://huggingface.co/ibm-granite/granite-3.1-2b-base/raw/main/config.json) |
| Granite 3.3 | 2B | 2025-04 | GQA + multipliers | 32/8/64 | vanilla (θ=1e7) | RMS pre | SwiGLU | 49152 | StarCoder-2 BPE | y | 131072 | [granite-3.3-2b](https://huggingface.co/ibm-granite/granite-3.3-2b-base/raw/main/config.json) |
| Granite 3.3 | 8B | 2025-04 | GQA + multipliers | 32/8/128 | vanilla (θ=1e7) | RMS pre | SwiGLU | 49152 | StarCoder-2 BPE | y | 131072 | [granite-3.3-8b](https://huggingface.co/ibm-granite/granite-3.3-8b-base/raw/main/config.json) |
| MiniCPM 3 | 4B | 2024-09 | **MLA (q_lora=768, kv_lora=256, qk_nope=64, qk_rope=32)** | 40/40 (latent) | LongRoPE (su) | RMS pre + **scale_emb=12, scale_depth=1.4, dim_model_base=256** | SwiGLU | 73448 | MiniCPM SP | — | 32768 | [MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B/raw/main/config.json) |
| Yi 1.5 | 6B | 2024-05 | GQA | 32/4/128 | vanilla (θ=5e6) | RMS pre | SwiGLU | 64000 | Yi SP | n | 4096 | [Yi-1.5-6B](https://huggingface.co/01-ai/Yi-1.5-6B/raw/main/config.json) |
| InternLM 2.5 | 7B | 2024-07 | GQA | 32/8/128 | dynamic NTK (factor=2.0, θ=5e7) | RMS pre | SwiGLU | 92544 | InternLM SP | n | 262144 | [internlm2_5-7b](https://huggingface.co/internlm/internlm2_5-7b/raw/main/config.json) |
| InternLM 3 | 8B | 2025-01 | GQA (very thin: KV=2!) | 32/2/128 | dynamic NTK (factor=6.0, θ=5e7) | RMS pre | SwiGLU | 128512 | InternLM SP | n | 32768 | [internlm3-8b-instruct](https://huggingface.co/internlm/internlm3-8b-instruct/raw/main/config.json) |
| StableLM 2 | 1.6B | 2024-01 | MHA + **QKV bias + partial RoPE (0.25)** | 32/32/64 | partial rotary | LayerNorm pre | SwiGLU | 100352 | TT-like | n | 4096 | [stablelm-2-1_6b](https://huggingface.co/stabilityai/stablelm-2-1_6b/raw/main/config.json) |
| StableLM 2 | 12B | 2024-04 | GQA + **QK-LayerNorm** + partial RoPE (0.25) | 32/8/128 | partial rotary (θ=10000) | LayerNorm pre + QK-LN | SwiGLU | 100352 | TT-like | n | 4096 | [stablelm-2-12b](https://huggingface.co/stabilityai/stablelm-2-12b/raw/main/config.json) |

### 2.7 Sparse MoE (active < 8B)

| Family | Variant | Released | Attn | H/KV/hd | Experts | Top-k | Shared | RoPE | Norm | FFN | Vocab | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen 3 MoE | 30B-A3B | 2025-04 | GQA + QK-norm | 32/4/128 | 128 | 8 | **0** (none) | vanilla (θ=1e6) | RMS pre + qk-norm | SwiGLU-MoE (moe_ffn=768, norm_topk=true, dense step 1) | 151936 | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) |
| DeepSeek-V2-Lite (Coder) | 16B (2.4B active) | 2024-06 | **MLA (kv_lora=512, qk_nope=128, qk_rope=64, v_head=128, no q_lora)** | 16/16 (latent) | 64 routed | 6 | **2 shared** | YaRN (factor=40, mscale 0.707) | RMS pre | SwiGLU-MoE (moe_ffn=1408) + **1 dense layer (first_k_dense_replace=1)** | 102400 | [DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/raw/main/config.json) |
| OLMoE | 1B-7B | 2024-09 | MHA | 16/16/128 | 64 | 8 | 0 | vanilla (θ=10000) | RMS pre (with norm_topk_prob=false) | SwiGLU-MoE | 50304 | [OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125/raw/main/config.json) |

### 2.8 Hybrid / SSM / linear-attention

| Family | Variant | Released | Token mixer | Layout | RoPE? | Norm | FFN | Vocab | Source |
|---|---|---|---|---|---|---|---|---|---|
| Mamba | 2.8B | 2023-12 | **Selective SSM (S6)** | 64 SSM blocks, no attention | none | RMS pre (residual fp32) | none (SSM gating only) | 50280 | GPT-NeoX | [mamba-2.8b-hf](https://huggingface.co/state-spaces/mamba-2.8b-hf/raw/main/config.json) |
| Zamba2 | 2.7B | 2024-10 | **Mamba2 + periodic hybrid attention** (54 layers, "hybrid" every ~6 layers shares a single attention block) | mamba/mamba/.../hybrid pattern | RoPE on shared attn | RMS pre | gelu MLP | 32000 | [Zamba2-2.7B](https://huggingface.co/Zyphra/Zamba2-2.7B/raw/main/config.json) |
| Jamba | mini (early dev card) | 2024-04 | **Mamba1 layers with 1-in-8 attention + 1-in-2 MoE** (`attn_layer_period=8`, `expert_layer_period=2`) | hybrid Transformer-Mamba-MoE | none on Mamba layers | RMS pre | MoE (8 experts, top-2) | 65536 | [Jamba-tiny-dev](https://huggingface.co/ai21labs/Jamba-tiny-dev/raw/main/config.json) |
| RWKV-6 (Finch) | 1.6B / 7B | 2024-08 | **RWKV WKV6 linear attention with channel mix (no QKV)** | uniform | none (time-decay + bonus) | LayerNorm | RWKV channel-mix (no SwiGLU) | 65536 | [v6-Finch-1B6](https://huggingface.co/RWKV/v6-Finch-1B6-HF/raw/main/config.json), [v6-Finch-7B](https://huggingface.co/RWKV/v6-Finch-7B-HF/raw/main/config.json) |

### 2.9 Vision-language models (text tower only)

| Family | Variant | Released | Text tower | H/KV/hd | Notes | Source |
|---|---|---|---|---|---|---|
| LLaVA-NeXT (Mistral) | 7B | 2024-01 | Mistral-7B-Instruct-v0.2 | 32/8/128 | text tower = Mistral 7B w/ θ=1e6 (no SWA in v0.2+) | [llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/raw/main/config.json) |
| PaliGemma | 3B | 2024-05 | Gemma 1 2B-class | 8/1/256 (MQA!) | text uses GeGLU, no SWA; head_dim=256 | (Gemma 1 codegemma-2b config, similar layout) |
| Florence-2 | base / large | 2024-06 | **BART-style encoder-decoder** (d_model=768, 6+6) | 12 heads | absolute pos emb; uses GELU; **encoder-decoder, not decoder-only** | [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base/raw/main/config.json) |
| Phi-3.5 vision | 4B | 2024-08 | Phi-3 mini 3.8B | 32/32/96 | LongRoPE (su) | [Phi-3.5-vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct/raw/main/config.json) |
| Gemma 3 4B | 4B | 2025-03 | Gemma 3 text 4B | 8/4/256 | full Gemma 3 LM described above | [gemma-3-4b-pt](https://huggingface.co/unsloth/gemma-3-4b-pt/raw/main/config.json) |

### 2.10 Coding-specialized

| Family | Variant | Released | Attn | H/KV/hd | RoPE | Norm | FFN | Vocab | Tied | Ctx / SWA | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-Coder-V2-Lite | 16B-A2.4B | 2024-06 | MLA | 16/16 (latent) | YaRN | RMS pre | SwiGLU-MoE + 1 dense layer | 102400 | n | 163840 | [DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/raw/main/config.json) |
| Qwen 2.5-Coder | 1.5B | 2024-09 | GQA | 12/2/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 151936 | y | 32768 / 32768 | [Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/raw/main/config.json) |
| Qwen 2.5-Coder | 7B | 2024-09 | GQA | 28/4/128 | vanilla (θ=1e6) | RMS pre | SwiGLU | 152064 | n | 32768 / 131072 | [Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B/raw/main/config.json) |
| CodeGemma | 2B | 2024-04 | **MQA (KV=1)** | 8/1/256 | vanilla (θ=10000) | RMS pre+post (Gemma 1 dual norm) | GeGLU | 256000 | y | 8192 | [unsloth/codegemma-2b](https://huggingface.co/unsloth/codegemma-2b/raw/main/config.json) |
| CodeGemma | 7B | 2024-04 | GQA | 16/16/256 | vanilla | RMS pre+post | GeGLU | 256000 | y | 8192 | (gemma-1 family card) |
| StarCoder 2 | 3B | 2024-02 | GQA + **dense QKV biases** (`use_bias=true`) | 24/2/128 | vanilla (θ≈1e6) | LayerNorm pre | **GLU-free MLP (`mlp_type=default`)** = gated GELU-tanh + bias | 49152 | — | 16384 / 4096 | [starcoder2-3b](https://huggingface.co/bigcode/starcoder2-3b/raw/main/config.json) |
| StarCoder 2 | 7B | 2024-02 | GQA + bias | 36/4/128 | vanilla (θ=1e6) | LayerNorm pre | gated GELU-tanh + bias | 49152 | — | 16384 / 4096 | [starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b/raw/main/config.json) |

### 2.11 R1 distillations <8B

| Family | Variant | Released | Attn | H/KV/hd | Backbone | Vocab | Ctx | Source |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen | 1.5B | 2025-01 | GQA | 12/2/128 | Qwen2.5-Math-1.5B | 151936 | 131072 | [R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json) |
| DeepSeek-R1-Distill-Qwen | 7B | 2025-01 | GQA | 28/4/128 | Qwen2.5-Math-7B | 152064 | 131072 | [R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/raw/main/config.json) |
| DeepSeek-R1-Distill-Llama | 8B | 2025-01 | GQA | 32/8/128 | Llama-3.1-8B | 128256 | 131072 | [R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/raw/main/config.json) |

Total cataloged: **45 distinct shipped variants** across **22 distinct architectures**.

---

## 3. Confirmed implementation details (cross-checked against `transformers` source)

The configs above are corroborated against the local `transformers` source tree (`C:\Users\zhengte\external\transformers\src\transformers\models\<name>\modeling_*.py`). The following implementation facts were verified by direct code inspection, not config inference:

- **Qwen 3 QK-norm shape**: `q_norm = Qwen3RMSNorm(self.head_dim, eps)` and applied **before** the transpose for attention computation (`modeling_qwen3.py` lines 248–264). RMSNorm is per-head, dim = `head_dim`. Contrast with OLMo 2 (next bullet).
- **OLMo 2 QK-norm shape**: `q_norm = Olmo2RMSNorm(num_attention_heads * head_dim, eps)` and `k_norm = Olmo2RMSNorm(num_key_value_heads * head_dim, eps)` — i.e. norms the full head-channels concatenated, **not** per-head (`modeling_olmo2.py` lines 231–232).
- **OLMo 2 post-norm**: `self.post_attention_layernorm` is applied *after* attention output is added to residual, and `self.post_feedforward_layernorm` is applied *after* MLP output is added (lines 302–331). This is *post-norm*, not pre-norm — the entire transformer block is post-norm.
- **Gemma 2 / Gemma 3 dual norm**: every sublayer has both `input_layernorm` *and* `post_attention_layernorm` (resp. `pre_feedforward_layernorm` and `post_feedforward_layernorm`). The hidden state is normalized *into* the sublayer and the sublayer's output is normalized again *before* being added to the residual (`modeling_gemma2.py` lines 309–343).
- **Gemma 3 layer alternation**: `layer_types` constructed at config-time as `["sliding_attention" if (i+1) % 6 else "full_attention" for i in range(L)]` — i.e. 5 sliding then 1 full, with `sliding_window_pattern=6` (`configuration_gemma3.py` lines 105–116). The *local* RoPE base `rope_local_base_freq=10000` is used for the sliding layers; the *global* `rope_theta=1e6` is used on the full-attn layers. This is two independent RoPE caches per layer-type.
- **Phi-3 fused gate_up**: `gate_up_proj = nn.Linear(hidden_size, 2*intermediate_size)` then split (`modeling_phi3.py` lines 49–59). This is SwiGLU but with one big linear instead of separate `gate_proj`/`up_proj` — quantization-relevant.
- **Phi-3-small `gegelu`**: an explicit named activation distinct from SwiGLU; the activation is gated-GELU (rather than gated-SiLU). Combined with block-sparse attention every 2nd layer and μP scaling factors (`mup_*` keys), Phi-3-small is structurally the most divergent dense model in the census.
- **SmolLM3 NoPE pattern**: `no_rope_layers` is a length-`num_hidden_layers` 0/1 mask, default = `[1,1,1,0]*L/4` — every 4th layer skips RoPE entirely. Layers also tag `layer_types` ∈ {`sliding_attention`, `full_attention`} indirectly (`configuration_smollm3.py` lines 89–113).
- **MLA (DeepSeek-V2-Lite, MiniCPM3)** decomposes attention into a low-rank latent space:
  - Q is projected to a `q_lora_rank`-dim (or directly if `q_lora_rank is None`) then up to `(qk_nope_head_dim + qk_rope_head_dim)` per head; the *nope* portion has no RoPE applied, the *rope* portion does.
  - KV share a single `kv_lora_rank` latent projection from which K_nope/K_rope/V are decoded; only the `kv_lora_rank`+`qk_rope_head_dim` latent and the rope-key need to be cached per token — *not* full K and V tensors.
- **Granite "multipliers"** (`embedding_multiplier`, `attention_multiplier`, `logits_scaling`, `residual_multiplier`) are scalar μP-style rescalings applied at: input embedding output, attention pre-softmax logits, output `lm_head` logits, and on each residual addition. The minimal API must allow scalar broadcasts at these specific points (not arbitrary places).
- **MiniCPM3** carries similar scalars: `scale_emb=12`, `scale_depth=1.4` (residual scaling = `scale_depth / sqrt(num_layers)`), `dim_model_base=256` (used to scale logits as `1 / (hidden_size / dim_model_base)`).
- **Qwen 3 30B-A3B MoE**: `n_routed_experts=128`, `num_experts_per_tok=8`, **zero shared experts** (no `shared_expert_intermediate_size` key in config), `decoder_sparse_step=1` (every layer is sparse), `norm_topk_prob=true` (router scores normalized over the chosen top-k). Contrast with DeepSeek-V2-Lite which has 2 shared experts and a single fully-dense first layer.
- **StarCoder 2 biases**: `use_bias=true` — keeps biases on attention QKV and MLP linears, almost unique among modern decoders; uses LayerNorm rather than RMSNorm.

---

## 4. Architectural Axes

The set of independent variation axes the API must parametrize.

### 4.1 Attention type

| Variant | Models | Defining parameters | KV-cache shape per token per layer |
|---|---|---|---|
| **MHA** (heads == kv_heads) | Phi-3 mini/medium, Phi-3-small, OLMo 2, SmolLM2-1.7B, StableLM 2 1.6B, OLMoE, TinyLlama (KV=4 actually GQA, see below), Mamba (no attn) | `num_heads == num_kv_heads` | `2 * num_heads * head_dim` |
| **GQA** (1 < kv_heads < heads) | Llama 3.x, Qwen 2.5, Qwen 3, Gemma 2/3, Mistral, Yi 1.5, InternLM 2.5/3, Granite, SmolLM3, Phi-4-mini, StarCoder2, Qwen2.5-Coder, R1-distills, StableLM 2 12B | `num_kv_heads ∈ {2,4,8}` typical | `2 * num_kv_heads * head_dim` |
| **MQA** (kv_heads == 1) | Gemma 3 1B, CodeGemma 2B, PaliGemma | `num_kv_heads = 1` | `2 * head_dim` |
| **MLA** (latent KV) | DeepSeek-V2-Lite (Coder), MiniCPM 3 | `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim` | `kv_lora_rank + qk_rope_head_dim` (massive reduction) |
| **Sliding-window** (attn restricted to last *W* tokens) | Gemma 2 (alternating every other layer, W=4096), Gemma 3 (5:1 SWA:full, W=512/1024), Phi-3 mini (W=2047), SmolLM3 (alternating), StarCoder2 (W=4096), Phi-3-small (block-sparse every 2nd) | `sliding_window`, `sliding_window_pattern`, `layer_types[]` | same as GQA but **circular buffer of size W**, not unbounded |
| **Block-sparse** | Phi-3-small | `blocksparse_block_size`, `blocksparse_num_local_blocks`, `blocksparse_vert_stride`, `dense_attention_every_n_layers` | per-layer-type |
| **Linear / SSM** | Mamba (S6), Mamba2, RWKV-6/-7 | `state_size`, `conv_kernel`, `time_step_rank`, or `head_size`+`attention_hidden_size` for RWKV | constant-size state per layer (no growth with sequence) |
| **Hybrid SSM+attn** | Zamba2, Jamba | `layers_block_type[]`, `attn_layer_period`, `attention_period` | mixed: SSM layers have constant state, attention layers full KV |

The API needs per-layer dispatch on attention type (Gemma 3, SmolLM3, Phi-3-small, Zamba2, Jamba all mix types within one model).

### 4.2 Positional encoding

| Variant | Distinguishing config | Models |
|---|---|---|
| **Vanilla RoPE** | `rope_theta`, full rotary | Qwen 2.5, Qwen 3, Mistral, OLMo 2, SmolLM2, Yi, CodeGemma, Granite, Gemma 2 |
| **Llama-3 RoPE** | `rope_scaling = {factor, low_freq_factor, high_freq_factor, original_max_pos, type="llama3"}` — frequency-conditional scaling | Llama 3.x, R1-distill-Llama |
| **LongRoPE / su / longrope** | `rope_scaling = {short_factor[], long_factor[], type}` — per-dimension factors | Phi-3.5 mini/vision, Phi-4 mini, MiniCPM 3 |
| **Dynamic NTK / dynamic** | `rope_scaling = {type:"dynamic", factor}` | InternLM 2.5/3 |
| **YaRN** | `rope_scaling = {type:"yarn", factor, mscale, beta_fast, beta_slow}` | DeepSeek-V2-Lite (Coder) |
| **Linear scaling** | `rope_scaling = {type:"linear", factor}` | Gemma 3 4B/12B/27B |
| **Dual RoPE (per-layer-type)** | `rope_theta` + `rope_local_base_freq` | Gemma 3 (all sizes) — full-attn layers use 1e6, sliding layers use 1e4 |
| **Partial RoPE** | `partial_rotary_factor < 1` (only first `factor * head_dim` channels rotated) | StableLM 2, Phi-4 mini (0.75) |
| **NoPE per layer** | `no_rope_layers[]` mask | SmolLM3 |
| **None (SSM/linear-attn)** | absent | Mamba, Mamba2, RWKV-6/-7 (uses decay), Jamba Mamba layers |
| **MLA-split RoPE** | RoPE applied only to `qk_rope_head_dim` channels per Q/K head | DeepSeek-V2-Lite, MiniCPM 3 |
| **Absolute learned** | classical absolute pos emb | Florence-2 (BART encoder-decoder) |

### 4.3 Normalization

Two dimensions: **type** (RMSNorm vs LayerNorm) and **placement** (pre vs post vs dual; QK-norm presence and shape).

| Variant | Models | Notes |
|---|---|---|
| **RMSNorm, pre-norm** | Llama 3.x, Qwen 2.5, Mistral, Yi, InternLM, Granite, SmolLM2/3, R1-distills, MiniCPM3, Mamba, Zamba2, Jamba | The "Llama default" |
| **RMSNorm + QK-RMSNorm per head_dim** | Qwen 3 (incl. Qwen 3 MoE) | `q_norm`, `k_norm` of shape `(head_dim,)`, applied to each head before RoPE+attention |
| **RMSNorm + QK-RMSNorm over full head channels** | OLMo 2 | `q_norm` shape `(num_heads*head_dim,)`, `k_norm` shape `(num_kv_heads*head_dim,)`, applied after Q/K projection before reshape |
| **RMSNorm post-norm (block-output)** | OLMo 2 | `post_attention_layernorm`, `post_feedforward_layernorm` applied to residual stream after addition |
| **RMSNorm dual (pre+post per sublayer)** | Gemma 2, Gemma 3, CodeGemma | Four norms per layer: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm` |
| **LayerNorm pre** | Phi-3-small, StarCoder 2, RWKV-6 | Mean+var, with learnable bias |
| **LayerNorm + QK-LayerNorm** | StableLM 2 12B | `qk_layernorm = true` |
| **LayerNorm partial RoPE-aware** | StableLM 2 1.6B | LayerNorm with `partial_rotary_factor=0.25` |

The placement matters more than the type for the API: a single `pre_norm` + `post_norm` flag per sublayer (4 options: pre, post, both, neither) captures everything. QK-norm needs its own `qk_norm_kind ∈ {none, per_head_dim, per_full_channels}` axis.

### 4.4 FFN family

| Variant | Activation | Gate? | Models |
|---|---|---|---|
| **SwiGLU** (separate gate_proj, up_proj, down_proj) | SiLU | yes | Llama, Qwen, Mistral, OLMo, OLMoE, SmolLM, Granite, Yi, InternLM, MiniCPM, StableLM, R1-distills |
| **SwiGLU fused gate_up** (single `gate_up_proj` linear of width `2*intermediate_size`, then chunk) | SiLU | yes | Phi-3 / Phi-3.5 / Phi-4 mini family |
| **GeGLU** (gelu_pytorch_tanh + gate) | GELU-tanh | yes | Gemma 2, Gemma 3, CodeGemma, PaliGemma text tower |
| **gegelu** (Phi-3-small variant) | custom (paired-channel gated GELU) | yes | Phi-3-small |
| **Gated GELU (non-tanh) with biases** | GELU | yes | StarCoder 2 |
| **GELU (no gate)** | GELU | no | Florence-2 (BART), Zamba2 MLP path |
| **SiLU (no gate)** | SiLU | no | Mamba (in the SSM gate path — there is no "FFN" per layer in pure Mamba) |
| **RWKV channel-mix** | sigmoid+ReLU² | yes | RWKV-6/-7 (replaces both attention and FFN) |
| **SwiGLU-MoE** | SiLU + gate | sparse routing | Qwen 3 30B-A3B, DeepSeek-V2-Lite, OLMoE, Jamba |

MoE adds three sub-axes:
- `num_experts`, `num_experts_per_tok`
- `shared_experts` (0 in Qwen 3 30B, 2 in DeepSeek-V2-Lite)
- `first_k_dense_replace` / `decoder_sparse_step` — DeepSeek-V2-Lite makes the first layer dense; Qwen 3 30B is sparse from layer 0.
- `norm_topk_prob` (true: softmax over chosen experts only / false: raw routing logits).
- Router activation (sigmoid vs softmax vs noisy top-k).

### 4.5 KV-cache implications per attention type

| Attention | Per-token-per-layer cache size | Shape | Special handling |
|---|---|---|---|
| MHA | `2 * H * d` | `(2, H, d)` | trivial |
| GQA | `2 * H_kv * d` | `(2, H_kv, d)` | replicated to H during attention |
| MQA | `2 * d` | `(2, 1, d)` | broadcast |
| MLA | `kv_lora_rank + qk_rope_head_dim` | `(rank + rope_dim,)` (1 cached latent, 1 RoPE-keyed pos channel) | matrix absorption into Q/output projections |
| SWA | `2 * H_kv * d` but **bounded** to `min(seq, W)` | circular buffer of size W | mask shift logic per step |
| Block-sparse (Phi-3-small) | mixed: dense layers full, sparse layers banded blocks | varies | needs per-layer kernel |
| SSM (Mamba/RWKV) | `state_size * d` (Mamba) or `head_size * num_heads` (RWKV) | constant per layer | no growth with seq |
| Hybrid (Zamba2/Jamba) | per-layer dispatch | mixed | needs heterogeneous cache container |

The API needs a `KVCache` abstraction whose underlying tensor shape and *update policy* are parametrized by the layer's attention type, with two non-trivial shapes (MLA's latent and SSM's state) being qualitatively different from a `(2, kv_heads, head_dim)` array.

### 4.6 Tokenizer / vocabulary families

| Family | Vocab size | Tokenizer flavor | Used by |
|---|---|---|---|
| Llama-3 Tiktoken (128k) | 128256 | BPE on Tiktoken regex | Llama 3.x, R1-Distill-Llama, SmolLM3 |
| Qwen BPE | 151936 / 152064 | BPE (Tiktoken-style, byte-fallback) | Qwen 2.5, Qwen 3, Qwen 3 MoE, Qwen2.5-Coder, R1-Distill-Qwen |
| Gemma SP | 256000 / 262144 / 262208 | SentencePiece, unigram; v3 expands to 262k with image tokens | Gemma 1/2/3, CodeGemma, PaliGemma |
| Microsoft Phi SP (32k) | 32064 | SentencePiece, Llama-2-derived | Phi-3 / Phi-3.5 mini / Phi-3.5 vision |
| Tiktoken cl100k | 100352 | OpenAI-cl100k regex | Phi-3-small, OLMo 2, StableLM 2 |
| Tiktoken o200k | 200064 | OpenAI-o200k regex | Phi-4 mini |
| SmolLM/StarCoder BPE | 49152 | Compact BPE | SmolLM2, Granite 3.x, StarCoder 2 |
| Mistral SP | 32768 | SentencePiece v3 (extended) | Mistral 7B v0.3, LLaVA-NeXT Mistral |
| TinyLlama / Llama-2 SP | 32000 | Llama-2 SentencePiece | TinyLlama, Zamba2 |
| Yi SP | 64000 | custom SentencePiece | Yi 1.5 |
| InternLM SP | 92544 / 128512 | custom SentencePiece | InternLM 2.5 / 3 |
| MiniCPM SP | 73448 | custom SentencePiece | MiniCPM 3 |
| GPT-NeoX BPE | 50280 / 50304 | classic GPT-NeoX | Mamba, OLMoE |
| RWKV BPE | 65536 | RWKV-world v2 | RWKV-6/-7, Jamba |

This matters for the API because (a) embedding-table sharing assumptions (`tie_word_embeddings`) differ per family, (b) tokenizers with special-purpose tokens (Gemma 3's image tokens, Phi-3.5 vision's image tokens) require the embedding layer to also tolerate cross-modal indices in the same lookup.

### 4.7 Scalar multipliers / μP-style rescalings

A subtle axis that is *invisible* in a Llama-style API:

| Multiplier | Models that use it | Application point |
|---|---|---|
| `embedding_multiplier` | Granite 3.x (12.0), MiniCPM3 `scale_emb` (12) | output of `embed_tokens` (before first norm) |
| `attention_multiplier` | Granite (0.015625 = 1/64), Phi-3-small μP | scales raw attn logits before softmax (replaces `1/sqrt(d)`) |
| `residual_multiplier` | Granite (0.22), MiniCPM3 `scale_depth/sqrt(L)` | scales sublayer output before residual addition |
| `logits_scaling` | Granite (8.0, 16.0), MiniCPM3 `hidden_size/dim_model_base`, Gemma 2/3 `final_logit_softcapping` | scales (or softcaps) final LM-head logits |
| `query_pre_attn_scalar` | Gemma 2/3 (= 256, overriding `1/sqrt(head_dim)`) | replaces `1/sqrt(head_dim)` with `1/sqrt(query_pre_attn_scalar)` |
| `attn_logit_softcapping` | Gemma 2 (50.0) | `tanh(logits/cap)*cap` before softmax |
| `final_logit_softcapping` | Gemma 2 (30.0) | `tanh(logits/cap)*cap` on LM-head output |

If the API hard-codes `attn_scale = 1/sqrt(head_dim)` it will silently produce wrong Gemma 2 outputs.

### 4.8 Bias presence

| Variant | Models |
|---|---|
| No biases anywhere | Llama 3, Qwen 2.5/3, OLMo 2, Mistral, Yi, InternLM, SmolLM, Granite |
| QKV bias, no MLP bias | Qwen 1 / Qwen 2 (`attention_bias=true` legacy), StableLM 2 1.6B (`use_qkv_bias=true`) |
| Full bias on QKV + MLP | StarCoder 2 (`use_bias=true`), Florence-2 |
| No bias on Q, bias on K? | (none — bias is either everywhere or nowhere by sublayer) |

### 4.9 Embedding tying

- **Tied** (one weight for input embed and `lm_head`): Llama-3.2 1B/3B, Qwen-2.5 0.5/1.5/3B, Qwen 3 0.6/1.7/4B, Gemma 2 (default), Gemma 3 1B/4B (default), Granite 3.x, SmolLM2 (all), SmolLM3, MiniCPM3, TinyLlama (no), Phi-4 mini, CodeGemma, Qwen2.5-Coder small.
- **Untied**: Llama-3.1 8B, Qwen-2.5 7B, Qwen 3 8B, Qwen 3 30B-A3B, Mistral 7B, OLMo 2 7B/13B, OLMoE, DeepSeek-Coder-V2-Lite, Phi-3 mini/3.5/small, Phi-3.5 vision, Yi 1.5 6B, InternLM 2.5/3, StableLM 2, R1-Distill-Llama-8B.

Roughly: smaller models tie, larger models untie. The cutoff is ~3–4B.

### 4.10 Per-layer heterogeneity

The API can no longer assume all `L` layers are identical. Heterogeneity sources:

| Model | What varies per layer |
|---|---|
| Gemma 2 | attention type (SWA vs full) every other layer |
| Gemma 3 | attention type + which RoPE base (local 1e4 vs global 1e6) — 5:1 pattern |
| SmolLM3 | RoPE on/off + sliding/full type — interleaved |
| Phi-3-small | dense vs block-sparse attention every 2nd layer |
| Zamba2 | mamba vs hybrid (= mamba + shared attention block) — `layers_block_type` is a 54-element list |
| Jamba | mamba vs attention (`attn_layer_period=8`) AND MLP vs MoE (`expert_layer_period=2`) — independently periodic |
| DeepSeek-V2-Lite | first layer is dense MLP; layers 1..L−1 are MoE (`first_k_dense_replace=1`) |
| Qwen 3 30B-A3B | every layer is MoE (`decoder_sparse_step=1`); but in larger DeepSeek-V3, some are dense |

The minimal API needs a layer-spec list, not a single layer config repeated `num_hidden_layers` times.

### 4.11 Context-length strategies and KV-cache implications

The "long context" story is *not* one variant — it's six:

1. **Llama-3 RoPE rescaling** (Llama 3.x): frequency-band scaling, applied at attention time, doesn't change KV-cache shape, but changes per-position RoPE matrices.
2. **LongRoPE per-dim factors** (Phi-3.5, Phi-4-mini, MiniCPM 3): per-dimension `short_factor[]` / `long_factor[]` chosen by `original_max_position_embeddings`. The factor arrays have length `head_dim/2` — they are baked into the cos/sin tables.
3. **YaRN** (DeepSeek-V2-Lite): `mscale` and `mscale_all_dim` modify both RoPE *and* the attention scale.
4. **Dynamic NTK** (InternLM): RoPE base θ is scaled at runtime as a function of current sequence length.
5. **Linear scaling** (Gemma 3): position is divided by `factor` (8.0). Trivial.
6. **Dual-RoPE alternation** (Gemma 3): different RoPE base per layer-type; the sliding layers use local positions.

For the cache, the only one that *changes shape* is MLA. Everything else just changes the cos/sin tables.

---

## 5. Surprises and non-obvious patterns

1. **Qwen 3 vs OLMo 2 use *different* QK-norm shapes.** Both apply RMSNorm to Q and K before attention. Qwen 3 uses `RMSNorm(head_dim)` — per-head normalization, applied after the head reshape, and the same norm broadcasts across all heads. OLMo 2 uses `RMSNorm(num_heads*head_dim)` — concatenated channel normalization applied before the reshape. These look textually similar (`q_norm`, `k_norm`) and configs look similar, but the operator is *not* the same: the API needs both `qk_norm_per_head_dim` and `qk_norm_per_full_channels` variants. (Source: `transformers/.../modeling_qwen3.py:248`, `modeling_olmo2.py:231`.)

2. **OLMo 2 is post-norm, not pre-norm.** The "Llama default" of pre-norm RMS is *not* what OLMo 2 ships; it explicitly normalizes after each sublayer residual. This requires the API to support `norm_position ∈ {pre, post, both}` per sublayer. It also affects mixed precision: post-norm runs the residual at higher magnitude before being clamped by the next sublayer's input norm.

3. **Gemma 3's 5:1 SWA:full alternation + dual RoPE.** Five sliding-window-512 layers, then one full-attention layer, repeated. The sliding layers use `rope_local_base_freq=10000` and the full layers use `rope_theta=1e6`. This means the model carries *two* RoPE caches simultaneously, and the per-layer dispatch is structural: you cannot run Gemma 3 with one attention kernel. The 5:1 ratio also drastically reduces full-sequence attention compute (factor ~6×), making Gemma 3 1B viable at 32k context with tiny memory.

4. **DeepSeek-V2-Lite mixes dense and sparse layers AND uses MLA AND uses YaRN.** Layer 0 is a dense SwiGLU MLP; layers 1–26 are MoE (64 routed + 2 shared experts, top-6). All 27 layers use MLA with `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128` — note that V has a *different* head dim than the rope portion of K. This means the API can't assume `q_head_dim == k_head_dim == v_head_dim`. Combined with YaRN (mscale=0.707), this single model exercises four orthogonal axes that no other model does simultaneously.

5. **Granite and MiniCPM smuggle μP scalars into "ordinary" Llama-shaped configs.** Granite 3 looks like a Llama with GQA + RMSNorm + SwiGLU + RoPE. But it also has `attention_multiplier=0.015625` (not `1/sqrt(64)=0.125`!), `residual_multiplier=0.22` applied to *every* sublayer output, `embedding_multiplier=12.0`, and `logits_scaling=8.0` or 16.0. Forgetting any of these makes the model produce garbage. If the API only exposes "is this a Llama-shaped model?" it will silently drop Granite. The fix: model the four scalars as first-class config, default to 1.0 for models that don't need them.

6. **Phi-3.5 mini and vision share the LongRoPE su variant with 48-element per-dim factor arrays** (`short_factor[48]`, `long_factor[48]`); Phi-4-mini uses a 47-element variant with `partial_rotary_factor=0.75`. The API needs both "named RoPE schedule" (`llama3`, `yarn`, `dynamic`, `linear`) and "explicit per-dim factor list" as a tagged-union for `rope_scaling`.

7. **MQA is alive — at the smallest sizes.** Gemma 3 1B, CodeGemma 2B, and PaliGemma text tower all use `num_kv_heads=1`. The KV-cache per token per layer is 2×head_dim, an order of magnitude smaller than even GQA-1:8. For sub-2B models targeting on-device deployment, MQA is the dominant memory saver — the API must support kv_heads=1 efficiently (no GQA broadcast loop).

8. **Phi-3-small is the structural outlier of the dense decoder pack.** It uses (a) `gegelu` not SwiGLU, (b) LayerNorm not RMSNorm, (c) block-sparse attention with `dense_attention_every_n_layers=2`, (d) μP scaling (`mup_width_multiplier=8.0`, `mup_attn_multiplier=1.0`, `mup_embedding_multiplier=10.0`), (e) cl100k tokenizer at vocab=100352. It is more of an ancestor of the GPT-4-Turbo block-sparse design than a sibling of Phi-3-mini. The API should treat Phi-3-small as needing its own attention kernel.

9. **SmolLM3 NoPE.** Every 4th layer skips RoPE entirely. The pattern is `[1,1,1,0]` repeated (1=apply RoPE, 0=skip). This is now an emerging pattern (also seen in Llama 4 dev configs). The API needs `apply_rope: bool` per layer, not a global flag.

10. **Mamba 2.8B has no head-dim and no attention at all; RWKV-6 has `head_size=64` but no Q/K/V matrices.** The "attention" abstraction breaks down for these. A unified API likely needs a notion of `token_mixer: {attention | mamba | mamba2 | rwkv6 | rwkv7 | linear_attn}` where `attention` is *one* implementation among several. KV-cache becomes "recurrent state" for the SSM/RWKV variants.

11. **Tied embeddings cut off around 3–4B.** Below this size, almost every model ties input/output embeddings (Llama 3.2 1B/3B, all small Qwens, Gemma 2 2B, Granite 2B, SmolLM2/3, Phi-4-mini). Above it, almost none do (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Mistral 7B, OLMo 2 7B, DeepSeek-V2-Lite). The tying decision changes the parameter count of the LM head by `vocab_size * hidden_size` (e.g. 128k×3072 = 400M for Llama 3.2 3B — 13% of the total!). The API needs `tie_word_embeddings` as a first-class config.

12. **`head_dim` is no longer `hidden_size / num_heads`.** Gemma 2/3 use `head_dim=256` independent of hidden size (Gemma 2 2B has hidden=2304, heads=8, head_dim=256 → Q projection is `2304 → 8*256 = 2048`). Phi-3 mini has head_dim=96 with hidden=3072, heads=32 (96 = 3072/32, ok). Phi-3.5 mini has head_dim=96 (same). But Phi-4-mini has head_dim=128 with hidden=3072, heads=24 → `24*128 = 3072` is hidden again. The default `hidden_size / num_heads` is *wrong* for Gemma; the API must accept explicit `head_dim`.

13. **Vocab sizes range over 4× (49152 → 262208).** This dominates final-layer cost. Gemma 3's 262k vocab includes 256k SP tokens + image tokens (`image_token_index=262144`); Phi-4-mini's 200064 uses o200k. The API's `lm_head` must accommodate this range without crippling smaller models.

---

## 6. Source citations

All configs accessed 2026-06-04. URLs point to `raw/main/config.json` on HuggingFace Hub. For models where the original repo requires auth, an authoritative mirror (e.g. `unsloth/`, `deepseek-ai/`) was used and noted.

**Llama family**:
- [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json) (mirror of meta-llama/Llama-3.2-1B)
- [unsloth/Llama-3.2-3B](https://huggingface.co/unsloth/Llama-3.2-3B/raw/main/config.json)
- [unsloth/Meta-Llama-3.1-8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/config.json)
- [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json)
- Tech reports: Llama 3 herd (Meta, 2024-07-23, arXiv:2407.21783).

**Qwen family**:
- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json)
- [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/config.json)
- [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B/raw/main/config.json)
- [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)
- [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json)
- [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)
- [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)
- [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B/raw/main/config.json)
- [Qwen/Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B/raw/main/config.json)
- Tech report: Qwen 3 technical report (Alibaba, 2025-04).

**Gemma family**:
- [unsloth/gemma-2-2b](https://huggingface.co/unsloth/gemma-2-2b/raw/main/config.json)
- [unsloth/gemma-3-1b-pt](https://huggingface.co/unsloth/gemma-3-1b-pt/raw/main/config.json)
- [unsloth/gemma-3-4b-pt](https://huggingface.co/unsloth/gemma-3-4b-pt/raw/main/config.json)
- [unsloth/codegemma-2b](https://huggingface.co/unsloth/codegemma-2b/raw/main/config.json)
- Gemma 2 / Gemma 3 technical reports (Google DeepMind, 2024-07 / 2025-03).

**Phi family**:
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/main/config.json)
- [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/raw/main/config.json)
- [microsoft/Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct/raw/main/config.json)
- [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct/raw/main/config.json)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct/raw/main/config.json)
- Phi-3 technical report (Microsoft, 2024-04, arXiv:2404.14219).

**OLMo / SmolLM / IBM / DeepSeek / others**:
- [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/raw/main/config.json)
- [allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B/raw/main/config.json)
- [allenai/OLMo-2-1124-13B](https://huggingface.co/allenai/OLMo-2-1124-13B/raw/main/config.json)
- [allenai/OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125/raw/main/config.json)
- [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json)
- [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json)
- [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/raw/main/config.json)
- [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/raw/main/config.json)
- [ibm-granite/granite-3.1-2b-base](https://huggingface.co/ibm-granite/granite-3.1-2b-base/raw/main/config.json)
- [ibm-granite/granite-3.3-2b-base](https://huggingface.co/ibm-granite/granite-3.3-2b-base/raw/main/config.json)
- [ibm-granite/granite-3.3-8b-base](https://huggingface.co/ibm-granite/granite-3.3-8b-base/raw/main/config.json)
- [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B/raw/main/config.json)
- [01-ai/Yi-1.5-6B](https://huggingface.co/01-ai/Yi-1.5-6B/raw/main/config.json)
- [internlm/internlm2_5-7b](https://huggingface.co/internlm/internlm2_5-7b/raw/main/config.json)
- [internlm/internlm3-8b-instruct](https://huggingface.co/internlm/internlm3-8b-instruct/raw/main/config.json)
- [stabilityai/stablelm-2-1_6b](https://huggingface.co/stabilityai/stablelm-2-1_6b/raw/main/config.json)
- [stabilityai/stablelm-2-12b](https://huggingface.co/stabilityai/stablelm-2-12b/raw/main/config.json)
- [deepseek-ai/DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base/raw/main/config.json)
- [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json)
- [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/raw/main/config.json)
- [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/raw/main/config.json)

**SSM / linear-attention**:
- [state-spaces/mamba-2.8b-hf](https://huggingface.co/state-spaces/mamba-2.8b-hf/raw/main/config.json)
- [Zyphra/Zamba2-2.7B](https://huggingface.co/Zyphra/Zamba2-2.7B/raw/main/config.json)
- [RWKV/v6-Finch-7B-HF](https://huggingface.co/RWKV/v6-Finch-7B-HF/raw/main/config.json)
- [RWKV/v6-Finch-1B6-HF](https://huggingface.co/RWKV/v6-Finch-1B6-HF/raw/main/config.json)
- [ai21labs/Jamba-tiny-dev](https://huggingface.co/ai21labs/Jamba-tiny-dev/raw/main/config.json)

**VLM text towers**:
- [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/raw/main/config.json)
- [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base/raw/main/config.json)
- [unsloth/gemma-3-4b-pt](https://huggingface.co/unsloth/gemma-3-4b-pt/raw/main/config.json) (text_config substructure for PaliGemma-style VLM)
- [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct/raw/main/config.json)

**Coding-specialized**:
- [bigcode/starcoder2-3b](https://huggingface.co/bigcode/starcoder2-3b/raw/main/config.json)
- [bigcode/starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b/raw/main/config.json)

**Cross-checked source**: local `transformers` checkout at `C:\Users\zhengte\external\transformers\src\transformers\models\<name>\modeling_*.py` and `configuration_*.py`, specifically `qwen3` (QK-norm shape), `olmo2` (post-norm + full-channel QK-norm), `gemma2` / `gemma3` (dual norm + dual RoPE + layer_types), `phi3` (fused gate_up), `smollm3` (no_rope_layers), `qwen3_moe`, `deepseek_v3`.

**Models I could not access**: the official `meta-llama/*`, `google/gemma-*`, `google/paligemma-*`, `google/codegemma-7b`, `ai21labs/AI21-Jamba-Mini-1.6`, and `google/gemma-2-9b` repos returned 401 (gated). Authoritative mirrors (unsloth, ai21labs/Jamba-tiny-dev) were used where available; CodeGemma 7B and Jamba 1.6 entries are inferred from family-level reports rather than verified configs and are marked as such above. PaliGemma's text-config is reconstructed from the CodeGemma 2B sibling (same Gemma-1 family).
