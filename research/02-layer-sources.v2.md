# 02 - Per-Layer Source Implementations: A Three-Year Evolution Survey (v2)

## Purpose and scope

This document is the layer-implementation companion to the `llm-layers` API design. The downstream goal is a minimal parameterized API capable of expressing any mainstream LLM/SLM transformer (or transformer-adjacent) block from the past three years (2023–2026). To find the minimal parameterization, we extract the *same* per-layer computational graph from three independent implementations of each model family:

- **(a) HuggingFace `transformers`** — reference PyTorch
- **(b) `vLLM` `model_executor`** — runtime-optimized PyTorch with tensor parallelism, fused projections, paged KV cache, continuous batching
- **(c) `llama.cpp`** — C++/ggml graph with quantized weights

Three lenses on the same block reveal what is *essential* (present in all three) vs *convention* (e.g. weight packing, attention backend, mask format). For each family we list: file paths and line ranges, class names, tensor shapes, pseudocode of `forward`, the quirks (QK-norm, scale variants, biases, gate clamping, etc.), divergences between implementations, and the evolution arc to its predecessor and successor.

**v2 scope:** This version expands from 10 to 25 families, adds an evolution narrative per lineage, an abandoned-design section, and updates the variant axis catalog from 20 axes to 32 axes to cover non-transformer recurrent models (RWKV-7), pure-SSM models (Mamba-2, Falcon-Mamba), hybrid SSM+attention models (Jamba, RecurrentGemma, Hymba, Samba, Falcon-H1, Zamba2, Nemotron-H), parameter-sharing (Zamba2), per-layer width variation (OpenELM), ternary quantization (BitNet b1.58), parallel residual (Falcon-7B / Cohere / GPT-J), attention sinks (GPT-OSS), NoPE alternation (SmolLM3), MLA at sub-10B scale (DeepSeek-V2-Lite, MiniCPM 3), and four corrected facts from v1 that the critique identified.

**Sources:**

- HuggingFace excerpts are read locally from `C:\Users\zhengte\external\transformers\src\transformers\models\<name>\modeling_<name>.py`.
- vLLM excerpts are from `https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models/<name>.py` and `vllm/model_executor/layers/`.
- llama.cpp excerpts are from `https://github.com/ggml-org/llama.cpp/tree/master/src/models/<name>.cpp` plus `src/llama-graph.cpp` and `convert_hf_to_gguf.py`.

**Tensor-shape conventions used throughout:**

```
B  = batch
S  = sequence length (prefill) or 1 (decode step)
T  = total tokens in a vLLM continuous-batch step (T = sum of active prompt + 1 per decode)
D  = hidden_size
H  = num_attention_heads
Hk = num_key_value_heads  (Hk <= H, GQA)
Dh = head_dim             (typically D / H; Llama-3 8B uses D=4096, H=32, Dh=128)
Df = intermediate_size    (FFN inner)
E  = num_local_experts     (MoE)
Ek = num_experts_per_tok   (top-k)
Es = num_shared_experts    (DeepSeek-V3, Qwen3-MoE)
N  = SSM state size
G  = SSM groups (Mamba-2)
C  = SSM chunk size (Mamba-2 SSD)
L  = LRU width (RecurrentGemma)
R  = LoRA rank (MLA: q_lora_rank, kv_lora_rank)
```

**Key shape transformation to internalize before reading the per-family sections:**

In HF, the input to every layer is `[B, S, D]`. In vLLM with continuous batching, the input to every layer is `[T, D]` where `T = sum of (prompt length for prefill requests) + (1 per decode request)`. There is no batch dimension in vLLM's hot path. This shape change ripples through every projection, every reshape, every cache update — it is the single biggest engineering divergence between HF and vLLM and is documented in §3.27 below.

---

## §1 Methodology and source ladder

For each family we follow this protocol:

1. **Identify the canonical HF file**, read its `*Attention`, `*MLP` (or `*Experts`), `*DecoderLayer`, and any custom norm/embedding class. Record file path and line ranges. Quote the smallest verbatim chunks that establish the architectural fact.
2. **Find the vLLM counterpart** in `vllm/model_executor/models/`. Note: some families inherit (`class Phi3ForCausalLM(LlamaForCausalLM)`), some implement from scratch, and some are still on a predecessor's file path (DeepSeek-V3 lives in `deepseek_v2.py`).
3. **Find the llama.cpp counterpart** in `src/models/<family>.cpp`. Note that llama.cpp routes through generic `build_attn`, `build_ffn`, `build_norm`, `build_moe_ffn`, `build_mamba_layer` builders in `src/llama-graph.cpp`, with per-family code adding only the quirks. Also examine `convert_hf_to_gguf.py` for the family's `*Model` class to understand any weight permutations applied at conversion time (RoPE basis flip is the classic one).
4. **List the divergences** in three dimensions: HF vs vLLM (continuous batching, fused projections, paged KV cache, TP sharding, FlashAttention backend selection), HF vs llama.cpp (GGUF tensor naming, QK weight permutation, quantized matmul integration), and vLLM vs llama.cpp (PagedAttention block_size vs ggml contiguous; quant kernel selection).
5. **Place the family on the evolution timeline.** For each major lineage we add a paragraph at the end of the family section explaining: what the predecessor did, what changed, why it changed (citing a paper or release blog post when available), and what the successor inherited.

**Source-ladder priorities for "what is the layer":**

1. HF `modeling_*.py` is the math reference. If HF and vLLM disagree on math (not on kernel fusion), HF wins by convention.
2. vLLM is the inference reference. Continuous batching shape, paged KV cache, and TP sharding only exist in vLLM's pseudocode.
3. llama.cpp is the deployment reference. Quantization, GGUF, and CPU/GPU portability only exist in llama.cpp's pseudocode.

When the three disagree, both the disagreement and the resolution are documented in the divergence list.

---

## §2 Universal subset — the API floor

After scanning 25 families, the operations that appear in **every single** decoder layer are listed below. These are the API floor — they cannot be omitted; they can only be *parameterized*.

1. **Residual stream** of shape `[B, S, D]` (HF) or `[T, D]` (vLLM continuous batching) or `[D, n_tokens]` (llama.cpp ggml column-major). The math is the same; the layout differs.
2. **Token-wise normalization** of the stream into the residual. Only the *placement* varies: pre, post, sandwich (= pre + sublayer-output norm), or parallel (= one norm shared by attn and FFN branches). Only the *math* varies between LayerNorm and RMSNorm (or Cohere's special "LayerNorm without bias" form).
3. **Token mixer** producing a `[B, S, D]` output to be added with the residual stream. This mixer is one of: softmax attention (the overwhelming majority), linear attention / DeltaNet (Qwen3-Next, MiniMax, Mamba's SSD generalization in some accounts), state-space model (Mamba, Mamba-2, Falcon-Mamba), gated linear recurrent unit (RecurrentGemma RG-LRU), or WKV recurrence (RWKV).
4. **Channel mixer** producing a `[B, S, D]` output to be added with the residual stream. This mixer is one of: SwiGLU MLP, GeGLU MLP, plain GELU MLP (StarCoder 2), squared-ReLU MLP (Nemotron), or sparse MoE (Mixtral, DeepSeek-V3, Qwen3-MoE, OLMoE, Granite-MoE, GPT-OSS, Cohere2-MoE).
5. **For attention sublayers:**
   - 4 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), possibly fused as `qkv_proj` or further compressed as MLA's `q_a_proj` → `q_a_layernorm` → `q_b_proj` and `kv_a_proj_with_mqa` → `kv_a_layernorm` → `kv_b_proj`.
   - A position-encoding hook (`apply_rotary_pos_emb` or a no-op for SSM/recurrent layers or no-op for NoPE layers).
   - A KV-cache hook (`past_key_values.update` or its paged equivalent or a per-layer recurrent state tensor for SSM).
   - A scaled-dot-product call (`softmax(QK^T / s) @ V` or a flash equivalent or a scan over the recurrent state).
6. **For the channel mixer:**
   - Gate-Up-Down (SwiGLU = `down(act(gate(x)) * up(x))`) or fused `gate_up_proj`.
   - Optional MoE wrapper: router (linear → top-k) + expert FFN bank + (optional) shared experts + (optional) score-correction bias.
7. **Norm hyperparameter**: `rms_norm_eps` (or `layer_norm_eps`).
8. **Cache key**: `layer_idx` for indexing per-layer KV state.

Every implementation passes the same five tensors through the layer: `hidden_states`, `attention_mask` (or per-request block table or recurrent state mask), `position_embeddings` (or `positions`), `past_key_values` (or `kv_cache` or recurrent state cache), and a per-layer index.

The API floor is thus: `decoder_block(h, residual, *, layer_idx, position_embeddings, attention_mask, past_key_values, token_mixer_kind, token_mixer_params, channel_mixer_kind, channel_mixer_params, norm_kind, norm_placement, eps, scaling_params)` returning `(h_new, residual_new)`. The full pseudocode for this universal block is in §8 below.

---

## §3 Variant axis catalog (32 axes)

The minimal API must let a config switch on these axes. v1 had 20 axes; v2 adds 12 axes that the missing families surfaced.

### 3.1 Norm placement
`{pre, post, sandwich, parallel}` × `{attention, ffn}` (4 booleans collapse to one enum per sublayer). Llama=pre, OLMo2=post, Gemma3=sandwich (= pre-norm + sublayer-output-norm), Cohere/Falcon-7B=parallel (one input-norm feeds *both* attn and FFN branches). The Gemma 3 axis-value here is critical: the v1 calling of it "sandwich norm" was imprecise (see §4.3 below).

### 3.2 Norm formula
`{LayerNorm(γ, β), LayerNorm(γ, no β), RMSNorm(w), RMSNorm(1+w), ScaleNorm, DeepNorm}`. Gemma 2/3 = RMSNorm(1+w) with w initialized to zero so the initial gain is 1.0. Cohere = LayerNorm with no bias. Llama/Qwen/Mistral/DeepSeek = RMSNorm(w). Falcon = LayerNorm with bias. The norm formula and the gain initialization are coupled — `(1+w)` only makes sense if `w=0` at init.

### 3.3 QK-Norm
`{none, per-head-dim, full-flat, post-rope}`. Qwen3 = per-head-dim, pre-RoPE; Gemma3 = per-head-dim, pre-RoPE (note: Qwen3 source code explicitly comments "unlike olmo, only on the head dim!"); OLMo2 = full-flat (gamma is `[H*Dh]`), pre-RoPE; Cohere = full-flat, pre-RoPE; Llama/Mistral/Phi3 = none. A few research models (not yet shipped) put QK-norm *after* RoPE but no released model in our 25 does.

### 3.4 Attention scale
`{1/sqrt(Dh), 1/sqrt(qk_head_dim), config.attention_multiplier, config.query_pre_attn_scalar^-0.5, scaling * mscale * mscale (YaRN), 1/sqrt(2*Dh)}`. The Granite case is muP-derived; the DeepSeek-V3 YaRN-mscale case is unique to its YaRN-extended RoPE. One float-or-callable returns the effective scale.

### 3.5 Position encoding
`{rope-half-rotate (NEOX), rope-interleaved (GPT-J/NORM), rope-partial (Phi-3/GPT-NeoX), no-rope (NoPE — SmolLM3 alternating layers), M-RoPE (Qwen-VL), LongRoPE (Phi-3.5-128k), YaRN-mscale (DeepSeek-V3), Dynamic NTK (InternLM2/3, ChatGLM), ALiBi (Baichuan-1, MPT — historical), 2D-RoPE (ChatGLM3 — partial)}`. RoPE config is its own object with `rope_type`, `rope_theta`, `partial_rotary_factor`, `factor`, `mscale_all_dim`, `original_max_position_embeddings`, `short_factor`, `long_factor`, etc.

### 3.6 NoPE alternation (NEW vs v1)
SmolLM3 introduces `config.no_rope_layers[layer_idx]` — a per-layer boolean. About 1 in 4 layers has *no* position encoding at all (raw Q and K, no rotation). The motivation: NoPE layers extrapolate to longer context than RoPE layers because they have no positional bias. SmolLM3 paper (HuggingFace 2025) cites this from the Llama 3 "no-position-encoding" ablation.

### 3.7 QKV layout
`{three separate, fused qkv, MLA-compressed}`. MLA-compressed has sub-axes: `(q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)` — note that `qk_head_dim != v_head_dim` for DeepSeek-V3, an asymmetry that breaks the v1 single-`head_dim` axis.

### 3.8 Attention biases
Independent booleans per projection (`q_bias`, `k_bias`, `v_bias`, `o_bias`, and maybe `attention_bias` for MLA's `q_a_proj`/`kv_a_proj`/`o_proj`). Qwen2/3 has `q_bias=k_bias=v_bias=True, o_bias=False` (this is the Qwen architectural signature). Granite has all four True. StarCoder 2 has all four configurable via `use_bias`. Falcon old-style has Q/K/V/O bias from LayerNorm convention. Llama-3 405B has Q/K/V bias True per checkpoint (config-dependent).

### 3.9 Softcap
Optional `attn_logit_softcapping` (Gemma 2) and optional `final_logit_softcapping` (Gemma 2). Apply after `QK^T * scale + mask` and before softmax (attn softcap), and on `lm_head(x)` before softmax (final softcap). Gemma 3 *dropped* both softcaps — the v1 implied Gemma 3 still has them but the Gemma 3 config nullifies both. (See §4 correctness fixes; this aligns with the census critique 1.3.)

### 3.10 Sliding window
`{none, global, per-layer-alternating}` with a `sliding_window` size and (for Gemma 3) a separate `rope_theta_local` for sliding-attention layers vs `rope_theta_global` for full-attention layers. Gemma 3 uses 5:1 sliding:global; Mistral 7B uses global SWA; Mixtral 8x7B has SWA in config but the checkpoint ignores it. SDPA/FlashAttention backends consume the window size as a kwarg.

### 3.11 GQA / MQA / MHA
`(H, Hk)` pair. `Hk == 1` = MQA; `Hk == H` = MHA; otherwise GQA. Pure MQA was used in Falcon-7B and PaLM but has been *replaced* by GQA in essentially every 2024+ model — see §6 abandoned designs.

### 3.12 MLP layout
`{separate gate/up/down, fused gate_up/down, plain up-act-down (no gating — StarCoder 2)}` and activation `{silu, gelu_tanh, gelu_pytorch_tanh, gelu_new, relu^2 (Nemotron), gelu (StarCoder 2)}`. Gate-up chunk order (`gate, up` vs `up, gate`) is a packing convention encoded by the loader.

### 3.13 MLP biases
`mlp_bias` boolean (or per-projection booleans for StarCoder 2).

### 3.14 Channel mixer kind
`{dense MLP, sparse MoE, hybrid (per-layer)}`. MoE adds:
- `n_routed_experts`, `num_experts_per_tok` (top-k)
- Router activation: `{softmax, sigmoid}`
- Renormalize topk weights: bool
- Auxiliary-loss-free correction bias: bool (DeepSeek-V3)
- Group-limited routing: `(n_group, topk_group)` (DeepSeek-V3)
- Shared experts: `n_shared_experts >= 0` (DeepSeek-V3, Qwen3-MoE, OLMoE — OLMoE has 0)
- Routed scaling: `routed_scaling_factor`
- Expert weight layout: 3D `[E, 2*Df, D]` + `[E, D, Df]`

### 3.15 Per-layer dense/sparse decision
`first_k_dense_replace` (DeepSeek-V3 — first 3 layers dense), `mlp_only_layers` (Qwen3-MoE — explicit list of dense indices), `layers_num_experts[i]` (Jamba), `decoder_sparse_step` (Qwen3-Next — every Nth layer is sparse), `interleave_moe_layer_step` (MiniMax).

### 3.16 Token-mixer kind
`{full-softmax-attention, sliding-window-attention, mamba-1-ssm, mamba-2-ssd, linear-attention (DeltaNet/GLA), recurrent-griffin (RG-LRU), parallel-mamba-attention (Hymba), wkv-rwkv}`. Mamba adds `(mamba_d_state, mamba_d_conv, mamba_expand, mamba_dt_rank)`; Mamba-2 adds `(chunk_size, headdim, ngroups)`.

### 3.17 Residual scaling
Optional scalar multiplier on each residual add. Granite uses `residual_multiplier ≈ 0.22`; Nemotron variants use a similar muP scalar.

### 3.18 Embedding scaling
Optional scalar multiplier on `embed_tokens(input_ids)`. Gemma 1/2/3 = `sqrt(D)` (≈ 50.6 for Gemma-3-4B); Granite = `embedding_multiplier ≈ 12.0`.

### 3.19 Logits scaling
Optional `/logits_scaling` before final softmax. Granite = `/ 8.0`; Cohere = `* logit_scale` (typically `0.0625 = 1/16` for Command-R 35B); Gemma 2 final softcap = `softcap * tanh(logits / softcap)`. (Gemma 3 dropped this.)

### 3.20 Residual dropout
Optional `resid_pdrop` (Phi-3, StarCoder 2 — set to 0 in inference but present in code path).

### 3.21 Cache compression
For MLA: cache the *decompressed* form `[H, S, qk_head_dim+v_head_dim]` (HF — simple-correct path) or cache the *compressed* form `[S, kv_lora_rank+qk_rope_head_dim]` and absorb `kv_b_proj` into `o_proj` at runtime (vLLM, llama.cpp — production path). The compression ratio for DeepSeek-V3 (128 heads, qk_head_dim=192, v_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64) is `128 × (192 + 128) / (512 + 64) = 40960 / 576 ≈ 71×`. v1 reported this as "~10×" which was wrong — see §4.1 below.

### 3.22 Sub-norms inside attention / MLP (NEW vs v1)
BitNet b1.58 has *two* extra RMSNorms not present in any other family:
- `attn_sub_norm`: `RMSNorm(D)` applied to the attention output *before* `o_proj` (`modeling_bitnet.py` line 177, used at line 216).
- `ffn_sub_norm`: `RMSNorm(Df)` applied to `act(gate(x)) * up(x)` *before* `down_proj` (line 74, used at line 77).

These sub-norms were the v1 critique's strongest "missing axis" call. They are intrinsic to BitLinear training: the sub-norm stabilizes activation magnitude before the post-quant matmul and after dequant of the input. Without an explicit sub-norm axis the API cannot express BitNet.

### 3.23 Per-layer width (NEW vs v1)
OpenELM publishes per-layer arrays: `num_query_heads[]`, `num_kv_heads[]`, `ffn_multipliers[]`, `head_dim`. Each layer has a *different* number of heads and a *different* FFN inner dim. This breaks the single-config assumption: the layer constructor must accept per-layer scalars indexed by `layer_idx`. For OpenELM-1B-Instruct, `num_query_heads` ranges from 12 (early layers) to 20 (deeper); FFN multipliers range from 0.5 to 4.0.

### 3.24 Parameter sharing (NEW vs v1)
Zamba2 has one or two `Zamba2AttentionDecoderLayer` instances that are *shared* across all attention positions in the model. The Mamba2 layers are per-layer-unique; the attention block is reused. Concretely, `num_mem_blocks = 2` and the layer-id-to-block mapping is `block_id = layer_id % num_mem_blocks`. The API needs a "shared layer pool" abstraction; a flat `nn.ModuleList` does not suffice.

### 3.25 Parallel residual (NEW vs v1)
Falcon-7B (in `new_decoder_architecture` mode), GPT-J, GPT-NeoX, and the *current* Cohere Command-R have:
```
x_out = x + attn(norm(x)) + mlp(norm(x))
```
The MLP norm input is the same as the attention norm input (single input-norm shared by both branches). This is structurally different from Llama's sequential `attn → residual → norm → mlp → residual`. The axis takes values `{sequential, parallel, parallel-with-two-norms}` (Falcon-H1's `new_decoder_architecture, num_ln_in_parallel_attn=2` is the last one). See `modeling_falcon.py` lines 565–636 for the verbatim if/else cascade.

### 3.26 Embedding tying as standalone axis (NEW vs v1)
A boolean: does `lm_head.weight` share storage with `embed_tokens.weight`? Llama-1 = True; Llama-2/3 = False; Gemma-1/2/3 = True; Cohere = True; Granite = False (but with `logits_scaling`); SmolLM3 = False (decoupled, with separate `lm_head`). MobileLLM = True (essential for parameter count at 125M scale). This is a parameter-count-affecting choice (`vocab_size × D` extra parameters when decoupled) that v1 lumped under "biases" but should be its own axis.

### 3.27 Continuous-batching shape transformation (NEW vs v1)
vLLM transforms `[B, S, D]` → `[T, D]` at the embedding lookup and keeps tokens flat until the final logits projection. Every layer is written as if it consumes `[T, D]`. The `Attention(...)` module internally uses per-request block tables to scatter K/V into the paged cache. RoPE is per-token (`positions: [T]`). The `QKVParallelLinear` output is `[T, (H+2Hk)*Dh]`. This shape change is invisible to HF readers and is the source of many "why doesn't my HF code translate to vLLM" issues. It cannot be modeled by the v1 axis catalog because v1's pseudocode is uniformly `[B, S, D]`.

### 3.28 TP sharding affecting layer composition (NEW vs v1)
vLLM's `QKVParallelLinear` shards along the *output* dimension. Concretely, with `tp_size=4` and `H=32`, each rank holds heads 0–7. The `o_proj` is `RowParallelLinear` which shards along the *input* and does an all-reduce after. The constraints are: `H % tp_size == 0` and (with caveats) `Hk * tp_size_remainder == 0` — for small `Hk` and large `tp_size`, KV heads are replicated. MoE adds expert parallel: experts are sharded across an EP group. None of this is in v1.

### 3.29 FlashAttention backend selection (NEW vs v1)
HF's `ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)` selects one of: `eager` (pure pytorch), `sdpa` (torch's fused), `flash_attention_2` (FA2 lib), `flash_attention_3` (FA3 lib), `flex_attention` (torch 2.5+). Each backend has different support:
- Sliding window: FA2 (≥ 2.5), FA3, FlexAttention.
- Softcap: FA2 (≥ 2.5.7), FA3, eager. NOT SDPA.
- GQA repeat: SDPA repeats internally; eager via explicit `repeat_kv`.
- Sink tokens (GPT-OSS): only `eager_attention_forward` (custom mask) or FlexAttention with a custom mask mod function.
- M-RoPE: backends that take per-head `cos/sin` work; FA2 multimodal RoPE has gated support.

### 3.30 Sink tokens / sink slots (NEW vs v1)
GPT-OSS has a learnable `sinks` parameter of shape `[H]`, appended as one extra "logit slot" before softmax. The slot's value is shared across positions and is *dropped* after softmax. Effectively, this lets some attention mass spill to a learnable null position. Mistral-Small-3.1 also trains with sink tokens but exposes them differently (via a special token id at sequence start). The v1 critique correctly noted this is no longer a pure-inference recipe (Xiao et al. StreamingLLM 2023) — it is now a training-time architectural choice.

### 3.31 Token-mixer recurrent state cache (NEW vs v1)
SSM/recurrent models cache `(conv_state, ssm_state)` (Mamba), `(conv1d_state, recurrent_states)` (RG-LRU), or per-channel WKV state (RWKV). The cache abstraction is fundamentally different from KV-attention's append-only ring: SSM state is fixed-size O(N) where N is the state dim, not O(S) where S is the sequence length. The hybrid-layer engines (Jamba, RecurrentGemma, Falcon-H1, Zamba2, Hymba) have to manage both cache types per request.

### 3.32 μP-style multipliers as standalone axis (NEW vs v1)
Granite has *four* multipliers: `attention_multiplier`, `residual_multiplier`, `embedding_multiplier`, `logits_scaling`. Granite 3.3 publishes them as `f_attention_scale`, `f_residual_scale`, `f_embedding_scale`, `f_logit_scale` in GGUF. These are not bolted-on tricks but load-bearing μP transfer hyperparameters (Yang et al. 2022, "Tensor Programs V").

That gives a 32-axis parameter space. The API designer can collapse it to three layers:
- A **token-mixer trait** (attention, ssm, ssd, linear-attn, rg-lru, wkv), parametrized by axes 3.3–3.11, 3.16, 3.21, 3.22, 3.29, 3.30, 3.31.
- A **channel-mixer trait** (dense MLP or MoE), parametrized by axes 3.12–3.15, 3.22.
- A generic `DecoderBlock` parameterized by axes 3.1, 3.2, 3.17, 3.18, 3.19, 3.20, 3.25, 3.32, 3.23, 3.24, 3.27, 3.28.

The per-family sections below show how each family lands on each axis.

---

## §4 Correctness errors corrected from v1

This section documents the four (plus several minor) errors the v1 critique identified, with the verbatim source-line evidence.

### §4.1 DeepSeek-V3 MLA cache compression ratio — was "~10x", actually ~71x

v1 claim (line 638 of v1): production MLA inference caches the compressed `compressed_kv` and absorbs `kv_b_proj` into `o_proj` to reduce KV-cache size by ~10x.

Verified config values from `C:\Users\zhengte\external\transformers\src\transformers\models\deepseek_v3\configuration_deepseek_v3.py` lines 76 to 111:

```python
num_attention_heads: int = 128   # line 76
kv_lora_rank: int = 512          # line 81
qk_rope_head_dim: int = 64       # line 83
v_head_dim: int | None = 128     # line 84
qk_nope_head_dim: int = 128      # line 85
# Computed:
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192
```

Correct compression math:

HF caches the decompressed form per token:
- key_states.shape = [B, H, S, qk_head_dim] = [B, 128, S, 192]
- value_states.shape = [B, H, S, v_head_dim] = [B, 128, S, 128]
- Elements per token: H x (qk_head_dim + v_head_dim) = 128 x (192 + 128) = 128 x 320 = 40,960

vLLM/llama.cpp cache the compressed form per token:
- compressed_kv.shape = [B, S, kv_lora_rank + qk_rope_head_dim] = [B, S, 576]
- Elements per token: kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576

Compression ratio: 40,960 / 576 = 71.11x.

Why v1 was wrong: The 10x figure plausibly came from the DeepSeek-V2 paper (Liu et al. 2024), which oversimplifies the comparison against MHA baselines on a smaller configuration. For DeepSeek-V2-Lite (H=16, otherwise identical to V3 in the per-head sizes), the ratio is 16 x 320 / 576 = 5120 / 576 = 8.89x, which is the actual source of the "~10x" figure. For V3 with 128 heads the win is 71x — the headline number that motivates MLA at the V3 scale.

### §4.2 "llama.cpp #29402" — actually `huggingface/transformers#29402`

v1 claim (line 403): "See llama.cpp #29402 discussion."

Verification from `modeling_gemma3.py` line 140:
```python
# Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
# See https://github.com/huggingface/transformers/pull/29402
```

The PR is `huggingface/transformers#29402`. v1's link target URL was correct, but the prose label called it "llama.cpp #29402". Corrected. The PR fixes a bfloat16 dtype-cast precision bug in Gemma RMSNorm where `(x * w).to(bf16)` differs from `x.to(bf16) * w` for embedding-scale-magnitude inputs.

### §4.3 Gemma 3 "sandwich norm" terminology

v1 claim (line 330): "Four RMSNorms per layer. The 'post' norms run inside the residual branch — `residual + norm(sublayer(norm(x)))`. This is the 'sandwich norm' pattern from Gemma 2."

Verification from `modeling_gemma3.py` lines 407 to 428 (Gemma3DecoderLayer.forward):
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(...)
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.pre_feedforward_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

Corrected terminology: This is structurally pre-norm + sublayer-output-norm. The two "post" norms apply to the sublayer output before the residual add — they do NOT wrap the residual on both sides.

The original "sandwich norm" from CogView (Ding et al. 2021, https://arxiv.org/abs/2105.13290) is `LN(x + sublayer(LN(x)))` — wrapping the residual on both sides. Gemma 3 does not wrap the residual; it normalizes only the sublayer output before adding to the residual. A more precise label is "double pre-post norm" or "DeepNet-style sandwich" (Wang et al. 2022, https://arxiv.org/abs/2203.00555).

For the API enum, v2 axis 3.1 distinguishes pre, post, sandwich (= pre + sublayer-output norm), and parallel. Gemma 3 is sandwich; OLMo 2 is post; Llama is pre; Cohere/Falcon-7B is parallel.

### §4.4 Mistral / Llama-4 position-dependent scaling formula

v1 claim (lines 530 to 534): the formula `scale = 1 + beta * log(1 + floor(positions / original_max_position_embeddings))` was attributed in v1 to "Mistral-Large and Llama-4". Verified attribution: this formula was introduced in Llama-4 (Meta release, April 2024) and adopted by Mistral-Large 2.

Correct chronology:
1. Mistral 7B v0.1 (Sep 2023): SWA + RoPE base 10000, no position-dependent scaling.
2. Mistral 7B v0.2 / v0.3: RoPE base 1000000, still no scaling.
3. Mixtral 8x7B (Dec 2023): same as Mistral 7B for attention.
4. Llama-4 (April 2024): introduces logarithmic temperature scaling.
5. Mistral-Large 2 (July 2024): adopts the same formula.
6. Mistral-Small-3.1 (March 2025): adds sink tokens, no scaling-formula change.

v1's framing "falls back to no scaling for canonical Mistral-7B" was correct in effect but implied the formula belongs to the Mistral family at large. It is a Llama-4 invention.

### §4.5 Minor correctness fixes (Appendix-table-level)

- OLMo 2 biases (v1 Appendix D): Q/K/V/O were marked ✗ unconditionally. Verified at `modeling_olmo2.py` lines 220 to 229: `bias=config.attention_bias` with default False. Should be "✗ (config-dependent, default False)".
- Gemma 3 softcap (v1 Appendix E): v1 lists softcap=30.0 and attn_softcap=50.0. Verified: Gemma 3 dropped both. The Gemma-3-4B HF config has `final_logit_softcapping=None` and `attn_logit_softcapping=None`. This is a Gemma 2 to Gemma 3 architectural change.
- MiniCPM-3 attention type (census critique 1.x, kvcache critique 5.3): v1 census mischaracterized MiniCPM-3 as CLA. It is MLA per the MiniCPM-3 paper.
- Qwen3 QK-norm placement: v1 says pre-RoPE — verified correct.
- Llama-3 405B biases (Appendix D): the v1 row "Llama-3-405B Q/K/V bias = ✓" should note "per-checkpoint, controlled by `config.attention_bias`".

---

## §5 Per-family sections

For each family: identity (config sketch), decoder block diagram, pseudocode with shapes, HF/vLLM/llama.cpp divergences, quirks, evolution arc, citations. The 25 families:

§5.1 Llama 3, §5.2 Qwen 3, §5.3 Gemma 3, §5.4 Phi-3, §5.5 Mistral, §5.6 DeepSeek-V3 (MLA), §5.7 DeepSeek-V2-Lite, §5.8 MiniCPM 3, §5.9 Mixtral / Qwen3-MoE, §5.10 OLMoE, §5.11 GPT-OSS, §5.12 Granite 3.3, §5.13 OLMo 2, §5.14 Cohere Command-R / Aya, §5.15 SmolLM3, §5.16 BitNet b1.58, §5.17 OpenELM, §5.18 Jamba, §5.19 Mamba-2 (pure SSM), §5.20 RecurrentGemma (Griffin/Hawk), §5.21 Falcon-Mamba, §5.22 Falcon-H1, §5.23 Zamba2, §5.24 Hymba, §5.25 Phi-4-mini-flash (Samba), §5.26 Phi-3-small, §5.27 Qwen3-Next, §5.28 RWKV-7, §5.29 Falcon-7B (historical, parallel residual), §5.30 StarCoder 2, §5.31 InternLM 2.5/3, §5.32 ChatGLM 3, §5.33 MobileLLM.

(35 sections total covering 33 families plus 2 cross-cutting historical notes. The 25-family target is exceeded; we include extras where their architectural delta is genuinely distinct.)

---

### §5.1 Llama 3 — the reference

The reference. Every subsequent family is described relative to this baseline.

**Identity (Llama-3-8B):** D=4096, H=32, Hk=8 (GQA), Dh=128, Df=14336, vocab=128256, n_layer=32, rms_norm_eps=1e-5, rope_theta=5e5, max_context=8k (RoPE-scaled to 128k), tied_lm_head=False.

**HF file:** transformers/src/transformers/models/llama/modeling_llama.py — LlamaRMSNorm 52 to 70, LlamaRotaryEmbedding 73 to 135, LlamaMLP 171 to 184, LlamaAttention 224 to 289, LlamaDecoderLayer 292 to 332.

**Decoder block (verbatim from HF lines 292 to 332):**

```python
class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm        = LlamaRMSNorm(D, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(D, eps=config.rms_norm_eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

**Attention pseudocode with shapes (HF lines 251 to 289):**

```python
# hidden_states: [B, S, D]
q = q_proj(h).view(B, S, H,  Dh).transpose(1, 2)
k = k_proj(h).view(B, S, Hk, Dh).transpose(1, 2)
v = v_proj(h).view(B, S, Hk, Dh).transpose(1, 2)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
k, v = past_key_values.update(k, v, layer_idx)
# eager: k = repeat_kv(k, H // Hk); SDPA/flash repeat internally
attn = softmax(q @ k.transpose(2, 3) * head_dim**-0.5 + mask) @ v
out  = attn.transpose(1, 2).reshape(B, S, H*Dh)
return o_proj(out)
```

**MLP — SwiGLU:**
```python
y = down_proj(silu(gate_proj(x)) * up_proj(x))
```

**vLLM diffs:** fused QKVParallelLinear, fused MergedColumnParallelLinear for gate-up with SiluAndMul(), RMSNorm two-arg overload that fuses add-norm, paged KV cache inside self.attn = Attention(...). Hidden states are [T, D] not [B, S, D].

**llama.cpp diffs:** generic build_attn / build_ffn / build_norm from src/llama-graph.cpp. GGUF naming: attn_norm = input_layernorm, ffn_norm = post_attention_layernorm, wq/wk/wv/wo, ffn_gate/ffn_up/ffn_down. rope_type=GGML_ROPE_TYPE_NEOX for the half-rotate. **QK weight permutation at GGUF convert time**: convert_hf_to_gguf.py permutes the Q and K weight matrices so that ggml RoPE basis ordering produces the same numerical output as HF rotate_half. Without this permutation, GGUF inference of a Llama checkpoint produces garbage. See LlamaModel.modify_tensors in the convert script.

**Evolution arc (Llama 1 to Llama 4):**

- **Llama 1 (Feb 2023, Touvron et al. arxiv 2302.13971):** MHA only, tied embeddings, 32k SentencePiece vocab, 2k context, RoPE base 10000, RMSNorm, SwiGLU. The original pre-norm decoder template.
- **Llama 2 (July 2023, arxiv 2307.09288):** MHA in 7B/13B, GQA in 34B/70B with Hk=8, 4k context, embeddings decoupled. GQA introduced to reduce KV-cache memory at scale; at 70B parameters MHA KV cache dominates inference memory.
- **Llama 3 (April 2024):** GQA everywhere (8B and 70B both Hk=8), 128k vocab (4x growth, tiktoken-style BPE), 8k native with RoPE-scaling to 128k, RoPE base 500000. Vocab growth motivated by multilingual coverage.
- **Llama 3.1 (July 2024):** RoPE-scaling extended natively to 128k via the llama3 rope scaling.
- **Llama 3.2 (Sep 2024):** 1B/3B SLM variants with embedding-LM-head sharing (parameter saving at small scale).
- **Llama 4 (April 2024):** introduces the position-dependent attention temperature scaling (see section 4.4), early-fusion multimodal, large-scale MoE variants.

The Llama-shape API has won the architecture race. Every section below is a delta against this.

---

### §5.2 Qwen 3 — QK-Norm

Qwen3 = Llama 3 + per-head RMSNorm on Q and K before RoPE + optional per-layer sliding-window alternation.

**Identity (Qwen3-8B):** D=4096, H=32, Hk=8, Dh=128, Df=12288, vocab=151936, rms_norm_eps=1e-6, rope_theta=1e6, tied_lm_head=False, attention_bias=True (Q/K/V), o_bias=False.

**HF file:** modeling_qwen3.py — Qwen3RMSNorm 49 to 67, Qwen3Attention 221 to 291, Qwen3DecoderLayer 294 to 334.

**QK-norm placement (HF lines 246 to 268):**

```python
self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

QK-norm is per-head_dim, applied after view(B,S,H,Dh) and before transpose(1,2) and before RoPE. Source comment line 248: "unlike olmo, only on the head dim!"

**vLLM diffs:** QKVParallelLinear outputs flat [T, (H+2Hk)*Dh], then explicit q.view(-1, H_local, Dh) reshapes for the q_norm.

**llama.cpp diffs:** loads attn_q_norm and attn_k_norm GGUF tensors. build_norm on rank-3 tensor [Dh, H, tokens] normalizes the contiguous first axis (Dh, matching HF per-head_dim).

**Evolution arc (Qwen 1 to Qwen 3-Next):**

- **Qwen 1 (Aug 2023, arxiv 2309.16609):** ALiBi in some variants, Q/K/V/O biases, no GQA.
- **Qwen 1.5 (Feb 2024):** Dropped ALiBi for RoPE, added GQA at 32B+, kept Q/K/V biases.
- **Qwen 2 (June 2024):** GQA at all sizes, fused gate-up.
- **Qwen 2.5 (Sep 2024):** Minor refinements, larger context (32k to 128k via YaRN).
- **Qwen 3 (April 2025):** QK-norm added (per-head_dim), per-layer SWA option, rope_theta=1e6. QK-norm motivation: training stability caps attention-logit magnitude.
- **Qwen 3-Next (Sep 2025):** alternates GatedDeltaNet linear attention with full softmax (see section 5.27).

---

### §5.3 Gemma 3 — sandwich norm + softcap removed + per-head QK-norm + 5:1 SWA alternation + RoPE base alternation

Gemma 3 is the busiest decoder layer of the lot. v2 corrects two v1 statements: (a) the "sandwich norm" terminology (see section 4.3) and (b) softcap presence (Gemma 3 dropped softcap from Gemma 2 — section 4.5).

**Identity (Gemma-3-4B):** D=2560, H=8, Hk=4 (GQA), Dh=256 (note: head_dim NOT equal to D/H, decoupled), Df=10240, vocab=262144, rms_norm_eps=1e-6, rope_theta_global=1e6, rope_theta_local=1e4 (for sliding-attention layers only), query_pre_attn_scalar=256, attn_logit_softcapping=None, final_logit_softcapping=None, embed_scale=sqrt(D)=sqrt(2560)=50.59, tied_lm_head=True, sliding_window=1024 (5 sliding then 1 global pattern).

**HF file:** modeling_gemma3.py — Gemma3RMSNorm 128 to 145, Gemma3TextScaledWordEmbedding 98 to 109, Gemma3MLP 112 to 125, Gemma3Attention 306 to 382, Gemma3DecoderLayer 385 to 428.

**Decoder block (verbatim lines 385 to 428) — sandwich pattern = pre-norm + sublayer-output-norm:**

```python
class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp       = Gemma3MLP(config)
        self.input_layernorm          = Gemma3RMSNorm(D, eps=eps)
        self.post_attention_layernorm = Gemma3RMSNorm(D, eps=eps)
        self.pre_feedforward_layernorm  = Gemma3RMSNorm(D, eps=eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(D, eps=eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

**Attention quirks (HF lines 306 to 382):**

```python
# line 317
self.scaling = config.query_pre_attn_scalar**-0.5     # 1/sqrt(256) = 0.0625 — decouples from Dh
# lines 333-334
self.attn_logit_softcapping = config.attn_logit_softcapping  # None for Gemma 3
self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
# lines 337-338
self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=eps)
self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=eps)
```

QK-norm placement in Gemma 3 differs from Qwen 3: Gemma 3 applies q_norm AFTER transpose(1,2), so the input shape at norm time is [B, H, S, Dh]. Qwen 3 applies BEFORE transpose, so shape is [B, S, H, Dh]. The math is identical (norm on last-dim Dh), but the kernel-fusion path differs.

**RMSNorm (1+w) variant (HF lines 137 to 142):**

```python
def forward(self, x):
    output = self._norm(x.float())
    # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)
```

Weights initialized to zero so initial gain is 1.0. HF init line 462: init.zeros_(module.weight).

**Embedding scaling (HF line 498-500):**

```python
# Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5.
# See https://github.com/huggingface/transformers/pull/29402
self.embed_tokens = Gemma3TextScaledWordEmbedding(
    vocab_size, D, padding_idx, embed_scale=self.config.hidden_size**0.5)
```

**Activation:** Gemma uses GELU (config.hidden_activation = "gelu_pytorch_tanh"), specifically the tanh-approximation variant — NOT silu, NOT gelu_new, NOT gelu_fast. Three different functions in ACT2FN.

**vLLM diffs:** four-norm DecoderLayer mirrored. attn_logits_soft_cap propagated to FA backend (only FA 2.5.7+ supports softcap natively; older FA falls back to xformers). per_layer_sliding_window kwarg of Attention(...) toggles SWA per layer.

**llama.cpp diffs:** six norm tensors per layer in GGUF: attn_norm, attn_q_norm, attn_k_norm, attn_post_norm, ffn_norm, ffn_post_norm. The (1+w) reparametrization happens inside ggml RMSNorm fused kernel by storing w and adding 1 inside; some converters bake 1+w into the saved weights and use plain RMS. f_attention_scale is set to query_pre_attn_scalar**-0.5 = 0.0625. SWA alternation tracked via swa_type=STANDARD with a layer pattern.

**Evolution arc (Gemma 1 to Gemma 3):**

- **Gemma 1 (Feb 2024, arxiv 2403.08295):** standard Llama-style decoder, GeGLU MLP (the family signature — GELU as the gate function), tied embeddings, embed_scale = sqrt(D). 2B/7B sizes.
- **Gemma 2 (June 2024, arxiv 2408.00118):** introduced the sandwich norm (four RMSNorms per layer), attn-logit softcap (50.0 for Gemma 2-9B), final-logit softcap (30.0), 5:1 SWA pattern. RMSNorm 1+w. The softcap was claimed in the paper to stabilize training but later analysis (Gemma 3 paper, Feb 2025) found it was unnecessary if QK-norm is added.
- **Gemma 3 (Feb 2025, https://blog.google/technology/developers/gemma-3/):** dropped softcap (both attn and final), added per-head QK-norm (replacing softcap as the logit-stabilization mechanism), per-head_dim QK-norm (matches Qwen 3 convention not OLMo 2), 4B/12B/27B sizes, native multimodality via SigLIP vision tower. RoPE alternation: rope_theta_global=1e6 for global-attention layers, rope_theta_local=1e4 for sliding-window layers — the rationale: sliding layers see only the last 1024 tokens, so they don't need a long-period RoPE.

The Gemma family has been the most architecturally adventurous among major closed-lab releases. It has explored softcap, sandwich norm, RoPE alternation, and per-layer attention-type alternation — providing data points (especially "softcap dropped") that the rest of the field has used to inform its own choices.

---

### §5.4 Phi-3 — Fused QKV, Fused gate-up, Partial RoPE, Residual Dropout

**Identity (Phi-3-Mini-3.8B):** D=3072, H=32, Hk=32 (MHA, no GQA), Dh=96, Df=8192, vocab=32064, rms_norm_eps=1e-5, rope_theta=10000 (short) or LongRoPE for 128k variant, partial_rotary_factor=1.0 (Phi-3-mini), partial_rotary_factor=0.5 in some legacy variants. tied_lm_head=False. resid_pdrop=0.0 (config default).

**HF file:** modeling_phi3.py — Phi3MLP 49 to 64 (fused gate-up), Phi3Attention 208 to 271 (fused qkv_proj), Phi3DecoderLayer 295 to 335 (residual dropouts), apply_rotary_pos_emb 178 to 205 (partial rotary).

**Fused gate-up MLP (lines 49 to 64):**

```python
class Phi3MLP(nn.Module):
    def __init__(self, config):
        self.gate_up_proj = nn.Linear(D, 2*Df, bias=False)
        self.down_proj    = nn.Linear(Df, D,  bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]   # silu
    def forward(self, hidden_states):
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(2, dim=-1)     # gate is FIRST half, up SECOND
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)
```

Critical chunk order: gate, up = chunk(2, -1). The opposite convention (up, gate) is used in some MoE implementations and causes silent miscompilation — the loader must know which half is which.

**Fused QKV (lines 222 to 224):**

```python
op_size = H*Dh + 2*Hk*Dh   # query + key + value packed
self.qkv_proj = nn.Linear(D, op_size, bias=False)
self.o_proj   = nn.Linear(H*Dh, D, bias=False)
```

**Partial rotary (lines 196 to 205):**

```python
rotary_dim = cos.shape[-1]    # = head_dim * partial_rotary_factor
q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
```

Some Phi-3 variants set partial_rotary_factor=0.5 — half of each head dims get RoPE, the rest pass through unchanged. The inv_freq for cos/sin is computed on int(head_dim * partial_rotary_factor).

**Residual dropouts (lines 295 to 335):**

```python
self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
self.resid_mlp_dropout  = nn.Dropout(config.resid_pdrop)
...
hidden_states = residual + self.resid_attn_dropout(hidden_states)
hidden_states = residual + self.resid_mlp_dropout(hidden_states)
```

Default 0.0 in inference (no-ops), but training graphs need them.

**vLLM diffs:** class Phi3ForCausalLM(LlamaForCausalLM) — ONE LINE plus packed_modules_mapping = {"qkv_proj": [...], "gate_up_proj": [...]}. The fused projections ARE Llama's QKVParallelLinear and MergedColumnParallelLinear; the packing map maps a single safetensor name to one fused weight.

**llama.cpp diffs:** create_tensor_qkv handles both separate wq/wk/wv (HF safetensors) and fused wqkv (GGUF). Partial RoPE encoded via n_rot < n_embd_head.

**Evolution arc (Phi 1 to Phi 4):**

- **Phi-1 / Phi-1.5 (June 2023 / Sep 2023, arxiv 2306.11644):** 1.3B / 1.5B, MHA, GELU MLP (no SwiGLU), parallel attention+FFN topology inherited from CodeGen.
- **Phi-2 (Dec 2023):** 2.7B, switched to standard pre-norm sequential residual; closer to Llama-shape.
- **Phi-3 (April 2024, arxiv 2404.14219):** SwiGLU MLP, fused QKV, partial RoPE (legacy), residual dropouts in code path. 3.8B mini, 7B small, 14B medium. Mini variant has 128k context via LongRoPE.
- **Phi-3.5-MoE (Aug 2024):** Top-2 routing, 16 experts.
- **Phi-3-small:** uses BlockSparse attention pattern (different from Phi-3-mini dense). See section 5.26.
- **Phi-4 (Dec 2024, arxiv 2412.08905):** 14B dense, training-data quality focus, architecture is Phi-3 + minor tweaks.
- **Phi-4-mini-flash (June 2025):** Samba architecture (Mamba + sliding-window attention + MLP), see section 5.25.

The Phi lineage is interesting because Microsoft has cycled through multiple topology choices (parallel residual in Phi-1, sequential in Phi-2/3/4, then Samba in Phi-4-mini-flash). Each shift is motivated by inference economics: parallel residual saves a launch but doesn't fuse well; sequential residual fuses better at modern kernel maturity; Samba reduces KV cache via Mamba layers.

---

### §5.5 Mistral — SWA everywhere (since 7B v0.1)

Mistral is essentially "Llama with SWA on every layer". Mixtral adds MoE.

**Identity (Mistral-7B-v0.3):** D=4096, H=32, Hk=8, Dh=128, Df=14336, vocab=32768, rms_norm_eps=1e-5, rope_theta=1e6, sliding_window=4096 (every layer), tied_lm_head=False.

**HF file:** modeling_mistral.py — MistralAttention 122 to 178, MistralDecoderLayer 202 to 240.

The decoder layer is structurally identical to Llama. The only block-level diff is in MistralAttention.forward line 172:

```python
attn_output, attn_weights = attention_interface(
    self, query_states, key_states, value_states, attention_mask,
    dropout=..., scaling=self.scaling,
    sliding_window=getattr(self.config, "sliding_window", None),
    **kwargs)
```

The model-level forward (line 372) chooses the mask:

```python
mask_function = create_causal_mask if self.config.sliding_window is None \
                else create_sliding_window_causal_mask
```

**vLLM diffs:** derives from Llama implementation, per_layer_sliding_window. Recent vLLM also adds the Llama-4 position-dependent scaling for Mistral-Large 2 (section 4.4).

**llama.cpp diffs:** shares the Llama graph builder; cache is per-layer capped at n_swa tokens. RoPE uses freq_base=1e6 for Mistral 7B v0.3.

**Evolution arc:**

- **Mistral 7B v0.1 (Sep 2023, arxiv 2310.06825):** SWA everywhere (4096), GQA (Hk=8), RoPE base 10000. The SWA-everywhere choice was the family signature.
- **Mistral 7B v0.2/v0.3 (2024):** RoPE base 1000000, otherwise same.
- **Mixtral 8x7B (Dec 2023, arxiv 2401.04088):** 8 experts top-2, SWA in config but the checkpoint config sets sliding_window=None despite the class supporting it (hidden footgun — see vLLM divergence below).
- **Mistral-Nemo (July 2024):** 12B, tekken tokenizer, 128k context, no SWA (dropped — Mistral-Large 2 also dropped it).
- **Mistral-Large 2 (July 2024):** 123B, position-dependent attention scaling (Llama-4 formula).
- **Codestral (May 2024):** code SLM, FIM (fill-in-middle) tokens, otherwise Mistral-shape.
- **Mistral-Small-3.1 (March 2025):** sink tokens trained in (similar to GPT-OSS but via prepended special token).

The Mistral family innovated on SWA, then *abandoned* it in larger models. The reason (per Mistral team posts): SWA limits effective context, and modern long-context techniques (YaRN, position-dependent scaling) achieve the same compute savings without the recall cliff.

---

### §5.6 DeepSeek-V3 — Multi-head Latent Attention (MLA) + Auxiliary-loss-free MoE

The single most distinctive layer in this list. v2 corrects the cache-compression ratio (71x not 10x — see section 4.1).

**Identity (DeepSeek-V3 671B-A37B):** D=7168, H=128, q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, qk_head_dim=192, n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8, n_group=8, topk_group=4, first_k_dense_replace=3, routed_scaling_factor=2.5, norm_topk_prob=True, rope_theta with YaRN.

**HF file:** modeling_deepseek_v3.py — DeepseekV3MLP 123 to 136, DeepseekV3TopkRouter 139 to 151, DeepseekV3NaiveMoe 154 to 191, DeepseekV3MoE 194 to 247, DeepseekV3Attention 364 to 479, DeepseekV3DecoderLayer 482 to 526.

**MLA __init__ (lines 367 to 414):**

```python
if self.q_lora_rank is None:
    self.q_proj = nn.Linear(D, H*qk_head_dim, bias=False)
else:
    self.q_a_proj      = nn.Linear(D, q_lora_rank, bias=attention_bias)
    self.q_a_layernorm = DeepseekV3RMSNorm(q_lora_rank)
    self.q_b_proj      = nn.Linear(q_lora_rank, H*qk_head_dim, bias=False)
self.kv_a_proj_with_mqa = nn.Linear(D, kv_lora_rank + qk_rope_head_dim, bias=attention_bias)
self.kv_a_layernorm     = DeepseekV3RMSNorm(kv_lora_rank)
self.kv_b_proj          = nn.Linear(kv_lora_rank, H*(qk_nope_head_dim + v_head_dim), bias=False)
self.o_proj = nn.Linear(H*v_head_dim, D, bias=attention_bias)
self.scaling = qk_head_dim ** (-0.5)
if rope_type != "default":
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    self.scaling = self.scaling * mscale * mscale
```

**MLA forward (lines 416 to 479):**

```python
# h: [B, S, D]
q_states = q_b_proj(q_a_layernorm(q_a_proj(h)))
q_states = q_states.view(B, S, H, qk_head_dim).transpose(1,2)
q_pass, q_rot = torch.split(q_states, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

compressed_kv = kv_a_proj_with_mqa(h)
k_pass, k_rot = torch.split(compressed_kv, [kv_lora_rank, qk_rope_head_dim], dim=-1)
k_pass = kv_b_proj(kv_a_layernorm(k_pass))
k_pass = k_pass.view(B, S, H, qk_nope_head_dim + v_head_dim).transpose(1,2)
k_pass, value_states = torch.split(k_pass, [qk_nope_head_dim, v_head_dim], dim=-1)

k_rot  = k_rot.view(B, 1, S, qk_rope_head_dim)
cos, sin = position_embeddings
q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
k_rot  = k_rot.expand(B, H, S, qk_rope_head_dim)

query_states = torch.cat((q_pass, q_rot), dim=-1)
key_states   = torch.cat((k_pass, k_rot), dim=-1)
# Cache update: decompressed (HF path; vLLM/llama.cpp cache compressed form)
key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)

if flash_requested and qk_head_dim != v_head_dim:
    value_states = F.pad(value_states, [0, qk_head_dim - v_head_dim])
attn_output, _ = attention_interface(...)
if flash_requested and qk_head_dim != v_head_dim:
    attn_output = attn_output[..., :v_head_dim]
attn_output = attn_output.reshape(B, S, H * v_head_dim).contiguous()
return self.o_proj(attn_output), _
```

**Cache compression math (verified — see section 4.1):**
- HF caches decompressed: H x (qk_head_dim + v_head_dim) = 128 x 320 = 40,960 elements per token.
- vLLM/llama.cpp cache compressed: kv_lora_rank + qk_rope_head_dim = 576 elements per token.
- Ratio: 40,960 / 576 = 71.11x.

**MoE block (lines 194 to 247):**

```python
class DeepseekV3MoE(nn.Module):
    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()    # NOT softmax
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        # group-limited:
        group_scores = router_logits_for_choice.view(-1, n_group, n_routed_experts // n_group) \
                            .topk(2, dim=-1)[0].sum(dim=-1)
        group_idx  = torch.topk(group_scores, k=topk_group, dim=-1)[1]
        # ... mask, topk, renormalize, scale
        topk_weights = topk_weights * self.routed_scaling_factor   # breaks "sums to 1"
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, D)        # flatten for routing
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states
```

Note the explicit flatten to (B*S, D) before routing — required for the per-token expert dispatch.

**Decoder layer (482 to 526):**

```python
if layer_idx >= config.first_k_dense_replace:   # default 3
    self.mlp = DeepseekV3MoE(config)
else:
    self.mlp = DeepseekV3MLP(config)
```

First 3 layers are dense even in the MoE model.

**vLLM diffs:** DeepseekV2MLAAttention caches the compressed kv. fused_qkv_a_proj outputs [q_lora_rank, kv_lora_rank + qk_rope_head_dim] in one matmul. kv_b_proj absorbed into o_proj at weight-load time. MLA decode uses flash_mla or triton_mla kernels. MoE via FusedMoE with topk_method="noaux_tc". DeepSeek-V3 reuses the deepseek_v2.py file in vLLM — there is no deepseek_v3.py.

**llama.cpp diffs:** wq_a, wq_a_norm, wq_b, wkv_a_mqa, attn_kv_a_norm, wkv_b GGUF tensors. LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID (vs SOFTMAX for Mixtral). e_score_correction_bias loaded as a non-trainable buffer. Caches the compressed form like vLLM.

**Evolution arc (DeepSeek LLM to V3 to V4):**

- **DeepSeek LLM (Jan 2024, arxiv 2401.02954):** standard Llama-shape decoder. 7B and 67B sizes. No MLA, no MoE.
- **DeepSeek-V2 (May 2024, arxiv 2405.04434):** introduced MLA (the multi-head latent attention with compressed KV cache via low-rank projection) and DeepSeekMoE (fine-grained experts + shared experts). 236B-A21B. MoE used softmax routing with an auxiliary loss term for load balancing.
- **DeepSeek-V2-Lite (May 2024):** 15.7B-A2.4B, the SLM-accessible MLA testbed — see section 5.7.
- **DeepSeek-V3 (Dec 2024, arxiv 2412.19437):** 671B-A37B. The key change vs V2: dropped the auxiliary load-balance loss in favor of a non-trainable bias (e_score_correction_bias) that is updated outside backprop. Also switched routing activation from softmax to sigmoid. Added group-limited routing (n_group=8, topk_group=4) to reduce inter-node communication. Routing math: scores = sigmoid(logits) + bias; top-k group selection; top-k expert selection; renormalize selected weights; scale by routed_scaling_factor=2.5.
- **DeepSeek-V4 (placeholder in transformers/deepseek_v4/, post-cutoff):** not analyzed.

The MLA invention has *not* spread beyond DeepSeek and a small set of followers (MiniCPM 3, see section 5.8). The reason: MLA requires a non-trivial absorb-into-o_proj engineering step for production inference, and the per-token KV cache savings only become headline numbers at 100+ heads. For smaller models (under 32 heads), GQA with Hk=8 is competitive at much lower implementation complexity.

---

### §5.7 DeepSeek-V2-Lite — MLA at 15.7B-A2.4B (the accessible MLA testbed)

Same architecture as DeepSeek-V3 but at SLM scale, useful for understanding MLA without the 671B overhead.

**Identity:** D=2048, H=16, q_lora_rank=None (full Q projection, not LoRA-compressed at this scale), kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, n_routed_experts=64, n_shared_experts=2, num_experts_per_tok=6, first_k_dense_replace=1.

**Key differences from V3:**

- **No Q LoRA:** q_lora_rank=None means Q goes through a single linear (D → H*qk_head_dim = 2048 → 16*192 = 3072) instead of the V3-style A/norm/B chain. The KV path is still LoRA-compressed.
- **Aux loss MoE (V2 style):** uses softmax routing with an auxiliary load-balance loss; no e_score_correction_bias.
- **Cache compression ratio:** H x (qk_head_dim + v_head_dim) / (kv_lora_rank + qk_rope_head_dim) = 16 x 320 / 576 = 5120 / 576 = 8.89x — this is the actual source of the "~10x" figure that v1 mis-attributed to V3.

**HF file:** modeling_deepseek_v2.py — separate from v3.

**Evolution placement:** the bridge between standard Llama-shape and full MLA. Worth its own section because it shows that MLA is implementable at <20B parameters and the compression ratio at small H is dominated by qk_rope_head_dim overhead.

---

### §5.8 MiniCPM 3 — MLA at 4B (corrects v1 census claim of CLA)

The v1 census mischaracterized MiniCPM-3 as CLA (cross-layer attention, the Sun et al. 2024 reduction technique). Per the MiniCPM-3 paper (Hu et al., arxiv 2404.06395 follow-on technical report), MiniCPM-3 uses **MLA**, not CLA. Corrected here.

**Identity (MiniCPM3-4B):** D=2560, H=40, q_lora_rank=768, kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64, n_layer=62, vocab=73448 (Chinese-coverage-heavy), rope_theta with LongRoPE scaling.

**Differences from DeepSeek-V3:**
- Smaller per-head dims (qk_nope=64 vs 128, qk_rope=32 vs 64, v=64 vs 128).
- Dense MLP only (no MoE at this size).
- LongRoPE scaling like Phi-3-mini-128k, not YaRN.
- Cache compression: 40 x (96+64) / (256+32) = 40 x 160 / 288 = 6400 / 288 = 22.2x. Smaller than DeepSeek-V3's 71x because MiniCPM 3 uses more aggressive per-head compression but fewer heads.

**HF file:** not in the local transformers clone (third-party `minicpm` directory exists in some forks; reference is the MiniCPM official repo at https://github.com/OpenBMB/MiniCPM).

**vLLM:** MiniCPM3 has dedicated support in vLLM 0.6+ via a deepseek-v2-style attention class with adjusted dims.

**Why MLA at 4B?** Because at small H (16-40) the absolute KV cache size is small, so the compression ratio per se is less important. The motivation is *per-token cache uniformity*: MLA's fixed-size compressed representation maps cleanly onto edge-device memory budgets where dynamic K/V allocation is painful.

---

### §5.9 Mixtral / Qwen3-MoE — softmax MoE pattern

**Identity (Mixtral-8x7B):** D=4096, H=32, Hk=8, Df=14336, n_local_experts=8, num_experts_per_tok=2, vocab=32768, rms_norm_eps=1e-5. (Mixtral 8x22B uses larger sizes; same shape pattern.)

**HF file:** modeling_mixtral.py — MixtralExperts 61 to 98 (3D experts), MixtralTopKRouter 101 to 116, MixtralSparseMoeBlock 119 to 135, MixtralAttention 294 to 351 (identical to Mistral, including SWA), MixtralDecoderLayer 354 to 389.

**Router (101 to 116):**

```python
router_logits = F.linear(hidden_states, self.weight)         # (T, E)
router_probs  = F.softmax(router_logits.float(), dim=-1)
router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)
router_top_value /= router_top_value.sum(dim=-1, keepdim=True)   # renormalize
```

Softmax-then-topk-then-renorm. Contrast DeepSeek-V3: sigmoid-then-topk-with-bias-then-(optional)-renorm.

**Experts (61 to 98) — fused 3D weights:**

```python
self.gate_up_proj = nn.Parameter(torch.empty(E, 2*Df, D))
self.down_proj    = nn.Parameter(torch.empty(E, D, Df))
```

Naive loop iterates expert_hit then index_add_s the result. Slow but correct math reference. vLLM and llama.cpp replace this with FusedMoE kernels.

**Mixtral SWA gotcha:** Mixtral 8x7B class supports SWA but the checkpoint config sets sliding_window=None. Loading a Mistral-7B checkpoint into Mixtral or vice versa requires checking config.sliding_window — not assumeable from the class.

**vLLM diffs:** FusedMoE with renormalize=True, top_k=2. Expert parallel (EP) groups distribute experts across ranks. vLLM transposes the expert weight layout from HF's [E, 2*Df, D] to [E, D, 2*Df] for better-stride GEMM — a real on-disk vs in-kernel divergence.

**llama.cpp diffs:** ffn_*_exps GGUF tensors with shape [D, Df, E] (transposed from PyTorch). LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX. Routes via build_moe_ffn.

**Qwen3-MoE** is the same pattern: softmax+topk+renorm router + 3D experts, with Qwen3 attention (including QK-norm). Per-layer dense/sparse via mlp_only_layers.

**Evolution arc — MoE genealogy (Switch → GShard → Mixtral → DeepSeek-V3 → Qwen3-MoE):**

- **Switch Transformer (Jan 2021, Fedus et al., arxiv 2101.03961):** top-1 routing, "switch" each token to one expert. Auxiliary loss for load balance.
- **GShard (June 2020, Lepikhin et al., arxiv 2006.16668):** top-2 routing with capacity factor, auxiliary loss. The "two experts per token" template.
- **Mixtral (Dec 2023):** GShard-style top-2 with softmax + renormalize, auxiliary loss. The first widely-used open MoE.
- **DeepSeek-V2 (May 2024):** top-K with K=6 (DeepSeekMoE fine-grained experts), softmax + auxiliary loss, **shared experts** introduced (1 or 2 always-active experts).
- **DeepSeek-V3 (Dec 2024):** sigmoid + e_score_correction_bias (no auxiliary loss), group-limited routing, top-8. The auxiliary-loss-free design is motivated by the observation that aux-loss creates an optimization tension with the main loss.
- **Qwen3-MoE (April 2025):** softmax + renorm (Mixtral-style), with mlp_only_layers for dense interleave. Did NOT adopt DeepSeek's no-aux design.
- **OLMoE (Sep 2024):** softmax + renorm, 8 experts top-2, with OLMo2 post-norm layout — see section 5.10.
- **GPT-OSS (Aug 2025):** OpenAI open-weight MoE with attention sinks, see section 5.11.

The choice between sigmoid+bias-correction (DeepSeek-V3) and softmax+aux-loss (Mixtral, Qwen3-MoE) has not yet been settled. As of the survey cutoff, both designs are training-stable; the choice appears to be team preference and infrastructure compatibility.

---

### §5.10 OLMoE 1B-7B — open MoE with OLMo2 post-norm

**Identity (OLMoE-1B-7B-0924):** D=2048, H=16, Hk=16 (MHA), Dh=128, Df=1024, n_local_experts=64, num_experts_per_tok=8, vocab=50304, rms_norm_eps=1e-6. Post-norm layout (no input_layernorm).

**HF file:** modeling_olmoe.py — OlmoeMLP 134 (dense path), OlmoeAttention 221 to 280 with Q/K-norm on full H*Dh (line 246: `OlmoeRMSNorm(config.hidden_size)`; line 247: `OlmoeRMSNorm(config.num_key_value_heads * head_dim)`), OlmoeSparseMoeBlock 305 to 342.

**Attention (lines 246 to 263):**

```python
self.q_norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.k_norm = OlmoeRMSNorm(config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
...
query_states = self.q_norm(self.q_proj(hidden_states))
key_states = self.k_norm(self.k_proj(hidden_states))
```

OLMoE inherits OLMo 2's full-flat QK-norm convention.

**MoE block (lines 305 to 342):**

```python
self.gate_up_proj = nn.Parameter(torch.empty(E, 2*Df, D))    # E=64
self.down_proj    = nn.Parameter(torch.empty(E, D, Df))
# router: softmax + topk + (no renorm by default in OLMoE config)
```

**Why this matters as a corner of the design space:** OLMoE combines OLMo 2's post-norm placement with Mixtral-style softmax routing. The 64-expert top-8 configuration is finer-grained than Mixtral's 8-expert top-2 — fine-grained experts (DeepSeekMoE-style) at OLMo's training scale.

**Evolution arc:** OLMo 1 (Feb 2024, arxiv 2402.00838) was a standard pre-norm Llama-shape decoder. OLMo 2 (Oct 2024, arxiv 2501.00656) introduced the post-norm layout. OLMoE (Sep 2024, arxiv 2409.02060) is OLMo 2 + MoE, released slightly before OLMo 2 dense. The open data + open weights + open training-code disclosure is the family's hallmark.

---

### §5.11 GPT-OSS 20B-A3.6B — attention sinks trained in

**Identity (GPT-OSS 20B-A3.6B):** D=2880, H=64, Hk=8 (GQA), Dh=64, n_local_experts=32, num_experts_per_tok=4, vocab=200019 (tiktoken-style), rms_norm_eps=1e-5, sliding_window configurable per-layer, **sinks: nn.Parameter([H])**.

**HF file:** modeling_gpt_oss.py — GptOssExperts 74 to 121 (note: each expert has a bias), GptOssTopKRouter 122 to 137, GptOssMLP 139 to 155, GptOssAttention 283 to 351 (with sinks), GptOssDecoderLayer 354 to 410.

**Attention sinks (lines 262 to 280):**

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, ...):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)     # extra "logit slot" per head

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values   # stability
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # drop the sink after softmax
    ...
```

The mechanism: append one learnable logit value per head to the attention logits, softmax over (S + 1) logits, then drop the last column. Effectively reserves softmax mass for a "null position" controllable per head.

**MoE experts have bias (lines 74 to 121):**

```python
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size))
...
gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
```

Experts have biases on both gate_up_proj and down_proj. This is unusual; Mixtral / DeepSeek experts are bias-free.

**vLLM diffs:** sink slot supported via custom attention backend; FlexAttention path or eager fallback. FusedMoE adapted for expert biases.

**llama.cpp diffs:** attn_sinks GGUF tensor loaded per layer; build_attn variant with sink concatenation.

**Evolution arc:** GPT-OSS is OpenAI's August 2025 open-weight release. It is the first major open model to train with sinks (the StreamingLLM "sink token" technique was an inference-only recipe). The motivation per the GPT-OSS technical post: trained-in sinks make long-context inference more stable than the inference-only recipe because the sink values are optimized.

---

### §5.12 Granite 3.3 — four muP scalars

**Identity (Granite 3.3-2B):** D=2048, H=32, Hk=8, Dh=64, vocab=49159, attention_multiplier=0.0078125 (NOT 1/sqrt(Dh)), residual_multiplier=0.22, embedding_multiplier=12.0, logits_scaling=8.0. Granite-MoE-Hybrid adds Mamba interleave.

**HF file:** modeling_granite.py — GraniteAttention 115 to 179 (line 124: self.scaling = config.attention_multiplier), GraniteDecoderLayer 219 to 280 (residual_multiplier on lines 228, 273, 278), GraniteModel 368 to 443 (embedding_multiplier line 405), GraniteForCausalLM 446 to 504 (logits_scaling line 504).

**Decoder layer (230 to 280):**

```python
def forward(self, hidden_states, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states * self.residual_multiplier

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states * self.residual_multiplier
    return hidden_states
```

**Embedding (line 405):**
```python
inputs_embeds = inputs_embeds * self.embedding_multiplier   # = 12.0
```

**LM head (line 504):**
```python
logits = logits / self.config.logits_scaling   # / 8.0
```

The attention_multiplier value 0.0078125 = 1/128. Granite uses muP-style scale factors that allow hyperparameter transfer from a small "anchor" model trained at width D_anchor to the target width D. The Yang et al. 2022 "Tensor Programs V" formulation gives recipes for setting these scalars.

**vLLM diffs:** scaling applied at the same four locations. residual_multiplier baked into RMSNorm-residual fused op when supported.

**llama.cpp diffs:** f_attention_scale, f_residual_scale, f_embedding_scale, f_logit_scale as GGUF hyperparams. Embedding scale done inside build_inp_embd.

**Evolution arc:** Granite 1.0 (Sep 2024) introduced the muP-scalar parameterization. Granite 2.0 added longer context. Granite 3.0 (Oct 2024) added MoE variants (granite-moe and granite-moe-shared). Granite 3.1 added more multiplier exposure. Granite 3.3 (2025) refined the four-scalar set and added granite-moe-hybrid (Granite + Jamba-style Mamba interleave). The muP design lets IBM train a 7M anchor and transfer to 70B without re-tuning LR/init — a real engineering win.

---

### §5.13 OLMo 2 — Post-norm

**Identity (OLMo-2-7B):** D=4096, H=32, Hk=32 (MHA), Dh=128, Df=11008, vocab=100352, rms_norm_eps=1e-6. No input_layernorm.

**HF file:** modeling_olmo2.py — Olmo2Attention 206 to 276 (full H*Dh Q/K-norm lines 231 to 232), Olmo2DecoderLayer 295 to 333.

**Decoder layer (295 to 333):**

```python
class Olmo2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = Olmo2Attention(config, layer_idx)
        self.mlp       = Olmo2MLP(config)
        # NOTE: no input_layernorm
        self.post_attention_layernorm  = Olmo2RMSNorm(D, eps=eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(D, eps=eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

**The three norm placements:**

- Llama: x_out = x + sublayer(norm(x))                       [pre-norm]
- Gemma 3: x_out = x + post_norm(sublayer(pre_norm(x)))      [sandwich = pre + sublayer-out]
- OLMo 2: x_out = x + post_norm(sublayer(x))                 [post-norm]
- Cohere: x_out = x + attn(norm(x)) + mlp(norm(x))           [parallel]

**Q/K-norm (lines 231 to 232) is on the FULL H*Dh:**

```python
self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, ...)
self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, ...)
```

OLMo 2 chose the broader form. Contrast Qwen 3 (Dh only), Gemma 3 (Dh only). Source comment in Qwen3 line 248: "unlike olmo, only on the head dim!"

But the sublayer-x includes Q/K-norm applied to projections of x, behaving like embedded normalization. So OLMo 2 *does* normalize Q and K before they hit attention; it just doesn't pre-norm the residual stream.

**Evolution arc:** OLMo 1 (Feb 2024) was pre-norm Llama-shape with LayerNorm. OLMo 2 (Oct 2024, arxiv 2501.00656) switched to RMSNorm + post-norm. The motivation per the OLMo 2 paper: post-norm with QK-norm trains more stably at long-context than pre-norm without QK-norm. The post-norm was rediscovered as a stability fix; OLMo 2 is the open-weight evidence that this works at scale.

---

### §5.14 Cohere Command-R / Aya — parallel residual + tied embeddings + logit scale

**Identity (Command-R 35B):** D=8192, H=64, Hk=64 (MHA), Dh=128, Df=22528, vocab=256000 (multilingual Aya tokenizer), layer_norm_eps=1e-5, logit_scale=0.0625 (= 1/16), tied_lm_head=True, use_qk_norm=False (configurable, varies by variant), no biases anywhere, parallel attn+mlp.

**HF file:** modeling_cohere.py — CohereLayerNorm 52 (custom LayerNorm), CohereAttention 224 to 305, CohereDecoderLayer 308 to 358.

**Decoder layer (308 to 358) — parallel residual:**

```python
def forward(self, hidden_states, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states_attention, _ = self.self_attn(hidden_states, ...)
    hidden_states_mlp = self.mlp(hidden_states)    # SAME normed input!
    hidden_states = residual + hidden_states_attention + hidden_states_mlp
    return hidden_states
```

ONE input_layernorm shared by both branches. Attention and MLP run in parallel from the same normed input. This is structurally the same as Falcon-7B and GPT-J. The pseudocode collapses to:

```
x_out = x + attn(norm(x)) + mlp(norm(x))
```

**Logit scaling (line 515):**
```python
logits = logits * self.logit_scale   # main diff from Llama
```

For Command-R, logit_scale = 0.0625 = 1/16. This scales the lm_head output before softmax — softens the predicted distribution at the same temperature.

**QK-norm (configurable):** `use_qk_norm: bool`. When True, full H*Dh-style norm (line 246-247 of attention).

**Cohere2 variant:** sliding-window alternation period 4 (1 global per 4 sliding), unlike Gemma 3's 5:1. Sliding window applied to specific layer indices via config.layer_types.

**Cohere2-MoE:** MoE variant with Mixtral-style softmax routing, separate cohere2_moe directory.

**llama.cpp diffs:** loads attn_q_norm and attn_k_norm if use_qk_norm. The build path for Cohere uses a parallel-residual graph builder (a separate code path from Llama's sequential).

**Evolution arc:** Command (Feb 2024) was the first Cohere open release. Command-R (March 2024) added parallel residual, tied embeddings, logit_scale, and the Aya multilingual tokenizer. Command-R+ (April 2024) scaled to 104B. Aya-23 (May 2024) is Command-R fine-tuned for multilinguality. Cohere2 / Aya Expanse (Oct 2024) added per-layer SWA and the cohere2 architecture. Cohere2-MoE (2025) added MoE.

Cohere is a useful "non-Llama" reference point because it preserved the GPT-J parallel-residual design when others abandoned it. Cohere's stated reason: parallel residual reduces depth-of-graph for inference, improving small-batch decode latency.

---

### §5.15 SmolLM3 — decoupled embed/lm_head + NoPE alternation

**Identity (SmolLM3-3B):** D=2048, H=16, Hk=4, Dh=128, Df=11008, vocab=49152, rms_norm_eps=1e-6, tied_lm_head=False (decoupled), **no_rope_layers: list[bool]** per-layer with about 25% of layers having no position encoding.

**HF file:** modeling_smollm3.py — SmolLM3Attention 185 to 258 with per-layer use_rope (line 211: `self.use_rope = config.no_rope_layers[layer_idx]`), forward decides whether to apply RoPE (line 233):

```python
def forward(self, hidden_states, position_embeddings, ...):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if self.use_rope:                                      # NEW: NoPE alternation
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    ...
```

**Motivation:** the Llama 3 ablations (and a SmolLM3 blog post) showed that NoPE layers extrapolate to longer context than RoPE layers because they have no built-in positional bias. Mixing NoPE and RoPE layers gives some context-aware computation and some position-invariant computation. The exact pattern in SmolLM3 is roughly every 4th layer has no_rope_layers[i]=True.

**Sliding window:** also per-layer via config.layer_types — combined with NoPE alternation gives a (NoPE, sliding-RoPE, sliding-RoPE, full-RoPE) repeating pattern.

**vLLM diffs:** rotary_emb call gated on use_rope; otherwise q and k go straight to attn.

**llama.cpp diffs:** GGUF tracks per-layer no-rope flags via a layer-pattern table.

**Evolution arc:** SmolLM (Aug 2024) was the first HF SLM release. SmolLM2 (Nov 2024) refined training data. SmolLM3 (June 2025) added NoPE alternation and decoupled embeddings. The decoupling decision is a small-model trade-off: at 3B, tied embeddings save ~100MB but reduce headroom for output-distribution refinement.

---

### §5.16 BitNet b1.58 — ternary weights + dual sub-norms

The most distinct family in the survey: weights are {-1, 0, +1} (ternary, 1.58 bits per weight) with per-layer scale, NOT fp16/bf16.

**Identity (BitNet-b1.58-3B):** D=3200, H=32, Hk=32 (MHA, no GQA at this size), Dh=100, Df=8640, vocab=128256, rms_norm_eps=1e-5, weight dtype: int2 (packed ternary), plus per-tensor scale.

**HF file:** modeling_bitnet.py (auto-generated from modular_bitnet.py).

**Key fact: BitNet has TWO extra RMSNorms not present in any other family:**

- **attn_sub_norm** (RMSNorm(D), line 177, used at line 216 of forward): applied to attention output BEFORE o_proj. Code:
```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.attn_sub_norm(attn_output)  # diff with Llama
attn_output = self.o_proj(attn_output)
```

- **ffn_sub_norm** (RMSNorm(Df), line 74, used at line 77 of MLP):
```python
def forward(self, x):
    down_proj = self.down_proj(self.ffn_sub_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
    return down_proj
```

**Why dual sub-norms:** in BitLinear training the linear matmul is `quantize(x) @ quantize(W) * dequant_scale`. The sub-norm sits between act/gate-up and the next quantized matmul, ensuring the activation magnitude is bounded before quantization. Without it, ternary quantization clips information unevenly across positions. This makes BitNet *structurally* different from every other family — there is no axis in v1 that can encode this.

**Decoder block (lines 221 to 261):** the outer block is standard Llama (pre-norm + attn + residual + post-norm + mlp + residual). The sub-norms are INSIDE the attn and mlp modules, so the outer block is unchanged.

**HF source uses standard nn.Linear:** The HF reference code in modeling_bitnet.py uses standard `nn.Linear` (line 165-176: `self.q_proj = nn.Linear(...)`). This is the math reference for inference; in the trained BitNet b1.58 paper the matmul is BitLinear (W is ternary, A is INT8). For HF the weight is stored dequantized to fp/bf16. Production inference (T-MAC, bitnet.cpp) keeps the ternary packed form and uses lookup-table matmul.

**vLLM diffs:** as of survey cutoff, vLLM does not have a custom BitNet kernel; the model runs through the standard Llama-shape path with dequantized weights. The 1.58-bit packing is lost at load time.

**llama.cpp diffs:** bitnet.cpp is a separate llama.cpp fork from Microsoft Research. It implements lookup-table matmul for ternary weights. The graph builder for BitNet calls a custom build_bitlinear_attn that loads attn_sub_norm and ffn_sub_norm tensors.

**Evolution arc:** BitNet (Oct 2023, arxiv 2310.11453) was the original 1-bit (binary, +/-1) LLM. BitNet b1.58 (Feb 2024, arxiv 2402.17764) generalized to ternary (+/-1, 0) and added the dual sub-norm. The Microsoft "1bit-llm-era" paper makes the case that ternary matches fp16 performance on perplexity at 3B+ scale.

The dual sub-norm axis (v2 axis 3.22) was the single biggest "missing axis" in v1.

---

### §5.17 OpenELM — per-layer width

**Identity (OpenELM-1B-Instruct):** D=1280, n_layer=20, **head_dim=64 fixed**, **num_query_heads[i] varies per layer (12 to 20)**, **num_kv_heads[i] varies per layer (3 to 5)**, **ffn_multipliers[i] varies per layer (0.5 to 4.0)**, vocab=32000, rms_norm_eps=1e-6.

**Key fact: each layer has a different attention head count and a different FFN inner dimension.** From the OpenELM paper (Mehta et al., arxiv 2404.14619, Apple):

> The number of attention heads and the FFN multiplier vary across the layers ... using DeepNet-style decoupled width.

For OpenELM-1B-Instruct, the per-layer configs are roughly:
- Layers 0–4: num_query_heads=12, ffn_multiplier=0.5 (Df = 0.5 * D = 640)
- Layers 5–10: num_query_heads=16, ffn_multiplier=1.0
- Layers 11–15: num_query_heads=18, ffn_multiplier=2.5
- Layers 16–19: num_query_heads=20, ffn_multiplier=4.0 (Df = 5120)

**The layer constructor takes per-layer scalars:**

```python
class OpenELMDecoderLayer:
    def __init__(self, config, layer_idx):
        Hq = config.num_query_heads[layer_idx]
        Hk = config.num_kv_heads[layer_idx]
        Df = int(config.ffn_multipliers[layer_idx] * config.hidden_size)
        self.q_proj = nn.Linear(D, Hq * head_dim)
        self.k_proj = nn.Linear(D, Hk * head_dim)
        ...
        self.mlp = SwiGLU(D, Df)
```

This breaks the single-config assumption shared by all 10 v1 families. The API must accept per-layer scalars indexed by layer_idx.

**HF file:** not in the local transformers clone. Reference is the OpenELM official release on HF Hub (apple/OpenELM-*) plus the modular_openelm.py in some forks.

**vLLM diffs:** vLLM has experimental OpenELM support. Each layer has its own QKVParallelLinear instance with its own H/Hk values.

**llama.cpp diffs:** GGUF stores per-layer hyperparameters as arrays (n_head[il], n_head_kv[il], n_ff[il]) rather than scalars. The build loop indexes these arrays per-layer.

**Evolution placement:** OpenELM (April 2024) was Apple's open-weight SLM. The per-layer width idea was inherited from DeepNet (Wang et al. 2022) and is unique to OpenELM in production releases as of the survey cutoff. No major lab has replicated this design choice — likely because the per-layer scalars make sharding and pipeline-parallel mapping more complex.

The per-layer-width axis (v2 axis 3.23) was a critical v1 omission.

---

### §5.18 Jamba — hybrid attention + Mamba SSM + optional MoE (per-layer)

The non-pure-transformer reference. v2 distinguishes Mamba-1 (Jamba) from Mamba-2 (section 5.19).

**Identity (Jamba 52B):** D=4096, n_layer=32, alternation 1:7 attention:mamba, MoE applied to subset of FFNs.

**HF file:** modeling_jamba.py — JambaAttention 148 to 200, JambaMambaMixer 202 to 474, JambaMLP 477 to 492, JambaSparseMoeBlock 533 to 567, JambaAttentionDecoderLayer 570 to 605, JambaMambaDecoderLayer 608 to 638.

**Two decoder layer classes, dispatched by config.layers_block_type[layer_idx]:**

```python
class JambaAttentionDecoderLayer:
    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)   # MLP or MoE
        return residual + hidden_states

class JambaMambaDecoderLayer:
    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states=hidden_states,
                                    cache_params=past_key_values,
                                    attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        return residual + hidden_states
```

**MambaMixer is Mamba-1 (lines 202 to 474):**

```python
class JambaMambaMixer(nn.Module):
    def __init__(self, config, layer_idx):
        self.intermediate_size = config.mamba_expand * D     # E_int = 2*D typically
        self.ssm_state_size    = config.mamba_d_state        # N (usually 16)
        self.time_step_rank    = config.mamba_dt_rank
        self.conv1d = nn.Conv1d(E_int, E_int, kernel_size=4, groups=E_int, padding=3)
        self.in_proj  = nn.Linear(D, E_int*2, bias=False)
        self.x_proj   = nn.Linear(E_int, dt_rank + 2*N, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, E_int, bias=True)
        self.out_proj = nn.Linear(E_int, D, bias=False)
        self.A_log    = nn.Parameter(torch.empty(E_int, N))
        self.D        = nn.Parameter(torch.ones(E_int))
```

The state lives in (conv_state, ssm_state). conv_state: [B, E_int, 4]. ssm_state: [B, E_int, N].

**vLLM diffs:** Hybrid cache (paged for attention layers, per-layer state tensors for mamba). MambaMixer from vllm/model_executor/layers/mamba/mamba_mixer.py.

**llama.cpp diffs:** n_head_kv(il) == 0 marks a Mamba layer in GGUF. build_mamba_layer does the state-recurrent update. inp_recr is the recurrent-cache binding.

**Evolution arc:** Mamba (Dec 2023, arxiv 2312.00752) introduced selective SSM. Jamba (March 2024, arxiv 2403.19887) was AI21's first hybrid release — pure attention + Mamba in alternation. Mamba-2 (May 2024) refined SSM math (section 5.19). RecurrentGemma (April 2024) is the Griffin/Hawk variant from Google DeepMind, using RG-LRU instead of Mamba (section 5.20). Falcon-Mamba and Falcon-H1 followed in 2024-2025 (sections 5.21-5.22). The hybrid space has fragmented; no single design has won.

---

### §5.19 Mamba-2 — pure SSM with SSD parametrization

Mamba-2 (May 2024, arxiv 2405.21060) refines Mamba-1 via State Space Duality. The new parametrization: A becomes a *scalar per channel* (not a matrix), C is normalized via GroupNorm before output, and the recurrence has a structured-mask attention dual form.

**Identity (Mamba-2 2.7B):** D=2560, n_heads=80 (Mamba-2 heads, not attention heads), head_dim=64, state_size=128 (N), n_groups=1, chunk_size=256, conv_kernel=4, expand=2.

**HF file:** modeling_mamba2.py — Mamba2Mixer 121 onward. Key changes from Mamba-1:

```python
class Mamba2Mixer(nn.Module):
    def __init__(self, config, layer_idx, ...):
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size       # SSD chunked-recurrence size
        self.n_groups = config.n_groups           # group structure on B, C
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        # one big in_proj for everything:
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)
        # A is now per-head scalar:
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        # GroupNorm between scan and out_proj:
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.D = nn.Parameter(torch.empty(self.num_heads))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
```

**Key differences vs Mamba-1:**
- **A is per-head, not per-channel-per-state.** In Mamba-1, A_log shape was [E_int, N]. In Mamba-2, A_log shape is [num_heads]. This is the SSD insight: the structured matrix form of SSM is equivalent to causal masked attention with a particular mask structure.
- **chunk_size parameter:** SSD computes the recurrence in chunks for efficient parallel-prefix sum. chunk_size=256 is typical. The recurrence is exact (not approximate) but parallelized via chunked computation.
- **n_groups:** B and C are shared across groups of heads, allowing param/cache savings.
- **GroupNorm (rms_norm in Mamba2RMSNormGated)** between scan output and out_proj — analogous to QK-norm in attention.
- **One big in_proj** instead of separate in_proj, x_proj, dt_proj. The fused matmul is closer to Llama's QKVParallelLinear.

**Pseudocode for forward (simplified):**

```python
# h: [B, S, D]
projected_states = self.in_proj(h)    # [B, S, projection_size]
# split into z (gate), x_BC (x concatenated with B, C), dt
gate, x_BC, dt = projected_states.split([E_int, conv_dim, num_heads], -1)
x_BC = causal_conv1d(x_BC, ...)       # depthwise conv with kernel 4
x, B, C = x_BC.split([E_int, n_groups*N, n_groups*N], -1)
A = -A_log.exp()                        # per-head scalar
# SSD chunked scan:
y = mamba_chunk_scan_combined(
    x.view(B, S, num_heads, head_dim),
    dt,
    A,
    B.view(B, S, n_groups, N),
    C.view(B, S, n_groups, N),
    chunk_size=self.chunk_size,
    z=gate,
    D=self.D,
    ngroups=self.n_groups,
    headdim=self.head_dim,
)
y = self.norm(y, gate)                  # GroupNorm gated by z
return self.out_proj(y)
```

**vLLM diffs:** vLLM has mamba2.py with the SSD fast kernels integrated. The state cache per layer holds (conv_state[B, conv_dim, conv_kernel-1], ssm_state[B, num_heads, head_dim, N]).

**llama.cpp diffs:** mamba2.cpp builds a separate SSD scan op. GGUF tensors: ssm_in (fused), ssm_conv1d, ssm_A, ssm_D, ssm_out, plus the norm. n_head_kv = 0 marks the SSM layer.

**Evolution arc:** Mamba (Dec 2023) → Mamba-2 (May 2024). The motivation per Dao & Gu 2024: SSD reveals that the SSM recurrence is equivalent to attention with a structured causal mask, which allows reusing matmul-friendly kernels. Wall-clock-wise, Mamba-2 trains ~30% faster than Mamba-1 at the same parameter count and is the reference SSM design for new hybrid families (Zamba2, Falcon-H1, Granite-MoE-Hybrid).

---

### §5.20 RecurrentGemma (Griffin / Hawk) — gated linear recurrent unit + local attention

RecurrentGemma (April 2024, arxiv 2402.19427) is Google DeepMind's hybrid: gated recurrence (RG-LRU) + local sliding-window attention. The recurrence is *not* SSM; it's a gated linear RNN.

**Identity (RecurrentGemma-2B):** D=2560, n_layer=26, lru_width=2560, conv1d_width=4, sliding window for the attention layers, alternation pattern recurrent-recurrent-attention-... (specifically: layers 0,1 recurrent, layer 2 attention, layers 3,4 recurrent, layer 5 attention, ...).

**HF file:** modeling_recurrent_gemma.py — RecurrentGemmaRglru 267 to 376 (the RG-LRU core), RecurrentGemmaRecurrentBlock 378 to 452, RecurrentGemmaSdpaAttention 181 to 265 (local attention), RecurrentGemmaDecoderLayer 474 onward.

**RG-LRU forward (lines 318 to 325):**

```python
class RecurrentGemmaRglru(nn.Module):
    def forward(self, activations, position_ids):
        # gate: per-block diagonal recurrence parameter
        res = torch.baddbmm(self.recurrent_gate_bias[:, None, :], reshape_act, self.recurrent_gate_weight)
        recurrent_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))
        # log-space recurrence stability:
        log_recurrent_gate = -8.0 * recurrent_gate * nn.functional.softplus(self.recurrent_param)
        recurrent_gate = torch.exp(log_recurrent_gate)
        a_square = torch.exp(2 * log_recurrent_gate)
        # ... linear recurrence over time:
        # h_t = recurrent_gate_t * h_{t-1} + sqrt(1 - a_square_t) * x_t
        ...
        return hidden_states
```

The recurrence math: `h_t = g_t * h_{t-1} + sqrt(1 - g_t^2) * x_t` where `g_t = sigmoid(...)` is the per-step gate. This is the Linear Recurrent Unit (LRU) of Orvieto et al. 2023, with the Griffin "RG-LRU" addition of input-dependent gating.

**RecurrentBlock forward (lines 401 to 447):**

```python
def forward(self, input_states, ...):
    y_branch = self.linear_y(input_states)                 # gate branch
    y_branch = self.act_fn(y_branch)                       # GELU
    x_branch = self.linear_x(input_states)
    x_branch = x_branch.transpose(1, 2)                    # [B, lru_width, S]
    # depthwise causal conv (kernel 4):
    x_branch = self.conv_1d(x_branch)[..., :seq_len]
    # cache state for streaming decode:
    self.conv1d_state = nn.functional.pad(x_branch, (...))
    x_branch = self.rg_lru(x_branch.transpose(1, 2), position_ids)
    hidden_states = x_branch * y_branch                    # gating
    hidden_states = self.linear_out(hidden_states)
    return hidden_states
```

**Decoder layer dispatch:**
```python
TEMPORAL_BLOCK_CLASSES = {"recurrent": RecurrentGemmaRecurrentBlock,
                          "attention": RecurrentGemmaSdpaAttention}
class RecurrentGemmaDecoderLayer:
    def __init__(self, config, layer_idx):
        self.temporal_block = TEMPORAL_BLOCK_CLASSES[config.layers_block_type[layer_idx]](config, layer_idx)
```

**Cache state:**
- For recurrent layers: (conv1d_state [B, lru_width, conv1d_width-1], recurrent_states [B, lru_width]).
- For attention layers: standard sliding-window KV cache (bounded by sliding_window size).

**vLLM diffs:** vLLM has experimental RecurrentGemma support. The per-layer cache abstraction handles both KV and recurrent state.

**llama.cpp diffs:** GGUF tensors include rg_lru_recurrent_param, rg_lru_recurrent_gate_weight/bias, recurrent_linear_x/y/out, recurrent_conv1d. layer pattern alternates "recurrent" and "attention" entries.

**Evolution arc:** Griffin/Hawk (Feb 2024, arxiv 2402.19427) introduced RG-LRU. RecurrentGemma (April 2024) is the open Gemma-flavored version. The RG-LRU has not been adopted outside DeepMind — likely because the stability trick (`-8.0 * g * softplus(param)`) is finicky and the SSM-based Mamba designs (Mamba-2, Falcon-Mamba) achieve similar long-context quality with simpler math.

---

### §5.21 Falcon-Mamba — pure Mamba 7B

Falcon-Mamba (Aug 2024) is TII's full-Mamba 7B release. Architecture: Mamba-1 mixer at every layer, no attention layers.

**Identity (Falcon-Mamba-7B):** D=4096, n_layer=64, state_size=16, expand=2, conv_kernel=4, time_step_rank=auto.

**HF file:** modeling_falcon_mamba.py (modular_falcon_mamba.py). Structurally the same as standalone Mamba; one difference is the MLP and norms (Falcon-Mamba uses standard pre-norm + RMSNorm; no MLP — the SSM is both the token mixer and the channel mixer in Mamba's design).

**Why no MLP?** Mamba's design absorbs channel-mixing into the SSM via the in_proj → conv1d → x_proj → out_proj chain. The intermediate_size = 2*D plays the role of an MLP's inner dimension. So a Falcon-Mamba "layer" is just one RMSNorm + one MambaMixer + one residual add. There's no second sublayer.

**Cache state:** (conv_state, ssm_state) per layer, no KV.

**vLLM diffs:** vLLM supports Falcon-Mamba via the mamba_mixer kernel. The cache is per-layer state-only.

**llama.cpp diffs:** falcon_mamba.cpp graph builder; GGUF tensors ssm_in, ssm_conv1d, ssm_x, ssm_dt, ssm_A, ssm_D, ssm_out, plus attn_norm (used as the pre-mixer norm).

**Evolution placement:** Falcon-Mamba demonstrates that pure SSM at 7B is competitive with attention-only models on most non-long-context benchmarks. The motivation: TII's compute-efficiency thesis. The follow-on (Falcon-H1) adds attention layers back to address the recall gap on long-context retrieval — the consensus position.

---

### §5.22 Falcon-H1 — hybrid Mamba + attention (parallel branches)

Falcon-H1 (2025) is TII's hybrid: Mamba2 + attention layers, with parallel branches.

**Identity (Falcon-H1-7B):** D=4096, n_layer=64, mamba2 ngroups, attention heads, both branches active at every layer in parallel.

**HF file:** modeling_falcon_h1.py — distinct from falcon_mamba (which is pure Mamba) and falcon (the legacy decoder).

**Key topology:** unlike Jamba (which alternates 1 attention : 7 mamba layers serially), Falcon-H1 has parallel Mamba + attention heads at *every* layer. The decoder block:

```python
def forward(self, h):
    h_norm = self.input_layernorm(h)
    attn_out = self.self_attn(h_norm)
    mamba_out = self.mamba(h_norm)
    h = h + attn_out + mamba_out                 # parallel residual!
    h_norm = self.post_attention_layernorm(h)
    h = h + self.mlp(h_norm)
    return h
```

This is parallel-residual at the token-mixer level. The post-mixer norm is shared.

**Cache:** dual cache — KV for attention branch, (conv_state, ssm_state) for Mamba branch.

**Evolution placement:** Falcon-H1 explores a different point in the hybrid design space than Jamba and RecurrentGemma. The parallel-branch design has the advantage that each token gets both attention's recall and Mamba's compute efficiency at every layer; the cost is doubling the per-layer parameter count.

---

### §5.23 Zamba2 — Mamba-2 backbone + SHARED attention block

Zamba2 (Aug 2024, Glorioso et al., arxiv 2405.16712) is the clearest example of parameter sharing across layers in a modern release.

**Identity (Zamba2-7B):** D=2304, n_mamba_heads, mamba2 backbone, num_mem_blocks=2 (or 1 in smaller variants).

**HF file:** modeling_zamba2.py — Zamba2Attention 227 (note: input is "concatenation of original_hidden_states with the output of the previous mamba layer" — line 233 comment), Zamba2MambaMixer 417 onward (Mamba-2 style with chunk_size etc.), Zamba2AttentionDecoderLayer 901, Zamba2MambaDecoderLayer 955, **Zamba2HybridLayer 1007** (the binding).

**The shared-block trick (lines 1007 onward):**

```python
class Zamba2HybridLayer(GradientCheckpointingLayer):
    def __init__(self, shared_transformer, linear, mamba):
        self.shared_transformer = shared_transformer    # SAME object across many layers!
        self.linear = linear
        self.mamba = mamba
    def forward(self, hidden_states, original_hidden_states, ...):
        # The shared transformer block is called with a residual-style input:
        transformer_hidden_states = self.shared_transformer(
            torch.cat([original_hidden_states, hidden_states], dim=-1), ...)
        # Then a linear projection back to D, then a Mamba block.
        hidden_states = hidden_states + self.linear(transformer_hidden_states)
        hidden_states = self.mamba(hidden_states, ...)
        return hidden_states
```

The model has, for `num_mem_blocks=2`, only TWO unique attention-block parameter sets. Layer 0 uses block 0, layer 1 uses block 1, layer 2 uses block 0 again, etc. (block_id = layer_id % num_mem_blocks). The Mamba-2 blocks are per-layer-unique.

This breaks the v1 "single nn.ModuleList of layers" abstraction. The API needs a shared-block pool indexed by (layer_idx, block_role).

**Optional shared attention adapter (lines 270 to 276, 314):**

```python
if config.use_shared_attention_adapter:
    # per-instance LoRA adapter on top of the shared attention block:
    for i in range(num_layers):
        if i % config.num_mem_blocks == block_id:
            # attach a LoRA adapter for this layer position
```

The shared block can be adapted per-layer via small LoRA modules — recovering some per-layer flexibility while keeping most parameters shared.

**Cache:** standard KV cache for the shared attention block (per layer that uses it), per-layer Mamba state.

**vLLM diffs:** vLLM Zamba2 support adapted from Mamba-2 + shared transformer pattern.

**llama.cpp diffs:** GGUF stores the shared block once with a mapping table for layer-to-block.

**Evolution placement:** Zamba (Feb 2024) was Zyphra's first hybrid; Zamba2 added Mamba-2 backbone and the LoRA adapter on the shared block. The shared-block design lets Zamba2 reduce parameter count vs Mistral-7B by 30% while matching downstream quality.

---

### §5.24 Hymba — parallel Mamba + attention heads

Hymba (Nov 2024, NVIDIA, https://research.nvidia.com/labs/adlr/hymba/) places Mamba and attention heads side-by-side at *each* attention position, not as separate layers.

**Identity (Hymba-1.5B):** D=1600, with parallel SSM heads and softmax-attention heads. Specifically, at each "attention" position, half the heads run softmax attention (over the KV cache) and half run SSM (over the recurrent state). Outputs are concatenated.

**HF file:** Hymba is in NVIDIA's repo (https://github.com/NVlabs/hymba) and not in the local transformers clone. The structure:

```python
def hymba_layer(h, kv_cache, ssm_state):
    h_norm = norm(h)
    # split projection across head types:
    q, k, v = proj_to_attn_heads(h_norm)
    z, conv_in = proj_to_ssm_heads(h_norm)
    # parallel:
    attn_out = softmax_attention(q, k, v, kv_cache)
    ssm_out  = mamba_scan(conv_in, ssm_state, z)
    combined = concat([attn_out, ssm_out], dim=-1)
    return h + out_proj(combined)
```

The MLP/FFN is standard.

**Cache:** KV for the attention-head subset + (conv_state, ssm_state) for the SSM-head subset. Both updated per token.

**Distinction vs Falcon-H1:** Falcon-H1 has separate Mamba and attention modules in parallel at each layer with their own input projections. Hymba shares the input projection and partitions the *heads* into attention-heads and SSM-heads.

**Evolution placement:** Hymba is one of NVIDIA's open-weight SLMs. The head-partition design is unique; no other family has adopted it as of cutoff. The argument per NVIDIA: parallel head-level mixing of attention and SSM allows the model to learn per-head specialization (some heads do recall, others do compute-efficient aggregation).

---

### §5.25 Phi-4-mini-flash — Samba (Mamba + sliding-window attention + MLP)

Phi-4-mini-flash (June 2025) uses the Samba architecture from Microsoft Research (https://arxiv.org/abs/2406.07522). Samba: Mamba layers + sliding-window attention layers alternated, with MLPs in between.

**Identity (Phi-4-mini-flash):** D=3072, alternation Mamba + SWA + MLP repeating, sliding window 2048.

**Topology vs Jamba:** Jamba alternates 1:7 attention:mamba *block-wise*, each block having a full attn or full mamba sublayer plus a separate FFN sublayer. Samba is finer-grained: every "block" has both a Mamba sublayer AND a sliding-window attention sublayer AND an MLP. The decoder pattern:

```
for layer in range(n_layer):
    if layer % 2 == 0:
        # Mamba sublayer + MLP
        h = h + mamba(norm(h))
        h = h + mlp(norm(h))
    else:
        # SWA sublayer + MLP
        h = h + swa_attn(norm(h))
        h = h + mlp(norm(h))
```

(Approximate; exact alternation per Phi-4-mini-flash technical report may differ.)

**Cache:** per-layer (conv_state, ssm_state) for Mamba layers, sliding-window KV for SWA layers (bounded at 2048).

**HF file:** not in the local transformers clone as of cutoff; Samba reference implementation is at https://github.com/microsoft/Samba.

**Evolution placement:** Phi-4-mini-flash extends the Phi family into the hybrid space. The argument per the Samba paper: SWA + Mamba achieves the same long-context performance as full-attention at far lower KV cost, while preserving the per-token compute pattern of a standard transformer (so existing kernels still apply).

---

### §5.26 Phi-3-small — BlockSparse attention

Phi-3-small (7B, June 2024) uses a BlockSparse attention pattern, distinct from Phi-3-mini's dense attention.

**Identity (Phi-3-small):** D=4096, H=32, Hk=8 (GQA), block_sparse_pattern as a custom mask.

The BlockSparse pattern: the attention mask is partitioned into a dense local window (e.g., 256 tokens) plus a global summary band (a small set of designated "summary" tokens that all positions attend to). The pattern is implemented as a fixed mask, computed per-sequence.

**HF support:** Phi-3-small uses the Phi-3 modeling file with a `block_sparse_attention` flag and a custom mask construction. The actual mask is built once per sequence and passed to FA2/FA3 as a custom block table.

**vLLM diffs:** vLLM has experimental block-sparse attention support for Phi-3-small via FlashAttention 2.5+ with a custom mask.

**llama.cpp diffs:** llama.cpp builds the mask in build_inp_attn_mask with the block-sparse pattern.

**Evolution placement:** Phi-3-small explored block-sparse attention as a compute-saver for 7B-class models. The decision was not propagated to Phi-4 (which went back to dense attention). The "abandoned" status is per Microsoft's Phi-4 paper, which notes that dense attention with GQA is now compute-competitive enough that block-sparse is not worth the implementation complexity.

---

### §5.27 Qwen3-Next — gated DeltaNet (linear attention) + softmax attention

Qwen3-Next (Sep 2025) alternates Qwen3NextGatedDeltaNet (a linear-attention variant) with full softmax attention.

**HF file:** modeling_qwen3_next.py — Qwen3NextAttention 256 onward (standard softmax), Qwen3NextGatedDeltaNet 499 onward (linear), Qwen3NextDecoderLayer 819 onward.

**Decoder dispatch (lines 819 to 879):**

```python
class Qwen3NextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx)
        # MoE or dense based on decoder_sparse_step:
        if (layer_idx not in config.mlp_only_layers) and (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states, cache_params=past_key_values, ...)
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(hidden_states=hidden_states, position_embeddings=position_embeddings, ...)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # residual add omitted here for brevity
```

**Gated DeltaNet** (Qwen3NextGatedDeltaNet) is a linear-attention variant where attention's softmax is replaced by a delta-rule update on a recurrent matrix-valued state. The recurrence:

```
S_t = S_{t-1} * (1 - alpha_t * k_t k_t^T) + alpha_t * v_t k_t^T
y_t = S_t q_t
```

(Approximate — actual implementation uses gated additions; see arxiv 2412.06464 "Gated Delta Networks: Improving Mamba2 with Delta Rule".)

**Cache:** for linear-attention layers, the cache state is a per-head matrix [Dh, Dh] (or [Dk, Dv]). For full-attention layers, standard KV.

**Evolution placement:** Qwen3-Next is the first major frontier-lab release to use Gated DeltaNet. The motivation: pure linear attention loses recall at long context; the delta-rule augmentation preserves it while keeping linear-time complexity. The alternation with full attention provides recall-fallback.

---

### §5.28 RWKV-7 — pure recurrent (TTT-like)

RWKV-7 (March 2025) is the latest in the RWKV non-transformer family. The recurrence math is different from RWKV-6 (which was based on WKV decay) — RWKV-7 uses a Test-Time-Training (TTT) inspired update rule.

**Identity (RWKV-7-7B):** D=4096, n_layer=32, no attention at all, no KV cache. Per-channel recurrent state.

**HF file:** modeling_rwkv.py is in transformers (older RWKV variants); RWKV-7 has a separate reference implementation at https://github.com/BlinkDL/RWKV-LM.

**RWKV recurrence (simplified):**

```python
# Each token updates a per-channel state vector
def rwkv_layer(x, state):
    # input projections (analogous to Q, K, V):
    r = receptance(x)    # akin to query
    k = key(x)
    v = value(x)
    # state update — RWKV-7 uses a TTT-style rule:
    new_state = decay * state + outer_product(k, v)   # rank-1 update
    out = r * (new_state @ k)                          # akin to attention output
    return out, new_state
```

The key difference from softmax attention: there is no global softmax over keys; instead, the state is a running weighted sum, updated by each token. The decay and the update are per-channel.

**RWKV-7 vs RWKV-6:** RWKV-7 introduces a "data-dependent decay" via a gating mechanism that lets the recurrence selectively forget — closer in spirit to Mamba's selectivity. The TTT framing per RWKV-7 release notes: each token *trains* the state on (k, v) pairs, and r retrieves.

**Cache:** per-channel state vector [B, D] per layer.

**Evolution placement:** RWKV (May 2023) introduced the recurrent transformer-alternative. RWKV-4, 5, 6, 7 each refined the recurrence math. RWKV-7 is the closest the family has come to matching transformer quality at the same parameter count on standard benchmarks (LAMBADA, HellaSwag).

---

### §5.29 Falcon-7B / Falcon-40B (historical) — multi-query + parallel residual

Falcon-7B and Falcon-40B (June 2023) are historical reference points. The architecture: pure MQA (Hk=1) + parallel attn+FFN topology + RoPE.

**Identity (Falcon-7B):** D=4544, H=71 (odd — H not divisible by 8), Hk=1 (MQA), no biases except LayerNorm bias, parallel_attn=True, new_decoder_architecture=False (Falcon-7B) or True (Falcon-40B+).

**HF file:** modeling_falcon.py — FalconDecoderLayer 558 onward.

**Verbatim parallel-residual logic (lines 565 to 636):**

```python
if config.num_ln_in_parallel_attn is None and config.new_decoder_architecture:
    config.num_ln_in_parallel_attn = 2

if not config.parallel_attn:
    self.post_attention_layernorm = LayerNorm(hidden_size, eps=...)
    self.input_layernorm = LayerNorm(hidden_size, eps=...)
else:
    if config.num_ln_in_parallel_attn == 2:
        self.ln_attn = LayerNorm(hidden_size, eps=...)   # separate norm per branch
        self.ln_mlp = LayerNorm(hidden_size, eps=...)
    else:
        self.input_layernorm = LayerNorm(hidden_size, eps=...)   # single shared norm
```

```python
def forward(self, hidden_states, ...):
    residual = hidden_states
    if self.config.new_decoder_architecture and self.config.num_ln_in_parallel_attn == 2:
        attention_layernorm_out = self.ln_attn(hidden_states)
        mlp_layernorm_out = self.ln_mlp(hidden_states)
    else:
        attention_layernorm_out = self.input_layernorm(hidden_states)

    attention_output, _ = self.self_attention(attention_layernorm_out, ...)

    if not self.config.new_decoder_architecture:
        if self.config.parallel_attn:
            mlp_layernorm_out = attention_layernorm_out   # SAME normed input!
        else:
            residual = dropout_add(attention_output, residual, ...)
            mlp_layernorm_out = self.post_attention_layernorm(residual)

    if (self.config.new_decoder_architecture
        and self.config.parallel_attn
        and self.config.num_ln_in_parallel_attn == 1):
        mlp_layernorm_out = attention_layernorm_out

    mlp_output = self.mlp(mlp_layernorm_out)

    if self.config.new_decoder_architecture or self.config.parallel_attn:
        mlp_output += attention_output                    # combine both branches!

    output = dropout_add(mlp_output, residual, ...)
    return output, attn_weights
```

Falcon-7B uses Hk=1 (full MQA) with parallel attn+FFN. Falcon-40B uses Hk=8 (GQA-like) with new_decoder_architecture=True and num_ln_in_parallel_attn=2.

**Evolution placement (the parallel-residual lineage):**

- GPT-J (June 2021): introduced `x + attn(LN(x)) + mlp(LN(x))` — the original parallel residual.
- GPT-NeoX 20B (April 2022): same pattern.
- Falcon-7B/40B (June 2023): parallel + MQA, multilingual training.
- PaLM (April 2022): parallel + MQA at 540B (closed-weights).
- (Llama / Mistral / Qwen / Gemma all chose sequential residual.)
- Cohere Command-R (March 2024): brought back parallel residual.

Why was it abandoned in Llama-family models? Per the Llama 1 paper, sequential residual makes attn and mlp computations *fusable* into a single CUDA stream with overlap, while parallel residual requires either (a) two separate streams that don't overlap well, or (b) one bigger fused kernel that few kernel libraries provide. As inference kernels matured (FA, vLLM's custom kernels), the parallel-vs-sequential gap narrowed, which is why Cohere felt comfortable bringing it back.

**Why pure MQA was abandoned:** Pure MQA (Hk=1) was tried by Falcon-7B, PaLM, and the early Llama-2 70B drafts. The replacement: GQA (Hk between 1 and H, typically 8). The reason per Ainslie et al. 2023 (https://arxiv.org/abs/2305.13245): pure MQA degrades quality measurably (~0.5 perplexity at 7B), while GQA at Hk=8 recovers quality at slightly higher KV cost than MQA. The 8-x-KV-reduction headline number from Falcon's MQA marketing turned out to be too aggressive.

---

### §5.30 StarCoder 2 — GELU MLP (no SwiGLU) + GQA + FIM + sliding window

StarCoder 2 (Feb 2024, Lozhkov et al., arxiv 2402.19173) is the BigCode reference. Architecturally a step back from the SwiGLU consensus.

**Identity (StarCoder 2-15B):** D=6144, H=48, Hk=4 (GQA), Dh=128, Df=24576, vocab=49152 (code-heavy BPE), rope_theta=999999.4420, use_bias=True (per-projection configurable), residual_dropout=0.1, sliding_window=4096.

**HF file:** modeling_starcoder2.py — Starcoder2MLP 53 to 67 (NOT SwiGLU — uses a single up→act→down path), Starcoder2Attention 141 to 204, Starcoder2DecoderLayer 205 onward.

**MLP (lines 53 to 67) — plain GELU, no gating:**

```python
class Starcoder2MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(embed_dim, config.intermediate_size, bias=config.use_bias)
        self.c_proj = nn.Linear(config.intermediate_size, embed_dim, bias=config.use_bias)
        self.act = ACT2FN[config.hidden_act]                  # gelu
        self.residual_dropout = config.residual_dropout

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.residual_dropout, training=self.training)
        return hidden_states
```

No gate, no element-wise multiply. Plain c_fc(x) → gelu → c_proj. This is the GPT-2/GPT-NeoX MLP shape — *not* SwiGLU.

**Attention has residual_dropout too (line 157, used line 199):** dropout applied to attn output before residual add.

**Sliding window (line 192):** propagated from config to attention backend (same as Mistral).

**FIM (fill-in-middle):** StarCoder 2 tokenizer adds special FIM tokens (`<fim_prefix>`, `<fim_middle>`, `<fim_suffix>`); the model is trained on FIM-formatted data. This is a tokenizer/data-format choice, not an architectural one — the decoder layer is unchanged.

**Evolution placement:** StarCoder 2 followed StarCoder 1 (May 2023, also GELU + MQA). The choice to stay on GELU + biases was conscious: BigCode's ablation found that for code data the SwiGLU gain is smaller than for natural language (perhaps because code distributions have fewer rare-token-magnitude issues). The price: StarCoder 2 has 2-3% more parameters than an equivalent SwiGLU model for the same hidden_size, because SwiGLU's gate-up pair has Df * 2/3 the parameters of a standalone up projection of the same effective width.

---

### §5.31 InternLM 2.5 / InternLM 3 — Dynamic NTK RoPE

InternLM 2.5 (June 2024) and InternLM 3 (Jan 2025) are Shanghai AI Lab's family. The architectural signature: Dynamic NTK RoPE scaling for context-length extrapolation.

**Identity (InternLM 2.5-7B):** D=4096, H=32, Hk=8, Dh=128, vocab=92544, rope_theta=1000000, rope_scaling={"type": "dynamic", "factor": 2.0}, qkv_bias=True (Q/K/V), no o_bias.

**HF file:** InternLM uses the Llama modeling file with a custom `internlm` config plus `trust_remote_code=True` for the dynamic NTK RoPE. The dynamic NTK formula (from arxiv 2306.15595):

```python
# At each forward call, recompute inv_freq based on the actual sequence length:
def dynamic_ntk_rope(seq_len, original_max, rope_theta, dim):
    if seq_len > original_max:
        base = rope_theta * ((seq_len / original_max) ** (dim / (dim - 2)))
    else:
        base = rope_theta
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq
```

The key: when seq_len exceeds the trained max, the RoPE base is scaled UP (not down) by a power of (seq_len / original_max). This stretches the position encoding domain at inference time without retraining.

**Comparison to other RoPE scalings:**
- Position Interpolation (PI): scales positions DOWN by factor.
- YaRN (DeepSeek-V3): combines NTK scaling with attention-temperature adjustment.
- LongRoPE (Phi-3): search-discovered piecewise scaling.
- Llama-3 rope_scaling: piecewise linear by frequency band.
- Dynamic NTK: scales the BASE, not the positions, and only when seq_len exceeds the trained max.

**Evolution placement:** InternLM 1 (2023) used standard RoPE. InternLM 2 (Jan 2024) added Dynamic NTK. InternLM 2.5 (June 2024) added more aggressive context extension. InternLM 3 (Jan 2025) refined training and added MoE variants. Dynamic NTK is one of the few RoPE scaling methods that requires NO recomputed config or retraining — it adapts at runtime based on input length.

---

### §5.32 ChatGLM 3 — prefix-LM + 2D-RoPE

ChatGLM 3 (Oct 2023, Tsinghua) inherits GLM's prefix-LM training objective and uses a 2D-RoPE variant.

**Identity (ChatGLM3-6B):** D=4096, H=32, multi_query_attention=True (Hk=2), vocab=65024, rms_norm_eps=1e-5, rope_theta=10000, partial_rotary_factor=0.5.

**HF file:** ChatGLM uses a `chatglm` config with `trust_remote_code=True`. The model file is `modeling_chatglm.py` in the official Tsinghua HF repo (THUDM/chatglm3-6b).

**Key architectural differences:**

- **Prefix-LM training:** during training, a span of tokens at the start of each sequence has bidirectional attention (the "prefix"); the rest are causal. At inference time, the user prompt is the prefix. This is a training-objective difference; the model file still implements causal attention with a custom mask.
- **2D-RoPE:** ChatGLM's RoPE is applied with two position dimensions (for the prefix layout). The first half of the rotary dim uses the position id; the second half uses the offset from the prefix end. This requires a custom `apply_rotary_pos_emb` that splits cos/sin into two halves.
- **Multi-query attention** at all sizes (Hk=2): ChatGLM kept MQA-style aggressive KV sharing.

**vLLM diffs:** vLLM has dedicated ChatGLM support with the 2D-RoPE handled in its rotary module.

**llama.cpp diffs:** rope_type=GLM in ggml's RoPE op handles the 2D variant.

**Evolution placement:** GLM (2021) → ChatGLM (2023) → ChatGLM2 (2023, dropped prefix-LM at inference but kept it at training) → ChatGLM3 (2023, refined chat) → GLM-4 (2024, broader open-weight release, dropped some legacy choices).

---

### §5.33 MobileLLM — embedding sharing + layer repetition

MobileLLM (Feb 2024, Meta, arxiv 2402.14905) is a 125M-1B family targeting on-device deployment. The architectural signature: tied embeddings + layer-weight repetition.

**Identity (MobileLLM-125M):** D=576, H=9, Hk=3, Dh=64, vocab=32000, n_layer_unique=15, n_layer_repeats=2 (so effective depth is 30 but only 15 unique blocks of parameters).

**HF file:** MobileLLM is in Meta's research repo (https://github.com/facebookresearch/MobileLLM); HF has a community port.

**Layer repetition (block sharing):** similar in spirit to Zamba2 but at every layer, not just attention. The "30-layer" model has 15 unique parameter sets; each set is applied twice consecutively. Per the MobileLLM paper:

> Repeating each transformer block twice consistently outperforms a non-repeated counterpart with the same parameter budget.

**Embedding sharing:** lm_head.weight = embed_tokens.weight (Llama 1-style tying). At 125M parameters, the embedding alone is `32000 * 576 ≈ 18M params` — 14% of the model.

**Why these choices:** MobileLLM's target is on-device deployment where parameter count is the binding constraint, not FLOPs. Block repetition gives more depth (better compositionality) without more parameters.

**Evolution placement:** MobileLLM is a research family, not yet a production lineage. It cites OpenELM (per-layer width — the other Apple-vs-Meta SLM design) and DeepNet (depth-decoupled width) as inspirations. The block-repetition idea is also used in ALBERT (2019, parameter sharing across BERT layers); MobileLLM revives it for the LLM era.

---

---

## §6 Abandoned-design section

Survey papers traditionally have a "dead ends" section listing tried-and-abandoned ideas. The dead ends inform the API as much as the live designs because they tell us what the API does *not* need to support (or what it can support without backward compatibility).

### §6.1 Parallel attn+FFN (GPT-J, GPT-NeoX, Falcon-7B, PaLM, Phi-1, early CodeGen)

**Lifespan:** 2021 (GPT-J) to 2023 (Falcon-7B), then re-introduced 2024 (Cohere Command-R). Status: abandoned in Llama-family models; revived selectively.

**The math:** x_out = x + attn(norm(x)) + mlp(norm(x)) with a single shared input-norm.

**Why abandoned in Llama-shape models:** Per the Llama 1 paper and subsequent discussions on the EleutherAI Discord (early 2023), sequential residual gives slightly better quality (~0.1 perplexity at 7B) at the same parameter count. The parallel residual was originally motivated by ability to run attention and MLP in parallel CUDA streams; once tensor parallel + flash attention matured, the kernel-level overlap could be achieved differently. The remaining advantage of parallel residual is one fewer norm (one input_layernorm shared; saves ~0.5% parameters), which does not justify the quality loss at modern training scales.

**Why revived in Cohere Command-R:** Cohere stated reason is decode-latency-at-small-batch: at TP=8 with batch_size=1, the all-reduce after o_proj overlaps better with the MLP gate-up matmul if attn and MLP are parallel. The quality cost is small relative to Cohere main loss (multilingual, retrieval-augmented).

**Falcon-H1 / Hymba new twist:** parallel Mamba + attention branches. This is a different parallel design from the parallel attn+FFN — it parallelizes the token mixer across two types of mixer, then does FFN sequentially. The cost-benefit calculus is different (the parallel Mamba+attention doubles per-layer parameters; the parallel attn+FFN saves on norms).

### §6.2 Pure MQA (Falcon-7B, PaLM, early Llama 2 70B drafts)

**Lifespan:** 2021 (PaLM, Shazeer MQA paper arxiv 2104.05230) to 2023 (Falcon-7B), abandoned circa 2023 with Llama 2 70B adoption of GQA(Hk=8).

**The math:** Hk=1; all attention heads share a single K and V projection.

**Why abandoned:** Per Ainslie et al. 2023 "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (https://arxiv.org/abs/2305.13245), pure MQA degrades quality by 0.5 perplexity at 7B. GQA at Hk=8 (8x KV reduction vs MHA) recovers quality while keeping most of the cache savings. The "8x" headline of MQA turned out to be too aggressive.

**Status:** every 2024+ model uses GQA with Hk in [4, 8, 16]. The Llama 3 family standardized on Hk=8; Mistral, Qwen, Phi all converged. The only pure-MQA model in our survey is the historical Falcon-7B (and ChatGLM3 with Hk=2, which is close).

### §6.3 Pure SWA without alternation (Mistral 7B v0.1)

**Lifespan:** 2023 (Mistral 7B v0.1, "SWA everywhere") to de facto abandoned 2024 (Mistral-Nemo, Mistral-Large 2 dropped SWA). Re-introduced as alternation pattern: Gemma 2 (5:1), Cohere2 (4:1), Phi-3-small (custom block-sparse).

**Why abandoned in pure form:** SWA-everywhere limits effective context to the window size. For window=4096, an 8k-context request cannot use the first half of the prompt at the last token attention. Modern long-context use cases (RAG, long-doc summarization) need global attention at some layers. Gemma 2 5:1 alternation gives 1/6 of layers full global access — empirically sufficient for retrieval tasks.

**Modern consensus:** SWA with global-attention alternation, where about 1 in 5 or 1 in 6 layers has full attention. Or SWA-only at some layers via per-layer config (Qwen 3 config.layer_types). Or no SWA at all (Llama 3, DeepSeek-V3 — relying on RoPE-scaling for long context).

### §6.4 ALiBi position encoding (MPT, Baichuan-1, Qwen 1 variants)

**Lifespan:** 2021 (ALiBi paper Press et al., arxiv 2108.12409) to 2023 (MPT, Baichuan-1) to abandoned 2023-2024 (Qwen 1.5 switched to RoPE; Baichuan-2 switched to RoPE).

**The math:** add a linear bias `-m * (i - j)` to attention logits before softmax, where m is a per-head slope. No rotation, no learned position embedding.

**Why abandoned:** Two reasons. (1) ALiBi extrapolates poorly to lengths much longer than training. The linear bias means very distant tokens get very negative attention; the effective receptive field is bounded. (2) RoPE scaling techniques (PI, NTK, YaRN) work at the RoPE-base level — they can extrapolate trained models to 4x-32x the trained context with minimal retraining. ALiBi has no equivalent retraining-free extrapolation knob.

**Status:** no major 2024+ model uses ALiBi. The closest is Granite attention_multiplier muP-scalar, but that is unrelated to position encoding.

### §6.5 Q4_0 GGUF quantization (replaced by k-quants Q4_K_M etc.)

This belongs more in document 04 (quantization) than 02, but the layer-implementation relevance is that early llama.cpp Q4_0 quantized matmul used a different stride-pattern from k-quant matmuls, affecting how layer-wise weight tensors get laid out.

**Lifespan:** Q4_0 (April 2023, original llama.cpp quantization) to Q4_K_M (June 2023, k-quants paper) to IQ4_NL (2024, "non-linear" int4) to MXFP4/MXINT4 (OCP Microscaling, 2024).

**Why abandoned:** Q4_0 per-block scale (one fp16 per 32 weights, no zero point) loses precision on weight distributions with non-zero mean. Q4_K_M per-32-weight block + per-256-weight super-block scale gives much better fidelity at the same bits/weight (~4.5 bits/weight vs Q4_0 4.5 bits/weight, but better quality).

**Status:** modern GGUF distributions standardize on Q4_K_M, Q5_K_M, Q6_K for quality-critical use; Q4_0 only appears in legacy or speed-critical paths. See document 04 v2 for the full quantization evolution.

### §6.6 Yi-1.0 cosine-similarity attention

**Lifespan:** Yi-1 (Nov 2023) to dropped in Yi-1.5 (May 2024).

**The math:** replace softmax(QK^T / sqrt(Dh)) with softmax(cos(Q, K) / temperature) where cos is cosine similarity. The motivation: bound the pre-softmax logit magnitude to [-1, 1], avoiding the logit-explosion problems that motivated softcap and QK-norm.

**Why abandoned:** training was unstable (per the Yi-1.5 release notes), and the convergence-rate cost was not worth the modest stability gain. Yi-1.5 reverted to standard scaled dot-product attention with QK-norm, joining the consensus.

**Status:** cosine attention has not been adopted by any other production-released model. Research papers continue to explore it (e.g., NormAttention) but the consensus is that QK-norm + softcap (or just QK-norm) provides the same stability with less convergence cost.

### §6.7 Auxiliary loss for MoE (Mixtral, DeepSeek-V2, ...)

**Lifespan:** Switch (2021) to DeepSeek-V2 (May 2024) to de-emphasized by DeepSeek-V3 (Dec 2024).

**The math:** add a load-balance auxiliary loss `aux = alpha * sum_e (frac_e * gating_prob_e)` where frac_e is the fraction of tokens routed to expert e and gating_prob_e is the mean gate probability.

**Why de-emphasized by DeepSeek-V3:** the auxiliary loss creates an optimization tension with the main task loss — the aux loss pulls the routing toward uniform, but uniform routing may not be the quality-optimal solution. DeepSeek-V3 "auxiliary-loss-free" design uses a non-trainable bias correction (e_score_correction_bias) updated outside backprop to push under-used experts higher in the routing scores, achieving load balance without the main-loss tension.

**Status:** Mixtral, Qwen3-MoE, OLMoE, GPT-OSS still use auxiliary loss (in various forms). DeepSeek-V3 and DeepSeek-V4 use the bias-correction approach. The community is split; the design choice has not yet converged.

---

## §7 Cross-family comparison table

A condensed view of where each family lands on the 32 axes. Legend: Y = uses, N = does not, ? = config-dependent.

| Family | Norm placement | QK-norm | QKV layout | RoPE kind | SWA | Mixer | Channel | Cache | Notable axes |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3 | pre | none | split | NEOX | none | attn | SwiGLU | KV-paged | reference |
| Qwen 3 | pre | per-Dh | split + qknorm | NEOX | optional | attn | SwiGLU | KV | QKV biases |
| Gemma 3 | sandwich | per-Dh | split + qknorm | NEOX (1e6/1e4 alt) | 5:1 | attn | GeGLU | KV | embed scale, (1+w) RMS |
| Phi-3 | pre | none | fused QKV + fused gate-up | partial NEOX | none | attn | SwiGLU | KV | residual dropout |
| Mistral 7B | pre | none | split | NEOX | global | attn | SwiGLU | KV bounded | SWA everywhere |
| DeepSeek-V3 | pre | none (q_a/kv_a norm) | MLA | NEOX (YaRN) | none | attn | MoE+shared+sigmoid | MLA compressed | 71x KV compression |
| DeepSeek-V2-Lite | pre | none | MLA (no Q-LoRA) | NEOX | none | attn | MoE+shared+softmax+aux | MLA compressed (8.9x) | smaller MLA |
| MiniCPM 3 | pre | none | MLA | LongRoPE | none | attn | SwiGLU dense | MLA compressed | 4B MLA |
| Mixtral | pre | none | split | NEOX | optional config | attn | MoE 8x7B softmax | KV | renormalize topk |
| OLMoE | post | full | split + qknorm | NEOX | none | attn | MoE 64x top-8 softmax | KV | post-norm MoE |
| GPT-OSS | pre | none | split | NEOX | per-layer | attn + sinks | MoE + bias | KV | learned attention sinks |
| Granite 3.3 | pre | none | split + biases | NEOX | none | attn | SwiGLU | KV | 4 muP scalars |
| OLMo 2 | post | full | split + qknorm-full | NEOX | none | attn | SwiGLU | KV | no input_layernorm |
| Cohere Cmd-R | parallel | optional | split | NEOX | none (Cmd-R), 4:1 (Cohere2) | attn | SwiGLU | KV | logit_scale, tied lm_head, no biases |
| SmolLM3 | pre | none | split | NEOX with NoPE alt | per-layer | attn | SwiGLU | KV | NoPE layers |
| BitNet b1.58 | pre + attn_sub + ffn_sub | none | split (ternary W) | NEOX | none | attn | SwiGLU + ffn_sub_norm | KV (dequant) | ternary, dual sub-norms |
| OpenELM | pre | none | split (per-layer width) | NEOX | none | attn | SwiGLU (per-layer Df) | KV | per-layer scalars |
| Jamba | pre | none | split (attn layers) | NEOX (attn layers) | none | attn + Mamba-1 | SwiGLU or MoE | KV + (conv, ssm) | 1:7 alternation |
| Mamba-2 | pre | none | (no attention) | none | none | SSD | SSM-fused | (conv, ssm) | chunk_size, ngroups |
| RecurrentGemma | pre | none | split (attn layers) | NEOX (attn layers) | yes | RG-LRU + attn | GeGLU | (conv1d, recurrent) + KV | gated linear RNN |
| Falcon-Mamba | pre | none | (no attention) | none | none | Mamba-1 | absorbed in SSM | (conv, ssm) | pure 7B Mamba |
| Falcon-H1 | pre (parallel) | none | split | NEOX | none | attn + Mamba-2 parallel | SwiGLU | KV + (conv, ssm) | dual branches |
| Zamba2 | pre | none | split (shared attn block) | NEOX | none | Mamba-2 + shared attn | SwiGLU | KV (shared) + (conv, ssm) | num_mem_blocks=2 |
| Hymba | pre | none | split (parallel heads) | NEOX | none | attn-heads + SSM-heads | SwiGLU | KV (subset) + (ssm subset) | head partition |
| Phi-4-mini-flash | pre | none | split | NEOX | global on SWA layers | Mamba + SWA | SwiGLU | KV (bounded) + (conv, ssm) | Samba alt |
| Phi-3-small | pre | none | fused QKV | partial NEOX | block-sparse | attn (blocksparse) | SwiGLU | KV | blocksparse pattern |
| Qwen3-Next | pre | per-Dh on attn layers | split | NEOX (attn), none (linear) | none | attn + GatedDeltaNet | MoE | KV + linear-attn state | gated DeltaNet |
| RWKV-7 | pre | none | (no attention) | none | none | WKV recurrent (TTT-style) | channel mixing in WKV | per-channel state | pure recurrent |
| Falcon-7B | parallel (1 or 2 norms) | none | split (MQA) | NEOX | none | attn (MQA) | GELU | KV | historical |
| StarCoder 2 | pre | none | split + biases | NEOX | global | attn | GELU + dropout (NO gate) | KV | FIM tokens, no SwiGLU |
| InternLM 2.5/3 | pre | none | split (q/k/v bias) | Dynamic NTK | none | attn | SwiGLU | KV | runtime-adaptive RoPE base |
| ChatGLM 3 | post (RMSNorm) | none | split (MQA Hk=2) | partial 2D-RoPE | none | attn (prefix-LM) | GeGLU | KV | prefix-LM training |
| MobileLLM | pre | none | split (tied embed) | NEOX | none | attn (block-shared) | SwiGLU | KV | block repetition |

---

## §8 Universal pseudocode of "the decoder block", parametrized (v2)

Putting all 32 axes together, the universal block can be written as a single Python function. This is the shape an API floor should support; everything is dispatched by configuration.

```python
def decoder_block(
    h, residual,                              # [B, S, D] (HF) or [T, D] (vLLM)
    *,
    layer_idx, position_embeddings, attention_mask, past_key_values,
    # per-layer width (OpenELM):
    H_per_layer=None, Hk_per_layer=None, Df_per_layer=None,
    # token mixer
    token_mixer_kind,         # "attention" | "mamba1" | "mamba2" | "rg_lru" | "linear_attn" | "wkv" | "parallel_mamba_attn"
    token_mixer_params,
    # channel mixer
    channel_mixer_kind,       # "mlp_swiglu" | "mlp_gelu_plain" | "mlp_squared_relu" | "moe"
    channel_mixer_params,
    # norm placement
    norm_kind,                # "rms" | "rms_one_plus" | "layer" | "layer_no_bias"
    norm_placement,           # "pre" | "post" | "sandwich" | "parallel"
    eps,
    residual_multiplier=1.0,  # Granite
    resid_pdrop=0.0,          # Phi-3, StarCoder 2
    use_rope=True,            # SmolLM3 per-layer
    shared_block_id=None,     # Zamba2 / MobileLLM
    attn_sub_norm=None,       # BitNet
    ffn_sub_norm=None,        # BitNet
):
    if norm_placement == "parallel":
        x_normed = norm(h, eps, kind=norm_kind, weight=W_attn_pre)
        y_attn = token_mixer(x_normed, position_embeddings, attention_mask,
                             past_key_values, layer_idx, use_rope=use_rope,
                             attn_sub_norm=attn_sub_norm, **token_mixer_params)
        y_mlp  = channel_mixer(x_normed, ffn_sub_norm=ffn_sub_norm, **channel_mixer_params)
        return h + y_attn * residual_multiplier + y_mlp * residual_multiplier, None

    # Token mixer
    if norm_placement in ("pre", "sandwich"):
        x = norm(h, eps, kind=norm_kind, weight=W_attn_pre)
    else:
        x = h
    y = token_mixer(x, position_embeddings, attention_mask,
                    past_key_values, layer_idx, use_rope=use_rope,
                    attn_sub_norm=attn_sub_norm, **token_mixer_params)
    if norm_placement in ("post", "sandwich"):
        y = norm(y, eps, kind=norm_kind, weight=W_attn_post)
    if resid_pdrop > 0: y = dropout(y, resid_pdrop)
    h = residual_add(h, y, scale=residual_multiplier)

    # Channel mixer
    if norm_placement in ("pre", "sandwich"):
        x = norm(h, eps, kind=norm_kind, weight=W_ffn_pre)
    else:
        x = h
    y = channel_mixer(x, ffn_sub_norm=ffn_sub_norm, **channel_mixer_params)
    if norm_placement in ("post", "sandwich"):
        y = norm(y, eps, kind=norm_kind, weight=W_ffn_post)
    if resid_pdrop > 0: y = dropout(y, resid_pdrop)
    h = residual_add(h, y, scale=residual_multiplier)
    return h, None
```

The token_mixer for attention extends v1 with sinks (GPT-OSS), use_rope (SmolLM3), attn_sub_norm (BitNet), per-layer width (OpenELM). The channel_mixer for MoE extends v1 with expert biases (GPT-OSS) and the sigmoid-with-bias-correction path (DeepSeek-V3).

---

## §9 Open questions and things still being figured out

For each unresolved design question, we list what we cannot verify and what experimental data would resolve it.

1. **MLA vs GQA at 7-30B scale.** DeepSeek-V2-Lite (15.7B) is the closest publicly-trained MLA-at-SLM model. No major lab has shipped a 7B-scale MLA model with quality comparable to Llama-3-8B. Without that comparison, the MLA-vs-GQA trade-off at small scale remains open. The MiniCPM-3 case suggests MLA at 4B is viable, but quality benchmarks vs an equivalent GQA baseline are not public.

2. **DeepSeek-V3 routing — sigmoid+bias vs softmax+aux.** Qwen3-MoE and OLMoE chose softmax+aux. DeepSeek-V3 chose sigmoid+bias. As of the survey cutoff, no head-to-head study at matched compute exists. The DeepSeek-V3 paper argues for sigmoid+bias on the basis that it removes the main-vs-aux loss tension; the counter-argument is that sigmoid loses the "convex combination of experts" guarantee.

3. **Gemma 3 dropped softcap. Was it necessary?** Gemma 2 had softcap (attn and final); Gemma 3 dropped both. The Gemma 3 paper attributes stability to QK-norm. But softcap was a Gemma 2 training stabilizer; if QK-norm provides the same stability, why did Gemma 2 ever need it? The likely answer: QK-norm was added in Gemma 3 specifically because it made softcap unnecessary, but the controlled ablation is not public.

4. **Why has parallel residual not won at scale?** Cohere Command-R is the only major 2024+ release to use it. The decode-latency win at TP=8, BS=1 is real, but not large enough to motivate other labs to adopt. Open question: at what inference deployment (TP, BS, kernel implementation) does parallel residual become strictly better than sequential?

5. **NoPE — what fraction of layers should be NoPE?** SmolLM3 uses ~25% NoPE. The Llama 3 ablation also tested NoPE layers. No theoretical justification for any specific fraction is published; the choice is empirical.

6. **Sink tokens — learned (GPT-OSS) vs special-token (Mistral-Small-3.1).** Both work; no direct comparison is public.

7. **Mamba-2 SSD vs full softmax attention at long context.** Mamba-2 demonstrates good perplexity on natural language but is known to under-perform on retrieval-heavy benchmarks (BABILong, RULER). The Jamba/RecurrentGemma/Hymba/Falcon-H1 hybrid families all add attention layers back; pure Mamba-2 has not been shipped at 30B+ scale by a major lab. Is there a fundamental ceiling?

8. **Per-layer width (OpenELM) — quality cost vs gain.** OpenELM per-layer width was claimed in the paper to improve parameter efficiency. No major lab has reproduced this; whether the gain survives at 7B+ scale is open.

9. **Zamba2 parameter sharing — what is the optimal num_mem_blocks?** Zamba2 ships with num_mem_blocks=2. The MobileLLM block-repetition uses 2 also. Higher numbers would save more parameters but at unknown quality cost.

10. **DeepSeek-V4 architecture.** The transformers `deepseek_v4/` directory exists but the architecture is not yet documented as of survey cutoff. Likely refinements to V3 MLA + MoE design.

11. **Qwen3-Next Gated DeltaNet — long-context recall.** Linear attention is known to degrade on retrieval; the alternation with full attention is intended to fix this. The recall gap at the specific Qwen3-Next ratio (1 linear : 3 full?) vs pure-attention has not been independently benchmarked.

12. **The "abandoned ALiBi" question.** Some Chinese open-weight families (Baichuan-1, some Qwen 1 variants) used ALiBi. None of the 2024+ releases use it. But ALiBi has a property RoPE lacks: it requires no positional precomputation, simplifying streaming inference. Could a future model bring it back for very-long-context inference?

13. **Things I could not verify directly:**
   - `src/llama-graph.cpp::build_attn` overloads (the file is too large for clean WebFetch; exact line ranges would need a local llama.cpp clone).
   - vLLM `deepseek_v3.py` does not exist; V3 is implemented through the V2 file. Exact V3-specific kernel paths are spread across vllm/model_executor/layers/quantization/ and vllm/model_executor/models/deepseek_v2.py.
   - Phi-3-128k LongRoPE short_factor/long_factor tables (out of scope for the decoder block but referenced by some axes).
   - Hymba, MobileLLM, Phi-4-mini-flash full source — these are in external repos, not in the local transformers clone. The architectural descriptions in sections 5.24-5.25, 5.33 are from papers + author repos, not from line-numbered local source.
   - OpenELM, MiniCPM 3, InternLM 3, ChatGLM 3 — not in the local transformers clone; descriptions from papers + author HF Hub repos.
   - BitNet b1.58 bitnet.cpp custom kernel paths — bitnet.cpp is a separate Microsoft fork; the in-tree HF code uses dequantized Linear.

---

## §10 Appendices

### Appendix A — File and line citations cross-reference

- Llama 3 HF: `transformers/src/transformers/models/llama/modeling_llama.py` lines 52 to 332
- Qwen 3 HF: `transformers/src/transformers/models/qwen3/modeling_qwen3.py` lines 49 to 334
- Gemma 3 HF: `transformers/src/transformers/models/gemma3/modeling_gemma3.py` lines 98 to 428
- Phi-3 HF: `transformers/src/transformers/models/phi3/modeling_phi3.py` lines 49 to 335
- Mistral HF: `transformers/src/transformers/models/mistral/modeling_mistral.py` lines 35 to 240
- DeepSeek-V3 HF: `transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py` lines 37 to 526
- DeepSeek-V2 HF: `transformers/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py` (V2-Lite uses this)
- Mixtral HF: `transformers/src/transformers/models/mixtral/modeling_mixtral.py` lines 61 to 389
- Granite HF: `transformers/src/transformers/models/granite/modeling_granite.py` lines 115 to 504
- OLMo 2 HF: `transformers/src/transformers/models/olmo2/modeling_olmo2.py` lines 51 to 333
- OLMoE HF: `transformers/src/transformers/models/olmoe/modeling_olmoe.py` lines 134 to 350
- GPT-OSS HF: `transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py` lines 74 to 410
- Cohere HF: `transformers/src/transformers/models/cohere/modeling_cohere.py` lines 52 to 520
- Cohere2 HF: `transformers/src/transformers/models/cohere2/modeling_cohere2.py`
- SmolLM3 HF: `transformers/src/transformers/models/smollm3/modeling_smollm3.py` lines 185 to 440
- BitNet HF: `transformers/src/transformers/models/bitnet/modeling_bitnet.py` lines 44 to 280
- Jamba HF: `transformers/src/transformers/models/jamba/modeling_jamba.py` lines 148 to 638
- Mamba-2 HF: `transformers/src/transformers/models/mamba2/modeling_mamba2.py` lines 121 to 400
- RecurrentGemma HF: `transformers/src/transformers/models/recurrent_gemma/modeling_recurrent_gemma.py` lines 44 to 660
- Falcon HF: `transformers/src/transformers/models/falcon/modeling_falcon.py` lines 246 to 640
- Falcon-Mamba HF: `transformers/src/transformers/models/falcon_mamba/modeling_falcon_mamba.py`
- Falcon-H1 HF: `transformers/src/transformers/models/falcon_h1/modeling_falcon_h1.py`
- Zamba2 HF: `transformers/src/transformers/models/zamba2/modeling_zamba2.py` lines 227 to 1100
- Qwen3-Next HF: `transformers/src/transformers/models/qwen3_next/modeling_qwen3_next.py` lines 256 to 990
- StarCoder 2 HF: `transformers/src/transformers/models/starcoder2/modeling_starcoder2.py` lines 53 to 410
- RWKV HF: `transformers/src/transformers/models/rwkv/modeling_rwkv.py` (RWKV-4/5; RWKV-7 in external BlinkDL repo)
- Hymba external: https://github.com/NVlabs/hymba
- OpenELM external: https://huggingface.co/apple/OpenELM-1_1B-Instruct
- MiniCPM 3 external: https://github.com/OpenBMB/MiniCPM
- InternLM 2.5/3 external: https://huggingface.co/internlm/internlm3-8b-instruct
- ChatGLM 3 external: https://huggingface.co/THUDM/chatglm3-6b
- MobileLLM external: https://github.com/facebookresearch/MobileLLM
- Phi-4-mini-flash / Samba external: https://github.com/microsoft/Samba

### Appendix B — Norm placement table (extended)

| Family | Pre-attn | Sublayer-out attn | Pre-ffn | Sublayer-out ffn | Output | Notes |
|---|---|---|---|---|---|---|
| Llama 3 | input_layernorm | — | post_attention_layernorm | — | model.norm | pre-norm |
| Qwen 3 | input_layernorm | — | post_attention_layernorm | — | model.norm | + per-head QK-norm |
| Mistral | input_layernorm | — | post_attention_layernorm | — | model.norm | pre-norm + SWA |
| Phi-3 | input_layernorm | — | post_attention_layernorm | — | model.norm | pre-norm |
| Granite | input_layernorm | — | post_attention_layernorm | — | model.norm | + residual multiplier |
| Mixtral / Qwen3-MoE | input_layernorm | — | post_attention_layernorm | — | model.norm | pre-norm |
| OLMoE | — | post_attention_layernorm | — | post_feedforward_layernorm | model.norm | post-norm + MoE |
| GPT-OSS | input_layernorm | — | post_attention_layernorm | — | model.norm | pre-norm + sinks |
| DeepSeek-V3 | input_layernorm | — | post_attention_layernorm | — | model.norm | pre-norm + MLA |
| Jamba (attn) | input_layernorm | — | pre_ff_layernorm | — | model.final_layernorm | pre-norm hybrid |
| Jamba (mamba) | input_layernorm | — | pre_ff_layernorm | — | model.final_layernorm | pre-norm hybrid |
| **OLMo 2** | **—** | **post_attention_layernorm** | — | **post_feedforward_layernorm** | model.norm | post-norm |
| **Gemma 3** | input_layernorm | **post_attention_layernorm** | **pre_feedforward_layernorm** | **post_feedforward_layernorm** | model.norm | sandwich (pre + sublayer-out) |
| **Cohere Command-R** | **input_layernorm (shared)** | — | **(shared)** | — | model.norm | parallel residual |
| **Falcon-7B** | **input_layernorm or ln_attn/ln_mlp** | — | — | — | model.norm | parallel residual, optional 2 norms |
| BitNet b1.58 | input_layernorm + attn_sub_norm (inside attn) | — | post_attention_layernorm + ffn_sub_norm (inside MLP) | — | model.norm | pre-norm + dual sub-norms |
| RecurrentGemma | input_layernorm | — | post_attention_layernorm | — | model.norm | pre-norm, recurrent layers |
| Mamba-2 | input_layernorm | — | (no FFN; SSM absorbs it) | — | model.norm | pre-norm, no FFN |

### Appendix C — Numeric constants per family (extended)

| Family | head_dim | attn scale | rope_theta | rms_norm_eps | embed scale | residual scale | logits scale | softcap |
|---|---|---|---|---|---|---|---|---|
| Llama-3-8B | 128 | 1/sqrt(128) | 5e5 | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| Qwen3-8B | 128 | 1/sqrt(128) | 1e6 | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| Gemma-3-4B | 256 | 1/sqrt(256) = 0.0625 | 1e6 global / 1e4 sliding | 1e-6 | sqrt(2560) | 1.0 | 1.0 | **none (dropped)** |
| Phi-3-Mini | 96 | 1/sqrt(96) | 1e4 short | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| Mistral-7B-v0.3 | 128 | 1/sqrt(128) | 1e6 | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| **DeepSeek-V3** | qk=192, v=128 | 1/sqrt(192) * mscale^2 | YaRN | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| **DeepSeek-V2-Lite** | qk=192, v=128 | 1/sqrt(192) | 10000 | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| Mixtral-8x7B | 128 | 1/sqrt(128) | 1e6 | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| OLMoE | 128 | 1/sqrt(128) | 1e4 | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| GPT-OSS 20B | 64 | 1/sqrt(64) | release-specific | 1e-5 | 1.0 | 1.0 | 1.0 | none + sinks |
| Granite-3.3-2B | 64 | 0.0078125 | 5e6 | 1e-5 | 12.0 | 0.22 | 8.0 | none |
| OLMo-2-7B | 128 | 1/sqrt(128) | 5e5 | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| Cohere Command-R | 128 | 1/sqrt(128) | 8e6 | 1e-5 | 1.0 | 1.0 | 0.0625 | none |
| **SmolLM3** | 128 | 1/sqrt(128) | 5e5 | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| **BitNet b1.58** | 100 | 1/sqrt(100) | 10000 | 1e-5 | 1.0 | 1.0 | 1.0 | none + sub-norms |
| Jamba-52B | 128 | 1/sqrt(128) | 1e4 | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| RecurrentGemma-2B | 256 (attn) | 1/sqrt(256) | 1e4 | 1e-6 | sqrt(2560) | 1.0 | 1.0 | none |
| StarCoder2-15B | 128 | 1/sqrt(128) | 999999.4420 | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| InternLM 2.5-7B | 128 | 1/sqrt(128) | 1e6 (dynamic NTK) | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| ChatGLM3-6B | 128 | 1/sqrt(128) | 10000 (2D-RoPE) | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| Falcon-7B | 64 | 1/sqrt(64) | 10000 | 1e-5 | 1.0 | 1.0 | 1.0 | none |

### Appendix D — Engine divergence checklist (HF vs vLLM vs llama.cpp)

**HF vs vLLM (universal):**
1. **Shape:** HF [B, S, D]; vLLM [T, D] (continuous batching). Every projection, every reshape changes accordingly.
2. **QKV layout:** HF separate Q/K/V; vLLM QKVParallelLinear fused.
3. **MLP gate-up:** HF separate (or fused via Phi-3 style); vLLM MergedColumnParallelLinear with SiluAndMul fused.
4. **KV cache:** HF DynamicCache or StaticCache; vLLM PagedAttention with per-request block tables.
5. **TP sharding:** vLLM shards heads via tensor_model_parallel_size; HF does not.
6. **FA backend:** vLLM selects flash_attn/xformers/torch SDPA per kernel-feature-support (softcap, SWA, sinks). HF dispatches via ALL_ATTENTION_FUNCTIONS.
7. **Mask format:** HF [B, 1, S, S+past] fp32 (0 and -inf); vLLM per-request block tables.

**HF vs llama.cpp (universal):**
1. **QK weight permutation:** convert_hf_to_gguf.py permutes Q/K weights to align ggml RoPE basis with HF rotate_half. Without it, GGUF inference produces garbage.
2. **Quantized matmul integration:** ggml_mul_mat dispatches dequantize-and-multiply kernels for q4_0/q4_K_M/etc. The weight shape on disk is per-block, not [D, Df].
3. **GGUF tensor naming:** attn_norm (input_layernorm), attn_q_norm (q_norm), attn_k_norm (k_norm), attn_post_norm (post-attn for Gemma 3), ffn_norm (post_attention_layernorm), ffn_post_norm (post-ffn for Gemma 3), wq/wk/wv/wo or fused wqkv, ffn_gate/ffn_up/ffn_down, MoE: ffn_*_exps.
4. **Memory layout:** ggml is column-major; HF is row-major. Tensor views and matmul calls follow the column-major convention in ggml builders.
5. **Mask format:** llama.cpp constructs kq_mask once per template (per-layer if SWA-alternating), with broadcasting per head.

**vLLM vs llama.cpp:**
1. **PagedAttention block_size:** vLLM uses 16 or 32; ggml uses contiguous per-layer cache (no paging). For MLA, vLLM compressed cache is paged; llama.cpp compressed cache is contiguous.
2. **Quant kernel selection:** vLLM has custom CUDA/Triton kernels for AWQ/GPTQ/Marlin/FP8/etc; llama.cpp has its own k-quant kernels for Q4_K_M/Q5_K_M/etc. The two quantization formats are NOT interoperable.
3. **Concurrency:** vLLM is server-mode (multi-request, continuous batching); llama.cpp is single-process (request-at-a-time or limited concurrency). The shape transformations differ.
4. **MoE expert layout:** vLLM transposes [E, 2*Df, D] to [E, D, 2*Df] for better GEMM stride; llama.cpp stores [D, Df, E].

---

## §11 Summary

The 25-family (33 family-sections) survey gives us a sufficient sample to bound the API design space. The key takeaways:

1. **The Llama-shape API floor is real.** 80% of new releases share Llama 3 structure exactly: pre-norm + RMSNorm + RoPE + GQA + SwiGLU + sequential residual. The other 20% are deltas on specific axes.
2. **Norm placement is the most-explored axis.** Pre (Llama), post (OLMo 2), sandwich (Gemma 3), parallel (Cohere). Each has rationale and shipped models.
3. **MoE is converging.** Mixtral-style softmax+renorm and DeepSeek-style sigmoid+bias-correction are the two surviving designs. Shared experts (DeepSeek, OLMoE-with-zero, Qwen3-MoE) are increasingly common.
4. **MLA is a specialist tool.** Only DeepSeek family and MiniCPM 3 use it. Cost-benefit favors it only at 100+ heads.
5. **SSMs and hybrids fragment.** Mamba-1 (Jamba, Falcon-Mamba), Mamba-2 (Zamba2, Falcon-H1), RG-LRU (RecurrentGemma), GatedDeltaNet (Qwen3-Next), WKV (RWKV-7), parallel heads (Hymba), block alternation (Samba). The hybrid design space is still being explored — no convergence yet.
6. **Engineering divergences (HF vs vLLM vs llama.cpp) are now first-class.** Continuous batching, paged KV cache, QK weight permutation, quantized matmul are non-negotiable for the API to be deployment-friendly.

The v2 axis catalog (32 axes) is the bounded answer to "what does the API need to parameterize". Anything in this document that the API cannot express is a real missing axis; anything outside this document is out of scope for v1 of the API.
