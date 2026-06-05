# 05 — KV Cache and Attention Variants Across Mainstream SLM Runtimes

**Stream:** Attention variants & KV cache designs
**Goal:** Distill the parameter axes a minimal SLM API must expose to express every shipped attention block, RoPE scheme, and KV-cache layout in mainstream small-language-model runtimes.
**Scope:** ≤ ~14 B parameter models (Qwen3, Llama-3.x 1B/3B/8B, Phi-3/3.5/4-mini, Gemma 2/3, Mistral 7B, SmolLM2, OLMo 2, DeepSeek-V2-Lite, Jamba-Mini, MiniCPM-3, StableLM-2, Granite 3) and the runtimes that serve them (HF transformers, vLLM, SGLang, llama.cpp, MLX, MLC, TensorRT-LLM, Core ML, ExecuTorch, ONNX Runtime GenAI).

---

## Part 0 — Why this matters for the API surface

A `forward()` on a transformer block is, to first approximation, three function calls: (1) build Q/K/V projections from `x`, (2) run attention against a cache, (3) feed-forward. Every shipped SLM differs in:

1. **Projection shape** — number of Q heads vs K/V heads vs latent dim (MHA / GQA / MQA / MLA).
2. **Positional embedding** — where the rotation goes, on which dims, with which θ schedule.
3. **Mask shape** — causal, sliding-window, sink-augmented, or block-diagonal (cross-doc).
4. **Cache layout** — how K/V (or compressed latent c_kv) are stored across time and layers.
5. **Norm placement** — pre/post on residual; pre/post on Q and K.

An API that wants to express *all* of these without per-model special casing must treat each of those as an *axis*, not a flag. The rest of this document enumerates the values that have been observed in production runtimes, then collapses them to the minimum axis set with a compatibility matrix.

References used throughout: Vaswani et al. 2017 (vanilla MHA); Shazeer 2019 (MQA); Ainslie et al. 2023 (GQA); DeepSeek-V2 paper (Liu et al. 2024) and DeepSeek-V3 tech report (MLA); Mistral 7B paper (sliding window); Gemma 2 (Riviere et al. 2024) and Gemma 3 tech report (interleaved local/global); Xiao et al. 2024 (StreamingLLM / sink); Sun et al. 2023 (RetNet); Gu & Dao 2023 (Mamba), Lieber et al. 2024 (Jamba); Brandon et al. 2024 (Cross-Layer Attention / CLA); Ye et al. 2024 (Differential Transformer); RoFormer (Su et al. 2021); NTK-aware (bloc97 reddit / Peng et al.); YaRN (Peng et al. 2023); LongRoPE / Phi-3 (Ding et al. 2024); Llama-3 RoPE scaling (`scaling.py` in `transformers`); ALiBi (Press et al. 2021); Dao et al. FlashAttention-2/3; Kwon et al. 2023 (PagedAttention / vLLM); llama.cpp `llama-kv-cache.cpp`; Apple Core ML `KvCacheState`; Chen et al. 2023 (Medusa, speculative decoding); Leviathan et al. 2023 (vanilla speculative decoding).

---

## Part 1 — Attention variants in shipped SLMs

### 1.1 The Q/K/V projection axis: MHA → GQA → MQA → MLA

The classical multi-head attention defines per-layer parameters `W_q, W_k, W_v ∈ R^{d × d}` where `d = n_heads · head_dim`. All four variants below can be expressed by parameterizing **how many K/V projections there are relative to Q projections, and what space they live in.**

| Variant | n_kv_heads | n_q_heads | KV space dim | KV cache size per token | Shipped in |
|---|---|---|---|---|---|
| **MHA** | `H` | `H` | `H · d_h` | `2 · H · d_h` | GPT-2, Phi-2, original BERT, OLMo-1, StableLM-2, SmolLM2-135M/360M, Pythia |
| **GQA** | `G` (`H/g`, g ∈ {2,4,8}) | `H` | `G · d_h` | `2 · G · d_h` | Llama-3 (1B/3B/8B: H=32, G=8), Qwen3 (0.6B–14B, G=H/4 typical), Mistral-7B-v0.2+, Gemma 2 (G=H/2), Gemma 3 (G=H/4), Phi-3-mini (4K) uses MHA but Phi-3.5-MoE GQA, Phi-4-mini GQA, MiniCPM-3 GQA, Granite-3.0 GQA |
| **MQA** | `1` | `H` | `d_h` | `2 · d_h` | Falcon-7B/40B, StarCoder, PaLM, original Gemini-Nano draft (per public reports), some Phi-Silica |
| **MLA** | n/a (single latent) | `H` | `d_c` (typ. 512) + `d_r` (rope) per token | `d_c + d_r` (~576) | DeepSeek-V2/V2.5/V3, DeepSeek-V2-Lite (16B A2.4B, the only mainstream "SLM-ish" MLA), MiniCPM-3 4B |

**MHA semantics.** Q, K, V each `[B, H, T, d_h]`. Standard scaled-dot-product. Cache is `K, V ∈ [B, H, T_cache, d_h]`.

**GQA semantics.** K, V are `[B, G, T, d_h]`. At attention time, K/V are repeated `H/G` times along the head axis (or, in fused kernels, indexed by `head_idx // (H/G)`). FlashAttention-2 natively supports `kv_heads ≠ q_heads`; FlashAttention-3 makes the broadcast a kernel intrinsic. Trades 1/g cache memory for negligible quality loss when `g ≤ 8` (Ainslie et al. show GQA-8 ≈ MHA on T5-XXL).

**MQA semantics.** Special case G=1. Pathologically small cache; quality regression is real on >2B params unless trained from scratch with MQA.

**MLA semantics (DeepSeek-V2 §2.1).** Replace K, V with a **shared low-rank latent** `c_kv ∈ R^{d_c}` per token, where `d_c ≈ 4 · d_h` (paper uses 512, head_dim 128). At attention time:
```
K_compressed = c_kv @ W_uk      # [d_c] → [H, d_h]
V_compressed = c_kv @ W_uv      # [d_c] → [H, d_h]
```
The trick: `W_uk` can be **absorbed into `W_q`** so the dot-product `Q · K^T` becomes `(Q · W_uk^T) · c_kv^T`, meaning the cache only needs to store `c_kv` (`d_c`) per token rather than `H · d_h`. RoPE is incompatible with this absorption (rotation does not commute through linear maps unless applied head-wise), so MLA splits each head into:
- `d_nope = 128` non-rotated dims, served from latent
- `d_rope = 64` rotated dims, served from a separate small per-token K cache (`k_pe`, single-head)

Total per-token KV cache: `d_c + d_rope = 512 + 64 = 576 fp16 = 1.15 KB`, vs DeepSeek-V2's MHA-equivalent `2 · 128 · 128 · 2 = 65 KB` (factor ~57× smaller, paper §2.2 Table 1). This is the **only** mainstream variant that breaks the "K and V are independent rank-`H · d_h` tensors" assumption an API normally makes.

### 1.2 Mask shape axis: causal → sliding → local/global → sink

| Variant | Mask pattern | Cache implication | Shipped in |
|---|---|---|---|
| **Full causal** | tril(ones) | grows linearly with T | Llama-3, Qwen3-dense, Phi-3, OLMo 2, StableLM-2 |
| **Sliding-window (SWA)** | `i - j < W` and causal | bounded by W tokens | Mistral-7B-v0.1 (W=4096), Gemma 2 (W=4096, alternating), Phi-3-mini-128k (blocksparse + SWA), Mistral-Nemo |
| **Interleaved local/global** | Layer-dependent: layers (1,2,3,4,5) SWA, layer 6 full | per-layer cache bound | Gemma 2 (1:1 alternation, W=4096), Gemma 3 (5:1 ratio, W=1024 local), Cohere Command-R7B (1:3), Mistral-Small-3 (some layers SWA) |
| **Sink** | First k tokens always visible + sliding window | bounded W + k | StreamingLLM inference recipe (Xiao et al. 2024); Qwen2.5-1M long-context fork; not natively trained into any base SLM yet, but enabled at inference for any SWA model |
| **Bidirectional + causal hybrid** | Prefix bidirectional, suffix causal | full cache | Used by some encoder-decoder distillations; not native to listed SLMs |
| **Block-diagonal (packed)** | Per-sequence diagonal blocks | cache per sub-seq | Used in cross-document packed pretraining; runtime concern for vLLM continuous batching |

**Gemma 3 specifics** (which are unusually idiosyncratic): every 6th layer is global full-attention, the other 5 are SWA with W=1024 (vs Gemma 2's W=4096). The local layers do **not** use RoPE base 1M; they use base 10K. Global layers use base 1M. So `rope_theta` is **per-layer**, not per-model. This is the cleanest counter-example to the assumption "one rope config per model".

### 1.3 SSM hybrid: Mamba/Jamba/Zamba/Samba

Pure attention-free SSMs (Mamba-2.8B, RWKV-5/6) exist but the *shipped hybrid* SLMs are more interesting because they expose the API design tension directly.

| Model | Pattern | Attention type | SSM type |
|---|---|---|---|
| **Jamba-Mini-1.6 (12B active)** | 1 attention every 8 layers, MoE every 2 | GQA + RoPE | Mamba-1 (selective scan, d_state=16) |
| **Zamba-2 (1.2B/2.7B/7B)** | 2 shared global attention blocks interleaved | GQA shared block | Mamba-2 |
| **Samba (Microsoft 3.8B)** | Mamba + SWA(2048) interleaved per layer | SWA-MHA | Mamba |
| **Phi-4-mini-flash** | Hybrid SSM + GQA, per Microsoft tech report | GQA | Mamba-style gated linear |
| **Falcon-Mamba-7B** | Pure Mamba, no attention | — | Mamba |

API implication: the block-level type is itself an axis. A "block" in the API is `{attention, mamba, mamba2, linear, identity}`. Hybrid models specify a per-layer schedule. Mamba state lives in a completely different cache (`ssm_state ∈ [B, n_inner, d_state]`, fixed size in T) — see §3 for layout.

### 1.4 Cross-layer KV sharing (CLA, YOCO, MiniCPM)

Brandon et al. 2024 (CLA) and Sun et al. 2024 (YOCO) propose that **multiple consecutive layers share one KV cache**. Only the layer that computes K, V writes; downstream layers read.

Shipped: **MiniCPM-3-4B** (sharing factor 2 across pairs of layers), **YOCO-1.3B reference impl**, some Apple OpenELM variants. Reduces cache by another integer factor on top of GQA.

API axis: `kv_share_group` — list of layer indices that share K/V with which "producer" layer. Default: each layer produces its own.

### 1.5 Differential transformer (Microsoft, Ye et al. 2024)

Each head computes **two** softmax attention maps and subtracts them: `attn = softmax(Q1 K^T) − λ · softmax(Q2 K^T)`. Doubles Q projections (2 × `d_q`), keeps K, V single. The scalar λ is learned per layer.

Shipped: not in a flagship SLM yet but in `DiffTransformer` reference release; reportedly used in early Phi-Diff experiments. **API axis:** `attention.kind ∈ {standard, differential}` with `n_q_splits` and per-layer λ.

### 1.6 Linear attention / RetNet / state-space attention

RetNet (Sun et al. 2023): replaces softmax with retention `Q · (D ⊙ exp(K))`, recurrent form has constant-size state. Not shipped in mainstream SLM (no >1B RetNet checkpoint widely used). Mentioned for completeness — API should permit `attention.kind = retention` but no production SLM forces it.

### 1.7 QK-norm placement

Three locations a runtime might apply norm to Q/K:

| Placement | Models |
|---|---|
| **No QK norm** | Llama-3 (all sizes), Mistral-7B, Phi-3-mini |
| **RMSNorm on Q and K, after projection, before RoPE** | OLMo 2 (all sizes), Chameleon, ViT-22B |
| **RMSNorm on Q and K, after projection, after RoPE** | Qwen3 (all dense + MoE variants), Gemma 3 |
| **Layer-wise scaling on Q only** | μP-style; not in shipped SLMs |

Qwen3's "QK-norm after RoPE" is unusual — most lit says "before RoPE preserves rotation". Qwen team reports it stabilizes training at high LR. **API must let QK-norm placement be `{none, pre_rope, post_rope}` and use independent RMSNorm γ vectors per head dim.**

### 1.8 ALiBi and other bias-based positional schemes

ALiBi (Press et al. 2021): no rotation; add `-m · (i - j)` to attention logits where m is per-head. **Shipped in**: MPT-7B, BLOOM, Falcon-180B (also has RoPE), Mosaic StoryWriter. **Not used** by any current top-tier SLM (Llama-3, Qwen3, Phi, Gemma all use RoPE). API axis: `position_bias ∈ {none, rope, alibi, learned_relative}`; `none + alibi_slopes per head` for ALiBi; `rope + freq config` for RoPE.

### 1.9 Logit cap / softmax variants

- **Soft-cap** (Gemma 2): `logits = soft_cap · tanh(logits / soft_cap)` with cap=50 attention, cap=30 final. Drops to none in Gemma 3. API axis: `attention.logit_softcap ∈ {None, float}`.
- **Attention scale** override: most models use `1/sqrt(d_h)` but DeepSeek-V2 MLA uses `1/sqrt(d_nope + d_rope) = 1/sqrt(192)` (the *combined* head dim), not 128. Phi-3 long-context uses an attention-scale tweak. API axis: `attn_scale: float | None`.
- **Sink token offset** (StreamingLLM): not a model param but an inference param.

---

## Part 2 — RoPE variants

### 2.1 Vanilla RoPE (Su et al. 2021)

For head dim `d_h`, frequencies `θ_i = base^(-2i/d_h)`, i ∈ [0, d_h/2). At position `m`, rotate pairs `(x_{2i}, x_{2i+1})` by angle `m · θ_i`.

Parameters: `base` (a.k.a. `rope_theta`), `d_rotated ∈ [0, d_h]`. **Base values across SLMs**: Llama-2 = 10000; Llama-3 = 500000; Llama-3.1 with scaling = 500000 then scaled; Mistral-7B-v0.1 = 10000; Mistral-7B-v0.3 = 1000000; Qwen2.5 = 1000000; Qwen3 = 5000000 (yes, 5M, Qwen3 0.6B-4B); Phi-3-mini-4k = 10000; Phi-3-mini-128k = 10000 (with LongRoPE rescaling, see 2.5); Gemma 2 = 10000; Gemma 3 = **per-layer**: 10000 (local) / 1000000 (global); DeepSeek-V2 = 10000 (with YaRN); SmolLM2 = 130000.

### 2.2 NTK-aware scaling

Replace `base` with `base · s^(d_h / (d_h − 2))` where `s = L_new / L_train`. Smooth interpolation of frequencies; high-frequency dims rotate slower. Shipped in early Llama-2 long-context forks (Together's LLongMA), CodeLlama-16K. Almost entirely superseded by YaRN / Llama-3 scaling in 2024+.

### 2.3 YaRN (Peng, Quesnelle, Kingma 2023)

Three-piece interpolation:
- For dims with rotation period < context length (`wavelength < L_train`): keep original frequency (extrapolate).
- For dims with rotation period > new context length (`wavelength > α · L_new`): apply linear position-interpolation factor `s = L_new / L_train`.
- In between: ramp linearly.

Also applies temperature scaling `1/t = 0.1 · ln(s) + 1` on the attention softmax to compensate for entropy increase. Parameters: `scaling_factor s`, `original_max_position_embeddings`, `attn_factor`, `beta_fast`, `beta_slow`. **Shipped**: DeepSeek-V2 (default with s=40 for 128K), Qwen2.5-1M, MiniCPM-3.

### 2.4 Llama-3 RoPE scaling

Smooth piecewise formula (`transformers/modeling_rope_utils.py::_compute_llama3_parameters`):
```python
low_freq_factor = 1.0          # below: scale by factor
high_freq_factor = 4.0         # above: identity
original_max_position = 8192
scale_factor = 8.0             # the LP-interpolation factor
wavelen = 2π / freqs
inv_freq_llama = where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
# smooth ramp in between, using (orig_max_pos / wavelen − low_freq_factor) /
#                              (high_freq_factor − low_freq_factor)
```
Shipped in Llama-3.1 8B/70B/405B, Llama-3.2 1B/3B (both with 128K context). Distinct from YaRN in that there is **no softmax temperature term** and the wavelength gating uses absolute wavelengths not periods relative to context length.

### 2.5 LongRoPE / Phi-3 long config (Ding et al. 2024)

Phi-3-mini-128k ships **two distinct frequency vectors** in config: `short_factor` (length `d_h/2`, used when `seq_len ≤ original_max`) and `long_factor` (length `d_h/2`, used when `seq_len > original_max`). Each is a learned per-dim multiplier applied as `inv_freq · factor_i`. Also has attention scale tweak: `attn_factor = sqrt(1 + log(scale)/log(original_max))`.

Shipped in: Phi-3-mini-128k, Phi-3-small-128k, Phi-3.5-mini-instruct, Phi-3.5-MoE-instruct, Phi-4 (some checkpoints). **This is the only SLM RoPE family that has per-dim learned rescaling**.

API axis: `rope.long_factor ∈ R^{d_h/2}`, `short_factor ∈ R^{d_h/2}`, threshold `original_max_position`.

### 2.6 Partial RoPE / non-rotated dims

Rotate only the first `d_rope` dims of each head, leave `d_h - d_rope` untouched. **Shipped**:
- Phi-1 / Phi-1.5 / Phi-2: `partial_rotary_factor = 0.5` (rotate 50%).
- GPT-J, GPT-NeoX-20B: `rotary_pct = 0.25` and 0.25 respectively.
- StableLM-2: `partial_rotary_factor = 0.25`.
- **DeepSeek-V2/V3 MLA**: structural — only the `d_rope = 64` channel is rotated, the `d_nope = 128` channel is not. This is **not** a tuning knob but architectural.

API axis: `rope.d_rope ∈ [0, d_h]` defaults to `d_h`.

### 2.7 2D / Multimodal RoPE (M-RoPE)

Qwen2-VL and Qwen2.5-VL split RoPE dims into temporal, height, width thirds; each gets its own position index. Required for video tokens. Phi-3.5-Vision uses a similar approach. Not relevant for pure text SLMs but the API must allow `rope.position_ids` to be an `[N_axes, B, T]` tensor rather than `[B, T]`. Most text-only SLMs use 1 axis.

### 2.8 Frequency basis layouts (interleaved vs split)

Two equivalent-in-math but different-in-memory ways to apply RoPE:
- **Interleaved** (HF default, original RoFormer): pairs `(x_0, x_1), (x_2, x_3), ...`
- **Split-half** (GPT-NeoX, used by llama.cpp, ggml): first half rotated with second half — `(x_0, x_{d/2}), (x_1, x_{d/2+1}), ...`

These produce **different numerical outputs** unless weights are re-shuffled. llama.cpp re-permutes Llama weights at load time to use the split form (the famous `llama_model_loader::tensor_permute_qk`). Without this permutation, ggml output differs from HF. **API must encode the basis convention** as `rope.basis ∈ {interleaved, split_half}`.

---

## Part 3 — KV cache layouts

### 3.1 Contiguous (HF transformers default)

For each layer, K and V are separate tensors of shape `[B, n_kv_heads, T_max, d_h]` allocated upfront (when `use_cache=True` and `max_cache_len` is set) or grown via `torch.cat` (legacy `DynamicCache`).

```python
# transformers/cache_utils.py::DynamicCache
self.key_cache:   List[Tensor[B, H_kv, T_seen, d_h]]
self.value_cache: List[Tensor[B, H_kv, T_seen, d_h]]
# append: torch.cat([prev, new], dim=-2)
```

Pros: simple, debuggable, single index per (layer, b, h, t, d).
Cons: O(T_max · B) memory always allocated; can't share across requests; reallocation on growth.

Dtype: model weight dtype (usually bf16) or quantized via `transformers.QuantizedCache` (HQQ/quanto, 4/8-bit).

### 3.2 Paged (vLLM PagedAttention, Kwon et al. 2023)

Cache is split into **blocks** of fixed size (default 16 tokens). Each request has a **block table** `[N_blocks]` mapping logical block index to physical block. K and V live in two large pools:

```python
# vLLM block_manager_v2
key_cache:   Tensor[N_blocks_total, block_size, n_kv_heads, d_h]
value_cache: Tensor[N_blocks_total, block_size, n_kv_heads, d_h]
block_tables: Tensor[num_seqs, max_blocks_per_seq]  # int32
```
The PagedAttention kernel takes `block_tables` and gathers K/V on-the-fly. The HND layout (`n_heads` before `d_h`) is what vLLM's CUDA kernel expects; NHD layout is used by some FlashInfer kernels.

Shipped in: **vLLM, SGLang, TensorRT-LLM (v0.10+)**, RayLLM. llama.cpp does **not** use paged attention — it uses a contiguous per-sequence cache with periodic defragmentation.

Block sizes in practice: vLLM defaults to 16; recent versions allow 1, 8, 16, 32. FlashInfer prefers 1 or 16. Page size 1 essentially removes paging overhead at cost of less coalesced loads.

### 3.3 Ring / circular buffer (sliding window models)

For Mistral-7B-v0.1 with W=4096, you only ever need W tokens of cache. llama.cpp implements this as a **ring buffer** with a write head:

```c
// llama.cpp/src/llama-kv-cache.cpp (paraphrased)
struct llama_kv_cache_unified {
    std::vector<llama_kv_cell> cells;  // size = cparams.n_ctx
    uint32_t head;   // next write position
    uint32_t size;   // n_ctx
};
```
For Mistral, llama.cpp also tracks `n_swa` and rotates positions modulo W. Once `T > W`, old slots are overwritten; attention mask is computed from `(absolute_pos % W)`.

For Gemma 2/3 interleaved layers, llama.cpp allocates **two pools**: a full-context pool for global layers and a W-sized ring for local layers (`llama_kv_cache_iswa` since b3389).

### 3.4 Per-layer separate vs unified

- **Per-layer separate** (HF default, vLLM): one cache tensor per layer. Layers can have different shapes (useful for CLA / hybrid).
- **Unified** (some Core ML / MLX builds): all layers concatenated into a single big tensor `[n_layers, ...]`. Simpler kernel dispatch but forces uniform shape; incompatible with hybrid Mamba/attention and with interleaved cache sizes.

### 3.5 HND vs NHD memory layout

For a given (B, T, H, D) cache, two memory orders matter:

- **HND** (`[B, H, T, D]`): heads outer, then time, then dim. Default in HF, vLLM, FlashAttention-2. Good for per-head sequential reads.
- **NHD** (`[B, T, H, D]`): time outer, then heads. Used by some Triton/FlashInfer kernels and by Mamba state for compatibility. Better for batched gather-by-position.

Some kernels (FlashInfer's `BatchPrefillWithPagedKVCache`) accept either via a `kv_layout` flag (`"NHD"` or `"HND"`). The conversion is a transpose — free in PyTorch view but materialized in a copy if the consumer is non-contiguous.

### 3.6 Quantized KV cache

| Scheme | Bits | Group size | Scale storage | Shipped where |
|---|---|---|---|---|
| FP8 E4M3 (per-head) | 8 | head | per-block scale | TensorRT-LLM, vLLM (>=0.5), SGLang |
| FP8 E5M2 | 8 | head | static scale | Some H100 paths |
| INT8 per-token | 8 | token | per-token | HF QuantizedCache (HQQ) |
| INT4 group | 4 | 32 or 64 | per-group | llama.cpp `--cache-type-k q4_0`, MLX-LM, ExecuTorch |
| INT2 / 1-bit | 2 / 1 | n/a | n/a | Experimental (KIVI 2-bit) |

KIVI (Liu et al. 2024) is notable for asymmetry: K cache quantized **per-channel**, V cache quantized **per-token**, because of the math of where the cache participates in the softmax. llama.cpp exposes this via `--cache-type-k` and `--cache-type-v` independently. **API axis:** quantization scheme is independently selectable for K and V.

### 3.7 Stateful (Core ML / MLX) vs explicit-pass (PyTorch/JAX)

Two paradigms:
- **Explicit-pass**: every forward takes `past_key_values` as input and returns updated `past_key_values`. The runtime owns lifetime. Default in HF, vLLM, llama.cpp.
- **Stateful**: the model owns mutable state internal to the compiled graph. Core ML 7+ introduced `MLState` for in-place KV updates (Apple Intelligence SLM, OpenELM, MLX-LM). JAX has `nn.scan` carry. ExecuTorch's mobile delegate uses `MutableBuffer`.

API implication: the API must allow attention modules to declare *who owns the cache*. Two options: (a) accept and return `Cache` objects; (b) hold mutable state and expose `reset()`. Most modern serving paths require (a) because requests are batched and time-multiplexed.

### 3.8 Continuous batching implications

With continuous batching (Orca / vLLM), token slots within a batch belong to **different requests** at different stages of generation. The cache layout must support per-request indexing:

```
slot_mapping: [B_tokens] int32      # for each token in batch, write to this physical slot
block_tables: [B_requests, MaxBlocks]
context_lens: [B_requests]
```

For an SLM API to compose with continuous batching, it must accept `slot_mapping` rather than `(b, t)` indices. PagedAttention takes this for granted; contiguous caches need a per-request shim.

### 3.9 Prefix / system-prompt cache reuse

vLLM `enable_prefix_caching=True`, SGLang `RadixAttention`, TensorRT-LLM `enable_kv_cache_reuse`. The kernel-side cost is zero — they all use the paged layout. What matters for the API: **the cache must be addressable by a content hash of the prefix tokens**. Concretely SGLang keeps a radix tree of `(token_id_seq, physical_block_id)`.

For non-paged caches (llama.cpp), prefix reuse is done via `llama_kv_cache_seq_cp` (copies cells from one seq_id to another). Slower but functional.

---

## Part 4 — Prefill vs decode distinction

### 4.1 Three regimes

| Regime | Q tokens | K tokens | Best kernel |
|---|---|---|---|
| **Prefill** | T (prompt) | T (prompt) | FlashAttention-2/3 (`causal=True`) |
| **Decode** | 1 | T (cache) | FlashDecoding / FlashAttention `M=1` path / vLLM `paged_attention_v2` |
| **Chunked prefill** | C (chunk, 256–2048) | T_so_far | FlashAttention with `cu_seqlens_q ≠ cu_seqlens_k` |

FlashDecoding (Dao 2023, blog) parallelizes the M=1 case across the **K dimension** rather than the M dimension; standard FA2 has zero parallelism in M=1 case. vLLM and TensorRT-LLM both use a FlashDecoding-style kernel for the decode step.

### 4.2 Chunked prefill (DeepSpeed-MII, Sarathi)

Splits a long prefill into chunks of size C and interleaves them with decode steps from other requests. Maximizes GPU utilization. Requires the attention kernel to accept arbitrary `Q.shape[0] = C` against full cache. Shipped in: vLLM (default since v0.5), TensorRT-LLM, SGLang.

### 4.3 Speculative decoding compatibility

Standard speculative (Leviathan et al. 2023): a smaller draft model proposes `k` tokens, verifier validates with one forward pass over the `k` proposed positions. The verifier's attention kernel must support `Q.shape[0] = k` with `K.shape[0] = T_cache + k − 1` and a **block-diagonal-ish mask** (each proposed token sees prior accepted tokens + earlier proposed tokens).

Medusa heads (Cai et al. 2024): single model with extra heads; same kernel as chunked prefill.

EAGLE-2 (Li et al. 2024): tree-form proposals; mask becomes **tree mask** (a static k×k pattern). Most kernels accept arbitrary 2D mask but only at FA2-fallback speed.

API implication: the attention call signature must accept `attn_mask` or a `mask_kind ∈ {causal, sliding, sink, custom_2d}` parameter to express tree masks.

### 4.4 KV cache state during speculation

Speculation requires **provisional writes**: candidate tokens are written into the cache, then rolled back if rejected. Paged caches handle this trivially (release blocks); ring buffers must explicitly rewind the head pointer. llama.cpp's `llama_kv_cache_seq_rm(seq_id, p0, p1)` supports this.

---

## Part 5 — Parameter spaces

### 5.1 The KV cache parameter space

After cataloguing all observed runtimes, every cache can be described by this struct:

```
KVCacheSpec {
  // Shape
  n_layers:        int
  per_layer_shape: list[KVLayerShape]    # len n_layers, supports hybrid

  // Layout
  layout:           {HND, NHD}
  storage:          {contiguous, paged, ring_swa}
  block_size:       int                  # paged: usually 16; contig: 1; ring: window W
  unified_layers:   bool                 # one big tensor vs per-layer list

  // Sharing
  kv_share_group:   list[int]            # producer-layer idx for each layer (CLA)

  // Dtype
  k_dtype:          DType                # bf16 / fp16 / fp8_e4m3 / int8 / int4_group
  v_dtype:          DType                # often differs from k_dtype (KIVI)
  k_group_size:     int | None
  v_group_size:     int | None
  k_quant_axis:     {channel, token, head}
  v_quant_axis:     {channel, token, head}

  // Addressing
  ownership:        {explicit_pass, stateful}
  request_addressing: {batch_index, slot_mapping}
  prefix_reuse:     {none, copy_seq, paged_share}

  // Bounds
  max_seq_len:      int
  ring_window:      int | None           # per-layer if interleaved
}

KVLayerShape {
  kind:             {standard_kv, mla_latent, none_ssm_state}
  n_kv_heads:       int                  # 1=MQA, G=GQA, H=MHA
  head_dim:         int
  // MLA-specific
  d_latent:         int | None           # c_kv dim (DeepSeek-V2: 512)
  d_rope:           int | None           # rotated channel dim (64)
  d_nope:           int | None           # non-rotated channel dim (128)
  // SSM-specific
  d_state:          int | None           # Mamba state dim (typ. 16)
  d_inner:          int | None           # Mamba inner dim
}
```

**Eight axes** describe every shipped cache layout. The biggest unification challenge: MLA's `d_latent` channel is shared across all heads (not multi-head), so a "per-head" tensor view breaks. The API must let a layer cache be either `[..., H, d_h]` *or* `[..., d_latent]`.

### 5.2 The attention parameter space

```
AttentionSpec {
  // Projection topology
  n_q_heads:        int
  n_kv_heads:       int                  # MHA: =n_q; GQA: <n_q; MQA: 1
  head_dim:         int                  # d_h
  kind:             {standard, mla, differential, retention, mamba, mamba2, linear}

  // MLA extras
  q_lora_rank:      int | None           # DeepSeek-V2 = 1536
  kv_lora_rank:     int | None           # = d_latent = 512
  qk_nope_dim:      int | None           # 128
  qk_rope_dim:      int | None           # 64
  v_dim:            int | None           # 128

  // Differential extras
  n_q_streams:      int                  # 2 for Diff
  lambda_init:      float | None

  // Mask
  mask:             {causal, sliding_window, sliding_with_sink, full, block_diag, custom}
  window_size:      int | None           # per-layer if interleaved
  n_sink_tokens:    int | None           # StreamingLLM: typ 4

  // Norm
  qk_norm:          {none, pre_rope_rms, post_rope_rms}
  qk_norm_axis:     {per_head, per_layer}

  // Numerics
  attn_scale:       float | None         # default 1/sqrt(head_dim)
  logit_softcap:    float | None         # Gemma 2: 50

  // RoPE
  rope:             RoPESpec | None

  // Cross-layer KV
  shares_kv_with:   int | None           # layer idx of producer (CLA)
}

RoPESpec {
  basis:            {interleaved, split_half}
  base_theta:       float                # 10000 / 500000 / 1000000 / 5000000 (per-layer in Gemma 3)
  d_rope:           int                  # = head_dim or partial
  position_axes:    int                  # 1 (text) or 3 (vision M-RoPE)

  scaling:          {none, linear_pi, ntk, yarn, llama3, longrope}
  scale_factor:     float | None
  original_max_pos: int | None

  // YaRN
  attn_factor:      float | None
  beta_fast:        float | None
  beta_slow:        float | None

  // Llama-3
  low_freq_factor:  float | None
  high_freq_factor: float | None

  // LongRoPE / Phi-3
  short_factor:     list[float] | None   # len d_h/2
  long_factor:      list[float] | None
  long_threshold:   int | None
}
```

**Eleven axes** in `AttentionSpec`, **eight** in `RoPESpec`. Combined, the model API needs roughly **19 attention-side axes**.

### 5.3 Counts (deliverable summary)

- Attention variants documented: **13** kinds (MHA, GQA, MQA, MLA, SWA, SWA+global, sink, linear/retention, Mamba, Mamba-2, CLA-sharing, Differential, ALiBi-bias).
- RoPE variants documented: **8** (vanilla, NTK, YaRN, Llama-3 smooth, LongRoPE / Phi-3, partial, M-RoPE 2D/3D, basis: interleaved vs split-half) plus the no-rope (ALiBi/none) baseline.
- Cache layouts documented: **6** primary (contiguous, paged, ring/SWA, unified-layer, hybrid SSM-attention, stateful) × **5+** dtype variants (bf16, fp16, fp8, int8-per-token, int4-group, int2).

### 5.4 Compatibility matrix — which axes pair

`✔` = combination shipped in production; `~` = possible but rare / unshipped; `✘` = mathematically excluded or no kernel.

| Attention ↓ \ RoPE → | Vanilla | NTK | YaRN | Llama-3 | LongRoPE | Partial | No-RoPE/ALiBi |
|---|---|---|---|---|---|---|---|
| MHA | ✔ Phi-3-mini-4k | ✔ legacy | ~ | ~ | ✔ Phi-3-mini-128k | ✔ Phi-2 | ✔ MPT (ALiBi) |
| GQA | ✔ Llama-3 base, Mistral | ✔ early Mistral | ✔ Qwen2.5-1M | ✔ Llama-3.1/3.2 | ✔ Phi-3.5-MoE | ✔ StableLM-2 | ~ |
| MQA | ✔ Falcon | ~ | ~ | ~ | ~ | ~ | ✘ no shipped |
| MLA | ✘ structural | ✘ | ✔ DeepSeek-V2 (on d_rope only) | ~ | ~ | ✔ **required** (only on d_rope=64 channel) | ✘ |
| SWA (local only) | ✔ Mistral-v0.1 | ~ | ~ | ✔ Mistral-Nemo | ~ | ~ | ~ |
| SWA + global mix | ✔ Gemma 2 | ✘ | ~ | ~ | ~ | ~ | ~ |
| Sink (StreamingLLM) | ✔ inference-time on any | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Differential | ✔ ref | ~ | ~ | ~ | ~ | ~ | ~ |
| Mamba / hybrid | partial — only attn layers use RoPE | — | — | ✔ Jamba attn | — | — | ✘ |

Key constraints:
- **MLA forces partial RoPE**: only the `d_rope = 64` channel is rotated; the rest comes from the absorbable latent. RoPE on the full head dim would break the W_uk absorption trick. So `attention.kind == mla ⇒ rope.d_rope == qk_rope_dim` and the rotation is fed by `k_pe` not by `c_kv`.
- **Gemma 3 forces per-layer RoPE base**: 1M for global layers, 10K for local. So `rope.base_theta` must be **per-layer**, not per-model. Most current libraries treat it as a single scalar — this is a real expressivity gap.
- **SWA + sink** is composable because sink is just "always include first k cells in the attended set". llama.cpp does this via cell flags.

| Attention ↓ \ Cache layout → | Contiguous | Paged | Ring/SWA | Stateful | Hybrid SSM |
|---|---|---|---|---|---|
| MHA | ✔ default | ✔ vLLM | n/a (no SWA) | ✔ Core ML | ~ |
| GQA | ✔ | ✔ | ✔ (Mistral) | ✔ MLX | ✔ Jamba |
| MQA | ✔ | ✔ | ~ | ✔ | ~ |
| MLA | ✔ DeepSeek ref | ~ (vLLM v0.6+ has MLA paging; non-trivial because of latent share) | ✘ | ✔ | ~ |
| SWA local | ✔ | ✔ (block_size matched to W) | ✔ ideal | ✔ | ✔ |
| Gemma 3 interleaved | ✔ (two pools) | ✔ (per-layer block table) | partial — only local layers ring | ✔ | ~ |
| Differential | ✔ (cache same shape as MHA) | ✔ | ✔ | ~ | ~ |
| Mamba SSM | per-layer state, fixed size | ~ | n/a | ✔ | ✔ inherent |

Key constraints:
- **MLA + paged**: vLLM 0.6 added MLA paged support but it stores `c_kv` not `K, V`. Block layout is `[N_blocks, block_size, d_c + d_rope]` — a single tensor, not two. So the "K cache" and "V cache" abstraction collapses into one for MLA.
- **Ring buffer + paged** is redundant; you pick one.
- **Hybrid SSM + paged**: SGLang and TensorRT-LLM handle this by allocating the attention layers' caches in paged form and the SSM layers' state in a flat `[B, L, d_inner, d_state]` tensor; the API must allow a heterogeneous per-layer cache.

| RoPE ↓ \ Cache layout → | Contiguous | Paged | Ring/SWA |
|---|---|---|---|
| Vanilla | ✔ | ✔ | ✔ |
| YaRN / Llama-3 | ✔ | ✔ | ✔ (position computed from absolute, not slot) |
| LongRoPE | ✔ | ✔ | ✔ |
| Partial | ✔ | ✔ | ✔ |
| MLA-style RoPE on `k_pe` only | ✔ | ✔ (but `k_pe` is the only rotated portion) | ✘ |

All RoPE variants are cache-layout-agnostic provided **position ids are absolute** (not slot indices). Ring buffers must therefore store absolute position in each cell. llama.cpp's `llama_kv_cell::pos` is exactly this. Skipping it (using slot index as position) breaks long-context behavior.

---

## Part 6 — Surprising findings / unintuitive constraints

1. **Gemma 3 has per-layer `rope_theta`** (10K local / 1M global). No other current SLM does this. Any library treating `rope_theta` as a model-scope scalar will silently miscompute Gemma 3.
2. **Qwen3 puts QK-norm *after* RoPE**, contradicting the conventional wisdom that this destroys the rotation invariance. The empirical justification in the Qwen3 report is training stability at LR ≥ 3e-4. So **QK-norm placement must be a 3-valued axis**, not a boolean.
3. **MLA's RoPE is structural, not optional**. The `d_rope = 64` channel exists precisely because RoPE doesn't commute with the W_uk absorption. Treating MLA as "GQA with a latent" misses this — you can't run MLA without splitting the head.
4. **llama.cpp re-permutes Q and K weights at load** so it can use the `split_half` RoPE basis. An export from HF (interleaved basis) to GGUF that skips this permutation produces nonsense outputs that *don't crash* — they just degrade fluency. This is a frequent source of silent bugs in third-party conversions.
5. **KIVI quantizes K per-channel and V per-token** — asymmetric. The intuition: K participates in `Q · K^T` (channel-dim is the inner dim, so per-channel quantization preserves the row sum), V participates in `attn · V` (token-dim is the inner dim, so per-token quantization preserves the col sum). An API that only exposes "KV quantization scheme" as one axis will miss this; K and V need independent quant config.
6. **vLLM's MLA cache stores `c_kv`, not K and V**, so the "K cache / V cache" duality disappears. The cache tensor shape is `[N_blocks, block_size, d_c + d_rope]` not `2 × [N_blocks, block_size, H, d_h]`. This is the single biggest deviation from the standard cache API.
7. **Phi-3 long-context uses *two* learned per-dim multiplier vectors** (`short_factor`, `long_factor`), one chosen at runtime based on observed sequence length. No other shipped SLM does this. It's also why Phi-3-mini-128k can't be served by libraries that lack LongRoPE support without a manual config patch.
8. **Sink attention is a pure inference-time recipe**, not a trained-in property. Any SWA model can be served with sink at inference; conversely, training with sink yields models that *don't* function without it (StreamingLLM § 3.2 — Llama-1 with sink train cannot do zero-sink inference).
9. **The "K and V are independent" assumption** is broken by MLA (shared latent), CLA (shared across layers), KIVI (different quant schemes), and StreamingLLM (sink cells distinguished only for K-side mask, not V-side). A robust API treats K and V slots as independent objects with potentially different dtypes, scopes, and lifetimes.
10. **For chunked prefill, the kernel call signature must accept `Q.shape[0] ≠ 1 ≠ T_cache`** — i.e., neither pure prefill nor pure decode. This is a serving-layer concern but pushes back on the model API: `attention(q, kv_cache)` must accept arbitrary `q` length, including 1 (decode), C (chunked prefill), and T (full prefill), uniformly.
11. **Hybrid models force per-layer block-type dispatch**: Jamba's 32 layers are 4× attention, 28× Mamba (with MoE on top). The "block" abstraction must be `Union[AttentionBlock, MambaBlock, ...]` with its own cache type. This is a stronger constraint than the per-layer cache shape — it's a per-layer *block class*.
12. **Sliding window size is sometimes per-layer**: in Gemma 3, only local layers have W; global layers have W = None (full). Even within the SWA family, "the window" is not model-scope.

---

## Part 7 — Minimal API expressivity claim

To express every shipped SLM in this stream's scope, the per-layer block API needs to expose:

- A `block_kind` discriminator (`{attention, mamba, mamba2, linear, identity}`).
- For `attention` blocks: the `AttentionSpec` of §5.2 (11 fields + nested `RoPESpec` of 8 fields = 19 axes).
- For `mamba`/`mamba2` blocks: `{d_state, d_inner, d_conv, expand_factor, use_bias}`.
- A `KVCacheSpec` of §5.1 (8 axes) **at the block level, not the model level** — because hybrid models, CLA, and Gemma 3 all need heterogeneous per-block caches.

And the model-level config exposes:
- `layers: list[BlockSpec]` (length = `n_layers`).
- `embedding`, `final_norm`, `lm_head` (out of scope for this stream).

A minimalist count: **19 + 8 + 1 (block discriminator) = 28 axes** per layer, plus the layer list itself. In practice, axes are highly correlated — a default-laden API can hide ~20 of them behind sensible defaults for the common GQA + RoPE + paged-cache case, exposing only the ~8 that vary across mainstream SLMs (n_q_heads, n_kv_heads, head_dim, rope_theta, rope_scaling_kind, window_size, qk_norm_placement, kv_share_with).

This is the parameter space the layer-0 API of `llm-layers` must cover.

---

## Appendix A — One-line model summaries

- **Llama-3.2-1B**: 16 layers, H=32, G=8, d_h=64, RoPE base=500K + Llama-3 scaling, no QK-norm, full-causal, contiguous cache.
- **Qwen3-0.6B**: 28 layers, H=16, G=8, d_h=128, RoPE base=5M, QK-norm post-RoPE, full-causal.
- **Phi-3-mini-128k**: 32 layers, MHA (H=32, G=32), d_h=96, partial RoPE? No — full RoPE with LongRoPE (short/long factors), no QK-norm.
- **Phi-4-mini (3.8B)**: GQA H=24 G=8 d_h=128, RoPE base=1M, no QK-norm, full-causal.
- **Gemma-3-1B**: 26 layers, GQA H=4 G=1 d_h=256, **per-layer RoPE base** (10K local / 1M global), QK-norm post-RoPE, SWA W=1024 on 5/6 layers, global on 1/6.
- **Mistral-7B-v0.1**: 32 layers, GQA H=32 G=8 d_h=128, RoPE base=10K, SWA W=4096 all layers.
- **DeepSeek-V2-Lite (16B/A2.4B)**: 27 layers, MLA (q_lora=1536, kv_lora=512, qk_nope=128, qk_rope=64, v=128), YaRN scaling, full-causal.
- **Jamba-Mini-1.6**: 32 layers, 4× GQA-attention + 28× Mamba, MoE every 2, attention has RoPE base 1M.
- **MiniCPM-3-4B**: MLA-like (kv_lora=256), CLA sharing across pairs, LongRoPE-style scaling.
- **OLMo-2-1B**: 16 layers, GQA H=16 G=4 d_h=64, RoPE base=500K, QK-norm pre-RoPE, no SWA.
- **SmolLM2-1.7B**: 24 layers, GQA H=32 G=8 d_h=64, RoPE base=130K, no QK-norm.
- **StableLM-2-1.6B**: 24 layers, MHA H=32 d_h=64, partial RoPE (25%), QK-norm pre-RoPE.

## Appendix B — Code-pointer cheatsheet

- HF cache utils: `transformers/src/transformers/cache_utils.py` (DynamicCache, StaticCache, OffloadedCache, QuantizedCache, HybridCache for Gemma).
- HF RoPE utils: `transformers/src/transformers/modeling_rope_utils.py` (all scaling variants).
- vLLM PagedAttention: `vllm/attention/ops/paged_attn.py`, `vllm/worker/cache_engine.py`.
- vLLM MLA: `vllm/attention/backends/mla/*.py` (added v0.6+).
- SGLang radix cache: `sglang/srt/managers/cache_controller.py`.
- llama.cpp KV cache: `src/llama-kv-cache.cpp`, `src/llama-kv-cache-iswa.cpp` (Gemma 2/3).
- FlashAttention-3: `flash-attention/hopper/flash_fwd_kernel_sm90.cu`.
- FlashInfer: `include/flashinfer/attention/decode.cuh`, `include/flashinfer/attention/prefill.cuh`.
- DeepSeek-V2 MLA reference: `DeepSeek-V2/inference/modeling_deepseek.py::DeepseekV2Attention`.
- Phi-3 LongRoPE: `transformers/src/transformers/models/phi3/modeling_phi3.py::Phi3RotaryEmbedding._compute_longrope_parameters`.
- Gemma 3 per-layer rope: `transformers/src/transformers/models/gemma3/modeling_gemma3.py::Gemma3DecoderLayer`.
- StreamingLLM sink: `streaming-llm/streaming_llm/kv_cache.py`.
- Mamba / Mamba-2 SSM state: `mamba/mamba_ssm/modules/mamba_simple.py`.

— end of document —
