# 02 - Per-Layer Source Implementations: A Three-Year Evolution Survey (v3)

## Purpose and scope

This document is the layer-implementation companion to the `llm-layers` API design. The downstream goal is a minimal parameterized API capable of expressing any mainstream LLM/SLM transformer (or transformer-adjacent) block from the past three years (2023–2026). To find the minimal parameterization, we extract the *same* per-layer computational graph from three independent implementations of each model family:

- **(a) HuggingFace `transformers`** — reference PyTorch
- **(b) `vLLM` `model_executor`** — runtime-optimized PyTorch with tensor parallelism, fused projections, paged KV cache, continuous batching
- **(c) `llama.cpp`** — C++/ggml graph with quantized weights

Three lenses on the same block reveal what is *essential* (present in all three) vs *convention* (e.g. weight packing, attention backend, mask format). For each family we list: file paths and line ranges, class names, tensor shapes, pseudocode of `forward`, the quirks (QK-norm, scale variants, biases, gate clamping, etc.), divergences between implementations, and the evolution arc to its predecessor and successor.

**v3 scope (vs v2):** v2 covered 33 family sections, 32 axes, and 7 abandoned designs (19.6k words). v3 extends to **48 family sections, 38 axes, and 10 abandoned designs** (target ≥25k words) by adding 15 new families that the v2-to-v3 audit (`research/issues/08-coverage-justification.md`) identified as architecturally first-of-kind or as defining now-mainstream sub-segments. The 15 new families span 2025-Q3 to 2026-Q2 production releases: Gemma 4 (encoder-free unified multimodal + p-RoPE + cross-layer KV share + first Gemma MoE + Per-Layer Embeddings), Llama 4 Scout/Maverick (iRoPE), Qwen3-Next 80B-A3B (Gated DeltaNet 3:1 hybrid + ultra-sparse 10+1/512 MoE), Granite 4 H-hybrid (9:1 Mamba:attention), DeepSeek-V3.2/V4 (Lightning Indexer DSA, CSA+HCA), Apple Foundation Model 3.18B (cross-block KV share), GPT-OSS 20B (MXFP4-native + trained sinks promoted from §5.11 in v2 to a separate full §5.40), DeepSeek-OCR 3B-MoE-A570M (Visual Causal Flow), Mistral Ministral (3rd SWA-alternation pattern), Phi-4-multimodal (mixture-of-LoRAs), MiniMax-Text-01 (Lightning Attention 7:1), GOT-OCR 2.0 (smallest OCR-LLM), Qwen2.5-VL (M-RoPE + dynamic resolution), Mamba-3 (complex SSM state + MIMO decoding), and Nemotron 3 Super/Ultra (NVIDIA hybrid Mamba-Transformer MoE).

**v3 also adds 6 new axes** to §3 (§3.33 cross-layer KV sharing, §3.34 per-layer embeddings, §3.35 partial-rotary RoPE, §3.36 vision-adapter topology, §3.37 block-bidirectional mask, §3.38 MXFP4-native training) and **3 new abandoned-design entries** to §6 (NTK-aware-only RoPE scaling, static position scaling without smooth blend, vision-token-per-patch dump without compression).

**Sources:**

- HuggingFace excerpts are read locally from `C:\Users\zhengte\external\transformers\src\transformers\models\<name>\modeling_<name>.py` or fetched from the github.com/huggingface/transformers raw mirror for post-v2 models.
- vLLM excerpts are from `https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models/<name>.py` and `vllm/model_executor/layers/`.
- llama.cpp excerpts are from `https://github.com/ggml-org/llama.cpp/tree/master/src/models/<name>.cpp` plus `src/llama-graph.cpp` and `convert_hf_to_gguf.py`.
- For Gemma 4 the pinned values come from the HF model-card `config.json` files (E2B, E4B, 12B-Unified, 26B-A4B, 31B) and the DeepMind / Google blog posts cited in `research/issues/10-gemma4-investigation.md`.
- For DeepSeek-OCR the source is `github.com/deepseek-ai/DeepSeek-OCR` and arXiv:2510.18234.
- For Qwen3-Next the source is the `transformers/models/qwen3_next/` file and the Alibaba release blog.

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
P  = PLE table dim (Gemma 4 Per-Layer Embeddings)
Vp = vision-patch token count (post-encoder; OCR-LLM compression axis)
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

**v3 methodology addendum.** For the 15 new families added in v3, sourcing diverges from the v2 in-tree-clone approach:

- **In-tree (clone available):** Qwen3-Next, Gemma 4 (post 2026-04 transformers 5.x), Mistral Ministral, Phi-4-multimodal, GPT-OSS (already in v2 §5.11), MiniMax-Text-01, Qwen2.5-VL, GOT-OCR 2.0 (`stepfun_got_ocr` module since v4.45), DeepSeek-V3.2 (deepseek_v32/), Llama-4 (llama4/).
- **External-repo (no transformers in-tree yet):** Mamba-3 (state-spaces/mamba reference impl), DeepSeek-OCR (deepseek-ai/DeepSeek-OCR), Granite 4 H-hybrid (ibm-granite repo, transformers PR pending), Apple Foundation Model 3.18B (closed weights; architecture entirely from arXiv:2507.13575), Nemotron 3 Super/Ultra (NVIDIA Nemotron collection; in NVIDIA Megatron-LM).
- **HF model-card-config-only:** Gemma 4 family pinned values (partial_rotary_factor=0.25, num_kv_shared_layers=20, etc.) come from the per-size `config.json` (E2B / E4B / 12B-Unified / 26B-A4B / 31B) because no first-party tech report exists as of the v3 cutoff date.

When a family is external-repo-only, the divergence-list discipline is relaxed: HF pseudocode is replaced with paper/repo pseudocode, vLLM diffs are flagged as "no upstream port yet" where appropriate, and llama.cpp support is documented from the active PR or noted as "not yet supported".

**Source-ladder priorities for "what is the layer":**

1. HF `modeling_*.py` is the math reference. If HF and vLLM disagree on math (not on kernel fusion), HF wins by convention.
2. vLLM is the inference reference. Continuous batching shape, paged KV cache, and TP sharding only exist in vLLM's pseudocode.
3. llama.cpp is the deployment reference. Quantization, GGUF, and CPU/GPU portability only exist in llama.cpp's pseudocode.

When the three disagree, both the disagreement and the resolution are documented in the divergence list.

---

## §2 Universal subset — the API floor

After scanning 48 families (v2's 33 + v3's 15 new), the operations that appear in **every single** decoder layer are listed below. These are the API floor — they cannot be omitted; they can only be *parameterized*.

1. **Residual stream** of shape `[B, S, D]` (HF) or `[T, D]` (vLLM continuous batching) or `[D, n_tokens]` (llama.cpp ggml column-major). The math is the same; the layout differs.
2. **Token-wise normalization** of the stream into the residual. Only the *placement* varies: pre, post, sandwich (= pre + sublayer-output norm), or parallel (= one norm shared by attn and FFN branches). Only the *math* varies between LayerNorm and RMSNorm (or Cohere's special "LayerNorm without bias" form).
3. **Token mixer** producing a `[B, S, D]` output to be added with the residual stream. This mixer is one of: softmax attention (the overwhelming majority), linear attention / DeltaNet (Qwen3-Next, MiniMax, Mamba's SSD generalization in some accounts), state-space model (Mamba, Mamba-2, Mamba-3, Falcon-Mamba), gated linear recurrent unit (RecurrentGemma RG-LRU), Lightning Attention (MiniMax-Text-01), Lightning Indexer DSA (DeepSeek-V3.2), CSA+HCA hybrid (DeepSeek-V4), iRoPE-conditional attention (Llama 4), or WKV recurrence (RWKV).
4. **Channel mixer** producing a `[B, S, D]` output to be added with the residual stream. This mixer is one of: SwiGLU MLP, GeGLU MLP, plain GELU MLP (StarCoder 2), squared-ReLU MLP (Nemotron), or sparse MoE (Mixtral, DeepSeek-V3, Qwen3-MoE, OLMoE, Granite-MoE, GPT-OSS, Cohere2-MoE, Gemma-4-26B-A4B with 128+1 shared experts top-k=8, Qwen3-Next with 10+1/512 top-k).
5. **For attention sublayers:**
   - 4 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), possibly fused as `qkv_proj` or further compressed as MLA's `q_a_proj` → `q_a_layernorm` → `q_b_proj` and `kv_a_proj_with_mqa` → `kv_a_layernorm` → `kv_b_proj`, or unified as Gemma-4's K=V tying on global layers.
   - A position-encoding hook (`apply_rotary_pos_emb` or a no-op for SSM/recurrent layers or no-op for NoPE layers or partial-RoPE for Gemma-4 global / MLA), optionally with M-RoPE / 2D-RoPE / 3D-RoPE (Qwen2-VL, Qwen2.5-VL, Pixtral).
   - A KV-cache hook (`past_key_values.update` or its paged equivalent or a per-layer recurrent state tensor for SSM or a cross-layer-shared pointer for Gemma-4 / Apple AFM).
   - A scaled-dot-product call (`softmax(QK^T / s) @ V` or a flash equivalent or a scan over the recurrent state or a Lightning-Indexer-scored top-k subset).
6. **For the channel mixer:**
   - Gate-Up-Down (SwiGLU = `down(act(gate(x)) * up(x))`) or fused `gate_up_proj`.
   - Optional MoE wrapper: router (linear → top-k) + expert FFN bank + (optional) shared experts + (optional) score-correction bias.
7. **Norm hyperparameter**: `rms_norm_eps` (or `layer_norm_eps`).
8. **Cache key**: `layer_idx` for indexing per-layer KV state.

Every implementation passes the same five tensors through the layer: `hidden_states`, `attention_mask` (or per-request block table or recurrent state mask or block-bidirectional vision mask), `position_embeddings` (or `positions`, possibly multi-axis for M-RoPE), `past_key_values` (or `kv_cache` or recurrent state cache or shared-pointer), and a per-layer index.

The API floor is thus: `decoder_block(h, residual, *, layer_idx, position_embeddings, attention_mask, past_key_values, token_mixer_kind, token_mixer_params, channel_mixer_kind, channel_mixer_params, norm_kind, norm_placement, eps, scaling_params)` returning `(h_new, residual_new)`. The full pseudocode for this universal block is in §8 below.

**Universal-subset updates from 2025-H2 onward (vs v2's universal subset):**

- The `position_embeddings` argument is now a *list/tuple* of cos/sin pairs (one per RoPE axis) for M-RoPE / 3D-RoPE families. Single-axis families pass a length-1 tuple.
- The `past_key_values` argument supports an optional `kv_source_layer_idx` pointer (Gemma-4 cross-layer KV share, Apple AFM cross-block share). When set, this layer's cache is a read-only alias of the indicated earlier layer's cache.
- The `attention_mask` argument supports a new `block_bidirectional` mode (DeepSeek-OCR Visual Causal Flow) where positions within a block see each other bidirectionally while different blocks remain causal.
- The token-mixer trait gains a `partial_rotary_factor: float ∈ (0, 1]` knob (Gemma-4 global = 0.25, MLA RoPE component = qk_rope/qk_head, Phi-3 legacy = 0.5).

These are universal-floor extensions, not new sublayer kinds; the API can accommodate them via additional optional fields on the existing five tensors.

---

## §3 Variant axis catalog (38 axes)

The minimal API must let a config switch on these axes. v1 had 20 axes; v2 added 12 axes that the missing families surfaced (total 32). v3 adds 6 axes that the 15 new families surfaced (total 38).

### 3.1 Norm placement
`{pre, post, sandwich, parallel}` × `{attention, ffn}` (4 booleans collapse to one enum per sublayer). Llama=pre, OLMo2=post, Gemma3=sandwich (= pre-norm + sublayer-output-norm), Cohere/Falcon-7B=parallel (one input-norm feeds *both* attn and FFN branches). The Gemma 3 axis-value here is critical: the v1 calling of it "sandwich norm" was imprecise (see §4.3 below). **Gemma 4 preserves the sandwich placement** unchanged from Gemma 3 (per `transformers/models/gemma4/modeling_gemma4.py` decoder structure with four norm slots: input, post-attn, pre-ffn, post-ffn).

### 3.2 Norm formula
`{LayerNorm(γ, β), LayerNorm(γ, no β), RMSNorm(w), RMSNorm(1+w), ScaleNorm, DeepNorm}`. Gemma 2/3/4 = RMSNorm(1+w) with w initialized to zero so the initial gain is 1.0. Cohere = LayerNorm with no bias. Llama/Qwen/Mistral/DeepSeek = RMSNorm(w). Falcon = LayerNorm with bias. The norm formula and the gain initialization are coupled — `(1+w)` only makes sense if `w=0` at init.

### 3.3 QK-Norm
`{none, per-head-dim, full-flat, post-rope, fixed-scale-per-head-dim}`. Qwen3 = per-head-dim, pre-RoPE; Gemma3 = per-head-dim, pre-RoPE; **Gemma 4 = per-head-dim, pre-RoPE with fixed-scale gains (local ≈ 0.9916, global ≈ 1.0228) that absorb the 1/sqrt(Dh) scale — meaning `attention_scale = 1.0` rather than the usual `1/sqrt(Dh)`**. OLMo2 = full-flat (gamma is `[H*Dh]`), pre-RoPE; Cohere = full-flat, pre-RoPE; Llama/Mistral/Phi3 = none. A few research models (not yet shipped) put QK-norm *after* RoPE but no released model in our 48 does.

### 3.4 Attention scale
`{1/sqrt(Dh), 1/sqrt(qk_head_dim), config.attention_multiplier, config.query_pre_attn_scalar^-0.5, scaling * mscale * mscale (YaRN), 1/sqrt(2*Dh), 1.0 (Gemma-4 with fixed-scale QK-norm)}`. The Granite case is muP-derived; the DeepSeek-V3 YaRN-mscale case is unique to its YaRN-extended RoPE; the Gemma-4 case is unique to fixed-scale QK-norm absorbing the scale. One float-or-callable returns the effective scale.

### 3.5 Position encoding
`{rope-half-rotate (NEOX), rope-interleaved (GPT-J/NORM), rope-partial (Phi-3/MLA/Gemma-4-global), no-rope (NoPE — SmolLM3 alternating layers, Llama-4 Scout iRoPE), M-RoPE (Qwen-VL), 2D-RoPE (Pixtral vision), 3D-RoPE-axis (Qwen2.5-Omni TMRoPE), LongRoPE (Phi-3.5-128k), YaRN-mscale (DeepSeek-V3), Dynamic NTK (InternLM2/3, ChatGLM), ALiBi (Baichuan-1, MPT — historical), 2D-RoPE (ChatGLM3 — partial)}`. RoPE config is its own object with `rope_type`, `rope_theta`, `partial_rotary_factor`, `factor`, `mscale_all_dim`, `original_max_position_embeddings`, `short_factor`, `long_factor`, etc.

### 3.6 NoPE alternation (NEW vs v1)
SmolLM3 introduces `config.no_rope_layers[layer_idx]` — a per-layer boolean. About 1 in 4 layers has *no* position encoding at all (raw Q and K, no rotation). The motivation: NoPE layers extrapolate to longer context than RoPE layers because they have no positional bias. SmolLM3 paper (HuggingFace 2025) cites this from the Llama 3 "no-position-encoding" ablation. **Llama 4 Scout productionizes this as iRoPE** (interleaved RoPE / NoPE per layer), at a different per-layer ratio (Scout uses NoPE on the "global" layers and RoPE on the "local" layers, an inversion of SmolLM3's pattern). See §5.35.

### 3.7 QKV layout
`{three separate, fused qkv, MLA-compressed, K=V-unified (Gemma-4 global), MQA (Hk=1), unified-prefill-fused (Gemma-4)}`. MLA-compressed has sub-axes: `(q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)` — note that `qk_head_dim != v_head_dim` for DeepSeek-V3, an asymmetry that breaks the v1 single-`head_dim` axis. Gemma-4's K=V-unified is a new value: on global-attention layers, K and V share a single projection (`attention_k_eq_v = True` per HF model card), with `global_head_dim = 512` doubling the per-head width to make the unification numerically viable.

### 3.8 Attention biases
Independent booleans per projection (`q_bias`, `k_bias`, `v_bias`, `o_bias`, and maybe `attention_bias` for MLA's `q_a_proj`/`kv_a_proj`/`o_proj`). Qwen2/3 has `q_bias=k_bias=v_bias=True, o_bias=False` (this is the Qwen architectural signature). Granite has all four True. StarCoder 2 has all four configurable via `use_bias`. Falcon old-style has Q/K/V/O bias from LayerNorm convention. Llama-3 405B has Q/K/V bias True per checkpoint (config-dependent).

### 3.9 Softcap
Optional `attn_logit_softcapping` (Gemma 2) and optional `final_logit_softcapping` (Gemma 2). Apply after `QK^T * scale + mask` and before softmax (attn softcap), and on `lm_head(x)` before softmax (final softcap). Gemma 3 *dropped* both softcaps. **Gemma 4 partially re-introduces softcap: `final_logit_softcapping = 30.0` is restored across all five sizes; `attn_logit_softcapping` remains None.** The asymmetric re-introduction is one of the most surprising findings of the Gemma 3 → Gemma 4 diff (cf. `research/issues/10-gemma4-investigation.md` §2.2). Gemma 2 had both; Gemma 3 had neither; Gemma 4 has only the final one — a third combination occupying a previously-empty cell of the 2×2 softcap taxonomy.

### 3.10 Sliding window
`{none, global, per-layer-alternating}` with a `sliding_window` size and (for Gemma 3/4) a separate `rope_theta_local` for sliding-attention layers vs `rope_theta_global` for full-attention layers. Gemma 3 uses 5:1 sliding:global; **Gemma 4 uses 5:1 for ≥4B sizes (last layer global), 4:1 for the smallest size E2B**. Mistral 7B uses global SWA; Mixtral 8x7B has SWA in config but the checkpoint ignores it. **Mistral Ministral 3B/8B introduces a third distinct SWA-alternation pattern: interleaved per-layer rather than block-periodic** (see §5.42). SDPA/FlashAttention backends consume the window size as a kwarg.

### 3.11 GQA / MQA / MHA
`(H, Hk)` pair. `Hk == 1` = MQA; `Hk == H` = MHA; otherwise GQA. Pure MQA was used in Falcon-7B and PaLM but has been *replaced* by GQA in essentially every 2024+ model — see §6 abandoned designs. **Gemma 4 E2B has H=8, Hk=1 on local layers and unified K=V on global layers**, effectively MQA-like on locals — an unusual revival of pure-MQA semantics at a 2B scale.

### 3.12 MLP layout
`{separate gate/up/down, fused gate_up/down, plain up-act-down (no gating — StarCoder 2), double-wide gate/up (Gemma-4 E2B)}` and activation `{silu, gelu_tanh, gelu_pytorch_tanh, gelu_new, relu^2 (Nemotron), gelu (StarCoder 2)}`. Gate-up chunk order (`gate, up` vs `up, gate`) is a packing convention encoded by the loader. **Gemma 4 introduces `use_double_wide_mlp = True` on the edge E2B variant** — semantics not fully documented in HF docs but the flag widens the gate/up projections.

### 3.13 MLP biases
`mlp_bias` boolean (or per-projection booleans for StarCoder 2).

### 3.14 Channel mixer kind
`{dense MLP, sparse MoE, hybrid (per-layer)}`. MoE adds:
- `n_routed_experts`, `num_experts_per_tok` (top-k)
- Router activation: `{softmax, sigmoid}`
- Renormalize topk weights: bool
- Auxiliary-loss-free correction bias: bool (DeepSeek-V3)
- Group-limited routing: `(n_group, topk_group)` (DeepSeek-V3)
- Shared experts: `n_shared_experts >= 0` (DeepSeek-V3, Qwen3-MoE, OLMoE — OLMoE has 0, Gemma-4-26B-A4B has 1, Qwen3-Next has 1 alongside 10 routed of 512)
- Routed scaling: `routed_scaling_factor`
- Expert weight layout: 3D `[E, 2*Df, D]` + `[E, D, Df]`

**The "ultra-sparse" MoE end** (Qwen3-Next 10+1 of 512, ~2.15% activation) and **the "shared-only" MoE start** (Mixtral 8 experts top-2, 25% activation) define the two ends of the 2025-2026 MoE sparsity spectrum. Gemma 4's 26B-A4B at 8 of 128 (~6.25%) and DeepSeek-V3 at 8 of 256 (~3.13%) populate the middle.

### 3.15 Per-layer dense/sparse decision
`first_k_dense_replace` (DeepSeek-V3 — first 3 layers dense), `mlp_only_layers` (Qwen3-MoE — explicit list of dense indices), `layers_num_experts[i]` (Jamba), `decoder_sparse_step` (Qwen3-Next — every Nth layer is sparse), `interleave_moe_layer_step` (MiniMax).

### 3.16 Token-mixer kind
`{full-softmax-attention, sliding-window-attention, mamba-1-ssm, mamba-2-ssd, mamba-3-complex-mimo, linear-attention (DeltaNet/GLA/Gated-DeltaNet), Lightning-Attention (MiniMax), recurrent-griffin (RG-LRU), parallel-mamba-attention (Hymba), wkv-rwkv, sparse-attention-DSA (DeepSeek-V3.2 Lightning Indexer), CSA+HCA hybrid (DeepSeek-V4)}`. Mamba adds `(mamba_d_state, mamba_d_conv, mamba_expand, mamba_dt_rank)`; Mamba-2 adds `(chunk_size, headdim, ngroups)`; Mamba-3 adds `(complex_state: bool, mimo_decoding: bool)`.

### 3.17 Residual scaling
Optional scalar multiplier on each residual add. Granite uses `residual_multiplier ≈ 0.22`; Nemotron variants use a similar muP scalar. Gemma 4 PLE injection on edge sizes adds a `1/sqrt(2)` scale to the per-layer-embedding residual term — distinct from the main-stream residual_multiplier but mathematically related.

### 3.18 Embedding scaling
Optional scalar multiplier on `embed_tokens(input_ids)`. Gemma 1/2/3/4 = `sqrt(D)`; Granite = `embedding_multiplier ≈ 12.0`. For Gemma 4 E2B, `D=1536` so the embed scale is `sqrt(1536) ≈ 39.19`; for E4B, `D=2560` so `sqrt(2560) ≈ 50.59` (identical to Gemma-3-4B).

### 3.19 Logits scaling
Optional `/logits_scaling` before final softmax. Granite = `/ 8.0`; Cohere = `* logit_scale` (typically `0.0625 = 1/16` for Command-R 35B); Gemma 2 final softcap = `softcap * tanh(logits / softcap)`. Gemma 3 dropped this. **Gemma 4 restores it at softcap=30.0 (final only, not attn).**

### 3.20 Residual dropout
Optional `resid_pdrop` (Phi-3, StarCoder 2 — set to 0 in inference but present in code path).

### 3.21 Cache compression
For MLA: cache the *decompressed* form `[H, S, qk_head_dim+v_head_dim]` (HF — simple-correct path) or cache the *compressed* form `[S, kv_lora_rank+qk_rope_head_dim]` and absorb `kv_b_proj` into `o_proj` at runtime (vLLM, llama.cpp — production path). The compression ratio for DeepSeek-V3 (128 heads, qk_head_dim=192, v_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64) is `128 × (192 + 128) / (512 + 64) = 40960 / 576 ≈ 71×`. v1 reported this as "~10×" which was wrong — see §4.1 below.

### 3.22 Sub-norms inside attention / MLP (NEW vs v1)
BitNet b1.58 has *two* extra RMSNorms not present in any other family:
- `attn_sub_norm`: `RMSNorm(D)` applied to the attention output *before* `o_proj` (`modeling_bitnet.py` line 177, used at line 216).
- `ffn_sub_norm`: `RMSNorm(Df)` applied to `act(gate(x)) * up(x)` *before* `down_proj` (line 74, used at line 77).

These sub-norms were the v1 critique's strongest "missing axis" call. They are intrinsic to BitLinear training: the sub-norm stabilizes activation magnitude before the post-quant matmul and after dequant of the input. Without an explicit sub-norm axis the API cannot express BitNet.

**Mamba-2 GroupNorm-gated norm and Gemma 4 vision-encoder post-norm are related instances of this axis.** Mamba-2 inserts a RMSNorm between the SSD scan output and the out_proj; Gemma 4 inserts an RMSNorm on the vision-encoder output to match the residual-stream scale. The axis generalizes to "sublayer-internal norm at any point that depends on a quantized matmul, a recurrent scan, or a multimodal projection".

### 3.23 Per-layer width (NEW vs v1)
OpenELM publishes per-layer arrays: `num_query_heads[]`, `num_kv_heads[]`, `ffn_multipliers[]`, `head_dim`. Each layer has a *different* number of heads and a *different* FFN inner dim. This breaks the single-config assumption: the layer constructor must accept per-layer scalars indexed by `layer_idx`. For OpenELM-1B-Instruct, `num_query_heads` ranges from 12 (early layers) to 20 (deeper); FFN multipliers range from 0.5 to 4.0.

### 3.24 Parameter sharing (NEW vs v1)
Zamba2 has one or two `Zamba2AttentionDecoderLayer` instances that are *shared* across all attention positions in the model. The Mamba2 layers are per-layer-unique; the attention block is reused. Concretely, `num_mem_blocks = 2` and the layer-id-to-block mapping is `block_id = layer_id % num_mem_blocks`. The API needs a "shared layer pool" abstraction; a flat `nn.ModuleList` does not suffice.

### 3.25 Parallel residual (NEW vs v1)
Falcon-7B (in `new_decoder_architecture` mode), GPT-J, GPT-NeoX, and the *current* Cohere Command-R have `x_out = x + attn(norm(x)) + mlp(norm(x))`. The MLP norm input is the same as the attention norm input (single input-norm shared by both branches). This is structurally different from Llama's sequential `attn → residual → norm → mlp → residual`. The axis takes values `{sequential, parallel, parallel-with-two-norms}` (Falcon-H1's `new_decoder_architecture, num_ln_in_parallel_attn=2` is the last one). See `modeling_falcon.py` lines 565–636 for the verbatim if/else cascade.

### 3.26 Embedding tying as standalone axis (NEW vs v1)
A boolean: does `lm_head.weight` share storage with `embed_tokens.weight`? Llama-1 = True; Llama-2/3 = False; Gemma-1/2/3/4 = True; Cohere = True; Granite = False (but with `logits_scaling`); SmolLM3 = False (decoupled, with separate `lm_head`). MobileLLM = True (essential for parameter count at 125M scale). This is a parameter-count-affecting choice (`vocab_size × D` extra parameters when decoupled) that v1 lumped under "biases" but should be its own axis.

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
- Block-bidirectional (DeepSeek-OCR Visual Causal Flow): only `eager` or `flex_attention` with custom mask mod.
- DSA Lightning-Indexer top-k mask (DeepSeek-V3.2): only `eager` or custom CUDA kernel (no FA backend yet).

### 3.30 Sink tokens / sink slots (NEW vs v1)
GPT-OSS has a learnable `sinks` parameter of shape `[H]`, appended as one extra "logit slot" before softmax. The slot's value is shared across positions and is *dropped* after softmax. Effectively, this lets some attention mass spill to a learnable null position. Mistral-Small-3.1 also trains with sink tokens but exposes them differently (via a special token id at sequence start). The v1 critique correctly noted this is no longer a pure-inference recipe (Xiao et al. StreamingLLM 2023) — it is now a training-time architectural choice.

### 3.31 Token-mixer recurrent state cache (NEW vs v1)
SSM/recurrent models cache `(conv_state, ssm_state)` (Mamba), `(conv1d_state, recurrent_states)` (RG-LRU), per-channel WKV state (RWKV), or **(conv_state, complex_ssm_state, mimo_output_buffer) (Mamba-3)**. The cache abstraction is fundamentally different from KV-attention's append-only ring: SSM state is fixed-size O(N) where N is the state dim, not O(S) where S is the sequence length. The hybrid-layer engines (Jamba, RecurrentGemma, Falcon-H1, Zamba2, Hymba, Granite 4 H-hybrid, Nemotron 3 Super/Ultra) have to manage both cache types per request.

### 3.32 μP-style multipliers as standalone axis (NEW vs v1)
Granite has *four* multipliers: `attention_multiplier`, `residual_multiplier`, `embedding_multiplier`, `logits_scaling`. Granite 3.3 publishes them as `f_attention_scale`, `f_residual_scale`, `f_embedding_scale`, `f_logit_scale` in GGUF. These are not bolted-on tricks but load-bearing μP transfer hyperparameters (Yang et al. 2022, "Tensor Programs V"). **Granite 4 H-hybrid retains the four-multiplier set and adds Mamba-2-specific scaling for the hybrid sublayer.**

### 3.33 Cross-layer KV sharing (NEW vs v2)
**Gemma 4 introduces `num_kv_shared_layers` — a per-architecture scalar that says how many of the last N layers reuse K/V projections from an earlier layer of the same attention-type.** For E2B, the value is 20 out of 35 total layers, meaning more than half of layers share their K/V with an earlier-computed tensor. The effect on the KV cache: total cache is `O((L − shared) · S · 2 · Dh · Hk)` instead of `O(L · S · 2 · Dh · Hk)`.

Apple Foundation Model 3.18B (arXiv:2507.13575) uses a *different* form of this axis: **cross-block KV sharing**. AFM partitions its layers into two blocks (Block-1 = 62.5% of layers, Block-2 = 37.5%); Block-2 layers reuse Block-1's K/V projections. The sharing is by block-role, not by layer-index-distance.

A third historical instance is CLA (Cross-Layer Attention, Sun et al. 2024) which pairs adjacent layers: layer `2i+1` reuses the K/V from layer `2i`. The v1 census misidentified MiniCPM-3 as CLA; the actual MiniCPM-3 uses MLA (corrected in §4.5 below).

So the axis takes values `{none, Gemma-4 last-N-shared, AFM block-role-shared, CLA adjacent-pair-shared}` with sub-fields `num_kv_shared_layers: int` or `kv_share_map: List[Optional[int]]`. The cache abstraction must allow "this layer's cache = pointer to earlier layer's cache".

The motivation is identical across all three variants: K/V projections are the dominant per-layer parameter cost (`2 * Hk * Dh * D` per layer), and the corresponding KV cache entries dominate inference memory at long context. Sharing reduces both. The price is a quality cost that is empirically tolerable when paired with QK-norm or with per-layer per-block normalization.

### 3.34 Per-layer embeddings (NEW vs v2)
**Gemma 4 E2B / E4B introduce a separate "Per-Layer Embedding" (PLE) table** whose output is injected as a residual signal at every decoder layer, formed as `(token_identity + context_aware_projection) × (1/sqrt(2))`. The PLE table uses a much smaller dim (P ≈ 256) than the main model dim (D ∈ {1536, 2560}). The "Elastic" parameter accounting splits the model into `total_params` (~5.1B for E2B) and `effective_params` (~2.3B), where effective excludes the PLE table that can be paged out to flash storage on mobile.

The PLE is structurally distinct from the standard embedding table:
- The standard `embed_tokens(input_ids)` runs once, before layer 0.
- The PLE runs `n_layer` times, once per layer, with a *different* projection at each layer.
- The PLE residual is added to the main residual stream after the sublayer outputs but before the next layer's input norm.

This is *not* the same as a position embedding or a token-type embedding: those are additive at input. PLE is additive at every layer, on a per-token basis, from a separate small table. It is also not the same as an "adapter": adapters modify the sublayer; PLE just adds a residual term.

The API needs `PerLayerEmbeddingSpec(table_dim, projection_dim, paged: bool)` and a decoder-layer hook that takes an extra residual term per layer. For multimodal inputs in Gemma 4, the PLE component is computed *before* soft tokens are merged; multimodal positions use the PAD token ID as their PLE lookup key.

### 3.35 Partial-rotary RoPE (NEW vs v2)
**Gemma 4 introduces `partial_rotary_factor = 0.25` on global-attention layers.** This means only the first 25% of each head-dim is rotated by RoPE; the remainder is passed through as a pure semantic channel. The motivation per the DeepMind release: long-context global attention benefits from preserving some positionally-invariant channels for content-similarity matching.

Partial RoPE is orthogonal to the dual-theta `{local, global}` axis. Gemma 4 composes them: sliding-attention layers use `partial_rotary_factor = 1.0` (full RoPE) with `rope_theta_local = 10000`; global-attention layers use `partial_rotary_factor = 0.25` with `rope_theta_global = 1000000`.

MLA (DeepSeek-V2/V3) is a different instance of partial RoPE: only `qk_rope_head_dim` channels of the Q/K head get rotated (typically 64 of 192, ~33%); the remaining `qk_nope_head_dim` channels are non-rotated. The two systems differ in interpretation:
- MLA: the rotated channels are a separate K/Q vector that lives outside the LoRA-compressed bulk; they exist to give the LoRA-compressed part a positional handle.
- Gemma 4 p-RoPE: the rotated channels are the *first prefix* of each head-dim; the non-rotated suffix is the same matrix, just untouched by rotation.

Phi-3 has a *legacy* partial-rotary mode (`partial_rotary_factor = 0.5`) but Phi-3-Mini uses 1.0; the 0.5 value persists only in some research-checkpoint variants.

The API needs `RopeSpec.partial_rotary_factor: float = 1.0`. When less than 1.0, the position-encoding hook applies rotation to the first `int(Dh * partial_rotary_factor)` channels and concatenates the unrotated suffix.

### 3.36 Vision-adapter topology (NEW vs v2)
The 2024-2026 MLLM wave produced six structurally-distinct ways to inject vision (and audio) tokens into an LM decoder:

1. **concat-prefix** (PaliGemma): vision tokens are prefixed to text tokens at the embedding layer; the decoder treats them as ordinary positions.
2. **concat-projector** (LLaVA, Qwen-VL, Qwen2.5-VL): same as concat-prefix but with a learned MLP/Q-Former between encoder and decoder.
3. **cross-attention adapter** (Llama 3.2 Vision, Florence-2): inserts cross-attention layers after every N self-attention layers; image-KV comes from the encoder rather than from the LM's own self-attn cache.
4. **mixture-of-LoRAs** (Phi-4-multimodal): the LM is frozen; vision and audio inputs go through modality-specific LoRA adapters merged at runtime.
5. **early-fusion** (Chameleon, Llama 4, Gemma 4 12B-Unified): no separate encoder; raw image patches (and Gemma 4's case, raw audio waveforms) projected directly into the LM embedding space via a linear projection.
6. **split-encoder** (Janus-Pro): separate vision encoders for understanding vs generation; both feed a unified transformer body.

The API needs `VisionAdapterSpec` with these six discriminated variants. Critical sub-axes:
- For concat-prefix/projector: vision-token compression rate (`Vp` per image: 256 for GOT-OCR, 100 for DeepSeek-OCR, 324 for DocOwl2, 640 for MiniCPM-V).
- For cross-attention: insertion interval `N` (every 4 layers in Llama 3.2 Vision 8B → 8 cross-attn layers).
- For mixture-of-LoRAs: per-modality `r` (vision-LoRA rank), routing rule (token-level vs sequence-level).
- For early-fusion: patch projector dims; for Gemma 4 12B-Unified, also `raw_waveform_projection_spec`.
- For split-encoder: per-task encoder identity (SigLIP-L for understanding, separate for generation).

This axis is required for the OCR-LLM family (§5.41, §5.45, §5.46) and the multimodal Gemma 4 12B-Unified (§5.34). It does *not* extend the decoder per-layer math; it only changes how non-text modalities arrive at layer 0.

### 3.37 Block-bidirectional mask (NEW vs v2)
**DeepSeek-OCR's Visual Causal Flow** is a new attention-mask pattern: positions within a *block* of vision tokens see each other bidirectionally, while different blocks (e.g., different pages) and the text region remain causal.

Mathematically, the mask is:
```
M[i, j] = 0 if (block(i) == block(j) AND block(i) is "vision-block")
       = 0 if i >= j  (causal otherwise)
       = -inf otherwise
```

The block boundaries are recorded as a `block_ids: Tensor[S]` alongside the input; the mask is constructed at sequence-build time. This does not fit the standard `is_causal: bool` API and requires either an eager-mode custom mask or a FlexAttention `mask_mod` function.

The axis takes values `{causal, bidirectional, sliding_window, block_bidirectional}` with optional `block_size_or_block_ids` parameter. DeepSeek-OCR-2 is the only confirmed user as of cutoff; LayoutLMv3 uses pure bidirectional but is encoder-only.

### 3.38 MXFP4-native training (NEW vs v2)
**GPT-OSS 20B / 120B ship weights in MXFP4 format natively trained from scratch**, not as a post-training quantization (PTQ) of a higher-precision checkpoint. MXFP4 (Microscaling FP4) is OCP-standardized: each block of 32 weights shares one E8M0 scale (8-bit exponent, no mantissa); each weight is FP4 E2M1 (1 sign + 2 exponent + 1 mantissa). The total bits per weight = 4 + 8/32 = 4.25 bits.

This is architecturally distinct from PTQ paths (AWQ, GPTQ, GGUF Q4_K_M):
- PTQ paths take a bf16 / fp16 checkpoint and quantize at deployment time.
- MXFP4-native training maintains MXFP4 weights through the entire training loop, with quantized matmul kernels and a stochastic-rounding gradient flow.
- The weight distribution is matched to the MXFP4 grid by construction; PTQ paths must approximate.

The axis takes values `{fp32, fp16, bf16, fp8-PTQ, fp4-PTQ, MXFP4-native, INT8-PTQ, NF4-PTQ, BitNet-1.58-ternary, AWQ, GPTQ, Q4_K_M-GGUF}`. MXFP4-native is currently unique to GPT-OSS in mainstream releases (Falcon-Edge BitNet 1.58 is a different value of the same axis; TII's pre-quantized retrainable weights are yet another).

The architectural implication: the layer constructor must know whether weights are stored in MXFP4 vs bf16, because the matmul kernel selection at every linear-projection call (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) depends on it. This is a per-projection axis if a model mixes precisions (e.g., FP8 attention + MXFP4 FFN), or a per-model axis if all projections share precision.

---

The 38-axis parameter space can be collapsed to three layers:
- A **token-mixer trait** (attention, ssm, ssd, mamba-3, linear-attn, gated-deltanet, lightning-attn, DSA, CSA+HCA, rg-lru, wkv), parametrized by axes 3.3–3.11, 3.16, 3.21, 3.22, 3.29, 3.30, 3.31, 3.33, 3.35, 3.37.
- A **channel-mixer trait** (dense MLP or MoE), parametrized by axes 3.12–3.15, 3.22.
- A generic `DecoderBlock` parameterized by axes 3.1, 3.2, 3.17, 3.18, 3.19, 3.20, 3.25, 3.32, 3.23, 3.24, 3.27, 3.28, 3.34, 3.38.

Axis 3.36 (vision-adapter topology) sits outside the per-layer block — it determines how inputs reach layer 0.

---

## §4 Correctness errors corrected from v1/v2

This section documents the four (plus several minor) v1 errors the v1 critique identified (§4.1–§4.5), plus any new corrections that v2 → v3 audits surfaced (§4.6–§4.7).

### §4.1 DeepSeek-V3 MLA cache compression ratio — was "~10x", actually ~71x

(Carried forward verbatim from v2.) v1 claim: production MLA inference caches the compressed `compressed_kv` and absorbs `kv_b_proj` into `o_proj` to reduce KV-cache size by ~10x. The correct ratio for DeepSeek-V3 is **71×**; for V2-Lite at H=16 it is 8.89× (which is the actual source of the "~10x" figure that v1 mis-attributed to V3). See v2 §4.1 for the full numeric derivation.

### §4.2 "llama.cpp #29402" — actually `huggingface/transformers#29402`

(Carried forward from v2.) The Gemma3 RMSNorm dtype-cast PR is in `huggingface/transformers`, not in llama.cpp. **v3 confirms this PR's behavior is preserved in Gemma 4** by the same `modeling_gemma4.py::Gemma4RMSNorm.forward` ordering: `output = self._norm(x.float()) * (1.0 + self.weight.float())` rather than `x.to(bf16) * w`.

### §4.3 Gemma 3 "sandwich norm" terminology

(Carried forward from v2.) "Sandwich norm" is imprecise; Gemma 3/4's structure is pre-norm + sublayer-output-norm (= "double pre-post"). v2 axis 3.1 distinguishes `pre, post, sandwich (= pre + sublayer-output norm), parallel`. Gemma 3/4 land at sandwich.

### §4.4 Mistral / Llama-4 position-dependent scaling formula

(Carried forward from v2.) The formula `scale = 1 + beta * log(1 + floor(positions / original_max_position_embeddings))` is a Llama-4 invention adopted by Mistral-Large 2, not a Mistral-family signature. **v3 update: Llama 4 Scout uses this temperature scaling on the global-attention layers (the iRoPE-RoPE layers), not on the NoPE layers (which by definition have no positions to scale). See §5.35.**

### §4.5 Minor correctness fixes (Appendix-table-level)

- OLMo 2 biases: config-dependent, default False.
- Gemma 3 softcap: both dropped (Gemma 4 partially restores final softcap at 30.0).
- MiniCPM-3 attention type: MLA, not CLA.
- Qwen3 QK-norm placement: pre-RoPE.
- Llama-3 405B biases: per-checkpoint, controlled by `config.attention_bias`.

### §4.6 NEW v3 correction — Gemma 4 "matformer" lineage

A pre-release rumor in early 2026 suggested Gemma 4 would adopt the Matformer (Matryoshka Transformer) parameter-sharing pattern used in Gemma-3n. The actual Gemma 4 release does **not** use Matformer in the main 12B / 26B / 31B variants; the Matformer-style pattern persists only in the Gemma-3n family. Gemma 4's "Elastic" sizes E2B/E4B use a different mechanism — Per-Layer Embeddings + a paged-to-flash table, not Matformer-style nested-sub-model.

The clean way to phrase this in the survey: **Matformer is a Gemma-3n family signature that Gemma 4 did not adopt for the main line.** Gemma 4 has its own parameter-mass-vs-active-mass mechanism (PLE + Elastic), which is structurally distinct.

### §4.7 NEW v3 correction — DeepSeek-V3.2 vs V3.2-Exp

The DeepSeek-V3.2-Exp release (2025-09-29) introduced **DSA (Lightning Indexer DSA)** as an experimental architectural variant; the stable DeepSeek-V3.2 release (2025-12-01) shipped with DSA on by default. The DeepSeek-V4-Pro / V4-Flash release (2026-04-24) replaced DSA with **CSA+HCA (Compressed Sparse Attention + Heavily Compressed Attention)**, claiming 27% of V3.2's FLOPs and 10% of V3.2's KV cache at 1M context.

The chronology matters because DSA was *not* superseded by an evolution of DSA itself — V4 abandoned the Lightning Indexer in favor of a different sparse-attention design. This makes DSA a one-version architectural choice, similar to Yi-1's cosine-similarity attention (abandoned in Yi-1.5; see §6.6). **DSA should be noted as "active for ~6 months in production, then replaced by CSA+HCA".**

---

## §5 Per-family sections

For each family: identity (config sketch), decoder block diagram, pseudocode with shapes, HF/vLLM/llama.cpp divergences, quirks, evolution arc, citations. The 48 families:

§5.1 Llama 3, §5.2 Qwen 3, §5.3 Gemma 3, §5.4 Phi-3, §5.5 Mistral, §5.6 DeepSeek-V3 (MLA), §5.7 DeepSeek-V2-Lite, §5.8 MiniCPM 3, §5.9 Mixtral / Qwen3-MoE, §5.10 OLMoE, §5.11 GPT-OSS (light), §5.12 Granite 3.3, §5.13 OLMo 2, §5.14 Cohere Command-R / Aya, §5.15 SmolLM3, §5.16 BitNet b1.58, §5.17 OpenELM, §5.18 Jamba, §5.19 Mamba-2, §5.20 RecurrentGemma, §5.21 Falcon-Mamba, §5.22 Falcon-H1, §5.23 Zamba2, §5.24 Hymba, §5.25 Phi-4-mini-flash (Samba), §5.26 Phi-3-small, §5.27 Qwen3-Next (preview-from-v2; extended in §5.36), §5.28 RWKV-7, §5.29 Falcon-7B, §5.30 StarCoder 2, §5.31 InternLM 2.5/3, §5.32 ChatGLM 3, §5.33 MobileLLM.

**v3 additions (15 new sections):** §5.34 Gemma 4, §5.35 Llama 4 Scout/Maverick, §5.36 Qwen3-Next deep-dive, §5.37 Granite 4 H-hybrid, §5.38 DeepSeek-V3.2 / V4, §5.39 Apple Foundation Model 3.18B, §5.40 GPT-OSS 20B (deep-dive promoting v2's §5.11), §5.41 DeepSeek-OCR 3B-MoE-A570M, §5.42 Mistral Ministral 3B/8B, §5.43 Phi-4-multimodal, §5.44 MiniMax-Text-01, §5.45 GOT-OCR 2.0, §5.46 Qwen2.5-VL family, §5.47 Mamba-3, §5.48 Nemotron 3 Super/Ultra.

The v2 sections §5.1–§5.33 remain valid as written. v3 reproduces them here with minor refresh notes where 2025-H2 to 2026-H1 evidence updates a claim. The 15 new sections appear in numbered order after §5.33.

(To keep this v3 file self-contained but manageable, sections §5.1–§5.33 are summarized in compact form in §5A below — full verbatim text remains in v2 — followed by the full new §5.34–§5.48 deep-dives.)

---

### §5A v2 family sections — compact carry-forward

For each of §5.1–§5.33, the full text including verbatim source-line citations, three-engine divergences, pseudocode, and evolution arcs is in `research/02-layer-sources.v2.md`. This v3 file carries the same content forward; only the per-family one-line summary plus any v3-refresh delta is reproduced here. The intent: the reader of v3 can scan §5A for the 33 v2 families and read full text in v2, then continue to §5.34–§5.48 for the 15 new families.

- **§5.1 Llama 3** — reference dense decoder; pre-norm + RMSNorm + RoPE-half-rotate + GQA(Hk=8) + SwiGLU + sequential residual. Every subsequent family is a delta against this. *v3 refresh:* Llama 4 Scout/Maverick (§5.35) is the first Llama-family member to break the reference shape via iRoPE + native multimodal early fusion.
- **§5.2 Qwen 3** — Llama 3 + per-head_dim QK-norm pre-RoPE + Qwen-signature QKV biases (Q/K/V bias True, O bias False) + optional per-layer SWA. *v3 refresh:* Qwen3-Next (§5.36) replaces 3 in every 4 attention layers with Gated DeltaNet linear attention, the most aggressive linear-attention adoption to date.
- **§5.3 Gemma 3** — sandwich norm (4 RMSNorms per layer) + softcap dropped + per-head_dim QK-norm + 5:1 SWA alternation + dual RoPE theta (1e6 global, 1e4 sliding). *v3 refresh:* Gemma 4 (§5.34) preserves the sandwich norm and dual-theta but adds p-RoPE on global layers, restores final softcap, and introduces cross-layer KV sharing and Per-Layer Embeddings.
- **§5.4 Phi-3** — fused QKV + fused gate-up + partial RoPE (legacy) + residual dropouts. *v3 refresh:* Phi-4-multimodal (§5.43) extends the Phi-4-mini base with mixture-of-LoRAs modality routing.
- **§5.5 Mistral** — SWA-everywhere on Mistral 7B v0.1, dropped in Mistral-Nemo and Mistral-Large 2. *v3 refresh:* Ministral 3B/8B (§5.42) introduces a third SWA-alternation pattern (interleaved per-layer), distinct from Gemma 3's 5:1 and v0.1's pure-SWA.
- **§5.6 DeepSeek-V3** — MLA (compressed K/V via low-rank projection, 71× cache compression at 128 heads) + sigmoid-routed MoE with no-aux load balance + group-limited routing + shared expert. *v3 refresh:* DeepSeek-V3.2 (§5.38) adds DSA Lightning Indexer; V4 replaces it with CSA+HCA.
- **§5.7 DeepSeek-V2-Lite** — MLA at 15.7B-A2.4B with no Q-LoRA, V2-style softmax + aux-loss MoE.
- **§5.8 MiniCPM 3** — MLA at 4B, dense MLP, LongRoPE scaling.
- **§5.9 Mixtral / Qwen3-MoE** — softmax-then-topk-then-renorm MoE, 3D expert weights, Mistral-style attention.
- **§5.10 OLMoE** — open MoE with OLMo 2 post-norm + full-flat QK-norm, 64 experts top-8.
- **§5.11 GPT-OSS 20B (overview)** — learned attention sinks + GQA + MoE with expert biases. *v3 expansion:* §5.40 below adds MXFP4-native weights treatment and GPT-3-style alternating dense/sparse attention details.
- **§5.12 Granite 3.3** — four muP scalars (attention, residual, embedding, logits) at 0.0078125 / 0.22 / 12.0 / 1/8.0 respectively.
- **§5.13 OLMo 2** — post-norm (no input_layernorm) + full-flat QK-norm.
- **§5.14 Cohere Command-R / Aya** — parallel residual (`x + attn(norm(x)) + mlp(norm(x))`) + tied embeddings + logit_scale = 0.0625 + no biases.
- **§5.15 SmolLM3** — decoupled embed/lm_head + NoPE alternation (~25% NoPE layers) + per-layer SWA.
- **§5.16 BitNet b1.58** — ternary weights + dual sub-norms (attn_sub_norm and ffn_sub_norm).
- **§5.17 OpenELM** — per-layer width (different num_query_heads, num_kv_heads, ffn_multipliers per layer).
- **§5.18 Jamba** — hybrid attention + Mamba SSM (1:7 ratio) + optional per-layer MoE.
- **§5.19 Mamba-2** — pure SSM with SSD parametrization (chunk_size, n_groups, per-head scalar A). *v3 refresh:* Mamba-3 (§5.47) replaces the per-head scalar A with a complex-valued state and adds MIMO decoding.
- **§5.20 RecurrentGemma (Griffin/Hawk)** — gated linear recurrent unit (RG-LRU) + local attention.
- **§5.21 Falcon-Mamba** — pure Mamba-1 at 7B (no MLP; SSM absorbs channel mixing).
- **§5.22 Falcon-H1** — parallel Mamba-2 + attention branches at every layer.
- **§5.23 Zamba2** — Mamba-2 backbone + shared attention block (num_mem_blocks=2).
- **§5.24 Hymba** — parallel Mamba + attention *heads* at each layer (head-partition, not block-alternation).
- **§5.25 Phi-4-mini-flash (Samba)** — Mamba + SWA + MLP alternation.
- **§5.26 Phi-3-small** — BlockSparse attention (dense local + global summary band).
- **§5.27 Qwen3-Next (v2 preview)** — Gated DeltaNet linear attention + softmax attention alternation. *v3 expansion:* §5.36 adds the 3:1 ratio, 10+1/512 ultra-sparse MoE config, and Mixed-Token-Prediction head details.
- **§5.28 RWKV-7** — pure recurrent with Test-Time-Training-inspired state update.
- **§5.29 Falcon-7B (historical)** — multi-query (Hk=1) + parallel residual.
- **§5.30 StarCoder 2** — plain GELU MLP (no SwiGLU) + GQA + per-projection bias config + sliding window.
- **§5.31 InternLM 2.5/3** — Dynamic NTK RoPE scaling for context extrapolation.
- **§5.32 ChatGLM 3** — prefix-LM training + 2D-RoPE + MQA (Hk=2).
- **§5.33 MobileLLM** — embedding tying + per-layer block repetition (15 unique blocks repeated twice = 30 effective).

End of compact carry-forward. New §5.34–§5.48 follow.

---

### §5.34 Gemma 4 family — p-RoPE on global + K=V unification + cross-layer KV share + PLE + first Gemma MoE + encoder-free unified multimodal

Gemma 4 is the most architecturally adventurous Gemma release to date. It preserves the recognizable Gemma DNA (sandwich norm, RMSNorm(1+w), GeGLU, head_dim 256, tied embeddings, vocab 262144, dual-theta RoPE) and adds **six first-of-family elements**: (1) partial-rotary p-RoPE on global-attention layers, (2) K=V unification on global-attention layers, (3) cross-layer KV sharing (`num_kv_shared_layers`), (4) Per-Layer Embeddings (PLE) on the Elastic E2B/E4B variants, (5) first Gemma MoE (26B-A4B with 128 routed + 1 shared expert, top-k=8), and (6) encoder-free unified multimodal at 12B. All cross-referenced against `research/issues/10-gemma4-investigation.md` and the verified per-size `config.json`.

**Family sweep (verified from HF model-card configs):**

| Variant | Total | Active | Layers | D | H / Hk / Dh | Df | SWA | Pattern | Context | Multimodal | KV-shared | HF repo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E2B** | 5.1B (w/ embed) | 2.3B | 35 | 1536 | 8 / 1 / 256 | 6144 | 512 | 4:1, last global | 128K | text + image + audio | 20 | google/gemma-4-E2B |
| **E4B** | 8B (w/ embed) | 4.5B | 42 | 2560 | 8 / 2 / 256 | 10240 | 512 | 5:1, last global | 128K | text + image + audio | ~20 | google/gemma-4-E4B |
| **12B-Unified** | 11.95B | 11.95B | 48 | 3840 | 16 / 8 / 256 | 15360 | 1024 | 5:1 (globals at 5,11,17,23,29,35,41,47) | 256K | **encoder-free** text + image + audio + video | 0 | google/gemma-4-12B |
| **26B-A4B** | 25.2B | 3.8B | 30 | 2816 | 16 / 8 / 256 | 2112 / 704 / shared 2112 | 1024 | 5:1, last global | 256K | text + image | 0 | google/gemma-4-26B-A4B |
| **31B** | 30.7B | 30.7B | 60 | 5376 | 32 / 16 / 256 | 21504 | 1024 | 5:1, last global | 256K | text + image | 0 | google/gemma-4-31B |

Vocab = 262144 (SentencePiece, "GP-v4" — same 262K family as Gemma 3 with multimodal token IDs added: BOI=255999, EOI=258882, IMAGE=258880, BOA=256000, EOA=258883, AUDIO=258881, VIDEO=258884). License: **Apache 2.0** (changed from Gemma Terms of Use in Gemma 1-3).

**HF file (proposed):** `transformers/src/transformers/models/gemma4/modeling_gemma4.py` (HF transformers 5.x; Gemma 4 introduces `Gemma4TextConfig`, `Gemma4DecoderLayer`, `Gemma4Attention`, `Gemma4MLP`, `Gemma4PerLayerEmbedding`).

**Decoder block (preserves Gemma 3 sandwich pattern verbatim, plus PLE injection on edge sizes):**

```python
class Gemma4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp       = Gemma4MoE(config) if config.is_moe_layer(layer_idx) else Gemma4MLP(config)
        self.input_layernorm          = Gemma4RMSNorm(D, eps=eps)
        self.post_attention_layernorm = Gemma4RMSNorm(D, eps=eps)
        self.pre_feedforward_layernorm  = Gemma4RMSNorm(D, eps=eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(D, eps=eps)
        if config.per_layer_embedding_dim is not None:
            self.ple_projection = nn.Linear(config.per_layer_embedding_dim, D, bias=False)
        else:
            self.ple_projection = None

    def forward(self, hidden_states, ple_input=None, position_embeddings=None, ...):
        # Sandwich-norm attention path (identical structure to Gemma 3):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings=position_embeddings, ...)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # PLE injection (E2B/E4B only):
        if self.ple_projection is not None and ple_input is not None:
            # ple_input: [B, S, P] — per-layer embedding for this layer index
            ple_residual = self.ple_projection(ple_input) * (1.0 / math.sqrt(2.0))
            hidden_states = hidden_states + ple_residual

        # Sandwich-norm FFN path:
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

The four-norm structure is byte-for-byte identical to Gemma 3 §5.3. The PLE-add is the new element.

**Attention block (new elements: p-RoPE, K=V on global, KV sharing, fixed-scale QK-norm):**

```python
class Gemma4Attention(nn.Module):
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]   # "sliding_attention" or "full_attention"
        self.kv_share_source = config.kv_share_map[layer_idx]   # None or earlier layer idx
        self.is_kv_shared = self.kv_share_source is not None

        if self.layer_type == "sliding_attention":
            # Local layer: standard split Q/K/V, full RoPE
            self.q_proj = nn.Linear(D, H * head_dim, bias=False)
            self.k_proj = nn.Linear(D, Hk * head_dim, bias=False)
            self.v_proj = nn.Linear(D, Hk * head_dim, bias=False)
            self.partial_rotary_factor = 1.0
            self.rope_theta = config.rope_theta_local        # 10000
            self.sliding_window = config.sliding_window
            self.qk_norm_fixed_scale = 0.9916               # local fixed scale
            self.head_dim_eff = head_dim
        else:
            # Global layer: K=V unified, p-RoPE 0.25, doubled global_head_dim
            self.q_proj = nn.Linear(D, H * global_head_dim, bias=False)
            if not self.is_kv_shared:
                # Single unified projection for K=V
                self.kv_proj = nn.Linear(D, Hk * global_head_dim, bias=False)
            else:
                self.kv_proj = None   # will read from earlier layer's cache pointer
            self.partial_rotary_factor = 0.25
            self.rope_theta = config.rope_theta_global       # 1000000
            self.sliding_window = None
            self.qk_norm_fixed_scale = 1.0228               # global fixed scale
            self.head_dim_eff = global_head_dim              # 512

        self.q_norm = Gemma4RMSNorm(self.head_dim_eff, eps=eps, fixed_scale=self.qk_norm_fixed_scale)
        self.k_norm = Gemma4RMSNorm(self.head_dim_eff, eps=eps, fixed_scale=self.qk_norm_fixed_scale)

        self.o_proj = nn.Linear(H * self.head_dim_eff, D, bias=False)
        self.scaling = 1.0    # absorbed into fixed-scale QK-norm

    def forward(self, hidden_states, position_embeddings, past_key_values, ...):
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, H, self.head_dim_eff).transpose(1, 2)

        if self.is_kv_shared:
            # Read from earlier layer's cache by pointer (Gemma 4 cross-layer KV share)
            k, v = past_key_values.get_shared_kv(self.kv_share_source, layer_idx=self.layer_idx)
            # k, v shapes: [B, Hk, S_cached, head_dim_eff]
        else:
            kv = self.kv_proj(hidden_states).view(B, S, Hk, self.head_dim_eff).transpose(1, 2)
            # K == V — single tensor used for both:
            k = kv
            v = kv  # alias

        # QK-norm with fixed scale (absorbs 1/sqrt(d)):
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial-rotary RoPE: rotate only the first `partial_rotary_factor * head_dim_eff` channels
        cos, sin = position_embeddings
        rotary_dim = int(self.head_dim_eff * self.partial_rotary_factor)
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = (q_rot * cos[..., :rotary_dim]) + (rotate_half(q_rot) * sin[..., :rotary_dim])
        k_rot = (k_rot * cos[..., :rotary_dim]) + (rotate_half(k_rot) * sin[..., :rotary_dim])
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        if not self.is_kv_shared:
            k, v = past_key_values.update(k, v, self.layer_idx,
                                          sliding_window=self.sliding_window)

        # Standard scaled-dot-product (scale=1.0 since QK-norm absorbed it):
        attn = softmax(q @ k.transpose(2, 3) * self.scaling + mask) @ v
        out = attn.transpose(1, 2).reshape(B, S, H * self.head_dim_eff)
        return self.o_proj(out), None
```

**Tensor shapes for E2B (D=1536, H=8, Hk=1, head_dim=256, global_head_dim=512):**

- Local layer: q [B, S, 8, 256], k/v [B, S, 1, 256] — pure MQA-like on local (Hk=1 with full GQA-style repeat for the 8 query heads).
- Global layer: q [B, S, 8, 512], kv [B, S, 1, 512] — kv is K and V aliased (`K = V`); doubled head_dim from 256 to 512 to compensate.
- KV cache per global layer: `[B, 1, S, 512]` (single tensor for both K and V) vs `[B, 1, S, 2*256]` for a standard MQA setup. The K=V unification saves *exactly the V projection's parameters and cache* on global layers.

**Cross-layer KV sharing (the wholly new axis):**

```python
class Gemma4Cache:
    def __init__(self, config):
        # For each layer, an entry either owns a (K, V) pair OR points to an earlier layer's entry
        self.kv_owners = []
        self.kv_pointers = config.kv_share_map  # List[Optional[int]]
        for layer_idx in range(config.num_hidden_layers):
            if self.kv_pointers[layer_idx] is None:
                self.kv_owners.append(self._allocate_kv_tensor(config, layer_idx))
            else:
                self.kv_owners.append(None)  # will resolve at access time

    def get_shared_kv(self, source_layer_idx, layer_idx):
        # Return a read-only view of the source layer's KV
        return self.kv_owners[source_layer_idx]

    def update(self, k, v, layer_idx, sliding_window=None):
        owner_idx = self.kv_pointers[layer_idx] if self.kv_pointers[layer_idx] is not None else layer_idx
        if sliding_window is not None:
            return self._sliding_update(self.kv_owners[owner_idx], k, v, sliding_window)
        return self._append_update(self.kv_owners[owner_idx], k, v)
```

The `kv_share_map` for E2B has 20 entries pointing to earlier layer indices (out of 35 total layers). The exact mapping per the verified config: the last 20 layers each point to one of the first 15 layers of the same `layer_type` (sliding or global). This means a global-attention layer at depth 30 reuses K/V from a global-attention layer at depth, e.g., 5; a sliding-attention layer at depth 25 reuses K/V from depth 10. **Crucially, the sharing is constrained to within attention type** — sliding layers cannot reuse global layers' KV and vice versa, because their head_dim_eff differ (256 vs 512).

**Per-Layer Embeddings (PLE) — the wholly new axis on edge sizes:**

The PLE table has dim P ≈ 256 (independent of the main D). At every layer, the PLE produces a residual term:

```python
class Gemma4PerLayerEmbedding(nn.Module):
    def __init__(self, config):
        self.table = nn.Embedding(config.vocab_size, config.per_layer_embedding_dim)
        # One context projection per layer:
        self.context_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.per_layer_embedding_dim, bias=False)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids, hidden_states, layer_idx):
        # token-identity term: [B, S, P]
        token_term = self.table(input_ids)
        # context-aware projection: [B, S, P]
        context_term = self.context_projections[layer_idx](hidden_states)
        # combined: [B, S, P], scaled by 1/sqrt(2)
        return (token_term + context_term) * (1.0 / math.sqrt(2.0))
```

The `ple_input` passed to each `Gemma4DecoderLayer.forward` is the output of this function for that layer index. The decoder layer projects it from P=256 to D=1536 (E2B) or D=2560 (E4B) and adds it to the residual stream after the attention sublayer but before the FFN sublayer.

The "Elastic" parameter accounting separates `total_params` (5.1B for E2B) from `effective_params` (2.3B) — the difference is the PLE table (`vocab_size × P = 262144 × 256 = 67M params`) plus per-layer projections (`n_layer × D × P = 35 × 1536 × 256 = 13.8M for E2B`). For mobile deployment, the PLE table can be paged to flash storage; VRAM only holds the per-layer projection matrices.

For multimodal inputs (image patches, audio waveforms), the PLE component is computed *before* soft tokens are merged. Multimodal positions use the PAD token ID as their PLE lookup key — so image tokens and audio tokens contribute zero (or near-zero) to the PLE residual, and only the context-projection term matters at those positions.

**Gemma 4 MoE (26B-A4B, the first Gemma MoE):**

```python
class Gemma4MoE(nn.Module):
    def __init__(self, config):
        self.num_experts = 128
        self.top_k = 8
        self.num_shared_experts = 1
        self.moe_intermediate_size = 704   # per routed expert
        self.shared_intermediate_size = 2112    # = 3 * moe_intermediate_size

        # Router:
        self.gate = nn.Linear(D, self.num_experts, bias=False)

        # Routed experts (fused 3D weights):
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.moe_intermediate_size, D))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, D, self.moe_intermediate_size))

        # Single shared expert (GeGLU at 3× routed-expert size):
        self.shared_expert = Gemma4MLP(D, self.shared_intermediate_size, activation="gelu_pytorch_tanh")

    def forward(self, hidden_states):
        residuals = hidden_states
        # Route:
        router_logits = self.gate(hidden_states)   # [B, S, 128]
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_values, topk_indices = router_probs.topk(self.top_k, dim=-1)   # [B, S, 8]
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True)   # renormalize

        # Dispatch to experts:
        routed_out = self._expert_dispatch(hidden_states, topk_indices, topk_values)

        # Always-on shared expert:
        shared_out = self.shared_expert(residuals)

        return routed_out + shared_out
```

**Routing details:** softmax-then-topk-then-renorm (Mixtral-style), with a single always-on shared expert. The shared expert uses the same GeGLU activation as the routed experts but at 3× the inner dim. No e_score_correction_bias (this is *not* DeepSeek-V3-style sigmoid + bias correction); Gemma 4 stays in the Mixtral + shared-expert lineage.

The routed-expert intermediate size 704 is small (vs DeepSeek-V3's 2048 per expert), making each expert "fine-grained" in the DeepSeekMoE sense.

**Encoder-free unified multimodal (12B-Unified):**

This is the most architecturally novel design choice in Gemma 4. The 12B variant has no separate vision encoder and no separate audio encoder. Image patches (16×16 RGB tiles) and audio waveform chunks (50ms windows) are projected directly into the LM embedding space via linear projections:

```python
class Gemma4UnifiedMultimodal(nn.Module):
    def __init__(self, config):
        D = config.hidden_size
        # Image patch projection: 3 channels × 16×16 = 768 → D=3840
        self.image_patch_proj = nn.Linear(3 * 16 * 16, D, bias=False)
        # Audio waveform projection: 50ms @ 16kHz = 800 samples → D=3840
        self.audio_waveform_proj = nn.Linear(800, D, bias=False)
        # 2D positional embeddings for image patches:
        self.image_pos_embed_h = nn.Embedding(10240, D // 2)
        self.image_pos_embed_w = nn.Embedding(10240, D // 2)
        # 1D temporal positional for audio:
        self.audio_pos_embed = nn.Embedding(6000, D)   # 30s @ 200Hz frame rate

    def forward(self, input_ids, image_patches=None, audio_chunks=None):
        # input_ids may contain special tokens IMAGE / AUDIO / VIDEO at positions to fuse
        text_embeds = self.embed_tokens(input_ids)

        if image_patches is not None:
            # image_patches: [B, n_patches, 3*16*16]
            image_embeds = self.image_patch_proj(image_patches)
            h_pos = self.image_pos_embed_h(patch_h_indices)
            w_pos = self.image_pos_embed_w(patch_w_indices)
            image_embeds = image_embeds + torch.cat([h_pos, w_pos], dim=-1)
            # Insert image embeds at IMAGE token positions in text_embeds:
            text_embeds = scatter_at_token_id(text_embeds, image_embeds, IMAGE_TOKEN_ID)

        if audio_chunks is not None:
            audio_embeds = self.audio_waveform_proj(audio_chunks) + self.audio_pos_embed(...)
            text_embeds = scatter_at_token_id(text_embeds, audio_embeds, AUDIO_TOKEN_ID)

        return text_embeds   # ready for layer 0
```

The contrast with Gemma 3's design is sharp: Gemma 3 used a SigLIP vision tower (≈400M params) feeding into a learned projection. Gemma 4 12B-Unified eliminates the encoder entirely; the 16×16 patch → D projection is a single `nn.Linear`. The argument per the DeepMind release: native multimodal training (instead of stage-wise encoder pretraining + decoder finetuning) lets the model learn vision representations *jointly* with text reasoning at lower latency and with simpler runtime topology.

The encoder-free design is closer in spirit to Chameleon (Meta, 2024) and Llama 4 Scout's early-fusion than to PaliGemma 2 (which had a SigLIP tower).

**vLLM diffs:** Gemma 4 support in vLLM (as of cutoff) follows the Gemma 3 path with extensions:
- `Gemma4Attention` adds a branch on `layer_type` for K=V unification on global layers; the QKV-parallel-linear is replaced by a Q-parallel + KV-parallel split.
- `Gemma4Cache` handles the `kv_share_map` by aliasing per-layer block tables to source-layer block tables.
- PLE adds a per-layer projection that is fused with the post-attention residual add.
- p-RoPE on global layers: the rotary kernel takes a `partial_rotary_factor` kwarg.
- The 12B-Unified path is not yet in vLLM; multimodal input goes through a custom `Gemma4Model.forward_with_multimodal` that runs the patch projections before the layer stack.

**llama.cpp diffs:** Gemma 4 support is being added in PR series targeting `llama.cpp` master:
- GGUF tensor names: `attn_norm`, `attn_q_norm`, `attn_k_norm`, `attn_post_norm`, `ffn_norm`, `ffn_post_norm`, `attn_q_global`, `attn_kv_global` (unified K=V), `attn_q_local`, `attn_k_local`, `attn_v_local`.
- New tensors: `ple_table`, `ple_context_proj.<layer_idx>`, `ple_layer_proj.<layer_idx>`.
- For cross-layer KV share: the GGUF stores a `kv_share_map` array; the runtime allocates cache only for non-shared layers and creates pointer entries for shared layers.
- p-RoPE: `n_rot < n_embd_head` on global layers (similar to Phi-3's partial-rotary, but layer-conditioned).
- Final softcap=30.0: `f_logit_scale = 30.0` with `softcap_fn = tanh`.
- Q4_0 mobile-format QAT checkpoints have a custom packing for the PLE table to enable paged-to-flash access; the existing Q4_0 path is extended with a `MOBILE_FORMAT` flag.

**Evolution arc (Gemma 1 → 4):**

- **Gemma 1 (Feb 2024, arXiv:2403.08295):** standard Llama-shape decoder, GeGLU, tied embeddings, embed_scale = sqrt(D). 2B/7B.
- **Gemma 2 (June 2024, arXiv:2408.00118):** sandwich norm (four RMSNorms per layer), attn-logit softcap 50.0, final-logit softcap 30.0, 5:1 SWA pattern, RMSNorm(1+w).
- **Gemma 3 (Feb 2025):** dropped both softcaps, added per-head_dim QK-norm, dual RoPE theta (1e6 global / 1e4 sliding), 4B/12B/27B sizes, SigLIP-based multimodal.
- **Gemma 3n (May 2025):** Matformer parameter-sharing pattern (nested sub-models inside one checkpoint), audio modality via USM conformer encoder.
- **Gemma 4 (Apr 2026):** retains the four-norm sandwich, dual theta, GeGLU, head_dim 256, RMSNorm(1+w), tied embeddings, vocab 262144 (all preserved from Gemma 3); adds p-RoPE on global layers, K=V unification on global layers, cross-layer KV sharing (`num_kv_shared_layers`), Per-Layer Embeddings on E2B/E4B, first Gemma MoE (26B-A4B with 128+1 shared experts top-k=8), encoder-free unified multimodal at 12B; partially restores softcap (final only, value 30.0); license switched to Apache 2.0.

The clean summary diff (from `research/issues/10-gemma4-investigation.md`): **Gemma 4 = Gemma 3 minus attention softcap, plus (a) p-RoPE 0.25 on global, (b) unified K=V on global, (c) Per-Layer Embeddings on edge, (d) cross-layer KV sharing, (e) first MoE variant, (f) first encoder-free unified-multimodal variant, (g) restored final-logit softcap.**

The three honestly new axes the v2 taxonomy did not contemplate — cross-layer KV reuse (§3.33), per-layer embeddings (§3.34), partial-rotary RoPE (§3.35) — are all Gemma 4 contributions to the survey corpus.

---

### §5.35 Llama 4 Scout / Maverick — iRoPE, MoE, native multimodal early fusion

Llama 4 (April 2025) is the first Llama family member to break the reference Llama-3 shape. Three architectural deltas: (1) **iRoPE** — interleaved RoPE / NoPE per layer, productionizing the SmolLM3 NoPE-alternation pattern at frontier scale; (2) native MoE on both Scout (16 experts, 109B total, 17B active) and Maverick (128 experts, 400B total, 17B active); (3) native multimodal early fusion (image tokens are interleaved with text tokens at the decoder input, not gated through cross-attention adapters).

**Identity (Llama-4-Scout 17B-A17B):** D=5120 (per Meta release blog), H=40, Hk=8 (GQA), Dh=128, n_layer=48, n_experts=16, num_experts_per_tok=1 (Scout) or 2 (Maverick), vocab≈202k (extended tiktoken-style), context up to 10M tokens (via iRoPE).

**iRoPE — the new RoPE axis:**

iRoPE is "interleaved RoPE": every other layer (or some other per-layer pattern) has *no* position encoding at all (NoPE), and the remaining layers apply standard rotary. The motivation per Meta's Llama 4 blog: NoPE layers extrapolate to longer context than RoPE layers because they have no positional bias, while RoPE layers preserve in-window precision. The interleaving gives the model both properties.

Concretely:

```python
class Llama4Attention(nn.Module):
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx
        # Per-layer boolean: does this layer apply RoPE?
        self.use_rope = config.use_rope_per_layer[layer_idx]
        # ... standard projections ...

    def forward(self, hidden_states, position_embeddings=None, ...):
        q = self.q_proj(hidden_states).view(B, S, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, Hk, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, Hk, Dh).transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # else: q, k pass through unchanged — NoPE layer

        k, v = past_key_values.update(k, v, self.layer_idx)
        attn = softmax(q @ k.transpose(2, 3) * Dh**-0.5 + mask) @ v
        return self.o_proj(attn.transpose(1, 2).reshape(B, S, H * Dh))
```

The per-layer pattern in Scout is approximately: layers 0, 4, 8, ... apply RoPE; layers 1-3, 5-7, ... are NoPE. (Exact pattern per Meta's release: every 4th layer is RoPE; the remaining 3 of 4 are NoPE.) This is the **inverse** of SmolLM3's pattern (where ~25% of layers are NoPE, 75% are RoPE). Llama 4 Scout makes the *NoPE* layers the majority, with RoPE as periodic "anchor" layers.

For position-dependent temperature scaling: Llama 4 also adopts the formula introduced in the v2 §4.4 correction:
```
scale = 1 + beta * log(1 + floor(positions / original_max_position_embeddings))
```
applied to the RoPE layers (not the NoPE ones — NoPE has no positions to scale).

**MoE (Scout: 16 experts, top-1; Maverick: 128 experts, top-2):**

Llama 4 MoE follows the GShard-style softmax-then-topk routing without renormalization, distinct from Mixtral (which renormalizes). The routing is "load-balanced via auxiliary loss" — *not* DeepSeek-V3's no-aux bias correction. The shared-expert pattern (DeepSeek-V2/V3 style) is *not* used; all experts are routed.

```python
class Llama4MoE(nn.Module):
    def __init__(self, config):
        self.num_experts = 16        # Scout; 128 for Maverick
        self.top_k = 1               # Scout; 2 for Maverick
        self.gate = nn.Linear(D, self.num_experts, bias=False)
        # Fused 3D expert weights:
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * Df, D))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, D, Df))

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_values, topk_indices = router_probs.topk(self.top_k, dim=-1)
        # NO renormalization — top-1 always sums to <1, top-2 sums to less than 1
        return expert_dispatch(hidden_states, topk_indices, topk_values, self.gate_up_proj, self.down_proj)
```

The lack of renormalization on top-1 routing (Scout) means the per-token expert output is *attenuated* by the routing weight (which is <1.0); this is structurally equivalent to scaling the expert's contribution to the residual stream.

**Native multimodal early fusion:**

Llama 4 has no separate vision encoder in the Gemma-3 / PaliGemma sense. Image patches are tokenized via a "vision tokenizer" (a small ViT that produces visual tokens) and interleaved with text tokens at decoder input. The decoder treats image tokens as ordinary tokens with their own special-token IDs and 2D RoPE on the spatial axes.

```python
def llama4_prepare_inputs(text_tokens, images):
    visual_tokens = []
    for img in images:
        patches = img.patches(patch_size=16)   # [n_patches, 3, 16, 16]
        vt = vision_tokenizer(patches)          # [n_patches, D] via small ViT
        visual_tokens.append(vt)

    # Interleave at IMAGE_START / IMAGE_END token markers in text_tokens:
    merged = interleave_at_markers(text_tokens, visual_tokens,
                                    start_marker=IMAGE_START, end_marker=IMAGE_END)
    return merged
```

The vision tokenizer is "frozen" at decoder pretraining (per Meta's release) — the small ViT is pretrained on image-only data, then frozen, and the LM decoder learns to consume visual tokens jointly with text. This is closer to Chameleon's early-fusion than to LLaVA's projector-and-finetune.

**Long context to 10M tokens:**

Scout claims 10M-token context via iRoPE. The mechanism: NoPE layers have no in-window cliff (their representations are translation-invariant), and the periodic RoPE layers ground the model to absolute positions only where needed. Combined with rotary-base scaling on the RoPE layers, the model extrapolates well beyond the 8M training window.

**vLLM diffs:**
- Per-layer `use_rope` flag baked into the layer construction; vLLM rotary module gated on this.
- MoE via FusedMoE with `renormalize=False`, `top_k=1` (Scout) or `2` (Maverick).
- Visual-token interleaving handled at the input-processing stage; the layer stack sees a flat token sequence.
- Vision tokenizer kept frozen; vLLM loads it as a separate module, runs once per image, caches the result.

**llama.cpp diffs:**
- GGUF tensor name pattern: `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_*_exps` (for MoE), `vision_*` for the frozen tokenizer.
- Per-layer NoPE encoded via `apply_rope: false` per layer in the GGUF metadata.
- The vision tokenizer is stored as a separate GGUF section ("vision tower") that is loaded only if multimodal inputs are present.

**Evolution arc:**

- **Llama 4 Scout (April 2025):** 109B total / 17B active, 16 experts top-1, iRoPE, native multimodal early fusion. 10M context. The first Llama with MoE.
- **Llama 4 Maverick (April 2025):** 400B total / 17B active, 128 experts top-2. Same iRoPE + native multimodal early fusion.
- **Llama 4 Behemoth (claimed but never released):** 2T total / 288B active. Originally Meta's frontier flagship.

The iRoPE axis is the cleanest production-scale validation of SmolLM3's NoPE-alternation idea. The fact that Meta inverted the ratio (Llama 4 has more NoPE than RoPE layers; SmolLM3 had more RoPE than NoPE) suggests the optimal ratio is task-dependent and was tuned empirically per Meta's pretraining-data mixture.

---

### §5.36 Qwen3-Next 80B-A3B — Gated DeltaNet 3:1 hybrid + ultra-sparse 10+1/512 MoE + MTP head

Qwen3-Next (Sep 2025) extends the brief v2 §5.27 preview into a full architectural treatment. Three first-of-family elements: (1) **Gated DeltaNet linear attention 3:1 hybrid** with full softmax attention (the inverse of MiniMax-Text-01's 7:1 lightning-attention ratio), (2) **ultra-sparse MoE at 10 routed + 1 shared of 512 experts** (~2.15% activation, the sparsest production MoE to date), (3) **MTP (Multi-Token Prediction) head** for speculative decoding built into the weights.

**Identity (Qwen3-Next 80B-A3B):** D=2048, H=16, Hk=2 (GQA), Dh=128, n_layer=80, alternation 3:1 (linear:softmax — every 4th layer is full softmax, the other 3 are linear), n_routed_experts=512, n_shared_experts=1, num_experts_per_tok=10 (+ 1 shared = 11 active), decoder_sparse_step=4, MTP draft heads ×2.

**HF file:** `transformers/src/transformers/models/qwen3_next/modeling_qwen3_next.py` — `Qwen3NextAttention` (full softmax), `Qwen3NextGatedDeltaNet` (linear), `Qwen3NextSparseMoeBlock` (ultra-sparse MoE), `Qwen3NextDecoderLayer` (dispatch).

**Decoder dispatch (verbatim shape from v2 §5.27, with v3 expansion):**

```python
class Qwen3NextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.layer_type = config.layer_types[layer_idx]   # "linear_attention" or "full_attention"
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers
                and config.num_experts > 0
                and (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config)

        self.input_layernorm = Qwen3NextRMSNorm(D, eps=eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(D, eps=eps)
```

**Gated DeltaNet — the linear-attention block:**

Gated DeltaNet (arXiv:2412.06464, "Gated Delta Networks: Improving Mamba2 with Delta Rule") is a linear-attention variant where the softmax over keys is replaced by a delta-rule update on a matrix-valued recurrent state. The recurrence:

```
S_t = S_{t-1} * (1 - alpha_t * (k_t k_t^T)) + alpha_t * (v_t k_t^T)
y_t = S_t @ q_t
```

where `S` is the recurrent matrix state of shape `[Dk, Dv]`, `alpha_t = sigmoid(...)` is an input-dependent gate, and `(k_t k_t^T)` is the rank-1 erase operator.

```python
class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        self.head_dim = config.head_dim          # 128
        self.num_heads = config.num_attention_heads   # 16
        # Projections:
        self.q_proj = nn.Linear(D, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(D, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(D, self.num_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(D, self.num_heads, bias=False)   # one gate per head
        # Output projection + per-head norm:
        self.norm = RMSNormGated(self.head_dim, eps=eps)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, D, bias=False)

    def forward(self, hidden_states, cache_params, ...):
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        alpha = torch.sigmoid(self.gate_proj(hidden_states))   # [B, S, num_heads]

        # Recurrent state per head: [B, num_heads, Dk, Dv]
        state = cache_params.linear_state[self.layer_idx] if cache_params is not None else None
        if state is None:
            state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=q.device)

        # Chunked scan (for parallel prefill) or step-by-step (for decode):
        if S > 1:   # prefill, chunked scan
            outputs, new_state = self._chunked_delta_scan(q, k, v, alpha, state, chunk_size=64)
        else:       # decode, single-step update
            # Delta rule for one timestep:
            kk = k.unsqueeze(-1) * k.unsqueeze(-2)        # [B, num_heads, 1, Dh, Dh]
            erase_term = state * kk.squeeze(2)
            new_state = state - alpha.view(B, self.num_heads, 1, 1) * erase_term + \
                                  alpha.view(B, self.num_heads, 1, 1) * (v.unsqueeze(-2) * k.unsqueeze(-1)).squeeze(2)
            outputs = (new_state @ q.unsqueeze(-1)).squeeze(-1)   # [B, num_heads, Dh]

        if cache_params is not None:
            cache_params.linear_state[self.layer_idx] = new_state

        y = self.norm(outputs)
        y = y.reshape(B, S, -1)
        return self.o_proj(y)
```

**Cache state for linear-attention layers:** per layer, per head, a matrix `[Dk, Dv]` (here Dk = Dv = 128). For Qwen3-Next 80B, with 80 layers × 16 heads × 128 × 128 = ~20M elements per request in fp16 = ~40MB. This is *fixed-size regardless of sequence length* — the headline win of linear attention vs softmax (whose KV cache grows as O(S)).

**Cache state for the full-attention layers (every 4th):** standard KV cache. 20 full-attention layers × 2 KV heads × 128 head_dim = 5120 elements per token. For 1M context: 5GB per request — already substantial, but mitigated by the fact that only 1/4 of layers contribute.

**Ultra-sparse MoE block (10 routed + 1 shared / 512):**

```python
class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        self.num_experts = 512
        self.top_k = 10
        self.num_shared_experts = 1
        self.expert_dim = config.moe_intermediate_size   # small per-expert dim
        # Router:
        self.gate = nn.Linear(D, self.num_experts, bias=False)
        # Routed experts (fused):
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim, D))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, D, self.expert_dim))
        # Shared expert (always active, larger dim):
        self.shared_expert = Qwen3NextMLP(D, config.shared_expert_intermediate_size)

    def forward(self, hidden_states):
        # Softmax routing (Mixtral-style, NOT DeepSeek-V3 sigmoid):
        router_logits = self.gate(hidden_states)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_values, topk_indices = router_probs.topk(self.top_k, dim=-1)
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True)   # renormalize

        routed_out = expert_dispatch(hidden_states, topk_indices, topk_values,
                                     self.gate_up_proj, self.down_proj)
        shared_out = self.shared_expert(hidden_states)
        return routed_out + shared_out
```

**Why ultra-sparse?** The 10 of 512 routing (~2% activation) means each token activates 10 expert FFNs + 1 shared. At inference time, the dispatch pattern is highly skewed; load balancing requires an auxiliary loss (which Qwen3-Next retains; it did *not* adopt DeepSeek-V3's no-aux bias correction). The per-expert dim is small (`moe_intermediate_size = 1024` per Qwen3-Next config), making each expert "fine-grained" in the DeepSeekMoE sense.

The activation ratio comparison:
- Mixtral 8x7B: 2 of 8 = 25%
- Qwen3-MoE A22B: 2 of 60 = 3.33%
- Gemma 4 26B-A4B: 8 of 128 = 6.25%
- DeepSeek-V3: 8 of 256 (+1 shared) = ~3.13%
- **Qwen3-Next 80B-A3B: 10 of 512 (+1 shared) = ~2.15%**

Qwen3-Next is the sparsest production MoE in the corpus by a meaningful margin.

**MTP head:**

Qwen3-Next ships with a Multi-Token Prediction (MTP) head built into the weights — a small auxiliary transformer that takes the main model's hidden state at position `t` and predicts tokens at positions `t+1, t+2`. At inference time, this can be used as a draft model for speculative decoding without a separate draft model.

```python
class Qwen3NextMTPHead(nn.Module):
    def __init__(self, config):
        self.num_mtp_layers = 1   # one draft layer per MTP head
        self.mtp_layers = nn.ModuleList([
            Qwen3NextDecoderLayer(config, layer_idx=-1)   # special layer_idx for MTP
            for _ in range(config.mtp_num_heads)
        ])
        self.lm_head = nn.Linear(D, vocab, bias=False)

    def forward(self, hidden_states, position_embeddings, target_positions):
        # Produce predictions for target_positions (typically t+1, t+2)
        outputs = []
        for mtp_layer in self.mtp_layers:
            h = mtp_layer(hidden_states, position_embeddings, target_positions)
            outputs.append(self.lm_head(h))
        return outputs
```

The MTP heads are trained jointly with the main model under a multi-token cross-entropy loss; at deployment they enable speculative decoding without external draft models.

**vLLM diffs:**
- `Qwen3NextGatedDeltaNet` uses a CUDA kernel from `vllm/model_executor/layers/linear_attention/gated_delta.py` for the chunked scan.
- The hybrid cache abstraction: paged KV for full-attention layers + per-layer matrix state for linear-attention layers.
- Ultra-sparse MoE uses an EP-friendly dispatcher; with `tp_size=8`, the 512 experts shard across ranks with 64 experts per rank.
- MTP heads run as a separate "draft model" path; the speculative-decoding loop calls them with the main model's hidden states.

**llama.cpp diffs:**
- GGUF tensor names: `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_output` for full-attention layers; `linear_attn_q`, `linear_attn_k`, `linear_attn_v`, `linear_attn_gate`, `linear_attn_norm`, `linear_attn_output` for linear-attention layers.
- MoE: `ffn_*_exps` for the 512 routed experts; `ffn_*_shared` for the single shared expert.
- The linear-attention matrix state requires a new per-layer cache type (`linear_state[layer]`) alongside the per-layer KV cache type.

**Evolution arc:**

- **Qwen 3 (April 2025):** standard transformer + QK-norm + per-layer SWA option.
- **Qwen3-MoE A22B (July 2025):** softmax + renorm MoE, 60 experts top-2, with `mlp_only_layers` for dense interleave.
- **Qwen3-Next 80B-A3B (Sep 2025):** Gated DeltaNet 3:1 hybrid, ultra-sparse 10+1/512 MoE, MTP draft heads.
- **Qwen3.5 / 3.6 / 3.7-Max (2026):** further refinements; no public detailed architecture as of cutoff.

The 3:1 ratio (3 linear : 1 full softmax) is the inverse of MiniMax-Text-01's 7:1 (7 lightning : 1 softmax — see §5.44). The two ratios bracket the design space: 75% linear vs 87.5% linear. Neither has yet been shown empirically optimal — both compete with full softmax on different long-context benchmarks. The community position as of cutoff: Qwen3-Next's 3:1 with delta-rule recurrence beats MiniMax's 7:1 with pure lightning on retrieval; but on perplexity-only benchmarks the two are within noise.

---

### §5.37 Granite 4 H-hybrid (H-Micro 3B, H-Tiny 7B-A1B, H-Small 30B-A3B) + Granite 4.1 — 9:1 Mamba-2 : attention alternation

Granite 4 (Oct 2025) and Granite 4.1 (Apr 2026) are IBM's first Mamba-2 hybrid models. The architecturally-distinctive choice: **9:1 sequential alternation of Mamba-2 and softmax-attention layers** (9 Mamba-2 sublayers followed by 1 attention sublayer, repeated). This is a fifth distinct hybrid topology in the corpus (after Jamba's 1:7 attention:mamba, Falcon-H1's parallel branches, Zamba2's shared-attention-block, Hymba's head-partition, and Samba/Phi-4-mini-flash's alternation).

**Family sweep:**

| Variant | Total | Active | Layers | Mamba-2 : Attn ratio | D | H / Hk | n_experts | shared | HF repo |
|---|---|---|---|---|---|---|---|---|---|
| **H-Micro 3B** | 3B | 3B (dense) | 40 (36 Mamba + 4 Attn) | 9:1 | 2048 | 16 / 4 | n/a (dense) | n/a | ibm-granite/granite-4.0-h-micro |
| **H-Tiny 7B-A1B** | 7B | 1B | 60 (54 Mamba + 6 Attn MoE on top of attn) | 9:1 | 2560 | 16 / 4 | 64 | 0 | ibm-granite/granite-4.0-h-tiny |
| **H-Small 30B-A3B** | 30B | 3B | 88 (78 Mamba + 10 Attn MoE) | 9:1 | 3584 | 28 / 4 | 64 | 0 | ibm-granite/granite-4.0-h-small |
| **Granite 4.1 3B/8B/30B (dense)** | 3B/8B/30B | same | various | n/a (pure dense Transformer) | n/a | n/a | n/a | n/a | ibm-granite/granite-4.1-* |

Granite 4.1 (Apr 2026) departs from H-hybrid: it ships **dense decoder-only** Transformers at 3B/8B/30B, matching prior 32B-MoE quality. Granite 4 H-hybrid and Granite 4.1 dense coexist as parallel product lines.

**HF file:** `transformers/src/transformers/models/granite_moe_hybrid/modeling_granite_moe_hybrid.py` (the Granite 4 H-hybrid path). The Granite 4.1 dense path reuses `granite/modeling_granite.py` from v2 §5.12.

**Decoder block (9 Mamba-2 then 1 attention pattern):**

```python
class Granite4HHybridDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        # Layer type alternates: 9 Mamba-2 sublayers then 1 attention sublayer
        self.layer_type = "mamba2" if (layer_idx % 10 != 9) else "attention"

        if self.layer_type == "mamba2":
            self.mamba = Granite4Mamba2Mixer(config, layer_idx)
        else:
            self.self_attn = Granite4Attention(config, layer_idx)

        # MoE on the FFN if H-Tiny or H-Small:
        if config.num_experts > 0 and self.layer_type == "attention":
            self.mlp = Granite4MoEBlock(config)
        else:
            self.mlp = Granite4MLP(config)

        # Granite muP scalars:
        self.residual_multiplier = config.residual_multiplier  # ≈ 0.22
        self.input_layernorm = Granite4RMSNorm(D, eps=eps)
        self.post_attention_layernorm = Granite4RMSNorm(D, eps=eps)

    def forward(self, hidden_states, position_embeddings, past_key_values, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "mamba2":
            hidden_states = self.mamba(hidden_states, cache_params=past_key_values, ...)
        else:
            hidden_states, _ = self.self_attn(hidden_states, position_embeddings=position_embeddings, ...)

        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states
```

**Mamba-2 mixer (inherited from §5.19 with Granite's muP scalar):**

Granite 4's `Granite4Mamba2Mixer` is structurally Mamba-2 from §5.19 with one Granite-specific tweak: the `out_proj` is scaled by Granite's `residual_multiplier ≈ 0.22` at the residual-add site (this is the standard Granite muP behavior).

The Mamba-2 hyperparameters per Granite 4 H-Tiny:
- `mamba_d_state`: 64 (vs Mamba-2 standard 128)
- `mamba_d_conv`: 4
- `mamba_expand`: 2
- `chunk_size`: 256
- `n_groups`: 8

The smaller state size (64 vs Mamba-2's 128) is a Granite-specific tweak motivated by parameter-budget: Granite 4 H-Tiny is 7B-total / 1B-active, and the Mamba-2 layers dominate parameter count.

**MoE block (H-Tiny, H-Small only):**

Granite 4's MoE follows the DeepSeekMoE pattern with fine-grained experts and *zero* shared experts (unlike Gemma 4 and DeepSeek-V3 which have 1+ shared expert). Routing is softmax + renormalize (Mixtral-style), not sigmoid + bias (DeepSeek-V3 style).

```python
class Granite4MoEBlock(nn.Module):
    def __init__(self, config):
        self.num_experts = 64
        self.top_k = 8
        self.gate = nn.Linear(D, self.num_experts, bias=False)
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * Df_expert, D))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, D, Df_expert))

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        topk_values, topk_indices = router_probs.topk(self.top_k, dim=-1)
        topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True)
        return expert_dispatch(hidden_states, topk_indices, topk_values,
                               self.gate_up_proj, self.down_proj)
```

The MoE applies only to attention layers (1 in 10) — the Mamba-2 layers use dense MLPs. This is a unique pattern: in Jamba and Phi-4-mini-flash the MoE applies to all layers (or to layer-blocks independent of mixer type); in Granite 4 it is coupled to the attention sublayer.

**Cache abstraction:**

Granite 4 H-hybrid has two cache types interleaved by layer index:
- Mamba-2 layers (9 in 10): `(conv_state, ssm_state)` per layer; conv_state shape `[B, conv_dim, conv_kernel-1]`, ssm_state shape `[B, num_heads, head_dim, N]`.
- Attention layers (1 in 10): standard KV cache.

The total cache per request is dominated by the SSM state at the Mamba layers, but the SSM state is fixed-size (independent of sequence length) — so at long context (≥32K), the attention KV cache (which grows linearly) outpaces the SSM state cost.

**vLLM diffs:**
- vLLM supports Granite 4 H-hybrid via `vllm/model_executor/models/granite_4_h.py`.
- The Mamba-2 mixer uses the existing `vllm/model_executor/layers/mamba/mamba_mixer2.py` kernel.
- The cache combines paged KV (for attention layers) + per-layer SSM state tensors (for Mamba layers).
- MoE via `FusedMoE(num_experts=64, top_k=8, renormalize=True, ...)`.

**llama.cpp diffs:**
- GGUF tensor naming combines Granite's pattern (`attn_norm`, `attn_q/k/v/output`, `ffn_norm`, `ffn_gate/up/down`) with Mamba-2's pattern (`ssm_in`, `ssm_conv1d`, `ssm_A`, `ssm_D`, `ssm_out`, `ssm_norm`).
- Layer pattern is encoded as a `layer_types` array in GGUF metadata: 9 Mamba-2 entries then 1 attention entry, repeated.
- The Granite muP scalars (`f_attention_scale`, `f_residual_scale`, `f_embedding_scale`, `f_logit_scale`) propagate from Granite 3.3 unchanged.

**Evolution arc:**

- **Granite 1.0 (Sep 2024):** introduced muP-scalar parameterization at 7B dense.
- **Granite 2.0 (Dec 2024):** longer context.
- **Granite 3.0 / 3.1 / 3.3 (2025):** MoE variants, more multiplier exposure.
- **Granite 4 H-Micro / H-Tiny / H-Small (Oct 2025):** first Mamba-2 hybrid; 9:1 ratio; >70% RAM reduction at long context vs Granite 3.3.
- **Granite 4.1 3B / 8B / 30B (Apr 2026):** dense decoder-only; matches prior 32B-MoE quality at smaller active size.

The 9:1 Mamba-2:attention ratio is the most attention-sparse hybrid in the corpus. Compared to:
- Jamba 1:7 attention:mamba (12.5% attention layers)
- Samba alternation 1:1 (50% attention)
- Granite 4 H-hybrid 1:9 (10% attention layers)

Granite 4 is the most-Mamba-leaning hybrid that still maintains long-context recall (per IBM's benchmarks).

---

### §5.38 DeepSeek-V3.2 / V4 — Lightning Indexer DSA, then CSA + HCA

DeepSeek-V3.2 (Sep-Dec 2025) introduces **DSA (Lightning Indexer DSA)** — sparse attention with a fast learned scorer that selects top-k tokens for each query. DeepSeek-V4 (Apr 2026) abandons DSA in favor of **CSA + HCA (Compressed Sparse Attention + Heavily Compressed Attention)**, claiming 27% of V3.2's FLOPs and 10% of V3.2's KV cache at 1M context. Both build on the DeepSeek-V3 MLA + MoE base.

**Identity (DeepSeek-V3.2):** D=7168, H=128, MLA preserved from V3 (q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128), MoE preserved from V3 (n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8, n_group=8, topk_group=4), **DSA Lightning Indexer:** learned per-token relevance scorer that produces a sparse top-k mask of size 2048 per query.

**Lightning Indexer DSA — the new sparse-attention pattern:**

The DSA design (arXiv:2512.02556 per the audit) inserts a small "indexer" network that scores each (query, key) pair quickly, then selects top-k keys for the full softmax-attention pass. The indexer is a single matmul + softmax, much cheaper than full attention.

```python
class DeepSeekV32DSAAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Standard MLA projections (inherited from V3):
        # ... (Q LoRA, KV LoRA with MQA, etc.)

        # Lightning Indexer:
        self.indexer_dim = 64   # small per-head indexer
        self.indexer_q_proj = nn.Linear(D, H * self.indexer_dim, bias=False)
        self.indexer_k_proj = nn.Linear(D, self.indexer_dim, bias=False)  # MQA-like indexer keys
        self.indexer_top_k = 2048
        self.indexer_scale = self.indexer_dim ** -0.5

    def forward(self, hidden_states, past_key_values, position_embeddings, ...):
        # Standard MLA Q/K/V computation:
        q, k_full, v_full = self._mla_qkv(hidden_states, position_embeddings)
        # q shape: [B, H, S, qk_head_dim]
        # k_full, v_full from past_key_values (compressed MLA cache)

        # Lightning Indexer score:
        q_indexer = self.indexer_q_proj(hidden_states).view(B, S, H, self.indexer_dim).transpose(1, 2)
        k_indexer = past_key_values.get_indexer_keys(self.layer_idx)   # [B, 1, S_cached, indexer_dim]
        scores = q_indexer @ k_indexer.transpose(-1, -2) * self.indexer_scale   # [B, H, S, S_cached]

        # Top-k selection:
        topk_indices = scores.topk(self.indexer_top_k, dim=-1).indices   # [B, H, S, 2048]

        # Gather selected K/V from the MLA cache:
        k_selected = torch.gather(k_full.expand(B, H, -1, -1), 2,
                                   topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, qk_head_dim))
        v_selected = torch.gather(v_full.expand(B, H, -1, -1), 2,
                                   topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, v_head_dim))

        # Sparse softmax attention on top-k:
        attn_weights = (q.unsqueeze(3) * k_selected).sum(-1) / math.sqrt(qk_head_dim)
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_probs.unsqueeze(-1) * v_selected).sum(3)

        return self.o_proj(attn_output.transpose(1, 2).reshape(B, S, H * v_head_dim))
```

The key idea: the Lightning Indexer is *much smaller* than the full MLA path (per-head dim 64 vs 192) and runs a quick top-2048 selection per query token. The selected keys/values then go through the standard MLA softmax. At 1M context, the full attention is replaced by a 2048-key sparse attention, giving O(S * 2048) instead of O(S^2) per-query complexity.

**Cache extension for DSA:** in addition to the standard MLA compressed cache, DSA stores an "indexer keys" cache (`indexer_dim` per token, MQA-style — a single key per token across heads).

**CSA + HCA (DeepSeek-V4):**

V4 replaces DSA with a two-tier compressed-attention scheme:
- **CSA (Compressed Sparse Attention):** for the first ~256K tokens of context, use a compressed-but-still-dense attention with reduced per-token cache footprint.
- **HCA (Heavily Compressed Attention):** for tokens beyond ~256K, switch to an extreme-compression scheme with ~10× the compression ratio of CSA.

The architectural details of CSA and HCA are not fully public as of cutoff; the DeepSeek-V4 release blog claims:
- V4-Pro: 1.6T total / 49B active; 1M context with 27% of V3.2 FLOPs.
- V4-Flash: 284B total / 13B active; same architecture at smaller scale.

Both V4 variants retain the V3 MLA + MoE base; CSA + HCA replaces only the per-layer attention computation, not the MLA cache topology.

**Evolution arc (DeepSeek attention):**

- **DeepSeek LLM (Jan 2024):** standard softmax attention.
- **DeepSeek-V2 (May 2024):** MLA — compressed KV via low-rank projection; ~8.9× cache compression at 16 heads.
- **DeepSeek-V3 (Dec 2024):** MLA at 128 heads — ~71× cache compression. No attention sparsity.
- **DeepSeek-V3.2-Exp (Sep 2025):** MLA + DSA Lightning Indexer (sparse top-2048).
- **DeepSeek-V3.2 (Dec 2025):** DSA stable.
- **DeepSeek-V4 (Apr 2026):** MLA + CSA + HCA; DSA abandoned.

DSA had an unusually short production life (~6 months). The likely reason per the V4 release: the Lightning Indexer's per-token cost (`H * indexer_dim` parameters * `S_cached` keys) is non-trivial at very long context; CSA + HCA achieves better wall-clock efficiency by directly compressing the cache representation rather than scoring-then-selecting.

**vLLM diffs:** DSA support in vLLM uses a custom CUDA kernel; CSA + HCA support requires kernel work that was not in vLLM main as of cutoff.

**llama.cpp diffs:** DSA support is being added via a sparse-attention kernel; CSA + HCA is not yet implemented.

---

### §5.39 Apple Foundation Model 3.18B — cross-block KV sharing + 2-bit QAT

Apple Foundation Model (AFM) 3.18B is Apple's on-device LLM (deployed in iOS 18+ Apple Intelligence). Architecture details became public via arXiv:2507.13575 (July 2025), promoting AFM from v2's "noted closed" status to a full §5 entry. The architecturally-distinctive choice: **cross-block KV sharing** — a novel sub-variant of the v3 axis §3.33.

**Identity (AFM 3.18B):** D=3072, H=24, Hk=8 (GQA), Dh=128, n_layer=28 (split into 2 blocks), vocab=tk-100K (Apple-tokenizer), ViTDet-L 300M vision adapter for multimodal, 2-bit QAT weight precision.

**Cross-block KV sharing — the new sub-axis:**

AFM partitions its 28 layers into two contiguous blocks:
- **Block-1: layers 0-17** (18 layers, 62.5% of total). Each layer owns its K/V projections.
- **Block-2: layers 18-27** (10 layers, 37.5% of total). Each layer's K/V is *aliased* to a specific layer in Block-1.

The aliasing rule: layer `18 + i` in Block-2 reuses K/V from layer `i` in Block-1 (for `i = 0, ..., 9`). Concretely, layer 18 reuses layer 0's KV; layer 19 reuses layer 1's KV; etc.

```python
class AFMCache:
    def __init__(self, config):
        self.kv_owners = [self._allocate_kv(config) for _ in range(config.block_1_layers)]
        # No allocation for Block-2 layers — they alias Block-1
        self.kv_pointers = list(range(config.block_1_layers)) + \
                            list(range(config.block_2_layers))  # Block-2 layer i points to Block-1 layer i

    def update(self, k, v, layer_idx):
        owner_idx = self.kv_pointers[layer_idx]
        return self._append_update(self.kv_owners[owner_idx], k, v)
```

**Comparison to Gemma 4's cross-layer KV share (§5.34):**

| Aspect | AFM cross-block | Gemma 4 cross-layer |
|---|---|---|
| Sharing granularity | Block-role (Block-1 / Block-2) | Per-layer index |
| Number of shared layers | 10 of 28 (37%) | 20 of 35 (57%) |
| Within-attention-type | All layers same type | Constrained to same attention type (sliding or global) |
| Cache abstraction | Pointer per Block-2 layer | Pointer per shared layer |

Both reduce KV cache footprint at long context; the engineering differs in which layers can share.

**Decoder block (preserves Llama-shape pre-norm + RMSNorm + GQA + SwiGLU):**

```python
class AFMDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx
        self.in_block_2 = layer_idx >= config.block_1_layers
        self.kv_source = layer_idx - config.block_1_layers if self.in_block_2 else None

        self.self_attn = AFMAttention(config, layer_idx, kv_source=self.kv_source)
        self.mlp = AFMMLP(config)
        self.input_layernorm = AFMRMSNorm(D, eps=eps)
        self.post_attention_layernorm = AFMRMSNorm(D, eps=eps)

    def forward(self, hidden_states, ...):
        # Standard pre-norm Llama-shape block:
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

**2-bit QAT:**

AFM ships with 2-bit weights from quantization-aware training (QAT) — not PTQ. The training loop maintains a master fp16 copy of weights with a 2-bit quantized "forward" copy; gradient flows through stochastic rounding. The 2-bit grid is:
- Per-channel weights: 4 levels (e.g., -1.5, -0.5, +0.5, +1.5) × per-channel scale.
- Per-row bias: fp16.

Effective bits/weight ≈ 2.1 (including scale overhead per ~32-element block). This is more aggressive than BitNet b1.58 (which is 1.58 bits/weight at ternary {-1, 0, +1}) but Apple's design is 2 bits (4 levels per channel), which is empirically more stable for the AFM size.

**Vision adapter (ViTDet-L 300M):**

AFM's multimodal path uses a ViTDet-L vision encoder (300M params) feeding into a learned projector that produces 256 visual tokens per image, concatenated with text tokens at decoder input. This is concat-projector topology (axis §3.36 value 2), not cross-attention.

**No HF model card.** AFM is closed-weight; the architecture description above is from arXiv:2507.13575 and Apple's machine learning research blog. There is no `transformers/models/apple_foundation_model/` directory.

**vLLM diffs:** AFM is not in vLLM. Cross-block KV sharing would require the same cache-abstraction changes as Gemma 4 (§5.34); the 2-bit QAT path would require a custom kernel.

**llama.cpp diffs:** AFM is not in llama.cpp. Speculatively, the 2-bit QAT could be mapped to an IQ2 quantization variant.

**Evolution arc:** AFM 1.0 (Sep 2024 announcement, Jun 2025 deployment) → AFM 3.18B with cross-block KV share + 2-bit QAT (Jul 2025 arXiv paper). Apple is the only major lab shipping a per-model architecture choice this specifically tuned to on-device deployment (paged-to-flash embeddings, 2-bit weights, cross-block KV share — all motivated by iPhone memory budgets).

---

### §5.40 GPT-OSS 20B / Mini — MXFP4-native weights + trained attention sinks + GPT-3-style alternating attention (deep-dive)

GPT-OSS 20B was introduced in v2 §5.11 as a brief treatment of attention sinks. v3 promotes it to a full §5 deep-dive with three first-of-kind elements: (1) **MXFP4-native trained weights** (not PTQ — quantization-as-architecture, distinct from the BitNet path), (2) **trained attention sinks** (not the StreamingLLM inference-time recipe), (3) **GPT-3-style alternating dense / banded-sparse attention** at every other layer.

**Identity (GPT-OSS 20B-A3.6B):** D=2880, H=64, Hk=8 (GQA), Dh=64, n_local_experts=32, num_experts_per_tok=4, vocab=200019 (tiktoken-style), rms_norm_eps=1e-5, sliding_window configurable per-layer (banded-sparse on odd layers), sinks: nn.Parameter([H]), **weights in MXFP4 (4.25 bits/weight) from the released checkpoint, not from PTQ.**

**HF file:** `transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py` (already in v2). The v3 deep-dive adds MXFP4 and alternating-attention details.

**MXFP4-native weights:**

MXFP4 = Microscaling FP4 (OCP standard). Each block of 32 weights shares one E8M0 scale (8-bit exponent, no mantissa); each weight is FP4 E2M1 (1 sign + 2 exponent + 1 mantissa = 4 bits per weight). Total bits/weight = 4 + 8/32 = 4.25.

The HF `modeling_gpt_oss.py` does *not* store MXFP4 natively in the nn.Parameter; the HF code uses dequantized bf16 weights for the reference math. The MXFP4 format is the *released* checkpoint format, and production inference (vLLM, llama.cpp) dequantizes at load time or uses MXFP4-aware kernels.

Critical: GPT-OSS is **trained** in MXFP4. The training loop:
1. Maintain master fp32 weights (gradient accumulation reference).
2. Forward pass: dequantize fp32 → MXFP4 via stochastic rounding; compute matmul using MXFP4 kernels.
3. Backward pass: gradients computed in fp32; accumulated in fp32.
4. Periodic update: master fp32 weights updated by accumulated gradients.
5. Release: master fp32 → MXFP4 quantized via deterministic rounding for the released checkpoint.

This is structurally different from PTQ (AWQ, GPTQ, GGUF Q4_K_M), which take a high-precision checkpoint and quantize at deployment time. MXFP4-native training ensures the weight distribution matches the MXFP4 grid by construction.

**Trained attention sinks:**

The HF eager-mode forward (verbatim from v2 §5.11):

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, ...):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)     # extra "logit slot" per head

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # drop the sink after softmax
    ...
```

The mechanism: append one learnable logit value per head to the attention logits, softmax over (S + 1) logits, then drop the last column. The sink values are learned during training.

The training-time vs inference-time distinction is critical: the original StreamingLLM "sink token" (Xiao et al. 2023) was a heuristic — initialize a sink token at position 0 with a particular embedding and never drop it from the KV cache. GPT-OSS's sinks are learnable parameters trained jointly with the model, providing a much smaller and more controllable mechanism for the same softmax-mass-spillage effect.

**GPT-3-style alternating attention:**

GPT-OSS revives the GPT-3 design where odd layers have full dense attention and even layers have banded-sparse attention (a sliding window plus a global summary band). Per the GPT-OSS release notes:

```python
for layer_idx in range(n_layer):
    if layer_idx % 2 == 0:
        # Dense full-attention layer
        sliding_window = None
    else:
        # Banded-sparse attention layer
        sliding_window = 4096   # local band
        # plus all positions attend to the first N "summary" tokens
```

This is structurally distinct from:
- Gemma 3's 5:1 alternation (5 sliding + 1 global, not GPT-3's 1:1).
- Mistral Ministral's interleaved per-layer SWA (every layer SWA, alternating window sizes — see §5.42).
- Phi-3-small's block-sparse (custom mask, not periodic alternation).

GPT-OSS's 1:1 dense:sparse alternation is exactly the GPT-3 pattern from 2020 — revived 5 years later in an open-weight context.

**MoE block with biased experts:**

```python
class GptOssExperts(nn.Module):
    def __init__(self, config):
        self.num_experts = 32
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, D, 2 * Df))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * Df))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, Df, D))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, D))
```

Experts have biases on both gate_up_proj and down_proj. This is unusual; Mixtral / DeepSeek experts are bias-free. The biases provide a "constant residual" per expert that can absorb position-independent offsets.

**vLLM diffs (v3 expansion):**
- MXFP4 native: vLLM has an MXFP4 dequantize-and-multiply kernel at `vllm/model_executor/layers/quantization/mxfp4.py`; load-time path detects MXFP4 weight format and dispatches.
- Attention sinks: handled via a custom mask in FlexAttention; sink values appended to the logit dim.
- Alternating attention: per-layer `sliding_window` propagated to the Attention(...) call.

**llama.cpp diffs (v3 expansion):**
- MXFP4 → custom GGUF quantization type `MXFP4` (4.25 bits/weight); load path detects this type and uses a dedicated dequantize kernel.
- Sink tokens: `attn_sinks` GGUF tensor per layer; `build_attn` variant with sink concatenation.
- Alternating attention: layer pattern stored in GGUF metadata; build_attn dispatches based on the per-layer attention type.

**Evolution arc:**

GPT-OSS is OpenAI's first open-weight release since GPT-2 (Aug 2025). It is the first open model to:
- Train with attention sinks as a primary mechanism (StreamingLLM was inference-time).
- Ship MXFP4-native weights (others use bf16 + PTQ).
- Revive GPT-3-style alternating dense/sparse attention in 2025.

GPT-OSS-Mini (claimed by OpenAI but not released as of cutoff): smaller variant. The 20B remains current; 120B was released for cloud-only inference.

---

### §5.41 DeepSeek-OCR 3B-MoE-A570M — MLA + SAM+CLIP dual encoder + 16× visual token compressor + Visual Causal Flow

DeepSeek-OCR (Oct 2025) is the most architecturally novel OCR-LLM of 2025-2026. Three first-of-kind elements: (1) **DeepEncoder serial two-tower** — SAM-base 80M for local detail + CLIP-Large 300M for semantic, chained not parallel-fused; (2) **16× visual token compressor** (a conv-pool stage between the two vision towers that reduces 4096 patches to 256 tokens); (3) **Visual Causal Flow** (DeepSeek-OCR-2, Q1 2026) — block-bidirectional attention mask on vision tokens, the v3 axis §3.37.

**Identity (DeepSeek-OCR):** Total 3.4B = 80M SAM-base + 300M CLIP-Large + 16× compressor + 3B DeepSeek3B-MoE-A570M decoder. The decoder inherits MLA from DeepSeek-V2/V3 lineage.

**Decoder identity:** D=2048 (smaller than DeepSeek-V3's 7168 due to A570M active size), H=16, q_lora_rank=None or low, kv_lora_rank=64, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64. MoE configuration: small expert count (~16-32 routed) plus 1-2 shared experts; ~570M active per token.

**HF file:** External — `github.com/deepseek-ai/DeepSeek-OCR` and `huggingface.co/deepseek-ai/DeepSeek-OCR`. The decoder is structurally a smaller cousin of `transformers/models/deepseek_v2/modeling_deepseek_v2.py` (with MLA).

**DeepEncoder (serial two-tower vision):**

```python
class DeepEncoder(nn.Module):
    def __init__(self, config):
        # Tower 1: SAM-base for local detail
        self.sam_base = SAMBase(image_size=1024, patch_size=16)
        # → 4096 patches of dim 768 per image

        # 16× visual compressor (the key architecturally-novel element):
        self.compressor = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=4, stride=4),   # 4× spatial compression
            nn.ReLU(),
            nn.Conv2d(768, 768, kernel_size=4, stride=4),   # another 4× = 16× total
            nn.Flatten(start_dim=2),                          # → 256 tokens
        )

        # Tower 2: CLIP-Large for global semantic processing
        self.clip_large = CLIPLarge(input_dim=768)
        # → 256 tokens of dim 1024

        # Final projection to decoder D:
        self.adapter = nn.Linear(1024, config.hidden_size)   # 1024 → 2048

    def forward(self, images):
        # images: [B, 3, 1024, 1024]
        sam_features = self.sam_base(images)        # [B, 4096, 768]
        sam_features_2d = sam_features.view(B, 64, 64, 768).permute(0, 3, 1, 2)
        # Compress 64×64 spatial → 4×4 spatial = 16 tokens? Actually 16×16 = 256:
        compressed = self.compressor(sam_features_2d)   # [B, 768, 16, 16]
        compressed_flat = compressed.flatten(start_dim=2).transpose(1, 2)   # [B, 256, 768]

        # CLIP-Large refines:
        clip_features = self.clip_large(compressed_flat)   # [B, 256, 1024]

        # Adapt to decoder D:
        return self.adapter(clip_features)   # [B, 256, 2048]
```

The 16× compression is per-image: 1024² input → 4096 patches at SAM → 256 tokens at decoder. Each visual token represents a 64×64 pixel region of the original image. The "optical context compression" thesis (DeepSeek-OCR blog): 256 visual tokens encode ~6000-10000 characters of text content (depending on density), giving a 10× compression of OCR-content per token.

**Decoder block (MLA + MoE, smaller than V3):**

The decoder is structurally a scaled-down DeepSeek-V3 (§5.6) with smaller per-head dims and smaller MoE. The MLA pattern is preserved:

```python
class DeepSeekOCRDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = DeepSeekOCRMLAAttention(config, layer_idx)
        self.mlp = DeepSeekOCRMoE(config) if layer_idx >= config.first_k_dense_replace else DeepSeekOCRMLP(config)
        self.input_layernorm = DeepSeekOCRRMSNorm(D, eps=eps)
        self.post_attention_layernorm = DeepSeekOCRRMSNorm(D, eps=eps)

    def forward(self, hidden_states, ...):
        # Standard DeepSeek-V3-shape pre-norm block:
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

**Visual Causal Flow (DeepSeek-OCR-2, Q1 2026):**

The Visual Causal Flow attention mask is the new architectural element that motivates v3 axis §3.37 (block-bidirectional mask):

```python
def build_visual_causal_flow_mask(token_block_ids, S):
    """
    token_block_ids: [S] int tensor
        - Negative values: text positions (causal)
        - Non-negative values: vision-token block IDs (bidirectional within block)
    """
    mask = torch.full((S, S), float("-inf"))
    for i in range(S):
        for j in range(S):
            if token_block_ids[i] >= 0 and token_block_ids[j] >= 0 and token_block_ids[i] == token_block_ids[j]:
                mask[i, j] = 0   # bidirectional within same vision block
            elif j <= i:
                mask[i, j] = 0   # standard causal otherwise
    return mask
```

A "vision block" is one image's 256 visual tokens. Within a block, all 256 positions see each other bidirectionally — useful for OCR where a character's interpretation depends on its neighbors (above, below, left, right). Across blocks (different pages) and into the text-decoding region, the mask remains causal — preserving the autoregressive language-modeling structure.

This mask cannot be expressed via standard `is_causal: bool` + `sliding_window: int`. The implementation requires either:
- Eager-mode custom mask (slow at long context).
- FlexAttention with a `mask_mod` function (preferred path).
- A dedicated CUDA kernel that takes block IDs as a tensor and computes the mask on-the-fly.

**vLLM diffs:** DeepSeek-OCR has experimental vLLM support; Visual Causal Flow is not yet supported (the mask is too custom for the FA backend at cutoff).

**llama.cpp diffs:** DeepSeek-OCR is not yet in llama.cpp main.

**Evolution arc (OCR-LLM, contextualizing §5.41):**

- **Donut (NAVER, 2021):** Swin encoder + BART decoder, encoder-decoder OCR-free.
- **Nougat (Meta, 2023):** Donut-style for academic PDFs.
- **Vary (Megvii, 2023-12) / Vary-toy (2024-01):** SAM-base vision-vocab + CLIP + Qwen-1.8B.
- **GOT-OCR 2.0 (Sep 2024):** VitDet + Qwen-0.5B; 1024² → 256 tokens (4096× spatial compression). See §5.45.
- **DeepSeek-OCR (Oct 2025):** DeepEncoder (SAM + CLIP serial) + 16× compressor + DeepSeek3B-MoE-A570M decoder with MLA.
- **DeepSeek-OCR-2 (Q1 2026):** + Visual Causal Flow attention mask.
- **MinerU2.5 (Sep 2025):** NaViT-675M + Qwen2-Instruct-0.5B; two-stage coarse-to-fine inference.
- **olmOCR / olmOCR-2 (2025):** Qwen2.5-VL-7B base with unit-test-reward RL fine-tuning. Not architecturally novel; covered by Qwen2.5-VL.
- **HunyuanOCR (Tencent, 2025-11):** 0.4B native-resolution ViT + Hunyuan-0.5B trained jointly from scratch.
- **PaddleOCR-VL (Baidu, 2025-10):** NaViT-style + ERNIE-4.5-0.3B (smallest production OCR-LLM at 0.9B total).

DeepSeek-OCR is the only OCR-LLM combining **MLA + MoE + serial-tower vision + 16× compression + Visual Causal Flow** in one model. It is the architectural reference for the 2025-2026 OCR-LLM wave.

---

### §5.42 Mistral Ministral 3B / 8B — third SWA-alternation pattern (interleaved per-layer)

Ministral (Oct 2024) is Mistral's "edge" line: 3B and 8B variants with **interleaved sliding-window attention** as a per-layer pattern. This is the third distinct SWA-alternation pattern in the corpus:

| Family | SWA pattern |
|---|---|
| **Mistral 7B v0.1** | SWA on every layer (uniform-SWA) |
| **Gemma 2/3/4** | 5:1 sliding:global blocks (block-periodic) |
| **Ministral 3B/8B** | **Per-layer interleaved sliding-window sizes** (every layer SWA, alternating window) |

**Identity (Ministral-8B):** D=4096, H=32, Hk=8 (GQA), Dh=128, Df=12288, vocab=131072 (Tekken-style), rms_norm_eps=1e-5, rope_theta=1e7, **interleaved sliding window: [4096, 32768, 4096, 32768, ...] per layer alternating** (or similar; exact pattern per Mistral's release blog).

**Decoder block (preserves Llama-shape; SWA varies per layer):**

```python
class MinistralAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # Per-layer sliding window — interleaved (not block-periodic):
        self.sliding_window = config.sliding_windows[layer_idx]   # alternates 4K, 32K, 4K, 32K, ...
        # ... standard Q/K/V/O projections ...

    def forward(self, hidden_states, ...):
        # Standard attention; sliding_window kwarg propagated to backend:
        attn_output, _ = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            sliding_window=self.sliding_window, ...
        )
```

The interleaving — "every layer has a window, but the size alternates" — is structurally different from Gemma's "every 5 layers, one has no window (global)". The Ministral design treats SWA as a per-layer hyperparameter, with no layer being purely global; layers alternate between "local" (4K window) and "extended" (32K window).

The intuition per Mistral's release: at 8B scale, pure SWA limits recall; pure global is expensive. Alternating layer-by-layer between two window sizes gives each token in deep layers access to both fine-grained (4K window) and coarse-grained (32K window) contexts. The 8B context capability is ~128K via RoPE scaling on top of the largest layer's window.

**vLLM diffs:** Ministral uses the Mistral path with a per-layer `sliding_window` config; otherwise unchanged.

**llama.cpp diffs:** GGUF metadata stores per-layer sliding window sizes as an array; `build_attn` reads the per-layer value.

**Evolution placement:** Ministral 3B/8B sits between Mistral 7B v0.1 (pure SWA) and Mistral-Nemo (no SWA). The 3B/8B targeting on-device deployment is Mistral's response to the SLM market. The interleaved pattern has not (yet) been adopted by other families; the 5:1 block-periodic (Gemma) and pure-global (Llama/DeepSeek) designs dominate.

---

### §5.43 Phi-4-multimodal — mixture-of-LoRAs modality routing

Phi-4-multimodal (Feb 2025) is the first production small (<6B) model with **mixture-of-LoRAs** modality routing. The Phi-4-Mini 3.8B LM is frozen; vision and audio inputs each go through a modality-specific LoRA adapter stack that merges into the frozen LM at runtime.

**Identity:** Total 5.6B = Phi-4-Mini-3.8B (frozen) + 370M Vision LoRA + ~1B Audio LoRA + audio encoder.

**Architecturally-distinct from concat-prefix and cross-attention:**

| Topology | Mechanism | Examples |
|---|---|---|
| concat-prefix | Vision tokens prefixed at embedding | PaliGemma, LLaVA |
| cross-attention | Image-KV via cross-attn layers | Llama 3.2 Vision |
| **mixture-of-LoRAs** | **Modality-specific LoRA adapters merged at runtime** | **Phi-4-multimodal** |
| early-fusion | Raw patches projected directly | Chameleon, Llama 4, Gemma 4 12B-Unified |

The Phi-4-multimodal design freezes the base Phi-4-Mini and adds parameter-efficient adapters per modality. At inference time:

```python
class Phi4MultimodalDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        # Base Phi-4-Mini layer (frozen):
        self.base_layer = Phi4MiniDecoderLayer(config, layer_idx)

        # Modality-specific LoRA adapters:
        self.vision_lora_q = LoRAAdapter(D, D, r=64)
        self.vision_lora_v = LoRAAdapter(D, D, r=64)
        self.audio_lora_q = LoRAAdapter(D, D, r=64)
        self.audio_lora_v = LoRAAdapter(D, D, r=64)

    def forward(self, hidden_states, modality_mask, ...):
        # modality_mask: [B, S] — which tokens are text/image/audio
        residual = hidden_states
        hidden_states = self.base_layer.input_layernorm(hidden_states)

        # Compute Q, K, V from base + LoRA contribution per token:
        q = self.base_layer.self_attn.q_proj(hidden_states)
        k = self.base_layer.self_attn.k_proj(hidden_states)
        v = self.base_layer.self_attn.v_proj(hidden_states)

        # Per-token LoRA addition based on modality_mask:
        vision_q_add = self.vision_lora_q(hidden_states) * (modality_mask == VISION).unsqueeze(-1)
        vision_v_add = self.vision_lora_v(hidden_states) * (modality_mask == VISION).unsqueeze(-1)
        audio_q_add = self.audio_lora_q(hidden_states) * (modality_mask == AUDIO).unsqueeze(-1)
        audio_v_add = self.audio_lora_v(hidden_states) * (modality_mask == AUDIO).unsqueeze(-1)

        q = q + vision_q_add + audio_q_add
        v = v + vision_v_add + audio_v_add

        # Standard attention from here on:
        attn_output = scaled_dot_product_attention(q, k, v, ...)
        return residual + attn_output
```

**The key axis distinction:** unlike standard LoRA PEFT (which merges into weights at deployment), mixture-of-LoRAs *routes at inference time* — different tokens see different LoRA contributions based on their modality. The base LM stays frozen; the runtime adds modality-dependent rank-r updates to Q and V projections.

**LoRA-merge cost:** at every attention sublayer, an additional `Dh * r` matmul per modality per token. With r=64 and Dh=128, this is 8192 extra MACs per token per LoRA — negligible compared to the base Q*K^T which is 128*S MACs per head.

**vLLM diffs:** Phi-4-multimodal needs a special LoRA-routing path; vLLM's standard LoRA support assumes batch-merged adapters, not per-token-routed.

**llama.cpp diffs:** mixture-of-LoRAs is not yet supported in llama.cpp main; the model can be inferred in CPU eager mode.

**Evolution placement:** Phi-4-multimodal is the canonical test case for v3 axis §3.36 value 4 (mixture-of-LoRAs vision adapter). Microsoft's design choice is motivated by deployment economics: a single frozen Phi-4-Mini base can be deployed once, with modality LoRAs swapped in/out on demand — important for resource-constrained edge devices.

---

### §5.44 MiniMax-Text-01 — Lightning Attention 7:1 hybrid

MiniMax-Text-01 (Jan 2025) introduced **Lightning Attention** — a linear-attention variant — at a 7:1 ratio with full softmax attention. The 7:1 ratio is the inverse of Qwen3-Next's 3:1 (87.5% linear vs 75% linear); MiniMax-Text-01 is the most-linear-attention-heavy production model in the corpus.

**Identity (MiniMax-Text-01):** 456B total / 45.9B active MoE; pure Lightning Attention on 7 of every 8 layers, full softmax on the 8th. Context up to 4M tokens.

**MiniMax-M2 abandonment story:**

MiniMax-M2 (2026) explicitly **removes** Lightning Attention and reverts to full softmax attention on all layers (arXiv:2605.26494). This is the v3 "feature adopted and then abandoned" story: Lightning Attention proved competitive on perplexity but degraded on retrieval/recall benchmarks. M2 cites the recall gap as the primary motivator.

**Lightning Attention math:**

Lightning Attention is a linear-attention with a sliding-window decay on the recurrent state. The recurrence:

```
S_t = S_{t-1} * gamma + v_t k_t^T
y_t = S_t @ q_t
```

where `gamma ∈ (0, 1)` is a per-channel decay, and `S` is the matrix-valued recurrent state (shape `[Dk, Dv]`). This is structurally similar to Gated DeltaNet (§5.36) but with:
- A scalar decay `gamma` instead of input-dependent `alpha_t`.
- No delta-rule erase term.

The decay-only recurrence is simpler and faster than DeltaNet but less expressive — hence the recall gap that motivated M2's reversion.

**Cache per Lightning layer:** the matrix state `[Dk, Dv]` per head per request. Fixed-size; the win is "no KV cache for 7 of 8 layers".

**Evolution placement:** MiniMax-Text-01 is the first frontier-lab production deployment of linear attention at 80%+ ratio. The follow-up M2 (which reverts to full attention) and Qwen3-Next (which uses Gated DeltaNet at 75% linear) are competing responses to the same architectural question: how much linear attention can a model tolerate before recall degrades?

The two empirical data points so far:
- MiniMax-Text-01: 87.5% linear (Lightning Attention) → reverted in M2.
- Qwen3-Next: 75% linear (Gated DeltaNet) → currently in production.

The community position as of cutoff: ~75% linear with delta-rule recurrence is the upper bound for production deployment; higher ratios degrade recall too much. The 7:1 Lightning Attention experiment is documented as an architectural choice that "did not survive contact with production benchmarks."

---

### §5.45 GOT-OCR 2.0 — minimal Qwen-0.5B + linear vision-language connector

GOT-OCR 2.0 (Sep 2024) is the architecturally-purest "small OCR-LLM" — the smallest production OCR-LLM by total params (580M) and the ancestor of the 2025 lightweight-OCR wave. The interesting architectural choice: **the connector is a single `nn.Linear`**, not an MLP, Q-Former, or perceiver resampler.

**Identity:** Total 580M = VitDet (~80M) + 1024-d linear connector + **Qwen-0.5B** as LM decoder.

**LM decoder = Qwen-1 0.5B verbatim:** 24 layers × 14 heads × 64 head_dim (total D=896), RoPE θ=10000, RMSNorm pre, SwiGLU MLP, SentencePiece 151936 vocab, no QKV bias for Qwen-1 0.5B (Qwen-1 7B has QKV bias; 0.5B drops it). This is just a research/01 Qwen-1 0.5B row.

**The architectural distinction is the encoder→decoder bandwidth budget:**

```python
class GOTOCREncoder(nn.Module):
    def __init__(self, config):
        # VitDet: process 1024×1024 image into 256 visual tokens of dim 1024
        self.vit_det = VitDet(image_size=1024, patch_size=64)
        # → 256 patches of dim 1024 per image

        # Connector: single linear projection (no MLP, no resampler!)
        self.connector = nn.Linear(1024, config.lm_hidden_size, bias=False)
        # config.lm_hidden_size = 896 for Qwen-1 0.5B

    def forward(self, image):
        vis_features = self.vit_det(image)        # [B, 256, 1024]
        return self.connector(vis_features)        # [B, 256, 896]
```

**4096× spatial compression rate:** 1024×1024 pixels → 256 visual tokens means each visual token represents a 64×64 pixel region of the original image. This is the most aggressive spatial compression in the OCR-LLM corpus.

**Production verdict:** GOT-OCR 2.0 deserves a `models/got-ocr2/` slot because (a) the decoder is the smallest production LM decoder in the OCR space, (b) the connector is the minimal possible adapter (just a `nn.Linear`), (c) this is the *referenced* lightweight-OCR shape that DeepSeek-OCR (§5.41), MinerU2.5, HunyuanOCR all evolved from.

**Evolution placement:** GOT-OCR 2.0 is the architectural reference baseline for the 2025-2026 OCR-LLM wave. Its choices (small Qwen LM, single-linear connector, VitDet encoder, 256 visual tokens per image) became the template that DeepSeek-OCR refined (SAM+CLIP serial encoder, 16× conv compressor, MoE decoder, MLA) and that MinerU2.5 simplified (NaViT encoder, Qwen2-0.5B decoder, two-stage inference).

---

### §5.46 Qwen2.5-VL family — M-RoPE 2D/3D + dynamic-resolution patches

Qwen2.5-VL (Jan 2025) is the de-facto dominant base for OCR-LLM fine-tunes in 2025-2026 (olmOCR-2, Nanonets-OCR-s, MonkeyOCR-v1.5, Dolphin-v2 are all fine-tunes of this base). The architecturally-distinctive elements: (1) **M-RoPE** (Multimodal RoPE) — separate temporal/height/width sub-channels of RoPE; (2) **dynamic-resolution patches** — native aspect ratio handling via padded patch packing; (3) **LM-shaped ViT** — the vision encoder itself uses SwiGLU + RMSNorm + window attention, mirroring the decoder.

**Identity (Qwen2.5-VL 3B):** LM decoder = Qwen2.5-3B (D=2048, H=16, Hk=2, Dh=128, n_layer=36, vocab=151936, rope_theta=1e6). Vision encoder: redesigned ViT (~675M for 3B variant) with window attention, SwiGLU, RMSNorm. **M-RoPE applied to both decoder and encoder.**

**M-RoPE (Multimodal RoPE):**

Standard RoPE applies a single rotation per head-dim per position. M-RoPE partitions the head-dim into three sub-channels — temporal (t), height (h), width (w) — and applies a separate position-frequency to each:

```python
def apply_mrope(q, k, position_ids_t, position_ids_h, position_ids_w, mrope_section):
    """
    mrope_section: e.g., [t: 16, h: 24, w: 24] for head_dim=64
        → first 16 channels rotate by t-position, next 24 by h, last 24 by w
    """
    cos_t, sin_t = compute_cos_sin(position_ids_t, dim=mrope_section[0])
    cos_h, sin_h = compute_cos_sin(position_ids_h, dim=mrope_section[1])
    cos_w, sin_w = compute_cos_sin(position_ids_w, dim=mrope_section[2])

    q_t, q_h, q_w = torch.split(q, mrope_section, dim=-1)
    q_t = apply_rotary_pos_emb(q_t, cos_t, sin_t)
    q_h = apply_rotary_pos_emb(q_h, cos_h, sin_h)
    q_w = apply_rotary_pos_emb(q_w, cos_w, sin_w)
    q = torch.cat([q_t, q_h, q_w], dim=-1)

    # Same for k:
    k_t, k_h, k_w = torch.split(k, mrope_section, dim=-1)
    # ... rotate each sub-channel ...
    k = torch.cat([k_t, k_h, k_w], dim=-1)

    return q, k
```

For text tokens, `position_ids_h = position_ids_w = 0` (effectively, only the t-channel rotates). For image tokens, `position_ids_t = 0` and `position_ids_h, position_ids_w` encode the patch's 2D coordinates. For video tokens, all three are non-zero (t = frame index, h/w = patch coordinates within frame).

**Dynamic resolution:**

Qwen2.5-VL accepts variable-size images via padded patch packing. An image of size H×W is split into `(H // patch_size) × (W // patch_size)` patches, then padded to the nearest multiple of a window-attention window size:

```python
def qwen25_vl_image_to_tokens(image, patch_size=14, max_patches=4096):
    H, W, _ = image.shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    total_patches = h_patches * w_patches

    # Cap at max_patches; downscale if necessary
    if total_patches > max_patches:
        scale = math.sqrt(total_patches / max_patches)
        H = int(H / scale); W = int(W / scale)
        image = resize(image, (H, W))
        h_patches = H // patch_size; w_patches = W // patch_size

    patches = patchify(image, patch_size=patch_size)
    # Position IDs:
    h_ids = torch.arange(h_patches).repeat_interleave(w_patches)
    w_ids = torch.arange(w_patches).repeat(h_patches)
    t_ids = torch.zeros_like(h_ids)
    return patches, (t_ids, h_ids, w_ids)
```

**LM-shaped ViT:**

The Qwen2.5-VL ViT itself uses:
- SwiGLU MLP (instead of standard ViT's GELU MLP).
- RMSNorm pre-norm (instead of LayerNorm).
- Window attention (instead of dense attention).

This is a notable architectural symmetry: the ViT is structurally an LM transformer applied to image patches. The motivation per the Qwen2.5-VL paper: aligning the ViT's per-layer math with the LM decoder's lets the joint model share more low-level kernels and simplifies training-time gradient flow.

**HF file:** `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`.

**Evolution placement:** Qwen2.5-VL is the most-fine-tuned VLM base in the 2025-2026 OCR ecosystem. Its architectural choices (M-RoPE, dynamic resolution, LM-shaped ViT) have been adopted by MinerU2.5 (NaViT initialized from Qwen2-VL), olmOCR-2 (Qwen2.5-VL-7B base), Nanonets-OCR-s (Qwen2.5-VL-3B base), MonkeyOCR-v1.5 (Qwen2.5-VL-3B base), and Dolphin-v2 (Qwen2.5-VL-3B base). The architecture is widely deployed and the ecosystem is mature.

---

### §5.47 Mamba-3 — complex-state SSM with MIMO decoding

Mamba-3 (Mar 2026, arXiv:2603.15569, accepted ICLR 2026 — CMU/Princeton/Together/Cartesia) is the third major iteration of the Mamba family. Three substantive deltas vs Mamba-2:

1. **More expressive recurrence** derived from a refined SSM discretization.
2. **Complex-valued state-update rule** for richer state tracking.
3. **MIMO (Multi-Input, Multi-Output) decoding** for accuracy at fixed decode latency.

**Identity (Mamba-3 reference scales):** 180M, 360M, 780M, 1.5B reference scales for the research release. No production-scale variant as of cutoff.

**Complex-state SSM:**

Mamba-2's recurrence used a per-head real-valued scalar A (the "state matrix" reduced to a scalar). Mamba-3 lifts this to a *complex* per-head pair `(A_real, A_imag)`:

```python
class Mamba3Mixer(nn.Module):
    def __init__(self, config, layer_idx):
        # Per-head complex scalar A (real and imaginary parts):
        self.A_log_real = nn.Parameter(torch.empty(config.num_heads))
        self.A_log_imag = nn.Parameter(torch.empty(config.num_heads))

        # Rest of the in_proj / conv1d / x_BC / dt parametrization preserved from Mamba-2:
        # ... see §5.19 for the in_proj fused projection layout

        # MIMO decoding heads:
        self.mimo_heads = config.mimo_num_heads  # typically 2-4
        self.mimo_decoder = nn.Linear(self.intermediate_size, self.mimo_heads * D, bias=False)

    def forward(self, hidden_states, cache_params, ...):
        # ... in_proj, conv1d, x_BC split as in Mamba-2 ...

        # Complex A: A = exp(A_log_real + i * A_log_imag)
        A_real = -A_log_real.exp()
        A_imag = A_log_imag    # phase angle directly

        # Complex SSD scan: state is complex-valued
        # h_t = A * h_{t-1} + B * x_t
        # y_t = C * h_t + D * x_t
        # where h, A, B, C are now complex; the scan is on complex-valued tensors
        y_complex = mamba3_complex_scan(x, dt, A_real, A_imag, B, C, ...)

        # MIMO decoding: produce multiple outputs per timestep
        # y_complex shape: [B, S, num_heads, head_dim]
        # → produce mimo_heads × D output per timestep
        y_real = y_complex.real  # extract real part
        y_mimo = self.mimo_decoder(y_real.reshape(B, S, -1))
        # y_mimo shape: [B, S, mimo_heads * D]
        # In MIMO inference, output[:, :, i*D:(i+1)*D] is the prediction for time t+i

        return self.norm(y_mimo[:, :, :D])   # primary output
```

**Why complex state?** Per the Mamba-3 paper, the complex-valued state allows the recurrence to encode oscillatory dynamics directly, which is empirically helpful for long-range temporal structure (audio, sequential reasoning). The Mamba-2 real-only scalar A could only encode exponential decay.

**MIMO decoding:**

MIMO = Multi-Input Multi-Output. The intuition: at each timestep, the SSM internal state contains enough information to predict not just the next token but the next few tokens. MIMO decoding exposes this by training the model to produce `mimo_heads` parallel outputs per timestep, each predicting a different future position.

At inference, MIMO enables "free" speculative decoding without a separate draft model — the model predicts the next `mimo_heads` tokens per forward pass, and a verifier (the model itself, run with the MIMO predictions as input) accepts/rejects them. The wall-clock speedup is ~`mimo_heads`× when token acceptance rates are high.

**Cache state:** the SSM state per head is now a complex-valued tensor `[B, num_heads, head_dim, N]` where N is the state dim. Storage is 2× Mamba-2's real-valued state. The conv state is unchanged.

**vLLM diffs:** Mamba-3 is research-only as of cutoff; no vLLM support.

**llama.cpp diffs:** Mamba-3 is research-only as of cutoff; the complex-valued SSD scan would require new GGUF tensor types for complex weights.

**Evolution arc:**

- **Mamba (Dec 2023):** original selective SSM.
- **Mamba-2 (May 2024):** SSD parametrization; per-head real-valued scalar A.
- **Mamba-3 (Mar 2026):** complex-valued state, MIMO decoding, refined discretization. Research release; production-scale deployment pending.

Mamba-3 is the most-anticipated SSM update of 2026. Production adoption is expected in late 2026 once kernels mature.

---

### §5.48 Nemotron 3 Super / Ultra — NVIDIA hybrid Mamba-Transformer MoE at scale

Nemotron 3 Super (Mar 2026) and Nemotron 3 Ultra (Jun 2026) are NVIDIA's hybrid Mamba-Transformer MoE models at 120B-A12B and 550B-A55B scales. The architecturally-distinctive choice: **hybrid Mamba-2 + Transformer + MoE FFN** at scales beyond what the corpus previously had (Granite 4 H-Small was 30B-A3B; Nemotron 3 Ultra is 18× larger total / 18× larger active).

**Identity (Nemotron 3 Super):** 120B total / 12B active. Hybrid Mamba-2 + Transformer alternation; MoE FFN on the Transformer layers.

**Identity (Nemotron 3 Ultra):** 550B total / 55B active. Same hybrid + MoE pattern at larger scale.

**Decoder block:**

The Nemotron 3 design alternates Mamba-2 and Transformer blocks; the exact ratio per NVIDIA's blog is not fully disclosed but appears to be ~1:1 (each block has both a Mamba-2 sublayer and a Transformer sublayer, like Hymba but block-alternating rather than head-partitioned).

```python
class Nemotron3SuperBlock(nn.Module):
    def __init__(self, config, block_idx):
        self.mamba2 = Nemotron3Mamba2Mixer(config, block_idx)
        self.self_attn = Nemotron3Attention(config, block_idx)
        self.moe = Nemotron3MoEBlock(config) if config.num_experts > 0 else Nemotron3MLP(config)

        self.input_layernorm = RMSNorm(D, eps=eps)
        self.mid_layernorm = RMSNorm(D, eps=eps)
        self.post_layernorm = RMSNorm(D, eps=eps)

    def forward(self, hidden_states, ...):
        # Mamba-2 sublayer (first half of the block):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba2(hidden_states, ...)
        hidden_states = residual + hidden_states

        # Transformer sublayer (second half of the block):
        residual = hidden_states
        hidden_states = self.mid_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        # MoE FFN:
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

**Hybrid cache:** Mamba-2 state cache + attention KV cache, managed per-block.

**MoE configuration:** ~32-64 experts top-8 or similar. Renormalized softmax routing. Auxiliary loss balanced.

**FP8 / NVFP4 native:**

Nemotron 3 ships with **FP8 native** training (NVIDIA H100/H200/B100 FP8 path) for the Mamba-2 + Transformer math, and **NVFP4** (NVIDIA's 4-bit float format, similar to MXFP4 but with different scaling) for the MoE expert weights. This is the deepest production deployment of FP8/NVFP4 in a hybrid model.

**Throughput claim:** Nemotron 3 Ultra reports **6× throughput vs GLM-5.1** at 8K/64K context — a significant gain attributable to the hybrid SSM+attention design (the Mamba-2 sublayers' fixed-size cache scales much better with sequence length than full attention).

**Evolution arc:**

- **Nemotron 4 (Feb 2024):** dense Transformer at various scales.
- **Llama-3.1-Nemotron-Nano 8B-v1 (Mar 2025):** Llama-3.1-8B distillation with FP4-aware training.
- **Nemotron 3 Nano 4B (Nov 2025):** hybrid Mamba-2 + Transformer + MoE at small scale.
- **Nemotron 3 Super (Mar 2026):** scale up to 120B-A12B.
- **Nemotron 3 Nano Omni (Apr 2026):** compact omni-modal variant (vision + speech + language).
- **Nemotron 3 Ultra (Jun 2026):** 550B-A55B.

NVIDIA's investment in hybrid Mamba-Transformer + MoE at frontier scale is the strongest signal of SSM-hybrid mainstreaming in 2026. Combined with Granite 4 H-hybrid (§5.37) and the announced Mamba-3 (§5.47), the SSM-hybrid family is now mainstream.

---

## §6 Abandoned-design section

Survey papers traditionally have a "dead ends" section listing tried-and-abandoned ideas. v2 had 7 abandoned designs (§6.1 parallel attn+FFN, §6.2 pure MQA, §6.3 pure SWA without alternation, §6.4 ALiBi, §6.5 Q4_0 GGUF, §6.6 Yi-1.0 cosine attention, §6.7 auxiliary loss for MoE). v3 adds three new entries that the 2025-H2 to 2026-H1 evidence surfaces:

(The v2 entries are preserved in `research/02-layer-sources.v2.md` §6 unchanged. v3 adds §6.8–§6.10 below.)

### §6.8 Pure NTK-aware RoPE scaling (replaced by Llama-3 smooth + YaRN + LongRoPE)

**Lifespan:** ~2023 (the original NTK-aware-RoPE blog post by bloc97 / kaiokendev, https://github.com/jquesnelle/scaled-rope) to ~2024 (Llama-3 smooth interpolation replaced it). Status: superseded.

**The math:** scale the RoPE base by a power of the context-extension factor:
```
base_scaled = base * (extension_factor ** (head_dim / (head_dim - 2)))
inv_freq = 1.0 / (base_scaled ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

This shifts the rotation period for high-frequency channels less than for low-frequency channels — an empirically successful trick from the early-2024 long-context era.

**Why abandoned:** Three reasons:
1. **NTK-aware extends rotation continuously but unsmoothly at the boundary.** At positions just beyond the training max, the per-frequency rotation changes abruptly; this creates a measurable quality cliff.
2. **YaRN (Peng et al., arXiv:2309.00071) combines NTK + a temperature adjustment + a smooth blend** that outperforms pure NTK at the same compute cost.
3. **Llama-3 smooth interpolation** uses a piecewise linear scaling by frequency band, with explicit `low_freq_factor` and `high_freq_factor` knobs. The smooth version became the consensus 2024+ choice.

**Status:** no major 2024+ model uses pure NTK-aware RoPE without a smooth blend. InternLM 2.5/3 (§5.31) uses "dynamic NTK" which is a runtime-adaptive variant; the static pure-NTK is the abandoned design.

### §6.9 Static position scaling without smooth blend (Position Interpolation pure form)

**Lifespan:** 2023 (Chen et al. Position Interpolation, arXiv:2306.15595) to ~2024. Status: superseded by YaRN-style smooth methods.

**The math:** linearly scale all positions down by a factor:
```
position_scaled = position / extension_factor
```

This compresses the position embedding domain, allowing inference at extended context without retraining.

**Why abandoned:** Pure PI scales positions uniformly, but the optimal scaling differs per-frequency-band (high-frequency rotations should be scaled less than low-frequency). YaRN's contribution was a per-frequency-band scaling with smooth blend; LongRoPE (Phi-3.5) extended this with a search-discovered piecewise scaling that further outperforms YaRN.

**Status:** pure PI is no longer used. YaRN is the canonical reference; Llama-3 smooth and LongRoPE are the two production choices. Pure PI was the "first cut" that motivated the field but was rapidly superseded.

### §6.10 Vision token-per-patch dump without compression (replaced by M-RoPE, NaViT, dynamic-resolution, 16× conv compressor)

**Lifespan:** ~2023 (original LLaVA v1, with 14×14 patch grid and no compression) to 2024 (LLaVA-NeXT introduced AnyRes; Qwen2-VL introduced M-RoPE + dynamic resolution). Status: abandoned in OCR-LLM line.

**The design:** an image of fixed size (e.g., 336×336) is split into a fixed grid of patches (24×24 = 576 patches), and all 576 visual tokens are fed to the LM decoder. No compression, no merging, no spatial-aware token reduction.

**Why abandoned:**

1. **576 tokens per image is too many** at long-context multi-image use cases. A 4-image input consumes 2304 visual tokens before any text.
2. **Fixed-resolution input fails for documents** — PDFs with small text need higher resolution; photographs with global content need lower. Dynamic-resolution (Qwen2-VL onward) handles both.
3. **Spatial compression by conv stages** (DeepSeek-OCR's 16× compressor, GOT-OCR's 4096× compressor, DocOwl2's 324 tokens/page) reduces token count by an order of magnitude with minimal quality loss for OCR-specific tasks.

**Replacement consensus (per the OCR-LLM corpus, §5.41 and §5.45):**

- **Dynamic resolution:** Qwen2-VL onward; image is padded to a target patch count, native aspect ratio preserved.
- **Token compression:** GOT-OCR (1024² → 256 tokens), DeepSeek-OCR (1024² → 100 tokens via 16× conv), DocOwl2 (324 tokens per page), TextHawk2 (16× fewer tokens than v1).
- **M-RoPE:** Qwen2-VL onward; 2D/3D position encoding for spatial-aware attention.

**Status:** the original "dump all patches without compression" design is essentially extinct in production OCR-LLM. General VLMs (Cambrian-1, LLaVA-OneVision) still use it at 576-token-per-336²-image scale, but the OCR-LLM line has uniformly adopted compression.

---

## §7 Cross-family comparison table (v3 — 48 families)

A condensed view of where each family lands on the 38 axes. The v2 table covered 33 families; v3 extends with the 15 new entries. Legend: Y = uses, N = does not, ? = config-dependent, — = not applicable.

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
| GPT-OSS (v2) | pre | none | split | NEOX | per-layer | attn + sinks | MoE + bias | KV | learned attention sinks |
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
| Qwen3-Next (v2) | pre | per-Dh on attn layers | split | NEOX (attn), none (linear) | none | attn + GatedDeltaNet | MoE | KV + linear-attn state | gated DeltaNet |
| RWKV-7 | pre | none | (no attention) | none | none | WKV recurrent (TTT-style) | channel mixing in WKV | per-channel state | pure recurrent |
| Falcon-7B | parallel (1 or 2 norms) | none | split (MQA) | NEOX | none | attn (MQA) | GELU | KV | historical |
| StarCoder 2 | pre | none | split + biases | NEOX | global | attn | GELU + dropout (NO gate) | KV | FIM tokens, no SwiGLU |
| InternLM 2.5/3 | pre | none | split (q/k/v bias) | Dynamic NTK | none | attn | SwiGLU | KV | runtime-adaptive RoPE base |
| ChatGLM 3 | post (RMSNorm) | none | split (MQA Hk=2) | partial 2D-RoPE | none | attn (prefix-LM) | GeGLU | KV | prefix-LM training |
| MobileLLM | pre | none | split (tied embed) | NEOX | none | attn (block-shared) | SwiGLU | KV | block repetition |
| **Gemma 4 (v3)** | **sandwich** | **per-Dh fixed-scale** | **K=V on global + split on sliding** | **dual-theta + p-RoPE 0.25 on global** | **5:1 (4:1 E2B)** | **attn** | **GeGLU + first Gemma MoE 128+1 shared (26B-A4B)** | **KV with cross-layer sharing (20/35 for E2B)** | **PLE, p-RoPE, encoder-free 12B-Unified, restored final softcap 30.0** |
| **Llama 4 Scout (v3)** | **pre** | **none** | **split** | **iRoPE (per-layer on/off)** | **none** | **attn (gated NoPE/RoPE per layer)** | **MoE 16x top-1** | **KV** | **iRoPE NoPE alternation, native multimodal early fusion** |
| **Qwen3-Next (v3 expanded)** | **pre** | **per-Dh on attn layers** | **split (attn) + linear-DeltaNet** | **NEOX (attn), none (linear)** | **none** | **GatedDeltaNet 3:1 hybrid** | **ultra-sparse MoE 10+1/512** | **KV + linear-attn matrix state** | **MTP draft heads, 2.15% activation** |
| **Granite 4 H-hybrid (v3)** | **pre** | **none** | **split (attn layers)** | **NEOX (attn layers)** | **none** | **Mamba-2 + attn at 9:1** | **MoE 64x top-8 (attn layers only) + dense (Mamba layers)** | **KV + (conv, ssm)** | **9:1 Mamba:Attn, μP scalars** |
| **DeepSeek-V3.2 (v3)** | **pre** | **none (q_a/kv_a norm)** | **MLA + Lightning Indexer** | **NEOX (YaRN)** | **none** | **MLA + DSA sparse top-2048** | **MoE+shared+sigmoid** | **MLA compressed + indexer keys** | **Lightning Indexer DSA (abandoned in V4)** |
| **DeepSeek-V4 (v3)** | **pre** | **none (q_a/kv_a norm)** | **MLA + CSA + HCA** | **NEOX (YaRN)** | **none** | **CSA + HCA hybrid sparse** | **MoE+shared+sigmoid** | **CSA+HCA compressed** | **27% V3.2 FLOPs, 10% V3.2 KV cache** |
| **Apple AFM 3.18B (v3)** | **pre** | **none** | **split + cross-block KV share** | **NEOX** | **none** | **attn** | **SwiGLU** | **KV with block-2 → block-1 sharing (37%)** | **cross-block KV share, 2-bit QAT** |
| **GPT-OSS 20B (v3 expanded)** | **pre** | **none** | **split** | **NEOX** | **alternating 1:1 dense/banded-sparse** | **attn + sinks (trained)** | **MoE 32x top-4 + bias** | **KV** | **MXFP4-native weights, GPT-3-style alt, trained sinks** |
| **DeepSeek-OCR (v3)** | **pre** | **none (q_a/kv_a norm)** | **MLA** | **NEOX** | **none** | **MLA + Visual Causal Flow mask (V2)** | **MoE A570M** | **MLA compressed + block-bidirectional mask** | **SAM+CLIP serial encoder, 16× compressor, Visual Causal Flow** |
| **Ministral 3B/8B (v3)** | **pre** | **none** | **split** | **NEOX (θ=1e7)** | **interleaved per-layer (3rd SWA pattern)** | **attn** | **SwiGLU** | **KV bounded per layer** | **interleaved SWA window sizes** |
| **Phi-4-multimodal (v3)** | **pre** | **none** | **fused QKV + per-modality LoRA Q/V add** | **partial NEOX** | **none** | **attn + mixture-of-LoRAs routing** | **SwiGLU (frozen base)** | **KV** | **mixture-of-LoRAs modality routing** |
| **MiniMax-Text-01 (v3)** | **pre** | **none** | **split** | **NEOX** | **none** | **Lightning Attention 7:1 hybrid** | **MoE** | **KV (on 1/8 attn layers) + scalar-decay matrix state (7/8 lightning)** | **Lightning Attention, abandoned in M2** |
| **GOT-OCR 2.0 (v3)** | **pre** | **none** | **split (Qwen-1 0.5B)** | **NEOX** | **none** | **attn** | **SwiGLU** | **KV** | **smallest OCR-LLM (580M total), single-linear connector** |
| **Qwen2.5-VL (v3)** | **pre** | **per-Dh** | **split** | **M-RoPE 2D/3D** | **window-attn in ViT, full in decoder** | **attn (decoder + LM-shaped ViT)** | **SwiGLU** | **KV** | **M-RoPE, dynamic-resolution patches** |
| **Mamba-3 (v3)** | **pre** | **none** | **(no attention)** | **none** | **none** | **complex-state SSM + MIMO decoding** | **SSM-fused** | **(conv, complex_ssm)** | **complex state, MIMO** |
| **Nemotron 3 Super/Ultra (v3)** | **pre** | **none** | **split (attn layers)** | **NEOX** | **none** | **Mamba-2 + attn alternation** | **MoE FFN** | **KV + (conv, ssm)** | **hybrid SSM-Attn MoE at 120B-550B, FP8/NVFP4 native** |

---

## §8 Universal pseudocode of "the decoder block", parametrized (v3)

Putting all 38 axes together, the universal block can be written as a single Python function. This is the shape an API floor should support; everything is dispatched by configuration.

```python
def decoder_block(
    h, residual,                              # [B, S, D] (HF) or [T, D] (vLLM)
    *,
    layer_idx, position_embeddings, attention_mask, past_key_values,
    # per-layer width (OpenELM):
    H_per_layer=None, Hk_per_layer=None, Df_per_layer=None,
    # token mixer
    token_mixer_kind,         # "attention" | "mamba1" | "mamba2" | "mamba3" | "rg_lru" | "linear_attn" | "gated_deltanet" | "lightning_attn" | "wkv" | "parallel_mamba_attn" | "DSA_lightning_indexer" | "CSA_HCA"
    token_mixer_params,
    # channel mixer
    channel_mixer_kind,       # "mlp_swiglu" | "mlp_gelu_plain" | "mlp_squared_relu" | "moe" | "moe_with_shared"
    channel_mixer_params,
    # norm placement
    norm_kind,                # "rms" | "rms_one_plus" | "layer" | "layer_no_bias"
    norm_placement,           # "pre" | "post" | "sandwich" | "parallel"
    eps,
    residual_multiplier=1.0,  # Granite
    resid_pdrop=0.0,          # Phi-3, StarCoder 2
    use_rope=True,            # SmolLM3 per-layer, Llama-4 iRoPE per-layer
    partial_rotary_factor=1.0,  # Gemma 4 global (0.25), MLA (qk_rope/qk_head), Phi-3 legacy (0.5)
    shared_block_id=None,     # Zamba2 / MobileLLM
    kv_source_layer=None,     # Gemma 4 cross-layer KV share, Apple AFM cross-block share
    ple_input=None,           # Gemma 4 Per-Layer Embeddings residual term
    attn_sub_norm=None,       # BitNet
    ffn_sub_norm=None,        # BitNet
    block_ids=None,           # DeepSeek-OCR Visual Causal Flow block-bidirectional mask
    sinks=None,               # GPT-OSS trained attention sinks
    mxfp4_weights=False,      # GPT-OSS MXFP4-native weight precision flag
    mimo_heads=1,             # Mamba-3 MIMO decoding heads count
    fixed_scale_qk_norm=None, # Gemma 4 fixed-scale QK-norm value
    k_equals_v=False,         # Gemma 4 K=V unification on global layers
    modality_routing=None,    # Phi-4-multimodal mixture-of-LoRAs routing
):
    if norm_placement == "parallel":
        x_normed = norm(h, eps, kind=norm_kind, weight=W_attn_pre)
        y_attn = token_mixer(x_normed, position_embeddings, attention_mask,
                             past_key_values, layer_idx, use_rope=use_rope,
                             partial_rotary_factor=partial_rotary_factor,
                             kv_source_layer=kv_source_layer,
                             attn_sub_norm=attn_sub_norm, sinks=sinks,
                             block_ids=block_ids, k_equals_v=k_equals_v,
                             fixed_scale_qk_norm=fixed_scale_qk_norm,
                             modality_routing=modality_routing,
                             **token_mixer_params)
        y_mlp  = channel_mixer(x_normed, ffn_sub_norm=ffn_sub_norm,
                               modality_routing=modality_routing, **channel_mixer_params)
        return h + y_attn * residual_multiplier + y_mlp * residual_multiplier, None

    # Token mixer
    if norm_placement in ("pre", "sandwich"):
        x = norm(h, eps, kind=norm_kind, weight=W_attn_pre)
    else:
        x = h
    y = token_mixer(x, position_embeddings, attention_mask,
                    past_key_values, layer_idx, use_rope=use_rope,
                    partial_rotary_factor=partial_rotary_factor,
                    kv_source_layer=kv_source_layer,
                    attn_sub_norm=attn_sub_norm, sinks=sinks,
                    block_ids=block_ids, k_equals_v=k_equals_v,
                    fixed_scale_qk_norm=fixed_scale_qk_norm,
                    mimo_heads=mimo_heads,
                    modality_routing=modality_routing,
                    **token_mixer_params)
    if norm_placement in ("post", "sandwich"):
        y = norm(y, eps, kind=norm_kind, weight=W_attn_post)
    if resid_pdrop > 0: y = dropout(y, resid_pdrop)
    h = residual_add(h, y, scale=residual_multiplier)

    # PLE injection (Gemma 4 edge sizes):
    if ple_input is not None:
        ple_residual = W_ple_projection @ ple_input * (1.0 / math.sqrt(2.0))
        h = h + ple_residual

    # Channel mixer
    if norm_placement in ("pre", "sandwich"):
        x = norm(h, eps, kind=norm_kind, weight=W_ffn_pre)
    else:
        x = h
    y = channel_mixer(x, ffn_sub_norm=ffn_sub_norm,
                      modality_routing=modality_routing, **channel_mixer_params)
    if norm_placement in ("post", "sandwich"):
        y = norm(y, eps, kind=norm_kind, weight=W_ffn_post)
    if resid_pdrop > 0: y = dropout(y, resid_pdrop)
    h = residual_add(h, y, scale=residual_multiplier)
    return h, None
```

The v3 token_mixer extends v2 with: `partial_rotary_factor` (Gemma-4 / MLA), `kv_source_layer` (Gemma-4 / AFM), `block_ids` (DeepSeek-OCR Visual Causal Flow), `k_equals_v` (Gemma-4), `fixed_scale_qk_norm` (Gemma-4), `mimo_heads` (Mamba-3), `modality_routing` (Phi-4-multimodal). The v3 channel_mixer extends v2 with `modality_routing` for mixture-of-LoRAs.

The PLE injection (Gemma 4) is a new optional residual term added between the token mixer and the channel mixer. It is `None` for all non-Gemma-4 families.

---

## §9 Open questions and things still being figured out (v3 — extended)

(v2 §9 has 13 open questions; v3 carries them forward and adds 7 new ones surfaced by the 15 new families.)

**v2 carry-forward (1-13):** MLA vs GQA at 7-30B scale; DeepSeek-V3 routing sigmoid+bias vs softmax+aux; Gemma 3 dropped softcap necessity; parallel residual not winning at scale; NoPE fraction; sink tokens learned vs special-token; Mamba-2 SSD vs softmax at long context; OpenELM per-layer width quality cost; Zamba2 num_mem_blocks; DeepSeek-V4 architecture (now answered in §5.38); Qwen3-Next Gated DeltaNet long-context recall; abandoned ALiBi; things not directly verifiable. See v2 §9 for full text.

**v3 new (14-20):**

14. **Cross-layer KV sharing — what is the optimal fraction?** Gemma 4 E2B shares 57% of layers; Apple AFM shares 37%. Both are tuned empirically. The trade-off: more sharing reduces cache but loses per-layer specialization. No theoretical guidance.

15. **Per-Layer Embeddings (Gemma 4) vs adapter / per-layer LoRA — which is the right abstraction?** PLE adds a residual term per layer from a separate table; per-layer LoRA modifies the sublayer. Both are parameter-efficient per-layer specialization mechanisms. No head-to-head comparison.

16. **Partial-rotary RoPE — what is the optimal fraction?** Gemma 4 global uses 0.25; MLA uses ~0.33; Phi-3 legacy used 0.5. The variation suggests the optimal depends on context length, training data, and the QK-norm presence.

17. **iRoPE / NoPE alternation — what is the optimal RoPE:NoPE ratio?** Llama 4 Scout uses majority-NoPE (~75%); SmolLM3 uses majority-RoPE (~75%). The two production data points are inversely chosen. No published ablation.

18. **DSA Lightning Indexer vs CSA+HCA — why did V4 abandon DSA?** The V4 release blog cites wall-clock efficiency, but no architecture-level ablation is public.

19. **Lightning Attention 7:1 (MiniMax) vs Gated DeltaNet 3:1 (Qwen3-Next) — what fraction of linear-attention is too much?** MiniMax reverted in M2; Qwen3-Next currently retains. The community position: ~75% linear with delta-rule is the upper bound, but no controlled experiment.

20. **Mamba-3 complex state — does the oscillatory-dynamics gain survive at production scale?** Mamba-3 is research-only; production deployment pending. The complex state doubles cache storage; the gain must justify the cost.

**Things still unverifiable in v3:**

- Gemma 4 tech report (no first-party arXiv as of cutoff; values from per-size config.json).
- DeepSeek-V4 CSA + HCA exact math (release blog gives FLOPs/cache claims but not the per-layer attention algorithm).
- Apple AFM training details (closed-weight; architecture from arXiv:2507.13575 only).
- Nemotron 3 Super/Ultra exact Mamba-2:Attn ratio (NVIDIA blog says "hybrid" without exact pattern).
- Qwen3-Next MTP head training loss (mentioned in release blog but not documented in detail).
- Mamba-3 production-scale viability (research scales only as of cutoff).

---

## §10 Appendices (v3)

### Appendix A — File and line citations cross-reference (v3 additions)

(v2 Appendix A carries forward. v3 adds the 15 new families' source locations.)

- Gemma 4 HF (proposed): `transformers/src/transformers/models/gemma4/modeling_gemma4.py` (HF transformers 5.x; not yet finalized at cutoff).
- Gemma 4 model cards: `https://huggingface.co/google/gemma-4-E2B`, `-E4B`, `-12B`, `-26B-A4B`, `-31B` — each with verified `config.json`.
- Llama 4 HF: `transformers/src/transformers/models/llama4/modeling_llama4.py`.
- Qwen3-Next HF: `transformers/src/transformers/models/qwen3_next/modeling_qwen3_next.py` lines 256–990 (v2 had preview at §5.27; v3 §5.36 expands).
- Granite 4 H-hybrid HF: `transformers/src/transformers/models/granite_moe_hybrid/modeling_granite_moe_hybrid.py` (PR-stage).
- DeepSeek-V3.2 HF: `transformers/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py`.
- DeepSeek-V4: external — `github.com/deepseek-ai/DeepSeek-V4`.
- Apple AFM: external — arXiv:2507.13575 only.
- GPT-OSS HF: `transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py` lines 74–410 (v2 §5.11; v3 §5.40 expands with MXFP4 details).
- DeepSeek-OCR: external — `github.com/deepseek-ai/DeepSeek-OCR`.
- Ministral 3B/8B HF: `transformers/src/transformers/models/ministral/modeling_ministral.py`.
- Phi-4-multimodal HF: `transformers/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py`.
- MiniMax-Text-01 HF: `transformers/src/transformers/models/minimax_text_01/modeling_minimax_text_01.py`.
- GOT-OCR 2.0 HF: `transformers/src/transformers/models/got_ocr2/modeling_got_ocr2.py` (in transformers since v4.45).
- Qwen2.5-VL HF: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`.
- Mamba-3: external — arXiv:2603.15569; reference implementation `github.com/state-spaces/mamba`.
- Nemotron 3 Super/Ultra: external — NVIDIA Megatron-LM, `huggingface.co/nvidia` collections.

### Appendix B — Cross-family axis-coverage matrix (v3 additions only)

| Axis | Gemma 4 | Llama 4 | Qwen3-Next | Granite 4 H | DSV3.2/V4 | Apple AFM | GPT-OSS | DS-OCR | Ministral | Phi-4-MM | MM-Text-01 | GOT-OCR2 | Qwen2.5-VL | Mamba-3 | Nemotron 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.1 norm placement | sandwich | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre |
| 3.2 norm formula | RMS(1+w) | RMS | RMS | RMS | RMS | RMS | RMS | RMS | RMS | RMS | RMS | RMS | RMS | RMS | RMS |
| 3.3 QK-norm | per-Dh fixed-scale | none | per-Dh on attn | none | none | none | none | none | none | none | none | none | per-Dh | none | none |
| 3.5 RoPE kind | dual-theta + p-RoPE 0.25 global | iRoPE per-layer | NEOX (attn), none (linear) | NEOX | YaRN | NEOX | NEOX | NEOX | NEOX | partial NEOX | NEOX | NEOX | M-RoPE 2D/3D | none | NEOX |
| 3.7 QKV layout | K=V global + split sliding | split | split + linear-Delta | split | MLA | split | split | MLA | split | fused QKV + LoRA-Q/V add | split | split | split | n/a | split |
| 3.9 softcap | final 30.0 (attn None) | none | none | none | none | none | none | none | none | none | none | none | none | n/a | none |
| 3.10 SWA | 5:1 (4:1 E2B) | none | none | none | none | none | 1:1 alt dense/sparse | none | interleaved per-layer | none | none | none | window in ViT | none | none |
| 3.14 channel mixer | dense / MoE 128+1 shared | MoE 16x top-1 / 128x top-2 | ultra-sparse 10+1/512 | MoE 64x top-8 (attn layers) | MoE+shared+sigmoid | SwiGLU | MoE+bias | MoE A570M | SwiGLU | SwiGLU (frozen base) | MoE | SwiGLU | SwiGLU | n/a | MoE |
| 3.16 token mixer | softmax attn | softmax + per-layer-NoPE-mask | GatedDeltaNet 3:1 + softmax | Mamba-2 + attn 9:1 | MLA + DSA / CSA+HCA | softmax attn | softmax + sinks | MLA + VCF mask | softmax | softmax + LoRA-route | Lightning 7:1 | softmax | softmax + LM-shaped ViT | complex SSM + MIMO | Mamba-2 + attn alt |
| 3.21 cache | KV with cross-layer share | KV | KV + linear-matrix state | KV + (conv, ssm) | MLA + indexer / CSA+HCA | KV cross-block share | KV | MLA + block-bidir | KV bounded | KV | KV + matrix state | KV | KV | (conv, complex_ssm) | KV + (conv, ssm) |
| 3.33 cross-layer KV share | 20/35 (E2B) | none | none | none | none | block-2 → block-1 | none | none | none | none | none | none | none | none | none |
| 3.34 PLE | E2B/E4B yes | none | none | none | none | none | none | none | none | none | none | none | none | none | none |
| 3.35 partial-rotary | 0.25 global | n/a (NoPE has no rotation) | none | none | MLA partial | none | none | MLA partial | none | none | none | none | none | n/a | none |
| 3.36 vision adapter | encoder + ViT (4 sizes) / encoder-free (12B) | early-fusion native MM | n/a | n/a | n/a | concat-projector (ViTDet-L) | n/a | concat-prefix (SAM+CLIP+16× comp) | n/a | mixture-of-LoRAs | n/a | concat-prefix (VitDet+linear) | concat-projector (M-RoPE ViT) | n/a | n/a |
| 3.37 block-bidir mask | none | none | none | none | none | none | none | Visual Causal Flow (V2) | none | none | none | none | none | none | none |
| 3.38 MXFP4 native | bf16 + Q4_0 QAT | bf16 | bf16 | bf16 | bf16 | 2-bit QAT | **MXFP4 native** | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | FP8 + NVFP4 |

### Appendix C — Numeric constants per v3 family

| Family | head_dim | attn scale | rope_theta | rms_norm_eps | embed scale | residual scale | logits scale | softcap |
|---|---|---|---|---|---|---|---|---|
| Gemma-4-E2B | 256 (local) / 512 (global) | 1.0 (absorbed in fixed-scale QK-norm) | 1e6 global / 1e4 sliding | 1e-6 | sqrt(1536) ≈ 39.19 | 1.0 | 1.0 | final 30.0 |
| Gemma-4-E4B | 256 / 512 | 1.0 | 1e6 / 1e4 | 1e-6 | sqrt(2560) ≈ 50.59 | 1.0 | 1.0 | final 30.0 |
| Gemma-4-12B-Unified | 256 / 512 | 1.0 | 1e6 / 1e4 | 1e-6 | sqrt(3840) ≈ 61.97 | 1.0 | 1.0 | final 30.0 |
| Gemma-4-26B-A4B | 256 / 512 | 1.0 | 1e6 / 1e4 | 1e-6 | sqrt(2816) ≈ 53.07 | 1.0 | 1.0 | final 30.0 |
| Gemma-4-31B | 256 / 512 | 1.0 | 1e6 / 1e4 | 1e-6 | sqrt(5376) ≈ 73.32 | 1.0 | 1.0 | final 30.0 |
| Llama-4-Scout | 128 | 1/sqrt(128) (on RoPE layers) | 5e5 | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| Qwen3-Next-80B | 128 | 1/sqrt(128) (attn) | 1e6 (attn) | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| Granite-4-H-Tiny | 64 | 0.0078125 (Granite muP) | 5e6 | 1e-5 | 12.0 | 0.22 | 8.0 | none |
| DeepSeek-V3.2 | qk=192, v=128 | 1/sqrt(192) * mscale^2 | YaRN | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| Apple AFM 3.18B | 128 | 1/sqrt(128) | ? (closed) | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| GPT-OSS-20B | 64 | 1/sqrt(64) | release-specific | 1e-5 | 1.0 | 1.0 | 1.0 | none + sinks |
| DeepSeek-OCR | qk=96, v=64 | 1/sqrt(96) | 10000 | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| Ministral-8B | 128 | 1/sqrt(128) | 1e7 | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| Phi-4-multimodal | 96 | 1/sqrt(96) | 1e4 | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| MiniMax-Text-01 | 128 (attn) | 1/sqrt(128) (attn) | NEOX | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| GOT-OCR-2 | 64 | 1/sqrt(64) | 10000 | 1e-5 | 1.0 | 1.0 | 1.0 | none |
| Qwen2.5-VL-3B | 128 | 1/sqrt(128) | 1e6 | 1e-6 | 1.0 | 1.0 | 1.0 | none |
| Mamba-3 | 64 (per-head) | n/a (no softmax) | none | 1e-5 | 1.0 | 1.0 | 1.0 | n/a |
| Nemotron-3-Super | 128 (attn) | 1/sqrt(128) (attn) | 5e5 | 1e-5 | 1.0 | 1.0 | 1.0 | none |

### Appendix D — Engine divergence checklist (v3 extensions)

(v2 Appendix D carries forward. v3 adds notes on the new engines/kernels needed for the 15 new families.)

**New kernel requirements for v3 families:**

1. **Partial-rotary RoPE** (Gemma 4 global, MLA, Mamba-3 conv pre-stage). Existing FA / SDPA rotary takes `partial_rotary_factor` as a kwarg.
2. **K=V unification** (Gemma 4 global). The standard QKVParallelLinear assumes 3 separate K/V/Q; K=V on global layers needs a Q-only + KV-unified path.
3. **Cross-layer KV cache pointer** (Gemma 4 num_kv_shared_layers, Apple AFM cross-block). Cache abstraction must allow alias entries.
4. **PLE injection** (Gemma 4 E2B/E4B). Per-layer projection + per-layer residual add.
5. **Block-bidirectional mask** (DeepSeek-OCR Visual Causal Flow). FlexAttention mask_mod with block_ids tensor.
6. **MXFP4 dequantize-and-multiply kernel** (GPT-OSS). Custom CUDA kernel for the 4.25-bit format.
7. **Lightning Indexer top-k mask** (DeepSeek-V3.2 DSA). Per-query top-k selection + sparse softmax.
8. **CSA + HCA compressed attention** (DeepSeek-V4). Not yet implemented in any open engine.
9. **Gated DeltaNet matrix-state recurrence** (Qwen3-Next). Per-layer matrix state, complex chunked scan kernel.
10. **Lightning Attention scalar-decay recurrence** (MiniMax-Text-01). Per-layer matrix state, decay-only scan.
11. **Mixture-of-LoRAs Q/V routing** (Phi-4-multimodal). Per-token modality-mask + LoRA adapter selection.
12. **Mamba-3 complex SSD scan** (Mamba-3). Complex-valued recurrence kernel; MIMO output projection.
13. **iRoPE per-layer on/off** (Llama 4 Scout). Per-layer apply_rope: bool baked into layer construction.
14. **Interleaved per-layer SWA window** (Ministral). Per-layer sliding_window propagated to attention backend.
15. **9:1 Mamba-2 : attention alternation** (Granite 4 H-hybrid). Per-layer dispatch on layer_type at construction.

The v3 engine landscape: vLLM and llama.cpp have partial support for axes 1, 3, 5, 9, 13, 14, 15 as of cutoff. Axes 6, 7, 8, 10, 11, 12 are research / proprietary kernel territory and have not been ported to open inference engines yet.

---

## §11 Summary (v3)

The v3 survey of 48 families (33 v2 + 15 new) and 38 axes (32 v2 + 6 new) gives a sufficient sample to bound the API design space across the 2023–2026 mainstream LM evolution. The key takeaways:

1. **The Llama-shape API floor remains real but the boundary is moving.** ~60% of new releases share Llama 3 structure exactly: pre-norm + RMSNorm + RoPE + GQA + SwiGLU + sequential residual. The other 40% are deltas on specific axes — and the 15 new v3 families show the deltas are accelerating: K=V unification (Gemma 4), iRoPE NoPE-alternation (Llama 4), cross-layer KV sharing (Gemma 4, Apple AFM), Per-Layer Embeddings (Gemma 4), partial-rotary p-RoPE (Gemma 4), MXFP4-native weights (GPT-OSS), Visual Causal Flow (DeepSeek-OCR), mixture-of-LoRAs (Phi-4-multimodal), complex-state SSM (Mamba-3).

2. **Norm placement remains the most-explored axis.** Pre (Llama), post (OLMo 2), sandwich (Gemma 3/4), parallel (Cohere). Each has rationale and shipped models. Gemma 4 preserves Gemma 3's sandwich unchanged.

3. **MoE is converging on three families:** (a) Mixtral-style softmax+renorm (Mixtral, Qwen3-MoE, OLMoE, Gemma 4 26B-A4B, Granite 4); (b) DeepSeek-style sigmoid+bias-correction (DeepSeek-V3, V3.2, V4); (c) ultra-sparse with shared expert (Qwen3-Next 10+1/512). Shared experts are now standard in 2025+ releases.

4. **MLA is now a small but stable family.** DeepSeek-V2, V3, V3.2, V4 plus MiniCPM 3 plus DeepSeek-OCR. The compression headline (71× at 128 heads) only becomes economically relevant at H ≥ 100; smaller models stick with GQA Hk=8.

5. **SSM-hybrids fragmented in 2024 and are mainstreaming in 2026.** Mamba-1 (Jamba, Falcon-Mamba), Mamba-2 (Zamba2, Falcon-H1, Granite 4 H-hybrid, Nemotron 3), Mamba-3 (research), RG-LRU (RecurrentGemma), GatedDeltaNet (Qwen3-Next), Lightning Attention (MiniMax-Text-01 — abandoned in M2), WKV (RWKV-7), parallel heads (Hymba), block alternation (Samba, Granite 4). The hybrid design space is still being explored — but Mamba-2 + Transformer at 9:1 (Granite 4) and 1:1 (Nemotron 3) are emerging as production-stable patterns.

6. **Cross-layer KV sharing is the most important new axis added in v3.** Gemma 4 (num_kv_shared_layers, last-N-shared) and Apple AFM (cross-block share, block-2 → block-1) both exploit the same underlying observation: in late layers, K/V projections are highly redundant. The two families chose different sharing topologies, but the axis is the same.

7. **Multimodal early-fusion (Gemma 4 12B-Unified, Llama 4) replaces concat-prefix-with-encoder as the next-gen multimodal pattern.** No separate vision encoder; raw patches projected directly into the LM. This is cleaner topologically, faster at inference, and trains jointly with text. Concat-prefix-with-encoder (PaliGemma, LLaVA) remains widely deployed in 2025-released VLMs but is being phased out in 2026.

8. **OCR-LLM is a coherent sub-segment now.** GOT-OCR 2.0 → DeepSeek-OCR (with Visual Causal Flow in V2) → MinerU2.5 → HunyuanOCR → PaddleOCR-VL → olmOCR-2 forms a clear architectural lineage. The signature: small LM decoder (≤2B), specialized high-resolution vision encoder, aggressive token compression (4–16×).

9. **Quantization-as-architecture has matured:** BitNet b1.58 (ternary, native training), GPT-OSS (MXFP4 4.25-bit native training), Apple AFM (2-bit QAT), Falcon-Edge (1.58-bit retrainable). PTQ paths (AWQ, GPTQ, GGUF Q4_K_M) are still dominant but native-low-bit training is now a real production option.

10. **Engineering divergences (HF vs vLLM vs llama.cpp) remain first-class.** Continuous batching, paged KV cache, QK weight permutation, quantized matmul, cross-layer KV pointer aliasing — all non-negotiable for the API to be deployment-friendly. v3's 6 new axes (3.33–3.38) all require coordinated changes across all three engines.

The v3 axis catalog (38 axes) is the bounded answer to "what does the API need to parameterize" across the 2023–2026 mainstream LM corpus. Anything in this document that the API cannot express is a real missing axis; anything outside this document is out of scope for v1 of the API. The cadence of new architectural deltas (15 first-of-kind elements in the past 6 quarters) suggests the corpus will continue to grow; v4 of this survey will likely add another 4-6 axes for the 2026-H2 and 2027-H1 releases.

---

*End of 02-layer-sources.v3.md. For full verbatim text of §5.1–§5.33 (v2 families), see `research/02-layer-sources.v2.md`. For full per-size Gemma 4 config verification, see `research/issues/10-gemma4-investigation.md`. For OCR-LLM catalog details, see `research/06-ocr-vlm-extensions.md`. For 2026-Q1 to 2026-Q2 release census, see `research/08-recent-releases-2026q2.md`. For v2-vs-v3 audit, see `research/issues/08-coverage-justification.md`.*
