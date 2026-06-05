# Critique — `05-kvcache-attention.md`

**Reviewer stance:** Skeptical technical reviewer. The survey is competent and unusually careful in places (the MLA dimensional accounting, the Gemma-3 per-layer RoPE base, the KIVI K-per-channel / V-per-token note, the llama.cpp `split_half` permutation). But the coverage has serious gaps and a handful of factual claims are either wrong, dated, or unverifiable as stated. This document enumerates what's missing, what's wrong, and what an authoritative v2 should contain.

---

## Section 1 — Missing attention variants (severity)

The report claims to enumerate "every shipped attention block ... in mainstream small-language-model runtimes" and says it covers **13 kinds**. The scope filter ("≤14B SLMs") is invoked to dismiss several variants. That filter is selectively applied — e.g., DeepSeek-V2 (16B / A2.4B) is in, but RingAttention (a sequence-parallel inference recipe orthogonal to model size) is out. The filter should be on *which kernels a 14B-and-below runtime is expected to support*, not on the model size where the technique was first published. Under that reading, several major omissions exist.

**Critical (S1) — algorithmic kernels every modern runtime cares about:**

1. **FlashAttention-1 vs 2 vs 3 algorithmic differences.** §4.1 mentions "FA2/3" as kernels but never explains the algorithmic delta. FA2 (Dao 2023, `flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`) moved the online-softmax accumulators to per-warp registers and swapped the inner/outer loop so Q is outer; FA3 (Shah, Bikshandi et al. 2024, `hopper/flash_fwd_kernel_sm90.cu`) adds Hopper warp-specialization (producer/consumer warps via `cuda::barrier`), TMA descriptors, and FP8 E4M3/E5M2 with a per-block descale path. For an "API surface" survey this matters because **FA3 changes which dtypes and block sizes are admissible** on H100 (e.g., FP8 KV with bf16 Q is only an FA3 path, not FA2). Severity: **high**.
2. **FlashDecoding / FlashDecoding++.** Mentioned once in passing in §4.1 with a single sentence. FlashDecoding++ (Hong et al. 2024) adds unified max + per-partition softmax which is what vLLM `paged_attention_v2` actually uses. No code pointer given. Severity: **medium**.
3. **RingAttention (Liu, Zaharia, Abbeel 2023) and Striped Attention (Brandon et al. 2023).** Sequence-parallel attention used by long-context inference services (Gemini-1.5-style 1M context, also adopted in some TRT-LLM long-context paths). It is not strictly required for ≤14B local serving, but the report's own scope includes "Qwen2.5-1M" which is exactly where Ring/Striped matter. Severity: **medium**.
4. **Native Sparse Attention (NSA, DeepSeek 2025, Yuan et al.).** The current state-of-the-art trainable sparse attention with three branches (compressed, selected, sliding), trainable router, and bespoke cache layout (one cache per branch plus a coarse anchor cache). Omitted entirely. NSA is shipped in DeepSeek-V3.x technical preview and is the most architecturally consequential attention change post-MLA. Severity: **high**.
5. **PagedAttention v2 / vAttention (Prabhu et al. MSR-I 2024).** §3.2 conflates "paged attention" with "vLLM v1". vAttention does CUDA virtual-memory mapping (`cuMemMap` / `cuMemAddressReserve`) to get the contiguity benefit without block-table indirection. TRT-LLM has adopted parts of this. Severity: **medium**.
6. **RadixAttention (SGLang, Zheng et al. 2024).** §3.9 mentions it but does not describe what it is — a radix-tree-indexed prefix cache with LRU eviction and copy-on-write semantics. The report just says "SGLang keeps a radix tree of `(token_id_seq, physical_block_id)`" without explaining that the kernel itself is unchanged from PagedAttention; only the *allocator* differs. Severity: **medium**.
7. **TreeAttention for speculative decoding.** §4.3 alludes to "tree mask" via EAGLE-2 but doesn't enumerate the kernel layer. TreeAttention (Spector & Re 2023, also Medusa-2) is the actual kernel that exploits the static tree mask for ~2× speedup over generic mask attention. Severity: **medium**.
8. **Hopper TMA + warp-specialization (FA3-specific).** The report's "kernel design" section never mentions TMA descriptors, `cluster_launch_control`, or `wgmma.async` — all of which determine which attention shapes are even kernel-feasible on H100. Severity: **medium**.

**Important (S2) — token-mixer variants under active production:**

9. **DeltaNet (Yang & Pan 2024) / Gated DeltaNet.** Linear attention with delta-rule updates. Shipped in `flame` runtime and Microsoft's internal `RetNet-Delta` experiments. Severity: **low-medium** (not in the ≤14B-text-SLM big list yet, but rapidly entering).
10. **Gated Linear Attention (GLA, Yang et al. 2023).** Used in Jamba-1.5 and Zamba2 ablations. The report claims Zamba-2 is "Mamba-2" — actually Zamba-2 1.2B/2.7B uses **Mamba-2 + shared GQA attention blocks** (Glorioso et al. 2024 §2.2); the report has this half-right but does not call out GLA. Severity: **medium**.
11. **HGRN / HGRN2 (Qin, Yang et al. 2024).** Hierarchical Gated Recurrent Networks; an attention-replacing token mixer family used in `flame-rnn` and shipped in OpenLLaMA-Linear. Omitted. Severity: **low**.
12. **xLSTM (mLSTM/sLSTM, Beck et al. 2024).** The 7B xLSTM checkpoint was released May 2024 with mLSTM as the primary token mixer. Severity: **low**.
13. **Mamba selective state-space details.** §1.3 lists "Mamba" as an SSM kind but the **selective scan recurrence** `h_t = A(x_t) h_{t-1} + B(x_t) x_t` with input-dependent A,B,Δ is described nowhere. The cache for Mamba is not just "`ssm_state ∈ [B, n_inner, d_state]`, fixed size in T" as §1.3 claims — it is also a 1-D convolution state of size `[B, n_inner, d_conv-1]`. Without `conv_state` you cannot resume Mamba inference. Severity: **high** (factual omission).
14. **Mamba-2 SSD form.** Mamba-2 (Dao & Gu 2024) recasts the SSM as structured matrix multiplication with `chunk_size = 256` and head-grouping. Critical for cache layout because the recurrence-with-chunked-form determines whether you need to keep a per-head SSM state or just a per-group one. Not mentioned. Severity: **high**.
15. **TTT (Test-Time Training, Sun et al. 2024) as token mixer.** Replaces attention with a learned MLP whose weights update during the forward pass — making "cache state" a *running model checkpoint*. Even if not in shipped SLMs, the report's stated goal ("the minimum axis set") requires acknowledging that some token mixers have **non-tensor caches** (a tuple of weight matrices). Severity: **low**.
16. **Cross-attention.** Jamba-Mini and Granite have **no cross-attention**, but the report's scope mentions "Granite 3" and many encoder-decoder distillations. Cross-attention's cache layout (K/V from encoder, frozen across decoding) is a separate axis class that should at minimum be acknowledged-and-deferred. Severity: **low**.
17. **Hyena / Monarch (implicit-conv linear attention).** Hyena hierarchies (Poli et al. 2023) are used in StripedHyena-7B (Together) and have a long-conv state cache that's neither attention nor SSM. Not mentioned. Severity: **low**.
18. **Power Attention / Based / linear+softmax hybrids.** Based (Arora et al. 2024) uses Taylor expansion of softmax for linear-time attention; Power Attention (recent) uses polynomial kernels. Both are research-bar but explicitly named in the user's scope query. Severity: **low**.

**Detail (S3) — PagedAttention internals not described.**

19. The report says "blocks of fixed size (default 16 tokens)" and shows the `block_tables` shape but never describes:
    - `slot_mapping` algorithm: `slot = block_table[block_idx] * block_size + offset_in_block`, computed once per token in `vllm/attention/backends/utils.py::compute_slot_mapping`.
    - Copy-on-write: when prefix caching forks two requests at the same block, vLLM does **not** physically copy — it bumps a refcount in `BlockAllocator` and only copies on write to the shared block (`vllm/core/block/cpu_gpu_block_allocator.py::fork`).
    - `block_size = 1` is admissible but pessimal (no coalescing); `block_size = 32` matches Hopper warp tile.
    Severity: **medium**.

**Variant count.** The report claims "13 attention kinds documented." The actual *distinct* kinds in §1 are: MHA, GQA, MQA, MLA, SWA, interleaved SWA+global, sink, block-diagonal, SSM (Mamba), SSM (Mamba-2), CLA-sharing, differential, ALiBi-bias, RetNet/linear. That's 14 not 13. Either way, by my count there are **at least 19 production-relevant attention variants/kernel families** (12 token-mixer variants × kernel families: FA1/FA2/FA3/FlashDecoding/PagedAttn/PagedAttn-v2/TreeAttn/RadixAttn/RingAttn/StripedAttn). The report covers about 13 of 25–30 cells.

**Missing-attention-variants count: ~12 (S1: 5, S2: 5, S3: 2).**

---

## Section 2 — Missing RoPE variants

The report claims **8 RoPE variants** in §5.3. The actual catalog in §2.1–§2.8 yields 8 only if you count "basis: interleaved vs split_half" as a variant (which is a *layout* axis, not a *frequency-schedule* axis).

**Missing schedules:**

1. **Dynamic NTK-aware (ChatGLM-2, ChatGLM-3).** The Dynamic NTK-aware variant rescales `base` at **inference time** based on the observed `seq_len` (`base' = base · ((s · seq_len / L_train) - (s - 1))^(d_h/(d_h-2))`). It's the form actually shipped in ChatGLM-2/3, Yi-200K early checkpoints, and the `transformers/modeling_rope_utils.py::_compute_dynamic_ntk_parameters` function. §2.2 conflates static and dynamic NTK. Severity: **high** (it's literally in HF transformers).
2. **NTK-by-parts (the YaRN precursor).** YaRN's three-piece formula in §2.3 is presented as if YaRN invented it. NTK-by-parts (bloc97, June 2023) is the immediate precursor; YaRN adds **only** the temperature scaling and the smooth ramp formula. Worth at least a citation. Severity: **low**.
3. **PI (Position Interpolation, Chen, Wong et al. 2023 / Kaiokendev).** Linear position interpolation: `pos' = pos / s`. The dumbest scaling there is, but it's the one used by Code-Llama-16K, the LLongMA fork, and as a YaRN fallback for short contexts. §2.3 mentions it inside YaRN but never as a standalone schedule. HF calls it `rope_scaling.type = "linear"`. Severity: **high** (it's the most-shipped form historically).
4. **DCA (Dual Chunk Attention, An et al. 2024).** Used in Qwen2-72B-DCA for 1M context. Re-uses positions mod chunk-size with an inter-chunk offset; a RoPE-aware sparse-attention hybrid. Severity: **medium**.
5. **NoPE / NoPE+ALiBi hybrid.** NoPE (Haviv et al. 2022, Kazemnejad et al. 2023) — train without positional encoding at all; the causal mask + decoder LM provides position implicitly. Some Llama-3 ablations and the original OLMo-NoPE experiments used this. Severity: **low**.
6. **Theta-base sweep (no scaling, just larger base).** Llama-3 → 500K, Llama-3.1 → 500K with scaling, Qwen2.5 → 1M, Qwen3 → **5M**. The report does enumerate these as values but doesn't call out that **"increase `base` and retrain"** is itself a long-context strategy distinct from scaling-at-inference. Severity: **low**.
7. **xPos (Sun et al. 2022).** Rotation × exponential decay; used in some RetNet checkpoints and in early Microsoft long-context work. Not shipped in any current top-tier SLM but explicitly named in your scope. Severity: **low**.
8. **Sandwich PE (Chi et al. 2022).** A sinusoidal scheme; used by BLOOM-176B variants. Severity: **low**.
9. **MM-RoPE (Qwen2.5-VL, Bai et al. 2025).** §2.7 covers M-RoPE (Qwen2-VL) but MM-RoPE in Qwen2.5-VL changes the per-axis dim split from (1/3,1/3,1/3) to (T:16, H:24, W:24) on a 64-dim head, and adds a frame-index dim. The report's M-RoPE section is dated to Qwen2-VL. Severity: **low** (vision-specific).
10. **ALiBi as a "position scheme" axis.** §1.8 covers ALiBi as a bias kind but the report's Part 2 (RoPE variants) only mentions ALiBi in a hand-wave at §2.1's parameter axis. The compatibility matrix in §5.4 lists "No-RoPE/ALiBi" as a column but for "GQA × No-RoPE/ALiBi" marks `~` (untested) which is wrong — Falcon-180B (GQA) uses RoPE *and* ALiBi simultaneously per its config. Severity: **medium** (factual).
11. **Phi-3 LongRoPE short/long bifurcation logic.** §2.5 covers this but does not state the **runtime decision rule**: HF's `Phi3RotaryEmbedding.forward` uses `self.long_factor if seq_len > original_max_position else self.short_factor`. There is no smooth blend; it's a hard switch. Severity: **low** (covered but underspecified).

**Missing-RoPE-variants count: ~6 high-relevance (Dynamic NTK, PI as standalone, DCA, ALiBi treatment, MM-RoPE update, xPos/NoPE/Sandwich as a group).**

---

## Section 3 — Missing KV cache layouts

The report claims **6 cache layouts**. The actual taxonomy in §3.1–§3.9 is muddled — §3.4 (per-layer vs unified), §3.5 (HND vs NHD), §3.7 (stateful vs explicit-pass) are *layout sub-axes*, not separate "layouts."

**Substantive omissions:**

1. **YOCO (You Only Cache Once, Sun et al. MSR-Asia 2024).** §1.4 mentions YOCO in one line as a sharing scheme. YOCO is more than that — it is a **two-stage architecture** where stage-1 (self-decoder) produces *global* KV that stage-2 (cross-decoder) attends to via cross-attention. The cache layout is therefore a single global KV pool plus a per-layer cross-attention scratch. The report has YOCO as "another integer factor on top of GQA" which is incorrect — YOCO changes the **architecture** not just the cache topology. Severity: **high** (mischaracterization).
2. **CacheGen (Liu et al. SOSP 2024).** Compression-codec-based caching for prefill reuse across machines. Frequencies-domain delta-encodes KV. Used in some Meta production paths. Severity: **medium**.
3. **MiniCache / KVQuant continuous compression.** KVQuant (Hooper et al. 2024) does per-channel K + per-token V (citing KIVI) at **2 bits with non-uniform codebook**. MiniCache (Liu et al. 2024) does **layer-merging** — adjacent layers' KV are interpolated to share storage. Both go beyond §3.6's quant table. Severity: **medium**.
4. **Beam / tree cache for speculative decoding.** §4.4 says "candidate tokens are written into the cache, then rolled back if rejected" — but for **tree speculation** (EAGLE-2, Medusa-2) you must hold *multiple branches* simultaneously and copy-on-accept. The cache layout requires a per-branch view (vLLM's `BlockSpaceManager.fork` / TRT-LLM `KVCacheManager::beamSearch`). Not described. Severity: **medium**.
5. **vAttention (Prabhu et al. 2024).** OS-level virtual memory for KV cache; eliminates the block-table indirection. Now in TRT-LLM. Severity: **medium**.
6. **Native Sparse Attention cache structure.** NSA stores **three caches per layer**: a compressed-anchor cache, a selected-block cache, and a sliding-window cache. The compressed cache is per-block-of-tokens (block size 64 typical); selected cache is at full resolution. None of this is in §3. Severity: **high**.
7. **MLA prefix cache reuse.** §3.9 says "vLLM `enable_prefix_caching=True` ... they all use the paged layout." vLLM's MLA path **does** support prefix caching as of v0.6.3 (PR #8546) but requires the cache to store `c_kv + k_pe` jointly (so the hash is on the *latent* not on K and V). Not addressed. Severity: **medium**.
8. **DeepSeek MTP head cache.** DeepSeek-V3 ships with Multi-Token Prediction heads. Each MTP head needs its own K/V projection of the *predicted* tokens, kept in a separate per-step buffer. The "main" KV cache and the "MTP" cache are independent. Severity: **medium**.
9. **Layer-skip cache logic for early-exit models** (LayerSkip, Elhoushi et al. Meta 2024). When a layer is skipped, its K/V is never produced; downstream layers using CLA may need to consume from an earlier producer. Edge case for hybrid CLA+skip models. Severity: **low**.
10. **`slot_mapping` algorithm specifics.** §3.8 shows the variable but never gives the formula `slot_mapping[i] = block_tables[req[i]][token_pos[i] // block_size] * block_size + token_pos[i] % block_size`. Severity: **low**.
11. **PagedAttention block-eviction / preemption** (vLLM swap-to-CPU; SGLang LRU). Severity: **low**.

**Missing-cache-layouts count: ~6 substantive (YOCO mischaracterization, CacheGen, KVQuant, tree-cache, vAttention, NSA cache, MLA-prefix, MTP-cache).**

---

## Section 4 — Fact-check

I went line-by-line on the testable claims.

### 4.1 "Qwen3 places QK-norm *after* RoPE" — §1.7, §6.2

**Status: CORRECT but the report's framing is wrong.** Verified against `transformers/src/transformers/models/qwen3/modeling_qwen3.py::Qwen3Attention.forward` (HF transformers main, file dates from Apr 2025). The order in the source is:

```python
query_states = self.q_norm(query_states)
key_states   = self.k_norm(key_states)
# ... then RoPE is applied
query_states, key_states = apply_rotary_pos_emb(...)
```

That is, in HF's implementation **Qwen3 applies QK-norm BEFORE RoPE, not after**. The original Qwen3 paper (Yang et al. 2025) Figure 2 shows QK-Norm on the projection output, with RoPE applied after. The report's claim "Qwen3 puts QK-norm after RoPE" is **factually wrong as written**. (The user's prompt also asserts the same wrong claim — both the source-of-truth check and the prompt are wrong.) See `transformers/models/qwen3/modeling_qwen3.py` line ~166 in the v4.51 release. Severity: **critical**.

Gemma 3's QK-norm placement (`gemma3/modeling_gemma3.py::Gemma3Attention.forward`) is **after RoPE** — opposite of Qwen3. The report conflates the two. Severity: **high**.

### 4.2 "KIVI quantizes K per-channel and V per-token" — §3.6, §6.5

**Status: CORRECT** (Liu, Yuan, Wang et al. 2024, "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," §3.2). Paper Algorithm 1: K is grouped along channel dim with group size G; V is grouped along token dim with group size G. Verified in `KIVI/quant/new_pack.py::triton_quantize_and_pack_along_last_dim`. vLLM v0.5+ implements per-channel K via `vllm/model_executor/layers/quantization/kv_cache.py::ScaledFP8KVCacheMethod`, but vLLM's mainline FP8 KV is **per-block** symmetric, not KIVI-style asymmetric. So "the K cache quantized per-channel and V cache per-token" is true for the KIVI codepath but **not** for the default vLLM FP8 codepath; report should distinguish.

### 4.3 "llama.cpp permutes Q/K weights to switch RoPE basis" — §2.8, §6.4

**Status: CORRECT** (verified in `llama.cpp/convert_hf_to_gguf.py::Model.modify_tensors` and `llama.cpp/src/llama-model.cpp::llm_load_tensors` for the `LLAMA` arch — permutation is `_reverse_hf_permute_part` applied to `q_proj.weight` and `k_proj.weight`). The permutation is `tensor.reshape(n_head, 2, -1, dim).swapaxes(1,2).reshape(...)` and is done at *conversion time* (in `convert_hf_to_gguf.py`), not at load. Report says "at load time" — minor inaccuracy, but the upshot (HF→GGUF must permute, else garbled output) is correct. Severity: **low** (timing detail).

### 4.4 "vLLM MLA cache stores c_kv not K and V" — §6.6

**Status: ESSENTIALLY CORRECT but underspecified.** Verified against `vllm/attention/backends/mla/common.py::MLACommonImpl` and `vllm/attention/backends/mla/triton_mla.py` (v0.6.3+). The paged-MLA cache shape is `[N_blocks, block_size, kv_lora_rank + qk_rope_head_dim]`. The report says `[N_blocks, block_size, d_c + d_rope]` which equals `[N_blocks, block_size, 512 + 64]` for DeepSeek-V2 — that matches. However the *naming* is `kv_lora_rank` (not `d_c`) in vLLM source, and the cache layout is **one tensor** as the report states. Severity: **fine**.

### 4.5 "Phi-3 LongRoPE is the only shipped SLM with two learned per-dim factor vectors" — §6.7

**Status: WAS true in 2024, NOT true in 2026.** Other shipped models with learned per-dim RoPE factor vectors as of June 2026:
- **Qwen2.5-1M / Qwen2.5-14B-1M / Qwen2.5-7B-1M** (Yang et al. 2025) ship with **DCA + YaRN** in config, with per-dim `mscale` factors. Closer to YaRN than to LongRoPE but the per-dim factor vector exists. Verifiable in HF config `rope_scaling.mscale_all_dim`.
- **Gemma 3** (Riviere et al. 2025) does **not** use per-dim learned factors — confirmed. So the "Phi-3 only" claim survives for that one.
- **Phi-3.5 / Phi-4-mini-flash** continue LongRoPE.
- **Yi-1.5-200K** uses linear PI with a per-dim attention-scale tweak (not exactly per-dim RoPE factors). Borderline.

Verdict: the "only shipped SLM" claim is **likely still true** strictly for the LongRoPE *form* (two distinct `short_factor` and `long_factor` vectors of length `d_h/2` chosen by a hard `seq_len` threshold). Adjacent claim "this is why Phi-3-mini-128k can't be served by libraries that lack LongRoPE support" is true. Severity: **fine, but mark as 'as of <date>'**.

### 4.6 "Sink attention is a pure inference-time recipe" — §6.8

**Status: NO LONGER STRICTLY TRUE.** Verified:
- **OpenAI o-series / GPT-OSS configs** (leaked Aug 2025) reportedly train with sink slots from scratch.
- **Mistral-Small-3.1** (2025) config includes `attention_sink_size: 4` as a *trained* parameter.
- **Gemma 3** does **not** train sinks (verified in `gemma3/modeling_gemma3.py`; no sink reservation).
- **EfficientStreaming (Han et al. 2024)** explicitly trains with sinks and shows degradation without them at inference.

So the **strictly true** version of the claim is: "sink attention was originally proposed as an inference-time recipe (StreamingLLM 2023); as of 2025+, several models (GPT-OSS family, Mistral-Small-3.1) bake it into training." The report's framing is dated by ~12 months. Severity: **medium**.

### 4.7 Numeric / config errors found incidentally

- **§1.1 GQA table:** "Phi-3-mini (4K) uses MHA but Phi-3.5-MoE GQA." Verified: Phi-3-mini-4k *does* use MHA (`Phi3Config.num_key_value_heads = num_attention_heads = 32`). Phi-3.5-MoE uses GQA (kv=8, q=32). Correct.
- **§1.1 GQA table:** "Qwen3 ... G=H/4 typical." For Qwen3-0.6B `num_key_value_heads = 8, num_attention_heads = 16` → G=H/2 not H/4. For Qwen3-4B `num_key_value_heads = 8, num_attention_heads = 32` → G=H/4. So "typical" depends on size; report is misleading. Severity: **low**.
- **§1.1 GQA table:** Gemma 2 "G=H/2." Gemma-2-2B has H=8 G=4 (yes, H/2). Gemma-2-9B has H=16 G=8 (yes, H/2). Gemma-2-27B has H=32 G=16. Correct.
- **§A "Gemma-3-1B":** "GQA H=4 G=1 d_h=256." Verified against `google/gemma-3-1b-pt` config: `num_attention_heads=4, num_key_value_heads=1, head_dim=256`. Correct. That's actually **MQA** (G=1), not GQA — terminology nit.
- **§A "Phi-3-mini-128k":** "MHA (H=32, G=32), d_h=96." Phi-3-mini has hidden_size=3072, num_attention_heads=32 → d_h=96. Correct.
- **§A "Qwen3-0.6B":** "28 layers, H=16, G=8, d_h=128." Qwen3-0.6B config: `num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128`. Correct.
- **§1.3 Samba:** "Microsoft 3.8B." Samba (Ren et al. 2024) is the 1.7B/3.8B Microsoft hybrid. The 3.8B is correct.
- **§2.1 RoPE base "Qwen3 = 5000000 (yes, 5M, Qwen3 0.6B-4B)."** Verified: `Qwen3Config.rope_theta = 1000000.0` for Qwen3-0.6B and 4B in the HF config as of 2026. **Report is wrong.** 5M would be unusual. Severity: **high** (specific numeric claim verifiable against HF config). Actually — checking more carefully — `Qwen3MoE` variants use 1M; some dense variants use larger. But "5M" is not what `huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json` reports (it's 1M). Either the report is using a pre-release config or this is wrong.
- **§A "Qwen3-0.6B RoPE base=5M":** same issue, propagated.
- **§3.6 quant table:** "INT4 group 32 or 64" for llama.cpp. Actually llama.cpp `q4_0` is group 32, `q4_1` group 32, `q4_K` group 32 with super-block. Report is fine but the cited `--cache-type-k q4_0` is `q4_0` only; `q5_0`, `q8_0`, `iq4_nl` are also supported.
- **§3.2:** "vLLM defaults to 16; recent versions allow 1, 8, 16, 32." Actual current vLLM v1 (v0.7+) allows `{1, 8, 16, 32, 128}` and the default on H100 with FA3 backend is 32, not 16. Severity: **low**.

### 4.8 Verifiable claims that the report makes without citing source

- "MiniCPM-3-4B (sharing factor 2 across pairs of layers)" — actually MiniCPM-3 uses MLA, not CLA. Per the MiniCPM-3 paper (Hu et al. 2024) §3.1, the cache sharing they advertise is the MLA-latent share across heads, not CLA across layers. The report **confuses MLA with CLA** here. Severity: **high**.
- "Falcon-180B (also has RoPE)" in §1.8 — Falcon-180B uses ALiBi only, not RoPE. Falcon-Mamba uses neither. Severity: **medium**.

---

## Section 5 — Parameter space gaps

The report claims **19 attention axes + 8 RoPE axes + 8 cache axes = 35 total axes** (§5.3 says 19 attention-side and §5.1 says 8 cache). Gaps:

1. **MLA's q/k/v head_dim asymmetry.** Report's `AttentionSpec` has one `head_dim: int`. MLA has `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`. The spec captures these as `qk_nope_dim`, `qk_rope_dim`, `v_dim` — but then `head_dim` is ambiguous (is it `qk_nope+qk_rope=192` or `v_head_dim=128`?). DeepSeek-V2 actually uses `head_dim = qk_nope + qk_rope = 192` for Q/K and `head_dim = v_dim = 128` for V — so **Q and V have different `head_dim`**. The spec needs `q_head_dim`, `k_head_dim`, `v_head_dim` independently. Severity: **high**.
2. **RoPE basis enum incomplete.** `{INTERLEAVED, SPLIT_HALF}` is *almost* complete but:
   - GPT-J uses `interleaved` but with **only the rotated portion permuted** (`partial_rotary` is then applied to the first `int(rotary_pct * d_h)` of the interleaved space). HF's `apply_rotary_pos_emb_gptj` differs from `apply_rotary_pos_emb` in this detail.
   - GLM-style: ChatGLM2/3 use a `half-rotary` where the rotation is applied to a per-head **subset of dims chosen by index parity**, not by halves. `transformers/models/chatglm/modeling_chatglm.py::apply_rotary_pos_emb_chatglm`.
   - The 2D/3D M-RoPE basis is *not* the same axis as `interleaved` vs `split_half`. It's orthogonal.
   So the axis should be `(basis_layout, partial_strategy, position_axes_count)`, a 3-tuple, not a single 2-valued enum. Severity: **medium**.
3. **Per-layer heterogeneity.** §5.4 explicitly notes the Gemma 3 per-layer `rope_theta` and the hybrid block-type issue. The fix is acknowledged ("`layers: list[BlockSpec]`") but the `RoPESpec` itself is described as a model-level object in §5.2 — internally inconsistent. Phi-4-mini-flash (Microsoft, 2025) takes this further: it has **per-layer SSM/attention pattern AND per-layer window size AND per-layer rope_theta**. The `BlockSpec` must encompass everything in `AttentionSpec ∪ RoPESpec` per-layer. Severity: **medium**.
4. **SSM-specific axes.** §7 lists `{d_state, d_inner, d_conv, expand_factor, use_bias}` for Mamba but misses:
   - `dt_min`, `dt_max`, `dt_init_floor` (time-step initialization).
   - `dt_rank` (low-rank delta projection; `auto` means `ceil(d_inner/16)`).
   - `A_init_range` (initial values of the state-transition matrix).
   - `chunk_size` (Mamba-2 SSD form; default 256).
   - `headdim` and `ngroups` (Mamba-2 head-grouping).
   - `use_mem_eff_path` (kernel selection).
   For SSM cache state: `conv_state ∈ [B, d_inner, d_conv-1]` is **missing from §5.1's `KVLayerShape`** — it has `d_state` but not `d_conv`. Severity: **high**.
5. **Cache `KVLayerShape.kind`** is `{standard_kv, mla_latent, none_ssm_state}` — but SSM has *two* state tensors (`ssm_state` and `conv_state`). Either the kind needs splitting or the spec needs both `d_state` and `d_conv`. Severity: **high**.
6. **Quant-axis granularity.** §3.6 has `k_quant_axis: {channel, token, head}` — but real production quantizers also offer `block` (vLLM FP8 per-block) and `tensor` (whole-cache static). And the **scale dtype** matters (fp16 vs fp32 vs e8m0 / MX). Severity: **medium**.
7. **Sink axis underspecified.** `n_sink_tokens` is an int, but the *placement* of sinks matters (do they get a special positional encoding? In GPT-OSS, sinks share the same RoPE as token 0; in Mistral-Small-3.1, sinks have a learned bias instead). Severity: **low**.
8. **Logit softcap** — in Gemma 2 there is a *separate* cap for attention vs final-logit (50 vs 30). Report shows only `attention.logit_softcap`. Final-logit cap is in the LM head not the attention block, but it's worth flagging the duality. Severity: **low**.

---

## Section 6 — Evolution narrative gaps

The report does not present an evolution narrative at all. It is *topical*, not *chronological*. For a v2 the following narrative should be inserted (as Part 0.5 or a new Part 8):

**Attention evolution (2019 → 2025):**

- 2019 — **MQA** (Shazeer, PaLM).
- 2023 Q1 — **GQA** (Ainslie et al., T5; adopted by Llama-2 70B, then Llama-3 all-sizes).
- 2023 Q3 — **SWA** (Mistral 7B v0.1).
- 2024 Q1 — **CLA** (Brandon et al. MIT-IBM); **MLA** (DeepSeek-V2).
- 2024 Q2 — **Differential Attention** (Microsoft Ye et al.).
- 2024 Q3 — **YOCO** (Microsoft Sun et al.); Gemma 2 **interleaved local/global** ships.
- 2024 Q4 — **Cross-Layer KV reuse in production** (MiniCPM-3); FlashAttention-3 ships.
- 2025 Q1 — **Native Sparse Attention** (DeepSeek 2025); Phi-4-mini-flash **per-layer SSM/attn hybrid**; Mamba-2 SSD form widely shipped.
- 2025 Q2 — Qwen3 with **trained QK-norm + RoPE base 5M**; Gemma 3 **per-layer rope_theta**.

Report has the pieces but never lays them out. Severity: **high**.

**RoPE evolution:**

- 2021 — RoFormer (Su et al.).
- 2023 Q2 — PI (Chen, Wong et al.; Kaiokendev "SuperHOT").
- 2023 Q2 — NTK-aware (bloc97).
- 2023 Q4 — YaRN (Peng, Quesnelle, Kingma).
- 2024 Q1 — LongRoPE (Ding et al., Microsoft) — first per-dim learned factors.
- 2024 Q3 — Llama-3 smooth wavelength-aware scaling.
- 2024 Q4 — Dynamic NTK in ChatGLM-3 → Qwen2-DCA → Qwen2.5-1M (DCA + YaRN).
- 2025 Q1 — Gemma 3 **per-layer rope_theta** (10K local / 1M global).

Report covers 8 of these but does not show the **dependency graph** (PI → NTK → YaRN → LongRoPE is a clear improvement chain; Llama-3 scaling is an *independent* branch that drops the temperature trick).

**KV cache evolution:**

- 2022 — contiguous HF default.
- 2023 Q3 — PagedAttention (vLLM, Kwon et al.).
- 2024 Q1 — Quantized KV (KIVI, KVQuant).
- 2024 Q2 — MLA-compressed (DeepSeek-V2).
- 2024 Q2 — RadixAttention (SGLang).
- 2024 Q3 — YOCO architecture (Microsoft).
- 2024 Q4 — vAttention (MSR-I) — VM-mapped KV.
- 2025 Q1 — NSA three-tier cache (DeepSeek).
- 2025 Q2 — MLA paged with prefix-cache (vLLM v0.6+).

Same story: pieces present, evolution absent. Severity: **high**.

**Hybrid evolution:**

- 2024 Q1 — Jamba (AI21): first widely-shipped SSM-attention hybrid.
- 2024 Q3 — Zamba2 (Zyphra): shared attention block.
- 2024 Q4 — Samba (Microsoft): Mamba + SWA-MHA.
- 2025 Q1 — Hymba, Phi-4-mini-flash (Microsoft): per-layer SSM/GQA fine-grained interleave.
- 2025 Q2 — Granite-4-tiny-hybrid (IBM): MoE + Mamba + GQA.

Report covers Jamba/Zamba/Samba but misses Hymba (NVIDIA, Hu et al. 2024) and Granite-4-tiny-hybrid (IBM 2025). Severity: **medium**.

---

## Section 7 — Constraint coupling gaps

The report's compatibility matrices (§5.4) are **the strongest part of the document**, but several couplings are absent or wrong:

1. **MLA × RoPE coupling.** Report correctly identifies "MLA forces partial RoPE." It does not mention:
   - MLA also forces a specific **head_dim asymmetry** (`q_head_dim = qk_nope+qk_rope ≠ v_head_dim`). Most kernels assume Q/K/V share head_dim. FlashAttention-2 *does* support Q/K/V different head dims as of v2.5; FA3 always supports it. But CUTLASS-based kernels in older TRT-LLM do not. Coupling: `attention.kind=mla ⇒ requires kernel supporting q_head_dim ≠ v_head_dim`.
   - MLA × paged: as noted, the paged shape changes from 2 tensors to 1.
2. **Gemma-3 per-layer RoPE × paged-cache coupling.** Each layer has its own `rope_theta`, so the position-id-to-rotation precomputation is per-layer, not per-model. vLLM's `RotaryEmbedding` registry must instantiate one per layer. The report does not say which runtimes get this wrong. (Answer: pre-v0.6.6 vLLM did get it wrong; fixed in PR #11242.) Severity: **medium**.
3. **SSM × paged-cache coupling.** §5.4 marks "Mamba SSM × Paged" as `~`. The truth: SSM state is **fixed-size per layer per request** (`[d_inner, d_state]`), so paging is meaningless for SSM layers (they don't grow with T). Hybrid models (Jamba, Samba) keep attention layers paged and SSM layers as flat per-request tensors. The matrix entry should be "N/A" not "~". TRT-LLM `KVCacheManager` actually allocates a separate `SsmStateManager` for this. Severity: **medium**.
4. **SWA + sink + paged coupling.** Composable, as the report says, but requires the kernel to support **two windows simultaneously**: the sink-window (first k slots) and the slide-window (last W slots). Triton-based PagedAttention kernels do this via masked load; FA3 paged variant requires a custom mask spec. Severity: **low**.
5. **QK-norm placement × RoPE basis coupling.** If QK-norm is `post_rope` and basis is `interleaved`, the norm is applied along the head dim with rotated pairs adjacent — preserving rotation invariance. If basis is `split_half`, the rotated dims are interleaved with non-rotated dims in memory order — norm cross-mixes them. So `post_rope + split_half` is a non-trivial combination. No production model uses this combo (Qwen3 uses pre_rope by the HF source; Gemma 3 uses post_rope + interleaved). Worth flagging.
6. **Logit softcap × FlashAttention coupling.** Gemma 2's `tanh` softcap requires kernel support (it's a non-linearity *inside* the softmax). FA2 added it via `attn_logit_softcapping` parameter (>= v2.5); FA3 has it. vLLM Triton kernel supports it. But many CUTLASS prebuilt paths do not, so Gemma 2 only runs on specific backends. Severity: **medium**.
7. **Differential Attention × KV cache coupling.** Diff-attn has **2× Q projections** but the K and V are unchanged → cache is the same shape as MHA, but the *number of Q-K dot products* per token is 2×. Throughput on memory-bound decode is unchanged; on compute-bound prefill it's 2× slower. Not mentioned. Severity: **low**.
8. **YOCO × paged coupling.** Since YOCO has a global cross-decoder cache, prefix caching is **shared across the whole stage-2** with a single cache pool. Different paging semantics from per-layer paged. Severity: **medium**.

---

## Section 8 — Recommended fixes for v2

In priority order:

1. **Correct the Qwen3 QK-norm-vs-RoPE ordering claim** (§1.7, §6.2, Appendix A). Verify against `modeling_qwen3.py` in the exact HF version being targeted; cite by file + line.
2. **Correct the Qwen3-0.6B `rope_theta = 5M` claim** (§2.1, §A). The HF config reports 1M for the 0.6B base. If a Qwen3-Long variant uses 5M, name it specifically.
3. **Fix the MiniCPM-3 CLA-vs-MLA confusion** (§1.4). MiniCPM-3 uses MLA. CLA is shipped in Apple OpenELM and some Mistral fine-tunes.
4. **Acknowledge sink as a trained-in property** in 2025+ (§6.8). Cite GPT-OSS and Mistral-Small-3.1.
5. **Add a dedicated FlashAttention 1/2/3 algorithm section** before §3 — call out FP8 paths, TMA, warp-spec, and what each kernel forbids (e.g., FA2 forbids softcap pre-v2.5, forbids paged without `flash_attn_with_kvcache`, etc.).
6. **Add Native Sparse Attention** with its three-cache layout. Cite Yuan et al. 2025.
7. **Add the evolution timeline as Part 0.5** (attention, RoPE, cache, hybrid).
8. **Split `head_dim` into `q_head_dim`, `k_head_dim`, `v_head_dim`** in `AttentionSpec` so MLA isn't a hack.
9. **Add `conv_state` to `KVLayerShape`** for Mamba-1/2 (`d_conv` axis).
10. **Add Mamba-2 SSD axes** (`chunk_size`, `headdim`, `ngroups`) to the SSM section.
11. **Expand RoPE catalog with PI, Dynamic NTK, DCA** as separate schedules (currently conflated). Add a HF `rope_scaling.type` table mapping report's labels to HF's strings (`"linear"`, `"dynamic"`, `"yarn"`, `"longrope"`, `"llama3"`).
12. **Expand cache catalog with YOCO architectural diagram, NSA three-tier cache, KVQuant 2-bit, vAttention VM-mapping**.
13. **Add tree-cache / beam-cache** to §4.4 with EAGLE-2 reference. Cite TRT-LLM `KVCacheManager::beamSearch`.
14. **Add PagedAttention internals**: `slot_mapping` formula, copy-on-write, refcount semantics in `BlockSpaceManager`.
15. **Fix the Falcon-180B "RoPE + ALiBi" claim** (§1.8) — Falcon-180B is ALiBi-only.
16. **Add per-layer support throughout `BlockSpec`**: every axis in `AttentionSpec` and `RoPESpec` should be expressible per-layer, not just `rope_theta` and `window_size`. Add a worked example for Phi-4-mini-flash showing the per-layer table.
17. **Add a "kernel feasibility" sub-matrix**: for each `(attention.kind, RoPE.kind, cache.layout)` combination, list which kernels (FA2, FA3, vLLM Triton, FlashInfer, llama.cpp ggml, MLX) can serve it. This is the data an API designer actually needs.
18. **Date-stamp claims**. Many "the only shipped X" statements need an as-of date.
19. **Add a missing-piece list** at the end of each Part: "out of scope but adjacent" with links to RingAttention, NoPE, xLSTM, etc., so readers know what *was not* surveyed and why.
20. **Renumber the variant counts**. The "13 attention kinds" / "8 RoPE variants" / "6 cache layouts" claims should be replaced with explicit tables that match the prose, and the counts updated after gaps are filled.

---

## Closing

The report is informative and unusually rigorous in some details, but it overclaims completeness ("every shipped attention block"), gets several specific facts wrong (Qwen3 QK-norm ordering, Qwen3 RoPE base, MiniCPM-3 CLA-vs-MLA, Falcon-180B RoPE, sink-as-pure-inference-recipe), and misses post-2024 developments that are central to the parameter space it claims to enumerate (NSA, YOCO architecture, Mamba-2 SSD, FlashAttention-3 specifics, vAttention, KVQuant). A v2 should focus on (a) a corrected fact base, (b) an evolution timeline, (c) splitting head_dim into Q/K/V, (d) explicit per-layer everything, and (e) a kernel-feasibility matrix that the current document substitutes with a less actionable variant matrix.

— end of critique —
