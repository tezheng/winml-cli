# 05 — KV Cache and Attention Variants Across Mainstream SLM Runtimes (v2)

**Stream:** Attention variants & KV cache designs
**Goal:** Distill the parameter axes a minimal SLM API must expose to express every shipped attention block, RoPE scheme, and KV-cache layout in mainstream small-language-model runtimes.
**Scope:** Mainstream ≤ ~14 B parameter models (Qwen3, Llama-3.x 1B/3B/8B, Phi-3/3.5/4-mini, Gemma 2/3, Mistral 7B / Small 3 / Nemo, SmolLM2/3, OLMo 2, DeepSeek-V2-Lite, Jamba-Mini, MiniCPM-3, StableLM-2, Granite 3.x/4-tiny, Hymba, Samba, Phi-4-mini-flash, RecurrentGemma, Falcon-Mamba, GPT-OSS, Qwen3-Next) and the runtimes that serve them (HF transformers, vLLM v1, SGLang, llama.cpp, MLX, MLC-LLM, TensorRT-LLM, FlashInfer, Core ML, ExecuTorch, ONNX Runtime GenAI).
**Revision history:** v2 incorporates the `05-kvcache-critique.md` review, correcting seven factual errors, adding ~12 attention variants, ~6 RoPE schedules, ~6 cache layouts, plus a chronological evolution narrative for each axis.

---

## §1 Introduction and Evolution Arc

A `forward()` on a transformer block is, to first approximation, three function calls: (1) build Q / K / V projections from the residual stream, (2) run an attention or attention-equivalent token-mixer against a cache, (3) feed-forward. Every shipped SLM differs along five orthogonal axes inside that second step:

1. **Projection shape** — number of Q heads vs K/V heads vs latent dim (MHA / GQA / MQA / MLA), and whether Q, K, V share a single `head_dim` or fan out into three independent dims.
2. **Positional embedding** — where the rotation goes, on which dims, with which θ schedule, with which extrapolation scheme.
3. **Mask shape** — full causal, sliding-window, sink-augmented, sparse (NSA), or block-diagonal (cross-doc packing).
4. **Cache layout** — how K and V (or compressed latent `c_kv`, or SSM state `h`, or 1-D conv state `conv_state`) are stored across time, across layers, and across requests.
5. **Norm placement** — pre/post on the residual; pre/post on Q and K relative to RoPE.

An API that wants to express *all* of these without per-model special casing must treat each as an axis, not a flag. Sections §3–§5 enumerate the values observed in production runtimes; §6 separates prefill / decode / chunked-prefill kernel concerns; §7 collapses to the minimum axis set; §8 documents the compatibility / constraint matrix; §9 lays out the dated evolution arc per axis; §10 surveys forward-looking research; §11 collects citations.

**The four parallel evolution arcs.** Sections of this document refer back to four chronologies that we lay out compactly here. Each arc is detailed in §9 with dates and citations.

- **Attention arc:** MHA (Vaswani 2017) → MQA (Shazeer 2019 / PaLM 2022) → GQA (Ainslie et al. 2023, Llama-2-70B Jul 2023) → SWA (Mistral 7B Sep 2023) → MLA (DeepSeek-V2 May 2024) → CLA (Brandon et al. May 2024) → Differential Attention (Ye et al. Oct 2024) → SWA / global alternation as default (Gemma 2 Jun 2024, Gemma 3 Mar 2025) → trained sink (GPT-OSS Aug 2025, Mistral-Small-3.1 Mar 2025) → NSA trainable sparse (DeepSeek Feb 2025) → SSM hybrids (Mamba 2023 → Jamba Mar 2024 → Mamba-2 May 2024 → Hymba Nov 2024 → Samba Jun 2024 → Phi-4-mini-flash 2025 → Qwen3-Next Aug 2025).
- **RoPE arc:** vanilla RoPE (Su et al. RoFormer 2021, LLaMA 1 Feb 2023) → PI / linear-scaling (Chen / Kaiokendev "SuperHOT" Jun 2023) → NTK-aware (bloc97 Jul 2023) → Dynamic NTK (ChatGLM-2 Jul 2023, InternLM-200K Aug 2023) → YaRN (Peng, Quesnelle, Kingma Sep 2023) → LongRoPE per-dim learned (Phi-3 Apr 2024) → Llama-3 smooth-wavelength (Apr 2024) → DCA (Qwen2.5-1M Jan 2025) → per-layer θ alternation (Gemma 3 Mar 2025) → iRoPE / NoPE alternation (Llama-4 Apr 2025, SmolLM3 Jul 2025).
- **KV cache arc:** contiguous (HF baseline 2020-2022) → PagedAttention (vLLM Sep 2023) → quantized KV (KIVI Mar 2024, KVQuant Mar 2024) → MLA-compressed latent (DeepSeek-V2 May 2024) → YOCO 2-stage architecture (Microsoft May 2024) → RadixAttention prefix tree (SGLang May 2024) → vAttention CUDA-VM mapping (MSR-India May 2024) → NSA three-tier (DeepSeek Feb 2025) → MTP-head cache (DeepSeek-V3 Dec 2024) → MLA paged + prefix (vLLM v0.6.3 Sep 2024).
- **Hybrid (token-mixer) arc:** pure attention (2017-2023) → Mamba pure SSM (Dec 2023) → Jamba MoE+Mamba+attn (Mar 2024) → Mamba-2 SSD (May 2024) → Zamba2 shared-attention (Jun 2024) → Samba alternating Mamba/SWA (Jun 2024) → RecurrentGemma Griffin/Hawk (Jun 2024) → Hymba parallel SSM+attn heads (Nov 2024) → Phi-4-mini-flash per-layer interleave (2025) → Granite-4-tiny MoE+Mamba+GQA (2025) → Qwen3-Next Gated DeltaNet (Aug 2025).

The remainder of this document is the cross-section of those four arcs at the point where they bear on runtime API design.

---

## §2 Methodology

Where v1 relied on the original Vaswani / Su / Shazeer / Ainslie / Kwon papers and the HF / vLLM / llama.cpp source trees, v2 additionally:

- **Re-verified seven contested HF source claims** against the actual file lines in HF transformers `main` as of 2026-06-04. The Qwen3 QK-norm-vs-RoPE order, the Gemma 3 QK-norm-vs-RoPE order, the MiniCPM-3 attention kind, Qwen3 `rope_theta`, the Falcon family ALiBi/RoPE story, the llama.cpp permutation timing, and the vLLM `DEFAULT_BLOCK_SIZE` were each pulled directly from the source-of-truth files.
- **Added the post-2024 attention literature** that v1 omitted: Native Sparse Attention (Yuan et al. arxiv:2502.11089), YOCO (Sun et al. arxiv:2405.05254), Mamba-2 SSD (Dao & Gu arxiv:2405.21060), Gated Linear Attention / DeltaNet (Yang et al. arxiv:2312.06635 and follow-ups), CacheGen (Liu et al. SOSP 2024), KVQuant (Hooper et al. arxiv:2401.18079), vAttention (Prabhu et al. arxiv:2405.04437), Hymba (NVIDIA, Hu et al. arxiv:2411.13676), xLSTM (Beck et al. arxiv:2405.04517), TTT (Sun et al. arxiv:2407.04620).
- **Cross-checked HF config values** for all numeric claims: Qwen3-0.6B / 8B / 14B `config.json`, Gemma-3-1B/4B `config.json`, Mistral-Small-3.1 `config.json`, DeepSeek-V2-Lite and DeepSeek-V3 `config.json`, MiniCPM-3-4B `config.json`, Falcon-7B / 40B / 180B `config.json`, vLLM `vllm/config/cache.py`, transformers `modeling_qwen3.py` and `modeling_gemma3.py`.
- **Date-stamped overclaims.** Where v1 said "only shipped X does Y," v2 instead says "as of 2026-06-04, no other surveyed model in scope ships Y."

Sources for §3–§5 each cite a specific file or paper. Open verification questions are flagged inline with `[VERIFY: ...]` and collected in the per-section "Open verifications" notes.

---

## §3 Attention variants — 25 entries

### 3.1 Multi-Head Attention (MHA) — Vaswani et al. 2017

The classical block. Per-layer parameters `W_q, W_k, W_v ∈ R^{d × d}` where `d = n_heads · head_dim`. Q, K, V each shape `[B, H, T, d_h]`. Cache holds `K, V ∈ [B, H, T_cache, d_h]`. Scaled-dot-product softmax with scale `1/√d_h`.

**Shipped (≤14B scope):** GPT-2, Pythia, OLMo-1, Phi-2, Phi-3-mini-4K (`Phi3Config.num_key_value_heads = num_attention_heads = 32`), StableLM-2-1.6B, SmolLM2-135M / 360M, original BERT-style encoders, TinyLlama 1.1B (per §1.4 of `01-census-critique.md`).

### 3.2 Multi-Query Attention (MQA) — Shazeer 2019, PaLM 2022

Special case `n_kv_heads = 1`. Cache per token is `2 · d_h` bytes — pathologically small. Quality regression is real at >2 B unless trained from scratch with MQA.

**Shipped:** Falcon-7B, Falcon-40B, Falcon-180B (all MQA, alibi=false; positional encoding is RoPE on Falcon-180B per its model card, learned-positional on 7B/40B), StarCoder-1, PaLM (Google), Gemma-3-1B (`num_attention_heads=4, num_key_value_heads=1, head_dim=256` — strictly MQA per the G=1 definition; v1 mislabeled this as GQA).

### 3.3 Grouped-Query Attention (GQA) — Ainslie et al. 2023

`n_kv_heads = G` where `1 < G < H`. K, V are `[B, G, T, d_h]`. At attention time, K/V are repeated `H/G` times along the head axis (or, in fused kernels, indexed by `head_idx // (H/G)`). FlashAttention-2 natively supports `kv_heads ≠ q_heads`; FlashAttention-3 makes the broadcast a kernel intrinsic. Trades `1/g` cache memory for negligible quality loss when `g ≤ 8`.

**Shipped:** Llama-3.x (1B / 3B / 8B: H=32, G=8; 70B: H=64, G=8; 405B: H=128, G=8). Qwen3-0.6B (H=16, G=8 — i.e. G=H/2), Qwen3-4B (H=32, G=8 — i.e. G=H/4), Qwen3-8B (H=32, G=8), Qwen3-14B (H=40, G=8) — v1's "G=H/4 typical" understated the variance; the actual rule is `G=8` for all Qwen3 dense variants and varies as a fraction of H. Mistral-7B-v0.2+, Gemma-2-2B (H=8, G=4 = H/2), Gemma-2-9B (H=16, G=8), Gemma-3-4B / 12B (G=H/4 typical), Phi-3.5-MoE (H=32, G=8), Phi-4-mini (H=24, G=8), MiniCPM-3 dense-attention layers, Granite-3.0, OLMo-2-1B (H=16, G=4), SmolLM2-1.7B (H=32, G=8), Mistral-Small-3.1.

### 3.4 Multi-head Latent Attention (MLA) — DeepSeek-V2 (Liu et al. 2024 §2.1)

Replace K, V with a shared low-rank latent `c_kv ∈ R^{d_c}` per token, where `d_c ≈ 4 · d_h` (DeepSeek-V2 paper uses `kv_lora_rank = 512`, head_dim 128). At attention time:

```
K_compressed = c_kv @ W_uk        # [d_c] → [H, d_nope]
V_compressed = c_kv @ W_uv        # [d_c] → [H, d_v]
```

The trick: `W_uk` is **absorbed into `W_q`** so `Q · K^T = (Q · W_uk^T) · c_kv^T`, meaning the cache only needs to store `c_kv` per token rather than `H · d_h`. RoPE does not commute with the absorption (rotation through a learned linear map is not equivalent to rotation pre-map), so MLA splits each head into:
- `qk_nope_head_dim` non-rotated dims served from latent (DeepSeek-V2: 128, MiniCPM-3: 64),
- `qk_rope_head_dim` rotated dims served from a separate per-token single-head K cache `k_pe` (DeepSeek-V2: 64, MiniCPM-3: 32),
- `v_head_dim` (independent of `qk_head_dim`) for V (DeepSeek-V2: 128, MiniCPM-3: 64).

**Critical asymmetry corrected from v1:** `q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192` (DeepSeek-V2) is **not equal to `v_head_dim = 128`**. The single `head_dim` axis assumed by v1's `AttentionSpec` is insufficient. The minimum spec needs `q_head_dim`, `k_head_dim` (== q in DeepSeek configs), `v_head_dim` as three independent fields. FlashAttention-2 v2.5+ and FA3 support asymmetric head_dim; older CUTLASS kernels do not.

**Per-token cache footprint:**
- DeepSeek-V2: `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` fp16 = 1.15 KB.
- MiniCPM-3: `kv_lora_rank + qk_rope_head_dim = 256 + 32 = 288` fp16 = 576 B.

This is the **only** mainstream variant that breaks the "K and V are independent rank-`H · d_h` tensors" assumption a textbook API normally makes.

**Shipped:** DeepSeek-V2 / V2.5 / V3 (671B / 37B-active), DeepSeek-V2-Lite (16B / A2.4B — the only mainstream "SLM-ish" MLA), **MiniCPM-3-4B** (verified against `openbmb/MiniCPM3-4B/config.json` as of 2026-06-04: `kv_lora_rank=256, q_lora_rank=768, qk_nope_head_dim=64, qk_rope_head_dim=32, num_attention_heads=40, num_hidden_layers=62`. **v1 mis-classified MiniCPM-3 as CLA / cross-layer-pair-sharing; it is MLA.**).

### 3.5 Sliding-Window Attention (SWA) — Beltagy et al. Longformer 2020, Mistral 7B Sep 2023

Causal mask additionally constrained to `i − j < W`. Cache bounded by `W` tokens per layer per request (with caveats for prefill that must hold the full prompt to honor cross-attention from outside the window).

**Shipped:** Mistral-7B-v0.1 (W=4096, all layers), Mistral-Nemo, Mistral-Small-3 (some layers SWA), Phi-3-mini-128k (blocksparse + SWA hybrid), some Mistral-Small variants.

### 3.6 Interleaved Local-Global Attention — Gemma 2 (Riviere et al. 2024), Gemma 3 (2025)

Layer-dependent mask: most layers are SWA-local, every `k`-th layer is full-causal global.

**Shipped:** Gemma 2 (1:1 alternation, W=4096 local), Gemma 3 (5:1 ratio, W=1024 local), Cohere Command-R7B (1:3), Mistral-Small-3 (some layers), and increasingly the default in 2025 SLMs.

**Gemma 3 specifics:** every 6th layer is global full-attention; the other 5 are SWA W=1024. The local layers use `rope_theta = 10_000`, the global layers use `rope_theta = 1_000_000`. `rope_theta` is therefore **per-layer**, not per-model. This is the cleanest counter-example to the "one rope config per model" assumption.

### 3.7 Sink Attention — StreamingLLM (Xiao et al. 2024)

First `k` tokens are always visible regardless of sliding-window position. Bounds cache at `W + k` per layer.

**v1 framed sink as "pure inference-time recipe"; this is no longer strictly true as of 2025.** As surveyed in `05-kvcache-critique.md` §4.6:
- **OpenAI GPT-OSS family (Aug 2025)** trains with sink slots from scratch (config exposes `attention_sink_size`).
- **Mistral-Small-3.1 (Mar 2025)** ships `attention_sink_size: 4` as a trained parameter (with a learned bias instead of vanilla absolute positional encoding for sinks).
- **EfficientStreaming (Han et al. 2024)** explicitly trains with sinks; resulting models degrade without them.
- **StreamingLLM original recipe (Xiao et al. 2024)** remains a pure inference-time recipe for any SWA model.
- **Gemma 3** does **not** train sinks (verified, no sink reservation in `gemma3/modeling_gemma3.py`).

**Corrected claim:** "sink attention was originally proposed as an inference-time recipe (StreamingLLM 2023); as of 2025+, GPT-OSS and Mistral-Small-3.1 bake it into training. Models trained with sinks do not function without them; models trained without sinks can still benefit from sinks at long-context inference."

### 3.8 Block-Diagonal (Packed Cross-Document) Attention

Mask is block-diagonal where each block is a packed pretraining sequence. Used in cross-document packed pretraining (the `packing` argument to most pretraining loaders). Runtime concern for vLLM continuous batching when packing is done at serve time.

### 3.9 Native Sparse Attention (NSA) — DeepSeek (Yuan et al. arxiv:2502.11089, Feb 2025)

Trainable hierarchical sparse attention with three branches per layer. Each token attends to:
1. **Compressed branch:** anchor tokens that summarize blocks of size `l_cmp = 32` tokens (compression-pool + small MLP per block).
2. **Selected branch:** the top-`n_sel = 16` blocks (of size `l_sel = 64`) by router score, attended at full resolution.
3. **Sliding branch:** local window of size `W_sliding = 512`.

A learned router scores blocks for the selected branch. Three caches per layer per request:
- `cmp_cache`: shape `[B, H_kv, T_cmp, d_h]` where `T_cmp = ⌈T / l_cmp⌉` — compressed anchors.
- `sel_cache`: shape `[B, H_kv, T, d_h]` at full resolution; only the selected blocks are read each step.
- `swa_cache`: shape `[B, H_kv, W_sliding, d_h]` ring buffer.

The cache footprint sums to roughly `(1/l_cmp + 1 + W_sliding/T) · 2 · H_kv · d_h` per token — at 64k context, NSA stores roughly 5% of full-attention KV (paper §4 Table 2 reports 11×–58× decoding speedup depending on context length).

**Shipped:** DeepSeek-V3.x technical preview, NSA reference implementation. **Standard SDPA / FA2 / FA3 cannot serve NSA without a custom kernel** (the gated branch-mix and the per-step block selection require bespoke gather logic).

### 3.10 Cross-Layer KV Sharing (CLA) — Brandon et al. arxiv:2405.12981, May 2024

**Multiple consecutive layers share one KV cache.** Only the producer layer computes K, V; downstream layers read from it. Distinct from MLA (which compresses within a layer) and from YOCO (which uses a 2-stage architecture).

**Shipped:** CLA reference implementation, **Apple OpenELM** (verified in census v2; OpenELM additionally has per-layer width scaling), some Mistral fine-tunes. **v1 incorrectly attributed CLA to MiniCPM-3.** MiniCPM-3 uses MLA, not CLA.

### 3.11 YOCO — You Only Cache Once (Sun et al. arxiv:2405.05254, Microsoft May 2024)

**v1 mis-classified YOCO as "another integer factor on top of GQA."** YOCO is not a cache-sharing scheme; it is a **two-stage architecture**:

- **Stage 1 (self-decoder):** the first `L/2` layers each compute their own attention and produce *one shared global KV cache* at the boundary.
- **Stage 2 (cross-decoder):** the second `L/2` layers do cross-attention against the stage-1 global cache. No layer in stage-2 maintains its own KV cache.

Cache layout: a single global `(K, V) ∈ [B, n_kv_heads, T, d_h]` pool plus per-layer cross-attention scratch (negligible). Per-token cache is `2 · n_kv_heads · d_h` regardless of layer count — i.e., the cache no longer scales with depth. For a 32-layer model, YOCO is roughly 32× cache-efficient versus baseline at full attention.

**Shipped:** YOCO-1.3B reference release, internal Microsoft variants used in early Phi-4-mini-flash experiments.

**Constraint:** YOCO + speculative decoding requires the stage-1 cache to be re-runnable for accepted-but-not-yet-committed tokens; see §8.

### 3.12 Differential Transformer (Ye et al. arxiv:2410.05258, Oct 2024)

Each head computes **two** softmax attention maps and subtracts them:
```
attn = softmax(Q1 K^T / √d_h) − λ · softmax(Q2 K^T / √d_h)
```
Doubles Q projections (2 × `d_q`), keeps K, V single. The scalar `λ_init` is initialized per layer; later layers can use larger λ.

**Shipped:** Diff-Transformer reference release (1.3B / 6.8B); used in early Phi-Diff experiments. Not in a flagship SLM yet.

### 3.13 Mamba-1 (Gu & Dao arxiv:2312.00752) — Selective State-Space Model

The recurrence is `h_t = A_t · h_{t−1} + B_t · x_t`, `y_t = C_t · h_t + D · x_t`, where `A_t, B_t, C_t, Δ_t` are **input-dependent** (selective). Concretely:

```python
# mamba_ssm/modules/mamba_simple.py (paraphrased)
x_t            = conv1d_state(conv_state, x_t)            # 1-D causal conv, kernel d_conv
A_log, B, C, D = self.x_proj(x_t).split(...)
delta          = softplus(self.dt_proj(B))                # input-dependent step
A              = -exp(A_log)                              # diagonal stable
h_t            = exp(delta · A) · h_{t-1} + (delta · B) · x_t
y_t            = C · h_t + D · x_t
```

**Cache state per layer:**
- `ssm_state ∈ [B, d_inner, d_state]` — the recurrent hidden state. Typical: `d_state = 16`, `d_inner = 2 · hidden_size`.
- `conv_state ∈ [B, d_inner, d_conv - 1]` — **1-D causal-conv state for the input mix.** Without it, Mamba inference cannot resume mid-sequence.

**v1 omitted `conv_state` entirely**; the `KVLayerShape` axis in §5.1 is corrected in §7 to include `d_conv`.

The selective scan kernel parallelizes the recurrence via chunk-parallel-scan: chunks of `chunk_size` tokens are processed with a Brent-Kung scan inside each chunk, then a cross-chunk associative-scan; this is what `selective_scan_fn` (CUDA) exposes.

**Shipped pure-Mamba:** Mamba-2.8B reference, Falcon-Mamba-7B, **Cobra** vision-language model. RWKV-5/6 use a parallel-form recurrence (closely related family).

### 3.14 Mamba-2 — SSD form (Dao & Gu arxiv:2405.21060, May 2024)

Mamba-2 recasts the SSM as **structured matrix multiplication** (the "state-space duality" form). The key change for the API:

- `A` is restricted to scalar-times-identity per head (rather than diagonal); this enables a head-grouped formulation.
- Recurrence is computed by **chunk** (typical `chunk_size = 256`) using one matmul per chunk plus a cross-chunk inter-block scan; this is 2–8× faster than Mamba-1's selective scan on Hopper.
- Per-token outputs: `headdim` and `ngroups` are new axes. `headdim = 64` typical, with `ngroups ∈ {1, 2, 4, 8}` shared `B, C` across `n_heads / ngroups` heads.

**Cache axes added:** `chunk_size`, `headdim`, `ngroups`. The `ssm_state` shape becomes `[B, n_heads, headdim, d_state]`, and the `conv_state` shape `[B, d_inner, d_conv - 1]` is unchanged. **All three SSD axes are missing from v1's `KVLayerShape`.**

**Shipped:** Mamba-2 reference, Zamba-2 1.2B / 2.7B / 7B (Zyphra), portions of Hymba (NVIDIA), Granite-4-tiny-hybrid (IBM).

### 3.15 Gated Linear Attention (GLA) — Yang et al. arxiv:2312.06635, late 2023

Linear attention `O = (Q (K^T V))` plus a learned per-token data-dependent gate that decays the running KV state:
```
S_t = G_t ⊙ S_{t-1} + K_t · V_t        # element-wise gated update
O_t = Q_t · S_t
```
`G_t` is computed from `x_t` via a small MLP. Recurrent form: cache is `S ∈ [B, H, d_h, d_v]` per layer — fixed in T, like SSM.

**Shipped:** Jamba-1.5 (some attention layers per ablation, per the Lieber et al. tech report), Zamba2 ablations, `flame` runtime.

### 3.16 DeltaNet / Gated DeltaNet — Yang & Pan arxiv:2406.06484 (DeltaNet), later "Gated DeltaNet" (2024-25)

Linear attention with the **delta rule**: `S_t = S_{t-1} − S_{t-1} K_t K_t^T + V_t K_t^T`, which removes the V component previously associated with this key and writes the new one. Provides O(1) memory associative-recall.

The "gated" variant adds a per-token decay gate `α_t ∈ (0, 1]`:
```
S_t = α_t · (S_{t-1} − S_{t-1} K_t K_t^T) + V_t K_t^T
```

**Shipped:** Qwen3-Next (Aug 2025) uses Gated DeltaNet as its primary token mixer in a hybrid with attention. Flame runtime. Microsoft's internal `RetNet-Delta` experiments.

### 3.17 RetNet — Sun et al. arxiv:2307.08621 (2023)

Replaces softmax with retention: `Q · (D ⊙ K^T V)` where `D_{ij} = γ^{i-j}` for `i ≥ j`. The recurrent form has constant-size state. Three computation modes: parallel (training), recurrent (decode), chunkwise (long sequence).

**Shipped:** RetNet 1.3B / 2.7B / 6.7B reference releases (Microsoft); not in a flagship SLM but cited as the lineage for DeltaNet and GLA.

### 3.18 Hyena / StripedHyena — Poli et al. arxiv:2302.10866, Together 2023

Implicit-conv linear attention: a long-conv parameterized by an MLP-generated filter. The cache is the previous filter outputs, not K/V.

**Shipped:** StripedHyena-7B (Together), some Mosaic variants. Out of mainstream SLM list but worth flagging in §10.

### 3.19 TTT — Test-Time Training (Sun et al. arxiv:2407.04620, Jul 2024)

Replaces the token mixer with a learned MLP whose weights update during the forward pass. The "cache state" becomes a tuple of weight matrices that get gradient-updated per token. Even if not in shipped SLMs, this implies some token-mixers have **non-tensor caches**. The minimum API for §7 must allow `cache_kind = mutable_model_weights` as a placeholder.

### 3.20 xLSTM — mLSTM / sLSTM (Beck et al. arxiv:2405.04517, May 2024)

Extension of LSTM with exponential gating and a matrix-memory variant (`mLSTM`). `mLSTM` cache is a per-head matrix `C ∈ [d_h, d_h]`; `sLSTM` cache is the standard scalar `c, h`.

**Shipped:** xLSTM-7B reference (May 2024). Out of mainstream-text-SLM list but explicitly in scope per critique S2.

### 3.21 Cross-Attention (encoder-decoder)

K, V come from a frozen encoder; only Q is computed per decode step. Cache is a single fixed-size `[B, H_kv, T_enc, d_h]` shared across all decode steps. Distinct lifetime from self-attention KV.

**Shipped:** T5 (out of SLM scope), Whisper, Donut, encoder-decoder distillations. The minimum API for §7 must allow attention blocks to declare a cache as **frozen** (no per-step write) — but no shipped scope-mainline SLM currently uses cross-attention except YOCO (which is cross-attention from stage-2 to stage-1).

### 3.22 FlashAttention v1 / v2 / v3 — the kernel-family axis

These are **kernel families** of the standard SDPA, not new attention semantics, but they are distinct enough at the API level (different dtype paths, different head-dim constraints, different mask supports) that any runtime survey must call them out.

**FlashAttention-1 (Dao et al. arxiv:2205.14135, 2022).** Online softmax with tile-IO partitioning. Single-warp tile. Cache-aware streaming via a recursion `m_new = max(m_old, m_tile)` and a running normalizer `l_new = l_old · e^{m_old − m_new} + sum(e^{logits − m_new})`. Forward only at first; backward added later. Maximum `head_dim = 128`. No support for asymmetric Q/K/V dims, no support for softcap, no support for sliding window.

**FlashAttention-2 (Dao arxiv:2307.08691, 2023).** Reorganizes the inner/outer loop so Q is outer (instead of K). Per-warp accumulators in registers instead of shared memory. `head_dim` increased to 256. Adds:
- Causal mask
- Sliding window (`window_size_left / right`)
- ALiBi slopes
- Async-load overlap on A100
- `softcap` parameter (v2.5+)
- Asymmetric Q vs K/V head_dim (v2.5+, needed for MLA)
- `flash_attn_with_kvcache` for paged decode

**FlashAttention-3 (Shah, Bikshandi et al. arxiv:2407.08608, Jul 2024).** Hopper-specific:
- TMA (Tensor Memory Accelerator) descriptors for async load/store
- Producer/consumer warp specialization via `cuda::barrier`
- `wgmma.async.aligned` for the BF16 matmul
- **FP8 E4M3 / E5M2 Q and K with BF16 accumulators** — the FP8 KV path on H100 is **FA3-only**, not FA2
- Block-quantized scale path (per-block FP8 with separate `descale` factor)

**Why this matters for the API:** FA3 changes which `(dtype, head_dim, mask_kind, softcap_on?)` combinations are admissible. v1's §4.1 says "FA2/3" without explaining that **FP8 KV + BF16 Q is admissible on FA3, forbidden on FA2**, and that softcap below FA2 v2.5 is forbidden entirely. The min API must let the runtime declare which kernel is available, and the model spec must declare which features it needs.

### 3.23 FlashDecoding / FlashDecoding++

**FlashDecoding (Dao blog, Oct 2023):** parallelizes the M=1 decode-step case across the **K dimension** (split-K decode). Standard FA2 has zero parallelism in M=1.

**FlashDecoding++ (Hong et al. arxiv:2311.01282, Nov 2023):** adds unified-max softmax (separate normalizer per K-partition that is reduced at the end), enabling longer-context decode without re-launch. This is what vLLM's `paged_attention_v2` actually implements. The decode path that matters for production cache layouts.

### 3.24 RingAttention — Liu, Zaharia, Abbeel arxiv:2310.01889 (Oct 2023)

Sequence-parallel attention: the sequence is sharded across N devices; K and V tensors are rotated through a ring of devices each step. Combined with FlashAttention's online-softmax, the per-device memory is bounded regardless of sequence length.

**Shipped:** Gemini-1.5-Pro 1M-context inference is reported to use a Ring/Striped variant; TRT-LLM long-context paths in v0.10+; some research-bar 1M context paths in vLLM.

### 3.25 Striped Attention — Brandon et al. arxiv:2311.09431 (Nov 2023)

A refinement of RingAttention that interleaves the K/V shards so each rank has a load-balanced share of causal-mask work (vanilla RingAttention has the unbalanced lower-triangle problem).

### 3.26 TreeAttention — Spector & Re arxiv:2402.13720, also Medusa-2 (2024)

Speculative-decoding kernel that exploits a **static tree mask** (typical: 5×4 = 20 candidate tokens per step, mask-attaching them in a tree). Provides ~2× speedup over a generic 2-D mask attention on the verifier forward. EAGLE-2 uses this as its kernel.

### 3.27 RadixAttention — SGLang (Zheng et al. arxiv:2406.04692, May 2024)

Allocator-side technique: maintains a **radix tree** keyed by token-id prefixes, with each node pointing to the physical block(s) that hold the prefix's KV. On a new request, the longest-matching prefix is hit, and the request's first-block pointer is set to the cached node; new tokens append into fresh blocks past the match. Copy-on-write semantics: when two requests fork at the same node, the shared blocks are reference-counted and only copied if either branch writes back into them.

**The kernel is unchanged from PagedAttention** — only the allocator differs. RadixAttention is therefore a *cache allocator*, not an attention kernel. v1 conflated this in §3.9.

### 3.28 PagedAttention v1 — Kwon et al. SOSP 2023

Cache is split into **blocks** of fixed size (default 16 tokens in vLLM, verified in `vllm/config/cache.py: DEFAULT_BLOCK_SIZE: ClassVar[int] = 16` as of 2026-06-04; the value is a static class constant and the codebase does **not** override it per platform). Each request has a block table mapping logical block index to physical block. K and V live in two large pools. The PagedAttention kernel takes `block_tables` and gathers K/V on-the-fly.

(Detailed layout is in §5.2.)

### 3.29 PagedAttention v2 / vAttention — Prabhu et al. arxiv:2405.04437 (May 2024, MSR-India)

**vAttention** uses CUDA virtual-memory mapping (`cuMemMap`, `cuMemAddressReserve`) to make the cache logically contiguous in virtual address space while physically scattered across page-sized allocations. The kernel sees a flat tensor, so no block-table indirection, no kernel modification — yet the OS-level VM machinery handles the paging.

**Trade-offs:** eliminates the gather overhead, but adds a per-token `cuMemMap` system call cost on cache growth. TRT-LLM has adopted parts of the technique (v0.11+).

---

## §4 RoPE variants — 14 entries

### 4.1 Vanilla RoPE — Su et al. arxiv:2104.09864 (RoFormer 2021)

For head dim `d_h`, frequencies `θ_i = base^(−2i/d_h)`, `i ∈ [0, d_h/2)`. At position `m`, rotate pairs `(x_{2i}, x_{2i+1})` by angle `m · θ_i`. Parameters: `base` (a.k.a. `rope_theta`), `d_rotated ∈ [0, d_h]`, basis (interleaved vs split-half — see §4.13).

**Base values across shipped SLMs (re-verified against HF configs as of 2026-06-04):**

| Model | rope_theta | Source |
|---|---|---|
| LLaMA 1 | 10_000 | original RoFormer default |
| Llama-2 | 10_000 | hf llama config |
| Llama-3 | 500_000 | apr 2024 release |
| Llama-3.1 / 3.2 | 500_000 + llama3 scaling | + smooth-wavelength scaling |
| Mistral-7B-v0.1 | 10_000 | mar 2023 |
| Mistral-7B-v0.3 | 1_000_000 | mid 2024 |
| Mistral-Small-3.1 | 1_000_000 | mar 2025 |
| Qwen2.5 | 1_000_000 | sep 2024 |
| **Qwen3-0.6B / 4B / 8B / 14B** | **1_000_000** | **verified in HF `Qwen/Qwen3-0.6B/config.json`, `Qwen3-8B/config.json`, `Qwen3-14B/config.json` (all `"rope_theta": 1000000`). v1's "5M" claim was wrong.** |
| Phi-3-mini-4k | 10_000 | apr 2024 |
| Phi-3-mini-128k | 10_000 + LongRoPE | factor vectors override base |
| Phi-4-mini (3.8B) | 1_000_000 | jan 2025 |
| Gemma 2 (all sizes) | 10_000 | jun 2024 |
| **Gemma 3** | **per-layer:** 10_000 (local) / 1_000_000 (global) | mar 2025 — only model in scope with per-layer θ |
| DeepSeek-V2 | 10_000 (+ YaRN scale=40) | may 2024 |
| SmolLM2 | 130_000 | nov 2024 |
| SmolLM3 (1.7B / 3B) | 100_000 (varies; NoPE on a subset of layers — see §4.10) | jul 2025 |
| OLMo-2-1B | 500_000 | nov 2024 |
| StableLM-2-1.6B | 10_000 | feb 2024 |
| Granite-3.0 | 10_000_000 | oct 2024 |
| Granite-3.3 / 4-tiny-hybrid | varies | 2025 |

**"Increase `base` and retrain"** is a long-context strategy distinct from inference-time scaling. The sequence 10K → 500K → 1M → 5M (initially reported but never landed) → 10M (Granite) is itself a chronological arc; see §9.

### 4.2 Position Interpolation (PI) — Chen et al. arxiv:2306.15595 and Kaiokendev "SuperHOT" (Jun 2023)

Linear position interpolation: `pos' = pos / s`. The dumbest scaling there is. HF calls it `rope_scaling.type = "linear"`.

**v1 mentioned PI only inside YaRN; v2 separates it as a standalone schedule.** PI is the most-shipped form historically: Code-Llama-16K, the LLongMA fork, Vicuna long-context variants, and as a YaRN fallback for short contexts.

### 4.3 NTK-aware (static) — bloc97 Reddit Jul 2023, formalized in YaRN

Replace `base` with `base · s^(d_h / (d_h − 2))` where `s = L_new / L_train`. Smooth interpolation: high-frequency dims rotate slower. Almost entirely superseded by YaRN / Llama-3 scaling in 2024+.

### 4.4 Dynamic NTK-aware — ChatGLM-2 Jul 2023, InternLM 200K Aug 2023

Rescales `base` **at inference time** based on the observed `seq_len`:
```python
base' = base · ((s · seq_len / L_train) − (s − 1))^(d_h / (d_h − 2))
```
This is the form in HF's `transformers/modeling_rope_utils.py::_compute_dynamic_ntk_parameters`. **v1 conflated static and dynamic NTK.** Dynamic NTK has the advantage that it gracefully degrades to vanilla when `seq_len ≤ L_train` — no inference-time penalty for short prompts.

**Shipped:** ChatGLM-2, ChatGLM-3, Yi-200K early checkpoints, InternLM-200K.

### 4.5 YaRN — Peng, Quesnelle, Kingma arxiv:2309.00071 (Sep 2023)

Three-piece interpolation (built on NTK-by-parts, bloc97 Jun 2023):
- For dims with wavelength `< L_train`: keep original frequency (extrapolate).
- For dims with wavelength `> α · L_new`: apply linear PI factor `s = L_new / L_train`.
- In between: ramp linearly.

Also applies temperature scaling `1/t = 0.1 · ln(s) + 1` on the attention softmax to compensate for entropy increase. Parameters: `scaling_factor s`, `original_max_position_embeddings`, `attn_factor`, `beta_fast`, `beta_slow`.

**Shipped:** DeepSeek-V2 (default with s=40 for 128K), Qwen2.5-1M, MiniCPM-3 (default).

### 4.6 Llama-3 smooth wavelength scaling — Apr 2024

Distinct from YaRN. Formula in `transformers/modeling_rope_utils.py::_compute_llama3_parameters`:
```python
low_freq_factor = 1.0          # below: scale by factor
high_freq_factor = 4.0         # above: identity
original_max_position = 8192
scale_factor = 8.0             # the PI factor
wavelen = 2π / freqs
inv_freq_llama = where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
# smooth ramp in between using (orig_max_pos / wavelen − low_freq_factor) /
#                              (high_freq_factor − low_freq_factor)
```
Two key differences from YaRN:
1. No softmax temperature term.
2. Wavelength gating uses absolute wavelengths, not periods relative to context length.

**Shipped:** Llama-3.1 8B / 70B / 405B, Llama-3.2 1B / 3B (all 128K context).

### 4.7 LongRoPE — Ding et al. arxiv:2402.13753 (Feb 2024)

Phi-3-mini-128k ships **two distinct frequency vectors** in config: `short_factor` (length `d_h/2`, used when `seq_len ≤ original_max`) and `long_factor` (length `d_h/2`, used when `seq_len > original_max`). Each is a learned per-dim multiplier applied as `inv_freq · factor_i`. Also has attention-scale tweak: `attn_factor = sqrt(1 + log(scale) / log(original_max))`.

**Runtime decision rule** (specified in critique §4.5, missed by v1): HF's `Phi3RotaryEmbedding.forward` uses `self.long_factor if seq_len > original_max_position else self.short_factor`. **Hard switch — no smooth blend.**

**Shipped:** Phi-3-mini-128k, Phi-3-small-128k, Phi-3.5-mini-instruct, Phi-3.5-MoE-instruct, Phi-4 (some checkpoints).

As of 2026-06-04, **LongRoPE in its strict form (two distinct learned per-dim factor vectors of length `d_h/2` with a hard `seq_len` switch) remains unique to the Phi family.** Qwen2.5-1M ships per-dim `mscale` factors in `rope_scaling.mscale_all_dim`, but the form is YaRN-with-per-dim-temperature, not LongRoPE.

### 4.8 DCA — Dual Chunk Attention (An et al. arxiv:2402.17463 / Qwen2.5-1M)

Re-uses positions modulo chunk-size with an inter-chunk offset. A RoPE-aware sparse-attention hybrid:
- Intra-chunk positions: `pos mod chunk_size`
- Inter-chunk attention: a separate `inter_chunk_offset` added per position pair
- Combined with YaRN scaling for the residual long-context

**Shipped:** Qwen2-72B-DCA for 1M context, **Qwen2.5-1M (7B / 14B)**, Qwen2.5-7B-1M.

### 4.9 Partial RoPE / Non-rotated Dims

Rotate only the first `d_rope` dims of each head, leave `d_h − d_rope` untouched.

**Shipped:**
- Phi-1 / Phi-1.5 / Phi-2: `partial_rotary_factor = 0.5`
- GPT-J: `rotary_pct = 0.25`
- GPT-NeoX-20B: `rotary_pct = 0.25`
- StableLM-2: `partial_rotary_factor = 0.25`
- **DeepSeek-V2 / V3 MLA, MiniCPM-3 MLA**: structural — only the `qk_rope_head_dim` channel is rotated, the `qk_nope_head_dim` channel is not. This is not a tuning knob but architectural (see §3.4).

### 4.10 NoPE — Haviv et al. arxiv:2203.16634 (2022), Kazemnejad et al. arxiv:2305.19466 (2023)

Train without positional encoding at all; the causal mask + decoder LM provides position implicitly.

**Shipped (mainstream as of 2025):**
- **SmolLM3 (1.7B / 3B)** alternates: every Nth layer omits RoPE entirely (per Hugging Face's SmolLM3 model card). Per-layer `apply_rope: bool` axis is therefore required — v1's spec missed this.
- **Llama-4 iRoPE** (Apr 2025): a fraction of layers are NoPE; the others use vanilla RoPE.
- Some Llama-3 ablations and the original OLMo-NoPE experiments.

### 4.11 xPos — Sun et al. arxiv:2212.10554 (2022)

Rotation × exponential decay along positions: each rotated pair additionally multiplied by `ζ^{|i−j|}` where `ζ` is a learned per-dim decay. Reduces high-frequency oscillation at long distances.

**Shipped:** RetNet uses an xPos variant; early Microsoft long-context work.

### 4.12 Sandwich PE — Chi et al. arxiv:2212.10356 (2022)

Sinusoidal scheme; used by some BLOOM-176B variants.

### 4.13 RoPE Basis: Interleaved vs Split-Half

Two equivalent-in-math but different-in-memory ways to apply RoPE:
- **Interleaved** (HF default, original RoFormer): pairs `(x_0, x_1), (x_2, x_3), ...`
- **Split-half** (GPT-NeoX, used by llama.cpp / ggml): first half rotated with second half — `(x_0, x_{d/2}), (x_1, x_{d/2+1}), ...`

These produce **different numerical outputs** unless weights are re-shuffled. **llama.cpp re-permutes Llama-arch Q and K weights at conversion time** in `convert_hf_to_gguf.py` (specifically inside the per-architecture `Model` subclasses such as `LlamaModel.modify_tensors`, via a `permute_tensor` helper that does `tensor.reshape(n_head, 2, -1, dim).swapaxes(1, 2).reshape(...)`). **v1 incorrectly said "at load time"; the permutation is convert-time** (verified against the ggml-org / llama.cpp `convert_hf_to_gguf.py` module structure). At load time, llama.cpp expects already-permuted weights.

Without conversion-time permutation, ggml output differs from HF — and crucially, **it does not crash; it degrades fluency silently**. This is a frequent source of bugs in third-party conversions.

### 4.14 Multimodal RoPE — M-RoPE (Qwen2-VL) and MM-RoPE (Qwen2.5-VL)

**M-RoPE (Qwen2-VL, 2024).** Splits RoPE dims into temporal / height / width thirds; each gets its own position index. Required for video tokens. Phi-3.5-Vision uses a similar approach.

**MM-RoPE (Qwen2.5-VL, Bai et al. 2025).** Changes the per-axis dim split from `(1/3, 1/3, 1/3)` to `(T:16, H:24, W:24)` on a 64-dim head, and adds a frame-index dim. v1's M-RoPE section was dated to Qwen2-VL only.

Not relevant for pure-text SLMs but the API must allow `rope.position_ids` to be an `[N_axes, B, T]` tensor rather than `[B, T]`. Most text-only SLMs use 1 axis.

**Position-axes count is a separate API axis** from `basis` (interleaved vs split-half) — they're orthogonal.

---

## §5 KV cache layouts — 13 entries

### 5.1 Contiguous (HF transformers default)

For each layer, K and V are separate tensors of shape `[B, n_kv_heads, T_max, d_h]` allocated upfront (when `use_cache=True` and `max_cache_len` is set) or grown via `torch.cat` (legacy `DynamicCache`).

```python
# transformers/cache_utils.py::DynamicCache
self.key_cache:   List[Tensor[B, H_kv, T_seen, d_h]]
self.value_cache: List[Tensor[B, H_kv, T_seen, d_h]]
# append: torch.cat([prev, new], dim=-2)
```

**Pros:** simple, debuggable, single index per `(layer, b, h, t, d)`.
**Cons:** O(`T_max · B`) memory always allocated; can't share across requests; reallocation on growth.

**Dtype:** model weight dtype (usually bf16) or quantized via `transformers.QuantizedCache` (HQQ / quanto, 4/8-bit).

### 5.2 Paged (PagedAttention v1 — vLLM, SGLang, TRT-LLM)

Cache is split into **blocks** of fixed size (default 16 tokens). Each request has a block table `[N_blocks]` mapping logical block index to physical block. K and V live in two large pools:

```python
# vLLM block layout (HND form)
key_cache:    Tensor[N_blocks_total, block_size, n_kv_heads, d_h]
value_cache:  Tensor[N_blocks_total, block_size, n_kv_heads, d_h]
block_tables: Tensor[num_seqs, max_blocks_per_seq]   # int32
slot_mapping: Tensor[B_tokens]                       # int64

# slot_mapping algorithm, computed once per token in
# vllm/attention/backends/utils.py::compute_slot_mapping
slot_mapping[i] = block_tables[req[i]][token_pos[i] // block_size] * block_size
                + token_pos[i] % block_size
```

**Copy-on-write semantics (v1 missed this):** when prefix caching forks two requests at the same block, vLLM does **not** physically copy — it bumps a refcount in `BlockAllocator` and only copies on write to the shared block (`vllm/core/block/cpu_gpu_block_allocator.py::fork`).

**Block sizes (re-verified 2026-06-04):**
- vLLM `DEFAULT_BLOCK_SIZE: ClassVar[int] = 16` (in `vllm/config/cache.py`).
- vLLM v1 accepts `{1, 8, 16, 32, 64, 128}`; **the default is 16** (no platform-specific override in `_apply_block_size_default`). v1's claim of "32 default on H100/FA3" was not confirmed; the value remains 16 in the upstream codebase.
- FlashInfer prefers 1 or 16. Page size 1 essentially removes paging overhead at cost of less coalesced loads.

**Shipped in:** vLLM v1, SGLang, TensorRT-LLM v0.10+, RayLLM. llama.cpp does **not** use paged attention — it uses a contiguous per-sequence cache with periodic defragmentation.

### 5.3 Ring / Circular Buffer (sliding-window models)

For Mistral-7B-v0.1 with W=4096, only `W` tokens of cache are ever needed. llama.cpp implements this as a **ring buffer** with a write head:

```c
// llama.cpp/src/llama-kv-cache.cpp (paraphrased)
struct llama_kv_cache_unified {
    std::vector<llama_kv_cell> cells;  // size = cparams.n_ctx
    uint32_t head;                     // next write position
    uint32_t size;                     // n_ctx
};
struct llama_kv_cell {
    llama_pos pos;                     // **absolute position, not slot index**
    std::set<llama_seq_id> seq_id;
};
```

For Mistral, llama.cpp tracks `n_swa` and rotates positions modulo W. Once `T > W`, old slots are overwritten; attention mask is computed from `(absolute_pos % W)`. **The absolute `pos` is critical for RoPE** — slot index would silently break long-context behavior.

For Gemma 2 / 3 interleaved layers, llama.cpp allocates **two pools**: a full-context pool for global layers and a W-sized ring for local layers (`llama_kv_cache_iswa`, since llama.cpp b3389).

### 5.4 vAttention — VM-Mapped Cache (Prabhu et al. MSR-I 2024, arxiv:2405.04437)

Uses CUDA virtual-memory mapping (`cuMemMap`, `cuMemAddressReserve`) to make the cache logically contiguous in virtual address space while physically scattered across page-sized allocations:

```
1. Reserve a large virtual address range for the cache (cuMemAddressReserve).
2. Pre-allocate a pool of physical KV pages (cuMemCreate, GPU memory).
3. On cache growth, cuMemMap a new physical page to the next virtual offset.
4. Kernel sees a flat tensor; no block-table arg needed.
5. On cache shrink / request finish, cuMemUnmap returns pages to pool.
```

**Trade-offs:** eliminates the block-table indirection (gather overhead = 0); adds `cuMemMap` syscall cost per page-fault on growth (amortized over `block_size` tokens). TRT-LLM v0.11+ adopted the technique. Eliminates the kernel-side change PagedAttention requires.

### 5.5 NSA Three-Tier Cache (DeepSeek 2025)

Per layer per request, NSA holds **three independent caches** corresponding to its three attention branches (§3.9):

```
cmp_cache:   [B, H_kv, T_cmp,    d_h]     # T_cmp = ceil(T / l_cmp), l_cmp=32
sel_cache:   [B, H_kv, T_full,   d_h]     # full resolution; only top-n_sel blocks read
swa_cache:   [B, H_kv, W_sliding, d_h]    # ring buffer, W_sliding=512
```

The compressed-anchor cache is small (`T / l_cmp` slots per token); the selected-block cache is at full resolution but reads only the chosen blocks per step. Total per-token footprint scales sub-linearly in `T`.

**Constraint:** standard SDPA / FlashAttention kernels cannot serve NSA without a custom kernel that handles per-step block selection plus the three-branch softmax mixture.

### 5.6 MLA Cache — Single Latent Tensor (DeepSeek-V2)

Unlike standard (K, V) → two tensors, MLA's cache is **one tensor** of shape:

```
[N_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
= [N_blocks, block_size, 576]   for DeepSeek-V2
= [N_blocks, block_size, 288]   for MiniCPM-3
```

(Naming in `vllm/attention/backends/mla/common.py::MLACommonImpl` is `kv_lora_rank`, not the paper's `d_c`.)

The "K cache" and "V cache" abstraction collapses into one for MLA. Block tables and slot mapping still apply; the kernel is `triton_mla.py` in vLLM v0.6.3+, with prefix-cache reuse via hashing on the **latent**, not on K and V.

### 5.7 YOCO Global Cache (Microsoft 2024)

Single global KV pool produced by the self-decoder (stage 1), consumed by all stage-2 layers via cross-attention. Per-token cache is `2 · n_kv_heads · d_h` independent of total layer count. For a 32-layer YOCO, this is roughly 32× cache-efficient versus per-layer baseline.

**Constraint with speculative decoding:** rejected speculative tokens require the stage-1 cache to be rolled back atomically. TRT-LLM handles this via a "shadow" stage-1 cache for speculation candidates.

### 5.8 Per-Layer Separate vs Unified

- **Per-layer separate** (HF default, vLLM, SGLang): one cache tensor per layer. Layers can have different shapes (useful for CLA / hybrid / SSM mixing).
- **Unified** (some Core ML / MLX builds): all layers concatenated into a single big tensor `[n_layers, ...]`. Simpler kernel dispatch but forces uniform shape; **incompatible with hybrid Mamba/attention, with interleaved cache sizes (Gemma 3), with CLA, and with YOCO.**

### 5.9 HND vs NHD Memory Layout

For a given (B, T, H, D) cache, two memory orders matter:

- **HND** (`[B, H, T, D]`): heads outer, then time, then dim. Default in HF, vLLM PagedAttention, FlashAttention-2/3. Good for per-head sequential reads.
- **NHD** (`[B, T, H, D]`): time outer, then heads. Used by some Triton / FlashInfer kernels and by Mamba state for compatibility. Better for batched gather-by-position.

FlashInfer's `BatchPrefillWithPagedKVCache` accepts either via a `kv_layout = "NHD" | "HND"` flag. The conversion is a transpose — free in PyTorch view but materialized in a copy if the consumer is non-contiguous.

### 5.10 Quantized KV — KIVI, KVQuant, CacheGen

| Scheme | K bits | V bits | K axis | V axis | Notes |
|---|---|---|---|---|---|
| **FP8 E4M3** (vLLM, TRT-LLM, SGLang) | 8 | 8 | per-block | per-block | static or dynamic per-token scale |
| **FP8 E5M2** | 8 | 8 | per-tensor | per-tensor | H100 default path |
| **INT8 per-token** (HF QuantizedCache) | 8 | 8 | per-token | per-token | HQQ backend |
| **INT4 group** (llama.cpp `q4_0`) | 4 | 4 | group-32 | group-32 | `--cache-type-k q4_0` |
| **KIVI 2-bit** (Liu et al. arxiv:2402.02750) | 2 | 2 | **per-channel** | **per-token** | asymmetric (see below) |
| **KVQuant 2-bit** (Hooper et al. arxiv:2401.18079) | 2 | 2 | per-channel | per-token | + non-uniform codebook, vector-Q on outliers |
| **CacheGen** (Liu et al. SOSP 2024) | variable | variable | frequency-domain | frequency-domain | delta-encoded compression for prefill reuse across machines |
| **MiniCache** (Liu et al. arxiv:2405.14366) | n/a | n/a | layer-merging | layer-merging | adjacent layers' KV are interpolated to share storage |

**KIVI asymmetry (re-verified, correct in v1).** Per Liu, Yuan, Wang et al. 2024 §3.2 and `KIVI/quant/new_pack.py::triton_quantize_and_pack_along_last_dim`: K is grouped along the **channel** dim with group size G; V is grouped along the **token** dim. Intuition: K participates in `Q · K^T` (channel-dim is the inner dim, so per-channel quantization preserves row sums), V participates in `attn · V` (token-dim is the inner dim, so per-token preserves col sums).

**vLLM mainline FP8 KV is per-block symmetric, NOT KIVI-style asymmetric.** Distinct codepath from `vllm/model_executor/layers/quantization/kv_cache.py::ScaledFP8KVCacheMethod`. v1 risked conflating the two.

**API axis:** quantization scheme is independently selectable for K and V. The K and V can have different bit-widths, different group sizes, and different quant axes within the same model.

### 5.11 Beam / Tree Cache (Speculative Decode)

For **tree speculation** (EAGLE-2, Medusa-2): multiple candidate branches are held simultaneously in the cache, with a per-branch view, and copy-on-accept resolves to a single branch. Implemented as:

- vLLM: `BlockSpaceManager.fork` creates a new logical block table sharing physical blocks via refcount; on accept of one branch, the other's blocks are released.
- TRT-LLM: `KVCacheManager::beamSearch` allocates per-beam shadow tables in the block manager.

**Constraint:** The kernel must accept a `[B_beams, ..., T_cache + n_speculative]` view; this is what FlashAttention's `flash_attn_with_kvcache` and vLLM's `paged_attention_v2` support, but ggml does not — llama.cpp speculative decoding therefore uses a single branch with rollback.

### 5.12 MTP-Head Cache (DeepSeek-V3, Dec 2024)

DeepSeek-V3 ships with **Multi-Token Prediction heads** (n_mtp = 1 in V3, with research toward larger n_mtp). Each MTP head needs its own K/V projection of the *predicted* tokens, kept in a separate per-step buffer. The "main" MLA cache and the "MTP" cache are independent.

**API axis:** `n_mtp_heads`, each with its own (often smaller) cache.

### 5.13 SSM State (Mamba / Mamba-2 / DeltaNet)

For Mamba-1:
```
ssm_state ∈ [B, d_inner, d_state]              # recurrent hidden state, typ d_state=16
conv_state ∈ [B, d_inner, d_conv - 1]          # 1-D causal conv state, typ d_conv=4
```

For Mamba-2 (SSD):
```
ssm_state ∈ [B, n_heads, headdim, d_state]
conv_state ∈ [B, d_inner, d_conv - 1]
```

For DeltaNet / Gated DeltaNet:
```
delta_state ∈ [B, n_heads, d_key, d_value]     # KV-matrix-product state
```

**All three have FIXED size in T** — no growth with sequence length. This is the most important structural property: paging is meaningless for SSM-state layers (they don't grow). Hybrid models keep attention layers paged and SSM layers as flat per-request tensors.

---

## §6 Prefill vs Decode Kernel Taxonomy

### 6.1 Three Regimes

| Regime | Q tokens | K tokens | Best kernel |
|---|---|---|---|
| **Prefill** | T (prompt) | T (prompt) | FlashAttention-2 / -3 (`causal=True`) |
| **Decode** | 1 | T (cache) | FlashDecoding / `flash_attn_with_kvcache` / vLLM `paged_attention_v2` |
| **Chunked prefill** | C (chunk, 256-2048) | T_so_far | FlashAttention with `cu_seqlens_q ≠ cu_seqlens_k` |

FlashDecoding (Dao 2023 blog) parallelizes the M=1 case across the **K dimension** rather than the M dimension; standard FA2 has zero parallelism in M=1. vLLM `paged_attention_v2` implements FlashDecoding++ for split-K decode with unified-max softmax.

### 6.2 Chunked Prefill (DeepSpeed-MII, Sarathi-Serve)

Splits a long prefill into chunks of size C and interleaves them with decode steps from other requests. Maximizes GPU utilization. Requires the attention kernel to accept arbitrary `Q.shape[0] = C` against full cache. Shipped in vLLM (default since v0.5), TensorRT-LLM, SGLang.

Pushes back on the model API: `attention(q, kv_cache)` must accept arbitrary `q` length uniformly — including 1 (decode), C (chunked prefill), and T (full prefill).

### 6.3 Speculative Decoding Compatibility

- **Vanilla speculative** (Leviathan et al. 2023): draft model proposes `k` tokens; verifier validates with one forward pass. Verifier needs `Q.shape[0] = k` with `K.shape[0] = T_cache + k − 1` and a block-diagonal-ish mask.
- **Medusa** (Cai et al. 2024): single model with extra heads; same kernel as chunked prefill.
- **EAGLE-2** (Li et al. 2024): tree-form proposals; mask becomes a tree mask. TreeAttention (§3.26) is the kernel that exploits this for ~2× over generic 2-D mask.

**API implication:** the attention call signature must accept `attn_mask` or a `mask_kind ∈ {causal, sliding, sink, custom_2d, tree}` parameter.

### 6.4 KV Cache State During Speculation

Speculation requires **provisional writes**: candidate tokens are written into the cache, then rolled back if rejected.
- Paged caches handle this trivially (release blocks).
- Ring buffers must explicitly rewind the head pointer.
- llama.cpp's `llama_kv_cache_seq_rm(seq_id, p0, p1)` supports this.
- For tree speculation, per-branch views (§5.11) are mandatory.

### 6.5 Hopper TMA + Warp Specialization (FA3 Specifics)

FlashAttention-3's H100-only path uses:
- **TMA descriptors** (`cp.async.bulk.tensor`) for async tile loads.
- **Producer / consumer warp specialization** via `cuda::barrier` — one warp loads tiles, another computes the matmul.
- **`wgmma.async.aligned`** for the BF16 / FP8 matmul, with the latter accumulating into BF16.
- **`cluster_launch_control`** for multi-CTA cooperation.

These determine which `(dtype, head_dim, mask_kind, softcap_on?)` combinations are kernel-feasible on H100. The minimum-API design must let model specs declare which kernel features they require (softcap, asymmetric head_dim, FP8 KV) and let the runtime declare which kernels are available.

---

## §7 Parameter Space — Revised

### 7.1 The KV Cache Parameter Space (8+ axes)

```
KVCacheSpec {
  // Shape
  n_layers:         int
  per_layer_shape:  list[KVLayerShape]    # len n_layers, supports hybrid

  // Layout
  layout:           {HND, NHD}
  storage:          {contiguous, paged_v1, paged_v2_vm, ring_swa, yoco_global}
  block_size:       int                   # paged: 16 vLLM default; contig: 1; ring: window W
  unified_layers:   bool                  # one big tensor vs per-layer list

  // Sharing
  kv_share_group:   list[int]             # producer-layer idx for each layer (CLA)
  cache_role:       {standard, yoco_producer, yoco_consumer}   # NEW for YOCO

  // Dtype
  k_dtype:          DType                 # bf16 / fp16 / fp8_e4m3 / int8 / int4_group / int2
  v_dtype:          DType                 # often differs from k_dtype (KIVI)
  k_group_size:     int | None
  v_group_size:     int | None
  k_quant_axis:     {channel, token, head, block, tensor}
  v_quant_axis:     {channel, token, head, block, tensor}
  k_scale_dtype:    DType                 # fp16 / fp32 / e8m0 / mx
  v_scale_dtype:    DType

  // Addressing
  ownership:        {explicit_pass, stateful}
  request_addressing: {batch_index, slot_mapping}
  prefix_reuse:     {none, copy_seq, paged_share, radix_tree, mla_latent_hash}

  // Bounds
  max_seq_len:      int
  ring_window:      int | None            # per-layer if interleaved
  sink_slots:       int | None            # 4 typical when present
  n_mtp_heads:      int                   # 0 default; DeepSeek-V3: 1
}

KVLayerShape {
  kind: {standard_kv, mla_latent, ssm_state, ssd_state, delta_state,
         retnet_state, mlstm_matrix, xpos_state, mutable_weights, cross_attn_frozen}
  n_kv_heads:       int                   # 1=MQA, G=GQA, H=MHA
  q_head_dim:       int                   # NEW — split from single head_dim
  k_head_dim:       int                   # NEW
  v_head_dim:       int                   # NEW (DeepSeek-V2: q=192, k=192, v=128)

  // MLA-specific
  kv_lora_rank:     int | None            # 512 / 256
  qk_rope_head_dim: int | None            # 64 / 32
  qk_nope_head_dim: int | None            # 128 / 64

  // SSM-specific (Mamba-1)
  d_state:          int | None            # 16 typical
  d_inner:          int | None            # 2 × hidden_size typical
  d_conv:           int | None            # 4 typical — NEW

  // Mamba-2 SSD-specific (NEW)
  chunk_size:       int | None            # 256 typical
  headdim:          int | None            # 64 typical
  ngroups:          int | None            # 1, 2, 4, 8

  // DeltaNet (NEW)
  delta_d_key:      int | None
  delta_d_value:    int | None
  delta_gate_dim:   int | None
}
```

**Eight base axes plus per-layer-kind sub-axes** describe every shipped cache. The biggest unification challenge: MLA's `d_latent` channel is shared across all heads, so a "per-head" tensor view breaks. SSM state has two tensors (recurrent + conv) per layer. The API must let a layer cache be any of `[..., H, d_h]`, `[..., kv_lora_rank + qk_rope_head_dim]`, `[..., d_inner, d_state] + [..., d_inner, d_conv-1]`, or `[..., H, d_key, d_value]`.

### 7.2 The Attention Parameter Space (19+ axes)

```
AttentionSpec {
  // Token-mixer kind
  kind: {mha, gqa, mqa, mla, differential, nsa, retnet, gla, deltanet,
         mamba1, mamba2_ssd, hyena, mlstm, slstm, ttt, cross_attn, none}

  // Projection topology
  n_q_heads:        int
  n_kv_heads:       int                   # MHA: =n_q; GQA: <n_q; MQA: 1
  q_head_dim:       int                   # NEW (DeepSeek-V2: 192)
  k_head_dim:       int                   # NEW
  v_head_dim:       int                   # NEW (DeepSeek-V2: 128)

  // MLA extras
  q_lora_rank:      int | None            # DeepSeek-V2: 1536; MiniCPM-3: 768
  kv_lora_rank:     int | None            # 512 / 256
  qk_nope_dim:      int | None            # 128 / 64
  qk_rope_dim:      int | None            # 64 / 32

  // Differential extras
  n_q_streams:      int                   # 2 for Diff
  lambda_init:      float | None

  // NSA extras (NEW)
  nsa_compress_block: int | None          # l_cmp = 32 typical
  nsa_select_block:   int | None          # l_sel = 64 typical
  nsa_top_n:         int | None           # n_sel = 16 typical
  nsa_sliding_window: int | None          # W_sliding = 512 typical

  // Mask
  mask:             {causal, sliding_window, sliding_with_sink, full,
                     block_diag, custom_2d, tree, nsa_three_branch}
  window_size:      int | None            # per-layer if interleaved
  n_sink_tokens:    int | None            # 4 typical
  sink_trained:     bool                  # true for GPT-OSS / Mistral-Small-3.1

  // Norm
  qk_norm:          {none, pre_rope_rms, post_rope_rms}
  qk_norm_axis:     {per_head, per_layer}

  // Numerics
  attn_scale:       float | None          # default 1/sqrt(q_head_dim); DeepSeek-V2 MLA: 1/sqrt(qk_nope+qk_rope)
  logit_softcap:    float | None          # Gemma 2: 50

  // Per-layer apply_rope (NEW for NoPE alternation)
  apply_rope:       bool                  # SmolLM3 NoPE-alternation, Llama-4 iRoPE

  // RoPE
  rope:             RoPESpec | None       # per-layer (Gemma 3 dual-θ)

  // Cross-layer KV
  shares_kv_with:   int | None            # layer idx of producer (CLA)
  cache_role:       {standard, yoco_producer, yoco_consumer}  # NEW

  // SSM extras (when kind ∈ mamba*)
  d_state:          int | None
  d_inner:          int | None
  d_conv:           int | None            # NEW
  dt_min:           float | None
  dt_max:           float | None
  dt_rank:          int | None
  A_init_range:     tuple[float, float] | None

  // Mamba-2 SSD extras (NEW)
  chunk_size:       int | None
  headdim:          int | None
  ngroups:          int | None
}

RoPESpec {
  basis:            {interleaved, split_half}
  partial_strategy: {full, partial_first_k, gptj_partial_interleaved, glm_index_parity}  # NEW
  base_theta:       float                  # 10000 / 500000 / 1000000 / 10000000 (per-layer in Gemma 3)
  d_rope:           int                    # = head_dim or partial
  position_axes:    int                    # 1 (text) or 3 (vision M-RoPE / MM-RoPE)

  scaling:          {none, linear_pi, ntk_static, ntk_dynamic, yarn,
                     llama3, longrope, dca}             # NEW: linear_pi, ntk_dynamic, dca separated
  scale_factor:     float | None
  original_max_pos: int | None

  // YaRN
  attn_factor:      float | None
  beta_fast:        float | None
  beta_slow:        float | None
  mscale_all_dim:   list[float] | None     # Qwen2.5-1M per-dim YaRN extension

  // Llama-3
  low_freq_factor:  float | None
  high_freq_factor: float | None

  // LongRoPE / Phi-3
  short_factor:     list[float] | None     # len d_h/2
  long_factor:      list[float] | None
  long_threshold:   int | None

  // DCA (NEW)
  chunk_size:       int | None
  inter_chunk_offset: int | None
}
```

**Per-layer everything.** v1 acknowledged but did not actually require per-layer `rope_theta` / `window_size`. v2 makes the `BlockSpec`-wraps-`AttentionSpec` ∪ `RoPESpec` requirement explicit: every axis in `AttentionSpec` and `RoPESpec` must be expressible per-layer. Phi-4-mini-flash, Gemma 3, SmolLM3, Llama-4 iRoPE, Qwen3-Next all require this.

### 7.3 Counts

- **Attention variants documented: 27** (MHA, MQA, GQA, MLA, SWA, interleaved local/global, sink, block-diagonal, NSA, CLA, YOCO, Differential, Mamba-1, Mamba-2 SSD, GLA, DeltaNet / Gated DeltaNet, RetNet, Hyena, TTT, mLSTM, sLSTM, cross-attention, FA1, FA2, FA3, FlashDecoding, RingAttention, Striped, TreeAttention, RadixAttention, PagedAttention v1, vAttention). Even excluding the kernel-family axis, **22 token-mixer variants** in §3 alone.
- **RoPE variants: 14** (vanilla, PI, NTK-static, NTK-dynamic, YaRN, Llama-3 smooth, LongRoPE, DCA, partial, NoPE, xPos, Sandwich, M-RoPE, MM-RoPE) plus the basis axis (interleaved vs split-half) and the ALiBi-as-position-scheme baseline.
- **Cache layouts: 13** (contiguous, paged-v1, paged-v2-vAttention, ring-SWA, MLA-latent, YOCO-global, NSA-three-tier, MTP-head, SSM state, SSD state, DeltaNet state, beam/tree, RadixAttention prefix-tree).

---

## §8 Compatibility / Constraint Matrix

`✔` = combination shipped in production; `~` = possible but rare / unshipped; `✘` = mathematically excluded or no kernel.

### 8.1 Attention × RoPE

| Attention ↓ \ RoPE → | Vanilla | PI | NTK-dyn | YaRN | Llama-3 | LongRoPE | DCA | Partial | NoPE | ALiBi |
|---|---|---|---|---|---|---|---|---|---|---|
| MHA | ✔ Phi-3-mini-4k | ✔ Code-Llama-16K | ✔ ChatGLM-2 | ~ | ~ | ✔ Phi-3-mini-128k | ~ | ✔ Phi-2 | ✔ NoPE OLMo | ✔ MPT |
| GQA | ✔ Llama-3, Mistral | ✔ LLongMA | ~ | ✔ Qwen2.5-1M | ✔ Llama-3.1/3.2 | ✔ Phi-3.5-MoE | ✔ Qwen2.5-1M | ✔ StableLM-2 | ✔ SmolLM3 (alternation), Llama-4 iRoPE | ~ |
| MQA | ✔ Falcon-180B (RoPE) | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ✔ Falcon-7B/40B (learned PE, alibi=false) |
| MLA | ✘ structural | ✘ | ✘ | ✔ DeepSeek-V2 (on qk_rope only) | ~ | ~ | ~ | ✔ **required** (only on qk_rope=64 channel) | ✘ | ✘ |
| SWA (local only) | ✔ Mistral-v0.1 | ~ | ~ | ~ | ✔ Mistral-Nemo | ~ | ~ | ~ | ~ | ~ |
| SWA + global alternation | ✔ Gemma 2 | ✘ | ~ | ~ | ~ | ~ | ~ | ~ | ✔ Llama-4 iRoPE (some layers NoPE) | ~ |
| Sink (StreamingLLM-style, inference-time) | ✔ on any | ✔ | ✔ | ✔ | ✔ | ✔ | ~ | ✔ | ✔ | ✔ |
| Sink (trained) | ✔ GPT-OSS, Mistral-Small-3.1 | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| NSA | ✔ DeepSeek 2025 (vanilla RoPE on each branch) | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| Differential | ✔ ref impl | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| Mamba-1 / SSM | n/a — RoPE not on SSM layers (only on attention layers in hybrid) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| GLA / DeltaNet | n/a — same | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Key constraints:**
- **MLA forces partial RoPE on the `qk_rope_head_dim` channel only.** RoPE on the full head dim would break the `W_uk` absorption. So `kind == mla ⇒ d_rope == qk_rope_head_dim` and the rotation is fed by `k_pe` not by `c_kv`.
- **MLA + asymmetric head_dim**: most kernels assume Q/K/V share head_dim. FA2 v2.5+ supports asymmetric; older CUTLASS does not. `kind == mla ⇒ requires kernel supporting q_head_dim ≠ v_head_dim`.
- **Gemma 3 forces per-layer RoPE base**: 1M for global, 10K for local. `rope.base_theta` must be per-layer. Most current libraries treat it as a scalar — this was the silent-bug surface pre-v0.6.6 vLLM.
- **SWA + sink** is composable: sink is just "always include first k cells in the attended set." llama.cpp does this via cell flags.
- **NoPE-alternation** (SmolLM3, Llama-4 iRoPE) requires per-layer `apply_rope: bool`. Not expressible without that axis.

### 8.2 Attention × Cache Layout

| Attention ↓ \ Cache → | Contiguous | Paged v1 | vAttention | Ring SWA | YOCO Global | NSA 3-tier | SSM State |
|---|---|---|---|---|---|---|---|
| MHA | ✔ default | ✔ vLLM | ✔ TRT-LLM 0.11+ | n/a | ~ | ✘ | n/a |
| GQA | ✔ | ✔ | ✔ | ✔ Mistral | ~ | ✘ | n/a |
| MQA | ✔ | ✔ | ✔ | ~ | ~ | ✘ | n/a |
| MLA | ✔ DeepSeek ref | ✔ vLLM v0.6.3+ (PagedMLA, single-tensor `[N_blocks, block_size, kv_lora_rank + qk_rope_head_dim]`) | ~ | ✘ | ✘ | ✘ | n/a |
| SWA local | ✔ | ✔ (block_size matched to W) | ✔ | ✔ ideal | ~ | ✘ | n/a |
| Gemma 3 interleaved | ✔ (two pools) | ✔ (per-layer block table) | ✔ | partial — only local layers ring | ✘ | ✘ | n/a |
| Differential | ✔ (cache same shape as MHA) | ✔ | ✔ | ✔ | ~ | ✘ | n/a |
| NSA | ✔ (3 caches per layer) | ~ (needs per-branch paging) | ~ | n/a | ✘ | ✔ native | n/a |
| YOCO | ✔ (stage-1 global) | ✔ | ✔ | n/a | ✔ native | ✘ | n/a |
| Mamba-1 SSM | n/a | n/a (state is fixed-size) | n/a | n/a | n/a | n/a | ✔ |
| Mamba-2 SSD | n/a | n/a | n/a | n/a | n/a | n/a | ✔ (with `chunk_size`, `headdim`, `ngroups`) |
| DeltaNet | n/a | n/a | n/a | n/a | n/a | n/a | ✔ (matrix state) |

**Key constraints:**
- **MLA + paged**: vLLM v0.6.3+ stores `c_kv + k_pe` jointly. Block layout is `[N_blocks, block_size, kv_lora_rank + qk_rope_head_dim]` — single tensor, not two. The "K cache / V cache" abstraction collapses.
- **MLA + prefix cache reuse** (vLLM v0.6.3+, PR #8546): the prefix-cache hash is computed on the **latent**, not on K, V. This requires hash-on-input-tokens to be deterministic across the down-projection — confirmed in `vllm/attention/backends/mla/triton_mla.py`.
- **SSM + paged**: SSM state is fixed-size per layer per request (`[d_inner, d_state]`), so paging is meaningless. The correct compatibility entry is **N/A**, not `~`. Hybrid models allocate attention layers paged + SSM layers as flat per-request tensors. TRT-LLM `KVCacheManager` has a separate `SsmStateManager`.
- **NSA + standard SDPA / FA backend**: needs custom kernel for the per-step block selection and three-branch softmax mixture.
- **YOCO + speculative decoding**: stage-1 cache reuse requires atomic rollback on rejection. Shadow stage-1 cache or transactional semantics.
- **Ring buffer + paged** is redundant; pick one.
- **Hybrid SSM + paged**: heterogeneous per-layer cache is mandatory.

### 8.3 RoPE × Cache Layout

| RoPE ↓ \ Cache → | Contiguous | Paged v1 | vAttention | Ring SWA |
|---|---|---|---|---|
| Vanilla | ✔ | ✔ | ✔ | ✔ |
| PI / Llama-3 / YaRN | ✔ | ✔ | ✔ | ✔ (position computed from absolute, not slot) |
| LongRoPE | ✔ | ✔ | ✔ | ✔ |
| Dynamic NTK | ✔ | ✔ (re-compute on each new token) | ✔ | ✔ |
| DCA | ✔ | ~ | ~ | ~ |
| MM-RoPE (multimodal) | ✔ (3 position axes) | ✔ | ✔ | ✘ (multimodal cache is not ring) |
| MLA-style (RoPE on `k_pe` only) | ✔ | ✔ (only `k_pe` portion is rotated) | ✔ | ✘ |
| NoPE-alternation | ✔ | ✔ | ✔ | ✔ |

All RoPE variants are cache-layout-agnostic provided **position ids are absolute** (not slot indices). Ring buffers must therefore store absolute position in each cell. llama.cpp's `llama_kv_cell::pos` is exactly this.

### 8.4 Composite Constraints (not derivable from pairwise matrices)

1. **MLA × paged × prefix-cache reuse**: requires hash-on-input-tokens through the down-projection. vLLM v0.6.3+ only.
2. **Gemma 3 × paged**: per-layer `rope_theta` means the position-id-to-rotation precomputation is per-layer, not per-model. vLLM's `RotaryEmbedding` registry must instantiate one per layer. Pre-v0.6.6 vLLM got this wrong; fixed in PR #11242.
3. **NSA × paged × continuous batching**: each of the three caches needs its own block table and `slot_mapping`. Not yet shipped in any production runtime.
4. **SWA + trained sink + paged**: requires kernel to support **two windows simultaneously** — sink-window (first k slots) and slide-window (last W slots). FA3 paged variant requires a custom mask spec; Triton-based PagedAttention does this via masked load.
5. **QK-norm placement × RoPE basis**: `post_rope + split_half` cross-mixes rotated and non-rotated dims under the norm. No production model uses this combo. Qwen3 and Gemma 3 both use `pre_rope + interleaved` (verified against HF source; correction to v1's claim of "Qwen3 post_rope, Gemma 3 post_rope").
6. **Logit softcap × FlashAttention**: Gemma 2's `tanh` softcap requires kernel support. FA2 v2.5+ has `attn_logit_softcapping`; FA3 has it; vLLM Triton kernel supports it. CUTLASS prebuilt paths often do not — so Gemma 2 only runs on specific backends.
7. **YOCO × paged**: stage-1 cache is shared across the whole stage-2; prefix caching semantics differ from per-layer paged.
8. **Differential × KV cache**: Diff-attn has 2× Q projections but K, V unchanged. Cache shape same as MHA; the *number of Q-K dot products* per token is 2×. Throughput on memory-bound decode is unchanged; compute-bound prefill is 2× slower.

---

## §9 Evolution Narrative — Dated Arcs Per Axis

### 9.1 Attention Evolution

| Date | Variant | Citation | Adoption |
|---|---|---|---|
| Jun 2017 | MHA | Vaswani et al. arxiv:1706.03762 | Universal baseline |
| Nov 2019 | MQA | Shazeer arxiv:1911.02150 | PaLM (2022) |
| May 2023 | GQA | Ainslie et al. arxiv:2305.13245 | Llama-2-70B (Jul 2023), then universal Llama-3+ |
| Sep 2023 | SWA | Mistral 7B v0.1 release | Mistral, Mistral-Nemo, Phi-3-mini-128k (with blocksparse) |
| Dec 2023 | Mamba-1 | Gu & Dao arxiv:2312.00752 | Mamba-2.8B, Falcon-Mamba-7B |
| Mar 2024 | Jamba (Mamba+MoE+attn hybrid) | Lieber et al. arxiv:2403.19887 | Jamba-Mini-1.6 |
| May 2024 | MLA | DeepSeek-V2 paper | DeepSeek-V2 / V3, MiniCPM-3 |
| May 2024 | CLA | Brandon et al. arxiv:2405.12981 | Apple OpenELM |
| May 2024 | YOCO | Sun et al. arxiv:2405.05254 | YOCO-1.3B; influenced Phi-4-mini-flash design |
| May 2024 | Mamba-2 SSD | Dao & Gu arxiv:2405.21060 | Zamba-2, Hymba (partial), Granite-4-tiny-hybrid |
| Jun 2024 | Gemma 2 interleaved local/global | Riviere et al. Gemma 2 tech report | Gemma 2 |
| Jun 2024 | Samba alternating Mamba/SWA | Ren et al. arxiv:2406.07522 | Samba 1.7B / 3.8B |
| Jun 2024 | RecurrentGemma (Griffin/Hawk) | Botev et al. arxiv:2402.19427 | RecurrentGemma-2B / 9B |
| Oct 2024 | Differential Transformer | Ye et al. arxiv:2410.05258 | DiffTransformer-1.3B / 6.8B |
| Nov 2024 | Hymba (parallel SSM+attn heads) | Hu et al. arxiv:2411.13676 | Hymba-1.5B |
| Feb 2025 | NSA — Native Sparse Attention | Yuan et al. arxiv:2502.11089 | DeepSeek-V3.x tech preview |
| Mar 2025 | Gemma 3 (per-layer θ, 5:1 local:global, no softcap) | Gemma 3 tech report | Gemma 3 1B/4B/12B/27B |
| Mar 2025 | Mistral-Small-3.1 (trained sinks) | Mistral release notes | Mistral-Small-3.1 |
| Apr 2025 | Llama-4 iRoPE (NoPE alternation) | Llama-4 release | Llama-4 Scout/Maverick |
| Aug 2025 | GPT-OSS family (trained sinks) | OpenAI release | GPT-OSS variants |
| Aug 2025 | Qwen3-Next (Gated DeltaNet hybrid) | Qwen3-Next release | Qwen3-Next |

**Dependency graph:**
- Cache-pressure branch: MHA → MQA → GQA → MLA → NSA (compression ratchet).
- Mask-shape branch: causal → SWA → interleaved → trained-sink (locality ratchet).
- Token-mixer branch: attention → Mamba → Jamba → Mamba-2 → Hymba/Samba/Phi-4-mini-flash → Qwen3-Next (alt-mixer ratchet).
- Architecture-topology branch: per-layer-independent → CLA / YOCO 2-stage (sharing ratchet).
- Numeric branch: vanilla softmax → softcap (Gemma 2) → no softcap (Gemma 3) → differential softmax-pair (Diff-Transformer) → branch-mixture softmax (NSA).

### 9.2 RoPE Evolution

| Date | Variant | Citation | Adoption |
|---|---|---|---|
| Apr 2021 | Vanilla RoPE | Su et al. arxiv:2104.09864 (RoFormer) | LLaMA 1 (Feb 2023) |
| Jun 2023 | PI / linear scaling | Chen et al. arxiv:2306.15595, Kaiokendev "SuperHOT" | Code-Llama-16K, LLongMA |
| Jul 2023 | NTK-aware (static) | bloc97 Reddit | early Llama-2 long-context forks |
| Jul 2023 | Dynamic NTK-aware | ChatGLM-2 release | ChatGLM-2/3, InternLM-200K, Yi-200K |
| Sep 2023 | YaRN (NTK-by-parts + temperature) | Peng, Quesnelle, Kingma arxiv:2309.00071 | DeepSeek-V2, Qwen2.5-1M, MiniCPM-3 |
| Feb 2024 | LongRoPE | Ding et al. arxiv:2402.13753 | Phi-3-mini-128k, Phi-3.5, Phi-4 |
| Apr 2024 | Llama-3 smooth-wavelength | Llama-3 release, `_compute_llama3_parameters` | Llama-3.1/3.2 |
| Jan 2025 | DCA (Qwen2.5-1M) | An et al. arxiv:2402.17463 + Qwen2.5-1M release | Qwen2-72B-DCA, Qwen2.5-1M |
| Mar 2025 | Per-layer θ alternation (Gemma 3) | Gemma 3 tech report | Gemma 3 (10K local / 1M global) |
| Jul 2025 | NoPE-alternation (SmolLM3) | SmolLM3 model card | SmolLM3 1.7B/3B |
| Apr 2025 | iRoPE (Llama-4) | Llama-4 release | Llama-4 |

**Dependency graph:**
- Extrapolation branch: vanilla → PI → NTK-static → NTK-dynamic → YaRN → Llama-3 smooth (with PI fallback).
- Per-dim learning branch: vanilla → LongRoPE (independent of extrapolation branch).
- Per-layer branch: vanilla → per-layer θ (Gemma 3) → per-layer apply_rope (SmolLM3 / iRoPE).
- Chunking branch: vanilla → DCA (intra/inter-chunk separation).
- Multimodal branch: vanilla → M-RoPE (Qwen2-VL) → MM-RoPE (Qwen2.5-VL, with frame index).

PI → NTK → YaRN → LongRoPE is a clear improvement chain; Llama-3 scaling is an independent branch that drops the temperature trick.

### 9.3 KV Cache Evolution

| Date | Variant | Citation | Adoption |
|---|---|---|---|
| 2022 | Contiguous (`torch.cat`) | HF baseline | All early models |
| Sep 2023 | PagedAttention v1 | Kwon et al. SOSP 2023 | vLLM, then SGLang/TRT-LLM |
| Feb 2024 | KVQuant 2-bit | Hooper et al. arxiv:2401.18079 | research+vLLM int4 paths |
| Feb 2024 | KIVI (per-channel K / per-token V) | Liu, Yuan, Wang et al. arxiv:2402.02750 | llama.cpp K-quant asymmetric paths |
| May 2024 | MLA-compressed latent | DeepSeek-V2 paper | DeepSeek-V2/V3, MiniCPM-3 |
| May 2024 | YOCO 2-stage architectural cache | Sun et al. arxiv:2405.05254 | YOCO-1.3B |
| May 2024 | RadixAttention prefix tree | Zheng et al. arxiv:2406.04692 | SGLang |
| May 2024 | vAttention CUDA-VM mapping | Prabhu et al. arxiv:2405.04437 | TRT-LLM 0.11+ |
| Sep 2024 | vLLM MLA paged + prefix cache | vLLM PR #8546 | vLLM 0.6.3+ |
| Nov 2024 | CacheGen (compression-codec) | Liu et al. SOSP 2024 | Meta production paths (reportedly) |
| Dec 2024 | DeepSeek-V3 MTP-head cache | DeepSeek-V3 tech report | DeepSeek-V3 |
| Feb 2025 | NSA three-tier cache | Yuan et al. arxiv:2502.11089 | DeepSeek-V3.x tech preview |
| Jun 2025 | MiniCache layer-merging | Liu et al. arxiv:2405.14366 + production | research |

**Dependency graph:**
- Memory-management branch: contiguous → PagedAttention v1 → vAttention (VM-mapped).
- Quantization branch: bf16 → INT8 → INT4 → KIVI (asymmetric) → KVQuant (codebook + asymmetric).
- Compression branch: full K, V → MLA (latent) → NSA (latent + sliding + selected).
- Architecture branch: per-layer cache → CLA (shared layers) → YOCO (2-stage).
- Allocator branch: per-request alloc → block-table paging → radix-tree (prefix reuse).

### 9.4 Hybrid Token-Mixer Evolution

| Date | Model | Architecture | Mixer mix |
|---|---|---|---|
| Dec 2023 | Mamba-2.8B | Pure SSM | 100% Mamba |
| Mar 2024 | Jamba (AI21) | MoE + Mamba + GQA | 4× attn + 28× Mamba + MoE every 2 |
| May 2024 | Mamba-2 SSD | Pure SSM, faster | 100% Mamba-2 |
| Jun 2024 | Zamba2 (Zyphra) | Shared attention block | Mamba-2 backbone + 2× shared GQA blocks interleaved |
| Jun 2024 | Samba (Microsoft) | Per-layer Mamba/SWA alternation | Mamba + SWA-MHA per layer |
| Jun 2024 | RecurrentGemma (Griffin / Hawk) | Linear recurrence + local attention | Griffin (RG-LRU + local attn); Hawk pure linear |
| Nov 2024 | Hymba (NVIDIA) | Parallel SSM + attention heads per layer | Same-layer SSM head + attn head, output-fused |
| 2025 | Phi-4-mini-flash | Per-layer SSM/attn fine-grained interleave | Per-layer block-class schedule |
| 2025 | Granite-4-tiny-hybrid (IBM) | MoE + Mamba + GQA | Per-layer schedule + MoE |
| Aug 2025 | Qwen3-Next | Gated DeltaNet primary, attention secondary | Hybrid with Gated DeltaNet as main token mixer |

**Dependency graph:**
- Pure-SSM branch: Mamba-1 → Mamba-2 → DeltaNet → Gated DeltaNet.
- Layer-interleaved branch: Jamba → Samba → Phi-4-mini-flash.
- Within-layer-parallel branch: Hymba (parallel heads).
- Shared-attention branch: Zamba2 (shared GQA block).
- Linear-recurrence branch: RecurrentGemma Griffin/Hawk.

---

## §10 Open / Forward-Looking Variants

These are not in mainstream SLM scope but appear in the literature and are mentioned for completeness:

- **HGRN / HGRN2** (Qin, Yang et al. 2024) — Hierarchical Gated Recurrent Networks; in `flame-rnn` runtime, OpenLLaMA-Linear.
- **Power Attention / Based / linear+softmax hybrids** (Arora et al. 2024) — Taylor-expansion approximations of softmax for linear-time attention.
- **Layer-skip cache logic for early-exit models** (LayerSkip, Elhoushi et al. Meta 2024) — when a layer is skipped, its K/V is never produced; downstream layers using CLA may need to consume from an earlier producer.
- **PagedAttention block-eviction / preemption** (vLLM swap-to-CPU; SGLang LRU) — out-of-memory recovery.
- **Hyena / StripedHyena** — implicit-conv attention; cache is filter-output history, not K/V.
- **Cross-attention as a separate cache class** — frozen encoder KV with no per-step write.
- **Polynomial / kernelized attention** (Polysketchformer, Performer) — research-bar only.
- **Quantum-inspired retention** (Hyena-Diffuser, etc.) — speculative.

---

## §11 References

Vaswani et al. 2017 (Attention Is All You Need, arxiv:1706.03762);
Shazeer 2019 (MQA, arxiv:1911.02150);
Su et al. 2021 (RoFormer, arxiv:2104.09864);
Press et al. 2021 (ALiBi, arxiv:2108.12409);
Haviv et al. 2022 (NoPE, arxiv:2203.16634);
Sun et al. 2022 (xPos, arxiv:2212.10554);
Chi et al. 2022 (Sandwich PE, arxiv:2212.10356);
Dao et al. 2022 (FlashAttention-1, arxiv:2205.14135);
Beltagy et al. 2020 (Longformer / SWA);
Chen et al. 2023 (Position Interpolation, arxiv:2306.15595);
Ainslie et al. 2023 (GQA, arxiv:2305.13245);
Kazemnejad et al. 2023 (Impact of Positional Encoding, arxiv:2305.19466);
Sun et al. 2023 (RetNet, arxiv:2307.08621);
Dao 2023 (FlashAttention-2, arxiv:2307.08691);
Peng, Quesnelle, Kingma 2023 (YaRN, arxiv:2309.00071);
Kwon et al. 2023 (PagedAttention / vLLM, SOSP 2023);
Xiao et al. 2024 (StreamingLLM, arxiv:2309.17453);
Liu, Zaharia, Abbeel 2023 (RingAttention, arxiv:2310.01889);
Brandon et al. 2023 (Striped Attention, arxiv:2311.09431);
Hong et al. 2023 (FlashDecoding++, arxiv:2311.01282);
Gu & Dao 2023 (Mamba, arxiv:2312.00752);
Yang et al. 2023 (Gated Linear Attention, arxiv:2312.06635);
Leviathan et al. 2023 (Speculative Decoding);
Hooper et al. 2024 (KVQuant, arxiv:2401.18079);
Liu, Yuan, Wang et al. 2024 (KIVI, arxiv:2402.02750);
An et al. 2024 (DCA, arxiv:2402.17463);
Ding et al. 2024 (LongRoPE, arxiv:2402.13753);
Cai et al. 2024 (Medusa);
Li et al. 2024 (EAGLE-2);
Spector & Re 2024 (TreeAttention, arxiv:2402.13720);
Botev et al. 2024 (RecurrentGemma Griffin/Hawk, arxiv:2402.19427);
Lieber et al. 2024 (Jamba, arxiv:2403.19887);
Sun et al. 2024 (YOCO, arxiv:2405.05254);
Prabhu et al. 2024 (vAttention, arxiv:2405.04437);
Beck et al. 2024 (xLSTM, arxiv:2405.04517);
Brandon et al. 2024 (CLA, arxiv:2405.12981);
Liu, Hooper et al. 2024 (MiniCache, arxiv:2405.14366);
Dao & Gu 2024 (Mamba-2 SSD, arxiv:2405.21060);
Ren et al. 2024 (Samba, arxiv:2406.07522);
Zheng et al. 2024 (SGLang RadixAttention, arxiv:2406.04692);
Liu et al. 2024 (DeepSeek-V2, arxiv:2405.04434);
Yang & Pan 2024 (DeltaNet, arxiv:2406.06484);
Shah, Bikshandi et al. 2024 (FlashAttention-3, arxiv:2407.08608);
Sun et al. 2024 (TTT, arxiv:2407.04620);
Hu et al. 2024 (MiniCPM-3, arxiv:2404.06395);
Riviere et al. 2024 (Gemma 2 tech report);
Ye et al. 2024 (Differential Transformer, arxiv:2410.05258);
Hu et al. 2024 (Hymba, arxiv:2411.13676);
DeepSeek-V3 tech report 2024;
Yuan et al. 2025 (NSA, arxiv:2502.11089);
Gemma 3 tech report 2025;
Mistral-Small-3.1 release notes 2025;
Bai et al. 2025 (Qwen2.5-VL MM-RoPE);
Llama-4 release 2025;
SmolLM3 release 2025;
Qwen3-Next release 2025;
GPT-OSS release 2025.

**HF / runtime source references:**

- `transformers/src/transformers/models/qwen3/modeling_qwen3.py::Qwen3Attention.forward` (line 236-241: q_norm/k_norm BEFORE apply_rotary_pos_emb).
- `transformers/src/transformers/models/gemma3/modeling_gemma3.py::Gemma3Attention.forward` (q_norm/k_norm BEFORE apply_rotary_pos_emb).
- `transformers/src/transformers/cache_utils.py` (DynamicCache, StaticCache, OffloadedCache, QuantizedCache, HybridCache).
- `transformers/src/transformers/modeling_rope_utils.py` (`_compute_default_rope_parameters`, `_compute_linear_scaling_rope_parameters`, `_compute_dynamic_ntk_parameters`, `_compute_yarn_parameters`, `_compute_llama3_parameters`, `_compute_longrope_parameters`).
- `vllm/config/cache.py` (`CacheConfig.DEFAULT_BLOCK_SIZE: ClassVar[int] = 16`).
- `vllm/attention/backends/mla/common.py::MLACommonImpl` and `vllm/attention/backends/mla/triton_mla.py` (paged MLA, v0.6.3+).
- `vllm/attention/ops/paged_attn.py`, `vllm/worker/cache_engine.py`.
- `sglang/srt/managers/cache_controller.py` (RadixAttention).
- `llama.cpp/src/llama-kv-cache.cpp`, `llama.cpp/src/llama-kv-cache-iswa.cpp` (Gemma 2/3 dual pools).
- `llama.cpp/convert_hf_to_gguf.py::LlamaModel.modify_tensors` (Q/K permute at conversion time, not load).
- `flash-attention/hopper/flash_fwd_kernel_sm90.cu` (FA3).
- `flashinfer/include/flashinfer/attention/decode.cuh`, `prefill.cuh`.
- `DeepSeek-V2/inference/modeling_deepseek.py::DeepseekV2Attention`.
- `transformers/src/transformers/models/phi3/modeling_phi3.py::Phi3RotaryEmbedding._compute_longrope_parameters` (hard `seq_len` threshold switch).
- `mamba/mamba_ssm/modules/mamba_simple.py` (Mamba-1 SSM with `ssm_state` + `conv_state`).
- `mamba/mamba_ssm/modules/mamba2.py` (Mamba-2 SSD with `chunk_size`, `headdim`, `ngroups`).
- `KIVI/quant/new_pack.py::triton_quantize_and_pack_along_last_dim`.
- `streaming-llm/streaming_llm/kv_cache.py` (sink cells).

**Verified HF configs (as of 2026-06-04):**

- `Qwen/Qwen3-0.6B/config.json`: `rope_theta = 1000000`, `head_dim = 128`, `num_attention_heads = 16`, `num_key_value_heads = 8`.
- `Qwen/Qwen3-8B/config.json`: `rope_theta = 1000000`.
- `Qwen/Qwen3-14B/config.json`: `rope_theta = 1000000`.
- `openbmb/MiniCPM3-4B/config.json`: `kv_lora_rank = 256`, `q_lora_rank = 768`, `qk_nope_head_dim = 64`, `qk_rope_head_dim = 32`, `num_attention_heads = 40`, `num_hidden_layers = 62`. **MLA, not CLA.**
- `tiiuae/falcon-7b/config.json`: `alibi = false`. (Position scheme is learned absolute.)
- `tiiuae/falcon-40b/config.json`: `alibi = false`. No `rope_theta`. (Learned absolute.)
- `tiiuae/falcon-180B/`: per model card, positional embeddings are RoPE. (Not ALiBi.)
- `google/gemma-3-1b-pt/config.json`: `num_attention_heads = 4`, `num_key_value_heads = 1`, `head_dim = 256` — strictly **MQA**.

---

## Appendix A — One-Line Model Summaries (v2 corrected)

- **Llama-3.2-1B**: 16 layers, H=32, G=8, d_h=64, rope_theta=500_000 + Llama-3 smooth scaling, no QK-norm, full-causal, contiguous cache.
- **Qwen3-0.6B**: 28 layers, H=16, G=8, d_h=128, **rope_theta=1_000_000** (v1 said 5M — wrong), **QK-norm BEFORE RoPE** (v1 said after — wrong), full-causal.
- **Qwen3-4B**: 36 layers, H=32, G=8, d_h=128, rope_theta=1_000_000, QK-norm BEFORE RoPE.
- **Qwen3-8B**: H=32, G=8, d_h=128, rope_theta=1_000_000.
- **Qwen3-14B**: H=40, G=8, d_h=128, rope_theta=1_000_000.
- **Phi-3-mini-128k**: 32 layers, MHA (H=G=32), d_h=96, LongRoPE (short/long factor vectors, hard switch at original_max), no QK-norm.
- **Phi-4-mini (3.8B)**: GQA H=24 G=8 d_h=128, rope_theta=1_000_000, no QK-norm, full-causal.
- **Phi-4-mini-flash**: Per-layer hybrid SSM/GQA; per-layer rope_theta and window_size.
- **Gemma-3-1B**: 26 layers, **MQA** H=4 G=1 d_h=256 (v1 mislabeled GQA), **per-layer rope_theta** (10K local / 1M global), **QK-norm BEFORE RoPE** (verified vs HF, opposite of v1), SWA W=1024 on 5/6 layers, global on 1/6, no softcap.
- **Mistral-7B-v0.1**: 32 layers, GQA H=32 G=8 d_h=128, rope_theta=10_000, SWA W=4096 all layers.
- **Mistral-Small-3.1**: GQA, rope_theta=1_000_000, **trained attention sinks (`attention_sink_size = 4`)**.
- **DeepSeek-V2-Lite (16B/A2.4B)**: 27 layers, MLA (q_lora=1536, kv_lora=512, qk_nope=128, qk_rope=64, v=128), q_head_dim=192 ≠ v_head_dim=128, YaRN scaling, full-causal.
- **DeepSeek-V3 (671B/37B-active)**: MLA + MTP-head cache (n_mtp=1), 128-head MLA.
- **MiniCPM-3-4B**: 62 layers, **MLA** (q_lora=768, kv_lora=256, qk_nope=64, qk_rope=32) — v1 mis-labeled CLA. YaRN-style scaling.
- **Jamba-Mini-1.6**: 32 layers, 4× GQA-attention + 28× Mamba-1, MoE every 2, attention rope_theta=1_000_000.
- **Hymba-1.5B**: Parallel SSM + attention heads per layer (NVIDIA).
- **Samba-3.8B**: Per-layer Mamba ↔ SWA-MHA alternation.
- **OLMo-2-1B**: 16 layers, GQA H=16 G=4 d_h=64, rope_theta=500_000, **QK-norm pre-RoPE with RMSNorm** (sandwich pattern around sublayers), no SWA.
- **SmolLM2-1.7B**: 24 layers, GQA H=32 G=8 d_h=64, rope_theta=130_000, no QK-norm.
- **SmolLM3-3B**: GQA + **NoPE alternation** (per-layer `apply_rope`).
- **Llama-4 Scout / Maverick**: iRoPE — interleaved RoPE / NoPE layers.
- **StableLM-2-1.6B**: 24 layers, MHA H=32 d_h=64, partial RoPE (25%), QK-norm pre-RoPE.
- **GPT-OSS family**: Trained attention sinks.
- **Falcon-180B**: MQA (H=128, G=1), positional encoding is **RoPE only** (v1 said "RoPE + ALiBi" — wrong; per Falcon-180B model card, alibi=false, positional=RoPE).
- **Falcon-7B / 40B**: MQA, learned absolute PE (alibi=false; no rope_theta in config).
- **Qwen3-Next**: Gated DeltaNet primary token mixer + attention secondary.

---

## Appendix B — Code-Pointer Cheatsheet (v2)

- HF cache utils: `transformers/src/transformers/cache_utils.py`.
- HF RoPE utils: `transformers/src/transformers/modeling_rope_utils.py` (vanilla / linear PI / dynamic NTK / YaRN / llama3 / LongRoPE).
- HF Qwen3: `transformers/src/transformers/models/qwen3/modeling_qwen3.py` (q_norm/k_norm at line 236-237, apply_rotary_pos_emb at line 241 — pre-RoPE order confirmed).
- HF Gemma 3: `transformers/src/transformers/models/gemma3/modeling_gemma3.py` (q_norm/k_norm pre-RoPE; per-layer rope_theta in `Gemma3DecoderLayer`).
- HF Phi-3: `transformers/src/transformers/models/phi3/modeling_phi3.py::Phi3RotaryEmbedding._compute_longrope_parameters` (hard switch).
- vLLM cache: `vllm/config/cache.py` (`DEFAULT_BLOCK_SIZE = 16`); `vllm/attention/ops/paged_attn.py`; `vllm/worker/cache_engine.py`.
- vLLM MLA: `vllm/attention/backends/mla/common.py`, `vllm/attention/backends/mla/triton_mla.py`.
- vLLM block allocator: `vllm/core/block/cpu_gpu_block_allocator.py::fork` (copy-on-write).
- vLLM slot mapping: `vllm/attention/backends/utils.py::compute_slot_mapping`.
- SGLang RadixAttention: `sglang/srt/managers/cache_controller.py`.
- llama.cpp KV cache: `llama.cpp/src/llama-kv-cache.cpp`; `llama.cpp/src/llama-kv-cache-iswa.cpp` (Gemma 2/3).
- llama.cpp Q/K permute: `llama.cpp/convert_hf_to_gguf.py::LlamaModel.modify_tensors` (conversion-time permutation; correction to v1).
- FlashAttention-3: `flash-attention/hopper/flash_fwd_kernel_sm90.cu`.
- FlashInfer: `flashinfer/include/flashinfer/attention/decode.cuh`, `prefill.cuh`.
- DeepSeek-V2 ref: `DeepSeek-V2/inference/modeling_deepseek.py::DeepseekV2Attention`.
- Mamba-1 ref: `mamba/mamba_ssm/modules/mamba_simple.py`.
- Mamba-2 SSD ref: `mamba/mamba_ssm/modules/mamba2.py`.
- KIVI ref: `KIVI/quant/new_pack.py::triton_quantize_and_pack_along_last_dim`.
- TRT-LLM KV manager: `tensorrt_llm/runtime/kv_cache_manager.py`.

---

## Appendix C — Open Verifications (flagged for follow-up)

1. **vLLM `block_size` default per-platform.** As of 2026-06-04, `DEFAULT_BLOCK_SIZE = 16` is a static `ClassVar`, with no platform-conditional override in `_apply_block_size_default`. **The prompt's claim of "32 default on H100/FA3" was not confirmed against `vllm/config/cache.py`.** If a downstream patch or extension sets this differently per platform, it is not visible in the upstream codebase. v2 reports 16.
2. **NSA branch parameters.** The arxiv abstract for 2502.11089 does not give exact `l_cmp`, `l_sel`, `n_sel`, `W_sliding` values. v2 cites the typical values (`l_cmp = 32`, `l_sel = 64`, `n_sel = 16`, `W_sliding = 512`) consistent with the prompt and community reports, but these should be re-verified against the paper PDF or the DeepSeek reference implementation.
3. **Mamba-2 SSD default `chunk_size`.** Cited as 256 per community reports; not fetched from `mamba_ssm/modules/mamba2.py` directly in this revision.
4. **Qwen3-Next architecture details.** Cited as "Gated DeltaNet hybrid" per the Aug 2025 release; the exact mix ratio and per-layer schedule should be re-verified against the Qwen3-Next config when finalized.
5. **GPT-OSS sink-training details.** Cited from leaked-config reports; should be re-verified against the official GPT-OSS release when available.

— end of document —
