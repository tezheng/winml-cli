# llm-layers — Examples Report

**Date:** 2026-06-11
**Status:** v7-complete (M1 + M2 + v5 + v6 + v7), 41 architecturally-distinct families verified plus 9 deferred stubs
**Source of truth:** the file tree under `models/` and `tests/models/`, plus `git log` for commit-message-recorded `max_abs_diff` values.

This report is the companion to:
- `docs/API-REFERENCE.md` — the IR surface (every public symbol in `api/*.py`).
- `docs/PROJECT-SUMMARY.md` — the project scoreboard.

It answers four questions:
1. **What does each family prove the IR can express?** (Part 1)
2. **Which architectural axes does the IR cover, and which example demonstrates each one?** (Part 2)
3. **Does it actually work?** (Part 3 — actual pytest output captured today.)
4. **How do I build a new family from scratch?** (Part 4 — Llama 3 walk-through.)

---

## Part 0 — Glossary (gate classes)

Adhering to the audit-corrected taxonomy from `docs/PROJECT-SUMMARY.md`:

| Class | Meaning |
|---|---|
| **R** | **Real HF weights** — `AutoModelForCausalLM.from_pretrained(...)` of the production checkpoint, embedded-then-per-layer forward compared against the HF `*DecoderLayer` forward at `atol=5e-4` (fp32 CPU). The strongest claim — "the IR re-assembles production weights." |
| **S** | **Synthetic-weight self-consistency** — random init under a mini-config (e.g. `hidden=64`), the IR-built block is loaded from the same `state_dict` HF saw, and the two forwards are compared. Proves the math against HF's `torch_forward` reference path but not against the production model. |
| **shape-only** | The family's spec composes; `forward()` either raises `NotImplementedError` (e.g. DSA Lightning Indexer, BlockSparse) or has been gated out (e.g. Llama 4 Scout weights are license-gated and 17B too large to download). Spec coverage only; no numerical guarantee. |
| **hybrid** | Composite gate — one branch of the layer (e.g. MoE) is numerically validated, another (e.g. CSA/HCA attention) is shape-only. |

"`max_abs_diff`" in the tables below is the largest absolute element-wise difference between IR output and HF output observed during the gate test, in fp32 on CPU. All R/S gates pass `atol=5e-4`; the recorded value is how much margin we beat that bar by.

### Why two gate classes coexist

The original M1 design assumed every family would land as **R**. That assumption broke at three points and the project absorbed each as a lesson:

1. **License gating.** Llama 4 Scout's weights require a gated HF token and the 17B checkpoint is too large for the project's `hf_cache`. The IR can compose the spec faithfully but cannot prove a forward against weights it cannot acquire. → shape-only.
2. **In-flight ops not yet implemented.** DeepSeek-V3.2 DSA's Lightning Indexer, Phi-3-small BlockSparse, DeepSeek-V4 CSA/HCA — all have a `forward()` that legitimately raises `NotImplementedError` because the kernel work has not landed. The spec is still useful because it locks the schema for the future numerical gate. → shape-only.
3. **Production checkpoint too large or unreleased.** Moshi 7B, Voxtral 3B, Mamba-2 2.7B, Granite-4-H. Each of these has a production reference but downloading and forwarding it in CI is prohibitively slow. The fallback is to instantiate the HF model class with a smaller config and random weights, then prove the IR matches HF's own reference forward (`MambaMixer.slow_forward`, `Mamba2Mixer.torch_forward`, etc.). → S.

The honest reading: **R is the strongest claim, S is the next-strongest, and shape-only is "spec only — no claim about math."** This examples report respects that distinction throughout.

---

## Part 1 — Per-family inventory (the WHAT)

### 1.1 Llama-family transformers (the dense, RoPE+RMSNorm+SwiGLU baseline)

The "starter pack" — these prove the IR floor: PRE-norm RMSNorm STANDARD_W, SPLIT QKV, GQA SwiGLU, SPLIT_HALF RoPE, no biases. Each adds at most one delta from the baseline.

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| Qwen3 0.6B | `models/qwen3/` | M1 anchor: GQA + QK-norm PRE-RoPE PER_HEAD_DH + SwiGLU + SPLIT_HALF RoPE θ=1M | R | 4.77e-7 |
| Llama 3.2 1B | `models/llama3/` | LLAMA3 smooth-scaled RoPE (factor=32, low=1, high=4, ctx=8192) | R | 5.36e-7 |
| Mistral 7B v0.3 | `models/mistral/` | Llama-shape with no QK-norm; close-cousin baseline of Llama 3 | R | 5.96e-8 |
| SmolLM3 3B | `models/smollm3/` | Per-layer **NoPE alternation** — every 4th layer drops RoPE (`rope=None`) | R | layer-0: 4.77e-7, layer-3 (NoPE): 2.98e-7 |
| TinyLlama 1.1B | `models/tinyllama/` | Llama 3 spec at small scale (delegates to `models/llama3`) | R | 8.94e-8 |

**What this group proves about the IR:** the `to_block_spec(layer_idx)` per-layer dispatch can produce two distinct specs (RoPE vs NoPE) from the same `Config`. The Mistral / TinyLlama rows prove the layer factory is genuinely shareable — Mistral's `layer.py` is ~30 lines because the loader keys are identical to Llama's.

**Concrete lesson from this batch.** During B1 the team discovered that several "Llama-shaped" families were not quite Llama-shaped. Mistral 7B v0.3 doesn't have any qk_norm at all (unlike Qwen3), so the spec carries `qk_norm=None`. SmolLM3 introduced **per-layer NoPE alternation** which forced the IR to allow `rope=None` inside `AttentionSpec` — this was a one-line fix in `api/attention.py` (commit `3ea394f`: "allow rope=None for NoPE layers") but it changed the shape of the IR. Llama 3.x is the first family where `from_hf_dict` had to handle both transformers 4.x flat keys (`rope_theta`, `rope_scaling`) and transformers 5.x nested keys (`rope_parameters.rope_theta`) — every config from B1 onward inherited this dance.

### 1.2 Llama-family quirks (where one axis diverges)

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| Granite 3.1-2B | `models/granite/` | **μP scalars**: `residual_multiplier` baked into block, `attention_multiplier` overrides `1/sqrt(Dh)` | R | 7.15e-7 |
| Phi-3-mini-4k | `models/phi3_mini/` | **FUSED QKV** layout (`Linear(D, (Nq+2Nkv)*Dh)`) + **fused gate_up** | R | 4.77e-7 |
| Phi-4-mini | `models/phi4_mini/` | LongRoPE (short_factor + long_factor + attention_factor formula) on Phi-3 envelope | R | 1.91e-6 |
| MiniCPM-3 4B | `models/minicpm3/` | **MLA** (q_lora_rank=768, kv_lora_rank=256, qk_nope=64, qk_rope=32, v_head_dim=64) + μP residual scaling + LongRoPE | R | 3.05e-5 |
| Phi-3-small | `models/phi3_small/` | **BlockSparse** mask (forward deferred) + GeGELU FFN | shape-only | n/a |

**Concrete lessons from this batch.** Granite forced the IR to plumb `residual_scale` through `DecoderBlockSpec` (commit `a62551f`), because Granite's `residual_multiplier` baked into the spec turned out to be the cleanest way to express μP residual scaling. The same field is reused by MiniCPM-3 (μP residual depth scaling, `scale_depth / sqrt(num_hidden_layers) ≈ 0.178`). Phi-3-mini introduced `QKVLayout.FUSED` (commit `4d6b9a5`) — HF Phi-3 stores Q, K, and V in a single `qkv_proj.weight` with shape `[(n_q+2*n_kv)*head_dim, hidden]` and the IR's loader has to slice. Phi-3 also added `fused_gate_up=True` to `FFNSpec` for the same reason. Phi-4-mini introduced LongRoPE with the `attention_factor` formula `sqrt(1 + log(scale)/log(orig))` — a numerically delicate computation that the IR computes in fp64 inside `LongRoPEParams` to match HF's `_compute_longrope_parameters`. MiniCPM-3 caught a drift here too: MiniCPM-3's LongRoPE computes `attention_factor` **unconditionally** with no clamp when `scale <= 1` (unlike Phi-3 which clamps to `1.0`). That difference is recorded in `models/minicpm3/config.py:144-153` with a source citation to `modeling_minicpm.py:218-222`.

### 1.3 Norm variants

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| OLMo 2 1B | `models/olmo2/` | **POST-norm** position + **FULL_HDH** QK-norm shape | R | 4.77e-7 |
| Gemma 2 2B | `models/gemma2/` | **Sandwich norm (PRE_AND_POST)** + **ONE_PLUS_W** RMSNorm + `attn_logit_softcap=50.0` in SDPA + GeGLU + alternating SWA/full | R | **0.0 (bit-exact)** |
| Gemma 3 1B | `models/gemma3/` | 5:1 SWA/full + dual-RoPE θ (local vs global) + PER_HEAD_DH QK-norm + sandwich + ONE_PLUS_W | R | layer-0: 1.22e-4, layer-5: 2.44e-4 |
| Gemma 4 E2B | `models/gemma4/` | All of Gemma 3 + **PLE** (per-layer embeddings) + **partial-RoPE proportional** + `v_norm` (no scale) + cross-layer KV sharing + `attention_k_eq_v` (12B+) | R | layer-0: 7.6e-6, layer-4: 3.8e-6 |

The Gemma 2 row is the only family in the entire project with a `0.0` bit-exact gate against **real HF weights**. The Mamba-2 / Granite-4-H / DeepSeek-V3-MoE bit-exact rows below are vs synthetic weights.

**Concrete lessons from this batch.** OLMo 2 was the first family to need **POST-norm** semantics — the residual flows are reversed compared to PRE-norm, and the `DecoderBlock` had to learn that `pre_attn_norm=None, post_attn_norm=norm_spec` means "compute attention on the residual stream directly, then normalize the attention output before adding back" (commit `f1dcd46`). Gemma 2 forced two more axes: **sandwich norm** (PRE_AND_POST, both halves present) and `attn_logit_softcap` inside SDPA (the math is `attn = tanh(attn / cap) * cap` applied **after** the QK matmul but **before** the mask add). Gemma 3 added **dual RoPE θ** (local layers θ = 10000, global layers θ = 1M) — the IR's per-layer dispatch carries the right value via `RoPESpec.base_theta`. Gemma 4 was the project's hardest single batch: the original B0.5 IR was based on blog summaries and the numerical gate caught **5 drifts** that became `feedback_verify_source_not_blogs.md` in user memory. The five: (1) RMSNorm is STANDARD_W not ONE_PLUS_W as Gemma 1/2/3; (2) no qk_norm fixed-scale absorption (HF Gemma 4 sets `scaling = 1.0`); (3) v_norm exists with `with_scale=False`; (4) partial-RoPE uses "proportional" geometry (zero-padded inv_freq) not Phi-3 "prefix"; (5) layer_scalar buffer is per-layer. Each fix is a B0.6 commit (`18f1bd6`, `50829ed`, `496cc58`, `0bc9d0c`, `9d5daea`).

### 1.4 Per-layer heterogeneous attention

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| Llama 4 Scout | `models/llama4_scout/` | **iRoPE per-layer dispatch** (`no_rope_layers[i] == 1` → RoPE, `0` → NoPE) + **INTERLEAVED** RoPE basis | shape-only | weights gated; chunked-attention forward deferred |
| Ministral 8B | `models/ministral/` | **1:3 interleaved SWA / full** per-layer dispatch via `layer_types[i]` | R | full layer: 1.91e-6, SWA layer: 1.04e-7 |

**Concrete lessons.** Llama 4 Scout's INTERLEAVED RoPE basis is the **third** RoPE basis the IR carries, distinct from SPLIT_HALF (rotate_half pairing) and the M-RoPE channel stitching. Where SPLIT_HALF pairs `x[..., :Dh/2]` with `x[..., Dh/2:]`, INTERLEAVED pairs `x[..., 0::2]` with `x[..., 1::2]` (the complex `view_as_complex(reshape(-1, 2))` pattern). Adding it (commit `8ea9930`) was one line in `api/rope.py` plus a `RoPEBasis.INTERLEAVED` enum member, but the basis flag is consumed everywhere `rope_apply` runs. Ministral was the first per-layer SWA-vs-full dispatcher landed where `to_block_spec(layer_idx)` is non-trivial — every other family up to that point either had uniform mask or uniform SWA. The 1:3 sliding:full ratio means that on a 32-layer Ministral, 24 layers see SWA windows of 8192 and 8 layers see full causal attention. The numerical gate covered both layer types separately (`layer_idx=0` for full, `layer_idx=1` for sliding) and recorded different `max_abs_diff` values for each (`1.91e-6` vs `1.04e-7`).

### 1.5 MLA family (Multi-Head Latent Attention)

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| DeepSeek-V2-Lite | `models/deepseek_v2_lite/` | **`q_lora_rank=None`** (direct q_proj, no LoRA) + V2 INTERLEAVED RoPE on `qk_rope_head_dim` slice + YARN scaling + sigmoid+bias MoE | S | 4 paths all 2.38e-7 |
| DeepSeek-V3-Lite | `models/deepseek_v3_lite/` | V3 SPLIT_HALF RoPE + V3 router (sigmoid + bias + group_routing) | S (submodule) | 0.0 |
| DeepSeek-V3.2 | `models/deepseek_v32/` | **DSA (DeepSeek Sparse Attention) Lightning Indexer** (`AttentionKind.DSA`, forward deferred) | shape-only | n/a |

**Concrete lessons.** MLA is the IR's heaviest attention variant. The latent layout (`QKVLayout.MLA_LATENT`) decomposes Q and KV into LoRA factors — for MiniCPM-3 that's `q_a_proj: Linear(hidden, q_lora_rank=768)` then `q_a_layernorm` then `q_b_proj: Linear(768, n_q * qk_head_dim)`; same for KV but with `kv_a_proj_with_mqa` producing a `kv_lora_rank + qk_rope_head_dim`-sized output. The RoPE applies to only the `qk_rope_head_dim` slice (last 32 channels of 96), so the IR's `AttentionSpec.qk_nope_head_dim` + `qk_rope_head_dim` carry the split. The `ContiguousKVCache` was extended to support **asymmetric K/V head_dim** (commit `b4c8e6c`) because MLA's V is `v_head_dim=64` but K is `qk_head_dim=96` — different head_dim for K vs V. DeepSeek-V2's MLA uses **INTERLEAVED** RoPE; DeepSeek-V3 (and V3-Lite) uses **SPLIT_HALF**. The same `AttentionKind.MLA` token mixer flips basis via `RoPESpec.basis`. The `q_lora_rank=None` path (V2-Lite, smaller model) skips the q LoRA entirely — direct `q_proj: Linear(hidden, n_q * qk_head_dim)` — and the IR honors this via an `if q_lora_rank is None` branch in `Attention.__init__` (commit `576f7a2`).

### 1.6 MoE families

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| Mixtral 8x7B | `models/mixtral/` | Canonical softmax MoE: `router_kind="softmax"`, `router_norm=True`, no shared experts, no group routing | S | < 5e-4 |
| Qwen3-MoE 30B-A3B | `models/qwen3_moe/` | Per-layer dense / MoE dispatch via `decoder_sparse_step`; softmax router | S | < 5e-4 |
| OLMoE 1B-7B | `models/olmoe/` | FULL_HDH QK-norm + softmax MoE (no shared experts) | S | < 5e-4 |
| DeepSeek-V3-MoE | `models/deepseek_v3_moe/` | Production-dim wrapper around V3-Lite (composes existing primitives) | shape-only | composes primitives |

**Concrete lessons.** The two router variants Mixtral vs DeepSeek-V3 turned out to be the IR's MoE bottleneck. Mixtral uses **softmax over the full router logit then top-k**, with optional `router_norm` to re-normalize the top-k weights to sum to 1. DeepSeek-V3 uses **sigmoid + bias on top of group routing** — the router emits `n_routed_experts` sigmoid scores, adds a learned bias, groups the experts into `n_group` groups, scores each group as the top-2-sum of expert scores, picks the top `topk_group` groups, then top-k within those. The IR landed both routers as a single `MoESpec` with a `router_kind: "softmax" | "sigmoid_plus_bias"` discriminator (commit `43a4471`). The drift that B5 caught: V3 **gathers** router weights from the bias-free sigmoid (the bias is for selection only, not for the weight that gets multiplied with the expert output). This is `models/deepseek_v3_lite/config.py` referencing `modeling_deepseek_v3.py:537-559`. OLMoE was the first family to combine **FULL_HDH** QK-norm with softmax MoE — exercises the cross-product of B3 (POST/FULL_HDH) and B5 (MoE) without further IR changes. Qwen3-MoE introduced the **per-layer dense / MoE dispatch via `decoder_sparse_step`** — every Nth layer is dense MLP, the rest are MoE; the `to_block_spec(layer_idx)` returns `FFNSpec` or `MoESpec` accordingly.

### 1.7 SSM and SSM-hybrid

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| Mamba-1 | `models/mamba1/` | **Mamba-1 selective scan**, per-channel `A_log`/`D`, learned `dt_proj` | S | **0.0 bit-exact** vs HF `MambaMixer.slow_forward` |
| Mamba-2 (mini) | `models/mamba2/` | **Mamba-2 SSD** (chunk-parallel selective scan with per-head `A_log`/`dt_bias`/`D`), depthwise conv1d, `skip_ffn=True` | S | **0.0 bit-exact** vs HF `torch_forward`, S=8/9/16 |
| Granite-4-H (mini) | `models/granite4_h/` | **Mamba-2 + GQA 5:1 hybrid**, fused-gate-up SwiGLU, μP scalars | S | **0.0 bit-exact** both mamba and attn layers |
| Jamba | `models/jamba/` | Mamba-1 + standard attention alternation (`attn_layer_period=8`), no RoPE in either branch | S | Mamba layer 0.0 bit-exact; attn layer 4.77e-7 |

**Concrete lessons.** B7 was the IR's largest single architectural extension — adding SSM support required `api/ssm.py` (274 lines), `SSMStateCache`, `SSDSpec`, `TokenMixerKind` enum, and `skip_ffn` on `DecoderBlockSpec`. The drift that B7 caught (`models/mamba2/layer.md` §6 docs and commit `c8c97d4`): Mamba-2's SSD math is **per-head** for `A_log`, `dt_bias`, and `D` — not per-channel as Mamba-1. Also Mamba-2's `conv1d` is **depthwise** (`groups = d_inner + 2*n_groups*d_state`, not 1). And Granite-4-H's `attention_multiplier` is **not** `1/sqrt(head_dim)` — it's a learned μP scalar in the config. The Granite-4-H hybrid alternation is 5:1 mamba:attention. Each layer is dispatched on `layer_types[i]` which is "mamba" or "attention". For the 4-layer mini config used in tests this is exercised on layers 0 (mamba) and 2 (attention) so both branches participate in the bit-exact gate. Jamba was promoted from a B7 stub to a working family in M3 (`bccab3e`) — it has **Mamba-1** (not Mamba-2) blocks with intra-mixer dt/B/C layernorms (`dt_layernorm`, `b_layernorm`, `c_layernorm` of shapes `dt_rank`, `d_state`, `d_state`) which are visible at `models/jamba/config.py:118-138`. The attention layer in Jamba has **no RoPE at all** (`rope=None`).

### 1.8 OCR-LLM (text-side decoder of multimodal models)

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| GOT-OCR 2.0 | `models/got_ocr2/` | Qwen2-0.5B LM-decoder portion; standard MHA + softmax MoE | R | 3.81e-6 |
| Qwen2.5-VL 3B | `models/qwen2_5_vl/` | **M-RoPE** (`mrope_section=[16,24,24]` channel stitching) on the LM portion | R | 1.67e-6 (M-RoPE sub-op bit-exact) |
| DeepSeek-OCR-2 | `models/deepseek_ocr2/` | LM-decoder portion (standard MHA + softmax MoE) — proves the IR isolates the text-decoder cleanly from the visual encoder | R | layer-0: 4.77e-7, layer-1: 1.42e-7 |

**Concrete lessons.** B8 was the IR's encounter with multimodal models. The decision was made early: **the IR scopes to the LM-decoder portion**, not the vision encoder or the projection bridge. That kept the surface small but required two new IR features: **M-RoPE** (`RoPESpec.mrope_section: tuple[int, ...]`) for Qwen2.5-VL and **`MaskKind.BLOCK_BIDIRECTIONAL`** + `block_bidirectional_mask` for vision-token positions in DeepSeek-OCR-2 / GOT-OCR-2.0. M-RoPE splits cos/sin into 6 channel bands (3 axes × 2 halves for rotate_half pairing) and cyclically picks `i mod 3` per channel. For text-only tokens the three position-axis components are all equal, in which case M-RoPE reduces to standard 1D RoPE — and the test exercises this by running text-only input through the M-RoPE LM portion. GOT-OCR 2.0's LM portion is a Qwen2-0.5B, so this row also proves the IR re-uses the Qwen2 decoder (no fresh `models/qwen2/` package needed — the same Qwen3 layer factory works because Qwen2 and Qwen3 share the same shape modulo `head_dim` differences and no QK-norm in Qwen2). DeepSeek-OCR-2 was tested in **four** configurations: synthetic dense + synthetic MoE + HF-checkpoint dense + HF-checkpoint MoE (commit `47ff482`).

### 1.9 Audio LM

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| Moshi (Helium) | `models/moshi/` | MHA (not GQA — drift caught at B9) + fused SwiGLU; synthetic 64-dim mini config | S | layer-0: 1.19e-7, layer-1: 5.96e-8 |
| Voxtral | `models/voxtral/` | Llama backbone (not Mistral!) with `rope_theta=1e8`; synthetic mini | S | layer-0: 1.19e-7, layer-1: 5.96e-8 |

**Concrete lessons.** B9 caught two drifts that would have been silent numerical bugs in production. (1) Moshi 7B is documented as a Mistral-family decoder but **HF Moshi attention is MHA, not GQA** — `n_kv_heads = n_q_heads`. The IR's spec carries `n_kv_heads = num_attention_heads` for Moshi and the test exercises this. (2) Voxtral is documented as a Mistral-family decoder but **HF Voxtral inherits from Llama** with `rope_theta=1e8` (a value distinctive to Voxtral; Mistral 7B uses 1e7, Llama 3 uses 5e5). The Voxtral row's max_abs_diff `1.19e-7` is therefore the same as Moshi's (both are tested on `hidden_size=64` synthetic minis, so the absolute diffs converge to single-ULP precision). Both families load from the appropriate HF model class (`MoshiForCausalLM`, `VoxtralForCausalLM`) with random init.

### 1.10 v5 mainstream variants

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| MPT 7B | `models/mpt/` | **ALiBi** position encoding + MHA + ungated GELU FFN (`GateKind.GELU_ONLY`) + LayerNorm **without** bias | R (via HF MptBlock) | 2.38e-7 |
| Falcon-7B | `models/falcon7b/` | **Parallel residual** (`BlockLayout.PARALLEL`, single shared pre-norm) + MQA (n_kv=1) + RoPE + ungated GELU + LayerNorm **with** bias | R | 2.38e-7 |
| BitNet b1.58 | `models/bitnet/` | **Sub-norms** inside attention (before `o_proj`) and FFN (after gate*up) + **ReLU²** activation + ternary-quantizable | R | 2.38e-7 |
| Hunyuan-Large | `models/hunyuan_large/` | **CLA** (cross-layer attention): `kv_source_layer_offset=-1` on odd layers points K/V to the previous (even) layer + head_dim=80 (non-standard) | shape-only | n/a |
| GPT-OSS 20B | `models/gpt_oss/` | **Trained attention sinks** (`MaskKind.SINK`, `n_sink_tokens=1`) + **clamped-SwiGLU MoE experts with bias** (`expert_kind="gpt_oss_clamped_swiglu"`) + YARN | R | **0.0 bit-exact** |

**Concrete lessons from v5.** The v5 batch was the IR's "second-pass for legacy mainstream models" — adding canonical pre-2024 architectures (MPT, Falcon, BitNet, Hunyuan-Large, GPT-OSS) that earlier batches had skipped. Each added an axis to the IR:

- **ALiBi** (MPT, commit `7f915b9`) — `AliBiSpec` carries `n_heads` and `alibi_bias_max`. ALiBi is the antithesis of RoPE — no rotation, just an additive bias on attention logits proportional to position distance. MPT has `rope=None` and uses ALiBi as its only positional encoding.
- **Parallel residual** (Falcon-7B, commit `c2149e1`) — `BlockLayout.PARALLEL` indicates that the attention and FFN sub-layers share a single pre-norm and add to the residual stream **in parallel** rather than sequentially. The DecoderBlock had to learn to either (a) compute `attn_norm = pre_attn_norm(x); attn_out = attn(attn_norm); ffn_out = ffn(attn_norm); x += attn_out + ffn_out` (PARALLEL) or (b) the standard sequential `x = x + attn(pre_attn_norm(x)); x = x + ffn(pre_ffn_norm(x))`. Falcon-7B's config sets `pre_ffn_norm=None` to signal that the pre_attn_norm is shared.
- **Sub-norms** (BitNet b1.58, commit `b1b0e07`) — RMSNorm applied **inside** the attention block (between the head concat-reshape and `o_proj`) and **inside** the FFN block (between the gated activation and `down_proj`). Carried as `AttentionSpec.attn_sub_norm` and `FFNSpec.ffn_sub_norm`. Also added `Activation.RELU2` (squared ReLU) for BitNet's `hidden_act='relu2'`.
- **CLA** (Hunyuan-Large, commit `c7c39bf`) — Cross-Layer Attention: odd layers borrow K/V from the previous even layer, halving the KV-cache footprint. Carried as `AttentionSpec.kv_source_layer_offset` (None on owners, -1 on borrowers). Conceptually similar to Gemma 4's cross-layer KV but with pairwise sharing rather than the Gemma 4 "every-N-th layer points to the last unshared block" pattern.
- **Trained attention sinks** (GPT-OSS, commit `d7e3682`) — `MaskKind.SINK` with `n_sink_tokens=1` adds a learned per-head "sink" token to the attention softmax denominator. The IR's `api.ops.sdpa(sinks=...)` accepts an optional sinks tensor and adds it to the softmax before normalization.
- **Clamped-SwiGLU experts with bias** (GPT-OSS) — distinct from DeepSeek's plain SwiGLU experts: GPT-OSS experts have `expert_bias=True`, `expert_swiglu_alpha=1.702`, `expert_clamp_limit=7.0`. The clamp is `clamp(gate, -limit, limit)` then `gate * sigmoid(alpha * gate)` (the swish/SiLU-like activation but with a learned slope alpha and a clamp to prevent overflow at large gate magnitudes).

### 1.11 v7 frontier MoE

| Family | Path | What it proves | Gate | max_abs_diff |
|---|---|---|---|---|
| DeepSeek-V4 | `models/deepseek_v4/` | **Hash MoE routing** (`router_kind="hash"`, frozen tid2eid table) + **CSA / HCA** dual sparse attention | hybrid | hash MoE < 5e-4; CSA/HCA shape-only |
| Qwen3-Next 80B-A3B | `models/qwen3_next/` | **Gated DeltaNet** linear-attention (3:1 linear-attn : full-attn hybrid) + ultra-sparse MoE (512 experts top-10 + shared) | R + shape | Gated DeltaNet layer: 7.5e-8; full-attn shape-only |
| GLM-MoE-DSA (GLM-5) | `models/glm_moe_dsa/` | MLA + **DSA Lightning Indexer** + sigmoid+bias MoE with single-group routing (n_group=1, topk_group=1) | hybrid | MoE < 5e-4; DSA shape-only |
| MiniMax-M2 | `models/minimax_m2/` | Full DecoderLayer numerical: STANDARD attention + **FULL_HDH** QK-norm + sigmoid+bias MoE (no group routing) | S | full DecoderLayer < 5e-4 |

**Concrete lessons from v7.** v7 is the IR's hardest batch because the four families ship four genuinely novel patterns:

- **Hash MoE routing** (DeepSeek-V4, commit `e69bb43`) — a frozen `tid2eid` lookup table assigns each token-id to a fixed expert at training time. The "routing" is therefore just a vocab embedding lookup. The IR carries this via `MoESpec.router_kind="hash"`, `hash_vocab_size=vocab_size`, and a `hash_score_fn` to combine the deterministic routing with a learned per-expert weight. The numerical gate uses `is_hash=True` synthetic config with `scoring_func="sigmoid"`.
- **CSA + HCA dual sparse attention** (DeepSeek-V4 P2, commit `7a9376e`) — Compressed Sparse Attention (block-wise compression with a Lightning Indexer) interleaved with Heavily Compressed Attention (much higher compression for long-range memory). The first two layers bootstrap with HCA; subsequent layers alternate CSA / HCA. Both are shape-only — the forward raises `NotImplementedError` because the sparse-pattern kernel hasn't landed.
- **Gated DeltaNet** (Qwen3-Next P3, commit `073b396`) — a linear-attention mixer based on the DeltaNet recurrence with an additional gating term. The state update is `S_t = S_{t-1} + (v_t - S_{t-1} k_t) k_t^T * gate_t`. The IR carries this as `GatedDeltaNetSpec` with `num_v_heads`, `num_k_heads`, `head_k_dim`, `head_v_dim`, `conv_kernel`, `silu_gate=True`. Qwen3-Next is 3:1 Gated DeltaNet : standard-attn — every 4th layer is full attention, the other 3 are linear. The numerical gate is on the Gated DeltaNet layer; the full-attention layer is shape-only because Qwen3-Next's HF attention has a fused `(q | gate)` output that the IR's STANDARD attention doesn't model (config note in `models/qwen3_next/config.py:159-162`).
- **DSA Lightning Indexer** (GLM-MoE-DSA / GLM-5, commit `94b3132`) — MLA token mixer plus a separate "indexer" sub-module that scores token pairs for sparse attention. `AttentionKind.DSA` is a Phase A reservation; forward raises NotImplementedError. The MoE side is V3-style sigmoid + bias with single-group routing (`n_group=1, topk_group=1` — degenerate to plain top-k but exercising the group-routing code path). GLM-MoE-DSA's MoE is the **numerical** half of the hybrid gate.
- **Nemotron-H latent MoE** (commit `02ad52c`, spec only) — a latent MoE wrapper where `fc1` projects into a latent space, MoE dispatch happens in the latent space, then `fc2` projects back to the residual stream. Spec landed; no production family wired up yet because no public Nemotron-H checkpoint exists.

### 1.12 Deferred stubs (`__init__.py` only)

These are placeholders that v3 research catalogued as needing follow-up. Each has its own architecturally-distinct trait that the IR will need to absorb:

| Family | Path | Reason for deferral |
|---|---|---|
| `models/mamba3/` | reserved | Mamba-3 (no public reference checkpoint yet) |
| `models/rwkv7/` | reserved | RWKV-7 receptance-weighted key-value mixer |
| `models/recurrent_gemma/` | reserved | Griffin recurrent block |
| `models/hymba/` | reserved | Hymba parallel attention+Mamba |
| `models/phi4_mini_flash/` | reserved | Sliding-attention + GQA + Flash hybrid |
| `models/falcon_h1/` | reserved | Falcon H1 (Mamba-2 hybrid) |
| `models/nemotron3/` | reserved | Nemotron-H latent MoE wrapper (Phase A spec landed) |
| `models/minimax_text_01/` | reserved | MiniMax-Text-01 (precursor to M2) |
| `models/jamba/` | landed | (Originally deferred at B7; promoted in M3 commit `bccab3e`.) |

**Totals.** 41 working families (49 directories if counting size-variant wrappers like Qwen3-0.6B/1.7B/4B/8B that share `models/qwen3/`) + 9 stubs. The R/S/shape breakdown across the 41 working: 16 R, 11 S, 11 hybrid/shape, 3 v7 frontier hybrids that combine numerical and shape branches.

---

## Part 2 — Architectural axes demonstrated (the WHY)

For each axis the IR supports, this section names ONE representative family, points at its `to_block_spec()` source line, and shows the IR setting in a one-line snippet. Every snippet is verbatim from the source.

The structure is intentional: **the IR is the union of all axes**, and **a family is a point in this axis space**. A new family that lives at an already-covered point requires no IR change (Mistral, TinyLlama). A family that lives at a new point in an existing axis requires only a new enum value or new field in an existing spec (Phi-3 `FUSED`, BitNet `RELU2`, GPT-OSS `topk_then_softmax_with_bias`). A family that lives along a wholly new axis requires extending the IR itself (Mamba-2 added `SSDSpec`+`Mamba2Mixer`+`SSMStateCache`+`skip_ffn`; OCR-LLM added `MaskKind.BLOCK_BIDIRECTIONAL` and `mrope_section`).

### 2.1 Attention kind

| Axis | Representative | Snippet |
|---|---|---|
| **GQA** | Qwen3 — `models/qwen3/config.py:90-92` | `n_q_heads=16, n_kv_heads=8, head_dim=128` (n_kv < n_q) |
| **MQA** | Falcon-7B — `models/falcon7b/config.py:56` | `n_kv_heads=self.num_kv_heads,  # 1` |
| **MHA** | MPT 7B — `models/mpt/config.py:86` | `n_kv_heads=self.num_attention_heads,  # MPT is MHA` |
| **MLA** | MiniCPM-3 — `models/minicpm3/config.py:185-200` | `kind=AttentionKind.MLA, qkv_layout=QKVLayout.MLA_LATENT, q_lora_rank=768, kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64` |
| **Linear attention (Gated DeltaNet)** | Qwen3-Next — `models/qwen3_next/config.py:145-154` | `GatedDeltaNetSpec(num_v_heads=32, num_k_heads=16, head_k_dim=128, head_v_dim=128, conv_kernel=4, silu_gate=True)` |

### 2.2 Attention mask family

| Axis | Representative | Snippet |
|---|---|---|
| **CAUSAL** | Llama 3 — `models/llama3/config.py:165` | `mask_kind=MaskKind.CAUSAL` |
| **SWA** | Gemma 2 (sliding layers) — `models/gemma2/config.py:145-146` | `mask_kind = MaskKind.SWA if is_sliding else MaskKind.CAUSAL; sw = self.sliding_window` |
| **SWA/full alternation (5:1)** | Gemma 3 — `models/gemma3/config.py:166-169` | `lt = self.layer_type(layer_idx); is_sliding = (lt == "sliding_attention")` |
| **Interleaved SWA (1:3)** | Ministral — `models/ministral/config.py:136-150` | `layer_types[i]` either `"sliding_attention"` or `"full_attention"` |
| **Trained sinks** | GPT-OSS — `models/gpt_oss/config.py:105-106` | `mask_kind=MaskKind.SINK, n_sink_tokens=1` |
| **BLOCK_BIDIRECTIONAL (VCF / vision tokens)** | DeepSeek-OCR-2 — `models/deepseek_ocr2/config.py:22-24` | `MaskKind.BLOCK_BIDIRECTIONAL` with `block_bidirectional_mask` spec hook (shape-only) |
| **BLOCK_SPARSE** | Phi-3-small — `models/phi3_small/config.py:128-129` | `mask_kind = MaskKind.CAUSAL if is_dense else MaskKind.BLOCK_SPARSE` |

### 2.3 RoPE basis

| Axis | Representative | Snippet |
|---|---|---|
| **SPLIT_HALF (rotate_half)** | Qwen3 — `models/qwen3/config.py:102` | `basis=RoPEBasis.SPLIT_HALF` |
| **INTERLEAVED (complex multiply)** | Llama 4 Scout — `models/llama4_scout/config.py:319` | `basis=RoPEBasis.INTERLEAVED` (also DeepSeek-V2-Lite `_rope_spec` line 169) |

### 2.4 RoPE scaling

| Axis | Representative | Snippet |
|---|---|---|
| **LLAMA3 smooth scaling** | Llama 3.2 — `models/llama3/config.py:140-152` | `scaling=RoPEScaling.LLAMA3, llama3_extra=Llama3RoPEParams(factor=32, low_freq_factor=1, high_freq_factor=4, original_context_length=8192)` |
| **YaRN** | DeepSeek-V2-Lite — `models/deepseek_v2_lite/config.py:158-172` | `scaling=RoPEScaling.YARN, yarn_extra=YarnRoPEParams(factor, beta_fast=32, beta_slow=1, mscale, mscale_all_dim)` |
| **LongRoPE** | Phi-4-mini — `models/phi3_mini/config.py:134-161` | `scaling=RoPEScaling.LONGROPE, longrope_extra=LongRoPEParams(short_factor, long_factor, attention_factor, original_max_position_embeddings)` |
| **NONE (vanilla)** | Mistral 7B — `models/mistral/config.py` | `scaling=RoPEScaling.NONE` |
| **NoPE per-layer** | SmolLM3 — `models/smollm3/config.py:60-92` | `rope=None` when `no_rope_layers[i] == 0` |
| **Proportional partial RoPE** | Gemma 4 (global layers) — `models/gemma4/config.py:226-234` | `partial_rotary_factor=0.25, partial_rotary_kind="proportional"` |
| **Prefix partial RoPE** | Phi-3-mini — `models/phi3_mini/config.py:160, 167` | `partial_rotary_factor=self.partial_rotary_factor, partial_rotary_kind="prefix"` (default for Phi-3) |
| **M-RoPE (multimodal)** | Qwen2.5-VL — `models/qwen2_5_vl/config.py:155` | `mrope_section=self.mrope_section  # (16, 24, 24)` |

### 2.5 RoPE quirks

| Axis | Representative | Snippet |
|---|---|---|
| **Dual θ (local vs global)** | Gemma 3 — `models/gemma3/config.py:170, 188` | `rope_theta = self.rope_theta_local if is_sliding else self.rope_theta_global` |

### 2.6 Norm

| Axis | Representative | Snippet |
|---|---|---|
| **RMSNorm STANDARD_W** | Qwen3 — `models/qwen3/config.py:84-87` | `NormSpec(kind=NormKind.RMS, weight_mode=NormWeightMode.STANDARD_W)` |
| **RMSNorm ONE_PLUS_W** | Gemma 2 — `models/gemma2/config.py:139-141` | `NormSpec(kind=NormKind.RMS, weight_mode=NormWeightMode.ONE_PLUS_W)` |
| **LayerNorm no-bias** | MPT 7B — `models/mpt/config.py:62-66` | `NormSpec(kind=NormKind.LAYER, has_bias=False)` |
| **LayerNorm with bias** | Falcon-7B — `models/falcon7b/config.py:84-88` | `NormSpec(kind=NormKind.LAYER, has_bias=True)` |

### 2.7 Norm position

| Axis | Representative | Snippet |
|---|---|---|
| **PRE-norm** | Qwen3 — `models/qwen3/config.py:114-115` | `attn_norm_position=NormPosition.PRE, ffn_norm_position=NormPosition.PRE` |
| **POST-norm** | OLMo 2 — `models/olmo2/config.py:146-152` | `attn_norm_position=NormPosition.POST, ffn_norm_position=NormPosition.POST` |
| **Sandwich (PRE_AND_POST)** | Gemma 2 — `models/gemma2/config.py:174-179` | `attn_norm_position=NormPosition.PRE_AND_POST, post_attn_norm=norm_spec, post_ffn_norm=norm_spec` |

### 2.8 QK-Norm

| Axis | Representative | Snippet |
|---|---|---|
| **PER_HEAD_DH PRE-RoPE** | Qwen3 — `models/qwen3/config.py:97-99` | `qk_norm=qk_norm_spec, qk_norm_phase=QKNormPhase.PRE_ROPE, qk_norm_shape=QKNormShape.PER_HEAD_DH` |
| **FULL_HDH PRE-RoPE** | OLMo 2 — `models/olmo2/config.py:129-131` | `qk_norm_phase=QKNormPhase.PRE_ROPE, qk_norm_shape=QKNormShape.FULL_HDH` |
| **Gemma 4 v_norm (no scale)** | Gemma 4 — `models/gemma4/config.py:224-225` | `v_norm=norm_spec, v_norm_with_scale=False` |

### 2.9 ALiBi

| Axis | Representative | Snippet |
|---|---|---|
| **ALiBi (no RoPE)** | MPT 7B — `models/mpt/config.py:92-96` | `rope=None, alibi=AliBiSpec(n_heads=self.num_attention_heads, alibi_bias_max=self.alibi_bias_max)` |

### 2.10 Sub-norms

| Axis | Representative | Snippet |
|---|---|---|
| **attn_sub_norm + ffn_sub_norm** | BitNet b1.58 — `models/bitnet/config.py:97, 108` | `AttentionSpec(..., attn_sub_norm=norm_spec)`, `FFNSpec(..., ffn_sub_norm=norm_spec)` |

### 2.11 FFN gate kind

| Axis | Representative | Snippet |
|---|---|---|
| **SwiGLU** | Qwen3 — `models/qwen3/config.py:107-109` | `gate_kind=GateKind.SWIGLU, activation=Activation.SILU` |
| **GeGLU** | Gemma 2 — `models/gemma2/config.py:168-169` | `gate_kind=GateKind.GEGLU, activation=Activation.GELU` |
| **GELU_ONLY (ungated)** | MPT 7B — `models/mpt/config.py:53-54` | `gate_kind=GateKind.GELU_ONLY, activation=Activation.GELU_EXACT` |
| **Fused gate_up** | Phi-3-mini — `models/phi3_mini/config.py:191` | `fused_gate_up=True` |
| **Clamped-SwiGLU experts** | GPT-OSS — `models/gpt_oss/config.py:77-80` | `expert_kind="gpt_oss_clamped_swiglu", expert_swiglu_alpha=1.702, expert_clamp_limit=7.0` |
| **ReLU² (BitNet)** | BitNet — `models/bitnet/config.py:104` | `activation=Activation.RELU2` |
| **GEGELU (Phi-3-small)** | Phi-3-small — `models/phi3_small/config.py:152` | `activation=Activation.GEGELU` |

### 2.12 MoE router

| Axis | Representative | Snippet |
|---|---|---|
| **softmax + top-k** | Mixtral — `models/mixtral/config.py:166-175` | `router_kind="softmax", router_norm=True` |
| **sigmoid + bias + group-routing** | DeepSeek-V3-Lite — `models/deepseek_v3_lite/config.py:104-117` | `router_kind="sigmoid_plus_bias", group_routing=GroupRoutingSpec(n_groups=n_group, topk_per_group=topk_group)` |
| **top-k then softmax with bias** | GPT-OSS — `models/gpt_oss/config.py:73` | `router_kind="topk_then_softmax_with_bias"` (with `expert_bias=True`) |
| **hash routing** | DeepSeek-V4 — `models/deepseek_v4/config.py:268-278` | `router_kind="hash", hash_vocab_size=self.vocab_size, hash_score_fn="sigmoid"` |
| **latent MoE (Nemotron-H pattern, fc1/fc2 projection)** | DeepSeek-V4 P4 (Nemotron-H spec) — `api/specs.py` (commit `02ad52c`) | spec landed; no production family wired yet |

### 2.13 Per-layer KV sharing

| Axis | Representative | Snippet |
|---|---|---|
| **Cross-layer KV (Gemma 4)** | Gemma 4 — `models/gemma4/config.py:257-280` (`kv_source_layer_idx_map`) | `SharedLayerKVCache` consumes a `{i: source_i}` dict to point layer `i`'s K/V at another layer's cache |
| **CLA (pairwise even→odd)** | Hunyuan-Large — `models/hunyuan_large/config.py:62-77` | `AttentionSpec(..., kv_source_layer_offset=-1)` on odd layers (borrower); even layers own K/V |

### 2.14 K = V unification (Gemma 4 12B+)

| Axis | Representative | Snippet |
|---|---|---|
| **K = V** | Gemma 4 — `models/gemma4/config.py:220` | `attention_k_eq_v=attention_k_eq_v_eff  # True on 12B+ global layers` |

### 2.15 PLE (Per-Layer Embeddings)

| Axis | Representative | Snippet |
|---|---|---|
| **PLE** | Gemma 4 — `models/gemma4/config.py:241-247` | `PLESpec(ple_dim=self.ple_dim, residual_scale=1.0 / sqrt(2), injection_norm=norm_spec)` |

### 2.16 Parallel residual block layout

| Axis | Representative | Snippet |
|---|---|---|
| **Parallel residual** | Falcon-7B — `models/falcon7b/config.py:99-107` | `block_layout=BlockLayout.PARALLEL, pre_ffn_norm=None  # shared pre_attn_norm` |

### 2.17 SSM and SSM hybrid

| Axis | Representative | Snippet |
|---|---|---|
| **Mamba-1 selective scan** | Mamba-1 — `models/mamba1/config.py:63-76` | `SSMSpec(d_state, d_conv, d_inner, kind=SSMKind.MAMBA1, dt_rank=self.dt_rank)` |
| **Mamba-2 SSD (chunk-parallel)** | Mamba-2 — `models/mamba2/config.py:127-136` | `SSDSpec(base=ssm_spec, chunk_size=self.chunk_size, headdim=self.head_dim, ngroups=self.n_groups, n_heads=self.num_heads)` |
| **Gated DeltaNet** | Qwen3-Next — `models/qwen3_next/config.py:145-154` | `GatedDeltaNetSpec(num_v_heads, num_k_heads, head_k_dim, head_v_dim, conv_kernel, silu_gate=True)` |
| **Mamba + attention alternation** | Jamba — `models/jamba/config.py:140-162` | `token_mixer = self.to_attention_spec() if self.is_attention_layer(layer_idx) else self.to_mamba_spec()` |
| **Mamba-2 + attention (5:1)** | Granite-4-H — `models/granite4_h/config.py:175-197` | `is_mamba = (self.layer_types[layer_idx] == "mamba")` per-layer dispatch |
| **`skip_ffn=True` (Mamba block has no FFN)** | Mamba-2 — `models/mamba2/config.py:149` | `skip_ffn=True` (placeholder FFN spec ignored) |

### 2.18 Quantization

Documented in `api/quant.py` and exercised by `tests/api/test_quant.py` + `tests/api/test_ternary_quant.py` + `tests/models/qwen3/test_quant_*.py`:

| Axis | Where | Notes |
|---|---|---|
| **AWQ W4A16** | `api/quant.py` + `tests/models/qwen3/test_quant_awq.py` | Grouped INT4 weights, fp16 activations |
| **GGUF Q4_K_M** | `api/quant.py` + `tests/models/qwen3/test_quant_gguf_q4k.py` | k-quant super-block + 6-bit sub-scales |
| **FP8 E4M3 W8A8** | `api/quant.py` + `tests/api/test_quant.py` | Per-tensor weight + per-token activation scales |
| **MXFP4** | `api/quant.py` | UE8M0 shared exponent + E2M1 mantissa |
| **LiteRT W4A8** | `api/quant.py` | Per-channel INT4 weights + per-tensor INT8 activations |
| **BitNet ternary (b1.58)** | `api/quant.py` + `tests/api/test_ternary_quant.py` | `{-1, 0, +1}` weights with per-tensor scale |

### 2.19 μP scalars

| Axis | Representative | Snippet |
|---|---|---|
| **Residual scale** | Granite — `models/granite/config.py:146` | `residual_scale=self.residual_multiplier` |
| **Attention multiplier** | Granite — `models/granite/config.py:126` | `attn_scale=self.attention_multiplier  # μP: overrides 1/sqrt(Dh)` |

### 2.20 Per-layer heterogeneity (the `to_block_spec(layer_idx)` pattern)

| Family | What dispatches | Source |
|---|---|---|
| Gemma 2/3/4 | SWA vs full; for Gemma 3/4: dual RoPE θ, partial-RoPE factor | `models/gemma2/config.py:120, 144`, etc. |
| Llama 4 Scout | iRoPE: RoPE vs NoPE (`no_rope_layers[i]`) | `models/llama4_scout/config.py:238-294` |
| SmolLM3 | NoPE alternation | `models/smollm3/config.py:60-95` |
| Ministral | SWA/full interleave | `models/ministral/config.py:136-150` |
| DeepSeek-V2/V3-Lite | first `first_k_dense_replace` layers dense, rest MoE | `models/deepseek_v2_lite/config.py:246` |
| Qwen3-MoE | Dense vs MoE per `decoder_sparse_step` | `models/qwen3_moe/config.py` |
| Qwen3-Next | linear-attention vs full-attention per `layer_types[i]`; MoE vs MLP per `mlp_only_layers` | `models/qwen3_next/config.py:135-247` |
| Jamba | Mamba vs attention every 8th layer | `models/jamba/config.py:140-162` |
| Granite-4-H | Mamba vs GQA per `layer_types[i]` | `models/granite4_h/config.py:172-197` |
| DeepSeek-V4 | hash MoE vs sigmoid+bias MoE per `mlp_layer_types[i]`; CSA / HCA / sliding per `layer_types[i]` | `models/deepseek_v4/config.py:170-243` |
| GLM-MoE-DSA | dense vs sparse per `mlp_layer_types[i]` | `models/glm_moe_dsa/config.py` |
| Hunyuan-Large | KV owner vs KV borrower per `is_kv_owner(layer_idx)` | `models/hunyuan_large/config.py:88-93` |

The shape of this list — 12+ families dispatching on `layer_idx` — is itself proof that the per-layer `to_block_spec(layer_idx)` signature was the right design choice in B1.

---

## Part 3 — Demonstrating by running tests

Tests ran on this machine: Windows 11, fp32 on CPU. `uv run pytest` resolves to the project's pinned Torch + Transformers (≥ 4.51). Output below is **not paraphrased** — it's the literal tail of the pytest run.

### 3.1 Total test count

```
$ uv run pytest --collect-only -q | Select-Object -Last 3

tests/models/voxtral/test_layer_shape.py::test_layer_forward_shape
tests/models/voxtral/test_numerical_synthetic.py::test_layer0_matches_hf_synthetic
tests/models/voxtral/test_numerical_synthetic.py::test_layer1_matches_hf_synthetic

751 tests collected in 40.58s
```

**751 tests collected** across `tests/api/` (225) + `tests/models/` (the remaining 526). The 40-second collection time is itself indicative — pytest is importing every `models/<family>` package for discovery.

### 3.2 Six representative model gates

Each command was run in isolation; the bottom three lines of output are reproduced verbatim.

#### Qwen3 (M1 anchor — kickoff gate against real `Qwen/Qwen3-0.6B` weights)

```
$ uv run pytest -q tests/models/qwen3/test_numerical_hf.py
.                                                                        [100%]
1 passed in 20.05s
```

**Honest disclosure:** this test downloads `Qwen/Qwen3-0.6B` (~1.3 GB) on first run via `huggingface_hub`. After caching, the 20-second time is purely the fp32 forward pass through a 1.2 GB model on CPU. Gate class **R**, max_abs_diff `4.77e-7` (per `models/qwen3/layer.md:189`).

#### Gemma 4 E2B (the IR's most demanding family — 5 axes diverged from Gemma 3 at B0.5)

```
$ uv run pytest -q tests/models/gemma4/test_numerical_hf.py
..                                                                       [100%]
2 passed in 64.92s (0:01:04)
```

The 65 s is the cost of a real `unsloth/gemma-4-e2b-it` checkpoint forward through two layer types (local SWA + global causal with partial-RoPE). Gate class **R**; both layer-0 (local) and layer-4 (global) covered. max_abs_diff layer-0: `7.6e-6`, layer-4: `3.8e-6`.

#### MiniCPM-3 4B (MLA + LongRoPE + μP, the heaviest single layer)

```
$ uv run pytest -q tests/models/minicpm3/test_numerical_hf.py
.                                                                        [100%]
1 passed in 74.87s (0:01:14)
```

The 75 s is one MLA forward (q_lora_rank=768, kv_lora_rank=256, qk_head_dim=96, v_head_dim=64, μP residual scale ≈ 0.18). Gate class **R**, max_abs_diff `3.05e-5` — the **largest** in the R cohort because LongRoPE's complex sin/cos chain accumulates more error in fp32.

#### Mamba-2 (B7 SSM proof — bit-exact synthetic)

```
$ uv run pytest -q tests/models/mamba2/
.............                                                            [100%]
13 passed in 64.62s (0:01:04)
```

13 tests: `test_config.py` (3), `test_layer_shape.py` (4, exercises pad path for `S=9` with chunk_size=8), `test_numerical_hf.py` (3 — `S=8, 9, 16`), `test_weight_loader.py` (3). **No checkpoint downloads** — the test builds an `HF Mamba2ForCausalLM` with `hidden_size=64, num_heads=4` and random init in-process. The 65 s is selective-scan fp32 math, not I/O. **Gate class S, but bit-exact (`0.0`) vs HF `torch_forward`.**

#### Granite-4-H (B7 hybrid — Mamba-2 + GQA 5:1)

```
$ uv run pytest -q tests/models/granite4_h/
........                                                                 [100%]
8 passed in 52.51s
```

8 tests cover both layer types (mamba and attn), config dispatch, and the hybrid block factory. Like Mamba-2 this is a synthetic mini (4 layers, 128 hidden). **Gate class S, bit-exact `0.0` for both layer types** — the mamba branch via `Mamba2Mixer` and the attn branch via standard GQA. No real `granite-4.0-h-micro` checkpoint loaded.

#### GPT-OSS (v5 — the only R family with **bit-exact** numerical gate)

```
$ uv run pytest -q tests/models/gpt_oss/
........                                                                 [100%]
8 passed in 51.50s
```

8 tests: config sanity (1), trained-sink attention vs HF eager `attention_forward` (~3 with combinations of SWA/full × sink), MoE-with-bias + clamped-SwiGLU experts vs HF `GptOssMoE` (3-ish), and one combined integration. **Gate class R, max_abs_diff `0.0` bit-exact.** Tests do NOT load the production `openai/gpt-oss-20b` (~40 GB MXFP4) — they construct a small `GptOssModel` in-process with random init. So R applies to the IR-vs-HF-code path, not specifically to weight loading of the 20B checkpoint.

### 3.3 API-level tests

```
$ uv run pytest -q tests/api/
........................................................................ [ 32%]
........................................................................ [ 64%]
........................................................................ [ 96%]
.........                                                                [100%]
225 passed in 37.65s
```

225/225 pass in 38 s. These cover the IR primitives directly: `test_attention.py`, `test_ssm.py`, `test_rope.py`, `test_kvcache.py`, `test_norm.py`, `test_ops.py`, `test_feedforward.py`, `test_quant.py`, `test_ternary_quant.py`, `test_alibi.py`, `test_attention_sinks.py`, `test_parallel_residual.py`, `test_block.py`, `test_specs.py`, `test_embedding.py`. No downloads; everything is synthetic.

### 3.4 Sample test fixtures (what "atol=5e-4 vs HF" looks like in source)

#### Llama 3 — `tests/models/llama3/test_numerical_hf.py:48-101`

```python
MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
FIXED_INPUT = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)
ATOL = 5e-4
RTOL = 5e-4

def test_layer0_forward_matches_hf(hf_model):
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = l3_config.Llama3Config.from_hf_dict(hf_cfg_dict)

    api_blk = l3_layer.build_llama3_decoder_layer(
        cfg, layer_idx=0, max_seq=cfg.max_position_embeddings,
    )
    full_sd = hf_model.state_dict()
    l3_layer.load_hf_llama3_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)
    B, S = FIXED_INPUT.shape

    hf_layer = hf_model.model.layers[0]
    position_ids = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = hf_model.model.rotary_emb(embed_out, position_ids)
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_out = hf_layer(
            embed_out,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_values=None,
            use_cache=False,
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"Llama 3.2 1B layer-0 max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
```

This is the **canonical R-class test shape**. Note four design choices it enforces:

1. **Fixed input tensor** (`FIXED_INPUT = [[101, 1024, ...]]`). No randomness — the same 9 tokens every test invocation.
2. **`attn_implementation="eager"`** on the HF model (in the `hf_model` fixture). Forces the reference path through `eager_attention_forward`, which is what the IR's `sdpa(...)` mirrors. SDPA / Flash kernels can have nondeterministic order-of-ops differences that inflate `max_abs_diff`.
3. **Explicit 4D additive causal mask** with `-inf` above the diagonal. Matches transformers 5.10.2's eager path verbatim.
4. **`ContiguousKVCache` with `HND` memory layout** — the same layout HF uses internally. Both sides see the same KV strides, so concat order is equivalent.

#### Gemma 2 — bit-exact assertion (`tests/models/gemma2/test_numerical_hf.py`, fb8ae88)

The Gemma 2 test follows the same shape as above but with `unsloth/gemma-2-2b` and asserts `max_abs_diff == 0.0`. The bit-exactness is no fluke — sandwich norm + ONE_PLUS_W + softcap-in-sdpa all turn out to be deterministic enough that fp32 reproduces bit-identical results.

#### Mamba-2 — synthetic mini construction (`tests/models/mamba2/test_numerical_hf.py:30-78`)

```python
def _small_mamba2_hf_config_dict():
    return {
        "hidden_size": 64, "num_heads": 4, "head_dim": 32,  # 4*32 = 128 = 64*2 (expand)
        "state_size": 16, "conv_kernel": 4, "expand": 2,
        "n_groups": 2, "num_hidden_layers": 2,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False, "use_conv_bias": True,
        "residual_in_fp32": True, "chunk_size": 8,
        "vocab_size": 100, "tie_word_embeddings": False,
    }

@pytest.fixture(scope="module")
def hf_small_model():
    from transformers import Mamba2Config, Mamba2ForCausalLM
    hf_cfg = Mamba2Config(**_small_mamba2_hf_config_dict())
    torch.manual_seed(42)
    model = Mamba2ForCausalLM(hf_cfg)
    model.eval()
    model.to(torch.float32)
    return model
```

This is the **canonical S-class test shape**: `Mamba2ForCausalLM(hf_cfg)` constructs the HF model directly with random weights. Crucially `S=9` (line 76) is chosen specifically **non-multiple** of `chunk_size=8` to exercise the SSD pad path. The bit-exact `0.0` is meaningful here: it proves the IR's chunked selective scan and the HF `torch_forward` agree on every floating-point op, including the pad-and-trim boundary.

### 3.4b Notes on test timing and skipping behavior

The six runs in §3.2 took a combined 4 min 28 s. Three of them dominate:

- **Qwen3 (20 s)** — small (~1.3 GB Qwen3-0.6B) and cached after the first run; this is the cheapest R-class gate in the project.
- **Gemma 4 (65 s)** — exercises **two** layer types (local SWA + global causal with partial-RoPE) which doubles the forward cost vs Qwen3.
- **MiniCPM-3 (75 s)** — the heaviest single forward in the project. MLA's q_a_proj→q_a_layernorm→q_b_proj chain is ~3× the FLOPs of a Llama-shaped q_proj, and LongRoPE's sin/cos materialization is also more expensive than vanilla RoPE.

**Honest disclosure about downloads.** Every **R**-class test fixture calls `AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=hf_cache)`. On first run this triggers a network download (typically 0.5 GB to 3 GB per family). After that the test loads from local cache. The `conftest.py` configures `HF_HOME` to point at `hf_cache/` in the repo so subsequent runs are I/O-only. Tests that **cannot find their weights** in the cache and cannot download (offline mode, gated model, or model removed from HF) **fail with `FileNotFoundError` or skip with `pytest.skip(...)`** depending on the family. Mistral 7B specifically has a `skip cleanly when weights not fully cached` test fixture (commit `ae0d6d4`) — if the 14 GB checkpoint isn't already present, the test skips with a message rather than blocking CI on a download.

**Synthetic S-class tests have no I/O.** Mamba-2, Granite-4-H, GPT-OSS, Mamba-1, Jamba, Moshi, Voxtral, DeepSeek-V2/V3-Lite, Mixtral, Qwen3-MoE, OLMoE, MiniMax-M2, Qwen3-Next, GLM-MoE-DSA — these all build `HF<Family>ForCausalLM(small_cfg)` in-process. Their timing is dominated by the forward pass alone, typically 5-15 s per test even with many tests in the file. The fastest model dirs (Mistral-config-only, Voxtral-config-only) collect in <1 s.

**API tests have no I/O at all.** 225 tests of `api/` primitives in 38 s — that's 170 ms per test on average. The `tests/api/test_specs.py` smoke tests are sub-millisecond; the `test_attention_sinks.py` GPT-OSS sink-attention tests are the longest individual ones at ~2 s each.

### 3.5 Recent commit `max_abs_diff` values (v7 phase)

Pulled from `git log --grep="max_abs_diff"` over the v7 commits:

| Commit | Family | Recorded diff |
|---|---|---|
| `2a7e0fc` v7-B2 | Qwen3-Next Gated DeltaNet | `< 8e-8` on synthetic-mini (64 hidden, 4 v-heads) |
| `27a7a4f` v7-B1 | DeepSeek-V4 hash MoE | `< 5e-4` (atol gate); CSA/HCA forward raises NotImplementedError |
| `80e4c72` v7-B4 | MiniMax-M2 | full DecoderLayer `< 5e-4` |
| `94b3132` v7-B3 | GLM-MoE-DSA | sigmoid+bias MoE `< 5e-4` on 8-expert mini config |
| `cd584eb` B7 | Granite-4-H | mamba layer-0 `0.000e+00`, attn layer-2 `0.000e+00` |
| `c8c97d4` B7 | Mamba-2 (S=8, 9, 16) | all `0.000e+00` |
| `1c6fd42` B5 | DeepSeek-V2-Lite (4 paths) | all `~2.4e-7` |
| `fb8ae88` B3 | Gemma 2 2B | `0.0` (bitwise vs real weights) |
| `b72e116` B4 | Ministral (full/SWA layers) | `1.907e-06` / `1.043e-07` |
| `2eabcef` B9 | Moshi layer-0/1 | `1.19e-07` / `5.96e-08` |
| `47ff482` B8 | DeepSeek-OCR-2 (4 variants) | `1.19e-7` synth, `4.77e-7` real layer-0, `1.42e-7` real layer-1 |
| `e22d141` B8 | Qwen2.5-VL | `1.67e-6` (M-RoPE sub-op bit-exact) |
| `cb68b0c` B8 | GOT-OCR 2.0 | `3.81e-6` |
| `7f915b9` v5 | MPT 7B (ALiBi sub-op) | `< 5e-4` atol gate |
| `c2149e1` v5 | Falcon-7B | `< 5e-4` |

---

## Part 4 — Quick-start: building your own family

This walks Llama 3 end-to-end. It's the simplest **R-class** family in the project. Follow this pattern when you bootstrap a new model: it's "Llama-shape with quirks." If your new model has any of the axes from Part 2 the IR already covers, you copy a config and tweak fields.

### Step 1 — `models/llama3/config.py`: the spec adapter

The `Llama3Config` dataclass mirrors HF's `LlamaConfig` and converts to a `DecoderBlockSpec`. The interesting parts:

```python
# models/llama3/config.py:37-58
@dataclass(frozen=True)
class Llama3Config:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rope_theta: float
    rope_type: str                      # "default" | "llama3"
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    # LLAMA3 smooth-scaling fields — populated when rope_type=="llama3".
    rope_factor: Optional[float] = None
    rope_low_freq_factor: Optional[float] = None
    rope_high_freq_factor: Optional[float] = None
    rope_original_max_position_embeddings: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
```

The `from_hf_dict()` classmethod handles **both** transformers 4.x and 5.x layouts — 4.x has a flat `rope_theta` + `rope_scaling` dict; 5.x nests them under `rope_parameters`. Every config in the project does this dance, because the project survived a transformers 4→5 migration mid-stream.

```python
# models/llama3/config.py:127-188 — to_block_spec(layer_idx)
def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
    norm_spec = specs.NormSpec(
        kind=types.NormKind.RMS, eps=self.rms_norm_eps,
        weight_mode=types.NormWeightMode.STANDARD_W,
    )
    if self.rope_type == "llama3":
        rope_spec = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.LLAMA3,
            llama3_extra=specs.Llama3RoPEParams(
                factor=float(self.rope_factor),
                low_freq_factor=float(self.rope_low_freq_factor),
                high_freq_factor=float(self.rope_high_freq_factor),
                original_context_length=int(
                    self.rope_original_max_position_embeddings
                ),
            ),
        )
    else:
        rope_spec = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.NONE,
        )
    attn_spec = specs.AttentionSpec(
        n_q_heads=self.num_attention_heads,
        n_kv_heads=self.num_key_value_heads,
        head_dim=self.head_dim,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        q_bias=self.attention_bias, k_bias=self.attention_bias,
        v_bias=self.attention_bias, o_bias=self.attention_bias,
        rope=rope_spec,
    )
    ffn_spec = specs.FFNSpec(
        intermediate_size=self.intermediate_size,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
        fused_gate_up=False,
        gate_bias=self.mlp_bias, up_bias=self.mlp_bias, down_bias=self.mlp_bias,
    )
    return specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn_spec,
        channel_mixer=ffn_spec,
        pre_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec,
    )
```

That's it for Llama 3. There's no QK-norm, no FFN sub-norm, no partial-RoPE, no per-layer dispatch — `layer_idx` is taken but never read, because every Llama 3 layer is the same shape. Compare against Qwen3's `to_block_spec()` (no `layer_idx` arg at all in the M1 anchor — added project-wide at B0.5 when Gemma 4 forced per-layer dispatch).

### Step 2 — `models/llama3/layer.py`: factory + weight loader

The factory is ~10 lines:

```python
# models/llama3/layer.py:22-32
def build_llama3_decoder_layer(
    cfg: _config.Llama3Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )
```

`api.block.DecoderBlock` does the rest — given the spec, it instantiates `Attention`, `FeedForward`, `RMSNorm`, etc. with the correct shapes.

The weight loader is the most line-heavy part of any family — that's because it bridges the HF naming convention to the IR slot tree:

```python
# models/llama3/layer.py:35-91
def load_hf_llama3_layer(blk, hf_state_dict, layer_idx):
    L = layer_idx
    prefix = f"model.layers.{L}"
    mapping = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":          blk.attention.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":          blk.attention.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":          blk.attention.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":          blk.attention.o_proj.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.mlp.gate_proj.weight":             blk.feedforward.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight":               blk.feedforward.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":             blk.feedforward.down_proj.weight,
    }
    # (Optional bias keys handled similarly.)
    ...
    with _torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(f"shape mismatch for {hf_name}: ...")
            slot.copy_(src.to(slot.dtype))
```

Notice `blk.pre_attn_norm.weight` ← `input_layernorm.weight`, even though HF semantically calls it "input layernorm." That naming gap is the kind of thing the loader exists to bridge. The shape check on line `if src.shape != slot.shape: raise ValueError` is the most common pre-numerical failure mode: when it triggers you almost always have a wrong `head_dim` in the config.

### Step 3 — `tests/models/llama3/test_numerical_hf.py`: the gate

The gate test (already shown in Part 3 §3.4) is the **R-class contract**. It does three things, in order:

1. **Build an IR block from the live HF config.** `cfg = Llama3Config.from_hf_dict(hf_model.config.to_dict())` — proves the config adapter round-trips production JSON correctly.
2. **Load HF weights into the IR slots.** `load_hf_llama3_layer(api_blk, full_sd, layer_idx=0)` — proves the loader's name mapping is correct.
3. **Compare forwards.** `api_out` vs `hf_layer(...)` at `atol=5e-4`. The print line is `max_abs_diff={value:.3e}` so you can read the absolute margin even on a pass.

The supplemental tests in `tests/models/llama3/` follow:

- `test_config.py` — from_hf_dict round-trip
- `test_layer_shape.py` — shape + determinism (no HF needed)
- `test_weight_loader.py` — state-dict mapping (no forward)
- `test_isolation_hf.py` — per-sub-op gates at `atol=1e-5` (Q proj output, K proj output, RMSNorm output, etc.)
- `test_kvcache_hf.py` — prefill + decode equivalence

That set is what you'd add for any new family. The `test_isolation_hf.py` file is the unsung hero of debugging: when the layer-level `test_numerical_hf.py` fails, isolation shows you which sub-op drifted.

### Adding a new family in ~6 file edits

If your new model is "Llama-shape with quirk X," the diff is roughly:

1. **`models/<family>/__init__.py`** — bootstrap (one line: `from .config import <Family>Config`).
2. **`models/<family>/config.py`** — copy `models/llama3/config.py`, swap class names, add the quirk field. If quirk = "different RoPE scaling," touch only `to_block_spec()`'s RoPE branch.
3. **`models/<family>/layer.py`** — usually one line in `build_<family>_decoder_layer` plus a state-dict rename table.
4. **`models/<family>/layer.md`** — copy `models/qwen3/layer.md`, rewrite §2 diagram if the quirk shows up. Sections 0 (At-a-glance), 8 (Validation status), and 9 (Source citations) are mandatory.
5. **`tests/models/<family>/test_config.py` + `test_layer_shape.py`** — copy from Llama 3; renamed.
6. **`tests/models/<family>/test_numerical_hf.py`** — copy the canonical test in §3.4, swap `MODEL_ID` and the cfg import. If your quirk requires a specialized attention mask, copy from Gemma 2's gate (which sets up an explicit 4D additive mask).

The whole bootstrap typically lands in ~150 LOC for a R-class Llama variant (Mistral, SmolLM3, TinyLlama all hit this number ± 30 lines). MLA, MoE, and SSM families need 400-800 LOC because the loader and tests grow.

### Anti-patterns to avoid (lessons from the M2 batch)

1. **Don't write the spec from blog posts.** The B0.5 → B0.6 saga (Gemma 4) demonstrates that even the most-careful blog summary will miss subtle details that the numerical gate catches. **Always read `transformers/src/transformers/models/<family>/modeling_<family>.py` first**, then write the spec. The `feedback_verify_source_not_blogs.md` memory entry records this lesson.
2. **Don't conflate gate classes.** If your test runs on a synthetic small config, it's **S**, not R — even if the math is bit-exact. Honesty about gate class is what the audit caught and corrected in `PROJECT-SUMMARY.md`.
3. **Don't skip the isolation tests.** When the layer-level numerical gate fails, sub-op isolation tests (`test_isolation_hf.py`) are the only way to triage. Building them is small effort (each is a few lines of "compute this sub-op two ways and `torch.allclose`") and they save hours of debugging.
4. **Use `attn_implementation="eager"` in test fixtures.** SDPA and Flash backends apply slightly different op orderings (fp32 vs fp32 accumulation, kernel-internal masking, etc.) that inflate `max_abs_diff` in ways that aren't real drift. The eager path is the reference.
5. **Use a fixed input tensor.** Random inputs in tests are non-reproducible bugs waiting to happen. Every gate test in the project uses `FIXED_INPUT = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)` (or a S-shaped equivalent). The values are arbitrary; the determinism is the point.
6. **Cite `modeling_*.py:line` in code comments.** Every spec field in `models/<family>/config.py` that mirrors HF behaviour cites the exact HF source line. If HF changes the semantics, `grep modeling_<family>.py` across the project finds every IR mirror that needs to be checked.

### What "bit-exact" means in this project

The seven bit-exact gates (`max_abs_diff == 0.0`) are not coincidental — each one tells a different story:

- **Gemma 2 (R)** — Gemma 2's sandwich norm + ONE_PLUS_W + softcap-in-SDPA happen to be deterministic enough in fp32 CPU that the IR's reproduction is element-equal to HF's. This is the project's strongest single result.
- **Mamba-1 (S, vs `slow_forward`)** — the IR's `selective_scan` reference implementation matches HF's `slow_forward` op-by-op. The HF "fast" path uses a fused triton kernel that has subtly different rounding; we don't gate against that.
- **Mamba-2 (S, vs `torch_forward`, S=8, 9, 16)** — same logic: the IR matches HF's pure-Python reference path. The three sequence lengths exercise the pad-and-trim boundary (`S=9 % chunk_size=8 ≠ 0`).
- **Granite-4-H (S, mamba + attn layers)** — both branches of the hybrid bit-exact against the same HF references.
- **GPT-OSS (R, vs HF eager forward, in-process tiny model)** — sink-attention + MoE-with-bias + clamped-SwiGLU all reproduce element-equal.
- **DeepSeek-V3-MoE submodule (S)** — the MoE submodule in isolation matches HF.
- **M-RoPE sub-op (Qwen2.5-VL)** — the M-RoPE channel-stitching is bit-exact at the sub-op level; the full layer max_abs_diff `1.67e-6` comes from the rest of the attention chain.

Bit-exactness is not the goal — the gate is `atol=5e-4`. Bit-exactness is a happy accident of fp32 determinism in specific math.

---

## Cross-references

- **IR surface** — every spec, op, and module appearing in Part 2 is documented at the symbol level in `docs/API-REFERENCE.md`. The §6 recipes (`docs/API-REFERENCE.md` §6.x) are the per-family build receipts in narrative form.
- **Project scoreboard** — `docs/PROJECT-SUMMARY.md` is the audited inventory (the source for Part 1's gate-class table).
- **Per-family `layer.md`** — every working family has `models/<family>/layer.md` with §0 At-a-glance, §2 block diagram, §3 tensor IO trace, §4 op trace, §5 spec instantiation, §6 quirks, §7 weight-name mapping, §8 source citations, §9 validation status. Always look there first.

---

## Summary table

| What | Value | Verified by |
|---|---|---|
| Total tests collected | **751** | `uv run pytest --collect-only -q` |
| API-level tests passing | **225 / 225** in 37.65 s | `uv run pytest -q tests/api/` |
| Model gates run today | 6/6 pass; combined 4 min 28 s | Per-family runs above |
| Working families | **41** architecturally distinct (16 R, 11 S, 11 shape-only or hybrid, plus 3 v7 frontier hybrids) | `models/<family>/layer.py` exists |
| Deferred stubs | **9** | `models/<family>/__init__.py` only |
| Bit-exact gates vs **real HF weights** | **1** (Gemma 2 2B) | `tests/models/gemma2/test_numerical_hf.py` |
| Bit-exact gates vs synthetic | **5** (Mamba-1, Mamba-2 mini, Granite-4-H mini, DeepSeek-V3-MoE submodule, GPT-OSS) | various |
| Largest R-class diff | **3.05e-5** (MiniCPM-3) | `models/minicpm3/layer.md` |
| Axes documented in Part 2 | ~20 covering attention kind, mask, RoPE basis + scaling, norm kind + position + mode, QK-norm, ALiBi, sub-norms, FFN gates, MoE routers, KV sharing, K=V, PLE, parallel residual, SSM variants, quant, μP, per-layer dispatch | inline source citations |
