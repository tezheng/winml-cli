# Audit A — Research-vs-IR Consistency

**Date:** 2026-06-10
**Auditor lens:** skeptical technical review. Hunts for drift between (a) the research-document promises about what `llm-layers` needs to express, and (b) what the working IR (`api/specs.py`, `api/types.py`, `api/ops.py`, building blocks) actually carries.
**Working tree:** `C:\Users\zhengte\external\llm-layers`, as of this writing.
**Sources cross-referenced:**
- `research/02-layer-sources.v3.md` §3 (38-axis catalog)
- `research/05-kvcache-attention.v3.md` §7 (KV/attention/RoPE parameter spaces)
- `research/04-quantization.v3.md` §10 (21 quant axes; 48 schemes documented in §3-§9)
- `research/01-model-census.v3.md` §5 (33 per-family evolution arcs)
- `research/03-ihv-opsets.v2.md` §4 (cross-runtime consensus op floor — NOT refreshed to v3)
- `api/specs.py` (`C:\Users\zhengte\external\llm-layers\api\specs.py`, 733 lines, 21 spec dataclasses post-v7)
- `api/types.py` (`...\api\types.py`, 211 lines, 17 enums)
- `api/ops.py` (`...\api\ops.py`, 866 lines, 26 public ops)
- `api/{attention,rope,norm,ssm,kvcache,quant,feedforward,block,embedding}.py`

---

## 1. A1 — Research-axis to IR-field mapping (38 rows)

For each axis in `research/02-layer-sources.v3.md §3.1-§3.38`, I traced the IR field(s) that implement it. Status enum: **MAPPED** (field exists and is read by `api/*.py`), **DEFERRED** (spec field present, forward raises `NotImplementedError` or is shape-only), **COLLAPSED** (semantically expressed via a different mechanism than the research-doc shape would suggest), **MISSING** (no IR coverage at all), **OUT-OF-SCOPE-BY-DESIGN** (research §3 itself states the axis is not per-layer math).

| # | Research axis (§3.x) | IR field(s) | File:line | Status |
|---|---|---|---|---|
| 3.1  | Norm placement {pre, post, sandwich, parallel} | `DecoderBlockSpec.attn_norm_position` + `ffn_norm_position` + `block_layout`; enums `NormPosition.{PRE,POST,PRE_AND_POST}` and `BlockLayout.{SEQUENTIAL,PARALLEL}` | `api/specs.py:701,702,733`; `api/types.py:110-131` | **MAPPED** |
| 3.2  | Norm formula {LayerNorm γβ, LayerNorm γ-no-β, RMSNorm, RMSNorm(1+w), ScaleNorm, DeepNorm} | `NormSpec.kind` (RMS/LAYER) + `weight_mode` (STANDARD_W/ONE_PLUS_W) + `has_bias` | `api/specs.py:14-26`; `api/types.py:100-107` | **MAPPED** — ScaleNorm / DeepNorm not enumerated (no released model uses them per research §3.2; deferral acceptable) |
| 3.3  | QK-Norm {none, per-head-dh, full-flat, post-rope, fixed-scale} | `AttentionSpec.qk_norm` + `qk_norm_phase` + `qk_norm_shape` + `qk_norm_fixed_scale` | `api/specs.py:189-191,197` | **MAPPED** — fixed-scale (Gemma 4 0.9916/1.0228) is wired but every shipping model factory passes `None` per the API-minimality audit §1.1a |
| 3.4  | Attention scale (5 variants incl. Gemma-4 1.0 + Granite muP) | `AttentionSpec.attn_scale` (Optional[float]) | `api/specs.py:185` | **MAPPED** |
| 3.5  | Position encoding {NEOX, GPT-J, partial, NoPE, M-RoPE, 2D-RoPE, 3D-RoPE, LongRoPE, YaRN, NTK, ALiBi} | `RoPESpec.basis` + `scaling` + `llama3_extra`/`yarn_extra`/`longrope_extra` + `mrope_section` + `AttentionSpec.alibi` | `api/specs.py:130-168,252` | **MAPPED** — NTK-static / NTK-dynamic / 2D-RoPE (vision tower) absent (research §3.5 lists them; no `<8B` model in scope uses them) |
| 3.6  | NoPE alternation (SmolLM3, Llama-4 iRoPE) | **COLLAPSED**: implemented as `spec.rope = None` per-layer instead of `apply_rope_per_layer` boolean. `Attention.__init__` sets `self.rope = None` when `spec.rope is None`; the forward skips `rope_apply` entirely. | `api/attention.py:290-300` ("B1: SmolLM3 NoPE per-layer — spec.rope may be None on layers where the model disables RoPE") | **COLLAPSED — accept** |
| 3.7  | QKV layout {three separate, fused, MLA, K=V unified, MQA, unified-prefill-fused} | `AttentionSpec.qkv_layout` (SPLIT/FUSED/MLA_LATENT) + `attention_k_eq_v` + MLA dims | `api/specs.py:177,196,221-225`; `api/types.py:61-65` | **MAPPED** |
| 3.8  | Attention biases per-projection | `AttentionSpec.{q_bias,k_bias,v_bias,o_bias}` | `api/specs.py:180-183` | **MAPPED** |
| 3.9  | Softcap (Gemma 2 attn + final; Gemma 4 final only) | `AttentionSpec.logit_softcap` (attn) + `ops.lm_head(softcap=...)` (final) | `api/specs.py:186`; `api/ops.py:71-82` | **MAPPED** — note: `DecoderBlockSpec.final_logit_softcap` documented as a dead set-but-never-read field by `docs/issues/API-minimality-coverage-audit.md` §1.1a; only `ops.lm_head` consumes the final softcap at model-assembly level. |
| 3.10 | Sliding window (none / global / per-layer-alternating; dual-θ local/global) | `AttentionSpec.sliding_window` per-layer (factory-set) + per-layer RoPESpec | `api/specs.py:187` | **MAPPED** |
| 3.11 | GQA / MQA / MHA = `(H, Hk)` | `AttentionSpec.{n_q_heads, n_kv_heads}` | `api/specs.py:173-174` | **MAPPED** |
| 3.12 | MLP layout {sep, fused, plain GELU, double-wide} + activation | `FFNSpec.gate_kind` (`GateKind.SWIGLU/GEGLU/GELU_ONLY/RELU2_ONLY`) + `fused_gate_up` + `activation` | `api/specs.py:312-316`; `api/types.py:134-152` | **MAPPED** — `use_double_wide_mlp` (Gemma 4 E2B) NOT carried as a discriminator; expressible only via `intermediate_size` doubling, which is a silent encoding |
| 3.13 | MLP biases | `FFNSpec.{gate_bias, up_bias, down_bias}` | `api/specs.py:317-319` | **MAPPED** |
| 3.14 | Channel mixer kind {dense MLP, sparse MoE, hybrid} + router / shared-expert / aux-loss-free correction / group-routing / routed-scaling | `DecoderBlockSpec.channel_mixer: Union[FFNSpec, MoESpec]` + `MoESpec` (`router_kind`, `router_norm`, `group_routing`, `routed_scaling_factor`, `n_shared_experts`) | `api/specs.py:498-557,710` | **MAPPED** |
| 3.15 | Per-layer dense/sparse decision | Implicit via per-layer factory build of `DecoderBlockSpec` (factories own the `layer_idx` -> mixer mapping) | `models/<family>/config.py` (e.g. `deepseek_v3_moe`, `qwen3_moe`) | **MAPPED-by-convention** — there is no IR-level field; each model factory owns its `first_k_dense_replace` / `mlp_only_layers` indexing. This is by-design (different families use different idioms) but worth flagging as "IR delegates this to model assembly" rather than carrying a discriminated field. |
| 3.16 | Token-mixer kind (12 values incl. mamba1/2/3, RWKV, RG-LRU, linear-attn, Lightning, DSA, CSA+HCA, Hymba parallel-mamba) | `TokenMixerKind.{ATTENTION, SSM_MAMBA1, SSM_MAMBA2, GATED_DELTANET}` + `DecoderBlockSpec.token_mixer: Union[AttentionSpec, SSMSpec, SSDSpec, GatedDeltaNetSpec]` | `api/types.py:22-43`; `api/specs.py:709` | **MAPPED-partial** — Mamba-3 complex/MIMO, RG-LRU (RecurrentGemma), WKV (RWKV), Hymba parallel-mamba, Lightning Attention, Mamba-1, MiniMax MSA all called out by research §3.16 but **not** enumerated as discriminated TokenMixerKind values. Mamba-1 IS partially live (`SSMKind.MAMBA1`); the others are DEFERRED via the existing stubs (`models/mamba3/`, `models/recurrent_gemma/`, `models/rwkv7/`, `models/minimax_text_01/`). |
| 3.17 | Residual scaling (Granite muP, Gemma 4 PLE 1/sqrt(2)) | `DecoderBlockSpec.residual_scale` + `PLESpec.residual_scale` | `api/specs.py:715,695` | **MAPPED** |
| 3.18 | Embedding scaling (Gemma sqrt(D); Granite 12.0) | OUT-OF-IR-PER-LAYER (handled at model assembly via `ops.embed(scale=...)`) | `api/ops.py:60-68` | **MAPPED** at op-level; not as a block field. Research §3 also flags this as not per-layer. |
| 3.19 | Logits scaling (Granite /8, Cohere *logit_scale, Gemma 2 final softcap, Gemma 4 30.0) | OUT-OF-IR-PER-LAYER (handled by `ops.lm_head(scale=..., softcap=...)`) | `api/ops.py:71-82` | **MAPPED** at op-level. `DecoderBlockSpec.logits_scale` exists per API-minimality audit but is set-only / never read (dead). |
| 3.20 | Residual dropout (Phi-3, StarCoder 2) | **MISSING** — no `resid_pdrop` field anywhere in `api/specs.py`; no `dropout` op | `api/` (grep returns no matches) | **MISSING-deliberate** — the research §3.20 itself says "set to 0 in inference but present in code path", so eval-only IR can defer. **Flag**: a model factory loading a Phi-3 train-time config will silently drop the field. |
| 3.21 | Cache compression (MLA decompressed vs compressed) | `KVCacheSpec.layout` (CacheLayout) + MLA dims in AttentionSpec | `api/specs.py:343-348`; `api/types.py:167-172` | **MAPPED-by-shape** — the IR consumes the "decompressed" HF-simple-correct form; the "compressed + absorb-kv_b" production form is not modeled (research §3.21 acknowledges this is a vLLM/llama.cpp-only optimization, not a per-layer math choice). |
| 3.22 | Sub-norms inside attn/MLP (BitNet attn_sub_norm + ffn_sub_norm; Mamba-2 group-norm-gated; Gemma 4 vision-encoder post-norm) | `AttentionSpec.attn_sub_norm` + `FFNSpec.ffn_sub_norm` (v5-phase2 V3) | `api/specs.py:260,327` | **MAPPED** for BitNet flavors. Mamba-2's gated RMSNorm lives inside `Mamba2Mixer` as a separate module (`api/ssm.py`), so it is not a `NormSpec` but is mathematically present. |
| 3.23 | Per-layer width (OpenELM) | `LayerScaleSpec` (dataclass with per-layer arrays) | `api/specs.py:673-677` | **DEFERRED-DEAD** — the dataclass exists but the API-minimality audit (`docs/issues/API-minimality-coverage-audit.md` §1.1a) confirms it is referenced only by `tests/api/test_specs.py`; no `models/openelm/` exists. |
| 3.24 | Parameter sharing (Zamba2 shared attention block; MobileLLM block-wise weight sharing) | **MISSING** — no IR construct for a "shared layer pool"; `nn.ModuleList` is the only assembly idiom. | n/a | **MISSING** — Zamba2 is out of corpus (no `models/zamba2/`); MobileLLM same. **Flag** for follow-up if either family is admitted. |
| 3.25 | Parallel residual (Falcon-7B, Cohere) | `BlockLayout.PARALLEL` + `DecoderBlock` PARALLEL forward branch | `api/types.py:116-131`; `api/block.py:255-313` | **MAPPED** — Falcon-H1's `num_ln_in_parallel_attn=2` (third variant) is not a separate enum value, but the existing PARALLEL with `pre_ffn_norm=None` covers it. |
| 3.26 | Embedding tying as standalone axis | OUT-OF-IR-PER-LAYER (model-assembly concern) | n/a | **OUT-OF-SCOPE-BY-DESIGN** — research §3.26 frames it as parameter-count-affecting, not a per-layer math change. |
| 3.27 | Continuous-batching shape transformation `[B, S, D]` → `[T, D]` | **MISSING** — no IR construct for vLLM-style flat token layout. | n/a | **MISSING-deliberate** — research §3.27 acknowledges this is a vLLM runtime axis. The IR is HF-shape only by design (per `docs/PROJECT-SUMMARY.md` HF-shape commitment). **Flag** but not an audit failure. |
| 3.28 | TP sharding affecting layer composition | **MISSING** — no IR construct for `QKVParallelLinear` / `RowParallelLinear` | n/a | **MISSING-deliberate**, same posture as 3.27. Research itself says "None of this is in v1." Acceptable for an HF-shape IR, but explicitly flagged in the audit as "TP is an out-of-scope axis." |
| 3.29 | FlashAttention backend selection (eager / sdpa / FA2 / FA3 / flex) | **MISSING** — `ops.sdpa` dispatches on `attn_bias`/`sinks`/`logit_softcap` presence, not on backend identity | `api/ops.py:340-438` | **MAPPED-by-feature-set** — the IR pays for backend differences by hard-coding the right fallback; it does not expose a backend selector. Research §3.29 frames this as a runtime axis, not a per-layer axis, so this is a conscious collapse. **Flag** as a conscious collapse, not drift. |
| 3.30 | Sink tokens / sink slots (GPT-OSS trained) | `AttentionSpec.n_sink_tokens` + `ops.sdpa(sinks=...)` | `api/specs.py:293`; `api/ops.py:395-412` | **MAPPED** |
| 3.31 | Token-mixer recurrent state cache (Mamba conv+ssm state; RG-LRU; WKV; Mamba-3 quad-state) | `KVCacheSpec.layout = SSM_STATE` + dedicated `Mamba1Mixer` / `Mamba2Mixer` / `GatedDeltaNetMixer` modules carrying their own state | `api/types.py:172`; `api/ssm.py` | **MAPPED-partial** — Mamba-1 (B7), Mamba-2 SSD (B7), Gated DeltaNet (v7 P3) all live. RG-LRU, WKV, Mamba-3 complex/MIMO state are DEFERRED (`models/recurrent_gemma/`, `models/rwkv7/`, `models/mamba3/` stubs). |
| 3.32 | μP-style multipliers as standalone axis (Granite 4-scalar set) | Threaded via `attn_scale`, `residual_scale`, `ops.embed(scale=...)`, `ops.lm_head(scale=...)` — no `MuPSpec` aggregator | various | **MAPPED-by-thread** — no dedicated discriminated dataclass. Research §3.32 calls for it as "load-bearing μP transfer hyperparameters." **Acceptable** but a `MuPSpec` would deduplicate the four-multiplier knowledge across model factories. |
| 3.33 | Cross-layer KV sharing (Gemma 4 same-block; Apple AFM block-role; CLA adjacent-pair) | `AttentionSpec.kv_source_layer_offset` (CLA) + `SharedLayerKVCache` (Gemma 4) | `api/specs.py:278`; `api/kvcache.py` (per minimality audit) | **MAPPED-partial** — CLA via `kv_source_layer_offset`; Gemma 4 via a separate `SharedLayerKVCache`. Apple AFM block-role split is NOT IR-modeled (no `share_block_split: float` field). **Flag** — research §3.33 explicitly enumerates three variants; IR covers two. |
| 3.34 | Per-layer embeddings (Gemma 4 E2B/E4B) | `PLESpec` + `PerLayerEmbedding` module + `DecoderBlockSpec.per_layer_embedding` + the PLE-at-end injection in `DecoderBlock.forward` | `api/specs.py:681-695,718`; `api/embedding.py:22-63`; `api/block.py:336-345` | **MAPPED** |
| 3.35 | Partial-rotary RoPE (Gemma 4 global 0.25; Phi-3 legacy 0.5; MLA qk_rope/qk_head) | `RoPESpec.partial_rotary_factor` + `partial_rotary_kind ∈ {"prefix", "proportional"}` | `api/specs.py:139,158` | **MAPPED** |
| 3.36 | Vision-adapter topology (6 discriminated variants) | OUT-OF-PER-LAYER (research §3.36 itself says "It does not extend the decoder per-layer math; it only changes how non-text modalities arrive at layer 0") | n/a | **OUT-OF-SCOPE-BY-DESIGN** |
| 3.37 | Block-bidirectional mask (DeepSeek-OCR Visual Causal Flow) | `MaskKind.BLOCK_BIDIRECTIONAL` + `AttentionSpec.block_bidirectional_mask: bool` | `api/types.py:85`; `api/specs.py:308` | **DEFERRED** — research §3.37 and the IR docstring agree the HF v5.10.2 deepseek_ocr2 implements standard causal; the IR carries the hook but does not exercise it numerically. |
| 3.38 | MXFP4-native training (GPT-OSS) | `QuantSpec.qdtype = ...` + `api/quant.py` mxfp4_quantize/dequantize | `api/quant.py:475-561` | **MAPPED-partial** — the MXFP4 round-trip is implemented; the `training_native` axis from `research/04 §10.3.2` is NOT a field on `QuantSpec`. The IR does not distinguish "MXFP4 PTQ" from "MXFP4-native training." **See A5 below.** |

**A1 verdict:** 38 axes audited.
- **MAPPED (full or partial-mapped acceptable):** 26
- **MAPPED-by-convention / MAPPED-by-thread / MAPPED-by-shape (no discriminated field):** 5 (3.15, 3.18, 3.19, 3.32, 3.21)
- **DEFERRED (spec hook present, forward shape-only or not exercised):** 4 (3.16 partial for Mamba-3/RG-LRU/WKV/Lightning/MSA; 3.23 LayerScaleSpec; 3.31 partial; 3.37)
- **COLLAPSED-acceptable (different mechanism than research-doc suggests but mathematically equivalent):** 1 (3.6 NoPE via `rope=None` instead of `apply_rope_per_layer` — equivalent; choice noted in `api/attention.py:290-300`)
- **OUT-OF-SCOPE-BY-DESIGN (research §3 itself says non-per-layer):** 4 (3.18, 3.26, 3.27, 3.36) — strictly speaking research §3 lists these as "axes" but acknowledges they're model-assembly or runtime concerns
- **MISSING-deliberate (research acknowledges optional/runtime/eval-irrelevant):** 3 (3.20 dropout; 3.28 TP; 3.29 backend select)
- **MISSING-flag (no IR coverage and research insists it's per-layer):** 1 (3.24 parameter sharing — Zamba2 / MobileLLM out of working corpus; would block admission)

So: **30 axes have IR coverage adequate for their corpus footprint, 4 are deferred via stubs, 4 are out-of-scope-by-design, 3 are deliberately-missing runtime concerns, and 1 (3.24) is the only "missing-flag" item where a planned-axis lacks IR coverage with no current covering family.**

The most consequential drift inside MAPPED is **3.6 NoPE alternation**: research/05 §7.2 lists `apply_rope_per_layer: list[bool] | None` as a NEW v3 axis (`research/05-kvcache-attention.v3.md:1019`); the IR implements it as `spec.rope = None` per-layer instead. The implementations are mathematically equivalent; the API surface diverges. The IR's choice is documented in `api/attention.py:290-300` citing `modeling_smollm3.py:211`. **Verdict: research-doc and IR diverge in field name but agree in semantics — DOCUMENT-LEVEL DRIFT, not behavioral drift. Recommended: add a footnote to `research/02 §3.6` and `research/05 §7.2` noting "IR implements this as `rope: Optional[RoPESpec]` per-layer rather than as a separate `apply_rope_per_layer` array."**

---

## 2. A2 — IR fields without research-catalog grounding

I traced each non-leaf spec field added in v5/v6/v7 to find any that exist in the IR but are NOT justified by `research/02 §3` or `research/05 §7`.

### 2.1 Ungrounded — exists in IR, NOT in `research/{02,05}` v3

| Spec field | Citation | Research justification? |
|---|---|---|
| `MoESpec.router_kind = "hash"` + `hash_vocab_size` + `hash_score_fn` | `api/specs.py:534-556` | **NO grounding in `research/02-layer-sources.v3.md §3.14`** (research §3.14 lists `{softmax, sigmoid}` only). The IR docstring cites `modeling_deepseek_v4.py:1050-1078` (`DeepseekV4HashRouter`), which is a primary source — but this is exactly the "Raschka-gallery driven addition" pattern flagged in `docs/issues/raschka-gallery-comparison.md` A.1. **The research catalog should add hash routing as a router-kind value in §3.14 before this lands.** |
| `AttentionKind.CSA_HCA` + `CSASpec` + `HCASpec` + `AttentionSpec.{csa, hca}` | `api/types.py:19`; `api/specs.py:242-243,596-648` | **PARTIAL grounding** — `research/02 §3.16` token-mixer-kind enumeration includes "CSA+HCA hybrid (DeepSeek-V4)" (`research/02-layer-sources.v3.md:177`) and `research/05 §7.2` carries `CSASpec`/`HCASpec` dataclasses (`research/05-kvcache-attention.v3.md:1055-1060`). **The IR's two-spec split (CSA + HCA separately, both optional on the same `AttentionSpec`) matches research/05's split. PASS.** |
| `GatedDeltaNetSpec` + `TokenMixerKind.GATED_DELTANET` | `api/specs.py:451-488`; `api/types.py:37-43` | **GROUNDED** — `research/02 §5.36` is a dedicated section on Qwen3-Next Gated DeltaNet; `research/02 §3.16` enumerates "linear-attention (DeltaNet/GLA/Gated-DeltaNet)" (line 177); research/05 §7.2 lists `kind = ... deltanet ...`. **PASS** — though the discriminator name varies (`GATED_DELTANET` in IR; `deltanet` in research/05 lowercase). |
| `MoESpec.routing_in_latent` + `latent_dim` + `latent_bias` | `api/specs.py:558-576` | **NO grounding** — Nemotron-H Latent MoE is mentioned in `docs/issues/raschka-gallery-comparison.md` A.5 as a Raschka-gallery axis (verified against `modeling_nemotron_h.py:672-745`) but **`research/02-layer-sources.v3.md §3.14` does not enumerate "latent-space MoE"** as a router-kind or as an FFN-around-experts axis. The IR adds it citing primary HF source code, which is in the spirit of `feedback_verify_source_not_blogs.md` — but the research catalog was not updated to reflect it. **Same pattern as `router_kind="hash"`: lands in IR ahead of research. Research §3.14 should add "latent-MoE wrapper" as an extension.** |
| `MoESpec.expert_kind = "gpt_oss_clamped_swiglu"` + `expert_swiglu_alpha` + `expert_clamp_limit` + `expert_bias` | `api/specs.py:578-592` | **PARTIAL grounding** — `research/02 §5.40` discusses GPT-OSS biased experts (research/02:1407-1419 shows the `gate_up_proj_bias` parameter); however **`research/02-layer-sources.v3.md §3.14` does not enumerate "clamped SwiGLU" as an MLP-layout variant in §3.12** (the enumeration is `{swiglu, geglu, gelu, relu2, double-wide}`), and the `alpha=1.702` / `clamp_limit=7.0` constants are not in any research v3 doc. The IR docstring cites `modeling_gpt_oss.py:73-119` as source. **Justifiable but undisclosed in the axis catalog — research §3.12 should be extended.** |
| `RoPESpec.partial_rotary_kind ∈ {"prefix", "proportional"}` | `api/specs.py:158` | **GROUNDED** — research/02 §3.35 says "The two systems differ in interpretation: MLA: the rotated channels are a separate K/Q vector... Gemma 4 p-RoPE: the rotated channels are the first prefix of each head-dim" (lines 268-271). The IR's two-value enum captures exactly this discriminator. **PASS.** |
| `AttentionSpec.attention_k_eq_v` + `qk_norm_fixed_scale` + `global_head_dim` (Gemma 4) | `api/specs.py:196,197` | **GROUNDED** — research/02 §3.7 (K=V), §3.3 (fixed-scale), §3.10 (dual-θ + sliding) all carry these axes. PASS. |
| `AttentionSpec.kv_source_layer_offset` (CLA) | `api/specs.py:278` | **GROUNDED** — research/02 §3.33 lists CLA as one of the three cross-layer-KV-sharing variants. PASS. |
| `MoESpec.router_kind = "topk_then_softmax_with_bias"` | `api/specs.py:541` | **PARTIAL grounding** — DeepSeek-V3 sigmoid-plus-bias is in research §3.14, but the IR's discriminated enum value names diverge from the research-doc nomenclature. Minor naming drift. |
| `AttentionSpec.attn_sub_norm` + `FFNSpec.ffn_sub_norm` (BitNet) | `api/specs.py:260,327` | **GROUNDED** — research/02 §3.22 explicitly calls out BitNet `attn_sub_norm` (line 196) and `ffn_sub_norm` (line 197). PASS. |
| `AttentionSpec.alibi: AliBiSpec` (MPT/Falcon) | `api/specs.py:252`; `api/specs.py:98-126` | **GROUNDED** — research/02 §3.5 enumerates "ALiBi (Baichuan-1, MPT — historical)." PASS. |

### 2.2 A2 verdict — count of "ungrounded" IR fields

**Strictly Raschka-only-or-source-direct, NOT in research/02 v3:** **3** ungrounded fields (a "field" here = one spec member; counts as one ungrounded item if its conceptual axis is missing from research §3):

1. `MoESpec.router_kind="hash"` + `hash_vocab_size` + `hash_score_fn` (3 fields, 1 axis)
2. `MoESpec.routing_in_latent` + `latent_dim` + `latent_bias` (3 fields, 1 axis)
3. `MoESpec.expert_kind="gpt_oss_clamped_swiglu"` + `expert_swiglu_alpha` + `expert_clamp_limit` + `expert_bias` (4 fields, 1 axis — though `expert_bias` is partially grounded in §5.40 source quote)

All three are documented in `docs/issues/raschka-gallery-comparison.md` as Raschka-gallery additions, all are verified against in-tree `modeling_*.py` source code per `feedback_verify_source_not_blogs.md`, and all carry primary-source citations in the IR docstrings. **None violates the "blog-not-source" guard — each cites HF source line ranges.** What they do violate is the research-doc → IR pipeline order: `research/02 §3` should have been refreshed to enumerate these axes BEFORE the IR fields landed.

**Verdict:** PARTIAL PASS. The IR additions are source-verified and not blog-driven, so the `feedback_verify_source_not_blogs.md` guard is respected. The drift is **documentation-only**: research/02 §3.12 (channel mixer / MLP layout) and §3.14 (router family + expert topology) need an addendum to enumerate hash-routing, latent-MoE wrapping, and clamped-SwiGLU expert kind.

---

## 3. A3 — Census evolution arcs vs IR families

`research/01-model-census.v3.md §5` lists 33 arcs (§5.1-§5.33) post-v3.1 (ChatGLM/GLM, Yi, Hunyuan added). The `models/` directory contains **49 family directories** (51 entries minus `__init__.py` and `__pycache__`).

### 3.1 Arcs in research §5 → models/ directory presence

I cross-walked each arc to the corresponding `models/<family>/` dir(s):

| § | Arc family | `models/<family>/` |
|---|---|---|
| 5.1 | Llama | `llama3/`, `llama4_scout/`, `tinyllama/` |
| 5.2 | Qwen | `qwen3/`, `qwen3_moe/`, `qwen3_next/`, `qwen2_5_vl/` |
| 5.3 | Phi | `phi3_mini/`, `phi3_small/`, `phi4_mini/`, `phi4_mini_flash/` |
| 5.4 | Gemma | `gemma2/`, `gemma3/`, `gemma4/` |
| 5.5 | DeepSeek | `deepseek_v2_lite/`, `deepseek_v3_lite/`, `deepseek_v3_moe/`, `deepseek_v32/`, `deepseek_v4/`, `deepseek_ocr2/` |
| 5.6 | Mistral | `mistral/`, `mixtral/`, `ministral/`, `voxtral/` (Voxtral is the audio sibling, present) |
| 5.7 | OLMo | `olmo2/`, `olmoe/` |
| 5.8 | BitNet → Falcon-Edge | `bitnet/` |
| 5.9 | Falcon | `falcon7b/`, `falcon_h1/` |
| 5.10 | SmolLM | `smollm3/` |
| 5.11 | Granite | `granite/`, `granite4_h/` |
| 5.12 | MiniCPM | `minicpm3/` |
| 5.13 | OpenELM | **NO** `models/openelm/` — arc with no model |
| 5.14 | Hybrid SSM | covers many of the above; `jamba/`, `hymba/` |
| 5.15 | StarCoder | **NO** `models/starcoder*/` — arc with no model |
| 5.16 | RWKV | `rwkv7/` (stub) |
| 5.17 | xLSTM | **NO** `models/xlstm/` |
| 5.18 | Mamba | `mamba1/`, `mamba2/`, `mamba3/` |
| 5.19 | Apple AFM | **NO** `models/apple_afm/` |
| 5.20 | OCR-LLM | `deepseek_ocr2/`, `got_ocr2/` |
| 5.21 | VLM text-tower | covered by `qwen2_5_vl/` etc. |
| 5.22 | Audio-LM | `moshi/`, `voxtral/` |
| 5.23 | EXAONE | **NO** |
| 5.24 | Hunyuan (recent) | covered by `hunyuan_large/` partially |
| 5.25 | Liquid AI LFM | **NO** |
| 5.26 | NVIDIA Nemotron | `nemotron3/` (stub) |
| 5.27 | Zhipu / Z.AI GLM | `glm_moe_dsa/` (covers GLM-5 line per the file present) |
| 5.28 | Cohere | **NO** `models/cohere/` |
| 5.29 | MiniMax | `minimax_text_01/` + `minimax_m2/` |
| 5.30 | MobileLLM | **NO** `models/mobilellm/` |
| 5.31 | ChatGLM/GLM | `glm_moe_dsa/` covers part of this (GLM-5 sparse); ChatGLM2/3 / GLM-4 not separately landed |
| 5.32 | Yi | **NO** `models/yi/` |
| 5.33 | Hunyuan-Large | `hunyuan_large/` |

### 3.2 Inventory delta

**Arcs without `models/<family>/` representation** (research arc exists, no IR family — these are census-only gaps):

1. **OpenELM** (§5.13) — per-layer width dataclass `LayerScaleSpec` exists in IR (dead per minimality audit), no model directory
2. **StarCoder 1/2** (§5.15) — research arc; no IR family; the canonical "biases-on-everything decoder" reference is missing
3. **xLSTM** (§5.17) — research arc; no IR family
4. **Apple AFM** (§5.19) — research arc; no IR family; cross-block KV-sharing variant (3.33 sub-axis) blocked
5. **EXAONE** (§5.23) — research arc; no IR family; standard Llama-shape so low-priority
6. **Liquid AI LFM** (§5.25) — research arc; no IR family
7. **Cohere** (§5.28) — research arc; no IR family. **NOTABLE**: parallel-residual axis `BlockLayout.PARALLEL` was added partly with Cohere in mind, but no `models/cohere/` exists — it is exercised only by `falcon7b/` (per `api/specs.py:733` docstring referencing Falcon).
8. **MobileLLM** (§5.30) — research arc; no IR family
9. **ChatGLM2/3, GLM-4-9B** (§5.31 early generations) — only GLM-5-line (`glm_moe_dsa/`) is represented; the partial-rotary ChatGLM heritage is documented in research but no `models/chatglm*/` family
10. **Yi** (§5.32) — research arc; no IR family; standard Llama shape, low-priority

**`models/<family>/` directories without a clean arc home in research §5:**

1. **`models/glm_moe_dsa/`** — fits §5.27 / §5.31 (GLM family) but no dedicated arc for the DSA-derivative variant
2. **`models/minimax_m2/`** — §5.29 (MiniMax) covers Text-01 and M2/M3 in narrative but `models/minimax_m2/` is not separately rowed as a model in census Table-of-49
3. **`models/voxtral/`** — §5.6 (Mistral) and §5.22 (Audio-LM) overlap; not surveyed which "owns" Voxtral
4. **`models/got_ocr2/`** — §5.20 (OCR-LLM family) is the arc; OK
5. **`models/moshi/`** — §5.22 owns; OK
6. **`models/recurrent_gemma/`** — RG-LRU is in §5.14 (Hybrid SSM) narrative; OK
7. **`models/mpt/`** — MPT is mentioned in §3.5 (ALiBi historical) but **has no §5 arc** at all. The only ALiBi reference family in the working corpus.

### 3.3 A3 verdict

**10 census arcs have no corresponding `models/<family>/` (missing-models delta = 10).** Most are deferred-by-priority (Yi, EXAONE are standard Llama-shape; xLSTM, Liquid LFM are below-priority recurrents). The notable absences are:

- **Cohere** — directly relevant for `BlockLayout.PARALLEL` validation; the IR's parallel-residual support is exercised only against Falcon-7B
- **Apple AFM** — its cross-block KV sharing is the only `research/02 §3.33` variant the IR doesn't model

**1 model directory has no corresponding arc:** `models/mpt/` lacks an evolution-arc home (only referenced as the ALiBi reference in §3.5).

**3 model directories are arc-orphan-but-narrative-covered:** `glm_moe_dsa/`, `minimax_m2/`, `voxtral/` need explicit row entries in census §3 model table (not just narrative mentions).

**Delta count: 10 arcs missing models; 1 model missing arc; 3 models missing census rows.**

---

## 4. A4 — IHV op-set survey consistency (research/03 v2)

`research/03-ihv-opsets.v2.md` was **NOT refreshed to v3**. Its op-floor claim is in §4.2:

> "The nine universal ops (`rms_norm`, `linear`, `rope`, `sdpa`, `residual_add`, `swiglu_ffn`, `silu`, `mul`, `lm_head`) cover ≥ 14/18 runtimes."
> — `research/03-ihv-opsets.v2.md:739`

**The audit task description says "16-op floor from research/03 §3."** This is a misquote of research/03. Research/03 v2 §4.2 claims a **9-op floor**; the "16-op floor" terminology comes from `docs/issues/API-REFERENCE-audit.md:48` ("16-op floor (7 stub ops added)") and refers to the IR's count milestone at the M1-fix-pass commit (`5aa8374`), NOT to the research-document claim.

### 4.1 Actual `api/ops.py` public-op count

Direct `grep '^def [a-z]' api/ops.py`:

```
silu, add, mul, linear, rms_norm, embed, lm_head, rope_apply,
rope_apply_mrope, rope_apply_partial, gelu_pytorch_tanh, gelu_exact,
relu2, build_alibi_slopes, apply_alibi, sdpa, layer_norm, softmax,
top_k, gather, scatter, conv1d, selective_scan_mamba1, selective_scan,
l2norm, gated_delta_step
```

→ **26 public ops** at HEAD (was 19 per `docs/issues/API-REFERENCE-audit.md` and 18 per `docs/API-REFERENCE.md`).

### 4.2 Drift quantification

- `research/03 §4.2` 9-op floor → currently 26 ops in `api/ops.py`.
- Net additions since v2 of research/03: `rope_apply_mrope` (B8 M-RoPE), `rope_apply_partial` (B0.5 Gemma 4 / Phi-3), `gelu_pytorch_tanh` (Gemma activation), `gelu_exact` (MPT/Falcon-7B ungated GELU), `relu2` (BitNet / Nemotron), `build_alibi_slopes` (v5-phase2 V1), `apply_alibi` (v5-phase2 V1), `layer_norm` (v5/v6 LayerNorm path for MPT/Falcon-7B), `softmax` (MoE), `top_k` (MoE), `gather`+`scatter` (MoE), `conv1d` (Mamba), `selective_scan_mamba1` (v6 B1), `selective_scan` (B7), `l2norm` (v7 P3 Gated DeltaNet), `gated_delta_step` (v7 P3).

### 4.3 A4 verdict

**Op-count drift: 9 (research/03 v2 floor) → 26 (current IR) = +17 ops.**

This is the expected drift after B7/v6/v7 SSM, Lightning-Indexer (deferred), ALiBi, and Gated DeltaNet rollouts. **Research/03 has not been refreshed.** The audit's quoted "16-op floor" appears to be a conflation with the IR's M1-fix-pass commit milestone, NOT with the research-doc claim. **research/03 v2 → v3 refresh is overdue** — every era beyond Era-5 (and the per-runtime op-set tables in §2) is unchanged since pre-rollout, despite the IR now supporting Mamba ssm/ssd, ALiBi, GatedDeltaNet, and sink-tokens that should ripple through the runtime support matrix.

Recommendation: refresh `research/03-ihv-opsets.v3.md` to (a) acknowledge the 26-op surface, (b) re-survey per-runtime support for `conv1d`, `selective_scan*`, `gated_delta_step`, `apply_alibi`, `rope_apply_partial`, `rope_apply_mrope`, and (c) decide whether to retract the "9-op floor" framing or extend it to a "9 + per-mixer-trait" pattern.

**Verdict: research/03 is stale; IR ships +17 ops since the v2 floor; needs v3 refresh.**

---

## 5. A5 — Quant scheme inventory grounding

`research/04-quantization.v3.md §3-§9` documents the quant schemes (§3.1-§3.28 weight-only INT/FP family + Family B-G). The audit asks "48 schemes" but `research/04` actually enumerates closer to 30 schemes by section across §3 (28 sub-sections plus Family B-G containing ~10 more). Either way, `api/quant.py` ships:

| Scheme in `api/quant.py` | Research §x grounding | Verdict |
|---|---|---|
| AWQ W4A16 (`awq_quantize` / `awq_dequantize` / AWQ_ORDER nibble permutation) | `research/04 §3.2` | **GROUNDED** |
| GGUF Q4_K_M (`gguf_q4_k_quantize` / `gguf_q4_k_dequantize`, super-block QK_K=256, K_SCALE_SIZE=12) | `research/04 §5.1` k-quant super-block structure | **GROUNDED** |
| FP8 E4M3 W8A8 (`fp8_e4m3_quantize_per_tensor` / `fp8_e4m3_quantize_per_token` / `fp8_e4m3_matmul`) | `research/04 §4.6` FP8 PTQ + `§3.28` DeepSeek-V3 native FP8 | **GROUNDED** |
| MXFP4 (`mxfp4_quantize` / `mxfp4_dequantize`, E2M1 + UE8M0 shared exponent, block 32) | `research/04 §3.21` MXFP4 PTQ + `§3.22` MXFP4-native training | **GROUNDED** |
| LiteRT mobile-INT4 W4A8 (`litert_quantize_weight` / `litert_quantize_activation_per_tensor`) | `research/04 §9.5` Gemma 4 mobile INT4 QAT | **GROUNDED** |
| BitNet b1.58 ternary (`ternary_quantize` / `ternary_dequantize`, base-3 LE packing) | `research/04 §9.1` BitNet b1.58 + `§9.2` Falcon-Edge retrainable | **GROUNDED** |
| `iq2_m_dequantize` / `aqlm_dequantize` (M3 stubs raising NotImplementedError) | `research/04 §3.11` AQLM + `§5.5` I-quants IQ-family | **GROUNDED-deferred** |

**A5 verdict:** All 6 shipping quant schemes (and 2 documented M3 stubs) are source-grounded in `research/04 §3-§9`. **No drift.** PASS.

The wider corpus (research/04 enumerates ~30 weight-only schemes across families A-G; the IR ships 6) — that gap is intentional, called out by `research/04 §10.4`'s coverage table where 23+ schemes are documented but not shipped. The minimal-shipping subset (AWQ representing weight-only-int, GGUF representing k-quant, FP8 representing W8A8 server, MXFP4 representing MX, LiteRT representing mobile QAT, BitNet representing sub-2-bit) covers the six axes of `research/04 §11` ("if you support six, support these six").

---

## 6. OVERALL VERDICT

| Check | Verdict |
|---|---|
| A1 — research axes have IR home | PARTIAL PASS (30/38 fully mapped or acceptably collapsed; 4 deferred via stubs; 4 out-of-scope-by-design; 1 missing-flag for Zamba2/MobileLLM) |
| A2 — IR fields have research home | PARTIAL PASS (3 axes ungrounded in `research/02 §3`: hash routing, latent MoE, clamped-SwiGLU expert. All three are source-verified against `modeling_*.py`, all three are flagged in `docs/issues/raschka-gallery-comparison.md`. **Documentation drift, not source-of-truth violation.**) |
| A3 — census arcs vs models/ | PARTIAL PASS (10 census arcs without models; 1 model orphan (mpt); 3 models lacking census rows) |
| A4 — op-set drift | **FAIL** — research/03 stale at v2; IR ships +17 ops since v2 floor; needs v3 refresh |
| A5 — quant grounding | **PASS** — all 6 shipped schemes source-grounded |

**Final score: PARTIAL.**

The IR is consistently more permissive (adds more) than research, never less. There are no "research says X, IR breaks X" instances — only "IR has X with a primary-source citation but research §3 catalog was not updated to enumerate X."

The drift is **documentation-asymmetric**: research/02 §3 (38-axis catalog) and research/03 (IHV op-set) lag behind the IR, while research/04 (quant) and research/05 (KV/attention parameter space) are roughly current.

---

## 7. Top 5 must-fix items

1. **Refresh `research/03-ihv-opsets.v3.md`** — survey the 26-op surface (was 9 in v2) against the 18 runtimes; re-rank per-runtime support for the SSM ops (`conv1d`, `selective_scan*`), Gated DeltaNet (`gated_delta_step`, `l2norm`), ALiBi (`build_alibi_slopes`, `apply_alibi`), partial-RoPE, M-RoPE. The pre-rollout v2 §4.2 "9-op floor" framing is now misleading. Cite `api/ops.py:17-866`.

2. **Extend `research/02-layer-sources.v3.md §3.14`** — add hash-routing (DeepSeek-V4 `tid2eid` lookup) and latent-MoE wrapper (Nemotron-H `fc1/fc2_latent_proj` around expert dispatch) as enumerated values of the channel-mixer axis. The IR carries both with primary-source citations (`modeling_deepseek_v4.py:1050-1078`, `modeling_nemotron_h.py:672-745`); the research catalog must catch up. Same for `expert_kind="gpt_oss_clamped_swiglu"` in §3.12.

3. **Document the 3.6 NoPE collapse in both `research/02 §3.6` and `research/05 §7.2`** — add a sentence: "The IR implements this as `RoPESpec | None` on `AttentionSpec.rope` per-layer rather than as a separate `apply_rope_per_layer: list[bool]` array; the semantics are equivalent. See `api/attention.py:290-300`." Without this, future reviewers will flag the missing field name.

4. **Add explicit Cohere or Apple-AFM coverage to `models/`** — `BlockLayout.PARALLEL` is currently exercised only by `models/falcon7b/`; Cohere is a more architecturally-distinct user (parallel-residual + LayerNorm-no-bias). Apple AFM is the only `research/02 §3.33` cross-layer-KV-sharing variant the IR doesn't model. Both block the "cross-block KV sharing" axis (3.33) from being fully validated.

5. **Reconcile arc-orphan models with `research/01 §5`** — `models/mpt/` lacks an evolution-arc entry; `models/glm_moe_dsa/`, `models/minimax_m2/`, `models/voxtral/` need explicit row entries in the census table. Either add §5.34 "MPT / parallel-residual ALiBi historical line", or fold MPT into a §5.x.

---

## 8. Recommendations beyond the must-fix list

- **A2-style guardrail**: when admitting a Raschka-gallery-driven IR field (per `docs/issues/raschka-gallery-comparison.md`), the same commit should patch `research/02-layer-sources.v3.md §3` to enumerate the axis. The current workflow lets IR additions land before the research catalog reflects them, which silently invalidates the "research → spec → IR" review-gate sequence (`feedback_review_gates.md`).

- **A3-style guardrail**: maintain a single-source-of-truth table mapping every `models/<family>/` to a `research/01 §5.x` arc (or explicitly "no arc" for ablation families). The current state has 3 narrative-covered orphans and 1 outright orphan.

- **A4 follow-up**: if research/03 is refreshed, the IHV-support matrix should explicitly note `selective_scan` and `gated_delta_step` as **mixer-trait ops** that runtimes will need to either fuse-compose or decompose, and update `research/03 §4.2` recommendation to "9 universal ops + per-mixer-trait op-set".

- **A5 follow-up**: even though all shipping schemes are grounded, the `QuantSpec` dataclass does NOT carry `training_native: bool` or `retrainable: bool` from `research/04 §10.3.2-10.3.3` — minor but a follow-up that would close the "IR carries every research/04 minimal axis" claim.

- **Dead-field cleanup**: per `docs/issues/API-minimality-coverage-audit.md` §1.1a, 17 spec fields are set-but-never-read (notably `DecoderBlockSpec.{final_logit_softcap, embedding_scale, logits_scale}`, `LayerScaleSpec` whole, `KVCacheSpec.{share_scheme, num_kv_shared_layers}`, `RoPESpec.scale_factor`, `RoPESpec.is_2d`). These should be pruned or wired to a real consumer; they distort the A1/A2 mapping by appearing to grant axis coverage that does not exist in the forward path.

---

## Appendix — citation index for spec fields surveyed

- `api/specs.py:14-26` NormSpec (kind, eps, weight_mode, has_bias)
- `api/specs.py:30-35` Llama3RoPEParams
- `api/specs.py:39-77` YarnRoPEParams
- `api/specs.py:80-95` LongRoPEParams
- `api/specs.py:99-126` AliBiSpec
- `api/specs.py:130-168` RoPESpec (incl. partial_rotary_factor / partial_rotary_kind / mrope_section)
- `api/specs.py:172-308` AttentionSpec — line-by-line spec
- `api/specs.py:312-327` FFNSpec (incl. ffn_sub_norm)
- `api/specs.py:331-339` QuantSpec
- `api/specs.py:343-348` KVCacheSpec
- `api/specs.py:352-364` ConvSpec
- `api/specs.py:368-411` SSMSpec (incl. SSMKind dispatcher, dt_rank, dt_layernorm)
- `api/specs.py:415-447` SSDSpec
- `api/specs.py:451-488` GatedDeltaNetSpec (v7 P3)
- `api/specs.py:492-495` GroupRoutingSpec
- `api/specs.py:498-592` MoESpec (incl. router_kind="hash", routing_in_latent, expert_kind)
- `api/specs.py:595-625` CSASpec (v7 P2)
- `api/specs.py:628-648` HCASpec (v7 P2)
- `api/specs.py:651-669` IndexerSpec
- `api/specs.py:672-677` LayerScaleSpec (DEAD per minimality audit)
- `api/specs.py:680-695` PLESpec
- `api/specs.py:699-733` DecoderBlockSpec
- `api/types.py:6-19` AttentionKind (incl. CSA_HCA)
- `api/types.py:22-43` TokenMixerKind (incl. GATED_DELTANET)
- `api/types.py:46-58` SSMKind
- `api/types.py:67-85` MaskKind (incl. BLOCK_BIDIRECTIONAL)
- `api/types.py:88-92` QKNormPhase
- `api/types.py:94-97` QKNormShape
- `api/types.py:116-131` BlockLayout (SEQUENTIAL / PARALLEL)
- `api/types.py:160-164` RoPEScaling (NONE / YARN / LLAMA3 / LONGROPE)
- `api/types.py:185-196` QDType (INT4, TERNARY)
- `api/types.py:204-209` QuantRole
- `api/ops.py:17-866` 26 public ops (silu, add, mul, linear, rms_norm, embed, lm_head, rope_apply, rope_apply_mrope, rope_apply_partial, gelu_pytorch_tanh, gelu_exact, relu2, build_alibi_slopes, apply_alibi, sdpa, layer_norm, softmax, top_k, gather, scatter, conv1d, selective_scan_mamba1, selective_scan, l2norm, gated_delta_step)
- `api/attention.py:290-300` NoPE-via-`rope=None` collapse note
- `api/block.py:255-313` PARALLEL block-layout forward branch
- `api/block.py:336-345` PLE-at-end injection forward
- `api/embedding.py:22-63` PerLayerEmbedding
- `api/quant.py` 6 shipped schemes (AWQ, GGUF Q4_K_M, FP8 E4M3, MXFP4, LiteRT W4A8, BitNet ternary) + 2 deferred (IQ2_M, AQLM)
