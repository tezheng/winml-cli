# Audit C — Example Correctness: 5 Model Family Deep Audit

**Date:** 2026-06-10  
**Scope:** Qwen3, Gemma 4, GPT-OSS, Qwen3-Next, DeepSeek-V4  
**Checks per family:** dtype handling, to_block_spec completeness, weight loader integrity, numerical gate validity, layer.md §0 accuracy  
**Verdict format:** PASS / PARTIAL / FAIL

---

## 1. Qwen3 — PASS

### Check 1 — Config dtype handling
`Qwen3Config.from_hf_dict` (config.py:47) handles both `dtype` (5.x) and `torch_dtype` (4.x) correctly via `hf.get("dtype", hf.get("torch_dtype", "float32"))`. It also handles `rope_theta` in both the flat key and the nested `rope_parameters` dict. **PASS.**

### Check 2 — to_block_spec completeness
`to_block_spec()` takes no `layer_idx` parameter, which is correct: Qwen3 is a uniform architecture. The method sets every field the architecture requires:
- `NormSpec` with `STANDARD_W` for both pre-attn and pre-ffn norms, and as `qk_norm_spec`
- `AttentionSpec` with explicit `head_dim` (critical — Qwen3-0.6B has hidden_size/n_q=64 vs head_dim=128), `QKNormPhase.PRE_ROPE`, `QKNormShape.PER_HEAD_DH`, `RoPEBasis.SPLIT_HALF`, `base_theta` sourced from config
- `FFNSpec` with `GateKind.SWIGLU`, `Activation.SILU`, no biases

Cross-referencing layer.md §5 spec instantiation: exact match to the code. **PASS.**

### Check 3 — Weight loader integrity
`load_hf_qwen3_layer` maps 9 tensors: input_layernorm (→ pre_attn_norm), q/k/v/o_proj (→ attention), post_attention_layernorm (→ pre_ffn_norm), gate/up/down_proj (→ feedforward). Q-norm and k-norm are conditionally added when present. Layer.md §7 lists the same mapping. **PASS.**

### Check 4 — Numerical gate validity
`tests/models/qwen3/test_numerical_hf.py`:
- `ATOL = 5e-4` explicitly set, not loosened.
- Tests against `Qwen3DecoderLayer` (via the full HF model's layer, not a sub-module).
- Layer.md §9 reports observed `max_abs_diff = 4.77e-7`, ~1000× tighter than the 5e-4 contract. **PASS.**

### Check 5 — Layer.md §0 accuracy
- **Signature:** Correct (GQA, QK-norm PRE-RoPE per-Dh, SwiGLU, θ=1M).
- **Active params:** Accurate range (0.6B / 1.7B / 4B / 8B).
- **Layer mix:** Correct (28/28/36/36 layers, all dense).
- **KV cache / token:** Formula `28 L × 8 KV × 128 Dh × 2 × 2 = 114 kB` uses decimal KB with floor truncation rather than binary kibibytes (true value is 114688 bytes = 112 KiB, but `114688 / 1000 = 114.7` → 114 KB decimal). The computation and the multiplication are correct; only the unit label is imprecise (uses KB₁₀ not KiB). Similarly, the 4B claim of 147 kB is actually 144 KiB. **Minor documentation imprecision, not a code bug.**

**Family verdict: PASS** (1 documentation unit-precision note, no code bugs).

---

## 2. Gemma 4 — PARTIAL

### Check 1 — Config dtype handling
`Gemma4Config.from_hf_dict` (config.py:69) handles both `dtype` and `torch_dtype` correctly. Default is `"bfloat16"` (appropriate for Gemma). **PASS.**

However, line 83 reads `rope_theta_local=float(d["rope_theta"])` — a **hard key access** that will raise `KeyError` if the caller passes a raw HF config dict where `rope_theta` is only stored under `rope_parameters.sliding_attention.rope_theta`. Qwen3's equivalent path uses `hf.get("rope_theta", ...)` with a fallback. The test works around this by manually bridging the nested path into `bridged["rope_theta"]` before calling `from_hf_dict`. **Silent fragility bug.**

Similarly, `partial_rotary_factor_global=float(d.get("partial_rotary_factor_global", 1.0))` (line 85) defaults to `1.0`. The correct value for Gemma 4 E2B global layers is `0.25`, which lives in the raw HF config under `rope_parameters.full_attention.partial_rotary_factor`. If `from_hf_dict` is called with a raw HF dict, this silently returns 1.0 instead of 0.25 — a **silent wrong-value bug** that would corrupt global-layer partial-RoPE. The test prevents this only by pre-bridging the value before calling `from_hf_dict`.

### Check 2 — to_block_spec completeness
`to_block_spec(layer_idx)` dispatches on `is_global_layer(layer_idx)` and sets:
- Local: `MaskKind.SWA`, `sliding_window`, `rope_theta_local`, `partial_rotary_factor=1.0`, `head_dim=local`
- Global: `MaskKind.CAUSAL`, `rope_theta_global`, `partial_rotary_factor=0.25`, `head_dim=global`, `attention_k_eq_v`
- Both: `v_norm`, `v_norm_with_scale=False`, `attn_scale=1.0`, `qk_norm_fixed_scale=None` (B0.6 absorb removed)
- `PLESpec` when `use_per_layer_embedding=True`
- `NormPosition.PRE_AND_POST` for both attn and ffn (sandwich norm)

Cross-referencing layer.md §5: matches. The `attn_norm_position=PRE_AND_POST` correctly maps to 4-norm sandwich. `FFNSpec` uses `GateKind.GEGLU` / `Activation.GELU` (GeGLU, Gemma signature). **PASS.**

### Check 3 — Weight loader integrity
`load_hf_gemma4_layer` covers: input_layernorm, post_attention_layernorm, pre/post_feedforward_layernorm, q/k/o_proj (v_proj conditional on non-K=V layers), q_norm/k_norm (conditional on q_norm not None), PLE tensors (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) when present, and `layer_scalar` buffer (conditional on key presence).

**Gap:** The synthetic `_random_state_dict` in `test_weight_loader.py` does not include `layer_scalar`, so the layer_scalar load path is never exercised by any weight-loader test. The loader itself handles the absent key gracefully, but there is no test that a real state dict with `layer_scalar` loads correctly. **Test coverage gap (not a loader bug).**

### Check 4 — Numerical gate validity
`tests/models/gemma4/test_numerical_hf.py`:
- `ATOL = 5e-4`, not loosened.
- Tests against HF `Gemma4TextDecoderLayer` (via `hf_text_model.layers[idx]`), which is the correct decoder layer class.
- Both layer-0 (local SWA) and layer-4 (first global, exercises partial-RoPE 0.25 + θ=1e6 + global_head_dim) are tested.
- Layer.md §9 reports the T17 test as passing. **PASS.**

**Note:** The test has a try/except around the weight-load step that `pytest.skip`s on `KeyError/ValueError`. This means a silent regression in the weight loader (e.g., due to a new HF naming convention) would cause the test to skip rather than fail. This is a gate-escape risk but not an atol loosening.

### Check 5 — Layer.md §0 accuracy
- **Signature:** Correct (sandwich norm PRE_AND_POST, proportional p-RoPE on global, PLE, SharedLayerKVCache, v_norm, layer_scalar).
- **Active params:** ~2.4B active / ~5B total (E2B) — consistent with publicly known E2B figures.
- **Layer mix:** 35 layers (28 local + 7 global, 4:1). Verified: `is_global_layer(i) = (i+1) % 5 == 0` gives layers 4, 9, 14, 19, 24, 29, 34 = 7 global. **Correct.**
- **KV cache / token:** `~18 kB` for `15 unshared layers: 12 local × 1 kv × 256 hd × 4 B + 3 global × 1 kv × 512 hd × 4 B = 12288 + 6144 = 18432 bytes = 18 kB`. Arithmetic is correct. **PASS.**

**Family verdict: PARTIAL** — two silent bugs in `from_hf_dict` (hard rope_theta key access; partial_rotary_factor_global defaults to 1.0 not 0.25 for raw HF configs). Test mitigation exists (pre-bridging in tests) but `from_hf_dict` would fail or silently misbehave if called directly with a raw HF config.

---

## 3. GPT-OSS — PARTIAL

### Check 1 — Config dtype handling
`GptOssConfig` (config.py) has **no `from_hf_dict` method**. The only constructor path is `gpt_oss_20b()` (a frozen `@classmethod` with hardcoded defaults) plus direct dataclass instantiation. The `dtype` field has a default of `torch.float32` on the dataclass. There is no `dtype`/`torch_dtype` bridging because there is no HF dict deserialization path. **NOT APPLICABLE by design** — GPT-OSS is a SHAPE-ONLY family and the `__init__.py` documents that full HF deserialization is out of scope. The Check 1 requirement is structurally inapplicable for this family. No bug, but zero HF config compatibility.

### Check 2 — to_block_spec completeness
`GptOssConfig` has **no `to_block_spec()` method**. It exposes `to_attention_spec(layer_idx)` and `to_moe_spec()` as separate builders. This is an intentional SHAPE-ONLY decomposition documented in the `__init__.py`: the GPT-OSS MoE experts use a custom forward (`gpt_oss_clamped_swiglu` with biases) that differs from the standard MoE path, and MXFP4 wiring is deferred. **NOT APPLICABLE by design.**

The `to_attention_spec(layer_idx)` spec is complete for the attention side: SINK mask, `n_sink_tokens=1`, YARN RoPE with `factor=32.0` / `base_theta=150000.0`, per-layer sliding_window dispatch, biases all True. The `to_moe_spec()` correctly sets `router_kind="topk_then_softmax_with_bias"`, `expert_kind="gpt_oss_clamped_swiglu"`, `expert_bias=True`, `expert_swiglu_alpha=1.702`, `expert_clamp_limit=7.0`. **PASS for what is scoped.**

### Check 3 — Weight loader integrity
**No `load_hf_gpt_oss_layer` function exists** — GPT-OSS has no `layer.py`. This is consistent with the SHAPE-ONLY designation. The MoE test (`test_moe_vs_hf.py`) uses a direct in-test weight copy. The sink attention test (`test_sink_attention_vs_hf.py`) uses synthetically constructed tensors without a weight loader. **NOT APPLICABLE by design.**

### Check 4 — Numerical gate validity
Two test files:
- `test_moe_vs_hf.py`: Tests `GptOssMLP` (the HF MoE block) vs api `MoE(expert_kind="gpt_oss_clamped_swiglu")`. Uses `ATOL = 5e-4`, not loosened. Tests against the correct HF class (`GptOssMLP`, not a wrapping `GptOssDecoderLayer`). Multiple test cases including top_k=1 corner case and shape assertions. **PASS.**
- `test_sink_attention_vs_hf.py`: Tests `ops.sdpa(sinks=...)` vs HF `eager_attention_forward`. Uses `ATOL = 5e-4`, not loosened. Tests both non-GQA and GQA configurations. **PASS.**

### Check 5 — Layer.md §0 accuracy
GPT-OSS has **no `layer.md` file**. The `__init__.py` serves as the family documentation. It accurately describes:
- The SHAPE-ONLY scope
- Which features are landed (SINK forward) vs deferred (MXFP4, RoPE channel-layout compat)
- The alternating sliding/full pattern

The `gpt_oss_20b()` method hard-codes `"sliding_attention" if (i+1)%2 else "full_attention"` for 36 layers. Verification: layer 0 → (0+1)%2=1 → sliding; layer 1 → (1+1)%2=0 → full. Pattern is correct.

The absence of a `layer.md` with §0 means the At-a-glance fields cannot be formally checked. No numerical KV cache formula to verify against. **NOT APPLICABLE (no layer.md).**

**Family verdict: PARTIAL** — intentionally incomplete (SHAPE-ONLY). All in-scope checks pass. No silent bugs. The family is correctly and honestly scoped. The PARTIAL reflects scope incompleteness, not bugs.

---

## 4. Qwen3-Next — PASS

### Check 1 — Config dtype handling
`Qwen3NextConfig.from_hf_dict` (config.py:74) handles both `dtype` and `torch_dtype` correctly: `hf.get("dtype", hf.get("torch_dtype", "float32"))`. **PASS.**

`rope_theta` is handled defensively via a `rope_parameters` sub-dict fallback chain, matching Qwen3's pattern (lines 96-103). **PASS.**

### Check 2 — to_block_spec completeness
`to_block_spec(layer_idx)` dispatches on `is_linear_attention_layer(layer_idx)`:

**Linear-attention path:** emits `GatedDeltaNetSpec` with all required fields (`num_v_heads`, `num_k_heads`, `head_k_dim`, `head_v_dim`, `conv_kernel`, `norm_eps`, `silu_gate=True`). `NormWeightMode.ONE_PLUS_W` correctly set on both `pre_attn_norm` and `pre_ffn_norm` (verified against layer.md: `modeling_qwen3_next.py:152-166`).

**Full-attention path (SHAPE-ONLY):** emits `AttentionSpec STANDARD` with `ONE_PLUS_W` qk_norm, `PRE_ROPE`, `PER_HEAD_DH`, partial_rotary_factor from config. Marked shape-only per spec and docstring (the HF q_proj is 2×-sized for (q|gate) and the IR's STANDARD does not model output-gating).

**MoE vs dense dispatch:** `is_mlp_only_layer(layer_idx)` correctly forces dense `FFNSpec` for layers in `mlp_only_layers` or when `num_experts <= 0` or sparse step doesn't fire. Otherwise `MoESpec(router_kind="softmax")` with `norm_topk_prob`. The `shared_expert_gate` sigmoid is explicitly called out as not modeled.

Cross-referencing layer.md §§ describing the block envelope: matches. **PASS.**

### Check 3 — Weight loader integrity
`load_hf_qwen3_next_linear_attn_layer` maps 9 tensors for the linear-attention path: input_layernorm, post_attention_layernorm, and 7 GatedDeltaNet tensors (`in_proj_qkvz.weight`, `in_proj_ba.weight`, `conv1d.weight`, `dt_bias`, `A_log`, `norm.weight`, `out_proj.weight`). Dense MLP path (gate/up/down_proj) is conditionally appended for `mlp_only` layers.

Cross-referencing layer.md §7 weight mapping table: exact 1:1 match. No orphan or missing tensors for the scoped (linear-attn + dense-mlp) case. Full-attention and MoE paths are explicitly left unloaded (documented reasons: output-gating shape mismatch; shared_expert_gate not modeled). **PASS for scoped paths.**

### Check 4 — Numerical gate validity
`tests/models/qwen3_next/test_numerical_synthetic.py`:
- `ATOL = 5e-4`, not loosened.
- Tests against `Qwen3NextDecoderLayer` (the full decoder layer class, not a sub-module). Layers 0 and 2 are tested (both `linear_attention` type).
- Uses `num_experts=0` to force dense MLP (avoids the un-modeled `shared_expert_gate`). This is the correct workaround and is documented.
- The full-attention layer (layer 3) is **NOT gated** — documented in the test file and layer.md as SHAPE-ONLY due to (q|gate) fusion + output gating. No atol loosening; simply no numerical test exists for that path. **PASS.**

### Check 5 — Layer.md §0 accuracy
- **Signature:** "Gated DeltaNet 3:1 linear-attention hybrid (75% DeltaNet, 25% softmax attn every 4th) + ultra-sparse MoE (512 experts, top-10, + 1 shared)" — matches production constants (48 layers at 3:1 → ~36 DeltaNet + ~12 attn). Note: at-a-glance says "~71 DeltaNet + ~23 attn" which is the 80B-A3B count at 94 layers, but `num_hidden_layers` from production constants is 48 (see §Production constants: `num_hidden_layers=48`). **Mild inconsistency:** the "71 + 23" figure would require 94 layers, but the listed production constant is `num_hidden_layers=48` (which gives ~36+12). Minor documentation drift; the ratio claim (3:1, every 4th) is correct.
- **KV cache / token:** `~46 kB (23 softmax-attn layers × 2 kv heads × 256 head_dim × 4 B)`. Computed: `23 × 2 × 256 × 4 = 47104 bytes = 46.0 kB`. **Correct** (uses decimal KB, ~same as Qwen3 treatment).
- **Active params:** `~3B active / ~80B total (4%)` — consistent with Qwen3-Next-80B-A3B designation. **PASS.**

**Family verdict: PASS** (1 minor layer-count drift in at-a-glance; no code bugs).

---

## 5. DeepSeek-V4 — PARTIAL

### Check 1 — Config dtype handling
`DeepSeekV4Config.from_hf_dict` (config.py:84) handles both `dtype` and `torch_dtype` correctly: `hf.get("dtype", hf.get("torch_dtype", "float32"))`. The `test_from_hf_dict_uses_dtype_key` test (test_config.py:116) explicitly verifies the `dtype` key takes precedence. **PASS.**

### Check 2 — to_block_spec completeness
`to_block_spec(layer_idx)` dispatches on both `_attn_spec(layer_idx)` (CSA / HCA / sliding) and `_moe_spec(layer_idx)` (hash vs sigmoid+bias). All three attention branches produce correctly typed `AttentionSpec`:
- CSA: `AttentionKind.CSA_HCA` with `CSASpec(compress_rate, block_size, indexer_n_heads, indexer_head_dim, indexer_topk)`
- HCA: `AttentionKind.CSA_HCA` with `HCASpec(compress_rate, hierarchy_levels=1)`
- Sliding: `AttentionKind.STANDARD` with `MaskKind.SWA`, `sliding_window`

RoPE is explicitly set to `SPLIT_HALF` (not `INTERLEAVED`) with `partial_rotary_factor = qk_rope_head_dim / head_dim`, with a comment explaining the INTERLEAVED IR limitation. The comment is accurate.

MoE spec: hash layers get `router_kind="hash"` + `hash_vocab_size` + `hash_score_fn="sigmoid"`; topk layers get `router_kind="sigmoid_plus_bias"` + `router_norm`. **PASS for scoped paths.**

### Check 3 — Weight loader integrity
`load_hf_deepseek_v4_hash_moe` maps: `gate.weight`, `gate.tid2eid` (hash path) or `gate.e_score_correction_bias` (topk path), `experts.gate_up_proj`, `experts.down_proj`, and conditionally `shared_experts.{gate,up,down}_proj.weight`.

**Silent bug (docstring vs code):** The docstring (layer.py lines 55–57) documents the shared-experts HF tensor names as:
```
model.layers.{L}.mlp.shared_experts.gate_proj.W
model.layers.{L}.mlp.shared_experts.up_proj.W
model.layers.{L}.mlp.shared_experts.down_proj.W
```
But the actual code (lines 87–92) uses `.weight` not `.W`. The code is correct (HF `DeepseekV4MLP` is a standard `nn.Linear` — its weight attribute is `.weight`). The test confirms this: `test_isolation_hf.py` line 106 copies via `hf_block.shared_experts.gate_proj.weight.data`. The docstring is wrong; the code will not break. **Docstring bug, not a functional bug.**

### Check 4 — Numerical gate validity
No `test_numerical_hf.py` or equivalent; the V4 numerical gate lives in `tests/models/deepseek_v4/test_isolation_hf.py`:
- `ATOL = 5e-4`, not loosened.
- Tests against `DeepseekV4SparseMoeBlock` (the correct HF MoE block class, `is_hash=True`).
- Verifies hash routing determinism (same `input_ids` → same expert selection regardless of hidden states).
- CSA/HCA attention forward intentionally raises `NotImplementedError` (shape-only P2) — tested in `test_layer_shape.py:test_csa_attention_forward_raises`. **PASS for hash MoE; CSA/HCA numerics are intentionally unscoped.**

### Check 5 — Layer.md §0 accuracy
**Silent inconsistency in layer count:**
- **At-a-glance (line 6):** "Layer mix: 64 hash-MoE (3 hash-router + **61** sigmoid+bias; CSA/HCA/SWA per-layer attn dispatch)"
- **Production constants (line 138):** `mlp_layer_types=["hash_moe"]*3 + ["moe"]*(43-3) = 3 hash + **40** moe`

These are contradictory: 3 + 61 = 64 total layers vs 3 + 40 = 43 total. The KV cache formula on line 7 uses the at-a-glance value: `128 kB (64 layers × 1 kv head × 512 head_dim × 4 B)`. If the real V4-Flash-Base has 43 layers, the KV figure should be `43 × 1 × 512 × 4 = 88064 bytes ≈ 86 kB`, not 128 kB.

This is a **silent documentation inconsistency** between two sections. It implies either (a) the at-a-glance is for a different V4 variant than "V4-Flash-Base", or (b) one of the two counts is wrong. The `num_hidden_layers` default in `from_hf_dict` is not set in the config (uses `hf["num_hidden_layers"]` as a mandatory field), so the codebase itself does not encode a single canonical count.

- **Active params:** `~49B active / ~685B total (7%)` — plausible for a full V4 Flash-Base model at 256 experts, 6 active, with hidden_size=4096 and 64 layers.
- **KV cache formula:** Arithmetic is internally consistent with the 64-layer count: `64 × 1 × 512 × 4 = 128 kB`. The formula itself is correct; the disagreement is between the two sections' layer counts.

**Family verdict: PARTIAL** — two bugs: (1) docstring `.W` vs `.weight` in layer.py (functional code is correct, docstring wrong); (2) layer.md §0 at-a-glance claims 64 layers while production constants section says 43 (3+40), making the KV cache claim internally inconsistent.

---

## Overall Verdict: PARTIAL

### Summary table

| Family | Check 1 (dtype) | Check 2 (to_block_spec) | Check 3 (loader) | Check 4 (numerical gate) | Check 5 (§0 accuracy) | Verdict |
|--------|----------------|------------------------|-----------------|--------------------------|----------------------|---------|
| Qwen3 | PASS | PASS | PASS | PASS (atol=5e-4) | PASS (minor unit precision) | **PASS** |
| Gemma 4 | PARTIAL (2 silent bugs) | PASS | PASS (layer_scalar test gap) | PASS (atol=5e-4) | PASS | **PARTIAL** |
| GPT-OSS | N/A (no from_hf_dict) | N/A (no to_block_spec) | N/A (no layer.py) | PASS (atol=5e-4) | N/A (no layer.md) | **PARTIAL** |
| Qwen3-Next | PASS | PASS | PASS | PASS (atol=5e-4) | PASS (minor layer count drift) | **PASS** |
| DeepSeek-V4 | PASS | PASS | PARTIAL (docstring bug) | PASS (atol=5e-4) | PARTIAL (layer count conflict) | **PARTIAL** |

---

## Top 5 Silent Bugs Found

### Bug 1 — Gemma 4: `from_hf_dict` silent wrong default for `partial_rotary_factor_global`
**File:** `models/gemma4/config.py:85`  
**Severity:** HIGH — would silently corrupt global-layer partial-RoPE  
`partial_rotary_factor_global=float(d.get("partial_rotary_factor_global", 1.0))` defaults to `1.0`. The correct E2B value is `0.25`, stored in the raw HF config under `rope_parameters.full_attention.partial_rotary_factor` (not a top-level key). Calling `from_hf_dict` with an unprocessed HF config dict returns the wrong value silently. All existing tests pre-bridge this field before calling `from_hf_dict`, so no test currently catches this path. Global layers would apply full RoPE rotation instead of partial, producing numerically wrong attention scores.

### Bug 2 — Gemma 4: `from_hf_dict` hard KeyError on `rope_theta`
**File:** `models/gemma4/config.py:83`  
**Severity:** MEDIUM — raises KeyError on unprocessed HF config dict  
`rope_theta_local=float(d["rope_theta"])` uses direct dict access with no fallback. Raw HF Gemma 4 config stores theta under `rope_parameters.sliding_attention.rope_theta`. Compare with Qwen3's defensive `hf.get("rope_theta", ...)` pattern. All existing tests pre-bridge `rope_theta` into the flat key before calling `from_hf_dict`. If a caller passes an unprocessed config, this raises `KeyError` instead of using a sensible default.

### Bug 3 — DeepSeek-V4: layer.md §0 layer count inconsistency (64 vs 43)
**File:** `models/deepseek_v4/layer.md`, lines 6 and 138  
**Severity:** MEDIUM — documentation inconsistency propagates to wrong KV cache figure  
At-a-glance states "64 hash-MoE (3 hash-router + 61 sigmoid+bias)" implying 64 total layers. Production constants section states `["hash_moe"]*3 + ["moe"]*(43-3) = 3 hash + 40 moe` implying 43 total layers. The KV cache formula (`128 kB`) is consistent with 64 layers (`64 × 1 × 512 × 4`), not with 43 layers (which would give ~86 kB). At minimum one of the two sections is wrong.

### Bug 4 — DeepSeek-V4: weight-loader docstring uses `.W` instead of `.weight`
**File:** `models/deepseek_v4/layer.py`, lines 55–57  
**Severity:** LOW — code is correct, docstring misleads  
The function docstring documents shared-expert HF tensor names as `shared_experts.gate_proj.W` / `.up_proj.W` / `.down_proj.W`. The actual mapping uses `.weight` (lines 87–92), which is correct (HF `DeepseekV4MLP` uses standard `nn.Linear` weight attribute). The test (`test_isolation_hf.py:106`) confirms `.weight` is correct. Anyone using the docstring to manually build a state dict for testing would create a state dict with the wrong keys, causing the `missing` check to raise `KeyError`.

### Bug 5 — Gemma 4: `test_weight_loader.py` does not include `layer_scalar` in random state dict
**File:** `tests/models/gemma4/test_weight_loader.py`, `_random_state_dict()`  
**Severity:** LOW — test coverage gap for the layer_scalar load path  
The `_random_state_dict` helper never includes the `"model.layers.{i}.layer_scalar"` key. The loader handles its absence gracefully (conditional copy on line 131). However, the critical path — loading a non-unit `layer_scalar` from a real checkpoint and verifying it lands in `blk.layer_scalar` — is never exercised. A regression in this path would go undetected by the weight-loader test suite (only caught if the T17 real-weight numerical test runs, which requires gated HF access).

---

## Atol-loosening detected?

**NO.** All five families use `ATOL = 5e-4` without exception:
- Qwen3 `test_numerical_hf.py`: `ATOL = 5e-4`, `RTOL = 5e-4`
- Gemma 4 `test_numerical_hf.py`: `ATOL = 5e-4`, `RTOL = 5e-4`
- GPT-OSS `test_moe_vs_hf.py`: `ATOL = 5e-4`, `RTOL = 5e-4`
- GPT-OSS `test_sink_attention_vs_hf.py`: `ATOL = 5e-4`, `RTOL = 5e-4`
- Qwen3-Next `test_numerical_synthetic.py`: `ATOL = 5e-4`, `RTOL = 5e-4`
- DeepSeek-V4 `test_isolation_hf.py`: `ATOL = 5e-4`, `RTOL = 5e-4`

No silent loosening found. The 5e-4 contract is uniformly held.

---

## Additional Notes

- **GPT-OSS Check 2 / from_hf_dict:** The absence of `from_hf_dict` and `to_block_spec` is by explicit design (SHAPE-ONLY family). This does not constitute a bug but does mean the family cannot be instantiated from a raw HF config dict, which limits its usability as a drop-in adapter.

- **Qwen3-Next at-a-glance layer count drift:** The "~71 DeltaNet + ~23 attn" figure (total ~94 layers) does not match the production constants' `num_hidden_layers=48` (which would give ~36 DeltaNet + ~12 attn). The ratio (3:1) is correct but the raw counts are inconsistent within the same file. The KV cache formula uses the `num_hidden_layers=48` figure (23 attn layers, assuming 48/4=12 attn but the formula uses 23, which is closer to the 94-layer figure divided differently). This is a documentation inconsistency that does not affect code correctness.

- **Gemma 4 numerical test skip path:** `test_numerical_hf.py` wraps both the weight-load and the HF forward in `try/except` blocks that `pytest.skip` rather than fail. While understandable for a gated model, this means a loader regression would silently skip the test rather than fail it. This is worth hardening if/when the model becomes ungated.
