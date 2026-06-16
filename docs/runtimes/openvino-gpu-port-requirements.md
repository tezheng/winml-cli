# OpenVINO GPU Port — Requirements (M1 + M2)

**Status:** Requirements / RFC. Awaiting kickoff for a new working thread.
**Audience:** Engineer continuing the OpenVINO port of `components/` in a new session.
**Last updated:** 2026-06-15.
**Target hardware:** Intel Arc 140V Xe2 iGPU (16 GB shared / 31 GB system), driver `32.0.101.8424`.
**Target runtime:** OpenVINO `2026.2.0`, with optional `openvino-genai` and `optimum-intel` (already in `pyproject.toml`'s `openvino` extra).
**Predecessor docs:** `docs/runtimes/openvino-gpu.md` (M0 verification), `components/api_spec.md` (normative spec API), `components/README.md` (intro).

---

## 1. Executive summary

This document specifies what the OpenVINO GPU port (`runtimes/openvino_port/`) MUST grow into in order to execute the full `components/` API surface on Intel client GPUs. M0 (already landed and verified) lowers a five-spec vertical slice — `NormStandardSpec`, vanilla `GQASpec`, full-rotary `RoPESpec`, `SwiGLUSpec`, and `AddSpec` wired through `pre_norm_block` — via direct opset16/opset13 node emission, and clears the `mean_rel < 5e-4` parity gate against the torch_port fp32 golden on Arc 140V (see `tests/parity/test_openvino_vs_torch.py:50`).

M1 is the priority for the new thread. It MUST extend each existing per-component builder to cover all variants used by the four reference layers shipped under `examples/v2/define_*.py` (Qwen3-0.6B, Qwen3.5-35B-A3B, Gemma 4 E2B, GPT-OSS-20B). M1 lands fp16 execution, the M1 attention extension fields (`qk_norm`, `output_gate`, `v_norm`, `qk_norm_fixed_scale`, `sliding_window`), the long-context RoPE family, `MRoPEInterleavedSpec`, `MLASpec`, `GQASinksSpec`, `GeGLUSpec`, `ClampedSwiGLUSpec`, `NormZeroCenteredSpec`, and the `sandwich_block` / `post_norm_reordered_block` / `parallel_block` topologies.

M2 is the stretch tier: stateful KV-cache via OpenVINO `MakeStateful`, NNCF INT4 / INT8 weight-only compression, `KVCacheSpec` family (dense / latent / sliding-ring / SSM-state), `MoEBlock`, `Embedding`, `LMHead`, and an `openvino_genai.LLMPipeline` wrapping layer for continuous-batching multi-stream serving. Success at M2 is a full Qwen3-0.6B model executed end-to-end on Arc 140V at ≥30 tok/s, INT4 weights, fp16 compute, stateful KV — matching the throughput already demonstrated via `optimum-intel` (`docs/runtimes/openvino-gpu.md` §3, 49.7 tok/s) but produced from the project's own builder path.

The work is **strictly incremental on M0's direct-emission architecture**. We do not switch to the PyTorch frontend (`ov.convert_model`) for the project itself. M0's builders fold cleanly into OV's pattern matchers (RMSFusion, RoPEFusion, GLU, SDPA→PagedAttention via `MakeStateful`) when the emitted IR matches the canonical pattern; the requirements below preserve that property.

---

## 2. Goals and non-goals

### 2.1 Goals

1. **Numerical equivalence with torch_port** for every `components/` spec variant on Intel Arc 140V Xe2 iGPU, gated at `mean_rel < 5e-4` against the torch fp32 golden.
2. **fp16 primary execution path.** `INFERENCE_PRECISION_HINT="f16"` MUST be the default once weight scaling lands; fp32 SHALL be a regression-only fallback.
3. **INT4 weight-only quantization** via NNCF (`nncf.compress_weights`) for every Linear-bearing component (attention Q/K/V/O, FFN gate/up/down, LM head). Group-wise asymmetric (`INT4_ASYM`, `group_size=128`) is the default mode.
4. **Stateful KV-cache** via `opset3::ReadValue` / `opset3::Assign` Variables, installed by either hand-emission inside `kv_cache.py` or the `MakeStateful` transformation pass.
5. **Direct opset14 SDPA emission** to unblock `GQASinksSpec` (which the PyTorch frontend cannot pass — see Section 4.2.4).
6. **Continuous-batching pipeline** via `openvino_genai.LLMPipeline` wrapping the assembled `ov.Model`, with PagedAttention KV management.
7. **Static-shape compilation as the baseline.** Dynamic-batch is a stretch goal and SHALL be enabled per-builder via an explicit constructor flag.
8. **No silent CPU fallback.** Every compiled model MUST be asserted `EXECUTION_DEVICES == ['GPU.0']` at runtime.
9. **Compile time ≤ 60 s** for a single decoder block on a warm OpenCL kernel cache.

### 2.2 Non-goals

1. **Other devices.** No CPU EP, NPU plugin, DML/D3D fallback, or ONNX Runtime integration. The ONNX port lives at `runtimes/onnx_port/` and is owned separately.
2. **Training / fine-tuning.** Weights are read-only.
3. **HuggingFace weight loaders.** Weight ingestion is out of scope; tests bind weights produced by `tests/parity/_canonical.py` and the torch_port golden.
4. **Multi-GPU.** Single Arc 140V iGPU.
5. **Custom OpenCL kernel authoring.** Only opset ops and (where strictly necessary) Python `openvino.Extension` custom ops are permitted. We do not author `.cl` files.
6. **PyTorch frontend (`ov.convert_model`).** M0's direct-emission philosophy stands; we keep the IR under our control so we can target specific fusion patterns and emit opset14 ops without translator coverage gaps.
7. **Server-side multi-tenant scheduling.** GenAI integration is for single-process continuous batching only, not for `ovms` deployment.

---

## 3. Background and current state

### 3.1 Existing port layout

```text
runtimes/openvino_port/
  __init__.py        # re-exports build_* + lower_block
  attention.py       # build_gqa (M0-vanilla only)
  block_lowerer.py   # walks BlockGraph, materializes Parameters, dispatches builders
  ffn.py             # build_swiglu (non-fused)
  misc.py            # build_add (N-way)
  norm.py            # build_rms_norm (NormStandardSpec only)
  rope.py            # build_rope_cos_sin + apply_rope_single (full rotary)
```

7 files. M0 capabilities:

- `runtimes/openvino_port/norm.py:25` — `build_rms_norm`, decomposes to `(x32 * rsqrt(mean(x32^2)+eps)) * w32` then `convert` back. Emits opset16. **RMSFusion does not currently fire** because the final `Convert` wraps the canonical pattern boundary — see Section 4.1 finding for M1.
- `runtimes/openvino_port/attention.py:57` — `build_gqa`, GQA vanilla with explicit `repeat_interleave` for K/V replication and `opset13::scaled_dot_product_attention` with `causal=True`. Lines 82–95 explicitly raise `NotImplementedError` for every M1 extension field (`sliding_window`, `output_gate`, `v_norm`, `qk_norm`, `qk_norm_fixed_scale`, `kv_source_layer_offset`, `logit_softcap`).
- `runtimes/openvino_port/rope.py:29` — `build_rope_cos_sin` + `apply_rope_single`. `inv_freq` constant-folded; the cos/sin emission matches OV's `RoPEFusionLlama` pattern.
- `runtimes/openvino_port/ffn.py:30` — `build_swiglu`, gated form; raises `NotImplementedError` on `fused_gate_up=True`.
- `runtimes/openvino_port/misc.py:8` — `build_add`, N-way (residual reuse).
- `runtimes/openvino_port/block_lowerer.py:56` — `lower_block`, walks the graph, materializes weight Parameters with name pattern `f"{node_name}.{slot}"`, emits a single `ov.Model`. RoPE is treated as a structural pass-through (`block_lowerer.py:103-106`), and `_find_rope_predecessor` hoists the rope spec into the attention builder (`block_lowerer.py:43`). Default dtype `f16` (constant `_DEFAULT_DTYPE` at `block_lowerer.py:40`), default input shape `(1, 8, 2048)`.

### 3.2 What M0 verified

`tests/parity/test_openvino_vs_torch.py:50` sets `MEAN_REL_TOL = 5e-4` and runs both an informational fp16 pass and the fp32 gate. The canonical-setup unscaled-Normal weights push output magnitudes past fp16 range; the gate therefore runs at fp32 to be numerically defined (`tests/parity/test_openvino_vs_torch.py:10-27`). `EXECUTION_DEVICES` is asserted to contain `"GPU"` (`tests/parity/test_openvino_vs_torch.py:119`). The runtime-model op summary is logged for fusion-pass visibility (`tests/parity/test_openvino_vs_torch.py:77-94`).

### 3.3 M0 limitations the new thread inherits

- **fp32 only at the parity gate.** fp16 path runs but is not asserted because canonical-setup weights overflow.
- **Static shapes.** B and S come from `input_shape=(1, 8, 2048)` and are folded into every Reshape constant.
- **No KV cache.** Q/K/V/O are recomputed every call; no `ReadValue`/`Assign`.
- **No quantization.** All weights are fp32 Parameters.
- **Single block.** Multi-block stacking lives outside the builder.
- **Five spec types only.** Every other dispatch in `block_lowerer.py:90-140` raises `NotImplementedError`.
- **No frontend integration.** `optimum-intel`'s `OVModelForCausalLM` is unused by the project's own path. (It is used by `docs/runtimes/openvino-gpu.md`'s separate verification, but that path produces a different IR.)

---

## 4. Functional requirements — per component, per variant

For every spec in the M1 + M2 scope, this section states (a) the OV opset path, (b) the OV ecosystem feature that recovers the fast kernel, and (c) the acceptance test.

Conformance keywords (MUST / SHOULD / MAY / SHALL / SHALL NOT) follow RFC 2119.

### 4.1 Norm

| Spec | OV opset path | SDK feature | Acceptance |
|---|---|---|---|
| `NormStandardSpec` | M0 — `power(-0.5)` decomposition (`runtimes/openvino_port/norm.py:50`) | `RMSFusion` common-transformation MUST collapse the pattern into the GPU `RMS` kernel | parity vs torch fp32 (existing); plus runtime-op summary MUST show `RMS` node |
| `NormZeroCenteredSpec` | extend `build_rms_norm` with a final `Add(weight, 1.0)` *before* the last `Multiply(normed, weight)` | `RMSFusion` SHOULD still fire — the `Add(w,1)` is folded into the `w` Constant when `w` is a constant Parameter, but it is *not* a constant Parameter (it is a graph Parameter input). The pass MAY fall back to RMS-without-gamma; investigate. | parity vs torch_port (gate `mean_rel < 5e-4`); runtime-op summary SHOULD show `RMS` node. If RMSFusion does not fire, the requirement is satisfied by passing parity at fp16; the fusion is an optimisation. |

**Finding (M0 holdover):** RMSFusion does *not* fire in M0 because the final `Convert` op wraps the canonical pattern and breaks the matcher. Cited canonical pattern: `x · 1/√(ReduceMean(x²) + ε) · γ` — the matcher allows a trailing learnable-affine multiplication but stops at a downstream `Convert`. M1 MUST emit the `Convert(scaled, out_type)` *outside* the recognised pattern by having `build_rms_norm` accept the activation dtype and emit fp16 weights directly when `dtype=f16`, eliminating the inner fp32 island. This also resolves the M0 fp16-overflow note in `tests/parity/test_openvino_vs_torch.py:11-27` once the canonical-setup weights are rescaled.

Source: OV `RMSFusion` pattern — `src/common/transformations/src/transformations/common_optimizations/rms_fusion.cpp` (canonical pattern: `x · 1/√(ReduceMean(x²) + ε) · γ`).

#### Indicative builder skeleton for `build_norm`

```python
def build_norm(spec: NormSpec, x: ov.Output, weight: ov.Output) -> ov.Output:
    # Dispatch on union variant. Both variants share the rsqrt path.
    out_type = x.get_element_type()
    accum_t  = _ov_type(spec.accumulator_dtype)   # f32 by default

    x_a = opset.convert(x, accum_t)
    w_a = opset.convert(weight, accum_t)
    sq  = opset.multiply(x_a, x_a)
    mean = opset.reduce_mean(sq, opset.constant([spec.axis], Type.i32), keep_dims=True)
    rsqrt = opset.power(opset.add(mean, opset.constant(spec.eps, accum_t)),
                        opset.constant(-0.5, accum_t))
    normed = opset.multiply(x_a, rsqrt)

    if isinstance(spec, NormZeroCenteredSpec):
        # (1 + w) * x_normed
        one = opset.constant(1.0, accum_t)
        w_a = opset.add(w_a, one)

    scaled = opset.multiply(normed, w_a)
    # IMPORTANT: keep the trailing Convert OUTSIDE the matched pattern so the
    # RMSFusion matcher recognises the scaled-by-gamma form.
    return opset.convert(scaled, out_type) if out_type != accum_t else scaled
```

The branch on `NormZeroCenteredSpec` adds an `Add(w_a, 1.0)` step. The matcher in `rms_fusion.cpp` allows variations of the post-rsqrt multiplication; the `(1 + w)` form MAY require a small custom pattern extension upstream. If RMSFusion does not fire on the zero-centered form, we accept the loss; the GPU kernel for the unfused form is ~10–15% slower per the M0 measurements but still functionally correct.

### 4.2 Attention

#### 4.2.1 `MHASpec`

Same code path as `GQASpec` with `num_kv_heads == num_heads` and no `repeat_interleave` broadcast (the `n_rep == 1` short-circuit at `runtimes/openvino_port/attention.py:45` already covers this). The new builder MUST dispatch `MHASpec` through `build_gqa` without code duplication; an `isinstance(spec, (MHASpec, GQASpec))` guard at the block-lowerer level is acceptable.

Acceptance: parity test against torch_port MHA reference at the M0 canonical setup, `mean_rel < 5e-4` at fp16 (post-rescaling).

#### 4.2.2 `GQASpec` extensions (the M1 priority bundle)

The five extension fields in `components/attention.py` GQA M1-extension fields (`api_spec.md:347-355`) MUST be implemented in `runtimes/openvino_port/attention.py`. Each is a separate sub-requirement.

| Field | OV implementation | SDK feature | Acceptance |
|---|---|---|---|
| `qk_norm: Optional[NormSpec]` | apply `build_rms_norm` (or `build_norm_zero_centered`) to Q and K **after** the per-head reshape and **before** RoPE. Norm acts over the last (Dh) axis. | `RMSFusion` MAY fold each per-head norm separately | parity vs torch_port — `qk_norm` set to either `NormStandardSpec` or `NormZeroCenteredSpec`. |
| `output_gate: Literal["sigmoid"]` | after the post-SDPA `transpose+reshape`, multiply by `sigmoid(matmul(x, W_gate))` per head. New weight slot `W_gate` MUST be plumbed through `block_lowerer.py`'s `ws` dict. | none required — element-wise | parity vs torch_port Qwen3.5 / Qwen3.6 layer canonical setup. |
| `v_norm: Optional[NormSpec]` | apply norm to V after the per-head reshape, before SDPA. | `RMSFusion` | parity vs torch_port Gemma 4 canonical setup. |
| `qk_norm_fixed_scale: Optional[float]` | when set, the SDPA `scale` input MUST be `1.0` (the scale is absorbed into `qk_norm`'s output via a constant multiplier applied inside the norm or as a separate `Multiply`). | none | parity vs Gemma 4 (`define_gemma4_layer.py:62`). |
| `sliding_window: Optional[int]` | M1 (stateless): emit an additive mask Constant of shape `[1, 1, S, K]` with `-inf` outside the window, feed via `attention_mask` input to `opset13::SDPA`. M2 (stateful): use `PagedAttention`'s `sliding_window` slot. | `opset13::SDPA.attention_mask` slot | parity at the M0 canonical setup with a synthetic window length covering S; equality with a torch_port reference using the same mask. |
| `kv_source_layer_offset: int` (negative) | M2 only — requires KV cache plumbing. The builder MUST accept a per-graph cache identity table (`{layer_idx: variable_id}`) and, when `offset != 0`, MUST resolve K and V from the cache of `layer_idx + offset` instead of the current layer's Q/K projections. | OV State API | parity vs Hunyuan-style cross-layer attention reference. |
| `logit_softcap: float` | when `!= 0.0`, the SDPA emission MUST be decomposed: `q·k^T → multiply(scale) → tanh → multiply(logit_softcap) → softmax → matmul(V)`. The decomposed path SHALL NOT use `opset13::SDPA` because `SDPA` provides no softcap hook. | none — manual decomposition | parity vs torch_port soft-capped attention reference. |

**Special case — `qk_norm` interaction with the sandwich-norm block.** When `sandwich_block` is used and `attention.qk_norm` is set, the block-lowerer MUST insert the qk-norm inside the attention builder, not as a separate `Node`. The existing M0 `_find_rope_predecessor` pattern (`block_lowerer.py:43`) generalises: a `_find_attention_internal_norms` helper SHALL be added.

##### Indicative dispatch for the five extension fields

```python
def build_gqa(spec, x, w_q, w_k, w_v, w_o, *, rope_spec=None, position_ids=None,
              batch_size=1, seq_len=8,
              w_qk_norm=None, w_v_norm=None, w_output_gate=None, sinks=None):
    # ... (Q, K, V projections and head reshape; same as M0) ...

    if spec.qk_norm is not None:
        # apply over last (Dh) axis with axis=-1 inside _NormBase
        q = build_norm(spec.qk_norm, q, w_qk_norm["q"])
        k = build_norm(spec.qk_norm, k, w_qk_norm["k"])

    if rope_spec is not None:
        q, k = _apply_rope_dispatch(rope_spec, q, k, position_ids)

    if spec.v_norm is not None:
        v = build_norm(spec.v_norm, v, w_v_norm["v"])

    if spec.sliding_window is not None:
        attn_mask = _build_sliding_window_mask(seq_len, spec.sliding_window, x.get_element_type())
    else:
        attn_mask = None

    if spec.qk_norm_fixed_scale is not None:
        scale = opset.constant(1.0, x.get_element_type())   # absorbed into qk_norm
    else:
        scale = opset.constant(spec.scale, x.get_element_type())

    if spec.logit_softcap != 0.0:
        attn = _build_softcapped_attention(q, k, v, scale, spec.logit_softcap, attn_mask, spec.causal)
    else:
        # opset13 for vanilla GQA; opset14 only for GQASinksSpec (separate dispatch)
        attn = opset13.scaled_dot_product_attention(
            q, k, v, attention_mask=attn_mask, scale=scale, causal=bool(spec.causal),
        )

    # ... (transpose + reshape + W_o matmul; same as M0) ...
    out = _matmul_xWT(attn, w_o)

    if spec.output_gate == "sigmoid":
        gate = opset.sigmoid(_matmul_xWT(x, w_output_gate))
        out = opset.multiply(out, gate)

    return out
```

The new weight slots (`w_qk_norm["q"]`, `w_qk_norm["k"]`, `w_v_norm["v"]`, `w_output_gate`, `sinks`) MUST be plumbed through `block_lowerer.py`'s `ws = weight_shapes[node.name]` dictionary. The naming convention extends the M0 convention without breaking it: slots `qk_norm.q.weight`, `qk_norm.k.weight`, `v_norm.v.weight`, `w_output_gate`, `sinks`.

##### Sliding-window mask Constant

```python
def _build_sliding_window_mask(S: int, window: int, dtype: Type) -> ov.Output:
    # Lower-triangular allowed region within a window, additive -inf elsewhere.
    import numpy as np
    mask = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(S):
            if j > i or i - j >= window:
                mask[i, j] = -np.inf
    m = opset.constant(mask, Type.f32)
    if dtype != Type.f32:
        m = opset.convert(m, dtype)
    # broadcast to [1, 1, S, S]
    m = opset.unsqueeze(m, opset.constant([0, 1], Type.i32))
    return m
```

For dynamic S (Phase 3), the mask MUST be built via `Range` + `Less` + `Select`; the static implementation above is the M1 form.

#### 4.2.3 `MLASpec` (DeepSeek-style)

OV has no MLA-fused operator. The builder MUST emit MLA as decomposed dense attention:

1. Read latent `c_kv` and rope sub-token `k_pe` from the per-token cache (latent_dim + rope_head_dim wide).
2. Up-project to `[H, nope_head_dim]` per-head K and `[H, nope_head_dim]` per-head V via `W_ukv`.
3. Concatenate K with the broadcast `k_pe` to `[H, nope_head_dim + rope_head_dim]`.
4. Q path: either full-rank `W_q` or `W_dq → W_uq` low-rank decomposition based on `q_lora_rank is None` (`api_spec.md:493-496`).
5. SDPA on the assembled Q/K/V using the MLA-overridden `scale = 1/sqrt(nope_head_dim + rope_head_dim)` from `MLASpec.scale` (`api_spec.md:378-385`).

This is a **degraded path**: there is no equivalent to FlashMLA's absorbed-matmul optimisation, so MLA on OV will spill the full `H × (nope + rope)` K tensor through SDPA. The requirements document explicitly acknowledges this. M1 acceptance: numerical parity, not throughput parity.

Acceptance: parity vs torch_port MLA reference (a new fixture under `tests/parity/_canonical.py` is required), `mean_rel < 5e-4` at fp16 once weight scaling is in place.

#### 4.2.4 `GQASinksSpec` — the central M1 unblock

`opset13::SDPA` does not accept a sink input. `opset14::SDPA` exposes a `sink` slot. As Section 1 noted, the PyTorch frontend emits opset13 only, so `ov.convert_model` cannot lift `GQASinksSpec`. **Direct opset14 emission is the only path on this hardware.**

Source: `openvino.runtime.opset14.scaled_dot_product_attention` signature accepts `sink` as an optional Output (added 2025.4); see `docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.runtime.opset14.scaled_dot_product_attention.html`. The C++ class `ov::op::v13::ScaledDotProductAttention` already accepts a `sink` argument in some constructor overloads, but the canonical IR exposure is via opset14.

Builder requirements:

1. New file `runtimes/openvino_port/attention_sinks.py` or branch in `attention.py`.
2. Use `from openvino import opset14`; emit `opset14.scaled_dot_product_attention(q, k, v, attention_mask=mask, scale=scale, causal=True, sink=sinks_per_head)`.
3. `sinks` shape MUST be `[H_q]` (matches `api_spec.md:521`), broadcast across S × K by the op itself.
4. The `softmax_scale_log_base` field (`api_spec.md:398`) is informational under opset14 — the op handles the sink-logit denominator internally. If a future opset variant exposes the log base, the builder MUST wire it; M1 SHOULD log a warning when the spec's value differs from `2`.
5. The `sinks` weight MUST be plumbed through `block_lowerer.py` using the shared-name convention (`api_spec.md:524-527`): the `sinks` Parameter is both a graph input *and* the value feeding the op's `sink` slot.

Acceptance: parity vs an opset14-reference torch implementation of attention-with-sinks at the GPT-OSS canonical setup (16 heads, head_dim=64, sliding_window=128). M1 MUST also assert that the runtime-model op summary contains `SDPA` and that the fused-path naming has not regressed.

##### Indicative opset14 SDPA-with-sinks call

```python
from openvino import opset14

def build_gqa_sinks(spec: GQASinksSpec, x, w_q, w_k, w_v, w_o, sinks,
                    *, rope_spec=None, position_ids=None,
                    batch_size=1, seq_len=8):
    # Q, K, V projections + head reshape (identical to vanilla GQA)
    # ...
    if rope_spec is not None:
        q, k = _apply_rope_dispatch(rope_spec, q, k, position_ids)
    # GQA broadcast
    if H_kv < H_q:
        k = _repeat_interleave_axis1(k, H_q // H_kv, B, H_kv, S, Dh)
        v = _repeat_interleave_axis1(v, H_q // H_kv, B, H_kv, S, Dh)
    # Sliding window (additive mask)
    attn_mask = (_build_sliding_window_mask(S, spec.sliding_window, x.get_element_type())
                 if spec.sliding_window is not None else None)
    # opset14 SDPA with sink — the sink shape is [H_q] (per-head additive denominator term)
    attn = opset14.scaled_dot_product_attention(
        q, k, v,
        attention_mask=attn_mask,
        scale=opset14.constant(spec.scale, x.get_element_type()),
        causal=True,
        sink=sinks,        # [H_q] — direct passthrough from the weight slot
    )
    # transpose + reshape + W_o matmul (identical to vanilla GQA)
    # ...
```

The `softmax_scale_log_base` attribute (default `2`, `api_spec.md:398`) is informational under opset14; the op handles the sink-logit denominator internally. M1 SHOULD log a warning when `spec.softmax_scale_log_base != 2`.

**Memory caveat (from `docs/runtimes/openvino-gpu.md` §5):** The smallest published GPT-OSS variant (gpt-oss-20b at ~40 GB fp16) does not fit on Arc 140V's 16 GB iGPU. M1 acceptance therefore uses a **synthetic** sinks layer matching the GPT-OSS-20B per-layer shapes (`hidden_size=2880`, `H_q=64`, `H_kv=8`, `Dh=64`), not an end-to-end model.

### 4.3 Positional encoding

| Spec | OV implementation | SDK feature | Acceptance |
|---|---|---|---|
| `RoPESpec` full rotary | M0 (`rope.py:29`) | `RoPEFusionLlama` MUST fold to the GPU RoPE kernel; per PR #27907 the GPU OCL kernel is the post-fusion target | parity (existing); runtime-op summary SHOULD show `RoPE` |
| `RoPESpec` `partial_rotary_factor < 1.0`, `kind="prefix"` | split head_dim into `rot_dim = round(Dh * factor)` and `pass_dim = Dh - rot_dim`; build cos/sin tables of width `rot_dim`; apply rotation to the *first* `rot_dim` channels; concatenate with pass-through. | `RoPEFusionLlama` SHOULD still match — investigate; the prefix layout matches HuggingFace's `apply_rotary_pos_emb` shape | parity vs Qwen3.5 partial-rotary reference (`define_qwen3_5_layer.py:48`). |
| `RoPESpec` `partial_rotary_factor < 1.0`, `kind="proportional"` | Gemma 4 — rotated and pass-through dims are interleaved across head_dim. Requires per-channel gather/scatter rather than a clean prefix split. Builder MUST use `Gather`(rope_dims) and `Gather`(pass_dims) with precomputed index Constants. | fusion MAY NOT fire on the proportional layout; document and accept | parity vs Gemma 4 (`define_gemma4_layer.py:91`). |
| `MRoPEInterleavedSpec` | `VariadicSplit` `position_ids` along the axis dim into `len(sections)` parts, build per-section inv_freq tables, concat the resulting freqs, then `cos`/`sin` and `apply_rope_single`. | RoPEFusion's vision variant — investigate; the proportional pattern may need an issue filed | parity vs Qwen3.5 MRoPE reference (`define_qwen3_5_layer.py:47`, sections `(11,11,10)`). |
| `RoPESmoothScalingSpec` | piecewise scaling of inv_freq: `wavelen < low_freq_wavelen → inv_freq`; `> high_freq_wavelen → inv_freq / factor`; else smooth interpolation. Implementable as a precomputed inv_freq Constant (all parameters are `kw_only=True` constants per `api_spec.md:570-578`). | Constant fold; `RoPEFusionLlama` fires on the rotation pattern | parity vs torch_port Llama3-scaling reference. |
| `RoPEYaRNSpec` | precompute the YaRN scaling correction term offline; build inv_freq as a Constant. mscale absorbed into attention scale at builder boundary. | Constant fold | parity vs YaRN torch reference. |
| `RoPELongRoPESpec` | tuple-valued `short_factor` / `long_factor` (`api_spec.md:611-612`) — precompute both Constants; the builder picks short vs long at compile time based on `seq_len < original_context_length`. | Constant fold | parity vs Phi-3-128k LongRoPE reference. |
| `RoPELinearSpec` | linear inv_freq scaling — `inv_freq_scaled = inv_freq / factor`. Constant fold. | Constant fold | parity vs torch reference. |
| `NoPESpec` | pass-through. Builder MUST emit identity outputs and reject the case where a `position_ids` Parameter is wired but unused (`block_lowerer.py` MUST detect this and elide the position_ids parameter to avoid a dangling unused input). | none | smoke parity. |

Sources for fusion claims: OV `RoPEFusion` PR #27907 (GPU RoPE kernel optimization, ~50% perf improvement) and PR #28447 (RoPEFusion fix after SDPA→PagedAttention conversion).

#### RoPE scaling-family inv_freq formulas (constant-foldable)

| Variant | `inv_freq[i]` formula (precompute at builder time as `Constant`) |
|---|---|
| `RoPESpec` | `1 / (theta ^ (2i/Dh))` |
| `RoPELinearSpec` | `(1 / (theta ^ (2i/Dh))) / factor` |
| `RoPESmoothScalingSpec` | piecewise: `wavelen = 2π/inv_freq`; `< low_freq_wavelen → inv_freq` (no scaling); `> high_freq_wavelen → inv_freq / factor`; smooth `lerp` in between by `smoothing = (original_context_length/wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)` |
| `RoPEYaRNSpec` | ramp-based interpolation between linear and NTK scaling; the `mscale` correction MUST be applied to the attention `scale` (not folded into inv_freq) |
| `RoPELongRoPESpec` | precompute *both* `short_factor` and `long_factor` Constants; at compile time the builder picks one based on `seq_len < original_context_length`. Both arrays are length `Dh/2`. |

All five families collapse to a single `Constant` plus the standard `apply_rope_single` rotation when the builder is given `seq_len` at compile time (static-shape M1 baseline). For dynamic-shape (Phase 3) the LongRoPE branch needs a runtime `Select` between the two precomputed Constants.

#### `MRoPEInterleavedSpec` sketch

```python
def build_mrope(spec: MRoPEInterleavedSpec, q, k, position_ids):
    # position_ids: [A, B, S] where A = len(sections)
    sections = spec.sections             # e.g. (11, 11, 10) — sums to Dh/2 under full rotary
    # 1. Build per-axis inv_freq Constants
    inv_freqs = [
        opset.constant(_make_inv_freq(s, spec.theta), Type.f32) for s in sections
    ]
    # 2. Split position_ids on axis 0 by A pieces
    axes = opset.constant([0], Type.i32)
    splits = opset.variadic_split(position_ids, axes,
                                  opset.constant([1]*len(sections), Type.i64)).outputs()
    # 3. Per-axis freq = pos · inv_freq, then concat in head_dim order
    per_axis = []
    for pos_a, inv_a in zip(splits, inv_freqs):
        pos_f = opset.convert(opset.squeeze(pos_a, axes), Type.f32)  # [B, S]
        per_axis.append(opset.multiply(opset.unsqueeze(pos_f, [-1]),
                                       opset.unsqueeze(inv_a, [0, 1])))
    freqs = opset.concat(per_axis, axis=-1)                          # [B, S, Dh/2]
    emb = opset.concat([freqs, freqs], axis=-1)                       # [B, S, Dh]
    cos, sin = opset.cos(emb), opset.sin(emb)
    if spec.partial_rotary_factor < 1.0:
        return _apply_partial_rope(q, k, cos, sin, spec)
    return apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)
```

### 4.4 FFN

| Spec | OV implementation | SDK feature | Acceptance |
|---|---|---|---|
| `SwiGLUSpec` non-fused | M0 (`ffn.py:30`) | `GLU` op (renamed from `SwiGLU` in PR #27683) MAY fold the silu+multiply pair | parity (existing). |
| `SwiGLUSpec` `fused_gate_up=True` | new branch: single `MatMul(x, W_gate_up)` of width `2 * intermediate_size`, then `VariadicSplit` into gate and up halves on the last axis. | `GLU` may still match the gate/up split | parity vs torch_port fused reference (Phi-style). |
| `GeGLUSpec` `activation="gelu_tanh"` | replace `_silu` with `opset7::gelu(x, mode="tanh")`. | `GLU` op detects gelu variants per PR #27683 ("can also be GeGLU") | parity vs Gemma 4 reference (`define_gemma4_layer.py:94`). |
| `GeGLUSpec` `activation="gelu_exact"` | `opset7::gelu(x, mode="erf")`. | same | parity vs reference. |
| `ClampedSwiGLUSpec` | pre-activation clamp on gate and up: `opset1::clamp(gate, -clamp, +clamp)` and `opset1::clamp(up, -clamp, +clamp)`; Swish-beta on gate: `gate * sigmoid(alpha * gate)` (note: `alpha` multiplies the sigmoid argument, not the gate itself). Then `multiply(swish_gate, clamped_up)` and the `W_down` matmul. | unlikely to fuse — the clamp + sigmoid(alpha·x) pattern is not a `GLU` op variant. Document this and accept the decomposed path. | parity vs GPT-OSS reference (`define_gpt_oss_layer.py:55-58`, `clamp=7.0, alpha=1.702`). |

#### Indicative `build_clamped_swiglu`

```python
def build_clamped_swiglu(spec: ClampedSwiGLUSpec, x, w_gate, w_up, w_down):
    g = _matmul_xWT(x, w_gate)
    u = _matmul_xWT(x, w_up)
    clamp_val = float(spec.clamp)
    g = opset.clamp(g, -clamp_val, clamp_val)
    u = opset.clamp(u, -clamp_val, clamp_val)
    alpha = opset.constant(spec.alpha, x.get_element_type())
    swish_g = opset.multiply(g, opset.sigmoid(opset.multiply(g, alpha)))
    gated = opset.multiply(swish_g, u)
    return _matmul_xWT(gated, w_down)
```

### 4.5 KV cache (M2)

`KVCacheSpec` is **not** in `components/` today (the package's `Status` section calls it out as M1 backlog — `components/README.md:316-318`). The component MUST be added in `components/kv_cache.py` (a separate spec PR), then ported here. The OV-port-side requirement is:

| Variant | OV opset path | SDK feature | Acceptance |
|---|---|---|---|
| `DenseKVSpec` | a pair `(ReadValue, Assign)` per `(layer, head_group)` over a fixed-shape Variable identity `f"layer_{i}.kv"`. The Variable element type SHALL be controllable via the `KV_CACHE_PRECISION` plugin property (default `f16`). | `opset3::ReadValue`, `opset3::Assign`, the State API, and `MakeStateful` transformation (when assembling end-to-end models from individually-built blocks) | parity vs torch_port multi-step decode reference; throughput ≥30 tok/s on Qwen3-0.6B. |
| `LatentKVSpec` (MLA) | identical Variable primitives, smaller width (`[B, S_cache, latent_dim + rope_head_dim]` per `api_spec.md:479`). MLA does not benefit from the OV-native compute fusion that dense KV gets; the requirement is functional only. | State API | parity vs torch_port MLA decode reference. |
| `SlidingRingSpec` | M2 option A: static window mask Constant of shape `[1, 1, S, K]` (`-inf` outside the window) and a non-ringed Variable. Option B: route through `PagedAttention.sliding_window` slot (PR #29248 added per-sequence sliding-window support; see also the score aggregation work in PR #30174). | PagedAttention | parity vs torch_port reference. M1 SHOULD adopt Option A (simpler); M2 SHOULD migrate to Option B once GenAI integration lands. |
| `SSMState` | `CausalConv1d` + `GatedDeltaNet` — OV 2025.4+ added a fused GatedDeltaNet GPU kernel pattern-matched via `optimum-intel` (per `docs/runtimes/openvino-gpu.md` §3). Direct emission is not currently feasible without HETERO carve-out; document as deferred. | not supported on M2 hardware footprint | deferred — Section 10. |

KV-cache precision policy: when the spec carries an explicit `kv_cache_precision: Optional[DType]` field, the builder MUST set the `KV_CACHE_PRECISION` plugin property accordingly. Default `f16` matches the M0 fp16 path.

#### Indicative `DenseKVSpec` builder

```python
def build_dense_kv(spec: DenseKVSpec, k_new, v_new, layer_idx: int,
                   *, batch_size, seq_len, num_kv_heads, head_dim, max_seq_len):
    from openvino import opset3
    # Variable identities — one per (layer, K or V). Used by MakeStateful pass.
    var_k_id = f"layer_{layer_idx}.k_cache"
    var_v_id = f"layer_{layer_idx}.v_cache"
    cache_shape = PartialShape([batch_size, num_kv_heads, max_seq_len, head_dim])

    # ReadValue with explicit initial value (zeros of correct shape).
    zero_k = opset.constant(np.zeros(cache_shape.to_shape(), dtype=np.float16))
    zero_v = opset.constant(np.zeros(cache_shape.to_shape(), dtype=np.float16))
    k_cache = opset3.read_value(zero_k, variable_id=var_k_id)
    v_cache = opset3.read_value(zero_v, variable_id=var_v_id)

    # Concat the new K/V onto the cache axis 2 (S axis).
    k_full = opset.concat([k_cache, k_new], axis=2)
    v_full = opset.concat([v_cache, v_new], axis=2)

    # Write back via Assign.
    opset3.assign(k_full, variable_id=var_k_id)
    opset3.assign(v_full, variable_id=var_v_id)

    return k_full, v_full
```

When assembling a multi-block model from individual blocks, the new thread MAY skip explicit `ReadValue`/`Assign` emission and instead use the `MakeStateful` transformation pass on the assembled `ov.Model` — mapping `(layer_{i}.k_in Parameter, layer_{i}.k_out Result)` pairs to a Variable identity. Both paths produce equivalent IR.

### 4.6 MoE block (M2)

`MoEBlock` is not in `components/` today. The OV port MUST plan for three sub-variants:

| Variant | OV implementation | Acceptance |
|---|---|---|
| `TopK-Softmax` | `TopK` (opset3) → `Softmax` → per-expert `MatMul` × `num_experts_per_tok` → `Scatter` aggregation. M2 MUST decompose, not rely on a fused MoE op. | parity vs torch_port reference (Qwen3-MoE shapes). |
| `TopK-Sigmoid` | Identical except the routing weight uses `Sigmoid` instead of `Softmax`. GPT-OSS uses this. | parity vs GPT-OSS reference. |
| `TopK-Softmax-SharedExperts` | Same as `TopK-Softmax` plus an always-on dense FFN path summed at the output. | parity vs DeepSeek-V2 reference. |

A fused `ov.MoE` extension op is an **open question** (Section 10). M2 MUST start with decomposition and benchmark; promotion to a custom OV extension op is a Phase 3 decision.

### 4.7 LM head and embedding (M2)

| Variant | OV implementation | NNCF | Acceptance |
|---|---|---|---|
| `LMHead Linear` | `MatMul(hidden, W_lm_head.T)`. INT4 weight via NNCF (`compress_weights(model, mode=INT4_ASYM, group_size=128, ratio=1.0)`). Tied-embedding case: see Section 10. | NNCF INT4_ASYM | parity vs torch_port logits at fp16, mean_rel < 5e-4. |
| `LMHead Linear-Softcap` | `MatMul → Tanh → Multiply(softcap_scalar)`. | NNCF INT4_ASYM on the MatMul | parity vs Gemma 2/3-style softcap reference. |
| `LMHead Linear-Scaled` | `MatMul → Multiply(scale_scalar)`. Constant-folds the scalar into the LM-head weight at compile time. | NNCF INT4_ASYM | parity vs reference. |
| `TokenEmbeddingSpec` | `opset8::Gather(W_embed, token_ids, axis=0)`. INT4 weight via NNCF. | NNCF INT4_ASYM (or `INT8_ASYM` for embedding rows where the accuracy degradation is too high — make the mode configurable) | parity vs torch_port. |

### 4.8 Block topologies

| Topology | OV-port status | Notes |
|---|---|---|
| `pre_norm_block` | M0 ✓ (`block_lowerer.py` already handles it) | gold path |
| `post_norm_reordered_block` | M1 — block_lowerer learns a new `isinstance(spec, NormStandardSpec)` branch reached *after* `AttentionSpec`. No new builder; only graph-walking. | parity vs torch_port reference. |
| `sandwich_block` | M1 — four norms in a single block. Block-lowerer's RoPE-predecessor finder MUST cope with the rope node sitting between `n_pre_a` and `attn` (already does). The `qk_norm` interaction (Section 4.2.2) MUST be tested here specifically. | parity vs Gemma 4 reference. |
| `parallel_block` | M1 — three-way `Add(input, attn, ffn)`; `build_add` already supports N-way (`misc.py:8`). | parity vs Falcon-7B / Cohere reference. |

---

## 5. Non-functional requirements

### 5.1 Precision

- **fp16** is the primary path. Activations Parameters, weights, all intermediate ops MUST be fp16 once weight scaling is canonicalised.
- **fp32** is the reference and the M0 parity gate.
- **INT4 weight-only** (M2) via NNCF; activations stay fp16.
- The reduction accumulator inside RMSNorm MUST run at fp32 even on the fp16 path. `_NormBase.accumulator_dtype` (`api_spec.md:249`) carries this and the builder MUST honour it.

### 5.2 Performance baseline (Arc 140V Xe2 iGPU)

| Workload | Target |
|---|---|
| Qwen3-0.6B single-block fp16 forward | ≥ 50× faster than the M0 fp32 path on the same setup |
| Qwen3-0.6B full model (M2, INT4 + stateful KV) | ≥ 30 tok/s greedy decode at S=1024 prompt; ≥ 50 tok/s with continuous batching at batch=4 |
| First-token p50 latency (M2) | < 200 ms for 1024-token prompt on Qwen3-0.6B |
| Block compile time | < 60 s on a warm OpenCL kernel cache (`ov_cache/`) |
| End-to-end model compile time | < 5 min for Qwen3-0.6B fp16 INT4 |

`docs/runtimes/openvino-gpu.md` §3 records 49.7 tok/s for Qwen3-0.6B via `optimum-intel` as the practical ceiling — the M2 ≥30 tok/s target is conservative on the direct-emission path.

### 5.3 Memory

- Block working set MUST fit in 16 GB shared (Arc 140V).
- Per-block weights MUST be released between blocks during compile (the per-block `ov.Model` is discarded as soon as `core.compile_model` returns).
- INT4 weight compression target: ≥4× reduction vs fp16 for any Linear weight (NNCF INT4_ASYM `group_size=128` achieves this).

### 5.4 Numerical parity

- M1 gate: `mean_rel < 5e-4` at fp16 vs torch_port fp32 reference. This matches the existing M0 gate at `tests/parity/test_openvino_vs_torch.py:50` adapted from fp32 to fp16.
- INT4 gate (M2): `mean_rel < 5e-3` for end-of-block hidden state, and Top-1 logit equality on a 256-token greedy decode against the torch_port reference.
- `max_rel` SHALL be reported but NOT gated (per the M0 precedent, `tests/parity/test_openvino_vs_torch.py:172-176`).

### 5.5 Compilation and caching

- `ov_cache/` MUST be enabled (already in `.gitignore` per `docs/runtimes/openvino-gpu.md` §6). Cold compile vs warm compile MUST be reported in the test logs.
- The compiler property `CACHE_DIR` MUST be set to `ov_cache/` by default.

### 5.6 Determinism

- For a fixed canonical setup, fp32 GPU runs MUST be bit-stable across invocations within the same compiled model. fp16 runs MAY drift in low bits due to MatMul tile-order; the `mean_rel` gate accommodates this.

---

## 6. SDK / tool leverage strategy

| Tool | Role in this port | Specific integration points |
|---|---|---|
| **OpenVINO opset16** (default for most ops) | Direct Node emission for everything except SDPA-with-sinks | every `runtimes/openvino_port/*.py` builder; `from openvino import opset16 as opset` |
| **OpenVINO opset13** | SDPA (current M0) and Variables (`ReadValue`/`Assign`) — opset13 is the highest one that already supports the sink overload at the C++ level via constructor, but Python exposure is via opset14 | `runtimes/openvino_port/attention.py:21`; KV-cache builder |
| **OpenVINO opset14** | `opset14::scaled_dot_product_attention` with `sink` input for `GQASinksSpec` | new `runtimes/openvino_port/attention_sinks.py` |
| **OpenVINO transformation passes** | Run automatically inside `core.compile_model()`; we shape our IR to match | `RMSFusion` (norm), `RoPEFusionLlama` (rope), `GLU` op (ffn), `MakeStateful` (KV) |
| **NNCF** (`pip install nncf`) | Post-training weight-only quantization | `runtimes/openvino_port/quantization.py` (NEW); calls `nncf.compress_weights(model, mode=CompressWeightsMode.INT4_ASYM, group_size=128, ratio=1.0)` over the assembled `ov.Model` before `compile_model` |
| **MLIR (internal to OV)** | OV 2026 has begun adopting MLIR for the NPU compiler (per OV release notes); the GPU plugin keeps its OCL-kernel path. We do NOT author MLIR IR. | Section 10 open question: should our project emit MLIR or only `ov.Model`? |
| **OpenVINO GenAI** (`openvino_genai`) | High-level continuous-batching pipeline wrapping our compiled IR | M2 — wrap our assembled `ov.Model` (saved to disk under `ov_models/<model-id>/`) via `ov_genai.LLMPipeline("path", "GPU", scheduler_config=SchedulerConfig(...))`. The cached IR is directly consumable (per `docs/runtimes/openvino-gpu.md` §7 "Throughput room to grow"). |
| **PagedAttention** | Continuous-batching KV cache management with sliding-window and sinks support | M2 — the GenAI scheduler-config path uses PagedAttention internally. We MUST verify the conversion from our `opset14::SDPA` emission to `PagedAttention` survives `MakeStateful` + the SDPA→PagedAttention pass (PR #28447 fixed a RoPEFusion-after-PagedAttention regression in this chain; this is the canonical pattern). |
| **optimum-intel** (`OVModelForCausalLM`) | Stateful KV wiring via `MakeStateful` for HuggingFace models | M2 reference / packaging layer ONLY. We do NOT use `OVModelForCausalLM` for the project's own builder path; it is the comparison point. |
| **Plugin properties** | Runtime knobs | `PERFORMANCE_HINT="LATENCY"` (M0 default; `block_lowerer.py` consumer at `examples/v2/03_run_openvino_gpu.py:98`), `INFERENCE_PRECISION_HINT="f16"` (M1 default), `KV_CACHE_PRECISION="f16"` (M2 default), `EXECUTION_MODE="ACCURACY"` (regression-only fallback), `CACHE_DIR="ov_cache"` |
| **Custom OV extension ops** | Last resort | M2 — possible candidates: a single fused `clamped_swiglu` op; a `mrope_apply` op for the proportional partial-rotary layout. Both are deferred until decomposed paths are profiled. |

### 6.1 NNCF integration sketch

```python
# runtimes/openvino_port/quantization.py
import nncf
from nncf import CompressWeightsMode
import openvino as ov

def compress_weights_int4(
    model: ov.Model,
    *,
    group_size: int = 128,
    ratio: float = 1.0,
    mode: CompressWeightsMode = CompressWeightsMode.INT4_ASYM,
    dataset=None,                  # nncf.Dataset; None = data-free
    awq: bool = False,
    scale_estimation: bool = False,
    gptq: bool = False,
) -> ov.Model:
    """Apply NNCF weight compression to every MatMul in the model.

    Data-aware paths (AWQ / Scale-Estimation / GPTQ) require a calibration
    dataset; data-free path lands the same INT4 with slightly worse accuracy.
    Default: data-free INT4_ASYM, group_size=128, ratio=1.0 — matches the
    common-case used in `optimum-intel` for sub-3B models.
    """
    return nncf.compress_weights(
        model,
        mode=mode,
        group_size=group_size,
        ratio=ratio,
        dataset=dataset,
        awq=awq,
        scale_estimation=scale_estimation,
        gptq=gptq,
    )
```

The data-aware paths (AWQ / Scale-Estimation / GPTQ — confirmed against NNCF's `Usage.md` source) MAY combine; for production accuracy NNCF docs recommend `AWQ + Scale-Estimation` for INT4_ASYM. M2 SHOULD ship the data-free path first and add a calibration-dataset flag later. Mode coverage from NNCF Usage.md: `INT8_ASYM`, `INT8_SYM`, `INT4_SYM`, `INT4_ASYM`, `NF4`, `CODEBOOK`, `CB4`, `MXFP4`, `MXFP8_E4M3`, `FP8_E4M3`, `FP4`, `NVFP4`.

### 6.2 GenAI integration sketch

```python
# runtimes/openvino_port/genai_wrapper.py
import openvino_genai as ov_genai
from openvino_genai import SchedulerConfig

def build_pipeline(model_path: str, *, batch_size: int = 1):
    """Wrap a saved ov.Model under model_path/ as an LLMPipeline.

    The saved IR MUST already contain stateful KV (ReadValue/Assign) for the
    PagedAttention backend to be selected.
    """
    scheduler = SchedulerConfig()
    scheduler.max_num_batched_tokens = 256
    scheduler.cache_size = 2          # GB of KV-cache pages
    scheduler.dynamic_split_fuse = True
    return ov_genai.LLMPipeline(
        model_path,
        device="GPU",
        scheduler_config=scheduler,
        # PERFORMANCE_HINT and INFERENCE_PRECISION_HINT flow through.
    )
```

GenAI's `LLMPipeline` inspects device + properties at construction and chooses among three backends: stateful, NPU, or PagedAttention. Passing `scheduler_config` MUST select the PagedAttention backend (per the GenAI source `llm_pipeline.hpp`). The Variable identities our `kv_cache.py` builder writes MUST line up with the names PagedAttention expects after `MakeStateful` runs.

### 6.3 Where MLIR fits (or doesn't)

OpenVINO's GPU plugin compiles `ov.Model` directly via its own OCL kernel selector, not MLIR. MLIR enters OV in 2026.0 as a **preview** compiler-in-plugin for the NPU (per OV 2026.0 release notes; preferred in 2026.1). The Arc 140V iGPU is *not* the NPU. **Therefore: this project SHALL NOT emit MLIR IR.** The MLIR question reduces to "do we author custom fusion-pass patterns at the OV transformations layer?" — answered no for M1, deferred for M2 Section 10.

The user's WinML CLI BYOM context (the consumer of this work) has its own MLIR `winml` dialect. The handoff between this Python port and the WinML CLI is via the `ov.Model` artefact serialized to XML+BIN; no MLIR exchange happens.

### 6.4 Plugin-property matrix

| Property | M0 | M1 default | M2 default | Notes |
|---|---|---|---|---|
| `PERFORMANCE_HINT` | `LATENCY` | `LATENCY` | `LATENCY` for single-stream, `THROUGHPUT` for continuous batching | `examples/v2/03_run_openvino_gpu.py:98` |
| `INFERENCE_PRECISION_HINT` | `f32` | `f16` | `f16` | M1 switch is **blocking** on weight rescaling (`tests/parity/_canonical.py`) |
| `KV_CACHE_PRECISION` | n/a | n/a | `f16` (or `u8` to halve cache memory at ~0.5pp accuracy cost) | only meaningful once Variables exist |
| `EXECUTION_MODE` | unset | unset | `PERFORMANCE`; `ACCURACY` for regression fallback | per OV 2026 GPU device docs |
| `CACHE_DIR` | unset | `ov_cache` | `ov_cache` | already gitignored; M1 SHOULD enable to cut cold-compile to ~1.4s (per `docs/runtimes/openvino-gpu.md` §2) |
| `GPU_HOST_TASK_PRIORITY` | unset | unset | unset | only relevant for shared-GPU contention |

---

## 7. Architecture changes from M0

The work is **incremental on the M0 file layout**. Specifically:

### 7.1 Files that grow

| File | M0 LoC | M1 expected LoC | Change |
|---|---|---|---|
| `runtimes/openvino_port/norm.py` | 57 | ~120 | add `build_norm_zero_centered` and a `build_norm(spec, x, weight)` dispatcher |
| `runtimes/openvino_port/attention.py` | 156 | ~600 | add MHA path, the five GQA extension branches, MLA dense decomposition, logit_softcap decomposition |
| `runtimes/openvino_port/rope.py` | 87 | ~400 | partial-rotary (both kinds), MRoPE-Interleaved, scaling families, NoPE |
| `runtimes/openvino_port/ffn.py` | 45 | ~200 | fused-gate-up, GeGLU, ClampedSwiGLU |
| `runtimes/openvino_port/block_lowerer.py` | 146 | ~350 | extend `isinstance` chain; add the four new topologies; add the qk_norm-as-attention-internal interaction; opset14 dispatch for GQASinksSpec |
| `runtimes/openvino_port/__init__.py` | 23 | ~50 | new re-exports |

### 7.2 New files

| File | Role |
|---|---|
| `runtimes/openvino_port/attention_sinks.py` | opset14 SDPA-with-sinks for `GQASinksSpec` (NEW) |
| `runtimes/openvino_port/attention_mla.py` | MLA decomposed path (NEW); kept separate from `attention.py` because the weight layout diverges (`api_spec.md:485-496`) |
| `runtimes/openvino_port/kv_cache.py` | `DenseKVSpec` / `LatentKVSpec` / `SlidingRingSpec` builders (M2) |
| `runtimes/openvino_port/moe_block.py` | `MoEBlock` decomposed builders (M2) |
| `runtimes/openvino_port/lm_head.py` | `LMHead` + softcap + scaled variants (M2) |
| `runtimes/openvino_port/embedding.py` | `TokenEmbeddingSpec` via `opset8::Gather` (M2) |
| `runtimes/openvino_port/quantization.py` | NNCF integration: `compress_weights_int4(model, ...)` (M2) |
| `runtimes/openvino_port/genai_wrapper.py` | `openvino_genai.LLMPipeline` wrapping (M2) |

### 7.3 Things that MUST NOT change

- M0 builders' public signatures (`build_rms_norm(spec, x, weight) -> ov.Output`, etc.). Existing parity tests depend on these.
- The graph Parameter naming convention `f"{node_name}.{slot}"` (`block_lowerer.py:111-115`). The torch_port golden and existing tests depend on it.
- The `_find_rope_predecessor` pattern. Extended, not replaced.

### 7.4 Things that SHOULD change

- `_DEFAULT_DTYPE = Type.f16` (`block_lowerer.py:40`) becomes the actual default after weight rescaling lands.
- The hard-coded `(1, 8, 2048)` input shape in `block_lowerer.py:60` becomes optional; default SHOULD come from the spec.
- A `compile_for_gpu(model, *, inference_precision="f16", kv_cache_precision=None, cache_dir="ov_cache")` convenience function SHOULD be added to `__init__.py` to centralize property setup.

---

## 8. Test contract

### 8.1 New parity tests (M1)

Each test under `tests/parity/test_v3_openvino_<variant>.py`. One test file per variant family:

| File | Covers | Gate |
|---|---|---|
| `test_v3_openvino_norm_zero_centered.py` | `NormZeroCenteredSpec` | `mean_rel < 5e-4` at fp16 |
| `test_v3_openvino_gqa_qk_norm.py` | `GQASpec.qk_norm` (both `NormStandard` and `NormZeroCentered`) | same |
| `test_v3_openvino_gqa_output_gate.py` | `GQASpec.output_gate="sigmoid"` | same |
| `test_v3_openvino_gqa_v_norm.py` | `GQASpec.v_norm` | same |
| `test_v3_openvino_gqa_fixed_scale.py` | `GQASpec.qk_norm_fixed_scale` (Gemma 4) | same |
| `test_v3_openvino_gqa_sliding_window.py` | `GQASpec.sliding_window` | same |
| `test_v3_openvino_gqa_sinks.py` | `GQASinksSpec` via opset14 | same |
| `test_v3_openvino_mla.py` | `MLASpec` decomposed | `mean_rel < 1e-3` (degraded path tolerance) |
| `test_v3_openvino_rope_partial.py` | `RoPESpec.partial_rotary_factor < 1.0` both kinds | same |
| `test_v3_openvino_mrope.py` | `MRoPEInterleavedSpec` | same |
| `test_v3_openvino_rope_scaling.py` | Smooth / YaRN / LongRoPE / Linear | same |
| `test_v3_openvino_ffn_geglu.py` | `GeGLUSpec` | same |
| `test_v3_openvino_ffn_clamped.py` | `ClampedSwiGLUSpec` | same |
| `test_v3_openvino_sandwich.py` | `sandwich_block` end-to-end (Gemma 4 layer) | same |
| `test_v3_openvino_post_norm.py` | `post_norm_reordered_block` (OLMo-2 layer) | same |
| `test_v3_openvino_parallel.py` | `parallel_block` (Falcon-7B layer) | same |

Each test MUST:

1. Assert `EXECUTION_DEVICES = ['GPU.0']`. No silent CPU fallback.
2. Report cold and warm compile times to stdout (matches the M0 precedent at `tests/parity/test_openvino_vs_torch.py:181-193`).
3. Dump the runtime-model op summary (via `_runtime_op_summary` helper) for fusion-pass visibility.
4. Use a synthetic canonical setup (rescaled to fit fp16) — a new helper `tests/parity/_canonical_v3.py` SHOULD parallel the existing `_canonical.py`.

### 8.2 Quantization tests (M2)

| File | Covers | Gate |
|---|---|---|
| `test_v3_openvino_int4_linear.py` | NNCF INT4_ASYM on a single MatMul layer | parity degradation ≤ 1.5× the fp16 baseline; weight-byte ratio ≥ 4× |
| `test_v3_openvino_int4_block.py` | INT4 over a full block (all Linears) | `mean_rel < 5e-3`; Top-1 logit equality on a 64-token greedy decode |

### 8.3 KV-cache tests (M2)

| File | Covers | Gate |
|---|---|---|
| `test_v3_openvino_stateful_kv.py` | `DenseKVSpec` multi-step decode | parity vs torch_port multi-step; throughput ≥30 tok/s |
| `test_v3_openvino_sliding_ring.py` | `SlidingRingSpec` | parity vs windowed torch reference |

### 8.4 GenAI integration tests (M2)

| File | Covers | Gate |
|---|---|---|
| `test_v3_openvino_genai_pipeline.py` | `openvino_genai.LLMPipeline` wrapping a built-from-spec model | generation matches `optimum-intel` reference on 30 tokens greedy; throughput ≥30 tok/s |
| `test_v3_openvino_continuous_batching.py` | `SchedulerConfig` with batch=4 | throughput ≥50 tok/s aggregate |

### 8.5 Golden regeneration

When `tests/parity/_canonical.py` changes (e.g. weight rescaling for fp16), `tests/parity/_save_golden.py` MUST be re-run before any port-side test runs. The new thread MUST document this in a `tests/parity/README.md` update.

---

## 9. Phasing / milestones

### 9.1 Phase 1 — family-defining specs (M1, **priority for the new thread**)

Scope:

1. `NormZeroCenteredSpec` (`norm.py` extension)
2. `GQASpec` with `qk_norm`, `output_gate="sigmoid"`, `v_norm`, `qk_norm_fixed_scale`, `sliding_window` (`attention.py` extension)
3. `GQASinksSpec` via opset14 (new `attention_sinks.py`)
4. `MRoPEInterleavedSpec` (`rope.py` extension)
5. `MLASpec` decomposed dense path (new `attention_mla.py`)
6. `RoPESpec` `partial_rotary_factor < 1.0`, both kinds (`rope.py` extension)
7. `GeGLUSpec` and `ClampedSwiGLUSpec` (`ffn.py` extension)
8. `sandwich_block` and `post_norm_reordered_block` and `parallel_block` (`block_lowerer.py` extension)
9. RMSFusion-light-up fix (move `Convert` outside the matched pattern; rescale canonical-setup weights to fit fp16)

Acceptance:

- One full-layer parity test passes at fp16 for **each of**: Qwen3-0.6B (M0 already covers), Qwen3.5-35B-A3B "full-attention" layer (`define_qwen3_5_layer.py`), Gemma 4 E2B "local" layer (`define_gemma4_layer.py`), GPT-OSS-20B sliding-attention layer (`define_gpt_oss_layer.py`).
- Every parity test MUST satisfy `mean_rel < 5e-4` and `EXECUTION_DEVICES == ['GPU.0']`.

LoC estimate: **~1500–2000 lines net add** (builders + tests + helpers).

### 9.2 Phase 2 — quantization and stateful KV (M2 part A)

Scope:

1. NNCF INT4 / INT8 weight-only compression integration (`quantization.py`).
2. `KVCacheSpec` family added to `components/` (out of OV-port scope; precondition).
3. `DenseKVSpec` builder (`kv_cache.py`) — `ReadValue` / `Assign` per layer-head-group; `MakeStateful` integration for end-to-end models.
4. `KV_CACHE_PRECISION` plugin-property plumbing.
5. `LatentKVSpec` builder for MLA (degraded but functional).
6. `SlidingRingSpec` via either static mask (Option A) or PagedAttention (Option B per Section 4.5).

Acceptance:

- Full decoder layer with INT4 weights + stateful KV cache runs on Arc 140V at ≥50 tok/s for Qwen3-0.6B (matches the `optimum-intel` ceiling).
- INT4 parity gate `mean_rel < 5e-3` and Top-1 logit equality on 64-token greedy decode.

LoC estimate: ~1000–1500 lines net add.

### 9.3 Phase 3 — end-to-end inference + GenAI integration (M2 part B)

Scope:

1. Multi-block model assembly — a `build_model(blocks: tuple[BlockGraph, ...], embedding, lm_head) -> ov.Model` function.
2. `Embedding` and `LMHead` builders (`embedding.py`, `lm_head.py`).
3. `MoEBlock` builders for the three variants (`moe_block.py`).
4. `openvino_genai.LLMPipeline` wrapper (`genai_wrapper.py`).
5. Continuous batching with `SchedulerConfig`; PagedAttention `sliding_window` slot adoption.
6. `optimum-intel` comparison harness — assert our IR-derived pipeline matches `OVModelForCausalLM` greedy output token-for-token at temperature 0.

Acceptance:

- Full Qwen3-0.6B model running end-to-end on Arc 140V via GenAI pipeline.
- Throughput ≥30 tok/s greedy decode; ≥50 tok/s aggregate at batch=4 continuous-batching.

LoC estimate: ~1500–2000 lines net add.

---

## 10. Open questions / decisions to make

The new thread MUST resolve (or escalate to the user) before deep implementation:

1. **Static vs dynamic shape.** M0 is static. M1 SHOULD remain static for parity speed; M2 SHOULD evaluate `PartialShape([B?, S?, D])` for the batched inference path. **Decision: static for M1 + Phase 2; dynamic-batch for Phase 3.** Confirm with the user.
2. **opset13 vs opset14 SDPA unconditionally.** M0 emits opset13. If `GQASinksSpec` requires opset14, the simplest path is to switch every attention emission to opset14 (sinks defaults to `None`). **Decision: emit opset13 by default; opset14 only when `isinstance(spec, GQASinksSpec)`.** Rationale: opset13 has more mature GPU-kernel selection paths; opset14 is recent.
3. **MoE: fused custom op vs decomposed.** Decomposed (Section 4.6) is portable and lands quickly; a single fused custom op (Python `Extension`) is faster but couples to the OV version. **Decision: decomposed in M2; revisit after profiling.**
4. **NNCF integration point.** Two options: (a) call `nncf.compress_weights` on the assembled `ov.Model` (single shot, simple); (b) per-Linear via a builder-level decorator (finer control, more code). **Decision: option (a) for M2 Phase 2; option (b) deferred until per-layer mixed-precision is needed.**
5. **Custom OV extension op for Clamped-SwiGLU.** The decomposed path (Section 4.4) likely runs slower than a single op. **Decision: ship decomposed; profile; if Phase 3 shows >20% overhead vs `GLU`-fused SwiGLU, file a custom-op work item.**
6. **MLIR emission.** Section 6.1's answer: no MLIR emission. **Confirm with the user** — the WinML BYOM context (`components/README.md:343`) may want this differently.
7. **Tied-embedding handling.** `docs/runtimes/openvino-gpu.md` §7 flags that the HF→OV exporter doesn't tie `lm_head.weight` with `model.embed_tokens.weight`. Our `LMHead` builder MUST decide: re-export the same Variable, or share a Constant. **Decision: share a Constant via a `tie_embeddings: bool` flag on `LMHead` / `TokenEmbedding` spec.**
8. **Weight rescaling for fp16.** The M0 canonical-setup unscaled Normal weights overflow fp16 (`tests/parity/test_openvino_vs_torch.py:14-18`). The new thread MUST update `tests/parity/_canonical.py` to use a scaled distribution (e.g. `Normal(0, 1/sqrt(d_model))`) so the fp16 gate is meaningful. This is **blocking** for the M1 fp16-default switch.
9. **`logit_softcap` reach.** Gemma 2/3 attention uses this; Gemma 4 does not. Decide whether Phase 1 includes this. **Recommendation: include it — it is one decomposed branch.**
10. **`kv_source_layer_offset` (cross-layer cache).** Requires the KVCache spec to exist. **Decision: defer to Phase 2.**
11. **Block compile time vs whole-model compile time.** Each block compiled independently is convenient for tests but does not enable model-level fusions. **Decision: keep per-block builders; add a `compile_model_from_blocks` helper that takes a tuple of `ov.Model`s and stitches them via `ConcatPass`.**

---

## 11. Reference / pointers

### 11.1 In-repo

| Artefact | Path |
|---|---|
| OpenVINO port M0 source | `runtimes/openvino_port/` (7 files) |
| Block lowering walker | `runtimes/openvino_port/block_lowerer.py:56-146` |
| M0 norm builder | `runtimes/openvino_port/norm.py:25-53` |
| M0 GQA builder (with M1-extension guards) | `runtimes/openvino_port/attention.py:57-156` |
| M0 RoPE builder | `runtimes/openvino_port/rope.py:29-84` |
| M0 SwiGLU builder | `runtimes/openvino_port/ffn.py:30-42` |
| Existing M0 parity gate | `tests/parity/test_openvino_vs_torch.py:50, 136-197` |
| Canonical fixtures | `tests/parity/_canonical.py` |
| Working M0 sample | `examples/v2/03_run_openvino_gpu.py` |
| Qwen3.5 layer (M1 target) | `examples/v2/define_qwen3_5_layer.py` |
| Gemma 4 layer (M1 target) | `examples/v2/define_gemma4_layer.py` |
| GPT-OSS layer (M1 target) | `examples/v2/define_gpt_oss_layer.py` |
| Qwen3-0.6B layer (M0 target) | `examples/v2/define_qwen3.py` |
| Earlier verification (separate path) | `docs/runtimes/openvino-gpu.md` |
| Normative API spec | `components/api_spec.md` |
| API intro | `components/README.md` |

### 11.2 OpenVINO documentation and source

| Reference | URL |
|---|---|
| OV 2026 GPU device docs | https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html |
| OV `ScaledDotProductAttention` spec | https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html |
| OV opset14 `scaled_dot_product_attention` Python | https://docs.openvino.ai/2024/api/ie_python_api/_autosummary/openvino.runtime.opset14.scaled_dot_product_attention.html |
| OV `RMSFusion` source | https://github.com/openvinotoolkit/openvino/blob/master/src/common/transformations/src/transformations/common_optimizations/rms_fusion.cpp |
| OV GPU plugin structure | https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/source_code_structure.md |
| Stateful models (`MakeStateful`) | https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-request/stateful-models/obtaining-stateful-openvino-model.html |
| `MakeStateful` Python API | https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.runtime.passes.MakeStateful.html |
| Obtaining a stateful OV model (KV-cache rationale) | https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-request/stateful-models/obtaining-stateful-openvino-model.html |
| NNCF weight compression docs | https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html |
| NNCF 4-bit quantization docs | https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression/4-bit-weight-quantization.html |
| NNCF Usage.md (modes, AWQ, GPTQ, scale_estimation) | https://github.com/openvinotoolkit/nncf/blob/develop/docs/usage/post_training_compression/weights_compression/Usage.md |
| Microscaling MX quantization | https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression/microscaling-quantization.html |
| OpenVINO GenAI `LLMPipeline` source | https://github.com/openvinotoolkit/openvino.genai/blob/master/src/cpp/include/openvino/genai/llm_pipeline.hpp |
| GenAI configuration | https://deepwiki.com/openvinotoolkit/openvino.genai/7.5-configuration-and-advanced-usage |
| GenAI LLM Pipeline overview | https://deepwiki.com/openvinotoolkit/openvino.genai/2.1-llm-pipeline |
| Custom GPU operations | https://docs.openvino.ai/2025/documentation/openvino-extensibility/custom-gpu-operations.html |
| OpenVINO project ideas 2026 | https://github.com/openvinotoolkit/openvino/wiki/Project-ideas-for-2026 |

### 11.3 Relevant pull requests

| PR | Topic |
|---|---|
| [#27683](https://github.com/openvinotoolkit/openvino/pull/27683) | Rename `SwiGLU` internal op → `GLU` (supports both SwiGLU and GeGLU based on `glu_type`) |
| [#27907](https://github.com/openvinotoolkit/openvino/pull/27907) | GPU RoPE OCL kernel optimisation (~50% perf improvement) |
| [#28447](https://github.com/openvinotoolkit/openvino/pull/28447) | Fix `RoPEFusion` after SDPA→PagedAttention conversion |
| [#28205](https://github.com/openvinotoolkit/openvino/pull/28205) | GPU PagedAttention scores output support |
| [#29248](https://github.com/openvinotoolkit/openvino/pull/29248) | GPU PagedAttention improvements (sliding_window slot) |
| [#29620](https://github.com/openvinotoolkit/openvino/pull/29620) | GPU RoPE + GroupNorm `ocl_v2` kernels + post-ops generator port |
| [#29527](https://github.com/openvinotoolkit/openvino/pull/29527) / [#29459](https://github.com/openvinotoolkit/openvino/pull/29459) | RoPEFusion pattern fixes for GLM4 model on GPU |
| [#29191](https://github.com/openvinotoolkit/openvino/pull/29191) | Update GLM RopeFusion pattern to support GLM4 with PagedAttention |
| [#30174](https://github.com/openvinotoolkit/openvino/pull/30174) | PagedAttention score aggregation inputs (`score_aggregation_window`) |

### 11.4 OV release-notes anchors

- OpenVINO 2025.4 — first release with `sink` input on SDPA at the opset14 level, GatedDeltaNet fused GPU kernel (`docs/runtimes/openvino-gpu.md` §3 cites this).
- OpenVINO 2026.0 — MLIR-based NPU compiler preview.
- OpenVINO 2026.1 — MLIR NPU compiler becomes the preferred path.
- OpenVINO 2026.2 — our pinned version (`pyproject.toml`'s `openvino` extra).

---

## 12. Glossary

Definitions for the OV-fluent / LLM-naive reader.

| Term | Definition |
|---|---|
| **GQA** (Grouped-Query Attention) | Attention where `num_kv_heads < num_heads`; each KV head is shared across `num_heads / num_kv_heads` query heads via broadcast. |
| **MHA** (Multi-Head Attention) | Classic Vaswani et al. attention; `num_kv_heads == num_heads`. The degenerate case of GQA with no broadcast. |
| **MLA** (Multi-head Latent Attention) | DeepSeek-V2 variant: K and V are compressed into a per-token latent of width `latent_dim + rope_head_dim`; per-head K/V are reconstructed via up-projection. Smaller cache, larger compute per token. |
| **SDPA** (Scaled Dot-Product Attention) | The fused `softmax(Q·K^T / sqrt(Dh)) · V` op. OV exposes it as `opset13::ScaledDotProductAttention` and `opset14::ScaledDotProductAttention`. |
| **Attention sinks** | A learned per-head additive logit prepended to the softmax denominator. Stabilises long-context generation. GPT-OSS uses this; appears as the `sinks` weight in `GQASinksSpec`. |
| **RMSNorm** | `x · 1/√(mean(x²) + ε) · γ`. Like LayerNorm but without the mean-centering. Cheaper, equally effective in transformer language models. |
| **Zero-centered RMSNorm** | Variant where the affine `γ` is initialized to 0 and the formula becomes `x · rsqrt(mean(x²) + ε) · (1 + γ)`. Gemma family. |
| **RoPE** (Rotary Positional Embedding) | Per-token rotation of Q and K by an angle derived from the position; `inv_freq[i] = 1 / theta^(2i/Dh)`. |
| **Partial RoPE** | Only a fraction (`partial_rotary_factor`) of head_dim is rotated; the rest passes through. Two layouts: `prefix` (rotated dims at the front) and `proportional` (interleaved). |
| **MRoPE** (Multi-axis RoPE) | Per-axis (T, H, W, ...) frequency sections; the `sections` tuple distributes `head_dim/2` slots across the axes. Used by vision-bearing positions in Qwen3.5. |
| **SwiGLU** | `down(silu(gate(x)) * up(x))`. The current default activation in Llama/Qwen/Mistral. |
| **GeGLU** | `down(gelu(gate(x)) * up(x))`. Gemma family. |
| **Clamped-SwiGLU** | GPT-OSS variant with pre-activation clamp and Swish-beta sigmoid; `clamp=7.0, alpha=1.702`. |
| **MoE** (Mixture of Experts) | The FFN is replaced by `num_experts` parallel FFNs; a router selects top-K for each token; outputs are weighted-summed. |
| **PagedAttention** | KV-cache management borrowed from vLLM; cache is split into fixed-size pages so multiple sequences can share GPU memory efficiently. OV exposes it as a `PagedAttentionExtension` op. |
| **`MakeStateful` transformation** | OV pass that converts a `(Parameter, Result)` pair into a `(ReadValue, Assign)` pair on a Variable identity — i.e. turns external KV-cache I/O into model-internal state. |
| **NNCF** (Neural Network Compression Framework) | OV's quantization library. `nncf.compress_weights` does post-training weight-only quantization to INT4/INT8 (plus MXFP4, NF4, etc.). |
| **opset** | OV's versioned set of operations. opset16 is the current general-purpose default; opset13/14 expose SDPA variants; opset3 owns Variables. |
| **`EXECUTION_DEVICES`** | OV runtime property listing which device(s) actually ran the compiled model. Must contain `'GPU.0'` for the iGPU path. |
| **`INFERENCE_PRECISION_HINT`** | OV plugin property; `f16` or `f32`. Selects the GPU's compute precision independent of the IR element type. |
| **`KV_CACHE_PRECISION`** | OV plugin property; sets the precision of `ReadValue`/`Assign` Variables independently of `INFERENCE_PRECISION_HINT`. Defaults to `u8` on some plugins; we force `f16` for parity. |
| **`PERFORMANCE_HINT`** | `LATENCY` (single stream, low latency) or `THROUGHPUT` (multi-stream, higher aggregate). M0 uses `LATENCY`. |
| **`EXECUTION_MODE`** | `PERFORMANCE` vs `ACCURACY`. `ACCURACY` disables some lossy optimisations. M2 regression fallback. |
| **BYOM** | Build Your Own Model — Microsoft WinML CLI Q3-Q4 2026 release feature that consumes the `components/` API as its source of truth. |
| **Cardinal Rule** | The org-wide rule (cited at `components/README.md:32-37`) forbidding model-family names in public symbols. Applies to this document too: no Llama / Qwen / Gemma in API symbols. |

---

## Appendix A — verification checklist for the new thread

Before declaring Phase 1 complete:

- [ ] Every M1 spec in `components/api_spec.md` Sections 4–7 has a builder dispatch in `block_lowerer.py`.
- [ ] Every M1 spec has a parity test under `tests/parity/test_v3_openvino_*.py`.
- [ ] Every parity test asserts `mean_rel < 5e-4` at fp16.
- [ ] Every parity test asserts `EXECUTION_DEVICES == ['GPU.0']`.
- [ ] One full-layer test passes for Qwen3-0.6B, Qwen3.5-35B-A3B-full-attn, Gemma 4 E2B local, and GPT-OSS-20B sliding (synthetic-shape).
- [ ] `tests/parity/_canonical_v3.py` exists with rescaled weights that fit fp16.
- [ ] RMSFusion fires on the new fp16 path (verified via runtime-model op summary).
- [ ] RoPEFusion fires on the new fp16 path (same).
- [ ] No regression in the existing M0 test (`test_openvino_vs_torch.py`).
- [ ] `examples/v2/03_run_openvino_gpu.py` runs and prints fp16 metrics (per the existing template at `examples/v2/03_run_openvino_gpu.py:152-167`).
- [ ] `docs/runtimes/openvino-gpu-port-requirements.md` (this file) is updated with any open-question decisions taken during implementation.

Before declaring Phase 2 complete: add INT4 + KV-cache items.
Before declaring Phase 3 complete: add GenAI + end-to-end-model items.

---

**End of document.**
