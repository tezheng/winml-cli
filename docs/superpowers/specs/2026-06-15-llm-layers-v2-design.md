# llm-layers v2 — Design Spec

**Date:** 2026-06-15
**Status:** Draft, awaiting user review (gates further work)
**Owner:** zhengte@microsoft.com
**Supersedes:** `2026-06-06-llm-layers-design.v3.md` for the *implementation contract* only — the v3 spec's axis catalog, op floor, and per-family rollout matrix all stay current; what v2 redefines is the **shape of the code that realises that IR**.
**Driver:** the 2026-06-15 audit surfaced two principle violations in the v1 implementation: (1) the IR layer is a thin `nn.Module` facade over torch, and (2) four family-named symbols leaked into `api/`. v1 is otherwise functionally correct (74 unit tests passing M1; Qwen3-0.6B parity at `max_abs_diff = 4.77e-7`). v2 keeps the design intent verbatim and replaces only the implementation substrate.

---

## 0. Executive summary

llm-layers v2 is a numpy-first re-implementation of the v3 IR. It separates the IR into three pure-data layers — **specs** (frozen-dataclass "what variant"), **params** (numpy-backed "weight data"), **functional** (top-level pure-function "compute") — and pushes every family name out of `api/` into `configs/<family>.py` factory functions. The torch dependency moves to an *optional* `runtimes/torch/` lane reserved for parity tests. PyTorch is no longer in the IR's import graph.

The two motivating principles are non-negotiable: (1) **no PyTorch in the IR layer** — tensors are `numpy.ndarray` plus `ml_dtypes` for bf16/fp8; (2) **no model-personal names in api class/function symbols** — variants live in parameters on neutral primitives (`DecoderBlock`, `SelectiveScanMixer`, `SmoothScaledRoPEParams`), and family names live only in `configs/<family>.py` filenames and citation comments. v1 violates both: 5,876 of 6,096 `api/` LOC are torch-coupled, and four symbols carry vendor names (`Llama3RoPEParams`, `Mamba1Mixer`, `Mamba2Mixer`, `_Qwen3NextRMSNormGated`).

Scope of this doc: the v2 module shape, the three IR layers, the configs/loaders structure, the migration phases, and the Qwen3-0.6B vertical-slice acceptance gate. Non-goals: building a runtime, optimising throughput, modifying v1 `api/` except for the Phase-0 renames (sequenced separately). The acceptance gate for the first deliverable is a numpy-only forward of Qwen3-0.6B matching HF torch eager within `atol=1e-3` at S=16 — wall-time ≤30s per test on the Win11 box. Everything else is sequenced after that gate.

---

## 1. Principles and non-goals

### 1.1 Principles (verbatim, non-negotiable)

**P1.** No PyTorch in the IR layer. Tensors are `numpy.ndarray`. bf16 / fp8 / packed types come from the `ml_dtypes` PyPI package. The IR's import graph must contain `numpy` and optionally `ml_dtypes` — and nothing else from the ML ecosystem.

**P2.** No model-personal names in `api_v2/` class or function symbols. The four v1 violations (`Llama3RoPEParams` — `api/specs.py:30`; `Mamba1Mixer` — `api/ssm.py:93`; `Mamba2Mixer` — `api/ssm.py:272`; `_Qwen3NextRMSNormGated` — `api/ssm.py:501`) are renamed to neutral primitives whose **parameters** select the variant. Family names live ONLY in `configs/<family>.py` filenames and in citation comments (`# Source: modeling_qwen3.py:42`).

### 1.2 Non-goals

- **Not a new runtime.** v2 is a *reference IR*. Slow numpy is fine. Throughput is the consumer's problem.
- **No optimisation kernels.** No torch.compile, no fused SDPA, no Triton.
- **No modification of v1 `api/`, `models/`, `tests/` during v2 build.** v1 stays intact as the working reference until v2 reaches feature parity. The only exception is the **Phase-0 rename** inside v1 (drop the four family-named symbols from `api/`) — sequenced separately, not blocked by v2.
- **No bit-exact equivalence with torch.** bf16 numerics in numpy via `ml_dtypes` will differ from torch CPU bf16 at round-to-nearest. Target is the same `atol=1e-3` band the v1 parity tests already use.
- **No quant code in Phase 1.** AWQ INT4 / GGUF Q4_K_M bit-packing in numpy is tedious to port (v1 spends 838 LOC on it). Defer to Phase 5 after structural primitives stabilise.
- **No training / fine-tuning.** Inherits the v3 non-goal; weight loaders are read-only.

---

## 2. Module layout

```
llm-layers/
  api_v2/                            # NEW — the numpy IR. No torch in import graph.
    __init__.py                      # public re-exports
    types.py                         # enums (DType, NormKind, AttentionKind, ...) — ported from api/types.py
    dtype.py                         # DType enum + ml_dtypes wiring; numpy/ml_dtypes dtype resolver
    specs.py                         # frozen dataclasses (AttentionSpec, FFNSpec, ...) — torch.dtype dropped
    ops.py                           # pure-function numpy primitives (silu, sdpa, rms_norm, rope_apply, ...)
    params.py                        # parameter dataclasses (RMSNormParams, AttentionParams, ...) — own np.ndarrays
    functional.py                    # pure forward functions (rms_norm_forward(params, x), attention_forward(...), ...)
    block.py                         # decoder_block_forward(...) — composite, also a top-level function
    kvcache.py                       # KV-cache helpers (numpy-backed; layout dispatch via spec)
    quant.py                         # quant primitives (numpy + ml_dtypes; Phase 5)
  configs/                           # NEW — per-family factories. NO CLASSES.
    __init__.py
    qwen3.py                         # block_spec(...), from_hf_dict(hf, layer_idx), HF_NAME_MAP
    qwen3_moe.py
    llama3.py
    llama4_scout.py                  # iRoPE NoPE pattern
    gemma2.py
    gemma3.py
    gemma4.py                        # K=V, p-RoPE, fixed-scale QK-norm, PLE
    mistral.py
    ministral.py
    mixtral.py
    olmo2.py
    olmoe.py
    phi3_mini.py
    phi3_small.py
    phi4_mini.py
    phi4_mini_flash.py
    smollm3.py
    tinyllama.py
    mpt.py                           # ALiBi, LayerNorm
    falcon7b.py                      # parallel block
    bitnet.py                        # ternary + sub-norms
    granite.py
    granite4_h.py                    # Mamba-2 9:1 hybrid
    gpt_oss.py                       # MoE, clamped-SwiGLU, MXFP4-native, sinks
    mamba1.py
    mamba2.py
    mamba3.py                        # NEW vs v1: complex-state
    jamba.py
    hymba.py
    falcon_h1.py
    rwkv7.py                         # NEW vs v1
    minimax_text_01.py               # Lightning Attention
    minimax_m2.py
    qwen3_next.py                    # 3:1 Gated DeltaNet hybrid + ultra-sparse MoE
    deepseek_v2_lite.py
    deepseek_v3_lite.py
    deepseek_v3_moe.py
    deepseek_v32.py                  # DSA Lightning Indexer
    deepseek_v4.py                   # CSA + HCA
    minicpm3.py                      # MLA
    nemotron3.py
    moshi.py
    voxtral.py
    hunyuan_large.py                 # CLA
    recurrent_gemma.py
    qwen2_5_vl.py                    # M-RoPE
    glm_moe_dsa.py
    got_ocr2.py
    deepseek_ocr2.py
  loaders/                           # NEW — generic HF / safetensors readers
    __init__.py
    hf.py                            # load_hf_state_dict(params, state_dict, layer_idx, name_map)
    safetensors.py                   # load_safetensors_into_params(path, params, layer_idx, name_map)
    torch_bridge.py                  # OPTIONAL — wraps torch tensors → np.ndarray at boundary
  runtimes/                          # OPTIONAL lane — parity-test wrappers
    torch/
      ops.py                         # torch-tensor mirror of api_v2.ops (only if needed)
      modules.py                     # nn.Module wrappers that hold params + delegate to api_v2.functional
    openvino/                        # FUTURE — ov.Model lowering
  docs/
    families/                        # NEW location for layer.md
      qwen3.md
      gemma4.md
      ...
  tests/
    api_v2/                          # numpy-only unit tests (replace tests/api/ over time)
    configs/                         # tests that block_spec_from_hf produces correct specs
    parity/                          # KEEPS torch — numpy IR vs HF torch reference
  api/                               # v1 — UNTOUCHED during v2 build (Phase-0 renames sequenced separately)
  models/                            # v1 — UNTOUCHED
```

Notes on key choices:

| Module | Rationale |
|---|---|
| `api_v2/` (parallel to `api/`) | lets v2 land incrementally without breaking v1; symlink-style aliasing avoided for clarity |
| `configs/` is data only | the 41 `Config` dataclasses + 76 `build_/load_hf_` functions in v1's `models/` collapse to one factory + one HF map per family |
| `loaders/` is generic | one name-template substituter handles all 38 families. Per-family logic stays in the `HF_NAME_MAP` dict, not in code |
| `runtimes/torch/` is optional | only needed for parity tests; not a public surface |
| `docs/families/<family>.md` | the v1 `models/<family>/layer.md` content moves out of code paths into docs |

---

## 3. The three IR layers

v2 separates the IR into three orthogonal layers. Each layer is pure (no side effects, no hidden state). The combination is composable in a way the v1 `nn.Module`-fused layout is not.

### 3.1 Layer A — `specs.py` (what)

Frozen dataclasses that declare *which variant* a block uses. No tensors, no compute.

v1's `api/specs.py` is ~99% torch-independent. The port is mostly verbatim, with these mechanical fixes plus the coverage gap fixes in §10:

| v1 location | v2 change |
|---|---|
| `api/specs.py:9` | drop `import torch` |
| `api/specs.py:335,338,346,347` (four `torch.dtype` fields on `QuantSpec`, `KVCacheSpec`) | replace with `DType` enum from `api_v2/dtype.py` |
| `api/specs.py:30` (`Llama3RoPEParams`) | rename `SmoothScaledRoPEParams` (P2 violation) |
| `api/specs.py:99` (`AliBiSpec`) | OK as-is (acronym, not vendor) |
| `api/specs.py:130` (`RoPESpec.llama3_extra`) | rename field `smooth_scaled_extra` (cross-cut with rename above) |
| `AttentionSpec` | add `output_gate: Literal["sigmoid"] | None` — Qwen3.5/3.6 sigmoid attention gate (§10) |
| `RoPESpec` | add `nope_layer_pattern: tuple[bool, ...] | None` — Llama 4 iRoPE per-layer NoPE alternation |
| `TokenMixerKind` | add `WKV7`, `LIGHTNING` |
| `SSMKind` | add `MAMBA3` |
| `QDType` | add `NVFP4` |

Skeleton for the renamed RoPE struct:

```python
# api_v2/specs.py
@dataclass(frozen=True)
class SmoothScaledRoPEParams:
    """Smooth wavelength-scaled RoPE parameters (introduced by Llama-3,
    reused by SmolLM3 / similar). Family-neutral name."""
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_context_length: int
```

### 3.2 Layer B — `params.py` (how — data)

For every spec there is a parallel `<X>Params` dataclass that owns the actual `np.ndarray` weights. Plain dataclasses; no methods, no inheritance from `nn.Module`. Pure data containers.

```python
# api_v2/params.py
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class RMSNormParams:
    weight: np.ndarray                  # [hidden] or [head_dim] (for QK-norm)

@dataclass
class LinearParams:
    weight: np.ndarray                  # [out, in]
    bias: Optional[np.ndarray] = None   # [out] or None

@dataclass
class AttentionParams:
    q_proj: LinearParams
    k_proj: LinearParams
    v_proj: LinearParams
    o_proj: LinearParams
    q_norm: Optional[RMSNormParams] = None    # Qwen3 / OLMo 2 / Gemma 3-4
    k_norm: Optional[RMSNormParams] = None
    v_norm: Optional[RMSNormParams] = None    # Gemma 4
    attn_sub_norm: Optional[RMSNormParams] = None   # BitNet
    sinks: Optional[np.ndarray] = None        # [n_heads] — GPT-OSS trained sinks
    # MLA additions (None for non-MLA):
    q_a_proj: Optional[LinearParams] = None
    q_a_layernorm: Optional[RMSNormParams] = None
    q_b_proj: Optional[LinearParams] = None
    kv_a_proj_with_mqa: Optional[LinearParams] = None
    kv_a_layernorm: Optional[RMSNormParams] = None
    kv_b_proj: Optional[LinearParams] = None

@dataclass
class FFNParams:
    gate_proj: LinearParams
    up_proj: LinearParams
    down_proj: LinearParams
    ffn_sub_norm: Optional[RMSNormParams] = None    # BitNet

@dataclass
class MoEParams:
    router: LinearParams
    experts_gate_up: list[LinearParams]     # n_experts × LinearParams
    experts_down: list[LinearParams]
    shared_expert: Optional[FFNParams] = None
    e_score_correction_bias: Optional[np.ndarray] = None  # DeepSeek-V3
    tid2eid: Optional[np.ndarray] = None    # DeepSeek-V4 hash router

@dataclass
class DecoderBlockParams:
    pre_attn_norm: RMSNormParams
    attention: AttentionParams
    pre_ffn_norm: RMSNormParams
    ffn: object                       # FFNParams | MoEParams — discriminated by spec
    # Sandwich norms (Gemma 2/3/4):
    post_attn_norm: Optional[RMSNormParams] = None
    post_ffn_norm: Optional[RMSNormParams] = None
    # PLE (Gemma 4):
    per_layer_input_gate: Optional[LinearParams] = None
    per_layer_projection: Optional[LinearParams] = None
    post_per_layer_input_norm: Optional[RMSNormParams] = None
    layer_scalar: Optional[np.ndarray] = None
    # Granite μP:
    residual_scale: Optional[float] = None
```

**Allocation pattern:** `params.zeros_like(spec, hidden_size, dtype)` produces a fully-zeroed param tree from a `DecoderBlockSpec`. The loader then fills in the arrays. Zero-init is the *only* default; weights without a state-dict entry stay zero and the loader raises if any expected key is missing.

### 3.3 Layer C — `functional.py` (how — compute)

Top-level pure functions that take `(params, spec, x, …)` and return the forward output. No closures over module state, no `self`. The functions dispatch on spec enums.

```python
# api_v2/functional.py
import numpy as np
from api_v2 import ops, params, specs, types

def rms_norm_forward(p: params.RMSNormParams, x: np.ndarray, *,
                     eps: float, one_plus_w: bool) -> np.ndarray:
    return ops.rms_norm(x, p.weight, eps=eps, one_plus_w=one_plus_w)

def linear_forward(p: params.LinearParams, x: np.ndarray) -> np.ndarray:
    y = x @ p.weight.T
    return y + p.bias if p.bias is not None else y

def attention_forward(
    p: params.AttentionParams,
    s: specs.AttentionSpec,
    x: np.ndarray,
    cache: "kvcache.KVCacheView | None",
    position_ids: np.ndarray,
    cos_sin: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, "kvcache.KVCacheUpdate | None"]:
    # spec-driven dispatch:
    #   STANDARD → q/k/v_proj + optional qk_norm + rope_apply + sdpa + o_proj
    #   MLA      → q_a/q_b + kv_a/kv_b lora path
    #   DSA      → STANDARD + indexer_score_topk on top
    #   CSA_HCA  → shape-only placeholder (Phase 4)
    ...

def decoder_block_forward(
    p: params.DecoderBlockParams,
    s: specs.DecoderBlockSpec,
    x: np.ndarray,
    cache: "kvcache.KVCacheView | None",
    position_ids: np.ndarray,
) -> tuple[np.ndarray, "kvcache.KVCacheUpdate | None"]:
    ...
```

The neutral mixer renames (no family names):

| v1 symbol (file:line) | v2 functional name | Selector |
|---|---|---|
| `Mamba1Mixer` (`api/ssm.py:93`) | `selective_scan_mixer_forward(p, s, x, ...)` | `s.kind == SSMKind.MAMBA1` |
| `Mamba2Mixer` (`api/ssm.py:272`) | `ssd_mixer_forward(p, s, x, ...)` | `s.kind == SSMKind.MAMBA2_SSD` |
| `_Qwen3NextRMSNormGated` (`api/ssm.py:501`) | `gated_rms_norm_forward(p, x, gate, eps)` | always (helper) |
| `Llama3RoPEParams` (`api/specs.py:30`) | `SmoothScaledRoPEParams` (spec rename) | scaling enum |

The token-mixer dispatcher in `decoder_block_forward` then branches on `TokenMixerKind` — `ATTENTION`, `SSM_MAMBA1`, `SSM_MAMBA2`, `SSM_MAMBA3`, `GATED_DELTANET`, `WKV7`, `LIGHTNING`, etc. — with no family name in the call graph.

### 3.4 Why three layers, not one `nn.Module`

| Property | v1 (`nn.Module` fused) | v2 (specs / params / functional) |
|---|---|---|
| Reload weights without rebuilding modules | no — must rebuild | yes — replace `params` arrays in place |
| Reorder spec field without rebuild | no | yes |
| Test forward with synthetic params dict | painful (need `nn.Module`) | trivial |
| Lower to a non-PyTorch runtime | requires re-implementation | spec + params are the lowering input |
| Carry torch in import graph | required | not required |

The pure-function layout is also the natural shape an OpenVINO / ONNX lowering would consume: `spec` becomes graph topology, `params` become initialiser tensors, `functional` is a reference implementation a backend can ignore.

---

## 4. `configs/` — per-family factories (no classes)

Every family is exactly one file: a `block_spec(...)` keyword factory, a `from_hf_dict(hf, layer_idx)` HF-config adapter, and an `HF_NAME_MAP` dict. **No classes.** Family name lives only in the filename and the docstring header. The whole module is data + light glue.

```python
# configs/qwen3.py
"""Qwen3 family — Qwen3-0.6B / 1.7B / 4B / 8B.

Source: transformers.Qwen3Config + modeling_qwen3.py.
Key facts (audited 2026-06-04, see research/02-layer-sources.v3.md §Qwen3):
- GQA, head_dim EXPLICIT (1024/16=64 vs head_dim=128 on 0.6B)
- QK-norm PRE-RoPE, weight shape PER_HEAD_DH
- RoPE SPLIT_HALF basis, base_theta=1_000_000 (Qwen3 uses 1M)
- RMSNorm STANDARD_W
- SwiGLU FFN
- tie_word_embeddings True for 0.6B/1.7B/4B, False for 8B
"""
from __future__ import annotations
from api_v2 import specs, types, dtype


def block_spec(
    *,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
    num_hidden_layers: int,
    rope_theta: float = 1_000_000.0,
    rms_norm_eps: float = 1e-6,
    layer_idx: int = 0,
    dtype_: dtype.DType = dtype.DType.BF16,
) -> specs.DecoderBlockSpec:
    norm = specs.NormSpec(
        kind=types.NormKind.RMS, eps=rms_norm_eps,
        weight_mode=types.NormWeightMode.STANDARD_W,
    )
    attn = specs.AttentionSpec(
        n_q_heads=num_attention_heads,
        n_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        qk_norm=norm,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        rope=specs.RoPESpec(
            base_theta=rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.NONE,
        ),
    )
    ffn = specs.FFNSpec(
        intermediate_size=intermediate_size,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )
    return specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn,
        channel_mixer=ffn,
        pre_attn_norm=norm,
        pre_ffn_norm=norm,
    )


def from_hf_dict(hf: dict, layer_idx: int = 0) -> specs.DecoderBlockSpec:
    # transformers 5.x nests rope_theta under rope_parameters; older flat key honoured.
    rp = hf.get("rope_parameters") or {}
    rope_theta = float(
        hf.get("rope_theta")
        or rp.get("rope_theta")
        or 1_000_000.0
    )
    return block_spec(
        hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_key_value_heads=hf["num_key_value_heads"],
        head_dim=hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]),
        intermediate_size=hf["intermediate_size"],
        num_hidden_layers=hf["num_hidden_layers"],
        rope_theta=rope_theta,
        rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
        layer_idx=layer_idx,
        dtype_=dtype.from_hf_dtype(hf.get("dtype", hf.get("torch_dtype", "float32"))),
    )


HF_NAME_MAP: dict[str, str] = {
    "model.layers.{L}.input_layernorm.weight":           "pre_attn_norm.weight",
    "model.layers.{L}.self_attn.q_proj.weight":          "attention.q_proj.weight",
    "model.layers.{L}.self_attn.k_proj.weight":          "attention.k_proj.weight",
    "model.layers.{L}.self_attn.v_proj.weight":          "attention.v_proj.weight",
    "model.layers.{L}.self_attn.o_proj.weight":          "attention.o_proj.weight",
    "model.layers.{L}.self_attn.q_norm.weight":          "attention.q_norm.weight",
    "model.layers.{L}.self_attn.k_norm.weight":          "attention.k_norm.weight",
    "model.layers.{L}.post_attention_layernorm.weight":  "pre_ffn_norm.weight",
    "model.layers.{L}.mlp.gate_proj.weight":             "ffn.gate_proj.weight",
    "model.layers.{L}.mlp.up_proj.weight":               "ffn.up_proj.weight",
    "model.layers.{L}.mlp.down_proj.weight":             "ffn.down_proj.weight",
}
```

Compare to v1: `models/qwen3/config.py` is 120 LOC of `class Qwen3Config:` plus `to_block_spec` (`models/qwen3/config.py:30-120`); `models/qwen3/layer.py` is 74 LOC of `build_qwen3_decoder_layer` + `load_hf_qwen3_layer` (`models/qwen3/layer.py:12,25`). v2 collapses both to ~80 LOC of data with the family name nowhere in a symbol.

The `qwen3 ↔ mistral` config diff in v1 is ~80% structurally identical; the `qwen3 ↔ mistral` layer.py diff is ~85% boilerplate. In v2 the diff *is* the data — only the `HF_NAME_MAP` rows and a few `block_spec` defaults change.

### 4.1 Optional sub-config families

A handful of families (DeepSeek-V2/V3/V4, Phi-3 short/long, Gemma 4 E2B/E4B/12B/26B/31B) have multi-size variants where the spec differs per size. Pattern: keep one `block_spec(...)` with all knobs as keyword arguments, then add named convenience constructors:

```python
# configs/gemma4.py
def block_spec(*, model_size: str, layer_idx: int = 0, ...) -> specs.DecoderBlockSpec: ...

def e2b(layer_idx: int = 0) -> specs.DecoderBlockSpec:
    return block_spec(model_size="E2B", layer_idx=layer_idx, ...)

def e4b(layer_idx: int = 0) -> specs.DecoderBlockSpec: ...
```

No classes; just named functions. Open question §11 covers granularity.

---

## 5. Loaders

The generic loader is one function — `load_hf_state_dict(params, state_dict, layer_idx, name_map)` — that handles every family. Family logic is in the per-family `HF_NAME_MAP` dict.

```python
# loaders/hf.py
import numpy as np
from ml_dtypes import bfloat16
from api_v2 import dtype as _dtype

def load_hf_state_dict(
    block_params: "params.DecoderBlockParams",
    state_dict: dict,                # HF state_dict, keys are torch tensors OR np arrays
    layer_idx: int,
    name_map: dict[str, str],
) -> None:
    """Walks name_map, substitutes {L}, copies into block_params attribute paths.

    Coerces dtype via ml_dtypes (bf16 / fp8) if the destination ndarray dtype
    differs from the source.
    """
    missing = []
    for hf_key_tmpl, params_path in name_map.items():
        hf_key = hf_key_tmpl.format(L=layer_idx)
        if hf_key not in state_dict:
            # Optional sub-keys (q_norm/k_norm on non-QK-norm families, etc.):
            slot = _resolve_slot(block_params, params_path)
            if slot is None:
                continue
            missing.append(hf_key)
            continue
        slot = _resolve_slot(block_params, params_path)
        if slot is None:
            continue
        src = _as_numpy(state_dict[hf_key])
        if src.shape != slot.shape:
            raise ValueError(f"shape mismatch {hf_key}: {src.shape} vs {slot.shape}")
        slot[...] = src.astype(slot.dtype, copy=False)
    if missing:
        raise KeyError(f"missing tensors: {missing}")


def _resolve_slot(root, path: str):
    """'attention.q_proj.weight' → root.attention.q_proj.weight (or None if any link is None)."""
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _as_numpy(t) -> np.ndarray:
    """Convert torch tensor or np.ndarray to np.ndarray (bf16 routed through ml_dtypes)."""
    if isinstance(t, np.ndarray):
        return t
    # torch.Tensor — convert without importing torch globally
    if hasattr(t, "detach"):
        if t.dtype.__class__.__module__.startswith("torch") and str(t.dtype) == "torch.bfloat16":
            # torch cannot .numpy() bf16 directly — view as int16 then cast
            return t.detach().cpu().view(np.int16).numpy().view(bfloat16)
        return t.detach().cpu().numpy()
    raise TypeError(f"unsupported tensor type: {type(t)}")
```

Coverage of harder cases:

| Case | Handling |
|---|---|
| Optional sub-keys (q_norm / k_norm) | `_resolve_slot` returns `None`; loop skips silently. Missing-tensor list excludes optional paths |
| MoE expert weights | Templated like `model.layers.{L}.mlp.experts.{E}.gate_proj.weight` — `name_map` can contain `{E}` and the loader iterates over `range(n_experts)` when it sees that token |
| Fused gate/up projections (Phi-3) | The mapping rule splits the source tensor into two slots; encoded as `{"phi3.fused_proj.weight": ("ffn.gate_proj.weight:0:I", "ffn.up_proj.weight:I:2I")}` — slice syntax in the value |
| Transposed-vs-not (HF convention vs ours) | Loader honours destination shape; no transpose logic in `name_map` |
| Tied embeddings | Out of scope of layer loader — model-level loader handles this |

The `safetensors.py` loader is a thin wrapper that opens a `.safetensors` file with the `safetensors` PyPI package and feeds the resulting dict into `load_hf_state_dict`.

---

## 6. `runtimes/torch/` — parity-test lane

The parity tests need to compare a numpy IR forward to an HF torch eager forward. Two options:

**Option A — Wrap.** Run `api_v2` end-to-end, then `torch.from_numpy(out)` at the test boundary. The parity test converts HF output back to numpy via `_as_numpy` from §5 and compares with `np.allclose`.

**Option B — Mirror.** `runtimes/torch/ops.py` mirrors every `api_v2/ops.py` signature but implements via torch. The mirror lane is consumable by users who want the API surface with torch tensors.

**Recommendation: A (Wrap).** Reasons:
1. Zero maintenance burden — `api_v2/ops.py` is the only implementation.
2. The mirror lane would be ~800 LOC duplicate of `ops.py` for a single consumer (the parity test) which already has `torch` available.
3. If a torch consumer ever appears, `runtimes/torch/modules.py` (thin `nn.Module` wrappers that hold `params` and call `functional.*` after `.numpy()`-ing the input) is a 100-LOC build at that time.

The parity tests under `tests/parity/` therefore look like:

```python
def test_qwen3_0_6b_numpy_vs_hf():
    hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    hf_layer = hf.model.layers[0]
    spec = configs.qwen3.from_hf_dict(hf.config.to_dict(), layer_idx=0)
    p = params.zeros_like(spec, hidden_size=1024, dtype=DType.FP32)
    loaders.hf.load_hf_state_dict(p, hf.state_dict(), 0, configs.qwen3.HF_NAME_MAP)
    x_np = np.random.randn(1, 16, 1024).astype(np.float32)
    pos = np.arange(16)[None, :]
    out_np, _ = functional.decoder_block_forward(p, spec, x_np, cache=None, position_ids=pos)
    with torch.no_grad():
        out_hf = hf_layer(torch.from_numpy(x_np))[0]
    assert np.allclose(out_np, out_hf.numpy(), atol=1e-3, rtol=1e-3)
```

`runtimes/openvino/` is reserved for future work; the spec + params layers are designed to be its lowering input.

---

## 7. Tensor / dtype model

| dtype | numpy origin | Notes |
|---|---|---|
| `np.float32` | native | default compute dtype for the reference |
| `np.float16` | native | accepted for fp16 inputs |
| `ml_dtypes.bfloat16` | `ml_dtypes` PyPI | weights / activations for bf16 models; storage-only by default |
| `ml_dtypes.float8_e4m3fn` | `ml_dtypes` | FP8 W8A8 (Hopper / DeepSeek-V3 native) |
| `ml_dtypes.float8_e5m2` | `ml_dtypes` | FP8 grad / alt activation format |
| custom packed `uint8` blocks | numpy + helpers | MXFP4, NVFP4 (separate group size), GGUF Q4_K_M, AWQ INT4, BitNet ternary |
| custom packed `int8` blocks | numpy | INT8 W8A8, INT4 QAT (Gemma 4 mobile) |

`ml_dtypes` is a Google package (~50KB, JAX dependency). Zero transitive deps beyond `numpy`. PyPI: `pip install ml_dtypes`. It exposes `bfloat16` and FP8 variants as full numpy dtype objects: arithmetic is supported via numpy's ufunc machinery (slow — but reference IR).

**`DType` enum.** v2 introduces a single `DType` enum in `api_v2/dtype.py` that wraps numpy + ml_dtypes for uniform reference everywhere specs talk about dtype:

```python
# api_v2/dtype.py
from enum import Enum
import numpy as np
import ml_dtypes

class DType(Enum):
    FP32  = "fp32"
    FP16  = "fp16"
    BF16  = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8  = "int8"
    INT4_PACKED = "int4_packed"        # packed nibbles in uint8
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"
    TERNARY = "ternary"

_NUMPY: dict[DType, np.dtype] = {
    DType.FP32: np.dtype(np.float32),
    DType.FP16: np.dtype(np.float16),
    DType.BF16: np.dtype(ml_dtypes.bfloat16),
    DType.FP8_E4M3: np.dtype(ml_dtypes.float8_e4m3fn),
    DType.FP8_E5M2: np.dtype(ml_dtypes.float8_e5m2),
    DType.INT8: np.dtype(np.int8),
    # packed types map to uint8 storage; the actual semantics live in quant.py
    DType.INT4_PACKED: np.dtype(np.uint8),
    DType.MXFP4: np.dtype(np.uint8),
    DType.NVFP4: np.dtype(np.uint8),
    DType.TERNARY: np.dtype(np.uint8),
}

def to_numpy(d: DType) -> np.dtype: return _NUMPY[d]

def from_hf_dtype(name: str | object) -> DType: ...  # str-or-torch.dtype → DType
```

`specs.QuantSpec` and `specs.KVCacheSpec` carry `DType` instead of `torch.dtype` (the four fields cited in §3.1).

**Numerics caveat.** bf16 in numpy through `ml_dtypes` is not bit-exact with torch CPU bf16: numpy uses a slightly different round-to-nearest convention on truncation. Expected relative error is ~1e-3 — the v1 parity tolerance band, so no test impact.

---

## 8. Migration phases

| Phase | Scope | Deliverables | Success criteria | Effort | Depends on |
|---|---|---|---|---|---|
| **0** | Rename 4 family-named symbols inside *v1* `api/` | `Llama3RoPEParams` → `SmoothScaledRoPEParams`; `Mamba1Mixer`/`Mamba2Mixer` → `SelectiveScanMixer`/`SSDMixer` with `kind` parameter; `_Qwen3NextRMSNormGated` → `_GatedRMSNorm` | All v1 tests still pass; 0 grep hits for `Llama3|Mamba1Mixer|Mamba2Mixer|Qwen3Next` inside `api/` | 1-2 days | — |
| **1** | Build `api_v2/` + `configs/qwen3.py` + `loaders/hf.py` end-to-end for Qwen3-0.6B | New: `api_v2/{types,dtype,specs,ops,params,functional,block,kvcache}.py`; `configs/qwen3.py`; `loaders/hf.py`; `tests/parity/qwen3/test_numpy_vs_hf.py` | Numpy forward of Qwen3-0.6B block matches HF eager within `atol=1e-3`, S=16 fixed input; wall-time ≤30s per test | 2-3 weeks | Phase 0 (or run in parallel) |
| **2** | Roll out 6 simple families | `configs/{llama3,mistral,gemma2,olmo2,smollm3,tinyllama}.py` + parity tests | Each family passes the same `atol=1e-3` gate on a small public checkpoint | 1-2 weeks | Phase 1 |
| **3** | Migrate tests | Port `tests/api/test_{norm,rope,ops,feedforward}.py` to `tests/api_v2/` against numpy IR. Keep `tests/parity/` torch-using | All v2 unit tests pass; coverage parity with v1 | 3 weeks | Phase 2 |
| **4** | Hard families | `configs/{deepseek_v2_lite,deepseek_v3_lite,minicpm3 (MLA), mixtral, olmoe, qwen3_moe (MoE), mamba1, mamba2, mamba3, jamba (SSM), qwen3_next (Gated DeltaNet), rwkv7, minimax_text_01 (Lightning)}.py` + parity | All listed families pass the parity gate | 2-3 weeks | Phase 3 |
| **5** | Quantization | Port `api/quant.py`'s 838 LOC: AWQ INT4 unpack, GGUF Q4_K_M, MXFP4, NVFP4, BitNet ternary, FP8 W8A8 | Quant round-trip tests pass; INT4 weight forwards within `atol=2e-3` | 3-4 weeks | Phase 4 |
| **6** | Delete v1 | Remove `api/`, `models/`, `tests/api/`, `tests/models/`. Re-point `pyproject.toml`, `conftest.py`. Move v1 layer.md content to `docs/families/` | `git rm -r api models tests/api tests/models`; full test suite green on v2 only | 1 week | Phases 1-5 |

Total estimated effort: 11-15 weeks. Phase 1 is the gating decision — if numpy fidelity is unsatisfactory on the Qwen3-0.6B parity test, v2 is reconsidered before any further investment.

---

## 9. Vertical slice — Qwen3-0.6B acceptance gate

Phase 1 lands as one PR with these new files (paths are absolute targets, not yet on disk):

| Path | Purpose | LOC est |
|---|---|---|
| `api_v2/__init__.py` | public exports | 30 |
| `api_v2/types.py` | enums; ported from `api/types.py` + new enum values from §10 | 250 |
| `api_v2/dtype.py` | `DType` enum, `to_numpy`, `from_hf_dtype` | 80 |
| `api_v2/specs.py` | frozen dataclasses; ported from `api/specs.py` with torch.dtype dropped + 4 new fields | 800 |
| `api_v2/ops.py` | numpy primitives: `rms_norm`, `silu`, `swiglu`, `rope_apply`, `sdpa`, `softmax`, `matmul` | 350 |
| `api_v2/params.py` | numpy param dataclasses + `zeros_like(spec, hidden, dtype)` factory | 220 |
| `api_v2/functional.py` | forward functions for Qwen3 path only (STANDARD attention + SwiGLU + RMSNorm + RoPE + QK-norm) | 350 |
| `api_v2/block.py` | `decoder_block_forward` + dispatcher (only PRE-norm path in Phase 1) | 150 |
| `api_v2/kvcache.py` | `KVCacheView`, `KVCacheUpdate`, contiguous layout only | 130 |
| `configs/__init__.py` | imports each family module on demand | 10 |
| `configs/qwen3.py` | per §4 sketch | 110 |
| `loaders/__init__.py` | — | 5 |
| `loaders/hf.py` | per §5 sketch | 150 |
| `tests/parity/__init__.py` | — | 0 |
| `tests/parity/qwen3/test_numpy_vs_hf.py` | the gate | 80 |

**Acceptance gate.** A single pytest:

```
tests/parity/qwen3/test_numpy_vs_hf.py::test_qwen3_0_6b_layer_0_numpy_vs_hf_eager
```

- Loads `Qwen/Qwen3-0.6B` via `AutoModelForCausalLM.from_pretrained` (HF cache).
- Pulls layer 0 weights into a `DecoderBlockParams` tree via `loaders.hf.load_hf_state_dict`.
- Runs numpy forward on a fixed input `x = np.random.default_rng(42).standard_normal((1, 16, 1024)).astype(np.float32)`.
- Runs HF torch eager forward on the same `x`.
- Asserts `np.allclose(out_numpy, out_torch.numpy(), atol=1e-3, rtol=1e-3)`.
- Wall-time target: **≤30s** on the Win11 box (numpy fp32, no GPU). If higher, profile `sdpa` and `matmul` first — those are the hot paths.

A second pytest exercises the same block at S=128 with `atol=2e-3` (looser bound for the longer sequence) — non-blocking, informational.

---

## 10. IR coverage gap fixes

The audit identified eight gaps in v1's `specs.py` that v2 fixes at port time. Each is a small, evidence-cited spec addition.

| Addition | Type | Why | Evidence |
|---|---|---|---|
| `AttentionSpec.output_gate: Literal["sigmoid"] \| None` | new field | Qwen3.5-35B-A3B and Qwen3.6-* use a sigmoid output gate; the q_proj is sized 2× hidden to carry both Q and the gate's pre-activation | research/02-layer-sources.v3.md §Qwen3.5 / Qwen3.6 |
| `RoPESpec.nope_layer_pattern: tuple[bool, ...] \| None` | new field | Llama 4 iRoPE: per-layer NoPE alternation (every 4th layer skips RoPE) | `modeling_llama4.py` config field `nope_layer_interval`, research/02-layer-sources.v3.md §Llama4 |
| `TokenMixerKind.WKV7` | new enum value | RWKV-7 WKV recurrence (not Mamba, not Mamba-2, not GatedDeltaNet — distinct math) | `modeling_rwkv7.py`, research/05-kvcache-attention.v3.md §RWKV-7 |
| `TokenMixerKind.LIGHTNING` | new enum value | MiniMax-Text-01 Lightning Attention (7:1 hybrid with full attention) | `modeling_minimax_text_01.py`, research/05-kvcache-attention.v3.md §MiniMax Lightning |
| `SSMKind.MAMBA3` | new enum value | Mamba-3 complex-valued state cache + MIMO decoding (Mar 2026) | research/02-layer-sources.v3.md §Mamba-3 |
| `QDType.NVFP4` | new enum value | NVIDIA NVFP4 — separate group size (16) from MXFP4 (32) | research/04-quantization.v3.md §NVFP4 |
| `MTPHeadSpec` (new dataclass) | new spec | Multi-token prediction head used by DeepSeek-V3, Qwen3-Next, Qwen3.5/3.6. Lives outside `DecoderBlockSpec` (a model-level spec) | `modeling_deepseek_v3.py` §MTP, `modeling_qwen3_next.py` |
| `KVCacheSpec.cross_block_sharing: bool` | new field | Apple AFM 2-block split (Block-1 carries full KV, Block-2 reuses) and YOCO producer/consumer fall under one boolean + an external block-membership table | research/05-kvcache-attention.v3.md §AFM, §YOCO |

These are *additive* and backward-compatible: every existing Phase-1 Qwen3 path takes the defaults and works unchanged. The new enum values and dataclasses are exercised by Phase 4 families.

The audit also confirmed two non-gaps: `AttentionSpec.attention_k_eq_v` (Gemma 4) and `RoPESpec.partial_rotary_factor` are already present in v1 (`api/specs.py:196`, `:139`). Port them verbatim.

---

## 11. Open questions for user review

**Q1. `configs/` granularity — per-family or per-size?** v1's `models/<family>/config.py` is per-family with a single `from_hf_dict` that adapts. The §4 sketch keeps the per-family model and adds optional named constructors for sizes that materially differ (`gemma4.e2b()`, `gemma4.e4b()`, ...). Alternative: one config file per (family, size). **Recommendation: per-family** with named constructors only for the families with structurally different sizes (Gemma 4, DeepSeek V2/V3/V4, Phi-3 short/long). Confirm?

**Q2. `ml_dtypes` dependency.** Adds one PyPI dep (~50KB, no transitive deps beyond numpy). Alternative: roll our own bf16 ↔ uint16 view helpers and skip FP8 entirely until Phase 5. **Recommendation: add `ml_dtypes`** — the maintenance burden of a hand-rolled bf16 dtype that round-trips correctly through HF state_dicts is non-trivial. Confirm?

**Q3. Tag v1 before v2 begins?** A `v7-complete` git tag (or equivalent) on v1 gives an easy revert point if v2's numpy fidelity proves insufficient and we need to walk back. **Recommendation: yes, tag before Phase 0**. Confirm?

**Q4. Long-term fate of `runtimes/torch/`.** After v2 stabilises, the parity tests are the only consumer of the torch lane. Two options: (a) delete `runtimes/torch/` and keep the wrap-at-boundary pattern from §6; (b) preserve `runtimes/torch/modules.py` as a public surface for downstream torch users who want the IR with torch tensors. **Recommendation: (a) delete** — keep one implementation, force torch users to write a 100-LOC wrapper if they ever appear. Confirm?

**Q5. Compute dtype for Phase 1.** Phase 1 uses fp32 for compute even when weights are bf16 (loaded then upcast). Alternative: do compute in bf16 via `ml_dtypes` for closer numerical match to torch. **Recommendation: fp32 compute, bf16 storage** — matches HF's default (`torch_dtype=bf16` for storage, fp32 accumulator in many ops); easier to bound numerics. Confirm?

---

## 12. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| numpy reference is slow enough to be unusable in CI | high (it will be slow) | Acceptance is for a *reference IR*. Parity tests run at S=16. CI runs a small set of single-block tests, not full models. If S=16 single-block exceeds 30s, profile `sdpa` and consider a numpy einsum-fused path |
| bf16 numerics in `ml_dtypes` diverge from torch CPU bf16 enough to fail `atol=1e-3` | medium | Phase 1 gate uses fp32 compute (only storage is bf16). If a future bf16-compute lane is added, expect to relax to `atol=2e-3`, same as v1 already does on longer sequences |
| Quant ports take longer than estimated (AWQ INT4 + GGUF Q4_K_M bit-packing in numpy is tedious) | high | Phase 5 is deferred until structural primitives are proven. v1's `api/quant.py` (838 LOC) is the upper bound on porting work |
| Selective-scan / SSD reference scales O(S·d_state·d_inner) in pure numpy — Mamba families could be O(seconds) at S=512 | medium | Phase 4 Mamba tests run at S=32; the chunk-parallel SSD path is the optimisation target if needed. Reference correctness > speed |
| Some HF state_dicts contain non-standard nested keys (Gemma 4 PLE, GPT-OSS sinks) | medium | `HF_NAME_MAP` is per-family — those families ship a richer map. `_resolve_slot` returns `None` for optional paths and the loader treats that as "not in this family's params tree" |
| `ml_dtypes` API surface changes break the bf16 view helper in `_as_numpy` | low | Pin a known-working version (`ml_dtypes >= 0.5, < 1.0`). The bf16 numpy view is the only sensitive API point — wrap it in a single helper to localise breakage |
| Phase-0 renames touch v1 and inadvertently break v1 tests | low | Phase 0 is mechanical (`grep -l` + `sed`). Run the full v1 test suite before merge. If a test fails, the rename touched a non-mechanical surface that needs a real fix — investigate, don't bypass |
| `configs/<family>.py` proliferation makes the dispatcher unwieldy | low | 38 modules of ~100 LOC each is ~4000 LOC of pure data — comparable to the v1 `models/<family>/config.py + layer.py` total (47 dirs × ~190 LOC = ~9000 LOC). v2 is *smaller* code in absolute terms |

---

## 13. Out of scope for this spec (explicit)

- **Embedding / unembedding layers.** v1's `api/embedding.py` (63 LOC) is trivial; ports in Phase 1 if needed. Not architecturally interesting.
- **Tokenizer / sampler.** Inherited v3 non-goal.
- **Full-model assembly.** v2 still focuses on the decoder block. A `models/` equivalent for full-model wiring is a Phase 7 (if ever).
- **OpenVINO lowering.** Reserved as `runtimes/openvino/`; not scheduled in this spec.
- **GPU backends.** numpy-only.
- **Phase-0 rename detail.** Already mechanical; sequenced separately. Symbol list: `Llama3RoPEParams`, `Mamba1Mixer`, `Mamba2Mixer`, `_Qwen3NextRMSNormGated` (file:line in §1.1).

---

## 14. Appendix A — v1 file references audited for this spec

| v1 path | What v2 borrows / replaces |
|---|---|
| `api/__init__.py` (9 LOC) | replaced by `api_v2/__init__.py` |
| `api/types.py` (211 LOC) | ported verbatim + 4 new enum values (§10) |
| `api/specs.py` (733 LOC) | ported with torch.dtype → DType swap + 4 new fields/dataclasses (§10) |
| `api/ops.py` (866 LOC) | torch → numpy rewrite; signatures preserved where possible |
| `api/norm.py` (115 LOC) | folded into `api_v2/functional.py::rms_norm_forward`, `layer_norm_forward` |
| `api/rope.py` (406 LOC) | folded into `api_v2/functional.py::rope_apply` + `cos_sin_*` helpers; family naming dropped from RoPE variant table |
| `api/attention.py` (811 LOC) | folded into `api_v2/functional.py::attention_forward` (dispatches on `AttentionSpec.kind`) |
| `api/feedforward.py` (750 LOC) | folded into `api_v2/functional.py::{ffn_forward, moe_forward}` |
| `api/block.py` (349 LOC) | folded into `api_v2/block.py::decoder_block_forward` |
| `api/kvcache.py` (222 LOC) | ported to `api_v2/kvcache.py` (numpy-backed; layout dispatch via spec) |
| `api/embedding.py` (63 LOC) | deferred (not block-internal) |
| `api/ssm.py` (723 LOC) | family symbols renamed; folded into `api_v2/functional.py::{selective_scan_mixer_forward, ssd_mixer_forward, gated_rms_norm_forward}` |
| `api/quant.py` (838 LOC) | deferred to Phase 5; numpy + ml_dtypes rewrite |
| `models/qwen3/config.py` (120 LOC), `models/qwen3/layer.py` (74 LOC) | collapsed to `configs/qwen3.py` (~110 LOC) |
| `tests/api/` (~16 test modules) | ported to `tests/api_v2/` in Phase 3 |
| `tests/models/<family>/` | merges into `tests/parity/<family>/` |

Total v1 IR LOC: 6,096. Estimated v2 IR LOC after all phases: ~3,800 (the `nn.Module` machinery is the LOC sink; pure functions are smaller).

---

## 15. Appendix B — neutral-name dictionary (P2 enforcement)

Quick reference for code review: family names that must NOT appear inside `api_v2/` as symbols.

| Banned in `api_v2/` symbols | Reason | Where it CAN appear |
|---|---|---|
| `Llama`, `Llama3`, `Llama4` | family | `configs/llama3.py`, `configs/llama4_scout.py`, citation comments |
| `Qwen`, `Qwen3`, `Qwen3Next` | family | `configs/qwen3*.py`, citation comments |
| `Gemma`, `Gemma2`, `Gemma3`, `Gemma4` | family | `configs/gemma*.py`, citation comments |
| `Mamba`, `Mamba1`, `Mamba2`, `Mamba3` | family | `configs/mamba*.py`, citation comments. Note: `MAMBA1`/`MAMBA2_SSD`/`MAMBA3` as **enum values** on `SSMKind` are OK (they name the *algorithm*, not the model family — analogous to `SWIGLU` or `RMS`) |
| `DeepSeek`, `DeepSeekV2`, `DeepSeekV3`, `DeepSeekV4` | family | `configs/deepseek_v*.py`, citation comments |
| `BitNet`, `Falcon`, `MPT`, `Phi`, `Mistral`, `Mixtral`, `OLMo`, `OLMoE` | family | `configs/<family>.py`, citation comments |
| `Hunyuan`, `Granite`, `Hymba`, `Jamba`, `Nemotron`, `MiniMax`, `MiniCPM`, `SmolLM`, `Tinyllama` | family | `configs/<family>.py`, citation comments |
| `RWKV` | family (also algorithm, but treat as family for P2) | `configs/rwkv7.py`, citation comments. Use `WKV7` as enum value (algorithm) |
| `Lightning` (attention) | family-coupled | use `TokenMixerKind.LIGHTNING` enum (algorithm name OK); no class names |
| `YaRN`, `LongRoPE`, `ALiBi`, `RoPE`, `NoPE` | algorithm names, not families | OK as class names (`YarnRoPEParams`, `LongRoPEParams`, `AliBiSpec`) |
| `SwiGLU`, `GeGLU`, `ReLU2` | algorithm names | OK |
| `SDPA`, `MLA`, `MoE`, `SSM`, `SSD`, `DSA`, `CSA`, `HCA` | algorithm acronyms | OK |
| `MTP` (multi-token prediction) | algorithm acronym (introduced by DeepSeek-V3, generalised) | OK as `MTPHeadSpec` |

A CI lint can grep `api_v2/` for any of the family names and fail. Recommended.
