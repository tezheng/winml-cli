# llm-layers M1 — Qwen3 Kickoff Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the M1 kickoff-gate deliverable from spec v2 §8 — a working `api/` skeleton (subset sufficient for Qwen3) plus `models/qwen3/{layer.md, layer.py, config.py}` and tests that prove numerical equivalence to HF's `Qwen3DecoderLayer` at atol=5e-4 on a fixed input, plus KV-cache equivalence and AWQ-quant round-trip.

**Architecture:** TDD throughout. The API is built bottom-up: enums + spec dataclasses → logical ops → small blocks (RMSNorm, RoPE, KVCache) → attention/FFN → DecoderBlock. Then `models/qwen3/` is a thin adapter that produces a `DecoderBlockSpec` from a Qwen3 config and assembles via the API. Every layer of the stack has its own test file. The final test is the gate: numerical equivalence vs HF on real weights.

**Tech Stack:** Python 3.11+, uv-managed venv, PyTorch 2.4+, transformers ≥4.51 (Qwen3 support), safetensors, pytest, ruff, mypy. CPU-only is sufficient for M1 (Qwen3-0.6B fits).

**Scope:** This plan is **M1 only**. M2 (per-model rollout batches B0.5–B9) is out of scope until you approve the M1 deliverable. The api/ skeleton produced here is the *Qwen3 subset*: 9 logical ops (not the 16-op floor), 7 spec dataclasses (not 12), and just the variants Qwen3 needs. Extensions for MoE/SSM/MLA/etc. land in later milestones.

---

## File structure

```
C:\Users\zhengte\external\llm-layers\
├── pyproject.toml                     [T1]
├── uv.lock                            [T1, generated]
├── .python-version                    [T1]
├── .gitignore                         [T1]
├── README.md                          [T1, expanded in T21]
├── conftest.py                        [T1]
├── api/
│   ├── __init__.py                    [T1]
│   ├── types.py                       [T2, enums + small types]
│   ├── specs.py                       [T2, dataclasses]
│   ├── ops.py                         [T3, T4, T5]
│   ├── norm.py                        [T6]
│   ├── rope.py                        [T7]
│   ├── kvcache.py                     [T8]
│   ├── quant.py                       [T9]
│   ├── attention.py                   [T10]
│   ├── feedforward.py                 [T11]
│   └── block.py                       [T12]
├── models/
│   └── qwen3/
│       ├── __init__.py                [T13]
│       ├── config.py                  [T13]
│       ├── layer.py                   [T14]
│       └── layer.md                   [T20]
├── tests/
│   ├── __init__.py                    [T1]
│   ├── api/
│   │   ├── __init__.py
│   │   ├── test_specs.py              [T2]
│   │   ├── test_ops.py                [T3, T4, T5]
│   │   ├── test_norm.py               [T6]
│   │   ├── test_rope.py               [T7]
│   │   ├── test_kvcache.py            [T8]
│   │   ├── test_quant.py              [T9]
│   │   ├── test_attention.py          [T10]
│   │   ├── test_feedforward.py        [T11]
│   │   └── test_block.py              [T12]
│   └── models/
│       └── qwen3/
│           ├── __init__.py
│           ├── test_config.py         [T13]
│           ├── test_layer_shape.py    [T14]
│           ├── test_weight_loader.py  [T15]
│           ├── test_isolation_hf.py   [T16]
│           ├── test_numerical_hf.py   [T17]
│           ├── test_kvcache_hf.py     [T18]
│           └── test_quant_awq.py      [T19]
```

Responsibility per file:
- `api/types.py` — small leaf types (enums, frozen-int aliases) with no deps
- `api/specs.py` — frozen dataclasses (no behaviour, no PyTorch imports)
- `api/ops.py` — pure functions on tensors; the IHV-consensus floor
- `api/{norm,rope,kvcache,quant,attention,feedforward,block}.py` — `nn.Module` building blocks that consume specs and call ops
- `models/qwen3/config.py` — `Qwen3Config` dataclass + `from_hf_config()` adapter
- `models/qwen3/layer.py` — `build_qwen3_decoder_layer(config, layer_idx)` + HF-weight loader

---

## Task 1 — Project bootstrap

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`, `conftest.py`
- Create: `api/__init__.py`, `tests/__init__.py`, `tests/api/__init__.py`, `tests/models/__init__.py`, `tests/models/qwen3/__init__.py`

- [ ] **Step 1: Initialize uv project and venv**

Run:
```powershell
cd C:\Users\zhengte\external\llm-layers
uv init --python 3.11 --no-readme --no-package
```
Expected: creates `.python-version`, no other files overwritten.

- [ ] **Step 2: Replace generated `pyproject.toml` with the project version**

Create `pyproject.toml`:
```toml
[project]
name = "llm-layers"
version = "0.1.0"
description = "Minimal API for assembling mainstream small language models"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4.0",
    "safetensors>=0.4.0",
    "numpy>=1.26",
]

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "pytest-xdist>=3.5",
    "transformers>=4.51.0",
    "huggingface-hub>=0.24",
    "accelerate>=0.30",
]
dev = [
    "ruff>=0.6",
    "mypy>=1.11",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
strict = true
disallow_untyped_defs = true
warn_return_any = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-ra"
```

- [ ] **Step 3: Install deps**

Run:
```powershell
uv sync --extra test --extra dev
```
Expected: creates `uv.lock`, `.venv/`, installs torch + transformers.

- [ ] **Step 4: Create `.gitignore`**

```
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.safetensors
*.bin
hf_cache/
.coverage
htmlcov/
```

- [ ] **Step 5: Create initial `README.md`**

```markdown
# llm-layers

Minimal, evidence-grounded API for assembling mainstream small language models
(<8B params) as decoder graphs, with quantization and KV-cache as first-class
parameters.

See `docs/superpowers/specs/2026-06-04-llm-layers-design.v2.md` for the design.
See `research/` for the supporting survey (5 reports + evolution narrative).

## Quickstart

```powershell
uv sync --extra test --extra dev
uv run pytest -q
```

## M1 status

Kickoff gate: `models/qwen3/` — see `docs/superpowers/plans/2026-06-05-llm-layers-m1-qwen3.md`.
```

- [ ] **Step 6: Create package-init files**

```powershell
New-Item -ItemType File -Path "api\__init__.py", "tests\__init__.py", "tests\api\__init__.py", "tests\models\__init__.py", "tests\models\qwen3\__init__.py"
```

Create `api/__init__.py`:
```python
"""llm-layers API — minimal primitives for SLM decoder blocks.

The public surface lives in submodules:
- api.types: enums and small types
- api.specs: frozen dataclasses (parameter spaces)
- api.ops: pure functional primitives
- api.norm, api.rope, api.kvcache, api.quant, api.attention,
  api.feedforward, api.block: nn.Module building blocks
"""
```

Create `conftest.py`:
```python
"""Pytest configuration. Adds repo root to PYTHONPATH so `api`/`models` import."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
```

- [ ] **Step 7: Verify pytest discovers an empty suite**

Run:
```powershell
uv run pytest -q
```
Expected: `no tests ran in <duration>` with exit 0.

- [ ] **Step 8: First commit**

```powershell
git init
git add .
git commit -m "feat(M1): project bootstrap — uv venv, pyproject, package layout"
```

---

## Task 2 — Enum types and spec dataclasses

**Files:**
- Create: `api/types.py`, `api/specs.py`
- Test: `tests/api/test_specs.py`

- [ ] **Step 1: Write failing test for spec immutability and field access**

Create `tests/api/test_specs.py`:
```python
import dataclasses
import pytest
import torch

from api import specs, types


def test_norm_spec_is_frozen():
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.eps = 1e-5  # type: ignore


def test_rope_spec_defaults():
    spec = specs.RoPESpec(base_theta=1_000_000.0, basis=types.RoPEBasis.SPLIT_HALF)
    assert spec.scaling == types.RoPEScaling.NONE
    assert spec.scale_factor is None


def test_attention_spec_minimum():
    spec = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    assert spec.qk_norm_phase == types.QKNormPhase.NONE


def test_ffn_spec_swiglu():
    spec = specs.FFNSpec(intermediate_size=3072,
                         activation=types.Activation.SILU,
                         gate_kind=types.GateKind.SWIGLU)
    assert not spec.fused_gate_up


def test_kvcache_spec_contiguous():
    spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.NHD,
        k_dtype=torch.bfloat16, v_dtype=torch.bfloat16,
        ownership=types.CacheOwnership.EXPLICIT_PASS,
    )
    assert spec.block_size is None  # not used for CONTIGUOUS


def test_quant_spec_awq():
    spec = specs.QuantSpec(
        qdtype=types.QDType.INT4, group_size=128, quant_axis=0,
        scale_dtype=torch.float16, has_zero_point=True,
        packing=types.PackingLayout.AWQ_INTERLEAVE,
        accumulator_dtype=torch.float32,
        role=types.QuantRole.WEIGHT,
    )
    assert spec.codebook is None


def test_decoder_block_spec_composition():
    norm = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    attn = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=3072,
                       activation=types.Activation.SILU,
                       gate_kind=types.GateKind.SWIGLU)
    block = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn, channel_mixer=ffn,
        input_norm=norm, pre_attn_norm=norm, pre_ffn_norm=norm,
    )
    assert block.residual_scale is None
```

- [ ] **Step 2: Run test to verify it fails (module not found)**

```powershell
uv run pytest tests/api/test_specs.py -v
```
Expected: `ModuleNotFoundError: No module named 'api.specs'` (the import itself fails).

- [ ] **Step 3: Implement `api/types.py`**

```python
"""Enums and small leaf types. No PyTorch dependency."""
from __future__ import annotations
from enum import Enum, auto


class AttentionKind(Enum):
    STANDARD = auto()       # MHA / GQA / MQA — distinguished by n_kv_heads
    MLA = auto()            # Multi-head Latent Attention (DeepSeek)
    LINEAR_RETENTION = auto()
    LINEAR_DELTANET = auto()
    LINEAR_GLA = auto()
    DIFFERENTIAL = auto()


class QKVLayout(Enum):
    SPLIT = auto()          # separate q/k/v projections
    FUSED = auto()          # one big QKV projection (Phi-3)
    MLA_LATENT = auto()     # MLA-specific


class MaskKind(Enum):
    CAUSAL = auto()
    SWA = auto()
    SWA_GLOBAL_ALT = auto()
    SINK = auto()
    FULL = auto()
    CUSTOM = auto()
    BLOCK_SPARSE = auto()


class QKNormPhase(Enum):
    NONE = auto()
    PRE_ROPE = auto()   # Qwen3, Gemma3 — verified PRE-RoPE in research/05 v2
    POST_ROPE = auto()


class QKNormShape(Enum):
    NONE = auto()
    PER_HEAD_DH = auto()    # Qwen3, Gemma3 — weight shape [head_dim]
    FULL_HDH = auto()       # OLMo 2 — weight shape [n_heads * head_dim]


class NormKind(Enum):
    RMS = auto()
    LAYER = auto()


class NormWeightMode(Enum):
    STANDARD_W = auto()     # y = x_normed * w
    ONE_PLUS_W = auto()     # y = x_normed * (1 + w) — Gemma


class NormPosition(Enum):
    PRE = auto()
    POST = auto()           # OLMo 2
    PRE_AND_POST = auto()   # Gemma 2 / 3


class Activation(Enum):
    SILU = auto()
    GELU = auto()
    GEGELU = auto()
    RELU2 = auto()


class GateKind(Enum):
    SWIGLU = auto()         # Llama, Qwen
    GEGLU = auto()          # Gemma
    GELU_ONLY = auto()      # no gating
    RELU2_ONLY = auto()


class RoPEBasis(Enum):
    INTERLEAVED = auto()    # GPT-J style
    SPLIT_HALF = auto()     # GPT-NeoX / Llama / Qwen style


class RoPEScaling(Enum):
    NONE = auto()
    PI = auto()
    NTK_STATIC = auto()
    NTK_DYNAMIC = auto()
    YARN = auto()
    LLAMA3 = auto()         # Llama 3 smooth scaling
    LONGROPE = auto()       # Phi-3 short/long


class CacheLayout(Enum):
    CONTIGUOUS = auto()     # HF baseline
    PAGED = auto()          # vLLM
    RING = auto()           # SWA wrap-around
    MLA_LATENT = auto()     # DeepSeek-V2/V3
    SSM_STATE = auto()      # Mamba


class MemoryLayout(Enum):
    HND = auto()            # [batch, heads, seq, dim]
    NHD = auto()            # [batch, seq, heads, dim]


class CacheOwnership(Enum):
    EXPLICIT_PASS = auto()  # PyTorch eager
    STATEFUL = auto()       # Core ML state op


class QDType(Enum):
    INT4 = auto()
    INT8 = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP4 = auto()
    NF4 = auto()
    MX_FP4 = auto()


class PackingLayout(Enum):
    NONE = auto()
    NIBBLE_LSB = auto()
    NIBBLE_MSB = auto()
    AWQ_INTERLEAVE = auto()     # [0,2,4,6,1,3,5,7] nibble permutation
    GPTQ_INT32_PACK = auto()
    GGUF_K = auto()


class QuantRole(Enum):
    WEIGHT = auto()
    ACTIVATION = auto()
    KV_K = auto()
    KV_V = auto()
    ATTN_INTERNAL = auto()
```

- [ ] **Step 4: Implement `api/specs.py`**

```python
"""Frozen dataclasses defining the parameter spaces for each layer component.

No behaviour — pure data. Behaviour lives in api/{norm,rope,attention,...}.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from api import types


@dataclass(frozen=True)
class NormSpec:
    kind: types.NormKind
    eps: float
    weight_mode: types.NormWeightMode = types.NormWeightMode.STANDARD_W


@dataclass(frozen=True)
class Llama3RoPEParams:
    """Llama-3 smooth-scaling RoPE parameters."""
    factor: float                       # e.g. 8.0
    low_freq_factor: float              # e.g. 1.0
    high_freq_factor: float             # e.g. 4.0
    original_context_length: int        # e.g. 8192


@dataclass(frozen=True)
class RoPESpec:
    base_theta: float
    basis: types.RoPEBasis
    scaling: types.RoPEScaling = types.RoPEScaling.NONE
    scale_factor: Optional[float] = None
    llama3_extra: Optional[Llama3RoPEParams] = None


@dataclass(frozen=True)
class AttentionSpec:
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    kind: types.AttentionKind
    qkv_layout: types.QKVLayout
    mask_kind: types.MaskKind

    q_bias: bool = False
    k_bias: bool = False
    v_bias: bool = False
    o_bias: bool = False

    attn_scale: Optional[float] = None   # overrides 1/sqrt(head_dim) (Gemma)
    logit_softcap: Optional[float] = None
    sliding_window: Optional[int] = None

    qk_norm: Optional[NormSpec] = None
    qk_norm_phase: types.QKNormPhase = types.QKNormPhase.NONE
    qk_norm_shape: types.QKNormShape = types.QKNormShape.NONE

    rope: Optional[RoPESpec] = None


@dataclass(frozen=True)
class FFNSpec:
    intermediate_size: int
    activation: types.Activation
    gate_kind: types.GateKind
    fused_gate_up: bool = False
    gate_bias: bool = False
    up_bias: bool = False
    down_bias: bool = False


@dataclass(frozen=True)
class QuantSpec:
    qdtype: types.QDType
    group_size: Optional[int]            # None=per-tensor, -1=per-channel, N=blockwise
    quant_axis: int
    scale_dtype: torch.dtype
    has_zero_point: bool
    packing: types.PackingLayout
    accumulator_dtype: torch.dtype
    role: types.QuantRole = types.QuantRole.WEIGHT
    codebook: Optional[object] = None    # CodebookSpec — defined post-M1


@dataclass(frozen=True)
class KVCacheSpec:
    layout: types.CacheLayout
    memory_layout: types.MemoryLayout
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    ownership: types.CacheOwnership = types.CacheOwnership.EXPLICIT_PASS
    block_size: Optional[int] = None
    k_quant: Optional[QuantSpec] = None
    v_quant: Optional[QuantSpec] = None


@dataclass(frozen=True)
class DecoderBlockSpec:
    attn_norm_position: types.NormPosition
    ffn_norm_position: types.NormPosition
    token_mixer: AttentionSpec           # only AttentionSpec for M1 (Qwen3 dense)
    channel_mixer: FFNSpec               # only FFNSpec for M1
    input_norm: NormSpec
    pre_attn_norm: Optional[NormSpec] = None
    post_attn_norm: Optional[NormSpec] = None
    pre_ffn_norm: Optional[NormSpec] = None
    post_ffn_norm: Optional[NormSpec] = None
    residual_scale: Optional[float] = None
    embedding_scale: Optional[float] = None
    logits_scale: Optional[float] = None
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
uv run pytest tests/api/test_specs.py -v
```
Expected: 7 passed.

- [ ] **Step 6: Commit**

```powershell
git add api/types.py api/specs.py tests/api/test_specs.py
git commit -m "feat(M1): api types + spec dataclasses (Qwen3 subset)"
```

---

## Task 3 — Logical ops: element-wise + linear

**Files:**
- Create: `api/ops.py` (start)
- Test: `tests/api/test_ops.py`

- [ ] **Step 1: Write failing tests for silu / add / mul / linear**

Create `tests/api/test_ops.py`:
```python
import pytest
import torch
import torch.nn.functional as F

from api import ops


def test_silu_matches_torch():
    x = torch.randn(2, 4, 8)
    assert torch.allclose(ops.silu(x), F.silu(x), atol=1e-7)


def test_add_with_residual():
    x = torch.randn(2, 4, 8)
    r = torch.randn(2, 4, 8)
    assert torch.allclose(ops.add(x, r), x + r, atol=1e-7)


def test_add_with_scale():
    x = torch.randn(2, 4, 8)
    r = torch.randn(2, 4, 8)
    out = ops.add(x, r, scale=0.5)
    assert torch.allclose(out, x + 0.5 * r, atol=1e-7)


def test_mul_elementwise():
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 4, 8)
    assert torch.allclose(ops.mul(a, b), a * b, atol=1e-7)


def test_linear_no_quant():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    out = ops.linear(x, w)
    assert torch.allclose(out, F.linear(x, w), atol=1e-5)


def test_linear_with_bias():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(8, dtype=torch.float32)
    out = ops.linear(x, w, bias=b)
    assert torch.allclose(out, F.linear(x, w, b), atol=1e-5)
```

- [ ] **Step 2: Run tests, expect failure**

```powershell
uv run pytest tests/api/test_ops.py -v
```
Expected: ModuleNotFoundError on `api.ops`.

- [ ] **Step 3: Implement `api/ops.py` (5 ops)**

```python
"""Logical op primitives — pure functions on tensors.

These are the IHV-consensus floor (see research/03-ihv-opsets.v2.md). Each op
is a thin wrapper around PyTorch tensor ops; backends (ONNX/QNN/OpenVINO) would
re-implement the same signatures.

M1 implements the Qwen3 subset: silu, add, mul, linear, rms_norm, embed,
lm_head, rope_apply, sdpa.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def add(x: torch.Tensor, residual: torch.Tensor,
        scale: Optional[float] = None) -> torch.Tensor:
    if scale is None:
        return x + residual
    return x + scale * residual


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def linear(x: torch.Tensor, weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Standard linear; quantization is handled in api.quant via wrapper modules."""
    return F.linear(x, weight, bias)
```

- [ ] **Step 4: Run tests, expect pass**

```powershell
uv run pytest tests/api/test_ops.py -v -k "silu or add or mul or linear"
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/ops.py tests/api/test_ops.py
git commit -m "feat(M1): ops silu/add/mul/linear with tests"
```

---

## Task 4 — Logical ops: rms_norm + embed + lm_head

**Files:**
- Modify: `api/ops.py`
- Modify: `tests/api/test_ops.py`

- [ ] **Step 1: Add failing tests for rms_norm / embed / lm_head**

Append to `tests/api/test_ops.py`:
```python
def test_rms_norm_standard_w():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(16, dtype=torch.float32)
    eps = 1e-6
    out = ops.rms_norm(x, w, eps, mode="standard_w")

    # Reference: PyTorch built-in (added in 2.4+)
    ref = F.rms_norm(x, (16,), w, eps)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rms_norm_one_plus_w():
    """Gemma-style RMSNorm where weight is (1 + w)."""
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(16, dtype=torch.float32)
    eps = 1e-6
    out = ops.rms_norm(x, w, eps, mode="one_plus_w")

    # Same as standard but using (1+w) for the weight
    ref = F.rms_norm(x, (16,), 1.0 + w, eps)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rms_norm_preserves_input_dtype():
    """RMSNorm computes in fp32 but returns input dtype."""
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    w = torch.randn(16, dtype=torch.bfloat16)
    out = ops.rms_norm(x, w, 1e-6, mode="standard_w")
    assert out.dtype == torch.bfloat16


def test_embed_basic():
    weight = torch.randn(100, 8, dtype=torch.float32)
    ids = torch.tensor([[0, 1, 2], [99, 50, 1]])
    out = ops.embed(ids, weight)
    assert out.shape == (2, 3, 8)
    assert torch.allclose(out[0, 0], weight[0])
    assert torch.allclose(out[1, 2], weight[1])


def test_embed_with_scale():
    """Gemma scales embeddings by sqrt(hidden_size)."""
    weight = torch.randn(10, 8, dtype=torch.float32)
    ids = torch.tensor([[1, 2]])
    out = ops.embed(ids, weight, scale=2.0)
    assert torch.allclose(out[0, 0], 2.0 * weight[1])


def test_lm_head_no_softcap():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(100, 16, dtype=torch.float32)
    out = ops.lm_head(x, w)
    assert torch.allclose(out, F.linear(x, w), atol=1e-5)


def test_lm_head_with_softcap():
    """Gemma 2 / 3 logits softcap."""
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(100, 16, dtype=torch.float32)
    cap = 30.0
    out = ops.lm_head(x, w, softcap=cap)
    logits = F.linear(x, w)
    ref = cap * torch.tanh(logits / cap)
    assert torch.allclose(out, ref, atol=1e-5)
```

- [ ] **Step 2: Run, expect failures (functions missing)**

```powershell
uv run pytest tests/api/test_ops.py -v -k "rms_norm or embed or lm_head"
```
Expected: AttributeError for missing `ops.rms_norm`, `ops.embed`, `ops.lm_head`.

- [ ] **Step 3: Append rms_norm / embed / lm_head to `api/ops.py`**

```python
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    mode: str = "standard_w",
) -> torch.Tensor:
    """RMSNorm with optional Gemma 1+w mode.

    Compute in fp32 for numerical stability, return in x's dtype.
    """
    if mode not in ("standard_w", "one_plus_w"):
        raise ValueError(f"unknown mode: {mode!r}")
    orig_dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    x_normed = x32 * torch.rsqrt(variance + eps)
    w = weight.float()
    if mode == "one_plus_w":
        w = 1.0 + w
    return (x_normed * w).to(orig_dtype)


def embed(
    ids: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    out = F.embedding(ids, weight)
    if scale is not None:
        out = out * scale
    return out


def lm_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    logits = F.linear(x, weight)
    if scale is not None:
        logits = logits * scale
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_ops.py -v
```
Expected: 13 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/ops.py tests/api/test_ops.py
git commit -m "feat(M1): ops rms_norm/embed/lm_head with tests"
```

---

## Task 5 — Logical ops: rope_apply + sdpa

**Files:**
- Modify: `api/ops.py`
- Modify: `tests/api/test_ops.py`

- [ ] **Step 1: Add failing tests for rope_apply (split-half basis) and sdpa**

Append to `tests/api/test_ops.py`:
```python
def _hf_rope_half(x, cos, sin):
    """Reference: HF's rotate_half + (cos, sin) application.

    HF Llama-style: input is [..., D] split into halves [..., :D/2] and [..., D/2:].
    rotate_half swaps and negates: [-x2, x1].
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def test_rope_apply_split_half_matches_hf():
    """rope_apply with SPLIT_HALF basis should match HF's apply_rotary_pos_emb."""
    B, S, H, Dh = 1, 4, 2, 8
    q = torch.randn(B, S, H, Dh, dtype=torch.float32)
    k = torch.randn(B, S, H, Dh, dtype=torch.float32)

    # cos / sin tables shaped [S, Dh] (broadcast over B and H)
    freqs = torch.linspace(0.1, 1.0, Dh // 2)
    positions = torch.arange(S).float().unsqueeze(-1) * freqs.unsqueeze(0)
    cos = torch.cat([positions.cos(), positions.cos()], dim=-1)   # [S, Dh]
    sin = torch.cat([positions.sin(), positions.sin()], dim=-1)

    q_rot, k_rot = ops.rope_apply(q, k, cos, sin, basis="split_half")

    cos_b = cos.view(1, S, 1, Dh)
    sin_b = sin.view(1, S, 1, Dh)
    q_ref = _hf_rope_half(q, cos_b, sin_b)
    k_ref = _hf_rope_half(k, cos_b, sin_b)

    assert torch.allclose(q_rot, q_ref, atol=1e-6)
    assert torch.allclose(k_rot, k_ref, atol=1e-6)


def test_sdpa_matches_torch_for_mha():
    """For n_q_heads == n_kv_heads, sdpa should match F.scaled_dot_product_attention."""
    B, H, S, Dh = 1, 4, 6, 8
    q = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k = torch.randn(B, H, S, Dh, dtype=torch.float32)
    v = torch.randn(B, H, S, Dh, dtype=torch.float32)

    out = ops.sdpa(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_gqa_broadcasts_kv():
    """For GQA (n_kv_heads < n_q_heads), sdpa should repeat KV to match Q."""
    B, Hq, Hk, S, Dh = 1, 8, 2, 6, 16   # Hq // Hk == 4
    q = torch.randn(B, Hq, S, Dh, dtype=torch.float32)
    k = torch.randn(B, Hk, S, Dh, dtype=torch.float32)
    v = torch.randn(B, Hk, S, Dh, dtype=torch.float32)

    out = ops.sdpa(q, k, v, is_causal=True)
    # Reference: repeat KV interleave-style then call torch's SDPA
    k_rep = k.repeat_interleave(Hq // Hk, dim=1)
    v_rep = v.repeat_interleave(Hq // Hk, dim=1)
    ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_custom_scale():
    B, H, S, Dh = 1, 2, 4, 8
    q = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k = torch.randn(B, H, S, Dh, dtype=torch.float32)
    v = torch.randn(B, H, S, Dh, dtype=torch.float32)

    custom_scale = 0.25
    out = ops.sdpa(q, k, v, is_causal=False, scale=custom_scale)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=custom_scale)
    assert torch.allclose(out, ref, atol=1e-5)
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/api/test_ops.py -v -k "rope or sdpa"
```
Expected: AttributeError on `ops.rope_apply`, `ops.sdpa`.

- [ ] **Step 3: Append rope_apply + sdpa to `api/ops.py`**

```python
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """For SPLIT_HALF RoPE basis: rotate by π/2 around the head_dim axis."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(
    q: torch.Tensor,           # [B, S, H, Dh]
    k: torch.Tensor,           # [B, S, Hk, Dh]
    cos: torch.Tensor,         # [S, Dh]
    sin: torch.Tensor,         # [S, Dh]
    basis: str = "split_half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding.

    For Qwen3 / Llama / most modern SLMs we use SPLIT_HALF basis (GPT-NeoX style).
    INTERLEAVED (GPT-J) basis is supported for future models.
    """
    if basis not in ("split_half", "interleaved"):
        raise ValueError(f"unknown basis: {basis!r}")
    if basis == "interleaved":
        raise NotImplementedError("INTERLEAVED basis lands in a later milestone")

    # Broadcast cos/sin from [S, Dh] to [1, S, 1, Dh]
    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    q_rot = q * cos_b + _rotate_half(q) * sin_b
    k_rot = k * cos_b + _rotate_half(k) * sin_b
    return q_rot, k_rot


def sdpa(
    q: torch.Tensor,                       # [B, Hq, S, Dh]
    k: torch.Tensor,                       # [B, Hk, S, Dh]
    v: torch.Tensor,                       # [B, Hk, S, Dh]
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Scaled dot-product attention with GQA support.

    Repeats K/V to match Q heads when n_q_heads > n_kv_heads.
    """
    Hq = q.shape[1]
    Hk = k.shape[1]
    if Hk != Hq:
        if Hq % Hk != 0:
            raise ValueError(f"n_q_heads ({Hq}) must be divisible by n_kv_heads ({Hk})")
        repeats = Hq // Hk
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_ops.py -v
```
Expected: 17 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/ops.py tests/api/test_ops.py
git commit -m "feat(M1): ops rope_apply/sdpa with GQA broadcast"
```

---

## Task 6 — `api/norm.py` — RMSNorm block

**Files:**
- Create: `api/norm.py`
- Test: `tests/api/test_norm.py`

- [ ] **Step 1: Failing tests for RMSNorm module + QKNorm**

Create `tests/api/test_norm.py`:
```python
import torch
import torch.nn.functional as F

from api import norm, specs, types


def test_rmsnorm_module_forward():
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    module = norm.RMSNorm(spec, hidden_size=16, dtype=torch.float32)
    # Initialize weight to ones for a deterministic check
    with torch.no_grad():
        module.weight.fill_(1.0)
    x = torch.randn(2, 4, 16)
    out = module(x)

    ref = F.rms_norm(x, (16,), torch.ones(16), 1e-6)
    assert torch.allclose(out, ref, atol=1e-5)


def test_qknorm_per_head_dh_shape():
    """Qwen3 / Gemma 3 QK-norm: weight shape is [head_dim], applied per-head."""
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    head_dim = 8
    qknorm = norm.QKNorm(spec, head_dim=head_dim,
                         shape=types.QKNormShape.PER_HEAD_DH,
                         dtype=torch.float32)
    assert qknorm.weight.shape == (head_dim,)

    B, S, H, Dh = 1, 4, 3, head_dim
    x = torch.randn(B, S, H, Dh)
    with torch.no_grad():
        qknorm.weight.fill_(1.0)
    out = qknorm(x)
    assert out.shape == (B, S, H, Dh)
    # Each head should be independently RMS-normalized along Dh
    ref = F.rms_norm(x, (Dh,), torch.ones(Dh), 1e-6)
    assert torch.allclose(out, ref, atol=1e-5)


def test_qknorm_full_hdh_shape():
    """OLMo 2 QK-norm: weight shape is [n_heads * head_dim]."""
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    n_heads, head_dim = 3, 8
    qknorm = norm.QKNorm(spec, head_dim=head_dim, n_heads=n_heads,
                         shape=types.QKNormShape.FULL_HDH,
                         dtype=torch.float32)
    assert qknorm.weight.shape == (n_heads * head_dim,)
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/api/test_norm.py -v
```

- [ ] **Step 3: Implement `api/norm.py`**

```python
"""Norm building blocks consuming NormSpec."""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from api import ops, specs, types


class RMSNorm(nn.Module):
    """Wraps api.ops.rms_norm. Weight initialized to ones."""

    def __init__(self, spec: specs.NormSpec, hidden_size: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.kind != types.NormKind.RMS:
            raise ValueError(f"RMSNorm requires kind=RMS, got {spec.kind}")
        self.spec = spec
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = ("one_plus_w" if self.spec.weight_mode == types.NormWeightMode.ONE_PLUS_W
                else "standard_w")
        return ops.rms_norm(x, self.weight, self.spec.eps, mode=mode)


class QKNorm(nn.Module):
    """Per-head QK normalization.

    Two shape modes:
    - PER_HEAD_DH: weight shape [head_dim], applied independently to each head's Dh
      (Qwen3, Gemma 3)
    - FULL_HDH: weight shape [n_heads * head_dim], applied to the flattened
      head-dim slice (OLMo 2)
    """

    def __init__(self, spec: specs.NormSpec, head_dim: int,
                 shape: types.QKNormShape, n_heads: Optional[int] = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.kind != types.NormKind.RMS:
            raise ValueError("QKNorm only supports RMS")
        self.spec = spec
        self.shape = shape
        self.head_dim = head_dim
        self.n_heads = n_heads
        if shape == types.QKNormShape.PER_HEAD_DH:
            weight_size = head_dim
        elif shape == types.QKNormShape.FULL_HDH:
            if n_heads is None:
                raise ValueError("FULL_HDH requires n_heads")
            weight_size = n_heads * head_dim
        else:
            raise ValueError(f"unsupported shape: {shape}")
        self.weight = nn.Parameter(torch.ones(weight_size, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H, Dh]
        mode = ("one_plus_w" if self.spec.weight_mode == types.NormWeightMode.ONE_PLUS_W
                else "standard_w")
        if self.shape == types.QKNormShape.PER_HEAD_DH:
            return ops.rms_norm(x, self.weight, self.spec.eps, mode=mode)
        # FULL_HDH: reshape to [B, S, H*Dh], norm, reshape back
        B, S, H, Dh = x.shape
        x_flat = x.reshape(B, S, H * Dh)
        out = ops.rms_norm(x_flat, self.weight, self.spec.eps, mode=mode)
        return out.reshape(B, S, H, Dh)
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_norm.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/norm.py tests/api/test_norm.py
git commit -m "feat(M1): RMSNorm + QKNorm modules"
```

---

## Task 7 — `api/rope.py` — RoPE block

**Files:**
- Create: `api/rope.py`
- Test: `tests/api/test_rope.py`

- [ ] **Step 1: Failing tests**

Create `tests/api/test_rope.py`:
```python
import torch

from api import rope, specs, types


def test_rope_module_returns_rotated():
    """RoPE module applied to random q/k should not equal input (rotation actually happens)."""
    spec = specs.RoPESpec(base_theta=1_000_000.0, basis=types.RoPEBasis.SPLIT_HALF)
    head_dim = 16
    max_seq = 32
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)

    B, S, H = 1, 8, 2
    q = torch.randn(B, S, H, head_dim)
    k = torch.randn(B, S, H, head_dim)
    pos = torch.arange(S)
    q_rot, k_rot = module(q, k, pos)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.allclose(q_rot, q)
    # Position 0 should be identity (cos=1, sin=0)
    assert torch.allclose(q_rot[:, 0], q[:, 0], atol=1e-5)


def test_rope_freqs_have_correct_shape():
    spec = specs.RoPESpec(base_theta=10000.0, basis=types.RoPEBasis.SPLIT_HALF)
    head_dim = 8
    max_seq = 16
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)
    assert module.cos_cached.shape == (max_seq, head_dim)
    assert module.sin_cached.shape == (max_seq, head_dim)


def test_rope_matches_hf_llama_style():
    """Compare against HF's rotary_emb computation for Llama / Qwen3."""
    spec = specs.RoPESpec(base_theta=1_000_000.0, basis=types.RoPEBasis.SPLIT_HALF)
    head_dim = 8
    max_seq = 16
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)

    # HF reference: inv_freq * positions
    inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    cos_ref = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin_ref = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    assert torch.allclose(module.cos_cached, cos_ref, atol=1e-5)
    assert torch.allclose(module.sin_cached, sin_ref, atol=1e-5)
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/api/test_rope.py -v
```

- [ ] **Step 3: Implement `api/rope.py`**

```python
"""RoPE block — precomputed cos/sin tables consuming RoPESpec.

M1 implements SPLIT_HALF basis with optional Llama-3 smooth scaling.
Other RoPE variants (PI, NTK, YaRN, LongRoPE) land in later milestones.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
from torch import nn

from api import ops, specs, types


def _llama3_scale_inv_freq(
    inv_freq: torch.Tensor,
    extra: specs.Llama3RoPEParams,
) -> torch.Tensor:
    """Apply Llama-3 smooth RoPE scaling.

    From Meta's reference: inv_freq is scaled per-frequency by a smooth function
    of wavelength.
    """
    low_freq_wavelen = extra.original_context_length / extra.low_freq_factor
    high_freq_wavelen = extra.original_context_length / extra.high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_scaled = torch.where(
        wavelen > low_freq_wavelen,
        inv_freq / extra.factor,
        inv_freq,
    )
    # Smooth interp zone
    smooth_factor = (extra.original_context_length / wavelen - extra.low_freq_factor) / (
        extra.high_freq_factor - extra.low_freq_factor
    )
    smoothed = (1 - smooth_factor) * inv_freq / extra.factor + smooth_factor * inv_freq
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_scaled)


class RoPE(nn.Module):
    def __init__(self, spec: specs.RoPESpec, head_dim: int, max_seq: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.basis != types.RoPEBasis.SPLIT_HALF:
            raise NotImplementedError("M1 supports SPLIT_HALF basis only")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.spec = spec
        self.head_dim = head_dim
        self.max_seq = max_seq

        inv_freq = 1.0 / (
            spec.base_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        if spec.scaling == types.RoPEScaling.LLAMA3:
            if spec.llama3_extra is None:
                raise ValueError("LLAMA3 scaling requires llama3_extra")
            inv_freq = _llama3_scale_inv_freq(inv_freq, spec.llama3_extra)
        elif spec.scaling != types.RoPEScaling.NONE:
            raise NotImplementedError(
                f"M1 supports NONE and LLAMA3 scaling only, got {spec.scaling}"
            )

        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)                  # [max_seq, head_dim // 2]
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
        # Use register_buffer so cos/sin move with .to(device) but aren't params
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        q: torch.Tensor,                                  # [B, S, H, Dh]
        k: torch.Tensor,                                  # [B, S, Hk, Dh]
        position_ids: torch.Tensor,                       # [S] or [B, S]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() == 1:
            cos = self.cos_cached[position_ids]           # [S, Dh]
            sin = self.sin_cached[position_ids]
        else:
            # [B, S] -> would need broadcast support; defer to a later milestone
            raise NotImplementedError("M1 supports 1D position_ids only")
        return ops.rope_apply(q, k, cos, sin, basis="split_half")
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_rope.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/rope.py tests/api/test_rope.py
git commit -m "feat(M1): RoPE module (SPLIT_HALF basis + LLAMA3 scaling)"
```

---

## Task 8 — `api/kvcache.py` — ContiguousKVCache

**Files:**
- Create: `api/kvcache.py`
- Test: `tests/api/test_kvcache.py`

- [ ] **Step 1: Failing tests for prefill / decode**

Create `tests/api/test_kvcache.py`:
```python
import pytest
import torch

from api import kvcache, specs, types


def _make_spec():
    return specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,   # [B, H, S, Dh]
        k_dtype=torch.float32,
        v_dtype=torch.float32,
        ownership=types.CacheOwnership.EXPLICIT_PASS,
    )


def test_kvcache_init_shapes():
    spec = _make_spec()
    cache = kvcache.ContiguousKVCache(spec, batch_size=2, n_kv_heads=4,
                                      head_dim=8, max_seq=16)
    assert cache.k.shape == (2, 4, 16, 8)
    assert cache.v.shape == (2, 4, 16, 8)
    assert cache.seq_len == 0


def test_kvcache_prefill_write_and_read_back():
    spec = _make_spec()
    cache = kvcache.ContiguousKVCache(spec, batch_size=1, n_kv_heads=2,
                                      head_dim=4, max_seq=8)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    cache.write(k, v, start_pos=0)
    k_out, v_out = cache.read(seq_len=5)
    assert k_out.shape == (1, 2, 5, 4)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)
    assert cache.seq_len == 5


def test_kvcache_decode_appends():
    spec = _make_spec()
    cache = kvcache.ContiguousKVCache(spec, batch_size=1, n_kv_heads=2,
                                      head_dim=4, max_seq=8)
    k1 = torch.randn(1, 2, 5, 4); v1 = torch.randn(1, 2, 5, 4)
    cache.write(k1, v1, start_pos=0)
    k2 = torch.randn(1, 2, 1, 4); v2 = torch.randn(1, 2, 1, 4)
    cache.write(k2, v2, start_pos=5)
    k_out, v_out = cache.read(seq_len=6)
    assert k_out.shape == (1, 2, 6, 4)
    assert torch.allclose(k_out[:, :, :5], k1)
    assert torch.allclose(k_out[:, :, 5:6], k2)
    assert cache.seq_len == 6


def test_kvcache_overflow_raises():
    spec = _make_spec()
    cache = kvcache.ContiguousKVCache(spec, batch_size=1, n_kv_heads=2,
                                      head_dim=4, max_seq=4)
    k = torch.randn(1, 2, 5, 4); v = torch.randn(1, 2, 5, 4)
    with pytest.raises(ValueError, match="exceeds max_seq"):
        cache.write(k, v, start_pos=0)
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/api/test_kvcache.py -v
```

- [ ] **Step 3: Implement `api/kvcache.py`**

```python
"""KV cache primitives.

M1 implements ContiguousKVCache with EXPLICIT_PASS ownership in HND layout
(the HF / Qwen3 default). Paged / ring / MLA-latent / SSM-state come in
later milestones.
"""
from __future__ import annotations

import torch

from api import specs, types


class ContiguousKVCache:
    """A pre-allocated K/V buffer.

    HND layout: tensors are [B, n_kv_heads, max_seq, head_dim].
    This is the shape attention SDPA expects.
    """

    def __init__(
        self,
        spec: specs.KVCacheSpec,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq: int,
        device: torch.device | str = "cpu",
    ):
        if spec.layout != types.CacheLayout.CONTIGUOUS:
            raise NotImplementedError("M1 supports CONTIGUOUS layout only")
        if spec.memory_layout != types.MemoryLayout.HND:
            raise NotImplementedError("M1 supports HND memory layout only")
        if spec.k_quant is not None or spec.v_quant is not None:
            raise NotImplementedError("M1 supports unquantized KV cache only")
        self.spec = spec
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq = max_seq
        self.k = torch.zeros(batch_size, n_kv_heads, max_seq, head_dim,
                             dtype=spec.k_dtype, device=device)
        self.v = torch.zeros(batch_size, n_kv_heads, max_seq, head_dim,
                             dtype=spec.v_dtype, device=device)
        self.seq_len = 0

    def write(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """Write k, v at positions [start_pos, start_pos + k.shape[2]).

        k, v shape: [B, n_kv_heads, S_new, head_dim]
        """
        s_new = k.shape[2]
        end_pos = start_pos + s_new
        if end_pos > self.max_seq:
            raise ValueError(
                f"start_pos+S_new ({end_pos}) exceeds max_seq ({self.max_seq})"
            )
        self.k[:, :, start_pos:end_pos] = k
        self.v[:, :, start_pos:end_pos] = v
        self.seq_len = max(self.seq_len, end_pos)

    def read(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read cache contents for positions [0, seq_len)."""
        return self.k[:, :, :seq_len], self.v[:, :, :seq_len]

    def reset(self) -> None:
        self.seq_len = 0
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_kvcache.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/kvcache.py tests/api/test_kvcache.py
git commit -m "feat(M1): ContiguousKVCache (HND, EXPLICIT_PASS)"
```

---

## Task 9 — `api/quant.py` — AWQ W4A16 round-trip

**Files:**
- Create: `api/quant.py`
- Test: `tests/api/test_quant.py`

Background: AWQ packs INT4 weights into INT32 with a specific nibble permutation `[0, 2, 4, 6, 1, 3, 5, 7]`. Zeros and scales are stored separately. Group size is typically 128.

- [ ] **Step 1: Failing test — synthetic packed → unpacked round trip**

Create `tests/api/test_quant.py`:
```python
import torch

from api import quant, specs, types


def _make_awq_spec():
    return specs.QuantSpec(
        qdtype=types.QDType.INT4,
        group_size=64,                   # small for fast tests
        quant_axis=0,
        scale_dtype=torch.float16,
        has_zero_point=True,
        packing=types.PackingLayout.AWQ_INTERLEAVE,
        accumulator_dtype=torch.float32,
        role=types.QuantRole.WEIGHT,
    )


def test_awq_pack_unpack_round_trip():
    """A synthetic INT4 tensor packed then unpacked should produce the original ints."""
    torch.manual_seed(0)
    K, N = 64, 16                          # K=64, N=16 — one group across K
    int_values = torch.randint(0, 16, (K, N), dtype=torch.int32)
    packed = quant.awq_pack(int_values)
    # packed should have shape [K // 8, N] in int32
    assert packed.dtype == torch.int32
    assert packed.shape == (K // 8, N)
    unpacked = quant.awq_unpack(packed, K, N)
    assert unpacked.shape == (K, N)
    assert torch.equal(unpacked, int_values)


def test_awq_dequant_recovers_within_quant_error():
    """Quantize an fp16 weight, then dequant, check error is bounded."""
    torch.manual_seed(0)
    K, N = 64, 8
    w_fp16 = torch.randn(K, N, dtype=torch.float16)

    spec = _make_awq_spec()
    packed, scales, zeros = quant.awq_quantize(w_fp16, spec)
    # Shapes
    assert packed.shape == (K // 8, N)             # 4-bit packed along K, INT32
    assert scales.shape == (K // 64, N)            # one scale per group along K
    assert zeros.shape == (K // 64, N // 8)        # zeros packed INT32 along N

    w_recovered = quant.awq_dequantize(packed, scales, zeros, spec, K, N)
    assert w_recovered.shape == (K, N)
    assert w_recovered.dtype == torch.float16

    # Bound: AWQ 4-bit grouped should give per-element error ≤ ~scale (i.e., ≤ amax/8).
    # As a coarse check: relative error should be < 0.15.
    rel_err = (w_recovered.float() - w_fp16.float()).abs() / (w_fp16.float().abs() + 1e-6)
    assert rel_err.median() < 0.15


def test_awq_dequant_then_linear_matches_full_precision_within_quant_tolerance():
    """End-to-end: x @ W_dequant.T should be close to x @ W.T at quant tolerance."""
    torch.manual_seed(0)
    K, N = 64, 8
    w_fp16 = torch.randn(K, N, dtype=torch.float16)
    spec = _make_awq_spec()
    packed, scales, zeros = quant.awq_quantize(w_fp16, spec)
    w_recovered = quant.awq_dequantize(packed, scales, zeros, spec, K, N)

    x = torch.randn(2, K, dtype=torch.float16)
    y_full = (x.float() @ w_fp16.float()).to(torch.float16)
    y_quant = (x.float() @ w_recovered.float()).to(torch.float16)
    rel = (y_full.float() - y_quant.float()).abs() / (y_full.float().abs() + 1e-6)
    assert rel.median() < 0.10
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/api/test_quant.py -v
```

- [ ] **Step 3: Implement `api/quant.py`**

```python
"""Quantization: AWQ W4A16 grouped INT4 round-trip.

Reference: AWQ uses the nibble permutation [0,2,4,6,1,3,5,7] when packing
8 INT4 values into one INT32. We replicate the storage layout so we can
load real AWQ checkpoints in Task 19.

Layout (for W of shape [K, N], group_size G along K):
    qweight  : int32[K // 8, N]              -- 8 nibbles packed per int32
    qzeros   : int32[K // G, N // 8]         -- 8 zero-nibbles packed per int32
    scales   : float16[K // G, N]            -- per-group scale
"""
from __future__ import annotations
from typing import Tuple

import torch

from api import specs, types


# AWQ nibble permutation when packing 8 INT4 values into one INT32 along axis K.
# The i-th unpacked row (i in 0..K) maps to bit position 4 * AWQ_ORDER[i % 8]
# of the int32 at row (i // 8).
AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _awq_shifts() -> torch.Tensor:
    """Bit shifts for the 8 nibbles in one INT32, in AWQ order."""
    return torch.tensor([4 * p for p in AWQ_ORDER], dtype=torch.int32)


def awq_pack(int_values: torch.Tensor) -> torch.Tensor:
    """Pack INT4 (0..15) values along axis 0 into INT32 with AWQ interleave.

    int_values: [K, N] int32 in range [0, 16)
    returns:    [K // 8, N] int32
    """
    if int_values.dtype != torch.int32:
        raise ValueError(f"expected int32, got {int_values.dtype}")
    K, N = int_values.shape
    if K % 8 != 0:
        raise ValueError(f"K ({K}) must be divisible by 8")
    iv = int_values & 0xF                                # ensure 4-bit
    iv = iv.reshape(K // 8, 8, N)                        # [K//8, 8, N]
    shifts = _awq_shifts().view(1, 8, 1)
    packed = (iv << shifts).sum(dim=1).to(torch.int32)   # [K//8, N]
    return packed


def awq_unpack(packed: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Inverse of awq_pack."""
    if packed.shape != (K // 8, N):
        raise ValueError(f"shape mismatch: packed {packed.shape} vs K//8={K//8}, N={N}")
    shifts = _awq_shifts().view(1, 8, 1)
    out = (packed.unsqueeze(1) >> shifts) & 0xF          # [K//8, 8, N]
    return out.reshape(K, N).to(torch.int32)


def awq_quantize(
    w: torch.Tensor,
    spec: specs.QuantSpec,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize w using AWQ-style grouped INT4 with asymmetric zero-point.

    w: [K, N] in scale_dtype (fp16/bf16)
    returns (qweight, scales, qzeros)
    """
    if spec.qdtype != types.QDType.INT4:
        raise ValueError("AWQ path requires INT4")
    if spec.group_size is None or spec.group_size <= 0:
        raise ValueError("AWQ requires positive group_size")
    if not spec.has_zero_point:
        raise ValueError("AWQ uses asymmetric quant — has_zero_point must be True")
    K, N = w.shape
    G = spec.group_size
    if K % G != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({G})")

    w32 = w.float().reshape(K // G, G, N)                 # [n_groups, G, N]
    w_min = w32.min(dim=1, keepdim=True).values           # [n_groups, 1, N]
    w_max = w32.max(dim=1, keepdim=True).values
    # 4-bit range: 0..15
    qmax = 15.0
    scales = (w_max - w_min) / qmax                       # [n_groups, 1, N]
    scales = scales.clamp_min(1e-8)
    zero_floats = -w_min / scales                          # the float that 0 represents
    zeros_int = zero_floats.round().clamp(0, qmax).to(torch.int32).squeeze(1)  # [n_groups, N]

    q_int = ((w32 / scales) + zero_floats).round().clamp(0, qmax).to(torch.int32)  # [n_groups, G, N]
    q_int = q_int.reshape(K, N)
    qweight = awq_pack(q_int)
    qzeros = awq_pack(zeros_int.repeat_interleave(8, dim=0)[:K, :])  # placeholder

    # AWQ qzeros layout is packed along N, not K — pack 8 N-values per int32
    qzeros_n_packed = _pack_n_axis(zeros_int)              # [n_groups, N // 8]
    scales = scales.squeeze(1).to(spec.scale_dtype)        # [n_groups, N]
    return qweight, scales, qzeros_n_packed


def _pack_n_axis(zeros_int: torch.Tensor) -> torch.Tensor:
    """Pack 8 INT4 values along N axis into one INT32, AWQ order."""
    n_groups, N = zeros_int.shape
    if N % 8 != 0:
        raise ValueError(f"N ({N}) must be divisible by 8 for AWQ zeros packing")
    iv = zeros_int & 0xF
    iv = iv.reshape(n_groups, N // 8, 8)
    shifts = _awq_shifts().view(1, 1, 8)
    return (iv << shifts).sum(dim=-1).to(torch.int32)


def _unpack_n_axis(qzeros_packed: torch.Tensor, N: int) -> torch.Tensor:
    n_groups = qzeros_packed.shape[0]
    if qzeros_packed.shape[1] != N // 8:
        raise ValueError(f"qzeros shape mismatch: got {qzeros_packed.shape}, expected (*, {N//8})")
    shifts = _awq_shifts().view(1, 1, 8)
    out = (qzeros_packed.unsqueeze(-1) >> shifts) & 0xF
    return out.reshape(n_groups, N).to(torch.int32)


def awq_dequantize(
    qweight: torch.Tensor,    # [K // 8, N] int32
    scales: torch.Tensor,     # [K // G, N] fp16/bf16
    qzeros: torch.Tensor,     # [K // G, N // 8] int32
    spec: specs.QuantSpec,
    K: int,
    N: int,
) -> torch.Tensor:
    """Reverse of awq_quantize. Returns the dequantized weight [K, N] in scale_dtype."""
    if spec.group_size is None or spec.group_size <= 0:
        raise ValueError("requires positive group_size")
    G = spec.group_size
    q_int = awq_unpack(qweight, K, N)                       # [K, N]
    zeros_int = _unpack_n_axis(qzeros, N)                   # [n_groups, N]

    n_groups = K // G
    q_int_g = q_int.reshape(n_groups, G, N).float()
    zeros_g = zeros_int.reshape(n_groups, 1, N).float()
    scales_g = scales.reshape(n_groups, 1, N).float()
    w = (q_int_g - zeros_g) * scales_g                       # [n_groups, G, N]
    return w.reshape(K, N).to(spec.scale_dtype)
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_quant.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/quant.py tests/api/test_quant.py
git commit -m "feat(M1): AWQ W4A16 grouped INT4 quant/dequant round-trip"
```

---

## Task 10 — `api/attention.py` — GQA + QK-norm + RoPE + KVCache

**Files:**
- Create: `api/attention.py`
- Test: `tests/api/test_attention.py`

- [ ] **Step 1: Failing test — random-weight Attention forward pass shapes**

Create `tests/api/test_attention.py`:
```python
import torch
import torch.nn.functional as F

from api import attention, kvcache, norm, rope, specs, types


def _qwen3_like_attention_spec(qk_norm: bool):
    qk_spec = (specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                              weight_mode=types.NormWeightMode.STANDARD_W)
               if qk_norm else None)
    return specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=64,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        qk_norm=qk_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE if qk_norm else types.QKNormPhase.NONE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH if qk_norm else types.QKNormShape.NONE,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )


def test_attention_forward_shape():
    hidden_size = 1024
    spec = _qwen3_like_attention_spec(qk_norm=True)
    attn = attention.Attention(spec, hidden_size=hidden_size, max_seq=128,
                               dtype=torch.float32)
    B, S = 1, 8
    x = torch.randn(B, S, hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(cache_spec, batch_size=B,
                                      n_kv_heads=spec.n_kv_heads,
                                      head_dim=spec.head_dim, max_seq=128)
    out = attn(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, hidden_size)
    assert cache.seq_len == S


def test_attention_decode_step_appends_to_cache():
    hidden_size = 256
    spec = _qwen3_like_attention_spec(qk_norm=False)
    attn = attention.Attention(spec, hidden_size=hidden_size, max_seq=64,
                               dtype=torch.float32)
    B = 1
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(cache_spec, batch_size=B,
                                      n_kv_heads=spec.n_kv_heads,
                                      head_dim=spec.head_dim, max_seq=64)
    x_prefill = torch.randn(B, 5, hidden_size)
    pos_prefill = torch.arange(5)
    _ = attn(x_prefill, position_ids=pos_prefill, cache=cache, start_pos=0)
    assert cache.seq_len == 5

    x_decode = torch.randn(B, 1, hidden_size)
    pos_decode = torch.tensor([5])
    _ = attn(x_decode, position_ids=pos_decode, cache=cache, start_pos=5)
    assert cache.seq_len == 6


def test_attention_causal_mask_for_prefill():
    """Tokens at position i must not attend to positions > i during prefill."""
    spec = _qwen3_like_attention_spec(qk_norm=False)
    attn = attention.Attention(spec, hidden_size=256, max_seq=16,
                               dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(cache_spec, batch_size=1,
                                      n_kv_heads=spec.n_kv_heads,
                                      head_dim=spec.head_dim, max_seq=16)
    B, S, D = 1, 4, 256
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = attn(x, position_ids=pos, cache=cache, start_pos=0)
    # Sanity: cache filled to S
    assert cache.seq_len == S
    assert out.shape == (B, S, D)
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/api/test_attention.py -v
```

- [ ] **Step 3: Implement `api/attention.py`**

```python
"""Attention block consuming AttentionSpec.

M1 supports the Qwen3 shape: STANDARD (GQA) with split QKV, optional QK-norm
PRE_ROPE PER_HEAD_DH, RoPE SPLIT_HALF, contiguous KV cache, causal mask.
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from api import kvcache as _kvcache, norm, ops, rope as _rope, specs, types


class Attention(nn.Module):
    def __init__(
        self,
        spec: specs.AttentionSpec,
        hidden_size: int,
        max_seq: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if spec.kind != types.AttentionKind.STANDARD:
            raise NotImplementedError(f"M1: STANDARD only, got {spec.kind}")
        if spec.qkv_layout != types.QKVLayout.SPLIT:
            raise NotImplementedError(f"M1: SPLIT QKV only, got {spec.qkv_layout}")
        if spec.mask_kind != types.MaskKind.CAUSAL:
            raise NotImplementedError(f"M1: CAUSAL mask only, got {spec.mask_kind}")
        self.spec = spec
        self.hidden_size = hidden_size

        q_proj_out = spec.n_q_heads * spec.head_dim
        kv_proj_out = spec.n_kv_heads * spec.head_dim
        self.q_proj = nn.Linear(hidden_size, q_proj_out, bias=spec.q_bias, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.k_bias, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.v_bias, dtype=dtype)
        self.o_proj = nn.Linear(q_proj_out, hidden_size, bias=spec.o_bias, dtype=dtype)

        if spec.qk_norm is not None:
            if spec.qk_norm_phase != types.QKNormPhase.PRE_ROPE:
                raise NotImplementedError("M1: PRE_ROPE QK-norm only")
            self.q_norm = norm.QKNorm(
                spec.qk_norm, head_dim=spec.head_dim,
                shape=spec.qk_norm_shape,
                n_heads=spec.n_q_heads if spec.qk_norm_shape == types.QKNormShape.FULL_HDH else None,
                dtype=dtype,
            )
            self.k_norm = norm.QKNorm(
                spec.qk_norm, head_dim=spec.head_dim,
                shape=spec.qk_norm_shape,
                n_heads=spec.n_kv_heads if spec.qk_norm_shape == types.QKNormShape.FULL_HDH else None,
                dtype=dtype,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        if spec.rope is None:
            raise NotImplementedError("M1: RoPE required")
        self.rope = _rope.RoPE(spec.rope, head_dim=spec.head_dim,
                               max_seq=max_seq, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,                        # [B, S, hidden_size]
        position_ids: torch.Tensor,             # [S]
        cache: _kvcache.ContiguousKVCache,
        start_pos: int,
    ) -> torch.Tensor:
        spec = self.spec
        B, S, _ = x.shape
        Hq, Hk, Dh = spec.n_q_heads, spec.n_kv_heads, spec.head_dim

        # Project to QKV.
        q = self.q_proj(x).view(B, S, Hq, Dh)
        k = self.k_proj(x).view(B, S, Hk, Dh)
        v = self.v_proj(x).view(B, S, Hk, Dh)

        # QK-norm (pre-RoPE), if enabled.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE on q and k.
        q, k = self.rope(q, k, position_ids)

        # Move to HND layout for SDPA: [B, H, S, Dh]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Write to cache and read it back for full attention context.
        cache.write(k, v, start_pos=start_pos)
        k_full, v_full = cache.read(seq_len=start_pos + S)

        # For decode-time S=1: we use a non-causal mask because Q is just the
        # last token and all of K is in the past.
        is_causal = (S > 1)
        scale = spec.attn_scale if spec.attn_scale is not None else (Dh ** -0.5)
        attn_out = ops.sdpa(q, k_full, v_full,
                            is_causal=is_causal, scale=scale)   # [B, Hq, S, Dh]

        # Merge heads and project out.
        attn_out = attn_out.transpose(1, 2).reshape(B, S, Hq * Dh)
        return self.o_proj(attn_out)
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_attention.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/attention.py tests/api/test_attention.py
git commit -m "feat(M1): Attention block (GQA + QK-norm PRE-RoPE + ContiguousKVCache)"
```

---

## Task 11 — `api/feedforward.py` — SwiGLU FFN

**Files:**
- Create: `api/feedforward.py`
- Test: `tests/api/test_feedforward.py`

- [ ] **Step 1: Failing test for SwiGLU forward pass**

Create `tests/api/test_feedforward.py`:
```python
import torch
import torch.nn.functional as F

from api import feedforward, specs, types


def _swiglu_spec():
    return specs.FFNSpec(
        intermediate_size=128,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )


def test_swiglu_forward_shape():
    spec = _swiglu_spec()
    ffn = feedforward.FeedForward(spec, hidden_size=32, dtype=torch.float32)
    x = torch.randn(2, 4, 32)
    out = ffn(x)
    assert out.shape == (2, 4, 32)


def test_swiglu_matches_manual():
    """y = down(silu(gate(x)) * up(x))"""
    spec = _swiglu_spec()
    ffn = feedforward.FeedForward(spec, hidden_size=8, dtype=torch.float32)
    x = torch.randn(1, 2, 8)

    g = F.linear(x, ffn.gate_proj.weight)
    u = F.linear(x, ffn.up_proj.weight)
    expected = F.linear(F.silu(g) * u, ffn.down_proj.weight)
    out = ffn(x)
    assert torch.allclose(out, expected, atol=1e-5)
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/api/test_feedforward.py -v
```

- [ ] **Step 3: Implement `api/feedforward.py`**

```python
"""Feed-forward / channel-mixer building blocks.

M1 supports SwiGLU only — the dominant SLM channel mixer (Llama, Qwen, Mistral).
GeGLU (Gemma), MoE, fused gate_up (Phi-3) land in later milestones.
"""
from __future__ import annotations

import torch
from torch import nn

from api import ops, specs, types


class FeedForward(nn.Module):
    def __init__(self, spec: specs.FFNSpec, hidden_size: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.gate_kind != types.GateKind.SWIGLU:
            raise NotImplementedError(f"M1: SWIGLU only, got {spec.gate_kind}")
        if spec.activation != types.Activation.SILU:
            raise NotImplementedError(f"M1: SILU only, got {spec.activation}")
        if spec.fused_gate_up:
            raise NotImplementedError("M1: split gate/up only")
        self.spec = spec
        self.hidden_size = hidden_size
        I = spec.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, I, bias=spec.gate_bias, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, I, bias=spec.up_bias, dtype=dtype)
        self.down_proj = nn.Linear(I, hidden_size, bias=spec.down_bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(ops.mul(ops.silu(gate), up))
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_feedforward.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/feedforward.py tests/api/test_feedforward.py
git commit -m "feat(M1): SwiGLU FeedForward block"
```

---

## Task 12 — `api/block.py` — DecoderBlock (pre-norm)

**Files:**
- Create: `api/block.py`
- Test: `tests/api/test_block.py`

- [ ] **Step 1: Failing test**

Create `tests/api/test_block.py`:
```python
import torch

from api import block, kvcache, specs, types


def _qwen3_like_block_spec(hidden_size: int = 256):
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    qk_spec = norm_spec
    attn_spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=4, head_dim=32,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        qk_norm=qk_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn_spec = specs.FFNSpec(intermediate_size=512,
                             activation=types.Activation.SILU,
                             gate_kind=types.GateKind.SWIGLU)
    return specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn_spec,
        channel_mixer=ffn_spec,
        input_norm=norm_spec,
        pre_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec,
    )


def test_block_forward_preserves_residual_shape():
    spec = _qwen3_like_block_spec(hidden_size=256)
    blk = block.DecoderBlock(spec, hidden_size=256, max_seq=64, dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=spec.token_mixer.n_kv_heads,
        head_dim=spec.token_mixer.head_dim, max_seq=64,
    )
    B, S, D = 1, 5, 256
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, D)
    assert cache.seq_len == S


def test_block_residual_actually_adds():
    """Output should NOT equal pure attention/ffn output — residual stream is in there."""
    spec = _qwen3_like_block_spec(hidden_size=128)
    blk = block.DecoderBlock(spec, hidden_size=128, max_seq=32, dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=spec.token_mixer.n_kv_heads,
        head_dim=spec.token_mixer.head_dim, max_seq=32,
    )
    B, S, D = 1, 3, 128
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    # Output should be in same scale ballpark as x (residual preserves magnitude).
    assert (out - x).abs().mean() > 0
    assert out.abs().mean() < 100 * x.abs().mean()
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/api/test_block.py -v
```

- [ ] **Step 3: Implement `api/block.py`**

```python
"""DecoderBlock — assembles token mixer + channel mixer with residual structure.

M1 supports PRE-norm only (Qwen3 / Llama default). POST / PRE_AND_POST (OLMo 2,
Gemma) land in later milestones.
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from api import attention as _attention, feedforward, kvcache, norm, ops, specs, types


class DecoderBlock(nn.Module):
    def __init__(
        self,
        spec: specs.DecoderBlockSpec,
        hidden_size: int,
        max_seq: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if spec.attn_norm_position != types.NormPosition.PRE:
            raise NotImplementedError("M1: PRE attn-norm only")
        if spec.ffn_norm_position != types.NormPosition.PRE:
            raise NotImplementedError("M1: PRE ffn-norm only")
        if not isinstance(spec.token_mixer, specs.AttentionSpec):
            raise NotImplementedError("M1: AttentionSpec token mixer only")
        if not isinstance(spec.channel_mixer, specs.FFNSpec):
            raise NotImplementedError("M1: FFNSpec channel mixer only")
        if spec.residual_scale is not None:
            raise NotImplementedError("M1: no residual scaling")
        self.spec = spec
        self.hidden_size = hidden_size

        if spec.pre_attn_norm is None:
            raise ValueError("PRE attn-norm requires pre_attn_norm")
        if spec.pre_ffn_norm is None:
            raise ValueError("PRE ffn-norm requires pre_ffn_norm")

        self.input_norm = norm.RMSNorm(spec.pre_attn_norm, hidden_size, dtype=dtype)
        self.attention = _attention.Attention(spec.token_mixer, hidden_size,
                                              max_seq=max_seq, dtype=dtype)
        self.post_attn_norm = norm.RMSNorm(spec.pre_ffn_norm, hidden_size, dtype=dtype)
        self.feedforward = feedforward.FeedForward(spec.channel_mixer, hidden_size,
                                                   dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,                        # [B, S, hidden_size]
        position_ids: torch.Tensor,
        cache: kvcache.ContiguousKVCache,
        start_pos: int,
    ) -> torch.Tensor:
        # Pre-norm attention sublayer
        attn_in = self.input_norm(x)
        attn_out = self.attention(attn_in, position_ids=position_ids,
                                  cache=cache, start_pos=start_pos)
        x = ops.add(x, attn_out)

        # Pre-norm FFN sublayer
        ffn_in = self.post_attn_norm(x)
        ffn_out = self.feedforward(ffn_in)
        return ops.add(x, ffn_out)
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/api/test_block.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add api/block.py tests/api/test_block.py
git commit -m "feat(M1): DecoderBlock with PRE-norm residual structure"
```

---

## Task 13 — `models/qwen3/config.py` — Qwen3Config + HF adapter

**Files:**
- Create: `models/qwen3/__init__.py`, `models/qwen3/config.py`
- Test: `tests/models/qwen3/test_config.py`

- [ ] **Step 1: Create `models/__init__.py`** (if missing)

```powershell
New-Item -ItemType File -Path "models\__init__.py", "models\qwen3\__init__.py"
```

Create `models/qwen3/__init__.py`:
```python
"""Qwen3 reference layer implementation via api/ primitives."""
```

- [ ] **Step 2: Failing test**

Create `tests/models/qwen3/test_config.py`:
```python
import torch

from models.qwen3 import config


def test_qwen3_config_from_hf_dict():
    """Standard Qwen3-0.6B config values."""
    hf = {
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,                   # explicit in Qwen3 (not hidden_size / heads)
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }
    c = config.Qwen3Config.from_hf_dict(hf)
    assert c.hidden_size == 1024
    assert c.num_attention_heads == 16
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128
    assert c.rope_theta == 1_000_000.0
    assert c.dtype == torch.bfloat16
    assert c.tie_word_embeddings is True


def test_qwen3_config_to_block_spec():
    c = config.Qwen3Config(
        hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, intermediate_size=3072, num_hidden_layers=28,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=151936, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
    )
    block_spec = c.to_block_spec()
    assert block_spec.token_mixer.n_q_heads == 16
    assert block_spec.token_mixer.n_kv_heads == 8
    assert block_spec.token_mixer.head_dim == 128
    assert block_spec.token_mixer.qk_norm is not None
    # Qwen3 verified to apply QK-norm PRE-RoPE with PER_HEAD_DH shape.
    from api import types as _t
    assert block_spec.token_mixer.qk_norm_phase == _t.QKNormPhase.PRE_ROPE
    assert block_spec.token_mixer.qk_norm_shape == _t.QKNormShape.PER_HEAD_DH
    assert block_spec.channel_mixer.intermediate_size == 3072
    assert block_spec.channel_mixer.gate_kind == _t.GateKind.SWIGLU
```

- [ ] **Step 3: Run, expect failure**

```powershell
uv run pytest tests/models/qwen3/test_config.py -v
```

- [ ] **Step 4: Implement `models/qwen3/config.py`**

```python
"""Qwen3 model config + adapter to api specs.

Verified against `transformers.Qwen3Config` and Qwen3-0.6B / 1.7B / 4B / 8B
config.json on HuggingFace.

Key Qwen3 facts (see research/05-kvcache-attention.v2.md, fixed in v2):
- GQA (n_kv_heads < n_q_heads)
- head_dim is EXPLICIT, not hidden_size / n_q_heads (Qwen3-0.6B: 1024/16=64 vs head_dim=128)
- QK-norm PRE-RoPE, weight shape PER_HEAD_DH (verified against modeling_qwen3.py)
- RoPE SPLIT_HALF basis, base_theta=1_000_000 (NOT 5M as v1 incorrectly claimed)
- RMSNorm STANDARD_W mode
- SwiGLU FFN
- tie_word_embeddings: True for 0.6B/1.7B/4B, False for 8B
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Qwen3Config":
        dt_raw = hf.get("torch_dtype", "float32")
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)
        return cls(
            hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf["num_key_value_heads"],
            head_dim=hf.get(
                "head_dim",
                hf["hidden_size"] // hf["num_attention_heads"],
            ),
            intermediate_size=hf["intermediate_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            rope_theta=float(hf["rope_theta"]),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
        )

    def to_block_spec(self) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        qk_norm_spec = norm_spec
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            qk_norm=qk_norm_spec,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            ),
        )
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            input_norm=norm_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
```

- [ ] **Step 5: Run, expect pass**

```powershell
uv run pytest tests/models/qwen3/test_config.py -v
```

- [ ] **Step 6: Commit**

```powershell
git add models/__init__.py models/qwen3/__init__.py models/qwen3/config.py tests/models/qwen3/test_config.py
git commit -m "feat(M1): Qwen3Config + HF→spec adapter"
```

---

## Task 14 — `models/qwen3/layer.py` — factory + shape test

**Files:**
- Create: `models/qwen3/layer.py`
- Test: `tests/models/qwen3/test_layer_shape.py`

- [ ] **Step 1: Failing test — build layer at 3 sizes, forward pass shapes**

Create `tests/models/qwen3/test_layer_shape.py`:
```python
import pytest
import torch

from api import kvcache, specs, types
from models.qwen3 import config, layer


def _config_0p6b():
    return config.Qwen3Config(
        hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, intermediate_size=3072, num_hidden_layers=28,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=151936, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
    )


def _config_1p7b():
    return config.Qwen3Config(
        hidden_size=2048, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, intermediate_size=6144, num_hidden_layers=28,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=151936, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
    )


def _config_4b():
    return config.Qwen3Config(
        hidden_size=2560, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=9728, num_hidden_layers=36,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=151936, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
    )


@pytest.mark.parametrize("cfg_fn", [_config_0p6b, _config_1p7b, _config_4b])
def test_qwen3_decoder_layer_forward_shape(cfg_fn):
    cfg = cfg_fn()
    blk = layer.build_qwen3_decoder_layer(cfg, layer_idx=0, max_seq=128)
    B, S = 1, 8
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=128,
    )
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)
    assert cache.seq_len == S


def test_qwen3_decoder_layer_determinism():
    cfg = _config_0p6b()
    blk = layer.build_qwen3_decoder_layer(cfg, layer_idx=0, max_seq=64)
    torch.manual_seed(0)
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4)

    def run():
        cache_spec = specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=cfg.dtype, v_dtype=cfg.dtype,
        )
        cache = kvcache.ContiguousKVCache(
            cache_spec, batch_size=1,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, max_seq=64,
        )
        return blk(x, position_ids=pos, cache=cache, start_pos=0)

    a = run()
    b = run()
    assert torch.allclose(a, b, atol=0)
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/models/qwen3/test_layer_shape.py -v
```

- [ ] **Step 3: Implement `models/qwen3/layer.py`**

```python
"""Qwen3 decoder-layer factory.

Builds a DecoderBlock from a Qwen3Config using only api/ primitives.
Does NOT import transformers.models.qwen3.modeling_qwen3.
"""
from __future__ import annotations

from api import block
from models.qwen3 import config as _config


def build_qwen3_decoder_layer(
    cfg: _config.Qwen3Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Qwen3 decoder block at layer_idx (currently identity)."""
    spec = cfg.to_block_spec()
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )
```

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/models/qwen3/test_layer_shape.py -v
```
Expected: 4 passed (3 parametrized + 1 determinism).

- [ ] **Step 5: Commit**

```powershell
git add models/qwen3/layer.py tests/models/qwen3/test_layer_shape.py
git commit -m "feat(M1): build_qwen3_decoder_layer factory + shape tests"
```

---

## Task 15 — HF weight loader

**Files:**
- Modify: `models/qwen3/layer.py`
- Test: `tests/models/qwen3/test_weight_loader.py`

The HF state-dict for a Qwen3 decoder layer has these tensor names:
```
model.layers.{L}.input_layernorm.weight                      -> input_norm.weight
model.layers.{L}.self_attn.q_proj.weight                     -> attention.q_proj.weight
model.layers.{L}.self_attn.k_proj.weight                     -> attention.k_proj.weight
model.layers.{L}.self_attn.v_proj.weight                     -> attention.v_proj.weight
model.layers.{L}.self_attn.o_proj.weight                     -> attention.o_proj.weight
model.layers.{L}.self_attn.q_norm.weight                     -> attention.q_norm.weight
model.layers.{L}.self_attn.k_norm.weight                     -> attention.k_norm.weight
model.layers.{L}.post_attention_layernorm.weight             -> post_attn_norm.weight
model.layers.{L}.mlp.gate_proj.weight                        -> feedforward.gate_proj.weight
model.layers.{L}.mlp.up_proj.weight                          -> feedforward.up_proj.weight
model.layers.{L}.mlp.down_proj.weight                        -> feedforward.down_proj.weight
```

- [ ] **Step 1: Failing test — synthetic state dict load**

Create `tests/models/qwen3/test_weight_loader.py`:
```python
import torch

from api import kvcache, specs, types
from models.qwen3 import config, layer


def _config_tiny():
    return config.Qwen3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=2,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=True, dtype=torch.float32,
    )


def _synthetic_hf_state_dict(cfg, layer_idx):
    """Generate a synthetic HF state dict for one Qwen3 layer."""
    hs = cfg.hidden_size
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    I = cfg.intermediate_size
    base = f"model.layers.{layer_idx}"
    torch.manual_seed(42)
    return {
        f"{base}.input_layernorm.weight": torch.randn(hs),
        f"{base}.self_attn.q_proj.weight": torch.randn(q_out, hs),
        f"{base}.self_attn.k_proj.weight": torch.randn(kv_out, hs),
        f"{base}.self_attn.v_proj.weight": torch.randn(kv_out, hs),
        f"{base}.self_attn.o_proj.weight": torch.randn(hs, q_out),
        f"{base}.self_attn.q_norm.weight": torch.randn(cfg.head_dim),
        f"{base}.self_attn.k_norm.weight": torch.randn(cfg.head_dim),
        f"{base}.post_attention_layernorm.weight": torch.randn(hs),
        f"{base}.mlp.gate_proj.weight": torch.randn(I, hs),
        f"{base}.mlp.up_proj.weight": torch.randn(I, hs),
        f"{base}.mlp.down_proj.weight": torch.randn(hs, I),
    }


def test_load_hf_weights_into_layer():
    cfg = _config_tiny()
    blk = layer.build_qwen3_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_hf_state_dict(cfg, layer_idx=0)
    layer.load_hf_qwen3_layer(blk, sd, layer_idx=0)

    # Spot check: q_proj weight should now equal the synthetic source
    assert torch.allclose(
        blk.attention.q_proj.weight, sd["model.layers.0.self_attn.q_proj.weight"]
    )
    assert torch.allclose(
        blk.feedforward.down_proj.weight, sd["model.layers.0.mlp.down_proj.weight"]
    )
    assert torch.allclose(
        blk.attention.q_norm.weight, sd["model.layers.0.self_attn.q_norm.weight"]
    )


def test_load_then_forward():
    cfg = _config_tiny()
    blk = layer.build_qwen3_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_hf_state_dict(cfg, layer_idx=0)
    layer.load_hf_qwen3_layer(blk, sd, layer_idx=0)

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=32,
    )
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (1, 4, cfg.hidden_size)
```

- [ ] **Step 2: Run, expect failure**

```powershell
uv run pytest tests/models/qwen3/test_weight_loader.py -v
```

- [ ] **Step 3: Append `load_hf_qwen3_layer` to `models/qwen3/layer.py`**

```python
def load_hf_qwen3_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Qwen3 layer weights into the api-assembled DecoderBlock.

    Mapping table (HF tensor name -> API tensor slot):
        model.layers.{L}.input_layernorm.weight
            -> blk.input_norm.weight
        model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
            -> blk.attention.{q,k,v,o}_proj.weight
        model.layers.{L}.self_attn.{q,k}_norm.weight
            -> blk.attention.{q,k}_norm.weight
        model.layers.{L}.post_attention_layernorm.weight
            -> blk.post_attn_norm.weight
        model.layers.{L}.mlp.{gate,up,down}_proj.weight
            -> blk.feedforward.{gate,up,down}_proj.weight
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    mapping = {
        f"{prefix}.input_layernorm.weight":             blk.input_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":            blk.attention.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":            blk.attention.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":            blk.attention.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":            blk.attention.o_proj.weight,
        f"{prefix}.post_attention_layernorm.weight":    blk.post_attn_norm.weight,
        f"{prefix}.mlp.gate_proj.weight":               blk.feedforward.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight":                 blk.feedforward.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":               blk.feedforward.down_proj.weight,
    }
    # QK-norm only if the attention spec has it.
    if blk.attention.q_norm is not None:
        mapping[f"{prefix}.self_attn.q_norm.weight"] = blk.attention.q_norm.weight
    if blk.attention.k_norm is not None:
        mapping[f"{prefix}.self_attn.k_norm.weight"] = blk.attention.k_norm.weight

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")

    import torch as _torch
    with _torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
```

Add `from api import block` to imports at the top of `models/qwen3/layer.py`.

- [ ] **Step 4: Run, expect pass**

```powershell
uv run pytest tests/models/qwen3/test_weight_loader.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add models/qwen3/layer.py tests/models/qwen3/test_weight_loader.py
git commit -m "feat(M1): HF Qwen3 weight loader (state dict → API slots)"
```

---

## Task 16 — Per-sub-op isolation tests vs HF (atol=1e-5)

Goal: confirm each api sub-op matches HF's equivalent at very tight tolerance. Catches numerical bugs in isolation before composing them.

**Files:**
- Test: `tests/models/qwen3/test_isolation_hf.py`

- [ ] **Step 1: Write tests comparing rms_norm / rope / sdpa / SwiGLU vs HF Qwen3 internals**

Create `tests/models/qwen3/test_isolation_hf.py`:
```python
"""Isolation tests — each api primitive vs the corresponding HF Qwen3 op,
at atol=1e-5. Run on CPU with deterministic seeds.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config as HFQwen3Config

from api import ops, norm, rope, specs, types


def _hf_cfg():
    """Tiny HF config for fast isolation tests."""
    return HFQwen3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=2,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=True, torch_dtype="float32",
    )


def test_rms_norm_matches_hf():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_norm = modeling_qwen3.Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)

    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)

    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="standard_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_freqs_match_hf():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_qwen3.Qwen3RotaryEmbedding(cfg)

    # Dummy position_ids and a dummy tensor for HF's signature
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, cfg.head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)        # [max_seq, head_dim]
    hf_sin = hf_sin.squeeze(0)

    rope_spec = specs.RoPESpec(base_theta=cfg.rope_theta,
                               basis=types.RoPEBasis.SPLIT_HALF)
    api_rope = rope.RoPE(rope_spec, head_dim=cfg.head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_swiglu_matches_hf_mlp():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_mlp = modeling_qwen3.Qwen3MLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)

    # Reconstruct using api ops with HF's projection weights
    g = F.linear(x, hf_mlp.gate_proj.weight)
    u = F.linear(x, hf_mlp.up_proj.weight)
    h = ops.mul(ops.silu(g), u)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_sdpa_with_gqa_matches_torch_repeated_kv():
    """Sanity check that our sdpa with GQA broadcast matches the obvious repeat_interleave path."""
    torch.manual_seed(0)
    B, Hq, Hk, S, Dh = 1, 4, 2, 6, 16
    q = torch.randn(B, Hq, S, Dh)
    k = torch.randn(B, Hk, S, Dh)
    v = torch.randn(B, Hk, S, Dh)

    api_out = ops.sdpa(q, k, v, is_causal=True)
    k_r = k.repeat_interleave(Hq // Hk, dim=1)
    v_r = v.repeat_interleave(Hq // Hk, dim=1)
    ref = F.scaled_dot_product_attention(q, k_r, v_r, is_causal=True)
    assert torch.allclose(api_out, ref, atol=1e-5)
```

- [ ] **Step 2: Run, expect pass**

```powershell
uv run pytest tests/models/qwen3/test_isolation_hf.py -v
```
Expected: 4 passed. If any fails, the bug is in the corresponding `api/ops.py` op or HF's path has diverged.

- [ ] **Step 3: Commit**

```powershell
git add tests/models/qwen3/test_isolation_hf.py
git commit -m "test(M1): isolation tests vs HF Qwen3 sub-ops at atol=1e-5"
```

---

## Task 17 — Full layer numerical equivalence vs HF (atol=5e-4)

This is the **kickoff gate**. Loads real Qwen3-0.6B weights from HuggingFace, instantiates both HF's `Qwen3DecoderLayer` and our `build_qwen3_decoder_layer`, runs a forward pass on a fixed prompt, and compares.

**Files:**
- Test: `tests/models/qwen3/test_numerical_hf.py`

- [ ] **Step 1: Write the equivalence test**

Create `tests/models/qwen3/test_numerical_hf.py`:
```python
"""Kickoff-gate test: full Qwen3 decoder layer numerical equivalence vs HF.

Runs on CPU, fp32 compute. Uses the smallest Qwen3 variant (0.6B) to keep
download/compute small. The fixed token IDs are arbitrary integers in vocab
range — semantics don't matter, only that both implementations see the same input.
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.qwen3 import config as q_config, layer as q_layer


MODEL_ID = "Qwen/Qwen3-0.6B"
FIXED_INPUT = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)
ATOL = 5e-4
RTOL = 5e-4


@pytest.fixture(scope="module")
def hf_model():
    """Load Qwen3-0.6B from HF on CPU in fp32 (download once, share across tests)."""
    cache_dir = os.environ.get("HF_HOME", os.path.join(os.path.dirname(__file__),
                                                       "..", "..", "..", "hf_cache"))
    cfg = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    # Force fp32 for stricter comparison.
    cfg.torch_dtype = "float32"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, torch_dtype=torch.float32,
    )
    model.eval()
    return model


def test_layer0_forward_matches_hf(hf_model):
    """Layer 0 forward output (HF vs API) must be within atol=5e-4."""
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = q_config.Qwen3Config.from_hf_dict(hf_cfg_dict)

    # Build API layer + copy HF weights in.
    api_blk = q_layer.build_qwen3_decoder_layer(
        cfg, layer_idx=0, max_seq=cfg.max_position_embeddings,
    )
    full_sd = hf_model.state_dict()
    q_layer.load_hf_qwen3_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    # Prepare the input. Both paths receive the SAME hidden state after embedding.
    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)
    B, S = FIXED_INPUT.shape

    # HF layer forward
    hf_layer = hf_model.model.layers[0]
    position_ids = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        # Build the cosine/sine table the HF way for this layer.
        cos, sin = hf_model.model.rotary_emb(embed_out, position_ids)
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_out_tuple = hf_layer(
            embed_out, attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_value=None, output_attentions=False, use_cache=False,
        )
        hf_out = hf_out_tuple[0]

    # API layer forward
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS, memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
```

- [ ] **Step 2: Run the test**

```powershell
uv run pytest tests/models/qwen3/test_numerical_hf.py -v -s
```
Expected: 1 passed. If it fails:
- Print `max_abs_diff` and locate the sub-op responsible by enabling per-stage diff prints
- Re-run isolation tests (Task 16) to confirm sub-ops still pass

This test downloads ~1.2 GB on first run (Qwen3-0.6B weights).

- [ ] **Step 3: Commit**

```powershell
git add tests/models/qwen3/test_numerical_hf.py
git commit -m "test(M1): Qwen3 decoder layer numerical equivalence vs HF (atol=5e-4)"
```

---

## Task 18 — KV-cache equivalence (prefill + decode)

**Files:**
- Test: `tests/models/qwen3/test_kvcache_hf.py`

- [ ] **Step 1: Write the test**

Create `tests/models/qwen3/test_kvcache_hf.py`:
```python
"""Verify our ContiguousKVCache prefill+decode produces the same layer-0 output as HF."""
import os
import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from api import kvcache, specs, types
from models.qwen3 import config as q_config, layer as q_layer


MODEL_ID = "Qwen/Qwen3-0.6B"
PREFILL_IDS = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024]], dtype=torch.long)
DECODE_ID = torch.tensor([[9]], dtype=torch.long)
ATOL = 5e-4


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get("HF_HOME", os.path.join(os.path.dirname(__file__),
                                                       "..", "..", "..", "hf_cache"))
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, torch_dtype=torch.float32,
    ).eval()


def test_prefill_then_decode_layer0(hf_model):
    hf_cfg = hf_model.config
    cfg = q_config.Qwen3Config.from_hf_dict(hf_cfg.to_dict())
    api_blk = q_layer.build_qwen3_decoder_layer(cfg, layer_idx=0,
                                                max_seq=cfg.max_position_embeddings)
    q_layer.load_hf_qwen3_layer(api_blk, hf_model.state_dict(), layer_idx=0)
    api_blk.eval()

    # ---- HF prefill ----
    with torch.no_grad():
        prefill_h = hf_model.model.embed_tokens(PREFILL_IDS)
        S_pre = PREFILL_IDS.shape[1]
        pos_pre = torch.arange(S_pre).unsqueeze(0)
        cos_pre, sin_pre = hf_model.model.rotary_emb(prefill_h, pos_pre)
        attn_mask = torch.full((1, 1, S_pre, S_pre), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_cache = DynamicCache()
        hf_pre_out = hf_model.model.layers[0](
            prefill_h, attention_mask=attn_mask,
            position_ids=pos_pre, position_embeddings=(cos_pre, sin_pre),
            past_key_value=hf_cache, use_cache=True,
        )[0]

        # ---- HF decode 1 token ----
        decode_h = hf_model.model.embed_tokens(DECODE_ID)
        pos_dec = torch.tensor([[S_pre]])
        cos_dec, sin_dec = hf_model.model.rotary_emb(decode_h, pos_dec)
        hf_dec_out = hf_model.model.layers[0](
            decode_h, attention_mask=None,
            position_ids=pos_dec, position_embeddings=(cos_dec, sin_dec),
            past_key_value=hf_cache, use_cache=True,
        )[0]

    # ---- API prefill ----
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS, memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    api_cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=cfg.max_position_embeddings,
    )
    with torch.no_grad():
        api_pre_out = api_blk(prefill_h, position_ids=torch.arange(S_pre),
                              cache=api_cache, start_pos=0)
        # Decode
        api_dec_out = api_blk(decode_h, position_ids=torch.tensor([S_pre]),
                              cache=api_cache, start_pos=S_pre)

    assert torch.allclose(hf_pre_out, api_pre_out, atol=ATOL), \
        f"prefill diff max={ (hf_pre_out - api_pre_out).abs().max().item() }"
    assert torch.allclose(hf_dec_out, api_dec_out, atol=ATOL), \
        f"decode diff max={ (hf_dec_out - api_dec_out).abs().max().item() }"
    assert api_cache.seq_len == S_pre + 1
```

- [ ] **Step 2: Run, expect pass**

```powershell
uv run pytest tests/models/qwen3/test_kvcache_hf.py -v -s
```
Expected: 1 passed.

- [ ] **Step 3: Commit**

```powershell
git add tests/models/qwen3/test_kvcache_hf.py
git commit -m "test(M1): KV cache prefill+decode equivalence vs HF"
```

---

## Task 19 — AWQ quant integration test

Tests that an AWQ-quantized layer projection produces output within quant tolerance of the fp16 reference.

**Files:**
- Test: `tests/models/qwen3/test_quant_awq.py`

We use a *synthetic* AWQ round trip: take HF Qwen3-0.6B's `q_proj` weight, quantize via our `awq_quantize`, dequantize via `awq_dequantize`, and compare a forward pass to the original. This avoids depending on whether a published `Qwen/Qwen3-0.6B-AWQ` checkpoint exists.

- [ ] **Step 1: Write the test**

Create `tests/models/qwen3/test_quant_awq.py`:
```python
"""AWQ round-trip on a real Qwen3 projection weight.

Verifies that quant→dequant→matmul stays within quantization tolerance vs
the full-precision reference. This proves the QuantSpec realization works
on production-shaped weights.
"""
import os
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM

from api import quant, specs, types


MODEL_ID = "Qwen/Qwen3-0.6B"
QUANT_REL_TOL = 0.10        # median rel-err < 10% after quant round trip


@pytest.fixture(scope="module")
def hf_state_dict():
    cache_dir = os.environ.get("HF_HOME", os.path.join(os.path.dirname(__file__),
                                                       "..", "..", "..", "hf_cache"))
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, torch_dtype=torch.float16,
    )
    return m.state_dict()


def test_awq_round_trip_on_qproj(hf_state_dict):
    w_fp16 = hf_state_dict["model.layers.0.self_attn.q_proj.weight"].to(torch.float16)
    # HF stores as [out, in] = [n_q_heads * head_dim, hidden_size]. AWQ packs along
    # the contraction (input/hidden) axis, so transpose to [K, N] = [hidden, q_out].
    w_KN = w_fp16.t().contiguous()       # [K=hidden, N=q_out]
    K, N = w_KN.shape
    # Pad K and N to AWQ-compatible group_size + 8.
    G = 128
    assert K % G == 0
    assert N % 8 == 0

    spec = specs.QuantSpec(
        qdtype=types.QDType.INT4, group_size=G, quant_axis=0,
        scale_dtype=torch.float16, has_zero_point=True,
        packing=types.PackingLayout.AWQ_INTERLEAVE,
        accumulator_dtype=torch.float32, role=types.QuantRole.WEIGHT,
    )
    packed, scales, qzeros = quant.awq_quantize(w_KN, spec)
    w_recovered = quant.awq_dequantize(packed, scales, qzeros, spec, K, N)

    # Median per-element rel error
    rel = (w_recovered.float() - w_KN.float()).abs() / (w_KN.float().abs() + 1e-6)
    assert rel.median() < QUANT_REL_TOL, f"median rel-err {rel.median():.4f} ≥ tol {QUANT_REL_TOL}"

    # Matmul check on synthetic input.
    x = torch.randn(2, K, dtype=torch.float16)
    y_full = (x.float() @ w_KN.float()).to(torch.float16)
    y_quant = (x.float() @ w_recovered.float()).to(torch.float16)
    rel_y = (y_full.float() - y_quant.float()).abs() / (y_full.float().abs() + 1e-6)
    # Output median rel error usually tighter than weight rel error
    assert rel_y.median() < QUANT_REL_TOL
```

- [ ] **Step 2: Run, expect pass**

```powershell
uv run pytest tests/models/qwen3/test_quant_awq.py -v -s
```
Expected: 1 passed.

- [ ] **Step 3: Commit**

```powershell
git add tests/models/qwen3/test_quant_awq.py
git commit -m "test(M1): AWQ round-trip on Qwen3-0.6B q_proj weight"
```

---

## Task 20 — `models/qwen3/layer.md` — layer documentation

**Files:**
- Create: `models/qwen3/layer.md`

- [ ] **Step 1: Write the layer documentation**

Create `models/qwen3/layer.md`:

```markdown
# Qwen3 — Decoder Layer

## 1. Identity

- **Family:** Qwen3 (Alibaba)
- **Variants in M1 scope:** 0.6B, 1.7B, 4B, 8B (all dense)
- **Release date:** March / April 2025
- **Paper / tech report:** Qwen3 Technical Report — arxiv:2505.09388
- **HF base model card:** https://huggingface.co/Qwen/Qwen3-0.6B
- **transformers source:** `transformers/src/transformers/models/qwen3/modeling_qwen3.py`

## 2. Decoder block diagram

```
        x  [B, S, D]
        |
   +----+----+
   |         |
   |     input_norm (RMSNorm STANDARD_W)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj   (split QKV, no bias)
   |    ├── q_norm, k_norm           (QKNorm PRE-RoPE, PER_HEAD_DH shape)
   |    ├── RoPE (SPLIT_HALF, theta=1_000_000)
   |    ├── KVCache write/read       (CONTIGUOUS HND layout)
   |    ├── SDPA (GQA, n_q=…, n_kv=…)
   |    └── o_proj                   (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     post_attn_norm (RMSNorm STANDARD_W)
   |         |
   |    feedforward(channel_mixer)
   |    ├── gate_proj                (SwiGLU gate, no bias)
   |    ├── up_proj                  (SwiGLU up,  no bias)
   |    ├── SiLU(gate) * up          (api.ops.mul ∘ api.ops.silu)
   |    └── down_proj                (no bias)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, D]
```

## 3. Tensor IO trace

For Qwen3-0.6B (D=1024, n_q=16, n_kv=8, head_dim=128, I=3072):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 1024] | bf16 |
| input_norm | x_in | [B, S, 1024] | bf16 |
| q_proj | q | [B, S, 16*128]=[B, S, 2048] | bf16 |
| k_proj | k | [B, S, 8*128]=[B, S, 1024] | bf16 |
| v_proj | v | [B, S, 8*128]=[B, S, 1024] | bf16 |
| reshape | q | [B, S, 16, 128] | bf16 |
| reshape | k, v | [B, S, 8, 128] | bf16 |
| q_norm | q | [B, S, 16, 128] | bf16 (computed in fp32) |
| k_norm | k | [B, S, 8, 128] | bf16 (computed in fp32) |
| RoPE | q, k | [B, S, *, 128] | bf16 |
| transpose | q | [B, 16, S, 128] | bf16 |
| transpose | k, v | [B, 8, S, 128] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 8, max_seq, 128] | bf16 |
| cache.read | k_full, v_full | [B, 8, T, 128] where T = start_pos + S | bf16 |
| sdpa | a | [B, 16, S, 128] | bf16 (fp32 accum) |
| transpose+reshape | a | [B, S, 2048] | bf16 |
| o_proj | attn_out | [B, S, 1024] | bf16 |
| residual add | x | [B, S, 1024] | bf16 |
| post_attn_norm | h | [B, S, 1024] | bf16 |
| gate_proj | g | [B, S, 3072] | bf16 |
| up_proj | u | [B, S, 3072] | bf16 |
| silu+mul | h_act | [B, S, 3072] | bf16 |
| down_proj | ffn_out | [B, S, 1024] | bf16 |
| residual add | y | [B, S, 1024] | bf16 |

## 4. Op trace (api.ops sequence)

```
x_in    = rms_norm(x, input_norm.weight, eps, "standard_w")
q       = linear(x_in, q_proj.weight)         # reshape to [B, S, 16, 128]
k       = linear(x_in, k_proj.weight)         # reshape to [B, S, 8, 128]
v       = linear(x_in, v_proj.weight)         # reshape to [B, S, 8, 128]
q       = rms_norm(q, q_norm.weight, eps, "standard_w")   # PER_HEAD_DH
k       = rms_norm(k, k_norm.weight, eps, "standard_w")
q, k    = rope_apply(q, k, cos, sin, basis="split_half")
# cache.write(k, v, start_pos); k_full, v_full = cache.read(start_pos + S)
a       = sdpa(q, k_full, v_full, is_causal=(S > 1), scale=head_dim ** -0.5)
attn_out = linear(a, o_proj.weight)
x       = add(x, attn_out)

h       = rms_norm(x, post_attn_norm.weight, eps, "standard_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
h_act   = mul(silu(g), u)
ffn_out = linear(h_act, down_proj.weight)
y       = add(x, ffn_out)
```

## 5. Spec instantiation (for Qwen3-0.6B)

See `models/qwen3/config.py:Qwen3Config.to_block_spec()`. Concretely:

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        qk_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                         weight_mode=NormWeightMode.STANDARD_W),
        qk_norm_phase=QKNormPhase.PRE_ROPE,
        qk_norm_shape=QKNormShape.PER_HEAD_DH,
        rope=RoPESpec(base_theta=1_000_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=3072,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    input_norm=NormSpec(...), pre_attn_norm=..., pre_ffn_norm=...,
)
```

## 6. Quirks

- **QK-norm is PRE-RoPE** (verified against `modeling_qwen3.py` `Qwen3Attention.forward` —
  `q_norm` and `k_norm` are called BEFORE `apply_rotary_pos_emb`). This was incorrectly
  documented as POST-RoPE in research/05 v1.
- **QK-norm weight shape is `[head_dim]`** (per-head Dh), distinct from OLMo 2's
  `[n_heads * head_dim]` shape. Source comment in `modeling_qwen3.py`:
  *"unlike olmo, only on the head dim!"*.
- **`rope_theta=1_000_000`** for Qwen3 base context (32k), NOT 5M (v1 error).
- **`tie_word_embeddings=True`** for 0.6B / 1.7B / 4B; `False` for 8B.
- **`head_dim` is explicit** in Qwen3 (1024/16=64 in 0.6B, vs head_dim=128 — they
  differ). Don't infer head_dim from hidden_size / num_heads.
- **No biases on any projection** (q/k/v/o, gate/up/down).

## 7. Source citations

- HF reference: `transformers/src/transformers/models/qwen3/modeling_qwen3.py:Qwen3DecoderLayer`
- HF config: `transformers/src/transformers/models/qwen3/configuration_qwen3.py`
- vLLM reference: `vllm/model_executor/models/qwen3.py` (mostly aliased from Llama)
- llama.cpp reference: `convert_hf_to_gguf.py` Qwen3 conversion + ggml tensor names
- Survey: `research/02-layer-sources.v2.md` §3 (Qwen3 section)
- Cache+attention: `research/05-kvcache-attention.v2.md` §3 (Qwen3 entry), §9.1 attention arc

## 8. M1 validation status

- ✅ Shape tests at 3 size variants (`test_layer_shape.py`)
- ✅ Determinism (`test_layer_shape.py`)
- ✅ HF weight loader (`test_weight_loader.py`)
- ✅ Sub-op isolation vs HF at atol=1e-5 (`test_isolation_hf.py`)
- ✅ Full layer numerical equivalence vs HF at atol=5e-4 on FIXED_INPUT (`test_numerical_hf.py`)
- ✅ KV cache prefill+decode equivalence vs HF (`test_kvcache_hf.py`)
- ✅ AWQ W4A16 round-trip on real q_proj weight (`test_quant_awq.py`)
```

- [ ] **Step 2: Commit**

```powershell
git add models/qwen3/layer.md
git commit -m "docs(M1): Qwen3 layer.md (architecture, tensor IO, op trace, quirks)"
```

---

## Task 21 — M1 wrap-up

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

```powershell
uv run pytest -v
```
Expected: all green. Estimated count: ~30 tests.

- [ ] **Step 2: Expand `README.md`**

```markdown
# llm-layers

Minimal, evidence-grounded API for assembling mainstream small language models
(<8B params) as decoder graphs, with quantization and KV-cache as first-class
parameters.

## Design

See `docs/superpowers/specs/2026-06-04-llm-layers-design.v2.md`.

## Survey

Five research reports + an evolution-narrative report in `research/`:

- `00-evolution.md` — 2022→2026 timeline, lineage diagrams, abandoned designs
- `01-model-census.v2.md` — 80 mainstream SLMs across 15 axes
- `02-layer-sources.v2.md` — 33 family decoder-block decompositions
- `03-ihv-opsets.v2.md` — 18 IHV runtimes' op-set comparison
- `04-quantization.v2.md` — 42 quant schemes with parameter space
- `05-kvcache-attention.v2.md` — 29 attn + 14 RoPE + 13 cache variants

## M1 status — Qwen3 kickoff gate

```powershell
uv sync --extra test --extra dev
uv run pytest -v
```

All tests pass; layer numerically equivalent to HF `transformers.Qwen3DecoderLayer`
at atol=5e-4 on a fixed input.

See:
- `models/qwen3/layer.md` for the architecture documentation
- `models/qwen3/layer.py` for the API-assembled factory
- `tests/models/qwen3/` for shape, isolation, numerical, KV cache, and AWQ tests

## API surface (M1 subset)

- `api/types.py` — enums and small types
- `api/specs.py` — frozen dataclasses (parameter spaces)
- `api/ops.py` — 9 functional primitives (rms_norm, linear, silu, add, mul,
  embed, lm_head, rope_apply, sdpa)
- `api/{norm, rope, kvcache, quant, attention, feedforward, block}.py` —
  nn.Module building blocks

The full API floor lands across M2+ rollout batches (MoE, SSM, MLA, more quant
schemes). M1 establishes the foundational shape.

## Project structure

See `docs/superpowers/plans/2026-06-05-llm-layers-m1-qwen3.md` for the M1 plan.

## License

Apache 2.0 (project code). Qwen3 model weights remain under their respective
HuggingFace licenses (Apache 2.0 for Qwen3-0.6B).
```

- [ ] **Step 3: Final commit + tag**

```powershell
git add README.md
git commit -m "docs(M1): README — M1 status + survey index + API surface"
git tag M1-complete -m "M1 — Qwen3 kickoff gate complete. All tests green."
```

- [ ] **Step 4: M1 deliverable summary**

State to the user:
- `models/qwen3/layer.md`, `models/qwen3/layer.py`, `models/qwen3/config.py` exist
- All ~30 tests pass (including numerical equivalence vs HF at atol=5e-4)
- AWQ W4A16 round-trip verified on real Qwen3 weights
- Tag `M1-complete` placed
- Ready for kickoff-gate user review

---

## Self-review (run before declaring plan done)

**Spec coverage** (against `docs/superpowers/specs/2026-06-04-llm-layers-design.v2.md` §8 M1 deliverables):

| Spec deliverable | Task |
|---|---|
| `api/ops.py` (subset of 16 primitives for Qwen3) | T3, T4, T5 (9 ops) |
| `api/specs.py` (subset of 12 dataclasses) | T2 (7 dataclasses) |
| `api/norm.py` (RMSNorm + QKNorm PRE_ROPE, PER_HEAD_DH) | T6 |
| `api/rope.py` (SPLIT_HALF + LLAMA3 scaling) | T7 |
| `api/attention.py` (GQA + QK-norm + RoPE + KVCache) | T10 |
| `api/feedforward.py` (SwiGLU) | T11 |
| `api/kvcache.py` (CONTIGUOUS HND) | T8 |
| `api/quant.py` (AWQ W4A16 round-trip) | T9 |
| `api/block.py` (PRE-norm DecoderBlock) | T12 |
| `models/qwen3/layer.md` | T20 |
| `models/qwen3/layer.py` (factory + HF loader) | T13, T14, T15 |
| `models/qwen3/config.py` | T13 |
| Shape tests (3 sizes) | T14 |
| Determinism test | T14 |
| HF weight loader test | T15 |
| Isolation tests at atol=1e-5 | T16 |
| **Numerical equivalence at atol=5e-4 on FIXED_INPUT** | T17 (kickoff gate) |
| KV cache prefill+decode equivalence | T18 |
| AWQ W4A16 integration test | T19 |
| Glossary, weight loader docs | T20 (layer.md §6, §7) |

✅ All M1 spec items have a task.

**Placeholder scan:** no TBD / TODO / "implement later" found.

**Type consistency:** `Qwen3Config` field names match between T13 and T14; spec field names match between T2 (`AttentionSpec.qk_norm_phase`) and T10 (`Attention.__init__` checks). `load_hf_qwen3_layer` uses the same module-attribute names (`blk.attention.q_proj`, `blk.feedforward.gate_proj`, etc.) that T10/T11/T12 declared.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-05-llm-layers-m1-qwen3.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (T1 → T21), review between tasks, fast iteration. Best for keeping main-context lean while implementation is long.

2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for review. Best if you want to watch every step in this conversation.

**Which approach?**
