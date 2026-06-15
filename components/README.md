# `components/` — the typed-spec API for assembling decoder layers

## What this is

The `components/` package is the **runtime-agnostic, typed-spec API** for
describing the parts of a transformer decoder layer. It defines a small set of
logical components — Norm, Attention, PositionalEncoding, FFN, plus graph
primitives (Add, Mul) — as plain `@dataclass(frozen=True)` Specs with
discriminated-union variants. The block itself is a `BlockGraph` DAG of those
component instances.

There is **zero ML-framework dependency in this package**. No `torch`, no
`numpy`, no `openvino`, no `onnx`. The only imports are stdlib (`dataclasses`,
`enum`, `typing`). The Specs are pure data: hashable, immutable, type-checked.

Those Specs are the input to the three independent ports under `runtimes/`:

- `runtimes/torch_port/` — PyTorch ground-truth tensor implementation (fp32).
- `runtimes/openvino_port/` — Intel Arc 140V GPU via OpenVINO 2026.2.
- `runtimes/onnx_port/` — ONNX Runtime CPU EP via Microsoft contrib ops.

Each port consumes the same Spec tree and produces equivalent computation;
parity tests under `tests/parity/` verify the three ports agree numerically.

## Design principles

These are load-bearing — most of the surface area follows from them.

- **No ML-framework dependency in `components/`.** `import torch`,
  `import numpy`, `import openvino`, and `import onnx` are forbidden inside
  this package. Tensor types belong to the runtime ports.
- **No model-personal names in symbols.** `Qwen3`, `Gemma4`, `Llama`,
  `DeepSeek`, etc. must not appear in any class name, slug, field name, or
  module name. Algorithm- and op-level names (`MHA`, `GQA`, `MLA`, `RoPE`,
  `SwiGLU`, `YaRN`) are allowed because they are model-agnostic. Model
  families are only allowed in `# Source: modeling_*.py:NN` citation
  comments.
- **Discriminated unions per component, not mega-classes.**
  `AttentionSpec = MHASpec | GQASpec | MLASpec | GQASinksSpec`. Each variant
  is its own dataclass with its own `kind: Literal[...]` tag. We do *not*
  collapse everything into one `AttentionSpec` with `Optional` fields plus
  runtime validation rules — that re-introduces the variant fork at the
  dispatch site.
- **Topology is data, not a class kind.** A decoder block is a
  `BlockGraph(nodes, output)` DAG. PreNorm, OLMo-2-style post-norm-reordered,
  Falcon-style parallel residual, and Gemma-style sandwich are all just
  different node lists, not different block subclasses. The four factory
  helpers (`pre_norm_block`, `post_norm_reordered_block`, `parallel_block`,
  `sandwich_block`) are sugar over `BlockGraph(...)` for the common shapes.
- **LoRA is a uniform composition slot.** Every component-instance spec
  inherits from `_ComponentBase` and carries a `lora: tuple[LoRAAdapterSpec,
  ...] = ()` field. Adapters attach by `target_slot` to weight names
  declared in the variant's `ApiSchema.weights`.
- **All collections are `tuple[...]`.** Frozen dataclasses must be hashable
  so the same Spec instance can be shared across layers and used as a cache
  key. `Node.__post_init__` and `_ComponentBase.__post_init__` reject
  `list` explicitly.
- **`ApiSchema.attributes` is auto-derived** from `dataclasses.fields(spec_cls)`
  joined with a class-level `_ATTR_META: dict[str, AttrMeta]` map. You
  hand-author only `inputs`, `outputs`, and `weights` in each Variant.
  `derive_api_attributes()` raises `ValueError` if a dataclass field is
  missing from `_ATTR_META`, which prevents the two-sources-of-truth bug.

## Module layout

```text
components/
  _types.py                # Spec base, _ComponentBase, DType, LoRAAdapterSpec,
                           # ApiField, AttrMeta, ApiSchema, ComponentMeta, Variant,
                           # derive_api_attributes()
  norm.py                  # NormStandardSpec | NormZeroCenteredSpec
  attention.py             # MHASpec | GQASpec | MLASpec | GQASinksSpec
  positional_encoding.py   # RoPESpec | RoPESmoothScalingSpec | RoPEYaRNSpec |
                           # RoPELongRoPESpec | RoPELinearSpec |
                           # MRoPEInterleavedSpec | NoPESpec
  ffn.py                   # SwiGLUSpec | GeGLUSpec | ClampedSwiGLUSpec
  misc.py                  # AddSpec, MulSpec  (graph primitives, not registry items)
  block_graph.py           # Node, BlockGraph + 4 factory helpers
  __init__.py              # re-exports + ALL_COMPONENTS / get_component()
```

The four "core" components (`norm`, `attention`, `positional-encoding`, `ffn`)
register a module-level `COMPONENT: ComponentMeta` and are collected into
`ALL_COMPONENTS`. `misc.py` defines graph primitives that are intentionally
*not* registry entries — they exist only to wire residuals and elementwise
combines inside a `BlockGraph`. `BlockGraph` itself is the
"DecoderBlock" component: a block has no kind tag, it has a topology.

The full 9-component target listed in the design doc — Embedding,
PositionalEncoding, Norm, Attention, FFN, MoEBlock, KVCache, LMHead,
DecoderBlock-via-BlockGraph — is the M0+M1 surface. M0 (current) ships Norm,
Attention, PositionalEncoding, FFN, and DecoderBlock-via-BlockGraph. The
rest are on the M1 backlog (see Status below).

## The 3-layer mental model

For every component you'll see three things stacked together. It pays to keep
them straight:

1. **`*Spec` — the WHAT.** A `@dataclass(frozen=True)` capturing every
   variant attribute. Type-checked, hashable, immutable. Carries `lora` via
   `_ComponentBase`. Carries `kind: Literal[...]` as the union discriminator.
   This is what runtime ports pattern-match on.

2. **`Variant` — the docs row.** Wraps one canonical `Spec` instance with an
   `ApiSchema` (inputs, outputs, attributes, weights). The `Variant` is what
   a documentation site or a downstream tooling registry consumes. The
   `Spec` field on a Variant is filled with realistic example numbers
   (e.g. `MHASpec(num_heads=32, head_dim=128)`), not zeros.

3. **`ComponentMeta` — the registry entry.** Per-component package metadata:
   `slug`, `name`, `oneliner`, `status`, `badges`, and the tuple of
   `Variant`s. Exposed via `get_component("attention")`.

A *block* is a separate concern. A block is a `BlockGraph`: a tuple of
`Node(name, spec, inputs)` entries plus an `output` node name. The reserved
input name `"input"` denotes the block's external input. `BlockGraph` enforces
its own invariants in `__post_init__`: no duplicate node names, every
referenced input must be `"input"` or a prior node, and `output` must name
an existing node.

## Five-minute walkthrough

Construct a single Llama-/Qwen-shape pre-norm decoder layer using a factory
plus four Spec instances:

```python
from components import (
    NormStandardSpec, GQASpec, RoPESpec, SwiGLUSpec, pre_norm_block,
)

block = pre_norm_block(
    block_norm = NormStandardSpec(eps=1e-6),
    attention = GQASpec(num_heads=16, head_dim=128, num_kv_heads=8),
    positional_encoding = RoPESpec(theta=10_000_000.0),
    post_attention_norm = NormStandardSpec(eps=1e-6),
    ffn = SwiGLUSpec(intermediate_size=3072),
)

for node in block.nodes:
    print(f"  {node.name:6s} <- {node.inputs}  ::  {type(node.spec).__name__}")
print(f"  output: {block.output}")
```

Output:

```text
  n0     <- ('input',)  ::  NormStandardSpec
  rope   <- ('n0',)  ::  RoPESpec
  attn   <- ('rope',)  ::  GQASpec
  r0     <- ('input', 'attn')  ::  AddSpec
  n1     <- ('r0',)  ::  NormStandardSpec
  ffn    <- ('n1',)  ::  SwiGLUSpec
  r1     <- ('r0', 'ffn')  ::  AddSpec
  output: r1
```

A few things to notice:

- The DAG is explicit: each `Node` lists its source-node names. The first
  residual sums `"input"` and `"attn"`; the second sums `"r0"` and
  `"ffn"`. There is no implicit residual; if it isn't a `Node`, it doesn't
  exist.
- The `rope` node is inserted between `n0` and `attn` because
  `GQASpec.rope_applied_inside` defaults to `False` (M0 keeps RoPE outside
  the attention op). When a future variant sets that flag, the factory
  drops the rope node and lets the attention op apply RoPE internally.
- The two residual nodes use the same `AddSpec()` instance type, which has
  no tunable attributes — its arity is determined by the surrounding
  `Node`'s `inputs` length. That's why one Spec can do both a 2-way
  residual and (in `parallel_block`) a 3-way residual.

Now the same component types, completely different model family. A
Gemma-style sandwich block with zero-centered norms and GeGLU:

```python
from components import (
    NormZeroCenteredSpec, GQASpec, RoPESpec, GeGLUSpec, sandwich_block,
)

block = sandwich_block(
    pre_attention_norm = NormZeroCenteredSpec(eps=1e-6),
    attention = GQASpec(num_heads=8, head_dim=256, num_kv_heads=4,
                        sliding_window=4096),
    positional_encoding = RoPESpec(theta=1_000_000.0),
    post_attention_norm = NormZeroCenteredSpec(eps=1e-6),
    pre_ffn_norm = NormZeroCenteredSpec(eps=1e-6),
    ffn = GeGLUSpec(intermediate_size=16384, activation="gelu_tanh"),
    post_ffn_norm = NormZeroCenteredSpec(eps=1e-6),
)

for node in block.nodes:
    print(f"  {node.name:10s} <- {node.inputs}  ::  {type(node.spec).__name__}")
print(f"  output: {block.output}")
```

Output:

```text
  n_pre_a    <- ('input',)  ::  NormZeroCenteredSpec
  rope       <- ('n_pre_a',)  ::  RoPESpec
  attn       <- ('rope',)  ::  GQASpec
  n_post_a   <- ('attn',)  ::  NormZeroCenteredSpec
  r0         <- ('input', 'n_post_a')  ::  AddSpec
  n_pre_f    <- ('r0',)  ::  NormZeroCenteredSpec
  ffn        <- ('n_pre_f',)  ::  GeGLUSpec
  n_post_f   <- ('ffn',)  ::  NormZeroCenteredSpec
  r1         <- ('r0', 'n_post_f')  ::  AddSpec
  output: r1
```

Same vocabulary (`Norm`, `GQA`, `RoPE`, `FFN`, `Add`); different `kind` tags
on the norms; different intermediate dims; an extra norm before each
residual add; and `sliding_window=4096` on the attention spec wakes up SWA
behavior in the runtime ports. The variant attributes are how one
`AttentionSpec` covers radically different model behaviors without growing
new Python classes.

Quick tour of the more interesting attribute knobs you'll see in M1
configurations:

- `GQASpec.output_gate="sigmoid"` — Qwen3.5/3.6 per-head output gate.
- `GQASpec.qk_norm=NormStandardSpec(eps=1e-6)` — Q/K norm before RoPE.
- `GQASpec.kv_source_layer_offset=-1` — Gemma-4 cross-layer / Hunyuan CLA;
  source K/V from an earlier layer's cache.
- `MLASpec(latent_dim, nope_head_dim, rope_head_dim, q_lora_rank)` —
  DeepSeek-style latent attention; `scale` is overridden to
  `1/sqrt(nope_head_dim + rope_head_dim)`.
- `GQASinksSpec(sinks)` — GPT-OSS softmax-sink logit; the `sinks` weight
  is also the runtime input slot of the same name (when input and weight
  share a name, the weight is the source of the input).
- `ClampedSwiGLUSpec(clamp=7.0, alpha=1.702)` — GPT-OSS asymmetric clamp.

## How to extend

Adding a new variant to an existing component is mechanical:

1. Pick the component module (e.g. `attention.py`).
2. Add a new `@dataclass(frozen=True, kw_only=True)` subclass of the
   relevant base (e.g. `_AttentionBase`). Give it a unique
   `kind: Literal["..."] = "..."` tag.
3. Add `_ATTR_META: dict[str, AttrMeta]` covering **every** dataclass
   field except `kind` and `lora`. `derive_api_attributes()` raises if
   anything is missing.
4. Hand-author the variant's `ApiSchema` (inputs, outputs, weights).
   `attributes` should be `derive_api_attributes(YourSpec)`.
5. Build a `Variant(slug, name, description, spec=..., api=...)` with
   a realistic example `spec`.
6. Append the new spec class to the type-alias union (e.g.
   `AttentionSpec = MHASpec | GQASpec | MLASpec | GQASinksSpec |
   YourNewSpec`) and append the new `Variant` to
   `COMPONENT.variants`.

Adding a new top-level *component* additionally requires creating the
`COMPONENT: ComponentMeta` and appending it to `ALL_COMPONENTS` in
`__init__.py`, plus re-exporting from `__all__`.

Adding a new *block topology* means writing a new factory helper in
`block_graph.py` that returns a `BlockGraph`. No new classes; topology is
data.

## Relation to the runtime ports

The three runtime ports share the `components/` API but lower it in three
different ways. The split matters because each port has different
constraints:

- **`runtimes/torch_port/`** is the numerical ground truth. PyTorch
  eager-mode tensor math, fp32 accumulators, no fused kernels. Every
  other port is checked against this implementation. If a runtime port
  disagrees with torch_port, torch_port wins by default.
- **`runtimes/openvino_port/`** lowers each Spec to an `ov.Model` op graph
  and runs on the Intel Arc 140V GPU under OpenVINO 2026.2. Many
  composite ops are emitted as builtin op patterns; some specs (e.g.
  `GQASinksSpec`) require lowering to lower-level ops.
- **`runtimes/onnx_port/`** emits an ONNX graph using Microsoft contrib
  ops (`com.microsoft.GroupQueryAttention`, etc.) and runs on the ORT
  CPU EP. For specs without a contrib-op match (e.g. `GQA+Sinks`), the
  port emits a direct opset-14 expansion.

Parity tests live under `tests/parity/`:

- `test_torch_self.py` — torch_port produces stable golden output.
- `test_openvino_vs_torch.py` — openvino_port matches torch_port.
- `test_onnx_vs_torch.py` — onnx_port matches torch_port.

The golden tensor is `tests/parity/golden_m0.pt`; the M0 ONNX export is
`tests/parity/m0_block.onnx`.

## Status

### M0 (current)

The five Spec variants used in the vertical-slice decoder block are wired
through all three runtime ports with passing parity tests:

- `NormStandardSpec` (RMSNorm)
- `GQASpec` (Grouped-Query Attention, no extension flags)
- `RoPESpec` (plain rotary)
- `SwiGLUSpec` (non-fused gate/up)
- `AddSpec` (residual)

### M1 (next)

Variants that are *defined* in `components/` today but do not yet have
all three runtime ports:

- `GQASinksSpec` — onnx_port via direct opset-14 emission (no contrib op).
- `MLASpec` — DeepSeek-style latent attention end-to-end.
- `MRoPEInterleavedSpec` — multi-axis RoPE for vision-bearing positions.
- `ClampedSwiGLUSpec` — GPT-OSS clamp/alpha.
- `NormZeroCenteredSpec` — Gemma `(1 + w) * x_normed` form.
- `RoPESmoothScalingSpec`, `RoPEYaRNSpec`, `RoPELongRoPESpec`,
  `RoPELinearSpec` — long-context RoPE families.
- `GeGLUSpec` — Gemma family.
- LoRA adapters (the `LoRAAdapterSpec` plumbing is in place; ports do
  not yet honor it).
- `KVCache` as a first-class component, and the
  `kv_source_layer_offset` cross-layer cache wiring on `GQASpec`.

### Open backlog

- **`_ATTR_META` and inheritance.** `_ATTR_META` is a plain class
  attribute, so a subclass dict shadows the parent's. If a base class
  later gains attributes, every subclass `_ATTR_META` must be updated by
  hand. A `__mro__`-merging helper is the planned fix; it isn't urgent
  while the bases are stable.
- **Canonical-setup weight rescaling for fp16 parity.** Some specs
  (notably MLA and Gemma sandwich) need pre-baked weight scaling to land
  inside the fp16 dynamic range during parity tests. The current
  `tests/parity/_canonical.py` hand-rolls this; a general
  `canonicalize(spec, weights)` is on the M1 list.
- **`sandwich_block` factory completion.** The factory exists and is
  validated for the GQA-only path. The
  `positional_encoding=None` short-circuit and the `qk_norm` interaction
  inside the sandwich norms need a small refactor before M1 ships.

## Pointers

- **Design spec (rationale, not duplicated here):**
  `docs/superpowers/specs/2026-06-15-llm-layers-v2-design.md`
- **Microsoft WinML CLI source-of-truth registry (TS reference):**
  `C:\Users\zhengte\BYOM\ModelKits\llm\docs\llm\registry\`
- **Org-wide Cardinal Rule (no model-personal symbols, etc.):**
  `C:\Users\zhengte\BYOM\ModelKits\llm\CLAUDE.md`
- **Runtime ports:**
  `runtimes/torch_port/`, `runtimes/openvino_port/`, `runtimes/onnx_port/`
- **Parity tests:** `tests/parity/`
