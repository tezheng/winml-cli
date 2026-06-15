# `components/` — API Specification

Normative reference for the public API exported by the `components/` package.
This document describes shape and contract; for rationale see
`docs/superpowers/specs/2026-06-15-llm-layers-v2-design.md`, and for an
introductory tour see [`README.md`](./README.md).

---

## 1. Scope and conformance

The **public API** is exactly the set of symbols re-exported by
`components.__init__` (enumerated in section 10). Symbols imported directly
from `components._types` or a sibling module (e.g. `components.attention.MHASpec`
rather than `components.MHASpec`), the per-module `COMPONENT` constants, the
`_ATTR_META` class attribute, and any name prefixed with `_`, are **internal**
and MAY change without notice.

This package is part of a proof-of-concept. **Backwards compatibility is not
guaranteed across commits.** Breaking changes land without a deprecation
window; consumers MUST pin to a specific revision.

A conforming consumer MUST treat every dataclass listed in this document as
frozen (section 2), MUST NOT mutate any field, and MUST NOT subclass any
concrete spec type that is part of a discriminated union (subclassing breaks
both `isinstance` narrowing on the union and the `_ATTR_META` contract for
`derive_api_attributes`).

---

## 2. Conventions

1. **Frozen dataclasses.** Every type holding API or component data is
   `@dataclass(frozen=True)`. Mutation raises `FrozenInstanceError`. Equality
   is structural; hashable iff all fields are hashable.
2. **Tuples, never lists.** Every collection-valued field has type
   `tuple[T, ...]`. Several types enforce this in `__post_init__` and raise
   `TypeError` on `list`; the rest rely on the declared type. Consumers MUST
   pass tuples.
3. **Discriminator field.** Every concrete spec subclass participating in a
   `TypeAlias = A | B | ...` union declares `kind: Literal["<TagName>"] =
   "<TagName>"`. Both `isinstance(spec, FooSpec)` and `spec.kind == "Foo"`
   discrimination are supported and stable within a commit.
4. **Field defaults are mandatory (with a `kw_only=True` escape hatch).**
   Because `_ComponentBase` introduces a defaulted `lora` field that all
   component-instance specs inherit, every added field is either defaulted
   or declared inside `@dataclass(frozen=True, kw_only=True)`. The latter is
   used for attention, FFN, and positional-encoding variants so required
   fields (`num_heads`, `head_dim`, `intermediate_size`) need no fake
   default; construction without the keyword raises `TypeError`.
5. **`_ATTR_META` class attribute.** Every spec subclass whose attributes
   surface through `ApiSchema.attributes` declares
   `_ATTR_META: dict[str, AttrMeta]` covering every dataclass field except
   the discriminator `kind` and the inherited `lora`. Omissions cause
   `derive_api_attributes` to raise `ValueError` at import time.
6. **No model-family names.** The Cardinal Rule: no public symbol, slug, or
   `kind` literal contains a model-family or vendor name (Llama, Qwen, Gemma,
   GPT-OSS, DeepSeek, Phi, etc.). In-source comments and docstrings MAY cite
   sources via `# Source: modeling_<family>.py:NNN`.

---

## 3. Type fundamentals (from `_types.py`)

All fundamentals are exported from the top-level `components` package.

### 3.1 `DType`

```python
class DType(str, Enum): ...
```

String-valued enum; the `.value` string is what consumers serialize.

| Member | Value | Member | Value | Member | Value |
|---|---|---|---|---|---|
| `F16` | `"f16"` | `F32` | `"f32"` | `I8` | `"i8"` |
| `BF16` | `"bf16"` | `F64` | `"f64"` | `I16` | `"i16"` |
| `I32` | `"i32"` | `I64` | `"i64"` | `U4` | `"u4"` |
| `U8` | `"u8"` | `BOOL` | `"bool"` | | |

Inheritance from `str` means `DType.F16 == "f16"` is `True`.

### 3.2 `Spec`

```python
@dataclass(frozen=True)
class Spec: ...
```

Marker base. No fields. Every other spec type in this package extends `Spec`
either directly (graph primitives, `Node`, `BlockGraph`) or via `_ComponentBase`
(component-instance specs).

### 3.3 `_ComponentBase`

```python
@dataclass(frozen=True)
class _ComponentBase(Spec): ...
```

Mixin for any spec that represents a **component instance** as opposed to a
graph primitive or a structural type. Provides the uniform LoRA attachment
slot.

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `lora` | `tuple[LoRAAdapterSpec, ...]` | `()` | MUST be a tuple. `__post_init__` raises `TypeError` if a `list` is passed (frozen specs require hashable fields). |

`_ComponentBase` is exported but the leading underscore signals: it is the
base class of component-instance specs, not something consumers instantiate
directly.

### 3.4 `LoRAAdapterSpec`

```python
@dataclass(frozen=True)
class LoRAAdapterSpec(Spec): ...
```

A single LoRA adapter attached to one weight slot of a component instance.

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `rank` | `int` | — (required) | LoRA bottleneck rank. |
| `alpha` | `float` | — (required) | LoRA scaling alpha. Effective scale is `alpha / rank`. |
| `target_slot` | `str` | — (required) | MUST equal a `name` from the parent component's `ApiSchema.weights`. Not enforced at construction; downstream consumers validate. |
| `dropout` | `float` | `0.0` | Adapter input dropout. |
| `init` | `Literal["zeros", "kaiming"]` | `"zeros"` | Initialization scheme for the adapter B factor. |

### 3.5 `ApiField`

```python
@dataclass(frozen=True)
class ApiField: ...
```

One slot in a component's API surface — an input, an output, an attribute, or
a weight.

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `name` | `str` | — (required) | Slot name. Unique within its parent collection. |
| `type` | `str` | — (required) | Type string, e.g. `"tensor<f16>"`, `"i64"`, `"f32"`, `"string"`. |
| `role` | `str` | — (required) | Human-readable role string. |
| `shape` | `Optional[str]` | `None` | Shape expression, e.g. `"[B, S, D]"`. `None` for scalars and attributes. |
| `default` | `Optional[str]` | `None` | Human-readable default value (a string, not a typed default). |
| `lora_attach` | `bool` | `False` | Only meaningful on weight slots; `True` means LoRA may attach here. |
| `optional` | `bool` | `False` | The slot may be omitted entirely at the call site. |
| `role_tag` | `Optional[str]` | `None` | LoRA-targeting role tag, e.g. `"attn.q"`, `"mlp.gate"`. Stable across variants of the same component. |

`ApiField` is positional-and-keyword constructible; the existing code uses
positional `ApiField(name, type, role, shape=..., ...)` for weights and inputs.

### 3.6 `AttrMeta`

```python
@dataclass(frozen=True)
class AttrMeta: ...
```

Per-field metadata for `derive_api_attributes`. Stored as the values of the
`_ATTR_META` dict on each spec class.

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `type` | `str` | — (required) | Attribute type tag, e.g. `"i64"`, `"f32"`, `"string"`, `"bool"`, `"f32[]"`, `"i64?"`, `"Norm?"`. The `?` suffix denotes optional; the `[]` suffix denotes a list/tuple. |
| `role` | `str` | — (required) | Human-readable role string. |
| `default_doc` | `Optional[str]` | `None` | Human-readable default; rendered into `ApiField.default`. |

### 3.7 `ApiSchema`

```python
@dataclass(frozen=True)
class ApiSchema: ...
```

The complete API surface of one component variant.

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `inputs` | `tuple[ApiField, ...]` | — (required) | Runtime inputs. |
| `outputs` | `tuple[ApiField, ...]` | — (required) | Runtime outputs. |
| `attributes` | `tuple[ApiField, ...]` | — (required) | Compile-time attributes. Typically built by `derive_api_attributes`. |
| `weights` | `tuple[ApiField, ...]` | — (required) | Trainable weights. Empty tuple for components with no weights (e.g. positional encodings). |

### 3.8 `Variant`

```python
@dataclass(frozen=True)
class Variant: ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `slug` | `str` | — (required) | Variant slug. Unique within parent ComponentMeta. |
| `name` | `str` | — (required) | Display name. |
| `description` | `str` | — (required) | One-paragraph description. |
| `spec` | `Spec` | — (required) | A representative typed spec instance for this variant (uses canonical defaults). |
| `api` | `ApiSchema` | — (required) | The variant's API surface. |

### 3.9 `ComponentMeta`

```python
@dataclass(frozen=True)
class ComponentMeta: ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `slug` | `str` | — (required) | Component slug. Unique within `ALL_COMPONENTS`. |
| `name` | `str` | — (required) | Display name. |
| `oneliner` | `str` | — (required) | One-line description. |
| `status` | `Literal["live", "stub"]` | — (required) | `"live"`: the component has at least one runtime port. `"stub"`: spec only. |
| `badges` | `tuple[str, ...]` | — (required) | Free-form taxonomy tags, e.g. `("core", "decoder-only")`. |
| `variants` | `tuple[Variant, ...]` | — (required) | Non-empty for live components. |

### 3.10 `derive_api_attributes`

```python
def derive_api_attributes(spec_cls: type) -> tuple[ApiField, ...]: ...
```

Walks `dataclasses.fields(spec_cls)` in declared order; for every field
whose name is neither `"kind"` nor `"lora"`, looks up
`spec_cls._ATTR_META[field.name]` and emits
`ApiField(name=field.name, type=meta.type, role=meta.role, default=meta.default_doc)`.
`shape`, `lora_attach`, `optional`, `role_tag` are left at their defaults
(attributes are scalars or scalar collections).

If `spec_cls` lacks `_ATTR_META`, returns `()`. If any non-skipped dataclass
field is missing from `_ATTR_META`, raises `ValueError` at import time.

---

## 4. Component: Norm (`norm.py`)

### 4.1 Spec types

```python
@dataclass(frozen=True)
class _NormBase(_ComponentBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `eps` | `float` | `1e-6` | Numerical stability epsilon inside the rsqrt. |
| `axis` | `int` | `-1` | Reduction axis (typically the last). |
| `accumulator_dtype` | `DType` | `DType.F32` | Accumulator dtype for the variance reduction. |

```python
@dataclass(frozen=True)
class NormStandardSpec(_NormBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["Standard"]` | `"Standard"` | Discriminator. |

Semantics: `y = x * rsqrt(mean(x^2, axis) + eps) * weight`.

```python
@dataclass(frozen=True)
class NormZeroCenteredSpec(_NormBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["ZeroCentered"]` | `"ZeroCentered"` | Discriminator. |

Semantics: `y = x * rsqrt(mean(x^2, axis) + eps) * (1 + weight)`.

```python
NormSpec = NormStandardSpec | NormZeroCenteredSpec
```

### 4.2 ApiSchema (identical across both variants)

- Inputs: `input: tensor<f16>` shape `[..., D]` — activation to normalize.
- Outputs: `output: tensor<f16>` shape `[..., D]`.
- Weights: `weight: tensor<f16>` shape `[D]`, `lora_attach=False`,
  `role_tag="norm.weight"`.
- Attributes: derived from `_ATTR_META` (`eps`, `axis`, `accumulator_dtype`).

### 4.3 ComponentMeta

| Property | Value |
|---|---|
| `slug` | `"norm"` |
| `name` | `"Norm"` |
| `status` | `"live"` |
| `badges` | `("core", "decoder-only")` |
| `variants` | `(rms-norm, rms-norm-zero-centered)` |

---

## 5. Component: Attention (`attention.py`)

### 5.1 Spec types

```python
@dataclass(frozen=True, kw_only=True)
class _AttentionBase(_ComponentBase): ...
```

All fields below are keyword-only (the dataclass is `kw_only=True`).

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `num_heads` | `int` | — (required) | Total query heads. |
| `head_dim` | `int` | — (required) | Per-head dim. First-class, not derived from `d_model/num_heads`. |
| `causal` | `bool` | `True` | Apply causal masking. |
| `qk_norm` | `Optional[NormSpec]` | `None` | Optional Q/K normalization before RoPE. |
| `rope_applied_inside` | `bool` | `False` | When `True`, the attention op applies RoPE internally; M0 keeps this `False`. |
| `accumulator_dtype` | `DType` | `DType.F32` | GEMM accumulator dtype. |
| `logit_softcap` | `float` | `0.0` | Tanh softcap on logits (`0.0` disables). |

```python
@property
def scale(self) -> float:
    return 1.0 / (self.head_dim ** 0.5)
```

Derived, not stored. Avoids cross-instance float-equality drift.

#### MHA

```python
@dataclass(frozen=True, kw_only=True)
class MHASpec(_AttentionBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["MHA"]` | `"MHA"` | Discriminator. |

No additional fields. Classical multi-head attention; `num_kv_heads ==
num_heads` is implicit.

#### GQA

```python
@dataclass(frozen=True, kw_only=True)
class GQASpec(_AttentionBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["GQA"]` | `"GQA"` | Discriminator. |
| `num_kv_heads` | `int` | `8` | K/V heads; MUST divide `num_heads`. |
| `sliding_window` | `Optional[int]` | `None` | Window size in tokens; `None` selects full attention. **M1 extension.** |
| `output_gate` | `Optional[Literal["sigmoid"]]` | `None` | Per-head output gating activation. **M1 extension.** |
| `v_norm` | `Optional[NormSpec]` | `None` | Optional V normalization. **M1 extension.** |
| `qk_norm_fixed_scale` | `Optional[float]` | `None` | When set, the qk-norm scale is fixed (absorbing `1/sqrt(head_dim)`) rather than learned. **M1 extension.** |
| `kv_source_layer_offset` | `int` | `0` | Negative offset to source K/V from an earlier layer's cache. `0` = own cache. **M1 extension.** |

**M0 conformance:** the five M1-extension fields are accepted at construction
time and are part of the typed surface, but no runtime port in M0 executes
them. A runtime port encountering a non-default value for any M1-extension
field SHALL raise `NotImplementedError`.

#### MLA

```python
@dataclass(frozen=True, kw_only=True)
class MLASpec(_AttentionBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["MLA"]` | `"MLA"` | Discriminator. |
| `num_kv_heads` | `int` | `16` | Equals `num_heads` for MLA (no GQA grouping). |
| `latent_dim` | `int` | `512` | KV latent compression dim (= `kv_lora_rank`). |
| `nope_head_dim` | `int` | `128` | Non-positional part of each head's Q/K. Also `v_head_dim`. |
| `rope_head_dim` | `int` | `64` | RoPE-rotated part of each head's Q/K. Shared across heads on the K side. |
| `q_lora_rank` | `Optional[int]` | `None` | Q low-rank bottleneck; `None` selects the full-rank Q path (DeepSeek-V2-Lite). |

```python
@property
def scale(self) -> float:
    return 1.0 / ((self.nope_head_dim + self.rope_head_dim) ** 0.5)
```

Overrides the base. The base's `head_dim` semantics still apply: callers
SHOULD set `head_dim == nope_head_dim + rope_head_dim`.

#### GQA + Sinks

```python
@dataclass(frozen=True, kw_only=True)
class GQASinksSpec(_AttentionBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["GQA+Sinks"]` | `"GQA+Sinks"` | Discriminator. |
| `num_kv_heads` | `int` | `8` | K/V heads. |
| `softmax_scale_log_base` | `int` | `2` | Base for sink-logit normalization. |
| `sliding_window` | `Optional[int]` | `None` | Window size in tokens; `None` selects full attention. |

```python
AttentionSpec = MHASpec | GQASpec | MLASpec | GQASinksSpec
```

### 5.2 ApiSchema — shared input fragments

A single `_MASK_INPUT` field appears in every attention variant's `inputs`:

| Name | Type | Shape | Optional | Role |
|---|---|---|---|---|
| `mask` | `tensor<f16>` | `[B?, H?, S, K]` | `True` | Optional attention mask. Dynamic dims accommodate sliding-window, ALiBi, static masks. |

### 5.3 MHA — ApiSchema

All tensors are `tensor<f16>` unless noted.

Inputs:

| Name | Shape | Optional |
|---|---|---|
| `query` | `[B, S, H*Dh]` | — |
| `key` | `[B, S, H*Dh]` | — |
| `value` | `[B, S, H*Dh]` | — |
| `mask` | `[B?, H?, S, K]` | `True` |
| `kv_cache_in` | `[2, B, H, S_cache, Dh]` | `True` |

Outputs:

| Name | Shape |
|---|---|
| `output` | `[B, S, H*Dh]` |
| `kv_cache_out` | `[2, B, H, S_cache + S, Dh]` |

Weights:

| Name | Shape | `lora_attach` | `role_tag` |
|---|---|---|---|
| `W_q` | `[H*Dh, D_model]` | `True` | `attn.q` |
| `W_k` | `[H*Dh, D_model]` | `True` | `attn.k` |
| `W_v` | `[H*Dh, D_model]` | `True` | `attn.v` |
| `W_o` | `[D_model, H*Dh]` | `True` | `attn.o` |

Attributes: derived from `MHASpec._ATTR_META`.

### 5.4 GQA — ApiSchema

Inputs (`H_q*Dh` query, `H_kv*Dh` key/value):

| Name | Shape | Optional |
|---|---|---|
| `query` | `[B, S, H_q*Dh]` | — |
| `key` | `[B, S, H_kv*Dh]` | — |
| `value` | `[B, S, H_kv*Dh]` | — |
| `mask` | `[B?, H?, S, K]` | `True` |
| `kv_cache_in` | `[2, B, H_kv, S_cache, Dh]` | `True` |

Outputs: `output` `[B, S, H_q*Dh]`, `kv_cache_out` `[2, B, H_kv, S_cache + S, Dh]`.

Weights:

| Name | Shape | `lora_attach` | `role_tag` |
|---|---|---|---|
| `W_q` | `[H_q*Dh, D_model]` | `True` | `attn.q` |
| `W_k` | `[H_kv*Dh, D_model]` | `True` | `attn.k` |
| `W_v` | `[H_kv*Dh, D_model]` | `True` | `attn.v` |
| `W_o` | `[D_model, H_q*Dh]` | `True` | `attn.o` |

Attributes: derived from `GQASpec._ATTR_META` (12 fields).

### 5.5 MLA — ApiSchema

Inputs (MLA omits the `mask` slot):

| Name | Shape | Optional |
|---|---|---|
| `query` | `[B, S, H*(nope_head_dim + rope_head_dim)]` | — |
| `key` | `[B, S, H*(nope_head_dim + rope_head_dim)]` | — |
| `value` | `[B, S, H*nope_head_dim]` | — |
| `kv_cache_in` | `[B, S_cache, latent_dim + rope_head_dim]` | `True` |

Outputs: `output` `[B, S, H*nope_head_dim]`, `kv_cache_out` `[B, S_cache + S, latent_dim + rope_head_dim]`.

Weights (six slots, three of which are mutually-exclusive on `q_lora_rank`):

| Name | Shape | Optional | `lora_attach` | `role_tag` |
|---|---|---|---|---|
| `W_dq` | `[q_lora_rank, D_model]` | `True` | `True` | `attn.q_down` |
| `W_uq` | `[H*(nope+rope), q_lora_rank]` | `True` | `True` | `attn.q_up` |
| `W_q` | `[H*(nope+rope), D_model]` | `True` | `True` | `attn.q` |
| `W_dkv` | `[latent_dim + rope_head_dim, D_model]` | — | `True` | `attn.kv_down` |
| `W_ukv` | `[H*(nope_head_dim + nope_head_dim), latent_dim]` | — | `True` | `attn.kv_up` |
| `W_o` | `[D_model, H*nope_head_dim]` | — | `True` | `attn.o` |

`W_dq` and `W_uq` are present iff `q_lora_rank is not None`; `W_q` is present
iff `q_lora_rank is None`. This is signaled by the `optional` flag on those
three slots.

### 5.6 GQA + Sinks — ApiSchema

Inputs:

| Name | Shape | Optional |
|---|---|---|
| `query` | `[B, S, H_q*Dh]` | — |
| `key` | `[B, S, H_kv*Dh]` | — |
| `value` | `[B, S, H_kv*Dh]` | — |
| `sinks` | `[H_q]` | — |
| `mask` | `[B?, H?, S, K]` | `True` |
| `kv_cache_in` | `[2, B, H_kv, S_cache, Dh]` | `True` |

Outputs: `output` `[B, S, H_q*Dh]`, `kv_cache_out` `[2, B, H_kv, S_cache + S, Dh]`.

Weights:

| Name | Shape | `lora_attach` | `role_tag` |
|---|---|---|---|
| `W_q` | `[H_q*Dh, D_model]` | `True` | `attn.q` |
| `W_k` | `[H_kv*Dh, D_model]` | `True` | `attn.k` |
| `W_v` | `[H_kv*Dh, D_model]` | `True` | `attn.v` |
| `W_o` | `[D_model, H_q*Dh]` | `True` | `attn.o` |
| `sinks` | `[H_q]` | `False` | `attn.sinks` |

**Shared-name convention.** When a weight and an input share a name — here,
`sinks` — the weight IS the source of the runtime input. There is no
projection between them; the weight tensor is fed directly into the
sublayer's input slot of the same name. Consumers MUST honor this when
materializing weights into a runtime graph.

### 5.7 ComponentMeta

| Property | Value |
|---|---|
| `slug` | `"attention"` |
| `name` | `"Attention"` |
| `status` | `"live"` |
| `badges` | `("core", "decoder-only")` |
| `variants` | `(mha, gqa, mla, gqa-sinks)` |

---

## 6. Component: PositionalEncoding (`positional_encoding.py`)

All variants extend `_ComponentBase` (so they inherit `lora`) and are declared
`kw_only=True`. All variants use the same I/O fragment except `NoPESpec`
(no `position_ids`) and `MRoPEInterleavedSpec` (rank-3 `position_ids`).

### 6.1 `RoPESpec`

```python
@dataclass(frozen=True, kw_only=True)
class RoPESpec(_ComponentBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["RoPE"]` | `"RoPE"` | Discriminator. |
| `theta` | `float` | `10000.0` | Base for inverse-frequency. |
| `partial_rotary_factor` | `float` | `1.0` | Fraction of `head_dim` rotated. |
| `partial_rotary_kind` | `Literal["prefix", "proportional"]` | `"prefix"` | Layout of the rotated/un-rotated split. |

### 6.2 `RoPESmoothScalingSpec`

```python
@dataclass(frozen=True, kw_only=True)
class RoPESmoothScalingSpec(_ComponentBase): ...
```

Renamed from the registry's `RoPE-Llama3Scaling` per the Cardinal Rule; same
algorithm, architecture-agnostic name.

| Field | Type | Default |
|---|---|---|
| `kind` | `Literal["RoPE-SmoothScaling"]` | `"RoPE-SmoothScaling"` |
| `theta` | `float` | `10000.0` |
| `factor` | `float` | `8.0` |
| `low_freq_factor` | `float` | `1.0` |
| `high_freq_factor` | `float` | `4.0` |
| `original_context_length` | `int` | `8192` |

### 6.3 `RoPEYaRNSpec`

```python
@dataclass(frozen=True, kw_only=True)
class RoPEYaRNSpec(_ComponentBase): ...
```

| Field | Type | Default |
|---|---|---|
| `kind` | `Literal["RoPE-YaRN"]` | `"RoPE-YaRN"` |
| `theta` | `float` | `10000.0` |
| `factor` | `float` | `40.0` |
| `attn_factor` | `float` | `1.0` |
| `beta_fast` | `float` | `32.0` |
| `beta_slow` | `float` | `1.0` |
| `mscale` | `float` | `1.0` |
| `mscale_all_dim` | `float` | `0.0` |
| `original_context_length` | `int` | `4096` |

### 6.4 `RoPELongRoPESpec`

```python
@dataclass(frozen=True, kw_only=True)
class RoPELongRoPESpec(_ComponentBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["RoPE-LongRoPE"]` | `"RoPE-LongRoPE"` | Discriminator. |
| `theta` | `float` | `10000.0` | |
| `factor` | `float` | `1.0` | |
| `short_factor` | `tuple[float, ...]` | `()` | MUST be tuple; `__post_init__` raises `TypeError` on `list`. |
| `long_factor` | `tuple[float, ...]` | `()` | MUST be tuple; `__post_init__` raises `TypeError` on `list`. |
| `original_context_length` | `int` | `4096` | |

### 6.5 `RoPELinearSpec`

```python
@dataclass(frozen=True, kw_only=True)
class RoPELinearSpec(_ComponentBase): ...
```

| Field | Type | Default |
|---|---|---|
| `kind` | `Literal["RoPE-Linear"]` | `"RoPE-Linear"` |
| `theta` | `float` | `10000.0` |
| `factor` | `float` | `1.0` |

### 6.6 `MRoPEInterleavedSpec`

```python
@dataclass(frozen=True, kw_only=True)
class MRoPEInterleavedSpec(_ComponentBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["MRoPE-Interleaved"]` | `"MRoPE-Interleaved"` | Discriminator. |
| `theta` | `float` | `10000.0` | |
| `sections` | `tuple[int, ...]` | `()` | Per-axis frequency slot counts (T, H, W, ...). MUST sum to `rotated_dim/2`. MUST be tuple; `__post_init__` raises `TypeError` on `list`. |
| `partial_rotary_factor` | `float` | `1.0` | |
| `partial_rotary_kind` | `Literal["prefix", "proportional"]` | `"prefix"` | |

### 6.7 `NoPESpec`

```python
@dataclass(frozen=True, kw_only=True)
class NoPESpec(_ComponentBase): ...
```

| Field | Type | Default |
|---|---|---|
| `kind` | `Literal["NoPE"]` | `"NoPE"` |

No tunable fields. `_ATTR_META = {}` (empty dict).

### 6.8 Type alias

```python
PositionalEncodingSpec = (
    RoPESpec | RoPESmoothScalingSpec | RoPEYaRNSpec | RoPELongRoPESpec
    | RoPELinearSpec | MRoPEInterleavedSpec | NoPESpec
)
```

### 6.9 ApiSchema

**RoPE family** (`RoPESpec`, `RoPESmoothScalingSpec`, `RoPEYaRNSpec`,
`RoPELongRoPESpec`, `RoPELinearSpec`):

- Inputs: `q: tensor<f16>` `[B, S, H_q, Dh]`, `k: tensor<f16>` `[B, S, H_kv, Dh]`,
  `position_ids: tensor<i64>` `[B, S]`.
- Outputs: `q_out` `[B, S, H_q, Dh]`, `k_out` `[B, S, H_kv, Dh]` (both `tensor<f16>`).

**MRoPE-Interleaved**: as above except `position_ids` is `[A, B, S]` where
`A` is the axis count (e.g. T/H/W => A=3).

**NoPE**: only `q` and `k` inputs (no `position_ids`); outputs are the
identity-passed `q_out` and `k_out`.

Weights: empty tuple `()` for every variant. Attributes: derived from each
variant's `_ATTR_META`.

### 6.10 ComponentMeta

| Property | Value |
|---|---|
| `slug` | `"positional-encoding"` |
| `name` | `"PositionalEncoding"` |
| `status` | `"live"` |
| `badges` | `("core", "decoder-only")` |
| `variants` | `(rope, rope-smooth-scaling, rope-yarn, rope-longrope, rope-linear, mrope-interleaved, nope)` |

---

## 7. Component: FFN (`ffn.py`)

All variants are `kw_only=True` with a required `intermediate_size`.

### 7.1 `SwiGLUSpec`

```python
@dataclass(frozen=True, kw_only=True)
class SwiGLUSpec(_ComponentBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["SwiGLU"]` | `"SwiGLU"` | Discriminator. |
| `intermediate_size` | `int` | — (required) | Inner/hidden dim of the MLP. |
| `fused_gate_up` | `bool` | `False` | When `True`, gate and up projections share one fused weight (Phi-style). |

Semantics: `y = down(silu(gate(x)) * up(x))`.

### 7.2 `GeGLUSpec`

```python
@dataclass(frozen=True, kw_only=True)
class GeGLUSpec(_ComponentBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["GeGLU"]` | `"GeGLU"` | Discriminator. |
| `intermediate_size` | `int` | — (required) | |
| `activation` | `Literal["gelu_tanh", "gelu_exact"]` | `"gelu_tanh"` | GeLU flavor. |

Semantics: `y = down(gelu(gate(x)) * up(x))`.

### 7.3 `ClampedSwiGLUSpec`

```python
@dataclass(frozen=True, kw_only=True)
class ClampedSwiGLUSpec(_ComponentBase): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["Clamped-SwiGLU"]` | `"Clamped-SwiGLU"` | Discriminator. |
| `intermediate_size` | `int` | — (required) | |
| `clamp` | `float` | `7.0` | Symmetric clamp magnitude on gate/up pre-activations. |
| `alpha` | `float` | `1.702` | Sigmoid-gain coefficient on the gate (Swish-beta). |

Semantics: pre-activation clamp on gate/up; Swish-beta sigmoid on gate.

### 7.4 Type alias

```python
FFNSpec = SwiGLUSpec | GeGLUSpec | ClampedSwiGLUSpec
```

### 7.5 ApiSchema (identical across the three variants)

- Inputs: `input: tensor<f16>` shape `[B, S, D_model]`.
- Outputs: `output: tensor<f16>` shape `[B, S, D_model]`.

Weights (gated form — used by all three variants in the current Variant
catalog):

| Name | Shape | `lora_attach` | `role_tag` |
|---|---|---|---|
| `W_gate` | `[I, D_model]` | `True` | `mlp.gate` |
| `W_up` | `[I, D_model]` | `True` | `mlp.up` |
| `W_down` | `[D_model, I]` | `True` | `mlp.down` |

A fused-weight alternative is defined internally as `_FFN_WEIGHTS_FUSED`
(`W_gate_up: [2*I, D_model]`, `W_down: [D_model, I]`, both `lora_attach=True`)
but is **not** wired into any current `Variant`; consumers SHOULD assume the
gated form when constructing from a `Variant.api.weights`. The
`SwiGLUSpec.fused_gate_up` flag selects the fused tensor layout at runtime;
the API schema does not bifurcate on it in this commit.

Attributes: derived from each variant's `_ATTR_META`.

### 7.6 ComponentMeta

| Property | Value |
|---|---|
| `slug` | `"ffn"` |
| `name` | `"FFN"` |
| `status` | `"live"` |
| `badges` | `("core", "decoder-only")` |
| `variants` | `(swiglu, geglu, clamped-swiglu)` |

---

## 8. Graph primitives (`misc.py`)

Two primitives, defined in `components.misc`. Both extend `Spec` directly (not
`_ComponentBase`); they carry no `lora` and no `_ATTR_META`. They are not
listed in `ALL_COMPONENTS` and do not have a `ComponentMeta` — they exist
solely to be wired into a `BlockGraph` `Node`.

### 8.1 `AddSpec`

```python
@dataclass(frozen=True)
class AddSpec(Spec): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["Add"]` | `"Add"` | Discriminator. |

Takes N inputs (arity comes from the surrounding `Node.inputs` length);
output is their elementwise sum.

### 8.2 `MulSpec`

```python
@dataclass(frozen=True)
class MulSpec(Spec): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `kind` | `Literal["Mul"]` | `"Mul"` | Discriminator. |

Takes 2 inputs; output is their elementwise (Hadamard) product. No weights.

---

## 9. `BlockGraph` and `Node` (`block_graph.py`)

A block topology is an explicit DAG, not an enum tag. Different wiring
patterns (pre-norm, post-norm-reordered, parallel residual, sandwich) are
different graphs sharing the same node vocabulary.

The reserved input name `"input"` denotes the block's external input.

### 9.1 `Node`

```python
@dataclass(frozen=True)
class Node(Spec): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `name` | `str` | — (required) | Local label. MUST NOT equal `"input"` (reserved). |
| `spec` | `Spec` | — (required) | Any `Spec` subclass: component-instance specs, `AddSpec`, `MulSpec`. |
| `inputs` | `tuple[str, ...]` | — (required) | Names of source nodes. Each entry MUST be `"input"` or the name of a prior node. |

`Node.__post_init__` enforces:

- `inputs` MUST be a tuple (not a list) — raises `TypeError` otherwise.
- `name` MUST NOT equal `"input"` — raises `ValueError` otherwise.

`Node` extends `Spec` but does NOT extend `_ComponentBase`; it has no `lora`
field and no `kind` discriminator (it is not part of a union).

### 9.2 `BlockGraph`

```python
@dataclass(frozen=True)
class BlockGraph(Spec): ...
```

| Field | Type | Default | Constraint / Notes |
|---|---|---|---|
| `nodes` | `tuple[Node, ...]` | — (required) | Ordered DAG nodes. |
| `output` | `str` | — (required) | Name of the node producing the block output. |

`BlockGraph.__post_init__` enforces all of the following; any violation
raises `ValueError` (or `TypeError` for the tuple check):

1. `nodes` MUST be a tuple, not a list (`TypeError`).
2. All node names MUST be unique.
3. Every entry in every `n.inputs` MUST be either `"input"` or the name of a
   node that appears earlier in `nodes`. This implicitly enforces
   acyclicity: a node cannot reference its own name or any later node's
   name, so there can be no cycle.
4. `output` MUST equal the name of some node in `nodes`.

The ordering rule (a referenced name must appear earlier) is strict —
forward references are rejected even if they would not form a cycle.

### 9.3 Factory helpers

All factories are keyword-only. All produce a `BlockGraph` whose `output`
field names the final residual-add node.

For all four factories, the rope node is inserted **only if** both
`positional_encoding is not None` and the attention spec has
`rope_applied_inside == False` (or the attribute is missing). When inserted,
its name is `"rope"`, its single input feeds from the immediately-preceding
node in the topology (block input or pre-norm output, depending on factory),
and the subsequent attention node takes `"rope"` as its sole input.

##### 9.3.1 `pre_norm_block`

```python
def pre_norm_block(*, block_norm, attention,
    positional_encoding, post_attention_norm, ffn) -> BlockGraph: ...
```

Topology `x -> norm -> [rope] -> attn -> +x -> norm -> ffn -> +`. Nodes (in
order): `n0=block_norm("input")`, optional `rope=pe("n0")`,
`attn=attention("n0" or "rope")`, `r0=Add("input","attn")`,
`n1=post_attention_norm("r0")`, `ffn=ffn("n1")`, `r1=Add("r0","ffn")`.
`output="r1"`. **M0-validated path.**

#### 9.3.2 `post_norm_reordered_block`

```python
def post_norm_reordered_block(*, block_norm, attention,
    positional_encoding, post_attention_norm, ffn) -> BlockGraph: ...
```

Topology `x -> [rope] -> attn -> norm -> +x -> ffn -> norm -> +`. Nodes:
optional `rope=pe("input")`, `attn=attention("input" or "rope")`,
`n0=block_norm("attn")`, `r0=Add("input","n0")`, `ffn=ffn("r0")`,
`n1=post_attention_norm("ffn")`, `r1=Add("r0","n1")`. `output="r1"`.
Defined; not end-to-end-tested in M0.

#### 9.3.3 `parallel_block`

```python
def parallel_block(*, block_norm, attention,
    positional_encoding, ffn) -> BlockGraph: ...
```

Topology `x -> norm -> {attn, ffn} -> 3-way add(x, attn, ffn)`. Nodes:
`n0=block_norm("input")`, optional `rope=pe("n0")`,
`attn=attention("n0" or "rope")`, `ffn=ffn("n0")`,
`r0=Add("input","attn","ffn")`. `output="r0"`. Note 3-way add. Defined; not
end-to-end-tested in M0.

#### 9.3.4 `sandwich_block`

```python
def sandwich_block(*, pre_attention_norm, attention,
    positional_encoding, post_attention_norm,
    pre_ffn_norm, ffn, post_ffn_norm) -> BlockGraph: ...
```

Topology: four norms surrounding the two sublayers. Nodes:
`n_pre_a=pre_attention_norm("input")`, optional `rope=pe("n_pre_a")`,
`attn=attention("n_pre_a" or "rope")`, `n_post_a=post_attention_norm("attn")`,
`r0=Add("input","n_post_a")`, `n_pre_f=pre_ffn_norm("r0")`,
`ffn=ffn("n_pre_f")`, `n_post_f=post_ffn_norm("ffn")`,
`r1=Add("r0","n_post_f")`. `output="r1"`. Defined; not end-to-end-tested
in M0.

---

## 10. Package-level public API (`__init__.py`)

The complete set of names re-exported by `components`:

- **Base types** (`components._types`): `DType`, `Spec`, `LoRAAdapterSpec`,
  `ApiField`, `ApiSchema`, `AttrMeta`, `ComponentMeta`, `Variant`,
  `derive_api_attributes`. (`_ComponentBase` is imported but not in
  `__all__`; treat as internal.)
- **Norm** (`components.norm`): `NormStandardSpec`, `NormZeroCenteredSpec`,
  `NormSpec`.
- **Attention** (`components.attention`): `MHASpec`, `GQASpec`, `MLASpec`,
  `GQASinksSpec`, `AttentionSpec`.
- **Positional encoding** (`components.positional_encoding`): `RoPESpec`,
  `RoPESmoothScalingSpec`, `RoPEYaRNSpec`, `RoPELongRoPESpec`,
  `RoPELinearSpec`, `MRoPEInterleavedSpec`, `NoPESpec`,
  `PositionalEncodingSpec`.
- **FFN** (`components.ffn`): `SwiGLUSpec`, `GeGLUSpec`, `ClampedSwiGLUSpec`,
  `FFNSpec`.
- **Graph primitives** (`components.misc`): `AddSpec`, `MulSpec`.
- **Block graph** (`components.block_graph`): `Node`, `BlockGraph`,
  `pre_norm_block`, `post_norm_reordered_block`, `parallel_block`,
  `sandwich_block`.

**Component registry** (defined in `components.__init__`):

```python
ALL_COMPONENTS: tuple[ComponentMeta, ...] = (
    norm_component,
    attention_component,
    positional_encoding_component,
    ffn_component,
)
```

Exactly four entries, in declaration order: norm, attention,
positional-encoding, ffn. Graph primitives are intentionally NOT in
`ALL_COMPONENTS`.

```python
def get_component(slug: str) -> Optional[ComponentMeta]:
    """Returns the ComponentMeta whose slug matches, or None."""
```

Linear scan over `ALL_COMPONENTS`; returns the first match. Slugs are
unique, so the scan order does not matter for correctness.

Per-module `COMPONENT` constants (`norm_component`,
`attention_component`, etc.) are NOT in `__all__` — they back
`ALL_COMPONENTS` and SHOULD NOT be referenced directly by consumers.

---

## 11. Error model

| Trigger | Exception |
|---|---|
| `derive_api_attributes(cls)` with non-skipped field absent from `cls._ATTR_META` | `ValueError` at import |
| Passing `list` where `tuple[...]` is declared on `_ComponentBase.lora`, `RoPELongRoPESpec.short_factor`/`long_factor`, `MRoPEInterleavedSpec.sections`, `Node.inputs`, or `BlockGraph.nodes` | `TypeError` in `__post_init__` |
| Constructing a `kw_only=True` spec without a required keyword | `TypeError` from `__init__` |
| Mutating a frozen field | `FrozenInstanceError` |
| `Node(name="input", ...)` | `ValueError` |
| `BlockGraph(...)` with duplicate node names | `ValueError` |
| `BlockGraph(...)` with `inputs` referencing an unknown or forward-declared name | `ValueError` |
| `BlockGraph(...)` with `output` not naming a defined node | `ValueError` |

The package does NOT validate domain semantics. Legal-but-nonsensical
constructions (e.g. `GQASpec(num_heads=32, head_dim=128, num_kv_heads=33)`,
`MLASpec` with `head_dim != nope_head_dim + rope_head_dim`,
`MRoPEInterleavedSpec.sections` not summing to `rotated_dim/2`) succeed at
the API layer; runtime ports MUST reject them.

---

## 12. Stability and extension protocol

### 12.1 Adding a new variant to an existing component union

(a) New `@dataclass(frozen=True[, kw_only=True])` extending the appropriate
base (`_AttentionBase`, `_ComponentBase`, ...) and declaring `kind:
Literal["<NewTag>"] = "<NewTag>"`.
(b) Declare `_ATTR_META` covering every dataclass field except `kind`/`lora`.
Omission raises `ValueError` at import.
(c) Construct a representative `Variant` with complete `ApiSchema`
(`inputs`, `outputs`, `weights`, `attributes=derive_api_attributes(NewSpec)`).
(d) Extend the `TypeAlias = A | B | ...` union.
(e) Append the `Variant` to the component's `ComponentMeta.variants`.
(f) Add the spec class to the module's `__all__` and to
`components.__init__`'s re-export and `__all__`.

### 12.2 Adding a new component

Steps (a)-(f) above, plus: (g) add the new module's `COMPONENT` to
`ALL_COMPONENTS` in `components/__init__.py`. The slug MUST be unique
across `ALL_COMPONENTS`.

### 12.3 Renaming a `kind` literal

ABI-affecting. `isinstance`-based dispatch is insulated; `spec.kind == "..."`
matching breaks. Treat as a breaking change.

### 12.4 The Cardinal Rule

No model-family or vendor name MAY appear in any public symbol, slug, or
`kind` literal. In-source citations such as `# Source: modeling_qwen3.py:NNN`
are permitted in comments and docstrings.

---

## 13. Implementation notes

Non-normative. Explains why the API has the shape it does.

### 13.1 `kw_only=True` on variant dataclasses

Stdlib `dataclasses` rejects a subclass field-without-default after an
inherited field-with-default. Because `_ComponentBase.lora` has a default,
every component-instance subclass adding a required field would otherwise
fail class construction. Declaring the subclass `kw_only=True` flips all
its fields to keyword-only and sidesteps the ordering constraint. The norm
specs do not need this — `_NormBase` introduces only defaulted fields.

### 13.2 Derived `scale` property on attention specs

`scale` is a `@property`, not a stored field, to avoid float-equality drift
across spec instances. `MLASpec.scale` overrides the base; consumers that
read `spec.scale` polymorphically get the correct value, while consumers
that re-derive `1/sqrt(spec.head_dim)` get the correct value for MLA only
when `head_dim == nope_head_dim + rope_head_dim` (the canonical
convention).

### 13.3 `head_dim` is first-class

The package treats `head_dim` as primary rather than deriving it from
`d_model / num_heads`. Real models break the identity (MLA, some Qwen
variants, Phi small models), so encoding it would be wrong.

### 13.4 Shared-name input/weight convention

`GQASinksSpec.sinks` appears in both `inputs` and `weights`: the weight IS
the runtime input, with no projection between them. The convention is
intentional and extensible.

### 13.5 Why `_ATTR_META` exists

An earlier design split attribute metadata across the dataclass field list
and a parallel hand-maintained list; the two drifted. `_ATTR_META` plus
`derive_api_attributes` collapses it to one source of truth, with a loud
import-time check.
