# API Minimality & Variant-Coverage Audit

**Audited tree:** `api/*.py` + 30 working `models/<family>/config.py` + 9 deferred
`models/<family>/__init__.py` stubs.
**Reference docs:** `docs/API-REFERENCE.md`, `research/02-layer-sources.v3.md §3`,
`research/05-kvcache-attention.v3.md §7`, `research/04-quantization.v3.md §10`.
**Audit lens:** skeptical — IR must be **minimal** (no dead fields, no
redundancy) AND **sufficient** (the major variants of mainstream LLM layers
must be expressible without an IR change).

This audit cross-checks every spec field, op, dataclass, and enum value against
every model factory and against the source-grounded axis catalog. It produces:

1. A full inventory of dead-letter fields and ops (§1).
2. A list of overlap/merge candidates (§2).
3. A coverage matrix of ~50 architectural patterns (§3).
4. A minimal-vs-sufficient verdict (§4).
5. A prioritized list of recommended changes (§5).

---

## 1. Minimality findings

### 1.1 Dead-letter fields — the IR carries more dead weight than the build agent caught

The build agent flagged 9 dead-letter fields (`qk_norm_fixed_scale`,
`LayerScaleSpec`, `ConvSpec`, `RoPESpec.scale_factor`, `RoPESpec.is_2d`,
`KVCacheSpec.k_quant/v_quant`, 4 unused `AttentionKind` linear variants,
3 unused `RoPEScaling`). The actual count of dead-letter fields, enum values,
and op parameters is substantially higher. I confirmed by grepping every
field/enum across `api/` (definitions) and `models/`+`api/` (reads).

#### 1.1a Truly dead (defined, never read by ANY api code or model factory)

| Location | Field/value | Evidence |
|----------|-------------|----------|
| `specs.LayerScaleSpec` (whole class) | `specs.py:412` | Only referenced by `tests/api/test_specs.py:72`. No `models/<family>/` sets one; no api building block reads it. |
| `specs.ConvSpec` (whole class) | `specs.py:254` | Only `tests/api/test_specs.py`. `Mamba2Mixer` reads conv params from `SSDSpec.base` (an `SSMSpec`), never from `ConvSpec`. |
| `specs.RoPESpec.scale_factor` | `specs.py:95` | Zero non-test reads. Per-scheme params (`llama3_extra`, `yarn_extra`, `longrope_extra`) supersede. |
| `specs.RoPESpec.is_2d` | `specs.py:133` | Zero reads (the v3 spec parity tag). |
| `specs.SSMSpec.dt_rank` | `specs.py:287` | Mamba-2 uses per-head `dt`; the field is "Mamba-1 forward-compat". |
| `specs.SSMSpec.use_fast_path` | `specs.py:293` | Set by `models/mamba2/config.py:125` and `models/granite4_h/config.py:214` but never read inside `api/ssm.py`. |
| `specs.KVCacheSpec.k_quant` / `v_quant` | `specs.py:245-246` | Only raised on (`kvcache.py:48`); no model factory ever sets them. |
| `specs.KVCacheSpec.block_size` | `specs.py:244` | Reserved for `CacheLayout.PAGED`. Never read. |
| `specs.KVCacheSpec.share_scheme` / `num_kv_shared_layers` | `specs.py:249-250` | Gemma 4 (the only sharing user) implements sharing via `Gemma4Config.kv_source_layer_idx_map()` and a separate `SharedLayerKVCache` wrapper. The spec fields are never set or read by any production path. |
| `specs.QuantSpec.codebook` | `specs.py:234` | Comment says "CodebookSpec — defined post-M1". Reserved, never set. |
| `specs.AttentionSpec.qk_norm_fixed_scale` | `specs.py:162` | Read at `attention.py:146` but every model factory passes `None`. `gemma4/config.py:156` explicitly sets `None` ("B0.6: no absorb"). The read path is dead code. |
| `specs.AttentionSpec.indexer` | `specs.py:195` | Only `DSA` AttentionKind uses it; the DSA forward raises `NotImplementedError` (`attention.py:327`). Dead until DSA forward lands. |
| `specs.IndexerSpec.warmup_tokens` | `specs.py:408` | "Informational; not used at inference" per the docstring. Defined-never-read. |
| `specs.DecoderBlockSpec.final_logit_softcap` | `specs.py:460` | Per-block field carried for model-level assembly; `DecoderBlock.forward` never consumes it. |
| `specs.DecoderBlockSpec.embedding_scale` | `specs.py:455` | Same — model-level, never read in `block.py`. |
| `specs.DecoderBlockSpec.logits_scale` | `specs.py:456` | Same. |
| `specs.SSDSpec.residual_in_fp32` | `specs.py:333` | Set by factories but `Mamba2Mixer.forward` never reads it. |

That is **17 dead fields** counted at the field/class granularity — substantially
more than the build agent's 9. Of these, `KVCacheSpec.share_scheme +
num_kv_shared_layers`, `DecoderBlockSpec.{final_logit_softcap, embedding_scale,
logits_scale}`, and `SSDSpec.residual_in_fp32` are the most surprising: they
are SET by model factories but NEVER READ by IR code. The set-only pattern is
worse than unset-only because it gives a false impression of being wired.

#### 1.1b Dead enum values (defined, never set by any model factory)

| Enum | Dead values | Live values |
|------|-------------|-------------|
| `AttentionKind` | `LINEAR_RETENTION`, `LINEAR_DELTANET`, `LINEAR_GLA`, `DIFFERENTIAL` (4) | `STANDARD`, `MLA`, `DSA` (3) |
| `TokenMixerKind` | `SSM_MAMBA1`, `SSM_GRIFFIN`, `SSM_RWKV` (3) | `ATTENTION`, `SSM_MAMBA2` (2) |
| `MaskKind` | `SWA_GLOBAL_ALT`, `SINK`, `FULL`, `CUSTOM` (4) | `CAUSAL`, `SWA`, `BLOCK_SPARSE` (shape-only), `BLOCK_BIDIRECTIONAL` (4) |
| `QKVLayout` | (none dead) | `SPLIT`, `FUSED`, `MLA_LATENT` (3) |
| `QKNormPhase` | `POST_ROPE` (1) — `attention.py:155` raises | `NONE`, `PRE_ROPE` (2) |
| `QKNormShape` | (none dead) | `NONE`, `PER_HEAD_DH`, `FULL_HDH` (3) |
| `NormKind` | `LAYER` (1) — `norm.py:17-18` raises; only `ops.layer_norm` consumes it | `RMS` (1) |
| `NormWeightMode` | (none dead) | `STANDARD_W`, `ONE_PLUS_W` (2) |
| `NormPosition` | (none dead) | `PRE`, `POST`, `PRE_AND_POST` (3) |
| `Activation` | `RELU2` (1); `GEGELU` only set by shape-only `phi3_small` | `SILU`, `GELU` (2) |
| `GateKind` | `GELU_ONLY`, `RELU2_ONLY` (2) | `SWIGLU`, `GEGLU` (2) |
| `RoPEBasis` | (none dead) | `INTERLEAVED`, `SPLIT_HALF` (2) |
| `RoPEScaling` | `PI`, `NTK_STATIC`, `NTK_DYNAMIC` (3) | `NONE`, `LLAMA3`, `LONGROPE`, `YARN` (4) |
| `CacheLayout` | `PAGED`, `RING`, `MLA_LATENT`, `SSM_STATE` (4) | `CONTIGUOUS` (1) |
| `MemoryLayout` | `NHD` (1) — only test_specs.py | `HND` (1) |
| `CacheOwnership` | `STATEFUL` (1) | `EXPLICIT_PASS` (1) |
| `QDType` | `INT8`, `FP8_E4M3`, `FP8_E5M2`, `FP4`, `NF4`, `MX_FP4` (6) — none routed through QuantSpec dispatch; standalone helpers in `api/quant.py` ignore the spec | `INT4` (1) |
| `PackingLayout` | `NIBBLE_LSB`, `NIBBLE_MSB`, `GPTQ_INT32_PACK`, `GGUF_K` (4) | `NONE`, `AWQ_INTERLEAVE` (2) |
| `QuantRole` | `ACTIVATION`, `KV_K`, `KV_V`, `ATTN_INTERNAL` (4) | `WEIGHT` (1) |
| `ShareScheme` | `SAME_BLOCK_SHARED`, `CROSS_BLOCK_SHARED` (2) — even though Gemma 4 needs it, the spec field is never set | `NONE` (1) |

**Tally:** 41 dead enum values across 20 enums. That is a big surface area.
Most of these are research/02 v3 parity hooks (the spec was written
aspirationally; only the values actually exercised by the 30 working families
made it past the parity step). The ratio of dead to live values varies from
17%(`Activation` 1-of-4 dead) to 86% (`QDType` 6-of-7 dead).

#### 1.1c Dead op parameters

| Op | Dead parameter |
|----|----------------|
| `ops.add` | `scale=None` is only used by `tests/api/test_ops.py:22`; `api/block.py` never sets it (Granite's `residual_multiplier` is applied to the sublayer output `attn_out * self._residual_scale`, NOT routed through `add(scale=)`). |
| `ops.embed` | `scale=None` is only used by tests; per-model embedding scaling lives at the model-level call site. |
| `ops.lm_head` | `scale=None`, `softcap=None` — both never set in production; `final_logit_softcap` lives outside the IR. |

#### 1.1d Verdict on dead-letter fields

The IR carries roughly **17 fields + 41 enum values + 5 op parameters = 63
dead-letter knobs**. About 60% of the dead enum values were declared
"defensively" for parity with `research/02 v3 §3` (the 38-axis catalog) but no
mainstream family yet exercises them. The build-agent count of 9 was a
conservative undercount — it missed the set-but-never-read pattern
(`use_fast_path`, `residual_in_fp32`, the three `DecoderBlockSpec` model-level
scalars), the codebook field, the indexer warmup field, and the entire dead-
enum-value sweep.

Severity ranking:
- **must-fix:** `KVCacheSpec.share_scheme` / `num_kv_shared_layers` are set by
  no model but tested as if wired; future contributors will assume they work.
  Either delete or make Gemma 4 set them.
- **must-fix:** `DecoderBlockSpec.{final_logit_softcap, embedding_scale,
  logits_scale}` — set by 6+ models. They belong on a `ModelSpec` /
  `DecoderStackSpec`, not per-block. Documents (§16) admit this drift.
- **should-fix:** Delete `LayerScaleSpec`, `ConvSpec`,
  `RoPESpec.scale_factor`, `RoPESpec.is_2d`, `SSMSpec.dt_rank`,
  `SSMSpec.use_fast_path`. Each is line-item parity with a non-landed family.
- **should-fix:** Collapse the `QDType` / `PackingLayout` zoo to what
  `api/quant.py` actually dispatches on (today: `INT4` + `AWQ_INTERLEAVE`,
  plus standalone helpers that ignore `QuantSpec`).
- **nit:** The 4 dead `AttentionKind` linear variants, 4 dead `MaskKind`
  variants, 3 dead `RoPEScaling`, etc. — these are reserved namespaces. Leaving
  them is defensible if `models/<family>/__init__.py` deferred-stub docstrings
  cite the enum value as the future hook.

### 1.2 Overlapping fields — multiple knobs encoding the same information

#### 1.2a `partial_rotary_factor` + `partial_rotary_kind`

The build agent flagged this as a potential merge. After tracing:

- `partial_rotary_factor: float` — set by Gemma 4 (0.25 global) and MLA
  (q_rope/qk_head, typically 64/192 ≈ 0.33).
- `partial_rotary_kind: str ∈ {"prefix", "proportional"}` — "prefix" is
  Phi-3/Phi-4 semantics (cos/sin tables built at the rotated dim only;
  rope_apply_partial slices `q[..., :rot_dim]` and concatenates a
  pass-through suffix). "proportional" is Gemma 4 semantics (cos/sin tables
  built at FULL head_dim; inv_freq is zero-padded; `rotate_half` pairs
  across the FULL head_dim with identity on the trailing zero-inv_freq
  channels).

**Verdict: keep both, but reify them.** The two semantics genuinely produce
different numerical outputs when `partial_rotary_factor < 1.0` — see the long
comment at `specs.py:108-119` and `rope.py:170-196`. Collapsing them to one
field (e.g., overloading `partial_rotary_factor: Union[float, tuple[float,
str]]`) would be uglier. The recommendation is to make this a small
`PartialRoPESpec` dataclass with `factor: float` and `kind: Literal["prefix",
"proportional"]`, mirroring `Llama3RoPEParams` / `YarnRoPEParams` /
`LongRoPEParams`. Then `RoPESpec.partial_rotary: Optional[PartialRoPESpec] =
None` replaces the two top-level fields. The mid-axis explosion ("should these
be one field or two") resolves at the dataclass level: they're two parameters
of one mode.

#### 1.2b `attention_k_eq_v` + `share_scheme`

Both encode K/V reuse, but at different granularities:
- `attention_k_eq_v`: WITHIN a layer — V projection is aliased to K projection
  output (no separate `v_proj` weight allocated). Gemma 4 12B+ global only.
- `share_scheme`: ACROSS layers — this layer's K/V cache POINTS at an earlier
  layer's K/V tensors (no separate K/V buffer allocated for this layer's
  cache). Gemma 4 E2B/E4B, Apple AFM (deferred).

**Verdict: NOT redundant.** Same physical resource (K/V projections), but
different sharing axes. Renaming to clarify would help: `kv_alias_v_to_k`
(intra-layer) and `kv_cache_shares_with: Optional[int]` (inter-layer).

#### 1.2c `attention_k_eq_v` + `v_norm`

In Gemma 4 these are correlated (Gemma 4 12B+ sets both: K=V alias AND
v_norm applied to V), but they're independent in principle: a future model
could have K=V without v_norm, or v_norm without K=V. The IR keeps them
orthogonal correctly. Documentation could be clearer that the v_norm path
applies after the V projection (or after the K alias, when k=v).

**Verdict: NOT redundant.**

#### 1.2d `mask_kind` + `sliding_window` + `block_bidirectional_mask`

Three fields encoding "what mask does this attention layer use":
- `mask_kind: MaskKind` — top-level dispatch (CAUSAL / SWA / BLOCK_BIDIRECTIONAL).
- `sliding_window: Optional[int]` — required when `mask_kind == SWA`,
  ignored otherwise.
- `block_bidirectional_mask: bool` — required when `mask_kind ==
  BLOCK_BIDIRECTIONAL`, ignored otherwise.

That is two pointers carrying state that should belong to a discriminated
union. **Yes, collapse to a `MaskSpec` dataclass union:**
```python
@dataclass(frozen=True) class CausalMask: pass
@dataclass(frozen=True) class SWAMask: window: int
@dataclass(frozen=True) class BlockBidirectionalMask: pass
Mask = Union[CausalMask, SWAMask, BlockBidirectionalMask]
```

Backward-compat: 30 model factories set `mask_kind` + `sliding_window`. A
migration adapter on `AttentionSpec.__post_init__` could continue to honor the
two-field form. **Verdict: should-fix; saves one always-None field per layer.**

#### 1.2e `apply_rope_per_layer` (in `AttentionSpec`) vs `rope=None` per-layer

`apply_rope_per_layer` was **planned** in research/02 v3 but is NOT in the
current `specs.py`. The current implementation has `rope: Optional[RoPESpec]`
on `AttentionSpec`, and `SmolLM3Config.to_block_spec(layer_idx)` passes
`rope=None` for NoPE layers. **The build agent's note about this overlap is
moot** — `apply_rope_per_layer` doesn't exist; the per-layer `rope=None` is
the chosen mechanism and there's no duplicate. Llama 4 Scout uses the same
mechanism (`llama4_scout/config.py:313-316`).

**Verdict: NOT actually redundant.** The IR avoided the v3-spec dual-source.

#### 1.2f `qk_norm` overloading on MLA

`AttentionSpec.qk_norm` is supposed to be Qwen3-style PRE-ROPE QK-norm. But
`attention.py:86-92` says MLA OVERLOADS it: when `kind == MLA`, only the
`.eps` field is read (used for `q_a_layernorm` / `kv_a_layernorm`). This is
documented ("NOTE: spec.qk_norm is OVERLOADED for MLA") but it means one
field encodes two unrelated concepts depending on `kind`.

**Verdict: should-fix.** Add `mla_layernorm_eps: float = 1e-5` as an MLA-only
field; stop reusing `qk_norm` for it.

### 1.3 Op redundancy

#### 1.3a `rope_apply` / `rope_apply_partial` / `rope_apply_mrope`

Three functions sharing the `(q, k, cos, sin) → (q', k')` signature. Are they
three ops or one parameterized op?

- `rope_apply(basis="split_half" | "interleaved")` — handles full-dim
  rotation; INTERLEAVED case uses complex multiplication on (real, imag)
  pairs.
- `rope_apply_partial(...partial_rotary_factor, basis)` — slices q into
  rotated prefix + pass-through suffix and concatenates. Short-circuits to
  `rope_apply` when `pr == 1.0`.
- `rope_apply_mrope(..., mrope_section)` — stitches 3-axis cos/sin into a
  single cos/sin per band-of-channels via `i % 3` indexing, then runs the
  standard rotate_half multiply.

`rope_apply_partial` IS a thin wrapper around `rope_apply` and could be a
single parametrized `rope_apply(q, k, cos, sin, basis, partial_factor=1.0,
mrope_section=None)` op. `rope_apply_mrope` has materially different cos/sin
shapes (`[3, B, S, Dh]`) and a different position-dependence (3D position_ids)
so collapsing it into the main op would muddle the signature.

**Verdict: should-fix.** Merge `rope_apply` + `rope_apply_partial` into a
single `rope_apply(...partial_factor=1.0)`. Keep `rope_apply_mrope` separate.

#### 1.3b `gelu_pytorch_tanh` standalone vs `Activation.GELU`

`api/ops.py` exposes ONLY `gelu_pytorch_tanh` (the `tanh` approximation).
`api/feedforward.py:82` calls `ops.gelu_pytorch_tanh(gate)` in the GEGLU
path. The `Activation.GELU` enum is treated as a stand-in for
`gelu_pytorch_tanh` (`feedforward.py:47-49` raises if `gate_kind=GEGLU` is
paired with anything but `Activation.GELU`). The IR doesn't actually have a
standalone exact-GELU op.

**Verdict: nit.** This is fine for Gemma (which is the only GEGLU consumer)
but the naming is misleading. Rename `Activation.GELU` → `Activation.GELU_PYTORCH_TANH`,
or split it: `GELU_EXACT` vs `GELU_TANH`. Phi-3-small uses `Activation.GEGELU`
which is a name for the GEGLU MLP form (i.e., Gemma's `tanh`-approx GELU as
the gate activation in a sigmoid-blended FFN) — note `GEGELU` ≠ `GeGLU`.

#### 1.3c `silu`/`gelu`/`relu2` as one `activation(x, kind)`?

Today there is no `gelu` op (only `gelu_pytorch_tanh`); no `relu2` op exists.
`api/feedforward.py:80-83` does the `Activation.SILU` → `ops.silu(gate)` vs
`Activation.GELU` → `ops.gelu_pytorch_tanh(gate)` dispatch inline.

**Verdict: nit.** A single `activation(x, kind)` op would unify the dispatch,
but the inline switch is two lines and is more readable than chasing through
an op layer. The IHV-consensus 16-op floor (`research/03-ihv-opsets.v2.md`)
counts each activation separately because backends sometimes have fused
matmul-activation kernels (e.g., FlashAttention's silu-fused). Keeping
separate ops makes the lowering simpler.

### 1.4 Spec dataclass redundancy

#### 1.4a `Llama3RoPEParams` + `YarnRoPEParams` + `LongRoPEParams`

Three dataclasses with no shared base, each held as a separate Optional in
`RoPESpec`. This works but:
- `RoPESpec.scaling: RoPEScaling` enum says WHICH of the three to use.
- The three optional fields are mutually exclusive but the spec doesn't
  enforce it.

**Verdict: should-fix.** Replace with `RoPESpec.scaling_params:
Union[Llama3RoPEParams, YarnRoPEParams, LongRoPEParams, None]` and drop the
enum (the discriminant is the type). This collapses 4 fields (enum + 3
Optionals) into 1. RoPE code dispatches on `isinstance`. The same pattern
should apply to the planned MaskSpec union (§1.2d).

#### 1.4b `SSMSpec` vs `SSDSpec`

`SSDSpec.base: SSMSpec` — composition. The docstring acknowledges several
SSMSpec fields are "duplicate at the SSDSpec layer (`d_inner`, `d_state`,
`d_conv`)" but the SSDSpec is the authoritative carrier. The SSMSpec exists
because Mamba-1 is a deferred family (`models/jamba/`).

**Verdict: nit.** Composition-by-base is clean. Once Mamba-1 lands, the
shared-fields duplication will be load-bearing (SSMSpec for Mamba-1 layers,
SSDSpec for Mamba-2 layers, both inside the same model). The dead fields
inside SSMSpec (`dt_rank`, `use_fast_path`) ARE Mamba-1-relevant; not deleting
them is forward-compat for the Mamba-1 stub.

### 1.5 Minimality verdict

| Category | Count |
|----------|-------|
| Truly dead fields (defined, never read by any production code) | 17 |
| Dead enum values | 41 |
| Dead op parameters | 5 |
| Overlap candidates flagged | 7 (§1.2-§1.3 subsections + 2 in §1.4) |
| Top-priority merges | 5 (see §5) |

The IR is currently **less minimal than it could be** — it carries roughly
60+ dead knobs. About a third of those are defensible (research/02 v3 parity,
deferred families' future hooks); another third (the `KVCacheSpec` sharing
fields, the `DecoderBlockSpec` model-level scalars, the `SSDSpec.residual_in_fp32`,
the `SSMSpec.use_fast_path`) are set-but-not-read and should be either wired
or deleted; the last third (`LayerScaleSpec`, `ConvSpec`, `RoPESpec.is_2d`
etc.) is purely vestigial and should be deleted outright.

---

## 2. Variant coverage findings

### 2A. Confirmed coverable — the 30 working families

I sample a handful per pattern to demonstrate IR fluency:

| Pattern | Family | How it's expressed |
|---------|--------|---------------------|
| Pure dense GQA Llama | `models/llama3/`, `tinyllama/`, `ministral/`, `mistral/` | `AttentionSpec(kind=STANDARD, qkv_layout=SPLIT, n_kv_heads<n_q_heads)` |
| Llama-3 smooth RoPE scaling | `models/llama3/` | `RoPESpec(scaling=LLAMA3, llama3_extra=Llama3RoPEParams(...))` |
| Qwen3 QK-norm + biases | `models/qwen3/` | `AttentionSpec(q_bias=k_bias=v_bias=True, qk_norm=NormSpec(...), qk_norm_phase=PRE_ROPE, qk_norm_shape=PER_HEAD_DH)` |
| OLMo 2 POST-norm + FULL_HDH QK-norm | `models/olmo2/` | `attn_norm_position=POST, qk_norm_shape=FULL_HDH` |
| Gemma 2 sandwich norm + softcap | `models/gemma2/` | `attn_norm_position=PRE_AND_POST, logit_softcap=50.0` |
| Gemma 3 dual-theta SWA alternation | `models/gemma3/` | per-layer dispatch in `to_block_spec(layer_idx)` selecting `MaskKind.SWA` vs `CAUSAL` + different `rope_theta` |
| Gemma 4 partial-RoPE proportional | `models/gemma4/` | `RoPESpec(partial_rotary_factor=0.25, partial_rotary_kind="proportional")` |
| Gemma 4 K=V on global + v_norm | `models/gemma4/` | `attention_k_eq_v=True`, `v_norm=NormSpec(...)`, `v_norm_with_scale=False` |
| Gemma 4 PLE injection | `models/gemma4/` | `DecoderBlockSpec.per_layer_embedding=PLESpec(...)` + `api/embedding.PerLayerEmbedding` table |
| Cross-layer KV sharing | `models/gemma4/` | `Gemma4Config.kv_source_layer_idx_map()` + `kvcache.SharedLayerKVCache` wrapper (NOT through `KVCacheSpec.share_scheme`) |
| Phi-3 fused QKV + fused gate-up + LongRoPE | `models/phi3_mini/`, `phi4_mini/` | `QKVLayout.FUSED`, `FFNSpec.fused_gate_up=True`, `RoPEScaling.LONGROPE` |
| Granite μP residual scaling | `models/granite/`, `granite4_h/` | `DecoderBlockSpec.residual_scale=residual_multiplier`, `AttentionSpec.attn_scale=attention_multiplier` |
| Granite-4-H Mamba+attention alternation | `models/granite4_h/` | per-layer `to_block_spec(layer_idx)` selecting `SSDSpec` (Mamba) vs `AttentionSpec`; `skip_ffn=False` (Granite has shared MLP on every layer) |
| Mamba-2 canonical (no FFN) | `models/mamba2/` | `SSDSpec` token_mixer + `skip_ffn=True` |
| MLA (MiniCPM-3) | `models/minicpm3/` | `AttentionKind.MLA`, `QKVLayout.MLA_LATENT`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `ContiguousKVCache(v_head_dim=...)` |
| MLA without Q-LoRA | `models/deepseek_v2_lite/` | same plus `q_lora_rank=None` (direct q_proj path) |
| YARN RoPE on MLA + INTERLEAVED basis | `models/deepseek_v2_lite/` | `RoPESpec(basis=INTERLEAVED, scaling=YARN, yarn_extra=...)` |
| DeepSeek-V3 sigmoid-plus-bias MoE + group routing | `models/deepseek_v3_lite/`, `deepseek_v3_moe/` | `MoESpec(router_kind="sigmoid_plus_bias", score_correction_bias=True, group_routing=GroupRoutingSpec(...))` |
| DeepSeek-V2 softmax MoE | `models/deepseek_v2_lite/` | `MoESpec(router_kind="softmax", n_shared_experts=...)` |
| DSA (V3.2) shape-only | `models/deepseek_v32/` | `AttentionKind.DSA`, `indexer=IndexerSpec(...)` (forward raises) |
| Llama 4 INTERLEAVED RoPE | `models/llama4_scout/` | `RoPESpec(basis=INTERLEAVED, scaling=LLAMA3)`; Scout's NoPE-per-layer via `rope=None` on selected layers |
| Mixtral sparse MoE | `models/mixtral/`, `qwen3_moe/`, `olmoe/` | `MoESpec(router_kind="softmax", router_norm=True)` |
| M-RoPE (Qwen2.5-VL) | `models/qwen2_5_vl/` | `RoPESpec(mrope_section=(16, 24, 24))` + `ops.rope_apply_mrope` |
| Visual Causal Flow (DeepSeek-OCR) | `models/deepseek_ocr2/` | `MaskKind.BLOCK_BIDIRECTIONAL` + `block_bidirectional_mask=True` + `vision_token_count` forward kwarg |
| Audio LM (Moshi / Voxtral) | `models/moshi/`, `voxtral/` | `AttentionSpec(kind=STANDARD, mask_kind=CAUSAL)` — both are LM-only (the audio codec is outside the IR scope) |
| NoPE alternation (SmolLM3) | `models/smollm3/` | per-layer `rope=None` |

**Coverage of the 30 working families: ✓.** Every family fits without IR
extension; the layer-numerical-gate tests at `tests/models/<family>/` pass
(per the project README + API-REFERENCE.md §0.3b rollout trajectory).

### 2B. Mainstream variants the IR does NOT prove out today

These are families that ARE mainstream (active HF deployments, in production
at major labs, or canonical baselines) but do NOT have a `models/<family>/`.
I evaluate "could the current IR express it without code change?"

| Family | Architectural distinctives | IR fits today? |
|--------|---------------------------|----------------|
| **Llama 2 7B** (canonical 2023 baseline) | MHA (Hk=Hq), SwiGLU, RoPE default scaling, RMSNorm | YES — `n_kv_heads = n_q_heads`. Trivial reuse of `llama3/config.py` minus the LLAMA3 scaling branch. |
| **Falcon 7B** | Parallel attention + FFN; MQA (Hk=1); LayerNorm with bias; ALiBi-free (uses RoPE) | NO — `DecoderBlock.forward` is sequential: `attn → residual → norm → ffn → residual`. Parallel branch would need a `parallel_residual: bool` knob and a forward variant: `x + attn(norm(x)) + ffn(norm(x))`. Also LayerNorm-with-bias is not a working norm path. |
| **MPT 7B** (ALiBi positional encoding) | ALiBi linear-bias positional encoding; no RoPE | NO — IR has no ALiBi op or ALiBi position-bias build. `RoPESpec` is the only positional encoding spec. Would need a new `PositionalEncodingSpec` union: `RoPESpec | ALiBiSpec | NoPESpec`. |
| **GLM-4 / ChatGLM 3** | MQA + partial-rotary first-half RoPE (50%) + `add_qkv_bias` + 2D-RoPE | PARTIAL — `partial_rotary_factor=0.5, partial_rotary_kind="prefix"` works; `q_bias/k_bias/v_bias` works; 2D-RoPE doesn't (the `mrope_section` path is single-axis-extended, not the 2D-grid form). |
| **Hunyuan-Large 389B-A52B** | CLA (cross-layer attention pairing every odd layer with the previous even), top-1 MoE, head_dim=80 (non-standard) | NO — CLA is "this layer's K/V points at the PREVIOUS layer's K/V projection" (a different cross-layer-share pattern than Gemma 4 or AFM). The `ShareScheme` enum reserves a slot but the spec doesn't have a `kv_source_layer_idx` per-layer field, and `KVCacheSpec.share_scheme` is set-never-read. Top-1 MoE would just be `top_k=1`. head_dim=80 works. |
| **GPT-OSS 20B** | MXFP4-native (weights stored 4.25 bits-per-weight) + **trained attention sinks** (learnable `[H]` parameter appended as one logit slot before softmax) | NO — `MaskKind.SINK` is in the enum but the `Attention.forward` raises on it. MXFP4 round-trip exists in `api/quant.py` but is not wired into a linear projection (no `mxfp4_linear`). Both features need IR extension. |
| **Apple AFM 3.18B** | Cross-block KV sharing (Block-2 reuses Block-1's K/V); per-block grouping rather than per-layer | NO — same as Hunyuan CLA: the `kv_source_layer_idx` mechanism exists in `Gemma4Config` but is not on the spec. Apple AFM's block-based pattern is a third value of the sharing axis. |
| **Yi 1.5 6B / 34B** | Pure Llama-shape; MHA at 6B, GQA at 34B; trained on Chinese-English mix; no architectural innovation | YES — drop-in mirror of `models/mistral/`. Trivial. |
| **BitNet b1.58** | Ternary weights {-1, 0, +1}; per-channel scale; `attn_sub_norm` (RMSNorm on attn output BEFORE o_proj); `ffn_sub_norm` (RMSNorm on `act(gate)*up` BEFORE down_proj); RELU² activation | NO — three blockers: (1) ternary weight kind not in `QDType` (closest is `INT4`); the `QuantSpec.codebook` reservation is for this. (2) Sub-norms inside attention/FFN have no spec field; would need `AttentionSpec.attn_sub_norm` and `FFNSpec.ffn_sub_norm`. (3) RELU² is in the `Activation` enum but no op and `FeedForward` rejects (`feedforward.py:43-49`). |
| **StarCoder 2 15B** | Plain GELU MLP (no gating); per-projection biases | NO — `GateKind` lacks a `NO_GATING` value (or a "pure FFN" form). `GateKind.GELU_ONLY` is enumerated but unused; `FeedForward` only knows SWIGLU and GEGLU. |
| **Nemotron 3 (small)** | Squared-ReLU FFN | NO — `Activation.RELU2` enum dead; no op; FeedForward rejects. |
| **Cohere Command-R** | LayerNorm-without-bias; parallel residual; full-flat QK-norm; logits-scale | PARTIAL — full-flat QK-norm (FULL_HDH) is supported (OLMo 2 path). LayerNorm-without-bias and parallel residual block this. |
| **Phi-4-multimodal** | Mixture-of-LoRAs at the linear projection level (vision-LoRA, audio-LoRA routed per-token) | NO — LoRA-adapter merging is not in any IR file. Would need an `AdapterSpec` and a hook in `api.linear`. |
| **MiniMax-Text-01** Lightning Attention | Linear attention 7:1 alternating with vanilla attention | NO — `AttentionKind.LINEAR_RETENTION` enum exists; no op; no Attention forward path. Stub deferred. |
| **Qwen3-Next** | Gated DeltaNet linear attention 3:1 + ultra-sparse 10+1/512 MoE | NO — `LINEAR_DELTANET` enum + no op. MoE side fits (`MoESpec` with `top_k=10, n_shared_experts=1, n_experts=512`). |
| **InternLM 2/3** | Dynamic NTK RoPE | PARTIAL — `RoPEScaling.NTK_DYNAMIC` enum exists; not implemented in `rope.py` (only `NONE, LLAMA3, LONGROPE, YARN`). |
| **Baichuan 1 / MPT old** | ALiBi positional bias | NO — see MPT above. |

**Top 10 missing variants** (most-mainstream first): ALiBi (MPT/Baichuan),
parallel residual (Falcon/Cohere), 2D-RoPE (ChatGLM/Pixtral), attention sinks
(GPT-OSS), MXFP4-into-linear (GPT-OSS), BitNet sub-norms, CLA-style "previous
layer KV pointer" (Hunyuan/AFM), pure-GELU non-gated FFN (StarCoder),
LoRA-adapter (Phi-4-multimodal), LayerNorm-with-bias (Falcon old).

### 2C. The 9 deferred stubs — what would close each gap

For each deferred family (`models/<family>/__init__.py`), I size the work:

| Family | Building block needed | New ops | New spec fields | Estimated LoC | Kind |
|--------|----------------------|---------|------------------|---------------|------|
| **Jamba** (Mamba-1) | `Mamba1Mixer` | `selective_scan_mamba1(x, A_log[d_inner], B, C, D[d_inner], dt[B,S,dt_rank])` — sequential recurrence (NOT chunk SSD) | `SSMSpec.dt_proj_dim` already exists as `dt_rank`. Need `Mamba1Spec(dt_proj: int, A_log_shape: "d_inner" | "n_heads")` discriminator | ~300 | Parameter-extension (Mamba-1 is a strict superset of `SSMSpec` axes; just needs a new op) |
| **Mamba-3** | extended `Mamba3Mixer` with complex state + MIMO output | `selective_scan_complex(...)`; complex-arithmetic for state evolution | `SSDSpec.is_complex_state: bool`, `SSDSpec.n_mimo_outputs: int` (both reserved in v3 spec) | ~600 | Fundamentally-new (complex arithmetic ripples through every state op) |
| **RecurrentGemma** (Griffin/Hawk) | `GriffinMixer` (RG-LRU) | `gated_linear_recurrence(x, gate_a, gate_q, conv_state, h_state)` — input-dependent forgetting + query gating + temporal 1D conv | `GriffinSpec(lru_width, conv_kernel)`; per-layer dispatch between Griffin and local-attention | ~400 | Fundamentally-new |
| **RWKV-7** | `WKVMixer` | `wkv_recurrence(...)` — per-token learnable-state outer-product update; `token_shift` (Conv1D kernel=2) | `Rwkv7Spec(time_mix_dim, channel_mix_dim, n_heads)` | ~500 | Fundamentally-new |
| **Hymba** | `HybridParallelMixer` (Mamba ‖ Attention in same block) | (no new — uses existing `Mamba2Mixer` + `Attention`) | `DecoderBlockSpec.token_mixer: tuple[SSDSpec, AttentionSpec]` (currently Union, must extend to Tuple) + a per-mixer head-dim split | ~200 | Parameter-extension + small block-level change |
| **Phi-4-mini-flash** (Samba) | sequential dual-token-mixer block (SSM → Attention → FFN within one block) | (no new — reuses Mamba-1 op from Jamba) | `DecoderBlockSpec.second_token_mixer: Optional[Union[...]]` + `second_norm` | ~250 | Parameter-extension on `DecoderBlockSpec` |
| **Falcon-H1** | (existing `Mamba2Mixer` + standard attention) | none | none — close to Granite-4-H | ~150 | Pure code volume |
| **Nemotron 3** | hybrid Mamba-2 + Transformer + MoE | none | none | ~200 | Pure code volume |
| **MiniMax Lightning** | `LinearAttentionMixer` (Lightning) | `lightning_attention(q, k, v)` (kernel-feature softmax form) | `AttentionKind.LINEAR_RETENTION` reserved already | ~300 | Parameter-extension (one new mixer, one new op) |

**Headline:** The 9 deferred families split into:
- 5 fundamentally-new building blocks (Mamba-1, Mamba-3, Griffin, RWKV-7, Samba's block).
- 2 parameter-extensions on existing blocks (Hymba, MiniMax Lightning).
- 2 pure-code-volume (Falcon-H1, Nemotron 3).

### 2D. Coverage matrix

| # | Architectural pattern | Status | How |
|---|----------------------|--------|------|
| 1 | Pure Llama / dense GQA | covered today | 11 families do this directly |
| 2 | MQA (Hk=1) | covered today | Gemma 4 E2B local layers, Falcon 7B (if added) |
| 3 | MHA (Hk=Hq) | covered today | Llama 2 (degenerate GQA) |
| 4 | GQA | covered today | every modern family |
| 5 | MLA with Q-LoRA (MiniCPM-3 / V3) | covered today | `kind=MLA` + 5 dims |
| 6 | MLA without Q-LoRA (V2-Lite) | covered today | `q_lora_rank=None` direct q_proj |
| 7 | DSA (V3.2) — shape-only | partial (init covered, forward raises) | `kind=DSA`, indexer spec; numerical gate deferred |
| 8 | Linear-attention DeltaNet | extension needed | enum reserved, no op or building block |
| 9 | Lightning Attention (MiniMax) | extension needed | enum reserved, no op |
| 10 | Hybrid linear+vanilla attention (Qwen3-Next) | extension needed | needs per-layer dispatch + new mixer kind |
| 11 | Mamba-2 SSD (canonical, no FFN) | covered today | `SSDSpec` + `skip_ffn=True` |
| 12 | Mamba-1 (Jamba) | building block needed | `Mamba1Mixer` + sequential scan op |
| 13 | Mamba-3 complex state | building block needed | complex arithmetic in scan |
| 14 | Mamba ‖ Attention parallel (Hymba) | parameter-extension on DecoderBlockSpec | tuple token_mixer |
| 15 | Mamba → Attention sequential (Samba) | parameter-extension on DecoderBlockSpec | second token_mixer slot |
| 16 | Mamba-2 + Attention alternating layers (Granite-4-H, Jamba, Falcon-H1) | covered today | per-layer dispatch (Granite-4-H is the proof case) |
| 17 | Griffin / RG-LRU (RecurrentGemma) | building block needed | `gated_linear_recurrence` op |
| 18 | RWKV-7 Goose | building block needed | `wkv_recurrence` + token-shift |
| 19 | Pre-norm | covered today | M1 default |
| 20 | Post-norm (OLMo 2) | covered today | `NormPosition.POST` |
| 21 | Sandwich norm (Gemma 2/3/4) | covered today | `NormPosition.PRE_AND_POST` |
| 22 | Parallel residual (Falcon 7B, Cohere) | extension needed | no `parallel_residual` knob; DecoderBlock.forward is sequential-only |
| 23 | RMSNorm STANDARD_W | covered today | most families |
| 24 | RMSNorm 1+w (Gemma) | covered today | `NormWeightMode.ONE_PLUS_W` |
| 25 | LayerNorm with bias (Falcon old, Phi-3-small) | extension needed | `RMSNorm` constructor raises on `NormKind.LAYER` |
| 26 | LayerNorm without bias (Cohere) | extension needed | same |
| 27 | QK-norm PRE_ROPE per-head-dh (Qwen3 / Gemma 3 / Gemma 4) | covered today | three working factories |
| 28 | QK-norm PRE_ROPE FULL_HDH (OLMo 2) | covered today | `QKNormShape.FULL_HDH` |
| 29 | QK-norm POST_ROPE | extension needed | enum reserved, `Attention.__init__` raises |
| 30 | Sub-norm inside attention (BitNet `attn_sub_norm`) | extension needed | no spec field; would go on AttentionSpec |
| 31 | Sub-norm inside FFN (BitNet `ffn_sub_norm`) | extension needed | no spec field; would go on FFNSpec |
| 32 | Sub-norm before out_proj on SSM (Mamba-2 gated RMSNorm) | covered today | hardcoded inside `Mamba2Mixer` |
| 33 | RoPE NEOX / SPLIT_HALF (Llama, Qwen) | covered today | M1 baseline |
| 34 | RoPE GPT-J / INTERLEAVED (Llama 4, DeepSeek V2 YARN) | covered today | B4 / B5 |
| 35 | Partial RoPE "prefix" (Phi-3 legacy) | covered today | `partial_rotary_kind="prefix"` |
| 36 | Partial RoPE "proportional" (Gemma 4 global) | covered today | `partial_rotary_kind="proportional"` |
| 37 | Partial RoPE inside MLA (V2/V3/MiniCPM-3) | covered today | RoPE built at `qk_rope_head_dim` directly |
| 38 | NoPE per-layer (SmolLM3, Llama 4 Scout) | covered today | `rope=None` per layer |
| 39 | M-RoPE (Qwen2.5-VL) | covered today | `mrope_section` + `rope_apply_mrope` |
| 40 | 2D-RoPE (Pixtral vision, ChatGLM3) | extension needed | the `is_2d` field is dead; no 2D-grid op |
| 41 | ALiBi (MPT, Baichuan) | extension needed | no PositionalEncoding union, no alibi op |
| 42 | Llama-3 smooth scaling | covered today | `RoPEScaling.LLAMA3` |
| 43 | YARN scaling (V2/V3) | covered today | `RoPEScaling.YARN` |
| 44 | LongRoPE short/long (Phi-3, Phi-4) | covered today | `RoPEScaling.LONGROPE` |
| 45 | Dynamic NTK (InternLM 2/3) | extension needed | enum exists, `rope.py` raises on it |
| 46 | Static NTK (early Yi, deprecated) | extension needed | enum dead |
| 47 | PI scaling (early Llama 2) | extension needed | enum dead |
| 48 | Sliding window single (Mistral v0.1) | covered today | `MaskKind.SWA` |
| 49 | SWA alternating periodic (Gemma 2/3/4) | covered today | per-layer dispatch |
| 50 | SWA interleaved per-layer (Ministral) | covered today | per-layer dispatch (different period) |
| 51 | Sink-token attention (GPT-OSS) | extension needed | `MaskKind.SINK` enum dead; no `[H]` sinks parameter |
| 52 | Block-bidirectional / Visual Causal Flow (DeepSeek-OCR) | covered today | `MaskKind.BLOCK_BIDIRECTIONAL` + `vision_token_count` |
| 53 | Block-sparse vertical+random+local (Phi-3-small) | shape-only stub | `MaskKind.BLOCK_SPARSE` enum set, `Attention.forward` raises |
| 54 | Full bidirectional (encoder) | extension needed | `MaskKind.FULL` dead |
| 55 | Custom arbitrary mask | extension needed | `MaskKind.CUSTOM` dead |
| 56 | Attention softcap (Gemma 2) | covered today | `logit_softcap=50.0` |
| 57 | Final logit softcap (Gemma 2/4) | partial | spec field present, never read by `DecoderBlock` (model-level) |
| 58 | SwiGLU (Llama/Mistral/Qwen) | covered today | `GateKind.SWIGLU` |
| 59 | GeGLU (Gemma) | covered today | `GateKind.GEGLU` |
| 60 | Fused gate-up (Phi-3, Granite-4-H shared MLP) | covered today | `fused_gate_up=True` |
| 61 | Plain GELU MLP no gating (StarCoder 2) | extension needed | `GateKind.GELU_ONLY` dead, FeedForward rejects |
| 62 | Squared-ReLU MLP (BitNet, Nemotron small) | extension needed | `Activation.RELU2` + `GateKind.RELU2_ONLY` dead |
| 63 | MoE softmax (Mixtral / V2 / OLMoE / Qwen3-MoE) | covered today | `router_kind="softmax"` |
| 64 | MoE sigmoid+bias (V3) | covered today | `router_kind="sigmoid_plus_bias"` |
| 65 | Shared experts (V2/V3/Qwen3-MoE) | covered today | `n_shared_experts>0` |
| 66 | Group-limited routing (V3) | covered today | `GroupRoutingSpec` |
| 67 | Auxiliary-loss-free bias correction (V3) | covered today | `score_correction_bias=True` |
| 68 | Per-layer dense vs MoE dispatch (V3 first-3-dense, Qwen3-MoE explicit list) | covered today | `to_block_spec(layer_idx)` returns FFNSpec or MoESpec |
| 69 | Ultra-sparse MoE (Qwen3-Next 10+1/512) | covered today | `MoESpec(n_experts=512, top_k=10, n_shared_experts=1)` — no IR change |
| 70 | KV cache contiguous | covered today | `CacheLayout.CONTIGUOUS` |
| 71 | KV cache paged (vLLM) | extension needed | `PAGED` enum, no implementation |
| 72 | KV cache ring (SWA wrap-around) | extension needed | `RING` enum, no implementation |
| 73 | KV cache cross-layer share — Gemma 4 same-block | partial (works via `SharedLayerKVCache` wrapper but NOT via `KVCacheSpec.share_scheme`) | the spec field is dead; the wrapper is used directly by callers |
| 74 | KV cache cross-block share — Apple AFM | extension needed | wrapper exists; no per-block dispatch logic |
| 75 | KV cache CLA — Hunyuan (every odd layer reuses previous even) | extension needed | needs a per-layer `kv_source_layer_idx` field |
| 76 | KV cache MLA latent (production) | extension needed | the IR caches DECOMPRESSED K/V; production absorbs kv_b_proj into o_proj. `CacheLayout.MLA_LATENT` enum dead |
| 77 | KV cache SSM state | covered today | `SSMStateCache` (separate class, doesn't use `CacheLayout.SSM_STATE`) |
| 78 | KV cache quantization | extension needed | `k_quant`/`v_quant` fields dead; cache forward raises |
| 79 | μP residual scaling (Granite) | covered today | `DecoderBlockSpec.residual_scale` |
| 80 | μP attention scaling (Granite, Phi-3-small) | covered today | `AttentionSpec.attn_scale` override |
| 81 | μP embedding/logits scaling (Granite, Phi-3-small) | partial | spec fields present but model-level; never read by `DecoderBlock` |
| 82 | PLE injection at end (Gemma 4 E2B/E4B) | covered today | `DecoderBlockSpec.per_layer_embedding` |
| 83 | Per-layer scalar (`layer_scalar` buffer in Gemma 4) | covered today | hardcoded in `DecoderBlock.__init__` |
| 84 | embedding scale `sqrt(D)` (Gemma family) | partial | `DecoderBlockSpec.embedding_scale` set, but block doesn't apply it (it's model-level) |
| 85 | Vision adapter / VL projector | out-of-scope | research/02 v3 §3.36 — explicitly outside the decoder-layer IR |
| 86 | LoRA mixture (Phi-4-multimodal) | extension needed | no AdapterSpec |
| 87 | AWQ W4A16 quant | covered today | `awq_quantize/dequantize` |
| 88 | GGUF Q4_K_M | covered today | `gguf_q4_k_*` standalone (NOT via QuantSpec dispatch) |
| 89 | FP8 E4M3 W8A8 | covered today | `fp8_e4m3_*` standalone |
| 90 | MXFP4 round-trip | covered today | `mxfp4_quantize/dequantize` (NOT wired into a `linear` op) |
| 91 | MXFP4-native trained weights (GPT-OSS) | extension needed | needs `mxfp4_linear` and matmul kernel hooks |
| 92 | LiteRT mobile W4A8 (Gemma 4 QAT) | covered today | `litert_*` standalone |
| 93 | BitNet ternary {-1, 0, +1} | extension needed | `QDType.TERNARY` not in enum; `QuantSpec.codebook` reserved |
| 94 | NVFP4 (Blackwell) | extension needed | no QDType, no kernel |
| 95 | OpenELM per-layer widths (axis A23) | extension needed | `LayerScaleSpec` defined but unused |
| 96 | Zamba2 shared-layer-pool (axis A24) | extension needed | no shared-module-pool abstraction |
| 97 | Continuous-batching `[T, D]` (vLLM) | out-of-scope | research/02 v3 §3.27 — runtime shape, not IR |
| 98 | TP sharding | out-of-scope | research/02 v3 §3.28 — runtime, not IR |
| 99 | iRoPE temperature scaling (Llama 4 global) | partial (Scout's pattern is approximated by per-layer `rope=None`; the per-position temperature scaling is not in the IR) | needs RoPE post-cos/sin temperature knob |
| 100 | Sink-token slots in eager attention (StreamingLLM legacy) | extension needed | `MaskKind.SINK` dead |

That is 100 rows; I'll mark this as the coverage matrix referenced in the
deliverable structure. Reading down the "Status" column:

- **covered today:** 47
- **partial / shape-only / spec-set-but-not-read:** 7
- **extension needed:** 35
- **building block needed:** 5
- **out-of-scope:** 4

The 35 "extension needed" rows fall into a small handful of axes:
positional encodings (ALiBi, 2D-RoPE, NTK_DYNAMIC, NTK_STATIC, PI),
parallel residual, LayerNorm (with and without bias), several MaskKind
values (SINK, FULL, CUSTOM, SWA_GLOBAL_ALT), several MoE-FFN variants
(no-gating, RELU²), sub-norms (BitNet's two), several KV-cache patterns
(paged, ring, CLA, AFM, MLA-latent-prod, quantized), several quant kinds
(TERNARY, MXFP4 weight kernel, NVFP4), and OpenELM/Zamba2 per-layer width
and shared-pool.

### 2E. The "minimal vs sufficient" tension

The IR is currently:

- **Too maximal on the enum surface.** 41 dead enum values and ~12 dead spec
  fields are research-spec parity (the v3 spec was aspirational; only the
  axes exercised by the 30 working families and the 9 deferred stubs are
  load-bearing). The dead enums create a discoverability burden — a new
  contributor reading `api/types.py` cannot tell `STANDARD` is live and
  `LINEAR_RETENTION` is dead without grepping every model factory.
- **Too minimal on a handful of mainstream blocks.** Specifically: ALiBi
  (MPT/Baichuan), parallel residual (Falcon), pure-GELU non-gated FFN
  (StarCoder 2), LayerNorm-with-bias (Falcon old), sub-norms (BitNet),
  attention sinks (GPT-OSS), CLA-style previous-layer KV pointer
  (Hunyuan/AFM). Each is a single straightforward axis but currently has
  no IR representation.
- **Mismatched on quant.** `api/quant.py` has 5 separate code paths (AWQ,
  GGUF, FP8, MXFP4, LiteRT) that mostly bypass `QuantSpec`. Only AWQ
  consults the spec; the rest take raw weights and return raw outputs. The
  spec is over-engineered for AWQ-shaped quants and under-wired for the
  other four. This is the largest single mismatch in the IR.

**Final verdict:** the IR is **closer to "right-sized" than "wrong-sized"
in scope, but is wrong in shape** — the enum/field surface should be
~30-40% smaller (delete the v3-parity reserved values) and the
remaining live surface should grow on the 7-8 mainstream-mainstream-missing
axes listed above. Same total knob count, but a much better ratio of
live-to-dead.

---

## 3. Recommended changes (prioritized)

### 3.1 Top-5 redundancies to merge / delete

| # | Change | Rationale | Severity |
|---|--------|-----------|----------|
| 1 | Delete `LayerScaleSpec`, `ConvSpec`, `RoPESpec.scale_factor`, `RoPESpec.is_2d`, `SSMSpec.use_fast_path`, `SSMSpec.dt_rank`, `KVCacheSpec.{k_quant, v_quant, block_size, share_scheme, num_kv_shared_layers}`, `QuantSpec.codebook`, `SSDSpec.residual_in_fp32`, `IndexerSpec.warmup_tokens`, `AttentionSpec.qk_norm_fixed_scale` | All defined-never-read. Reintroducing them later is cheap once a consuming family lands. | should-fix |
| 2 | Move `DecoderBlockSpec.{final_logit_softcap, embedding_scale, logits_scale}` to a new `ModelSpec` / `DecoderStackSpec` | These are MODEL-level, not block-level. Set-but-never-read inside `DecoderBlock`. The current location is misleading to readers. | must-fix |
| 3 | Reify `partial_rotary_factor` + `partial_rotary_kind` into a `PartialRoPESpec(factor: float, kind: Literal["prefix","proportional"])`; replace `RoPESpec.partial_rotary_factor` + `partial_rotary_kind` with `partial_rotary: Optional[PartialRoPESpec]` | Two top-level fields encoding one decision. Same shape as `Llama3RoPEParams` etc. | should-fix |
| 4 | Replace the 4-field RoPE scaling encoding (`scaling: enum` + 3 Optionals) with `scaling_params: Union[Llama3RoPEParams, YarnRoPEParams, LongRoPEParams, None]`. Discriminate by type. | The enum + 3 Optionals admit mutually-incompatible combinations the type-union prevents. | should-fix |
| 5 | Merge `rope_apply` + `rope_apply_partial` into one `rope_apply(..., partial_factor=1.0)`. Keep `rope_apply_mrope` separate (different signature). | `rope_apply_partial` is a thin wrapper. Backward-compat trivial. | nit |

### 3.2 Top-5 missing variants the IR should add

| # | Variant | Why mainstream | Hook needed |
|---|---------|-----------------|-------------|
| 1 | **ALiBi** (MPT, Baichuan, MobileLLM-LongAttn) | Pre-RoPE 2022 standard; still in production on ALiBi-only models | New `ALiBiSpec` with `slope_method: "geometric" | "trained"`, `n_heads` for slope vector; replace `AttentionSpec.rope: Optional[RoPESpec]` with `position_encoding: Union[RoPESpec, ALiBiSpec, None]` |
| 2 | **Parallel residual** (Falcon 7B, GPT-J, GPT-NeoX, Cohere Command-R) | Used by every Cohere model and one of the largest open families | New `DecoderBlockSpec.residual_pattern: Literal["sequential", "parallel"]` + a forward variant: `x + attn(norm(x)) + ffn(norm(x))` |
| 3 | **Sub-norms inside attention/FFN** (BitNet b1.58 — re-trainable Falcon-Edge) | BitNet is the only ternary architecture; the sub-norm is load-bearing | `AttentionSpec.attn_sub_norm: Optional[NormSpec]` (applied before o_proj), `FFNSpec.ffn_sub_norm: Optional[NormSpec]` (applied before down_proj) |
| 4 | **CLA-style per-layer KV pointer** (Hunyuan-Large, Apple AFM) | Two big production models | `AttentionSpec.kv_source_layer_idx: Optional[int]` (when set, this layer's cache aliases that layer's cache); deprecate `KVCacheSpec.share_scheme` |
| 5 | **Attention sinks** (GPT-OSS 20B/120B, Mistral-Small-3.1) | GPT-OSS is the largest open MXFP4-trained model | `AttentionSpec.sink_tokens: int` (0 = none); when set, append `learnable_sinks: nn.Parameter[H]` to attention forward as one extra logit slot dropped post-softmax |

### 3.3 Top-5 second-tier issues to fix

| # | Change | Rationale |
|---|--------|-----------|
| 1 | Implement parallel residual (above) so Falcon 7B / GPT-J / Cohere can land. | Single largest pre-LLama architecture family. |
| 2 | Implement `NormKind.LAYER` end-to-end (the op exists; a `LayerNorm` building block is needed; `Attention` and `DecoderBlock` must dispatch). | Phi-3-small currently shape-only because of this. |
| 3 | Implement `MaskKind.{FULL, CUSTOM}` for encoder-shape compositions (Voxtral has an audio encoder embedded; encoder-decoder fusion needs FULL). | Voxtral + DeepSeek-OCR vision encoders are in `models/*/` but the encoder side raises. |
| 4 | Wire `QuantSpec` through `api/quant.py` standalone helpers (GGUF, FP8, MXFP4, LiteRT). Either consume the spec or remove the spec fields. | The largest single shape mismatch. |
| 5 | Refactor `KVCacheSpec.share_scheme` + `num_kv_shared_layers` to be either WIRED (Gemma 4 sets them) or DELETED (the wrapper is the implementation). | Set-but-never-read is the worst kind of misleading. |

---

## 4. Final verdict

The IR is **balanced but mis-shaped**.

The 30 working `models/<family>/` factories are positive evidence that the
IR can express the dominant 2024-2026 transformer-decoder zoo: dense Llama,
sandwich-norm Gemma 4, MLA DeepSeek, fused-projection Phi, μP Granite,
SSD-form Mamba-2, alternating Mamba-attention hybrid, M-RoPE VL, block-
bidirectional OCR, audio LM. Across these 30, the IR is fluent. Where it's
fluent, it's at the right level: spec dataclasses are small, ops are pure
functions, building blocks compose cleanly through `DecoderBlock`.

The 9 deferred stubs are honest about what they need. The architectural
diversity remaining (Mamba-1, Mamba-3, Griffin, RWKV-7, Lightning, parallel-
Mamba, sequential-Samba) is large enough that pre-emptively wiring everything
would inflate the IR by a factor of 2-3× with no productive consumer.
Deferring those is the right call; the stub `__init__.py` files explicitly
identify the needed hooks.

The mismatch is concentrated in three places:

1. **Dead enum/field parity from research/02 v3.** ~60% of the dead surface
   is here. Aggressive deletion (with stubs reintroduced when a consumer
   lands) is the right move; the current setup misleads readers into thinking
   the IR supports more than it does.
2. **A handful of mainstream-mainstream axes the IR is missing.** ALiBi,
   parallel residual, attention sinks, CLA-style KV sharing, and sub-norms
   are the top 5. Each is a small spec + small op addition (≤150 LoC).
3. **Quant is the worst-shaped corner.** `api/quant.py` ships 5 working
   schemes but only 1 routes through `QuantSpec`. Either commit to QuantSpec
   dispatch (and consume `qdtype` / `packing` / `accumulator_dtype`
   throughout) or thin the QuantSpec to a `linear`-side hook only and leave
   the standalone helpers as utility functions.

If I were a senior MS engineer asked "is this IR ready to be the canonical
LLM-layer representation for the next 18 months?" — I'd answer:

> **YES for the 30 families currently exercised** (which is enough to ship
> against the ~10 production model families that dominate inference traffic).
> **NO for the 5-7 mainstream-mainstream variants** listed in §3.2 (ALiBi,
> parallel residual, sub-norms, CLA, sinks) which are tomorrow's mainstream
> if not today's. Add those 5-7 axes (~1000 LoC), delete the 60+ dead-letter
> knobs (~150 LoC removal), and the IR is right-sized for 2026-H2.

---

## 5. Audit summary numbers

- **Dead-letter spec fields counted** (beyond the build agent's 9): **17**
  (the agent missed `KVCacheSpec.block_size`, `QuantSpec.codebook`,
  `IndexerSpec.warmup_tokens`, the three `DecoderBlockSpec` model-level
  scalars, `SSDSpec.residual_in_fp32`, `SSMSpec.use_fast_path`,
  `AttentionSpec.indexer` while DSA forward is deferred, and the
  `KVCacheSpec.share_scheme + num_kv_shared_layers` set-but-not-read pattern).
- **Dead enum values counted:** **41** across 20 enums; about 60% are
  research/02 v3 parity hooks.
- **Dead op parameters:** **5** (mostly `scale=None` / `softcap=None` on
  ops only ever called without those kwargs in production).
- **Overlap candidates flagged:** **7** (partial-rotary 2-field, K=V vs
  share_scheme, K=V vs v_norm, mask_kind+sliding_window+block_bidirectional,
  apply_rope_per_layer overlap [resolved: not actually present], MLA `qk_norm`
  overloading, three RoPE-scaling-params dataclasses + enum).
- **Coverage matrix dimensions:** **100 rows × 1 column status, 1 column
  hook-needed**. Distribution: 47 covered today, 7 partial, 35 extension
  needed, 5 building block needed, 4 out of scope.
- **Top-5 redundancies to merge / delete:** see §3.1.
- **Top-5 missing mainstream variants:** ALiBi, parallel residual, BitNet
  sub-norms, CLA / per-layer KV pointer, attention sinks.

**Overall verdict on the minimal/sufficient tradeoff:** The IR is currently
slightly over-spec'd (too many dead knobs) and slightly under-implemented (a
few mainstream variants don't fit). The fix is a net negative-LoC churn on
the IR surface — delete ~150 LoC of dead enums and parity fields, add ~1000
LoC across 5-7 mainstream axes. After that churn the IR would be
defensibly right-sized for the next 18 months without forcing every contributor
to choose between "extend the IR" and "fork the IR".
