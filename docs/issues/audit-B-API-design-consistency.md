# Audit B — API Design Consistency

**Date:** 2026-06-10
**Scope:** `api/specs.py`, `api/types.py`, `api/ops.py` (ground truth) vs
`docs/API-REFERENCE.md` (documentation claim) vs
`docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md` (stale pre-rollout intent).
**Method:** Source-first — every finding is grounded in `grep` results or line-counted reads.
**Status:** FAIL on three checks, PARTIAL on two.

---

## B1 — Op Count Drift

### Ground truth: `api/ops.py` at HEAD

Public ops are non-underscore-prefixed `def` at module scope:

| # | Op name |
|---|---------|
| 1 | `silu` |
| 2 | `add` |
| 3 | `mul` |
| 4 | `linear` |
| 5 | `rms_norm` |
| 6 | `embed` |
| 7 | `lm_head` |
| 8 | `rope_apply` |
| 9 | `rope_apply_mrope` |
| 10 | `rope_apply_partial` |
| 11 | `gelu_pytorch_tanh` |
| 12 | `gelu_exact` |
| 13 | `relu2` |
| 14 | `build_alibi_slopes` |
| 15 | `apply_alibi` |
| 16 | `sdpa` |
| 17 | `layer_norm` |
| 18 | `softmax` |
| 19 | `top_k` |
| 20 | `gather` |
| 21 | `scatter` |
| 22 | `conv1d` |
| 23 | `selective_scan_mamba1` |
| 24 | `selective_scan` |
| 25 | `l2norm` |
| 26 | `gated_delta_step` |

**Real op count at HEAD: 26**

Private helpers (excluded): `_rotate_half`, `_pad_tensor_by_size`,
`_reshape_into_chunks`, `_segment_sum`.

### Comparison

| Source | Claimed count | Actual |
|--------|---------------|--------|
| v3 spec §5.1 | **16** | 26 |
| API-REFERENCE.md §1 | **19** (corrected from 18 in audit note) | 26 |
| HEAD | — | **26** |

**Verdict: FAIL.** API-REFERENCE.md §1 claims "The current count is **19**" in the
opening paragraph of that section. HEAD has 26. The gap is 7 ops added after the last
doc update: `gelu_exact` (v6), `relu2` (v6), `build_alibi_slopes` (v5), `apply_alibi`
(v5), `selective_scan_mamba1` (v6 B1), `l2norm` (v7 P3), `gated_delta_step` (v7 P3).
Audit A's prior count of 26 is confirmed correct.

---

## B2 — Spec Dataclass Count

### Ground truth: `api/specs.py` at HEAD

Grep for `@dataclass(frozen=True)` returned **22 occurrences** across `api/specs.py`.

The 22 classes are:

| # | Class |
|---|-------|
| 1 | `NormSpec` |
| 2 | `Llama3RoPEParams` |
| 3 | `YarnRoPEParams` |
| 4 | `LongRoPEParams` |
| 5 | `AliBiSpec` |
| 6 | `RoPESpec` |
| 7 | `AttentionSpec` |
| 8 | `FFNSpec` |
| 9 | `QuantSpec` |
| 10 | `KVCacheSpec` |
| 11 | `ConvSpec` |
| 12 | `SSMSpec` |
| 13 | `SSDSpec` |
| 14 | `GatedDeltaNetSpec` |
| 15 | `GroupRoutingSpec` |
| 16 | `MoESpec` |
| 17 | `CSASpec` |
| 18 | `HCASpec` |
| 19 | `IndexerSpec` |
| 20 | `LayerScaleSpec` |
| 21 | `PLESpec` |
| 22 | `DecoderBlockSpec` |

**Real dataclass count: 22**

### Comparison

| Source | Claimed count | Actual |
|--------|---------------|--------|
| API-REFERENCE.md §3 header | **18** ("The 18 frozen dataclasses below") | 22 |
| v3 spec §5.2 | **≥18** (aspirational) | 22 |

**Verdict: FAIL.** API-REFERENCE.md §3 states "The 18 frozen dataclasses below" but the
ground truth is 22. Four dataclasses added after the doc's last count were not reflected:
`AliBiSpec` (v5), `GatedDeltaNetSpec` (v7 P3), `CSASpec` (v7 P2), `HCASpec` (v7 P2).

---

## B3 — v5/v6/v7 Additions Documented?

API-REFERENCE.md was searched for each symbol by name. Results:

### v5 additions

| Symbol | In API-REFERENCE.md? |
|--------|----------------------|
| `AliBiSpec` | NOT MENTIONED |
| `BlockLayout` enum | NOT MENTIONED |
| `AttentionSpec.attn_sub_norm` | NOT MENTIONED |
| `AttentionSpec.v_norm` | MENTIONED (B0.6 section, §3.6, §4.7, coverage matrix, §6.3, §8) |
| `AttentionSpec.kv_source_layer_offset` | NOT MENTIONED |
| `AttentionSpec.n_sink_tokens` | NOT MENTIONED |
| `FFNSpec.ffn_sub_norm` | NOT MENTIONED |

### v6 additions

| Symbol | In API-REFERENCE.md? |
|--------|----------------------|
| `Activation.GELU_EXACT` | NOT MENTIONED |
| `SSMKind` enum | NOT MENTIONED |
| `TokenMixerKind.SSM_MAMBA1` | MENTIONED (§2.2, §9 open extension points) |
| `QDType.TERNARY` | NOT MENTIONED |
| `NormSpec.has_bias` | NOT MENTIONED |
| `GateKind.GELU_ONLY` | MENTIONED (§2.11, listed as "declared; not used") |

### v7 additions

| Symbol | In API-REFERENCE.md? |
|--------|----------------------|
| `MoESpec.router_kind="hash"` | NOT MENTIONED |
| `AttentionKind.CSA_HCA` | NOT MENTIONED (only `CSA_HCA` in §9 extension list) |
| `CSASpec` | NOT MENTIONED |
| `HCASpec` | NOT MENTIONED |
| `TokenMixerKind.GATED_DELTANET` | NOT MENTIONED |
| `GatedDeltaNetSpec` | NOT MENTIONED |
| `MoESpec.routing_in_latent` | NOT MENTIONED |
| `MoESpec.latent_dim` | NOT MENTIONED |
| `MoESpec.latent_bias` | NOT MENTIONED |
| `MoESpec.expert_kind="gpt_oss_clamped_swiglu"` | NOT MENTIONED |

### Summary

- **Total v5/v6/v7 additions checked:** 20
- **Mentioned in API-REFERENCE.md:** 3 (`v_norm`, `SSM_MAMBA1`, `GELU_ONLY`)
- **NOT mentioned:** 17

**Verdict: FAIL.** 17 of 20 v5/v6/v7 additions are absent from API-REFERENCE.md. The
doc is correctly current through B8/B9 attention variants but has not been updated for
v5 ALiBi additions, v6 SSMKind/GELU_EXACT/TERNARY/has_bias additions, or any of the
v7 P1–P4 additions (hash routing, CSA/HCA, Gated DeltaNet, Nemotron latent MoE,
GPT-OSS clamped SwiGLU).

---

## B4 — Dead-Code Re-emergence

For each v5/v6/v7 addition, I searched `api/`, `models/`, and `tests/` to determine
whether the field is READ anywhere (as opposed to merely declared in `api/specs.py`).

### v5 additions

| Field | Read outside specs.py? | Verdict |
|-------|------------------------|---------|
| `AliBiSpec` | YES — `api/attention.py` (16 hits), `api/ops.py` (9 hits) | LIVE |
| `BlockLayout` | YES — `api/block.py` (5 hits) | LIVE |
| `AttentionSpec.attn_sub_norm` | YES — `api/attention.py` (10 hits) | LIVE |
| `AttentionSpec.kv_source_layer_offset` | YES — `api/attention.py` (3 hits) | LIVE |
| `AttentionSpec.n_sink_tokens` | YES — `api/attention.py` (6 hits) | LIVE |
| `FFNSpec.ffn_sub_norm` | YES — `api/feedforward.py` (11 hits) | LIVE |

### v6 additions

| Field | Read outside specs.py? | Verdict |
|-------|------------------------|---------|
| `Activation.GELU_EXACT` | YES — `api/feedforward.py` (2 hits) | LIVE |
| `SSMKind` | YES — `api/ssm.py` (1 hit) | LIVE |
| `TokenMixerKind.SSM_MAMBA1` | Only in types.py and doc references — NOT dispatched in api/block.py | DEAD |
| `QDType.TERNARY` | YES — `api/quant.py` (3 hits) | LIVE |
| `NormSpec.has_bias` | YES — `api/norm.py` (1 hit), `api/feedforward.py` (1 hit) | LIVE |
| `GateKind.GELU_ONLY` | YES — `api/feedforward.py` (11 hits) | LIVE (raises NotImplementedError) |

### v7 additions

| Field | Read outside specs.py? | Verdict |
|-------|------------------------|---------|
| `MoESpec.router_kind="hash"` | YES — `api/feedforward.py` (16 hits) | LIVE |
| `AttentionKind.CSA_HCA` | YES — `api/attention.py` (4 hits) | LIVE (shape-only) |
| `CSASpec` | YES — `api/attention.py`, `api/specs.py`, `api/types.py` | LIVE |
| `HCASpec` | YES — `api/attention.py`, `api/specs.py` | LIVE |
| `TokenMixerKind.GATED_DELTANET` | YES — `api/block.py` (4 hits) | LIVE |
| `GatedDeltaNetSpec` | YES — `api/block.py`, `api/ssm.py`, `api/ops.py` | LIVE |
| `MoESpec.routing_in_latent` | YES — `api/feedforward.py` (19 hits) | LIVE |
| `MoESpec.latent_dim` | YES — `api/feedforward.py` (19 hits) | LIVE |
| `MoESpec.latent_bias` | YES — `api/feedforward.py` (19 hits) | LIVE |
| `MoESpec.expert_kind="gpt_oss_clamped_swiglu"` | YES — `api/feedforward.py` (12 hits) | LIVE |

### Dead fields among v5/v6/v7 additions

**Count: 1** — `TokenMixerKind.SSM_MAMBA1`

This enum value is declared in `api/types.py` and referenced in `api/specs.py` comments
and the doc, but `api/block.py`'s dispatch only handles `isinstance(spec.token_mixer,
(AttentionSpec, SSDSpec, GatedDeltaNetSpec))`. The `SSM_MAMBA1` / `SSMSpec`-as-token-mixer
path is not dispatched. It is listed as "Reserved" in both `api/types.py` and the doc,
so this is an intentional stub — but it is functionally dead.

**Verdict: PARTIAL.** One dead field among v5/v6/v7 additions, and it is flagged as
reserved/not-wired in both source comments and docs. No re-emergent dead code found
from prior removals.

---

## B5 — ShareScheme Stale References

### Ground truth

`ShareScheme` was searched in:
- `api/types.py` — **0 occurrences** (the enum does NOT exist in `api/types.py`)
- `api/specs.py` — **0 occurrences** (no field uses `ShareScheme`)
- `api/kvcache.py` — **0 occurrences**

`ShareScheme` was searched in:
- `docs/API-REFERENCE.md` — **8 occurrences**

### The 8 references in API-REFERENCE.md

1. `§0.3` change-log table: "`ShareScheme`" listed as a B0.5 addition.
2. `§2.20` — a full subsection titled "`ShareScheme` (`types.py:169`)" with `NONE`,
   `SAME_BLOCK_SHARED`, `CROSS_BLOCK_SHARED` values documented as current.
3. `§3.9 KVCacheSpec` field table: `share_scheme` listed as a field of `KVCacheSpec`.
4. `§6.3` Gemma 4 recipe code block: `share_scheme=ShareScheme.SAME_BLOCK_SHARED`.
5. `§6.13` CLA section: "The hooks are already present (`ShareScheme.CROSS_BLOCK_SHARED`
   enum value)."
6. `§8` drift log: "Commit `6dbe232` — `ShareScheme` enum for cross-layer KV sharing."
7. `§8` change table: `ShareScheme` enum listed as B0.5 enum addition.
8. `§9.6` open extension points: "`ShareScheme.CROSS_BLOCK_SHARED` — Apple AFM."

### Ground truth vs documentation

`ShareScheme` is completely absent from `api/types.py` — the enum does not exist.
`KVCacheSpec` in `api/specs.py` has no `share_scheme` field. There is no
`num_kv_shared_layers` field on `KVCacheSpec` either.

The cross-layer sharing mechanism is implemented via `AttentionSpec.kv_source_layer_offset`
(a v5-phase2 addition on `api/specs.py`) and `api/kvcache.py`'s `SharedLayerKVCache`
class — neither references `ShareScheme`.

**Stale reference count: 8**

**Verdict: FAIL (CRITICAL).** API-REFERENCE.md contains 8 references to an enum
(`ShareScheme`) and spec field (`KVCacheSpec.share_scheme`) that were REMOVED from the
codebase. The doc's §2.20 reads as authoritative ("§2.20 `ShareScheme` (`types.py:169`)"),
presents three valid-sounding enum values, and is internally consistent — making it
particularly dangerous for readers who trust the doc. A user following §6.3's Gemma 4
recipe would produce broken code (`ShareScheme.SAME_BLOCK_SHARED` is undefined).

---

## Overall — Top 5 Must-Fix

### MF-1 (CRITICAL): ShareScheme ghost in §2.20, §3.9, §6.3

**What:** `ShareScheme` enum and `KVCacheSpec.share_scheme` field documented as current
in 8 places but completely absent from `api/types.py` and `api/specs.py`.

**Impact:** Any code written from §6.3's Gemma 4 recipe will fail at runtime with
`NameError: name 'ShareScheme' is not defined`.

**Fix:** Remove all 8 `ShareScheme` references. §2.20 should be deleted. §3.9 should
remove the `share_scheme` and `num_kv_shared_layers` rows and replace with a note about
`AttentionSpec.kv_source_layer_offset`. §6.3 should use the actual `kv_source_layer_offset`
field pattern shown in the Hunyuan-Large model family.

---

### MF-2 (HIGH): Op count wrong in §1 header

**What:** §1 opens with "The current count is **19**" but HEAD has 26 public ops.
The 7 missing ops are: `gelu_exact`, `relu2`, `build_alibi_slopes`, `apply_alibi`,
`selective_scan_mamba1`, `l2norm`, `gated_delta_step`.

**Impact:** A reader using §1 as an enumeration will miss 7 ops and the families they
serve (ALiBi, squared-ReLU, Mamba-1, Gated DeltaNet).

**Fix:** Update the count to 26, add §1.19–§1.26 entries for the 7 missing ops with
signatures and "Where used" notes matching the existing style.

---

### MF-3 (HIGH): Dataclass count wrong in §3 header

**What:** §3 states "The 18 frozen dataclasses below" but HEAD has 22.
The 4 undocumented classes are: `AliBiSpec`, `GatedDeltaNetSpec`, `CSASpec`, `HCASpec`.

**Impact:** Users extending the IR for ALiBi-based models (MPT, Falcon), Gated DeltaNet
(Qwen3-Next), or DeepSeek-V4 CSA/HCA will not find these specs in the reference.

**Fix:** Update count to 22, add §3.5b `AliBiSpec`, §3.19 `GatedDeltaNetSpec`,
§3.20 `CSASpec`, §3.21 `HCASpec` with field tables in the established style.

---

### MF-4 (MEDIUM): 17 v5/v6/v7 field additions absent from §3 tables

**What:** The field tables in §3.1 (`NormSpec`), §3.6 (`AttentionSpec`), §3.7 (`FFNSpec`),
and §3.14 (`MoESpec`) do not list fields added in v5–v7 phases:
- `NormSpec.has_bias` (v6)
- `AttentionSpec.attn_sub_norm`, `.kv_source_layer_offset`, `.n_sink_tokens` (v5)
- `FFNSpec.ffn_sub_norm` (v5)
- `MoESpec.router_kind="hash"`, `.routing_in_latent`, `.latent_dim`, `.latent_bias`,
  `.expert_kind="gpt_oss_clamped_swiglu"`, `.expert_bias`, `.expert_swiglu_alpha`,
  `.expert_clamp_limit`, `.hash_vocab_size`, `.hash_score_fn` (v7)
- `SSMSpec.kind` / `SSMKind` (v6)
- `Activation.GELU_EXACT`, `QDType.TERNARY` (v6) missing from §2.10 / §2.17

**Impact:** Medium — the fields are live and documented in source comments, but the
reference doc tables present an incomplete picture for any user targeting
ALiBi/BitNet/GPT-OSS/DeepSeek-V4/Nemotron-H families.

**Fix:** Add rows to the affected field tables and enum value lists in §2 and §3.

---

### MF-5 (MEDIUM): `BlockLayout` enum and PARALLEL block layout missing from §2

**What:** `types.py` declares `BlockLayout.SEQUENTIAL` and `BlockLayout.PARALLEL`
(v5-phase2 V2 for Falcon-7B). `api/block.py` dispatches on this field. It is
referenced in `api/specs.py`'s `DecoderBlockSpec.block_layout`. It appears nowhere
in API-REFERENCE.md.

**Impact:** A reader building a Falcon-7B or Cohere-style PARALLEL block from the
reference doc has no guidance — the coverage matrix shows Falcon-7B as an open
extension but the required spec knob is not described.

**Fix:** Add `§2.21 BlockLayout` enum with `SEQUENTIAL` / `PARALLEL` descriptions,
and add `block_layout` row to the `DecoderBlockSpec` field table in §3.18.

---

## Overall Verdict: FAIL

| Check | Result | Severity |
|-------|--------|----------|
| B1 — Op count (19 claimed vs 26 actual) | FAIL | HIGH |
| B2 — Dataclass count (18 claimed vs 22 actual) | FAIL | HIGH |
| B3 — v5/v6/v7 additions documented (3/20 in doc) | FAIL | HIGH |
| B4 — Dead-code re-emergence (1 dead field, intentional stub) | PARTIAL | LOW |
| B5 — ShareScheme stale refs (8 references, enum removed) | FAIL | CRITICAL |

The documentation is structurally sound for the M1–B8 landing strip (Qwen3 through
Qwen2.5-VL) and contains genuinely useful content. The failure mode is that it stopped
being updated after B8: three entire phases of post-B8 additions (v5-phase2, v6, v7)
are simply absent from the reference, and one removed abstraction (`ShareScheme`) was
never cleaned up and now actively misleads.

**Priority order for repair:** MF-1 (ShareScheme ghost, breaks user code) → MF-2 and
MF-3 (op + dataclass counts, fixes reader trust) → MF-4 (field tables, completeness)
→ MF-5 (BlockLayout, coverage gap).
