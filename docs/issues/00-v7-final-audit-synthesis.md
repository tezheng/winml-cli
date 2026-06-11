# v7 final audit — 4-reviewer synthesis

**Date:** 2026-06-10
**Sources:**
- `docs/issues/audit-A-research-vs-IR.md` — research alignment
- `docs/issues/audit-B-API-design-consistency.md` — API design consistency
- `docs/issues/audit-C-example-correctness.md` — model examples
- `docs/issues/audit-D-cross-doc-pitfalls.md` — cross-doc pitfalls

> **Status banner (2026-06-11):** Cleanup wave 2026-06-10 → 2026-06-11
> completed. See the ✅ RESOLVED annotations on each finding below. The
> 🔲 OPEN items below are backlog (not blocking) — research/03 IHV refresh,
> 10 inventory-gap families, M3 backlog items.

## Headline verdict

**Code is clean. Docs are stale. 1 critical doc bug.**

| Dimension | Verdict | Severity |
|---|---|---|
| Layer-analysis research | PARTIAL | Doc-asymmetric drift (research lags code; no behavior bugs) |
| API-design consistency | **FAIL** | Critical: ShareScheme ghost in §6.3 recipe → NameError at runtime if copy-pasted |
| Example correctness | PARTIAL | 2 silent bugs in Gemma 4 `from_hf_dict` (mitigated by tests pre-bridging) |
| Cross-doc + pitfalls | **FAIL** | All top-level docs frozen at B10; missing 3 release waves of v5/v6/v7 |

**Importantly:** zero atol-loosening detected across 41 families. The numerical-gate discipline holds. The damage is all doc-level.

## What's clean (the foundation is sound)

- All 41 model families' code is consistent with the IR
- All 26 ops in `api/ops.py` source-grounded in HF `modeling_*.py`
- All numerical gates use atol=5e-4; none silently loosened
- The 41 families correctly use whatever subset of the IR they need
- Audit B found only **1** truly dead field among 20 v5/v6/v7 additions (and it's an intentional stub: `TokenMixerKind.SSM_MAMBA1`)
- Audit A confirmed all "Raschka-only" additions cite primary HF source — the `feedback_verify_source_not_blogs.md` rule is respected

## Critical findings (must-fix)

### 🚨 1. `ShareScheme` ghost in API-REFERENCE.md (CRITICAL)

✅ **RESOLVED (2026-06-10, commit `fe0d280`)** — `docs: forward-port top-level docs to v7-complete + scrub ShareScheme`. All 12-13 references in README.md and API-REFERENCE.md were rewritten to cite `AttentionSpec.kv_source_layer_offset` + the per-family `kv_source_layer_idx_map()` dispatcher (canonical: `models/gemma4/config.py:257-285`). §2.20 of API-REFERENCE.md was added to document the post-v5 mechanism.

`ShareScheme` enum was removed in v5 Phase 1 (commit `8532879`) — verified absent from `api/types.py`, `api/specs.py`, `api/kvcache.py`. But:
- **API-REFERENCE.md** still references it 8+ times (§2.13, §3.14, §6.3, §8.12, §14, §18.2, §18.12)
- **README.md** references it 2x (lines 44, 56)
- **§6.3 Gemma 4 recipe** emits code calling `ShareScheme.SAME_BLOCK_SHARED` — anyone copy-pasting this gets `NameError` at runtime

**Action:** scrub the 12-13 ShareScheme references; the actual Gemma 4 mechanism is `AttentionSpec.kv_source_layer_offset` (v5) + the `kv_source_layer_idx_map()` in `models/gemma4/config.py`.

### 🟠 2. Top-level docs frozen at B10 / M2-complete

✅ **RESOLVED (2026-06-10 commit `fe0d280` + 2026-06-11 cleanup wave)** — README.md, PROJECT-SUMMARY.md, and API-REFERENCE.md were all forward-ported to v7-complete. The 2026-06-11 cleanup wave additionally removed the stale M1/B0.5/B8/B9/B10 status blocks from README.md (replaced with a one-paragraph "Current status" + "Release history" log), fixed the §5 LOC table in PROJECT-SUMMARY.md (3,626 → ~6,100 across 13 files), updated the §7 test inventory (604 → 751 collected; 30 → 41 families), and bumped API-REFERENCE.md to v5 (post-v7-cleanup, current as of v7-complete 2026-06-10) with §18 expanded from ~30 to 41 family entries.

| Doc | Last documented release | Releases missing |
|---|---|---|
| `README.md` | M1 + B0.5 status blocks | B1-B10, M2, v5, v6, v7 |
| `docs/PROJECT-SUMMARY.md` | M2-complete | v5, v6, v7 |
| `docs/API-REFERENCE.md` | B10 (post-rollout) | v5, v6, v7 |

**Action:** forward-port all three to v7-complete. Ground truth:
- Family count: 30 → **41** (39 with layer.py + 2 shape-only)
- Op count: 19 → **26** (ALiBi build_slopes/apply_alibi, relu2, gelu_exact, selective_scan_mamba1, l2norm, gated_delta_step added)
- Dataclass count: 18 → **22** (AliBiSpec, CSASpec, HCASpec, GatedDeltaNetSpec added)
- Tag count: 14 → **21** (v5-phase1, v5-phase2, v5, v6, v7-phase-a, v7-phase-b, v7)
- Quant scheme count: 5 → **6** (BitNet ternary added v6)
- `api/` LOC: 3,636 → **6,096**

### 🟠 3. 17 of 20 v5/v6/v7 IR additions undocumented in API-REFERENCE.md

✅ **RESOLVED (2026-06-10 commit `fe0d280` + 2026-06-11 cleanup wave)** — §1 op count and §3 dataclass count updated and ground-truth (`grep -c` against `api/ops.py` and `api/specs.py`). §18 family roster expanded to 41 entries (added MPT, Falcon-7B, BitNet, Hunyuan-Large, GPT-OSS, Mamba-1, Jamba, DeepSeek-V4, Qwen3-Next, GLM-MoE-DSA, MiniMax-M2 — 11 new families). Deferred-stubs list cleaned: Mamba-1 + Jamba removed (they landed in v6). Per-family layer.md remains the canonical reference for any axis missing from the API-REFERENCE roster.


Audit B grepped each addition:
- v5: AliBiSpec, BlockLayout enum, AttentionSpec.{attn_sub_norm, v_norm, kv_source_layer_offset, n_sink_tokens}, FFNSpec.ffn_sub_norm
- v6: Activation.GELU_EXACT, SSMKind enum, QDType.TERNARY, NormSpec.has_bias, GateKind.GELU_ONLY
- v7: MoESpec.{router_kind="hash", routing_in_latent, latent_dim, latent_bias, expert_kind="gpt_oss_clamped_swiglu"}, AttentionKind.CSA_HCA, CSASpec, HCASpec, TokenMixerKind.GATED_DELTANET, GatedDeltaNetSpec

Only **3 of 20** are mentioned. ALiBi/BitNet/GPT-OSS/DeepSeek-V4/Nemotron-H users would be looking at the wrong reference.

### 🟡 4. Gemma 4 `from_hf_dict` silent bugs

✅ **RESOLVED (2026-06-10, commit `bbc975c`)** — `fix(gemma4): from_hf_dict supports raw HF nested rope_parameters`. `Gemma4Config.from_hf_dict` now natively handles the nested-rope_parameters shape: it descends into `rope_parameters.full_attention.partial_rotary_factor` for the global partial-RoPE factor, and `rope_parameters.sliding_attention.rope_theta` for theta. Callers passing `hf_model.config.to_dict()` directly no longer hit `KeyError` or silently-corrupted partial-RoPE.


From Audit C reading `models/gemma4/config.py`:

a) **`partial_rotary_factor_global` defaults to 1.0**; correct E2B value is `0.25`. The HF config stores this nested under `rope_parameters.full_attention.partial_rotary_factor`. Raw HF config → silently corrupted global-layer partial-RoPE.

b) **`rope_theta` lookup hard-codes `d["rope_theta"]`** — would `KeyError` on a raw HF dict (which stores it nested under `rope_parameters.sliding_attention.rope_theta`).

Both bugs are **mitigated** because tests pre-bridge by flattening the nested config before calling `from_hf_dict`. The real-weights gate still passes. But anyone calling `from_hf_dict(hf_model.config.to_dict())` directly would hit either bug.

**Action:** make `from_hf_dict` handle the nested-rope_parameters shape natively.

### 🟡 5. DeepSeek-V4 layer.md / docstring inconsistencies

✅ **RESOLVED (2026-06-10, commit `07e5729`)** — `fix(deepseek_v4): reconcile layer count + correct loader docstring`. The §0 At-a-glance and §"Production constants" sections of `models/deepseek_v4/layer.md` were reconciled (the 64-layer figure is canonical; the 43-layer mention was a stale mini-config holdover). The weight-loader docstring was corrected from `.W` to `.weight` to match the code.


- §0 At-a-glance says **64 layers** (3 hash + 61 sigmoid); §"Production constants" says **43** (3 hash + 40 sigmoid). KV cache figure of 128 kB is only consistent with 64. One of them is wrong.
- Weight-loader docstring uses `.W` but code (correctly) uses `.weight`. Anyone using the docstring to build a test state dict would get `KeyError`.

### 🟡 6. Memory file frozen pre-M2

✅ **RESOLVED (2026-06-10)** — `~/.claude/projects/.../memory/project_llm_layers.md` rewritten as a 1-page v7-complete summary pointing at `docs/PROJECT-SUMMARY.md`. Future sessions now see the post-v7 state.

`~/.claude/projects/.../memory/project_llm_layers.md` ends with "Awaiting user review of v2 set" — frozen at 2026-06-04. Project is now at v7 (5 days of intense work invisible to future sessions).

**Action:** rewrite as a 1-page v7-complete summary pointing at PROJECT-SUMMARY.md.

## Should-fix (lower priority)

### Research lag (Audit A)

🔲 **OPEN (backlog)** — `research/03-ihv-opsets.v2.md` refresh + research/02 §3.12/§3.14 catalog updates remain pending. Not blocking — per-family layer.md is the canonical source for any architectural axis, and the IR fields are individually source-verified.

- **`research/03-ihv-opsets.v2.md` never refreshed to v3** — still claims 9-op floor; actual IR is 26 ops. Cross-runtime synthesis tables stale.
- **`research/02-layer-sources.v3.md §3.12 / §3.14` missing entries** for hash routing, latent-MoE wrapper, clamped SwiGLU expert kind. These IR fields are source-verified individually but not catalogued.
- **`§3.6` NoPE alternation** describes `apply_rope_per_layer: list[bool]` (which v5 Phase 1 removed); IR actually uses `spec.rope = None` per-layer. Documentation drift only — math is equivalent.

### Inventory gap (Audit A)

🔲 **OPEN (backlog)** — 10 census arcs without implementation remain backlog (no IR change required for any; see M3 plan).

10 census arcs without `models/<family>/` implementation:
- OpenELM (per-layer head/FFN scaling — unique axis)
- StarCoder, xLSTM, Apple AFM (cross-block KV — unique axis), EXAONE, Liquid LFM, Cohere (PARALLEL exercised only by Falcon-7B), MobileLLM, ChatGLM2/3, Yi

And **1 model orphan**: MPT has no §5 arc home (only mentioned in §3.5 as ALiBi reference).

### Spec v3 + M2 plan staleness

✅ **RESOLVED (2026-06-10)** — both files were stamped SUPERSEDED-BY-CODE at the top pointing readers to `docs/API-REFERENCE.md` (post-rollout) instead.

`docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md` and `docs/superpowers/plans/2026-06-06-llm-layers-m2-rollout.md` were the pre-rollout design intent. They should be marked **SUPERSEDED-BY-CODE** at the top so future readers don't follow them.

## Recommended fix sequence

1. **Top priority — scrub `ShareScheme` references** in README + API-REFERENCE.md (2-3 hours; eliminates the only critical-severity issue)
2. **Forward-port the 3 top-level docs** to v7-complete state (half a day; closes the doc-lag-3-releases gap)
3. **Fix Gemma 4 `from_hf_dict`** to handle nested rope_parameters natively (1 hour; eliminates latent silent bug)
4. **Fix DeepSeek-V4 layer.md** layer count + weight-loader docstring (30 min)
5. **Rewrite memory `project_llm_layers.md`** as 1-page v7 summary (30 min)
6. **Mark spec v3 and M2 plan as superseded** with 1-line stamps (15 min)
7. **(optional) Refresh `research/03-ihv-opsets.v2.md → v3`** with current 26-op surface (half a day; should-fix for survey-paper publication, not blocking)
8. **(optional) Bootstrap 1-2 of the inventory gaps**: Cohere (parallel residual is exercised only by Falcon-7B currently) and OpenELM (unique per-layer scaling axis)

## Overall

The code base is healthy. The IR is sound. 41 families verified, 26 ops implemented, 747 tests passing. The numerical-gate discipline (atol=5e-4) is maintained throughout. The "verify source not blogs" rule is respected for every v7 frontier addition.

The damage is entirely **documentation lag** — top-level docs stopped tracking at M2-complete and missed 3 subsequent release waves. After ~1 working day of doc maintenance the entire project moves from PARTIAL/FAIL → PASS.
