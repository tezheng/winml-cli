# Audit D — Cross-Document Inconsistencies, Stale Claims & Pitfalls

**Auditor scope:** project-level docs (`README.md`, `docs/PROJECT-SUMMARY.md`,
`docs/API-REFERENCE.md`), spec/plan docs (`docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md`,
`docs/superpowers/plans/2026-06-06-llm-layers-m2-rollout.md`), `research/issues/*.md`
audit trail, prior issues (`docs/issues/{API-REFERENCE-audit, API-minimality-coverage-audit,
raschka-gallery-comparison}.md`), `~/.claude/projects/.../memory/project_llm_layers.md`,
and the git tag list / commit log.
**Working directory:** `C:\Users\zhengte\external\llm-layers`.
**Cut date / HEAD:** 2026-06-10, HEAD `dd0d68a` (final v7 test commit).
**Method:** ground truth re-counted on disk (`models/*/layer.py`, `models/*/__init__.py`,
`tests/models/*/test_*.py`, `git tag -l`, `git log --oneline`, `api/*.py` symbol set);
each doc claim cross-checked against ground truth or current `api/*.py`.

**Verdict (TL;DR):** **FAIL — must-fix before any external read.** All three
top-of-tree narrative docs (`README.md`, `docs/PROJECT-SUMMARY.md`,
`docs/API-REFERENCE.md`) stop documenting the project at HEAD `4010e74`
(M2-complete / B10). Three subsequent release waves (v5-phase-{1,2}, v6,
v7-phase-{a,b}) have shipped — 10 additional model families, 1 new
quant scheme (BitNet ternary), one major IR removal (the `ShareScheme` enum
+ `KVCacheSpec.num_kv_shared_layers` and miscellaneous dead-spec scalars),
plus 5 new public ops — but none of this is reflected in any production
narrative doc. The audit-corrected `API-REFERENCE.md` is no longer "post-
rollout" — it is mid-rollout at a stale snapshot. **The single highest-
impact fix is to bring `README`, `PROJECT-SUMMARY`, and `API-REFERENCE` up
to v7-complete; everything else (broken cross-references, dead enum
documentation, project-memory drift) is downstream of that.**

---

## 1. D1 verdict — Family-count consistency: **FAIL**

### 1.1 Actual ground truth (re-counted 2026-06-10)

Direct count on disk:

| Metric | Count | How counted |
|---|---|---|
| `models/<family>/` directories | **49** | `ls models/` minus `__init__.py` and `__pycache__` |
| Families with `layer.py` (implemented) | **39** | `ls models/*/layer.py` |
| Families with only `config.py` (shape-only at the layer level) | **2** | `gpt_oss`, `hunyuan_large` — both have config + tests but no `layer.py` |
| Deferred stubs (`__init__.py` docstring only) | **8** | `falcon_h1`, `mamba3`, `hymba`, `minimax_text_01`, `nemotron3`, `phi4_mini_flash`, `recurrent_gemma`, `rwkv7` |
| Families with `layer.md` (production-doc) | **35** | `ls models/*/layer.md` |
| `tests/models/<family>/` directories | **42** | `ls -d tests/models/*/` |
| Per-family test files (`test_*.py`) | **155** | `ls tests/models/*/test_*.py` |
| Total commits | **164** | `git log --oneline \| wc -l` |
| Total tags | **21** | `git tag -l` (see D2) |

The "v7 = 41" headline in the audit brief is **plausible** under the
inclusive count: `39 layer.py implementations + 2 shape-only families
(gpt_oss, hunyuan_large) = 41 working`. Under the strict
"`models/<family>/layer.py` exists" criterion the count is **39**.
Either count is defensible; the project picked the inclusive count at
v5+ where shape-only landings were explicitly part of the milestone.
But **no current doc reflects either 39 or 41**.

### 1.2 Per-doc drift

**`README.md`** (205 lines):
- Lines 30–48: "M1 / B0.5 status" block frames the project as if M1 + B0.5
  are the only completed milestones. "117 tests pass" (line 30) — but
  `PROJECT-SUMMARY.md:17` already updated to 604 collected, and the
  current `tests/api/` has 16 files plus 155 per-family tests, so the
  actual collected count is even higher (well past 700 with v5–v7 tests).
- Lines 104–195: B8 / B9 / B10 status blocks present, but
  **nothing exists for v5 / v6 / v7**. The reader is left thinking the
  project ends at B10.
- Line 167: "B10 status — Quantization 'support 6' baseline (5 schemes
  landed, 1 stubbed)" — see D4: BitNet ternary is the 6th and it
  landed in v6 (`85b0927 feat(v6): BitNet ternary QDType (1.58-bit
  weights, per-tensor scale)`). README still says "5 landed, 1 stubbed".
- Lines 9–10: "See `docs/superpowers/specs/2026-06-04-llm-layers-design.v2.md`
  (M1) and `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md`
  (M2 / B0.5)." — no pointer to API-REFERENCE.md, which is the only doc
  the reader should be looking at for the IR floor at v6+.

**`docs/PROJECT-SUMMARY.md`** (255 lines):
- Line 4: **"M1 + M2 complete (B0.5 → B10), 30 model families verified"**.
  Ground truth is 39 implemented families. Off by 9. Off by 11 under the
  inclusive count.
- Line 17: "Tags placed | 14". Actual: **21** (see D2).
- Line 18: "Working model families | 30 (incl. 3 shape-only) + 9 deferred
  stubs". Ground truth: 39 implemented + 2 shape-only + 8 deferred.
- Line 23: "Bit-exact gate vs real HF weights | 1 family (Gemma 2)" —
  this was correct at B10; need to re-audit whether v5/v6/v7 added any.
- Line 25: "Quant schemes operational | 5 (AWQ, GGUF Q4_K_M, FP8 E4M3,
  MXFP4, LiteRT W4A8)" — BitNet ternary missing (D4).
- Line 84: "Total: 3,636 lines across 13 files" for `api/`. Actual
  `wc -l api/*.py = 6,096 lines` across 13 files. Off by **2,460 lines**
  (~67% under-count — v5/v6/v7 added ALiBi, BitNet ternary, parallel
  residual, attention sinks, Mamba-1 + selective_scan_mamba1, GLM/MiniMax
  MoE routers, Gated DeltaNet, CSA+HCA composition, hash-MoE routing,
  l2norm, gated_delta_step, ternary_quantize / ternary_dequantize, etc.).
- Lines 113–146: "Working families (30) grouped by batch" — the per-batch
  table stops at B9 / B10. No v5 (Falcon-7B, MPT, BitNet b1.58, GPT-OSS
  shape, Hunyuan-Large shape), no v6 (Mamba-1, Jamba), no v7 (DeepSeek-
  V4, Qwen3-Next, GLM-MoE-DSA, MiniMax-M2).
- Line 196: timeline ends at "M2-complete → tagged at HEAD (covers B0.5 →
  B10)" — wrong. M2-complete is at commit `4010e74`; HEAD is now `dd0d68a`.
- Line 199: "from `5068bf7` (project bootstrap) to `4010e74` (HEAD)" —
  HEAD has moved.

**`docs/API-REFERENCE.md`** (4,255 lines):
- Line 1: `# llm-layers — API Reference v4 (post-rollout)`. **Stale** —
  the rollout did not end at B10; "post-rollout" is now misleading.
- Line 3: "**Status:** Post-rollout reference, current as of B10
  (2026-06-08)". Five batches behind reality.
- §3292 / §4021 / §4028: "Across all 30 families…" — 39 implemented + 2
  shape-only.
- §4019–4253 §18 "The 30-Family Roster (one-paragraph each)" — names
  18.1 Qwen3 through 18.30 Voxtral. v5–v7 families (MPT, Falcon-7B,
  BitNet b1.58, GPT-OSS, Hunyuan-Large, Mamba-1, Jamba, DeepSeek-V4,
  Qwen3-Next, GLM-MoE-DSA, MiniMax-M2) ARE NOT in the roster. They
  exist in `models/<family>/` with `layer.py`, `layer.md`, and tests, but
  are invisible to a reader of API-REFERENCE.
- Line 4255: `*End of API Reference v4 — generated 2026-06-08 post-B10.*`
  — same problem.

**`docs/issues/*.md`** (audit trail):
- `docs/issues/API-REFERENCE-audit.md` was the 2026-06-08 audit that drove
  the v4 "audit-corrected" reference. It correctly flagged "30 vs 39
  count" (§3a) but the corrections it produced never extended past B10.
  No fault of this doc — it is point-in-time correct for its own date.
- `docs/issues/API-minimality-coverage-audit.md` lines 3, 89, 331, 364,
  550, 613, 646: repeatedly cites "30 working families". Same
  point-in-time situation; this audit pre-dates v5.
- `docs/issues/raschka-gallery-comparison.md:5,68,70,350`: cites HEAD as
  `4010e74` and "30 working families". Doc was written 2026-06-09 — already
  one day after v5-phase1/2 + v6 + v7 had landed at HEAD `dd0d68a`. **The
  raschka comparison is stale ON ARRIVAL** — it explicitly cites the
  wrong HEAD. (See D6 for orphan-doc handling.)

### 1.3 Spec v3 §10 rollout

Spec v3 §10 (line 1626 onward) lays out batches B0.5 → B10 with
`Status` "pending"; the file header (line 4) says
**"Status: Draft v3, awaiting user review"**. The spec was never
"closed" / "superseded" formally after the rollout. Strictly correct as
a design-time artifact, but if a reader uses the spec as a current source
of truth they will be misled — `API-REFERENCE.md:4` calls itself the
"Replaces:" doc, but the spec doesn't reciprocate. **Recommend: spec v3
should add `Status: Superseded by docs/API-REFERENCE.md as of B10; v5–v7
extensions documented in §8 drift log of API-REFERENCE.md (when updated)`.**

### 1.4 Research issues (audit trail)

Critique files in `research/issues/` cite `v2` counts (e.g.,
`01-census-critique.md` references "80 rows / ~30 families" — see
`08-coverage-justification.md:5`). These are **historical audit-trail
artifacts**, not current-state docs, so the dated counts are fine.
**No fix needed** — but the directory should grow a 1-line README
clarifying "these are point-in-time critiques, not current status".

### D1 verdict: **FAIL.** Family count is wrong in all three production
narrative docs. Top-of-tree count should be 39 implemented + 2
shape-only + 8 deferred (49 total directories under `models/`).

---

## 2. D2 verdict — Tag-list reconciliation: **PARTIAL**

### 2.1 Actual `git tag -l`

21 tags total. Sorted into rollout order:

```
M1-complete
B0.5-complete
B1-complete
B2a-complete
B2b-complete
B3-complete
B4-complete
B5-complete
B6-complete
B7-complete
B8-complete
B9-complete
B10-complete
M2-complete
v5-phase1-complete
v5-phase2-complete
v5-complete
v6-complete
v7-phase-a-complete
v7-phase-b-complete
v7-complete
```

### 2.2 Tag drift vs the audit brief

The audit brief said: "should now show: M1, B0.5..B10, M2, v5-phase1,
v5-phase2, v5, v6, v7-phase-a, v7-phase-b, v7". Actual tags carry the
`-complete` suffix on every tag (e.g., `v5-phase1-complete`, not
`v5-phase1`). The suffix is consistent across the entire tag list, so the
naming convention is internally OK. **No tag drift on the rollout
landings.**

There is no `M3` tag, but there are `feat(M3)` commits at HEAD:
`bccab3e feat(M3): Jamba family` and `8e7cb00 feat(M3): Mamba-1
selective_scan + Mamba1Mixer`. These commits land AFTER `M2-complete`
and BEFORE `v5-phase1-complete` chronologically (per `git log`). They
appear to be early M3 work that was rolled INTO v5+/v6+ rather than
shipped as M3. **Pitfall:** the `feat(M3):` prefix is misleading — these
commits are NOT under an M3 tag and never will be. Recommend a single
"chore: re-label M3 commits" follow-up isn't worth it, but flag this in
any "list of milestones" doc.

### 2.3 PROJECT-SUMMARY tag-count mismatch

`docs/PROJECT-SUMMARY.md:17` claims "Tags placed | 14". Actual: 21.
Off by 7 (all from v5/v6/v7).

### 2.4 Commit-message accuracy

Spot-checked the v5/v6/v7 tags by reading the commits that immediately
precede each tag in `git log`. All commit titles read as accurate
descriptions of what landed (no e.g. "feat(v7 B4)" actually landing v6
content). Below is the spot-check table:

| Tag | Last feat-commit before tag | Title | Verdict |
|---|---|---|---|
| `v5-phase1-complete` | `7098631` | `chore(v5): remove dead enum values + RoPESpec.{scale_factor,is_2d}` | ✓ matches "phase-1 = dead-spec cleanup" |
| `v5-phase2-complete` | `427da59` | `test(v5): cover SWA+sinks combination and SEQUENTIAL block-layout default` | ✓ phase-2 added sinks (GPT-OSS) and parallel residual (Falcon, MPT) |
| `v5-complete` | `427da59` | (same) | ✓ |
| `v6-complete` | `85b0927` | `feat(v6): BitNet ternary QDType (1.58-bit weights, per-tensor scale)` | ✓ |
| `v7-phase-a-complete` | `d60273b` | `test(v7): consolidated spec coverage for the 4 v7 Phase-A patterns` | ✓ phase-a = 4 spec extensions (hash MoE, CSA+HCA, Gated DeltaNet, latent MoE) |
| `v7-phase-b-complete` | `dd0d68a` | `test(v7 B1-B4): seed + initialise MoE expert tensors before forward` | ✓ phase-b = the 4 family bootstraps |
| `v7-complete` | `dd0d68a` | (same) | ✓ |

### D2 verdict: **PARTIAL.** Tags themselves are clean and commit titles
match the content. But `PROJECT-SUMMARY.md:17` ("14 tags") and the
implicit M3 milestone (commits exist, no tag, never tagged) are both
small drift. No tag-level data corruption.

---

## 3. D3 verdict — "M2-complete" stale claims: **PASS but PARTIAL.**

### 3.1 Direct "M2-complete" mentions

Only 2 files outside spec/plan contain the literal "M2-complete":

1. **`docs/PROJECT-SUMMARY.md:4`** (`**Status:** M1 + M2 complete`) —
   ALREADY FLAGGED in D1. Headline-wrong.
2. **`docs/PROJECT-SUMMARY.md:196`** (timeline entry `M2-complete →
   tagged at HEAD`) — wrong: M2-complete is no longer at HEAD; v7-
   complete is.
3. **`docs/issues/raschka-gallery-comparison.md:5,68,70,350`** — written
   at HEAD `4010e74` (= M2-complete commit). The doc explicitly cites
   that HEAD, so the text "M2-complete" is correct ABOUT THE DOC ITSELF
   but the doc-level claim "current frontier of llm-layers" is wrong as of
   2026-06-10. (See D6 orphan handling.)

### 3.2 Indirect "M2 is the current state" claims

These are subtler but more numerous:

- **`README.md:9–10,40–88,167–193`** — entire body of README implies the
  project ends at B10 / M2. No mention of v5/v6/v7 (subjective: the
  reader of README leaves thinking M2 is current state, even though the
  literal string "M2-complete" doesn't appear).
- **`docs/API-REFERENCE.md:1,3,4255`** — "v4 (post-rollout)", "current
  as of B10", "End of API Reference v4 — generated 2026-06-08 post-B10".

### 3.3 Count of stale "M2 is current state" claims

Direct literal-string hits: **2** (PROJECT-SUMMARY.md lines 4, 196).
Indirect "B10 is the current state" framing claims: **3 more** (README,
API-REFERENCE.md three places). Plus the entire body of README has no
v5+ section even though it has a B8 / B9 / B10 section, which is
implicitly an "M2 is the current state" claim. Generous count: **5
to 7 stale "M2 is current state" claims** across the three top docs.

### D3 verdict: **PARTIAL.** The literal-string drift is bounded (just
PROJECT-SUMMARY); the framing drift is widespread.

---

## 4. D4 verdict — Quant scheme count: **FAIL**

### 4.1 What landed

Reading `api/quant.py` symbol set:
- `awq_quantize` / `awq_dequantize` (M1)
- `gguf_q4_k_quantize` / `gguf_q4_k_dequantize` (B10)
- `fp8_e4m3_quantize_per_tensor` / `fp8_e4m3_quantize_per_token` /
  `fp8_e4m3_matmul` (B10)
- `mxfp4_quantize` / `mxfp4_dequantize` (B10)
- `litert_quantize_weight` / `litert_quantize_activation_per_tensor` /
  `litert_w4a8_matmul` / `litert_w4_pack` / `litert_w4_unpack` (B10)
- `ternary_quantize` / `ternary_dequantize` (**v6**, line 747, line 808
  in `api/quant.py`)
- Stubs: `iq2_m_dequantize`, `aqlm_dequantize` (B10)

That's **6 schemes operational** + 2 stubs — not "5 + 1 stubbed" as
README says.

`api/types.py:185–196` confirms `QDType.TERNARY` is in the enum, with
the v6 docstring "BitNet b1.58 ternary weights — values in {-1, 0, +1}".

### 4.2 Where the count is wrong

- **`README.md:167`** — `## B10 status — Quantization "support 6"
  baseline (5 schemes landed, 1 stubbed)`. The B10-cut headline is
  stuck. After v6 added BitNet ternary, the headline should be
  "Quantization 'support 6' baseline — 6 schemes landed (5 + BitNet
  ternary)" + a new "v6 status" block under it.
- **`README.md:174–181`** — the quant table has 6 rows for AWQ, GGUF
  Q4_K_M, FP8 E4M3, MXFP4, LiteRT W4A8 + IQ2_M / AQLM stubs. BitNet
  ternary is missing.
- **`README.md:183–193`** — the "`api/quant.py` now exposes:" bullet
  list is missing BitNet ternary entirely.
- **`docs/PROJECT-SUMMARY.md:25`** — "Quant schemes operational | 5".
  Missing BitNet.
- **`docs/API-REFERENCE.md`** — `grep -n "ternary\|TERNARY\|BitNet"`
  → no hits. The full quantization §16 of the reference is silent on
  the v6 addition.

### 4.3 Drift narrative

The B10 README block actually says "support 6 baseline" in the
**headline** but lists only 5 in the body (line 167's "5 schemes
landed, 1 stubbed"). This is internally inconsistent even at B10 —
which is what motivated v6's BitNet addition.

### D4 verdict: **FAIL.** Six schemes operational; all docs say 5. Triv-
ial fix once an editor knows. Recommend: README B10 block stays as-is
(historical), add a new "v6 status" block beneath it that adds the
BitNet row and updates the table.

---

## 5. D5 verdict — Memory consistency: **FAIL**

### 5.1 Project-llm-layers memory

`C:/Users/zhengte/.claude/projects/C--Users-zhengte-external/memory/project_llm_layers.md`
(the project status file referenced from the user's MEMORY.md index)
opens with:

> **Status as of 2026-06-04 (after v2 critique-and-fix pass):**

and discusses M1-COMPLETE (2026-06-05) as the only milestone past
research-Phase-A. **No B0.5, no M2, no v5/v6/v7.**

The file ends with:

> **Awaiting user review of v2 set** before invoking writing-plans
> skill for M1 (Qwen3 kickoff gate) implementation plan

— but M1 was completed weeks ago and the project is now post-v7.

This is the file the user's MEMORY.md `[llm-layers project]
(project_llm_layers.md)` link points to as the source of current
project status. Reader (or future Claude session) starting with this
memory file will form a wildly outdated mental model.

### 5.2 What's correct in memory

The other memory entries are time-stamped lessons:
- `feedback_verify_source_not_blogs.md` ("Gemma 4 IR drift lesson
  2026-06-06") — still applies.
- `feedback_evidence_first.md`, `feedback_review_gates.md`,
  `user_role.md` — all evergreen.

So only `project_llm_layers.md` needs updating. The fix is small but
the impact is large (sets the framing for any future llm-layers session).

### D5 verdict: **FAIL.** Memory file is at the M1 milestone; project is at
v7. Recommend: rewrite `project_llm_layers.md` to a 1-page summary that
tracks `PROJECT-SUMMARY.md` (once PROJECT-SUMMARY is updated).

---

## 6. D6 verdict — Pitfall hunt: **PARTIAL — 7 pitfalls found**

### 6.1 Stale enum references in production docs

(See D8 for the full enumeration; counted here for the pitfall-hunt
tally.) `ShareScheme` was removed by `8532879 chore(v5): remove dead
KVCacheSpec fields + ShareScheme enum`. README and API-REFERENCE still
document it as if extant. **Count: 2 docs broken.**

### 6.2 Numerical-gate claims with missing tests

Not a finding from this audit (would require running the full pytest
suite). Defer to a future D9 audit. **Count: 0 from this pass.**

### 6.3 "TBD" / "TODO" / "PLACEHOLDER" in production docs

`grep -rn "TBD\|TODO\|PLACEHOLDER\|FIXME" README.md docs/` →
no hits in production docs. `grep` does find them in `docs/superpowers/
plans/*` and `research/06-ocr-vlm-extensions.md`, both of which are
historical artifacts (plans and v2-era research). **No production-doc
TBDs.**

### 6.4 Orphan files

A. **`docs/issues/raschka-gallery-comparison.md`** — written 2026-06-09,
   cited nowhere outside itself. The doc title implies it's a
   point-comparison study with a frontier external resource, but no
   project doc surfaces or links it. The doc is also stale (D3
   §3.1 #3). **Orphan — recommend linking from PROJECT-SUMMARY.md §2
   "Critique trail" or removing.**

B. **`research/00-evolution.md` vs. `research/00-evolution.md` v3-era**
   — PROJECT-SUMMARY.md:31–32 cites `research/00-evolution.md` as the
   "v3-era" version, but no `00-evolution.v2.md` or v3.md is on disk.
   `research/00-evolution.md` exists at 72 KB but is undated within the
   file. **Possible orphan or possible canonical-version: needs explicit
   front-matter "v3 era, current as of ..." to disambiguate.**

C. **`docs/superpowers/specs/2026-06-04-llm-layers-design.md`** (v1) and
   `2026-06-04-llm-layers-design.v2.md` (v2) — both still on disk.
   `PROJECT-SUMMARY.md:69–71` correctly lists all three (v1, v2, v3).
   `README.md:9` only mentions v2 (M1) and v3 (M2 / B0.5) — that's the
   right pointer. Not an orphan, just version-archived.

D. **`docs/superpowers/plans/2026-06-05-llm-layers-m1-qwen3.md`** and
   **`2026-06-06-llm-layers-m2-rollout.md`** — both still on disk. Both
   referenced. No v5/v6/v7 plans on disk (per the audit brief, these
   were executed without a plan file — `docs/superpowers/plans/` should
   gain a `2026-06-XX-llm-layers-v5-cleanup-and-extension.md` etc. or
   the choice to skip plans should be documented).

E. **`research/issues/09-*.md` slot** — `00-master-issues, 01..08,
   10, 11` exist but no `09`. Issue 09 was either deleted or never
   written. Recommend a 1-line `09-skipped.md` placeholder noting why.
   Minor.

### 6.5 Numerical gates not as a test

The audit brief asked: "Any layer.md claiming a numerical gate that
doesn't actually exist as a test". A full spot-check would need to
iterate over 35 layer.md files. Spot-checked 5:
- `models/qwen3/layer.md` cites the numerical gate at atol=5e-4 with
  observed max_abs_diff ~5e-7 — matches `tests/models/qwen3/test_numerical_hf.py`
  (the file exists).
- `models/gemma2/layer.md` cites bit-exact numerical gate — matches
  `tests/models/gemma2/test_numerical_hf.py` (exists).
- `models/deepseek_v4/layer.md` (one of the v7 families) cites
  shape-only CSA/HCA + numerical gate on hash MoE — matches
  `tests/models/deepseek_v4/` (3 tests).
- `models/qwen3_next/layer.md` — gated-deltanet numerical gate; matches
  `tests/models/qwen3_next/`.
- `models/minimax_m2/layer.md` — full DecoderLayer numerical gate;
  matches `tests/models/minimax_m2/`.
5/5 spot-checked OK. **No false numerical-gate claims found in the
sample.** Full sweep deferred.

### 6.6 Removed enum values still referenced

ShareScheme (D8). `RoPESpec.scale_factor` and `RoPESpec.is_2d` (`7098631
chore(v5): remove dead enum values + RoPESpec.{scale_factor,is_2d}`)
should be checked.

Grep for `scale_factor` in docs:
- `docs/API-REFERENCE.md` — line 3801–3805 (audit-corrected section
  notes `RoPESpec.scale_factor` as DEAD). PARTIAL HIT: the doc
  correctly flags it as dead but doesn't say "removed".
- `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md` — older
  spec, expected to mention historical field names. Not a problem.

Grep for `is_2d`:
- `docs/API-REFERENCE.md` — references the m-rope path. Need to check
  if `is_2d` is the surviving name or the removed one. (`grep is_2d
  api/specs.py` returns 0 hits — confirmed removed.)
- Spec v3 §5.2 references `is_2d`. Spec v3 historical.

Grep for `KVCacheSpec.num_kv_shared_layers` (also dropped):
- `README.md:43` cites "the four new spec fields, the new `PLESpec` and
  `ShareScheme` enum, the `SharedLayerKVCache` wrapper". Implicit
  reference. (D8.)

### 6.7 Summary count

Distinct production-doc pitfalls catalogued:

1. ShareScheme references in README + API-REFERENCE (×2 docs, ~12 line hits — D8).
2. `num_kv_shared_layers` references in README (×1 doc, indirect).
3. `RoPESpec.scale_factor` / `is_2d` references in API-REFERENCE (×1 doc).
4. Orphan `docs/issues/raschka-gallery-comparison.md` (no inbound link).
5. Project memory file 6 weeks stale (D5).
6. Spec v3 not marked superseded (D1.3).
7. Missing v5/v6/v7 plan files in `docs/superpowers/plans/` (D6.4 D).

**Pitfall count: 7 distinct categories.**

### D6 verdict: **PARTIAL.** Catalogued; most are downstream of D1
(updating top-of-tree narrative would clear half of them).

---

## 7. D7 verdict — Spec v3 → API-REFERENCE → code triangle: **2 minor drifts**

Spot-checked 5 spec v3 claims against API-REFERENCE.md against the code.

### 7.1 ShareScheme enum (spec v3 §5.2.5 area, lines 518, 839, 856)

- Spec v3 §5.2.7 / §5.2.13: `ShareScheme.NONE / CLA / YOCO / GEMMA4 /
  APPLE_AFM` (line 839, 856).
- API-REFERENCE.md §2.20 line 686: `ShareScheme` with `NONE`,
  `SAME_BLOCK_SHARED`, `CROSS_BLOCK_SHARED`.
- Code `api/types.py`: **`ShareScheme` does not exist** (D8).

**Triangle drift:** spec ≠ ref ≠ code. Spec was forward-looking (5
values), ref documented the 3 that landed at B0.5, code has 0 (removed
in v5).

### 7.2 `num_kv_shared_layers` field on KVCacheSpec

- Spec v3 §5.2.13 (line 839): `kv_source_layer: Optional[int] = None`
  is the field name in the spec.
- API-REFERENCE.md §0.3 row B0.5 (line 56): `num_kv_shared_layers`.
- Code: `grep num_kv_shared_layers api/specs.py` → 0 hits. Dropped by
  `8532879 chore(v5)`. The CLA `kv_source_layer_offset` field on
  AttentionSpec (line 278) is the surviving cross-layer-share mechanism.

**Triangle drift:** spec uses one name, ref uses a different name, code
has neither name (uses `kv_source_layer_offset` on AttentionSpec
instead).

### 7.3 PLESpec (Gemma 4 Per-Layer Embedding)

- Spec v3 §5.2.X: `PLESpec(embedding_dim=256, paged_to_storage=True)`
  with the building block `PLE`.
- API-REFERENCE.md §4.10: `PerLayerEmbedding` module, `PLESpec`
  dataclass.
- Code `api/specs.py:681` (`class PLESpec`) and `api/embedding.py`
  (`PerLayerEmbedding`).

**Triangle clean:** spec field semantic matches code; doc cites both.
✓

### 7.4 `final_logit_softcap` on AttentionSpec (Gemma 2/4)

- Spec v3 mentions `final_logit_softcap=30.0` as a B0.5 addition.
- API-REFERENCE.md §3.6 (line 988 area) documents it.
- Code `api/specs.py` — present. ✓

### 7.5 Op floor count

- Spec v3 §5.1: "16 logical ops" (the "ihv-consensus floor").
- API-REFERENCE.md §1: "**19** — the 16-op floor plus rope_apply_partial,
  rope_apply_mrope, and gelu_pytorch_tanh".
- Code `api/ops.py`: `grep '^def [a-z]' | grep -v '^def _'` → **26
  public functions** (including 5 new v5/v6/v7 ops: `gelu_exact`,
  `relu2`, `build_alibi_slopes`, `apply_alibi`, `l2norm`,
  `gated_delta_step`, `selective_scan_mamba1`).

Wait, that's 7 new public functions if we count generously. Let me
recount: `silu, add, mul, linear, rms_norm, embed, lm_head, rope_apply,
rope_apply_mrope, rope_apply_partial, gelu_pytorch_tanh, gelu_exact,
relu2, build_alibi_slopes, apply_alibi, sdpa, layer_norm, softmax,
top_k, gather, scatter, conv1d, selective_scan_mamba1, selective_scan,
l2norm, gated_delta_step` = **26 public ops**.

**Triangle drift:** spec says 16, ref says 19, code has 26. The ref is
already 7 ops behind v5/v6/v7 reality.

### 7.6 Triangle drift count

Spec→ref→code triangle drifts spot-checked: **2 sharp ones (ShareScheme,
op-floor count)** + 1 indirect (KVCacheSpec.num_kv_shared_layers field
name). Plus an arbitrary number more for v5/v6/v7 fields not in the
reference (BitNet QDType.TERNARY, BlockLayout.PARALLEL, AliBiSpec,
attn_sub_norm/ffn_sub_norm fields, selective_scan_mamba1 op, l2norm op,
gated_delta_step op, MaskKind.SINK forward semantics, ...) — these are
all "code added; ref silent", not strictly triangle disagreements.

### D7 verdict: **2-3 explicit drifts**, dozens of "ref silent on
code-added field" gaps. Bulk fix is the v4 → v5 reference update.

---

## 8. D8 verdict — Broken cross-references from refactors: **3 classes,
~14 doc-level hits**

### 8.1 `ShareScheme` enum (REMOVED in v5)

`8532879 chore(v5): remove dead KVCacheSpec fields + ShareScheme enum`.

- `api/types.py` — gone. ✓ (no broken code).
- `api/specs.py` — gone. ✓.
- `models/*/layer.py` and `tests/` — clean (`grep -rn ShareScheme
  models/ tests/` → 0 hits). ✓.
- **`README.md:44`** — "the new `PLESpec` and `ShareScheme` enum".
  BROKEN.
- **`README.md:56`** — "`types.ShareScheme` (NONE / SAME_BLOCK_SHARED /
  CROSS_BLOCK_SHARED)". BROKEN.
- **`docs/API-REFERENCE.md:56`** — table row "B0.5 | `PLESpec`,
  `ShareScheme`, ...". BROKEN.
- **`docs/API-REFERENCE.md:686–688`** — full §2.20 entry on
  `ShareScheme`. BROKEN.
- **`docs/API-REFERENCE.md:988,991,1694,1897,2027,2434,2648,3894`** —
  8 more references to `ShareScheme` / `share_scheme` /
  `SAME_BLOCK_SHARED` / `CROSS_BLOCK_SHARED`. BROKEN.
- Spec v3, plan v3, audit-trail files — historical artifacts;
  not broken in their context.

**Hits in production docs:** 2 in README + ~10 in API-REFERENCE = **12
doc-level references to a removed symbol.**

### 8.2 `DecoderBlockSpec.input_norm` → `block.pre_attn_norm` rename

`grep input_norm api/ models/` shows:
- `api/block.py` — only `post_per_layer_input_norm` (a different field,
  for PLE). Clean.
- `models/gemma4/layer.{py,md}` — `post_per_layer_input_norm` only.
  Clean.

But:
- **`docs/API-REFERENCE.md:2408`** — `**Commit `12e53f6`** — drop dead
  `DecoderBlockSpec.input_norm`; rename` — this is INTENTIONAL (the
  drift log entry that DOCUMENTS the rename). Not broken.
- **`docs/API-REFERENCE.md:3805`** — `**`DecoderBlockSpec.input_norm`**
  was a single field in spec v3 §5.2.8.` — again, intentional history.
- **`docs/superpowers/specs/2026-06-04-llm-layers-design.v2.md:894,
  964`** — `block_0.input_norm.weight` in the weight-name table.
  Historical, but a reader following the v2 spec for guidance would be
  confused.
- **`docs/superpowers/plans/2026-06-06-llm-layers-m2-rollout.md:1189,
  1192,1214,1757`** — `self.input_norm = ...` in plan pseudo-code.
  Historical.
- **`docs/superpowers/plans/2026-06-05-llm-layers-m1-qwen3.md:2269,
  2284,2783,2796`** — same.

**Hits in production docs:** 0 broken references in production docs;
the API-REFERENCE drift log entries are intentional history. **The
rename is cleanly handled.** ✓

### 8.3 `RoPESpec.scale_factor` and `is_2d` (REMOVED in v5)

`7098631 chore(v5): remove dead enum values + RoPESpec.{scale_factor,
is_2d}`.

- `api/specs.py` — gone. ✓.
- `docs/API-REFERENCE.md` — `grep scale_factor` shows the rotation-
  factor field name in YarnRoPEParams and Llama3RoPEParams, which is
  DIFFERENT (`factor` vs. `scale_factor`). No broken refs in
  API-REFERENCE for this name. ✓ (though it's worth a confirming pass).
- `docs/API-REFERENCE.md` — `grep is_2d` no hits. ✓.

**Hits in production docs:** 0. ✓

### 8.4 `KVCacheSpec.num_kv_shared_layers` (REMOVED in v5)

- `api/specs.py` — gone. ✓.
- **`docs/API-REFERENCE.md:56`** — `B0.5 | `PLESpec`, `ShareScheme`,
  `num_kv_shared_layers`, ...`. BROKEN.
- **`README.md`** — implicit reference via "the four new spec fields"
  (line 43); the four don't include `num_kv_shared_layers` by name so
  not literally broken, but the spec line that backed this claim is gone.

### 8.5 `KVCacheSpec.k_quant` / `v_quant` (dead)

Both still in code (per the audit `API-REFERENCE-audit.md` §4 #10) but
both raise NotImplementedError in `ContiguousKVCache`. Document as dead.
**Code-level: present but dead. Doc-level: §16b flags them. ✓**

### 8.6 Broken-ref summary

| Symbol | Production doc hits | Resolution |
|---|---|---|
| `ShareScheme` (removed v5) | **~12** (2 in README, ~10 in API-REFERENCE) | Update docs to "removed in v5, replaced by per-layer `AttentionSpec.kv_source_layer_offset` for CLA-style sharing" |
| `num_kv_shared_layers` (removed v5) | **1** (API-REFERENCE.md:56) | Same — replaced by `kv_source_layer_offset` |
| `DecoderBlockSpec.input_norm` rename | **0** broken (drift log entries only) | ✓ |
| `RoPESpec.scale_factor` removed | **0** | ✓ |
| `RoPESpec.is_2d` removed | **0** | ✓ |

**Total broken cross-references: ~13.**

### D8 verdict: **PARTIAL — 13 broken-ref hits in production docs, all
concentrated in README (2) + API-REFERENCE (~11). All point at the same
two removed symbols: `ShareScheme` and `num_kv_shared_layers`. Fixing
those two symbols across the two docs clears the bulk of D8.**

---

## 9. OVERALL — Must-fix items, prioritized

### 9.1 Verdict: **FAIL** for the project as a whole at HEAD `dd0d68a`.

The audit-corrected `API-REFERENCE.md` was excellent at its commit
(`4010e74`), but it has not been forward-ported through v5 / v6 / v7.
`README.md` and `PROJECT-SUMMARY.md` stop at B10 with no v5+ section.
`project_llm_layers.md` (the user's auto-memory) stops at M1.

The good news: there is **no internally-broken `api/` code** — the
removed symbols are gone, the rename is consistent, the tests pass.
The damage is entirely doc-level.

### 9.2 Top 5 must-fix items, in priority order

1. **Bring API-REFERENCE.md to v7-complete.** Add §0.3 rows for v5
   phase-1 cleanup (ShareScheme removal, scale_factor/is_2d removal,
   dead spec scalar removal); v5 phase-2 ALiBi (MPT), parallel residual
   (Falcon-7B), BitNet b1.58 + attn_sub_norm/ffn_sub_norm + ReLU²
   activation, attention sinks (GPT-OSS), CLA (Hunyuan-Large);
   v6 MPT/Falcon-7B full decoder, GPT-OSS MoE-with-bias, BitNet ternary
   QDType, Mamba-1 + Jamba; v7 phase-A DeepSeek-V4 CSA+HCA + hash MoE,
   Qwen3-Next Gated DeltaNet, Nemotron-H latent MoE, plus the 4 v7
   phase-B family bootstraps. Update §1 op count from 19 to 26.
   Update §18 roster to add 10 new families. Remove the 12 stale
   `ShareScheme` references; document as removed-in-v5 with the
   replacement (`AttentionSpec.kv_source_layer_offset`).

2. **Update PROJECT-SUMMARY.md to v7-complete.** Line 4 → "v7 complete,
   39 model families with `layer.py` (+ 2 shape-only, + 8 deferred
   stubs)". Line 17 → "Tags placed | 21". Line 25 → "Quant schemes
   operational | 6 (AWQ, GGUF Q4_K_M, FP8 E4M3, MXFP4, LiteRT W4A8,
   BitNet ternary)". Line 84 → `api/` line count = 6,096. Lines 113–146
   → add v5/v6/v7 batches to the per-batch table. Line 179–197 → extend
   the git timeline to v7-complete. Line 199 → HEAD = `dd0d68a`.

3. **Update README.md to v7-complete.** Add a "v5 phase-1 status",
   "v5 phase-2 status", "v6 status", "v7 phase-A status", "v7 phase-B
   status" block under the existing B10 block. Remove the
   `types.ShareScheme` reference (line 44 + 56) and replace with the
   surviving CLA mechanism. Line 167 → "Quantization '6 operational'"
   not "5 schemes landed, 1 stubbed". Optionally collapse all the B0.5–
   B10 blocks into a single "M2 complete" link to PROJECT-SUMMARY.md
   for clarity.

4. **Rewrite `project_llm_layers.md` memory file.** Replace the 2026-06-04
   M1 status with a 1-page summary at v7-complete + a pointer to
   `docs/PROJECT-SUMMARY.md` for the current snapshot. Keep
   evidence-first / verify-source-not-blogs / review-gates feedback
   notes — they are evergreen.

5. **Mark spec v3 as superseded.** Add a 1-line "Status: SUPERSEDED by
   docs/API-REFERENCE.md as of v7-complete (see §0.3 for the v3→v7 drift
   trail)" at the top of `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md`.
   Same for the M2 rollout plan — it was executed; mark "Status:
   EXECUTED (M2-complete tag); v5–v7 followed without a separate
   rollout plan".

### 9.3 Secondary cleanup (low priority)

- Link `docs/issues/raschka-gallery-comparison.md` from PROJECT-SUMMARY
  §2 "Critique trail", or delete if it has no ongoing value.
- Add 1-line `research/issues/09-skipped.md` placeholder (or rename
  `10-gemma4-investigation.md` to `09-`).
- Optionally write a `docs/superpowers/plans/2026-06-XX-llm-layers-v5-v7-extensions.md`
  retrospective to capture the v5+ extension trajectory (or accept
  that v5+ was extension-by-discovery without a plan).
- Make the test count consistent: re-run `pytest --collect-only -q`
  and update PROJECT-SUMMARY.md:17 (currently 604, post-v7 it will be
  higher — perhaps 700+).

### 9.4 What's NOT broken

- `api/*.py` symbol set, removed-symbol hygiene, tests under `tests/`.
- The B0.5 → B10 documentation. (It is correct for its date; the
  problem is forward-porting.)
- The audit-trail files in `docs/issues/` (point-in-time correct).
- The rename `block.input_norm` → `block.pre_attn_norm` (cleanly
  propagated to code; drift log entries in API-REFERENCE are
  intentional history).
- Spec v3 + plan v3 as design-time artifacts (point-in-time correct).
- All evergreen memory files (`user_role.md`,
  `feedback_evidence_first.md`, `feedback_verify_source_not_blogs.md`,
  `feedback_review_gates.md`).

### 9.5 Estimated cost of fixes

- Top-of-tree forward-port (README, PROJECT-SUMMARY, API-REFERENCE):
  **~half a working day for an editor familiar with the project**
  (mostly mechanical: copy-paste the existing B-batch block format and
  fill in the v5/v6/v7 specifics). Highest single ROI.
- Memory file rewrite: **15 minutes.**
- Spec-supersedence stamps: **5 minutes each, 2 files.**
- Secondary cleanup: **1 hour.**

After items 1–5 are applied the project doc set goes from **FAIL** to
**PASS**.

---

*End of Audit D — 2026-06-10. Auditor cut-date HEAD: `dd0d68a` (post
v7-complete + the final test-init commit). Tally: 1 FAIL doc-set verdict
overall; D1 FAIL; D2 PARTIAL; D3 PARTIAL; D4 FAIL; D5 FAIL; D6 PARTIAL
(7 pitfalls); D7 2–3 explicit drifts + many code-ahead-of-doc gaps; D8
~13 broken-ref hits.*
