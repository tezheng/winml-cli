# Reading-path critique — skeptical audit of the proposed 7-step "what was achieved + current status" path

**Date:** 2026-06-11
**Auditor:** llm-layers-investigator sub-agent (skeptical lens)
**Scope:** verify each of the 7 documents on the proposed path against ground truth on disk, then assess the path overall.

---

## Path under audit

1. `README.md`
2. `docs/PROJECT-SUMMARY.md`
3. `docs/issues/00-v7-final-audit-synthesis.md`
4. `docs/API-REFERENCE.md`
5. `research/00-evolution.md`
6a. `models/qwen3/layer.md`
6b. `models/gemma4/layer.md`
7. `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md`

---

## 1. Per-doc verdicts

### 1.1 `README.md` — verdict: **PARTIAL** (improved, but still has remnants)

**Honesty / currency:**

- Headline line 7: `Status — M1 → M2 → v5 → v6 → v7 complete` — present and correct.
- Line 9-11: `41 architecturally distinct model families`, `751 tests`, `22 git tags placed (latest v7-complete)` — present.
- Line 18-22: v5→v7 release-wave summary present (ALiBi, parallel residual, BitNet ternary, GPT-OSS sinks, CLA, Mamba-1, Jamba alternation, DeepSeek-V4 hash routing, Qwen3-Next Gated DeltaNet, GLM-MoE-DSA, MiniMax-M2). Reasonable forward-port.

**Problem 1 — old M1 + B0.5 status blocks still present (lines 40-108).** After the new top status block (lines 7-22), the README still contains the original "M1 status — Qwen3 kickoff gate PASSED" section AND the "B0.5 status — Gemma 4 v3 IR seeded, IR-locked, gate deferred" section (with its now-obsolete `pytest.skip` discussion of the four IR drifts that B0.6 already fixed). A new reader will read all 108 lines as if those were the current status — they are historical context, not current truth. The Gemma 4 numerical gate now **passes** at atol=5e-4 (verified in `models/gemma4/layer.md` §9 — "T17 real-weight layer-0/4 numerical equivalence at atol=5e-4 — passes"). The README's B0.5 block says it `pytest.skip`s; that's stale.

**Problem 2 — B8/B9/B10 status blocks still present (lines 123-213).** Same problem on a smaller scale: a reader sees a list of "B8 status", "B9 status", "B10 status" blocks but no "v5 / v6 / v7" status blocks. The status taxonomy in the README is inconsistent across release waves.

**Problem 3 — no quickstart for "review what landed".** The README has a `uv sync && pytest -v` invocation but no pointer to PROJECT-SUMMARY.md or to the §6 family roster. A reader who wants to know "what families landed?" has to scroll the README itself.

**Verdict:** the new top block is honest; the body below it is doc-debt that contradicts the top block on Gemma 4 gate status. **PARTIAL.**

---

### 1.2 `docs/PROJECT-SUMMARY.md` — verdict: **PARTIAL** (good content; 3 internal contradictions)

**Honesty / currency:**

- Date 2026-06-11, "M1 + M2 + v5/v6/v7 complete" — current.
- §6 working-families table lists all 41 families across batches M1 → v7 — present and correctly enumerated. Spot-checked: MPT, Falcon-7B, BitNet, Hunyuan-Large, GPT-OSS, Mamba-1, Jamba, DeepSeek-V4, Qwen3-Next, GLM-MoE-DSA, MiniMax-M2 all appear with correct gate classes and atol numbers. **This is the cleanest doc on the path.**

**Problem 1 — tag count off-by-one.** §0 scoreboard line 16: `Tags placed | 21`. README line 11: `22 git tags placed`. One of these is wrong (the actual tag count is what matters; the docs disagree).

**Problem 2 — test count contradiction within the same doc.** §0 scoreboard line 17: `Tests collected | 751`. §7 header line 165: `Test inventory — tests/ (604 collected)`. §7 line 171: `160 test files across 30 families` — yet line 18 says 41 families. The §7 paragraph is forward-porting-incomplete: the count number was updated at the top but not in §7.

**Problem 3 — internal contradiction on Jamba.** Line 152 lists Jamba as a landed v6 family (Mamba layer bit-exact, attention layer 4.77e-7). Line 161 simultaneously lists `models/jamba/` in the "Deferred stubs (B7+ follow-up)" section. Both can't be true. The reader can't reconcile this — they have to look at `models/jamba/` on disk to discover Jamba did in fact land in v6.

**Problem 4 — drift count is inconsistent.** Line 22 scoreboard says `~70` source-grounded drifts caught. §10 says `~55 across batches`. §10's "What this proves" §6 also says `~55`. Three numbers, two values.

**Problem 5 — `api/` LOC scoreboard contradiction.** Line 19 says `~6,100` lines. §5 line 84 says **3,626 lines**. The §5 number is a forward-port omission — the line numbers below it sum to ~3,636. The 6,100 number is the post-v7 figure that was added to §0 but never propagated to §5.

**Verdict:** the content is largely accurate, but a careful reader will find 5 distinct internal contradictions. **PARTIAL** — the numbers need to be reconciled top-to-bottom.

---

### 1.3 `docs/issues/00-v7-final-audit-synthesis.md` — verdict: **PARTIAL** (the audit is honest, but it's STALE — most cleanup it lists has now been done)

**Critical issue:** this audit was written 2026-06-10 with the headline "Code is clean. Docs are stale. 1 critical doc bug." The 1 critical bug was 12-13 `ShareScheme` references in API-REFERENCE.md / README.md.

**Verified against current state:**

- ShareScheme refs in README.md: **0** (down from 2). Done.
- ShareScheme refs in API-REFERENCE.md: **3** (down from 8+). Spot-checked — all 3 are historical-context references ("removed in v5 Phase 1 commit 8532879") in §2.20, §8 commit history, and the §8.15 SHA archive table. **None of them would cause a `NameError` if copy-pasted.** Verified §6.3 Gemma 4 recipe (the audit's worst-case example) now uses `kv_source_layer_offset` + `kv_source_layer_idx_map()`, NOT `ShareScheme`. The critical bug is fixed.
- "Top-level docs frozen at B10": README has the new top status block (M1→v7); PROJECT-SUMMARY has the new top status block; API-REFERENCE.md still has `Status: Post-rollout reference, current as of B10` (line 3) and `## §1 The 16 Logical Ops` with "current count is 19" (line 110) — neither updated.

So:
- Finding 1 (ShareScheme critical bug): **RESOLVED** but audit still reads as if must-fix.
- Finding 2 (3 top-level docs frozen at B10): **PARTIALLY RESOLVED** (README + PROJECT-SUMMARY forward-ported; API-REFERENCE.md still at B10).
- Finding 3 (17 of 20 v5/v6/v7 IR additions undocumented in API-REFERENCE): **NOT verified by this audit** — likely still true.
- Finding 4 (Gemma 4 `from_hf_dict` silent bugs): not investigated this turn.
- Finding 5 (DeepSeek-V4 layer.md inconsistencies): not investigated.

The audit doc itself has no "RESOLVED" markers added since 2026-06-10 — a reader will read it as live must-fix list and panic about ShareScheme even though it's already cleaned.

**Verdict:** **PARTIAL.** Useful as a snapshot, but reads as if all findings still apply when ~half are fixed. Should be marked "STATUS AS OF 2026-06-10 — see §X for resolution status".

---

### 1.4 `docs/API-REFERENCE.md` — verdict: **FAIL** (still frozen at B10; misleading on op/spec/family counts)

This is the largest and most technical doc (4,279 lines, ~25k words). It is also the most stale.

**Stale claims (verified):**

- Line 1: title `API Reference v4 (post-rollout)`.
- Line 3: `Status: Post-rollout reference, current as of B10 (2026-06-08).` — three release waves stale (v5/v6/v7).
- Line 101: `## §1 The 16 Logical Ops (api/ops.py)` — wrong count.
- Line 110: `The current count is 19` — Audit B says 26. Op floor never updated through v5/v6/v7 (ALiBi `build_alibi_slopes` / `apply_alibi`, `relu2`, `gelu_exact`, `selective_scan_mamba1`, `l2norm`, `gated_delta_step` added in v5-v7 are not catalogued).
- §3 line 706: `The 18 frozen dataclasses below…` — Audit B says 22 (AliBiSpec, CSASpec, HCASpec, GatedDeltaNetSpec added).
- §18 line 4043: `## §18 The 41-Family Roster` — header says 41 but only enumerates ~30 entries (18.1 through 18.30 + 30b/c). The v5 families (MPT, Falcon-7B, BitNet, Hunyuan-Large, GPT-OSS), v6 families (Mamba-1, Jamba), and v7 families (DeepSeek-V4, Qwen3-Next, GLM-MoE-DSA, MiniMax-M2) are missing from the enumerated entries.
- §18 "Deferred families (stubs)" at line 4263 lists **Mamba-1** and **Jamba** as deferred — but both landed in v6 (PROJECT-SUMMARY §6 confirms; `models/mamba1/layer.py` and `models/jamba/layer.py` exist).

**Honesty on ShareScheme (audit's worst critique):** the 3 remaining references are correctly framed as "removed in v5 Phase 1" historical context. §6.3 Gemma 4 recipe is clean. The audit's "ghost in §6.3" finding is no longer accurate. ✓

**Verdict:** **FAIL** for currency. The doc cleanly documents what existed at B10 but anyone reading it post-v7 will believe:
- The IR has 19 ops (wrong — 26).
- The IR has 18 dataclasses (wrong — 22).
- DeepSeek-V4 / Qwen3-Next / GLM-MoE-DSA / MiniMax-M2 / Mamba-1 / Jamba / BitNet / GPT-OSS / Hunyuan-Large / Falcon-7B / MPT aren't part of the project (they are — they landed in v5-v7).
- Mamba-1 and Jamba are still deferred (they aren't — they landed in v6).

---

### 1.5 `research/00-evolution.md` — verdict: **PARTIAL** (timeline & axes are solid; v5-v7 era invisible)

Dated 2026-06-04. Content: 2022 Nov → 2026 Q1 evolution narrative. 9 family lineage diagrams. Per-axis evolution arcs (attention, RoPE, norm, FFN, vocab, quant, KV cache). Abandoned-design graveyard. 2026 consensus-vs-divergence table.

**Strengths:**

- High pedagogical value. A reader new to the project gets the "why" of every IR axis.
- Section structure is clean (1 Intro, 2 Timeline, 3 Lineages, 4 Per-axis, 5 Abandoned, 6 Consensus, 7 Open, 8 Cross-org, 9 Future-proofing, 10 References).

**Weakness — currency:**

- Title says `2022 Nov → 2026 Q1`. Content stops at 2025-09 (line 170: `DeepSeek-V3.1 / Qwen3-Next`). DeepSeek-V4 (Apr 2026), GLM-MoE-DSA / GLM-5, MiniMax-M2 (Jun 2026), Gemma 4 official release (Apr 2026) — all post-2025-Q4 — are largely absent. Qwen3-Next is the only one that appears in the timeline.
- The doc is dated 2026-06-04. The spec v3 was written 2026-06-06 and references all the post-2026-Q1 families. So the evolution narrative is older than the spec it claims to back-fill.

**Verdict:** **PARTIAL.** This is still the best high-level explainer for axes 1-9, but a v3 extension covering the 2026-H1 frontier additions would be needed for a publication-quality survey paper. For the proposed reading-path use case (understand what was achieved), it's a good foundation but misses the most recent 6 months of architectural innovation that this very project just implemented.

---

### 1.6a `models/qwen3/layer.md` — verdict: **PASS**

- Clean diagram, tensor IO trace, op trace, spec instantiation, weight-name mapping, source citations, validation status.
- Numerical-gate claim (`max_abs_diff observed: 4.77e-7`) matches the M1 entry in PROJECT-SUMMARY §6 exactly.
- The §6 "Quirks" section accurately calls out:
  - QK-norm is PRE-RoPE (verified against `modeling_qwen3.py`).
  - QK-norm shape `[head_dim]` (per-head Dh, distinct from OLMo 2).
  - `rope_theta=1_000_000` (32k context).
  - `tie_word_embeddings=True` for 0.6B/1.7B/4B; `False` for 8B.
  - Explicit `head_dim` (don't infer from hidden_size / num_heads).
  - No biases on any projection.
  - transformers 5.10.2 RoPE config nesting.

A clean canonical example. **PASS** — this is what a per-family layer doc should look like.

---

### 1.6b `models/gemma4/layer.md` — verdict: **PASS**

- Diagram includes the four sandwich norms (pre-attn, post-attn, pre-ffn, post-ffn) AND the B0.6 corrections (PLE-at-END injection, layer_scalar trailing multiply, v_norm, no QK fixed-scale).
- §9 validation status says "T17 real-weight layer-0/4 numerical equivalence at atol=5e-4 — passes" — this **contradicts** the README's B0.5 status block which says T17 skips with `~1.5e+03` diff. The layer.md is current (post-B0.6); the README has stale B0.5 text below the new v7 top block.
- Spot-checks: cross-layer KV sharing description correctly uses `kv_source_layer_idx_map()` (the v5 Phase 1 mechanism), NOT `ShareScheme`. ✓
- 35 layers (28 local + 7 global, 4:1 pattern) matches `models/gemma4/config.py`.
- E2B-specific numbers (D=1536, Hq=8, Hk=1, Dh_local=256, Dh_global=512, I=8192) self-consistent.

**PASS.** Most architecturally complex example in the project, accurately documented.

---

### 1.7 `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md` — verdict: **PASS** (with SUPERSEDED stamp)

- Line 3: `**Status: SUPERSEDED.** This spec was the pre-rollout design intent for M2 (B0.5→B10). The post-rollout reference is docs/API-REFERENCE.md and the cumulative scoreboard is docs/PROJECT-SUMMARY.md.` — the superseded stamp the audit asked for is present.
- The reader gets a clear "this is historical, go look at API-REFERENCE.md" pointer at line 1.
- §0 "Survey-paper preamble" is a beautifully written evolution narrative that genuinely covers the 2025-H2 / 2026-H1 frontier (Llama 4 iRoPE, DSA, DeepSeek-V4 CSA+HCA, Mamba-3, MiniMax M3.0, Gemma 4 K=V, Qwen3-Next Gated DeltaNet, MiniMax Lightning Attention, Granite 4, Nemotron 3, Visual Causal Flow) — much more current than `research/00-evolution.md`.

**PASS.** But: the SUPERSEDED stamp means the path #7 is essentially historical reading — useful for "how the project was designed" but not for "current truth". A reader on a short timeline might rightly skip it after seeing the stamp.

---

## 2. Path-level critique

### 2.1 Top-level navigation

The path's CONNECT-the-dots problem: the user has to **manually triangulate** between docs whose claims disagree.

- README says 22 tags, PROJECT-SUMMARY says 21.
- PROJECT-SUMMARY §0 says 751 tests, §7 says 604 collected.
- README B0.5 block says Gemma 4 T17 skips; layer.md says T17 passes.
- PROJECT-SUMMARY lists Jamba as both v6-landed AND deferred-stub.
- API-REFERENCE.md says 19 ops; audit-synthesis says 26.
- API-REFERENCE.md says 18 dataclasses; audit-synthesis says 22.

A reader who reads in path order will end up confused about which numbers to trust. The numerical-gate claims (Gemma 4 passes vs skips) are the most worrying — they affect the project's headline claim.

### 2.2 Order

**My recommendation: change the order.**

The proposed #3 (audit-synthesis) is most useful AFTER #4 (API-REFERENCE.md), not before. Reading the audit first tells the reader "the docs are stale and there's a critical ShareScheme bug" — but most of those findings are now resolved or stale themselves. Reading API-REFERENCE.md first lets the reader form their own opinion; then the audit becomes a useful "here are the rough edges you may have noticed".

The proposed #7 (spec v3) is correctly placed last (the SUPERSEDED stamp makes it clear it's historical).

#6a (Qwen3) and #6b (Gemma 4) are well-placed: after the IR doc, before the historical spec.

### 2.3 Gaps in the path

**Gap 1 — no pointer to tests.** "Where do I see the test results that prove the bit-exact / atol=5e-4 claims?" None of the 7 docs tell the reader to run `uv run pytest -v` and observe the gate tests, or to read `tests/models/gemma2/test_numerical_hf.py` for the one genuine bit-exact case.

**Gap 2 — no IHV comparison doc.** `docs/issues/raschka-gallery-comparison.md` exists in the repo but is not on the path. If the user wants "how does this compare to other LLM IR projects (Hugging Face, llama.cpp, vLLM)?", that doc is the answer.

**Gap 3 — no "what the IR cannot do" backlog visible from the path.** The audit-synthesis mentions §9 "open extension points" of API-REFERENCE.md and 10 census families without `models/<family>/`. But there is no consolidated M3 backlog doc — and the path doesn't tell the reader to look at the `Deferred families (stubs)` section of API-REFERENCE §18 (which itself wrongly lists Mamba-1 and Jamba as still-deferred).

**Gap 4 — no memory file refresh pointer.** Audit finding §6 (memory file frozen pre-M2) is not actionable from the path. A reader who wants to know "what does future-me know about this project" would have to know to look at `~/.claude/projects/.../memory/project_llm_layers.md`.

---

## 3. Top 5 pitfalls a user could hit

1. **Gemma 4 gate status confusion.** README lines 82-108 say T17/T18 skip with ~1.5e+03 diff. `models/gemma4/layer.md` §9 says T17/T18 pass at atol=5e-4. Same project, two docs, opposite claims. A reader who reads the README cover-to-cover concludes "Gemma 4 is broken"; a reader who reads layer.md concludes "Gemma 4 is the most complex landed family". The truth is the latter (B0.6 fixed the drifts).

2. **41 vs 30 vs 18 family enumeration.** PROJECT-SUMMARY §6 enumerates 41. API-REFERENCE §18 header says 41 but only enumerates ~30 entries (no v5/v6/v7 entries). README mentions all 41 batches in the v5→v7 status paragraph. Three different mental models depending on which doc you read.

3. **"19 ops" trap in API-REFERENCE.** A new contributor reading §1 will believe the op floor is 19 and start adding their new op as the 20th. Actually it's 26 — they'd be writing duplicate `relu2` / `gelu_exact` / `selective_scan_mamba1` / `l2norm` etc.

4. **Bit-exact misinterpretation.** PROJECT-SUMMARY §0 clearly distinguishes "1 family bit-exact vs real weights" (Gemma 2) from "3 synthetic-mini bit-exact" (Mamba-2 mini, Granite-4-H mini, DeepSeek-V3 MoE submodule). A speed-reader could conflate them and tell colleagues "the project achieved bit-exact on 4 families". The §6 Mamba-2 row even explicitly says "2.7B variant is shape-only" — but a speed-reader skips that.

5. **Audit-as-todo.** The audit at #3 reads as a punch-list. A reader who follows the recommended fix sequence (steps 1-8) will spend hours scrubbing ShareScheme refs that are already cleaned. The audit needs a STATUS AS OF stamp and ideally RESOLVED markers on individual findings.

---

## 4. Recommended path changes

### Suggested replacement path (7 steps, reordered)

1. `README.md` — entry. **But:** advise the user "read lines 1-22 only; the M1/B0.5/B8/B9/B10 status blocks below are historical".
2. `docs/PROJECT-SUMMARY.md` — scoreboard. **But:** advise the user the 21-vs-22, 751-vs-604, and Jamba-twice contradictions are doc-debt, not signal.
3. `models/qwen3/layer.md` — canonical example FIRST (was 6a). Reading a working layer-doc grounds the reader in the IR before they hit the abstract reference.
4. `models/gemma4/layer.md` — most complex example (was 6b). Demonstrates the IR can express a frontier family.
5. `docs/API-REFERENCE.md` — IR reference. **But:** warn the reader §1 op count is stale (19 → 26), §3 dataclass count is stale (18 → 22), §18 family roster is stale (missing v5/v6/v7 entries; Mamba-1/Jamba wrongly in deferred).
6. `docs/issues/00-v7-final-audit-synthesis.md` — audit verdict, AFTER reading IR reference. **But:** the user now has context to see that the ShareScheme critical bug is already fixed and most "must-fix" items are cleaned. Tell the user this is a 2026-06-10 snapshot.
7. `research/00-evolution.md` — survey backbone. Pedagogically valuable, but stops at 2025-Q4. For 2026-H1 frontier (DeepSeek-V4 / Qwen3-Next / GLM-MoE-DSA / MiniMax-M2), point the user instead at the §0 preamble of `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md`.

### Drop from path

- `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md` as a standalone item — it's clearly SUPERSEDED. Cite only its §0 preamble for the 2026-H1 frontier narrative.

### Add to path (or at least make discoverable)

- `tests/models/gemma2/test_numerical_hf.py` — the one genuine bit-exact-vs-real-weights gate (5 minutes of reading).
- `docs/issues/raschka-gallery-comparison.md` — for "how does this relate to other LLM IR projects".
- `docs/issues/API-minimality-coverage-audit.md` — for "what could the IR do that it doesn't yet".

---

## 5. Specific edits the docs still need

1. **README.md** — drop lines 40-108 (the M1 + B0.5 status blocks). Replace with a 5-line "see PROJECT-SUMMARY.md §6 for per-family validation status".
2. **README.md** — drop lines 123-213 (B8/B9/B10 blocks; superseded by the v5→v7 top block).
3. **README.md** — reconcile tag count: 21 or 22 (run `git tag -l | wc -l`).
4. **PROJECT-SUMMARY.md** §7 — update from `604 collected` and `30 families` to `751 collected` and `41 families`.
5. **PROJECT-SUMMARY.md** §5 line 84 — update `Total: 3,626 lines` to the post-v7 LOC (matches scoreboard `~6,100`).
6. **PROJECT-SUMMARY.md** line 161 — remove Jamba from the deferred-stubs list (it landed in v6).
7. **PROJECT-SUMMARY.md** — reconcile drift count: 55 or ~70. Pick one and use it three places.
8. **API-REFERENCE.md** §0 / §1 / §3 — forward-port to v7 (26 ops, 22 dataclasses, v5/v6/v7 entries in §18, remove Mamba-1/Jamba from deferred §18 list).
9. **API-REFERENCE.md** — update `Status: current as of B10` → `current as of v7`.
10. **audit-synthesis.md** — add a "STATUS AS OF 2026-06-11" preamble + RESOLVED markers on §1 (ShareScheme cleaned), §2 (README + PROJECT-SUMMARY forward-ported; API-REFERENCE not yet).
11. **research/00-evolution.md** — either bump the date or add a v3 supplement with the 2026-H1 frontier (DeepSeek-V4 / Qwen3-Next / GLM-MoE-DSA / MiniMax-M2 / Mamba-1 landing / Jamba alternation).

---

## 6. Overall verdict — would I send a colleague through this path?

**Y-but.**

**Y:** The 7 documents together do cover what was achieved. The combination of (PROJECT-SUMMARY §6 table + qwen3/layer.md + gemma4/layer.md) is sufficient to convince a sophisticated reader that 41 families landed and the IR is sound. The audit-synthesis honestly catalogues remaining doc-debt. The README top block + PROJECT-SUMMARY scoreboard give the headline numbers.

**But:** the path has at least 5 internal contradictions that will trip a careful reader, and the API-REFERENCE.md (the technical centerpiece, ~25k words) is 3 release waves stale on family count, op count, and dataclass count. A colleague who is asked to "review current status" will spend roughly half their reading time triangulating doc-vs-doc disagreements rather than learning the IR.

Recommendation: **before sending the colleague through this path, do the 2-hour cleanup of items 1-9 above** (it's exactly what the audit-synthesis recommended on 2026-06-10 and what was partially done since). After that cleanup, the path becomes **unambiguously Y** — and the colleague's reading-time goes to architectural depth (the strength of qwen3/layer.md and gemma4/layer.md) rather than reconciling stale numbers.

The strongest documents on the path are the per-family layer.md files (Qwen3 and Gemma 4). The weakest is the API-REFERENCE.md, precisely because it's the most ambitious doc — and the most exposed to currency drift across release waves.

---

*End of critique. Generated 2026-06-11 against the on-disk state of `C:\Users\zhengte\external\llm-layers`.*
