# llm-layers

Minimal, evidence-grounded API for assembling mainstream small language models
(<8B params) as decoder graphs, with quantization and KV-cache as first-class
parameters.

## Design

See `docs/superpowers/specs/2026-06-04-llm-layers-design.v2.md` (M1) and
`docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md` (M2 / B0.5).

## Survey

Five research reports + an evolution-narrative report in `research/`:

- `00-evolution.md` — 2022→2026 timeline, lineage diagrams, abandoned designs
- `01-model-census.v2.md` — 80 mainstream SLMs across 15 axes
- `02-layer-sources.v2.md` — 33 family decoder-block decompositions
- `03-ihv-opsets.v2.md` — 18 IHV runtimes' op-set comparison
- `04-quantization.v2.md` — 42 quant schemes with parameter space
- `05-kvcache-attention.v2.md` — 29 attn + 14 RoPE + 13 cache variants

## M1 status — Qwen3 kickoff gate PASSED

```powershell
uv sync --extra test --extra dev
uv run pytest -v
```

117 tests pass, 4 documented xfails, 4 documented IR-divergence skips on the
Gemma 4 HF gate (see B0.5 section below). Layer-0 is numerically equivalent
to HF `transformers.Qwen3DecoderLayer` at **atol=5e-4** on a fixed input —
observed max_abs_diff: **~5e-7** (≈1000× tighter than the contract).

See:
- `models/qwen3/layer.md` for the architecture documentation
- `models/qwen3/layer.py` for the API-assembled factory
- `tests/models/qwen3/` for shape, isolation, numerical, KV cache, and AWQ tests

## B0.5 status — Gemma 4 v3 IR seeded, IR-locked, gate deferred

B0.5 lands the v3 IR extensions needed by Gemma 4 (the most architecturally
divergent SLM family in the M2 census). All spec and code is in place; the
four new spec fields, the new `PLESpec` and `ShareScheme` enum, the
`SharedLayerKVCache` wrapper, and the new `api/embedding.py` module are
exercised by 30+ B0.5 tests. See `models/gemma4/layer.md` for the layer
documentation.

New spec fields:
- `AttentionSpec.attention_k_eq_v` — Gemma 4 12B+ global K=V aliasing.
- `AttentionSpec.qk_norm_fixed_scale` — fixed-scale absorption hook.
- `RoPESpec.partial_rotary_factor` — Gemma 4 global partial-RoPE 0.25, Phi-3.
- `DecoderBlockSpec.per_layer_embedding` (`PLESpec`) — Gemma 4 E2B PLE.

New enums / types:
- `types.ShareScheme` (NONE / SAME_BLOCK_SHARED / CROSS_BLOCK_SHARED).
- `specs.PLESpec` (Per-Layer Embedding parameters).

New modules:
- `api/embedding.py` — `PerLayerEmbedding` (Gemma 4 PLE table + projection).
- `api/kvcache.SharedLayerKVCache` — alias-only KV-cache wrapper.

### B0.5 HF gate — IR-divergence skip path

The T17 numerical-equivalence gate (`tests/models/gemma4/test_numerical_hf.py`)
and the T18 KV-cache gate (`tests/models/gemma4/test_kvcache_hf.py`) run
end-to-end against `google/gemma-4-e2b-it`. With the B0.5 IR they record
`max_abs_diff` ≈ 1.5e+03 (layer-0, local SWA) and 4.25e+01 (layer-4, global,
partial-RoPE 0.25), substantially above the `atol=5e-4` contract.

Per the plan pragma — "do not weaken atol; skip with reason" — the gate tests
`pytest.skip()` with the diff and the four documented IR drifts captured by
the `xfail` tests in `test_isolation_hf.py`:

1. **RMSNorm weight_mode** — HF Gemma 4 uses `STANDARD_W`; our IR emits
   `ONE_PLUS_W` (carried over from Gemma 3).
2. **No QK fixed-scale absorption** — HF attention uses `self.scaling = 1.0`;
   the speculative `qk_norm_fixed_scale` (0.9916 local / 1.0228 global) baked
   into B0.5 has no counterpart in the released source.
3. **Missing `v_norm`** — HF has a unit-scale `Gemma4RMSNorm(with_scale=False)`
   on V before the cache write; our `AttentionSpec` lacks the field.
4. **Partial-RoPE geometry** — HF uses `proportional` rope_type with cos/sin
   of FULL `head_dim` (zero-padded inv_freq); our `rope_apply_partial`
   rotates a `Dh_rot` prefix with cos/sin of length `Dh_rot`. Geometrically
   distinct.

These drifts are the scope of the planned B0.6 IR-correction batch. Both T17
and T18 test stubs will assert-pass without modification once the IR is
updated; no test code changes are required to flip the skip into a pass.

## API surface (M1 + B0.5)

- `api/types.py` — enums and small types
- `api/specs.py` — frozen dataclasses (parameter spaces)
- `api/ops.py` — functional primitives (rms_norm, linear, silu, gelu_pytorch_tanh,
  add, mul, embed, lm_head, rope_apply, rope_apply_partial, sdpa, ...)
- `api/{norm, rope, kvcache, quant, attention, feedforward, block, embedding}.py` —
  nn.Module building blocks

The full API floor lands across M2+ rollout batches (MoE, SSM, MLA, more quant
schemes). M1 establishes the foundational shape; B0.5 extends with Gemma 4
primitives.

## B8 status — OCR-LLM family (3 families landed)

B8 adds the LM-decoder portion of three OCR / Vision-LLM families:

- **GOT-OCR 2.0** (`models/got_ocr2/`) — Qwen2-0.5B backbone with concat-prefix
  vision fusion. Numerical gate vs `stepfun-ai/GOT-OCR-2.0-hf`:
  max_abs_diff = 3.81e-6 (~131x tighter than atol=5e-4).

- **Qwen2.5-VL 3B** (`models/qwen2_5_vl/`) — Qwen2.5 backbone with **M-RoPE**
  (multimodal rotary position embedding). New api hooks:
  `RoPESpec.mrope_section`, `ops.rope_apply_mrope`. Numerical gate vs
  `Qwen/Qwen2.5-VL-3B-Instruct`: max_abs_diff = 1.67e-6 (~299x tighter).

- **DeepSeek-OCR-2** (`models/deepseek_ocr2/`) — DeepseekV3-style MoE
  (12 layers; layer 0 dense, 1-11 sparse MoE). Numerical gates:
  - Synthetic-weight gates (dense + MoE): max_abs_diff = 1.19e-7 / 1.19e-7
  - HF-checkpoint gates (dense + MoE):    max_abs_diff = 4.77e-7 / 1.42e-7

**Source-grounded drift caught for DeepSeek-OCR-2:** the HF transformers
v5.10.2 `deepseek_ocr2` reference uses **STANDARD MHA + softmax MoE +
standard causal mask** — NO MLA and NO block-bidirectional "Visual Causal
Flow" mask described in the original paper §3.2. The VCF mask is landed as
a v3-spec hook (`MaskKind.BLOCK_BIDIRECTIONAL` +
`AttentionSpec.block_bidirectional_mask` + `Attention.forward(vision_token_count=...)`)
exercised by shape-only tests; the canonical numerical gate runs against
CAUSAL per the HF source.

## Project structure

See `docs/superpowers/plans/2026-06-05-llm-layers-m1-qwen3.md` for the M1 plan,
and `docs/superpowers/plans/2026-06-06-llm-layers-m2-rollout.md` for the M2 / B0.5
plan.

## License

Apache 2.0 (project code). Model weights remain under their respective
HuggingFace licenses (Apache 2.0 for Qwen3-0.6B; Gemma terms for Gemma 4 E2B).
