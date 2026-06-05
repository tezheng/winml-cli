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

## M1 status — Qwen3 kickoff gate ✅ PASSED

```powershell
uv sync --extra test --extra dev
uv run pytest -v
```

All 59 tests pass. Layer-0 is numerically equivalent to HF
`transformers.Qwen3DecoderLayer` at **atol=5e-4** on a fixed input —
observed max_abs_diff: **~5e-7** (≈1000× tighter than the contract).

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
