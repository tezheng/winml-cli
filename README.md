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
