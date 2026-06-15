# llm-layers

Minimal, evidence-grounded API for assembling mainstream small language models
(<8B params) as decoder graphs, with quantization and KV-cache as first-class
parameters.

## Quickstart

```powershell
uv sync --extra test --extra dev
uv run pytest -v
```

751 tests pass at HEAD `ee84053`. Each family ships with a numerical-equivalence
gate at `atol=5e-4` against the canonical HuggingFace `modeling_*.py` reference;
see `models/<family>/layer.md` for the per-family architecture documentation.

## Current status — M1 → M2 → v5 → v6 → v7 complete

41 architecturally distinct model families landed (39 with full `layer.py` +
2 shape-only stubs `gpt_oss`, `hunyuan_large`), 6 quant schemes operational,
751 tests collected, 21 git tags placed (latest `v7-complete`).

The v5 → v7 release waves extended the M2 IR to cover ALiBi, parallel
residual blocks, BitNet ternary, GPT-OSS trained attention sinks, CLA
cross-layer KV pointers, Mamba-1 selective scan, the Jamba Mamba/attention
alternation, and the four v7 frontier-MoE patterns: DeepSeek-V4 hash routing
+ CSA/HCA, Qwen3-Next Gated DeltaNet, GLM-MoE-DSA, MiniMax-M2.

## What's in here

- **`api/`** — the IR layer (~6,100 LoC). 26 functional ops in `api/ops.py`,
  22 frozen dataclasses in `api/specs.py`, plus building-block `nn.Module`s
  (`norm`, `rope`, `kvcache`, `quant`, `attention`, `feedforward`, `block`,
  `embedding`, `ssm`). This is the parameter space every family slots into.
- **`models/`** — per-family `config.py` + `layer.py` + `layer.md`. Each
  family builds `DecoderBlockSpec` objects from a HuggingFace config and
  assembles a `DecoderBlock` from the api primitives. Production HF weight
  loading is provided per family via `load_<family>_layer`.
- **`research/`** — five v1/v2/v3 survey reports (model census, layer
  decomposition, IHV op-sets, quantization, KV-cache + attention), plus the
  `00-evolution.md` 2022→2026 timeline and the v3-era coverage extensions
  (`06-ocr-vlm-extensions.md`, `08-recent-releases-2026q2.md`).
- **`docs/`** — design specs (v1/v2/v3), implementation plans (M1, M2),
  post-rollout reference (`API-REFERENCE.md`), and the `PROJECT-SUMMARY.md`
  scoreboard. See `docs/issues/` for the audit + critique trail.
- **`tests/`** — 751 collected tests. `tests/api/` exercises the IR
  primitives; `tests/models/<family>/` runs per-family shape + isolation +
  numerical + KV-cache + weight-loader gates.

## Reading path

Recommended doc order for new readers:

1. This README — project surface
2. `docs/PROJECT-SUMMARY.md` — full scoreboard + artifact inventory
3. `docs/API-REFERENCE.md` — post-rollout reference for `api/*.py`
4. `research/00-evolution.md` — 2022→2026 SLM architectural timeline
5. `research/01-model-census.v3.md` — 148 SLMs × 19 axes
6. `models/qwen3/layer.md` — anchor family worked example
7. `models/gemma4/layer.md` — most architecturally divergent landed family
8. `docs/runtimes/openvino-gpu.md` — Intel GPU (Arc 140V) vertical slice with
   op-coverage audit and family classification (GREEN/YELLOW/RED)

## Release history

- **M1** (Qwen3 anchor) — kickoff gate passing at atol=5e-4 with
  max_abs_diff ≈ 5e-7.
- **B0.5 / B0.6** (Gemma 4 IR + corrections) — IR-divergence drifts caught
  end-to-end; T17 now passes at atol=5e-4 with max_abs_diff 7.6e-6
  (layer-0) and 3.8e-6 (layer-4).
- **B1 → B7** (M2 rollout) — Llama-dense families, Granite μP, Phi family,
  MiniCPM-3 MLA, OLMo 2 POST-norm, Gemma 2/3, Llama 4 Scout, Ministral
  per-layer SWA, DeepSeek V2/V3/V3.2, Mixtral / Qwen3-MoE / OLMoE,
  Mamba-2 + Granite-4-H SSM hybrid.
- **B8 / B9 / B10** — OCR-LM (GOT-OCR 2.0, Qwen2.5-VL, DeepSeek-OCR-2),
  audio-LM (Moshi, Voxtral), quantization "support 6" (AWQ + GGUF Q4_K_M
  + FP8 E4M3 + MXFP4 + LiteRT W4A8 + IQ2/AQLM stubs).
- **v5** — ALiBi (MPT), parallel residual (Falcon-7B), BitNet ternary,
  Hunyuan-Large CLA, GPT-OSS trained attention sinks.
- **v6** — Mamba-1 selective scan, Jamba (Mamba-1 + attention alternation),
  BitNet ternary `QDType`.
- **v7** — frontier-MoE wave: DeepSeek-V4 (hash routing + CSA/HCA shape),
  Qwen3-Next (Gated DeltaNet), GLM-MoE-DSA, MiniMax-M2.
- **OpenVINO GPU vertical slice (2026-06-15)** — Qwen3-0.6B verified
  end-to-end on Intel Arc 140V iGPU; argmax + top-5 bit-exact vs torch
  fp32 reference. 49.7 tok/s fp16. See `docs/runtimes/openvino-gpu.md`.

Full timeline browsable via `git log --oneline` (`5068bf7` bootstrap →
`ee84053` HEAD). Detailed milestone-by-milestone state is in
`docs/PROJECT-SUMMARY.md`.

## Design + plan documents

- `docs/superpowers/specs/2026-06-04-llm-layers-design.v2.md` — M1 design
- `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md` — M2 / B0.5 design
  (superseded by `docs/API-REFERENCE.md` post-rollout)
- `docs/superpowers/plans/2026-06-05-llm-layers-m1-qwen3.md` — M1 implementation plan
- `docs/superpowers/plans/2026-06-06-llm-layers-m2-rollout.md` — M2 / B0.5 → B10 plan
  (superseded by the actual rollout)

## Contributing

Per-family work follows the source-first rule: read `modeling_<family>.py`
in HF Transformers BEFORE writing the IR spec. Every architectural claim in
`models/<family>/layer.md` cites a `modeling_*.py:<line>` reference. See
`docs/issues/` for the critique-trail discipline.

## License

Apache 2.0 (project code). Model weights remain under their respective
HuggingFace licenses (Apache 2.0 for Qwen3-0.6B; Gemma terms for Gemma 4 E2B;
upstream HF licenses for every other family).
