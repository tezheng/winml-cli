# `examples/v2/` — components API samples

Runnable demos of the v2 components API at `components/` and the three runtime
ports under `runtimes/`. New to the repo? Open these in order.

All samples are runnable from the repo root:

```powershell
.venv\Scripts\python.exe examples\v2\<filename>.py
```

Each runnable sample either does its work or prints a clear skip message
(e.g. "OpenVINO GPU device not available"). No silent failure.

## Prerequisites

Before running samples that compare against torch ground truth, generate the
golden tensor once:

```powershell
.venv\Scripts\python.exe tests\parity\_save_golden.py
```

This writes `tests/parity/golden_m0.pt`. Samples 02, 03, and 04 reuse it.

## The samples

### Runnable (executes a layer end-to-end)

| File | One-liner |
| --- | --- |
| `01_build_block.py` | Construct a `BlockGraph` with `pre_norm_block` and print its DAG. No ML deps. |
| `02_run_torch.py` | Run the canonical M0 block on the PyTorch reference port; print shape, mean, std, sha256, and 10-run latency. |
| `03_run_openvino_gpu.py` | Lower the same M0 block to `ov.Model`, compile for the Intel Arc 140V GPU (`EXECUTION_DEVICES == ['GPU.0']`), compare against the torch golden, and print parity metrics. |
| `04_run_onnx_cpu.py` | Emit the same M0 block as ONNX (com.microsoft contrib ops), run on ORT CPU EP, compare against the torch golden. Writes `_sample_block.onnx` (gitignored via `*.onnx`). |

### Spec-only definitions (no runtime work)

These show how the same spec primitives express different model families.
Each file constructs a `BlockGraph` for the model layer and prints a summary;
none of them run the layer.

| File | Model | What it demonstrates |
| --- | --- | --- |
| `define_qwen3.py` | Qwen3-0.6B | Plain `GQASpec` + per-head-`Dh` `qk_norm` + RoPE `theta=1_000_000` (Qwen3-specific). |
| `define_qwen3_5_layer.py` | Qwen3.5-35B-A3B `full_attention` layer | M1 fields: `NormZeroCenteredSpec`, `GQASpec(output_gate="sigmoid")`, `MRoPEInterleavedSpec`. |
| `define_gemma4_layer.py` | Gemma 4 E2B local (sliding) layer | `sandwich_block` topology, `NormZeroCenteredSpec`, `GQASpec(v_norm, qk_norm_fixed_scale, sliding_window)`, `RoPESpec(partial_rotary_factor=0.25, partial_rotary_kind="proportional")`, `GeGLUSpec`. |
| `define_gpt_oss_layer.py` | GPT-OSS-20B sliding layer | `GQASinksSpec` (per-head learned sink logit), `ClampedSwiGLUSpec(clamp=7.0, alpha=1.702)` as the per-expert inner activation. The MoE wrapping (128 experts, top-4) is M1+ work. |

## How the runnable samples relate

All three runtimes (`02`, `03`, `04`) consume the **same** `BlockGraph` produced
by `tests/parity/_canonical.py::_canonical_m0_setup()`. The torch port output
is the golden reference; the OV and ONNX ports are compared against it. The
parity gates baked into `tests/parity/` are:

- ORT CPU: `max_rel < 1e-3`, `mean_abs < 1.0` (vs torch fp32)
- OV GPU (fp32 precision): `mean_rel < 5e-4`

The samples print the actual metrics so you can see the numbers; the test files
under `tests/parity/` are the authoritative gates.

## Reading order for new engineers

1. `01_build_block.py` — see that a block is data.
2. `define_qwen3.py` — see real model parameters slotted into the same primitives.
3. `02_run_torch.py` — see the torch reference execute the canonical M0 fixture.
4. `03_run_openvino_gpu.py` and `04_run_onnx_cpu.py` — see the same fixture
   produce matching outputs on two other backends.
5. `define_gemma4_layer.py`, `define_qwen3_5_layer.py`,
   `define_gpt_oss_layer.py` — see the spec extension surface used by other families.
