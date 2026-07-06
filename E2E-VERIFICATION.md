# End-to-End Verification — cc7a6650

Verification checklist for the squash commit `cc7a6650`
(unified-source EP refactor + P0 fixes + simplification pass).

**Scope**: exercise the CLI binary with real args on real inputs.
This is NOT unit testing (that's `uv run pytest tests/unit/`).
This is end-to-end: the `winml` binary, real DLLs, real ONNX files.

**Assumes**: `uv run winml <cmd>` invokes the built CLI, PowerShell shell.
For bash, substitute `$env:X = "y"` → `X=y`, `Remove-Item Env:X` → `unset X`.

**All flags/commands below verified against src at `cc7a6650`.**

---

## Fixtures

- `microsoft/resnet-50` — HF ID, downloadable on first use (~50 MB)
- `t5-small` — HF composite (encoder + decoder), validates B3/B6 (~250 MB)
- `./convnext/model_opt.onnx` — in-repo raw ONNX
- `./convnext/model_opt_qdq.onnx` — in-repo raw ONNX with QDQ nodes
- `./x64/Release/` — local dir with EP DLLs (set via `WINMLCLI_EP_PATH`)

Note: `winml compile -m <hf-id>` fails because `--model` is a `click.Path(exists=True)`.
Compile only against local .onnx paths.

## Timing expectations

| Command class | Cold cache | Warm cache | Timeout to set |
|---|---|---|---|
| `--help`, `--version` | <2s | <2s | 30s |
| `winml sys` / `sys --list-ep` | 5-15s | 3-8s | 60s |
| `winml perf` HF (resnet-50) | 30-300s (download) + 90-180s benchmark | 90-180s | 600s |
| `winml perf` HF (t5-small) | 300-600s (download) + 30-120s | 30-120s | 900s |
| `winml perf` raw ONNX | 20-60s | 20-60s | 180s |
| `winml compile` local ONNX | 30-90s | 30-90s | 300s |
| `winml build` local ONNX | 60-180s (may hit ENV before finish) | 60-180s | 600s |

## Expected noise (NOT regressions)

The following are ORT native warnings on stderr — expected, not regressions:

- `[W:onnxruntime:...InitProviderBridgeORT] provider bridge failed`
- `[W:onnxruntime:...] tried to load ... plugin, failed`
- `Init provider bridge failed`

These originate from `onnxruntime` C++ code, not from `src/winml/modelkit/`. Only
tracebacks originating from `src/winml/modelkit/...` should be treated as regressions.

---

## Minimum smoke (4 commands, ~5 minutes with warm cache)

```powershell
# 1. EP discovery + source tags
uv run winml sys --format json

# 2. Auto-resolve + T-01 typing + Option B pre-bench panels + live chart
uv run winml perf -m microsoft/resnet-50 --device auto --ep auto --monitor

# 3. Raw ONNX + op-tracing + QNNMonitor
uv run winml perf -m ./convnext/model_opt_qdq.onnx --ep qnn --device npu --op-tracing basic

# 4. Composite model (B3/B6)
uv run winml perf -m t5-small --device cpu
```

The `perf` cases exercise `_open_ep_monitor_or_exit` (D1),
`_pre_bench_kwargs_from_ep_device` (D2), and — for t5-small — the composite
dispatch fixed in B3/B6.

---

## Full CLI matrix

Top-level commands exposed by `cli.py::LazyGroup.list_commands`:
`sys`, `perf`, `compile`, `build`, `config`, `analyze`, `eval`, `hub`,
`inspect`, `optimize`, `quantize`, `export`, `expand_rules`.

Disabled: `run`, `serve` (in `_DISABLED_COMMANDS`; both must print an
actionable error and exit 2 — NEVER stack trace).

### `winml sys` — hardware + EP discovery

```powershell
uv run winml sys                          # baseline text
uv run winml sys --format json            # JSON output
uv run winml sys --format compact         # compact plain-text summary
uv run winml sys --list-ep                # EP inventory (hierarchical tree)
uv run winml sys --list-ep --format json  # EP inventory as JSON
uv run winml sys --list-device            # devices in preference order
uv run winml sys --verbose                # additional diagnostics
```

**Verify**:
- EP source tags render (Title/UPPER-cased in CLI display) as one of:
  `bundled`, `PyPI`, `NuGet`, `MSIX`, `Catalog`, `Directory`.
  Internal tags (lowercase) are `bundled`, `pypi`, `nuget`, `msix`, `winml-catalog`, `directory`.
  The pre-collapse forms `msix-microsoft` / `msix-workload` (or their Title-cased
  variants `MSIX-Microsoft` / `MSIX-Workload`) must NEVER appear.
- `--list-ep` renders each plugin EP source with `[status]` marker
  (`primary` / `shadowed` / `incompatible`), followed by indented `Version:` /
  `Path:` / `Devices:` blocks.
- Built-in EPs (CPU, DML, Azure) render `[primary] bundled` on the source line
  with a `Version:` block. Path is omitted (built-ins ship inside ORT).
- `--format json` and `--list-ep --format json` output must be parseable JSON.

**Expected `sys --format compact` output (PASS example)**:
```
Python: 3.11.15 (Windows)
torch: 2.11.0 | transformers: 5.8.1 | onnx: 1.21.0
QNN: N/A | OpenVINO: N/A
Export Ready: ONNX OK
```

**Expected `sys --list-ep` structure (PASS example)**:
```
Available Execution Providers
  OpenVINOExecutionProvider
    [primary]   PyPI      onnxruntime-ep-openvino 1.4.1
                Version: 1.4.1+f33af4f
                Path:    <venv>\Lib\site-packages\onnxruntime_ep_openvino\onnxruntime_providers_openvino_plugin.dll
                Devices:
                  NPU: Memory: 16.0 GB  |  Capabilities: FP16, INT8
                  GPU: Memory: 16.3 GB  |  Capabilities: FP32, BIN, FP16, INT8, MatMul, USM
                  CPU: Capabilities: BF16, FP32, FP16, INT8, BIN
    [shadowed]  Catalog   (catalog default)
                Version: ...
                Path:    C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_...\onnxruntime_providers_openvino_plugin.dll
                Devices:
                  ...
    [shadowed]  MSIX      WindowsWorkload.EP.Intel.OpenVINO.1.8 v1.8.61.0
                Version: ...
                Path:    C:\Program Files\WindowsApps\WindowsWorkload.EP.Intel.OpenVINO.1.8_...\onnxruntime_providers_openvino_plugin.dll
                Devices:
                  NPU: (no metadata published)
                  ...
  QNNExecutionProvider  [incompatible]
    [incompatible] PyPI   onnxruntime-qnn 2.1.1
                Version: 0.1.0
                Path:    <venv>\Lib\site-packages\onnxruntime_qnn\libs\amd64\onnxruntime_providers_qnn.dll
  CPUExecutionProvider
    [primary]   bundled
                Version: 1.24.5
                Devices:
                  CPU: (no metadata published)
  DmlExecutionProvider
    [primary]   bundled
                Version: 1.24.5
                Devices:
                  GPU: Memory: 128.0 MB
```

### `winml perf` — benchmark

```powershell
# HF ID + auto (add --monitor for live chart)
uv run winml perf -m microsoft/resnet-50 --device auto --ep auto --monitor

# Raw ONNX
uv run winml perf -m ./convnext/model_opt.onnx --device auto --ep auto
uv run winml perf -m ./convnext/model_opt_qdq.onnx --device auto

# EP@source pinning (verify D1 helper on both entry paths)
uv run winml perf -m microsoft/resnet-50 --ep qnn@pypi --device npu
uv run winml perf -m microsoft/resnet-50 --ep openvino@pypi --device cpu

# Op-tracing (--op-tracing requires basic|detail — bare flag fails at parse)
uv run winml perf -m ./convnext/model_opt_qdq.onnx --ep qnn --device npu --op-tracing basic
uv run winml perf -m ./convnext/model_opt.onnx --ep openvino --op-tracing basic
#   ^ MUST refuse with the Intel-wheel-lacks-CSV message

# Directory EP source
$env:WINMLCLI_EP_PATH = "./x64/Release"
uv run winml perf -m microsoft/resnet-50 --device auto
Remove-Item Env:WINMLCLI_EP_PATH

# GPU utilization chart (verify orange+ / bright_yellow patch)
uv run winml perf -m microsoft/resnet-50 --device gpu --monitor

# Throughput knobs
uv run winml perf -m microsoft/resnet-50 --iterations 100 --warmup 10 --batch-size 4
```

**Verify**:
- Two `rich.Panel`s at pre-bench (Model + Device) — rendered to **stderr**;
  use `2>&1` when piping.
- With `--monitor`: 4-row status (progress / NPU-CPU-GPU `now%/avg%` /
  memory 2-line / latency+throughput); GPU line uses `orange+` / `bright_yellow`.
- `EP:` line in the Device panel always renders as `<name>@<source>`
  (e.g. `openvino@pypi v1.4.1`), never as `(tuple, tuple)`.
- Pre-bench identity block matches across both perf entry paths (D1/D2).

**Expected pre-bench Device panel snippet (PASS example)**:
```
+------------------- Device -----------------+
| Device: gpu / Intel Iris Xe                |
| EP: openvino@pypi v1.4.1                   |
| EP DLL: <path>\openvino_provider.dll       |
+--------------------------------------------+
```

### `winml compile` — verify T-01 typing + list mode

```powershell
# Requires an existing local ONNX file (--model is validated with exists=True)
uv run winml compile -m ./convnext/model_opt.onnx --ep qnn --device npu
uv run winml compile -m ./convnext/model_opt.onnx --ep openvino --device cpu

# List available compilers for a device (works without --model)
uv run winml compile --list --device npu
uv run winml compile --list --device gpu

# @source ACCEPTED and silently stripped by winml compile (does NOT reject)
# Verify EP renders as bare 'qnn', NOT '('qnn', 'pypi')'
uv run winml compile -m ./convnext/model_opt.onnx --ep qnn@pypi --device npu
```

**Verify**:
- `EP:` line in compile output renders as bare short-name (e.g. `qnn`),
  never `('qnn', None)` or `('qnn', 'pypi')` — T-01 leak test for compile.
- `compile` accepts `@source` (silently strips) — `build` and `config` reject.
  The `<ep>@<source>` render is exclusive to `perf`'s pre-bench block.

**Expected `compile` output snippet (PASS example)**:
```
EP: qnn
Device: npu
Compiler: ort
```

### `winml build` — verify D3 consolidation

Build requires a config file (via `winml config` first). Uses `-o` for output.

```powershell
# 1. Generate a build config (config REQUIRES -m even when just generating)
uv run winml config -m ./convnext/model_opt.onnx --device cpu -o ./tmp/build.json

# 2. HF ID path (uses build/hf.py which delegates to build/common.py)
uv run winml build -c ./tmp/build.json -m microsoft/resnet-50 -o ./tmp/hf

# 3. Raw ONNX path (uses build/onnx.py which delegates to build/common.py)
uv run winml build -c ./tmp/build.json -m ./convnext/model_opt.onnx -o ./tmp/onnx

# 4. Composite path (validates B3/B6 — from_pretrained with ep_device threading)
uv run winml build -c ./tmp/build.json -m t5-small -o ./tmp/t5

# @source REJECTED by build with clean UsageError (not a stack trace)
uv run winml build -c ./tmp/build.json -m ./convnext/model_opt.onnx --ep qnn@pypi -o ./tmp/reject
```

**Verify**:
- Both hf and onnx paths delegate to `build/common.py::run_build_stages`
  (confirmed at `build/hf.py:31,232` and `build/onnx.py:24,138`).
- Stages [3]-[6] identical between hf and onnx invocations.
- Composite build reaches sub-model construction without `TypeError`.
- Required flags: `-c`, `-o`.
- `@source` rejection returns exit != 0 with text containing
  `does not yet support source pinning` — no traceback.

### `winml config` — generate build config

Config REQUIRES `-m <model>` (either HF ID or local .onnx path).

```powershell
uv run winml config -m ./convnext/model_opt.onnx --device npu -o ./tmp/config-npu.json
uv run winml config -m ./convnext/model_opt.onnx --device cpu --ep qnn -o ./tmp/config-cpu-qnn.json

# @source REJECTED by config with clean UsageError
uv run winml config -m ./convnext/model_opt.onnx --device cpu --ep qnn@pypi -o ./tmp/config-reject.json
```

**Verify**:
- `EP:` in stdout renders bare short-name (no tuple leak).
- `--ep qnn@pypi` rejected with actionable UsageError — no traceback.

### `winml analyze` — verify P0-A / P0-B / D4 fixes

```powershell
uv run winml analyze -m microsoft/resnet-50
uv run winml analyze -m ./convnext/model_opt.onnx
uv run winml analyze -m ./convnext/model_opt_qdq.onnx --device npu --ep qnn
```

**Verify** (all failure modes should be absent):
- No `ImportError` from removed `sysinfo.resolve_device` (P0-A).
- No `ImportError` from deleted `sysinfo.device.get_ep_device_map` (P0-B).
- No `NameError` from consolidated NPU checkers (D4).

### `winml eval` — evaluate a model

```powershell
uv run winml eval -m ./tmp/hf/model.onnx --dataset imagenet-1k --samples 100
uv run winml eval -m ./tmp/hf/model.onnx --task image-classification
```

### `winml hub` / `inspect` / `optimize` / `quantize` / `export` / `expand_rules`

Smoke-test each with `--help` (proves import hygiene after 5 deletions).

```powershell
uv run winml hub --help
uv run winml inspect --help
uv run winml inspect -m microsoft/resnet-50 --format json

uv run winml optimize --help
uv run winml quantize --help
uv run winml export --help
uv run winml expand_rules --help
```

**Verify**: no `ImportError` mentioning any of the 5 deleted modules
(`optracing`, `optimum_loader`, `hub_utils`, `config_generator`,
`pattern_matching`) in the import chain.

**Python import hygiene check** (fastest way to catch a stray reference):
```powershell
uv run python -c "from winml.modelkit.commands import hub, inspect, optimize, quantize, export, expand_rules, analyze; print('OK')"
uv run python -c "from winml.modelkit.build import hf, onnx, common; print('OK')"
uv run python -c "from winml.modelkit.models.winml import composite_model; print('OK')"
uv run python -c "from winml.modelkit.inference import engine; print('OK')"
uv run python -c "from winml.modelkit.session import ep_registry, ep_device; print('OK')"
```

---

## Composite model verification (B2 / B3 / B6 critical path)

These paths were broken pre-fix. Must succeed post-fix:

```powershell
# T5-small (composite: encoder + decoder)
uv run winml perf -m t5-small --device cpu

# Compile a composite sub-model
uv run winml compile -m ./tmp/t5/encoder.onnx --ep openvino --device cpu
```

**Verify**: none of the following appear in stderr:
- `TypeError: from_pretrained() got unexpected keyword argument 'device'`
- `TypeError: from_onnx() got unexpected keyword argument 'ep'`
- `NameError: name 'device' is not defined`
- `NameError: name 'ep' is not defined`

Any of the four means B2/B3 regressed.

**Composite export sanity** (unrelated to B2/B3/B6 but caught by t5-small perf):
- In perf output for `t5-small`, check `hierarchy_modules > 0` (composite
  dispatch reached sub-model construction — the actual B3/B6 fix scope).
- If perf reports `onnx_nodes: 0` followed by `Error: Benchmark failed:
  tuple index out of range`, that's a DOWNSTREAM composite ONNX export bug,
  NOT this session's B2/B3/B6 fix. Flag separately for the composite export
  pipeline owner — do NOT count as this commit's regression.

---

## Sanity: help + version

```powershell
uv run winml --help
uv run winml sys --help
uv run winml perf --help
uv run winml --version
```

---

## Output patterns to grep

Run each command with `2>&1 | Select-String -Pattern '<pattern>'` (PowerShell).
Pre-bench Panels emit to stderr, so `2>&1` is required for the perf checks.

| Pattern | Should appear | Meaning if hit / missed |
|---|---|---|
| `EP: (` | never | T-01 tuple leak in perf/compile/config/build output |
| `TypeError: from_pretrained` | never | B3 regression |
| `TypeError: from_onnx` | never | B2 regression |
| `NameError: name 'device'` | never | B2/B3/B6 regression |
| `(?i)msix-microsoft` or `(?i)msix-workload` | never | source-tag collapse not applied (case-insensitive — pre-collapse tags render as `MSIX-Microsoft` / `MSIX-Workload`) |
| `ImportError.*resolve_device` | never | P0-A regression |
| `sysinfo.device` in traceback | never | P0-B regression |
| `Op-tracing --ep openvino is not currently supported` | perf --ep openvino --op-tracing basic | OpenVINO stub refusal working |
| `bundled\|PyPI\|NuGet\|MSIX\|Catalog\|Directory` | sys --list-ep | source tag rendering correct (CLI display casing) |
| `does not yet support source pinning` | build/config --ep qnn@pypi | `_reject_ep_source` working |
| `onnx_nodes: 0` followed by `tuple index out of range` | never | composite ONNX export bug (downstream — flag separately) |

---

## Environmental failure triage

The following failures are **environmental** (hardware / EP availability),
**NOT** shipping regressions. Do not open bugs for these.

| Error text | Cause | Verdict |
|---|---|---|
| `No source for QNNExecutionProvider/npu exposed device class 'NPU'` | Host has no Qualcomm NPU (e.g. Intel AI Boost NPU instead) | ENV |
| `Could not deserialize by device xml header` | OpenVINO plugin internal error, model/device mismatch | ENV |
| `CompiledModel was not initialized` | OpenVINO EP failed to compile the graph for the target device | ENV |
| `DeviceNotFound` after registration | EP registered but no matching `OrtEpDevice` (see F-07) | ENV |
| `DeviceNotAvailableError` | Requested device class absent on host | ENV |
| `HF Hub connection error` / timeout during download | Network — retry or use cached fixture | ENV |
| `Access is denied` when reading DLL | Windows permissions / DLL locked by another process | ENV |
| `[W:onnxruntime:...] provider bridge failed` | ORT native warning, not a code regression | Expected noise |

If the error is from `src/winml/modelkit/...` with a Python traceback,
it is a **REGRESSION**, not ENV.

---

## When to run

- After any change to `commands/perf.py`, `session/ep_registry.py`,
  `session/ep_device.py`, `session/monitor/*`, `build/*`, `models/auto.py`,
  `models/winml/composite_model.py`, `inference/engine.py`
- Before pushing this branch to origin
- Not needed for test-only or docs-only changes

---

## Not covered here

- Unit-level: `uv run pytest tests/unit/ --tb=line -q`
- Hardware-specific validation (QNN NPU on Snapdragon, OpenVINO NPU on Intel):
  requires real hardware — those failures are environmental, not regressions
- Perf regression (throughput / latency deltas): use dedicated perf harness
- `winml run` / `winml serve`: disabled in `cli.py:_DISABLED_COMMANDS`; verify
  they exit with an actionable message rather than a stack trace
- Composite ONNX export completeness (`onnx_nodes` count): separate concern
  from this commit's B2/B3/B6 composite-dispatch fix

---

## Convert to script?

If this becomes routine, extract into `scripts/verify-e2e.ps1` at repo root
with each command wrapped in `try/catch` and a pass/fail summary line.
Keep this doc as the human-readable reference. Estimated one-time setup:
90 minutes; ongoing savings: ~15 minutes per verification pass.
