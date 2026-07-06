# HWMonitor: GPU Utilization Monitoring

Status: implemented on branch `feat/update-pkg-deps`, as of 2026-06-24. Extends the existing PDH-based `HWMonitor` (CPU/RAM/NPU) with a third utilization series — GPU — rendered live alongside CPU and NPU during `winml perf --monitor`. Windows-only (PDH / `pdh.dll`).

**See also**: the NPU half of this monitor lives in the same modules; this document only covers the GPU additions and the research that shaped them. The chart/poller architecture it builds on is in [`src/winml/modelkit/session/monitor/_pdh.py`](../src/winml/modelkit/session/monitor/_pdh.py) and [`src/winml/modelkit/commands/_live_chart.py`](../src/winml/modelkit/commands/_live_chart.py).

## Executive summary

- The `perf --monitor` command already drew a live two-line chart (NPU green, CPU cyan) using `plotext` inside a Rich `Live` panel, fed by a background `PdhPoller` thread reading Windows PDH performance counters. This change adds a **GPU** series (yellow).
- The hard part is not plumbing — it is **what "GPU %" even means**. An NPU exposes a single `Compute` engine, so one counter is the whole story. A GPU exposes *many* engines (`3D`, `Compute`, `Copy`, `Video*`, `Security`, `Timer`...), and DirectML inference does not land on a fixed one. So GPU % must be **aggregated across engines**, and the aggregation choice is a real design decision, not an implementation detail.
- Research (five parallel agents, sources cited below) converged on: **Task Manager reports the busiest single engine — a `max` across engines, capped at 100% — not a sum or average.** We reproduce that.
- To honour the repo's **No-Hardcoded-Logic** cardinal rule, we never name an engine type. We enumerate *every* engine type each GPU exposes, register a utilization counter for each, and `max` across them at poll time. DirectML's preferred engine (`3D` on most consumer GPUs) is captured automatically without being named.
- All metrics flow into the existing `HWMonitor.to_dict()` → benchmark JSON automatically, and into the live display via two new `update()` kwargs.

## Background: the existing monitor pipeline

Three layers, unchanged in shape, extended in content:

1. **`PdhPoller`** ([`_pdh.py`](../src/winml/modelkit/session/monitor/_pdh.py)) — a daemon thread that opens one PDH query, registers named counters, and polls them every 200 ms (100 ms for the live display) into thread-safe sample lists. Already collected `cpu_pct`, `ram_committed_bytes`, and NPU `utilization_pct` / memory.
2. **`HWMonitor`** ([`hw_monitor.py`](../src/winml/modelkit/session/monitor/hw_monitor.py)) — a context-manager facade over `PdhPoller` exposing `mean_*`/`peak_*` properties, time-series sample lists for the chart, and a JSON-serializable `to_dict()`.
3. **`LiveMonitorDisplay`** ([`_live_chart.py`](../src/winml/modelkit/commands/_live_chart.py)) — renders the `plotext` chart + a 3-row status block inside a Rich `Live` panel, called once per inference iteration from `_run_monitored_loop` in [`perf.py`](../src/winml/modelkit/commands/perf.py).

GPU support threads a new sample series through all three.

## OS-level API: how we call Windows PDH

There is **no third-party dependency** for hardware monitoring. We talk to the Windows **Performance Data Helper** (PDH) subsystem directly through `ctypes`, binding `pdh.dll` once at module load:

```python
import ctypes, ctypes.wintypes as wintypes
_pdh = ctypes.windll.pdh   # _pdh.py:53, pdh_adapters.py:46
```

This is the same OS facility Task Manager, `typeperf`, and PerfMon read from; the GPU/NPU engine data ultimately originates in the WDDM graphics kernel (VidSch/VidMm) and is exposed as the **"GPU Engine"** and **"GPU Process Memory"** counter sets. A WDDM 2.0+ driver is required.

### Counter query lifecycle (`PdhQuery` in `_pdh.py`)

Five `pdh.dll` entry points, wrapped by the `PdhQuery` class:

| `pdh.dll` function | Wrapper | Purpose |
| --- | --- | --- |
| `PdhOpenQueryW` | `PdhQuery.open` (`_pdh.py:117`) | Allocate a query handle (`HQUERY`). |
| `PdhAddEnglishCounterW` | `PdhQuery.add_counter` (`_pdh.py:138`) | Register one counter path against the query, returning a counter handle. |
| `PdhCollectQueryData` | `PdhQuery.prime` / `_collect_once` (`_pdh.py:152,156`) | Sample **all** registered counters at once. |
| `PdhGetFormattedCounterValue` | `_collect_once` (`_pdh.py:168,176`) | Read one counter's formatted value from the last collect. |
| `PdhCloseQuery` | `PdhQuery.close` (`_pdh.py:216`) | Release the query handle and all counters. |

Two OS-level details that the wrapper exists to manage:

1. **English (locale-independent) counter paths.** We use `PdhAddEnglish­CounterW`, not `PdhAddCounterW`. The English variant resolves counter paths by their canonical US-English names (`\GPU Engine(...)\Utilization Percentage`, `\Processor(_Total)\% Processor Time`) regardless of the machine's display language, so paths built in code work on localized Windows. This is why no counter name is ever localized in our source.

2. **Rate counters need two samples.** `Utilization Percentage` and `% Processor Time` are rate-derived: a single `PdhCollectQueryData` yields no valid value (the formatter needs a previous + current sample to compute the delta over time). `PdhQuery.prime()` performs the throwaway first collect; `collect()` then retries until every counter yields non-`None` or a timeout elapses. Microsoft guidance is ≥1 s between samples for accuracy; our background poller's 100–200 ms interval trades some precision for chart responsiveness, which is acceptable for a live "is it busy" signal.

### Reading typed counter values via ctypes structures

`PdhGetFormattedCounterValue` writes into a caller-allocated union struct. We mirror the two variants we need as `ctypes.Structure` subclasses (`_pdh.py:58–70`) and pass a format flag telling PDH which field to populate:

```python
_PDH_FMT_DOUBLE = 0x00000200   # percentages → struct.doubleValue
_PDH_FMT_LARGE  = 0x00000400   # bytes / nanoseconds → struct.largeValue (int64)
```

Each call also receives a `DWORD CStatus` out-field; we treat a value as valid only when both the call status and `CStatus` are `ERROR_SUCCESS` (0), otherwise the sample becomes `None` and is skipped. `_PdhFmtLarge` includes an explicit 4-byte `_padding` field to match the native struct's 8-byte alignment of the `LONGLONG` — getting this wrong silently corrupts the read.

### Instance enumeration (`enumerate_adapters` in `pdh_adapters.py`)

GPU/NPU adapters are **discovered**, never hardcoded. `PdhEnumObjectItemsW` (`pdh_adapters.py:108,126`) lists every instance of the "GPU Engine" counter object. It uses the classic Win32 **two-call sizing idiom**:

```python
# 1st call: pass NULL buffers to learn required sizes (returns PDH_MORE_DATA)
# 2nd call: pass buffers of the returned size to receive the data
_PDH_MORE_DATA       = 0x800007D2   # expected status from the sizing call
_PERF_DETAIL_WIZARD  = 400          # detail level: include all instances
```

The instance names come back as a **REG_MULTI_SZ** buffer (NUL-separated, double-NUL terminated), parsed by `_parse_multi_sz` via `ctypes.wstring_at`. Each name (`pid_<PID>_luid_<HI>_<LO>_phys_<N>_eng_<N>_engtype_<TYPE>`) is split to group instances by LUID and engine type into `AdapterInfo`. `discover_gpu_luids()` then selects adapters exposing a `3D` engine (GPUs) versus the Compute-only NPU — a structural fingerprint, not a vendor/device-name match.

### Threading & cleanup

All PDH calls for the live monitor happen on a single daemon thread (`PdhPoller._poll_loop`); sample lists are guarded by a `threading.Lock`. The query handle is opened in `start()` and always closed in `stop()` (including a final collect to capture end-of-run running-time deltas), so no `pdh.dll` handle leaks even if the benchmark raises — verified by `TestMonitorExceptionSafety`.

## The core problem: aggregating a multi-engine GPU into one number

A Windows GPU Engine PDH instance is named per-process, per-engine:

```text
pid_<PID>_luid_<HI>_<LO>_phys_<N>_eng_<N>_engtype_<TYPE>
```

On the dev machine, enumerating our own process's GPU registers (one of two adapters shown):

```text
gpu_util_0x00000000_0x000138C0_3D
gpu_util_0x00000000_0x000138C0_Copy
gpu_util_0x00000000_0x000138C0_Compute 0
gpu_util_0x00000000_0x000138C0_Timer 0
gpu_util_0x00000000_0x000138C0_Security 1
gpu_util_0x00000000_0x000138C0_Video JPEG 0
gpu_util_0x00000000_0x000138C0_Video Processor
gpu_util_0x00000000_0x000138C0_Video Codec Engine
gpu_util_0x00000000_0x00015675_3D
```

Each engine reports its own `Utilization Percentage` (fraction of the interval that engine was busy). Three findings determined how we collapse these into one "GPU %":

### Finding 1 — Task Manager uses MAX, not SUM

> "we opted to pick the percentage utilization of the busiest engine as a representative of the overall GPU usage."
> — [GPUs in the Task Manager, DirectX Developer Blog](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)

Averaging was explicitly rejected (a 10-engine GPU saturating only `3D` would aggregate to a misleading 10%). Summing is wrong too: engines run in parallel, so summed per-engine utilization routinely **exceeds 100%**. PDH caps a *single* instance at 100% by default, but cross-instance sums are not bounded ([PdhGetFormattedCounterArray](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw), `PDH_FMT_NOCAP100`).

### Finding 2 — DirectML work is not on a fixed engine

DirectML dispatches ML operators as DX12 compute work on a DIRECT/COMPUTE queue. On most consumer GPUs (notably NVIDIA and Intel iGPUs) that surfaces under the **`3D`** engine, *not* `Compute`, because WDDM requires the node exposing the GPU's general shader cores to advertise the 3D capability ([Enumerating GPU nodes](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/enumerating-gpu-nodes); [DirectML EP docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)). AMD may route via dedicated compute (ACE) paths. **Conclusion: we cannot hardcode an engine type** — it differs by vendor and is not queryable.

### Finding 3 — PDH has no `_Total`, no auto-sum

Every GPU Engine instance carries a `pid_` token; there is no system-wide aggregate instance. Wildcard paths (`\GPU Engine(*)\...`) *expand* but never *sum* — a wildcard handle must be read with `PdhGetFormattedCounterArray`, which the existing `_collect_once` (singular `PdhGetFormattedCounterValue`) does not use. So we register **explicit per-engine counters** — mirroring the proven NPU path — and aggregate in Python. ([Microsoft Q&A: per-process GPU usage via PDH](https://learn.microsoft.com/en-us/answers/questions/5641645/how-to-get-the-special-process-gpu-usage-with-the))

### The resulting rule

```
GPU % = min(100, max(utilization of each registered GPU engine for this PID))
```

A pure function, hardware-independent, unit-testable on CI without a GPU. It naturally yields the NPU-equivalent single value too, so the same code shape serves both.

## Implementation

### 1. `_pdh.py` — discovery, registration, aggregation, poller

`discover_gpu_luids()` already existed in [`sysinfo/pdh_adapters.py`](../src/winml/modelkit/sysinfo/pdh_adapters.py) (adapters with a `3D` engine, excluding the Compute-only NPU). Three new pieces in `_pdh.py`:

- **`aggregate_gpu_utilization(values) -> float | None`** — the pure rule above. Ignores `None` entries (engines that returned no data this interval); returns `None` if every value was `None` (so the poll loop appends nothing rather than a spurious 0).

- **`add_gpu_engine_counters(query, gpu_luids, *, pid=None) -> list[str]`** — for each GPU LUID, iterates *every* engine type in the adapter's `engine_map` and registers a `\GPU Engine(...)\Utilization Percentage` counter named `gpu_util_<luid>_<engtype>`. Returns the names that registered successfully, for the poller to aggregate. No engine type is named in code.

- **`PdhPoller`** gains `_gpu_luids` / `_gpu_counter_names` / `_gpu_samples`; `start()` discovers GPUs and registers their counters into the *same* query (so one poll covers CPU+RAM+NPU+GPU); `_poll_loop()` maxes the GPU counter values via `aggregate_gpu_utilization` and appends. New properties: `gpu_luids`, `gpu_samples`, `mean_gpu_pct`, `peak_gpu_pct`, `gpu_sample_count`, and static `is_gpu_available()`.

**Why registration works before inference starts:** at `start()` the current process has not yet touched the GPU, so its per-PID engine instances do not exist. `PdhAddEnglishCounterW` nonetheless registers them as *deferred* instances and begins returning data once the instance appears (i.e. once inference runs). This is the identical mechanism the NPU path already relies on; confirmed on real hardware (counters registered for both adapters while idle).

### 2. `hw_monitor.py` — facade + JSON

Added `mean_gpu_pct` / `peak_gpu_pct` / `gpu_samples` properties and a `"gpu"` block in `to_dict()`:

```json
"gpu": { "mean_pct": 0.0, "peak_pct": 0.0, "sample_count": 0,
         "luids": ["0x00000000_0x000138C0", "0x00000000_0x00015675"] }
```

Because `BenchmarkResult` serializes `hw.to_dict()` verbatim, GPU metrics reach the benchmark JSON with no change to `perf.py`'s result path.

### 3. `_live_chart.py` — third series

`update()` and `_render_chart()` take optional `gpu_samples`; `_render_status()` takes `gpu_pct`. GPU plots in **yellow** (NPU green / CPU cyan / GPU yellow), the legend is now built dynamically so each series only appears when present, and the status row gains a `GPU: x.x%` cell:

```text
Utilization (██ NPU %  ██ CPU %  ██ GPU %)
...
NPU: 60.0% avg (60.0% now) | CPU: 12.0% | GPU: 44.0% | Sys Mem: ... | Device Mem: ...
```

### 4. `perf.py` — one call site

`_run_monitored_loop` passes `gpu_samples=hw.gpu_samples, gpu_pct=hw.mean_gpu_pct` into `display.update()`. That is the only edit outside the monitor/chart modules.

## Testing (TDD)

13 tests written first, watched fail, then made pass ([`tests/unit/session/test_ep_monitor.py`](../tests/unit/session/test_ep_monitor.py)):

- **`TestGpuUtilizationAggregation`** — the pure rule: max-across-engines, cap at 100, ignore `None`, all-`None`→`None`, empty→`None`. Hardware-independent (runs on any Windows CI, GPU or not).
- **`TestPdhPollerGpu`** — `is_gpu_available()` type; GPU property types; graceful degradation when `discover_gpu_luids()` returns `[]` (patched) → empty samples, zero metrics, CPU still collected.
- **`TestHWMonitorGpu`** — `gpu_*` properties accessible; `to_dict()` has a JSON-serializable `"gpu"` section.
- **`TestLiveMonitorDisplay`** (extended) — status row includes the GPU cell; `update()` accepts GPU kwargs without a live context.

Results: 13/13 new pass; full `test_ep_monitor.py` 76/76; commands suite 413 passed / 1 skipped; ruff clean. Real-hardware smoke test confirmed 2 GPUs discovered, counters registered, status renders `GPU: 44.0%`, valid JSON `gpu` section.

## Design tradeoffs & limitations

- **`max` per engine *type*, v1 simplification.** The fully faithful Task Manager algorithm is *sum within an engtype, then max across engtypes* — relevant only when one engine type has multiple physical nodes (`eng_0`, `eng_1`) simultaneously busy. The adapter `engine_map` keeps one representative node per engine type, so we effectively max across types. Since DirectML's `3D` node dominates inference, this matches Task Manager in the common case. Documented in a code comment; revisit if multi-node engines matter.
- **Time-busy, not saturation.** PDH `Utilization Percentage` (like NVML) measures the fraction of wall-clock time an engine was busy, not compute occupancy — it can read 100% while SM occupancy is far lower. For true intensity, pair with vendor tools (DCGM, etc.). Adequate for a "is the GPU working" live signal.
- **Per-process scope.** We monitor the current process. Idle process → empty GPU samples (identical to NPU behaviour). Live non-zero values require an actual GPU/DML inference run, which the unit tests do not exercise — verify visually on a real `perf --monitor` with a DML model.
- **`GPU Process Memory` deliberately omitted.** That PDH counter has a [documented over-reporting bug](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/gpu-process-memory-counters-report-wrong-value); we only add GPU *utilization*, not GPU memory.
- **Windows-only.** PDH is `pdh.dll`. Cross-platform GPU monitoring (psutil/NVML) is out of scope, consistent with the existing NPU monitor.

## Sources

- [GPUs in the Task Manager — DirectX Developer Blog](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/) — busiest-engine (max) semantics, >100% rationale.
- [Enumerating GPU nodes — Microsoft Learn](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/enumerating-gpu-nodes) — why general compute reports as the 3D engine.
- [DirectML Execution Provider — ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) — DML = DX12 on DIRECT/COMPUTE queue.
- [How to get per-process GPU usage with the PDH API — Microsoft Q&A](https://learn.microsoft.com/en-us/answers/questions/5641645/how-to-get-the-special-process-gpu-usage-with-the) — per-process instance naming, enumerate-and-aggregate.
- [PdhGetFormattedCounterArray / PDH_FMT_NOCAP100 — Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw) — wildcards expand but do not sum; 100% cap behaviour.
- [GPU Process Memory counters report wrong value — Microsoft Learn](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/gpu-process-memory-counters-report-wrong-value) — why GPU memory counter is omitted.
