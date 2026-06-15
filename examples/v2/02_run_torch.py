"""Build the canonical M0 block and run it end-to-end on the torch_port.

Demonstrates: the torch_port executor (`run_block`) walks a `BlockGraph`,
dispatches by spec type, and produces the fp32 reference output. This is the
ground truth all other ports compare against.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

# Make `components`, `runtimes`, and `tests` importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from runtimes.torch_port import run_block
from tests.parity._canonical import _canonical_m0_setup


def main() -> None:
    setup = _canonical_m0_setup()
    print(f"BlockGraph: {len(setup.graph.nodes)} nodes, output='{setup.graph.output}'")
    print(
        f"Inputs: input.shape={tuple(setup.inputs['input'].shape)}, "
        f"position_ids.shape={tuple(setup.inputs['position_ids'].shape)}"
    )
    n_params = sum(p.numel() for ws in setup.params.values() for p in ws.values())
    print(f"Params: {len(setup.params)} parameterised nodes, {n_params:,} total elements")

    # --- One reference forward -----------------------------------------------
    output = run_block(setup.graph, setup.params, setup.inputs)
    h = hashlib.sha256(output.detach().contiguous().numpy().tobytes()).hexdigest()

    print("\n[torch_port forward]")
    print(f"  shape   = {tuple(output.shape)}")
    print(f"  dtype   = {output.dtype}")
    print(f"  mean    = {output.mean().item():.4e}")
    print(f"  std     = {output.std().item():.4e}")
    print(f"  absmax  = {output.abs().max().item():.4e}")
    print(f"  sha256  = {h}")

    # --- Latency over 10 runs -------------------------------------------------
    n_runs = 10
    # Warmup once so the first-call JIT cost doesn't skew the average.
    _ = run_block(setup.graph, setup.params, setup.inputs)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = run_block(setup.graph, setup.params, setup.inputs)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / n_runs

    print(f"\n[latency] {n_runs} runs, average {avg_ms:.2f} ms/run (CPU, fp32)")


if __name__ == "__main__":
    main()


# Expected output (representative numbers; latency varies):
#
# BlockGraph: 7 nodes, output='r1'
# Inputs: input.shape=(1, 8, 2048), position_ids.shape=(1, 8)
# Params: 5 parameterised nodes, 8,394,752 total elements
#
# [torch_port forward]
#   shape   = (1, 8, 2048)
#   dtype   = torch.float32
#   mean    = ...
#   std     = ...
#   absmax  = ~1.5e+05  (synthetic randn weights — see runtimes/openvino_port/
#                        test note on fp16 saturation)
#   sha256  = <stable 64-char hex>
#
# [latency] 10 runs, average <few> ms/run (CPU, fp32)
