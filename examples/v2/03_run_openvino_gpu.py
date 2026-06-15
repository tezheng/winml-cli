"""Build the canonical M0 block, lower it to ``ov.Model``, and run on Arc 140V GPU.

Demonstrates: a single `BlockGraph` flows through `runtimes.openvino_port.lower_block`
to produce an OpenVINO model that can be compiled for the Intel iGPU. The
output is compared against the torch_port golden reference.

Precision note: the canonical setup uses unscaled Normal(0,1) weights through a
2048-wide matmul stack and produces ~1.5e5 outputs — past fp16 range. We
therefore run fp32 on the GPU for parity (matches the parity gate in
``tests/parity/test_openvino_vs_torch.py``).

Skips cleanly if no GPU is available or if the golden_m0.pt artefact is missing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `components`, `runtimes`, and `tests` importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

_GOLDEN_PATH = _REPO_ROOT / "tests" / "parity" / "golden_m0.pt"


def _have_openvino() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except Exception:
        return False


def _gpu_available() -> bool:
    import openvino as ov
    try:
        return "GPU" in ov.Core().available_devices
    except Exception:
        return False


def _weight_shapes(params):
    return {
        node_name: {slot: tuple(t.shape) for slot, t in ws.items()}
        for node_name, ws in params.items()
    }


def main() -> int:
    if not _have_openvino():
        print("openvino not installed — skipping.")
        return 0
    if not _gpu_available():
        print("OpenVINO GPU device not available — skipping.")
        return 0
    if not _GOLDEN_PATH.exists():
        print(
            f"golden file missing: {_GOLDEN_PATH}\n"
            "Run `.venv\\Scripts\\python.exe tests\\parity\\_save_golden.py` first."
        )
        return 0

    import openvino as ov
    from openvino import Type

    from runtimes.openvino_port import lower_block
    from tests.parity._canonical import _canonical_m0_setup

    setup = _canonical_m0_setup()
    golden = torch.load(_GOLDEN_PATH, weights_only=False)
    params = golden["params"]
    inputs = golden["inputs"]
    expected = golden["output"].detach().numpy().astype(np.float32)

    # --- Lower to ov.Model and compile for GPU at fp32 -----------------------
    B, S, D = inputs["input"].shape
    model = lower_block(
        setup.graph,
        _weight_shapes(params),
        input_shape=(int(B), int(S), int(D)),
        dtype=Type.f32,
    )
    print(f"ov.Model: {len(model.get_ordered_ops())} ops, "
          f"{len(model.get_parameters())} parameters")

    core = ov.Core()
    print(f"Available devices: {core.available_devices}")

    compiled = core.compile_model(
        model,
        device_name="GPU",
        config={
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f32",
        },
    )
    exec_devices = compiled.get_property("EXECUTION_DEVICES")
    print(f"EXECUTION_DEVICES: {exec_devices}")
    assert "GPU" in str(exec_devices), f"GPU not engaged: {exec_devices}"

    # --- Bind inputs ----------------------------------------------------------
    feed: dict[str, np.ndarray] = {
        "input": inputs["input"].numpy().astype(np.float32),
        "position_ids": inputs["position_ids"].numpy().astype(np.int64),
    }
    for node_name, weights in params.items():
        for slot, tensor in weights.items():
            feed[f"{node_name}.{slot}"] = tensor.detach().numpy().astype(np.float32)

    # --- Inference ------------------------------------------------------------
    request = compiled.create_infer_request()
    # Warmup.
    request.infer(feed)

    n_runs = 10
    t0 = time.perf_counter()
    for _ in range(n_runs):
        request.infer(feed)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / n_runs

    result = request.infer(feed)
    out = np.asarray(list(result.values())[0]).astype(np.float32)

    # --- Parity metrics -------------------------------------------------------
    denom = np.maximum(np.abs(expected), 1.0)
    rel = np.abs(out - expected) / denom
    abs_diff = np.abs(out - expected)

    print("\n[OV GPU vs torch_port golden]")
    print(f"  shape    : {out.shape}")
    print(f"  expected absmax = {np.abs(expected).max():.4e}")
    print(f"  ov absmax       = {np.abs(out).max():.4e}")
    print(f"  max_abs  = {abs_diff.max():.4e}")
    print(f"  mean_abs = {abs_diff.mean():.4e}")
    print(f"  max_rel  = {rel.max():.4e}")
    print(f"  mean_rel = {rel.mean():.4e}   (parity gate: < 5e-4)")
    print(f"\n[latency] {n_runs} runs, average {avg_ms:.2f} ms/run (Arc iGPU, fp32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Expected output (representative):
#
# ov.Model: <N> ops, <P> parameters
# Available devices: ['CPU', 'GPU', ...]
# EXECUTION_DEVICES: ['GPU.0']
#
# [OV GPU vs torch_port golden]
#   shape    : (1, 8, 2048)
#   expected absmax = 1.4862e+05
#   ov absmax       = ~1.5e+05
#   max_abs  = small
#   mean_abs = small
#   max_rel  = <varies, isolated small denominators amplify>
#   mean_rel = ~1e-4 to 4e-4   (parity gate: < 5e-4)
#
# [latency] 10 runs, average <few> ms/run (Arc iGPU, fp32)
