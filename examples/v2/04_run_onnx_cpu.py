"""Build the canonical M0 block, emit it as ONNX, and run on ORT CPU.

Demonstrates: `runtimes.onnx_port.emit_block` walks the same `BlockGraph`,
producing an ONNX ``ModelProto`` that uses Microsoft contrib ops
(``com.microsoft.GroupQueryAttention``, ``com.microsoft.RotaryEmbedding``,
``SimplifiedLayerNormalization``) plus stock ONNX ops. The output is compared
against the torch_port golden reference at fp32.

Skips cleanly if onnx/onnxruntime aren't installed, or if golden_m0.pt is
missing.
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
_MODEL_PATH = _REPO_ROOT / "examples" / "v2" / "_sample_block.onnx"


def _have(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False


def main() -> int:
    if not (_have("onnx") and _have("onnxruntime")):
        print("onnx and/or onnxruntime not installed — skipping.")
        return 0
    if not _GOLDEN_PATH.exists():
        print(
            f"golden file missing: {_GOLDEN_PATH}\n"
            "Run `.venv\\Scripts\\python.exe tests\\parity\\_save_golden.py` first."
        )
        return 0

    import onnx
    import onnxruntime as ort

    from runtimes.onnx_port import emit_block
    from tests.parity._canonical import _canonical_m0_setup

    setup = _canonical_m0_setup()
    golden = torch.load(_GOLDEN_PATH, weights_only=False)
    params = golden["params"]
    inputs = golden["inputs"]
    expected = golden["output"].detach().numpy().astype(np.float32)

    weight_shapes = {
        node_name: {slot: tuple(t.shape) for slot, t in slots.items()}
        for node_name, slots in params.items()
    }

    # --- Emit ONNX -----------------------------------------------------------
    model = emit_block(
        setup.graph,
        weight_shapes,
        input_shape=tuple(inputs["input"].shape),
        position_ids_shape=tuple(inputs["position_ids"].shape),
    )
    onnx.save(model, _MODEL_PATH)

    op_counts: dict[str, int] = {}
    ms_ops: list[str] = []
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.domain == "com.microsoft":
            ms_ops.append(node.op_type)

    print(f"ONNX model: {len(model.graph.node)} nodes, "
          f"{len(model.graph.initializer)} initializers, "
          f"{_MODEL_PATH.stat().st_size:,} bytes")
    print(f"  saved to: {_MODEL_PATH}")
    top = sorted(op_counts.items(), key=lambda kv: -kv[1])[:8]
    print("  top ops:  " + ", ".join(f"{k}={v}" for k, v in top))
    print(f"  com.microsoft ops present: {sorted(set(ms_ops))}")

    # --- Run on ORT CPU ------------------------------------------------------
    sess = ort.InferenceSession(str(_MODEL_PATH), providers=["CPUExecutionProvider"])
    print(f"ORT providers: {sess.get_providers()}")

    feed: dict[str, np.ndarray] = {
        "input": inputs["input"].numpy(),
        "position_ids": inputs["position_ids"].numpy(),
    }
    for node_name, weights in params.items():
        for slot, t in weights.items():
            feed[f"{node_name}.{slot}"] = t.detach().numpy()

    # Warmup, then timed runs.
    sess.run(["r1"], feed)
    n_runs = 10
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = sess.run(["r1"], feed)[0]
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / n_runs

    # --- Parity metrics -------------------------------------------------------
    abs_diff = np.abs(out - expected)
    out_scale = float(np.abs(expected).max())
    rel_max = float(abs_diff.max() / out_scale)
    mean_abs = float(abs_diff.mean())

    print("\n[ORT CPU vs torch_port golden]")
    print(f"  shape    : {out.shape}")
    print(f"  expected absmax = {out_scale:.4e}")
    print(f"  max_abs  = {abs_diff.max():.4e}")
    print(f"  mean_abs = {mean_abs:.4e}   (gate: < 1.0)")
    print(f"  max_rel  = {rel_max:.4e}   (gate: < 1e-3)")
    print(f"\n[latency] {n_runs} runs, average {avg_ms:.2f} ms/run (ORT CPU, fp32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Expected output (representative):
#
# ONNX model: ~30 nodes, several initializers, ~few KB
#   saved to: ...\examples\v2\_sample_block.onnx
#   top ops:  MatMul=N, Add=M, SimplifiedLayerNormalization=2, ...
#   com.microsoft ops present: ['GroupQueryAttention']
# ORT providers: ['CPUExecutionProvider']
#
# [ORT CPU vs torch_port golden]
#   shape    : (1, 8, 2048)
#   expected absmax = 1.4862e+05
#   max_abs  = O(10)
#   mean_abs = small
#   max_rel  = ~1e-4   (gate: < 1e-3)
#
# [latency] 10 runs, average <few> ms/run (ORT CPU, fp32)
