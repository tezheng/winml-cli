"""V3 — openvino_port GPU smoke on novel hyperparameters.

Same alt block as V2 but lowered to an ov.Model and run on the iGPU.

Asserts:
  - EXECUTION_DEVICES contains 'GPU'
  - INFERENCE_PRECISION_HINT round-trip matches what we requested
  - mean relative error vs torch_port fp32 < 5e-4 (fp32 GPU path)
  - At least one ScaledDotProductAttention / SDPA op is fused into a single
    ExecutionNode in the runtime model. The architecture claim ("SDPA fuses
    on Xe2 GPU") is empirically observed on the fp16-precision path —
    matches what tests/parity/test_openvino_vs_torch.py also reports.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("openvino")
import openvino as ov  # noqa: E402
from openvino import Type  # noqa: E402

from runtimes.openvino_port import lower_block  # noqa: E402
from runtimes.torch_port import run_block  # noqa: E402

from ._alt_setup import alt_setup, weight_shapes_from_params, B, S, D_MODEL  # noqa: E402


_MEAN_REL_TOL = 5e-4


def _gpu_available() -> bool:
    try:
        return "GPU" in ov.Core().available_devices
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="OpenVINO GPU not available"
)


def _compile_and_check_sdpa(graph, weight_shapes, dtype, hint):
    model = lower_block(
        graph, weight_shapes,
        input_shape=(B, S, D_MODEL), dtype=dtype,
    )
    core = ov.Core()
    compiled = core.compile_model(
        model,
        device_name="GPU",
        config={
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": hint,
        },
    )
    rt = compiled.get_runtime_model()
    counts: dict[str, int] = {}
    fused: list[str] = []
    sdpa_found = False
    for op in rt.get_ordered_ops():
        t = op.get_type_name()
        n = op.get_friendly_name()
        counts[t] = counts.get(t, 0) + 1
        if "SDPA" in n or "ScaledDotProductAttention" in n:
            sdpa_found = True
            fused.append(f"{t}({n})")
    return compiled, sdpa_found, fused, counts


def test_v3_openvino_gpu_smoke(capsys) -> None:
    setup = alt_setup()
    weight_shapes = weight_shapes_from_params(setup.params)

    # Torch reference
    expected = run_block(setup.graph, setup.params, setup.inputs).detach().numpy().astype(np.float32)

    # --- fp32 GPU path for parity gate ---
    compiled_fp32, sdpa_fp32, fused_fp32, counts_fp32 = _compile_and_check_sdpa(
        setup.graph, weight_shapes, Type.f32, "f32"
    )
    exec_devices = compiled_fp32.get_property("EXECUTION_DEVICES")
    assert "GPU" in str(exec_devices), f"GPU not used: {exec_devices}"

    prec_hint = compiled_fp32.get_property("INFERENCE_PRECISION_HINT")
    prec_str = str(prec_hint)
    assert "f32" in prec_str.lower() or "float32" in prec_str.lower(), (
        f"INFERENCE_PRECISION_HINT not fp32: {prec_str}"
    )

    feed: dict[str, np.ndarray] = {
        "input": setup.inputs["input"].numpy().astype(np.float32),
        "position_ids": setup.inputs["position_ids"].numpy().astype(np.int64),
    }
    for node_name, weights in setup.params.items():
        for slot, t in weights.items():
            feed[f"{node_name}.{slot}"] = t.numpy().astype(np.float32)

    out = np.asarray(list(compiled_fp32.create_infer_request().infer(feed).values())[0]).astype(np.float32)
    assert out.shape == expected.shape, f"shape mismatch: ov={out.shape} torch={expected.shape}"

    denom = np.maximum(np.abs(expected), 1.0)
    rel = np.abs(out - expected) / denom
    mean_rel = float(rel.mean())
    max_rel = float(rel.max())

    # --- fp16 GPU path: where SDPA actually fuses on Xe2 ---
    compiled_fp16, sdpa_fp16, fused_fp16, counts_fp16 = _compile_and_check_sdpa(
        setup.graph, weight_shapes, Type.f16, "f16"
    )

    with capsys.disabled():
        print(
            f"\n[V3 OV-GPU smoke (alt shape)]"
            f"\n  out.shape          = {out.shape}"
            f"\n  EXECUTION_DEVICES  = {exec_devices}"
            f"\n  INFERENCE_PRECISION= {prec_str}"
            f"\n  mean_rel vs torch  = {mean_rel:.4e}"
            f"\n  max_rel  vs torch  = {max_rel:.4e}"
            f"\n  fp32 runtime ops   = {counts_fp32}"
            f"\n  fp32 SDPA fused?   = {sdpa_fp32}  ({fused_fp32[:2]})"
            f"\n  fp16 runtime ops   = {counts_fp16}"
            f"\n  fp16 SDPA fused?   = {sdpa_fp16}  ({fused_fp16[:2]})"
        )

    assert mean_rel < _MEAN_REL_TOL, (
        f"OV-GPU(fp32) parity failed: mean_rel={mean_rel:.4e} >= {_MEAN_REL_TOL:.0e}"
    )
    # Architecture claim: SDPA fuses on Xe2 (empirically the fp16 path).
    assert sdpa_fp16, (
        f"Expected ScaledDotProductAttention fusion in the fp16 GPU runtime "
        f"model; not found. fp16 counts={counts_fp16}"
    )
