"""V4 — onnx_port CPU smoke on novel hyperparameters.

Emit ONNX model from the alt shape, validate structural claims, run via ORT
CPU EP, compare to torch_port.

Structural claims:
  - At least one SimplifiedLayerNormalization op (default-domain RMSNorm fast path)
  - At least one GroupQueryAttention op (com.microsoft contrib)
  - At least one node with domain == "com.microsoft"
  - ir_version == 10
  - ORT loads successfully and CPUExecutionProvider is in providers

Numerical:
  - mean_rel error vs torch_port fp32 < 5e-4
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from runtimes.onnx_port import emit_block
from runtimes.torch_port import run_block

from ._alt_setup import alt_setup, weight_shapes_from_params, B, S, D_MODEL


_MEAN_REL_TOL = 5e-4
_MODEL_PATH = Path(__file__).resolve().parent / "_v4_alt_block.onnx"


def test_v4_onnx_cpu_smoke(capsys) -> None:
    setup = alt_setup()
    weight_shapes = weight_shapes_from_params(setup.params)

    # Emit
    model = emit_block(
        setup.graph,
        weight_shapes,
        input_shape=tuple(setup.inputs["input"].shape),
        position_ids_shape=tuple(setup.inputs["position_ids"].shape),
    )
    onnx.save(model, _MODEL_PATH)

    # Reload from disk so we exercise the on-disk path too.
    on_disk = onnx.load(str(_MODEL_PATH))

    # ---- Structural checks --------------------------------------------------
    node_count = len(on_disk.graph.node)
    op_types: dict[str, int] = {}
    ms_node_count = 0
    sln_count = 0
    gqa_count = 0
    for n in on_disk.graph.node:
        op_types[n.op_type] = op_types.get(n.op_type, 0) + 1
        if n.domain == "com.microsoft":
            ms_node_count += 1
        if n.op_type == "SimplifiedLayerNormalization":
            sln_count += 1
        if n.op_type == "GroupQueryAttention":
            gqa_count += 1

    assert sln_count >= 1, f"no SimplifiedLayerNormalization in graph; op_types={op_types}"
    assert gqa_count >= 1, f"no GroupQueryAttention in graph; op_types={op_types}"
    assert ms_node_count >= 1, f"no com.microsoft-domain nodes; op_types={op_types}"
    assert on_disk.ir_version == 10, f"ir_version={on_disk.ir_version}, expected 10"

    # ---- ORT load + provider check -----------------------------------------
    sess = ort.InferenceSession(str(_MODEL_PATH), providers=["CPUExecutionProvider"])
    loaded_providers = sess.get_providers()
    assert "CPUExecutionProvider" in loaded_providers, (
        f"CPUExecutionProvider not loaded; got {loaded_providers}"
    )

    feed: dict[str, np.ndarray] = {
        "input": setup.inputs["input"].numpy(),
        "position_ids": setup.inputs["position_ids"].numpy(),
    }
    for node_name, weights in setup.params.items():
        for slot, t in weights.items():
            feed[f"{node_name}.{slot}"] = t.numpy()

    out_names = [o.name for o in sess.get_outputs()]
    out = sess.run(out_names, feed)[0]

    # ---- Torch reference + parity ------------------------------------------
    expected = run_block(setup.graph, setup.params, setup.inputs).detach().numpy().astype(np.float32)
    assert out.shape == expected.shape

    denom = np.maximum(np.abs(expected), 1.0)
    rel = np.abs(out - expected) / denom
    mean_rel = float(rel.mean())
    max_rel = float(rel.max())

    with capsys.disabled():
        print(
            f"\n[V4 ONNX CPU smoke (alt shape)]"
            f"\n  out.shape            = {out.shape}"
            f"\n  graph nodes          = {node_count}  ops={op_types}"
            f"\n  SimplifiedLayerNorm  = {sln_count}"
            f"\n  GroupQueryAttention  = {gqa_count}"
            f"\n  com.microsoft nodes  = {ms_node_count}"
            f"\n  ir_version           = {on_disk.ir_version}"
            f"\n  providers loaded     = {loaded_providers}"
            f"\n  mean_rel vs torch    = {mean_rel:.4e}"
            f"\n  max_rel  vs torch    = {max_rel:.4e}"
        )

    assert mean_rel < _MEAN_REL_TOL, (
        f"ONNX CPU parity failed: mean_rel={mean_rel:.4e} >= {_MEAN_REL_TOL:.0e}"
    )
