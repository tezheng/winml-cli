"""torch_port self-parity: determinism + shape on the canonical M0 graph.

This test produces the *golden* reference tensor that every other runtime
port (openvino_port, onnx_port, ...) compares against. If this test ever
changes output, all golden-comparison tests must be re-baselined.
"""
from __future__ import annotations

import hashlib

import pytest
import torch

from runtimes.torch_port import run_block

from ._canonical import D_MODEL, B, S, _canonical_m0_setup


@pytest.fixture(scope="module")
def setup():
    """Build the canonical (graph, params, inputs) triple once per module."""
    return _canonical_m0_setup()


def test_run_block_deterministic(setup) -> None:
    """Two consecutive invocations must produce bit-identical tensors."""
    out1 = run_block(setup.graph, setup.params, setup.inputs)
    out2 = run_block(setup.graph, setup.params, setup.inputs)
    # Bit-exact, not allclose — the runner has no randomness, and both runs
    # share the same fp32 reduction order.
    assert torch.equal(out1, out2), "run_block produced non-deterministic output"


def test_run_block_output_shape(setup) -> None:
    """Block output must preserve [B, S, D_model]."""
    out = run_block(setup.graph, setup.params, setup.inputs)
    assert out.shape == (B, S, D_MODEL), f"expected (B,S,D)=({B},{S},{D_MODEL}), got {tuple(out.shape)}"


def test_run_block_finite(setup) -> None:
    """No NaN/Inf in the canonical-setup output (catches obvious math errors)."""
    out = run_block(setup.graph, setup.params, setup.inputs)
    assert torch.isfinite(out).all(), "block output contains non-finite values"


def test_block_graph_node_names(setup) -> None:
    """Pin the node names emitted by pre_norm_block for M0 — guards against
    silent reshuffling that would break other ports' param dicts."""
    names = [n.name for n in setup.graph.nodes]
    assert names == ["n0", "rope", "attn", "r0", "n1", "ffn", "r1"], (
        f"pre_norm_block emitted unexpected node names: {names}"
    )
    assert setup.graph.output == "r1"


def test_golden_summary_print(setup, capsys) -> None:
    """Emit a small fingerprint line so the test log records the golden
    output's sha256 for cross-machine verification."""
    out = run_block(setup.graph, setup.params, setup.inputs)
    # Hash the raw fp32 bytes (contiguous) — stable across runs on the same arch.
    h = hashlib.sha256(out.detach().contiguous().numpy().tobytes()).hexdigest()
    with capsys.disabled():
        print(
            f"\nOK: torch_port deterministic; shape={tuple(out.shape)} "
            f"mean={out.mean().item():.4e} std={out.std().item():.4e} "
            f"sha256={h}"
        )
