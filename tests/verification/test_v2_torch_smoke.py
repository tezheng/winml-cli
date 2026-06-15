"""V2 — torch_port smoke on novel hyperparameters (B=2, S=16, H_q=8, H_kv=4, Dh=64, D_ff=256)."""
from __future__ import annotations

import torch

from runtimes.torch_port import run_block

from ._alt_setup import alt_setup, B, S, D_MODEL


def test_v2_torch_shape_and_finite_and_deterministic() -> None:
    setup = alt_setup()
    out = run_block(setup.graph, setup.params, setup.inputs)

    # Shape
    expected_shape = (B, S, D_MODEL)
    assert tuple(out.shape) == expected_shape, (
        f"unexpected output shape: got {tuple(out.shape)} want {expected_shape}"
    )

    # Finite
    assert torch.isfinite(out).all(), "output contains NaN/Inf"

    # Determinism: rebuild the setup from the same seed and re-run; expect
    # bit-exact equality (CPU fp32, no nondeterministic kernels).
    setup2 = alt_setup()
    out2 = run_block(setup2.graph, setup2.params, setup2.inputs)
    assert torch.equal(out, out2), "two consecutive runs differ — nondeterministic!"

    print(
        f"\n[V2 torch smoke]"
        f"\n  out.shape = {tuple(out.shape)}"
        f"\n  finite    = True"
        f"\n  bit-exact = True"
        f"\n  out abs mean = {out.abs().mean().item():.4e}"
        f"  out abs max = {out.abs().max().item():.4e}"
    )
