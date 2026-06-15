"""V5 — parallel_block topology smoke on torch_port.

Verifies a different topology factory than the canonical/alt setup which both
use pre_norm_block. parallel_block is the Falcon/Cohere-style 3-way residual:

    x -> norm -> {attn, ffn} -> 3-way add(x, attn, ffn)

We construct the graph with the same alt hyperparameters, build matching
params, run it on torch_port, and verify shape + finiteness.
"""
from __future__ import annotations

import torch

from components import (
    GQASpec,
    NormStandardSpec,
    RoPESpec,
    SwiGLUSpec,
    parallel_block,
)
from runtimes.torch_port import run_block

from ._alt_setup import B, S, H_Q, H_KV, DH, D_FF, D_MODEL, SEED


def test_v5_parallel_block_topology_torch(capsys) -> None:
    torch.manual_seed(SEED)

    attn = GQASpec(num_heads=H_Q, head_dim=DH, num_kv_heads=H_KV)
    rope = RoPESpec(theta=10000.0)
    ffn = SwiGLUSpec(intermediate_size=D_FF)
    norm0 = NormStandardSpec(eps=1e-6)

    graph = parallel_block(
        block_norm=norm0,
        attention=attn,
        positional_encoding=rope,
        ffn=ffn,
    )

    # Inspect topology: parallel_block has 5 nodes: n0, rope, attn, ffn, r0
    node_names = [n.name for n in graph.nodes]
    assert node_names == ["n0", "rope", "attn", "ffn", "r0"], (
        f"unexpected parallel_block node names: {node_names}"
    )
    assert graph.output == "r0", f"unexpected output: {graph.output}"

    # Verify r0 is a 3-way add taking (input, attn, ffn) — distinguishes
    # this from pre_norm_block where r0 is 2-way (input, attn).
    r0 = next(n for n in graph.nodes if n.name == "r0")
    assert r0.inputs == ("input", "attn", "ffn"), (
        f"r0 should be 3-way add(input,attn,ffn); got {r0.inputs}"
    )

    # Build params. parallel_block has only ONE norm (n0) — no n1.
    s = 1.0 / (D_MODEL ** 0.5)
    params: dict[str, dict[str, torch.Tensor]] = {
        "n0":   {"weight": torch.randn(D_MODEL, dtype=torch.float32)},
        "attn": {
            "w_q": torch.randn(H_Q * DH, D_MODEL, dtype=torch.float32) * s,
            "w_k": torch.randn(H_KV * DH, D_MODEL, dtype=torch.float32) * s,
            "w_v": torch.randn(H_KV * DH, D_MODEL, dtype=torch.float32) * s,
            "w_o": torch.randn(D_MODEL, H_Q * DH, dtype=torch.float32) * s,
        },
        "ffn": {
            "w_gate": torch.randn(D_FF, D_MODEL, dtype=torch.float32) * s,
            "w_up":   torch.randn(D_FF, D_MODEL, dtype=torch.float32) * s,
            "w_down": torch.randn(D_MODEL, D_FF, dtype=torch.float32) * s,
        },
    }

    inputs: dict[str, torch.Tensor] = {
        "input": torch.randn(B, S, D_MODEL, dtype=torch.float32),
        "position_ids": torch.arange(S, dtype=torch.int64).unsqueeze(0).repeat(B, 1),
    }

    out = run_block(graph, params, inputs)

    assert tuple(out.shape) == (B, S, D_MODEL), (
        f"unexpected output shape: {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "output contains NaN/Inf"

    with capsys.disabled():
        print(
            f"\n[V5 parallel_block on torch_port]"
            f"\n  topology       = {node_names} -> output={graph.output}"
            f"\n  r0 inputs      = {r0.inputs}  (3-way add)"
            f"\n  out.shape      = {tuple(out.shape)}"
            f"\n  finite         = True"
            f"\n  out abs mean   = {out.abs().mean().item():.4e}"
            f"  out abs max = {out.abs().max().item():.4e}"
        )
