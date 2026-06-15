"""Topological executor for BlockGraph — PyTorch reference port.

Walks ``graph.nodes`` in order, dispatching to the matching ``*_forward``
function by spec ``isinstance``. Tensor values flow through an ``env`` dict
keyed by node name; the reserved name ``"input"`` is seeded from ``inputs``.

RoPE handling (M0): the RoPE node in the graph is treated as a structural
pass-through (its output tensor is the same as its input). When an
attention node consumes the RoPE node, the runner detects that the input is
a RoPE-typed node and forwards the RoPE spec plus ``position_ids`` into
``gqa_forward`` so the attention op applies RoPE internally to its
projected Q/K. This matches Hugging Face wiring (RoPE applied between Q/K
projection and SDPA) without needing multi-output nodes.
"""
from __future__ import annotations

import torch

from components import (
    AddSpec,
    BlockGraph,
    GQASpec,
    NormStandardSpec,
    RoPESpec,
    SwiGLUSpec,
)

from .attention import gqa_forward
from .ffn import swiglu_forward
from .misc import add_forward
from .norm import rms_norm_forward


_POSITION_IDS_KEY = "position_ids"


def run_block(
    graph: BlockGraph,
    params: dict[str, dict[str, torch.Tensor]],
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Execute a BlockGraph forward.

    Args:
        graph:  BlockGraph (topologically ordered ``nodes`` tuple).
        params: ``{node_name: {weight_slot: tensor}}`` for every parameterised
                node. RoPE and Add nodes need no entries.
        inputs: Must contain ``"input"`` (the block input tensor). If any RoPE
                node is present, must also contain ``"position_ids"``.

    Returns:
        The tensor at ``graph.output``.
    """
    env: dict[str, torch.Tensor] = dict(inputs)

    # Index nodes by name so attention can look up its predecessor's spec.
    nodes_by_name: dict[str, object] = {n.name: n for n in graph.nodes}

    for node in graph.nodes:
        spec = node.spec

        if isinstance(spec, NormStandardSpec):
            (src,) = node.inputs
            x = env[src]
            w = params[node.name]["weight"]
            env[node.name] = rms_norm_forward(spec, x, w)

        elif isinstance(spec, RoPESpec):
            # Structural pass-through. The attention node consuming this RoPE
            # node detects it via predecessor lookup and applies RoPE to its
            # projected Q/K. We keep the activation flowing so downstream
            # shape contracts hold.
            (src,) = node.inputs
            env[node.name] = env[src]

        elif isinstance(spec, GQASpec):
            (src,) = node.inputs
            # If the immediate predecessor is a RoPE node, lift its spec out
            # so gqa_forward can apply rotary embeddings.
            pred_node = nodes_by_name.get(src)
            rope_spec: RoPESpec | None = None
            if pred_node is not None and isinstance(pred_node.spec, RoPESpec):
                rope_spec = pred_node.spec
            x = env[src]
            p = params[node.name]
            position_ids = env.get(_POSITION_IDS_KEY) if rope_spec is not None else None
            env[node.name] = gqa_forward(
                spec,
                x,
                p["w_q"], p["w_k"], p["w_v"], p["w_o"],
                rope_spec=rope_spec,
                position_ids=position_ids,
            )

        elif isinstance(spec, SwiGLUSpec):
            (src,) = node.inputs
            x = env[src]
            p = params[node.name]
            env[node.name] = swiglu_forward(spec, x, p["w_gate"], p["w_up"], p["w_down"])

        elif isinstance(spec, AddSpec):
            input_tensors = [env[name] for name in node.inputs]
            env[node.name] = add_forward(*input_tensors)

        else:
            raise NotImplementedError(
                f"torch_port has no forward for spec type {type(spec).__name__} "
                f"(node '{node.name}')."
            )

    return env[graph.output]


__all__ = ["run_block"]
