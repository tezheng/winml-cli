"""Walk a BlockGraph and emit an ``ov.Model``.

Mirrors ``runtimes.torch_port.block_runner.run_block`` semantics:

- The RoPE Node is a structural pass-through. When an attention Node's
  immediate predecessor is a RoPE Node, the RoPE spec is hoisted into the
  attention sub-graph builder so RoPE is applied to Q/K internally.
- Weight parameters are exposed as graph inputs named ``"{node_name}.{slot}"``
  (e.g. ``"attn.w_q"``), matched against the param dict at infer time.
- The hidden state is exposed as an input named ``"input"`` and
  ``position_ids`` as an input named ``"position_ids"``.

The graph is constructed with static shapes (B, S, D, etc. taken from
``weight_shapes`` and an explicit ``input_shape``). Static shapes maximise GPU
kernel selection and let the fusion passes (RMSFusion, RoPEFusion) match.
"""
from __future__ import annotations

from typing import Any

import openvino as ov
from openvino import PartialShape, Type
from openvino import opset16 as opset

from components import (
    AddSpec,
    BlockGraph,
    GQASpec,
    NormStandardSpec,
    RoPESpec,
    SwiGLUSpec,
)

from .attention import build_gqa
from .ffn import build_swiglu
from .misc import build_add
from .norm import build_rms_norm


_DEFAULT_DTYPE = Type.f16


def _find_rope_predecessor(graph: BlockGraph, node_name: str) -> RoPESpec | None:
    """If a Node's immediate predecessor is a RoPE node, return that spec."""
    by_name = {n.name: n for n in graph.nodes}
    target = by_name[node_name]
    for inp in target.inputs:
        if inp == "input":
            continue
        pred = by_name.get(inp)
        if pred is not None and isinstance(pred.spec, RoPESpec):
            return pred.spec
    return None


def lower_block(
    graph: BlockGraph,
    weight_shapes: dict[str, dict[str, tuple[int, ...]]],
    *,
    input_shape: tuple[int, int, int] = (1, 8, 2048),  # [B, S, D_model]
    dtype: Type = _DEFAULT_DTYPE,
) -> ov.Model:
    """Build an ``ov.Model`` corresponding to ``graph``.

    Args:
        graph:         The BlockGraph DAG to lower.
        weight_shapes: ``{node_name: {weight_slot: shape}}`` — used to size
                       graph parameters for each weight (no values yet).
        input_shape:   Static [B, S, D_model] for the block input parameter.
        dtype:         Element type for activations and weights (default f16).

    Returns:
        An ``ov.Model`` with parameters
        ``["input", "position_ids", *flattened weight params]`` and a single
        result corresponding to ``graph.output``.
    """
    B, S, _D = input_shape

    # ---- Block-level parameters ---------------------------------------------
    input_param = opset.parameter(PartialShape(list(input_shape)), dtype, name="input")
    pos_ids_param = opset.parameter(
        PartialShape([B, S]), Type.i64, name="position_ids",
    )

    env: dict[str, Any] = {"input": input_param}
    parameters: list[Any] = [input_param, pos_ids_param]

    nodes_by_name = {n.name: n for n in graph.nodes}

    for node in graph.nodes:
        spec = node.spec
        sources = [env[name] for name in node.inputs]

        if isinstance(spec, NormStandardSpec):
            (src,) = sources
            w_shape = weight_shapes[node.name]["weight"]
            w_param = opset.parameter(
                PartialShape(list(w_shape)), dtype, name=f"{node.name}.weight",
            )
            parameters.append(w_param)
            env[node.name] = build_rms_norm(spec, src, w_param)

        elif isinstance(spec, RoPESpec):
            # Structural pass-through (same as torch_port).
            (src,) = sources
            env[node.name] = src

        elif isinstance(spec, GQASpec):
            (src,) = sources
            ws = weight_shapes[node.name]
            w_q = opset.parameter(PartialShape(list(ws["w_q"])), dtype, name=f"{node.name}.w_q")
            w_k = opset.parameter(PartialShape(list(ws["w_k"])), dtype, name=f"{node.name}.w_k")
            w_v = opset.parameter(PartialShape(list(ws["w_v"])), dtype, name=f"{node.name}.w_v")
            w_o = opset.parameter(PartialShape(list(ws["w_o"])), dtype, name=f"{node.name}.w_o")
            parameters.extend([w_q, w_k, w_v, w_o])

            rope_spec = _find_rope_predecessor(graph, node.name)
            env[node.name] = build_gqa(
                spec, src, w_q, w_k, w_v, w_o,
                rope_spec=rope_spec,
                position_ids=pos_ids_param if rope_spec is not None else None,
                batch_size=B, seq_len=S,
            )

        elif isinstance(spec, SwiGLUSpec):
            (src,) = sources
            ws = weight_shapes[node.name]
            w_gate = opset.parameter(PartialShape(list(ws["w_gate"])), dtype, name=f"{node.name}.w_gate")
            w_up = opset.parameter(PartialShape(list(ws["w_up"])), dtype, name=f"{node.name}.w_up")
            w_down = opset.parameter(PartialShape(list(ws["w_down"])), dtype, name=f"{node.name}.w_down")
            parameters.extend([w_gate, w_up, w_down])
            env[node.name] = build_swiglu(spec, src, w_gate, w_up, w_down)

        elif isinstance(spec, AddSpec):
            env[node.name] = build_add(*sources)

        else:
            raise NotImplementedError(
                f"openvino_port has no lowering for {type(spec).__name__} (node '{node.name}')."
            )

    result = opset.result(env[graph.output])
    return ov.Model(results=[result], parameters=parameters, name="m0_block")


__all__ = ["lower_block"]
