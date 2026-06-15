"""BlockGraph -> onnx.ModelProto emitter.

Mirrors ``runtimes/torch_port/block_runner.run_block`` walking semantics. RoPE
nodes in the graph are structural pass-throughs: when an attention node
consumes a RoPE node, the emitter folds the RoPE spec into
``GroupQueryAttention`` (via ``do_rotary=1`` + cos/sin cache initializers).
This matches torch_port's wiring where RoPE is applied internally inside
``gqa_forward`` rather than as a standalone op on the Q/K tensors.

Weights are exposed as graph *inputs* (not initializers) so the caller binds
them at session-run time. Cos/sin RoPE caches ARE baked in as initializers
because they depend only on (theta, head_dim, max_seq_len) — fixed at emit time.
"""
from __future__ import annotations

from typing import Any

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

from components import (
    AddSpec,
    BlockGraph,
    GQASpec,
    NormStandardSpec,
    RoPESpec,
    SwiGLUSpec,
)

from .attention import build_gqa_nodes, build_rope_cos_sin_initializers
from .ffn import build_swiglu_nodes
from .misc import build_add_node
from .norm import build_rms_norm_nodes


_BLOCK_INPUT = "input"
_POSITION_IDS = "position_ids"
_SEQLENS_K = "seqlens_k"
_TOTAL_SEQLEN = "total_sequence_length"


def emit_block(
    graph: BlockGraph,
    weight_shapes: dict[str, dict[str, tuple[int, ...]]],
    input_shape: tuple[int, ...],
    position_ids_shape: tuple[int, ...],
    *,
    max_seq_len: int | None = None,
) -> onnx.ModelProto:
    """Emit an ONNX ModelProto from a BlockGraph.

    Args:
        graph:              BlockGraph (topologically ordered).
        weight_shapes:      ``{node_name: {slot: shape_tuple}}`` for every
                            parameterised node. Same key layout as torch_port
                            params (e.g. attn -> w_q/w_k/w_v/w_o).
        input_shape:        Shape of the block input tensor, ``[B, S, D]``.
        position_ids_shape: Shape of position_ids, ``[B, S]``. Required even
                            when no RoPE node exists (the graph input slot
                            stays present for consistency).
        max_seq_len:        Max sequence length for the RoPE cos/sin cache.
                            Defaults to ``input_shape[1]`` (== S).

    Returns:
        onnx.ModelProto ready to ``onnx.save`` or pass into ORT.
    """
    if max_seq_len is None:
        max_seq_len = int(input_shape[1])

    nodes_by_name: dict[str, Any] = {n.name: n for n in graph.nodes}

    # Identify the attention node so we can build cos/sin caches with the right Dh.
    attn_node = next((n for n in graph.nodes if isinstance(n.spec, GQASpec)), None)
    rope_node = next((n for n in graph.nodes if isinstance(n.spec, RoPESpec)), None)

    initializers: list[TensorProto] = []
    if attn_node is not None and rope_node is not None:
        # Validate: the RoPE node must be a direct predecessor of attention.
        if attn_node.inputs[0] != rope_node.name:
            raise ValueError(
                "Emitter expects the RoPE node to feed directly into the GQA "
                f"node; attn '{attn_node.name}' input is '{attn_node.inputs[0]}'."
            )
        gqa_spec: GQASpec = attn_node.spec
        initializers.extend(build_rope_cos_sin_initializers(
            rope_node.spec, gqa_spec.head_dim, max_seq_len,
            cos_name="rope_cos_cache",
            sin_name="rope_sin_cache",
        ))

    # seqlens_k and total_sequence_length: for fresh-prompt (no past KV), they
    # are constant per (B, S). Bake them in as initializers so the caller
    # doesn't have to bind them.
    B = int(input_shape[0])
    S = int(input_shape[1])
    if attn_node is not None:
        initializers.append(numpy_helper.from_array(
            np.full((B,), S - 1, dtype=np.int32), name=_SEQLENS_K))
        initializers.append(numpy_helper.from_array(
            np.array([S], dtype=np.int32), name=_TOTAL_SEQLEN))

    # Build graph inputs: block input, position_ids, all weight tensors.
    graph_inputs = [
        helper.make_tensor_value_info(_BLOCK_INPUT, TensorProto.FLOAT, list(input_shape)),
        helper.make_tensor_value_info(_POSITION_IDS, TensorProto.INT64, list(position_ids_shape)),
    ]
    for node_name in sorted(weight_shapes.keys()):
        for slot, shape in weight_shapes[node_name].items():
            graph_inputs.append(helper.make_tensor_value_info(
                f"{node_name}.{slot}", TensorProto.FLOAT, list(shape),
            ))

    # Walk the DAG.
    env: dict[str, str] = {_BLOCK_INPUT: _BLOCK_INPUT}
    onnx_nodes = []

    for node in graph.nodes:
        spec = node.spec
        out_name = node.name

        if isinstance(spec, NormStandardSpec):
            (src,) = node.inputs
            w_name = f"{node.name}.weight"
            onnx_nodes.extend(build_rms_norm_nodes(spec, env[src], w_name, out_name))

        elif isinstance(spec, RoPESpec):
            # Structural pass-through (matches torch_port semantics).
            (src,) = node.inputs
            onnx_nodes.append(helper.make_node(
                "Identity", [env[src]], [out_name], name=f"{out_name}/rope_passthrough"))

        elif isinstance(spec, GQASpec):
            (src,) = node.inputs
            pred = nodes_by_name.get(src)
            rope_spec_for_attn: RoPESpec | None = None
            if pred is not None and isinstance(pred.spec, RoPESpec):
                rope_spec_for_attn = pred.spec
            onnx_nodes.extend(build_gqa_nodes(
                spec,
                env[src],
                f"{node.name}.w_q",
                f"{node.name}.w_k",
                f"{node.name}.w_v",
                f"{node.name}.w_o",
                _SEQLENS_K,
                _TOTAL_SEQLEN,
                "rope_cos_cache",
                "rope_sin_cache",
                out_name,
                rope_spec=rope_spec_for_attn,
                name_scope=node.name,
            ))

        elif isinstance(spec, SwiGLUSpec):
            (src,) = node.inputs
            onnx_nodes.extend(build_swiglu_nodes(
                spec,
                env[src],
                f"{node.name}.w_gate",
                f"{node.name}.w_up",
                f"{node.name}.w_down",
                out_name,
                name_scope=node.name,
            ))

        elif isinstance(spec, AddSpec):
            onnx_nodes.extend(build_add_node(
                [env[name] for name in node.inputs],
                out_name,
                name_scope=node.name,
            ))

        else:
            raise NotImplementedError(
                f"onnx_port has no emitter for spec type {type(spec).__name__} "
                f"(node '{node.name}')."
            )

        env[node.name] = out_name

    graph_outputs = [
        helper.make_tensor_value_info(graph.output, TensorProto.FLOAT, list(input_shape)),
    ]

    graph_proto = helper.make_graph(
        onnx_nodes,
        "m0_block",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )

    opset_imports = [
        helper.make_opsetid("", 17),                # standard opset
        helper.make_opsetid("com.microsoft", 1),    # contrib ops
    ]
    model = helper.make_model(graph_proto, opset_imports=opset_imports)
    # ORT 1.26 supports up to IR 10/11; onnx 1.21's default IR_VERSION is 13.
    model.ir_version = 10
    model.producer_name = "llm-layers/runtimes/onnx_port"
    return model


__all__ = ["emit_block"]
