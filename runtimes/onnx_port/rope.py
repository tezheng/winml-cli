"""RoPE ONNX emit — standalone ``com.microsoft.RotaryEmbedding``.

In the block_emitter wiring, RoPE is fused into ``GroupQueryAttention`` via
``do_rotary=1`` and is NOT emitted as a separate node. This module exists for
completeness (test scaffolding, future ports that want a standalone op).

Both this op and the GQA fused path consume the same ``cos_cache``/``sin_cache``
initializers, shaped ``[max_seq_len, head_dim // 2]``. The cache construction
math lives in ``attention.build_rope_cos_sin_initializers`` so a single source
serves both call sites.
"""
from __future__ import annotations

from onnx import NodeProto, helper

from components import RoPESpec


def build_rope_nodes(
    spec: RoPESpec,
    input_name: str,
    position_ids_name: str,
    cos_cache_name: str,
    sin_cache_name: str,
    out_name: str,
) -> list[NodeProto]:
    """Emit a single RotaryEmbedding node.

    Args:
        spec:               RoPESpec; only theta is consumed (cos/sin cache must
                            be precomputed and bound via initializer).
        input_name:         Tensor name shaped ``[B, H, S, Dh]`` (or
                            ``[B, S, H*Dh]``).
        position_ids_name:  Int64 positions, shape ``[B, S]``.
        cos_cache_name:     Tensor name of the precomputed cos cache,
                            ``[max_seq_len, Dh/2]``.
        sin_cache_name:     Tensor name of the sin cache.
        out_name:           Tensor name for the rotated output.

    Returns:
        Single-element list.
    """
    if spec.partial_rotary_factor != 1.0:
        raise NotImplementedError(
            "M0 only supports full rotary (partial_rotary_factor == 1.0); "
            f"got {spec.partial_rotary_factor}."
        )
    node = helper.make_node(
        "RotaryEmbedding",
        inputs=[input_name, position_ids_name, cos_cache_name, sin_cache_name],
        outputs=[out_name],
        name=f"{out_name}/RotaryEmbedding",
        domain="com.microsoft",
        interleaved=0,  # Llama-style: first half real, second half imag
    )
    return [node]


__all__ = ["build_rope_nodes"]
