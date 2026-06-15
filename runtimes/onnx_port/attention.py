"""GQA ONNX emit — Q/K/V/O projections + fused ``com.microsoft.GroupQueryAttention``.

The contrib op takes the post-projection (q, k, v) tensors shaped
``[B, S, H*Dh]`` directly. RoPE is folded inside via ``do_rotary=1`` consuming
precomputed ``cos_cache`` / ``sin_cache`` initializers — both ``do_rotary=1``
and standalone ``com.microsoft.RotaryEmbedding`` + ``do_rotary=0`` produce the
same parity against torch_port on M0 (within ~1e-4 relative on the final block
output), so we use the fused path for graph simplicity.

Layout quirks worth noting:
  * ``F.linear(x, W)`` in torch is ``x @ W.T`` because torch stores weight as
    ``[out, in]``. ONNX ``MatMul`` does no transpose, so we transpose each
    weight to ``[in, out]`` before binding via a ``Transpose`` node.
  * ``head_size`` MUST be a multiple of 16 for the CPU GQA kernel. Canonical M0
    uses Dh=128, so this is fine.
  * ``seqlens_k`` (int32, shape ``[B]``) is the per-batch last-K-index
    (``S - 1`` for a fresh prompt with no past). ``total_sequence_length``
    (int32, shape ``[1]``) is the total K length including past — equal to ``S``
    when there's no past KV cache. Both are non-optional even when ``past_key``/
    ``past_value`` are empty.
  * ``cos_cache``/``sin_cache`` shape: ``[max_seq_len, head_dim // 2]``. We
    take the first half because torch_port builds
    ``emb = cat(freqs, freqs)`` where halves are equal.
"""
from __future__ import annotations

import numpy as np
from onnx import NodeProto, TensorProto, helper, numpy_helper

from components import GQASpec, RoPESpec


def _build_rope_cos_sin_numpy(
    theta: float,
    head_dim: int,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct cos/sin caches matching torch_port.rope.build_rope_cos_sin.

    The torch_port form does ``emb = cat(freqs, freqs)`` then cos/sin over the
    full Dh, but ContribOperators' RotaryEmbedding wants only the first half
    (``[max_seq_len, Dh/2]``). Because the first and second halves are equal,
    taking the first half is exact.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE; got {head_dim}.")
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    positions = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    assert cos.shape == (max_seq_len, half)
    return cos, sin


def build_rope_cos_sin_initializers(
    rope_spec: RoPESpec,
    head_dim: int,
    max_seq_len: int,
    cos_name: str,
    sin_name: str,
) -> list[TensorProto]:
    """Return cos/sin TensorProtos for ``graph.initializer``.

    Shape: ``[max_seq_len, head_dim/2]`` each (matches contrib op's expectation).
    """
    if rope_spec.partial_rotary_factor != 1.0:
        raise NotImplementedError(
            "M0 only supports partial_rotary_factor == 1.0; "
            f"got {rope_spec.partial_rotary_factor}."
        )
    cos, sin = _build_rope_cos_sin_numpy(rope_spec.theta, head_dim, max_seq_len)
    return [
        numpy_helper.from_array(cos, name=cos_name),
        numpy_helper.from_array(sin, name=sin_name),
    ]


def build_gqa_nodes(
    spec: GQASpec,
    x_name: str,
    w_q_name: str,
    w_k_name: str,
    w_v_name: str,
    w_o_name: str,
    seqlens_k_name: str,
    total_seqlen_name: str,
    cos_cache_name: str,
    sin_cache_name: str,
    out_name: str,
    *,
    rope_spec: RoPESpec | None,
    name_scope: str,
) -> list[NodeProto]:
    """Emit MatMul(Q) | MatMul(K) | MatMul(V) -> GQA -> MatMul(O)."""
    if spec.sliding_window is not None:
        raise NotImplementedError("GQA sliding_window not supported in M0.")
    if spec.output_gate is not None:
        raise NotImplementedError("GQA output_gate not supported in M0.")
    if spec.v_norm is not None:
        raise NotImplementedError("GQA v_norm not supported in M0.")
    if spec.qk_norm is not None:
        raise NotImplementedError("GQA qk_norm not supported in M0.")
    if spec.qk_norm_fixed_scale is not None:
        raise NotImplementedError("GQA qk_norm_fixed_scale not supported in M0.")
    if spec.kv_source_layer_offset != 0:
        raise NotImplementedError("GQA kv_source_layer_offset not supported in M0.")
    if spec.logit_softcap != 0.0:
        raise NotImplementedError("GQA logit_softcap not supported in M0.")
    if spec.head_dim % 16 != 0:
        raise ValueError(
            "com.microsoft.GroupQueryAttention CPU kernel requires "
            f"head_dim % 16 == 0; got head_dim={spec.head_dim}."
        )

    p = name_scope
    nodes: list[NodeProto] = []

    # Transpose [out, in] weights to [in, out] for MatMul.
    nodes.append(helper.make_node("Transpose", [w_q_name], [f"{p}/w_q_T"], perm=[1, 0]))
    nodes.append(helper.make_node("Transpose", [w_k_name], [f"{p}/w_k_T"], perm=[1, 0]))
    nodes.append(helper.make_node("Transpose", [w_v_name], [f"{p}/w_v_T"], perm=[1, 0]))
    nodes.append(helper.make_node("Transpose", [w_o_name], [f"{p}/w_o_T"], perm=[1, 0]))

    nodes.append(helper.make_node("MatMul", [x_name, f"{p}/w_q_T"], [f"{p}/q_proj"],
                                  name=f"{p}/q_proj_matmul"))
    nodes.append(helper.make_node("MatMul", [x_name, f"{p}/w_k_T"], [f"{p}/k_proj"],
                                  name=f"{p}/k_proj_matmul"))
    nodes.append(helper.make_node("MatMul", [x_name, f"{p}/w_v_T"], [f"{p}/v_proj"],
                                  name=f"{p}/v_proj_matmul"))

    do_rotary = 1 if rope_spec is not None else 0
    gqa_inputs = [
        f"{p}/q_proj",
        f"{p}/k_proj",
        f"{p}/v_proj",
        "",  # past_key (empty — no KV cache in M0 forward parity)
        "",  # past_value
        seqlens_k_name,
        total_seqlen_name,
        cos_cache_name if rope_spec is not None else "",
        sin_cache_name if rope_spec is not None else "",
    ]
    nodes.append(helper.make_node(
        "GroupQueryAttention",
        inputs=gqa_inputs,
        outputs=[f"{p}/gqa_out", f"{p}/present_k", f"{p}/present_v"],
        name=f"{p}/GroupQueryAttention",
        domain="com.microsoft",
        num_heads=int(spec.num_heads),
        kv_num_heads=int(spec.num_kv_heads),
        do_rotary=do_rotary,
        scale=float(spec.scale),
    ))

    nodes.append(helper.make_node("MatMul", [f"{p}/gqa_out", f"{p}/w_o_T"], [out_name],
                                  name=f"{p}/o_proj_matmul"))
    return nodes


__all__ = [
    "build_gqa_nodes",
    "build_rope_cos_sin_initializers",
]
