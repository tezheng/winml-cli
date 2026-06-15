"""GQA forward as an OpenVINO opset13 sub-graph.

Mirror of ``runtimes.torch_port.attention.gqa_forward``:

    q = matmul(x, W_q.T).view(B, S, H_q,  Dh).transpose(1, 2)   # [B, H_q, S, Dh]
    k = matmul(x, W_k.T).view(B, S, H_kv, Dh).transpose(1, 2)
    v = matmul(x, W_v.T).view(B, S, H_kv, Dh).transpose(1, 2)
    [optional] q, k = apply_rope(q, k, cos, sin)
    k = repeat_interleave(k, n_rep, dim=1)
    v = repeat_interleave(v, n_rep, dim=1)
    attn = SDPA(q, k, v, scale=1/sqrt(Dh), causal=True)         # opset13
    out  = matmul(attn.transpose(1, 2).reshape(B, S, H_q*Dh), W_o.T)
"""
from __future__ import annotations

from typing import Optional

import openvino as ov
from openvino import Type
from openvino import opset13 as opset13
from openvino import opset16 as opset

from components import GQASpec, RoPESpec

from .rope import apply_rope_single, build_rope_cos_sin


def _matmul_xWT(x: ov.Output, w: ov.Output) -> ov.Output:
    """torch.nn.functional.linear semantics: out = x @ w.T.

    OV MatMul supports ``transpose_b=True`` which fuses the transpose.
    """
    return opset.matmul(x, w, transpose_a=False, transpose_b=True)


def _repeat_interleave_axis1(
    t: ov.Output, n_rep: int, b: int, h_kv: int, s: int, dh: int,
) -> ov.Output:
    """Expand [B, H_kv, S, Dh] to [B, H_kv*n_rep, S, Dh] by repeating each head.

    Implementation: reshape to [B, H_kv, 1, S, Dh], broadcast on axis-2 to
    [B, H_kv, n_rep, S, Dh], reshape to [B, H_kv*n_rep, S, Dh]. Equivalent to
    ``torch.repeat_interleave(t, n_rep, dim=1)``.
    """
    if n_rep == 1:
        return t
    # [B, H_kv, S, Dh] -> [B, H_kv, 1, S, Dh]
    t = opset.unsqueeze(t, opset.constant([2], Type.i32))
    # broadcast to [B, H_kv, n_rep, S, Dh] — fully static shape.
    target_shape = opset.constant([b, h_kv, n_rep, s, dh], Type.i64)
    t = opset.broadcast(t, target_shape, broadcast_spec="BIDIRECTIONAL")
    # reshape to [B, H_kv*n_rep, S, Dh]
    out_shape = opset.constant([b, h_kv * n_rep, s, dh], Type.i64)
    return opset.reshape(t, out_shape, special_zero=False)


def build_gqa(
    spec: GQASpec,
    x: ov.Output,                       # [B, S, D_model]
    w_q: ov.Output,                     # [H_q*Dh, D_model]
    w_k: ov.Output,                     # [H_kv*Dh, D_model]
    w_v: ov.Output,                     # [H_kv*Dh, D_model]
    w_o: ov.Output,                     # [D_model, H_q*Dh]
    *,
    rope_spec: Optional[RoPESpec] = None,
    position_ids: Optional[ov.Output] = None,
    batch_size: int = 1,
    seq_len: int = 8,
) -> ov.Output:
    """Build the full GQA graph and return the output node.

    Args:
        spec:        GQASpec (M0 subset only).
        x:           Hidden state node, shape ``[B, S, D_model]``.
        w_q/k/v/o:   Weight parameter nodes (see shape doc above).
        rope_spec:   When non-None RoPE is applied to Q/K.
        position_ids: i64 ``[B, S]`` node, required if ``rope_spec`` is set.
        batch_size:  Static B; needed for reshape constants (graph is static).
        seq_len:     Static S; needed for reshape constants.
    """
    # ---- M0 unsupported-feature guards (mirror torch_port) ------------------
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

    H_q, H_kv, Dh = spec.num_heads, spec.num_kv_heads, spec.head_dim
    D_model = H_q * Dh
    B, S = batch_size, seq_len

    x_dtype = x.get_element_type()

    # ---- Q/K/V projections + head reshape -----------------------------------
    q = _matmul_xWT(x, w_q)   # [B, S, H_q*Dh]
    k = _matmul_xWT(x, w_k)   # [B, S, H_kv*Dh]
    v = _matmul_xWT(x, w_v)   # [B, S, H_kv*Dh]

    # Reshape to [B, S, H, Dh] then transpose to [B, H, S, Dh].
    q = opset.reshape(q, opset.constant([B, S, H_q, Dh], Type.i64), special_zero=False)
    k = opset.reshape(k, opset.constant([B, S, H_kv, Dh], Type.i64), special_zero=False)
    v = opset.reshape(v, opset.constant([B, S, H_kv, Dh], Type.i64), special_zero=False)
    perm = opset.constant([0, 2, 1, 3], Type.i64)
    q = opset.transpose(q, perm)
    k = opset.transpose(k, perm)
    v = opset.transpose(v, perm)

    # ---- RoPE on Q and K ----------------------------------------------------
    if rope_spec is not None:
        if position_ids is None:
            raise ValueError("position_ids is required when rope_spec is provided.")
        if rope_spec.partial_rotary_factor != 1.0:
            raise NotImplementedError(
                "M0 only supports full rotary (partial_rotary_factor == 1.0)."
            )
        cos, sin = build_rope_cos_sin(
            position_ids, head_dim=Dh, theta=rope_spec.theta, cos_sin_dtype=x_dtype,
        )
        q = apply_rope_single(q, cos, sin)
        k = apply_rope_single(k, cos, sin)

    # ---- GQA broadcast on K/V -----------------------------------------------
    if H_kv < H_q:
        if H_q % H_kv != 0:
            raise ValueError(f"H_q ({H_q}) must be divisible by H_kv ({H_kv}).")
        n_rep = H_q // H_kv
        k = _repeat_interleave_axis1(k, n_rep, B, H_kv, S, Dh)
        v = _repeat_interleave_axis1(v, n_rep, B, H_kv, S, Dh)

    # ---- SDPA (opset13) -----------------------------------------------------
    # Use explicit scale to match torch_port (1 / sqrt(Dh)).
    scale_const = opset13.constant(spec.scale, x_dtype)
    attn = opset13.scaled_dot_product_attention(
        q, k, v,
        attention_mask=None,
        scale=scale_const,
        causal=bool(spec.causal),
    )  # [B, H_q, S, Dh]

    # ---- Concat heads + O projection ----------------------------------------
    # transpose back to [B, S, H_q, Dh] then reshape to [B, S, D_model].
    attn = opset.transpose(attn, opset.constant([0, 2, 1, 3], Type.i64))
    attn = opset.reshape(attn, opset.constant([B, S, D_model], Type.i64), special_zero=False)
    return _matmul_xWT(attn, w_o)


__all__ = ["build_gqa"]
