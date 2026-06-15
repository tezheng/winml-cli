"""RoPE (cos/sin build + apply) as OpenVINO opset16 sub-graphs.

Mirror of ``runtimes.torch_port.rope``:

    inv_freq = 1 / (theta ** (arange(0, Dh, 2) / Dh))         # [Dh/2]
    freqs    = position_ids[..., None] * inv_freq             # [B, S, Dh/2]
    emb      = concat(freqs, freqs, axis=-1)                  # [B, S, Dh]
    cos, sin = cos(emb), sin(emb)

Rotation (``apply_rotary_pos_emb_single``) for x shaped ``[B, H, S, Dh]``:

    cos_b = cos.unsqueeze(1)                                  # [B, 1, S, Dh]
    sin_b = sin.unsqueeze(1)
    rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2], -1)
    y = x * cos_b + rotate_half(x) * sin_b

OpenVINO's ``RoPEFusion`` pass collapses this Split/Mul/Negative/Concat
pattern into the dedicated GPU ``RoPE`` kernel.
"""
from __future__ import annotations

import numpy as np

import openvino as ov
from openvino import Type
from openvino import opset16 as opset


def build_rope_cos_sin(
    position_ids: ov.Output,   # [B, S] i64
    head_dim: int,
    theta: float,
    cos_sin_dtype: Type,
) -> tuple[ov.Output, ov.Output]:
    """Build cos/sin tables of shape [B, S, head_dim] for SPLIT_HALF basis."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim; got {head_dim}.")

    # inv_freq constant in fp32: [Dh/2]
    inv_freq_np = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    inv_freq = opset.constant(inv_freq_np, Type.f32)  # [Dh/2]

    pos_f = opset.convert(position_ids, Type.f32)                       # [B, S]
    pos_u = opset.unsqueeze(pos_f, opset.constant([-1], Type.i32))       # [B, S, 1]
    # Reviewer-found: axes must be unique per ONNX/OV Unsqueeze spec. [0, 0] was
    # silently accepted but is undefined behavior; [0, 1] is the correct two-axis insertion.
    inv_u = opset.unsqueeze(inv_freq, opset.constant([0, 1], Type.i32))  # [1, 1, Dh/2]

    freqs = opset.multiply(pos_u, inv_u)                                # [B, S, Dh/2]
    emb = opset.concat([freqs, freqs], axis=-1)                          # [B, S, Dh]
    cos = opset.cos(emb)
    sin = opset.sin(emb)
    if cos_sin_dtype != Type.f32:
        cos = opset.convert(cos, cos_sin_dtype)
        sin = opset.convert(sin, cos_sin_dtype)
    return cos, sin


def _rotate_half(x: ov.Output) -> ov.Output:
    """concat(-x[..., d/2:], x[..., :d/2]) on the last axis.

    Uses opset.split which returns equal halves — head_dim is even by RoPE
    contract.
    """
    # Split along axis -1 into 2 equal halves.
    halves = opset.split(x, opset.constant(-1, Type.i32), num_splits=2).outputs()
    x1, x2 = halves[0], halves[1]
    neg_x2 = opset.negative(x2)
    return opset.concat([neg_x2, x1], axis=-1)


def apply_rope_single(
    x: ov.Output,        # [B, H, S, Dh]
    cos: ov.Output,      # [B, S, Dh]
    sin: ov.Output,      # [B, S, Dh]
) -> ov.Output:
    """Apply rope rotation. cos/sin broadcast over the head axis."""
    # cos/sin: [B, S, Dh] -> [B, 1, S, Dh]
    cos_b = opset.unsqueeze(cos, opset.constant([1], Type.i32))
    sin_b = opset.unsqueeze(sin, opset.constant([1], Type.i32))
    rotated = _rotate_half(x)
    return opset.add(opset.multiply(x, cos_b), opset.multiply(rotated, sin_b))


__all__ = ["build_rope_cos_sin", "apply_rope_single"]
