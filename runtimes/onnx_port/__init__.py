"""ONNX runtime port — emits ONNX graphs that compute the same outputs as
``runtimes/torch_port``.

Uses Microsoft contrib ops (``com.microsoft`` domain) for the heavy lifting
(``GroupQueryAttention``, ``RotaryEmbedding``) plus the default-domain
``SimplifiedLayerNormalization``. Target runtime: ONNX Runtime CPU EP at fp32,
parity against the torch_port golden tensor to ``atol <= 1e-5``.
"""
from __future__ import annotations

from .attention import build_gqa_nodes, build_rope_cos_sin_initializers
from .block_emitter import emit_block
from .ffn import build_swiglu_nodes
from .misc import build_add_node
from .norm import build_rms_norm_nodes
from .rope import build_rope_nodes

__all__ = [
    "build_rms_norm_nodes",
    "build_gqa_nodes",
    "build_rope_cos_sin_initializers",
    "build_rope_nodes",
    "build_swiglu_nodes",
    "build_add_node",
    "emit_block",
]
