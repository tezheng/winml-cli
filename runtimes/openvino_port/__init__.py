"""OpenVINO GPU port — assembles opset graphs equivalent to torch_port.

Each module exposes a ``build_*`` function returning an ``ov.Output`` node.
``block_lowerer.lower_block`` walks a ``components.BlockGraph`` and stitches
the nodes into a single ``ov.Model`` ready for ``core.compile_model("GPU")``.
"""
from __future__ import annotations

from .attention import build_gqa
from .block_lowerer import lower_block
from .ffn import build_swiglu
from .misc import build_add
from .norm import build_rms_norm
from .rope import build_rope_cos_sin

__all__ = [
    "build_rms_norm",
    "build_gqa",
    "build_rope_cos_sin",
    "build_swiglu",
    "build_add",
    "lower_block",
]
