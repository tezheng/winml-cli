"""PyTorch ground-truth port — pure forward functions for each Spec.

Every other runtime port compares against the tensors produced here. All math
runs in torch.float32 (no quantization, no half precision); operations follow
the Hugging Face transformers reference implementations cited in each module.
"""
from __future__ import annotations

from .attention import gqa_forward
from .block_runner import run_block
from .ffn import swiglu_forward
from .misc import add_forward
from .norm import rms_norm_forward
from .rope import apply_rotary_pos_emb_single, build_rope_cos_sin, rope_forward

__all__ = [
    "rms_norm_forward",
    "gqa_forward",
    "rope_forward",
    "apply_rotary_pos_emb_single",
    "build_rope_cos_sin",
    "swiglu_forward",
    "add_forward",
    "run_block",
]
