"""Build a decoder block from components/ and print its structure.

Demonstrates: BlockGraph is data; you can construct one without
torch/numpy/openvino loaded. Only stdlib imports happen under the hood.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `components` importable when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from components import (
    GQASpec,
    NormStandardSpec,
    RoPESpec,
    SwiGLUSpec,
    pre_norm_block,
)


def main() -> None:
    block = pre_norm_block(
        block_norm=NormStandardSpec(eps=1e-6),
        attention=GQASpec(num_heads=16, head_dim=128, num_kv_heads=8),
        positional_encoding=RoPESpec(theta=10_000_000.0),
        post_attention_norm=NormStandardSpec(eps=1e-6),
        ffn=SwiGLUSpec(intermediate_size=3072),
    )

    print(f"BlockGraph: {len(block.nodes)} nodes")
    for n in block.nodes:
        inputs_str = str(n.inputs)
        print(f"  {n.name:8s} <- {inputs_str:24s} :: {type(n.spec).__name__}")
    print(f"output: {block.output}")

    # Show the spec details we care about.
    print("\nSpec details:")
    attn = next(n.spec for n in block.nodes if isinstance(n.spec, GQASpec))
    rope = next(n.spec for n in block.nodes if isinstance(n.spec, RoPESpec))
    ffn = next(n.spec for n in block.nodes if isinstance(n.spec, SwiGLUSpec))
    print(
        f"  Attention: {attn.num_heads} heads, "
        f"{attn.num_kv_heads} kv heads, head_dim={attn.head_dim}, "
        f"scale={attn.scale:.6f}"
    )
    print(f"  RoPE:      theta={rope.theta:g}")
    print(f"  FFN:       intermediate_size={ffn.intermediate_size}")


if __name__ == "__main__":
    main()


# Expected output (when run with `.venv\Scripts\python.exe examples\v2\01_build_block.py`):
#
# BlockGraph: 7 nodes
#   n0       <- ('input',)               :: NormStandardSpec
#   rope     <- ('n0',)                  :: RoPESpec
#   attn     <- ('rope',)                :: GQASpec
#   r0       <- ('input', 'attn')        :: AddSpec
#   n1       <- ('r0',)                  :: NormStandardSpec
#   ffn      <- ('n1',)                  :: SwiGLUSpec
#   r1       <- ('r0', 'ffn')            :: AddSpec
# output: r1
#
# Spec details:
#   Attention: 16 heads, 8 kv heads, head_dim=128, scale=0.088388
#   RoPE:      theta=1e+07
#   FFN:       intermediate_size=3072
