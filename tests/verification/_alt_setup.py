"""Alt M0 block setup with DIFFERENT hyperparameters than tests/parity/_canonical.

Distinct from canonical to confirm the runtime ports work on novel shapes,
not just the one fixed configuration the existing parity tests cover.

Differences from canonical:
- B=2 (vs 1), S=16 (vs 8)
- num_heads=8 (vs 16), num_kv_heads=4 (vs 8), head_dim=64 (vs 128)
- D_ff=256 (vs 512)
- Weights scaled by 1/sqrt(D_model) to keep magnitudes inside fp16 range
  (canonical uses unscaled randn which overflows fp16 — irrelevant for V2/V4
  but lets V3 fp32 GPU still operate on well-conditioned numbers).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from components import (
    BlockGraph,
    GQASpec,
    NormStandardSpec,
    RoPESpec,
    SwiGLUSpec,
    pre_norm_block,
)


B: int = 2
S: int = 16
H_Q: int = 8
H_KV: int = 4
DH: int = 64
D_FF: int = 256
D_MODEL: int = H_Q * DH  # 512
SEED: int = 1729


@dataclass(frozen=True)
class AltSetup:
    graph: BlockGraph
    params: dict[str, dict[str, torch.Tensor]]
    inputs: dict[str, torch.Tensor]


def alt_setup() -> AltSetup:
    torch.manual_seed(SEED)

    norm0 = NormStandardSpec(eps=1e-6)
    norm1 = NormStandardSpec(eps=1e-6)
    attn = GQASpec(num_heads=H_Q, head_dim=DH, num_kv_heads=H_KV)
    rope = RoPESpec(theta=10000.0)
    ffn = SwiGLUSpec(intermediate_size=D_FF)

    graph = pre_norm_block(
        block_norm=norm0,
        attention=attn,
        positional_encoding=rope,
        post_attention_norm=norm1,
        ffn=ffn,
    )

    # Scale weights down to keep activations well within fp16 range, so V3
    # has a chance of running fp16 cleanly even though we still gate on fp32.
    s = 1.0 / math.sqrt(D_MODEL)

    params: dict[str, dict[str, torch.Tensor]] = {
        "n0": {"weight": torch.randn(D_MODEL, dtype=torch.float32)},
        "attn": {
            "w_q": torch.randn(H_Q * DH, D_MODEL, dtype=torch.float32) * s,
            "w_k": torch.randn(H_KV * DH, D_MODEL, dtype=torch.float32) * s,
            "w_v": torch.randn(H_KV * DH, D_MODEL, dtype=torch.float32) * s,
            "w_o": torch.randn(D_MODEL, H_Q * DH, dtype=torch.float32) * s,
        },
        "n1": {"weight": torch.randn(D_MODEL, dtype=torch.float32)},
        "ffn": {
            "w_gate": torch.randn(D_FF, D_MODEL, dtype=torch.float32) * s,
            "w_up":   torch.randn(D_FF, D_MODEL, dtype=torch.float32) * s,
            "w_down": torch.randn(D_MODEL, D_FF, dtype=torch.float32) * s,
        },
    }

    inputs: dict[str, torch.Tensor] = {
        "input": torch.randn(B, S, D_MODEL, dtype=torch.float32),
        "position_ids": torch.arange(S, dtype=torch.int64).unsqueeze(0).repeat(B, 1),
    }

    return AltSetup(graph=graph, params=params, inputs=inputs)


def weight_shapes_from_params(
    params: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, tuple[int, ...]]]:
    return {
        node_name: {slot: tuple(tensor.shape) for slot, tensor in weights.items()}
        for node_name, weights in params.items()
    }


__all__ = [
    "AltSetup", "alt_setup", "weight_shapes_from_params",
    "B", "S", "H_Q", "H_KV", "DH", "D_FF", "D_MODEL", "SEED",
]
