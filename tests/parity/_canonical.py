"""Canonical M0 block setup — shared by every runtime-port parity test.

The graph, parameters, and inputs are built deterministically (fixed seed) so
the torch_port output saved to ``golden_m0.pt`` is reproducible bit-for-bit
across machines (modulo cpu vendor — all in fp32, no fused kernels). Other
ports import this same builder to keep weight/input layout identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from components import (
    BlockGraph,
    GQASpec,
    NormStandardSpec,
    RoPESpec,
    SwiGLUSpec,
    pre_norm_block,
)


# ---- Dimensions (M0 canonical) ---------------------------------------------
# Picked to stay small (test runs <1 s on CPU) while exercising:
#   - GQA broadcast (H_q != H_kv, 16/8 = 2x repeat)
#   - even Dh for RoPE
#   - D_ff != D_model so SwiGLU shapes are distinct
B: int = 1
S: int = 8
H_Q: int = 16
H_KV: int = 8
DH: int = 128
D_FF: int = 512
D_MODEL: int = H_Q * DH  # 2048
SEED: int = 42


@dataclass(frozen=True)
class CanonicalSetup:
    graph: BlockGraph
    params: dict[str, dict[str, torch.Tensor]]
    inputs: dict[str, torch.Tensor]


def _canonical_m0_setup() -> CanonicalSetup:
    """Return the canonical M0 (graph, params, inputs) triple.

    Uses ``torch.manual_seed(SEED)`` so two calls produce identical tensors.
    """
    torch.manual_seed(SEED)

    # ---- Specs --------------------------------------------------------------
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

    # ---- Parameters (deterministic via fixed seed) --------------------------
    # Weight names are torch_port-internal slot names ("w_q" etc.) — the public
    # ApiSchema uses W_q/W_k/... but the runner consumes whatever the forward
    # function expects.
    params: dict[str, dict[str, torch.Tensor]] = {
        "n0":   {"weight": torch.randn(D_MODEL, dtype=torch.float32)},
        # No params for "rope".
        "attn": {
            "w_q": torch.randn(H_Q * DH,  D_MODEL, dtype=torch.float32),
            "w_k": torch.randn(H_KV * DH, D_MODEL, dtype=torch.float32),
            "w_v": torch.randn(H_KV * DH, D_MODEL, dtype=torch.float32),
            "w_o": torch.randn(D_MODEL,   H_Q * DH, dtype=torch.float32),
        },
        # No params for "r0".
        "n1":   {"weight": torch.randn(D_MODEL, dtype=torch.float32)},
        "ffn":  {
            "w_gate": torch.randn(D_FF,    D_MODEL, dtype=torch.float32),
            "w_up":   torch.randn(D_FF,    D_MODEL, dtype=torch.float32),
            "w_down": torch.randn(D_MODEL, D_FF,    dtype=torch.float32),
        },
        # No params for "r1".
    }

    inputs: dict[str, torch.Tensor] = {
        "input":        torch.randn(B, S, D_MODEL, dtype=torch.float32),
        "position_ids": torch.arange(S, dtype=torch.int64).unsqueeze(0),  # [1, S]
    }

    return CanonicalSetup(graph=graph, params=params, inputs=inputs)


__all__ = [
    "CanonicalSetup",
    "_canonical_m0_setup",
    "B", "S", "H_Q", "H_KV", "DH", "D_FF", "D_MODEL", "SEED",
]
