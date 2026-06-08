"""Falcon-7B config + spec adapter.

Verified against `transformers.FalconConfig` for `tiiuae/falcon-7b`.

The decoder block uses PARALLEL residual + MQA. See models/falcon7b/
__init__.py for the FFN-form caveat (SwiGLU substitution for the
ungated GELU FFN).
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from api import specs, types


@dataclass(frozen=True)
class Falcon7BConfig:
    hidden_size: int                # 4544 for Falcon-7B
    num_attention_heads: int        # 71
    num_kv_heads: int               # 1 (MQA)
    num_hidden_layers: int          # 32
    intermediate_size: int          # 4 * hidden_size = 18176
    max_position_embeddings: int    # 2048
    rope_theta: float = 10000.0
    layer_norm_epsilon: float = 1e-5
    vocab_size: int = 65024
    dtype: torch.dtype = torch.float32

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def falcon_7b(cls) -> "Falcon7BConfig":
        return cls(
            hidden_size=4544, num_attention_heads=71, num_kv_heads=1,
            num_hidden_layers=32, intermediate_size=4 * 4544,
            max_position_embeddings=2048,
        )

    def to_attention_spec(self) -> specs.AttentionSpec:
        """Falcon-7B attention sublayer. MQA (n_kv=1), RoPE SPLIT_HALF,
        no biases. Source: configuration_falcon.py:66-89."""
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_kv_heads,                  # 1 → MQA
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
            ),
        )

    def to_block_spec(self) -> specs.DecoderBlockSpec:
        """Falcon-7B decoder block.

        block_layout=PARALLEL, num_ln_in_parallel_attn=1 — one shared
        LayerNorm feeds both attn and FFN. We substitute RMSNorm for LN
        and SwiGLU for the ungated GELU FFN (both substitutions are
        documented Phase-3 gaps — the PARALLEL topology and MQA
        attention are the V2 contribution).
        """
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.layer_norm_epsilon,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # FFN substitution: Falcon's actual FFN is `dense_h_to_4h` →
        # GELU → `dense_4h_to_h` (no gating). We use SwiGLU as a
        # placeholder — same shape budget, different math. See
        # models/falcon7b/__init__.py.
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=self.to_attention_spec(),
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=None,          # PARALLEL: shared pre_attn_norm
            block_layout=types.BlockLayout.PARALLEL,
        )
