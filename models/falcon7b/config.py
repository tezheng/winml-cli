"""Falcon-7B config + spec adapter.

Verified against `transformers.FalconConfig` for `tiiuae/falcon-7b`:
- hidden_size=4544, num_attention_heads=71, num_kv_heads=1 (MQA)
- num_hidden_layers=32, intermediate_size = 4 * hidden_size = 18176
- parallel_attn=True, new_decoder_architecture=False (=> num_ln_in_parallel_attn=1)
- bias=False (Linear bias OFF), but LayerNorm bias is the PyTorch default (=True)
- activation="gelu" (exact, via get_activation('gelu') → nn.GELU(approximate='none'))
- alibi=False (Falcon-7B uses RoPE)

The decoder block uses PARALLEL residual (shared pre-norm fed to both
attention and FFN) + MQA + RoPE SPLIT_HALF + ungated exact GELU FFN +
LayerNorm-with-bias.

v6 A2 lands the full decoder block; v5-phase2 V2 landed the attention
sublayer only.
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

    def to_ffn_spec(self) -> specs.FFNSpec:
        """Falcon-7B FFN: ungated exact GELU.
        Source: modeling_falcon.py:531-544 (FalconMLP — `dense_h_to_4h`,
        `act = get_activation('gelu')`, `dense_4h_to_h`)."""
        return specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.GELU_EXACT,
            gate_kind=types.GateKind.GELU_ONLY,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )

    def to_norm_spec(self) -> specs.NormSpec:
        """Falcon-7B LayerNorm with bias (Falcon `bias` config field only
        affects Linears; LayerNorm uses PyTorch defaults i.e. bias=True).
        Source: modeling_falcon.py:574-578."""
        return specs.NormSpec(
            kind=types.NormKind.LAYER,
            eps=self.layer_norm_epsilon,
            has_bias=True,
        )

    def to_block_spec(self) -> specs.DecoderBlockSpec:
        """Falcon-7B decoder block — PARALLEL residual + MQA + RoPE + ungated
        GELU FFN + LayerNorm with bias.

        For Falcon-7B parallel_attn=True with new_decoder_architecture=False,
        HF sets num_ln_in_parallel_attn=1 → ONE shared `input_layernorm` feeds
        both the attention and the MLP. Source: modeling_falcon.py:565-578,
        594-634.
        """
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=self.to_attention_spec(),
            channel_mixer=self.to_ffn_spec(),
            pre_attn_norm=self.to_norm_spec(),
            pre_ffn_norm=None,          # PARALLEL: shared pre_attn_norm
            block_layout=types.BlockLayout.PARALLEL,
        )
