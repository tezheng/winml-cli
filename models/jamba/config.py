"""Jamba (ai21labs) config + per-layer block spec adapter (v6 B2).

The Jamba block at index L is:
- attention if `L % attn_layer_period == attn_layer_offset`
- mamba otherwise

For ai21labs/Jamba-tiny-dev:
- num_hidden_layers=4 (tiny), attn_layer_period=2, attn_layer_offset=1
  → layers 0=mamba, 1=attn, 2=mamba, 3=attn.
For ai21labs/Jamba-v0.1:
- num_hidden_layers=32, attn_layer_period=8, attn_layer_offset=4
  → layers 4, 12, 20, 28 are attention; rest are mamba.

Source: `transformers/models/jamba/configuration_jamba.py:50-94`.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch

from api import specs, types


@dataclass(frozen=True)
class JambaConfig:
    hidden_size: int                # 4096 for Jamba-v0.1
    intermediate_size: int          # 14336
    num_hidden_layers: int          # 32
    num_attention_heads: int        # 32
    num_key_value_heads: int        # 8 (GQA)
    rms_norm_eps: float = 1e-6
    attn_layer_period: int = 8
    attn_layer_offset: int = 4
    expert_layer_period: int = 2
    expert_layer_offset: int = 1
    num_experts: int = 16
    num_experts_per_tok: int = 2
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int = 0          # 0 → auto = ceil(hidden_size / 16)
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    max_position_embeddings: int = 262144
    vocab_size: int = 65536
    dtype: torch.dtype = torch.float32

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def dt_rank(self) -> int:
        if self.mamba_dt_rank > 0:
            return self.mamba_dt_rank
        return math.ceil(self.hidden_size / 16)

    @property
    def mamba_d_inner(self) -> int:
        return self.mamba_expand * self.hidden_size

    @classmethod
    def jamba_tiny_dev(cls) -> "JambaConfig":
        """ai21labs/Jamba-tiny-dev: 4 layers, tiny dims, all-dense MLPs.

        Verified vs Jamba-tiny-dev config.json:
            hidden_size=128, intermediate_size=384, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=4,
            attn_layer_period=2, attn_layer_offset=1,
            expert_layer_period=2, expert_layer_offset=0,
            num_experts=2, num_experts_per_tok=2, vocab_size=65536.
        """
        return cls(
            hidden_size=128, intermediate_size=384, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=4,
            attn_layer_period=2, attn_layer_offset=1,
            expert_layer_period=2, expert_layer_offset=0,
            num_experts=2, num_experts_per_tok=2,
        )

    def is_attention_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.attn_layer_period == self.attn_layer_offset

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.expert_layer_period == self.expert_layer_offset

    def to_norm_spec(self) -> specs.NormSpec:
        return specs.NormSpec(kind=types.NormKind.RMS, eps=self.rms_norm_eps)

    def to_attention_spec(self) -> specs.AttentionSpec:
        """Jamba attention: GQA + NO RoPE.

        Source: modeling_jamba.py:147-199 — JambaAttention has NO rotary
        embedding (apply_rotary_pos_emb is defined but never called).
        """
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            rope=None,                  # CRITICAL: Jamba has no RoPE
        )

    def to_ffn_spec(self) -> specs.FFNSpec:
        """Standard SwiGLU FFN (modeling_jamba.py:477-490)."""
        return specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )

    def to_mamba_spec(self) -> specs.SSMSpec:
        """Jamba's Mamba-1 spec with intra-mixer dt/B/C layernorms.

        Source: modeling_jamba.py:249-251 — `dt_layernorm`, `b_layernorm`,
        `c_layernorm` of shapes `dt_rank`, `d_state`, `d_state`.
        """
        norm = specs.NormSpec(kind=types.NormKind.RMS, eps=self.rms_norm_eps)
        return specs.SSMSpec(
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            d_inner=self.mamba_d_inner,
            expand_factor=self.mamba_expand,
            conv_bias=self.mamba_conv_bias,
            bias=self.mamba_proj_bias,
            activation=types.Activation.SILU,
            kind=types.SSMKind.MAMBA1,
            dt_rank=self.dt_rank,
            dt_layernorm=norm,
            b_layernorm=norm,
            c_layernorm=norm,
        )

    def to_block_spec(self, layer_idx: int) -> specs.DecoderBlockSpec:
        """Per-layer dispatch: attention or mamba.

        For v6 B2 we ship the DENSE-only path (always FFN; never MoE).
        The MoE composition with Mamba layers is deferred — see
        `models/jamba/__init__.py`.

        Source: modeling_jamba.py:670-684 (`ALL_DECODER_LAYER_TYPES`
        + JambaModel.__init__).
        """
        norm = self.to_norm_spec()
        if self.is_attention_layer(layer_idx):
            token_mixer = self.to_attention_spec()
        else:
            token_mixer = self.to_mamba_spec()
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=token_mixer,
            channel_mixer=self.to_ffn_spec(),
            pre_attn_norm=norm,
            pre_ffn_norm=norm,
        )
