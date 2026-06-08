"""Hunyuan-Large config + per-layer block spec adapter.

Verified against Hunyuan-Large public config.json. Note: HF
transformers v5.10.2 carries `hunyuan_v1_dense` and `hunyuan_v1_moe`
modules but neither implements the CLA (cross-layer attention) hook
that v5-phase2 V4 lands. The reference architecture is the original
Hunyuan-Large release.

The full Hunyuan-Large architecture is MoE (16 routed experts top-1 +
1 shared); v5-phase2 V4 lands the CLA shape only — the MoE channel
mixer is composed via existing `MoESpec`.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from api import specs, types


@dataclass(frozen=True)
class HunyuanLargeConfig:
    hidden_size: int                # 6400 (Hunyuan-Large public)
    num_attention_heads: int        # 80
    num_key_value_heads: int        # 8
    head_dim: int                   # 80 (non-standard; 6400/80=80)
    intermediate_size: int          # 18304 (moe_intermediate_size)
    num_hidden_layers: int          # 64
    max_position_embeddings: int    # 32768
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128000
    use_cla: bool = True
    cla_share_factor: int = 2       # even layers own K/V; odd borrow from L-1
    dtype: torch.dtype = torch.float32

    @classmethod
    def hunyuan_large(cls) -> "HunyuanLargeConfig":
        return cls(
            hidden_size=6400, num_attention_heads=80, num_key_value_heads=8,
            head_dim=80, intermediate_size=18304, num_hidden_layers=64,
            max_position_embeddings=32768,
        )

    def _attn_spec_owner(self) -> specs.AttentionSpec:
        """K/V-owning layer: full attention with split QKV."""
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
            ),
            kv_source_layer_offset=None,
        )

    def _attn_spec_borrower(self) -> specs.AttentionSpec:
        """K/V-borrowing layer: only q_proj + o_proj built; K/V from L-1."""
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
            ),
            kv_source_layer_offset=-1,
        )

    def _ffn_spec(self) -> specs.FFNSpec:
        return specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )

    def is_kv_owner(self, layer_idx: int) -> bool:
        """Even-indexed layers (0, 2, 4, ...) own K/V when use_cla=True
        and cla_share_factor=2. Source: Hunyuan-Large config."""
        if not self.use_cla:
            return True
        return (layer_idx % self.cla_share_factor) == 0

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        attn_spec = (self._attn_spec_owner() if self.is_kv_owner(layer_idx)
                     else self._attn_spec_borrower())
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=self._ffn_spec(),
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
