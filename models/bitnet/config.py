"""BitNet b1.58 config + spec adapter.

Verified against `transformers.BitNetConfig` for
`microsoft/bitnet-b1.58-2B-4T`.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class BitNetConfig:
    hidden_size: int                # 2560
    num_attention_heads: int        # 20
    num_key_value_heads: int        # 5 (GQA)
    intermediate_size: int          # 6912
    num_hidden_layers: int          # 30
    max_position_embeddings: int    # 2048
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128256
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    dtype: torch.dtype = torch.float32

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def bitnet_b158_2b(cls) -> "BitNetConfig":
        """microsoft/bitnet-b1.58-2B-4T defaults."""
        return cls(
            hidden_size=2560, num_attention_heads=20, num_key_value_heads=5,
            intermediate_size=6912, num_hidden_layers=30,
            max_position_embeddings=2048,
        )

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "BitNetConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)
        rp = hf.get("rope_parameters", {}) or {}
        rope_theta = float(rp.get("rope_theta", hf.get("rope_theta", 500000.0)))
        return cls(
            hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf.get("num_key_value_heads",
                                       hf["num_attention_heads"]),
            intermediate_size=hf["intermediate_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            max_position_embeddings=hf["max_position_embeddings"],
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=hf["vocab_size"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            dtype=dtype,
        )

    def to_block_spec(self) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # Sub-norms: hidden-size weight for attn_sub_norm (post-attn
        # concat-reshape carries hidden_size dim); intermediate-size
        # weight for ffn_sub_norm (acts on the gated I-dim activation).
        # Both are STANDARD_W RMSNorm with rms_norm_eps. Source:
        # modeling_bitnet.py:74, 177.
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
            ),
            attn_sub_norm=norm_spec,                   # diff with Llama
        )
        # BitNet FFN: GateKind.SWIGLU + Activation.RELU2 (squared ReLU,
        # not SILU). Plus ffn_sub_norm on the gated activation.
        # Source: modeling_bitnet.py:64-78.
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.RELU2,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
            ffn_sub_norm=norm_spec,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
