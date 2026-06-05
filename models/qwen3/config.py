"""Qwen3 model config + adapter to api specs.

Verified against `transformers.Qwen3Config` and Qwen3-0.6B / 1.7B / 4B / 8B
config.json on HuggingFace.

Key Qwen3 facts (see research/05-kvcache-attention.v2.md, fixed in v2):
- GQA (n_kv_heads < n_q_heads)
- head_dim is EXPLICIT, not hidden_size / n_q_heads (Qwen3-0.6B: 1024/16=64 vs head_dim=128)
- QK-norm PRE-RoPE, weight shape PER_HEAD_DH (verified against modeling_qwen3.py)
- RoPE SPLIT_HALF basis, base_theta=1_000_000 (NOT 5M as v1 incorrectly claimed)
- RMSNorm STANDARD_W mode
- SwiGLU FFN
- tie_word_embeddings: True for 0.6B/1.7B/4B, False for 8B
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
class Qwen3Config:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Qwen3Config":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # transformers 5.x nests rope_theta under `rope_parameters`; older flat
        # `rope_theta` key is still accepted for backward compatibility.
        if "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        elif (
            isinstance(hf.get("rope_parameters"), dict)
            and "rope_theta" in hf["rope_parameters"]
        ):
            rope_theta = float(hf["rope_parameters"]["rope_theta"])
        else:
            rope_theta = 10000.0

        return cls(
            hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf["num_key_value_heads"],
            head_dim=hf.get(
                "head_dim",
                hf["hidden_size"] // hf["num_attention_heads"],
            ),
            intermediate_size=hf["intermediate_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
        )

    def to_block_spec(self) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        qk_norm_spec = norm_spec
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            qk_norm=qk_norm_spec,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            ),
        )
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
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            input_norm=norm_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
