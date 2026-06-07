"""Llama 3 / 3.1 / 3.2 model config + adapter to api specs.

Verified against `transformers.LlamaConfig` and HF model-card config.json for
unsloth/Llama-3.2-1B-Instruct (3.2 1B), meta-llama/Meta-Llama-3-8B (3.0 8B),
and meta-llama/Meta-Llama-3.1-8B (3.1 8B).

Key Llama 3 facts (verified against modeling_llama.py):
- GQA (n_kv_heads < n_q_heads for 3.x variants).
- head_dim is explicit (3.2 1B: 2048/32=64 = head_dim=64; 3.0 8B: 4096/32=128).
- RMSNorm STANDARD_W mode (modeling_llama.py:62-67).
- SwiGLU FFN (modeling_llama.py:171-184), `mlp_bias` defaults False.
- No QK-norm.
- No attention biases (attention_bias defaults False).
- RoPE: SPLIT_HALF basis (rotate_half at modeling_llama.py:138-142).
  - 3.0:  rope_type="default", θ=500_000.
  - 3.1:  rope_type="llama3", factor=8, low=1, high=4, ctx=8192, θ=500_000.
  - 3.2:  rope_type="llama3", factor=32 (NOT 8 — that is 3.1), low=1, high=4,
          ctx=8192, θ=500_000.
- tie_word_embeddings=True for 3.2 1B/3B; False for 8B.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Llama3Config:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rope_theta: float
    rope_type: str                      # "default" | "llama3"
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    # LLAMA3 smooth-scaling fields — populated when rope_type=="llama3".
    rope_factor: Optional[float] = None
    rope_low_freq_factor: Optional[float] = None
    rope_high_freq_factor: Optional[float] = None
    rope_original_max_position_embeddings: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Llama3Config":
        # transformers 5.x emits `dtype`; older 4.x emits `torch_dtype`.
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # Read rope_parameters (transformers 5.x). Backward compat: also accept
        # flat `rope_theta` + `rope_scaling` (transformers 4.x).
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 500_000.0))
            rope_type = rp.get("rope_type", "default")
            rope_factor = rp.get("factor")
            rope_low_freq_factor = rp.get("low_freq_factor")
            rope_high_freq_factor = rp.get("high_freq_factor")
            rope_original_max_position_embeddings = rp.get(
                "original_max_position_embeddings"
            )
        else:
            rope_theta = float(hf.get("rope_theta", 500_000.0))
            rs = hf.get("rope_scaling")
            if isinstance(rs, dict) and rs.get("rope_type") == "llama3":
                rope_type = "llama3"
                rope_factor = rs.get("factor")
                rope_low_freq_factor = rs.get("low_freq_factor")
                rope_high_freq_factor = rs.get("high_freq_factor")
                rope_original_max_position_embeddings = rs.get(
                    "original_max_position_embeddings"
                )
            else:
                rope_type = "default"
                rope_factor = None
                rope_low_freq_factor = None
                rope_high_freq_factor = None
                rope_original_max_position_embeddings = None

        head_dim = hf.get("head_dim")
        if head_dim is None:
            head_dim = hf["hidden_size"] // hf["num_attention_heads"]

        return cls(
            hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf.get(
                "num_key_value_heads", hf["num_attention_heads"]
            ),
            head_dim=head_dim,
            intermediate_size=hf["intermediate_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            rope_theta=rope_theta,
            rope_type=rope_type,
            rope_factor=rope_factor,
            rope_low_freq_factor=rope_low_freq_factor,
            rope_high_freq_factor=rope_high_freq_factor,
            rope_original_max_position_embeddings=rope_original_max_position_embeddings,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            attention_bias=bool(hf.get("attention_bias", False)),
            mlp_bias=bool(hf.get("mlp_bias", False)),
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        if self.rope_type == "llama3":
            if (self.rope_factor is None or self.rope_low_freq_factor is None
                    or self.rope_high_freq_factor is None
                    or self.rope_original_max_position_embeddings is None):
                raise ValueError(
                    "rope_type='llama3' requires factor, low_freq_factor, "
                    "high_freq_factor, original_max_position_embeddings"
                )
            rope_spec = specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.LLAMA3,
                llama3_extra=specs.Llama3RoPEParams(
                    factor=float(self.rope_factor),
                    low_freq_factor=float(self.rope_low_freq_factor),
                    high_freq_factor=float(self.rope_high_freq_factor),
                    original_context_length=int(
                        self.rope_original_max_position_embeddings
                    ),
                ),
            )
        else:
            rope_spec = specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            )
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=self.attention_bias,
            k_bias=self.attention_bias,
            v_bias=self.attention_bias,
            o_bias=self.attention_bias,
            rope=rope_spec,
        )
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=self.mlp_bias,
            up_bias=self.mlp_bias,
            down_bias=self.mlp_bias,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
