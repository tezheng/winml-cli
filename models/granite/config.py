"""Granite 3.x dense model config + adapter to api specs.

Verified against `transformers.models.granite.configuration_granite.GraniteConfig`
and HF config.json for `ibm-granite/granite-3.1-2b-base`,
`ibm-granite/granite-3.0-2b-base`.

Key Granite facts (verified against modeling_granite.py):
- Llama-shaped block: PRE-norm RMSNorm STANDARD_W, SPLIT QKV, GQA SwiGLU, no
  QK-norm. (modeling_granite.py:115-179 GraniteAttention; 203-216 GraniteMLP).
- μP scalar overrides:
    * `attention_multiplier` overrides softmax scale (modeling_granite.py:124,
      `self.scaling = config.attention_multiplier`). Not 1/sqrt(Dh).
    * `residual_multiplier` scales BOTH sublayer outputs before residual add
      (modeling_granite.py:273,278).
    * `embedding_multiplier` scales input embeddings (modeling_granite.py:405).
    * `logits_scaling` DIVIDES the final lm_head output (modeling_granite.py:504).
- biases controlled by `attention_bias` and `mlp_bias` (Granite 3.1 base: False).
- tie_word_embeddings=True for 3.1-2B-Base.
- RoPE default, no scaling, theta=5_000_000 for 3.1-2B; partial_rotary_factor=1.0.
- `attention_dropout` ignored at inference (we operate in eval mode).
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
class GraniteConfig:
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
    # μP scalars — the Granite distinguishers vs vanilla Llama.
    attention_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    embedding_multiplier: float = 1.0
    logits_scaling: float = 1.0
    attention_bias: bool = False
    mlp_bias: bool = False

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "GraniteConfig":
        # transformers 5.x emits `dtype`; older 4.x emits `torch_dtype`.
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # transformers 5.x nests rope_theta under `rope_parameters`.
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 10_000.0))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        else:
            rope_theta = 10_000.0

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
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            attention_multiplier=float(hf.get("attention_multiplier", 1.0)),
            residual_multiplier=float(hf.get("residual_multiplier", 1.0)),
            embedding_multiplier=float(hf.get("embedding_multiplier", 1.0)),
            logits_scaling=float(hf.get("logits_scaling", 1.0)),
            attention_bias=bool(hf.get("attention_bias", False)),
            mlp_bias=bool(hf.get("mlp_bias", False)),
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
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
            # μP: attention_multiplier replaces 1/sqrt(head_dim).
            attn_scale=self.attention_multiplier,
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
            # μP residual scaling — applied inside DecoderBlock.forward.
            residual_scale=self.residual_multiplier,
        )
