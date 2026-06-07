"""Phi-3-mini-4k-instruct model config + adapter to api specs.

Verified against `transformers.models.phi3.configuration_phi3.Phi3Config` and
HF config.json for `microsoft/Phi-3-mini-4k-instruct`.

Key Phi-3 mini 4k facts (verified against modeling_phi3.py):
- Llama-shaped block (PRE-norm RMSNorm STANDARD_W) but with FUSED QKV + FUSED
  gate_up — modeling_phi3.py:222-241 (qkv_proj), 49-64 (Phi3MLP gate_up_proj).
- MHA for the 4k variant (n_q=n_kv=32, head_dim=96).
- Default RoPE θ=10000, partial_rotary_factor=1.0, NO LongRoPE for the 4k
  variant (rope_scaling=null in config).
- Sliding window 2047 (Phi-3 uses SWA for ALL layers, not alternating).
- No QK-norm.
- No biases (attention_bias=False, mlp_bias=False).
- tie_word_embeddings=False.
- resid_pdrop=0 — dropout disabled at inference (we use eval mode).
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
class Phi3MiniConfig:
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
    original_max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    sliding_window: Optional[int] = None
    partial_rotary_factor: float = 1.0
    attention_bias: bool = False
    mlp_bias: bool = False
    # LongRoPE — None for Phi-3-mini-4k (rope_scaling=null).
    rope_type: str = "default"  # "default" | "longrope"
    longrope_short_factor: Optional[tuple[float, ...]] = None
    longrope_long_factor: Optional[tuple[float, ...]] = None

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Phi3MiniConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # Read rope_parameters (transformers 5.x) or fall back to flat
        # rope_theta + rope_scaling (transformers 4.x).
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 10_000.0))
            partial_rotary_factor = float(rp.get("partial_rotary_factor", 1.0))
            rope_type = rp.get("rope_type", "default")
            short_factor = rp.get("short_factor")
            long_factor = rp.get("long_factor")
        else:
            rope_theta = float(hf.get("rope_theta", 10_000.0))
            partial_rotary_factor = float(hf.get("partial_rotary_factor", 1.0))
            rs = hf.get("rope_scaling")
            if isinstance(rs, dict):
                rope_type = rs.get("type", rs.get("rope_type", "default"))
                # Map legacy types ("su", "yarn") to "longrope" per
                # configuration_phi3.py:103-104.
                if rope_type in ("su", "yarn"):
                    rope_type = "longrope"
                short_factor = rs.get("short_factor")
                long_factor = rs.get("long_factor")
            else:
                rope_type = "default"
                short_factor = None
                long_factor = None

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
            original_max_position_embeddings=int(
                hf.get("original_max_position_embeddings",
                       hf["max_position_embeddings"])
            ),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            sliding_window=hf.get("sliding_window"),
            partial_rotary_factor=partial_rotary_factor,
            attention_bias=bool(hf.get("attention_bias", False)),
            mlp_bias=bool(hf.get("mlp_bias", False)),
            rope_type=rope_type,
            longrope_short_factor=(
                tuple(float(x) for x in short_factor) if short_factor else None
            ),
            longrope_long_factor=(
                tuple(float(x) for x in long_factor) if long_factor else None
            ),
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # RoPE — default for Phi-3 mini 4k, LongRoPE for Phi-4-mini etc.
        if self.rope_type == "longrope":
            if (self.longrope_short_factor is None
                    or self.longrope_long_factor is None):
                raise ValueError(
                    "rope_type='longrope' requires short_factor and long_factor"
                )
            # attention_factor per modeling_rope_utils.py:_compute_longrope_parameters
            # lines 529-537.
            import math
            factor = self.max_position_embeddings / self.original_max_position_embeddings
            if factor <= 1.0:
                attention_factor = 1.0
            else:
                attention_factor = math.sqrt(
                    1.0 + math.log(factor) / math.log(self.original_max_position_embeddings)
                )
            rope_spec = specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.LONGROPE,
                longrope_extra=specs.LongRoPEParams(
                    short_factor=self.longrope_short_factor,
                    long_factor=self.longrope_long_factor,
                    original_max_position_embeddings=self.original_max_position_embeddings,
                    attention_factor=attention_factor,
                ),
                partial_rotary_factor=self.partial_rotary_factor,
            )
        else:
            rope_spec = specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
                partial_rotary_factor=self.partial_rotary_factor,
            )
        if self.sliding_window is not None:
            mask_kind = types.MaskKind.SWA
            sw_eff = self.sliding_window
        else:
            mask_kind = types.MaskKind.CAUSAL
            sw_eff = None
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.FUSED,           # ← Phi-3 fused QKV
            mask_kind=mask_kind,
            sliding_window=sw_eff,
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            rope=rope_spec,
        )
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=True,                         # ← Phi-3 fused gate_up
            gate_bias=self.mlp_bias, up_bias=self.mlp_bias, down_bias=self.mlp_bias,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
