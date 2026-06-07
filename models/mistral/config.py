"""Mistral 7B model config + adapter to api specs.

Verified against `transformers.MistralConfig` and HF config.json for
`mistralai/Mistral-7B-v0.3`.

Key Mistral 7B facts (verified against modeling_mistral.py):
- GQA: n_q=32, n_kv=8, head_dim=128 (4096/32 = 128 = explicit).
- RMSNorm STANDARD_W (modeling_mistral.py:182-199, identical to Llama).
- SwiGLU FFN (modeling_mistral.py:35-48), gate/up/down with bias=False.
- No QK-norm.
- No attention biases (hard-coded bias=False in MistralAttention init).
- RoPE: SPLIT_HALF basis, theta=1_000_000 (v0.3 raised from v0.1's 10000),
  rope_type="default" — NO LLAMA3 scaling.
- Sliding window:
    v0.1/v0.2: sliding_window=4096 (SWA in attention)
    v0.3: sliding_window=None (Mistral dropped SWA for v0.3 — official release).
  HF source path: modeling_mistral.py:172 (`sliding_window=getattr(
  self.config, "sliding_window", None)` — None means causal mask only).
- tie_word_embeddings=False.
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
class MistralConfig:
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
    sliding_window: Optional[int] = None     # v0.3 has None; v0.1/v0.2 = 4096

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "MistralConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # transformers 5.x nests rope_theta under `rope_parameters`.
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 1_000_000.0))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        else:
            rope_theta = 1_000_000.0

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
            sliding_window=hf.get("sliding_window"),
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # Mask kind: when sliding_window is set use SWA; otherwise CAUSAL.
        # v0.3 sets sliding_window=None and uses plain causal masking.
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
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=mask_kind,
            sliding_window=sw_eff,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
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
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
