"""Voxtral LM-decoder config + adapter to api specs.

Verified against `transformers.VoxtralConfig` (configuration_voxtral.py:74-131)
and the canonical `mistralai/Voxtral-Mini-3B-2507` config.

The Voxtral text decoder is a Llama-style block (the `text_config.model_type`
defaults to `"llama"`; the canonical 3B checkpoint confirms this). The block
is architecturally identical to Mistral 7B v0.3 — PRE-norm RMSNorm STANDARD_W,
SPLIT QKV, SwiGLU, no biases, no QK-norm, SPLIT_HALF RoPE (theta=1e8).

Canonical Voxtral-Mini-3B-2507 LM-decoder dims:
  hidden_size=3072, num_hidden_layers=30, num_q=32, num_kv=8 (GQA), head_dim=128,
  intermediate_size=8192, rope_theta=1e8, rms_norm_eps=1e-5, vocab=131072,
  max_pos=131072, sliding_window=None, hidden_act='silu',
  text_config.tie_word_embeddings=False (top-level tie_word_embeddings=True
  — that tie connects the LM head to the embed_tokens of the language_model).

Key source citations:
- configuration_voxtral.py:74-131 VoxtralConfig (sub-configs + defaults).
- modeling_voxtral.py:384-481 VoxtralModel.forward (LM is `language_model`,
  which is `AutoModel.from_config(text_config)` — a LlamaModel here).
- The Llama-decoder layer code is the canonical Llama 3 / Mistral block we
  already exercise in models/mistral and models/llama3.
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
class VoxtralConfig:
    """LM-decoder config for Voxtral (mirrors HF text_config sub-config)."""
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
    sliding_window: Optional[int] = None

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "VoxtralConfig":
        """Accepts either a flat LlamaConfig (text_config) dict OR a top-level
        VoxtralConfig dict containing a nested ``text_config``.

        Per HF's VoxtralConfig the TOP-level `tie_word_embeddings` controls
        whether the model's lm_head is tied to the inner language_model's
        embed_tokens (configuration_voxtral.py:113 — defaults to True). The
        INNER text_config has its own `tie_word_embeddings` which is the
        Llama backbone-internal flag; for B9 we surface the inner one
        because the per-layer code doesn't use embed_tokens.
        """
        if "text_config" in hf and isinstance(hf["text_config"], dict):
            inner = hf["text_config"]
        else:
            inner = hf
        dt_raw = inner.get("dtype", inner.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        elif dt_raw is None:
            dtype = torch.float32
        else:
            dtype = _TORCH_DTYPE_MAP.get(str(dt_raw), torch.float32)

        # rope_theta lives nested under `rope_parameters` in transformers 5.x.
        rp = inner.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 1e8))
        elif "rope_theta" in inner and inner["rope_theta"] is not None:
            rope_theta = float(inner["rope_theta"])
        else:
            # Voxtral's `_default_text_config_kwargs["rope_theta"] = 1e8`
            # — configuration_voxtral.py:105.
            rope_theta = 1e8

        head_dim = inner.get("head_dim")
        if head_dim is None:
            head_dim = inner["hidden_size"] // inner["num_attention_heads"]

        sw_raw = inner.get("sliding_window")
        sliding_window = int(sw_raw) if sw_raw is not None else None

        return cls(
            hidden_size=int(inner["hidden_size"]),
            num_attention_heads=int(inner["num_attention_heads"]),
            num_key_value_heads=int(
                inner.get("num_key_value_heads", inner["num_attention_heads"])
            ),
            head_dim=int(head_dim),
            intermediate_size=int(inner["intermediate_size"]),
            num_hidden_layers=int(inner["num_hidden_layers"]),
            rope_theta=rope_theta,
            rms_norm_eps=float(inner.get("rms_norm_eps", 1e-5)),
            vocab_size=int(inner["vocab_size"]),
            max_position_embeddings=int(inner["max_position_embeddings"]),
            tie_word_embeddings=bool(inner.get("tie_word_embeddings", False)),
            dtype=dtype,
            sliding_window=sliding_window,
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # Voxtral-Mini-3B sets sliding_window=None — pure causal masking.
        if self.sliding_window is not None:
            mask_kind = types.MaskKind.SWA
            sw_eff: Optional[int] = self.sliding_window
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
            # Llama-style: NO biases on q/k/v/o.
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
