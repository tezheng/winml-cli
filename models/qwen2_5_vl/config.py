"""Qwen2.5-VL model config — LM-decoder portion only.

Verified against `transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py`
and the publicly-loaded `Qwen/Qwen2.5-VL-3B-Instruct` config.

Qwen2.5-VL architecture summary:
- Vision encoder: dynamic-resolution ViT with M-RoPE on (H, W) — OUT OF
  SCOPE for B8.
- Vision connector: PatchMerger (4-tap mean pool + linear). OUT OF SCOPE.
- LM decoder backbone: **Qwen2.5** (same shape as Qwen2 with M-RoPE).
  For Qwen2.5-VL-3B-Instruct the inner config is:
    hidden_size=2048, num_attention_heads=16, num_kv_heads=2 (GQA),
    head_dim=128, intermediate_size=11008, num_hidden_layers=36,
    rope_theta=1_000_000, mrope_section=[16, 24, 24],
    rms_norm_eps=1e-6, vocab=151936, tie_word_embeddings=False
    (3B-Instruct has its own LM head).

Key Qwen2.5 facts (verified against modeling_qwen2_5_vl.py L609-696):
- QKV projections HAVE biases (q/k/v_proj `bias=True`); o_proj `bias=False`
  — same as Qwen2.
- NO QK-norm (predates Qwen3's PRE-RoPE QK-norm).
- RoPE basis: SPLIT_HALF rotate_half pairing, but applied via M-RoPE
  channel-stitching (apply_multimodal_rotary_pos_emb at L564-606). The
  mrope_section field of len 3 (T/H/W) splits cos/sin into 6 bands
  (3 axes * 2 halves for rotate_half) and cyclically picks i%3.
- For text-only tokens, the 3 position-axis components are all equal —
  in which case M-RoPE reduces to standard 1D RoPE since each output
  channel just samples ANY of the 3 (equal) tables. This is the property
  that lets text-only inference work without M-RoPE-specific behavior.
- RMSNorm STANDARD_W, SwiGLU FFN (no biases on gate/up/down).

Source citations:
- modeling_qwen2_5_vl.py:485-545 (Qwen2_5_VLRotaryEmbedding — inv_freq
  expanded over (3, B, S) position_ids).
- modeling_qwen2_5_vl.py:564-606 (apply_multimodal_rotary_pos_emb —
  mrope_section channel stitching).
- modeling_qwen2_5_vl.py:609-696 (Qwen2_5_VLAttention — Qwen2-style QKV
  with biases).
- modeling_qwen2_5_vl.py:699-764 (Qwen2_5_VLDecoderLayer — PRE-norm
  RMSNorm + Qwen2MLP).
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
class Qwen2_5_VLConfig:
    """LM-decoder config for Qwen2.5-VL (mirrors HF text_config sub-config)."""
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rope_theta: float
    mrope_section: tuple[int, ...]
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    sliding_window: Optional[int] = None

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Qwen2_5_VLConfig":
        """Accepts either a flat Qwen2_5_VLTextConfig dict OR a top-level
        Qwen2_5_VLConfig dict that contains a nested ``text_config``."""
        if "text_config" in hf and isinstance(hf["text_config"], dict):
            top = hf
            inner = hf["text_config"]
        else:
            top = hf
            inner = hf
        dt_raw = inner.get("dtype", inner.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        elif dt_raw is None:
            dtype = torch.float32
        else:
            dtype = _TORCH_DTYPE_MAP.get(str(dt_raw), torch.float32)

        rp = inner.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 1_000_000.0))
            mrope_section_raw = rp.get("mrope_section", [16, 24, 24])
        else:
            rope_theta = float(inner.get("rope_theta", 1_000_000.0))
            mrope_section_raw = inner.get("mrope_section", [16, 24, 24])
        mrope_section = tuple(int(x) for x in mrope_section_raw)

        head_dim = inner.get("head_dim")
        if head_dim is None:
            head_dim = inner["hidden_size"] // inner["num_attention_heads"]

        use_sw = bool(inner.get("use_sliding_window", False))
        sw_raw = inner.get("sliding_window")
        sliding_window = int(sw_raw) if (use_sw and sw_raw is not None) else None

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
            mrope_section=mrope_section,
            rms_norm_eps=float(inner.get("rms_norm_eps", 1e-6)),
            vocab_size=int(inner["vocab_size"]),
            max_position_embeddings=int(inner["max_position_embeddings"]),
            tie_word_embeddings=bool(
                top.get("tie_word_embeddings",
                        inner.get("tie_word_embeddings", False))
            ),
            dtype=dtype,
            sliding_window=sliding_window,
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # All Qwen2.5-VL-3B-Instruct layers are full_attention (no SWA).
        mask_kind = types.MaskKind.CAUSAL
        sw_eff: Optional[int] = None
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=mask_kind,
            sliding_window=sw_eff,
            # Qwen2.5-VL inherits Qwen2's QKV biases (modeling_qwen2_5_vl.py:641-644).
            q_bias=True, k_bias=True, v_bias=True, o_bias=False,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
                mrope_section=self.mrope_section,
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
