"""GOT-OCR 2.0 model config — LM-decoder portion only.

Verified against `transformers/models/got_ocr2/configuration_got_ocr2.py` and
the publicly-loaded `stepfun-ai/GOT-OCR-2.0-hf` config.

GOT-OCR 2.0 architecture summary:
- Vision encoder: SAM-style ViT (OUT OF SCOPE for B8 — only the LM decoder
  is reproduced here).
- Vision connector: a single Linear from vision_dim -> text hidden_size
  (`multi_modal_projector` per modeling_got_ocr2.py L535-547).
- LM decoder backbone: **Qwen2** (NOT Qwen3). For the canonical
  stepfun-ai/GOT-OCR-2.0-hf checkpoint the inner config is:
    hidden_size=1024, num_attention_heads=16, num_kv_heads=16,
    head_dim=64, intermediate_size=2816, num_hidden_layers=24,
    rope_theta=1_000_000, rms_norm_eps=1e-6, vocab=151860,
    tie_word_embeddings=True.
- Concat-prefix fusion: vision tokens (image_seq_length=576 in the canonical
  config) are written into the embedding sequence by `get_placeholder_mask`
  at the positions of `image_token_index`; the LM decoder sees a single
  unified [B, S, hidden] tensor where vision and text tokens are
  interleaved. There is NO architectural change in the LM decoder itself
  — concat-prefix is purely an embedding-level fusion.

Key Qwen2 facts (verified against modeling_qwen2.py L186-245):
- QKV projections HAVE biases (q/k/v_proj `bias=True`); o_proj `bias=False`.
- NO QK-norm (Qwen2 predates the Qwen3 QK-norm change).
- RoPE SPLIT_HALF basis (apply_rotary_pos_emb / rotate_half).
- RMSNorm STANDARD_W.
- SwiGLU FFN (gate_proj / up_proj / down_proj, all `bias=False`).
- Sliding window OPTIONAL — for stepfun-ai/GOT-OCR-2.0-hf, sliding_window=None
  (use_sliding_window=False) so the mask is pure causal.

Source citations:
- transformers/models/got_ocr2/configuration_got_ocr2.py:67-129
  (GotOcr2Config; default text_config is a Qwen2 config).
- transformers/models/qwen2/modeling_qwen2.py:186-245 (Qwen2Attention),
  L269-309 (Qwen2DecoderLayer).
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
class GotOcr2Config:
    """LM-decoder config for GOT-OCR 2.0 (mirrors HF text_config sub-config)."""
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
    # Vision-side knobs kept for processor pairing — NOT used by the
    # decoder layer factory (the vision encoder is out of scope).
    image_token_index: int = 151859
    image_seq_length: int = 576

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "GotOcr2Config":
        """Accepts either a flat Qwen2 text_config dict OR a top-level
        GotOcr2Config dict that contains a nested ``text_config``."""
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

        # transformers 5.x: rope_theta lives under rope_parameters.
        rp = inner.get("rope_parameters")
        if isinstance(rp, dict) and "rope_theta" in rp:
            rope_theta = float(rp["rope_theta"])
        elif "rope_theta" in inner:
            rope_theta = float(inner["rope_theta"])
        else:
            rope_theta = 1_000_000.0

        head_dim = inner.get("head_dim")
        if head_dim is None:
            head_dim = inner["hidden_size"] // inner["num_attention_heads"]

        # sliding_window: HF stores None when use_sliding_window=False.
        # Source: modeling_qwen2.py:204 (`sliding_window =
        #   getattr(self.config, 'sliding_window', None)`).
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
            rms_norm_eps=float(inner.get("rms_norm_eps", 1e-6)),
            vocab_size=int(inner["vocab_size"]),
            max_position_embeddings=int(inner["max_position_embeddings"]),
            tie_word_embeddings=bool(
                top.get("tie_word_embeddings",
                        inner.get("tie_word_embeddings", True))
            ),
            dtype=dtype,
            sliding_window=sliding_window,
            image_token_index=int(top.get("image_token_index", 151859)),
            image_seq_length=int(top.get("image_seq_length", 576)),
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        """Build the DecoderBlockSpec for one Qwen2 layer.

        layer_idx is accepted for parity with other configs that vary spec
        per layer (e.g. SmolLM3 NoPE). For GOT-OCR 2.0 / Qwen2 the spec is
        uniform across all layers — sliding-window only kicks in for
        layers >= max_window_layers when use_sliding_window is True, and
        the canonical checkpoint disables it.
        """
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # Per-layer sliding window — only active for layer_idx >=
        # max_window_layers when sliding_window is set. The canonical
        # GOT-OCR-2.0 disables this (sliding_window=None), so all layers
        # are CAUSAL. Source: modeling_qwen2.py:204 + config layer_types.
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
            # Qwen2 has BIASES on Q/K/V; o_proj has NO bias.
            # Source: modeling_qwen2.py:200-203.
            q_bias=True, k_bias=True, v_bias=True, o_bias=False,
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
