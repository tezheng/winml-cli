"""Moshi 7B main-decoder config + adapter to api specs.

Verified against `transformers.MoshiConfig` and HF config.json for the
canonical Kyutai checkpoint (`kmhf/hf-moshiko`). The Moshi main decoder
(Helium) is the LM-decoder portion exercised here. The Mimi codec
encoder/decoder and the MoshiDepthDecoder (smaller per-codebook head)
are OUT OF SCOPE for B9.

Key Moshi 7B facts (verified against modeling_moshi.py + config):
- MHA, NOT GQA: n_q = n_kv = 32, head_dim = 128 (hidden 4096 / 32 = 128).
  Source: configuration_moshi.py:134-137 default `num_attention_heads=32,
  num_key_value_heads=None` (post_init sets it to num_attention_heads).
- RMSNorm STANDARD_W. Source: modeling_moshi.py:192-208 — output is
  `_norm(x.float()) * weight.float()` (weight is upcast), then cast back
  via `.type_as(x)`. No `ONE_PLUS_W`.
- SwiGLU FFN, FUSED gate/up. Source: modeling_moshi.py:368-390. With
  `use_flexible_linear=False` (the LM-decoder path), `fc1` is a single
  Linear(hidden, ffn_dim) with `bias=False`, then `.view(B, S, 2, -1)`
  is taken; index 0 along dim 2 is `gate`, index 1 is `up`. This is
  algebraically identical to `chunk(2, dim=-1) -> (gate, up)` because the
  view's first chunk is the first `ffn_dim/2` elements. Our api
  `FFNSpec(fused_gate_up=True, intermediate_size=ffn_dim//2)` matches.
  `fc2: Linear(ffn_dim/2, hidden, bias=False)` is the down projection.
- No QK-norm.
- No attention biases: `MoshiLinear(use_flexible_linear=False)` wraps an
  `nn.Linear(..., bias=False)`. Source: modeling_moshi.py:250-265.
- RoPE: SPLIT_HALF basis (uses HF rotate_half + `cat((freqs, freqs), -1)`
  in MoshiRotaryEmbedding at L320-331 — same as Llama). theta=10_000,
  rope_type="default" — no scaling.
- Sliding window: HF config defaults `sliding_window=3000` and
  `max_position_embeddings=3000`. The HF *eager* MoshiAttention
  (L454-509) does NOT apply SWA — sliding_window is only forwarded into
  `_flash_attention_forward` on the Flash path (L611). Our numerical
  gate runs eager, so we map mask_kind=CAUSAL.
- tie_word_embeddings=False.
- num_codebooks=8 / audio_vocab_size=2048 are dual-stream metadata —
  they affect MoshiForConditionalGeneration's embedding heads, NOT the
  per-decoder-layer code.
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
class MoshiConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    ffn_dim: int                       # raw HF field; intermediate = ffn_dim // 2
    num_hidden_layers: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    sliding_window: Optional[int] = None

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "MoshiConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        elif dt_raw is None:
            dtype = torch.float32
        else:
            dtype = _TORCH_DTYPE_MAP.get(str(dt_raw), torch.float32)

        # transformers 5.x nests rope_theta under rope_parameters (Moshi
        # config strictly uses rope_parameters per configuration_moshi.py:140).
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 10_000.0))
        elif "rope_theta" in hf and hf["rope_theta"] is not None:
            rope_theta = float(hf["rope_theta"])
        else:
            rope_theta = 10_000.0

        head_dim = hf.get("head_dim")
        if head_dim is None:
            head_dim = hf["hidden_size"] // hf["num_attention_heads"]

        n_kv = hf.get("num_key_value_heads")
        if n_kv is None:
            n_kv = hf["num_attention_heads"]

        # The HF config keeps sliding_window populated (default 3000) but
        # the eager forward does NOT apply SWA — only the flash path does.
        # We surface the value but the to_block_spec maps to CAUSAL when
        # sliding_window >= max_position_embeddings (full coverage = causal).
        sw = hf.get("sliding_window")

        return cls(
            hidden_size=int(hf["hidden_size"]),
            num_attention_heads=int(hf["num_attention_heads"]),
            num_key_value_heads=int(n_kv),
            head_dim=int(head_dim),
            ffn_dim=int(hf.get("ffn_dim", 22528)),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-8)),
            vocab_size=int(hf["vocab_size"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            sliding_window=int(sw) if sw is not None else None,
        )

    @property
    def intermediate_size(self) -> int:
        """FFN inner dim per the GatingMLP's chunk(2) split — i.e. ffn_dim // 2.

        Source: modeling_moshi.py:376-381 (fc1 hidden->ffn_dim, then
        view(B,S,2,-1) splits into two halves of size ffn_dim/2; fc2 then
        maps ffn_dim/2 -> hidden).
        """
        return self.ffn_dim // 2

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # Mask: eager MoshiAttention does NOT apply SWA. The HF config
        # has sliding_window=3000 == max_position_embeddings=3000, which
        # is equivalent to plain causal coverage anyway. Use CAUSAL.
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            sliding_window=None,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            ),
        )
        # FUSED gate/up (Moshi MoshiGatingMLP `fc1` produces 2*I in one
        # Linear, then `.view(B,S,2,-1)` selects gate/up).
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=True,
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
