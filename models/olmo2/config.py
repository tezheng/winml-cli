"""OLMo 2 model config + adapter to api specs.

Verified against `transformers.Olmo2Config` and the public configs of
`allenai/OLMo-2-0425-1B` and `allenai/OLMo-2-1124-7B`.

Key OLMo 2 architectural facts (verified at modeling_olmo2.py):
- POST-norm decoder block. ``Olmo2DecoderLayer.forward`` (L295-333) skips
  pre-attn and pre-FFN norms entirely: ``residual = x; x, _ = attn(x);
  x = post_attention_layernorm(x); x = residual + x`` and the same for MLP.
- QK-norm: ``Olmo2RMSNorm(num_attention_heads * head_dim, eps)`` and
  ``Olmo2RMSNorm(num_key_value_heads * head_dim, eps)``. Weight shape is
  ``[H * Dh]`` — our QKNormShape.FULL_HDH. Applied to the flattened
  ``[B, S, H*Dh]`` tensor BEFORE the ``view(*, H, Dh)`` reshape; algebraic
  equivalence with reshape-first then norm-last holds. Source: L231-249.
- QK-norm phase: PRE_ROPE. The order in ``Olmo2Attention.forward`` is
  proj -> q_norm/k_norm -> view -> transpose -> apply_rotary_pos_emb.
  Source: L242-254.
- No biases on q/k/v/o projections (``attention_bias=False`` by default,
  always set so in the public 1B and 7B configs).
- SwiGLU FFN (gate/up/down projections, activation=silu). Source: L279-292.
- Vanilla full RoPE on a single dim, SPLIT_HALF basis, rope_theta from config
  (500_000 for 1B and 7B). Source: L71-132 and L172-202.
- RMSNorm STANDARD_W mode. Source: L50-65.
- Final model norm (``self.norm`` in Olmo2Model) is a STANDARD_W RMSNorm
  applied AFTER the last layer's residual add. Source: L366, L422.
- BF16 residual stream is the recommended precision per the OLMo 2 paper.
  Our tests run in FP32 for the gate (atol=5e-4) — we document this in
  layer.md but the runtime dtype is whatever the caller passes via
  ``Olmo2Config.dtype``.
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
class Olmo2Config:
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
    attention_bias: bool = False

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Olmo2Config":
        # transformers 5.x emits `dtype`; older 4.x emits `torch_dtype`. Accept either.
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # transformers 5.x nests rope_theta under `rope_parameters`; older
        # flat `rope_theta` key still accepted.
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
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            attention_bias=bool(hf.get("attention_bias", False)),
        )

    def to_block_spec(self) -> specs.DecoderBlockSpec:
        """Build a POST-norm DecoderBlockSpec.

        Verified against modeling_olmo2.py:
        - attn_norm_position = POST (no pre-norm).  Source L302-326.
        - QK-norm: FULL_HDH shape, PRE_ROPE phase. Source L231-254.
        - rope_theta from config; SPLIT_HALF basis. Source L109-118.
        - SwiGLU FFN, no biases on projections. Source L279-292.
        - mask = CAUSAL.
        """
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # QK-norm uses the SAME eps and STANDARD_W mode, but its weight shape is
        # FULL_HDH (n_q_heads * head_dim or n_kv_heads * head_dim, depending on
        # which side). The spec carries one NormSpec; the api.norm.QKNorm
        # constructor consumes shape + n_heads at build time.
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            qk_norm=norm_spec,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.FULL_HDH,
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
            attn_norm_position=types.NormPosition.POST,
            ffn_norm_position=types.NormPosition.POST,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            # POST-norm: pre_*_norm explicitly None; post_*_norm carries the spec.
            pre_attn_norm=None, post_attn_norm=norm_spec,
            pre_ffn_norm=None, post_ffn_norm=norm_spec,
        )
