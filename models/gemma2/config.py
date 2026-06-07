"""Gemma 2 model config + adapter to api specs.

Verified against `transformers.Gemma2Config` and the public configs of
`unsloth/gemma-2-2b` (mirror of google/gemma-2-2b which is license-gated).

Key Gemma 2 architectural facts (verified at modeling_gemma2.py):
- Sandwich norm — pre-attn, post-attn, pre-FFN, post-FFN. Source L309-313
  (Gemma2DecoderLayer.__init__) and L324-344 (forward).
- ONE_PLUS_W RMSNorm: weight initialised to ZEROS; forward computes
  ``output * (1 + weight)``; final cast to input dtype. Source L49-65.
- Alternating SWA / full attention every other layer. Default pattern:
  `"sliding_attention" if (i+1) % 2 else "full_attention"`. So layer 0 is
  SWA, layer 1 is full, layer 2 is SWA, ... Source: configuration_gemma2.py
  L95-98.
- `attn_logit_softcapping=50.0` (model-wide). Applied AFTER matmul*scale
  and BEFORE the mask add: `attn = tanh(attn/cap)*cap`. Source L212-217
  (eager_attention_forward) and L293 (softcap arg in attention_interface
  call).
- `final_logit_softcapping=30.0`. Applied to lm_head output. Source
  L537-540 (Gemma2ForCausalLM.forward).
- attn_scale = `query_pre_attn_scalar ** -0.5` — overrides head_dim**-0.5.
  For 2B/9B these may differ from head_dim. Source L240.
- GeGLU FFN with `gelu_pytorch_tanh` activation. Source L69-82.
- Embedding scale = `hidden_size ** 0.5`. Source L399.
- NO QK-norm in Gemma 2 (verify: Gemma2Attention.__init__ L244-258 has
  no q_norm / k_norm).
- Vanilla RoPE single-theta (default 10000) at FULL head_dim. Source L85-147.

NOT YET CAPTURED in this config (carried as DecoderBlockSpec
`final_logit_softcap` for model-level assembly, but the per-layer block
itself doesn't consume final_logit_softcap):
- Per-layer-type RoPE / scaling differences (none for Gemma 2; Gemma 3 splits
  rope_theta per layer type, Gemma 2 does NOT).
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
class Gemma2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    sliding_window: int
    query_pre_attn_scalar: int
    attn_logit_softcap: Optional[float]
    final_logit_softcap: Optional[float]
    attention_bias: bool = False
    # Override per-layer dispatch with an explicit list; if None we use the
    # HF default of "sliding_attention" iff (i+1) % 2.
    layer_types: Optional[tuple[str, ...]] = None

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Gemma2Config":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "bfloat16"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.bfloat16)

        if "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        elif (
            isinstance(hf.get("rope_parameters"), dict)
            and "rope_theta" in hf["rope_parameters"]
        ):
            rope_theta = float(hf["rope_parameters"]["rope_theta"])
        else:
            rope_theta = 10000.0

        layer_types = hf.get("layer_types")
        if layer_types is not None:
            layer_types = tuple(layer_types)

        return cls(
            hidden_size=hf["hidden_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf["num_key_value_heads"],
            head_dim=hf.get(
                "head_dim",
                hf["hidden_size"] // hf["num_attention_heads"],
            ),
            intermediate_size=hf["intermediate_size"],
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", True)),
            dtype=dtype,
            sliding_window=int(hf.get("sliding_window", 4096)),
            query_pre_attn_scalar=int(hf.get("query_pre_attn_scalar", 256)),
            attn_logit_softcap=hf.get("attn_logit_softcapping"),
            final_logit_softcap=hf.get("final_logit_softcapping"),
            attention_bias=bool(hf.get("attention_bias", False)),
            layer_types=layer_types,
        )

    def layer_type(self, layer_idx: int) -> str:
        """Return "sliding_attention" or "full_attention" for the given layer.

        Defaults to the HF rule: `(i+1) % 2` → SWA. Source: configuration_gemma2.py:96-98.
        """
        if self.layer_types is not None:
            return self.layer_types[layer_idx]
        return "sliding_attention" if bool((layer_idx + 1) % 2) else "full_attention"

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        """Per-layer DecoderBlockSpec, dispatching on layer type.

        Sliding layers: MaskKind.SWA, sliding_window=self.sliding_window.
        Full layers:    MaskKind.CAUSAL, sliding_window=None.

        Both share: GeGLU FFN, sandwich norm with ONE_PLUS_W RMSNorm,
        attn_scale=query_pre_attn_scalar**-0.5, attn_logit_softcap=50.
        """
        # ONE_PLUS_W matches Gemma2RMSNorm exactly. Source: modeling_gemma2.py:49-65.
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        )
        lt = self.layer_type(layer_idx)
        is_sliding = (lt == "sliding_attention")
        mask_kind = types.MaskKind.SWA if is_sliding else types.MaskKind.CAUSAL
        sw = self.sliding_window if is_sliding else None

        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=mask_kind,
            sliding_window=sw,
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            attn_scale=float(self.query_pre_attn_scalar) ** -0.5,
            logit_softcap=self.attn_logit_softcap,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            ),
        )
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.GELU,
            gate_kind=types.GateKind.GEGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE_AND_POST,
            ffn_norm_position=types.NormPosition.PRE_AND_POST,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
            embedding_scale=self.hidden_size ** 0.5,
            final_logit_softcap=self.final_logit_softcap,
        )
