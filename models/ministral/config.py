"""Ministral model config + adapter to api specs (per-layer SWA dispatch).

Verified against `transformers.MinistralConfig` and HF config.json for
`mistralai/Ministral-8B-Instruct-2410`.

Key Ministral architectural facts (verified at modeling_ministral.py and
configuration_ministral.py):

- GQA: n_q=32, n_kv=8, head_dim=128 (explicit field; not derived). Source:
  configuration_ministral.py:67-72.
- RMSNorm STANDARD_W (modeling_ministral.py:200-217 — identical to Llama;
  the `MinistralRMSNorm` class mirrors `MistralRMSNorm`). eps default 1e-6
  via configuration_ministral.py:76; the published 8B-2410 config sets 1e-5.
- SwiGLU FFN with bias=False (modeling_ministral.py:50-63 — gate/up/down
  Linear without bias).
- No QK-norm (modeling_ministral.py:140-156 — no q_norm/k_norm modules).
- No attention biases (modeling_ministral.py:151-154 — bias=False on
  q/k/v/o_proj, hard-coded to match Mistral parity).
- RoPE: SPLIT_HALF basis (cat-half rotate via modeling_ministral.py:66-96).
  rope_theta varies by release: 8B-2410 uses `rope_theta=100_000_000.0`.

NOVEL: per-layer interleaved SWA dispatch.
- `layer_types[i]` is either "full_attention" or "sliding_attention" (one
  entry per decoder layer). Source: configuration_ministral.py:86-95
  (post_init default = uniform from sliding_window flag) and
  modeling_ministral.py:142 (`self.layer_type = config.layer_types[layer_idx]`).
- For Ministral-8B-Instruct-2410 the published `layer_types` is the
  1:3 pattern `[full, sliding, sliding, sliding] * 9` over 36 layers.
- modeling_ministral.py:155 then encodes the dispatch on the attention
  module: `self.sliding_window = config.sliding_window if self.layer_type
  == "sliding_attention" else None`. Full layers ignore the window.

IMPORTANT — divergence between Mistral and Ministral HF modeling:
- HF Mistral's MistralModel (modeling_mistral.py:372-379) applies ONE mask
  to ALL layers based on `config.sliding_window is None`. It DOES NOT
  honour `layer_types` even if present in the config.
- HF Ministral's MinistralModel (modeling_ministral.py:393-414) builds BOTH
  masks and dispatches per layer via `causal_mask_mapping[layer_types[i]]`.
- The published `Ministral-8B-Instruct-2410` advertises
  `architectures: ["MistralForCausalLM"]` — so AutoModelForCausalLM loads it
  via Mistral modeling, which then ignores the per-layer pattern. The
  per-layer dispatch only takes effect when the user explicitly loads the
  weights via MinistralForCausalLM. This config preserves both modes (the
  IR carries the layer_types tuple even when sliding_window is None) so
  that downstream consumers always see the truth.
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
class MinistralConfig:
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
    # Per-layer SWA dispatch. ``layer_types[i]`` is either
    # ``"full_attention"`` or ``"sliding_attention"``. If empty the dispatch
    # rule degrades to "all sliding when sliding_window is not None" to match
    # configuration_ministral.py:92-95 default.
    layer_types: tuple[str, ...] = ()

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "MinistralConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # transformers 5.x nests rope_theta under `rope_parameters`.
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 100_000_000.0))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        else:
            rope_theta = 100_000_000.0

        head_dim = hf.get("head_dim")
        if head_dim is None:
            head_dim = hf["hidden_size"] // hf["num_attention_heads"]

        num_layers = hf["num_hidden_layers"]
        sw = hf.get("sliding_window")
        layer_types = hf.get("layer_types")
        if layer_types is None:
            # Default rule per configuration_ministral.py:92-95: all layers
            # are sliding when `sliding_window` is set; otherwise all full.
            layer_types = [
                "sliding_attention" if sw is not None else "full_attention"
            ] * num_layers
        else:
            layer_types = list(layer_types)

        return cls(
            hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf.get(
                "num_key_value_heads", hf["num_attention_heads"]
            ),
            head_dim=head_dim,
            intermediate_size=hf["intermediate_size"],
            num_hidden_layers=num_layers,
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            sliding_window=sw,
            layer_types=tuple(layer_types),
        )

    def layer_type(self, layer_idx: int) -> str:
        """Return ``"sliding_attention"`` or ``"full_attention"`` for the
        given layer.

        Source: modeling_ministral.py:142.
        """
        if layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx]
        # Fallback when the tuple is empty / shorter than num_layers (only
        # possible in synthetic configs): respect the global SWA flag.
        return (
            "sliding_attention"
            if self.sliding_window is not None
            else "full_attention"
        )

    def to_block_spec(self, layer_idx: int) -> specs.DecoderBlockSpec:
        """Per-layer DecoderBlockSpec.

        Sliding layers: SWA mask, sliding_window from config.
        Full layers:    CAUSAL mask, sliding_window=None.

        Both share: RMSNorm STANDARD_W eps, RoPE SPLIT_HALF with
        ``rope_theta``, SwiGLU FFN with no biases. Source:
        modeling_ministral.py:140-196.
        """
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        lt = self.layer_type(layer_idx)
        is_sliding = (lt == "sliding_attention")
        # Per modeling_ministral.py:155: sliding_window only applied on
        # sliding-type layers; full layers explicitly disable it.
        if is_sliding and self.sliding_window is not None:
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
