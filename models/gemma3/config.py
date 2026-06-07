"""Gemma 3 model config + adapter to api specs.

Verified against `transformers.Gemma3TextConfig` and the public config of
`unsloth/gemma-3-1b-it` (mirror of `google/gemma-3-1b-it`, license-gated).

Key Gemma 3 architectural facts (verified at modeling_gemma3.py and
configuration_gemma3.py):

- Sandwich norm (same as Gemma 2/4) with ONE_PLUS_W RMSNorm. Source
  modeling_gemma3.py:128-145, L385-396 (Gemma3DecoderLayer.__init__),
  L398-428 (forward).
- 5:1 SWA:full layer alternation by default. Pattern: `(i+1) %
  sliding_window_pattern == 0 → full`. For 1B with pattern=6: full layers
  at i = 5, 11, 17, 23. Source: configuration_gemma3.py:109-115.
  For 4B with pattern=6: same rule applies — every 6th layer global.
- Dual rope_theta per layer-type. Local layers (sliding) use
  `rope_local_base_freq` (default 10_000); global layers (full) use
  `rope_theta` (default 1_000_000). Source: configuration_gemma3.py:127-150,
  modeling_gemma3.py:148-225 (per-layer-type RoPE module).
- QK-norm PER_HEAD_DH, PRE_ROPE. q_norm and k_norm are
  `Gemma3RMSNorm(head_dim, eps)` — ONE_PLUS_W. Applied AFTER reshape+
  transpose to [B, H, S, Dh] and BEFORE apply_rotary_pos_emb. Source L337-356.
- NO QK-norm fixed-scale absorb (Gemma 4 ONLY uses that; Gemma 3 has plain
  QK-norm). Source L317 sets `scaling = config.query_pre_attn_scalar**-0.5`
  unconditionally.
- NO partial RoPE — full head_dim rotation. Source L197-206 uses the full
  head_dim in the inv_freq denominator. So Gemma 3 does NOT regress B0.5's
  partial-RoPE IR; it just uses partial_rotary_factor=1.0 (the IR default).
- attn_logit_softcap=None and final_logit_softcap=None in 1B and 4B configs.
  Confirmed in unsloth/gemma-3-1b-it config.json. Source: HF
  configuration_gemma3.py:99-100.
- GeGLU FFN with `gelu_pytorch_tanh` activation. Source L112-125.
- attn_scale = query_pre_attn_scalar**-0.5 (same as Gemma 2). Source L317.
- sliding_window from config (512 for 1B-it). Applied only to SWA layers.
  Source L334.
- Embedding scale = sqrt(hidden_size). Source L498-501.
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
class Gemma3Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta_local: float       # sliding-layer base theta (10_000 default)
    rope_theta_global: float      # full-layer base theta (1_000_000 default)
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    sliding_window: int
    sliding_window_pattern: int   # 6 = every 6th layer is full
    query_pre_attn_scalar: int
    attn_logit_softcap: Optional[float]
    final_logit_softcap: Optional[float]
    attention_bias: bool = False
    layer_types: Optional[tuple[str, ...]] = None

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Gemma3Config":
        # Some Gemma 3 configs nest text under text_config (multimodal); accept
        # either flat or nested.
        if "text_config" in hf and isinstance(hf["text_config"], dict):
            hf = hf["text_config"]

        dt_raw = hf.get("dtype", hf.get("torch_dtype", "bfloat16"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.bfloat16)

        # Dual theta:
        # - Newer HF (5.x) uses `rope_parameters = {"sliding_attention": {...},
        #   "full_attention": {...}}` with `rope_theta` inside each.
        # - Older flat config uses `rope_local_base_freq` + `rope_theta`.
        rope_params = hf.get("rope_parameters")
        if isinstance(rope_params, dict) and isinstance(rope_params.get("sliding_attention"), dict):
            rope_theta_local = float(
                rope_params["sliding_attention"].get("rope_theta", 10_000.0)
            )
            rope_theta_global = float(
                rope_params.get("full_attention", {}).get("rope_theta", 1_000_000.0)
            )
        else:
            rope_theta_local = float(hf.get("rope_local_base_freq", 10_000.0))
            rope_theta_global = float(hf.get("rope_theta", 1_000_000.0))

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
            rope_theta_local=rope_theta_local,
            rope_theta_global=rope_theta_global,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", True)),
            dtype=dtype,
            sliding_window=int(hf.get("sliding_window", 4096)),
            sliding_window_pattern=int(hf.get("sliding_window_pattern", 6)),
            query_pre_attn_scalar=int(hf.get("query_pre_attn_scalar", 256)),
            attn_logit_softcap=hf.get("attn_logit_softcapping"),
            final_logit_softcap=hf.get("final_logit_softcapping"),
            attention_bias=bool(hf.get("attention_bias", False)),
            layer_types=layer_types,
        )

    def layer_type(self, layer_idx: int) -> str:
        """Return "sliding_attention" or "full_attention" for the given layer.

        Default rule from configuration_gemma3.py:111-115:
            "sliding_attention" if (i+1) % pattern else "full_attention"

        So for pattern=6: full layers at i = 5, 11, 17, 23, ...
        """
        if self.layer_types is not None:
            return self.layer_types[layer_idx]
        return (
            "sliding_attention"
            if bool((layer_idx + 1) % self.sliding_window_pattern)
            else "full_attention"
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        """Per-layer DecoderBlockSpec.

        Sliding layers: SWA mask, rope_theta_local, sliding_window from config.
        Full layers:    CAUSAL mask, rope_theta_global, sliding_window=None.

        Both share: QK-norm PER_HEAD_DH PRE_ROPE ONE_PLUS_W, sandwich norm,
        GeGLU, attn_scale = query_pre_attn_scalar**-0.5, no softcap on
        attention or final logits (1B and 4B configs both null).
        """
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        )
        lt = self.layer_type(layer_idx)
        is_sliding = (lt == "sliding_attention")
        mask_kind = types.MaskKind.SWA if is_sliding else types.MaskKind.CAUSAL
        sw = self.sliding_window if is_sliding else None
        rope_theta = self.rope_theta_local if is_sliding else self.rope_theta_global

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
            logit_softcap=self.attn_logit_softcap,    # None in 1B / 4B
            qk_norm=norm_spec,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
            rope=specs.RoPESpec(
                base_theta=rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
                # Gemma 3 uses FULL head_dim rotation — no partial RoPE.
                partial_rotary_factor=1.0,
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
            final_logit_softcap=self.final_logit_softcap,    # None in 1B/4B
        )
