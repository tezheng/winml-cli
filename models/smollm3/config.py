"""SmolLM3 model config + adapter to api specs (per-layer dispatch).

Verified against `transformers.SmolLM3Config` and HF config.json for
`HuggingFaceTB/SmolLM3-3B`.

Key SmolLM3-3B facts (verified against modeling_smollm3.py + configuration_smollm3.py):
- GQA: n_q=16, n_kv=4, head_dim=128 (derived from hidden=2048 / n_q=16).
- RMSNorm STANDARD_W (modeling_smollm3.py:262-279, identical to Llama).
- SwiGLU FFN (modeling_smollm3.py:282-295).
- No QK-norm.
- No biases (attention_bias=False, mlp_bias=False).
- RoPE: SPLIT_HALF, theta=5_000_000, rope_type="default" — NO LLAMA3 scaling.

NOVEL: per-layer NoPE dispatch.
- `no_rope_layers[i] == 1`: layer i uses RoPE.
- `no_rope_layers[i] == 0`: layer i is NoPE (skips apply_rotary_pos_emb).
- Default pattern: `(i+1) % no_rope_layer_interval != 0`, with
  no_rope_layer_interval=4 — so NoPE layers are at indices 3, 7, 11, ...
- Verified at `modeling_smollm3.py:211` (`self.use_rope =
  config.no_rope_layers[layer_idx]`) and 233-235 (the `if self.use_rope:`
  guard around `apply_rotary_pos_emb`).

NOT exercised in SmolLM3-3B but supported by the IR:
- Sliding window: `use_sliding_window=False` and `sliding_window=None` in
  SmolLM3-3B. If the future SmolLM3 release enables SWA on the NoPE layers,
  `layer_types[i] == "sliding_attention"` triggers it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Smollm3Config:
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
    # Per-layer NoPE pattern. `no_rope_layers[i] == 1` means layer i uses RoPE;
    # `0` means layer i is NoPE.
    no_rope_layers: tuple[int, ...] = ()
    no_rope_layer_interval: int = 4
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    layer_types: tuple[str, ...] = ()
    attention_bias: bool = False
    mlp_bias: bool = False

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Smollm3Config":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 5_000_000.0))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        else:
            rope_theta = 5_000_000.0

        head_dim = hf.get("head_dim")
        if head_dim is None:
            head_dim = hf["hidden_size"] // hf["num_attention_heads"]

        num_layers = hf["num_hidden_layers"]
        no_rope_interval = int(hf.get("no_rope_layer_interval", 4))
        no_rope = hf.get("no_rope_layers")
        if no_rope is None:
            no_rope = [
                int((i + 1) % no_rope_interval != 0) for i in range(num_layers)
            ]
        layer_types = hf.get("layer_types") or [
            "full_attention" for _ in range(num_layers)
        ]

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
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", True)),
            dtype=dtype,
            no_rope_layers=tuple(no_rope),
            no_rope_layer_interval=no_rope_interval,
            use_sliding_window=bool(hf.get("use_sliding_window", False)),
            sliding_window=hf.get("sliding_window"),
            layer_types=tuple(layer_types),
            attention_bias=bool(hf.get("attention_bias", False)),
            mlp_bias=bool(hf.get("mlp_bias", False)),
        )

    def layer_uses_rope(self, layer_idx: int) -> bool:
        """`no_rope_layers[i] == 1` means RoPE; `0` means NoPE."""
        if layer_idx >= len(self.no_rope_layers):
            return True
        return self.no_rope_layers[layer_idx] == 1

    def to_block_spec(self, layer_idx: int) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # Per-layer RoPE dispatch — NoPE layers have rope=None.
        if self.layer_uses_rope(layer_idx):
            rope_spec: Optional[specs.RoPESpec] = specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            )
        else:
            rope_spec = None

        # SWA dispatch on per-layer `layer_types` (SmolLM3-3B disables SWA).
        lt = (self.layer_types[layer_idx]
              if layer_idx < len(self.layer_types) else "full_attention")
        if (self.use_sliding_window and self.sliding_window is not None
                and lt == "sliding_attention"):
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
            q_bias=self.attention_bias,
            k_bias=self.attention_bias,
            v_bias=self.attention_bias,
            o_bias=self.attention_bias,
            rope=rope_spec,
        )
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=self.mlp_bias,
            up_bias=self.mlp_bias,
            down_bias=self.mlp_bias,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
