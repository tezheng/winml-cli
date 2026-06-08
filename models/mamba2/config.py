"""Mamba-2 model config + adapter to api specs.

Mirrors `transformers.models.mamba2.configuration_mamba2.Mamba2Config`.
Verified against `state-spaces/mamba2-2.7b/config.json` and the
strict-typed defaults in `configuration_mamba2.py:18-89`.

Key Mamba-2 facts:
- Token mixer is `Mamba2Mixer`, not attention.
- No FFN sublayer — each block has only `norm + mixer + residual`.
  Source: `modeling_mamba2.py:617-640`.
- `num_heads * head_dim == hidden_size * expand == d_inner`
  (validated in `Mamba2Config.validate_architecture()` lines 97-103).
- `num_heads % n_groups == 0` (HF code does `repeat_interleave` by
  `num_heads // n_groups` at lines 510-511).
- Final RMSNorm `norm_f` after the last block; LM head is untied
  (`tie_word_embeddings: False`).

The reference model `state-spaces/mamba2-2.7b` has:
  hidden_size=2560, num_heads=80, head_dim=64, n_groups=1, state_size=128,
  conv_kernel=4, expand=2, num_hidden_layers=64, vocab_size=50288 (rounded
  up from 50257 for kernel alignment per Mamba-1 convention).
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
class Mamba2Config:
    """Mamba-2 config (subset of fields required to build a single decoder block)."""
    hidden_size: int
    num_heads: int
    head_dim: int
    state_size: int               # d_state
    conv_kernel: int              # d_conv
    expand: int
    n_groups: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    use_bias: bool                # bias on in_proj / out_proj
    use_conv_bias: bool           # bias on conv1d
    residual_in_fp32: bool
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    time_step_limit: tuple[float, float]
    chunk_size: int
    vocab_size: int
    tie_word_embeddings: bool
    dtype: torch.dtype

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Mamba2Config":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # time_step_limit can be a 2-tuple or a list; some configs omit it
        # entirely and rely on the HF default `(0.0, float("inf"))`.
        # Source: configuration_mamba2.py:78.
        tsl_raw = hf.get("time_step_limit", (0.0, float("inf")))
        if isinstance(tsl_raw, (list, tuple)) and len(tsl_raw) == 2:
            time_step_limit = (float(tsl_raw[0]), float(tsl_raw[1]))
        else:
            time_step_limit = (0.0, float("inf"))

        return cls(
            hidden_size=hf["hidden_size"],
            num_heads=hf.get("num_heads", 128),
            head_dim=hf.get("head_dim", 64),
            state_size=hf.get("state_size", 128),
            conv_kernel=hf.get("conv_kernel", 4),
            expand=hf.get("expand", 2),
            n_groups=hf.get("n_groups", 8),
            num_hidden_layers=hf["num_hidden_layers"],
            layer_norm_epsilon=float(hf.get("layer_norm_epsilon", 1e-5)),
            use_bias=bool(hf.get("use_bias", False)),
            use_conv_bias=bool(hf.get("use_conv_bias", True)),
            residual_in_fp32=bool(hf.get("residual_in_fp32", True)),
            time_step_min=float(hf.get("time_step_min", 0.001)),
            time_step_max=float(hf.get("time_step_max", 0.1)),
            time_step_floor=float(hf.get("time_step_floor", 1e-4)),
            time_step_limit=time_step_limit,
            chunk_size=hf.get("chunk_size", 256),
            vocab_size=hf["vocab_size"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
        )

    @property
    def d_inner(self) -> int:
        return self.hidden_size * self.expand

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        # Mamba2RMSNorm is a STANDARD_W RMSNorm with eps from config.
        # Source: modeling_mamba2.py:600-614.
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS,
            eps=self.layer_norm_epsilon,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # SSMSpec inside SSDSpec carries the per-head defaults.
        ssm_spec = specs.SSMSpec(
            d_state=self.state_size,
            d_conv=self.conv_kernel,
            d_inner=self.d_inner,
            expand_factor=self.expand,
            dt_min=self.time_step_min,
            dt_max=self.time_step_max,
            dt_init_floor=self.time_step_floor,
            conv_bias=self.use_conv_bias,
            bias=self.use_bias,
            activation=types.Activation.SILU,
        )
        ssd_spec = specs.SSDSpec(
            base=ssm_spec,
            chunk_size=self.chunk_size,
            headdim=self.head_dim,
            ngroups=self.n_groups,
            n_heads=self.num_heads,
            time_step_limit_low=self.time_step_limit[0],
            time_step_limit_high=self.time_step_limit[1],
            layer_norm_epsilon=self.layer_norm_epsilon,
        )
        # Placeholder FFN spec — ignored due to skip_ffn=True.
        placeholder_ffn = specs.FFNSpec(
            intermediate_size=self.hidden_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=ssd_spec,
            channel_mixer=placeholder_ffn,
            pre_attn_norm=norm_spec,
            skip_ffn=True,
        )
