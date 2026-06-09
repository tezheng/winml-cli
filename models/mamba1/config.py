"""Mamba-1 model config + adapter to api specs.

Mirrors `transformers.models.mamba.configuration_mamba.MambaConfig`
(verified against `state-spaces/mamba-130m-hf/config.json`).

Mamba-1 key facts:
- expand=2 → d_inner = expand * hidden_size
- dt_rank = ceil(hidden_size / 16) when "auto"
- state_size=16 (vs 128 in Mamba-2)
- conv_kernel=4
- use_bias=False (in_proj/out_proj), use_conv_bias=True (conv1d)
- hidden_act="silu"
- No FFN sublayer — Mamba-1 layers are `norm + mixer + residual` only.

Source: `transformers/models/mamba/configuration_mamba.py:24-103`.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import torch

from api import specs, types


@dataclass(frozen=True)
class Mamba1Config:
    hidden_size: int                # 768 for mamba-130m
    state_size: int                 # 16
    conv_kernel: int                # 4
    expand: int                     # 2
    num_hidden_layers: int          # 24 for mamba-130m
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = False
    use_conv_bias: bool = True
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_rank: int = 0         # 0 → auto = ceil(hidden / 16)
    vocab_size: int = 50280
    dtype: torch.dtype = torch.float32

    @property
    def intermediate_size(self) -> int:
        return self.expand * self.hidden_size

    @property
    def dt_rank(self) -> int:
        if self.time_step_rank > 0:
            return self.time_step_rank
        return math.ceil(self.hidden_size / 16)

    @classmethod
    def mamba_130m(cls) -> "Mamba1Config":
        """state-spaces/mamba-130m-hf."""
        return cls(
            hidden_size=768, state_size=16, conv_kernel=4, expand=2,
            num_hidden_layers=24, vocab_size=50280,
        )

    def to_ssm_spec(self) -> specs.SSMSpec:
        """Mamba-1 SSM spec — per-channel A_log/D, learned dt_proj."""
        return specs.SSMSpec(
            d_state=self.state_size,
            d_conv=self.conv_kernel,
            d_inner=self.intermediate_size,
            expand_factor=self.expand,
            dt_min=self.time_step_min,
            dt_max=self.time_step_max,
            dt_init_floor=self.time_step_floor,
            conv_bias=self.use_conv_bias,
            bias=self.use_bias,
            activation=types.Activation.SILU,
            kind=types.SSMKind.MAMBA1,
            dt_rank=self.dt_rank,
        )

    def to_norm_spec(self) -> specs.NormSpec:
        return specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.layer_norm_epsilon,
        )

    def to_block_spec(self) -> specs.DecoderBlockSpec:
        """Mamba-1 decoder block: RMSNorm + Mamba1Mixer + residual.

        No FFN sublayer — `skip_ffn=True`. Source: modeling_mamba.py:401-424
        (MambaBlock — `norm + mixer + residual`).

        Note: HF Mamba-1 has `residual_in_fp32=True` which means the
        residual add happens in fp32. The api `ops.add` uses standard
        elementwise add in the input dtype; for fp32 weights/activations
        the distinction is moot, and we don't propagate the fp32 hint here.
        """
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=self.to_ssm_spec(),
            channel_mixer=self.to_ssm_spec(),  # unused under skip_ffn
            pre_attn_norm=self.to_norm_spec(),
            skip_ffn=True,
        )
