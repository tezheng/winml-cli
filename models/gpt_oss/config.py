"""GPT-OSS 20B config + attention-level spec adapter.

For v5-phase2 V5 we expose `to_attention_spec(layer_idx)` only —
the full decoder block is SHAPE-ONLY for the FFN side (MoE expert
forward differs from the DeepSeek-V2/V3 MoE we have, MXFP4 wiring
deferred). See __init__.py for the rationale.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from api import specs, types


@dataclass(frozen=True)
class GptOssConfig:
    hidden_size: int                # 2880
    num_attention_heads: int        # 64
    num_key_value_heads: int        # 8
    head_dim: int                   # 64 (explicit, not hidden/n_heads)
    intermediate_size: int          # 2880
    num_hidden_layers: int          # 36
    sliding_window: int             # 128
    max_position_embeddings: int    # 131072
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    vocab_size: int = 201088
    layer_types: Optional[tuple[str, ...]] = None
    dtype: torch.dtype = torch.float32

    @classmethod
    def gpt_oss_20b(cls) -> "GptOssConfig":
        """openai/gpt-oss-20b canonical defaults."""
        # Per __post_init__ in HF: alternating sliding/full starting with
        # sliding on layer 0 (`(i+1) % 2` truthy on even i).
        layer_types = tuple(
            "sliding_attention" if (i + 1) % 2 else "full_attention"
            for i in range(36)
        )
        return cls(
            hidden_size=2880, num_attention_heads=64, num_key_value_heads=8,
            head_dim=64, intermediate_size=2880, num_hidden_layers=36,
            sliding_window=128, max_position_embeddings=131072,
            layer_types=layer_types,
        )

    def is_sliding_layer(self, layer_idx: int) -> bool:
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "sliding_attention"

    def to_moe_spec(self) -> specs.MoESpec:
        """GPT-OSS MoE — top-k softmax + clamped-SwiGLU experts with bias.

        Source: modeling_gpt_oss.py:73-151 (GptOssExperts +
        GptOssTopKRouter + GptOssMLP).
        """
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,    # ignored for gpt_oss
            gate_kind=types.GateKind.SWIGLU,     # ignored
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        return specs.MoESpec(
            n_experts=self.num_local_experts,
            top_k=self.num_experts_per_tok,
            n_shared_experts=0,
            router_kind="topk_then_softmax_with_bias",
            router_norm=False,
            routed_scaling_factor=1.0,
            expert_ffn=expert_ffn,
            expert_kind="gpt_oss_clamped_swiglu",
            expert_bias=True,
            expert_swiglu_alpha=1.702,
            expert_clamp_limit=7.0,
        )

    def to_attention_spec(self, layer_idx: int = 0) -> specs.AttentionSpec:
        """GPT-OSS attention sublayer with trained sinks.

        Per-layer fields:
        - mask_kind: SINK (carries sinks param) for ALL layers; the
          sliding/full alternation rides on `sliding_window`.
          (Functionally GPT-OSS sinks apply regardless of
          sliding_window — both layer types own a `sinks` parameter.)
        - sliding_window: 128 on sliding layers, None on full layers.

        Note: we use SPLIT_HALF basis as the closest IR analog of
        GPT-OSS's pair-wise rotary on chunk(2)-halves. The two are
        mathematically equivalent under a weight reordering — exact
        weight compat with GPT-OSS checkpoints is not landed.
        """
        sw = self.sliding_window if self.is_sliding_layer(layer_idx) else None
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.SINK,
            n_sink_tokens=1,
            sliding_window=sw,
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            rope=specs.RoPESpec(
                base_theta=150000.0,                 # config.default_theta
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.YARN,
                yarn_extra=specs.YarnRoPEParams(
                    factor=32.0,
                    original_max_position_embeddings=4096,
                    beta_fast=32.0, beta_slow=1.0,
                    truncate=False,
                ),
            ),
        )
