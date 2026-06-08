"""DeepSeek-V3-Lite synthetic config.

Verified against `transformers/models/deepseek_v3/modeling_deepseek_v3.py`
and `transformers/models/deepseek_v3/configuration_deepseek_v3.py`.

V3 architectural deltas vs V2 (must round-trip through to_block_spec):
- Router: sigmoid + score_correction_bias. Source:
  modeling_deepseek_v3.py:139-237.
- Group routing: ALWAYS on. n_group=8, topk_group=4 in V3-full.
- routed_scaling_factor = 2.5 (V2: 1.0).
- norm_topk_prob = True (V2: False).
- RoPE: SPLIT_HALF basis (rotate_half pattern). Source:
  modeling_deepseek_v3.py:250-280.
  But V3 has a `rope_interleave` boolean — when True it uses
  apply_rotary_pos_emb_interleave which is YET DIFFERENT from V2's
  complex-multiply path (it does a view+transpose reshape before the
  standard rotate_half multiply, source:
  modeling_deepseek_v3.py:320-355). For B5 we model V3 as SPLIT_HALF
  only (rope_interleave=False). This is the architectural family we
  expose; an extension for rope_interleave=True is deferred.
- MoE expert layout identical to V2.
- MLA dims: V3-full has q_lora_rank=1536 (q-LoRA path). V3-Lite synthetic
  config likewise uses the q-LoRA path so we exercise it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from api import specs, types


@dataclass(frozen=True)
class DeepSeekV3LiteConfig:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: Optional[int]
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    dtype: torch.dtype

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def _attn_spec(self) -> specs.AttentionSpec:
        # V3 RoPE: SPLIT_HALF basis (rotate_half on a single tensor).
        # Source: modeling_deepseek_v3.py:250-280.
        rope = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.NONE,
        )
        norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_attention_heads,
            head_dim=self.qk_head_dim,
            kind=types.AttentionKind.MLA,
            qkv_layout=types.QKVLayout.MLA_LATENT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=self.attention_bias,
            k_bias=self.attention_bias,
            v_bias=self.attention_bias,
            o_bias=self.attention_bias,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_norm=norm,
            rope=rope,
        )

    def _moe_spec(self) -> specs.MoESpec:
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.moe_intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
        )
        gr = specs.GroupRoutingSpec(
            n_groups=self.n_group,
            topk_per_group=self.topk_group,
        )
        return specs.MoESpec(
            n_experts=self.n_routed_experts,
            top_k=self.num_experts_per_tok,
            n_shared_experts=self.n_shared_experts,
            router_kind="sigmoid_plus_bias",
            router_norm=self.norm_topk_prob,
            group_routing=gr,
            routed_scaling_factor=self.routed_scaling_factor,
            expert_ffn=expert_ffn,
        )

    def _dense_ffn_spec(self) -> specs.FFNSpec:
        return specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        attn = self._attn_spec()
        if layer_idx >= self.first_k_dense_replace:
            channel = self._moe_spec()
        else:
            channel = self._dense_ffn_spec()
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn,
            channel_mixer=channel,
            pre_attn_norm=norm,
            pre_ffn_norm=norm,
        )
