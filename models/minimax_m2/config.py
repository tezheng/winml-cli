"""MiniMax-M2 config + adapter to api specs.

Mirrors `transformers.models.minimax_m2.MiniMaxM2Config`. Verified against
`MiniMaxAI/MiniMax-Text-01-hf/config.json` shapes (the v7 Raschka agent
confirmed M2 = plain attention + V3-style sigmoid+bias MoE, no Lightning
attention).
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
class MiniMaxM2Config:
    """MiniMax-M2 config (subset for one decoder layer)."""
    hidden_size: int
    intermediate_size: int                # per-expert intermediate (HF: intermediate_size = per-expert dim)
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_theta: float
    num_experts_per_tok: int
    num_local_experts: int
    dtype: torch.dtype

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "MiniMaxM2Config":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        rope_theta = 5_000_000.0
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", rope_theta))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])

        # `num_experts` is an alias of `num_local_experts` (attribute_map).
        nle = hf.get("num_local_experts", hf.get("num_experts", 256))

        return cls(
            hidden_size=hf.get("hidden_size", 3072),
            intermediate_size=hf.get("intermediate_size", 1536),
            num_hidden_layers=hf["num_hidden_layers"],
            num_attention_heads=hf.get("num_attention_heads", 48),
            num_key_value_heads=hf.get("num_key_value_heads", 8),
            head_dim=hf.get("head_dim", 128),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf.get("vocab_size", 200064),
            max_position_embeddings=hf.get("max_position_embeddings", 196608),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            rope_theta=rope_theta,
            num_experts_per_tok=hf.get("num_experts_per_tok", 8),
            num_local_experts=nle,
            dtype=dtype,
        )

    def _attn_spec(self) -> specs.AttentionSpec:
        # MiniMax-M2 q_norm / k_norm: applied to the FULL flattened
        # projection (num_heads * head_dim resp. num_kv_heads * head_dim)
        # BEFORE view+transpose — that's FULL_HDH PRE_ROPE (OLMo-2 style).
        # Source: modeling_minimax_m2.py:312-313 (`q_norm = MiniMaxM2RMSNorm(
        # num_attention_heads * head_dim, eps)`), 326 (`query_states =
        # q_norm(q_proj(hidden_states))`).
        qk_norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        rope = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.NONE,
        )
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            qk_norm=qk_norm,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.FULL_HDH,
            rope=rope,
        )

    def _moe_spec(self) -> specs.MoESpec:
        """V3-style sigmoid+bias router, NO group routing, NO shared experts.

        Source: modeling_minimax_m2.py:46-124 (TopKRouter + SparseMoeBlock).
        The router math:
            router_logits = F.linear(x, weight)
            routing_weights = sigmoid(router_logits.float())
            scores_for_choice = routing_weights + e_score_correction_bias
            _, top_k_idx = topk(scores_for_choice, top_k)
            top_k_w = routing_weights.gather(1, top_k_idx)     # BIAS-FREE
            top_k_w = top_k_w / top_k_w.sum(-1, keepdim=True)  # always normed
            # No routed_scaling_factor.

        NOTE: `e_score_correction_bias` lives on the SparseMoeBlock (HF line 114),
        not on the gate. Our `_SigmoidRouter` puts it on the gate
        (`gate.e_score_correction_bias`). The weight loader bridges this.
        """
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
        )
        return specs.MoESpec(
            n_experts=self.num_local_experts,
            top_k=self.num_experts_per_tok,
            n_shared_experts=0,
            router_kind="sigmoid_plus_bias",
            router_norm=True,         # M2 ALWAYS normalises top-k weights.
            group_routing=None,       # M2: no group routing.
            routed_scaling_factor=1.0,
            expert_ffn=expert_ffn,
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=self._attn_spec(),
            channel_mixer=self._moe_spec(),
            pre_attn_norm=norm,
            pre_ffn_norm=norm,
        )
