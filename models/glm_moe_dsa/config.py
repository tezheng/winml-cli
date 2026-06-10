"""GLM-MoE-DSA config + adapter to api specs.

Mirrors `transformers.models.glm_moe_dsa.GlmMoeDsaConfig`. Verified at
`zai-org/GLM-5/config.json` shapes.

Per-layer MLP dispatch (mlp_layer_types[L]):
- "dense"  -> dense SwiGLU FFN (FFNSpec).
- "sparse" -> MoE with sigmoid+bias router (V3-style).
  Source: modeling_glm_moe_dsa.py:478-591.

Token mixer is always MLA + DSA Lightning Indexer. The IR carries this as
`AttentionKind.DSA` (shape-only forward — Phase A reservation).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class GlmMoeDsaConfig:
    """GLM-MoE-DSA config (subset for one decoder layer)."""
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    # MLA dims
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    # RoPE
    rope_theta: float
    # MoE
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    # DSA Indexer
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    # Per-layer dispatch
    mlp_layer_types: Tuple[str, ...]
    indexer_types: Tuple[str, ...]
    dtype: torch.dtype

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "GlmMoeDsaConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        n = hf["num_hidden_layers"]
        mlp_layer_types = hf.get("mlp_layer_types")
        if mlp_layer_types is None:
            mlp_layer_types = ["dense"] * min(3, n) + ["sparse"] * max(0, n - 3)
        mlp_layer_types = tuple(mlp_layer_types[:n])

        indexer_types = hf.get("indexer_types")
        if indexer_types is None:
            # Default: every layer "full" with freq=1 (configuration:142).
            indexer_types = ["full"] * n
        indexer_types = tuple(indexer_types[:n])

        rope_theta = 10000.0
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", rope_theta))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])

        return cls(
            hidden_size=hf.get("hidden_size", 6144),
            intermediate_size=hf.get("intermediate_size", 12288),
            moe_intermediate_size=hf.get("moe_intermediate_size", 2048),
            num_hidden_layers=n,
            num_attention_heads=hf.get("num_attention_heads", 64),
            num_key_value_heads=hf.get("num_key_value_heads", 64),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=hf.get("vocab_size", 154880),
            max_position_embeddings=hf.get("max_position_embeddings", 202752),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            kv_lora_rank=hf.get("kv_lora_rank", 512),
            q_lora_rank=hf.get("q_lora_rank", 2048),
            qk_nope_head_dim=hf.get("qk_nope_head_dim", 192),
            qk_rope_head_dim=hf.get("qk_rope_head_dim", 64),
            v_head_dim=hf.get("v_head_dim", 256),
            rope_theta=rope_theta,
            n_routed_experts=hf.get("n_routed_experts", 256),
            n_shared_experts=hf.get("n_shared_experts", 1),
            num_experts_per_tok=hf.get("num_experts_per_tok", 8),
            routed_scaling_factor=float(hf.get("routed_scaling_factor", 2.5)),
            norm_topk_prob=bool(hf.get("norm_topk_prob", True)),
            n_group=hf.get("n_group", 1),
            topk_group=hf.get("topk_group", 1),
            index_topk=hf.get("index_topk", 2048),
            index_head_dim=hf.get("index_head_dim", 128),
            index_n_heads=hf.get("index_n_heads", 32),
            mlp_layer_types=mlp_layer_types,
            indexer_types=indexer_types,
            dtype=dtype,
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_sparse_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "sparse"

    def _attn_spec(self) -> specs.AttentionSpec:
        # MLA + DSA Indexer (shape-only).
        norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        rope = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.NONE,
        )
        indexer = specs.IndexerSpec(
            indexer_dim=self.index_head_dim,
            top_k=self.index_topk,
        )
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_attention_heads,    # MLA expands fully (modeling_glm:374-378)
            head_dim=self.qk_head_dim,
            kind=types.AttentionKind.DSA,
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
            indexer=indexer,
        )

    def _moe_spec(self) -> specs.MoESpec:
        """V3-style sigmoid+bias MoE with group routing.

        GLM-5's group routing: `n_group=1, topk_group=1` by default — i.e.
        single-group routing which is functionally equivalent to no group
        routing, but the HF code path still goes through the group-routing
        scoring (sum of top-2 per group). Our `sigmoid_plus_bias` MoE handles
        this correctly.
        """
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.moe_intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
        )
        gr = specs.GroupRoutingSpec(
            n_groups=self.n_group, topk_per_group=self.topk_group,
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
        if self.is_sparse_layer(layer_idx):
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
