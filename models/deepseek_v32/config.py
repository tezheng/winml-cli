"""DeepSeek-V3.2 (DSA Lightning Indexer) shape-only config.

DSA = DeepSeek Sparse Attention. V3.2 keeps V3's MLA + MoE composition and
adds a Lightning Indexer head that scores (Q, K) pairs cheaply via a
separate Q/K projection of smaller `indexer_dim`. The main SDPA is then
run only on the top-k keys per query.

There is no published HF model file for V3.2 yet, so this is SHAPE-ONLY:
the spec composes correctly through to AttentionSpec.indexer +
AttentionKind.DSA, and the api allocates the indexer modules. Forward
raises NotImplementedError — the full DSA runtime is deferred to a future
batch.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from api import specs, types


@dataclass(frozen=True)
class DeepSeekV32Config:
    # Backbone: V3-shaped MLA + MoE.
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
    # Lightning Indexer params.
    indexer_dim: int = 64
    indexer_top_k: int = 2048
    indexer_warmup_tokens: int = 0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
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
            indexer_dim=self.indexer_dim,
            top_k=self.indexer_top_k,
        )
        attn = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_attention_heads,
            head_dim=self.qk_head_dim,
            kind=types.AttentionKind.DSA,
            qkv_layout=types.QKVLayout.MLA_LATENT,
            mask_kind=types.MaskKind.CAUSAL,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_norm=norm,
            rope=rope,
            indexer=indexer,
        )

        if layer_idx >= self.first_k_dense_replace:
            expert_ffn = specs.FFNSpec(
                intermediate_size=self.moe_intermediate_size,
                activation=types.Activation.SILU,
                gate_kind=types.GateKind.SWIGLU,
            )
            channel = specs.MoESpec(
                n_experts=self.n_routed_experts,
                top_k=self.num_experts_per_tok,
                n_shared_experts=self.n_shared_experts,
                router_kind="sigmoid_plus_bias",
                router_norm=self.norm_topk_prob,
                group_routing=specs.GroupRoutingSpec(
                    n_groups=self.n_group, topk_per_group=self.topk_group,
                ),
                routed_scaling_factor=self.routed_scaling_factor,
                expert_ffn=expert_ffn,
            )
        else:
            channel = specs.FFNSpec(
                intermediate_size=self.intermediate_size,
                activation=types.Activation.SILU,
                gate_kind=types.GateKind.SWIGLU,
            )

        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn,
            channel_mixer=channel,
            pre_attn_norm=norm,
            pre_ffn_norm=norm,
        )
