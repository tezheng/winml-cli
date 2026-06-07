"""DeepSeek-V3 (671B) decoder-layer config — full-scale dimensions.

This package presents the V3 family at production scale: same MoE module
(sigmoid+bias router + group routing + shared experts) as V3-Lite, but
with the FULL-V3 dims (hidden=7168, n_q=128, n_routed_experts=256, ...).

Architecturally identical to V3-Lite (modeled in
``models/deepseek_v3_lite``) — this wrapper exists to:
1. Validate that our MoE + MLA infra ASSEMBLES at production dims
   (synthetic-weight shape sanity).
2. Provide a single source of truth for the V3 production dim profile
   (mirrors ``configuration_deepseek_v3.DeepseekV3Config`` defaults).

Numerical verification at production dims is NOT exercised (671B weights
are out of scope); the per-module numerical gate lives in the B5 isolation
test ``tests/models/deepseek_v3_lite/test_isolation_hf.py``.

Defaults verified against
``transformers/models/deepseek_v3/configuration_deepseek_v3.py:71-104``.
``rope_interleave`` is set to False for parity with V3-Lite; the
True branch is deferred (see deepseek_v3_lite/config.py docstring).
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
class DeepSeekV3MoEConfig:
    """Production-scale V3 config (671B baseline).

    Defaults mirror configuration_deepseek_v3.py:71-104.
    """
    hidden_size: int = 7168
    num_attention_heads: int = 128
    num_hidden_layers: int = 61
    intermediate_size: int = 18432            # dense FFN size (layers < first_k_dense_replace)
    moe_intermediate_size: int = 2048         # per-expert FFN size
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    n_group: int = 8
    topk_group: int = 4
    first_k_dense_replace: int = 3
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: Optional[int] = 1536
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 129280
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    dtype: torch.dtype = torch.bfloat16

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "DeepSeekV3MoEConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "bfloat16"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(str(dt_raw), torch.bfloat16)
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict) and "rope_theta" in rp:
            rope_theta = float(rp["rope_theta"])
        else:
            rope_theta = float(hf.get("rope_theta", 10_000.0))
        return cls(
            hidden_size=int(hf.get("hidden_size", 7168)),
            num_attention_heads=int(hf.get("num_attention_heads", 128)),
            num_hidden_layers=int(hf.get("num_hidden_layers", 61)),
            intermediate_size=int(hf.get("intermediate_size", 18432)),
            moe_intermediate_size=int(hf.get("moe_intermediate_size", 2048)),
            n_routed_experts=int(hf.get("n_routed_experts", 256)),
            n_shared_experts=int(hf.get("n_shared_experts", 1)),
            num_experts_per_tok=int(hf.get("num_experts_per_tok", 8)),
            routed_scaling_factor=float(hf.get("routed_scaling_factor", 2.5)),
            norm_topk_prob=bool(hf.get("norm_topk_prob", True)),
            n_group=int(hf.get("n_group", 8) or 8),
            topk_group=int(hf.get("topk_group", 4) or 4),
            first_k_dense_replace=int(hf.get("first_k_dense_replace", 3) or 3),
            kv_lora_rank=int(hf.get("kv_lora_rank", 512)),
            qk_nope_head_dim=int(hf.get("qk_nope_head_dim", 128)),
            qk_rope_head_dim=int(hf.get("qk_rope_head_dim", 64)),
            v_head_dim=int(hf.get("v_head_dim", 128)),
            q_lora_rank=(None if hf.get("q_lora_rank") in (None, 0)
                         else int(hf["q_lora_rank"])),
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=int(hf.get("vocab_size", 129280)),
            max_position_embeddings=int(hf.get("max_position_embeddings", 4096)),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            dtype=dtype,
        )

    # The to_block_spec / build / load logic is identical to V3-Lite —
    # delegate to it. Importing here lazily to avoid circular imports if
    # the lite package ever depends on this one (it won't).
    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        # V3 RoPE: SPLIT_HALF basis (rotate_half on a single tensor) — we
        # mirror V3-Lite which represents the rope_interleave=False branch.
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
        attn = specs.AttentionSpec(
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
        if layer_idx < self.first_k_dense_replace:
            channel = specs.FFNSpec(
                intermediate_size=self.intermediate_size,
                activation=types.Activation.SILU,
                gate_kind=types.GateKind.SWIGLU,
            )
        else:
            expert_ffn = specs.FFNSpec(
                intermediate_size=self.moe_intermediate_size,
                activation=types.Activation.SILU,
                gate_kind=types.GateKind.SWIGLU,
            )
            gr = specs.GroupRoutingSpec(
                n_groups=self.n_group,
                topk_per_group=self.topk_group,
            )
            channel = specs.MoESpec(
                n_experts=self.n_routed_experts,
                top_k=self.num_experts_per_tok,
                n_shared_experts=self.n_shared_experts,
                router_kind="sigmoid_plus_bias",
                router_norm=self.norm_topk_prob,
                score_correction_bias=True,
                group_routing=gr,
                routed_scaling_factor=self.routed_scaling_factor,
                expert_ffn=expert_ffn,
            )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn,
            channel_mixer=channel,
            pre_attn_norm=norm,
            pre_ffn_norm=norm,
        )
