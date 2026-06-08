"""DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite) config + adapter to api specs.

Verified against:
- `deepseek-ai/DeepSeek-V2-Lite/config.json` (downloaded snapshot).
- `transformers/models/deepseek_v2/modeling_deepseek_v2.py` (HF native).
- `transformers/models/deepseek_v2/configuration_deepseek_v2.py`.

Key V2-Lite production facts (config.json values):
- hidden_size=2048, num_attention_heads=16, num_hidden_layers=27.
- MLA dims: q_lora_rank=null (direct q_proj!), kv_lora_rank=512,
  qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128.
  qk_head_dim = qk_nope + qk_rope = 192. The "head_dim" field in the
  HF config is set to qk_rope_head_dim=64 in __post_init__ (used only by
  RoPE, NOT by the attention head shape).
- attention_bias=False; no biases on any attention linears.
- Dense+MoE alternation: first_k_dense_replace=1 → layer 0 is a dense
  SwiGLU FFN with intermediate_size=10944; layers 1..26 are MoE.
- MoE: n_routed_experts=64, num_experts_per_tok=6, n_shared_experts=2,
  moe_intermediate_size=1408, routed_scaling_factor=1.0,
  topk_method="greedy", scoring_func="softmax", norm_topk_prob=False.
  n_group=1, topk_group=1 (degenerate group routing — equivalent to no
  group routing). We disable group_routing in MoESpec when n_group==1
  AND topk_group==1.
- RoPE: rope_theta=10000, rope_scaling type "yarn" with factor=40,
  beta_fast=32, beta_slow=1, mscale=0.707, mscale_all_dim=0.707,
  original_max_position_embeddings=4096.
- norm: rms_norm_eps=1e-6, STANDARD_W mode.
- Block envelope: PRE-norm (input_layernorm + post_attention_layernorm
  both fed BEFORE the sublayer, with the sublayer output added to the
  residual). No μP residual scaling on V2 (unlike MiniCPM-3).
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
class DeepSeekV2LiteConfig:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int                # dense FFN size (layer 0)
    moe_intermediate_size: int            # per-expert FFN size
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
    # YARN scaling parameters.
    rope_type: str = "default"            # "default" | "yarn"
    yarn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 0.0
    yarn_original_max_position_embeddings: int = 4096

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "DeepSeekV2LiteConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(str(dt_raw), torch.float32)

        rs = hf.get("rope_scaling")
        if isinstance(rs, dict) and rs.get("type") == "yarn":
            rope_type = "yarn"
            yarn_factor = float(rs["factor"])
            yarn_beta_fast = float(rs.get("beta_fast", 32))
            yarn_beta_slow = float(rs.get("beta_slow", 1))
            yarn_mscale = float(rs.get("mscale", 1.0))
            yarn_mscale_all_dim = float(rs.get("mscale_all_dim", 0.0))
            yarn_orig = int(rs.get("original_max_position_embeddings",
                                   hf["max_position_embeddings"]))
        else:
            rope_type = "default"
            yarn_factor = 1.0
            yarn_beta_fast = 32.0
            yarn_beta_slow = 1.0
            yarn_mscale = 1.0
            yarn_mscale_all_dim = 0.0
            yarn_orig = int(hf["max_position_embeddings"])

        return cls(
            hidden_size=int(hf["hidden_size"]),
            num_attention_heads=int(hf["num_attention_heads"]),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            intermediate_size=int(hf["intermediate_size"]),
            moe_intermediate_size=int(hf["moe_intermediate_size"]),
            n_routed_experts=int(hf["n_routed_experts"]),
            n_shared_experts=int(hf["n_shared_experts"]),
            num_experts_per_tok=int(hf["num_experts_per_tok"]),
            routed_scaling_factor=float(hf.get("routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(hf.get("norm_topk_prob", False)),
            n_group=int(hf.get("n_group", 1) or 1),
            topk_group=int(hf.get("topk_group", 1) or 1),
            first_k_dense_replace=int(hf.get("first_k_dense_replace", 0)),
            kv_lora_rank=int(hf["kv_lora_rank"]),
            qk_nope_head_dim=int(hf["qk_nope_head_dim"]),
            qk_rope_head_dim=int(hf["qk_rope_head_dim"]),
            v_head_dim=int(hf["v_head_dim"]),
            q_lora_rank=(None if hf.get("q_lora_rank") in (None, 0)
                         else int(hf["q_lora_rank"])),
            rope_theta=float(hf.get("rope_theta", 10_000.0)),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=int(hf["vocab_size"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            dtype=dtype,
            rope_type=rope_type,
            yarn_factor=yarn_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            yarn_mscale=yarn_mscale,
            yarn_mscale_all_dim=yarn_mscale_all_dim,
            yarn_original_max_position_embeddings=yarn_orig,
        )

    # ------------------------------------------------------------------
    # to_block_spec — per-layer dispatch (dense vs MoE).
    # ------------------------------------------------------------------
    def _rope_spec(self) -> specs.RoPESpec:
        # V2 RoPE: complex-multiply on (real, imag) pairs — INTERLEAVED basis.
        # Source: modeling_deepseek_v2.py:271-284 (view_as_complex on
        # reshape(*, -1, 2) + complex multiply with polar(ones, freqs)).
        if self.rope_type == "yarn":
            yarn = specs.YarnRoPEParams(
                factor=self.yarn_factor,
                original_max_position_embeddings=self.yarn_original_max_position_embeddings,
                beta_fast=self.yarn_beta_fast,
                beta_slow=self.yarn_beta_slow,
                mscale=self.yarn_mscale,
                mscale_all_dim=self.yarn_mscale_all_dim,
            )
            return specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.INTERLEAVED,
                scaling=types.RoPEScaling.YARN,
                yarn_extra=yarn,
            )
        else:
            return specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.INTERLEAVED,
                scaling=types.RoPEScaling.NONE,
            )

    def _norm_spec(self) -> specs.NormSpec:
        return specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )

    def _attn_spec(self) -> specs.AttentionSpec:
        # Source: modeling_deepseek_v2.py:287-335.
        # V2 has `attention_bias` flag; V2-Lite sets it False (config.json).
        norm = self._norm_spec()
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_attention_heads,    # MLA: H heads everywhere
            head_dim=self.qk_head_dim,               # 192 for V2-Lite
            kind=types.AttentionKind.MLA,
            qkv_layout=types.QKVLayout.MLA_LATENT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=self.attention_bias,
            k_bias=self.attention_bias,
            v_bias=self.attention_bias,
            o_bias=self.attention_bias,
            q_lora_rank=self.q_lora_rank,           # None for V2-Lite
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_norm=norm,       # MLA reuses qk_norm slot for q_a/kv_a norm eps.
            rope=self._rope_spec(),
        )

    def _dense_ffn_spec(self) -> specs.FFNSpec:
        return specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )

    def _moe_spec(self) -> specs.MoESpec:
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.moe_intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
        )
        # V2-Lite has n_group=1, topk_group=1 — degenerate group routing
        # equivalent to greedy top-k over all experts. We model this as
        # group_routing=None.
        if self.n_group > 1 or self.topk_group > 1:
            group = specs.GroupRoutingSpec(
                n_groups=self.n_group,
                topk_per_group=self.topk_group,
            )
        else:
            group = None
        return specs.MoESpec(
            n_experts=self.n_routed_experts,
            top_k=self.num_experts_per_tok,
            n_shared_experts=self.n_shared_experts,
            router_kind="softmax",
            router_norm=self.norm_topk_prob,
            group_routing=group,
            routed_scaling_factor=self.routed_scaling_factor,
            expert_ffn=expert_ffn,
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        """Per-layer dispatch.

        Source: modeling_deepseek_v2.py:404
            self.mlp = DeepseekV2Moe(config) if layer_idx >= first_k_dense_replace
                       else DeepseekV2MLP(config)
        """
        norm = self._norm_spec()
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
