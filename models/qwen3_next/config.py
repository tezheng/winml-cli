"""Qwen3-Next config + adapter to api specs.

Mirrors `transformers.models.qwen3_next.Qwen3NextConfig`. Verified at
`Qwen/Qwen3-Next-80B-A3B-Instruct/config.json` shapes (carry-over of the
80B-A3B production defaults).

Per-layer dispatch on `layer_types[i]`:
- "linear_attention" -> GatedDeltaNetSpec token mixer (Phase A P3 forward).
- "full_attention"   -> AttentionSpec STANDARD with q_norm/k_norm PRE-ROPE
  per-Dh (carries SHAPE-ONLY semantics — Qwen3-Next's HF attention has a
  fused (q | gate) projection and output-gating which the IR does NOT model).
  Source: modeling_qwen3_next.py:295-330.

Channel mixer (MoE layers — most of them):
- Qwen3NextSparseMoeBlock: softmax router + plain SwiGLU experts + shared
  expert with its own learned gate sigmoid. We model as MoESpec with
  router_kind="softmax", router_norm=norm_topk_prob; the shared_expert_gate
  sigmoid is NOT modeled.
- `mlp_only_layers[i]` (typically empty) forces a plain dense MLP. We
  default to the MoE path always when n_experts > 0.
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
class Qwen3NextConfig:
    """Qwen3-Next config (subset for one decoder layer)."""
    hidden_size: int
    intermediate_size: int                     # dense MLP intermediate
    moe_intermediate_size: int                 # per-expert intermediate
    shared_expert_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int                              # 256 for Qwen3-Next 80B
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    rope_theta: float
    partial_rotary_factor: float
    dtype: torch.dtype
    # Per-layer dispatch
    layer_types: Tuple[str, ...]               # ("linear_attention" | "full_attention", ...)
    mlp_only_layers: Tuple[int, ...]
    # Gated DeltaNet (linear-attention) params
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    # MoE
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    norm_topk_prob: bool

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Qwen3NextConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        n = hf["num_hidden_layers"]
        layer_types = hf.get("layer_types")
        if layer_types is None:
            # Qwen3-Next default: every 4th layer is full_attention.
            # Source: configuration_qwen3_next.py:121-127.
            interval = hf.get("full_attention_interval", 4)
            layer_types = [
                "linear_attention" if bool((i + 1) % interval) else "full_attention"
                for i in range(n)
            ]
        layer_types = tuple(layer_types[:n])

        mlp_only_layers = tuple(hf.get("mlp_only_layers") or ())

        rope_theta = 10000.0
        partial_rotary = 0.25
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", rope_theta))
            partial_rotary = float(rp.get("partial_rotary_factor", partial_rotary))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        if "partial_rotary_factor" in hf:
            partial_rotary = float(hf["partial_rotary_factor"])

        return cls(
            hidden_size=hf.get("hidden_size", 2048),
            intermediate_size=hf.get("intermediate_size", 5632),
            moe_intermediate_size=hf.get("moe_intermediate_size", 512),
            shared_expert_intermediate_size=hf.get("shared_expert_intermediate_size", 512),
            num_hidden_layers=n,
            num_attention_heads=hf.get("num_attention_heads", 16),
            num_key_value_heads=hf.get("num_key_value_heads", 2),
            head_dim=hf.get("head_dim", 256),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf.get("vocab_size", 151936),
            max_position_embeddings=hf.get("max_position_embeddings", 32768),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary,
            dtype=dtype,
            layer_types=layer_types,
            mlp_only_layers=mlp_only_layers,
            linear_conv_kernel_dim=hf.get("linear_conv_kernel_dim", 4),
            linear_key_head_dim=hf.get("linear_key_head_dim", 128),
            linear_value_head_dim=hf.get("linear_value_head_dim", 128),
            linear_num_key_heads=hf.get("linear_num_key_heads", 16),
            linear_num_value_heads=hf.get("linear_num_value_heads", 32),
            num_experts=hf.get("num_experts", 512),
            num_experts_per_tok=hf.get("num_experts_per_tok", 10),
            decoder_sparse_step=hf.get("decoder_sparse_step", 1),
            norm_topk_prob=bool(hf.get("norm_topk_prob", True)),
        )

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "linear_attention"

    def is_mlp_only_layer(self, layer_idx: int) -> bool:
        return (
            layer_idx in self.mlp_only_layers
            or self.num_experts <= 0
            or (layer_idx + 1) % self.decoder_sparse_step != 0
        )

    def _gated_deltanet_spec(self) -> specs.GatedDeltaNetSpec:
        return specs.GatedDeltaNetSpec(
            num_v_heads=self.linear_num_value_heads,
            num_k_heads=self.linear_num_key_heads,
            head_k_dim=self.linear_key_head_dim,
            head_v_dim=self.linear_value_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
            norm_eps=self.rms_norm_eps,
            silu_gate=True,
        )

    def _full_attention_spec(self) -> specs.AttentionSpec:
        # Qwen3-Next q_norm / k_norm: per-head Dh, PRE-RoPE.
        # Source: modeling_qwen3_next.py:280-302.
        # NOTE: Qwen3-Next's q_proj is sized 2 * num_heads * head_dim and the
        # output is split into (q, gate) with attn_output * sigmoid(gate). The
        # IR's STANDARD attention does NOT model this output-gating; this
        # layer is therefore SHAPE-ONLY for numerical purposes.
        # Qwen3NextRMSNorm uses (1 + weight) — see ONE_PLUS_W below.
        # The RMSNorm used inside attention's q_norm/k_norm is the SAME
        # Qwen3NextRMSNorm (line 280), so ONE_PLUS_W applies.
        qk_norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        )
        rope = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.NONE,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=self.attention_bias,
            k_bias=self.attention_bias,
            v_bias=self.attention_bias,
            o_bias=self.attention_bias,
            qk_norm=qk_norm,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
            rope=rope,
        )

    def _moe_spec(self) -> specs.MoESpec:
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.moe_intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        # Qwen3-Next: softmax router with norm_topk_prob; n_shared_experts is
        # ONE shared expert with intermediate = shared_expert_intermediate_size.
        # Our MoE shared_experts uses `I * n_shared_experts` as intermediate,
        # so set n_shared_experts so that I * n_shared = shared_expert_intermediate_size.
        # When the two intermediates differ we can't express it cleanly; for
        # the Qwen3-Next 80B production config both are 512.
        I = self.moe_intermediate_size
        if self.shared_expert_intermediate_size % I != 0:
            raise ValueError(
                f"shared_expert_intermediate_size ({self.shared_expert_intermediate_size}) "
                f"must be a multiple of moe_intermediate_size ({I}) — "
                f"api.MoE represents shared via n_shared * I."
            )
        n_shared = self.shared_expert_intermediate_size // I
        return specs.MoESpec(
            n_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            n_shared_experts=n_shared,
            router_kind="softmax",
            router_norm=self.norm_topk_prob,
            routed_scaling_factor=1.0,
            expert_ffn=expert_ffn,
        )

    def _dense_ffn_spec(self) -> specs.FFNSpec:
        return specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        # Qwen3-Next RMSNorm uses (1 + weight) — modeling_qwen3_next.py:152-166.
        norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        )
        if self.is_linear_attention_layer(layer_idx):
            token_mixer: object = self._gated_deltanet_spec()
        else:
            token_mixer = self._full_attention_spec()
        if self.is_mlp_only_layer(layer_idx):
            channel = self._dense_ffn_spec()
        else:
            channel = self._moe_spec()
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=token_mixer,
            channel_mixer=channel,
            pre_attn_norm=norm,
            pre_ffn_norm=norm,
        )
