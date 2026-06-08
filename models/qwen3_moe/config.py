"""Qwen3-MoE model config + adapter to api specs.

Verified against `transformers/models/qwen3_moe/configuration_qwen3_moe.py`
(defaults at L48-117) and `modeling_qwen3_moe.py` (attention L126-195,
SparseMoeBlock + DecoderLayer L275-353).

Architectural facts (source-grounded):

- Qwen3-style decoder: PRE-norm + QK-norm PRE_ROPE PER_HEAD_DH.
  Source: modeling_qwen3_moe.py:152-153 (``q_norm = RMSNorm(head_dim)`` /
  ``k_norm = RMSNorm(head_dim)``) and 167-172 (q_norm/k_norm applied
  AFTER view but BEFORE apply_rotary_pos_emb).

- GQA: ``num_attention_heads=32``, ``num_key_value_heads=4`` (defaults).
  ``head_dim = hidden_size // num_attention_heads`` (also overridable via
  ``head_dim`` config field per modeling_qwen3_moe.py:134).

- RoPE: SPLIT_HALF basis (``rotate_half`` at modeling_qwen3_moe.py:56-60),
  ``rope_theta`` from ``config.rope_parameters["rope_theta"]``.

- Sliding window: gated by ``use_sliding_window``. If False (default),
  ``sliding_window`` is forced to None in ``__post_init__``
  (configuration_qwen3_moe.py:115). When True the window applies; we
  honor whichever value the config carries via ``sliding_window``.

- RMSNorm STANDARD_W, ``rms_norm_eps`` default 1e-6.

- ``tie_word_embeddings = False`` by default (configuration_qwen3_moe.py:96).

- MoE block (modeling_qwen3_moe.py:215-286):
  * ``num_experts=128`` routed experts, ``num_experts_per_tok=8`` (top-8).
    Defaults at configuration_qwen3_moe.py:104-105.
  * Each expert is a SwiGLU FFN with ``moe_intermediate_size`` (default
    768 in the small reference config; 1408 in Qwen3-30B-A3B per
    HuggingFace's Qwen/Qwen3-30B-A3B config.json).
  * Router math (modeling_qwen3_moe.py:263-272):
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, top_k, ...)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
    ``norm_topk_prob`` is a config flag (default False per
    configuration_qwen3_moe.py:106). Qwen3-30B-A3B sets it True.
  * NO shared experts (SparseMoeBlock has only `gate` + `experts` per
    modeling_qwen3_moe.py:275-286).
  * NO group routing.
  * NO ``routed_scaling_factor``.

- Per-layer dense/MoE dispatch (modeling_qwen3_moe.py:314-319):
        if (layer_idx not in mlp_only_layers) and (
            num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)
  Default ``decoder_sparse_step=1`` and ``mlp_only_layers=[]`` →
  every layer is MoE. The dense MLP uses ``config.intermediate_size``,
  NOT ``moe_intermediate_size``.

- Vocab: 151936 (default).

- Attention biases: ``attention_bias = False`` (default).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Qwen3MoeConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int                # dense MLP intermediate size
    moe_intermediate_size: int            # per-expert SwiGLU intermediate size
    num_hidden_layers: int
    num_experts: int                      # = num_local_experts (HF alias)
    num_experts_per_tok: int              # top-k
    norm_topk_prob: bool                  # default False, Qwen3-30B-A3B sets True
    decoder_sparse_step: int              # default 1
    mlp_only_layers: Tuple[int, ...]      # default ()
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    dtype: torch.dtype
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None   # honored only if use_sliding_window=True

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Qwen3MoeConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(str(dt_raw), torch.float32)

        # transformers 5.x nests rope_theta under rope_parameters.
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict) and "rope_theta" in rp:
            rope_theta = float(rp["rope_theta"])
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        else:
            rope_theta = 10_000.0

        head_dim = hf.get("head_dim")
        if head_dim is None:
            head_dim = int(hf["hidden_size"]) // int(hf["num_attention_heads"])

        use_sw = bool(hf.get("use_sliding_window", False))
        sw_raw = hf.get("sliding_window")
        # Mirror configuration_qwen3_moe.py:115:
        # `self.sliding_window = self.sliding_window if self.use_sliding_window else None`
        sw_eff = (int(sw_raw) if (use_sw and sw_raw is not None) else None)

        # HF aliases num_experts → num_local_experts; accept either.
        n_experts = int(hf.get("num_experts", hf.get("num_local_experts", 128)))

        mlp_only = hf.get("mlp_only_layers") or []
        mlp_only_t = tuple(int(L) for L in mlp_only)

        return cls(
            hidden_size=int(hf["hidden_size"]),
            num_attention_heads=int(hf["num_attention_heads"]),
            num_key_value_heads=int(hf.get(
                "num_key_value_heads", hf["num_attention_heads"]
            )),
            head_dim=int(head_dim),
            intermediate_size=int(hf["intermediate_size"]),
            moe_intermediate_size=int(hf.get("moe_intermediate_size", 768)),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            num_experts=n_experts,
            num_experts_per_tok=int(hf.get("num_experts_per_tok", 8)),
            norm_topk_prob=bool(hf.get("norm_topk_prob", False)),
            decoder_sparse_step=int(hf.get("decoder_sparse_step", 1)),
            mlp_only_layers=mlp_only_t,
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=int(hf["vocab_size"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            dtype=dtype,
            use_sliding_window=use_sw,
            sliding_window=sw_eff,
        )

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """Per-layer dispatch (modeling_qwen3_moe.py:314-319)."""
        if layer_idx in self.mlp_only_layers:
            return False
        if self.num_experts <= 0:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0

    def _attn_spec(self) -> specs.AttentionSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        if self.sliding_window is not None:
            mask_kind = types.MaskKind.SWA
            sw_eff = self.sliding_window
        else:
            mask_kind = types.MaskKind.CAUSAL
            sw_eff = None
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=mask_kind,
            sliding_window=sw_eff,
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            qk_norm=norm_spec,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            ),
        )

    def _dense_ffn_spec(self) -> specs.FFNSpec:
        # Source: modeling_qwen3_moe.py:319 — Qwen3MoeMLP uses
        # `intermediate_size=config.intermediate_size` (NOT moe size).
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
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        return specs.MoESpec(
            n_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            n_shared_experts=0,
            router_kind="softmax",
            router_norm=self.norm_topk_prob,
            group_routing=None,
            routed_scaling_factor=1.0,
            expert_ffn=expert_ffn,
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        attn = self._attn_spec()
        if self._is_moe_layer(layer_idx):
            channel = self._moe_spec()
        else:
            channel = self._dense_ffn_spec()
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn,
            channel_mixer=channel,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
