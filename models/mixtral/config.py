"""Mixtral 8x7B config + adapter to api specs.

Verified against `transformers/models/mixtral/configuration_mixtral.py`
(defaults at lines 61-83) and `modeling_mixtral.py` (router/MoE block
behaviour at lines 101-135, attention at lines 295-351, decoder layer at
lines 354-389).

Architectural facts (source-grounded):

- GQA: ``num_attention_heads=32``, ``num_key_value_heads=8``,
  ``head_dim = hidden_size // num_attention_heads`` (4096/32 = 128) unless
  set explicitly. Source: configuration_mixtral.py:65-67.

- Attention biases are all False (HF hard-codes ``bias=False`` on every
  Mixtral projection at modeling_mixtral.py:307-310).

- RoPE: ``rope_theta = 1e6`` (``default_theta`` at configuration_mixtral.py:44),
  SPLIT_HALF basis (``rotate_half`` at modeling_mixtral.py:224-228), no scaling.

- ``sliding_window`` defaults to None (configuration_mixtral.py:77). HF v0.1
  shipped without SWA. Mask kind: CAUSAL.

- NO QK-norm — Mixtral attention has no q_norm / k_norm layers (compare with
  Qwen3MoE which does, at modeling_qwen3_moe.py:152-153).

- RMSNorm STANDARD_W (modeling_mixtral.py:138-156).

- MoE block (modeling_mixtral.py:101-135):
  * ``num_local_experts = 8`` routed experts (config L80).
  * ``num_experts_per_tok = 2`` top-2 (config L79).
  * Each expert is a SwiGLU FFN with ``intermediate_size = 14336`` (config L63).
  * Router math (modeling_mixtral.py:109-116):
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, top_k, ...)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    The renormalization is ALWAYS ON — Mixtral has no ``norm_topk_prob``
    flag (compare with Qwen3MoE/OLMoE which do). We therefore set
    ``router_norm=True`` unconditionally.
  * NO ``routed_scaling_factor`` (the renormalized top-k weights are used
    directly). We set ``routed_scaling_factor = 1.0``.
  * NO shared experts (no ``shared_experts`` module on MixtralSparseMoeBlock).
    We set ``n_shared_experts = 0``.
  * NO group routing.

- Vocab: 32000 (config L61). ``tie_word_embeddings = False`` (config L76).

- ``rms_norm_eps = 1e-5`` (config L71).
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
class MixtralConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int                # per-expert SwiGLU intermediate size
    num_hidden_layers: int
    num_local_experts: int                # = num_experts (HF aliases)
    num_experts_per_tok: int              # top-k
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    sliding_window: Optional[int] = None  # default None — v0.1 has no SWA

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "MixtralConfig":
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
            rope_theta = 1_000_000.0   # default_theta per configuration_mixtral.py:44

        head_dim = hf.get("head_dim")
        if head_dim is None:
            head_dim = int(hf["hidden_size"]) // int(hf["num_attention_heads"])

        return cls(
            hidden_size=int(hf["hidden_size"]),
            num_attention_heads=int(hf["num_attention_heads"]),
            num_key_value_heads=int(hf.get(
                "num_key_value_heads", hf["num_attention_heads"]
            )),
            head_dim=int(head_dim),
            intermediate_size=int(hf["intermediate_size"]),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            num_local_experts=int(hf.get(
                "num_local_experts", hf.get("num_experts", 8)
            )),
            num_experts_per_tok=int(hf.get("num_experts_per_tok", 2)),
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=int(hf["vocab_size"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            sliding_window=hf.get("sliding_window"),
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # Mask kind: when sliding_window is set use SWA; otherwise CAUSAL.
        # Mixtral 8x7B v0.1 (and the public Instruct variant) ship
        # sliding_window=None in the config.
        if self.sliding_window is not None:
            mask_kind = types.MaskKind.SWA
            sw_eff = self.sliding_window
        else:
            mask_kind = types.MaskKind.CAUSAL
            sw_eff = None
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=mask_kind,
            sliding_window=sw_eff,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            ),
        )
        # MoE: softmax router with ALWAYS-on top-k normalization
        # (modeling_mixtral.py:114). No shared experts, no group routing,
        # no routed_scaling_factor.
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        moe_spec = specs.MoESpec(
            n_experts=self.num_local_experts,
            top_k=self.num_experts_per_tok,
            n_shared_experts=0,
            router_kind="softmax",
            router_norm=True,                    # Mixtral always renormalizes
            score_correction_bias=False,
            group_routing=None,
            routed_scaling_factor=1.0,
            expert_ffn=expert_ffn,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=moe_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
