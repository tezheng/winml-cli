"""OLMoE 1B-7B config + adapter to api specs.

Verified against `transformers/models/olmoe/configuration_olmoe.py` (L23-89)
and `modeling_olmoe.py`.

Architectural facts (source-grounded):

- PRE-norm decoder (Llama-style envelope). Source: modeling_olmoe.py:378-416
  — input_layernorm BEFORE self_attn; post_attention_layernorm BEFORE mlp.

- FULL_HDH QK-norm PRE-RoPE.
  Source: modeling_olmoe.py:246-249 (
      q_norm = OlmoeRMSNorm(hidden_size, eps=...)
      k_norm = OlmoeRMSNorm((hidden_size // num_attention_heads) * num_key_value_heads, eps=...)
  ) and 262-275 (norms applied to flat tensor BEFORE view+transpose+RoPE).
  Note that ``q_norm`` operates on the flattened ``[B, S, H_q * Dh]``
  tensor (FULL_HDH), and ``k_norm`` on ``[B, S, H_kv * Dh]`` — same shape
  pattern as OLMo 2.

- ``clip_qkv``: optional float that clamps Q/K/V to ``[-clip, clip]`` after
  the projection norms. The public ``allenai/OLMoE-1B-7B-0924`` checkpoint
  has ``clip_qkv = None`` (verified at the HF model-card config.json).
  When non-None this would need a runtime clamp — we assert it's None and
  raise NotImplementedError otherwise. Source: modeling_olmoe.py:266-269.

- Attention biases: ``attention_bias=False`` default.

- GQA: ``num_attention_heads=16``, ``num_key_value_heads`` defaults to
  ``num_attention_heads`` when null (configuration_olmoe.py:83-85).
  OLMoE-1B-7B-0924 uses n_q=16, n_kv=16 (MHA, NOT GQA — verified at the
  HF config.json).

- RoPE: SPLIT_HALF basis (modeling_olmoe.py:150-154 ``rotate_half``);
  ``rope_theta`` from ``config.rope_parameters["rope_theta"]`` (default
  10000 unless set).

- ``rms_norm_eps`` default 1e-5.

- Mask: CAUSAL only (modeling_olmoe.py:490 — ``create_causal_mask`` with
  no sliding-window branch — "diff with mixtral: no sliding").

- MoE block (modeling_olmoe.py:301-375):
  * ``num_experts=64`` routed experts, ``num_experts_per_tok=8`` top-8.
    Defaults at configuration_olmoe.py:77-78.
  * Each expert is a SwiGLU FFN with ``intermediate_size`` (one number
    used for both the dense-style block in OlmoeMLP — which doesn't
    exist in the decoder layer for OLMoE — and the per-expert size).
    Source: modeling_olmoe.py:309 (``intermediate_dim = config.intermediate_size``).
  * Router (modeling_olmoe.py:341-359): same shape as Qwen3MoeTopKRouter
    — softmax(fp32) → topk → optional norm_topk_prob renormalization.
  * NO shared experts (SparseMoeBlock at modeling_olmoe.py:362-375 has
    only gate + experts).
  * NO group routing.
  * NO routed_scaling_factor.
  * NO ``mlp_only_layers`` / ``decoder_sparse_step`` (every layer is MoE).

- Vocab: 50304 (default). ``tie_word_embeddings = False``.

- ``norm_topk_prob`` flag (default False per configuration_olmoe.py:81).
  The public ``allenai/OLMoE-1B-7B-0924`` config sets it to False.
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
class OlmoeConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int                # per-expert SwiGLU intermediate size
    num_hidden_layers: int
    num_experts: int                      # = num_local_experts (HF alias)
    num_experts_per_tok: int              # top-k
    norm_topk_prob: bool                  # default False
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    dtype: torch.dtype
    clip_qkv: Optional[float] = None      # assert None until we implement clamp

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "OlmoeConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(str(dt_raw), torch.float32)

        rp = hf.get("rope_parameters")
        if isinstance(rp, dict) and "rope_theta" in rp:
            rope_theta = float(rp["rope_theta"])
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])
        else:
            rope_theta = 10_000.0

        n_q = int(hf["num_attention_heads"])
        n_kv_raw = hf.get("num_key_value_heads")
        n_kv = int(n_kv_raw) if n_kv_raw is not None else n_q
        head_dim = hf.get("head_dim")
        if head_dim is None:
            head_dim = int(hf["hidden_size"]) // n_q

        # HF aliases num_local_experts → num_experts (attribute_map).
        n_experts = int(hf.get("num_experts", hf.get("num_local_experts", 64)))

        return cls(
            hidden_size=int(hf["hidden_size"]),
            num_attention_heads=n_q,
            num_key_value_heads=n_kv,
            head_dim=int(head_dim),
            intermediate_size=int(hf["intermediate_size"]),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            num_experts=n_experts,
            num_experts_per_tok=int(hf.get("num_experts_per_tok", 8)),
            norm_topk_prob=bool(hf.get("norm_topk_prob", False)),
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=int(hf["vocab_size"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            dtype=dtype,
            clip_qkv=hf.get("clip_qkv"),
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        if self.clip_qkv is not None:
            raise NotImplementedError(
                "OLMoE clip_qkv is not yet implemented in api.attention; "
                f"got clip_qkv={self.clip_qkv}. The public OLMoE-1B-7B-0924 "
                "checkpoint has clip_qkv=None."
            )
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,        # no SWA on OLMoE
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            qk_norm=norm_spec,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.FULL_HDH,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            ),
        )
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        moe_spec = specs.MoESpec(
            n_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            n_shared_experts=0,
            router_kind="softmax",
            router_norm=self.norm_topk_prob,
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
