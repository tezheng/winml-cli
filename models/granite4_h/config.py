"""Granite 4 H config + adapter to api specs.

Mirrors `transformers.models.granitemoehybrid.GraniteMoeHybridConfig`.
Verified against `ibm-granite/granite-4.0-h-micro/config.json`.

Per-layer dispatch:
- `layer_types[i] == "mamba"` -> SSDSpec token mixer (Mamba2Mixer) +
  skip_ffn=True residual structure for SSM block. Wait — actually NO:
  the Granite hybrid block structure ALWAYS runs the
  post_attention_layernorm + shared_mlp sublayer AFTER the token mixer,
  even for mamba layers. Source: modeling_granitemoehybrid.py:1085-1094.
- `layer_types[i] == "attention"` -> AttentionSpec token mixer.

So in our framework:
- mamba layer: token_mixer=SSDSpec, channel_mixer=FFNSpec (shared_mlp),
  residual_scale=residual_multiplier, skip_ffn=False.
- attention layer: token_mixer=AttentionSpec, channel_mixer=FFNSpec,
  residual_scale=residual_multiplier, skip_ffn=False.

This is the FIRST B7 family where SSM layers do have a downstream FFN
sublayer — contrast canonical Mamba-2 which has none.

Key drifts caught at source-reading time:
- modeling_granitemoehybrid.py:130 — attention scale is `attention_multiplier`,
  NOT 1/sqrt(head_dim). We use AttentionSpec.attn_scale.
- modeling_granitemoehybrid.py:1141 — rotary_emb is None when
  `position_embedding_type != "rope"`. For granite-4.0-h-micro it's "nope".
  AttentionSpec.rope = None.
- modeling_granitemoehybrid.py:1167 — embeddings are pre-scaled by
  embedding_multiplier (carried via DecoderBlockSpec.embedding_scale).
- modeling_granitemoehybrid.py:1084,1094 — residual_multiplier is applied
  to the SUBLAYER OUTPUT before the residual add. Mirrors Granite B2a
  exactly (api/block.py self._residual_scale).
- modeling_granitemoehybrid.py:1088-1092 — when has_experts is False
  (granite-4-h-micro: num_local_experts == 0), the channel mixer is JUST
  the shared_mlp. When experts > 0 (larger variants), the channel mixer is
  shared_mlp(x) + block_sparse_moe(x). We land the num_local_experts == 0
  case for B7; MoE variant deferred (the shared+moe sum is not a single
  FFNSpec or MoESpec — needs a small MoE+shared composition).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Granite4HConfig:
    """Granite 4 H hybrid config (subset for one decoder layer)."""
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int             # shared_mlp intermediate
    shared_intermediate_size: int      # alias used by HF; equals intermediate_size for dense
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    # μP scales
    embedding_multiplier: float
    logits_scaling: float
    residual_multiplier: float
    attention_multiplier: float
    # Per-layer dispatch
    layer_types: tuple[str, ...]
    # Mamba params (same as Mamba2)
    mamba_n_heads: int
    mamba_n_groups: int
    mamba_d_state: int
    mamba_d_head: int
    mamba_d_conv: int
    mamba_expand: int
    mamba_chunk_size: int
    mamba_conv_bias: bool
    mamba_proj_bias: bool
    time_step_min: float
    time_step_max: float
    time_step_floor: float
    time_step_limit: tuple[float, float]
    # Attention params
    attention_bias: bool
    position_embedding_type: str       # "rope" | "nope"
    rope_theta: float                  # only used if position_embedding_type == "rope"
    num_local_experts: int             # 0 means dense (no MoE composition)

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Granite4HConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        layer_types = hf.get("layer_types") or hf.get("layers_block_type")
        if layer_types is None:
            n = hf["num_hidden_layers"]
            layer_types = ["mamba"] * n
        layer_types = tuple(layer_types)

        tsl_raw = hf.get("time_step_limit", (0.0, float("inf")))
        if isinstance(tsl_raw, (list, tuple)) and len(tsl_raw) == 2:
            tsl = (float(tsl_raw[0]), float(tsl_raw[1]))
        else:
            tsl = (0.0, float("inf"))

        rope_theta = 10000.0
        if isinstance(hf.get("rope_parameters"), dict):
            rope_theta = float(hf["rope_parameters"].get("rope_theta", 10000.0))
        elif "rope_theta" in hf:
            rope_theta = float(hf["rope_theta"])

        # Determine mamba_d_head: defaults to mamba_expand * hidden // n_heads
        # when the HF config has "auto".
        m_n_heads = hf.get("mamba_n_heads", 128)
        m_expand = hf.get("mamba_expand", 2)
        hidden = hf["hidden_size"]
        m_d_head = hf.get("mamba_d_head", "auto")
        if m_d_head == "auto" or m_d_head is None:
            m_d_head = (m_expand * hidden) // m_n_heads

        return cls(
            hidden_size=hidden,
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            intermediate_size=hf.get("intermediate_size", 11008),
            shared_intermediate_size=hf.get("shared_intermediate_size", 1024),
            num_hidden_layers=hf["num_hidden_layers"],
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf.get("max_position_embeddings", 2048),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            embedding_multiplier=float(hf.get("embedding_multiplier", 1.0)),
            logits_scaling=float(hf.get("logits_scaling", 1.0)),
            residual_multiplier=float(hf.get("residual_multiplier", 1.0)),
            attention_multiplier=float(hf.get("attention_multiplier", 1.0)),
            layer_types=layer_types,
            mamba_n_heads=m_n_heads,
            mamba_n_groups=hf.get("mamba_n_groups", 1),
            mamba_d_state=hf.get("mamba_d_state", 256),
            mamba_d_head=m_d_head,
            mamba_d_conv=hf.get("mamba_d_conv", 4),
            mamba_expand=m_expand,
            mamba_chunk_size=hf.get("mamba_chunk_size", 256),
            mamba_conv_bias=bool(hf.get("mamba_conv_bias", True)),
            mamba_proj_bias=bool(hf.get("mamba_proj_bias", False)),
            time_step_min=float(hf.get("time_step_min", 0.001)),
            time_step_max=float(hf.get("time_step_max", 0.1)),
            time_step_floor=float(hf.get("time_step_floor", 1e-4)),
            time_step_limit=tsl,
            attention_bias=bool(hf.get("attention_bias", False)),
            position_embedding_type=hf.get("position_embedding_type") or "nope",
            rope_theta=rope_theta,
            num_local_experts=hf.get("num_local_experts", 0) or 0,
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def is_mamba_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "mamba"

    def to_block_spec(self, layer_idx: int) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS,
            eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )

        # Shared MLP — GraniteMoeHybridMLP is a FUSED-gate-up SwiGLU.
        # Source: modeling_granitemoehybrid.py:768 (`input_linear =
        # Linear(hidden, 2 * shared_intermediate_size, bias=False)`),
        # 772-775 (`up = input_linear(x); gate, up = up.chunk(2, -1);
        # x = activation(gate) * up; output_linear(x)`).
        # This is the SAME shape and order as Phi-3 (we already support
        # fused_gate_up=True via api/feedforward.FeedForward).
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.shared_intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=True,
            gate_bias=False, up_bias=False, down_bias=False,
        )

        is_mamba = self.is_mamba_layer(layer_idx)
        if is_mamba:
            ssm_inner = self.mamba_expand * self.hidden_size
            assert self.mamba_n_heads * self.mamba_d_head == ssm_inner, (
                f"Granite-4-H: n_heads*d_head={self.mamba_n_heads*self.mamba_d_head} "
                f"!= expand*hidden={ssm_inner}"
            )
            ssm = specs.SSMSpec(
                d_state=self.mamba_d_state,
                d_conv=self.mamba_d_conv,
                d_inner=ssm_inner,
                expand_factor=self.mamba_expand,
                dt_min=self.time_step_min,
                dt_max=self.time_step_max,
                dt_init_floor=self.time_step_floor,
                conv_bias=self.mamba_conv_bias,
                bias=self.mamba_proj_bias,
                activation=types.Activation.SILU,
            )
            token_mixer: object = specs.SSDSpec(
                base=ssm,
                chunk_size=self.mamba_chunk_size,
                headdim=self.mamba_d_head,
                ngroups=self.mamba_n_groups,
                n_heads=self.mamba_n_heads,
                time_step_limit_low=self.time_step_limit[0],
                time_step_limit_high=self.time_step_limit[1],
                layer_norm_epsilon=self.rms_norm_eps,
            )
        else:
            # NoPE attention with `attention_multiplier` as scale.
            rope_spec = None
            if self.position_embedding_type == "rope":
                rope_spec = specs.RoPESpec(
                    base_theta=self.rope_theta,
                    basis=types.RoPEBasis.SPLIT_HALF,
                )
            token_mixer = specs.AttentionSpec(
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
                attn_scale=self.attention_multiplier,
                rope=rope_spec,
            )

        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=token_mixer,
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
            residual_scale=self.residual_multiplier,
            skip_ffn=False,                  # Granite-4-H always has the shared_mlp
        )
