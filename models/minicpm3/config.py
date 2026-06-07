"""MiniCPM-3 (openbmb/MiniCPM3-4B) config + adapter to api specs.

Verified against:
- hf_cache/.../openbmb--MiniCPM3-4B/configuration_minicpm.py and config.json
- hf_cache/.../openbmb--MiniCPM3-4B/modeling_minicpm.py (MLA attention,
  decoder layer scaling).

Key MiniCPM-3 facts:
- MLA attention. q_lora_rank=768, kv_lora_rank=256, qk_nope_head_dim=64,
  qk_rope_head_dim=32, v_head_dim = hidden_size / num_attention_heads = 64.
  Source: modeling_minicpm.py:351-357.
- Llama-shaped block envelope: PRE-norm RMSNorm STANDARD_W, SwiGLU FFN.
- μP residual scaling: per-sublayer multiply by `scale_depth / sqrt(L)` BEFORE
  the residual add. With scale_depth=1.4 and L=62, the factor ≈ 0.1778.
  Source: modeling_minicpm.py:941,948.
- LongRoPE with short_factor == long_factor (16 entries each = qk_rope/2).
  original_max_position_embeddings == max_position_embeddings == 32768 — at
  the boundary the SHORT table is selected (HF condition is
  `seq_len > original_max_position_embeddings`). attention_factor is
  sqrt(1 + log(scale)/log(orig)); with scale==1 → attention_factor == 1.0.
  Source: modeling_minicpm.py:218-222, 225-240.
- scale_emb scales input embeddings; dim_model_base / hidden_size scales
  lm_head output (modeling_minicpm.py:1163 and 1262-ish).
- tie_word_embeddings is typically False for MiniCPM-3 (check from config).
- attention_bias defaults False; no biases in linear layers besides what
  config.attention_bias enables on q_a_proj / kv_a_proj_with_mqa / o_proj.
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
class MiniCPM3Config:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    original_max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    # MiniCPM μP scalars
    scale_emb: float = 1.0
    scale_depth: float = 1.0
    dim_model_base: int = 1
    # LongRoPE factors (None → default RoPE).
    rope_type: str = "default"      # "default" | "longrope"
    longrope_short_factor: Optional[tuple[float, ...]] = None
    longrope_long_factor: Optional[tuple[float, ...]] = None
    attention_bias: bool = False

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "MiniCPM3Config":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # rope_scaling (HF 4.x layout used by MiniCPM3's custom config).
        # Source: modeling_minicpm.py:412-420.
        rs = hf.get("rope_scaling")
        if isinstance(rs, dict) and rs.get("type") == "longrope":
            rope_type = "longrope"
            short_factor = tuple(float(x) for x in rs["short_factor"])
            long_factor = tuple(float(x) for x in rs["long_factor"])
            orig_max = int(rs.get("original_max_position_embeddings",
                                  hf["max_position_embeddings"]))
        else:
            rope_type = "default"
            short_factor = None
            long_factor = None
            orig_max = int(hf.get("original_max_position_embeddings",
                                  hf["max_position_embeddings"]))

        # v_head_dim is NOT in the public config — it is derived as
        # hidden_size // num_attention_heads (modeling_minicpm.py:354).
        v_head_dim = int(hf["hidden_size"]) // int(hf["num_attention_heads"])

        return cls(
            hidden_size=int(hf["hidden_size"]),
            num_attention_heads=int(hf["num_attention_heads"]),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            intermediate_size=int(hf["intermediate_size"]),
            q_lora_rank=int(hf["q_lora_rank"]),
            kv_lora_rank=int(hf["kv_lora_rank"]),
            qk_nope_head_dim=int(hf["qk_nope_head_dim"]),
            qk_rope_head_dim=int(hf["qk_rope_head_dim"]),
            v_head_dim=v_head_dim,
            rope_theta=float(hf.get("rope_theta", 10_000.0)),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=int(hf["vocab_size"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            original_max_position_embeddings=orig_max,
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            scale_emb=float(hf.get("scale_emb", 1.0)),
            scale_depth=float(hf.get("scale_depth", 1.0)),
            dim_model_base=int(hf.get("dim_model_base", 1)),
            rope_type=rope_type,
            longrope_short_factor=short_factor,
            longrope_long_factor=long_factor,
            attention_bias=bool(hf.get("attention_bias", False)),
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        import math
        # μP residual scale: scale_depth / sqrt(num_hidden_layers).
        # Source: modeling_minicpm.py:941, 948.
        residual_scale = self.scale_depth / math.sqrt(self.num_hidden_layers)

        # RoPE on the qk_rope_head_dim slice. MiniCPM-3 rotates ALL of the rope
        # subspace; from the api/rope perspective head_dim = qk_rope_head_dim
        # and partial_rotary_factor = 1.0.
        # LongRoPE attention_factor: sqrt(1 + log(scale)/log(orig_max)) where
        # scale = max_position_embeddings / original_max_position_embeddings.
        # When scale==1 this is 1.0. Source: modeling_minicpm.py:218-222.
        if self.rope_type == "longrope":
            assert self.longrope_short_factor is not None
            assert self.longrope_long_factor is not None
            # MiniCPM-3's LongRoPE attention_factor formula is computed
            # unconditionally — no clamp when scale ≤ 1 (unlike Phi-3 LongRoPE
            # which clamps). Source: modeling_minicpm.py:218-222.
            # For the production config (max == orig == 32768), scale=1 →
            # attention_factor = 1.0 either way.
            scale = self.max_position_embeddings / self.original_max_position_embeddings
            attention_factor = math.sqrt(
                1.0 + math.log(scale) / math.log(self.original_max_position_embeddings)
            )
            rope_spec = specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.LONGROPE,
                longrope_extra=specs.LongRoPEParams(
                    short_factor=self.longrope_short_factor,
                    long_factor=self.longrope_long_factor,
                    original_max_position_embeddings=self.original_max_position_embeddings,
                    attention_factor=attention_factor,
                ),
                partial_rotary_factor=1.0,
                partial_rotary_kind="prefix",  # irrelevant when pr==1.0
            )
        else:
            rope_spec = specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            )

        # NormSpec for q_a_layernorm / kv_a_layernorm (and the pre-attn / pre-ffn
        # block norms). MiniCPM uses one rms_norm_eps for everything.
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )

        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_attention_heads,    # MLA: q/k/v all have H heads
            head_dim=self.qk_head_dim,               # 96
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
            qk_norm=norm_spec,            # reused for q_a_layernorm /
                                          # kv_a_layernorm eps; the spec stays
                                          # generic — the MLA branch consumes
                                          # its eps for the LoRA layernorms.
            rope=rope_spec,
        )
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
            # μP residual scaling — applied inside DecoderBlock.forward.
            # Source: modeling_minicpm.py:941, 948.
            residual_scale=residual_scale,
            # μP model-level scalars (consumed by embedding layer and lm_head
            # respectively; the block does not apply them).
            embedding_scale=self.scale_emb,
            # logits_scale convention in this repo: lm_head MULTIPLIES by it.
            # MiniCPM divides by hidden/dim_model_base, so the equivalent
            # multiplier is dim_model_base / hidden_size.
            logits_scale=self.dim_model_base / self.hidden_size,
        )
