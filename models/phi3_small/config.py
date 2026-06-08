"""Phi-3-small (microsoft/Phi-3-small-8k-instruct) config.

SHAPE-ONLY scope for B2b — the spec wiring suffices to construct a layer
and verify weight shapes; numerical equivalence vs HF is deferred.

Verified against `hf_cache/.../Phi-3-small-8k-instruct/config.json` and
`modeling_phi3_small.py` (microsoft custom; no native HF package).
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
class BlockSparseParams:
    block_size: int
    num_local_blocks: int
    vert_stride: int
    kernel_block_size: int
    homo_head_pattern: bool


@dataclass(frozen=True)
class Phi3SmallConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rope_theta: float
    layer_norm_epsilon: float
    vocab_size: int
    max_position_embeddings: int
    dense_attention_every_n_layers: int
    dtype: torch.dtype
    # GeGELU
    gegelu_limit: float = 20.0
    # BlockSparse
    blocksparse: BlockSparseParams = BlockSparseParams(
        block_size=64, num_local_blocks=16, vert_stride=8,
        kernel_block_size=64, homo_head_pattern=False,
    )
    # μP scalars
    mup_attn_multiplier: float = 1.0
    mup_embedding_multiplier: float = 1.0
    mup_width_multiplier: float = 1.0
    mup_use_scaling: bool = True
    attention_bias: bool = False

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Phi3SmallConfig":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)
        head_dim = int(hf["hidden_size"]) // int(hf["num_attention_heads"])
        bsp = BlockSparseParams(
            block_size=int(hf.get("blocksparse_block_size", 64)),
            num_local_blocks=int(hf.get("blocksparse_num_local_blocks", 16)),
            vert_stride=int(hf.get("blocksparse_vert_stride", 8)),
            kernel_block_size=int(hf.get(
                "blocksparse_triton_kernel_block_size", 64,
            )),
            homo_head_pattern=bool(hf.get("blocksparse_homo_head_pattern", False)),
        )
        return cls(
            hidden_size=int(hf["hidden_size"]),
            num_attention_heads=int(hf["num_attention_heads"]),
            num_key_value_heads=int(hf["num_key_value_heads"]),
            head_dim=head_dim,
            intermediate_size=int(hf["ff_intermediate_size"]),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            rope_theta=float(hf.get("rope_embedding_base", 10_000.0)),
            layer_norm_epsilon=float(hf.get("layer_norm_epsilon", 1e-5)),
            vocab_size=int(hf["vocab_size"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            dense_attention_every_n_layers=int(hf.get(
                "dense_attention_every_n_layers", 2,
            )),
            dtype=dtype,
            gegelu_limit=float(hf.get("gegelu_limit", 20.0)),
            blocksparse=bsp,
            mup_attn_multiplier=float(hf.get("mup_attn_multiplier", 1.0)),
            mup_embedding_multiplier=float(hf.get("mup_embedding_multiplier", 1.0)),
            mup_width_multiplier=float(hf.get("mup_width_multiplier", 1.0)),
            mup_use_scaling=bool(hf.get("mup_use_scaling", True)),
            attention_bias=bool(hf.get("attention_bias", False)),
        )

    def is_dense_layer(self, layer_idx: int) -> bool:
        """Source: modeling_phi3_small.py:213. Dense if (L+1) %
        dense_attention_every_n_layers == 0 (1-indexed)."""
        if not self.dense_attention_every_n_layers:
            return True
        return (layer_idx + 1) % self.dense_attention_every_n_layers == 0

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        # Phi-3-small uses LayerNorm (not RMSNorm). The api.norm.RMSNorm only
        # supports RMS — for shape-only this is acceptable, but we still need
        # the NormSpec to carry LAYER kind. The api Block constructor will
        # raise NotImplementedError for LAYER which is consistent with the
        # shape-only contract; the test below constructs the spec but only
        # asserts on weight shapes built by direct nn.Linear, not by
        # api.block.DecoderBlock.
        norm_spec = specs.NormSpec(
            kind=types.NormKind.LAYER, eps=self.layer_norm_epsilon,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        rope_spec = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.NONE,
        )
        is_dense = self.is_dense_layer(layer_idx)
        mask_kind = (types.MaskKind.CAUSAL if is_dense
                     else types.MaskKind.BLOCK_SPARSE)
        # softmax_scale = mup_attn_multiplier / head_dim (NOT 1/sqrt) when
        # mup_use_scaling. Source: modeling_phi3_small.py:200-206.
        if self.mup_use_scaling:
            attn_scale = self.mup_attn_multiplier / self.head_dim
        else:
            attn_scale = None     # api defaults to 1/sqrt(head_dim)
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.FUSED,
            mask_kind=mask_kind,
            q_bias=self.attention_bias,
            k_bias=self.attention_bias,
            v_bias=self.attention_bias,
            o_bias=self.attention_bias,
            attn_scale=attn_scale,
            rope=rope_spec,
        )
        ffn_spec = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.GEGELU,
            gate_kind=types.GateKind.GEGLU,
            fused_gate_up=True,
            gate_bias=self.attention_bias,
            up_bias=self.attention_bias,
            down_bias=self.attention_bias,
        )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=ffn_spec,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
