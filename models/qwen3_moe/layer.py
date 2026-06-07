"""Qwen3-MoE decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a Qwen3MoeConfig using only api/ primitives.

Verified against `transformers/models/qwen3_moe/modeling_qwen3_moe.py`:
- 126-195  Qwen3MoeAttention (Q/K-norm PRE-RoPE PER_HEAD_DH, no biases).
- 198-211  Qwen3MoeMLP (dense SwiGLU using config.intermediate_size).
- 215-251  Qwen3MoeExperts (packed gate_up_proj [E, 2I, H] / down_proj).
- 254-272  Qwen3MoeTopKRouter (softmax + optional norm_topk_prob).
- 275-286  Qwen3MoeSparseMoeBlock (router → experts; no shared experts).
- 310-353  Qwen3MoeDecoderLayer (PRE-norm; per-layer dense vs MoE).
"""
from __future__ import annotations
from typing import Optional

import torch

from api import block, specs
from models.qwen3_moe import config as _config


def build_qwen3_moe_decoder_layer(
    cfg: _config.Qwen3MoeConfig,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_qwen3_moe_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    cfg: _config.Qwen3MoeConfig,
) -> None:
    """Copy HF Qwen3-MoE layer tensors into the api DecoderBlock.

    Norms + attention tensors are the same across dense / MoE layers; the
    ``mlp.*`` half differs.
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    attn = blk.attention
    mapping: dict[str, torch.Tensor] = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":          attn.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":          attn.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":          attn.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":          attn.o_proj.weight,
    }
    if attn.q_norm is not None:
        mapping[f"{prefix}.self_attn.q_norm.weight"] = attn.q_norm.weight
        mapping[f"{prefix}.self_attn.k_norm.weight"] = attn.k_norm.weight
    if cfg.attention_bias:
        mapping[f"{prefix}.self_attn.q_proj.bias"] = attn.q_proj.bias
        mapping[f"{prefix}.self_attn.k_proj.bias"] = attn.k_proj.bias
        mapping[f"{prefix}.self_attn.v_proj.bias"] = attn.v_proj.bias
        mapping[f"{prefix}.self_attn.o_proj.bias"] = attn.o_proj.bias

    # Channel mixer half: dense vs MoE.
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    if isinstance(spec.channel_mixer, specs.MoESpec):
        moe = blk.feedforward
        mapping[f"{prefix}.mlp.gate.weight"] = moe.gate.weight
        mapping[f"{prefix}.mlp.experts.gate_up_proj"] = moe.experts_gate_up
        mapping[f"{prefix}.mlp.experts.down_proj"]   = moe.experts_down
    else:
        # Dense Qwen3MoeMLP (modeling_qwen3_moe.py:198-211).
        mapping[f"{prefix}.mlp.gate_proj.weight"] = blk.feedforward.gate_proj.weight
        mapping[f"{prefix}.mlp.up_proj.weight"]   = blk.feedforward.up_proj.weight
        mapping[f"{prefix}.mlp.down_proj.weight"] = blk.feedforward.down_proj.weight

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing[:10]}")
    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
