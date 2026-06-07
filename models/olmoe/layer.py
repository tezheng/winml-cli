"""OLMoE decoder-layer factory + HF weight loader.

Verified against `transformers/models/olmoe/modeling_olmoe.py`:
- 220-298  OlmoeAttention (FULL_HDH q_norm / k_norm PRE-RoPE, no biases).
- 301-339  OlmoeExperts (packed gate_up_proj [E, 2I, H] / down_proj).
- 341-359  OlmoeTopKRouter (softmax + optional norm_topk_prob).
- 362-375  OlmoeSparseMoeBlock (router → experts; no shared experts).
- 378-416  OlmoeDecoderLayer (PRE-norm).
"""
from __future__ import annotations
from typing import Optional

import torch

from api import block
from models.olmoe import config as _config


def build_olmoe_decoder_layer(
    cfg: _config.OlmoeConfig,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_olmoe_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    cfg: _config.OlmoeConfig,
) -> None:
    """Copy HF OLMoE layer tensors into the api DecoderBlock.

    HF state-dict names (verified at modeling_olmoe.py:378-416 + 246-249 +
    301-375):
      model.layers.{L}.input_layernorm.weight
      model.layers.{L}.post_attention_layernorm.weight
      model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
      model.layers.{L}.self_attn.q_norm.weight   # FULL Q dim
      model.layers.{L}.self_attn.k_norm.weight   # FULL K dim
      model.layers.{L}.mlp.gate.weight
      model.layers.{L}.mlp.experts.gate_up_proj
      model.layers.{L}.mlp.experts.down_proj
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    attn = blk.attention
    moe = blk.feedforward
    mapping: dict[str, torch.Tensor] = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":          attn.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":          attn.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":          attn.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":          attn.o_proj.weight,
        f"{prefix}.self_attn.q_norm.weight":          attn.q_norm.weight,
        f"{prefix}.self_attn.k_norm.weight":          attn.k_norm.weight,
        f"{prefix}.mlp.gate.weight":                  moe.gate.weight,
        f"{prefix}.mlp.experts.gate_up_proj":         moe.experts_gate_up,
        f"{prefix}.mlp.experts.down_proj":            moe.experts_down,
    }
    if cfg.attention_bias:
        mapping[f"{prefix}.self_attn.q_proj.bias"] = attn.q_proj.bias
        mapping[f"{prefix}.self_attn.k_proj.bias"] = attn.k_proj.bias
        mapping[f"{prefix}.self_attn.v_proj.bias"] = attn.v_proj.bias
        mapping[f"{prefix}.self_attn.o_proj.bias"] = attn.o_proj.bias

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
