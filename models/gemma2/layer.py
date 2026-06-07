"""Gemma 2 decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a Gemma2Config using only api/ primitives. Does NOT
import transformers.models.gemma2.modeling_gemma2.

Layer-index dispatch:
- Sliding layers (default: even-1-indexed i.e. 0, 2, 4, ...): SWA mask with
  sliding_window from config.
- Full layers (1, 3, 5, ...): full causal mask.

Both layer types share the same weight names and tensors; the only difference
is the attention mask kind (carried in the AttentionSpec). The weight loader
maps identically for both.
"""
from __future__ import annotations

import torch

from api import block
from models.gemma2 import config as _config


def build_gemma2_decoder_layer(
    cfg: _config.Gemma2Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Gemma 2 decoder block at layer_idx."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_gemma2_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Gemma 2 layer weights into the api-assembled DecoderBlock.

    Mapping table (HF tensor name -> API tensor slot). Source for HF naming:
    `Gemma2DecoderLayer.__init__` (modeling_gemma2.py L302-313) and
    `Gemma2Attention.__init__` (L233-258).

        model.layers.{L}.input_layernorm.weight
            -> blk.pre_attn_norm.weight
        model.layers.{L}.post_attention_layernorm.weight
            -> blk.post_attn_sublayer_norm.weight
        model.layers.{L}.pre_feedforward_layernorm.weight
            -> blk.pre_ffn_norm.weight
        model.layers.{L}.post_feedforward_layernorm.weight
            -> blk.post_ffn_sublayer_norm.weight
        model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
            -> blk.attention.{q,k,v,o}_proj.weight
        model.layers.{L}.mlp.{gate,up,down}_proj.weight
            -> blk.feedforward.{gate,up,down}_proj.weight
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    mapping = {
        f"{prefix}.input_layernorm.weight":            blk.pre_attn_norm.weight,
        f"{prefix}.post_attention_layernorm.weight":   blk.post_attn_sublayer_norm.weight,
        f"{prefix}.pre_feedforward_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.post_feedforward_layernorm.weight": blk.post_ffn_sublayer_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":           blk.attention.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":           blk.attention.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":           blk.attention.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":           blk.attention.o_proj.weight,
        f"{prefix}.mlp.gate_proj.weight":              blk.feedforward.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight":                blk.feedforward.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":              blk.feedforward.down_proj.weight,
    }
    # Optional biases — Gemma 2 doesn't use them but accept if present.
    if blk.attention.q_proj.bias is not None:
        mapping[f"{prefix}.self_attn.q_proj.bias"] = blk.attention.q_proj.bias
        mapping[f"{prefix}.self_attn.k_proj.bias"] = blk.attention.k_proj.bias
        mapping[f"{prefix}.self_attn.v_proj.bias"] = blk.attention.v_proj.bias
        mapping[f"{prefix}.self_attn.o_proj.bias"] = blk.attention.o_proj.bias

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")

    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
