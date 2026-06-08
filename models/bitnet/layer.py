"""BitNet b1.58 decoder-layer factory.

Builds a DecoderBlock from a BitNetConfig using api/ primitives.
"""
from __future__ import annotations

import torch

from api import block
from models.bitnet import config as _config


def build_bitnet_decoder_layer(
    cfg: _config.BitNetConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec()
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(spec=spec, hidden_size=cfg.hidden_size,
                              max_seq=seq, dtype=cfg.dtype)


def load_hf_bitnet_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    prefix: str = "model.layers",
) -> None:
    """Copy HF BitNet layer weights into the api-assembled DecoderBlock.

    Mapping (HF tensor name -> API tensor slot):
        {prefix}.{L}.input_layernorm.weight
            -> blk.pre_attn_norm.weight
        {prefix}.{L}.self_attn.{q,k,v,o}_proj.weight
            -> blk.attention.{q,k,v,o}_proj.weight
        {prefix}.{L}.self_attn.attn_sub_norm.weight
            -> blk.attention.attn_sub_norm.weight
        {prefix}.{L}.post_attention_layernorm.weight
            -> blk.pre_ffn_norm.weight
        {prefix}.{L}.mlp.{gate,up,down}_proj.weight
            -> blk.feedforward.{gate,up,down}_proj.weight
        {prefix}.{L}.mlp.ffn_sub_norm.weight
            -> blk.feedforward.ffn_sub_norm.weight
    """
    L = layer_idx
    p = f"{prefix}.{L}"
    mapping = {
        f"{p}.input_layernorm.weight": blk.pre_attn_norm.weight,
        f"{p}.self_attn.q_proj.weight": blk.attention.q_proj.weight,
        f"{p}.self_attn.k_proj.weight": blk.attention.k_proj.weight,
        f"{p}.self_attn.v_proj.weight": blk.attention.v_proj.weight,
        f"{p}.self_attn.o_proj.weight": blk.attention.o_proj.weight,
        f"{p}.self_attn.attn_sub_norm.weight": blk.attention.attn_sub_norm.weight,
        f"{p}.post_attention_layernorm.weight": blk.pre_ffn_norm.weight,
        f"{p}.mlp.gate_proj.weight": blk.feedforward.gate_proj.weight,
        f"{p}.mlp.up_proj.weight": blk.feedforward.up_proj.weight,
        f"{p}.mlp.down_proj.weight": blk.feedforward.down_proj.weight,
        f"{p}.mlp.ffn_sub_norm.weight": blk.feedforward.ffn_sub_norm.weight,
    }
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
