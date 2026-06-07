"""OLMo 2 decoder-layer factory + HF weight loader.

Builds a DecoderBlock from an Olmo2Config using only api/ primitives. Does NOT
import transformers.models.olmo2.modeling_olmo2.
"""
from __future__ import annotations

import torch

from api import block
from models.olmo2 import config as _config


def build_olmo2_decoder_layer(
    cfg: _config.Olmo2Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one OLMo 2 decoder block at layer_idx (currently identity)."""
    spec = cfg.to_block_spec()
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_olmo2_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF OLMo 2 layer weights into the api-assembled DecoderBlock.

    Mapping table (HF tensor name -> API tensor slot). Source for HF naming:
    `Olmo2DecoderLayer` and `Olmo2Attention` (modeling_olmo2.py L209-303).

        model.layers.{L}.self_attn.q_proj.weight
            -> blk.attention.q_proj.weight
        model.layers.{L}.self_attn.k_proj.weight
            -> blk.attention.k_proj.weight
        model.layers.{L}.self_attn.v_proj.weight
            -> blk.attention.v_proj.weight
        model.layers.{L}.self_attn.o_proj.weight
            -> blk.attention.o_proj.weight
        model.layers.{L}.self_attn.q_norm.weight                (FULL_HDH = [Hq*Dh])
            -> blk.attention.q_norm.weight
        model.layers.{L}.self_attn.k_norm.weight                (FULL_HDH = [Hk*Dh])
            -> blk.attention.k_norm.weight
        model.layers.{L}.post_attention_layernorm.weight
            -> blk.post_attn_sublayer_norm.weight                # POST-norm slot
        model.layers.{L}.post_feedforward_layernorm.weight
            -> blk.post_ffn_sublayer_norm.weight                 # POST-norm slot
        model.layers.{L}.mlp.gate_proj.weight
            -> blk.feedforward.gate_proj.weight
        model.layers.{L}.mlp.up_proj.weight
            -> blk.feedforward.up_proj.weight
        model.layers.{L}.mlp.down_proj.weight
            -> blk.feedforward.down_proj.weight
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    mapping = {
        f"{prefix}.self_attn.q_proj.weight":         blk.attention.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":         blk.attention.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":         blk.attention.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":         blk.attention.o_proj.weight,
        f"{prefix}.self_attn.q_norm.weight":         blk.attention.q_norm.weight,
        f"{prefix}.self_attn.k_norm.weight":         blk.attention.k_norm.weight,
        f"{prefix}.post_attention_layernorm.weight": blk.post_attn_sublayer_norm.weight,
        f"{prefix}.post_feedforward_layernorm.weight": blk.post_ffn_sublayer_norm.weight,
        f"{prefix}.mlp.gate_proj.weight":            blk.feedforward.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight":              blk.feedforward.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":            blk.feedforward.down_proj.weight,
    }
    # Biases are optional — only when attention_bias=True (1B / 7B: False).
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
