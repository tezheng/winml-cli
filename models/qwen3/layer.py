"""Qwen3 decoder-layer factory.

Builds a DecoderBlock from a Qwen3Config using only api/ primitives.
Does NOT import transformers.models.qwen3.modeling_qwen3.
"""
from __future__ import annotations

from api import block
from models.qwen3 import config as _config


def build_qwen3_decoder_layer(
    cfg: _config.Qwen3Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Qwen3 decoder block at layer_idx (currently identity)."""
    spec = cfg.to_block_spec()
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_qwen3_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Qwen3 layer weights into the api-assembled DecoderBlock.

    Mapping table (HF tensor name -> API tensor slot):
        model.layers.{L}.input_layernorm.weight
            -> blk.pre_attn_norm.weight
        model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
            -> blk.attention.{q,k,v,o}_proj.weight
        model.layers.{L}.self_attn.{q,k}_norm.weight
            -> blk.attention.{q,k}_norm.weight
        model.layers.{L}.post_attention_layernorm.weight
            -> blk.pre_ffn_norm.weight
        model.layers.{L}.mlp.{gate,up,down}_proj.weight
            -> blk.feedforward.{gate,up,down}_proj.weight
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    mapping = {
        f"{prefix}.input_layernorm.weight":             blk.pre_attn_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":            blk.attention.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":            blk.attention.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":            blk.attention.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":            blk.attention.o_proj.weight,
        f"{prefix}.post_attention_layernorm.weight":    blk.pre_ffn_norm.weight,
        f"{prefix}.mlp.gate_proj.weight":               blk.feedforward.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight":                 blk.feedforward.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":               blk.feedforward.down_proj.weight,
    }
    if blk.attention.q_norm is not None:
        mapping[f"{prefix}.self_attn.q_norm.weight"] = blk.attention.q_norm.weight
    if blk.attention.k_norm is not None:
        mapping[f"{prefix}.self_attn.k_norm.weight"] = blk.attention.k_norm.weight

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")

    import torch as _torch
    with _torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
