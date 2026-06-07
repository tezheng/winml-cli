"""MiniCPM-3 decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a MiniCPM3Config using only api/ primitives.
Does NOT import any minicpm/modeling_minicpm.py module from HF (the openbmb
custom file is only used as a reference and target for numerical equivalence).

Verified against (hf_cache/.../openbmb--MiniCPM3-4B/modeling_minicpm.py):
- 331-385  MiniCPMAttention __init__ (q_a_proj / q_a_layernorm / q_b_proj,
           kv_a_proj_with_mqa / kv_a_layernorm / kv_b_proj, o_proj).
- 427-526  MiniCPMAttention.forward (the MLA flow now in api.attention._forward_mla).
- 886-958  MiniCPMDecoderLayer.forward (input_layernorm → self_attn → residual
           with `* (scale_depth/sqrt(L))`; post_attention_layernorm → mlp →
           same residual scale).
"""
from __future__ import annotations

import torch

from api import block
from models.minicpm3 import config as _config


def build_minicpm3_decoder_layer(
    cfg: _config.MiniCPM3Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one MiniCPM-3 decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_minicpm3_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF MiniCPM-3 layer weights into the api-assembled DecoderBlock.

    HF tensor names (verified against modeling_minicpm.py:331-385 + decoder
    layer 886-958):

        model.layers.{L}.input_layernorm.weight
            -> blk.pre_attn_norm.weight
        model.layers.{L}.post_attention_layernorm.weight
            -> blk.pre_ffn_norm.weight

        model.layers.{L}.self_attn.q_a_proj.weight
            -> blk.attention.q_a_proj.weight
        model.layers.{L}.self_attn.q_a_layernorm.weight
            -> blk.attention.q_a_layernorm.weight
        model.layers.{L}.self_attn.q_b_proj.weight
            -> blk.attention.q_b_proj.weight
        model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight
            -> blk.attention.kv_a_proj_with_mqa.weight
        model.layers.{L}.self_attn.kv_a_layernorm.weight
            -> blk.attention.kv_a_layernorm.weight
        model.layers.{L}.self_attn.kv_b_proj.weight
            -> blk.attention.kv_b_proj.weight
        model.layers.{L}.self_attn.o_proj.weight
            -> blk.attention.o_proj.weight

        model.layers.{L}.mlp.gate_proj.weight
            -> blk.feedforward.gate_proj.weight
        model.layers.{L}.mlp.up_proj.weight
            -> blk.feedforward.up_proj.weight
        model.layers.{L}.mlp.down_proj.weight
            -> blk.feedforward.down_proj.weight

    Optional biases (only when config.attention_bias=True):
        q_a_proj.bias, kv_a_proj_with_mqa.bias, o_proj.bias
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    attn = blk.attention
    mapping = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.self_attn.q_a_proj.weight":        attn.q_a_proj.weight,
        f"{prefix}.self_attn.q_a_layernorm.weight":   attn.q_a_layernorm.weight,
        f"{prefix}.self_attn.q_b_proj.weight":        attn.q_b_proj.weight,
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": attn.kv_a_proj_with_mqa.weight,
        f"{prefix}.self_attn.kv_a_layernorm.weight":  attn.kv_a_layernorm.weight,
        f"{prefix}.self_attn.kv_b_proj.weight":       attn.kv_b_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":          attn.o_proj.weight,
        f"{prefix}.mlp.gate_proj.weight":             blk.feedforward.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight":               blk.feedforward.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":             blk.feedforward.down_proj.weight,
    }
    # Optional biases.
    for hf_name, slot in (
        (f"{prefix}.self_attn.q_a_proj.bias",            getattr(attn.q_a_proj, "bias", None)),
        (f"{prefix}.self_attn.kv_a_proj_with_mqa.bias",  getattr(attn.kv_a_proj_with_mqa, "bias", None)),
        (f"{prefix}.self_attn.o_proj.bias",              getattr(attn.o_proj, "bias", None)),
    ):
        if slot is not None and hf_name in hf_state_dict:
            mapping[hf_name] = slot

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
