"""Voxtral LM-decoder-layer factory + HF weight loader.

Voxtral's text decoder is a Llama-style block (the canonical
`mistralai/Voxtral-Mini-3B-2507` checkpoint sets `text_config.model_type='llama'`).
The block is architecturally identical to Mistral 7B v0.3 / Llama 3 with
`rope_theta=1e8` and `sliding_window=None`.

We don't reinvent the layer — we use api.block.DecoderBlock directly.

HF state-dict layout: Voxtral wraps the text decoder under
`model.language_model`, so layer-i tensors live at
`model.language_model.layers.{i}.*`. Within each layer the names are the
canonical Llama names (`input_layernorm`, `self_attn.{q,k,v,o}_proj`,
`post_attention_layernorm`, `mlp.{gate,up,down}_proj`).

Source: modeling_voxtral.py:384-481 (VoxtralModel — `self.language_model
= AutoModel.from_config(text_config)`). The state-dict prefix is verified
by building a VoxtralForConditionalGeneration with a tiny LlamaConfig and
inspecting keys.
"""
from __future__ import annotations

import torch

from api import block
from models.voxtral import config as _config


def build_voxtral_decoder_layer(
    cfg: _config.VoxtralConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Voxtral LM-decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_voxtral_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    prefix: str = "model.language_model.layers",
) -> None:
    """Copy HF Voxtral LM-decoder layer weights into the api DecoderBlock.

    Default prefix matches the VoxtralForConditionalGeneration state-dict
    naming (`model.language_model.layers.{L}.*`). Pass an alternate prefix
    if the caller pre-strips the `model.` namespace.
    """
    L = layer_idx
    pfx = f"{prefix}.{L}"
    mapping = {
        f"{pfx}.input_layernorm.weight":             blk.pre_attn_norm.weight,
        f"{pfx}.self_attn.q_proj.weight":            blk.attention.q_proj.weight,
        f"{pfx}.self_attn.k_proj.weight":            blk.attention.k_proj.weight,
        f"{pfx}.self_attn.v_proj.weight":            blk.attention.v_proj.weight,
        f"{pfx}.self_attn.o_proj.weight":            blk.attention.o_proj.weight,
        f"{pfx}.post_attention_layernorm.weight":    blk.pre_ffn_norm.weight,
        f"{pfx}.mlp.gate_proj.weight":               blk.feedforward.gate_proj.weight,
        f"{pfx}.mlp.up_proj.weight":                 blk.feedforward.up_proj.weight,
        f"{pfx}.mlp.down_proj.weight":               blk.feedforward.down_proj.weight,
    }
    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing[:6]}")

    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
