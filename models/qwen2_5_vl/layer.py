"""Qwen2.5-VL LM-decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a Qwen2_5_VLConfig using only api/ primitives.
Does NOT import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.

Architectural parity vs Qwen2: identical projection shapes and biases —
the ONLY runtime difference is M-RoPE (multimodal rotary position
embedding), which is dispatched inside api.rope when position_ids has
shape [3, B, S] and `RoPESpec.mrope_section` is set.

For text-only inference (mrope_section unused at runtime), the three
position-axis components are all equal and M-RoPE reduces algebraically
to standard 1D RoPE. For mixed image+text inputs, the (T, H, W) axes
are populated by the HF processor's `get_rope_index` routine.

Source citations:
- modeling_qwen2_5_vl.py:485-545 (Qwen2_5_VLRotaryEmbedding — 3-axis
  position_ids).
- modeling_qwen2_5_vl.py:564-606 (apply_multimodal_rotary_pos_emb).
- modeling_qwen2_5_vl.py:609-696 (Qwen2_5_VLAttention).
- modeling_qwen2_5_vl.py:699-764 (Qwen2_5_VLDecoderLayer).
- modeling_qwen2_5_vl.py:766-840 (Qwen2_5_VLTextModel — `.language_model`
  attribute prefix on the multimodal-model state-dict).
"""
from __future__ import annotations

import torch

from api import block
from models.qwen2_5_vl import config as _config


def build_qwen2_5_vl_decoder_layer(
    cfg: _config.Qwen2_5_VLConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Qwen2.5-VL LM-decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_qwen2_5_vl_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    prefix: str = "model.language_model.layers",
) -> None:
    """Copy HF Qwen2.5-VL LM-decoder layer weights into the api DecoderBlock.

    HF model surface: `Qwen2_5_VLModel.language_model` is a
    `Qwen2_5_VLTextModel`. Layer tensors live at
    `model.language_model.layers.{L}.*` with the same intra-layer names
    as Qwen2.
    """
    L = layer_idx
    pfx = f"{prefix}.{L}"
    mapping = {
        f"{pfx}.input_layernorm.weight":             blk.pre_attn_norm.weight,
        f"{pfx}.self_attn.q_proj.weight":            blk.attention.q_proj.weight,
        f"{pfx}.self_attn.q_proj.bias":              blk.attention.q_proj.bias,
        f"{pfx}.self_attn.k_proj.weight":            blk.attention.k_proj.weight,
        f"{pfx}.self_attn.k_proj.bias":              blk.attention.k_proj.bias,
        f"{pfx}.self_attn.v_proj.weight":            blk.attention.v_proj.weight,
        f"{pfx}.self_attn.v_proj.bias":              blk.attention.v_proj.bias,
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
