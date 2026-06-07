"""Llama 3 / 3.1 / 3.2 decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a Llama3Config using only api/ primitives. Does NOT
import transformers.models.llama.modeling_llama.

Architectural parity vs Qwen3: same shape (PRE-norm RMSNorm STANDARD_W, SPLIT
QKV, GQA SwiGLU, no biases), differs only in:
- No QK-norm (Llama 3 attention has no q_norm / k_norm).
- RoPE variant: 3.0 = default (rope_type='default'); 3.1 / 3.2 = LLAMA3 smooth
  scaling (rope_type='llama3').

Verified against modeling_llama.py:225-289 (LlamaAttention — q_proj/k_proj/
v_proj/o_proj without QK-norm), and modeling_llama.py:292-332 (LlamaDecoderLayer
forward order: residual + attn(input_layernorm(x)); residual + mlp(post_attn_ln(x))).
"""
from __future__ import annotations

from api import block
from models.llama3 import config as _config


def build_llama3_decoder_layer(
    cfg: _config.Llama3Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Llama 3 decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_llama3_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Llama 3 layer weights into the api-assembled DecoderBlock.

    Mapping table (verified against modeling_llama.py:296-301 + 238-249):
        model.layers.{L}.input_layernorm.weight
            -> blk.pre_attn_norm.weight
        model.layers.{L}.self_attn.{q,k,v,o}_proj.weight (+ optional .bias)
            -> blk.attention.{q,k,v,o}_proj.{weight,bias}
        model.layers.{L}.post_attention_layernorm.weight
            -> blk.pre_ffn_norm.weight
        model.layers.{L}.mlp.{gate,up,down}_proj.weight (+ optional .bias)
            -> blk.feedforward.{gate,up,down}_proj.{weight,bias}
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    mapping = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":          blk.attention.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":          blk.attention.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":          blk.attention.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":          blk.attention.o_proj.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.mlp.gate_proj.weight":             blk.feedforward.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight":               blk.feedforward.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":             blk.feedforward.down_proj.weight,
    }
    # Optional biases (Llama 3 has none, but the loader honours the spec).
    for hf_name, slot in (
        (f"{prefix}.self_attn.q_proj.bias", getattr(blk.attention.q_proj, "bias", None)),
        (f"{prefix}.self_attn.k_proj.bias", getattr(blk.attention.k_proj, "bias", None)),
        (f"{prefix}.self_attn.v_proj.bias", getattr(blk.attention.v_proj, "bias", None)),
        (f"{prefix}.self_attn.o_proj.bias", getattr(blk.attention.o_proj, "bias", None)),
        (f"{prefix}.mlp.gate_proj.bias", getattr(blk.feedforward.gate_proj, "bias", None)),
        (f"{prefix}.mlp.up_proj.bias", getattr(blk.feedforward.up_proj, "bias", None)),
        (f"{prefix}.mlp.down_proj.bias", getattr(blk.feedforward.down_proj, "bias", None)),
    ):
        if slot is not None and hf_name in hf_state_dict:
            mapping[hf_name] = slot

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
