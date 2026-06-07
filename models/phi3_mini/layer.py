"""Phi-3 mini decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a Phi3MiniConfig using only api/ primitives. Does NOT
import transformers.models.phi3.modeling_phi3.

Architectural differences vs Llama 3:
- FUSED QKV (one qkv_proj of shape [hidden, (Hq+2*Hk)*Dh]).
- FUSED gate_up (one gate_up_proj of shape [hidden, 2*I]).
- SWA (sliding_window=2047) for the 4k variant.
- Default RoPE θ=10000 (no scaling) for the 4k variant; partial_rotary_factor=1.0.

Verified against:
- modeling_phi3.py:208-271 (Phi3Attention — qkv_proj + slicing).
- modeling_phi3.py:49-64 (Phi3MLP — gate_up_proj + chunk(2,-1)).
- modeling_phi3.py:295-335 (Phi3DecoderLayer.forward — order is
  input_layernorm → self_attn → residual; post_attention_layernorm → mlp →
  residual).
"""
from __future__ import annotations

from api import block
from models.phi3_mini import config as _config


def build_phi3_mini_decoder_layer(
    cfg: _config.Phi3MiniConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_phi3_mini_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Phi-3 layer weights into the api-assembled DecoderBlock.

    Phi-3 HF tensor names (verified against modeling_phi3.py:295-302):
        model.layers.{L}.input_layernorm.weight
            -> blk.pre_attn_norm.weight
        model.layers.{L}.self_attn.qkv_proj.weight                ← FUSED
            -> blk.attention.qkv_proj.weight
        model.layers.{L}.self_attn.o_proj.weight
            -> blk.attention.o_proj.weight
        model.layers.{L}.post_attention_layernorm.weight
            -> blk.pre_ffn_norm.weight
        model.layers.{L}.mlp.gate_up_proj.weight                  ← FUSED
            -> blk.feedforward.gate_up_proj.weight
        model.layers.{L}.mlp.down_proj.weight
            -> blk.feedforward.down_proj.weight

    The fused projections come straight from HF — we copy them verbatim. The
    slicing/chunking happens at forward time inside api.attention.Attention
    and api.feedforward.FeedForward.
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    mapping = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.self_attn.qkv_proj.weight":        blk.attention.qkv_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":          blk.attention.o_proj.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.mlp.gate_up_proj.weight":          blk.feedforward.gate_up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":             blk.feedforward.down_proj.weight,
    }
    # Optional biases (Phi-3 has none).
    for hf_name, slot in (
        (f"{prefix}.self_attn.qkv_proj.bias", getattr(blk.attention.qkv_proj, "bias", None)),
        (f"{prefix}.self_attn.o_proj.bias", getattr(blk.attention.o_proj, "bias", None)),
        (f"{prefix}.mlp.gate_up_proj.bias", getattr(blk.feedforward.gate_up_proj, "bias", None)),
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
