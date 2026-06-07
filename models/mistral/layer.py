"""Mistral 7B decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a MistralConfig using only api/ primitives. Does NOT
import transformers.models.mistral.modeling_mistral.

Architectural parity vs Llama 3: identical shape (PRE-norm RMSNorm STANDARD_W,
SPLIT QKV, GQA SwiGLU, no biases, no QK-norm). The only attention-side
differences are RoPE theta (default 1_000_000) and sliding window (v0.3 = None;
v0.1/v0.2 = 4096).

Verified against modeling_mistral.py:122-178 (MistralAttention),
35-48 (MistralMLP), 202-240 (MistralDecoderLayer.forward).
"""
from __future__ import annotations

from api import block
from models.mistral import config as _config


def build_mistral_decoder_layer(
    cfg: _config.MistralConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Mistral decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_mistral_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Mistral layer weights into the api-assembled DecoderBlock.

    HF Mistral uses the same tensor names as HF Llama 3 (verified against
    modeling_mistral.py:202-209 — input_layernorm / self_attn.{q,k,v,o}_proj /
    post_attention_layernorm / mlp.{gate,up,down}_proj).
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
