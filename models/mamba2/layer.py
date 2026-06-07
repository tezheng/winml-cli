"""Mamba-2 decoder-layer factory.

Builds a DecoderBlock from a Mamba2Config using only api/ primitives.
Does NOT import transformers.models.mamba2.modeling_mamba2 at construction
time (test paths do, to drive the numerical gate).
"""
from __future__ import annotations

import torch

from api import block
from models.mamba2 import config as _config


def build_mamba2_decoder_layer(
    cfg: _config.Mamba2Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Mamba-2 decoder block at layer_idx."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    # max_seq is irrelevant to SSM blocks (no KV cache); pass a sane default
    # so the DecoderBlock constructor doesn't complain (it forwards to
    # Mamba2Mixer which ignores max_seq).
    seq = max_seq if max_seq is not None else 4096
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_mamba2_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Mamba-2 layer weights into the api-assembled DecoderBlock.

    HF tensor names (verified against state-spaces/mamba2-2.7b state_dict):
        backbone.layers.{L}.norm.weight                  -> blk.pre_attn_norm.weight
        backbone.layers.{L}.mixer.in_proj.weight         -> blk.attention.in_proj.weight
        backbone.layers.{L}.mixer.in_proj.bias?          -> blk.attention.in_proj.bias
        backbone.layers.{L}.mixer.conv1d.weight          -> blk.attention.conv1d.weight
        backbone.layers.{L}.mixer.conv1d.bias?           -> blk.attention.conv1d.bias
        backbone.layers.{L}.mixer.dt_bias                -> blk.attention.dt_bias
        backbone.layers.{L}.mixer.A_log                  -> blk.attention.A_log
        backbone.layers.{L}.mixer.D                      -> blk.attention.D
        backbone.layers.{L}.mixer.norm.weight            -> blk.attention.norm.weight
        backbone.layers.{L}.mixer.out_proj.weight        -> blk.attention.out_proj.weight
        backbone.layers.{L}.mixer.out_proj.bias?         -> blk.attention.out_proj.bias

    Source: modeling_mamba2.py:121-220 (Mamba2Mixer init) + 600-640
    (Mamba2RMSNorm + Mamba2Block init).
    """
    L = layer_idx
    prefix = f"backbone.layers.{L}"
    mixer_prefix = f"{prefix}.mixer"

    pairs: list[tuple[str, torch.Tensor]] = [
        (f"{prefix}.norm.weight",            blk.pre_attn_norm.weight),
        (f"{mixer_prefix}.in_proj.weight",   blk.attention.in_proj.weight),
        (f"{mixer_prefix}.conv1d.weight",    blk.attention.conv1d.weight),
        (f"{mixer_prefix}.dt_bias",          blk.attention.dt_bias),
        (f"{mixer_prefix}.A_log",            blk.attention.A_log),
        (f"{mixer_prefix}.D",                blk.attention.D),
        (f"{mixer_prefix}.norm.weight",      blk.attention.norm.weight),
        (f"{mixer_prefix}.out_proj.weight",  blk.attention.out_proj.weight),
    ]
    # Conditional biases.
    if blk.attention.in_proj.bias is not None:
        pairs.append((f"{mixer_prefix}.in_proj.bias", blk.attention.in_proj.bias))
    if blk.attention.conv1d.bias is not None:
        pairs.append((f"{mixer_prefix}.conv1d.bias", blk.attention.conv1d.bias))
    if blk.attention.out_proj.bias is not None:
        pairs.append((f"{mixer_prefix}.out_proj.bias", blk.attention.out_proj.bias))

    missing = [k for k, _ in pairs if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")

    with torch.no_grad():
        for hf_name, slot in pairs:
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {tuple(src.shape)} "
                    f"vs slot {tuple(slot.shape)}"
                )
            slot.copy_(src.to(slot.dtype))
