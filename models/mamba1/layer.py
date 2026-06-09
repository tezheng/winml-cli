"""Mamba-1 decoder-layer factory (v6 B1)."""
from __future__ import annotations

import torch

from api import block
from models.mamba1 import config as _config


def build_mamba1_decoder_layer(
    cfg: _config.Mamba1Config,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Mamba-1 decoder block."""
    spec = cfg.to_block_spec()
    seq = max_seq if max_seq is not None else 4096
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_mamba1_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    prefix: str = "backbone.layers",
) -> None:
    """Copy HF Mamba-1 layer weights into the api-assembled DecoderBlock.

    HF tensor names (verified against `state-spaces/mamba-130m-hf` state_dict):
        {prefix}.{L}.norm.weight        -> blk.pre_attn_norm.weight
        {prefix}.{L}.mixer.in_proj.weight  -> blk.attention.in_proj.weight
        {prefix}.{L}.mixer.conv1d.weight   -> blk.attention.conv1d.weight
        {prefix}.{L}.mixer.conv1d.bias?    -> blk.attention.conv1d.bias
        {prefix}.{L}.mixer.x_proj.weight   -> blk.attention.x_proj.weight
        {prefix}.{L}.mixer.dt_proj.weight  -> blk.attention.dt_proj.weight
        {prefix}.{L}.mixer.dt_proj.bias    -> blk.attention.dt_proj.bias
        {prefix}.{L}.mixer.A_log           -> blk.attention.A_log
        {prefix}.{L}.mixer.D               -> blk.attention.D
        {prefix}.{L}.mixer.out_proj.weight -> blk.attention.out_proj.weight

    Source: modeling_mamba.py:58-120 (MambaMixer init) + 400-424 (MambaBlock).
    """
    L = layer_idx
    p = f"{prefix}.{L}"
    mp = f"{p}.mixer"

    pairs: list[tuple[str, torch.Tensor]] = [
        (f"{p}.norm.weight",         blk.pre_attn_norm.weight),
        (f"{mp}.in_proj.weight",     blk.attention.in_proj.weight),
        (f"{mp}.conv1d.weight",      blk.attention.conv1d.weight),
        (f"{mp}.x_proj.weight",      blk.attention.x_proj.weight),
        (f"{mp}.dt_proj.weight",     blk.attention.dt_proj.weight),
        (f"{mp}.dt_proj.bias",       blk.attention.dt_proj.bias),
        (f"{mp}.A_log",              blk.attention.A_log),
        (f"{mp}.D",                  blk.attention.D),
        (f"{mp}.out_proj.weight",    blk.attention.out_proj.weight),
    ]
    if blk.attention.in_proj.bias is not None:
        pairs.append((f"{mp}.in_proj.bias", blk.attention.in_proj.bias))
    if blk.attention.conv1d.bias is not None:
        pairs.append((f"{mp}.conv1d.bias", blk.attention.conv1d.bias))
    if blk.attention.out_proj.bias is not None:
        pairs.append((f"{mp}.out_proj.bias", blk.attention.out_proj.bias))

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
