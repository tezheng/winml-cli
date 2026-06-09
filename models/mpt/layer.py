"""MPT decoder-layer factory (v6 A1).

Builds a full MPT decoder block from MptConfig using api/ primitives:
    LayerNorm(no-bias) -> ALiBi-MHA -> residual
    LayerNorm(no-bias) -> ungated-GELU FFN -> residual

Source: `transformers/models/mpt/modeling_mpt.py:158-212` (MptBlock).
"""
from __future__ import annotations

import torch

from api import block
from models.mpt import config as _config


def build_mpt_decoder_layer(
    cfg: _config.MptConfig,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one MPT decoder block."""
    spec = cfg.to_block_spec()
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_mpt_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    prefix: str = "transformer.blocks",
) -> None:
    """Copy HF MPT layer weights into the api-assembled DecoderBlock.

    Mapping (HF tensor name -> API tensor slot):
        {prefix}.{L}.norm_1.weight        -> blk.pre_attn_norm.weight
        {prefix}.{L}.attn.Wqkv.weight     -> chunked into blk.attention.{q,k,v}_proj.weight
        {prefix}.{L}.attn.out_proj.weight -> blk.attention.o_proj.weight
        {prefix}.{L}.norm_2.weight        -> blk.pre_ffn_norm.weight
        {prefix}.{L}.ffn.up_proj.weight   -> blk.feedforward.up_proj.weight
        {prefix}.{L}.ffn.down_proj.weight -> blk.feedforward.down_proj.weight

    Source: modeling_mpt.py:158-178 (MptBlock module layout) and
    modeling_mpt.py:82-83 (Wqkv fused projection).
    """
    L = layer_idx
    p = f"{prefix}.{L}"
    H = blk.hidden_size
    Wqkv_key = f"{p}.attn.Wqkv.weight"
    if Wqkv_key not in hf_state_dict:
        raise KeyError(f"missing {Wqkv_key}")
    Wqkv = hf_state_dict[Wqkv_key]
    if Wqkv.shape != (3 * H, H):
        raise ValueError(
            f"unexpected Wqkv shape {tuple(Wqkv.shape)} (expected {(3 * H, H)})"
        )
    Wq, Wk, Wv = Wqkv.chunk(3, dim=0)

    rms_pairs = [
        (f"{p}.norm_1.weight",         blk.pre_attn_norm.weight),
        (f"{p}.norm_2.weight",         blk.pre_ffn_norm.weight),
        (f"{p}.attn.out_proj.weight",  blk.attention.o_proj.weight),
        (f"{p}.ffn.up_proj.weight",    blk.feedforward.up_proj.weight),
        (f"{p}.ffn.down_proj.weight",  blk.feedforward.down_proj.weight),
    ]
    missing = [k for k, _ in rms_pairs if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")
    with torch.no_grad():
        for hf_name, slot in rms_pairs:
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: {tuple(src.shape)} vs {tuple(slot.shape)}"
                )
            slot.copy_(src.to(slot.dtype))
        # Wqkv split
        blk.attention.q_proj.weight.copy_(Wq.to(blk.attention.q_proj.weight.dtype))
        blk.attention.k_proj.weight.copy_(Wk.to(blk.attention.k_proj.weight.dtype))
        blk.attention.v_proj.weight.copy_(Wv.to(blk.attention.v_proj.weight.dtype))
