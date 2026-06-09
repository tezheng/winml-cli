"""Falcon-7B decoder-layer factory (v6 A2).

Builds the full Falcon-7B decoder block from Falcon7BConfig using api/
primitives:

    shared = input_layernorm(x)
    y = x + attention(shared) + mlp(shared)

with LayerNorm-with-bias, MQA + RoPE attention, ungated GELU FFN.

Source: `transformers/models/falcon/modeling_falcon.py:554-636`
(FalconDecoderLayer + forward with parallel_attn=True path).
"""
from __future__ import annotations

import torch

from api import block
from models.falcon7b import config as _config


def build_falcon7b_decoder_layer(
    cfg: _config.Falcon7BConfig,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Falcon-7B decoder block."""
    spec = cfg.to_block_spec()
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_falcon7b_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    prefix: str = "transformer.h",
) -> None:
    """Copy HF Falcon-7B layer weights into the api-assembled DecoderBlock.

    For Falcon-7B (parallel_attn=True, new_decoder_architecture=False,
    num_ln_in_parallel_attn=1):

        {prefix}.{L}.input_layernorm.{weight,bias}
            -> blk.pre_attn_norm.{weight,bias}
        {prefix}.{L}.self_attention.query_key_value.weight (fused)
            -> sliced into blk.attention.{q,k,v}_proj.weight
        {prefix}.{L}.self_attention.dense.weight
            -> blk.attention.o_proj.weight
        {prefix}.{L}.mlp.dense_h_to_4h.weight
            -> blk.feedforward.up_proj.weight
        {prefix}.{L}.mlp.dense_4h_to_h.weight
            -> blk.feedforward.down_proj.weight

    The fused QKV layout for Falcon-7B's multi_query=True (from
    modeling_falcon.py:316-335 `_split_heads`):
        fused.view(B, S, num_heads + 2, head_dim)
        -> q = fused[..., :-2, :]  (rows 0..Hq*Dh of the weight)
        -> k = fused[..., [-2], :] (rows Hq*Dh..(Hq+1)*Dh)
        -> v = fused[..., [-1], :] (rows (Hq+1)*Dh..(Hq+2)*Dh)
    """
    L = layer_idx
    p = f"{prefix}.{L}"
    Hq = blk.spec.token_mixer.n_q_heads
    Dh = blk.spec.token_mixer.head_dim
    H = blk.hidden_size

    qkv_key = f"{p}.self_attention.query_key_value.weight"
    if qkv_key not in hf_state_dict:
        raise KeyError(f"missing {qkv_key}")
    W = hf_state_dict[qkv_key]
    expected = ((Hq + 2) * Dh, H)
    if W.shape != expected:
        raise ValueError(
            f"unexpected qkv shape {tuple(W.shape)} (expected {expected})"
        )
    Wq = W[: Hq * Dh, :]
    Wk = W[Hq * Dh : (Hq + 1) * Dh, :]
    Wv = W[(Hq + 1) * Dh :, :]

    weight_pairs = [
        (f"{p}.input_layernorm.weight",          blk.pre_attn_norm.weight),
        (f"{p}.input_layernorm.bias",            blk.pre_attn_norm.bias),
        (f"{p}.self_attention.dense.weight",     blk.attention.o_proj.weight),
        (f"{p}.mlp.dense_h_to_4h.weight",        blk.feedforward.up_proj.weight),
        (f"{p}.mlp.dense_4h_to_h.weight",        blk.feedforward.down_proj.weight),
    ]
    missing = [k for k, _ in weight_pairs if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")
    with torch.no_grad():
        for hf_name, slot in weight_pairs:
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: "
                    f"{tuple(src.shape)} vs {tuple(slot.shape)}"
                )
            slot.copy_(src.to(slot.dtype))
        blk.attention.q_proj.weight.copy_(Wq.to(blk.attention.q_proj.weight.dtype))
        blk.attention.k_proj.weight.copy_(Wk.to(blk.attention.k_proj.weight.dtype))
        blk.attention.v_proj.weight.copy_(Wv.to(blk.attention.v_proj.weight.dtype))
