"""Granite 4 H decoder-layer factory.

Per-layer dispatch on `Granite4HConfig.layer_types[layer_idx]`:
- "mamba"      -> SSDSpec token mixer + shared_mlp channel mixer
- "attention"  -> AttentionSpec token mixer + shared_mlp channel mixer
"""
from __future__ import annotations

import torch

from api import block
from models.granite4_h import config as _config


def build_granite4_h_decoder_layer(
    cfg: _config.Granite4HConfig,
    layer_idx: int,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Granite 4 H decoder block at layer_idx."""
    spec = cfg.to_block_spec(layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_granite4_h_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    cfg: _config.Granite4HConfig,
    layer_idx: int,
) -> None:
    """Copy HF GraniteMoeHybrid layer weights into the api block.

    HF naming (verified vs GraniteMoeHybridModel.state_dict()):
    Common to all layers:
        model.layers.{L}.input_layernorm.weight             -> blk.pre_attn_norm.weight
        model.layers.{L}.post_attention_layernorm.weight    -> blk.pre_ffn_norm.weight
        model.layers.{L}.shared_mlp.input_linear.weight     -> blk.feedforward.gate_up_proj.weight
        model.layers.{L}.shared_mlp.output_linear.weight    -> blk.feedforward.down_proj.weight

    Mamba layers (`layer_types[L] == "mamba"`):
        model.layers.{L}.mamba.in_proj.weight        -> blk.attention.in_proj.weight
        model.layers.{L}.mamba.conv1d.weight         -> blk.attention.conv1d.weight
        model.layers.{L}.mamba.conv1d.bias?          -> blk.attention.conv1d.bias
        model.layers.{L}.mamba.dt_bias               -> blk.attention.dt_bias
        model.layers.{L}.mamba.A_log                 -> blk.attention.A_log
        model.layers.{L}.mamba.D                     -> blk.attention.D
        model.layers.{L}.mamba.norm.weight           -> blk.attention.norm.weight
        model.layers.{L}.mamba.out_proj.weight       -> blk.attention.out_proj.weight

    Attention layers (`layer_types[L] == "attention"`):
        model.layers.{L}.self_attn.{q,k,v,o}_proj.weight -> blk.attention.{q,k,v,o}_proj.weight
    """
    L = layer_idx
    prefix = f"model.layers.{L}"

    pairs: list[tuple[str, torch.Tensor]] = [
        (f"{prefix}.input_layernorm.weight",          blk.pre_attn_norm.weight),
        (f"{prefix}.post_attention_layernorm.weight", blk.pre_ffn_norm.weight),
        (f"{prefix}.shared_mlp.input_linear.weight",  blk.feedforward.gate_up_proj.weight),
        (f"{prefix}.shared_mlp.output_linear.weight", blk.feedforward.down_proj.weight),
    ]

    if cfg.is_mamba_layer(L):
        mp = f"{prefix}.mamba"
        pairs.extend([
            (f"{mp}.in_proj.weight",   blk.attention.in_proj.weight),
            (f"{mp}.conv1d.weight",    blk.attention.conv1d.weight),
            (f"{mp}.dt_bias",          blk.attention.dt_bias),
            (f"{mp}.A_log",            blk.attention.A_log),
            (f"{mp}.D",                blk.attention.D),
            (f"{mp}.norm.weight",      blk.attention.norm.weight),
            (f"{mp}.out_proj.weight",  blk.attention.out_proj.weight),
        ])
        if blk.attention.conv1d.bias is not None:
            pairs.append((f"{mp}.conv1d.bias", blk.attention.conv1d.bias))
        if blk.attention.in_proj.bias is not None:
            pairs.append((f"{mp}.in_proj.bias", blk.attention.in_proj.bias))
        if blk.attention.out_proj.bias is not None:
            pairs.append((f"{mp}.out_proj.bias", blk.attention.out_proj.bias))
    else:
        sp = f"{prefix}.self_attn"
        pairs.extend([
            (f"{sp}.q_proj.weight", blk.attention.q_proj.weight),
            (f"{sp}.k_proj.weight", blk.attention.k_proj.weight),
            (f"{sp}.v_proj.weight", blk.attention.v_proj.weight),
            (f"{sp}.o_proj.weight", blk.attention.o_proj.weight),
        ])
        if blk.attention.q_proj.bias is not None:
            pairs.append((f"{sp}.q_proj.bias", blk.attention.q_proj.bias))
            pairs.append((f"{sp}.k_proj.bias", blk.attention.k_proj.bias))
            pairs.append((f"{sp}.v_proj.bias", blk.attention.v_proj.bias))
            pairs.append((f"{sp}.o_proj.bias", blk.attention.o_proj.bias))

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
