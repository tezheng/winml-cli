"""MiniMax-M2 decoder-layer factory."""
from __future__ import annotations
from typing import Optional

import torch

from api import block
from models.minimax_m2 import config as _config


def build_minimax_m2_decoder_layer(
    cfg: _config.MiniMaxM2Config,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else min(cfg.max_position_embeddings, 4096)
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_minimax_m2_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    cfg: _config.MiniMaxM2Config,
    layer_idx: int,
) -> None:
    """Copy HF MiniMaxM2DecoderLayer weights into the api block.

    HF naming (verified vs MiniMaxM2Model state dict, modeling_minimax_m2.py
    :360-395 — DecoderLayer; :295-330 — Attention; :107-124 — SparseMoeBlock):

    Block-level norms:
        model.layers.{L}.input_layernorm.weight             -> blk.pre_attn_norm.weight
        model.layers.{L}.post_attention_layernorm.weight    -> blk.pre_ffn_norm.weight

    Attention (FULL_HDH q/k norms PRE-RoPE):
        model.layers.{L}.self_attn.q_proj.weight            -> blk.attention.q_proj.weight
        model.layers.{L}.self_attn.k_proj.weight            -> blk.attention.k_proj.weight
        model.layers.{L}.self_attn.v_proj.weight            -> blk.attention.v_proj.weight
        model.layers.{L}.self_attn.o_proj.weight            -> blk.attention.o_proj.weight
        model.layers.{L}.self_attn.q_norm.weight            -> blk.attention.q_norm.weight
        model.layers.{L}.self_attn.k_norm.weight            -> blk.attention.k_norm.weight

    MoE — IMPORTANT NAMING:
        model.layers.{L}.mlp.gate.weight                    -> blk.feedforward.gate.weight
        model.layers.{L}.mlp.e_score_correction_bias        -> blk.feedforward.gate.e_score_correction_bias
                  ^                                                            ^
                  on the MoE block (HF line 114), NOT on the gate. The api
                  `_SigmoidRouter` puts it under `gate.` — we re-key here.
        model.layers.{L}.mlp.experts.gate_up_proj           -> blk.feedforward.experts_gate_up
        model.layers.{L}.mlp.experts.down_proj              -> blk.feedforward.experts_down
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    attn = blk.attention
    moe = blk.feedforward

    pairs: list[tuple[str, torch.Tensor]] = [
        (f"{prefix}.input_layernorm.weight",            blk.pre_attn_norm.weight),
        (f"{prefix}.post_attention_layernorm.weight",   blk.pre_ffn_norm.weight),
        # Attention.
        (f"{prefix}.self_attn.q_proj.weight",           attn.q_proj.weight),
        (f"{prefix}.self_attn.k_proj.weight",           attn.k_proj.weight),
        (f"{prefix}.self_attn.v_proj.weight",           attn.v_proj.weight),
        (f"{prefix}.self_attn.o_proj.weight",           attn.o_proj.weight),
        (f"{prefix}.self_attn.q_norm.weight",           attn.q_norm.weight),
        (f"{prefix}.self_attn.k_norm.weight",           attn.k_norm.weight),
        # MoE.
        (f"{prefix}.mlp.gate.weight",                   moe.gate.weight),
        # NOTE re-keyed: HF stores on the block, we hang on gate.
        (f"{prefix}.mlp.e_score_correction_bias",       moe.gate.e_score_correction_bias),
        (f"{prefix}.mlp.experts.gate_up_proj",          moe.experts_gate_up),
        (f"{prefix}.mlp.experts.down_proj",             moe.experts_down),
    ]

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
