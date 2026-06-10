"""Qwen3-Next decoder-layer factory.

Per-layer dispatch on `Qwen3NextConfig.layer_types[layer_idx]`:
- "linear_attention" -> GatedDeltaNetSpec token mixer (numerical-equivalent
  to HF Qwen3NextGatedDeltaNet at the synthetic-mini config).
- "full_attention"   -> AttentionSpec STANDARD (shape-only — HF's q_proj is
  2× sized for (q | gate) and the attention output is gated; the IR does
  not model output-gating).
"""
from __future__ import annotations
from typing import Optional

import torch

from api import block
from models.qwen3_next import config as _config


def build_qwen3_next_decoder_layer(
    cfg: _config.Qwen3NextConfig,
    layer_idx: int,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else min(cfg.max_position_embeddings, 4096)
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_qwen3_next_linear_attn_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    cfg: _config.Qwen3NextConfig,
    layer_idx: int,
) -> None:
    """Copy HF Qwen3NextDecoderLayer (linear-attention variant) weights into the
    api block.

    HF naming for the linear-attention path (verified vs Qwen3NextModel state
    dict, modeling_qwen3_next.py:499-717, 819-839):

    Block-level norms:
        model.layers.{L}.input_layernorm.weight             -> blk.pre_attn_norm.weight
        model.layers.{L}.post_attention_layernorm.weight    -> blk.pre_ffn_norm.weight

    GatedDeltaNet token mixer (`self.linear_attn` on HF):
        model.layers.{L}.linear_attn.in_proj_qkvz.weight    -> blk.attention.in_proj_qkvz.weight
        model.layers.{L}.linear_attn.in_proj_ba.weight      -> blk.attention.in_proj_ba.weight
        model.layers.{L}.linear_attn.conv1d.weight          -> blk.attention.conv1d.weight
        model.layers.{L}.linear_attn.dt_bias                -> blk.attention.dt_bias
        model.layers.{L}.linear_attn.A_log                  -> blk.attention.A_log
        model.layers.{L}.linear_attn.norm.weight            -> blk.attention.norm.weight
        model.layers.{L}.linear_attn.out_proj.weight        -> blk.attention.out_proj.weight

    Channel mixer (dense MLP — not the MoE path; weight loader supports the
    `mlp_only` case for synthetic tests):
        model.layers.{L}.mlp.gate_proj.weight               -> blk.feedforward.gate_proj.weight
        model.layers.{L}.mlp.up_proj.weight                 -> blk.feedforward.up_proj.weight
        model.layers.{L}.mlp.down_proj.weight               -> blk.feedforward.down_proj.weight

    Raises if `layer_idx` is not a linear-attention layer.
    """
    if not cfg.is_linear_attention_layer(layer_idx):
        raise ValueError(
            f"layer {layer_idx} is full_attention; this loader handles linear_attention only"
        )

    L = layer_idx
    prefix = f"model.layers.{L}"
    mixer = blk.attention

    pairs: list[tuple[str, torch.Tensor]] = [
        (f"{prefix}.input_layernorm.weight",            blk.pre_attn_norm.weight),
        (f"{prefix}.post_attention_layernorm.weight",   blk.pre_ffn_norm.weight),
        # GatedDeltaNet mixer.
        (f"{prefix}.linear_attn.in_proj_qkvz.weight",   mixer.in_proj_qkvz.weight),
        (f"{prefix}.linear_attn.in_proj_ba.weight",     mixer.in_proj_ba.weight),
        (f"{prefix}.linear_attn.conv1d.weight",         mixer.conv1d.weight),
        (f"{prefix}.linear_attn.dt_bias",               mixer.dt_bias),
        (f"{prefix}.linear_attn.A_log",                 mixer.A_log),
        (f"{prefix}.linear_attn.norm.weight",           mixer.norm.weight),
        (f"{prefix}.linear_attn.out_proj.weight",       mixer.out_proj.weight),
    ]

    # Dense MLP path (mlp_only or num_experts == 0).
    if cfg.is_mlp_only_layer(layer_idx):
        pairs.extend([
            (f"{prefix}.mlp.gate_proj.weight",   blk.feedforward.gate_proj.weight),
            (f"{prefix}.mlp.up_proj.weight",     blk.feedforward.up_proj.weight),
            (f"{prefix}.mlp.down_proj.weight",   blk.feedforward.down_proj.weight),
        ])
    # MoE path is NOT loaded — the HF Qwen3NextSparseMoeBlock has a
    # `shared_expert_gate` (Linear hidden -> 1) which the api MoE does not
    # model. The synthetic-config numerical gate uses mlp_only_layers.

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
