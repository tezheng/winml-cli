"""GLM-MoE-DSA decoder-layer factory.

The MLA + DSA attention is SHAPE-ONLY (Phase A reservation, AttentionKind.DSA);
the MoE channel mixer is full numerical via the `sigmoid_plus_bias` router
+ group routing (V3-style).
"""
from __future__ import annotations
from typing import Optional

import torch

from api import block
from models.glm_moe_dsa import config as _config


def build_glm_moe_dsa_decoder_layer(
    cfg: _config.GlmMoeDsaConfig,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else min(cfg.max_position_embeddings, 4096)
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_glm_moe_dsa_moe(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    cfg: _config.GlmMoeDsaConfig,
    layer_idx: int,
) -> None:
    """Copy HF GlmMoeDsaMoE weights into the api MoE module.

    HF naming (modeling_glm_moe_dsa.py:538-591 + GlmMoeDsaTopkRouter:478-495):

    Block-level norms (also loaded):
        model.layers.{L}.input_layernorm.weight             -> blk.pre_attn_norm.weight
        model.layers.{L}.post_attention_layernorm.weight    -> blk.pre_ffn_norm.weight

    Sparse layer (mlp_layer_types[L] == "sparse"):
        model.layers.{L}.mlp.gate.weight                        -> blk.feedforward.gate.weight
        model.layers.{L}.mlp.gate.e_score_correction_bias       -> blk.feedforward.gate.e_score_correction_bias
        model.layers.{L}.mlp.experts.gate_up_proj               -> blk.feedforward.experts_gate_up
        model.layers.{L}.mlp.experts.down_proj                  -> blk.feedforward.experts_down
        model.layers.{L}.mlp.shared_experts.gate_proj.weight    -> blk.feedforward.shared_experts.gate_proj.weight
        model.layers.{L}.mlp.shared_experts.up_proj.weight      -> blk.feedforward.shared_experts.up_proj.weight
        model.layers.{L}.mlp.shared_experts.down_proj.weight    -> blk.feedforward.shared_experts.down_proj.weight

    Raises if `layer_idx` is dense.
    """
    if not cfg.is_sparse_layer(layer_idx):
        raise ValueError(
            f"layer {layer_idx} is dense; this loader only handles sparse layers"
        )
    L = layer_idx
    prefix = f"model.layers.{L}"
    moe = blk.feedforward

    pairs: list[tuple[str, torch.Tensor]] = [
        (f"{prefix}.input_layernorm.weight",            blk.pre_attn_norm.weight),
        (f"{prefix}.post_attention_layernorm.weight",   blk.pre_ffn_norm.weight),
        (f"{prefix}.mlp.gate.weight",                   moe.gate.weight),
        (f"{prefix}.mlp.gate.e_score_correction_bias",  moe.gate.e_score_correction_bias),
        (f"{prefix}.mlp.experts.gate_up_proj",          moe.experts_gate_up),
        (f"{prefix}.mlp.experts.down_proj",             moe.experts_down),
    ]
    if moe.shared_experts is not None:
        pairs.extend([
            (f"{prefix}.mlp.shared_experts.gate_proj.weight",
             moe.shared_experts.gate_proj.weight),
            (f"{prefix}.mlp.shared_experts.up_proj.weight",
             moe.shared_experts.up_proj.weight),
            (f"{prefix}.mlp.shared_experts.down_proj.weight",
             moe.shared_experts.down_proj.weight),
        ])

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
