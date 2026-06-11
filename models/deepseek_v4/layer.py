"""DeepSeek-V4 decoder-layer factory.

Wraps `api.block.DecoderBlock` with the V4-specific per-layer dispatch carried
on the config. The V4 attention is shape-only (Phase A P2 raises
NotImplementedError in forward); the V4 hash MoE channel mixer has a full
numerical forward (Phase A P1).

The HF V4 weight loader maps the V4 hash-MoE state dict into the api MoE
buffers. Names verified at modeling_deepseek_v4.py:1081-1098 +
DeepseekV4HashRouter init (:1059-1067) + DeepseekV4Experts init (:992-1000)
+ DeepseekV4MLP shared experts (:970-985).
"""
from __future__ import annotations
from typing import Optional

import torch

from api import block
from models.deepseek_v4 import config as _config


def build_deepseek_v4_decoder_layer(
    cfg: _config.DeepSeekV4Config,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else min(cfg.max_position_embeddings, 4096)
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_deepseek_v4_hash_moe(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    cfg: _config.DeepSeekV4Config,
    layer_idx: int,
) -> None:
    """Copy HF DeepseekV4SparseMoeBlock weights into the api MoE module.

    Only the channel-mixer (MoE) side is loaded — the V4 attention is
    shape-only and its forward raises NotImplementedError. For numerical
    parity tests, instantiate `DeepseekV4SparseMoeBlock` directly and copy
    weights using this helper's mapping.

    HF naming (verified vs DeepseekV4ForCausalLM.state_dict() with `layers.{L}.`
    prefix, modeling_deepseek_v4.py:1081-1098):

    Hash MoE (mlp_layer_types[L] == "hash_moe"):
        model.layers.{L}.mlp.gate.weight                 -> blk.feedforward.gate.weight
        model.layers.{L}.mlp.gate.tid2eid                -> blk.feedforward.gate.tid2eid
        model.layers.{L}.mlp.experts.gate_up_proj        -> blk.feedforward.experts_gate_up
        model.layers.{L}.mlp.experts.down_proj           -> blk.feedforward.experts_down
        model.layers.{L}.mlp.shared_experts.gate_proj.weight  -> blk.feedforward.shared_experts.gate_proj.weight
        model.layers.{L}.mlp.shared_experts.up_proj.weight    -> blk.feedforward.shared_experts.up_proj.weight
        model.layers.{L}.mlp.shared_experts.down_proj.weight  -> blk.feedforward.shared_experts.down_proj.weight

    The V4 TopKRouter ("moe" layer) additionally carries
    `gate.e_score_correction_bias`. We don't load that here; the
    sigmoid_plus_bias path uses the same name on api's `_SigmoidRouter`.
    """
    L = layer_idx
    prefix = f"model.layers.{L}.mlp"
    moe = blk.feedforward

    pairs: list[tuple[str, torch.Tensor]] = []

    if cfg.is_hash_layer(L):
        pairs.extend([
            (f"{prefix}.gate.weight",                moe.gate.weight),
            (f"{prefix}.gate.tid2eid",               moe.gate.tid2eid),
        ])
    else:
        # sigmoid_plus_bias router — gate.weight + gate.e_score_correction_bias.
        pairs.extend([
            (f"{prefix}.gate.weight",                moe.gate.weight),
            (f"{prefix}.gate.e_score_correction_bias", moe.gate.e_score_correction_bias),
        ])

    pairs.extend([
        (f"{prefix}.experts.gate_up_proj",       moe.experts_gate_up),
        (f"{prefix}.experts.down_proj",          moe.experts_down),
    ])
    if moe.shared_experts is not None:
        pairs.extend([
            (f"{prefix}.shared_experts.gate_proj.weight",
             moe.shared_experts.gate_proj.weight),
            (f"{prefix}.shared_experts.up_proj.weight",
             moe.shared_experts.up_proj.weight),
            (f"{prefix}.shared_experts.down_proj.weight",
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
