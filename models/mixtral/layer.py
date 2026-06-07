"""Mixtral decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a MixtralConfig using only api/ primitives.

Verified against `transformers/models/mixtral/modeling_mixtral.py`:
- 295-351  MixtralAttention init + forward (GQA, RoPE θ=1e6, no QK-norm).
- 354-389  MixtralDecoderLayer init + forward (PRE-norm + MoE block).
- 119-135  MixtralSparseMoeBlock (router gate + experts packed tensors).
- 101-116  MixtralTopKRouter (softmax + ALWAYS-on top-k renormalization).
- 62-98    MixtralExperts (gate_up_proj [E, 2I, H], down_proj [E, H, I]).
"""
from __future__ import annotations
from typing import Optional

import torch

from api import block
from models.mixtral import config as _config


def build_mixtral_decoder_layer(
    cfg: _config.MixtralConfig,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_mixtral_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Mixtral layer tensors into the api DecoderBlock.

    HF state-dict tensor names (verified at modeling_mixtral.py:354-389,
    119-135, 101-107, 62-72):

      model.layers.{L}.input_layernorm.weight
      model.layers.{L}.post_attention_layernorm.weight
      model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
      model.layers.{L}.block_sparse_moe.gate.weight
      model.layers.{L}.block_sparse_moe.experts.gate_up_proj
      model.layers.{L}.block_sparse_moe.experts.down_proj

    NB: the HF-5.x ``MixtralExperts`` keeps the experts packed as 3D
    parameters ``gate_up_proj [E, 2I, H]`` and ``down_proj [E, H, I]`` —
    identical layout to our ``MoE.experts_gate_up`` / ``experts_down``.
    The MLP submodule is named ``block_sparse_moe`` on the decoder layer
    (modeling_mixtral.py:361). Older HF Mixtral checkpoints used the
    ``mlp`` name and per-expert ``mlp.experts.{i}.{w1,w2,w3}.weight``
    layout; we try both lazily here for forward-compat against gated
    checkpoints converted in either schema.
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    attn = blk.attention
    moe = blk.feedforward

    # Mandatory norms + attention projections.
    mapping: dict[str, torch.Tensor] = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":          attn.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":          attn.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":          attn.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":          attn.o_proj.weight,
    }

    # MoE: support both naming schemes — "block_sparse_moe" (modular HF)
    # and "mlp" (some older converted checkpoints).
    moe_prefix = None
    for cand in (f"{prefix}.block_sparse_moe.gate.weight",
                 f"{prefix}.mlp.gate.weight"):
        if cand in hf_state_dict:
            moe_prefix = cand.rsplit(".gate.weight", 1)[0]
            break
    if moe_prefix is None:
        raise KeyError(
            f"could not find Mixtral MoE block prefix for layer {L}: "
            "tried block_sparse_moe.gate.weight and mlp.gate.weight"
        )

    mapping[f"{moe_prefix}.gate.weight"] = moe.gate.weight

    # Packed-experts (HF 5.x): one big 3D tensor.
    packed_gu = f"{moe_prefix}.experts.gate_up_proj"
    packed_d = f"{moe_prefix}.experts.down_proj"
    if packed_gu in hf_state_dict and packed_d in hf_state_dict:
        mapping[packed_gu] = moe.experts_gate_up
        mapping[packed_d] = moe.experts_down
    else:
        # Per-expert split layout (older HF Mixtral / community converts):
        #   experts.{i}.w1.weight  → gate (rows [0, I))   of experts_gate_up[i]
        #   experts.{i}.w3.weight  → up   (rows [I, 2I))  of experts_gate_up[i]
        #   experts.{i}.w2.weight  → down                  of experts_down[i]
        # We do not stage these into the `mapping` dict — they go through a
        # separate copy because the destination is a slice of a Parameter.
        E = moe.n_experts
        I = moe.intermediate_size
        H = moe.hidden_size
        with torch.no_grad():
            for e in range(E):
                w1 = hf_state_dict[f"{moe_prefix}.experts.{e}.w1.weight"]   # [I, H]
                w3 = hf_state_dict[f"{moe_prefix}.experts.{e}.w3.weight"]   # [I, H]
                w2 = hf_state_dict[f"{moe_prefix}.experts.{e}.w2.weight"]   # [H, I]
                if w1.shape != (I, H) or w3.shape != (I, H) or w2.shape != (H, I):
                    raise ValueError(
                        f"expert {e} weight shape mismatch: "
                        f"w1={tuple(w1.shape)}, w3={tuple(w3.shape)}, "
                        f"w2={tuple(w2.shape)} vs (I={I}, H={H})"
                    )
                # Pack: rows [0..I) ← w1 (gate), rows [I..2I) ← w3 (up).
                moe.experts_gate_up[e, :I, :].copy_(
                    w1.to(moe.experts_gate_up.dtype)
                )
                moe.experts_gate_up[e, I:, :].copy_(
                    w3.to(moe.experts_gate_up.dtype)
                )
                moe.experts_down[e].copy_(w2.to(moe.experts_down.dtype))

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing[:10]}")
    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
