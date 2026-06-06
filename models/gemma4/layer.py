"""Gemma 4 reference layer assembly.

Layer index dispatch:
- Local layers: SWA, head_dim=local, rope theta=10000, partial_rotary=1.0.
- Global layers: full causal, head_dim=global, rope theta=1_000_000,
  partial_rotary=0.25, attention_k_eq_v=True on 12B+ (no v_proj allocated).
- Shared K/V layers (E2B/E4B last 20/35 layers): the source attention block's
  ContiguousKVCache is aliased via SharedLayerKVCache and passed in by the
  model-assembly layer (out of scope for B0.5); the layer factory itself is
  unaware of sharing - the cache instance dispatches.

Weight-loader contract: at load time we absorb the Gemma 4 QK fixed-scale and
1/sqrt(Dh) into the ONE_PLUS_W norm weight so that the runtime attention path
keeps effective_scale=1.0 (no per-step rescale needed). This matches
api.attention.Attention's contract when AttentionSpec.qk_norm_fixed_scale is set.
"""
from __future__ import annotations
from typing import Optional

import torch

from api import block as _block
from models.gemma4 import config as gemma4_config


def build_gemma4_decoder_layer(
    cfg: gemma4_config.Gemma4Config,
    layer_idx: int,
    max_seq: Optional[int] = None,
) -> _block.DecoderBlock:
    """Build a Gemma 4 decoder block for the given layer index.

    The returned DecoderBlock can be called with an optional `per_layer_residual`
    argument (from a separate PerLayerEmbedding module owned at the model level).
    Shared-K/V layers should be invoked with a SharedLayerKVCache instead of a
    fresh ContiguousKVCache; this is wired at the model assembly level, not here.
    """
    block_spec = cfg.to_block_spec(layer_idx=layer_idx)
    max_seq_eff = max_seq if max_seq is not None else cfg.max_position_embeddings
    return _block.DecoderBlock(
        block_spec, hidden_size=cfg.hidden_size, max_seq=max_seq_eff,
        dtype=cfg.dtype,
    )


def load_hf_gemma4_layer(
    blk: _block.DecoderBlock,
    hf_state_dict: dict,
    layer_idx: int,
    cfg: gemma4_config.Gemma4Config,
) -> None:
    """Map an HF Gemma 4 state dict for a single layer into our DecoderBlock.

    HF tensor naming (from transformers.models.gemma4.modeling_gemma4):
      model.layers.{i}.input_layernorm.weight
      model.layers.{i}.self_attn.q_proj.weight
      model.layers.{i}.self_attn.k_proj.weight
      model.layers.{i}.self_attn.v_proj.weight        (absent on 12B+ global w/ K=V)
      model.layers.{i}.self_attn.o_proj.weight
      model.layers.{i}.self_attn.q_norm.weight
      model.layers.{i}.self_attn.k_norm.weight
      model.layers.{i}.post_attention_layernorm.weight
      model.layers.{i}.pre_feedforward_layernorm.weight
      model.layers.{i}.post_feedforward_layernorm.weight
      model.layers.{i}.mlp.gate_proj.weight
      model.layers.{i}.mlp.up_proj.weight
      model.layers.{i}.mlp.down_proj.weight

    QK-norm fixed-scale absorption (Gemma 4 specific):
    - HF Gemma 4 QKNorm uses RMSNorm with ONE_PLUS_W weight mode, with a
      multiplicative fixed scale (~0.99 local / ~1.02 global) applied OUTSIDE
      the norm, AND attention then divides by sqrt(Dh).
    - To collapse all this into a single weight gain at load time we set
      effective_gain = (1 + w_on_disk) * fixed_scale * sqrt(Dh)
      and write back w_new = effective_gain - 1 (ONE_PLUS_W mode).
    - The runtime attention path then sets effective_scale = 1.0 (no /sqrt(Dh))
      and skips the fixed-scale multiplication.
    """
    prefix = f"model.layers.{layer_idx}."

    is_global = cfg.is_global_layer(layer_idx)
    head_dim_eff = cfg.global_head_dim if is_global else cfg.head_dim
    fixed_scale = (cfg.qk_norm_global_fixed_scale if is_global
                   else cfg.qk_norm_local_fixed_scale)

    # absorb = fixed_scale / (1/sqrt(Dh)) = fixed_scale * sqrt(Dh)
    absorb = fixed_scale * (head_dim_eff ** 0.5)

    # Build the explicit HF-name -> tensor slot mapping (excluding QK-norm,
    # which gets the absorb transform).
    mapping = {
        prefix + "input_layernorm.weight":             blk.pre_attn_norm.weight,
        prefix + "post_attention_layernorm.weight":    blk.post_attn_sublayer_norm.weight,
        prefix + "pre_feedforward_layernorm.weight":   blk.pre_ffn_norm.weight,
        prefix + "post_feedforward_layernorm.weight":  blk.post_ffn_sublayer_norm.weight,
        prefix + "self_attn.q_proj.weight":            blk.attention.q_proj.weight,
        prefix + "self_attn.k_proj.weight":            blk.attention.k_proj.weight,
        prefix + "self_attn.o_proj.weight":            blk.attention.o_proj.weight,
        prefix + "mlp.gate_proj.weight":               blk.feedforward.gate_proj.weight,
        prefix + "mlp.up_proj.weight":                 blk.feedforward.up_proj.weight,
        prefix + "mlp.down_proj.weight":               blk.feedforward.down_proj.weight,
    }
    # v_proj is absent only when this layer aliases V := K (12B+ global only).
    if blk.attention.v_proj is not None:
        mapping[prefix + "self_attn.v_proj.weight"] = blk.attention.v_proj.weight

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")

    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))

        # QK norms: absorb fixed_scale * sqrt(Dh) into ONE_PLUS_W weight.
        # On disk Gemma stores `w` such that gain = (1 + w). We want effective
        # gain = (1 + w_on_disk) * absorb, so the stored w_new = (1+w)*absorb - 1.
        if blk.attention.q_norm is not None:
            q_key = prefix + "self_attn.q_norm.weight"
            k_key = prefix + "self_attn.k_norm.weight"
            if q_key not in hf_state_dict or k_key not in hf_state_dict:
                raise KeyError(f"missing QK-norm weights: {q_key} or {k_key}")
            learned_q = hf_state_dict[q_key].to(blk.attention.q_norm.weight.dtype)
            learned_k = hf_state_dict[k_key].to(blk.attention.k_norm.weight.dtype)
            blk.attention.q_norm.weight.copy_((learned_q + 1.0) * absorb - 1.0)
            blk.attention.k_norm.weight.copy_((learned_k + 1.0) * absorb - 1.0)
