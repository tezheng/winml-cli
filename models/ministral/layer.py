"""Ministral decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a MinistralConfig at a given `layer_idx`. Same
factory is used for both full and sliding layers — the per-layer dispatch
lives in `MinistralConfig.to_block_spec(layer_idx)` (mask + sliding_window
field on the AttentionSpec).

Verified against modeling_ministral.py:137-196 (MinistralAttention with
per-layer ``self.layer_type`` / ``self.sliding_window`` set), 50-63
(MinistralMLP), 220-260 (MinistralDecoderLayer).
"""
from __future__ import annotations

from api import block
from models.ministral import config as _config


def build_ministral_decoder_layer(
    cfg: _config.MinistralConfig,
    layer_idx: int,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate the Ministral decoder block at ``layer_idx``.

    Per-layer dispatch:
    - If ``cfg.layer_type(layer_idx) == "sliding_attention"`` AND
      ``cfg.sliding_window is not None``, the block uses SWA with the
      configured window.
    - Otherwise (full layer or sliding_window unset) the block uses CAUSAL.
    """
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_ministral_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Ministral layer weights into the api-assembled DecoderBlock.

    HF Ministral uses the same tensor names as HF Mistral / Llama 3 (verified
    at modeling_ministral.py:220-260 -- input_layernorm / self_attn.{q,k,v,o}_proj
    / post_attention_layernorm / mlp.{gate,up,down}_proj). No q_norm / k_norm
    and no biases. The weight names are identical for full and sliding layers
    -- the per-layer dispatch is purely in the module configuration, not in
    the state dict.

    The published ``mistralai/Ministral-8B-Instruct-2410`` checkpoint declares
    ``architectures: ["MistralForCausalLM"]`` and uses Mistral-shaped tensor
    names; this loader therefore works for both load paths.
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    mapping = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.self_attn.q_proj.weight":          blk.attention.q_proj.weight,
        f"{prefix}.self_attn.k_proj.weight":          blk.attention.k_proj.weight,
        f"{prefix}.self_attn.v_proj.weight":          blk.attention.v_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":          blk.attention.o_proj.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        f"{prefix}.mlp.gate_proj.weight":             blk.feedforward.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight":               blk.feedforward.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight":             blk.feedforward.down_proj.weight,
    }
    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")

    import torch as _torch
    with _torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
