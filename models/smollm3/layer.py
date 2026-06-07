"""SmolLM3 decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a Smollm3Config at a given `layer_idx`. The same
factory is used for both RoPE layers and NoPE layers — the per-layer dispatch
lives in `Smollm3Config.to_block_spec(layer_idx)`.

Verified against modeling_smollm3.py:185-258 (SmolLM3Attention with use_rope
gate), 282-295 (SmolLM3MLP), 298-338 (SmolLM3DecoderLayer.forward).
"""
from __future__ import annotations

from api import block
from models.smollm3 import config as _config


def build_smollm3_decoder_layer(
    cfg: _config.Smollm3Config,
    layer_idx: int,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate the SmolLM3 decoder block at `layer_idx`.

    Per-layer dispatch:
    - If `cfg.no_rope_layers[layer_idx] == 1`, the block carries a RoPE module.
    - If `cfg.no_rope_layers[layer_idx] == 0`, the block omits RoPE entirely
      (NoPE layer).
    """
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_smollm3_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF SmolLM3 layer weights into the api-assembled DecoderBlock.

    HF SmolLM3 uses the same tensor names as HF Llama 3 (verified against
    modeling_smollm3.py:298-307 — input_layernorm / self_attn.{q,k,v,o}_proj /
    post_attention_layernorm / mlp.{gate,up,down}_proj). No q_norm / k_norm.
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
