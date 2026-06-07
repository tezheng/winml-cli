"""Granite 3.x decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a GraniteConfig using only api/ primitives. Does NOT
import transformers.models.granite.modeling_granite.

Architectural parity vs Llama 3:
- Identical Llama-shaped block (PRE-norm RMSNorm STANDARD_W, SPLIT QKV, GQA
  SwiGLU, no QK-norm).
- Differs ONLY in four μP scalars:
    * `attention_multiplier` overrides 1/sqrt(Dh) (api wires via attn_scale).
    * `residual_multiplier` scales sublayer outputs (api wires via
      DecoderBlockSpec.residual_scale, applied in block.forward).
    * `embedding_multiplier` and `logits_scaling` are model-level scalars; they
      live on the BlockSpec for assembly metadata but are NOT applied inside
      the decoder block (the input embed layer applies embedding_multiplier,
      the lm_head applies 1/logits_scaling).

Verified against:
- modeling_granite.py:115-179 (GraniteAttention; scale = attention_multiplier).
- modeling_granite.py:203-216 (GraniteMLP — SwiGLU with bias=mlp_bias).
- modeling_granite.py:219-280 (GraniteDecoderLayer — residual_multiplier on
  attn_out and ffn_out before add).
"""
from __future__ import annotations

from api import block
from models.granite import config as _config


def build_granite_decoder_layer(
    cfg: _config.GraniteConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Granite decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_granite_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Copy HF Granite layer weights into the api-assembled DecoderBlock.

    HF Granite uses the same per-layer tensor names as HF Llama (verified
    against modeling_granite.py:219-228 — input_layernorm /
    self_attn.{q,k,v,o}_proj / post_attention_layernorm / mlp.{gate,up,down}_proj).
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
    # Optional biases (Granite 3.1-2B-Base: False; loader respects spec).
    for hf_name, slot in (
        (f"{prefix}.self_attn.q_proj.bias", getattr(blk.attention.q_proj, "bias", None)),
        (f"{prefix}.self_attn.k_proj.bias", getattr(blk.attention.k_proj, "bias", None)),
        (f"{prefix}.self_attn.v_proj.bias", getattr(blk.attention.v_proj, "bias", None)),
        (f"{prefix}.self_attn.o_proj.bias", getattr(blk.attention.o_proj, "bias", None)),
        (f"{prefix}.mlp.gate_proj.bias", getattr(blk.feedforward.gate_proj, "bias", None)),
        (f"{prefix}.mlp.up_proj.bias", getattr(blk.feedforward.up_proj, "bias", None)),
        (f"{prefix}.mlp.down_proj.bias", getattr(blk.feedforward.down_proj, "bias", None)),
    ):
        if slot is not None and hf_name in hf_state_dict:
            mapping[hf_name] = slot

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
