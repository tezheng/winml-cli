"""GOT-OCR 2.0 LM-decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a GotOcr2Config using only api/ primitives. Does
NOT import transformers.models.got_ocr2.modeling_got_ocr2.

Architectural parity vs Qwen2 LM-decoder: identical (PRE-norm RMSNorm
STANDARD_W, SPLIT QKV with biases, SwiGLU FFN without biases, SPLIT_HALF
RoPE). The B8 wrapper exists so the vision-fusion entry point has its own
factory + the HF weight loader knows where to read the LM weights from
the nested ``model.language_model.layers.{L}.*`` prefix (vs Qwen2's bare
``model.layers.{L}.*``).

Source citations:
- transformers/models/qwen2/modeling_qwen2.py:186-309
  (Qwen2Attention init/forward, Qwen2DecoderLayer).
- transformers/models/got_ocr2/modeling_got_ocr2.py:531-617
  (GotOcr2Model — wraps `vision_tower`, `multi_modal_projector`, and
  `language_model` which is Qwen2Model). The "language_model" attribute
  prefix is the only HF-side rename vs vanilla Qwen2.
"""
from __future__ import annotations

import torch

from api import block
from models.got_ocr2 import config as _config


def build_got_ocr2_decoder_layer(
    cfg: _config.GotOcr2Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one GOT-OCR 2.0 LM-decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_got_ocr2_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    prefix: str = "model.language_model.layers",
) -> None:
    """Copy HF GOT-OCR 2.0 LM-decoder layer weights into the api DecoderBlock.

    HF model surface: ``GotOcr2Model.language_model`` is a ``Qwen2Model``,
    so the layer tensors live under ``model.language_model.layers.{L}.*``
    with the same intra-layer names as vanilla Qwen2 (input_layernorm,
    self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj,
    post_attention_layernorm, mlp.gate_proj, mlp.up_proj, mlp.down_proj).

    For the standalone-Qwen2 path (running tests against a bare Qwen2-0.5B
    checkpoint with `model.layers.*`), pass ``prefix="model.layers"``.
    """
    L = layer_idx
    pfx = f"{prefix}.{L}"
    mapping = {
        f"{pfx}.input_layernorm.weight":             blk.pre_attn_norm.weight,
        f"{pfx}.self_attn.q_proj.weight":            blk.attention.q_proj.weight,
        f"{pfx}.self_attn.q_proj.bias":              blk.attention.q_proj.bias,
        f"{pfx}.self_attn.k_proj.weight":            blk.attention.k_proj.weight,
        f"{pfx}.self_attn.k_proj.bias":              blk.attention.k_proj.bias,
        f"{pfx}.self_attn.v_proj.weight":            blk.attention.v_proj.weight,
        f"{pfx}.self_attn.v_proj.bias":              blk.attention.v_proj.bias,
        f"{pfx}.self_attn.o_proj.weight":            blk.attention.o_proj.weight,
        f"{pfx}.post_attention_layernorm.weight":    blk.pre_ffn_norm.weight,
        f"{pfx}.mlp.gate_proj.weight":               blk.feedforward.gate_proj.weight,
        f"{pfx}.mlp.up_proj.weight":                 blk.feedforward.up_proj.weight,
        f"{pfx}.mlp.down_proj.weight":               blk.feedforward.down_proj.weight,
    }

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
