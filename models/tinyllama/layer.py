"""TinyLlama 1.1B decoder-layer factory + HF weight loader.

TinyLlama is architecturally a Llama (vanilla RoPE, GQA, SwiGLU, no biases,
RMSNorm STANDARD_W), so the factory delegates to
`models.llama3.layer.build_llama3_decoder_layer` and the loader delegates to
`models.llama3.layer.load_hf_llama3_layer`.
"""
from __future__ import annotations

from api import block
from models.llama3 import layer as _llama3_layer
from models.tinyllama import config as _config


def build_tinyllama_decoder_layer(
    cfg: _config.TinyLlamaConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    return _llama3_layer.build_llama3_decoder_layer(
        cfg.inner, layer_idx=layer_idx, max_seq=max_seq,
    )


def load_hf_tinyllama_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
) -> None:
    """Use the Llama 3 loader — TinyLlama state-dict uses identical tensor names."""
    _llama3_layer.load_hf_llama3_layer(blk, hf_state_dict, layer_idx=layer_idx)
