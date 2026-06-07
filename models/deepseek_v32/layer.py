"""DeepSeek-V3.2 (DSA) decoder layer — shape-only factory.

Forward raises NotImplementedError. Use this package only for spec
composition exercises and tooling smoke-tests.
"""
from __future__ import annotations
from typing import Optional

from api import block
from models.deepseek_v32 import config as _config


def build_deepseek_v32_decoder_layer(
    cfg: _config.DeepSeekV32Config,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )
