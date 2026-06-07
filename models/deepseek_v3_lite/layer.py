"""DeepSeek-V3-Lite decoder-layer factory.

Source: `transformers/models/deepseek_v3/modeling_deepseek_v3.py:482-526`
(DeepseekV3DecoderLayer init + forward — same envelope as V2).
"""
from __future__ import annotations
from typing import Optional

from api import block
from models.deepseek_v3_lite import config as _config


def build_deepseek_v3_lite_decoder_layer(
    cfg: _config.DeepSeekV3LiteConfig,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )
