"""DeepSeek-V3-MoE decoder-layer factory (production dims wrapper).

This is a thin wrap of the same MLA + MoE machinery already exercised in
B5 against ``transformers.models.deepseek_v3``; it merely advertises the
production V3 dims (hidden=7168, n_q=128, ...). The numerical gate at
synthetic dims lives in ``tests/models/deepseek_v3_lite/test_isolation_hf.py``.

Verified against ``transformers/models/deepseek_v3/modeling_deepseek_v3.py``:
- 482-526  DeepseekV3DecoderLayer init + forward.
"""
from __future__ import annotations
from typing import Optional

from api import block
from models.deepseek_v3_moe import config as _config


def build_deepseek_v3_moe_decoder_layer(
    cfg: _config.DeepSeekV3MoEConfig,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )
