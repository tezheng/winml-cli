"""Jamba decoder-layer factory (v6 B2)."""
from __future__ import annotations

import torch

from api import block
from models.jamba import config as _config


def build_jamba_decoder_layer(
    cfg: _config.JambaConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Jamba decoder block at layer_idx."""
    spec = cfg.to_block_spec(layer_idx)
    seq = max_seq if max_seq is not None else 4096
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )
