"""Falcon7BConfig sanity checks."""
from __future__ import annotations

from api import types
from models.falcon7b import config as falcon_config


def test_falcon7b_shape():
    cfg = falcon_config.Falcon7BConfig.falcon_7b()
    assert cfg.hidden_size == 4544
    assert cfg.num_attention_heads == 71
    assert cfg.num_kv_heads == 1                   # MQA
    assert cfg.head_dim == 64                      # 4544 / 71
    spec = cfg.to_block_spec()
    assert spec.block_layout == types.BlockLayout.PARALLEL
    assert spec.pre_ffn_norm is None
    assert spec.token_mixer.n_kv_heads == 1
    assert spec.token_mixer.rope is not None
    assert spec.token_mixer.alibi is None          # Falcon-7B uses RoPE, not ALiBi


def test_falcon7b_block_assembles():
    """The block must instantiate without raising."""
    from api import block as api_block
    cfg = falcon_config.Falcon7BConfig(
        hidden_size=64, num_attention_heads=4, num_kv_heads=1,
        num_hidden_layers=1, intermediate_size=128,
        max_position_embeddings=32,
    )
    blk = api_block.DecoderBlock(cfg.to_block_spec(), hidden_size=64,
                                 max_seq=32)
    blk.eval()
    assert blk.pre_attn_norm is not None
    assert blk.pre_ffn_norm is None                # PARALLEL has no pre_ffn_norm
