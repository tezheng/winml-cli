"""MptConfig sanity checks."""
from __future__ import annotations

import torch

from api import specs, types
from models.mpt import config as mpt_config


def test_mpt_7b_shape():
    cfg = mpt_config.MptConfig.mpt_7b()
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.head_dim == 128
    spec = cfg.to_attention_spec()
    assert spec.kind == types.AttentionKind.STANDARD
    assert spec.rope is None
    assert spec.alibi is not None
    assert spec.alibi.n_heads == 32
    assert spec.alibi.alibi_bias_max == 8.0


def test_mpt_attention_init_builds_slopes_buffer():
    """The Attention module must materialize the ALiBi slopes buffer at
    init when the spec carries alibi."""
    from api import attention as api_attention
    cfg = mpt_config.MptConfig(
        hidden_size=64, num_attention_heads=4, num_hidden_layers=1,
        max_position_embeddings=32,
    )
    spec = cfg.to_attention_spec()
    attn = api_attention.Attention(spec, hidden_size=64, max_seq=32,
                                   dtype=torch.float32)
    assert hasattr(attn, "alibi_slopes")
    assert attn.alibi_slopes is not None
    assert attn.alibi_slopes.shape == (4,)


def test_alibi_and_rope_mutually_exclusive():
    """Setting both alibi and rope on AttentionSpec must raise at init."""
    import pytest
    from api import attention as api_attention
    spec = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=16,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
        alibi=specs.AliBiSpec(n_heads=4),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        api_attention.Attention(spec, hidden_size=64, max_seq=32)
