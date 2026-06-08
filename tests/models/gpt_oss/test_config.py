"""GptOssConfig sanity + sink-attention shape tests."""
from __future__ import annotations

import torch

from api import attention as api_attention, types
from models.gpt_oss import config as g_config


def test_gpt_oss_20b_shape():
    cfg = g_config.GptOssConfig.gpt_oss_20b()
    assert cfg.hidden_size == 2880
    assert cfg.num_attention_heads == 64
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 64
    assert cfg.sliding_window == 128
    assert cfg.layer_types is not None
    assert len(cfg.layer_types) == cfg.num_hidden_layers
    # Layer 0 is sliding (i=0 → (0+1)%2 = 1 → sliding_attention).
    assert cfg.is_sliding_layer(0)
    # Layer 1 is full (i=1 → (1+1)%2 = 0).
    assert not cfg.is_sliding_layer(1)


def test_attention_spec_per_layer():
    cfg = g_config.GptOssConfig.gpt_oss_20b()
    # Sliding layer has sliding_window set, full layer doesn't.
    a0 = cfg.to_attention_spec(layer_idx=0)
    a1 = cfg.to_attention_spec(layer_idx=1)
    assert a0.sliding_window == 128
    assert a1.sliding_window is None
    # Both have SINK mask + n_sink_tokens=1
    for a in (a0, a1):
        assert a.mask_kind == types.MaskKind.SINK
        assert a.n_sink_tokens == 1
        assert a.q_bias is True
        assert a.k_bias is True


def test_gpt_oss_attention_module_has_sinks_and_runs():
    """Build a tiny GPT-OSS attention; verify sinks param + forward works."""
    cfg = g_config.GptOssConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=64, num_hidden_layers=2,
        sliding_window=8, max_position_embeddings=32,
        layer_types=("sliding_attention", "full_attention"),
    )
    spec = cfg.to_attention_spec(layer_idx=0)
    attn = api_attention.Attention(spec, hidden_size=64, max_seq=32,
                                   dtype=torch.float32)
    attn.eval()
    assert attn.sinks is not None
    assert attn.sinks.shape == (4,)
    # Forward smoke test
    from api import kvcache, specs as _specs
    cache = kvcache.ContiguousKVCache(
        _specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=1, n_kv_heads=2, head_dim=16, max_seq=32,
    )
    x = torch.randn(1, 5, 64)
    pos = torch.arange(5)
    with torch.no_grad():
        y = attn(x, position_ids=pos, cache=cache, start_pos=0)
    assert y.shape == (1, 5, 64)
