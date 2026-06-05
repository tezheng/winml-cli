import pytest
import torch

from api import kvcache, specs, types
from models.qwen3 import config, layer


def _config_0p6b():
    return config.Qwen3Config(
        hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, intermediate_size=3072, num_hidden_layers=28,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=151936, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
    )


def _config_1p7b():
    return config.Qwen3Config(
        hidden_size=2048, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, intermediate_size=6144, num_hidden_layers=28,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=151936, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
    )


def _config_4b():
    return config.Qwen3Config(
        hidden_size=2560, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=9728, num_hidden_layers=36,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=151936, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
    )


@pytest.mark.parametrize("cfg_fn", [_config_0p6b, _config_1p7b, _config_4b])
def test_qwen3_decoder_layer_forward_shape(cfg_fn):
    cfg = cfg_fn()
    blk = layer.build_qwen3_decoder_layer(cfg, layer_idx=0, max_seq=128)
    B, S = 1, 8
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=128,
    )
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)
    assert cache.seq_len == S


def test_qwen3_decoder_layer_determinism():
    cfg = _config_0p6b()
    blk = layer.build_qwen3_decoder_layer(cfg, layer_idx=0, max_seq=64)
    torch.manual_seed(0)
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4)

    def run():
        cache_spec = specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=cfg.dtype, v_dtype=cfg.dtype,
        )
        cache = kvcache.ContiguousKVCache(
            cache_spec, batch_size=1,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, max_seq=64,
        )
        return blk(x, position_ids=pos, cache=cache, start_pos=0)

    a = run()
    b = run()
    assert torch.allclose(a, b, atol=0)
