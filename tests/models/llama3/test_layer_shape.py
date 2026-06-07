import pytest
import torch

from api import kvcache, specs, types
from models.llama3 import config, layer


def _config_3p2_1b():
    """Llama 3.2 1B with rope_type='llama3', factor=32."""
    return config.Llama3Config(
        hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
        head_dim=64, intermediate_size=8192, num_hidden_layers=16,
        rope_theta=500000.0, rope_type="llama3",
        rope_factor=32.0, rope_low_freq_factor=1.0, rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
        rms_norm_eps=1e-5, vocab_size=128256,
        max_position_embeddings=131072, tie_word_embeddings=True,
        dtype=torch.float32,
    )


def _config_3p0_8b():
    """Llama 3.0 8B with rope_type='default'."""
    return config.Llama3Config(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=14336, num_hidden_layers=32,
        rope_theta=500000.0, rope_type="default",
        rms_norm_eps=1e-5, vocab_size=128256,
        max_position_embeddings=8192, tie_word_embeddings=False,
        dtype=torch.float32,
    )


def _config_3p1_8b():
    """Llama 3.1 8B with rope_type='llama3', factor=8."""
    return config.Llama3Config(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=14336, num_hidden_layers=32,
        rope_theta=500000.0, rope_type="llama3",
        rope_factor=8.0, rope_low_freq_factor=1.0, rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
        rms_norm_eps=1e-5, vocab_size=128256,
        max_position_embeddings=131072, tie_word_embeddings=False,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("cfg_fn", [_config_3p2_1b, _config_3p0_8b, _config_3p1_8b])
def test_llama3_decoder_layer_forward_shape(cfg_fn):
    cfg = cfg_fn()
    blk = layer.build_llama3_decoder_layer(cfg, layer_idx=0, max_seq=128)
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


def test_llama3_decoder_layer_no_qk_norm():
    """Llama 3 has NO QK-norm — verify the attention module reflects that."""
    cfg = _config_3p2_1b()
    blk = layer.build_llama3_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.q_norm is None
    assert blk.attention.k_norm is None


def test_llama3_decoder_layer_determinism():
    cfg = _config_3p2_1b()
    blk = layer.build_llama3_decoder_layer(cfg, layer_idx=0, max_seq=64)
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
