"""OLMo 2 layer-shape tests — verify the built decoder block has the
expected POST-norm topology and runs a forward pass.
"""
import pytest
import torch

from api import kvcache, specs, types
from models.olmo2 import config, layer


def _config_1b():
    """OLMo-2-0425-1B shape."""
    return config.Olmo2Config(
        hidden_size=2048, num_attention_heads=16, num_key_value_heads=16,
        head_dim=128, intermediate_size=8192, num_hidden_layers=16,
        rope_theta=500000.0, rms_norm_eps=1e-6,
        vocab_size=100352, max_position_embeddings=4096,
        tie_word_embeddings=False, dtype=torch.float32,
        attention_bias=False,
    )


def _config_7b():
    """OLMo-2-1124-7B shape."""
    return config.Olmo2Config(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=32,
        head_dim=128, intermediate_size=11008, num_hidden_layers=32,
        rope_theta=500000.0, rms_norm_eps=1e-6,
        vocab_size=100352, max_position_embeddings=4096,
        tie_word_embeddings=False, dtype=torch.float32,
        attention_bias=False,
    )


@pytest.mark.parametrize("cfg_fn", [_config_1b, _config_7b])
def test_olmo2_decoder_layer_forward_shape(cfg_fn):
    cfg = cfg_fn()
    blk = layer.build_olmo2_decoder_layer(cfg, layer_idx=0, max_seq=64)
    B, S = 1, 6
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
        head_dim=cfg.head_dim, max_seq=64,
    )
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)
    assert cache.seq_len == S


def test_olmo2_decoder_layer_has_post_norm_topology():
    """B3: the built block must NOT have pre-norm modules and MUST have
    post-norm modules — POST-norm structural invariant.
    """
    cfg = _config_1b()
    blk = layer.build_olmo2_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.pre_attn_norm is None
    assert blk.pre_ffn_norm is None
    assert blk.post_attn_sublayer_norm is not None
    assert blk.post_ffn_sublayer_norm is not None


def test_olmo2_decoder_layer_qk_norm_full_hdh_shape():
    """B3: q_norm.weight shape == [Hq*Dh]; k_norm.weight shape == [Hk*Dh].
    Source: modeling_olmo2.py:231-232.
    """
    cfg = _config_1b()
    blk = layer.build_olmo2_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.q_norm is not None
    assert blk.attention.k_norm is not None
    assert blk.attention.q_norm.weight.shape == (cfg.num_attention_heads * cfg.head_dim,)
    assert blk.attention.k_norm.weight.shape == (cfg.num_key_value_heads * cfg.head_dim,)
