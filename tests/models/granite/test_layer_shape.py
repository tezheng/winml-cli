import torch

from api import kvcache, specs, types
from models.granite import config, layer


def _config_3_1_2b():
    """ibm-granite/granite-3.1-2b-base."""
    return config.GraniteConfig(
        hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
        head_dim=64, intermediate_size=8192, num_hidden_layers=40,
        rope_theta=5_000_000.0, rms_norm_eps=1e-5,
        vocab_size=49152, max_position_embeddings=131072,
        tie_word_embeddings=True, dtype=torch.float32,
        attention_multiplier=0.015625,
        residual_multiplier=0.22,
        embedding_multiplier=12.0,
        logits_scaling=8.0,
    )


def test_granite_decoder_layer_forward_shape():
    cfg = _config_3_1_2b()
    blk = layer.build_granite_decoder_layer(cfg, layer_idx=0, max_seq=64)
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
        head_dim=cfg.head_dim, max_seq=64,
    )
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_granite_decoder_layer_attn_scale_overridden():
    """The Attention block's effective_scale must be attention_multiplier,
    not 1/sqrt(head_dim). Source: modeling_granite.py:124."""
    cfg = _config_3_1_2b()
    blk = layer.build_granite_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.effective_scale == 0.015625
    # NOT 1/sqrt(64) = 0.125.
    assert blk.attention.effective_scale != cfg.head_dim ** -0.5


def test_granite_decoder_layer_residual_scale_wired():
    """The DecoderBlock's _residual_scale must equal residual_multiplier."""
    cfg = _config_3_1_2b()
    blk = layer.build_granite_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk._residual_scale == 0.22


def test_granite_decoder_layer_no_qk_norm():
    cfg = _config_3_1_2b()
    blk = layer.build_granite_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.q_norm is None
    assert blk.attention.k_norm is None
