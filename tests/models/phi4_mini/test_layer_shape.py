import torch

from api import kvcache, specs, types
from models.phi4_mini import config, layer


def _phi4_mini():
    """microsoft/Phi-4-mini-instruct sized config."""
    return config.Phi4MiniConfig(
        hidden_size=3072, num_attention_heads=24, num_key_value_heads=8,
        head_dim=128, intermediate_size=8192, num_hidden_layers=32,
        rope_theta=10000.0, rms_norm_eps=1e-5,
        vocab_size=200064, max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        tie_word_embeddings=True, dtype=torch.float32,
        sliding_window=None, partial_rotary_factor=0.75,
        rope_type="longrope",
        longrope_short_factor=tuple([1.0] * 48),
        longrope_long_factor=tuple([2.0] * 48),
    )


def test_phi4_mini_decoder_layer_forward_shape():
    cfg = _phi4_mini()
    blk = layer.build_phi4_mini_decoder_layer(cfg, layer_idx=0, max_seq=64)
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


def test_phi4_mini_gqa_dims():
    """24 Q / 8 KV → group ratio 3. qkv_proj output = (24 + 16) * 128 = 5120."""
    cfg = _phi4_mini()
    blk = layer.build_phi4_mini_decoder_layer(cfg, layer_idx=0, max_seq=64)
    expected = (24 + 2 * 8) * 128
    assert blk.attention.qkv_proj.weight.shape == (expected, cfg.hidden_size)


def test_phi4_mini_partial_rotary_factor_built():
    """The RoPE module must have rope_angles = int(0.75 * 128 // 2) = 48."""
    cfg = _phi4_mini()
    blk = layer.build_phi4_mini_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.rope.rope_angles == 48


def test_phi4_mini_longrope_table_present():
    """LongRoPE built two tables; the LONG one differs from SHORT."""
    cfg = _phi4_mini()
    blk = layer.build_phi4_mini_decoder_layer(cfg, layer_idx=0, max_seq=64)
    r = blk.attention.rope
    assert hasattr(r, "cos_cached_long")
    assert hasattr(r, "sin_cached_long")
    assert not torch.allclose(r.cos_cached, r.cos_cached_long)
