"""MiniMax-M2 layer-shape tests."""
import torch

from api import feedforward, kvcache, specs, types
from models.minimax_m2 import config as _c, layer as _l


def _small_cfg(**overrides):
    base = dict(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        rope_theta=5_000_000.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        dtype=torch.float32,
    )
    base.update(overrides)
    return _c.MiniMaxM2Config(**base)


def test_layer_forward_shape():
    cfg = _small_cfg()
    blk = _l.build_minimax_m2_decoder_layer(cfg, layer_idx=0)
    blk.eval()
    assert isinstance(blk.feedforward, feedforward.MoE)
    B, S = 1, 6
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )
    kv_cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=cfg.max_position_embeddings,
    )
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    with torch.no_grad():
        y = blk(x, position_ids=pos, cache=kv_cache, start_pos=0)
    assert y.shape == (B, S, cfg.hidden_size)
    assert torch.isfinite(y).all()


def test_qk_norm_full_hdh_shapes():
    cfg = _small_cfg()
    blk = _l.build_minimax_m2_decoder_layer(cfg, layer_idx=0)
    # FULL_HDH q_norm: weight sized num_heads * head_dim
    expected_q = cfg.num_attention_heads * cfg.head_dim
    expected_k = cfg.num_key_value_heads * cfg.head_dim
    assert blk.attention.q_norm.weight.shape == (expected_q,)
    assert blk.attention.k_norm.weight.shape == (expected_k,)
