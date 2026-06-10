"""Qwen3-Next layer-shape tests."""
import torch

from api import kvcache, specs, types
from models.qwen3_next import config as _c, layer as _l


def _small_cfg(**overrides):
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attention_bias=False,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        dtype=torch.float32,
        layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention"),
        mlp_only_layers=(0, 1, 2, 3),       # dense MLP everywhere for shape tests
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=0,                       # force dense regardless
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
    )
    base.update(overrides)
    return _c.Qwen3NextConfig(**base)


def test_linear_attention_layer_forward_shape():
    cfg = _small_cfg()
    blk = _l.build_qwen3_next_decoder_layer(cfg, layer_idx=0)
    blk.eval()
    B, S = 2, 5
    x = torch.randn(B, S, cfg.hidden_size)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == (B, S, cfg.hidden_size)
    assert torch.isfinite(y).all()


def test_full_attention_layer_forward_shape():
    cfg = _small_cfg()
    blk = _l.build_qwen3_next_decoder_layer(cfg, layer_idx=3)
    blk.eval()
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


def test_dispatch_attribute_set():
    cfg = _small_cfg()
    blk_lin = _l.build_qwen3_next_decoder_layer(cfg, layer_idx=0)
    blk_full = _l.build_qwen3_next_decoder_layer(cfg, layer_idx=3)
    assert blk_lin._is_gated_deltanet is True
    assert blk_full._is_gated_deltanet is False
    assert blk_full._is_ssm_block is False
