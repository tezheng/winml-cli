"""Granite-4 H layer shape tests (no HF download)."""
import pytest
import torch

from api import kvcache, specs, types
from models.granite4_h import config, layer


def _small_cfg():
    return config.Granite4HConfig(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=256, shared_intermediate_size=256,
        num_hidden_layers=4, rms_norm_eps=1e-5, vocab_size=100,
        max_position_embeddings=128, tie_word_embeddings=False,
        dtype=torch.float32,
        embedding_multiplier=12.0, logits_scaling=8.0,
        residual_multiplier=0.22, attention_multiplier=0.25,
        layer_types=("mamba", "mamba", "attention", "mamba"),
        mamba_n_heads=4, mamba_n_groups=2, mamba_d_state=16,
        mamba_d_head=64, mamba_d_conv=4, mamba_expand=2,
        mamba_chunk_size=8,
        mamba_conv_bias=True, mamba_proj_bias=False,
        time_step_min=0.001, time_step_max=0.1, time_step_floor=1e-4,
        time_step_limit=(0.0, float("inf")),
        attention_bias=False,
        position_embedding_type="nope", rope_theta=10000.0,
        num_local_experts=0,
    )


def test_granite4_h_mamba_layer_forward_shape():
    cfg = _small_cfg()
    blk = layer.build_granite4_h_decoder_layer(cfg, layer_idx=0)
    blk.eval()
    B, S = 1, 10
    x = torch.randn(B, S, cfg.hidden_size)
    mixer = blk.attention
    cache = kvcache.SSMStateCache(
        batch_size=B, conv_dim=mixer.conv_dim,
        conv_kernel=mixer.conv_kernel,
        n_heads=mixer.num_heads, head_dim=mixer.head_dim,
        d_state=mixer.d_state,
    )
    with torch.no_grad():
        y = blk(x, cache=cache)
    assert y.shape == (B, S, cfg.hidden_size)
    # FFN sublayer ran — residual added twice with scale.


def test_granite4_h_attention_layer_forward_shape():
    cfg = _small_cfg()
    # layer_idx=2 is "attention" per _small_cfg layer_types.
    blk = layer.build_granite4_h_decoder_layer(cfg, layer_idx=2)
    blk.eval()
    B, S = 1, 8
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
    assert kv_cache.seq_len == S


def test_granite4_h_per_layer_dispatch():
    cfg = _small_cfg()
    # 0 mamba, 1 mamba, 2 attention, 3 mamba.
    blk_mamba = layer.build_granite4_h_decoder_layer(cfg, layer_idx=0)
    blk_attn = layer.build_granite4_h_decoder_layer(cfg, layer_idx=2)
    assert blk_mamba._is_ssm_block is True
    assert blk_attn._is_ssm_block is False
    # Both have the shared_mlp.
    assert blk_mamba.feedforward is not None
    assert blk_attn.feedforward is not None
