"""B6: Qwen3-MoE decoder-layer shape & dispatch sanity tests."""
import pytest
import torch

from api import feedforward, kvcache, specs, types
from models.qwen3_moe import config as m_config, layer as m_layer


def _small_cfg(decoder_sparse_step=1, mlp_only_layers=()):
    return m_config.Qwen3MoeConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=decoder_sparse_step,
        mlp_only_layers=mlp_only_layers,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        dtype=torch.float32,
    )


def test_qwen3_moe_layer_forward_shape_moe():
    cfg = _small_cfg()
    blk = m_layer.build_qwen3_moe_decoder_layer(cfg, layer_idx=0, max_seq=32)
    blk.eval()
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=32,
    )
    B, S = 1, 5
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    with torch.no_grad():
        y = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert y.shape == (B, S, cfg.hidden_size)
    # Layer must have been built as MoE.
    assert isinstance(blk.feedforward, feedforward.MoE)


def test_qwen3_moe_layer_forward_shape_dense_via_mlp_only():
    """When mlp_only_layers contains layer_idx, the block uses a dense MLP
    with `intermediate_size`. Source: modeling_qwen3_moe.py:314-319."""
    cfg = _small_cfg(mlp_only_layers=(0,))
    blk = m_layer.build_qwen3_moe_decoder_layer(cfg, layer_idx=0, max_seq=32)
    blk.eval()
    # The dense path uses FeedForward, not MoE.
    assert isinstance(blk.feedforward, feedforward.FeedForward)
    assert blk.feedforward.spec.intermediate_size == cfg.intermediate_size
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=32,
    )
    B, S = 1, 3
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    with torch.no_grad():
        y = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert y.shape == (B, S, cfg.hidden_size)


def test_qwen3_moe_module_layout():
    """Verify MoE module exposes HF-compatible tensors.
    Source: modeling_qwen3_moe.py:215-272."""
    cfg = _small_cfg()
    blk = m_layer.build_qwen3_moe_decoder_layer(cfg, layer_idx=0, max_seq=32)
    moe = blk.feedforward
    assert isinstance(moe, feedforward.MoE)
    # Router gate.
    assert moe.gate.weight.shape == (cfg.num_experts, cfg.hidden_size)
    # Experts.
    assert moe.experts_gate_up.shape == (
        cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size,
    )
    assert moe.experts_down.shape == (
        cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size,
    )
    # No shared experts.
    assert moe.shared_experts is None


def test_qwen3_moe_layer_qk_norm_is_per_head_dh():
    """B6 invariant — Qwen3MoE uses PER_HEAD_DH QK-norm (not FULL_HDH).
    The weight shape must equal [head_dim]."""
    cfg = _small_cfg()
    blk = m_layer.build_qwen3_moe_decoder_layer(cfg, layer_idx=0, max_seq=32)
    qn = blk.attention.q_norm
    kn = blk.attention.k_norm
    assert qn is not None and kn is not None
    assert qn.weight.shape == (cfg.head_dim,)
    assert kn.weight.shape == (cfg.head_dim,)
