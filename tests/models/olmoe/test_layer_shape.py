"""B6: OLMoE decoder-layer shape & invariants tests."""
import pytest
import torch

from api import feedforward, kvcache, specs, types
from models.olmoe import config as m_config, layer as m_layer


def _small_cfg() -> m_config.OlmoeConfig:
    return m_config.OlmoeConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        rope_theta=10_000.0,
        rms_norm_eps=1e-5,
        vocab_size=100,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        dtype=torch.float32,
    )


def test_olmoe_layer_forward_shape():
    cfg = _small_cfg()
    blk = m_layer.build_olmoe_decoder_layer(cfg, layer_idx=0, max_seq=32)
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


def test_olmoe_moe_module_layout_no_shared_experts():
    cfg = _small_cfg()
    blk = m_layer.build_olmoe_decoder_layer(cfg, layer_idx=0, max_seq=32)
    moe = blk.feedforward
    assert isinstance(moe, feedforward.MoE)
    assert moe.shared_experts is None
    assert moe.gate.weight.shape == (cfg.num_experts, cfg.hidden_size)
    assert moe.experts_gate_up.shape == (
        cfg.num_experts, 2 * cfg.intermediate_size, cfg.hidden_size,
    )
    assert moe.experts_down.shape == (
        cfg.num_experts, cfg.hidden_size, cfg.intermediate_size,
    )


def test_olmoe_qk_norm_is_full_hdh():
    """OLMoE q_norm weight shape is [hidden_size]; k_norm weight shape is
    [(hidden_size / num_attention_heads) * num_key_value_heads].
    Source: modeling_olmoe.py:246-249."""
    cfg = _small_cfg()
    blk = m_layer.build_olmoe_decoder_layer(cfg, layer_idx=0, max_seq=32)
    qn = blk.attention.q_norm
    kn = blk.attention.k_norm
    assert qn is not None and kn is not None
    # FULL_HDH for q: n_q * head_dim = hidden_size.
    assert qn.weight.shape == (cfg.num_attention_heads * cfg.head_dim,)
    # FULL_HDH for k: n_kv * head_dim.
    assert kn.weight.shape == (cfg.num_key_value_heads * cfg.head_dim,)
