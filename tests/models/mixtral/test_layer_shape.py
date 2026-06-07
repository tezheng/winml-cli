"""B6: Mixtral decoder-layer shape & MoE infra sanity tests."""
import pytest
import torch

from api import feedforward, kvcache, specs, types
from models.mixtral import config as m_config, layer as m_layer


def _small_cfg(sliding_window=None) -> m_config.MixtralConfig:
    return m_config.MixtralConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-5,
        vocab_size=100,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )


def test_mixtral_layer_forward_shape():
    cfg = _small_cfg()
    blk = m_layer.build_mixtral_decoder_layer(cfg, layer_idx=0, max_seq=32)
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


def test_mixtral_moe_module_layout():
    """Verify the MoE module exposes the expected tensors with HF-compatible
    shapes. Source: modeling_mixtral.py:62-72, 101-107."""
    cfg = _small_cfg()
    blk = m_layer.build_mixtral_decoder_layer(cfg, layer_idx=0, max_seq=32)
    moe = blk.feedforward
    assert isinstance(moe, feedforward.MoE)
    # Router gate: nn.Linear(hidden, n_experts, bias=False), weight at
    # [n_experts, hidden] per modeling_mixtral.py:107.
    assert moe.gate.weight.shape == (cfg.num_local_experts, cfg.hidden_size)
    # Experts: gate_up_proj [E, 2*I, H]; down_proj [E, H, I].
    assert moe.experts_gate_up.shape == (
        cfg.num_local_experts, 2 * cfg.intermediate_size, cfg.hidden_size,
    )
    assert moe.experts_down.shape == (
        cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size,
    )
    # No shared experts on Mixtral.
    assert moe.shared_experts is None


def test_mixtral_router_always_normalizes_topk():
    """B6 invariant — Mixtral renormalizes top-k weights to sum 1, ALWAYS.
    Source: modeling_mixtral.py:114.

    We feed random inputs through MoE._route_softmax and assert the returned
    weights sum to 1 per token."""
    cfg = _small_cfg()
    blk = m_layer.build_mixtral_decoder_layer(cfg, layer_idx=0, max_seq=32)
    moe = blk.feedforward
    # Realistic random gate weights.
    torch.manual_seed(0)
    with torch.no_grad():
        moe.gate.weight.normal_(0, 1)
    x = torch.randn(7, cfg.hidden_size)   # 7 tokens.
    with torch.no_grad():
        topk_idx, topk_w = moe._route_softmax(x)
    assert topk_idx.shape == (7, cfg.num_experts_per_tok)
    assert topk_w.shape == (7, cfg.num_experts_per_tok)
    # Each row sums to 1.0 within tolerance.
    s = topk_w.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), s


def test_mixtral_layer_supports_swa_when_set():
    cfg = _small_cfg(sliding_window=8)
    blk = m_layer.build_mixtral_decoder_layer(cfg, layer_idx=0, max_seq=32)
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
    B, S = 1, 3
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    with torch.no_grad():
        y = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert y.shape == (B, S, cfg.hidden_size)
