"""B6: DeepSeek-V3-MoE wrapper builds at SCALED-DOWN dims.

We can't build the production 671B-param layer in a unit test, but we can
verify the assembly pipeline composes correctly at reduced dims by
constructing a ``DeepSeekV3MoEConfig`` with overridden small fields.
"""
import torch

from api import feedforward, kvcache, specs, types
from models.deepseek_v3_moe import config as m_config, layer as m_layer


def _scaled_cfg() -> m_config.DeepSeekV3MoEConfig:
    # Override the production defaults with tiny dims for shape-only test.
    return m_config.DeepSeekV3MoEConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=24,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=4,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        n_group=4,
        topk_group=2,
        first_k_dense_replace=1,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=24,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        dtype=torch.float32,
    )


def test_v3_moe_dense_layer_forward_shape():
    cfg = _scaled_cfg()
    blk = m_layer.build_deepseek_v3_moe_decoder_layer(cfg, layer_idx=0, max_seq=32)
    blk.eval()
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=32,
        v_head_dim=cfg.v_head_dim,
    )
    B, S = 1, 5
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    with torch.no_grad():
        y = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert y.shape == (B, S, cfg.hidden_size)
    # Layer 0 is dense FFN (first_k_dense_replace=1).
    assert isinstance(blk.feedforward, feedforward.FeedForward)


def test_v3_moe_moe_layer_forward_shape():
    cfg = _scaled_cfg()
    blk = m_layer.build_deepseek_v3_moe_decoder_layer(cfg, layer_idx=1, max_seq=32)
    blk.eval()
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=32,
        v_head_dim=cfg.v_head_dim,
    )
    B, S = 1, 5
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    with torch.no_grad():
        y = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert y.shape == (B, S, cfg.hidden_size)
    moe = blk.feedforward
    assert isinstance(moe, feedforward.MoE)
    # V3 MoE layout: 16 experts, shared experts present, gate has bias buffer.
    assert moe.n_experts == cfg.n_routed_experts
    assert moe.shared_experts is not None
    assert hasattr(moe.gate, "e_score_correction_bias")
