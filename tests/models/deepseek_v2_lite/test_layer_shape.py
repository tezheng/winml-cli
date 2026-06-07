"""Shape & forward smoke tests for DeepSeek-V2-Lite decoder layer."""
import torch

from api import kvcache, specs, types
from models.deepseek_v2_lite import config as m_config, layer as m_layer


def _small_cfg() -> m_config.DeepSeekV2LiteConfig:
    """A tiny V2-Lite config — preserves architecture, scaled down for speed."""
    return m_config.DeepSeekV2LiteConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=24,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=None,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        vocab_size=128,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        dtype=torch.float32,
        rope_type="default",       # no yarn for synthetic smoke
    )


def test_v2_lite_layer0_dense_forward_shape():
    cfg = _small_cfg()
    blk = m_layer.build_deepseek_v2_lite_decoder_layer(cfg, layer_idx=0, max_seq=16)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=16,
        v_head_dim=cfg.v_head_dim,
    )
    x = torch.randn(1, 5, cfg.hidden_size)
    out = blk(x, position_ids=torch.arange(5), cache=cache, start_pos=0)
    assert out.shape == (1, 5, cfg.hidden_size)
    # Layer 0 is dense — feedforward is FeedForward, NOT MoE.
    from api import feedforward as _ff
    assert isinstance(blk.feedforward, _ff.FeedForward)


def test_v2_lite_layer1_moe_forward_shape():
    cfg = _small_cfg()
    blk = m_layer.build_deepseek_v2_lite_decoder_layer(cfg, layer_idx=1, max_seq=16)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=16,
        v_head_dim=cfg.v_head_dim,
    )
    x = torch.randn(1, 5, cfg.hidden_size)
    out = blk(x, position_ids=torch.arange(5), cache=cache, start_pos=0)
    assert out.shape == (1, 5, cfg.hidden_size)
    from api import feedforward as _ff
    assert isinstance(blk.feedforward, _ff.MoE)
    # MoE expert packed tensor shapes.
    assert blk.feedforward.experts_gate_up.shape == (
        cfg.n_routed_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
    assert blk.feedforward.experts_down.shape == (
        cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    # Shared experts (intermediate = moe_intermediate * n_shared).
    assert blk.feedforward.shared_experts is not None
    assert blk.feedforward.shared_experts.gate_proj.weight.shape == (
        cfg.moe_intermediate_size * cfg.n_shared_experts, cfg.hidden_size)


def test_v2_lite_attention_mla_direct_q_proj():
    cfg = _small_cfg()
    blk = m_layer.build_deepseek_v2_lite_decoder_layer(cfg, layer_idx=0, max_seq=16)
    attn = blk.attention
    assert attn.q_proj is not None
    assert attn.q_a_proj is None
    # Shape: hidden -> H * qk_head_dim.
    assert attn.q_proj.weight.shape == (
        cfg.num_attention_heads * cfg.qk_head_dim, cfg.hidden_size,
    )
