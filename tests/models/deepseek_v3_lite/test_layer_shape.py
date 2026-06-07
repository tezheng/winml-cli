"""V3-Lite shape tests."""
import torch

from api import kvcache, specs, types
from models.deepseek_v3_lite import config as m_config, layer as m_layer


def _small_cfg() -> m_config.DeepSeekV3LiteConfig:
    return m_config.DeepSeekV3LiteConfig(
        hidden_size=64, num_attention_heads=4, num_hidden_layers=4,
        intermediate_size=128, moe_intermediate_size=24,
        n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=4,
        routed_scaling_factor=2.5, norm_topk_prob=True,
        n_group=4, topk_group=2, first_k_dense_replace=1,
        kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
        v_head_dim=16, q_lora_rank=24,
        rope_theta=10000.0, rms_norm_eps=1e-6,
        vocab_size=100, max_position_embeddings=64,
        tie_word_embeddings=False, attention_bias=False,
        dtype=torch.float32,
    )


def test_v3_lite_layer1_moe_forward_shape_and_router_module_layout():
    cfg = _small_cfg()
    blk = m_layer.build_deepseek_v3_lite_decoder_layer(cfg, layer_idx=1, max_seq=16)
    from api import feedforward as _ff
    assert isinstance(blk.feedforward, _ff.MoE)
    # V3 router has gate.weight Parameter + e_score_correction_bias buffer.
    assert hasattr(blk.feedforward.gate, "weight")
    assert hasattr(blk.feedforward.gate, "e_score_correction_bias")
    # The bias buffer must be of shape [n_experts] and fp32.
    bias = blk.feedforward.gate.e_score_correction_bias
    assert bias.shape == (cfg.n_routed_experts,)
    assert bias.dtype == torch.float32

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


def test_v3_lite_layer0_dense_is_swiglu_ffn():
    cfg = _small_cfg()
    blk = m_layer.build_deepseek_v3_lite_decoder_layer(cfg, layer_idx=0, max_seq=16)
    from api import feedforward as _ff
    assert isinstance(blk.feedforward, _ff.FeedForward)


def test_v3_lite_q_lora_path_active():
    """V3-Lite uses q_lora_rank>0 → q_a_proj/q_a_layernorm/q_b_proj."""
    cfg = _small_cfg()
    blk = m_layer.build_deepseek_v3_lite_decoder_layer(cfg, layer_idx=1, max_seq=16)
    attn = blk.attention
    assert attn.q_proj is None
    assert attn.q_a_proj is not None
    assert attn.q_a_layernorm is not None
    assert attn.q_b_proj is not None
