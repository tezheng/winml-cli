"""V3.2 (DSA) shape-only tests."""
import pytest
import torch

from api import block, kvcache, specs, types
from models.deepseek_v32 import config as m_config, layer as m_layer


def _cfg() -> m_config.DeepSeekV32Config:
    return m_config.DeepSeekV32Config(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=3,
        intermediate_size=128,
        moe_intermediate_size=24,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
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
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        dtype=torch.float32,
        indexer_dim=8,
        indexer_top_k=16,
        indexer_warmup_tokens=1000,
    )


def test_v32_block_spec_composes_with_dsa_kind():
    cfg = _cfg()
    blk_spec = cfg.to_block_spec(layer_idx=1)
    attn = blk_spec.token_mixer
    assert attn.kind == types.AttentionKind.DSA
    assert attn.indexer is not None
    assert attn.indexer.indexer_dim == 8
    assert attn.indexer.top_k == 16


def test_v32_block_allocates_indexer_modules():
    cfg = _cfg()
    blk = m_layer.build_deepseek_v32_decoder_layer(cfg, layer_idx=1, max_seq=16)
    assert hasattr(blk.attention, "indexer_q_proj")
    assert hasattr(blk.attention, "indexer_k_proj")
    # indexer_q_proj: hidden -> H * indexer_dim.
    assert blk.attention.indexer_q_proj.weight.shape == (
        cfg.num_attention_heads * cfg.indexer_dim, cfg.hidden_size,
    )
    # indexer_k_proj: hidden -> indexer_dim.
    assert blk.attention.indexer_k_proj.weight.shape == (
        cfg.indexer_dim, cfg.hidden_size,
    )


def test_v32_forward_raises_not_implemented():
    cfg = _cfg()
    blk = m_layer.build_deepseek_v32_decoder_layer(cfg, layer_idx=1, max_seq=16)
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
    x = torch.randn(1, 3, cfg.hidden_size)
    with pytest.raises(NotImplementedError, match="DSA"):
        blk(x, position_ids=torch.arange(3), cache=cache, start_pos=0)


def test_v32_indexer_spec_rejected_without_dsa_kind():
    """Verify the spec gate: DSA kind requires .indexer field set."""
    spec = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=24,
        kind=types.AttentionKind.DSA,
        qkv_layout=types.QKVLayout.MLA_LATENT,
        mask_kind=types.MaskKind.CAUSAL,
        q_lora_rank=16, kv_lora_rank=8,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        indexer=None,                  # missing
    )
    with pytest.raises(ValueError, match="indexer"):
        from api import attention as _attn
        _attn.Attention(spec, hidden_size=64, max_seq=8, dtype=torch.float32)
