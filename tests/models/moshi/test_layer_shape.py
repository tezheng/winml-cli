"""Moshi LM-decoder shape tests."""
from __future__ import annotations

import torch

from api import kvcache, specs, types
from models.moshi import config as m_config, layer as m_layer


_HF_MOSHI_TINY = {
    "model_type": "moshi",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,        # MHA
    "head_dim": 16,
    "ffn_dim": 128,                  # intermediate = 64
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "rms_norm_eps": 1e-8,
    "vocab_size": 64,
    "max_position_embeddings": 64,
    "sliding_window": 64,
    "tie_word_embeddings": False,
    "num_codebooks": 8,
    "audio_vocab_size": 32,
}


def _cfg() -> m_config.MoshiConfig:
    return m_config.MoshiConfig.from_hf_dict(_HF_MOSHI_TINY)


def test_layer_assembles_correct_shapes():
    cfg = _cfg()
    blk = m_layer.build_moshi_decoder_layer(cfg, layer_idx=0, max_seq=32)
    blk.eval()
    # MHA — n_q == n_kv heads, no biases.
    assert blk.attention.q_proj.in_features == cfg.hidden_size
    assert blk.attention.q_proj.out_features == cfg.num_attention_heads * cfg.head_dim
    assert blk.attention.q_proj.bias is None
    assert blk.attention.k_proj.out_features == cfg.num_key_value_heads * cfg.head_dim
    assert blk.attention.k_proj.bias is None
    assert blk.attention.v_proj.bias is None
    assert blk.attention.o_proj.bias is None
    # Fused gate/up SwiGLU. gate_up_proj.out_features == 2 * intermediate_size.
    ff = blk.feedforward
    assert ff.gate_proj is None and ff.up_proj is None
    assert ff.gate_up_proj.in_features == cfg.hidden_size
    assert ff.gate_up_proj.out_features == 2 * cfg.intermediate_size
    assert ff.down_proj.in_features == cfg.intermediate_size
    assert ff.down_proj.out_features == cfg.hidden_size


def test_layer_forward_shape():
    cfg = _cfg()
    blk = m_layer.build_moshi_decoder_layer(cfg, layer_idx=0, max_seq=32)
    blk.eval()
    B, S = 2, 5
    hidden = torch.randn(B, S, cfg.hidden_size)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=32,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        out = blk(hidden, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)
