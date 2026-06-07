"""Voxtral LM-decoder shape tests."""
from __future__ import annotations

import torch

from api import kvcache, specs, types
from models.voxtral import config as v_config, layer as v_layer


_HF_VOXTRAL_TINY_TEXT = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,           # GQA 2x reduction
    "head_dim": 16,
    "intermediate_size": 128,
    "rope_parameters": {"rope_theta": 1e8, "rope_type": "default"},
    "rms_norm_eps": 1e-5,
    "vocab_size": 64,
    "max_position_embeddings": 64,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}


def _cfg() -> v_config.VoxtralConfig:
    return v_config.VoxtralConfig.from_hf_dict(_HF_VOXTRAL_TINY_TEXT)


def test_layer_assembles_correct_shapes():
    cfg = _cfg()
    blk = v_layer.build_voxtral_decoder_layer(cfg, layer_idx=0, max_seq=32)
    blk.eval()
    # GQA — q is n_q*Dh, k/v are n_kv*Dh.
    assert blk.attention.q_proj.in_features == cfg.hidden_size
    assert blk.attention.q_proj.out_features == cfg.num_attention_heads * cfg.head_dim
    assert blk.attention.q_proj.bias is None
    assert blk.attention.k_proj.out_features == cfg.num_key_value_heads * cfg.head_dim
    assert blk.attention.k_proj.bias is None
    assert blk.attention.v_proj.bias is None
    assert blk.attention.o_proj.bias is None
    # SwiGLU NOT fused.
    ff = blk.feedforward
    assert ff.gate_up_proj is None
    assert ff.gate_proj.out_features == cfg.intermediate_size
    assert ff.up_proj.out_features == cfg.intermediate_size
    assert ff.down_proj.in_features == cfg.intermediate_size


def test_layer_forward_shape():
    cfg = _cfg()
    blk = v_layer.build_voxtral_decoder_layer(cfg, layer_idx=0, max_seq=32)
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
