"""Qwen2.5-VL LM-decoder shape tests (tiny config + synthetic forward)."""
from __future__ import annotations

import torch

from api import kvcache, specs, types
from models.qwen2_5_vl import config as q_config, layer as q_layer


_HF_QWEN2_5_VL_TINY = {
    "model_type": "qwen2_5_vl",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen2_5_vl_text",
        "vocab_size": 200,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            # head_dim = 128 / 4 = 32. mrope_section must sum to 32/2 = 16.
            "type": "mrope",
            "mrope_section": [4, 6, 6],
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
        "use_sliding_window": False,
        "sliding_window": None,
    },
}


def _build_block():
    cfg = q_config.Qwen2_5_VLConfig.from_hf_dict(_HF_QWEN2_5_VL_TINY)
    blk = q_layer.build_qwen2_5_vl_decoder_layer(cfg, layer_idx=0, max_seq=32)
    return cfg, blk


def test_block_has_qkv_biases_and_gqa_shapes():
    cfg, blk = _build_block()
    attn = blk.attention
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None
    assert attn.q_proj.out_features == cfg.num_attention_heads * cfg.head_dim
    assert attn.k_proj.out_features == cfg.num_key_value_heads * cfg.head_dim
    assert attn.v_proj.out_features == cfg.num_key_value_heads * cfg.head_dim
    # GQA: 4 q-heads, 2 kv-heads → repeat 2x.
    assert cfg.num_attention_heads // cfg.num_key_value_heads == 2


def test_forward_with_3d_position_ids():
    """The key Qwen2.5-VL feature: when position_ids has shape [3, B, S],
    the attention runs M-RoPE. Sanity: forward returns the expected
    [B, S, hidden] shape."""
    cfg, blk = _build_block()
    blk.eval()
    B, S = 1, 7
    hidden = torch.randn(B, S, cfg.hidden_size)
    # Position IDs: distinct T/H/W. For a real model the processor builds
    # them per token — we just need any [3, B, S] long tensor.
    pos_t = torch.arange(S)
    pos_h = torch.arange(S) + 1
    pos_w = torch.arange(S) + 2
    position_ids = torch.stack([pos_t, pos_h, pos_w], dim=0).unsqueeze(1).expand(3, B, S)

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
    with torch.no_grad():
        out = blk(hidden, position_ids=position_ids, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_forward_with_1d_position_ids_text_only_path():
    """When all 3 axes are equal we can pass [3, B, S] with equal rows
    — this is the 'text-only' fallback that M-RoPE supports."""
    cfg, blk = _build_block()
    blk.eval()
    B, S = 1, 5
    hidden = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    # Equal-axis [3, B, S].
    position_ids = pos[None, None, :].expand(3, B, S)
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
    with torch.no_grad():
        out = blk(hidden, position_ids=position_ids, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)
