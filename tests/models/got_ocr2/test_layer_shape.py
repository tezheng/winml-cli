"""GOT-OCR 2.0 LM-decoder shape tests.

Verifies the assembled DecoderBlock has the expected projection shapes
including the Qwen2-distinctive QKV biases, no QK-norm modules, and the
correct concat-prefix-friendly forward signature (vision tokens are
already in the embedding sequence).
"""
from __future__ import annotations

import torch

from api import kvcache, specs, types
from models.got_ocr2 import config as g_config, layer as g_layer


_HF_GOT_OCR2_TINY = {
    "model_type": "got_ocr2",
    "tie_word_embeddings": True,
    "image_token_index": 99,
    "image_seq_length": 4,
    "text_config": {
        "model_type": "qwen2",
        "vocab_size": 200,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
        "use_sliding_window": False,
        "sliding_window": None,
    },
}


def _build_block():
    cfg = g_config.GotOcr2Config.from_hf_dict(_HF_GOT_OCR2_TINY)
    blk = g_layer.build_got_ocr2_decoder_layer(cfg, layer_idx=0, max_seq=64)
    return cfg, blk


def test_block_assembles_with_qkv_biases():
    cfg, blk = _build_block()
    attn = blk.attention
    # Q biases present (Qwen2 distinctive).
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None
    # No QK-norm (Qwen2 predates Qwen3 QK-norm).
    assert attn.q_norm is None
    assert attn.k_norm is None
    # Sizes match config.
    head_dim = cfg.head_dim
    assert attn.q_proj.out_features == cfg.num_attention_heads * head_dim
    assert attn.k_proj.out_features == cfg.num_key_value_heads * head_dim
    assert attn.v_proj.out_features == cfg.num_key_value_heads * head_dim
    assert attn.o_proj.in_features == cfg.num_attention_heads * head_dim
    # FFN sizes.
    assert blk.feedforward.gate_proj.out_features == cfg.intermediate_size
    assert blk.feedforward.up_proj.out_features == cfg.intermediate_size
    assert blk.feedforward.down_proj.in_features == cfg.intermediate_size


def test_forward_shape_with_vision_prefix_tokens():
    """Concat-prefix fusion: vision tokens are prepended to text tokens
    in the SAME hidden_states tensor; the LM layer sees a single
    [B, S_total, H] sequence and operates uniformly. We mimic this here
    by sampling a random hidden_states tensor of size
    (image_seq_length + n_text_tokens)."""
    cfg, blk = _build_block()
    blk.eval()
    B = 1
    n_text = 5
    S = cfg.image_seq_length + n_text                # 4 + 5 = 9
    hidden = torch.randn(B, S, cfg.hidden_size)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32,
        v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq=64,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        out = blk(hidden, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_load_then_forward_does_not_change_shape():
    """Sanity: synthesize a random state-dict for the LM layer and load
    it. Numerical equivalence vs HF is checked separately."""
    cfg, blk = _build_block()
    sd = {}
    L = 0
    prefix = f"model.language_model.layers.{L}"
    # Hand-craft the canonical Qwen2 tensor set (mirrors load_hf_got_ocr2_layer
    # mapping). Shapes must match the slots in the assembled block.
    H = cfg.hidden_size
    Hq = cfg.num_attention_heads * cfg.head_dim
    Hk = cfg.num_key_value_heads * cfg.head_dim
    I = cfg.intermediate_size
    sd[f"{prefix}.input_layernorm.weight"] = torch.ones(H)
    sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(Hq, H) * 0.02
    sd[f"{prefix}.self_attn.q_proj.bias"]   = torch.randn(Hq) * 0.02
    sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(Hk, H) * 0.02
    sd[f"{prefix}.self_attn.k_proj.bias"]   = torch.randn(Hk) * 0.02
    sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(Hk, H) * 0.02
    sd[f"{prefix}.self_attn.v_proj.bias"]   = torch.randn(Hk) * 0.02
    sd[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(H, Hq) * 0.02
    sd[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(H)
    sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(I, H) * 0.02
    sd[f"{prefix}.mlp.up_proj.weight"]   = torch.randn(I, H) * 0.02
    sd[f"{prefix}.mlp.down_proj.weight"] = torch.randn(H, I) * 0.02

    g_layer.load_hf_got_ocr2_layer(blk, sd, layer_idx=0)
    blk.eval()
    B, S = 1, 9
    hidden = torch.randn(B, S, cfg.hidden_size)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=64,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        out = blk(hidden, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)
