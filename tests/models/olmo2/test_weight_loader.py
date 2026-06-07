"""OLMo 2 weight-loader unit tests — synthetic state dict, no HF download."""
import pytest
import torch

from models.olmo2 import config, layer


def _config_tiny():
    """Tiny config for fast unit testing of the loader."""
    return config.Olmo2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
        head_dim=16, intermediate_size=128, num_hidden_layers=2,
        rope_theta=500000.0, rms_norm_eps=1e-5,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=False, dtype=torch.float32,
        attention_bias=False,
    )


def _synthetic_layer0_state_dict(cfg):
    H = cfg.num_attention_heads
    Hk = cfg.num_key_value_heads
    Dh = cfg.head_dim
    D = cfg.hidden_size
    I = cfg.intermediate_size
    torch.manual_seed(0)
    return {
        "model.layers.0.self_attn.q_proj.weight":              torch.randn(H * Dh, D),
        "model.layers.0.self_attn.k_proj.weight":              torch.randn(Hk * Dh, D),
        "model.layers.0.self_attn.v_proj.weight":              torch.randn(Hk * Dh, D),
        "model.layers.0.self_attn.o_proj.weight":              torch.randn(D, H * Dh),
        "model.layers.0.self_attn.q_norm.weight":              torch.randn(H * Dh),
        "model.layers.0.self_attn.k_norm.weight":              torch.randn(Hk * Dh),
        "model.layers.0.post_attention_layernorm.weight":      torch.randn(D),
        "model.layers.0.post_feedforward_layernorm.weight":    torch.randn(D),
        "model.layers.0.mlp.gate_proj.weight":                 torch.randn(I, D),
        "model.layers.0.mlp.up_proj.weight":                   torch.randn(I, D),
        "model.layers.0.mlp.down_proj.weight":                 torch.randn(D, I),
    }


def test_load_hf_olmo2_weights_into_layer():
    cfg = _config_tiny()
    blk = layer.build_olmo2_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_layer0_state_dict(cfg)
    layer.load_hf_olmo2_layer(blk, sd, layer_idx=0)
    # Confirm a few weights actually copied.
    assert torch.allclose(
        blk.attention.q_proj.weight,
        sd["model.layers.0.self_attn.q_proj.weight"],
    )
    assert torch.allclose(
        blk.post_attn_sublayer_norm.weight,
        sd["model.layers.0.post_attention_layernorm.weight"],
    )
    assert torch.allclose(
        blk.attention.q_norm.weight,
        sd["model.layers.0.self_attn.q_norm.weight"],
    )


def test_load_hf_olmo2_missing_tensor_raises():
    cfg = _config_tiny()
    blk = layer.build_olmo2_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_layer0_state_dict(cfg)
    del sd["model.layers.0.self_attn.q_norm.weight"]
    with pytest.raises(KeyError, match="missing tensors"):
        layer.load_hf_olmo2_layer(blk, sd, layer_idx=0)


def test_load_then_forward_olmo2():
    """End-to-end: build, load synthetic weights, run a forward pass."""
    from api import kvcache, specs, types
    cfg = _config_tiny()
    blk = layer.build_olmo2_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_layer0_state_dict(cfg)
    layer.load_hf_olmo2_layer(blk, sd, layer_idx=0)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=32,
    )
    x = torch.randn(1, 5, cfg.hidden_size)
    pos = torch.arange(5)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
