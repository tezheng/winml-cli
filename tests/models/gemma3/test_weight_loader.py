"""Gemma 3 weight-loader unit tests — synthetic state dict."""
import pytest
import torch

from models.gemma3 import config, layer


def _config_tiny():
    return config.Gemma3Config(
        hidden_size=64, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128,
        rope_theta_local=10_000.0, rope_theta_global=1_000_000.0,
        rms_norm_eps=1e-6,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=True, dtype=torch.float32,
        sliding_window=8, sliding_window_pattern=6,
        query_pre_attn_scalar=16,
        attn_logit_softcap=None, final_logit_softcap=None,
        attention_bias=False,
    )


def _synthetic_sd(cfg, layer_idx=0):
    H = cfg.num_attention_heads
    Hk = cfg.num_key_value_heads
    Dh = cfg.head_dim
    D = cfg.hidden_size
    I = cfg.intermediate_size
    torch.manual_seed(0)
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.input_layernorm.weight":            torch.randn(D),
        f"{prefix}.post_attention_layernorm.weight":   torch.randn(D),
        f"{prefix}.pre_feedforward_layernorm.weight":  torch.randn(D),
        f"{prefix}.post_feedforward_layernorm.weight": torch.randn(D),
        f"{prefix}.self_attn.q_proj.weight":           torch.randn(H * Dh, D),
        f"{prefix}.self_attn.k_proj.weight":           torch.randn(Hk * Dh, D),
        f"{prefix}.self_attn.v_proj.weight":           torch.randn(Hk * Dh, D),
        f"{prefix}.self_attn.o_proj.weight":           torch.randn(D, H * Dh),
        f"{prefix}.self_attn.q_norm.weight":           torch.randn(Dh),
        f"{prefix}.self_attn.k_norm.weight":           torch.randn(Dh),
        f"{prefix}.mlp.gate_proj.weight":              torch.randn(I, D),
        f"{prefix}.mlp.up_proj.weight":                torch.randn(I, D),
        f"{prefix}.mlp.down_proj.weight":              torch.randn(D, I),
    }


def test_load_hf_gemma3_weights_into_layer():
    cfg = _config_tiny()
    blk = layer.build_gemma3_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_sd(cfg, 0)
    layer.load_hf_gemma3_layer(blk, sd, layer_idx=0)
    assert torch.allclose(
        blk.attention.q_norm.weight,
        sd["model.layers.0.self_attn.q_norm.weight"],
    )
    assert torch.allclose(
        blk.pre_attn_norm.weight,
        sd["model.layers.0.input_layernorm.weight"],
    )


def test_load_hf_gemma3_missing_tensor_raises():
    cfg = _config_tiny()
    blk = layer.build_gemma3_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_sd(cfg, 0)
    del sd["model.layers.0.self_attn.q_norm.weight"]
    with pytest.raises(KeyError, match="missing tensors"):
        layer.load_hf_gemma3_layer(blk, sd, layer_idx=0)


def test_load_then_forward_gemma3():
    from api import kvcache, specs, types
    cfg = _config_tiny()
    blk = layer.build_gemma3_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_sd(cfg, 0)
    layer.load_hf_gemma3_layer(blk, sd, layer_idx=0)
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
