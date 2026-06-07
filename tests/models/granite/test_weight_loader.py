"""Synthetic weight loader test — no HF model required."""
import torch

from api import kvcache, specs, types
from models.granite import config, layer


def _config_small():
    return config.GraniteConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=2,
        rope_theta=10_000.0, rms_norm_eps=1e-5,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=False, dtype=torch.float32,
        attention_multiplier=0.25,
        residual_multiplier=0.5,
        embedding_multiplier=2.0,
        logits_scaling=4.0,
    )


def _synth_state_dict(cfg: config.GraniteConfig, layer_idx: int = 0) -> dict:
    Hq, Hk, Dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    D, I = cfg.hidden_size, cfg.intermediate_size
    prefix = f"model.layers.{layer_idx}"
    torch.manual_seed(0)
    return {
        f"{prefix}.input_layernorm.weight":           torch.randn(D),
        f"{prefix}.self_attn.q_proj.weight":          torch.randn(Hq * Dh, D),
        f"{prefix}.self_attn.k_proj.weight":          torch.randn(Hk * Dh, D),
        f"{prefix}.self_attn.v_proj.weight":          torch.randn(Hk * Dh, D),
        f"{prefix}.self_attn.o_proj.weight":          torch.randn(D, Hq * Dh),
        f"{prefix}.post_attention_layernorm.weight":  torch.randn(D),
        f"{prefix}.mlp.gate_proj.weight":             torch.randn(I, D),
        f"{prefix}.mlp.up_proj.weight":               torch.randn(I, D),
        f"{prefix}.mlp.down_proj.weight":             torch.randn(D, I),
    }


def test_load_hf_weights_into_layer():
    cfg = _config_small()
    blk = layer.build_granite_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synth_state_dict(cfg)
    layer.load_hf_granite_layer(blk, sd, layer_idx=0)
    # Spot-check a tensor was actually copied (not still random init).
    assert torch.allclose(
        blk.pre_attn_norm.weight, sd["model.layers.0.input_layernorm.weight"]
    )
    assert torch.allclose(
        blk.feedforward.down_proj.weight, sd["model.layers.0.mlp.down_proj.weight"]
    )


def test_load_then_forward():
    cfg = _config_small()
    blk = layer.build_granite_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synth_state_dict(cfg)
    layer.load_hf_granite_layer(blk, sd, layer_idx=0)
    blk.eval()
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
    with torch.no_grad():
        out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (1, 5, cfg.hidden_size)
    assert torch.isfinite(out).all()
