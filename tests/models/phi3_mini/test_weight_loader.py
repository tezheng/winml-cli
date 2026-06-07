"""Synthetic weight loader test — no HF model required."""
import torch

from api import kvcache, specs, types
from models.phi3_mini import config, layer


def _cfg_small():
    return config.Phi3MiniConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
        head_dim=16, intermediate_size=128, num_hidden_layers=2,
        rope_theta=10_000.0, rms_norm_eps=1e-5,
        vocab_size=100, max_position_embeddings=32,
        original_max_position_embeddings=32,
        tie_word_embeddings=False, dtype=torch.float32,
        sliding_window=16,
        rope_type="default",
    )


def _synth_state_dict(cfg, layer_idx: int = 0) -> dict:
    Hq, Hk, Dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    D, I = cfg.hidden_size, cfg.intermediate_size
    prefix = f"model.layers.{layer_idx}"
    torch.manual_seed(0)
    return {
        f"{prefix}.input_layernorm.weight":           torch.randn(D),
        f"{prefix}.self_attn.qkv_proj.weight":        torch.randn((Hq + 2*Hk) * Dh, D),
        f"{prefix}.self_attn.o_proj.weight":          torch.randn(D, Hq * Dh),
        f"{prefix}.post_attention_layernorm.weight":  torch.randn(D),
        f"{prefix}.mlp.gate_up_proj.weight":          torch.randn(2 * I, D),
        f"{prefix}.mlp.down_proj.weight":             torch.randn(D, I),
    }


def test_load_hf_weights_into_layer():
    cfg = _cfg_small()
    blk = layer.build_phi3_mini_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synth_state_dict(cfg)
    layer.load_hf_phi3_mini_layer(blk, sd, layer_idx=0)
    assert torch.allclose(
        blk.attention.qkv_proj.weight, sd["model.layers.0.self_attn.qkv_proj.weight"]
    )
    assert torch.allclose(
        blk.feedforward.gate_up_proj.weight, sd["model.layers.0.mlp.gate_up_proj.weight"]
    )


def test_load_then_forward():
    cfg = _cfg_small()
    blk = layer.build_phi3_mini_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synth_state_dict(cfg)
    layer.load_hf_phi3_mini_layer(blk, sd, layer_idx=0)
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
