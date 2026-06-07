import torch

from api import kvcache, specs, types
from models.mistral import config, layer


def _config_tiny():
    return config.MistralConfig(
        hidden_size=64, num_attention_heads=8, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=2,
        rope_theta=1_000_000.0, rms_norm_eps=1e-5,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=False, dtype=torch.float32,
        sliding_window=None,
    )


def _synthetic_hf_state_dict(cfg, layer_idx):
    hs = cfg.hidden_size
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    I = cfg.intermediate_size
    base = f"model.layers.{layer_idx}"
    torch.manual_seed(42)
    return {
        f"{base}.input_layernorm.weight": torch.randn(hs),
        f"{base}.self_attn.q_proj.weight": torch.randn(q_out, hs),
        f"{base}.self_attn.k_proj.weight": torch.randn(kv_out, hs),
        f"{base}.self_attn.v_proj.weight": torch.randn(kv_out, hs),
        f"{base}.self_attn.o_proj.weight": torch.randn(hs, q_out),
        f"{base}.post_attention_layernorm.weight": torch.randn(hs),
        f"{base}.mlp.gate_proj.weight": torch.randn(I, hs),
        f"{base}.mlp.up_proj.weight": torch.randn(I, hs),
        f"{base}.mlp.down_proj.weight": torch.randn(hs, I),
    }


def test_load_hf_weights_into_layer():
    cfg = _config_tiny()
    blk = layer.build_mistral_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_hf_state_dict(cfg, layer_idx=0)
    layer.load_hf_mistral_layer(blk, sd, layer_idx=0)
    assert torch.allclose(
        blk.attention.q_proj.weight, sd["model.layers.0.self_attn.q_proj.weight"]
    )
    assert torch.allclose(
        blk.feedforward.down_proj.weight, sd["model.layers.0.mlp.down_proj.weight"]
    )


def test_load_then_forward():
    cfg = _config_tiny()
    blk = layer.build_mistral_decoder_layer(cfg, layer_idx=0, max_seq=32)
    sd = _synthetic_hf_state_dict(cfg, layer_idx=0)
    layer.load_hf_mistral_layer(blk, sd, layer_idx=0)
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
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (1, 4, cfg.hidden_size)
