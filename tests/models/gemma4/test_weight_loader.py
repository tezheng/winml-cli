import torch

from api import kvcache, specs, types
from models.gemma4 import config as gemma4_config, layer as gemma4_layer


def test_weight_loader_round_trip_random_state_dict_local_layer():
    """Build random state-dict tensors that match the HF naming and shapes; load."""
    cfg = _smallified()
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=0)
    sd = _random_state_dict(cfg, layer_idx=0, is_global=False)
    gemma4_layer.load_hf_gemma4_layer(blk, sd, layer_idx=0, cfg=cfg)
    # Verify a forward pass after weight loading still succeeds.
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (1, 4, cfg.hidden_size)


def test_weight_loader_round_trip_random_state_dict_global_layer():
    """Global layer uses global_head_dim and the global fixed_scale on absorb."""
    cfg = _smallified()
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=4)  # first global
    sd = _random_state_dict(cfg, layer_idx=4, is_global=True)
    gemma4_layer.load_hf_gemma4_layer(blk, sd, layer_idx=4, cfg=cfg)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.global_head_dim, max_seq=64,
    )
    x = torch.randn(1, 3, cfg.hidden_size)
    pos = torch.arange(3)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (1, 3, cfg.hidden_size)


def test_weight_loader_qk_norm_loads_straight_local():
    """B0.6: QK-norm weights load straight from HF — no absorb.

    Pre-B0.6 the loader baked `(1 + w) * fixed_scale * sqrt(Dh) - 1` into the
    weight to absorb the attention scale. HF Gemma 4 has no fixed scale and
    runtime scaling = 1.0 (modeling_gemma4.py:1195), so the weight loads as-is.
    """
    cfg = _smallified()
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=0)  # local
    sd = _random_state_dict(cfg, layer_idx=0, is_global=False)
    expected_q = torch.randn(cfg.head_dim)
    expected_k = torch.randn(cfg.head_dim)
    sd[f"model.layers.0.self_attn.q_norm.weight"] = expected_q.clone()
    sd[f"model.layers.0.self_attn.k_norm.weight"] = expected_k.clone()
    gemma4_layer.load_hf_gemma4_layer(blk, sd, layer_idx=0, cfg=cfg)
    assert torch.allclose(blk.attention.q_norm.weight, expected_q, atol=1e-6)
    assert torch.allclose(blk.attention.k_norm.weight, expected_k, atol=1e-6)


def test_weight_loader_qk_norm_loads_straight_global():
    """Global layer also loads QK-norm straight (Dh=global_head_dim)."""
    cfg = _smallified()
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=4)  # global
    sd = _random_state_dict(cfg, layer_idx=4, is_global=True)
    expected_q = torch.randn(cfg.global_head_dim)
    expected_k = torch.randn(cfg.global_head_dim)
    sd[f"model.layers.4.self_attn.q_norm.weight"] = expected_q.clone()
    sd[f"model.layers.4.self_attn.k_norm.weight"] = expected_k.clone()
    gemma4_layer.load_hf_gemma4_layer(blk, sd, layer_idx=4, cfg=cfg)
    assert torch.allclose(blk.attention.q_norm.weight, expected_q, atol=1e-6)
    assert torch.allclose(blk.attention.k_norm.weight, expected_k, atol=1e-6)


def _smallified():
    return gemma4_config.Gemma4Config(
        hidden_size=128, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, global_head_dim=64,
        intermediate_size=256,
        rope_theta_local=10000.0, rope_theta_global=1_000_000.0,
        partial_rotary_factor_global=0.25,
        sliding_window=16, sliding_window_pattern=4,
        rms_norm_eps=1e-6,
        vocab_size=64, max_position_embeddings=32,
        tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh",
        dtype=torch.float32,
        final_logit_softcap=30.0, attn_logit_softcap=None,
        num_kv_shared_layers=0, use_per_layer_embedding=False,
        ple_dim=32, attention_k_eq_v=False,
        qk_norm_local_fixed_scale=0.9916,
        qk_norm_global_fixed_scale=1.0228,
    )


def _random_state_dict(cfg, layer_idx, is_global):
    p = f"model.layers.{layer_idx}."
    D = cfg.hidden_size
    Hq = cfg.num_attention_heads
    Hk = cfg.num_key_value_heads
    Dh = cfg.global_head_dim if is_global else cfg.head_dim
    return {
        p + "input_layernorm.weight": torch.zeros(D),
        p + "post_attention_layernorm.weight": torch.zeros(D),
        p + "pre_feedforward_layernorm.weight": torch.zeros(D),
        p + "post_feedforward_layernorm.weight": torch.zeros(D),
        p + "self_attn.q_proj.weight": torch.randn(Hq * Dh, D),
        p + "self_attn.k_proj.weight": torch.randn(Hk * Dh, D),
        p + "self_attn.v_proj.weight": torch.randn(Hk * Dh, D),
        p + "self_attn.o_proj.weight": torch.randn(D, Hq * Dh),
        p + "self_attn.q_norm.weight": torch.zeros(Dh),
        p + "self_attn.k_norm.weight": torch.zeros(Dh),
        p + "mlp.gate_proj.weight": torch.randn(cfg.intermediate_size, D),
        p + "mlp.up_proj.weight": torch.randn(cfg.intermediate_size, D),
        p + "mlp.down_proj.weight": torch.randn(D, cfg.intermediate_size),
    }
