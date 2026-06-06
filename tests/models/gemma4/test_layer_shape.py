import torch

from api import kvcache, specs, types
from models.gemma4 import config as gemma4_config, layer as gemma4_layer


def test_gemma4_e2b_layer_0_local_forward_shape():
    cfg = _e2b_smallified_config()  # smaller version for fast test
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=0)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    B, S = 1, 5
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_gemma4_e2b_layer_4_global_forward_shape():
    cfg = _e2b_smallified_config()
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=4)  # first global
    # Global layer uses global_head_dim (doubled in E2B) - the KV cache must allocate
    # to the global head_dim.
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.global_head_dim, max_seq=64,
    )
    B, S = 1, 3
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def _e2b_smallified_config():
    # A small variant of E2B that fits in CPU memory for shape tests
    return gemma4_config.Gemma4Config(
        hidden_size=128, num_hidden_layers=10,
        num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, global_head_dim=64,
        intermediate_size=512,
        rope_theta_local=10000.0, rope_theta_global=1_000_000.0,
        partial_rotary_factor_global=0.25,
        sliding_window=16, sliding_window_pattern=4,
        rms_norm_eps=1e-6,
        vocab_size=512, max_position_embeddings=64,
        tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh",
        dtype=torch.float32,
        final_logit_softcap=30.0, attn_logit_softcap=None,
        num_kv_shared_layers=0,  # disable sharing for the shape test
        use_per_layer_embedding=False,
        ple_dim=64,
        attention_k_eq_v=False,
        qk_norm_local_fixed_scale=0.9916,
        qk_norm_global_fixed_scale=1.0228,
    )
