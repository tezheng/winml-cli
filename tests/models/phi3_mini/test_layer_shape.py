import torch

from api import kvcache, specs, types
from models.phi3_mini import config, layer


def _phi3_mini_4k():
    return config.Phi3MiniConfig(
        hidden_size=3072, num_attention_heads=32, num_key_value_heads=32,
        head_dim=96, intermediate_size=8192, num_hidden_layers=32,
        rope_theta=10000.0, rms_norm_eps=1e-5,
        vocab_size=32064, max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        tie_word_embeddings=False, dtype=torch.float32,
        sliding_window=2047, partial_rotary_factor=1.0,
        rope_type="default",
    )


def test_phi3_mini_decoder_layer_forward_shape():
    cfg = _phi3_mini_4k()
    blk = layer.build_phi3_mini_decoder_layer(cfg, layer_idx=0, max_seq=64)
    B, S = 1, 8
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_phi3_mini_uses_fused_qkv_and_gate_up():
    cfg = _phi3_mini_4k()
    blk = layer.build_phi3_mini_decoder_layer(cfg, layer_idx=0, max_seq=64)
    # FUSED QKV: weight shape (Hq + 2*Hk) * Dh = (32 + 64) * 96 = 9216.
    assert blk.attention.qkv_proj is not None
    assert blk.attention.qkv_proj.weight.shape == (
        (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim,
        cfg.hidden_size,
    )
    assert blk.attention.q_proj is None
    assert blk.attention.k_proj is None
    assert blk.attention.v_proj is None
    # FUSED gate_up: weight shape 2 * I = 16384.
    assert blk.feedforward.gate_up_proj is not None
    assert blk.feedforward.gate_up_proj.weight.shape == (
        2 * cfg.intermediate_size, cfg.hidden_size,
    )
    assert blk.feedforward.gate_proj is None
    assert blk.feedforward.up_proj is None


def test_phi3_mini_no_qk_norm():
    cfg = _phi3_mini_4k()
    blk = layer.build_phi3_mini_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.q_norm is None
    assert blk.attention.k_norm is None


def test_phi3_mini_swa_wired():
    cfg = _phi3_mini_4k()
    blk = layer.build_phi3_mini_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert blk.spec.token_mixer.sliding_window == 2047
