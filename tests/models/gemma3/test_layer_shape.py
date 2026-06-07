"""Gemma 3 layer-shape tests."""
import pytest
import torch

from api import kvcache, specs, types
from models.gemma3 import config, layer


def _config_1b():
    return config.Gemma3Config(
        hidden_size=1152, num_hidden_layers=26,
        num_attention_heads=4, num_key_value_heads=1,
        head_dim=256, intermediate_size=6912,
        rope_theta_local=10_000.0, rope_theta_global=1_000_000.0,
        rms_norm_eps=1e-6,
        vocab_size=262144, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
        sliding_window=512, sliding_window_pattern=6,
        query_pre_attn_scalar=256,
        attn_logit_softcap=None, final_logit_softcap=None,
        attention_bias=False,
    )


@pytest.mark.parametrize("layer_idx", [0, 5])     # 0=sliding, 5=first global
def test_gemma3_decoder_layer_forward_shape(layer_idx):
    cfg = _config_1b()
    blk = layer.build_gemma3_decoder_layer(cfg, layer_idx=layer_idx, max_seq=64)
    B, S = 1, 6
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


def test_gemma3_qk_norm_per_head_dh_shape():
    """B3: q_norm.weight shape == [head_dim], k_norm.weight shape == [head_dim].
    Source: modeling_gemma3.py:337-338.
    """
    cfg = _config_1b()
    blk = layer.build_gemma3_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.q_norm is not None
    assert blk.attention.q_norm.weight.shape == (cfg.head_dim,)
    assert blk.attention.k_norm.weight.shape == (cfg.head_dim,)


def test_gemma3_dispatch_layer_0_vs_5():
    """Layer 0 is sliding (theta=10000), layer 5 is global (theta=1000000)."""
    cfg = _config_1b()
    blk_local = layer.build_gemma3_decoder_layer(cfg, layer_idx=0, max_seq=64)
    blk_global = layer.build_gemma3_decoder_layer(cfg, layer_idx=5, max_seq=64)
    assert blk_local.spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert blk_local.spec.token_mixer.sliding_window == 512
    assert blk_local.spec.token_mixer.rope.base_theta == 10_000.0
    assert blk_global.spec.token_mixer.mask_kind == types.MaskKind.CAUSAL
    assert blk_global.spec.token_mixer.sliding_window is None
    assert blk_global.spec.token_mixer.rope.base_theta == 1_000_000.0


def test_gemma3_no_softcap_in_attention_spec():
    cfg = _config_1b()
    blk = layer.build_gemma3_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.spec.token_mixer.logit_softcap is None
