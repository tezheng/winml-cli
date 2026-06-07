"""Gemma 2 layer-shape tests."""
import pytest
import torch

from api import kvcache, specs, types
from models.gemma2 import config, layer


def _config_2b():
    return config.Gemma2Config(
        hidden_size=2304, num_hidden_layers=26,
        num_attention_heads=8, num_key_value_heads=4,
        head_dim=256, intermediate_size=9216,
        rope_theta=10000.0, rms_norm_eps=1e-6,
        vocab_size=256000, max_position_embeddings=8192,
        tie_word_embeddings=True, dtype=torch.float32,
        sliding_window=4096, query_pre_attn_scalar=256,
        attn_logit_softcap=50.0, final_logit_softcap=30.0,
        attention_bias=False,
    )


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_gemma2_decoder_layer_forward_shape(layer_idx):
    """Both sliding (idx 0) and full (idx 1) layers run a forward."""
    cfg = _config_2b()
    blk = layer.build_gemma2_decoder_layer(cfg, layer_idx=layer_idx, max_seq=64)
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


def test_gemma2_decoder_layer_has_sandwich_norm_topology():
    """B3: built block must have ALL four norm modules (PRE_AND_POST sandwich)."""
    cfg = _config_2b()
    blk = layer.build_gemma2_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.pre_attn_norm is not None
    assert blk.post_attn_sublayer_norm is not None
    assert blk.pre_ffn_norm is not None
    assert blk.post_ffn_sublayer_norm is not None


def test_gemma2_layer_idx_0_is_sliding_layer_idx_1_is_full():
    """Default Gemma 2 pattern verification at the block-build level."""
    cfg = _config_2b()
    blk0 = layer.build_gemma2_decoder_layer(cfg, layer_idx=0, max_seq=64)
    blk1 = layer.build_gemma2_decoder_layer(cfg, layer_idx=1, max_seq=64)
    assert blk0.spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert blk0.spec.token_mixer.sliding_window == 4096
    assert blk1.spec.token_mixer.mask_kind == types.MaskKind.CAUSAL
    assert blk1.spec.token_mixer.sliding_window is None
