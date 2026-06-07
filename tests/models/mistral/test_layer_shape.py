import pytest
import torch

from api import kvcache, specs, types
from models.mistral import config, layer


def _config_7b_v03():
    """Mistral 7B v0.3 — sliding_window=None."""
    return config.MistralConfig(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=14336, num_hidden_layers=32,
        rope_theta=1_000_000.0, rms_norm_eps=1e-5,
        vocab_size=32768, max_position_embeddings=32768,
        tie_word_embeddings=False, dtype=torch.float32,
        sliding_window=None,
    )


def _config_7b_v01():
    """Mistral 7B v0.1 — sliding_window=4096, theta=10000."""
    return config.MistralConfig(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=14336, num_hidden_layers=32,
        rope_theta=10000.0, rms_norm_eps=1e-5,
        vocab_size=32000, max_position_embeddings=32768,
        tie_word_embeddings=False, dtype=torch.float32,
        sliding_window=4096,
    )


@pytest.mark.parametrize("cfg_fn", [_config_7b_v03, _config_7b_v01])
def test_mistral_decoder_layer_forward_shape(cfg_fn):
    cfg = cfg_fn()
    blk = layer.build_mistral_decoder_layer(cfg, layer_idx=0, max_seq=128)
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
        head_dim=cfg.head_dim, max_seq=128,
    )
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_mistral_decoder_layer_no_qk_norm():
    cfg = _config_7b_v03()
    blk = layer.build_mistral_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.q_norm is None
    assert blk.attention.k_norm is None


def test_mistral_v01_swa_mask_path():
    """v0.1 uses SWA mask — verify the spec is wired through."""
    cfg = _config_7b_v01()
    blk = layer.build_mistral_decoder_layer(cfg, layer_idx=0, max_seq=64)
    from api import types as _t
    assert blk.spec.token_mixer.mask_kind == _t.MaskKind.SWA
    assert blk.spec.token_mixer.sliding_window == 4096
