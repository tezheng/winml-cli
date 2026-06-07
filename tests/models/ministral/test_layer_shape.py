"""Shape-only tests for the Ministral decoder layer.

Exercises both full and sliding layers built from a synthetic 1:3 config
(matching the published Ministral-8B-Instruct-2410 pattern). Verifies that
the per-layer dispatch sets the SWA / CAUSAL mask correctly and that the
sliding_window field flows through to the AttentionSpec.
"""
import pytest
import torch

from api import kvcache, specs, types
from models.ministral import config, layer


def _synthetic_cfg(num_hidden_layers=8, sliding_window=4, max_seq=64):
    # 1:3 pattern: full, sliding, sliding, sliding, ...
    layer_types = []
    for i in range(num_hidden_layers):
        layer_types.append(
            "full_attention" if i % 4 == 0 else "sliding_attention"
        )
    return config.MinistralConfig(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, intermediate_size=256, num_hidden_layers=num_hidden_layers,
        rope_theta=100_000_000.0, rms_norm_eps=1e-5,
        vocab_size=512, max_position_embeddings=max_seq,
        tie_word_embeddings=False, dtype=torch.float32,
        sliding_window=sliding_window,
        layer_types=tuple(layer_types),
    )


@pytest.mark.parametrize("layer_idx", [0, 1, 2, 3, 4, 7])
def test_ministral_decoder_layer_forward_shape(layer_idx):
    cfg = _synthetic_cfg()
    blk = layer.build_ministral_decoder_layer(cfg, layer_idx=layer_idx, max_seq=64)
    B, S = 2, 6
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


def test_ministral_full_vs_sliding_layer_attention_spec():
    cfg = _synthetic_cfg()
    full_blk = layer.build_ministral_decoder_layer(cfg, layer_idx=0, max_seq=64)
    slide_blk = layer.build_ministral_decoder_layer(cfg, layer_idx=1, max_seq=64)
    assert full_blk.attention.spec.mask_kind == types.MaskKind.CAUSAL
    assert full_blk.attention.spec.sliding_window is None
    assert slide_blk.attention.spec.mask_kind == types.MaskKind.SWA
    assert slide_blk.attention.spec.sliding_window == cfg.sliding_window


def test_ministral_sliding_window_actually_masks_at_small_window():
    """At sliding_window=4 and S=8, the sliding-layer output must differ
    from the full-layer output (when both use the same weights). This is the
    smoking-gun for the per-layer dispatch — without dispatch, the two would
    be identical at S=8 because causal_mask == swa_mask for small S only
    when S <= W."""
    cfg = _synthetic_cfg(sliding_window=4, max_seq=64)
    full_blk = layer.build_ministral_decoder_layer(cfg, layer_idx=0, max_seq=64)
    slide_blk = layer.build_ministral_decoder_layer(cfg, layer_idx=1, max_seq=64)
    # Copy weights from full to slide so the only difference is the mask.
    with torch.no_grad():
        slide_blk.load_state_dict(full_blk.state_dict())
    B, S = 1, 8     # S > sliding_window=4 -> mask divergence
    torch.manual_seed(0)
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )

    def run(blk):
        cache = kvcache.ContiguousKVCache(
            cache_spec, batch_size=B,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, max_seq=64,
        )
        return blk(x, position_ids=pos, cache=cache, start_pos=0)

    full_out = run(full_blk)
    slide_out = run(slide_blk)
    assert not torch.allclose(full_out, slide_out, atol=1e-4), (
        "Full and sliding outputs must differ when S > W. The per-layer "
        "dispatch isn't actually changing the mask."
    )


def test_ministral_full_equals_sliding_at_short_seq():
    """When S <= W, the SWA mask coincides with the causal mask, so a
    sliding layer with the same weights as a full layer produces an
    identical output. This guards the converse of the smoking-gun test:
    with S=4, W=4, the dispatch is observable only structurally, not
    numerically — and the layer should still be correct."""
    cfg = _synthetic_cfg(sliding_window=4, max_seq=64)
    full_blk = layer.build_ministral_decoder_layer(cfg, layer_idx=0, max_seq=64)
    slide_blk = layer.build_ministral_decoder_layer(cfg, layer_idx=1, max_seq=64)
    with torch.no_grad():
        slide_blk.load_state_dict(full_blk.state_dict())
    B, S = 1, 4      # S <= sliding_window=4 -> masks identical
    torch.manual_seed(1)
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )

    def run(blk):
        cache = kvcache.ContiguousKVCache(
            cache_spec, batch_size=B,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, max_seq=64,
        )
        return blk(x, position_ids=pos, cache=cache, start_pos=0)

    full_out = run(full_blk)
    slide_out = run(slide_blk)
    assert torch.allclose(full_out, slide_out, atol=1e-5, rtol=1e-5)
