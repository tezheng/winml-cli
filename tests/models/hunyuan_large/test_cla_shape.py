"""Hunyuan-Large CLA shape tests (v5-phase2 V4).

CLA (Cross-Layer Attention) is a per-layer KV-sharing pattern:
- Even-indexed layers own K/V (k_proj, v_proj exist).
- Odd-indexed layers borrow K/V from the previous layer
  (k_proj=v_proj=None, `kv_shared` arg required at forward).

This test validates the SHAPE of the IR extension; an end-to-end
numerical gate against the Hunyuan-Large checkpoint is out of scope
(no HF v5.10.2 reference module implements CLA).
"""
from __future__ import annotations

import pytest
import torch

from api import attention as api_attention, block as api_block, kvcache, specs, types
from models.hunyuan_large import config as hl_config


def _tiny_cfg() -> hl_config.HunyuanLargeConfig:
    return hl_config.HunyuanLargeConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=4,
        max_position_embeddings=32, cla_share_factor=2,
    )


def test_owner_layer_has_kv_projections():
    cfg = _tiny_cfg()
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.token_mixer.kv_source_layer_offset is None
    blk = api_block.DecoderBlock(spec=spec, hidden_size=64, max_seq=32)
    assert blk.attention.q_proj is not None
    assert blk.attention.k_proj is not None
    assert blk.attention.v_proj is not None


def test_borrower_layer_has_no_kv_projections():
    cfg = _tiny_cfg()
    spec = cfg.to_block_spec(layer_idx=1)
    assert spec.token_mixer.kv_source_layer_offset == -1
    blk = api_block.DecoderBlock(spec=spec, hidden_size=64, max_seq=32)
    assert blk.attention.q_proj is not None      # Q always built
    assert blk.attention.k_proj is None
    assert blk.attention.v_proj is None


def test_cla_pattern_across_layers():
    """For cla_share_factor=2, layers 0, 2, 4, ... own; 1, 3, 5, ... borrow."""
    cfg = _tiny_cfg()
    for L in range(cfg.num_hidden_layers):
        spec = cfg.to_block_spec(layer_idx=L)
        if L % 2 == 0:
            assert spec.token_mixer.kv_source_layer_offset is None, (
                f"layer {L} should be owner"
            )
        else:
            assert spec.token_mixer.kv_source_layer_offset == -1, (
                f"layer {L} should be borrower (offset=-1)"
            )


def test_borrower_forward_requires_kv_shared():
    """A borrower layer's forward must raise if kv_shared not provided."""
    cfg = _tiny_cfg()
    spec = cfg.to_block_spec(layer_idx=1)
    blk = api_block.DecoderBlock(spec=spec, hidden_size=64, max_seq=32)
    blk.eval()
    cache = kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=1, n_kv_heads=2, head_dim=16, max_seq=32,
    )
    x = torch.randn(1, 4, 64)
    pos = torch.arange(4)
    with pytest.raises(ValueError, match="kv_shared"):
        blk(x, position_ids=pos, cache=cache, start_pos=0)


def test_borrower_forward_uses_kv_shared():
    """Pass kv_shared explicitly — forward must run end-to-end and
    produce output of the expected shape."""
    cfg = _tiny_cfg()
    spec_o = cfg.to_block_spec(layer_idx=0)
    spec_b = cfg.to_block_spec(layer_idx=1)
    owner = api_block.DecoderBlock(spec_o, hidden_size=64, max_seq=32)
    borrower = api_block.DecoderBlock(spec_b, hidden_size=64, max_seq=32)
    owner.eval(); borrower.eval()

    B, S = 1, 4
    x = torch.randn(B, S, 64)
    pos = torch.arange(S)
    cache_o = kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=B, n_kv_heads=2, head_dim=16, max_seq=32,
    )
    cache_b = kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=B, n_kv_heads=2, head_dim=16, max_seq=32,
    )

    # Run owner; pull its written K/V from the cache.
    with torch.no_grad():
        out_o = owner(x, position_ids=pos, cache=cache_o, start_pos=0)
    k_shared, v_shared = cache_o.read(seq_len=S)

    # Run borrower with kv_shared.
    with torch.no_grad():
        out_b = borrower(out_o, position_ids=pos, cache=cache_b,
                         start_pos=0, kv_shared=(k_shared, v_shared))
    assert out_b.shape == out_o.shape


def test_attention_init_skips_kv_proj_on_cla_borrower():
    """Direct Attention init with kv_source_layer_offset=-1 must skip
    k_proj/v_proj allocation."""
    spec = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=2, head_dim=16,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
        kv_source_layer_offset=-1,
    )
    attn = api_attention.Attention(spec, hidden_size=64, max_seq=32)
    assert attn.q_proj is not None
    assert attn.k_proj is None
    assert attn.v_proj is None
    # Param count: only q_proj + o_proj (no k/v)
    n_params = sum(p.numel() for p in attn.parameters())
    # q_proj: 64 * (4*16) = 4096; o_proj: (4*16) * 64 = 4096; sum 8192
    assert n_params == 8192, (
        f"borrower attention should have only q+o projections, got {n_params}"
    )
