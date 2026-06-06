import pytest
import torch

from api import kvcache, specs, types


def _make_spec():
    return specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32,
        v_dtype=torch.float32,
        ownership=types.CacheOwnership.EXPLICIT_PASS,
    )


def test_kvcache_init_shapes():
    spec = _make_spec()
    cache = kvcache.ContiguousKVCache(spec, batch_size=2, n_kv_heads=4,
                                      head_dim=8, max_seq=16)
    assert cache.k.shape == (2, 4, 16, 8)
    assert cache.v.shape == (2, 4, 16, 8)
    assert cache.seq_len == 0


def test_kvcache_prefill_write_and_read_back():
    spec = _make_spec()
    cache = kvcache.ContiguousKVCache(spec, batch_size=1, n_kv_heads=2,
                                      head_dim=4, max_seq=8)
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    cache.write(k, v, start_pos=0)
    k_out, v_out = cache.read(seq_len=5)
    assert k_out.shape == (1, 2, 5, 4)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)
    assert cache.seq_len == 5


def test_kvcache_decode_appends():
    spec = _make_spec()
    cache = kvcache.ContiguousKVCache(spec, batch_size=1, n_kv_heads=2,
                                      head_dim=4, max_seq=8)
    k1 = torch.randn(1, 2, 5, 4); v1 = torch.randn(1, 2, 5, 4)
    cache.write(k1, v1, start_pos=0)
    k2 = torch.randn(1, 2, 1, 4); v2 = torch.randn(1, 2, 1, 4)
    cache.write(k2, v2, start_pos=5)
    k_out, v_out = cache.read(seq_len=6)
    assert k_out.shape == (1, 2, 6, 4)
    assert torch.allclose(k_out[:, :, :5], k1)
    assert torch.allclose(k_out[:, :, 5:6], k2)
    assert cache.seq_len == 6


def test_kvcache_write_rejects_dtype_mismatch():
    spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
        ownership=types.CacheOwnership.EXPLICIT_PASS,
    )
    cache = kvcache.ContiguousKVCache(spec, batch_size=1, n_kv_heads=2,
                                      head_dim=4, max_seq=8)
    k_bad = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
    v_ok = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="k dtype mismatch"):
        cache.write(k_bad, v_ok, start_pos=0)


def test_kvcache_overflow_raises():
    spec = _make_spec()
    cache = kvcache.ContiguousKVCache(spec, batch_size=1, n_kv_heads=2,
                                      head_dim=4, max_seq=4)
    k = torch.randn(1, 2, 5, 4); v = torch.randn(1, 2, 5, 4)
    with pytest.raises(ValueError, match="exceeds max_seq"):
        cache.write(k, v, start_pos=0)


def test_shared_layer_kv_cache_aliases_source():
    """Gemma 4 E2B: layer i with i in shared range reads K/V from layer kv_source_idx[i]."""
    spec_base = _make_spec()
    source = kvcache.ContiguousKVCache(spec_base, batch_size=1, n_kv_heads=2,
                                       head_dim=4, max_seq=8)
    k = torch.randn(1, 2, 5, 4); v = torch.randn(1, 2, 5, 4)
    source.write(k, v, start_pos=0)

    shared = kvcache.SharedLayerKVCache(source_cache=source)
    k_out, v_out = shared.read(seq_len=5)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)
    assert shared.seq_len == 5


def test_shared_layer_kv_cache_write_is_noop():
    """Writes to a shared cache do NOT modify the source — the writing layer's K/V
    is computed but discarded (Gemma 4 shared-layer behavior: just consume the source)."""
    spec_base = _make_spec()
    source = kvcache.ContiguousKVCache(spec_base, batch_size=1, n_kv_heads=2,
                                       head_dim=4, max_seq=8)
    k_src = torch.randn(1, 2, 3, 4); v_src = torch.randn(1, 2, 3, 4)
    source.write(k_src, v_src, start_pos=0)

    shared = kvcache.SharedLayerKVCache(source_cache=source)
    k_new = torch.randn(1, 2, 1, 4); v_new = torch.randn(1, 2, 1, 4)
    shared.write(k_new, v_new, start_pos=3)  # should be no-op

    k_out, v_out = source.read(seq_len=3)
    assert torch.allclose(k_out, k_src)  # source unchanged
    assert torch.allclose(v_out, v_src)
