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


def test_kvcache_asymmetric_v_head_dim():
    """B2b: MLA stores K at qk_head_dim and V at v_head_dim — asymmetric.

    MiniCPM-3 ref path: K is [B, H, S, qk_head_dim=96], V is [B, H, S, v_head_dim=64].
    Source: modeling_minicpm.py:457 (kv view uses qk_nope_head_dim + v_head_dim),
    481-483 (key_states assembled at q_head_dim=96), 516 (attn_output uses v_head_dim).
    """
    spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        spec, batch_size=1, n_kv_heads=2,
        head_dim=96, max_seq=8, v_head_dim=64,
    )
    assert cache.k.shape == (1, 2, 8, 96)
    assert cache.v.shape == (1, 2, 8, 64)
    k = torch.randn(1, 2, 3, 96); v = torch.randn(1, 2, 3, 64)
    cache.write(k, v, start_pos=0)
    k_out, v_out = cache.read(seq_len=3)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)


def test_kvcache_default_v_head_dim_matches_head_dim():
    """Default v_head_dim == head_dim — backward-compat."""
    spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(spec, batch_size=1, n_kv_heads=2,
                                      head_dim=8, max_seq=4)
    assert cache.v_head_dim == 8
    assert cache.k.shape == cache.v.shape


def test_ssm_state_cache_init_shapes():
    """B7: SSMStateCache holds conv_state and ssm_state of correct shapes."""
    cache = kvcache.SSMStateCache(
        batch_size=2, conv_dim=64, conv_kernel=4,
        n_heads=8, head_dim=16, d_state=32,
    )
    assert cache.conv_state.shape == (2, 64, 4)
    assert cache.ssm_state.shape == (2, 8, 16, 32)
    assert cache.has_previous_state is False
    assert cache.conv_state.sum().item() == 0.0
    assert cache.ssm_state.sum().item() == 0.0


def test_ssm_state_cache_update_conv_state_marks_has_previous():
    cache = kvcache.SSMStateCache(
        batch_size=1, conv_dim=8, conv_kernel=4,
        n_heads=2, head_dim=4, d_state=4,
    )
    new = torch.randn(1, 8, 4)
    cache.update_conv_state(new)
    assert cache.has_previous_state
    assert torch.allclose(cache.conv_state, new)


def test_ssm_state_cache_update_recurrent_state_marks_has_previous():
    cache = kvcache.SSMStateCache(
        batch_size=1, conv_dim=8, conv_kernel=4,
        n_heads=2, head_dim=4, d_state=4,
    )
    new = torch.randn(1, 2, 4, 4)
    cache.update_recurrent_state(new)
    assert cache.has_previous_state
    assert torch.allclose(cache.ssm_state, new)


def test_ssm_state_cache_update_rejects_shape_mismatch():
    cache = kvcache.SSMStateCache(
        batch_size=1, conv_dim=8, conv_kernel=4,
        n_heads=2, head_dim=4, d_state=4,
    )
    with pytest.raises(ValueError, match="conv_state shape mismatch"):
        cache.update_conv_state(torch.randn(1, 8, 5))
    with pytest.raises(ValueError, match="ssm_state shape mismatch"):
        cache.update_recurrent_state(torch.randn(1, 2, 4, 5))


def test_ssm_state_cache_reset_clears_state():
    cache = kvcache.SSMStateCache(
        batch_size=1, conv_dim=4, conv_kernel=2,
        n_heads=1, head_dim=2, d_state=2,
    )
    cache.update_conv_state(torch.ones_like(cache.conv_state))
    cache.update_recurrent_state(torch.ones_like(cache.ssm_state))
    assert cache.has_previous_state
    cache.reset()
    assert not cache.has_previous_state
    assert cache.conv_state.sum().item() == 0.0
    assert cache.ssm_state.sum().item() == 0.0
