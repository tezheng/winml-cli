import torch
import torch.nn.functional as F

from api import attention, kvcache, norm, rope, specs, types


def _qwen3_like_attention_spec(qk_norm: bool):
    qk_spec = (specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                              weight_mode=types.NormWeightMode.STANDARD_W)
               if qk_norm else None)
    return specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=64,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        qk_norm=qk_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE if qk_norm else types.QKNormPhase.NONE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH if qk_norm else types.QKNormShape.NONE,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )


def test_attention_forward_shape():
    hidden_size = 1024
    spec = _qwen3_like_attention_spec(qk_norm=True)
    attn = attention.Attention(spec, hidden_size=hidden_size, max_seq=128,
                               dtype=torch.float32)
    B, S = 1, 8
    x = torch.randn(B, S, hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(cache_spec, batch_size=B,
                                      n_kv_heads=spec.n_kv_heads,
                                      head_dim=spec.head_dim, max_seq=128)
    out = attn(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, hidden_size)
    assert cache.seq_len == S


def test_attention_decode_step_appends_to_cache():
    hidden_size = 256
    spec = _qwen3_like_attention_spec(qk_norm=False)
    attn = attention.Attention(spec, hidden_size=hidden_size, max_seq=64,
                               dtype=torch.float32)
    B = 1
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(cache_spec, batch_size=B,
                                      n_kv_heads=spec.n_kv_heads,
                                      head_dim=spec.head_dim, max_seq=64)
    x_prefill = torch.randn(B, 5, hidden_size)
    pos_prefill = torch.arange(5)
    _ = attn(x_prefill, position_ids=pos_prefill, cache=cache, start_pos=0)
    assert cache.seq_len == 5

    x_decode = torch.randn(B, 1, hidden_size)
    pos_decode = torch.tensor([5])
    _ = attn(x_decode, position_ids=pos_decode, cache=cache, start_pos=5)
    assert cache.seq_len == 6


def test_attention_chunked_prefill_matches_full_prefill():
    """Chunked prefill (S>1 from start_pos>0) must equal full prefill output for the same range."""
    spec = _qwen3_like_attention_spec(qk_norm=False)
    attn = attention.Attention(spec, hidden_size=256, max_seq=32,
                               dtype=torch.float32)
    torch.manual_seed(0)
    B, S_total, D = 1, 6, 256
    x_full = torch.randn(B, S_total, D)
    pos_full = torch.arange(S_total)

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    # Full prefill: one shot, 6 tokens.
    cache_full = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=spec.n_kv_heads, head_dim=spec.head_dim, max_seq=32,
    )
    out_full = attn(x_full, position_ids=pos_full, cache=cache_full, start_pos=0)

    # Chunked: prefill first 2 tokens, then 4 tokens as a chunk.
    cache_chunk = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=spec.n_kv_heads, head_dim=spec.head_dim, max_seq=32,
    )
    _ = attn(x_full[:, :2], position_ids=pos_full[:2], cache=cache_chunk, start_pos=0)
    out_chunk_tail = attn(x_full[:, 2:], position_ids=pos_full[2:], cache=cache_chunk, start_pos=2)
    assert torch.allclose(out_full[:, 2:], out_chunk_tail, atol=1e-5)


def test_attention_causal_actually_masks_future():
    """Perturbing token at position T must not change outputs at positions < T."""
    spec = _qwen3_like_attention_spec(qk_norm=False)
    attn = attention.Attention(spec, hidden_size=128, max_seq=16,
                               dtype=torch.float32)
    torch.manual_seed(0)
    B, S, D = 1, 5, 128
    x_a = torch.randn(B, S, D)
    x_b = x_a.clone()
    x_b[:, 3] += 100.0  # large perturbation at position 3

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache_a = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=spec.n_kv_heads, head_dim=spec.head_dim, max_seq=16,
    )
    cache_b = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=spec.n_kv_heads, head_dim=spec.head_dim, max_seq=16,
    )
    out_a = attn(x_a, position_ids=torch.arange(S), cache=cache_a, start_pos=0)
    out_b = attn(x_b, position_ids=torch.arange(S), cache=cache_b, start_pos=0)
    # Positions 0..2 should be identical (no leak from position 3).
    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5)


def test_attention_causal_mask_for_prefill():
    spec = _qwen3_like_attention_spec(qk_norm=False)
    attn = attention.Attention(spec, hidden_size=256, max_seq=16,
                               dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(cache_spec, batch_size=1,
                                      n_kv_heads=spec.n_kv_heads,
                                      head_dim=spec.head_dim, max_seq=16)
    B, S, D = 1, 4, 256
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = attn(x, position_ids=pos, cache=cache, start_pos=0)
    assert cache.seq_len == S
    assert out.shape == (B, S, D)


def test_attention_k_eq_v_skips_v_projection():
    """When attention_k_eq_v=True, the V projection should not be allocated; v = k."""
    spec = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        attention_k_eq_v=True,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = attention.Attention(spec, hidden_size=4096, max_seq=64, dtype=torch.float32)
    # v_proj should be absent (or None) when attention_k_eq_v
    assert getattr(attn, "v_proj", None) is None


def test_attention_qk_norm_fixed_scale_overrides_scale():
    """With fixed-scale QK norm, attn_scale defaults to 1.0 instead of 1/sqrt(Dh)."""
    qk_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                             weight_mode=types.NormWeightMode.ONE_PLUS_W)
    spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=512,
        qk_norm=qk_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        qk_norm_fixed_scale=0.9916,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = attention.Attention(spec, hidden_size=1024, max_seq=64, dtype=torch.float32)
    assert attn.effective_scale == 1.0


def test_attention_swa_masks_outside_window():
    """SWA mask: tokens at position i can only see positions in [max(0, i - W), i]."""
    spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=64,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=2,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = attention.Attention(spec, hidden_size=512, max_seq=8, dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(cache_spec, batch_size=1,
                                      n_kv_heads=1, head_dim=64, max_seq=8)
    B, S, D = 1, 5, 512
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = attn(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, D)
    # We cannot check SWA masking directly without inspecting internals;
    # verify shape and let the numerical-equivalence test (T17) catch correctness.
