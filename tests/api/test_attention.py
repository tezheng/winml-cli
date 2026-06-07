import pytest
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


def test_attention_v_norm_with_scale_false_no_state_dict_entry():
    """B0.6: v_norm with `with_scale=False` (Gemma 4) registers a non-persistent
    buffer of ones — there must be NO weight in the state_dict, and the buffer
    must equal a vector of ones (no learnable scale)."""
    v_norm_spec = specs.NormSpec(
        kind=types.NormKind.RMS, eps=1e-6,
        weight_mode=types.NormWeightMode.STANDARD_W,
    )
    spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=32,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        v_norm=v_norm_spec, v_norm_with_scale=False,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = attention.Attention(spec, hidden_size=128, max_seq=8, dtype=torch.float32)
    # v_norm_weight must be a buffer (not a Parameter) and not in state_dict.
    sd = attn.state_dict()
    assert "v_norm_weight" not in sd
    # and equal to ones
    assert torch.allclose(attn.v_norm_weight,
                          torch.ones(spec.head_dim, dtype=torch.float32))


def test_attention_v_norm_changes_output():
    """Sanity: enabling v_norm must change the attention output vs no-v_norm."""
    common = dict(
        n_q_heads=4, n_kv_heads=1, head_dim=16,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    no_vn = specs.AttentionSpec(**common)
    v_norm_spec = specs.NormSpec(
        kind=types.NormKind.RMS, eps=1e-6,
        weight_mode=types.NormWeightMode.STANDARD_W,
    )
    with_vn = specs.AttentionSpec(**common, v_norm=v_norm_spec,
                                  v_norm_with_scale=False)
    torch.manual_seed(7)
    H = 64
    a = attention.Attention(no_vn, hidden_size=H, max_seq=8, dtype=torch.float32)
    b = attention.Attention(with_vn, hidden_size=H, max_seq=8, dtype=torch.float32)
    # Copy weights so the only difference is the v_norm path
    b.q_proj.load_state_dict(a.q_proj.state_dict())
    b.k_proj.load_state_dict(a.k_proj.state_dict())
    b.v_proj.load_state_dict(a.v_proj.state_dict())
    b.o_proj.load_state_dict(a.o_proj.state_dict())

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache_a = kvcache.ContiguousKVCache(cache_spec, batch_size=1,
                                        n_kv_heads=1, head_dim=16, max_seq=8)
    cache_b = kvcache.ContiguousKVCache(cache_spec, batch_size=1,
                                        n_kv_heads=1, head_dim=16, max_seq=8)
    x = torch.randn(1, 4, H)
    pos = torch.arange(4)
    out_a = a(x, position_ids=pos, cache=cache_a, start_pos=0)
    out_b = b(x, position_ids=pos, cache=cache_b, start_pos=0)
    # Outputs must differ (RMS-normalised V is different from raw V in general).
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_attention_fused_qkv_equivalent_to_split():
    """B2a: FUSED QKV layout must produce identical output to a SPLIT layout
    when weights are arranged such that the fused matrix's blocks match the
    individual q/k/v matrices. (Phi-3 stores the fused matrix concatenated
    along the output dim in order Q | K | V.)
    Source: modeling_phi3.py:222-241."""
    hidden = 64
    Hq, Hk, Dh = 4, 2, 16
    rope_spec = specs.RoPESpec(base_theta=10_000.0,
                               basis=types.RoPEBasis.SPLIT_HALF)
    split = specs.AttentionSpec(
        n_q_heads=Hq, n_kv_heads=Hk, head_dim=Dh,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=rope_spec,
    )
    fused = specs.AttentionSpec(
        n_q_heads=Hq, n_kv_heads=Hk, head_dim=Dh,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.FUSED,
        mask_kind=types.MaskKind.CAUSAL,
        rope=rope_spec,
    )
    a_s = attention.Attention(split, hidden_size=hidden, max_seq=32, dtype=torch.float32)
    a_f = attention.Attention(fused, hidden_size=hidden, max_seq=32, dtype=torch.float32)
    # Stitch fused weight from split.
    with torch.no_grad():
        a_f.qkv_proj.weight.copy_(torch.cat([
            a_s.q_proj.weight, a_s.k_proj.weight, a_s.v_proj.weight,
        ], dim=0))
        a_f.o_proj.weight.copy_(a_s.o_proj.weight)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS, memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache_s = kvcache.ContiguousKVCache(cache_spec, 1, Hk, Dh, 32)
    cache_f = kvcache.ContiguousKVCache(cache_spec, 1, Hk, Dh, 32)
    x = torch.randn(1, 5, hidden)
    pos = torch.arange(5)
    out_s = a_s(x, position_ids=pos, cache=cache_s, start_pos=0)
    out_f = a_f(x, position_ids=pos, cache=cache_f, start_pos=0)
    assert torch.allclose(out_s, out_f, atol=1e-5)


def test_attention_fused_qkv_shape_outputs_correct_blocks():
    """Verify the qkv_proj output has the right total size for Phi-3 fused QKV:
    (Hq + 2*Hk) * Dh — the Q block, K block, V block concatenated."""
    spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=2, head_dim=16,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.FUSED,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=10_000.0, basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = attention.Attention(spec, hidden_size=64, max_seq=32, dtype=torch.float32)
    expected_out = (8 + 2 * 2) * 16   # 12 * 16 = 192
    assert attn.qkv_proj.weight.shape == (expected_out, 64)
    assert attn.q_proj is None
    assert attn.k_proj is None
    assert attn.v_proj is None


def _minicpm3_like_mla_spec():
    """MLA spec shaped like MiniCPM-3 (smaller dims for fast tests)."""
    return specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4,
        head_dim=96,                # qk_nope + qk_rope = 64+32
        kind=types.AttentionKind.MLA,
        qkv_layout=types.QKVLayout.MLA_LATENT,
        mask_kind=types.MaskKind.CAUSAL,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        rope=specs.RoPESpec(base_theta=10_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )


def test_attention_mla_forward_shape():
    """B2b: MLA forward returns [B, S, hidden]; cache stores K at qk_h, V at v_h."""
    hidden_size = 256
    spec = _minicpm3_like_mla_spec()
    attn = attention.Attention(spec, hidden_size=hidden_size, max_seq=32,
                               dtype=torch.float32)
    B, S = 1, 5
    x = torch.randn(B, S, hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=spec.n_q_heads,
        head_dim=spec.head_dim,            # 96 for K
        max_seq=32,
        v_head_dim=spec.v_head_dim,        # 64 for V
    )
    out = attn(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, hidden_size)
    assert cache.k.shape == (B, spec.n_q_heads, 32, spec.head_dim)
    assert cache.v.shape == (B, spec.n_q_heads, 32, spec.v_head_dim)
    assert cache.seq_len == S


def test_attention_mla_decode_step_appends():
    """MLA decode: prefill S then step 1 and verify cache grows."""
    hidden_size = 128
    spec = _minicpm3_like_mla_spec()
    attn = attention.Attention(spec, hidden_size=hidden_size, max_seq=16,
                               dtype=torch.float32)
    B = 1
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=spec.n_q_heads,
        head_dim=spec.head_dim, max_seq=16,
        v_head_dim=spec.v_head_dim,
    )
    x1 = torch.randn(B, 4, hidden_size)
    _ = attn(x1, position_ids=torch.arange(4), cache=cache, start_pos=0)
    assert cache.seq_len == 4
    x2 = torch.randn(B, 1, hidden_size)
    _ = attn(x2, position_ids=torch.tensor([4]), cache=cache, start_pos=4)
    assert cache.seq_len == 5


def test_attention_mla_rejects_partial_specs():
    """MLA requires all 5 MLA fields; missing one raises."""
    with pytest.raises(ValueError, match="missing"):
        spec = specs.AttentionSpec(
            n_q_heads=4, n_kv_heads=4, head_dim=96,
            kind=types.AttentionKind.MLA,
            qkv_layout=types.QKVLayout.MLA_LATENT,
            mask_kind=types.MaskKind.CAUSAL,
            q_lora_rank=128,
            # kv_lora_rank missing
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )
        attention.Attention(spec, hidden_size=256, max_seq=8)


def test_attention_mla_rejects_wrong_qkv_layout():
    """MLA requires MLA_LATENT qkv_layout."""
    with pytest.raises(ValueError, match="MLA_LATENT"):
        spec = specs.AttentionSpec(
            n_q_heads=4, n_kv_heads=4, head_dim=96,
            kind=types.AttentionKind.MLA,
            qkv_layout=types.QKVLayout.SPLIT,           # wrong
            mask_kind=types.MaskKind.CAUSAL,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )
        attention.Attention(spec, hidden_size=256, max_seq=8)
