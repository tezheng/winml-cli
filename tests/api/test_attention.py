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
