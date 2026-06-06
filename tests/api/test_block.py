import torch

from api import block, kvcache, specs, types


def _qwen3_like_block_spec(hidden_size: int = 256):
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    qk_spec = norm_spec
    attn_spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=4, head_dim=32,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        qk_norm=qk_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn_spec = specs.FFNSpec(intermediate_size=512,
                             activation=types.Activation.SILU,
                             gate_kind=types.GateKind.SWIGLU)
    return specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn_spec,
        channel_mixer=ffn_spec,
        pre_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec,
    )


def test_block_forward_preserves_residual_shape():
    spec = _qwen3_like_block_spec(hidden_size=256)
    blk = block.DecoderBlock(spec, hidden_size=256, max_seq=64, dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=spec.token_mixer.n_kv_heads,
        head_dim=spec.token_mixer.head_dim, max_seq=64,
    )
    B, S, D = 1, 5, 256
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, D)
    assert cache.seq_len == S


def test_block_residual_actually_adds():
    spec = _qwen3_like_block_spec(hidden_size=128)
    blk = block.DecoderBlock(spec, hidden_size=128, max_seq=32, dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=spec.token_mixer.n_kv_heads,
        head_dim=spec.token_mixer.head_dim, max_seq=32,
    )
    B, S, D = 1, 3, 128
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert (out - x).abs().mean() > 0
    assert out.abs().mean() < 100 * x.abs().mean()


def test_block_sandwich_norm_runs():
    """Gemma 4 sandwich: PRE + POST norms around each sublayer."""
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.ONE_PLUS_W)
    attn = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=64,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=128,
        qk_norm=norm_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        qk_norm_fixed_scale=0.9916,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=512,
                       activation=types.Activation.GELU,
                       gate_kind=types.GateKind.GEGLU)
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE_AND_POST,
        ffn_norm_position=types.NormPosition.PRE_AND_POST,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
    )
    blk = block.DecoderBlock(block_spec, hidden_size=512, max_seq=64,
                             dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=1, head_dim=64, max_seq=64,
    )
    B, S, D = 1, 4, 512
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, D)


def test_block_with_per_layer_embedding_residual():
    """Block adds a PLE residual at the very end of the block (or as configured)."""
    # Build a minimal block with PLESpec
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.ONE_PLUS_W)
    ple_spec = specs.PLESpec(
        ple_dim=8, residual_scale=1.0 / (2 ** 0.5),
        injection_norm=norm_spec,
    )
    # Block construction — only used here for spec assembly; the PLE injection
    # is *called from outside* the DecoderBlock (because PLE depends on input_ids,
    # which the block does not know about). Instead the model assembly passes
    # the per-layer PLE residual into block.forward as an optional argument.
    attn = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=32,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=64,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=64,
                       activation=types.Activation.GELU,
                       gate_kind=types.GateKind.GEGLU)
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE_AND_POST,
        ffn_norm_position=types.NormPosition.PRE_AND_POST,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
        per_layer_embedding=ple_spec,
    )
    blk = block.DecoderBlock(block_spec, hidden_size=256, max_seq=32,
                             dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=1, head_dim=32, max_seq=32,
    )
    B, S, D = 1, 3, 256
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    ple_residual = torch.randn(B, S, D)  # provided by the model assembly
    out_with_ple = blk(x, position_ids=pos, cache=cache, start_pos=0,
                       per_layer_residual=ple_residual)
    cache.reset()
    out_without_ple = blk(x, position_ids=pos, cache=cache, start_pos=0)
    # The PLE residual should change the output
    assert not torch.allclose(out_with_ple, out_without_ple, atol=1e-3)
