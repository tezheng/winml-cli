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
                               weight_mode=types.NormWeightMode.STANDARD_W)
    attn = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=64,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=128,
        qk_norm=norm_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        attn_scale=1.0,
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


def test_block_with_per_layer_embedding_at_end():
    """B0.6: Block applies the Gemma 4 PLE-at-END injection when given a
    `per_layer_input: [B, S, ple_dim]` tensor. Steps in the block:
        residual = x
        x = per_layer_input_gate(x)             # Linear hidden → ple_dim
        x = gelu_pytorch_tanh(x)
        x = x * per_layer_input
        x = per_layer_projection(x)             # Linear ple_dim → hidden
        x = post_per_layer_input_norm(x)        # RMSNorm
        x = residual + x
        x = x * layer_scalar                    # registered buffer of ones
    """
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    ple_spec = specs.PLESpec(
        ple_dim=8, residual_scale=1.0 / (2 ** 0.5),
        injection_norm=norm_spec,
    )
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
    D = 256
    blk = block.DecoderBlock(block_spec, hidden_size=D, max_seq=32,
                             dtype=torch.float32)
    # PLE plumbing should be present.
    assert blk.per_layer_input_gate is not None
    assert blk.per_layer_input_gate.in_features == D
    assert blk.per_layer_input_gate.out_features == ple_spec.ple_dim
    assert blk.per_layer_projection.in_features == ple_spec.ple_dim
    assert blk.per_layer_projection.out_features == D
    assert blk.post_per_layer_input_norm is not None
    # And layer_scalar is a registered buffer of ones.
    assert hasattr(blk, "layer_scalar")
    assert torch.allclose(blk.layer_scalar, torch.ones(1))

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=1, head_dim=32, max_seq=32,
    )
    B, S = 1, 3
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    ple_input = torch.randn(B, S, ple_spec.ple_dim)  # provided by the model assembly
    out_with_ple = blk(x, position_ids=pos, cache=cache, start_pos=0,
                       per_layer_input=ple_input)
    cache.reset()
    out_without_ple = blk(x, position_ids=pos, cache=cache, start_pos=0,
                          per_layer_input=None)
    # The PLE injection should change the output.
    assert not torch.allclose(out_with_ple, out_without_ple, atol=1e-3)


def test_block_layer_scalar_scales_output():
    """B0.6: changing `layer_scalar` from 1 to 2 must double the output."""
    spec = _qwen3_like_block_spec(hidden_size=128)
    blk = block.DecoderBlock(spec, hidden_size=128, max_seq=32, dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache_a = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=4, head_dim=32, max_seq=32,
    )
    cache_b = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=4, head_dim=32, max_seq=32,
    )
    B, S, D = 1, 3, 128
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out_a = blk(x, position_ids=pos, cache=cache_a, start_pos=0)
    blk.layer_scalar.fill_(2.0)
    out_b = blk(x, position_ids=pos, cache=cache_b, start_pos=0)
    assert torch.allclose(out_b, out_a * 2.0, atol=1e-5)
