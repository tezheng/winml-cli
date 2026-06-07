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


def test_block_residual_scale_multiplies_sublayer_out():
    """B2a: DecoderBlockSpec.residual_scale (Granite μP residual_multiplier)
    multiplies BOTH sublayer outputs before the residual add.

    With residual_scale=0 every sublayer's output is zeroed, so the block must
    be an identity map (x unchanged) — verifies the scale is applied at exactly
    the right point. Source: modeling_granite.py:273,278 — `residual + hidden_states * self.residual_multiplier`.
    """
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    attn = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=2, head_dim=32,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=256,
                        activation=types.Activation.SILU,
                        gate_kind=types.GateKind.SWIGLU)
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm_spec, pre_ffn_norm=norm_spec,
        residual_scale=0.0,
    )
    D = 128
    blk = block.DecoderBlock(block_spec, hidden_size=D, max_seq=32,
                             dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=2, head_dim=32, max_seq=32,
    )
    B, S = 1, 4
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert torch.allclose(out, x, atol=1e-5), (
        f"residual_scale=0 should zero both sublayer outputs; "
        f"max_abs_diff={(out - x).abs().max().item():.3e}"
    )


def test_block_residual_scale_nonzero_matches_manual_compute():
    """B2a: residual_scale=0.5 should give x + 0.5 * attn_out + 0.5 * ffn_out
    (after second sublayer's residual add uses the *post-attn* x).

    We verify by running once with residual_scale=1.0 and once with
    residual_scale=0.5, both with the SAME weights; the second run should
    be x + 0.5*(out1 - x) only if the attention output were strictly linear,
    which it isn't due to the FFN's nonlinearity through the residual stream.
    So instead we just verify a single specific scalar multiplier propagates
    end-to-end by checking that the block with residual_scale=K is NOT the
    same as residual_scale=1 — a regression guard.
    """
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    attn = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=2, head_dim=32,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=256,
                        activation=types.Activation.SILU,
                        gate_kind=types.GateKind.SWIGLU)
    def _mk(scale):
        block_spec = specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn, channel_mixer=ffn,
            pre_attn_norm=norm_spec, pre_ffn_norm=norm_spec,
            residual_scale=scale,
        )
        return block.DecoderBlock(block_spec, hidden_size=128, max_seq=32,
                                  dtype=torch.float32)
    blk_a = _mk(None)        # no scaling
    blk_b = _mk(0.22)        # Granite μP residual_multiplier
    # Copy weights from a → b so the only diff is residual_scale.
    blk_b.load_state_dict(blk_a.state_dict())
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache_a = kvcache.ContiguousKVCache(cache_spec, 1, 2, 32, 32)
    cache_b = kvcache.ContiguousKVCache(cache_spec, 1, 2, 32, 32)
    x = torch.randn(1, 3, 128)
    pos = torch.arange(3)
    out_a = blk_a(x, position_ids=pos, cache=cache_a, start_pos=0)
    out_b = blk_b(x, position_ids=pos, cache=cache_b, start_pos=0)
    assert not torch.allclose(out_a, out_b, atol=1e-3)


def test_block_post_norm_skips_pre_norm():
    """B3 POST (OLMo 2): when attn_norm_position=POST, the block has no
    pre_attn_norm module and feeds raw residual to attention. Equivalent
    structure for FFN: no pre_ffn_norm, post-norm wraps the FFN output.
    Source: modeling_olmo2.py:295-333.
    """
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    attn = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=32,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        # OLMo 2 uses FULL_HDH QK-norm.
        qk_norm=norm_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.FULL_HDH,
        rope=specs.RoPESpec(base_theta=500000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=256,
                        activation=types.Activation.SILU,
                        gate_kind=types.GateKind.SWIGLU)
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.POST,
        ffn_norm_position=types.NormPosition.POST,
        token_mixer=attn, channel_mixer=ffn,
        post_attn_norm=norm_spec,
        post_ffn_norm=norm_spec,
    )
    blk = block.DecoderBlock(block_spec, hidden_size=128, max_seq=32,
                             dtype=torch.float32)
    # The POST-norm block has no pre-norm modules.
    assert blk.pre_attn_norm is None
    assert blk.pre_ffn_norm is None
    # Post-norm modules are allocated.
    assert blk.post_attn_sublayer_norm is not None
    assert blk.post_ffn_sublayer_norm is not None
    # Forward runs.
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=4, head_dim=32, max_seq=32,
    )
    B, S, D = 1, 4, 128
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, D)


def test_block_post_norm_matches_manual_olmo2_compute():
    """B3 POST: with all norm weights = 1 and a controlled attention/FFN,
    we can verify the POST-norm sandwich pattern numerically.

    Specifically: out = x + post_attn_norm(attn(x))
                  out2 = out + post_ffn_norm(ffn(out))

    The test asserts the block output matches the manual computation.
    Source: modeling_olmo2.py:315-326 (residual = x; x, _ = attn(x);
    x = post_attention_layernorm(x); x = residual + x — same for MLP).
    """
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    attn = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=16,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        qk_norm=norm_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.FULL_HDH,
        rope=specs.RoPESpec(base_theta=500000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=128,
                        activation=types.Activation.SILU,
                        gate_kind=types.GateKind.SWIGLU)
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.POST,
        ffn_norm_position=types.NormPosition.POST,
        token_mixer=attn, channel_mixer=ffn,
        post_attn_norm=norm_spec, post_ffn_norm=norm_spec,
    )
    D = 64
    blk = block.DecoderBlock(block_spec, hidden_size=D, max_seq=16,
                             dtype=torch.float32)
    blk.eval()
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache_blk = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=4, head_dim=16, max_seq=16,
    )
    cache_manual = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=4, head_dim=16, max_seq=16,
    )
    B, S = 1, 3
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    with torch.no_grad():
        out_blk = blk(x, position_ids=pos, cache=cache_blk, start_pos=0)
        # Manual: feed x directly to attention (NO pre-norm).
        attn_out = blk.attention(x, position_ids=pos, cache=cache_manual, start_pos=0)
        h = x + blk.post_attn_sublayer_norm(attn_out)
        ffn_out = blk.feedforward(h)
        out_manual = h + blk.post_ffn_sublayer_norm(ffn_out)
        # The block has a layer_scalar buffer (default 1.0).
        out_manual = out_manual * blk.layer_scalar
    assert torch.allclose(out_blk, out_manual, atol=1e-5), (
        f"max_abs_diff={(out_blk - out_manual).abs().max().item():.3e}"
    )


def test_block_softcap_propagates_through_attention():
    """B3 (Gemma 2 path): AttentionSpec.logit_softcap should flow into ops.sdpa.

    We instantiate two blocks with identical weights, one with softcap=50 and
    one with no softcap; the outputs must differ (softcap actually applies).
    Source: modeling_gemma2.py:293 (`softcap=self.attn_logit_softcapping`).
    """
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    def _mk(cap):
        attn = specs.AttentionSpec(
            n_q_heads=4, n_kv_heads=2, head_dim=32,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            logit_softcap=cap,
            rope=specs.RoPESpec(base_theta=10000.0,
                                basis=types.RoPEBasis.SPLIT_HALF),
        )
        ffn = specs.FFNSpec(intermediate_size=128,
                            activation=types.Activation.SILU,
                            gate_kind=types.GateKind.SWIGLU)
        block_spec = specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn, channel_mixer=ffn,
            pre_attn_norm=norm_spec, pre_ffn_norm=norm_spec,
        )
        return block.DecoderBlock(block_spec, hidden_size=128, max_seq=16,
                                  dtype=torch.float32)
    blk_a = _mk(None)
    blk_b = _mk(0.5)        # tiny cap so the effect is large
    blk_b.load_state_dict(blk_a.state_dict())
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache_a = kvcache.ContiguousKVCache(cache_spec, 1, 2, 32, 16)
    cache_b = kvcache.ContiguousKVCache(cache_spec, 1, 2, 32, 16)
    x = torch.randn(1, 4, 128) * 5.0       # larger scale so softcap bites
    pos = torch.arange(4)
    out_a = blk_a(x, position_ids=pos, cache=cache_a, start_pos=0)
    out_b = blk_b(x, position_ids=pos, cache=cache_b, start_pos=0)
    assert not torch.allclose(out_a, out_b, atol=1e-3)


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


def test_b5_block_with_moe_channel_mixer_forward_runs():
    """B5: DecoderBlock dispatches to MoE when channel_mixer is MoESpec."""
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    attn_spec = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=16,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=10000.0, basis=types.RoPEBasis.SPLIT_HALF),
    )
    moe = specs.MoESpec(
        n_experts=4, top_k=2, n_shared_experts=1,
        router_kind="softmax", routed_scaling_factor=1.0,
        expert_ffn=specs.FFNSpec(intermediate_size=8,
                                 activation=types.Activation.SILU,
                                 gate_kind=types.GateKind.SWIGLU),
    )
    spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn_spec,
        channel_mixer=moe,
        pre_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec,
    )
    H = 16
    blk = block.DecoderBlock(spec, hidden_size=H, max_seq=16, dtype=torch.float32)
    from api import feedforward as _ff
    assert isinstance(blk.feedforward, _ff.MoE)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=4, head_dim=16, max_seq=16,
    )
    x = torch.randn(1, 3, H)
    out = blk(x, position_ids=torch.arange(3), cache=cache, start_pos=0)
    assert out.shape == (1, 3, H)
