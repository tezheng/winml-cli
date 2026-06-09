"""B7: api/ssm.py — Mamba-2 mixer tests.

These tests exercise:
1. Mixer instantiation with various canonical Mamba-2 shapes.
2. Forward pass shape correctness.
3. Determinism.
4. KV cache writes (conv_state + ssm_state populated after a prefill).
5. End-to-end DecoderBlock with SSM token mixer.
"""
import pytest
import torch

from api import block as _block, kvcache, specs, ssm, types


def _make_ssd_spec(hidden_size=128, n_heads=8, head_dim=32, n_groups=2,
                   d_state=16, conv_kernel=4, chunk_size=8):
    """A small SSD spec that satisfies hidden_size*expand == n_heads*head_dim."""
    # d_inner = hidden_size * 2 (expand factor)
    d_inner = hidden_size * 2
    assert n_heads * head_dim == d_inner
    assert n_heads % n_groups == 0
    base = specs.SSMSpec(d_state=d_state, d_conv=conv_kernel, d_inner=d_inner)
    return specs.SSDSpec(
        base=base, headdim=head_dim, ngroups=n_groups, n_heads=n_heads,
        chunk_size=chunk_size, layer_norm_epsilon=1e-5,
    )


def test_mamba2_mixer_init_basic():
    spec = _make_ssd_spec()
    mixer = ssm.Mamba2Mixer(spec, hidden_size=128)
    # in_proj input dim = hidden_size; output = d_inner + conv_dim + num_heads
    # d_inner = hidden * expand = 128*2 = 256
    # conv_dim = d_inner + 2 * n_groups * d_state = 256 + 2*2*16 = 320
    # output = 256 + 320 + 8 = 584
    assert mixer.d_inner == 256
    assert mixer.conv_dim == 320
    assert mixer.in_proj.weight.shape == (584, 128)
    # conv1d: depthwise (groups = conv_dim = 320), kernel=4
    assert mixer.conv1d.weight.shape == (320, 1, 4)
    assert mixer.conv1d.groups == 320
    # dt_bias, A_log, D per-head
    assert mixer.dt_bias.shape == (8,)
    assert mixer.A_log.shape == (8,)
    assert mixer.D.shape == (8,)
    # out_proj: d_inner -> hidden_size
    assert mixer.out_proj.weight.shape == (128, 256)


def test_mamba2_mixer_rejects_invalid_shapes():
    base = specs.SSMSpec(d_state=16, d_conv=4, d_inner=200)
    bad_spec = specs.SSDSpec(base=base, headdim=64, ngroups=1, n_heads=4)
    # n_heads * head_dim = 256 != d_inner = 200
    with pytest.raises(ValueError, match="must equal d_inner"):
        ssm.Mamba2Mixer(bad_spec, hidden_size=100)

    base2 = specs.SSMSpec(d_state=16, d_conv=4, d_inner=64)
    # n_heads = 5 not divisible by ngroups = 2
    bad_spec2 = specs.SSDSpec(base=base2, headdim=64 // 5, ngroups=2, n_heads=5)
    # but invariants...
    base3 = specs.SSMSpec(d_state=16, d_conv=4, d_inner=64)
    bad_spec3 = specs.SSDSpec(base=base3, headdim=16, ngroups=3, n_heads=4)
    # n_heads(4) % ngroups(3) != 0
    with pytest.raises(ValueError, match="divisible"):
        ssm.Mamba2Mixer(bad_spec3, hidden_size=32)


def test_mamba2_mixer_forward_shape_no_cache():
    spec = _make_ssd_spec(hidden_size=64, n_heads=4, head_dim=32, n_groups=2,
                          d_state=8, chunk_size=4)
    mixer = ssm.Mamba2Mixer(spec, hidden_size=64)
    mixer.eval()
    B, S = 2, 9   # S not divisible by chunk_size -> exercises pad path
    x = torch.randn(B, S, 64)
    with torch.no_grad():
        y = mixer(x)
    assert y.shape == (B, S, 64)


def test_mamba2_mixer_forward_shape_with_cache_prefill():
    spec = _make_ssd_spec(hidden_size=64, n_heads=4, head_dim=32, n_groups=2,
                          d_state=8, chunk_size=4)
    mixer = ssm.Mamba2Mixer(spec, hidden_size=64)
    mixer.eval()
    B, S = 1, 8
    cache = kvcache.SSMStateCache(
        batch_size=B, conv_dim=mixer.conv_dim,
        conv_kernel=mixer.conv_kernel,
        n_heads=mixer.num_heads, head_dim=mixer.head_dim,
        d_state=mixer.d_state,
    )
    x = torch.randn(B, S, 64)
    with torch.no_grad():
        y = mixer(x, cache=cache)
    assert y.shape == (B, S, 64)
    assert cache.has_previous_state
    # conv_state should now be populated (non-zero); ssm_state should be
    # populated (non-zero with random weights and inputs).
    assert cache.conv_state.abs().sum().item() > 0
    assert cache.ssm_state.abs().sum().item() > 0


def test_mamba2_mixer_forward_determinism():
    spec = _make_ssd_spec()
    mixer = ssm.Mamba2Mixer(spec, hidden_size=128)
    mixer.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, 128)
    with torch.no_grad():
        y1 = mixer(x)
        y2 = mixer(x)
    assert torch.allclose(y1, y2, atol=0)


def test_mamba2_mixer_decode_path_raises():
    """B7: decode (single-step) is not implemented yet."""
    spec = _make_ssd_spec(hidden_size=64, n_heads=4, head_dim=32, n_groups=2,
                          d_state=8, chunk_size=4)
    mixer = ssm.Mamba2Mixer(spec, hidden_size=64)
    cache = kvcache.SSMStateCache(
        batch_size=1, conv_dim=mixer.conv_dim,
        conv_kernel=mixer.conv_kernel,
        n_heads=mixer.num_heads, head_dim=mixer.head_dim,
        d_state=mixer.d_state,
    )
    cache.has_previous_state = True  # simulate post-prefill
    x = torch.randn(1, 1, 64)
    with pytest.raises(NotImplementedError, match="decode"):
        mixer(x, cache=cache)


def test_decoder_block_with_ssd_token_mixer_no_ffn():
    """B7: DecoderBlock + SSDSpec token_mixer + skip_ffn produces correct
    residual structure: x = x + mixer(norm(x)). No FFN sublayer."""
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-5,
                              weight_mode=types.NormWeightMode.STANDARD_W)
    ssd = _make_ssd_spec(hidden_size=64, n_heads=4, head_dim=32, n_groups=2,
                         d_state=8, chunk_size=4)
    # channel_mixer is required by the union type but ignored when skip_ffn=True.
    placeholder_ffn = specs.FFNSpec(intermediate_size=128,
                                    activation=types.Activation.SILU,
                                    gate_kind=types.GateKind.SWIGLU)
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=ssd, channel_mixer=placeholder_ffn,
        pre_attn_norm=norm_spec, pre_ffn_norm=norm_spec,
        skip_ffn=True,
    )
    blk = _block.DecoderBlock(block_spec, hidden_size=64, max_seq=128)
    blk.eval()
    assert blk.feedforward is None
    assert blk.pre_ffn_norm is None
    assert blk._is_ssm_block is True

    B, S = 1, 8
    x = torch.randn(B, S, 64)
    cache = kvcache.SSMStateCache(
        batch_size=B, conv_dim=blk.attention.conv_dim,
        conv_kernel=blk.attention.conv_kernel,
        n_heads=blk.attention.num_heads,
        head_dim=blk.attention.head_dim,
        d_state=blk.attention.d_state,
    )
    with torch.no_grad():
        y = blk(x, cache=cache)
    assert y.shape == (B, S, 64)
    assert cache.has_previous_state


# ---------------------------------------------------------------------------
# v7 P3: Qwen3-Next Gated DeltaNet tests
# ---------------------------------------------------------------------------


def _gated_deltanet_spec(
    num_v_heads: int = 4,
    num_k_heads: int = 2,
    head_k_dim: int = 16,
    head_v_dim: int = 16,
) -> specs.GatedDeltaNetSpec:
    return specs.GatedDeltaNetSpec(
        num_v_heads=num_v_heads,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel=4,
        norm_eps=1e-6,
        silu_gate=True,
    )


def test_v7_p3_gated_deltanet_mixer_shapes():
    """v7 P3: GatedDeltaNetMixer composes and forward returns [B,S,hidden]."""
    spec = _gated_deltanet_spec()
    hidden = 64
    mixer = ssm.GatedDeltaNetMixer(spec, hidden_size=hidden, dtype=torch.float32)
    # Projection shapes.
    expected_qkvz = 2 * (spec.head_k_dim * spec.num_k_heads) + 2 * (spec.head_v_dim * spec.num_v_heads)
    expected_ba = 2 * spec.num_v_heads
    assert mixer.in_proj_qkvz.weight.shape == (expected_qkvz, hidden)
    assert mixer.in_proj_ba.weight.shape == (expected_ba, hidden)
    # Conv1d shape: groups = conv_dim = 2*key_dim + value_dim, depthwise.
    conv_dim = 2 * mixer.key_dim + mixer.value_dim
    assert mixer.conv1d.weight.shape == (conv_dim, 1, spec.conv_kernel)
    # A_log / dt_bias per head.
    assert mixer.A_log.shape == (spec.num_v_heads,)
    assert mixer.dt_bias.shape == (spec.num_v_heads,)
    # Output projection: value_dim -> hidden.
    assert mixer.out_proj.weight.shape == (hidden, mixer.value_dim)

    B, S = 2, 6
    x = torch.randn(B, S, hidden)
    out = mixer(x)
    assert out.shape == (B, S, hidden)
    assert torch.isfinite(out).all()


def test_v7_p3_gated_delta_step_matches_hf_inline_reference():
    """v7 P3: api.ops.gated_delta_step matches an inline reference written
    directly from modeling_qwen3_next.py:455-496 (torch_recurrent_gated_delta_rule)."""
    from api import ops as _ops
    torch.manual_seed(0)
    B, S, H, dk, dv = 2, 5, 3, 8, 8
    q = torch.randn(B, S, H, dk)
    k = torch.randn(B, S, H, dk)
    v = torch.randn(B, S, H, dv)
    beta = torch.rand(B, S, H)              # already in (0,1)
    g = -torch.rand(B, S, H)                # log-decay (negative)
    out, state = _ops.gated_delta_step(
        q, k, v, g, beta, initial_state=None, use_qk_l2norm=True,
    )

    # Inline reference — direct port of HF.
    q_n = _ops.l2norm(q, dim=-1, eps=1e-6)
    k_n = _ops.l2norm(k, dim=-1, eps=1e-6)
    qT = q_n.transpose(1, 2).float()
    kT = k_n.transpose(1, 2).float()
    vT = v.transpose(1, 2).float()
    betaT = beta.transpose(1, 2).float()
    gT = g.transpose(1, 2).float()
    scale = 1.0 / (dk ** 0.5)
    qT = qT * scale
    state_ref = torch.zeros(B, H, dk, dv)
    out_ref = torch.zeros(B, H, S, dv)
    for i in range(S):
        gi = gT[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        bi = betaT[:, :, i].unsqueeze(-1)
        ki = kT[:, :, i]
        vi = vT[:, :, i]
        qi = qT[:, :, i]
        state_ref = state_ref * gi
        kv_mem = (state_ref * ki.unsqueeze(-1)).sum(dim=-2)
        delta = (vi - kv_mem) * bi
        state_ref = state_ref + ki.unsqueeze(-1) * delta.unsqueeze(-2)
        out_ref[:, :, i] = (state_ref * qi.unsqueeze(-1)).sum(dim=-2)
    out_ref = out_ref.transpose(1, 2).to(out.dtype)

    assert torch.allclose(out, out_ref, atol=1e-5), (
        f"max_abs_diff={(out - out_ref).abs().max().item():.3e}"
    )
    assert torch.allclose(state, state_ref, atol=1e-5)


def test_v7_p3_gated_deltanet_decoder_block_forward():
    """v7 P3: DecoderBlock dispatches to GatedDeltaNetMixer when token_mixer
    is GatedDeltaNetSpec."""
    spec = _gated_deltanet_spec()
    hidden = 64
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=spec,
        channel_mixer=specs.FFNSpec(
            intermediate_size=128,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
        ),
        pre_attn_norm=specs.NormSpec(
            kind=types.NormKind.RMS, eps=1e-6,
            weight_mode=types.NormWeightMode.STANDARD_W,
        ),
        pre_ffn_norm=specs.NormSpec(
            kind=types.NormKind.RMS, eps=1e-6,
            weight_mode=types.NormWeightMode.STANDARD_W,
        ),
    )
    blk = _block.DecoderBlock(block_spec, hidden_size=hidden, max_seq=128)
    blk.eval()
    assert blk._is_gated_deltanet is True
    assert blk.feedforward is not None
    B, S = 1, 6
    x = torch.randn(B, S, hidden)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == (B, S, hidden)
    assert torch.isfinite(y).all()


def test_v7_p3_gated_deltanet_num_v_heads_divisibility_required():
    """v7 P3: num_v_heads must be divisible by num_k_heads (GQA constraint
    inherited from `Qwen3NextGatedDeltaNet.fix_query_key_value_ordering`)."""
    spec = specs.GatedDeltaNetSpec(
        num_v_heads=5, num_k_heads=2,
        head_k_dim=16, head_v_dim=16,
    )
    with pytest.raises(ValueError, match="divisible"):
        ssm.GatedDeltaNetMixer(spec, hidden_size=32, dtype=torch.float32)


def test_v7_p3_gated_deltanet_no_gqa_when_v_eq_k():
    """v7 P3: when num_v_heads == num_k_heads, no repeat_interleave applies;
    Q and K stay at num_k_heads heads."""
    spec = _gated_deltanet_spec(num_v_heads=4, num_k_heads=4)
    mixer = ssm.GatedDeltaNetMixer(spec, hidden_size=64, dtype=torch.float32)
    x = torch.randn(1, 4, 64)
    out = mixer(x)
    assert out.shape == (1, 4, 64)


def test_v7_p3_qwen3next_norm_gated_matches_hf_inline():
    """v7 P3: _Qwen3NextRMSNormGated matches modeling_qwen3_next.py:73-82
    exactly."""
    import torch.nn.functional as F
    norm = ssm._Qwen3NextRMSNormGated(hidden_size=16, eps=1e-6, dtype=torch.float32)
    with torch.no_grad():
        norm.weight.normal_(std=0.5)
    x = torch.randn(3, 4, 16)
    gate = torch.randn(3, 4, 16)
    out = norm(x, gate)

    # Reference inline (modeling_qwen3_next.py:73-82).
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + 1e-6)
    h = norm.weight * h.to(x.dtype)
    h = h * F.silu(gate.float())
    expected = h.to(x.dtype)
    assert torch.allclose(out, expected, atol=1e-6)
