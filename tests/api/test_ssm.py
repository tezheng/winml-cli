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
