import torch

from api import rope, specs, types


def test_rope_module_returns_rotated():
    spec = specs.RoPESpec(base_theta=1_000_000.0, basis=types.RoPEBasis.SPLIT_HALF)
    head_dim = 16
    max_seq = 32
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)
    B, S, H = 1, 8, 2
    q = torch.randn(B, S, H, head_dim)
    k = torch.randn(B, S, H, head_dim)
    pos = torch.arange(S)
    q_rot, k_rot = module(q, k, pos)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.allclose(q_rot, q)
    assert torch.allclose(q_rot[:, 0], q[:, 0], atol=1e-5)


def test_rope_freqs_have_correct_shape():
    spec = specs.RoPESpec(base_theta=10000.0, basis=types.RoPEBasis.SPLIT_HALF)
    head_dim = 8
    max_seq = 16
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)
    assert module.cos_cached.shape == (max_seq, head_dim)
    assert module.sin_cached.shape == (max_seq, head_dim)


def test_rope_matches_hf_llama_style():
    spec = specs.RoPESpec(base_theta=1_000_000.0, basis=types.RoPEBasis.SPLIT_HALF)
    head_dim = 8
    max_seq = 16
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)
    inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    cos_ref = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin_ref = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    assert torch.allclose(module.cos_cached, cos_ref, atol=1e-5)
    assert torch.allclose(module.sin_cached, sin_ref, atol=1e-5)


def test_llama3_scaling_matches_hf_compute_llama3_parameters():
    """LLAMA3 smooth scaling math must match HF's reference _compute_llama3_parameters."""
    import math
    # Llama-3-8B params per the official Llama 3 release.
    factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    original_context_length = 8192
    head_dim = 128
    base_theta = 500_000.0

    # Build our RoPE.
    extra = specs.Llama3RoPEParams(
        factor=factor, low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        original_context_length=original_context_length,
    )
    rope_spec = specs.RoPESpec(
        base_theta=base_theta, basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LLAMA3, llama3_extra=extra,
    )
    api_rope = rope.RoPE(rope_spec, head_dim=head_dim, max_seq=32,
                         dtype=torch.float32)

    # HF reference math, replicated inline:
    inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    low_wlen = original_context_length / low_freq_factor
    high_wlen = original_context_length / high_freq_factor
    wlen = 2 * math.pi / inv_freq
    inv_freq_scaled = torch.where(wlen > low_wlen, inv_freq / factor, inv_freq)
    smooth = (original_context_length / wlen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * inv_freq / factor + smooth * inv_freq
    is_med = (wlen >= high_wlen) & (wlen <= low_wlen)
    inv_freq_ref = torch.where(is_med, smoothed, inv_freq_scaled)

    t = torch.arange(32).float()
    freqs_ref = torch.outer(t, inv_freq_ref)
    cos_ref = torch.cat([freqs_ref.cos(), freqs_ref.cos()], dim=-1)
    sin_ref = torch.cat([freqs_ref.sin(), freqs_ref.sin()], dim=-1)

    assert torch.allclose(api_rope.cos_cached, cos_ref, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, sin_ref, atol=1e-5)


def test_rope_module_partial_rotary_factor():
    """B0.6 — RoPE module with partial_rotary_factor=0.25 uses HF Gemma 4
    proportional geometry. cos/sin tables are at FULL head_dim with zero-padded
    inv_freq. With rope_angles = pr * head_dim // 2 = 8, the channels that pass
    through identity are [rope_angles, head_dim/2) AND
    [head_dim/2 + rope_angles, head_dim) — NOT a contiguous prefix split.
    """
    spec = specs.RoPESpec(
        base_theta=1_000_000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        partial_rotary_factor=0.25,
        partial_rotary_kind="proportional",   # Gemma 4 semantic — full head_dim
    )
    head_dim = 64
    max_seq = 32
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)
    # B0.6: cos/sin tables are FULL head_dim wide (zero-padded inv_freq).
    assert module.cos_cached.shape == (max_seq, head_dim)
    rope_angles = int(0.25 * head_dim // 2)
    assert rope_angles == 8

    # The trailing channels of inv_freq must be zero (so cos=1, sin=0 there).
    # Position 0 is special (any inv_freq * 0 = 0, so cos=1 everywhere).
    # Use position 1: cos[1, rope_angles:Dh/2] should be 1, sin[1, ...] = 0.
    cos1 = module.cos_cached[1]
    sin1 = module.sin_cached[1]
    Dh_half = head_dim // 2
    assert torch.allclose(cos1[rope_angles:Dh_half], torch.ones(Dh_half - rope_angles),
                          atol=1e-6)
    assert torch.allclose(sin1[rope_angles:Dh_half], torch.zeros(Dh_half - rope_angles),
                          atol=1e-6)
    # Same for the second half (mirror of the inv_freq zeros).
    assert torch.allclose(cos1[Dh_half + rope_angles:], torch.ones(Dh_half - rope_angles),
                          atol=1e-6)
    assert torch.allclose(sin1[Dh_half + rope_angles:], torch.zeros(Dh_half - rope_angles),
                          atol=1e-6)

    B, S, H = 1, 8, 4
    q = torch.randn(B, S, H, head_dim)
    k = torch.randn(B, S, H, head_dim)
    pos = torch.arange(S)
    q_rot, k_rot = module(q, k, pos)
    assert q_rot.shape == q.shape
    # The "identity" channel slabs: [rope_angles, Dh/2) and [Dh/2 + rope_angles, Dh).
    # By rotate_half pairing (i ↔ i + Dh/2 with sign flip on second), channel i
    # in [rope_angles, Dh/2) has cos[i]=1, sin[i]=0 → output[i] = x[i]. And
    # channel i' = i + Dh/2 in [Dh/2 + rope_angles, Dh) has cos[i']=1, sin[i']=0
    # → output[i'] = x[i'].
    assert torch.allclose(q_rot[..., rope_angles:Dh_half],
                          q[..., rope_angles:Dh_half], atol=1e-6)
    assert torch.allclose(q_rot[..., Dh_half + rope_angles:],
                          q[..., Dh_half + rope_angles:], atol=1e-6)


def test_longrope_short_factor_below_boundary():
    """B2a: LongRoPE — for positions ≤ original_max, RoPE uses short_factor.
    Built table must match the HF formula:
        inv_freq[i] = 1 / (short[i] * base ** (2i/dim_rot))
    """
    head_dim = 8
    base = 10_000.0
    short_factor = (1.0, 1.0, 1.0, 1.0)         # length dim_rot/2 = 4 (no scaling)
    long_factor = (2.0, 2.0, 2.0, 2.0)
    spec = specs.RoPESpec(
        base_theta=base, basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LONGROPE,
        longrope_extra=specs.LongRoPEParams(
            short_factor=short_factor, long_factor=long_factor,
            original_max_position_embeddings=16, attention_factor=1.0,
        ),
    )
    r = rope.RoPE(spec, head_dim=head_dim, max_seq=32, dtype=torch.float32)
    # The short table is the default; with short_factor=1.0 it equals plain RoPE.
    spec_plain = specs.RoPESpec(base_theta=base, basis=types.RoPEBasis.SPLIT_HALF,
                                scaling=types.RoPEScaling.NONE)
    r_plain = rope.RoPE(spec_plain, head_dim=head_dim, max_seq=32, dtype=torch.float32)
    assert torch.allclose(r.cos_cached, r_plain.cos_cached, atol=1e-6)
    assert torch.allclose(r.sin_cached, r_plain.sin_cached, atol=1e-6)


def test_longrope_dispatches_long_above_boundary():
    """B2a: LongRoPE forward must pick LONG table when max(position_ids)+1 > boundary."""
    head_dim = 8
    spec = specs.RoPESpec(
        base_theta=10_000.0, basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LONGROPE,
        longrope_extra=specs.LongRoPEParams(
            short_factor=(1.0, 1.0, 1.0, 1.0),
            long_factor=(2.0, 2.0, 2.0, 2.0),
            original_max_position_embeddings=4,
            attention_factor=1.0,
        ),
    )
    r = rope.RoPE(spec, head_dim=head_dim, max_seq=16, dtype=torch.float32)
    # Short and long tables differ.
    assert not torch.allclose(r.cos_cached, r.cos_cached_long)
    # Dispatch check: small positions → cos_cached values.
    q = torch.randn(1, 3, 2, head_dim)
    k = torch.randn(1, 3, 2, head_dim)
    pos_short = torch.arange(3)                                 # max+1 = 3 ≤ 4 → short
    q_s, k_s = r(q, k, pos_short)
    # Manually build the expected output using cos_cached[pos] directly.
    cos_s, sin_s = r.cos_cached[pos_short], r.sin_cached[pos_short]
    from api import ops
    q_exp, k_exp = ops.rope_apply(q, k, cos_s, sin_s, basis="split_half")
    assert torch.allclose(q_s, q_exp)
    # And large positions → cos_cached_long.
    pos_long = torch.tensor([5, 6, 7])                          # max+1 = 8 > 4 → long
    q_l, k_l = r(q, k, pos_long)
    cos_l, sin_l = r.cos_cached_long[pos_long], r.sin_cached_long[pos_long]
    q_exp_l, k_exp_l = ops.rope_apply(q, k, cos_l, sin_l, basis="split_half")
    assert torch.allclose(q_l, q_exp_l)


def test_longrope_attention_factor_scales_cos_sin():
    """B2a: attention_factor multiplies BOTH cos and sin in the cached table.
    Source: modeling_phi3.py:128-129 (cos = emb.cos() * attention_scaling)."""
    head_dim = 8
    spec_a = specs.RoPESpec(
        base_theta=10_000.0, basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LONGROPE,
        longrope_extra=specs.LongRoPEParams(
            short_factor=(1.0,)*4, long_factor=(1.0,)*4,
            original_max_position_embeddings=16, attention_factor=1.0,
        ),
    )
    spec_b = specs.RoPESpec(
        base_theta=10_000.0, basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LONGROPE,
        longrope_extra=specs.LongRoPEParams(
            short_factor=(1.0,)*4, long_factor=(1.0,)*4,
            original_max_position_embeddings=16, attention_factor=2.5,
        ),
    )
    r_a = rope.RoPE(spec_a, head_dim=head_dim, max_seq=16, dtype=torch.float32)
    r_b = rope.RoPE(spec_b, head_dim=head_dim, max_seq=16, dtype=torch.float32)
    assert torch.allclose(r_b.cos_cached, r_a.cos_cached * 2.5, atol=1e-6)
    assert torch.allclose(r_b.sin_cached, r_a.sin_cached * 2.5, atol=1e-6)


# ---------------------------------------------------------------------------
# B4: INTERLEAVED basis (Llama 4) — rotates consecutive (real, imag) pairs.
# ---------------------------------------------------------------------------


def test_interleaved_rope_cos_sin_shape():
    """INTERLEAVED cos/sin tables have shape [S, Dh/2], not [S, Dh]."""
    spec = specs.RoPESpec(base_theta=500_000.0, basis=types.RoPEBasis.INTERLEAVED)
    head_dim = 16
    max_seq = 8
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)
    assert module.cos_cached.shape == (max_seq, head_dim // 2)
    assert module.sin_cached.shape == (max_seq, head_dim // 2)


def test_interleaved_rope_matches_hf_llama4_complex_multiply():
    """Verify the INTERLEAVED rotation matches HF Llama 4's
    `apply_rotary_emb` (modeling_llama4.py:245-254) which uses
    view_as_complex + complex multiply.
    """
    spec = specs.RoPESpec(base_theta=500_000.0, basis=types.RoPEBasis.INTERLEAVED)
    head_dim = 8
    max_seq = 16
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)

    B, S, H = 1, 4, 2
    torch.manual_seed(0)
    q = torch.randn(B, S, H, head_dim)
    k = torch.randn(B, S, H, head_dim)
    pos = torch.arange(S)
    q_rot, k_rot = module(q, k, pos)

    # Reference: rebuild HF Llama 4's complex-multiply path.
    inv_freq = 1.0 / (500_000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)               # [max_seq, head_dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex
    # Sub-select positions [0..S).
    freqs_cis = freqs_cis[:S]
    # Apply per HF: view_as_complex on (real, imag) pairs.
    q_ = torch.view_as_complex(q.reshape(B, S, H, head_dim // 2, 2))
    k_ = torch.view_as_complex(k.reshape(B, S, H, head_dim // 2, 2))
    # Broadcast: freqs_cis[None, S, None, head_dim/2].
    freqs_b = freqs_cis[None, :, None, :]
    q_ref = torch.view_as_real(q_ * freqs_b).flatten(3)
    k_ref = torch.view_as_real(k_ * freqs_b).flatten(3)
    assert torch.allclose(q_rot, q_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_rot, k_ref, atol=1e-5, rtol=1e-5)


def test_interleaved_rope_identity_at_position_0():
    """At position 0 the rotation is identity (cos=1, sin=0)."""
    spec = specs.RoPESpec(base_theta=10000.0, basis=types.RoPEBasis.INTERLEAVED)
    head_dim = 16
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=8, dtype=torch.float32)
    B, S, H = 1, 1, 2
    q = torch.randn(B, S, H, head_dim)
    k = torch.randn(B, S, H, head_dim)
    pos = torch.tensor([0])
    q_rot, k_rot = module(q, k, pos)
    assert torch.allclose(q_rot, q, atol=1e-6)
    assert torch.allclose(k_rot, k, atol=1e-6)


def test_b5_yarn_inv_freq_matches_hf_compute_yarn_parameters():
    """B5: YARN inv_freq must match HF's _compute_yarn_parameters exactly.

    Uses DeepSeek-V2-Lite's actual production rope_scaling block as a fixture.
    Source: deepseek-ai/DeepSeek-V2-Lite/config.json (factor=40, beta_fast=32,
    beta_slow=1, mscale=0.707, mscale_all_dim=0.707, original=4096).
    """
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    from api.rope import _yarn_inv_freq_and_scale

    class _FakeCfg:
        head_dim = 64
        hidden_size = 1024
        num_attention_heads = 16
        rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 40.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
        }
        max_position_embeddings = 163840
        def standardize_rope_params(self):
            pass

    hf_inv_freq, hf_att = _compute_yarn_parameters(_FakeCfg())
    extra = specs.YarnRoPEParams(
        factor=40.0, original_max_position_embeddings=4096,
        beta_fast=32, beta_slow=1, mscale=0.707, mscale_all_dim=0.707,
    )
    ours_inv_freq, ours_att = _yarn_inv_freq_and_scale(10000.0, 64, extra)
    assert torch.allclose(hf_inv_freq, ours_inv_freq, atol=1e-6)
    assert abs(hf_att - ours_att) < 1e-9


def test_b5_yarn_attention_factor_non_unity_case():
    """B5: when mscale_all_dim is 0, attention_factor falls back to get_mscale(factor)."""
    from api.rope import _yarn_inv_freq_and_scale
    extra = specs.YarnRoPEParams(
        factor=40.0, original_max_position_embeddings=4096,
        beta_fast=32, beta_slow=1, mscale=1.0, mscale_all_dim=0.0,
    )
    _, att = _yarn_inv_freq_and_scale(10000.0, 64, extra)
    # get_mscale(40, 1.0) = 0.1 * 1.0 * log(40) + 1.0
    import math
    expected = 0.1 * 1.0 * math.log(40.0) + 1.0
    assert abs(att - expected) < 1e-9


def test_b5_yarn_interleaved_matches_v2_complex_multiply():
    """B5: INTERLEAVED basis + YARN scaling reproduces V2 RoPE math.

    DeepSeek-V2 uses `view_as_complex(reshape(*, -1, 2))` + complex multiply
    with `polar(ones, freqs)` (modeling_deepseek_v2.py:271-284). Our
    INTERLEAVED basis is the same complex multiply on (real, imag) pairs.
    """
    yarn = specs.YarnRoPEParams(
        factor=40.0, original_max_position_embeddings=4096,
        beta_fast=32, beta_slow=1, mscale=0.707, mscale_all_dim=0.707,
    )
    spec = specs.RoPESpec(
        base_theta=10000.0,
        basis=types.RoPEBasis.INTERLEAVED,
        scaling=types.RoPEScaling.YARN,
        yarn_extra=yarn,
    )
    mod = rope.RoPE(spec, head_dim=64, max_seq=128, dtype=torch.float32)
    assert mod.cos_cached.shape == (128, 32)

    # Replicate V2 forward inline.
    from api.rope import _yarn_inv_freq_and_scale
    inv_freq_yarn, _ = _yarn_inv_freq_and_scale(10000.0, 64, yarn)
    positions = torch.arange(8).float()
    freqs = positions.view(8, 1) * inv_freq_yarn.view(1, 32)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex [8, 32]

    torch.manual_seed(7)
    B, H, S, D = 1, 4, 8, 64
    q = torch.randn(B, H, S, D)
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    fc = freqs_cis.unsqueeze(0).unsqueeze(0)        # [1, 1, S, D/2]
    hf_out = torch.view_as_real(q_ * fc).flatten(3).type_as(q)

    q_api = q.transpose(1, 2)                       # [B, S, H, D]
    k_api = torch.zeros_like(q_api)
    q_rot, _ = mod(q_api, k_api, positions.long())
    q_rot_b = q_rot.transpose(1, 2)
    diff = (q_rot_b - hf_out).abs().max().item()
    assert diff < 1e-5, f"max_abs_diff={diff:.3e}"
