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
