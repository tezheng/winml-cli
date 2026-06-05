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
