import pytest
import torch
import torch.nn.functional as F

from api import ops


def test_silu_matches_torch():
    x = torch.randn(2, 4, 8)
    assert torch.allclose(ops.silu(x), F.silu(x), atol=1e-7)


def test_add_with_residual():
    x = torch.randn(2, 4, 8)
    r = torch.randn(2, 4, 8)
    assert torch.allclose(ops.add(x, r), x + r, atol=1e-7)


def test_add_with_scale():
    x = torch.randn(2, 4, 8)
    r = torch.randn(2, 4, 8)
    out = ops.add(x, r, scale=0.5)
    assert torch.allclose(out, x + 0.5 * r, atol=1e-7)


def test_mul_elementwise():
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 4, 8)
    assert torch.allclose(ops.mul(a, b), a * b, atol=1e-7)


def test_linear_no_quant():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    out = ops.linear(x, w)
    assert torch.allclose(out, F.linear(x, w), atol=1e-5)


def test_linear_with_bias():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(8, dtype=torch.float32)
    out = ops.linear(x, w, bias=b)
    assert torch.allclose(out, F.linear(x, w, b), atol=1e-5)


def test_rms_norm_standard_w():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(16, dtype=torch.float32)
    eps = 1e-6
    out = ops.rms_norm(x, w, eps, mode="standard_w")
    ref = F.rms_norm(x, (16,), w, eps)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rms_norm_one_plus_w():
    """Gemma-style RMSNorm where weight is (1 + w)."""
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(16, dtype=torch.float32)
    eps = 1e-6
    out = ops.rms_norm(x, w, eps, mode="one_plus_w")
    ref = F.rms_norm(x, (16,), 1.0 + w, eps)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rms_norm_preserves_input_dtype():
    """RMSNorm computes in fp32 but returns input dtype."""
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    w = torch.randn(16, dtype=torch.bfloat16)
    out = ops.rms_norm(x, w, 1e-6, mode="standard_w")
    assert out.dtype == torch.bfloat16


def test_embed_basic():
    weight = torch.randn(100, 8, dtype=torch.float32)
    ids = torch.tensor([[0, 1, 2], [99, 50, 1]])
    out = ops.embed(ids, weight)
    assert out.shape == (2, 3, 8)
    assert torch.allclose(out[0, 0], weight[0])
    assert torch.allclose(out[1, 2], weight[1])


def test_embed_with_scale():
    """Gemma scales embeddings by sqrt(hidden_size)."""
    weight = torch.randn(10, 8, dtype=torch.float32)
    ids = torch.tensor([[1, 2]])
    out = ops.embed(ids, weight, scale=2.0)
    assert torch.allclose(out[0, 0], 2.0 * weight[1])


def test_lm_head_no_softcap():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(100, 16, dtype=torch.float32)
    out = ops.lm_head(x, w)
    assert torch.allclose(out, F.linear(x, w), atol=1e-5)


def test_lm_head_with_softcap():
    """Gemma 2 / 3 logits softcap."""
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(100, 16, dtype=torch.float32)
    cap = 30.0
    out = ops.lm_head(x, w, softcap=cap)
    logits = F.linear(x, w)
    ref = cap * torch.tanh(logits / cap)
    assert torch.allclose(out, ref, atol=1e-5)


def _hf_rope_half(x, cos, sin):
    """Reference: HF's rotate_half + (cos, sin) application.

    HF Llama-style: input is [..., D] split into halves [..., :D/2] and [..., D/2:].
    rotate_half swaps and negates: [-x2, x1].
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def test_rope_apply_split_half_matches_hf():
    """rope_apply with SPLIT_HALF basis should match HF's apply_rotary_pos_emb."""
    B, S, H, Dh = 1, 4, 2, 8
    q = torch.randn(B, S, H, Dh, dtype=torch.float32)
    k = torch.randn(B, S, H, Dh, dtype=torch.float32)

    # cos / sin tables shaped [S, Dh] (broadcast over B and H)
    freqs = torch.linspace(0.1, 1.0, Dh // 2)
    positions = torch.arange(S).float().unsqueeze(-1) * freqs.unsqueeze(0)
    cos = torch.cat([positions.cos(), positions.cos()], dim=-1)   # [S, Dh]
    sin = torch.cat([positions.sin(), positions.sin()], dim=-1)

    q_rot, k_rot = ops.rope_apply(q, k, cos, sin, basis="split_half")

    cos_b = cos.view(1, S, 1, Dh)
    sin_b = sin.view(1, S, 1, Dh)
    q_ref = _hf_rope_half(q, cos_b, sin_b)
    k_ref = _hf_rope_half(k, cos_b, sin_b)

    assert torch.allclose(q_rot, q_ref, atol=1e-6)
    assert torch.allclose(k_rot, k_ref, atol=1e-6)


def test_sdpa_matches_torch_for_mha():
    """For n_q_heads == n_kv_heads, sdpa should match F.scaled_dot_product_attention."""
    B, H, S, Dh = 1, 4, 6, 8
    q = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k = torch.randn(B, H, S, Dh, dtype=torch.float32)
    v = torch.randn(B, H, S, Dh, dtype=torch.float32)

    out = ops.sdpa(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_gqa_broadcasts_kv():
    """For GQA (n_kv_heads < n_q_heads), sdpa should repeat KV to match Q."""
    B, Hq, Hk, S, Dh = 1, 8, 2, 6, 16   # Hq // Hk == 4
    q = torch.randn(B, Hq, S, Dh, dtype=torch.float32)
    k = torch.randn(B, Hk, S, Dh, dtype=torch.float32)
    v = torch.randn(B, Hk, S, Dh, dtype=torch.float32)

    out = ops.sdpa(q, k, v, is_causal=True)
    k_rep = k.repeat_interleave(Hq // Hk, dim=1)
    v_rep = v.repeat_interleave(Hq // Hk, dim=1)
    ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_custom_scale():
    B, H, S, Dh = 1, 2, 4, 8
    q = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k = torch.randn(B, H, S, Dh, dtype=torch.float32)
    v = torch.randn(B, H, S, Dh, dtype=torch.float32)

    custom_scale = 0.25
    out = ops.sdpa(q, k, v, is_causal=False, scale=custom_scale)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=custom_scale)
    assert torch.allclose(out, ref, atol=1e-5)
