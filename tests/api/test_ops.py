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
