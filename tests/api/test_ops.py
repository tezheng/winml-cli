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
