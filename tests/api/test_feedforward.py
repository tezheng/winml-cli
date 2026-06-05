import torch
import torch.nn.functional as F

from api import feedforward, specs, types


def _swiglu_spec():
    return specs.FFNSpec(
        intermediate_size=128,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )


def test_swiglu_forward_shape():
    spec = _swiglu_spec()
    ffn = feedforward.FeedForward(spec, hidden_size=32, dtype=torch.float32)
    x = torch.randn(2, 4, 32)
    out = ffn(x)
    assert out.shape == (2, 4, 32)


def test_swiglu_matches_manual():
    """y = down(silu(gate(x)) * up(x))"""
    spec = _swiglu_spec()
    ffn = feedforward.FeedForward(spec, hidden_size=8, dtype=torch.float32)
    x = torch.randn(1, 2, 8)
    g = F.linear(x, ffn.gate_proj.weight)
    u = F.linear(x, ffn.up_proj.weight)
    expected = F.linear(F.silu(g) * u, ffn.down_proj.weight)
    out = ffn(x)
    assert torch.allclose(out, expected, atol=1e-5)
