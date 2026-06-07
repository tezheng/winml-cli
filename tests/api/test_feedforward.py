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


def test_geglu_forward_shape():
    spec = specs.FFNSpec(
        intermediate_size=128,
        activation=types.Activation.GELU,
        gate_kind=types.GateKind.GEGLU,
    )
    ffn = feedforward.FeedForward(spec, hidden_size=32, dtype=torch.float32)
    x = torch.randn(2, 4, 32)
    out = ffn(x)
    assert out.shape == (2, 4, 32)


def test_geglu_uses_gelu_pytorch_tanh_for_gemma():
    """Gemma's GeGLU uses gelu_pytorch_tanh, not plain gelu."""
    spec = specs.FFNSpec(
        intermediate_size=8,
        activation=types.Activation.GELU,
        gate_kind=types.GateKind.GEGLU,
    )
    ffn = feedforward.FeedForward(spec, hidden_size=4, dtype=torch.float32)
    x = torch.randn(1, 2, 4)
    g = F.linear(x, ffn.gate_proj.weight)
    u = F.linear(x, ffn.up_proj.weight)
    expected = F.linear(F.gelu(g, approximate="tanh") * u, ffn.down_proj.weight)
    assert torch.allclose(ffn(x), expected, atol=1e-5)


def test_fused_gate_up_equivalent_to_split():
    """B2a: Phi-3 fused gate/up — one big projection of size 2*I, chunked into
    (gate, up). When the fused matrix is built by concatenating split
    gate/up along the output dim, the two paths produce identical output.
    Source: modeling_phi3.py:54-64."""
    hidden, I = 32, 64
    split = specs.FFNSpec(intermediate_size=I, activation=types.Activation.SILU,
                          gate_kind=types.GateKind.SWIGLU, fused_gate_up=False)
    fused = specs.FFNSpec(intermediate_size=I, activation=types.Activation.SILU,
                          gate_kind=types.GateKind.SWIGLU, fused_gate_up=True)
    f_s = feedforward.FeedForward(split, hidden_size=hidden, dtype=torch.float32)
    f_f = feedforward.FeedForward(fused, hidden_size=hidden, dtype=torch.float32)
    with torch.no_grad():
        # Phi-3 chunks along last dim — `gate, up = chunk(2, dim=-1)` — which
        # means rows 0..I are gate and rows I..2I are up in the fused weight.
        f_f.gate_up_proj.weight.copy_(torch.cat(
            [f_s.gate_proj.weight, f_s.up_proj.weight], dim=0
        ))
        f_f.down_proj.weight.copy_(f_s.down_proj.weight)
    x = torch.randn(1, 4, hidden)
    out_s = f_s(x)
    out_f = f_f(x)
    assert torch.allclose(out_s, out_f, atol=1e-5)


def test_fused_gate_up_shape():
    spec = specs.FFNSpec(intermediate_size=128,
                         activation=types.Activation.SILU,
                         gate_kind=types.GateKind.SWIGLU,
                         fused_gate_up=True)
    ffn = feedforward.FeedForward(spec, hidden_size=64, dtype=torch.float32)
    assert ffn.gate_up_proj.weight.shape == (2 * 128, 64)
    assert ffn.gate_proj is None
    assert ffn.up_proj is None
