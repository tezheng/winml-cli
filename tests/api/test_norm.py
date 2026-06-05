import torch
import torch.nn.functional as F

from api import norm, specs, types


def test_rmsnorm_module_forward():
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    module = norm.RMSNorm(spec, hidden_size=16, dtype=torch.float32)
    with torch.no_grad():
        module.weight.fill_(1.0)
    x = torch.randn(2, 4, 16)
    out = module(x)
    ref = F.rms_norm(x, (16,), torch.ones(16), 1e-6)
    assert torch.allclose(out, ref, atol=1e-5)


def test_qknorm_per_head_dh_shape():
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    head_dim = 8
    qknorm = norm.QKNorm(spec, head_dim=head_dim,
                         shape=types.QKNormShape.PER_HEAD_DH,
                         dtype=torch.float32)
    assert qknorm.weight.shape == (head_dim,)
    B, S, H, Dh = 1, 4, 3, head_dim
    x = torch.randn(B, S, H, Dh)
    with torch.no_grad():
        qknorm.weight.fill_(1.0)
    out = qknorm(x)
    assert out.shape == (B, S, H, Dh)
    ref = F.rms_norm(x, (Dh,), torch.ones(Dh), 1e-6)
    assert torch.allclose(out, ref, atol=1e-5)


def test_qknorm_full_hdh_shape():
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    n_heads, head_dim = 3, 8
    qknorm = norm.QKNorm(spec, head_dim=head_dim, n_heads=n_heads,
                         shape=types.QKNormShape.FULL_HDH,
                         dtype=torch.float32)
    assert qknorm.weight.shape == (n_heads * head_dim,)
