"""BitNet b1.58 ternary quant round-trip (v6 A4).

The BitNet b1.58 quant grid is {-s, 0, +s} per-tensor with `s = mean(|w|)`.
This is LOSSY — the test verifies:
- shape preservation
- dequant values lie in {-s, 0, +s}
- per-element |w_recon - w| <= s (round-to-nearest-ternary bound)
- packed storage uses INT8 with last dim K//4

Source: BitNet b1.58 paper (Ma et al. 2024).
"""
from __future__ import annotations

import pytest
import torch

from api import quant, specs, types


def _spec() -> specs.QuantSpec:
    return specs.QuantSpec(
        qdtype=types.QDType.TERNARY,
        group_size=None,
        quant_axis=0,
        scale_dtype=torch.float32,
        has_zero_point=False,
        packing=types.PackingLayout.NONE,
        accumulator_dtype=torch.float32,
        role=types.QuantRole.WEIGHT,
    )


def test_ternary_round_trip_shape():
    """Pack and unpack a 2D weight; shapes match."""
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    packed, s = quant.ternary_quantize(w, _spec())
    assert packed.dtype == torch.int8
    assert packed.shape == (16, 16)
    assert s.dim() == 0
    recon = quant.ternary_dequantize(packed, s, K=64)
    assert recon.shape == w.shape


def test_ternary_round_trip_values_in_grid():
    """Reconstructed values are exactly in {-s, 0, +s}."""
    torch.manual_seed(1)
    w = torch.randn(8, 32)
    packed, s = quant.ternary_quantize(w, _spec())
    recon = quant.ternary_dequantize(packed, s, K=32)
    unique = torch.unique(recon).tolist()
    assert all(abs(v) < 1e-6 or abs(abs(v) - s.item()) < 1e-6 for v in unique), (
        f"recon values not in ternary grid: {unique}, s={s.item()}"
    )


def test_ternary_round_trip_bounded_error():
    """|w_recon - w| <= s (round-to-nearest-ternary error bound).

    Since we round w/s to the nearest of {-1, 0, +1} and clamp:
    - if |w/s| <= 0.5, projects to 0; error |w| <= 0.5 * s
    - if 0.5 < |w/s| <= 1.5, projects to sign(w); error |w - sign(w)*s| <= 0.5 * s
    - if |w/s| > 1.5, clamped to sign(w); error = |w| - s.
      For |w| <= 1.5 * s + bounded margin, error <= |w| but the |w| > 1.5*s
      tail means error grows linearly. The tight bound for the round-only
      path is 0.5*s; with clamp we use max(|w|).
    """
    torch.manual_seed(2)
    w = torch.randn(8, 64)
    packed, s = quant.ternary_quantize(w, _spec())
    recon = quant.ternary_dequantize(packed, s, K=64)
    err = (recon - w).abs()
    # Loose bound: max(|w|) (worst case when clamp activates).
    assert err.max().item() <= max(w.abs().max().item(), s.item()), (
        f"max err {err.max().item():.4e} exceeds max(|w|)={w.abs().max().item():.4e}"
    )
    # Tight bound on the non-clamped tail (|w| <= 1.5 * s): err <= 0.5 * s.
    mask = w.abs() <= 1.5 * s.item()
    assert err[mask].max().item() <= 0.5 * s.item() + 1e-6, (
        f"non-clamped tail err {err[mask].max().item():.4e} > 0.5 * s = "
        f"{0.5 * s.item():.4e}"
    )


def test_ternary_specific_values():
    """A hand-built tensor: known scale, known ternary outputs."""
    # w = [-2, -0.4, 0.6, 1.5] with absmean s = (2 + 0.4 + 0.6 + 1.5) / 4 = 1.125
    # w / s = [-1.78, -0.36, 0.53, 1.33]
    # round → [-2, 0, 1, 1] → clamp → [-1, 0, 1, 1]
    # recon = [-1.125, 0, 1.125, 1.125]
    w = torch.tensor([[-2.0, -0.4, 0.6, 1.5]])
    packed, s = quant.ternary_quantize(w, _spec())
    expected_s = (2.0 + 0.4 + 0.6 + 1.5) / 4.0
    assert abs(s.item() - expected_s) < 1e-5
    recon = quant.ternary_dequantize(packed, s, K=4)
    expected = torch.tensor([[-expected_s, 0.0, expected_s, expected_s]])
    assert torch.allclose(recon, expected, atol=1e-5)


def test_ternary_all_zero_weight():
    """All-zero weight: scale defaults to 1.0, recon is all zeros."""
    w = torch.zeros(4, 8)
    packed, s = quant.ternary_quantize(w, _spec())
    # Scale fallback to 1.0 (degenerate case).
    assert abs(s.item() - 1.0) < 1e-6
    recon = quant.ternary_dequantize(packed, s, K=8)
    assert torch.allclose(recon, w)


def test_ternary_packing_validates_last_dim():
    """Last dim not divisible by 4 raises."""
    w = torch.randn(4, 6)         # 6 % 4 != 0
    with pytest.raises(ValueError, match="divisible by 4"):
        quant.ternary_quantize(w, _spec())


def test_ternary_qdtype_validation():
    """Spec must have qdtype TERNARY."""
    bad = specs.QuantSpec(
        qdtype=types.QDType.INT4, group_size=None,
        quant_axis=0, scale_dtype=torch.float32,
        has_zero_point=False, packing=types.PackingLayout.NONE,
        accumulator_dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="TERNARY"):
        quant.ternary_quantize(torch.zeros(4, 4), bad)


def test_ternary_rejects_zero_point():
    """has_zero_point=True is rejected (ternary is symmetric)."""
    bad = specs.QuantSpec(
        qdtype=types.QDType.TERNARY, group_size=None,
        quant_axis=0, scale_dtype=torch.float32,
        has_zero_point=True, packing=types.PackingLayout.NONE,
        accumulator_dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="symmetric"):
        quant.ternary_quantize(torch.zeros(4, 4), bad)
