"""Unit tests for the ALiBi position encoding op (v5-phase2 V1).

Source-grounded against `transformers/models/mpt/modeling_mpt.py:42-62`
(`build_mpt_alibi_tensor`) and 121 (additive bias before softmax).
"""
from __future__ import annotations

import math

import pytest
import torch

from api import ops


def _hf_mpt_alibi(num_heads: int, sequence_length: int,
                  alibi_bias_max: int = 8) -> torch.Tensor:
    """Verbatim copy of `transformers.models.mpt.modeling_mpt.
    build_mpt_alibi_tensor` (lines 42-62) — used to verify our slope build
    and bias values."""
    alibi = torch.arange(1 - sequence_length, 1, dtype=torch.int32).view(
        1, 1, 1, sequence_length
    )
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.int64).float()
    base = base * (alibi_bias_max / num_heads_power_of_2)
    slopes = 1.0 / torch.pow(2, base)
    slopes = slopes.view(1, num_heads_power_of_2, 1, 1)
    if num_heads_power_of_2 != num_heads:
        slopes = torch.concat(
            [slopes[:, 1::2, ...], slopes[:, ::2, ...]], dim=1
        )[:, :num_heads, ...]
    alibi = alibi * slopes
    return alibi.squeeze(0)


def test_build_alibi_slopes_power_of_2():
    """For n_heads=32 (MPT 7B), slopes are 2^(-(h+1)*8/32) for h=0..31."""
    slopes = ops.build_alibi_slopes(n_heads=32, alibi_bias_max=8.0)
    expected = torch.tensor(
        [2.0 ** (-((h + 1) * 8.0 / 32.0)) for h in range(32)],
        dtype=torch.float32,
    )
    assert slopes.shape == (32,)
    assert torch.allclose(slopes, expected, atol=1e-7)


def test_build_alibi_slopes_matches_mpt_reference():
    """Our slopes (for n_heads=32) must match the diagonal of the HF
    `build_mpt_alibi_tensor` output (the per-head multiplier)."""
    n_heads, S = 32, 16
    hf = _hf_mpt_alibi(n_heads, S, alibi_bias_max=8)   # [num_heads, 1, S]
    # The "slope" per head is `hf[h, 0, -2]` since position offset -1 (last
    # but one token) corresponds to (j - i) = -1 and so bias = -slope * 1.
    # The last col has offset 0, so bias=0. Use the second-to-last col.
    inferred_slopes = -hf[:, 0, -2]
    ours = ops.build_alibi_slopes(n_heads)
    assert torch.allclose(ours, inferred_slopes, atol=1e-6)


def test_build_alibi_slopes_non_power_of_2():
    """For n_heads=14 (not a power of 2): MPT interleaves odd+even from
    the n_heads_p2=16 slopes set."""
    n_heads = 14
    slopes = ops.build_alibi_slopes(n_heads=n_heads, alibi_bias_max=8.0)
    assert slopes.shape == (n_heads,)
    # The interleave must keep values in (0, 1) and roughly geometric.
    assert (slopes > 0).all().item()
    assert (slopes < 1).all().item()


def test_apply_alibi_bias_offset():
    """Bias[h, i, j] = slope[h] * (j - i). For causal (j <= i), bias <= 0."""
    n_heads = 4
    S = 6
    slopes = ops.build_alibi_slopes(n_heads)
    scores = torch.zeros(1, n_heads, S, S)
    biased = ops.apply_alibi(scores, slopes, start_pos=0)
    # Diagonal (i=j): bias is 0.
    for i in range(S):
        for h in range(n_heads):
            assert biased[0, h, i, i].item() == pytest.approx(0.0)
    # Off-diagonal: slope[h] * (j - i)
    for h in range(n_heads):
        for i in range(S):
            for j in range(S):
                expected = slopes[h].item() * (j - i)
                assert biased[0, h, i, j].item() == pytest.approx(expected, abs=1e-6)


def test_apply_alibi_softmax_invariance_vs_hf_mpt():
    """MPT adds a per-row-CONSTANT bias `slope[h] * (j - S + 1)` while our
    op uses the per-query-relative `slope[h] * (j - i)`. The two differ by
    a row-constant `slope[h] * (i - S + 1)` — softmax is translation-
    invariant in the row dim, so AFTER softmax (with the same causal
    mask applied) the two MUST be equal.

    This is the same invariance MPT exploits (modeling_mpt.py:42-48
    docstring) to use a compact 1-row bias tensor."""
    n_heads, S = 8, 12
    hf_alibi = _hf_mpt_alibi(n_heads, S)                # [H, 1, S]
    scores = torch.randn(1, n_heads, S, S)

    # HF path: scores + hf_alibi + causal mask, then softmax.
    hf_biased = scores + hf_alibi[None, :, :, :S]
    causal_mask = torch.triu(
        torch.full((S, S), float("-inf")), diagonal=1,
    )[None, None, :, :]
    hf_probs = torch.softmax(hf_biased + causal_mask, dim=-1, dtype=torch.float32)

    # Our path: scores + ops.apply_alibi + causal mask, then softmax.
    slopes = ops.build_alibi_slopes(n_heads)
    ours_biased = ops.apply_alibi(scores, slopes, start_pos=0)
    ours_probs = torch.softmax(
        ours_biased + causal_mask, dim=-1, dtype=torch.float32,
    )
    diff = (ours_probs - hf_probs).abs().max().item()
    assert torch.allclose(ours_probs, hf_probs, atol=1e-5), (
        f"post-softmax max_abs_diff={diff:.6e}"
    )


def test_apply_alibi_with_start_pos():
    """When start_pos > 0 (cached decoding), bias[h, i, j] = slope[h] *
    (j - (start_pos + i)). Verify by constructing a known case."""
    n_heads = 2
    S = 1
    start_pos = 5
    T = start_pos + S   # 6
    slopes = ops.build_alibi_slopes(n_heads)
    scores = torch.zeros(1, n_heads, S, T)
    biased = ops.apply_alibi(scores, slopes, start_pos=start_pos)
    # Query is at position 5 (start_pos=5, i=0). For j in [0, 6):
    for j in range(T):
        for h in range(n_heads):
            expected = slopes[h].item() * (j - start_pos)
            assert biased[0, h, 0, j].item() == pytest.approx(expected, abs=1e-6)


def test_apply_alibi_rejects_bad_slopes_shape():
    n_heads = 4
    scores = torch.zeros(1, n_heads, 3, 3)
    bad = ops.build_alibi_slopes(n_heads + 1)
    with pytest.raises(ValueError):
        ops.apply_alibi(scores, bad)
