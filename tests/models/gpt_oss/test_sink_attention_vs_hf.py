"""GPT-OSS sink attention numerical gate vs HF eager_attention_forward.

We exercise the SINK math directly: build (q, k, v) tensors, call our
sdpa(sinks=...) AND call HF's `eager_attention_forward` (which is the
GPT-OSS reference sink implementation), then verify they agree at
atol=5e-4.

This avoids the GPT-OSS RoPE channel-layout mismatch (HF GPT-OSS uses
`emb = freqs` with cos/sin of shape [..., Dh/2] and a chunk(2)-style
rotation; our IR uses SPLIT_HALF basis with cos/sin of shape [..., Dh].
The two are mathematically equivalent under a weight reshuffle but
NOT bit-identical without the reshuffle.) The SINK forward is the v5
contribution and IS directly comparable.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.gpt_oss.modeling_gpt_oss import (  # noqa: E402
    eager_attention_forward,
)

from api import ops  # noqa: E402


class _SinkOnlyModule:
    """Stand-in module exposing the fields HF's eager_attention_forward
    reads: `num_key_value_groups`, `sinks`, `training`."""
    def __init__(self, n_kv_groups: int, sinks: torch.Tensor):
        self.num_key_value_groups = n_kv_groups
        self.sinks = sinks
        self.training = False


ATOL = 5e-4
RTOL = 5e-4


def test_gpt_oss_sink_attention_matches_hf_eager():
    """B=1, Hq=8, Hk=8 (no GQA), S=T=6, Dh=16."""
    B, Hq, Hk, S, T, Dh = 1, 8, 8, 6, 6, 16
    torch.manual_seed(0)
    q = torch.randn(B, Hq, S, Dh)
    k = torch.randn(B, Hk, T, Dh)
    v = torch.randn(B, Hk, T, Dh)
    sinks = torch.randn(Hq) * 0.1
    causal_mask = torch.triu(
        torch.full((S, T), float("-inf"), dtype=q.dtype), diagonal=1,
    )[None, None, :, :]
    scale = Dh ** -0.5

    # HF reference
    module = _SinkOnlyModule(n_kv_groups=Hq // Hk, sinks=sinks)
    hf_out, _ = eager_attention_forward(
        module, q, k, v, causal_mask, scaling=scale, dropout=0.0,
    )
    # eager_attention_forward returns [B, S, H, Dh] after transpose+contiguous.
    # Our ops.sdpa returns [B, H, S, Dh] — transpose to compare.
    hf_out = hf_out                          # [B, S, H, Dh]
    ours = ops.sdpa(q, k, v, attn_mask=causal_mask, scale=scale, sinks=sinks)
    ours_t = ours.transpose(1, 2)            # [B, S, H, Dh]
    diff = (hf_out - ours_t).abs().max().item()
    assert torch.allclose(hf_out, ours_t, atol=ATOL, rtol=RTOL), (
        f"GPT-OSS sink attention max_abs_diff={diff:.6e}, atol={ATOL}"
    )
    print(f"GPT-OSS sink attention max_abs_diff={diff:.2e}")


def test_gpt_oss_sink_attention_matches_hf_eager_gqa():
    """GQA case: Hq=8, Hk=2 (groups of 4)."""
    B, Hq, Hk, S, T, Dh = 1, 8, 2, 6, 6, 16
    torch.manual_seed(1)
    q = torch.randn(B, Hq, S, Dh)
    k = torch.randn(B, Hk, T, Dh)
    v = torch.randn(B, Hk, T, Dh)
    sinks = torch.randn(Hq) * 0.1
    causal_mask = torch.triu(
        torch.full((S, T), float("-inf"), dtype=q.dtype), diagonal=1,
    )[None, None, :, :]
    scale = Dh ** -0.5

    module = _SinkOnlyModule(n_kv_groups=Hq // Hk, sinks=sinks)
    hf_out, _ = eager_attention_forward(
        module, q, k, v, causal_mask, scaling=scale, dropout=0.0,
    )
    ours = ops.sdpa(q, k, v, attn_mask=causal_mask, scale=scale, sinks=sinks)
    ours_t = ours.transpose(1, 2)
    diff = (hf_out - ours_t).abs().max().item()
    assert torch.allclose(hf_out, ours_t, atol=ATOL, rtol=RTOL), (
        f"GQA sink attention max_abs_diff={diff:.6e}, atol={ATOL}"
    )
