"""Trained attention sinks (v5-phase2 V5 — GPT-OSS).

GPT-OSS introduces a LEARNABLE per-head scalar `sinks` appended as
one extra logit column before softmax, then dropped post-softmax.
Mathematically: probabilities sum to 1 over (T + 1) slots (T keys + 1
sink), and the sink slot consumes some softmax mass.

Source: `transformers/models/gpt_oss/modeling_gpt_oss.py:267-275`
(eager_attention_forward) and 309 (`self.sinks = nn.Parameter(
torch.empty(num_attention_heads))`).

Note on terminology: the user-facing v5-phase2 spec brief mentions
"first N sink tokens always attended". That is the StreamingLLM
"attention sinks" idea (first 4 tokens are always KV-cached); GPT-OSS
uses a DIFFERENT mechanism — a LEARNED per-head scalar parameter.
This test exercises the GPT-OSS form (which is the form HF transformers
actually implements and what `MaskKind.SINK` + `n_sink_tokens` wires
in our IR).
"""
from __future__ import annotations

import pytest
import torch

from api import attention as api_attention, kvcache, ops, specs, types


def test_sdpa_sinks_matches_hf_gpt_oss_eager():
    """Hand-rolled GPT-OSS eager attention sink path vs our sdpa(sinks=...)."""
    B, Hq, S, T, Dh = 1, 4, 6, 6, 8
    torch.manual_seed(0)
    q = torch.randn(B, Hq, S, Dh)
    k = torch.randn(B, Hq, T, Dh)
    v = torch.randn(B, Hq, T, Dh)
    sinks = torch.randn(Hq)
    scale = Dh ** -0.5

    # HF reference: build attn_weights = q@kT*scale + mask, then append
    # sinks broadcast, subtract max, softmax, drop the sink col.
    # Source: modeling_gpt_oss.py:263-277.
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(
        torch.full((S, T), float("-inf"), dtype=q.dtype), diagonal=1,
    )[None, None, :, :]
    attn = attn + causal_mask
    sink_broadcast = sinks.view(1, Hq, 1, 1).expand(B, Hq, S, 1)
    combined = torch.cat([attn, sink_broadcast], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined, dim=-1, dtype=combined.dtype)
    probs = probs[..., :-1]
    hf_out = torch.matmul(probs, v)

    # Our op
    ours = ops.sdpa(q, k, v, attn_mask=causal_mask, scale=scale, sinks=sinks)
    diff = (hf_out - ours).abs().max().item()
    assert torch.allclose(hf_out, ours, atol=5e-7), (
        f"sinks max_abs_diff={diff:.6e}"
    )


def test_sdpa_sinks_reduces_to_sdpa_without_sinks_when_sink_is_neg_inf():
    """If sinks → -inf, the sink slot's softmax weight goes to 0, so the
    output should match plain sdpa."""
    B, Hq, S, T, Dh = 1, 2, 4, 4, 8
    torch.manual_seed(1)
    q = torch.randn(B, Hq, S, Dh)
    k = torch.randn(B, Hq, T, Dh)
    v = torch.randn(B, Hq, T, Dh)
    scale = Dh ** -0.5
    causal_mask = torch.triu(
        torch.full((S, T), float("-inf"), dtype=q.dtype), diagonal=1,
    )[None, None, :, :]
    plain = ops.sdpa(q, k, v, attn_mask=causal_mask, scale=scale)
    # Use very negative sink so its softmax contribution vanishes (numerically).
    sinks = torch.full((Hq,), -1e9, dtype=q.dtype)
    with_sinks = ops.sdpa(q, k, v, attn_mask=causal_mask, scale=scale,
                          sinks=sinks)
    diff = (plain - with_sinks).abs().max().item()
    assert torch.allclose(plain, with_sinks, atol=1e-5), (
        f"sinks=-inf max_abs_diff={diff:.6e}"
    )


def test_attention_module_allocates_sinks_parameter():
    spec = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=8,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SINK,
        n_sink_tokens=1,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = api_attention.Attention(spec, hidden_size=32, max_seq=16,
                                   dtype=torch.float32)
    assert hasattr(attn, "sinks")
    assert attn.sinks is not None
    assert attn.sinks.shape == (4,)
    # The sinks parameter is a learnable Parameter (not a buffer)
    assert any(p is attn.sinks for p in attn.parameters())


def test_attention_module_forward_with_sinks():
    """End-to-end forward through Attention with mask_kind=SINK."""
    torch.manual_seed(2)
    spec = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=8,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SINK,
        n_sink_tokens=1,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = api_attention.Attention(spec, hidden_size=32, max_seq=16,
                                   dtype=torch.float32)
    # Init sinks so they're not all zero (which would tie the test to
    # default-init behavior).
    with torch.no_grad():
        attn.sinks.normal_(mean=0.0, std=0.1)
    attn.eval()
    cache = kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=1, n_kv_heads=4, head_dim=8, max_seq=16,
    )
    x = torch.randn(1, 5, 32)
    pos = torch.arange(5)
    with torch.no_grad():
        y_sinks = attn(x, position_ids=pos, cache=cache, start_pos=0)

    # Compare against the same attention WITHOUT sinks. Build a parallel
    # spec without n_sink_tokens, copy weights, run, and confirm the
    # outputs differ (proves sinks impact the forward).
    spec_no = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=8,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn_no = api_attention.Attention(spec_no, hidden_size=32, max_seq=16,
                                       dtype=torch.float32)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if name == "sinks":
                continue
            target = dict(attn_no.named_parameters())[name]
            target.copy_(p)
    cache2 = kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=1, n_kv_heads=4, head_dim=8, max_seq=16,
    )
    with torch.no_grad():
        y_no = attn_no(x, position_ids=pos, cache=cache2, start_pos=0)
    diff = (y_sinks - y_no).abs().max().item()
    assert diff > 1e-4, f"sinks had no effect: max_abs_diff={diff:.6e}"


def test_attention_rejects_n_sink_tokens_gt_1():
    spec = specs.AttentionSpec(
        n_q_heads=4, n_kv_heads=4, head_dim=8,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SINK,
        n_sink_tokens=4,                           # > 1 not supported
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    with pytest.raises(NotImplementedError, match="n_sink_tokens=1"):
        api_attention.Attention(spec, hidden_size=32, max_seq=16)
