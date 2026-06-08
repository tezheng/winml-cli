"""Parallel residual block layout (v5-phase2 V2 — Falcon-7B / Cohere).

The PARALLEL layout computes
    shared = pre_attn_norm(x)
    y = x + attn(shared) + ffn(shared)
instead of the standard SEQUENTIAL
    y1 = x + attn(pre_attn_norm(x))
    y  = y1 + ffn(pre_ffn_norm(y1))

This is a topological change in the residual graph — the two MUST
produce different outputs on the same weights (modulo trivial
degenerate cases).

Source: `transformers/models/falcon/modeling_falcon.py:594-634`
(`parallel_attn=True, num_ln_in_parallel_attn=1`).
"""
from __future__ import annotations

import pytest
import torch

from api import block, kvcache, specs, types


def _baseline_attn_spec() -> specs.AttentionSpec:
    """Llama-shape MQA + RoPE (mirrors Falcon-7B's attention)."""
    return specs.AttentionSpec(
        n_q_heads=4,
        n_kv_heads=1,                       # Falcon-7B is MQA
        head_dim=16,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )


def _ffn_spec() -> specs.FFNSpec:
    return specs.FFNSpec(
        intermediate_size=128,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
        fused_gate_up=False,
        gate_bias=False, up_bias=False, down_bias=False,
    )


def _norm_spec() -> specs.NormSpec:
    return specs.NormSpec(kind=types.NormKind.RMS, eps=1e-5,
                          weight_mode=types.NormWeightMode.STANDARD_W)


def _make_block(layout: types.BlockLayout) -> block.DecoderBlock:
    norm_spec = _norm_spec()
    if layout == types.BlockLayout.PARALLEL:
        # PARALLEL: only one shared pre-norm (no pre_ffn_norm).
        spec = specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=_baseline_attn_spec(),
            channel_mixer=_ffn_spec(),
            pre_attn_norm=norm_spec,
            pre_ffn_norm=None,
            block_layout=layout,
        )
    else:
        spec = specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=_baseline_attn_spec(),
            channel_mixer=_ffn_spec(),
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
            block_layout=layout,
        )
    blk = block.DecoderBlock(spec=spec, hidden_size=64, max_seq=32,
                             dtype=torch.float32)
    return blk


def _build_cache(B: int, n_kv: int = 1, Dh: int = 16, S: int = 32):
    return kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=B, n_kv_heads=n_kv, head_dim=Dh, max_seq=S,
    )


def test_parallel_differs_from_sequential():
    """Given identical weights and input, the PARALLEL output must
    differ from the SEQUENTIAL output."""
    seq = _make_block(types.BlockLayout.SEQUENTIAL)
    par = _make_block(types.BlockLayout.PARALLEL)
    seq.eval(); par.eval()

    # Copy the SEQUENTIAL weights into PARALLEL block. They share all
    # tensor names except PARALLEL has no `pre_ffn_norm`.
    with torch.no_grad():
        for name, p in seq.named_parameters():
            if name.startswith("pre_ffn_norm"):
                continue
            target = dict(par.named_parameters()).get(name)
            assert target is not None, f"missing {name} in PARALLEL block"
            target.copy_(p)
        for name, b in seq.named_buffers():
            if name.startswith("pre_ffn_norm"):
                continue
            target = dict(par.named_buffers()).get(name)
            if target is not None:
                target.copy_(b)

    torch.manual_seed(1)
    B, S = 1, 6
    x = torch.randn(B, S, 64)

    cache_s = _build_cache(B)
    cache_p = _build_cache(B)
    pos = torch.arange(S)
    with torch.no_grad():
        y_seq = seq(x, position_ids=pos, cache=cache_s, start_pos=0)
        y_par = par(x, position_ids=pos, cache=cache_p, start_pos=0)
    diff = (y_seq - y_par).abs().max().item()
    assert diff > 1e-3, (
        f"PARALLEL and SEQUENTIAL produced identical outputs "
        f"(max_abs_diff={diff:.2e}); the block_layout knob is not wired"
    )


def test_parallel_matches_hand_rolled_reference():
    """The PARALLEL forward must equal
        x + attn(norm(x)) + ffn(norm(x))
    when implemented by hand using the same sub-modules."""
    par = _make_block(types.BlockLayout.PARALLEL)
    par.eval()
    torch.manual_seed(2)
    B, S = 1, 5
    x = torch.randn(B, S, 64)
    cache = _build_cache(B)
    pos = torch.arange(S)
    cache_ref = _build_cache(B)
    with torch.no_grad():
        # Hand-rolled reference
        shared = par.pre_attn_norm(x)
        attn_out = par.attention(shared, position_ids=pos,
                                 cache=cache_ref, start_pos=0)
        ffn_out = par.feedforward(shared)
        ref = x + attn_out + ffn_out
        # Module forward
        out = par(x, position_ids=pos, cache=cache, start_pos=0)
    assert torch.allclose(out, ref, atol=1e-6), (
        f"PARALLEL forward != hand-rolled reference; "
        f"max_abs_diff={(out-ref).abs().max().item():.6e}"
    )


def test_parallel_block_requires_no_pre_ffn_norm():
    """Setting pre_ffn_norm on a PARALLEL block must raise at init."""
    norm_spec = _norm_spec()
    spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=_baseline_attn_spec(),
        channel_mixer=_ffn_spec(),
        pre_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec,                   # invalid on PARALLEL
        block_layout=types.BlockLayout.PARALLEL,
    )
    with pytest.raises(ValueError, match="PARALLEL"):
        block.DecoderBlock(spec=spec, hidden_size=64, max_seq=32,
                           dtype=torch.float32)
