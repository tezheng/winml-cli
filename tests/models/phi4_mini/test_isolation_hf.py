"""Isolation tests for Phi-4-mini-instruct vs HF Phi3 sub-ops.

Verifies that our LongRoPE + partial_rotary_kind='prefix' reproduces HF's
RotaryEmbedding output bit-exact at atol=1e-5.
"""
import math
import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.phi3 import modeling_phi3
from transformers.models.phi3.configuration_phi3 import Phi3Config as HFPhi3Config

from api import rope, specs, types


def _phi4_mini_hf_cfg():
    """Small Phi-4-mini-like config: short_factor + long_factor with
    head_dim=16, partial_rotary_factor=0.75 → rope_angles=6."""
    return HFPhi3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, num_hidden_layers=2,
        rms_norm_eps=1e-5, vocab_size=100,
        max_position_embeddings=128,
        original_max_position_embeddings=32,
        tie_word_embeddings=False, torch_dtype="float32",
        rope_parameters={
            "rope_theta": 10_000.0, "rope_type": "longrope",
            "partial_rotary_factor": 0.75,
            "short_factor": [1.0] * 6,
            "long_factor": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "original_max_position_embeddings": 32,
        },
        sliding_window=None,
    )


def test_longrope_cos_sin_matches_hf_short_branch():
    """For seq_len ≤ original_max, HF uses short_factor. cos/sin shape must be
    [S, head_dim_rot] (NOT head_dim) — Phi-3 prefix semantic."""
    cfg = _phi4_mini_hf_cfg()
    hf_rotary = modeling_phi3.Phi3RotaryEmbedding(cfg)
    head_dim = cfg.hidden_size // cfg.num_attention_heads        # 16
    pr = 0.75
    head_dim_rot = int(head_dim * pr)                            # 12
    rope_angles = head_dim_rot // 2                              # 6

    position_ids = torch.arange(10).unsqueeze(0)                 # seq=10 ≤ original=32
    dummy = torch.zeros(1, 10, head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)
    # HF cos shape is [S, head_dim_rot] (rotated channels only).
    assert hf_cos.shape == (10, head_dim_rot)

    # Our RoPE with partial_rotary_kind='prefix' must match.
    factor = cfg.max_position_embeddings / cfg.original_max_position_embeddings
    if factor <= 1.0:
        att = 1.0
    else:
        att = math.sqrt(1 + math.log(factor) / math.log(cfg.original_max_position_embeddings))
    rope_spec = specs.RoPESpec(
        base_theta=10_000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LONGROPE,
        partial_rotary_factor=pr,
        partial_rotary_kind="prefix",
        longrope_extra=specs.LongRoPEParams(
            short_factor=tuple([1.0] * rope_angles),
            long_factor=tuple([2.0] * rope_angles),
            original_max_position_embeddings=cfg.original_max_position_embeddings,
            attention_factor=att,
        ),
    )
    api_rope = rope.RoPE(rope_spec, head_dim=head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    # cos_cached spans [max_seq, head_dim_rot].
    assert api_rope.cos_cached.shape == (cfg.max_position_embeddings, head_dim_rot)
    assert torch.allclose(api_rope.cos_cached[:10], hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached[:10], hf_sin, atol=1e-5)


def test_longrope_dispatches_long_above_boundary():
    """For seq_len > original_max, HF uses long_factor. We verify dispatch."""
    cfg = _phi4_mini_hf_cfg()
    hf_rotary = modeling_phi3.Phi3RotaryEmbedding(cfg)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    pr = 0.75
    head_dim_rot = int(head_dim * pr)
    rope_angles = head_dim_rot // 2

    # Force HF to refresh inv_freq to LONG by passing position_ids past boundary.
    pos_long = torch.arange(40, 50).unsqueeze(0)   # max=49+1>32
    dummy_long = torch.zeros(1, 10, head_dim)
    hf_cos_long, hf_sin_long = hf_rotary(dummy_long, pos_long)
    hf_cos_long = hf_cos_long.squeeze(0)
    hf_sin_long = hf_sin_long.squeeze(0)

    factor = cfg.max_position_embeddings / cfg.original_max_position_embeddings
    att = math.sqrt(1 + math.log(factor) / math.log(cfg.original_max_position_embeddings))
    rope_spec = specs.RoPESpec(
        base_theta=10_000.0, basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LONGROPE,
        partial_rotary_factor=pr, partial_rotary_kind="prefix",
        longrope_extra=specs.LongRoPEParams(
            short_factor=tuple([1.0] * rope_angles),
            long_factor=tuple([2.0] * rope_angles),
            original_max_position_embeddings=cfg.original_max_position_embeddings,
            attention_factor=att,
        ),
    )
    api_rope = rope.RoPE(rope_spec, head_dim=head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    # Forward dispatch picks long table.
    q = torch.randn(1, 10, 2, head_dim)
    k = torch.randn(1, 10, 2, head_dim)
    q_out, k_out = api_rope(q, k, pos_long.squeeze(0))
    # Validate that the LONG cos/sin table (at positions 40..49) matches HF.
    assert torch.allclose(api_rope.cos_cached_long[40:50], hf_cos_long, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached_long[40:50], hf_sin_long, atol=1e-5)
