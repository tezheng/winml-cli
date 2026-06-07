"""Sub-op isolation tests for Qwen2.5-VL's M-RoPE.

We don't need the full HF model checkpoint to verify the M-RoPE op —
the apply_multimodal_rotary_pos_emb function is pure stateless math.
Bit-exactness here is necessary for the LM-decoder gate to hold.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    apply_multimodal_rotary_pos_emb,
)

from api import ops  # noqa: E402


def _build_cos_sin(B, S, head_dim, mrope_section, rope_theta):
    """Replicate Qwen2_5_VLRotaryEmbedding.forward for synthetic positions."""
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    pos_t = torch.arange(S, dtype=torch.float32) + 1
    pos_h = torch.arange(S, dtype=torch.float32) + 100
    pos_w = torch.arange(S, dtype=torch.float32) + 200
    position_ids = (
        torch.stack([pos_t, pos_h, pos_w], dim=0)[:, None, :]
        .expand(3, B, S).long()
    )
    inv_freq_exp = inv_freq[None, None, :, None].expand(3, B, -1, 1)
    pos_exp = position_ids[:, :, None, :].float()
    freqs = (inv_freq_exp @ pos_exp).transpose(2, 3)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().float()
    sin = emb.sin().float()
    return cos, sin


def test_mrope_bit_exact_vs_hf():
    """rope_apply_mrope must produce identical tensors to
    apply_multimodal_rotary_pos_emb on the same cos/sin inputs."""
    torch.manual_seed(0)
    mrope_section = [16, 24, 24]
    B, S, H, Dh = 1, 8, 4, 128
    rope_theta = 1_000_000.0

    q_bhsd = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k_bhsd = torch.randn(B, H, S, Dh, dtype=torch.float32)
    cos, sin = _build_cos_sin(B, S, Dh, mrope_section, rope_theta)

    q_hf, k_hf = apply_multimodal_rotary_pos_emb(
        q_bhsd, k_bhsd, cos, sin, mrope_section, unsqueeze_dim=1,
    )

    # Our op uses [B, S, H, Dh] layout.
    q_bshd = q_bhsd.transpose(1, 2).contiguous()
    k_bshd = k_bhsd.transpose(1, 2).contiguous()
    q_ours, k_ours = ops.rope_apply_mrope(
        q_bshd, k_bshd, cos, sin, tuple(mrope_section),
    )

    # Transpose ours back to [B, H, S, Dh] for diff.
    diff_q = (q_hf - q_ours.transpose(1, 2)).abs().max().item()
    diff_k = (k_hf - k_ours.transpose(1, 2)).abs().max().item()
    assert diff_q == 0.0, f"q M-RoPE mismatch: {diff_q}"
    assert diff_k == 0.0, f"k M-RoPE mismatch: {diff_k}"


def test_mrope_reduces_to_1d_when_axes_equal():
    """When T/H/W positions are all equal (text-only), M-RoPE produces
    the same result as standard 1D RoPE on the same positions."""
    torch.manual_seed(0)
    mrope_section = [16, 24, 24]
    B, S, H, Dh = 1, 6, 4, 128
    rope_theta = 1_000_000.0

    q = torch.randn(B, S, H, Dh, dtype=torch.float32)
    k = torch.randn(B, S, H, Dh, dtype=torch.float32)

    # cos/sin via M-RoPE path, equal T/H/W.
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, Dh, 2).float() / Dh))
    pos = torch.arange(S).float()
    position_ids = pos[None, None, :].expand(3, B, S).long()
    inv_freq_exp = inv_freq[None, None, :, None].expand(3, B, -1, 1)
    pos_exp = position_ids[:, :, None, :].float()
    freqs = (inv_freq_exp @ pos_exp).transpose(2, 3)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().float()
    sin = emb.sin().float()

    q_m, k_m = ops.rope_apply_mrope(q, k, cos, sin, tuple(mrope_section))

    # Standard 1D RoPE — cos/sin from axis 0 (equal to 1, 2).
    q_1d, k_1d = ops.rope_apply(
        q, k, cos[0, 0], sin[0, 0], basis="split_half",
    )
    assert torch.allclose(q_m, q_1d, atol=0.0, rtol=0.0)
    assert torch.allclose(k_m, k_1d, atol=0.0, rtol=0.0)


def test_mrope_differs_from_1d_when_axes_distinct():
    """Sanity: when T/H/W are distinct, M-RoPE output materially differs
    from any single 1D RoPE applied with one of the axes."""
    torch.manual_seed(1)
    mrope_section = [16, 24, 24]
    B, S, H, Dh = 1, 4, 4, 128
    rope_theta = 1_000_000.0

    q = torch.randn(B, S, H, Dh)
    k = torch.randn(B, S, H, Dh)
    cos, sin = _build_cos_sin(B, S, Dh, mrope_section, rope_theta)
    q_m, k_m = ops.rope_apply_mrope(q, k, cos, sin, tuple(mrope_section))

    q_1d, _ = ops.rope_apply(
        q, k, cos[0, 0], sin[0, 0], basis="split_half",
    )
    # Output should be materially different from any 1D variant.
    assert (q_m - q_1d).abs().max().item() > 0.05
