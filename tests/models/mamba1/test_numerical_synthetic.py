"""Mamba-1 mixer numerical gate (v6 B1).

Tests api.ssm.Mamba1Mixer.forward against HF MambaMixer.slow_forward at
atol=5e-4. Uses synthetic (tiny) configs so the test runs without
downloading the public `state-spaces/mamba-130m-hf` checkpoint.

The HF `slow_forward` is the same sequential time-step recurrence that
api.ops.selective_scan_mamba1 implements, so this is a direct path
match (not a kernel comparison).

Source: `transformers/models/mamba/modeling_mamba.py:270-363`.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.mamba.configuration_mamba import MambaConfig  # noqa: E402
from transformers.models.mamba.modeling_mamba import MambaMixer  # noqa: E402

from api import ssm  # noqa: E402
from models.mamba1 import config as m1_config, layer as m1_layer  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _build_hf_mixer(hidden_size: int, num_layers: int = 1):
    cfg = MambaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        vocab_size=128,
        state_size=8,                   # small for test speed
        conv_kernel=4,
        expand=2,
        use_bias=False,
        use_conv_bias=True,
        use_mambapy=False,              # force slow_forward path
        use_associative_scan=False,
        time_step_rank="auto",
        hidden_act="silu",
        layer_norm_epsilon=1e-5,
    )
    torch.manual_seed(0)
    mixer = MambaMixer(cfg, layer_idx=0, initialize_mixer_weights=True)
    mixer.eval()
    return cfg, mixer


def _build_api_mixer(hidden_size: int):
    cfg = m1_config.Mamba1Config(
        hidden_size=hidden_size, state_size=8, conv_kernel=4, expand=2,
        num_hidden_layers=1, vocab_size=128,
    )
    spec = cfg.to_ssm_spec()
    return ssm.Mamba1Mixer(spec, hidden_size=hidden_size, dtype=torch.float32)


def _copy_weights(hf: MambaMixer, api: ssm.Mamba1Mixer) -> None:
    with torch.no_grad():
        api.in_proj.weight.copy_(hf.in_proj.weight)
        api.conv1d.weight.copy_(hf.conv1d.weight)
        if api.conv1d.bias is not None and hf.conv1d.bias is not None:
            api.conv1d.bias.copy_(hf.conv1d.bias)
        api.x_proj.weight.copy_(hf.x_proj.weight)
        api.dt_proj.weight.copy_(hf.dt_proj.weight)
        api.dt_proj.bias.copy_(hf.dt_proj.bias)
        api.A_log.copy_(hf.A_log)
        api.D.copy_(hf.D)
        api.out_proj.weight.copy_(hf.out_proj.weight)


def test_mamba1_mixer_matches_hf_slow_forward():
    """Single-layer Mamba-1 mixer; eval slow_forward path."""
    hidden_size = 96
    cfg, hf = _build_hf_mixer(hidden_size)
    api = _build_api_mixer(hidden_size)
    _copy_weights(hf, api)
    api.eval()
    torch.manual_seed(1)
    B, S = 2, 12
    x = torch.randn(B, S, hidden_size)
    with torch.no_grad():
        hf_out = hf.slow_forward(x, cache_params=None)
        api_out = api(x)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Mamba-1 mixer max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    print(f"Mamba-1 mixer max_abs_diff={max_abs_diff:.2e}")


def test_mamba1_mixer_longer_sequence():
    """Longer S exercises the sequential loop."""
    hidden_size = 64
    cfg, hf = _build_hf_mixer(hidden_size)
    api = _build_api_mixer(hidden_size)
    _copy_weights(hf, api)
    api.eval()
    torch.manual_seed(2)
    B, S = 1, 32
    x = torch.randn(B, S, hidden_size)
    with torch.no_grad():
        hf_out = hf.slow_forward(x, cache_params=None)
        api_out = api(x)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Mamba-1 mixer S=32 max_abs_diff={max_abs_diff:.6e}"
    )


def test_mamba1_decoder_block_assembles():
    """Smoke test: build a Mamba-1 DecoderBlock end-to-end."""
    cfg = m1_config.Mamba1Config(
        hidden_size=64, state_size=8, conv_kernel=4, expand=2,
        num_hidden_layers=1, vocab_size=128,
    )
    blk = m1_layer.build_mamba1_decoder_layer(cfg)
    assert blk.feedforward is None   # skip_ffn=True
    assert blk._is_ssm_block
    torch.manual_seed(3)
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        out = blk(x)
    assert out.shape == (1, 8, 64)


def test_mamba1_spec_kind_validation():
    """Mamba1Mixer rejects spec.kind != MAMBA1."""
    from api import specs, types
    spec = specs.SSMSpec(
        d_state=8, d_conv=4, d_inner=64,
        kind=types.SSMKind.MAMBA2_SSD, dt_rank=4,
    )
    with pytest.raises(ValueError, match="MAMBA1"):
        ssm.Mamba1Mixer(spec, hidden_size=32)
