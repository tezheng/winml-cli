"""Mamba-2 layer shape tests (no HF download)."""
import pytest
import torch

from api import kvcache
from models.mamba2 import config, layer


def _small_cfg():
    """A small Mamba-2 config that satisfies all invariants."""
    return config.Mamba2Config(
        hidden_size=128, num_heads=8, head_dim=32, state_size=16,
        conv_kernel=4, expand=2, n_groups=2, num_hidden_layers=4,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        residual_in_fp32=True, time_step_min=0.001, time_step_max=0.1,
        time_step_floor=1e-4, time_step_limit=(0.0, float("inf")),
        chunk_size=8, vocab_size=1000, tie_word_embeddings=False,
        dtype=torch.float32,
    )


def _ssd_2p7b_cfg():
    """state-spaces/mamba2-2.7b shape (for shape-only assertions)."""
    return config.Mamba2Config(
        hidden_size=2560, num_heads=80, head_dim=64, state_size=128,
        conv_kernel=4, expand=2, n_groups=1, num_hidden_layers=64,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        residual_in_fp32=True, time_step_min=0.001, time_step_max=0.1,
        time_step_floor=1e-4, time_step_limit=(0.0, float("inf")),
        chunk_size=256, vocab_size=50288, tie_word_embeddings=False,
        dtype=torch.float32,
    )


def test_mamba2_layer_forward_shape_small():
    cfg = _small_cfg()
    blk = layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    blk.eval()
    B, S = 2, 10  # 10 not a multiple of chunk_size=8 → exercises pad
    x = torch.randn(B, S, cfg.hidden_size)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == (B, S, cfg.hidden_size)


def test_mamba2_layer_forward_with_state_cache():
    cfg = _small_cfg()
    blk = layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    blk.eval()
    B, S = 1, 8
    mixer = blk.attention
    cache = kvcache.SSMStateCache(
        batch_size=B, conv_dim=mixer.conv_dim,
        conv_kernel=mixer.conv_kernel,
        n_heads=mixer.num_heads, head_dim=mixer.head_dim,
        d_state=mixer.d_state,
    )
    x = torch.randn(B, S, cfg.hidden_size)
    with torch.no_grad():
        y = blk(x, cache=cache)
    assert y.shape == (B, S, cfg.hidden_size)
    assert cache.has_previous_state
    assert cache.ssm_state.abs().sum().item() > 0


def test_mamba2_2p7b_shape_constructs():
    """Shape-only check that the 2.7B production config builds without crashing."""
    cfg = _ssd_2p7b_cfg()
    # Don't actually instantiate the full block — that allocates ~50M params.
    # Instead, just validate the to_block_spec invariants pass.
    block_spec = cfg.to_block_spec()
    ssd = block_spec.token_mixer
    assert ssd.n_heads * ssd.headdim == cfg.d_inner == 5120
    assert ssd.n_heads % ssd.ngroups == 0


def test_mamba2_layer_determinism():
    cfg = _small_cfg()
    blk = layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    blk.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, cfg.hidden_size)
    with torch.no_grad():
        y1 = blk(x)
        y2 = blk(x)
    assert torch.allclose(y1, y2, atol=0)
