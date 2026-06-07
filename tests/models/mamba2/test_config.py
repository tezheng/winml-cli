"""Mamba-2 config adapter tests."""
import torch

from api import specs, types
from models.mamba2 import config


def test_mamba2_config_from_hf_dict_defaults():
    hf = {
        "hidden_size": 2560,
        "num_heads": 80,
        "head_dim": 64,
        "state_size": 128,
        "conv_kernel": 4,
        "expand": 2,
        "n_groups": 1,
        "num_hidden_layers": 64,
        "vocab_size": 50288,
    }
    cfg = config.Mamba2Config.from_hf_dict(hf)
    assert cfg.hidden_size == 2560
    assert cfg.num_heads == 80
    assert cfg.head_dim == 64
    assert cfg.state_size == 128
    assert cfg.expand == 2
    assert cfg.n_groups == 1
    assert cfg.chunk_size == 256  # default
    assert cfg.layer_norm_epsilon == 1e-5
    assert cfg.residual_in_fp32 is True
    assert cfg.tie_word_embeddings is False
    assert cfg.d_inner == 5120        # 2560 * 2


def test_mamba2_to_block_spec_carries_ssd_spec_with_skip_ffn():
    cfg = config.Mamba2Config(
        hidden_size=128, num_heads=8, head_dim=32, state_size=16,
        conv_kernel=4, expand=2, n_groups=2, num_hidden_layers=2,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        residual_in_fp32=True, time_step_min=0.001, time_step_max=0.1,
        time_step_floor=1e-4, time_step_limit=(0.0, float("inf")),
        chunk_size=8, vocab_size=1000, tie_word_embeddings=False,
        dtype=torch.float32,
    )
    block_spec = cfg.to_block_spec(layer_idx=0)
    assert isinstance(block_spec.token_mixer, specs.SSDSpec)
    assert block_spec.skip_ffn is True
    ssd = block_spec.token_mixer
    assert ssd.n_heads == 8
    assert ssd.headdim == 32
    assert ssd.ngroups == 2
    assert ssd.chunk_size == 8
    assert ssd.base.d_state == 16
    assert ssd.base.d_conv == 4
    assert ssd.base.d_inner == 256          # 128 * 2
    assert block_spec.pre_attn_norm.kind == types.NormKind.RMS
    assert block_spec.attn_norm_position == types.NormPosition.PRE


def test_mamba2_time_step_limit_passthrough():
    hf = {
        "hidden_size": 128, "num_heads": 4, "head_dim": 64,
        "state_size": 16, "conv_kernel": 4, "expand": 2, "n_groups": 1,
        "num_hidden_layers": 1, "vocab_size": 100,
        "time_step_limit": [0.001, 100.0],
    }
    cfg = config.Mamba2Config.from_hf_dict(hf)
    assert cfg.time_step_limit == (0.001, 100.0)
