"""Mamba-2 weight loader tests with a synthetic state dict (no HF download)."""
import pytest
import torch

from models.mamba2 import config, layer


def _small_cfg(use_bias=False, use_conv_bias=True):
    return config.Mamba2Config(
        hidden_size=64, num_heads=4, head_dim=32, state_size=16,
        conv_kernel=4, expand=2, n_groups=2, num_hidden_layers=4,
        layer_norm_epsilon=1e-5, use_bias=use_bias,
        use_conv_bias=use_conv_bias,
        residual_in_fp32=True, time_step_min=0.001, time_step_max=0.1,
        time_step_floor=1e-4, time_step_limit=(0.0, float("inf")),
        chunk_size=4, vocab_size=100, tie_word_embeddings=False,
        dtype=torch.float32,
    )


def _make_synthetic_hf_state_dict(blk, layer_idx=0):
    """Build a fake HF state dict from the block's actual tensors.

    Mirrors the names produced by `Mamba2Model.state_dict()`.
    """
    L = layer_idx
    prefix = f"backbone.layers.{L}"
    mp = f"{prefix}.mixer"
    sd: dict[str, torch.Tensor] = {
        f"{prefix}.norm.weight":            torch.randn_like(blk.pre_attn_norm.weight),
        f"{mp}.in_proj.weight":             torch.randn_like(blk.attention.in_proj.weight),
        f"{mp}.conv1d.weight":              torch.randn_like(blk.attention.conv1d.weight),
        f"{mp}.dt_bias":                    torch.randn_like(blk.attention.dt_bias),
        f"{mp}.A_log":                      torch.randn_like(blk.attention.A_log),
        f"{mp}.D":                          torch.randn_like(blk.attention.D),
        f"{mp}.norm.weight":                torch.randn_like(blk.attention.norm.weight),
        f"{mp}.out_proj.weight":            torch.randn_like(blk.attention.out_proj.weight),
    }
    if blk.attention.in_proj.bias is not None:
        sd[f"{mp}.in_proj.bias"] = torch.randn_like(blk.attention.in_proj.bias)
    if blk.attention.conv1d.bias is not None:
        sd[f"{mp}.conv1d.bias"] = torch.randn_like(blk.attention.conv1d.bias)
    if blk.attention.out_proj.bias is not None:
        sd[f"{mp}.out_proj.bias"] = torch.randn_like(blk.attention.out_proj.bias)
    return sd


def test_load_synthetic_state_dict_no_bias():
    cfg = _small_cfg(use_bias=False, use_conv_bias=True)
    blk = layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    sd = _make_synthetic_hf_state_dict(blk, layer_idx=0)
    layer.load_hf_mamba2_layer(blk, sd, layer_idx=0)
    # Confirm a few tensors were copied (random sources are unique).
    assert torch.allclose(blk.pre_attn_norm.weight, sd["backbone.layers.0.norm.weight"])
    assert torch.allclose(blk.attention.A_log, sd["backbone.layers.0.mixer.A_log"])
    assert torch.allclose(blk.attention.D, sd["backbone.layers.0.mixer.D"])
    assert torch.allclose(blk.attention.conv1d.weight,
                          sd["backbone.layers.0.mixer.conv1d.weight"])
    assert torch.allclose(blk.attention.out_proj.weight,
                          sd["backbone.layers.0.mixer.out_proj.weight"])


def test_load_synthetic_state_dict_with_conv_bias():
    cfg = _small_cfg(use_bias=False, use_conv_bias=True)
    blk = layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    sd = _make_synthetic_hf_state_dict(blk, layer_idx=0)
    # Ensure conv1d.bias key is present and populated
    assert "backbone.layers.0.mixer.conv1d.bias" in sd
    layer.load_hf_mamba2_layer(blk, sd, layer_idx=0)
    assert torch.allclose(blk.attention.conv1d.bias,
                          sd["backbone.layers.0.mixer.conv1d.bias"])


def test_load_rejects_missing_tensor():
    cfg = _small_cfg()
    blk = layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    sd = _make_synthetic_hf_state_dict(blk, layer_idx=0)
    del sd["backbone.layers.0.mixer.A_log"]
    with pytest.raises(KeyError, match="missing tensors"):
        layer.load_hf_mamba2_layer(blk, sd, layer_idx=0)


def test_load_then_forward_runs():
    """After loading random weights, the layer still produces a finite output."""
    cfg = _small_cfg()
    blk = layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    sd = _make_synthetic_hf_state_dict(blk, layer_idx=0)
    # Make weights small to keep softplus / exp tame.
    for v in sd.values():
        v.mul_(0.1)
    layer.load_hf_mamba2_layer(blk, sd, layer_idx=0)
    blk.eval()
    x = torch.randn(1, 8, cfg.hidden_size)
    with torch.no_grad():
        y = blk(x)
    assert y.shape == (1, 8, cfg.hidden_size)
    assert torch.isfinite(y).all()
