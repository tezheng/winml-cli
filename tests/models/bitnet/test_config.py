"""BitNetConfig sanity checks."""
from __future__ import annotations

import torch

from api import block as api_block, types
from models.bitnet import config as b_config


def test_bitnet_b158_shape():
    cfg = b_config.BitNetConfig.bitnet_b158_2b()
    assert cfg.hidden_size == 2560
    assert cfg.num_attention_heads == 20
    assert cfg.num_key_value_heads == 5
    spec = cfg.to_block_spec()
    assert spec.token_mixer.attn_sub_norm is not None
    assert spec.channel_mixer.ffn_sub_norm is not None
    assert spec.channel_mixer.activation == types.Activation.RELU2
    assert spec.channel_mixer.gate_kind == types.GateKind.SWIGLU


def test_bitnet_block_assembles_and_has_sub_norms():
    cfg = b_config.BitNetConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, num_hidden_layers=1,
        max_position_embeddings=32,
    )
    blk = api_block.DecoderBlock(cfg.to_block_spec(), hidden_size=64,
                                 max_seq=32, dtype=torch.float32)
    blk.eval()
    # attn_sub_norm wired
    assert blk.attention.attn_sub_norm is not None
    assert blk.attention.attn_sub_norm.weight.shape == (64,)
    # ffn_sub_norm wired
    assert blk.feedforward.ffn_sub_norm is not None
    assert blk.feedforward.ffn_sub_norm.weight.shape == (128,)


def test_relu2_activation_in_swiglu():
    """SWIGLU + RELU2 must work end-to-end (BitNet's `act_fn(gate)*up`
    with act_fn = relu2)."""
    from api import feedforward, specs
    spec = specs.FFNSpec(
        intermediate_size=16,
        activation=types.Activation.RELU2,
        gate_kind=types.GateKind.SWIGLU,
    )
    ff = feedforward.FeedForward(spec, hidden_size=8, dtype=torch.float32)
    ff.eval()
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        y = ff(x)
    assert y.shape == (1, 4, 8)
