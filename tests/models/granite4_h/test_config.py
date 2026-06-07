"""Granite-4 H config adapter tests."""
import torch

from api import specs, types
from models.granite4_h import config


def test_granite4_h_config_from_hf_dict():
    hf = {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "shared_intermediate_size": 8192,
        "num_hidden_layers": 40,
        "vocab_size": 100096,
        "layers_block_type": ["mamba"] * 5 + ["attention"] + ["mamba"] * 5 + ["attention"] +
                             ["mamba"] * 5 + ["attention"] + ["mamba"] * 5 + ["attention"] +
                             ["mamba"] * 5 + ["attention"] + ["mamba"] * 5 + ["attention"] +
                             ["mamba"] * 4,
        "mamba_n_heads": 64,
        "mamba_n_groups": 1,
        "mamba_d_state": 128,
        "mamba_d_head": 64,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "mamba_chunk_size": 256,
        "embedding_multiplier": 12,
        "logits_scaling": 8,
        "residual_multiplier": 0.22,
        "attention_multiplier": 0.015625,
        "position_embedding_type": "nope",
        "num_local_experts": 0,
    }
    cfg = config.Granite4HConfig.from_hf_dict(hf)
    assert cfg.hidden_size == 2048
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.mamba_n_heads == 64
    assert cfg.mamba_d_head == 64
    assert cfg.embedding_multiplier == 12.0
    assert cfg.residual_multiplier == 0.22
    assert cfg.attention_multiplier == 0.015625
    assert cfg.is_mamba_layer(0)
    assert not cfg.is_mamba_layer(5)
    assert cfg.is_mamba_layer(6)


def test_to_block_spec_mamba_layer():
    hf = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "shared_intermediate_size": 256,
        "num_hidden_layers": 4,
        "vocab_size": 100,
        "layers_block_type": ["mamba", "mamba", "attention", "mamba"],
        "mamba_n_heads": 4,
        "mamba_d_state": 16,
        "mamba_d_head": 64,                 # 4 heads * 64 = 256 = 128*2
        "mamba_n_groups": 2,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "mamba_chunk_size": 8,
        "residual_multiplier": 0.5,
        "attention_multiplier": 0.25,
        "position_embedding_type": "nope",
    }
    cfg = config.Granite4HConfig.from_hf_dict(hf)
    bs0 = cfg.to_block_spec(0)             # mamba
    assert isinstance(bs0.token_mixer, specs.SSDSpec)
    assert bs0.token_mixer.n_heads == 4
    assert bs0.token_mixer.headdim == 64
    assert bs0.residual_scale == 0.5
    assert bs0.skip_ffn is False           # always has shared_mlp
    assert bs0.channel_mixer.fused_gate_up is True


def test_to_block_spec_attention_layer_no_rope():
    hf = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "shared_intermediate_size": 256,
        "num_hidden_layers": 4,
        "vocab_size": 100,
        "layers_block_type": ["mamba", "mamba", "attention", "mamba"],
        "mamba_n_heads": 4, "mamba_d_state": 16, "mamba_d_head": 64,
        "mamba_n_groups": 2, "mamba_d_conv": 4, "mamba_expand": 2,
        "mamba_chunk_size": 8,
        "attention_multiplier": 0.25,
        "position_embedding_type": "nope",
    }
    cfg = config.Granite4HConfig.from_hf_dict(hf)
    bs2 = cfg.to_block_spec(2)             # attention
    assert isinstance(bs2.token_mixer, specs.AttentionSpec)
    assert bs2.token_mixer.attn_scale == 0.25   # attention_multiplier
    assert bs2.token_mixer.rope is None
    assert bs2.token_mixer.n_q_heads == 4
    assert bs2.token_mixer.n_kv_heads == 2
    assert bs2.skip_ffn is False
