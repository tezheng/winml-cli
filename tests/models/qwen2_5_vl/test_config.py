"""Qwen2.5-VL config tests."""
from __future__ import annotations

import torch

from api import specs, types
from models.qwen2_5_vl import config as q_config


_HF_QWEN2_5_VL_3B = {
    "model_type": "qwen2_5_vl",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen2_5_vl_text",
        "vocab_size": 151936,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 128000,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "type": "mrope",
            "mrope_section": [16, 24, 24],
            "rope_theta": 1_000_000.0,
            "rope_type": "default",
        },
        "use_sliding_window": False,
        "sliding_window": None,
    },
}


def test_from_hf_dict_parses_mrope_section():
    cfg = q_config.Qwen2_5_VLConfig.from_hf_dict(_HF_QWEN2_5_VL_3B)
    assert cfg.hidden_size == 2048
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2                # GQA
    assert cfg.head_dim == 128                         # 2048 // 16
    assert cfg.intermediate_size == 11008
    assert cfg.num_hidden_layers == 36
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.mrope_section == (16, 24, 24)
    assert sum(cfg.mrope_section) * 2 == cfg.head_dim  # invariant
    assert cfg.tie_word_embeddings is False
    assert cfg.sliding_window is None


def test_to_block_spec_carries_mrope_section():
    cfg = q_config.Qwen2_5_VLConfig.from_hf_dict(_HF_QWEN2_5_VL_3B)
    spec = cfg.to_block_spec(layer_idx=0)
    attn = spec.token_mixer
    assert isinstance(attn, specs.AttentionSpec)
    assert attn.kind == types.AttentionKind.STANDARD
    assert attn.q_bias is True and attn.k_bias is True and attn.v_bias is True
    assert attn.o_bias is False
    assert attn.qk_norm is None
    assert attn.rope is not None
    assert attn.rope.mrope_section == (16, 24, 24)
    assert attn.rope.basis == types.RoPEBasis.SPLIT_HALF
    assert attn.rope.scaling == types.RoPEScaling.NONE


def test_to_block_spec_ffn_no_biases():
    cfg = q_config.Qwen2_5_VLConfig.from_hf_dict(_HF_QWEN2_5_VL_3B)
    spec = cfg.to_block_spec(layer_idx=0)
    ffn = spec.channel_mixer
    assert isinstance(ffn, specs.FFNSpec)
    assert ffn.gate_kind == types.GateKind.SWIGLU
    assert ffn.gate_bias is False
    assert ffn.up_bias is False
    assert ffn.down_bias is False
    assert ffn.intermediate_size == 11008
