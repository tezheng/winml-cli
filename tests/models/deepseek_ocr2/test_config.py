"""DeepSeek-OCR-2 config tests."""
from __future__ import annotations

import torch

from api import specs, types
from models.deepseek_ocr2 import config as ds_config


_HF_DS_OCR2_FULL = {
    "model_type": "deepseek_ocr2",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "deepseek_ocr2_text",
        "vocab_size": 129280,
        "hidden_size": 1280,
        "intermediate_size": 6848,
        "num_hidden_layers": 12,
        "num_attention_heads": 10,
        "num_key_value_heads": 10,
        "hidden_act": "silu",
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "attention_bias": False,
        "mlp_bias": False,
        "head_dim": 128,
        "n_group": 1,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "routed_scaling_factor": 1.0,
        "topk_group": 1,
        "topk_method": "greedy",
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 896,
        "mlp_layer_types": [
            "dense", "sparse", "sparse", "sparse", "sparse", "sparse",
            "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
        ],
    },
}


def test_from_hf_dict_parses_text_config():
    cfg = ds_config.DeepseekOcr2Config.from_hf_dict(_HF_DS_OCR2_FULL)
    assert cfg.hidden_size == 1280
    assert cfg.num_attention_heads == 10
    assert cfg.num_key_value_heads == 10           # standard MHA (no GQA)
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 6848
    assert cfg.moe_intermediate_size == 896
    assert cfg.num_hidden_layers == 12
    assert cfg.rope_theta == 10000.0
    assert cfg.n_routed_experts == 64
    assert cfg.n_shared_experts == 2
    assert cfg.num_experts_per_tok == 6
    assert cfg.topk_method == "greedy"
    assert cfg.n_group == 1
    assert cfg.attention_bias is False
    assert cfg.mlp_bias is False
    assert cfg.mlp_layer_types[0] == "dense"
    assert cfg.mlp_layer_types[1] == "sparse"
    assert len(cfg.mlp_layer_types) == 12


def test_layer0_is_dense_ffn():
    cfg = ds_config.DeepseekOcr2Config.from_hf_dict(_HF_DS_OCR2_FULL)
    spec = cfg.to_block_spec(layer_idx=0)
    ffn = spec.channel_mixer
    assert isinstance(ffn, specs.FFNSpec)
    assert ffn.intermediate_size == 6848
    assert ffn.gate_kind == types.GateKind.SWIGLU
    assert ffn.gate_bias is False


def test_layer1_is_moe():
    cfg = ds_config.DeepseekOcr2Config.from_hf_dict(_HF_DS_OCR2_FULL)
    spec = cfg.to_block_spec(layer_idx=1)
    moe = spec.channel_mixer
    assert isinstance(moe, specs.MoESpec)
    assert moe.n_experts == 64
    assert moe.top_k == 6
    assert moe.n_shared_experts == 2
    assert moe.router_kind == "softmax"
    assert moe.router_norm is False                # HF source does NOT renorm
    assert moe.score_correction_bias is False      # NOT sigmoid+bias
    assert moe.group_routing is None               # n_group=1 collapses
    assert moe.routed_scaling_factor == 1.0
    assert moe.expert_ffn is not None
    assert moe.expert_ffn.intermediate_size == 896


def test_attention_spec_is_standard_mha_no_mla():
    """KEY DRIFT: HF DeepSeek-OCR-2 uses STANDARD MHA, NOT MLA."""
    cfg = ds_config.DeepseekOcr2Config.from_hf_dict(_HF_DS_OCR2_FULL)
    spec = cfg.to_block_spec(layer_idx=0)
    attn = spec.token_mixer
    assert isinstance(attn, specs.AttentionSpec)
    assert attn.kind == types.AttentionKind.STANDARD
    assert attn.qkv_layout == types.QKVLayout.SPLIT
    assert attn.mask_kind == types.MaskKind.CAUSAL  # NOT BLOCK_BIDIRECTIONAL
    assert attn.block_bidirectional_mask is False
    assert attn.q_lora_rank is None                 # MLA fields unused
    assert attn.kv_lora_rank is None
    assert attn.qk_nope_head_dim is None
    assert attn.qk_rope_head_dim is None
    assert attn.v_head_dim is None
    assert attn.rope is not None
    assert attn.rope.mrope_section is None          # no M-RoPE either


def test_group_limited_greedy_collapses_when_n_group_1():
    """The `topk_method=group_limited_greedy` only kicks in when n_group > 1."""
    hf2 = dict(_HF_DS_OCR2_FULL)
    hf2["text_config"] = dict(hf2["text_config"])
    hf2["text_config"]["topk_method"] = "group_limited_greedy"
    hf2["text_config"]["n_group"] = 8
    hf2["text_config"]["topk_group"] = 4
    cfg = ds_config.DeepseekOcr2Config.from_hf_dict(hf2)
    spec = cfg.to_block_spec(layer_idx=1)
    moe = spec.channel_mixer
    assert isinstance(moe, specs.MoESpec)
    assert moe.group_routing is not None
    assert moe.group_routing.n_groups == 8
    assert moe.group_routing.topk_per_group == 4
