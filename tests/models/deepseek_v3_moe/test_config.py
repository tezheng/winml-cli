"""DeepSeek-V3-MoE config tests — production-default round-trip."""
import torch

from api import specs, types
from models.deepseek_v3_moe import config as m_config


def _hf_dict_v3_defaults():
    """Production DeepSeek-V3 defaults — verified against
    configuration_deepseek_v3.py:71-104."""
    return {
        "vocab_size": 129280,
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "routed_scaling_factor": 2.5,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "qk_nope_head_dim": 128,
        "n_group": 8,
        "topk_group": 4,
        "num_experts_per_tok": 8,
        "first_k_dense_replace": 3,
        "norm_topk_prob": True,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "rope_theta": 10_000.0,
        "attention_bias": False,
        "torch_dtype": "bfloat16",
    }


def test_v3_moe_production_defaults_round_trip():
    cfg = m_config.DeepSeekV3MoEConfig.from_hf_dict(_hf_dict_v3_defaults())
    assert cfg.hidden_size == 7168
    assert cfg.num_attention_heads == 128
    assert cfg.num_hidden_layers == 61
    assert cfg.q_lora_rank == 1536
    assert cfg.kv_lora_rank == 512
    assert cfg.qk_head_dim == 128 + 64
    assert cfg.v_head_dim == 128
    assert cfg.n_routed_experts == 256
    assert cfg.n_shared_experts == 1
    assert cfg.num_experts_per_tok == 8
    assert cfg.n_group == 8 and cfg.topk_group == 4
    assert cfg.routed_scaling_factor == 2.5
    assert cfg.norm_topk_prob is True
    assert cfg.first_k_dense_replace == 3
    assert cfg.dtype == torch.bfloat16


def test_v3_moe_layer_0_through_2_are_dense():
    cfg = m_config.DeepSeekV3MoEConfig.from_hf_dict(_hf_dict_v3_defaults())
    for L in (0, 1, 2):
        blk = cfg.to_block_spec(layer_idx=L)
        assert isinstance(blk.channel_mixer, specs.FFNSpec), f"layer {L}"
        assert blk.channel_mixer.intermediate_size == 18432


def test_v3_moe_layer_3_plus_is_moe_with_sigmoid_router_and_group_routing():
    cfg = m_config.DeepSeekV3MoEConfig.from_hf_dict(_hf_dict_v3_defaults())
    blk = cfg.to_block_spec(layer_idx=3)
    moe = blk.channel_mixer
    assert isinstance(moe, specs.MoESpec)
    assert moe.router_kind == "sigmoid_plus_bias"
    assert moe.router_norm is True
    assert moe.routed_scaling_factor == 2.5
    assert moe.n_experts == 256
    assert moe.top_k == 8
    assert moe.n_shared_experts == 1
    assert moe.group_routing is not None
    assert moe.group_routing.n_groups == 8
    assert moe.group_routing.topk_per_group == 4


def test_v3_moe_attention_is_mla_with_q_lora_rank_1536():
    cfg = m_config.DeepSeekV3MoEConfig.from_hf_dict(_hf_dict_v3_defaults())
    blk = cfg.to_block_spec(layer_idx=3)
    attn = blk.token_mixer
    assert isinstance(attn, specs.AttentionSpec)
    assert attn.kind == types.AttentionKind.MLA
    assert attn.qkv_layout == types.QKVLayout.MLA_LATENT
    assert attn.q_lora_rank == 1536
    assert attn.kv_lora_rank == 512
    assert attn.qk_nope_head_dim == 128
    assert attn.qk_rope_head_dim == 64
    assert attn.v_head_dim == 128
    assert attn.head_dim == 192
    # V3 RoPE: SPLIT_HALF basis (rope_interleave=False branch).
    assert attn.rope is not None
    assert attn.rope.basis == types.RoPEBasis.SPLIT_HALF
