"""Config tests for DeepSeek-V2-Lite (synthetic + actual production config)."""
import torch

from api import specs, types
from models.deepseek_v2_lite import config as m_config


_V2_LITE_HF = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 100000,
    "eos_token_id": 100001,
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 10944,
    "kv_lora_rank": 512,
    "max_position_embeddings": 163840,
    "moe_intermediate_size": 1408,
    "moe_layer_freq": 1,
    "n_group": 1,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "norm_topk_prob": False,
    "num_attention_heads": 16,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 27,
    "num_key_value_heads": 16,
    "pretraining_tp": 1,
    "q_lora_rank": None,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.0,
    "scoring_func": "softmax",
    "tie_word_embeddings": False,
    "topk_group": 1,
    "topk_method": "greedy",
    "torch_dtype": "bfloat16",
    "v_head_dim": 128,
    "vocab_size": 102400,
}


def test_v2_lite_from_hf_dict_round_trip():
    cfg = m_config.DeepSeekV2LiteConfig.from_hf_dict(_V2_LITE_HF)
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 27
    assert cfg.q_lora_rank is None              # V2-Lite direct q_proj.
    assert cfg.kv_lora_rank == 512
    assert cfg.qk_head_dim == 128 + 64
    assert cfg.v_head_dim == 128
    assert cfg.first_k_dense_replace == 1
    assert cfg.n_routed_experts == 64
    assert cfg.num_experts_per_tok == 6
    assert cfg.n_shared_experts == 2
    assert cfg.moe_intermediate_size == 1408
    assert cfg.routed_scaling_factor == 1.0
    assert cfg.norm_topk_prob is False
    assert cfg.n_group == 1
    assert cfg.topk_group == 1
    assert cfg.rope_type == "yarn"
    assert cfg.yarn_factor == 40.0
    assert cfg.yarn_original_max_position_embeddings == 4096


def test_v2_lite_layer0_is_dense():
    """B5: with first_k_dense_replace=1, layer 0 must be dense FFN."""
    cfg = m_config.DeepSeekV2LiteConfig.from_hf_dict(_V2_LITE_HF)
    blk = cfg.to_block_spec(layer_idx=0)
    assert isinstance(blk.channel_mixer, specs.FFNSpec)
    assert blk.channel_mixer.intermediate_size == 10944


def test_v2_lite_layer1_plus_is_moe():
    """B5: with first_k_dense_replace=1, layer 1+ must be MoE."""
    cfg = m_config.DeepSeekV2LiteConfig.from_hf_dict(_V2_LITE_HF)
    for L in (1, 5, 26):
        blk = cfg.to_block_spec(layer_idx=L)
        assert isinstance(blk.channel_mixer, specs.MoESpec), f"layer {L}"
        assert blk.channel_mixer.n_experts == 64
        assert blk.channel_mixer.top_k == 6
        assert blk.channel_mixer.n_shared_experts == 2
        # n_group==1, topk_group==1 → degenerate → group_routing=None.
        assert blk.channel_mixer.group_routing is None


def test_v2_lite_attention_spec_is_mla_interleaved_yarn():
    """V2 RoPE is complex-multiply (INTERLEAVED basis) + YARN scaling.
    Source: modeling_deepseek_v2.py:271-284 (complex multiply);
    deepseek-ai/DeepSeek-V2-Lite/config.json (rope_scaling=yarn)."""
    cfg = m_config.DeepSeekV2LiteConfig.from_hf_dict(_V2_LITE_HF)
    blk = cfg.to_block_spec(layer_idx=0)
    attn = blk.token_mixer
    assert isinstance(attn, specs.AttentionSpec)
    assert attn.kind == types.AttentionKind.MLA
    assert attn.qkv_layout == types.QKVLayout.MLA_LATENT
    assert attn.q_lora_rank is None
    assert attn.kv_lora_rank == 512
    assert attn.qk_nope_head_dim == 128
    assert attn.qk_rope_head_dim == 64
    assert attn.v_head_dim == 128
    assert attn.head_dim == 192            # qk_nope + qk_rope
    rope = attn.rope
    assert rope.basis == types.RoPEBasis.INTERLEAVED
    assert rope.scaling == types.RoPEScaling.YARN
    assert rope.yarn_extra.factor == 40.0
    assert rope.yarn_extra.original_max_position_embeddings == 4096
