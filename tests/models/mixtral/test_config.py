"""Mixtral config tests — adapter and block-spec correctness.

Source-grounded at `transformers/models/mixtral/configuration_mixtral.py`
and `transformers/models/mixtral/modeling_mixtral.py`.
"""
import torch

from api import specs, types
from models.mixtral import config as m_config


def _fake_hf_dict_8x7b():
    """Subset of mistralai/Mixtral-8x7B-v0.1 config.json — only the keys our adapter reads."""
    return {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32000,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "sliding_window": None,
    }


def test_mixtral_config_from_hf_dict():
    cfg = m_config.MixtralConfig.from_hf_dict(_fake_hf_dict_8x7b())
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128                    # derived 4096/32
    assert cfg.intermediate_size == 14336
    assert cfg.num_hidden_layers == 32
    assert cfg.num_local_experts == 8
    assert cfg.num_experts_per_tok == 2
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.dtype == torch.bfloat16
    assert cfg.tie_word_embeddings is False
    assert cfg.sliding_window is None


def test_mixtral_config_accepts_num_experts_alias():
    """HF aliases num_experts to num_local_experts (attribute_map). Adapter
    must read either key."""
    d = _fake_hf_dict_8x7b()
    d.pop("num_local_experts")
    d["num_experts"] = 16
    cfg = m_config.MixtralConfig.from_hf_dict(d)
    assert cfg.num_local_experts == 16


def test_mixtral_config_explicit_head_dim_honored():
    d = _fake_hf_dict_8x7b()
    d["head_dim"] = 64
    cfg = m_config.MixtralConfig.from_hf_dict(d)
    assert cfg.head_dim == 64


def test_mixtral_block_spec_uses_pre_norm():
    cfg = m_config.MixtralConfig.from_hf_dict(_fake_hf_dict_8x7b())
    blk = cfg.to_block_spec()
    assert blk.attn_norm_position == types.NormPosition.PRE
    assert blk.ffn_norm_position == types.NormPosition.PRE
    assert blk.pre_attn_norm is not None
    assert blk.pre_ffn_norm is not None
    assert blk.post_attn_norm is None
    assert blk.post_ffn_norm is None


def test_mixtral_block_spec_uses_split_gqa_attention():
    cfg = m_config.MixtralConfig.from_hf_dict(_fake_hf_dict_8x7b())
    blk = cfg.to_block_spec()
    attn = blk.token_mixer
    assert isinstance(attn, specs.AttentionSpec)
    assert attn.kind == types.AttentionKind.STANDARD
    assert attn.qkv_layout == types.QKVLayout.SPLIT
    assert attn.n_q_heads == 32 and attn.n_kv_heads == 8
    assert attn.q_bias is False and attn.k_bias is False
    assert attn.v_bias is False and attn.o_bias is False
    # No QK-norm on Mixtral (modeling_mixtral.py:298-311).
    assert attn.qk_norm is None
    assert attn.qk_norm_phase == types.QKNormPhase.NONE
    assert attn.qk_norm_shape == types.QKNormShape.NONE


def test_mixtral_block_spec_rope_split_half_theta_1e6():
    cfg = m_config.MixtralConfig.from_hf_dict(_fake_hf_dict_8x7b())
    blk = cfg.to_block_spec()
    rope = blk.token_mixer.rope
    assert rope is not None
    assert rope.basis == types.RoPEBasis.SPLIT_HALF
    assert rope.base_theta == 1_000_000.0
    assert rope.scaling == types.RoPEScaling.NONE


def test_mixtral_block_spec_moe_softmax_always_norm():
    """B6: Mixtral router ALWAYS renormalizes top-k weights — no
    norm_topk_prob flag in HF (modeling_mixtral.py:114). The spec must
    carry router_norm=True, no shared experts, no group routing, no
    routed_scaling_factor != 1.0."""
    cfg = m_config.MixtralConfig.from_hf_dict(_fake_hf_dict_8x7b())
    blk = cfg.to_block_spec()
    moe = blk.channel_mixer
    assert isinstance(moe, specs.MoESpec)
    assert moe.router_kind == "softmax"
    assert moe.router_norm is True
    assert moe.group_routing is None
    assert moe.routed_scaling_factor == 1.0
    assert moe.n_experts == 8
    assert moe.top_k == 2
    assert moe.n_shared_experts == 0
    assert moe.expert_ffn is not None
    assert moe.expert_ffn.intermediate_size == 14336
    assert moe.expert_ffn.gate_kind == types.GateKind.SWIGLU
    assert moe.expert_ffn.activation == types.Activation.SILU


def test_mixtral_block_spec_mask_causal_when_no_swa():
    cfg = m_config.MixtralConfig.from_hf_dict(_fake_hf_dict_8x7b())
    blk = cfg.to_block_spec()
    assert blk.token_mixer.mask_kind == types.MaskKind.CAUSAL
    assert blk.token_mixer.sliding_window is None


def test_mixtral_block_spec_mask_swa_when_set():
    d = _fake_hf_dict_8x7b()
    d["sliding_window"] = 4096
    cfg = m_config.MixtralConfig.from_hf_dict(d)
    blk = cfg.to_block_spec()
    assert blk.token_mixer.mask_kind == types.MaskKind.SWA
    assert blk.token_mixer.sliding_window == 4096


def test_mixtral_config_accepts_rope_parameters_nested():
    """transformers 5.x nests rope_theta under rope_parameters."""
    d = _fake_hf_dict_8x7b()
    d.pop("rope_theta")
    d["rope_parameters"] = {"rope_type": "default", "rope_theta": 1_000_000.0}
    cfg = m_config.MixtralConfig.from_hf_dict(d)
    assert cfg.rope_theta == 1_000_000.0
