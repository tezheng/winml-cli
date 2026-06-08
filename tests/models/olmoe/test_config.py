"""OLMoE config tests — adapter and block-spec correctness.

Source-grounded at `transformers/models/olmoe/configuration_olmoe.py` and
`modeling_olmoe.py`.
"""
import pytest
import torch

from api import specs, types
from models.olmoe import config as m_config


def _fake_hf_dict_1b_7b_0924():
    """Subset of allenai/OLMoE-1B-7B-0924 config.json — only the keys we read."""
    return {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,            # MHA, not GQA on 1B-7B-0924
        "intermediate_size": 1024,
        "num_hidden_layers": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "norm_topk_prob": False,
        "rope_theta": 10_000.0,
        "rms_norm_eps": 1e-5,
        "vocab_size": 50304,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "torch_dtype": "float32",
        "clip_qkv": None,
    }


def test_olmoe_config_from_hf_dict():
    cfg = m_config.OlmoeConfig.from_hf_dict(_fake_hf_dict_1b_7b_0924())
    assert cfg.hidden_size == 2048
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 16
    assert cfg.head_dim == 128                  # derived 2048/16
    assert cfg.intermediate_size == 1024
    assert cfg.num_experts == 64
    assert cfg.num_experts_per_tok == 8
    assert cfg.norm_topk_prob is False
    assert cfg.rope_theta == 10_000.0
    assert cfg.dtype == torch.float32
    assert cfg.clip_qkv is None
    assert cfg.tie_word_embeddings is False


def test_olmoe_config_num_kv_heads_defaults_to_num_q_heads_when_null():
    """configuration_olmoe.py:83-85 — when num_key_value_heads is None it
    defaults to num_attention_heads."""
    d = _fake_hf_dict_1b_7b_0924()
    d["num_key_value_heads"] = None
    cfg = m_config.OlmoeConfig.from_hf_dict(d)
    assert cfg.num_key_value_heads == cfg.num_attention_heads == 16


def test_olmoe_config_accepts_num_local_experts_alias():
    d = _fake_hf_dict_1b_7b_0924()
    d.pop("num_experts")
    d["num_local_experts"] = 64
    cfg = m_config.OlmoeConfig.from_hf_dict(d)
    assert cfg.num_experts == 64


def test_olmoe_block_spec_uses_pre_norm():
    cfg = m_config.OlmoeConfig.from_hf_dict(_fake_hf_dict_1b_7b_0924())
    blk = cfg.to_block_spec()
    assert blk.attn_norm_position == types.NormPosition.PRE
    assert blk.ffn_norm_position == types.NormPosition.PRE


def test_olmoe_block_spec_uses_full_hdh_qk_norm_pre_rope():
    """B6: OLMoE uses FULL_HDH QK-norm PRE-RoPE (same shape as OLMo 2, but
    PRE-norm envelope). Source: modeling_olmoe.py:246-249 + 262-275."""
    cfg = m_config.OlmoeConfig.from_hf_dict(_fake_hf_dict_1b_7b_0924())
    blk = cfg.to_block_spec()
    attn = blk.token_mixer
    assert attn.qk_norm is not None
    assert attn.qk_norm_phase == types.QKNormPhase.PRE_ROPE
    assert attn.qk_norm_shape == types.QKNormShape.FULL_HDH


def test_olmoe_block_spec_moe_no_shared_no_group():
    cfg = m_config.OlmoeConfig.from_hf_dict(_fake_hf_dict_1b_7b_0924())
    blk = cfg.to_block_spec()
    moe = blk.channel_mixer
    assert isinstance(moe, specs.MoESpec)
    assert moe.router_kind == "softmax"
    assert moe.n_experts == 64
    assert moe.top_k == 8
    assert moe.n_shared_experts == 0
    assert moe.group_routing is None
    assert moe.routed_scaling_factor == 1.0
    # OLMoE-0924 ships norm_topk_prob=False.
    assert moe.router_norm is False
    assert moe.expert_ffn.intermediate_size == 1024


def test_olmoe_block_spec_mask_is_causal_no_sliding():
    """OLMoE has no sliding window (modeling_olmoe.py:490 comment)."""
    cfg = m_config.OlmoeConfig.from_hf_dict(_fake_hf_dict_1b_7b_0924())
    blk = cfg.to_block_spec()
    assert blk.token_mixer.mask_kind == types.MaskKind.CAUSAL
    assert blk.token_mixer.sliding_window is None


def test_olmoe_block_spec_rejects_nonnull_clip_qkv():
    """B6: clip_qkv is not yet implemented in api.attention; the spec
    builder must raise rather than silently dropping the clamp."""
    d = _fake_hf_dict_1b_7b_0924()
    d["clip_qkv"] = 8.0
    cfg = m_config.OlmoeConfig.from_hf_dict(d)
    with pytest.raises(NotImplementedError, match="clip_qkv"):
        cfg.to_block_spec()


def test_olmoe_block_spec_rope_split_half():
    cfg = m_config.OlmoeConfig.from_hf_dict(_fake_hf_dict_1b_7b_0924())
    blk = cfg.to_block_spec()
    rope = blk.token_mixer.rope
    assert rope.basis == types.RoPEBasis.SPLIT_HALF
    assert rope.base_theta == 10_000.0
