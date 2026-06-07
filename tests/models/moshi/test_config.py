"""Moshi config tests — verifies from_hf_dict and to_block_spec match the
HF source-of-truth for the canonical kmhf/hf-moshiko config.
"""
import pytest
import torch

from api import specs, types
from models.moshi import config as m_config


def _canonical_moshiko_hf_dict() -> dict:
    """Mirror of `kmhf/hf-moshiko` MoshiConfig (LM-decoder-relevant keys).

    Verified against configuration_moshi.py:93-218 + a live AutoConfig
    fetch of `kmhf/hf-moshiko`.
    """
    return dict(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,         # MHA — equals num_attention_heads
        head_dim=128,
        ffn_dim=22528,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rms_norm_eps=1e-8,
        vocab_size=32000,
        max_position_embeddings=3000,
        sliding_window=3000,
        tie_word_embeddings=False,
        num_codebooks=8,
        audio_vocab_size=2048,
        dtype="float32",
    )


def test_from_hf_dict_canonical_dims():
    cfg = m_config.MoshiConfig.from_hf_dict(_canonical_moshiko_hf_dict())
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 32      # MHA — not GQA
    assert cfg.head_dim == 128
    assert cfg.ffn_dim == 22528
    assert cfg.intermediate_size == 11264    # ffn_dim // 2
    assert cfg.num_hidden_layers == 32
    assert cfg.rope_theta == 10000.0
    assert cfg.rms_norm_eps == 1e-8
    assert cfg.vocab_size == 32000
    assert cfg.max_position_embeddings == 3000
    assert cfg.sliding_window == 3000
    assert cfg.tie_word_embeddings is False
    assert cfg.dtype == torch.float32


def test_from_hf_dict_handles_missing_num_kv_heads():
    """HF config defaults `num_key_value_heads=None` then post_init sets it
    to `num_attention_heads`. Our adapter must reproduce that."""
    hf = _canonical_moshiko_hf_dict()
    hf.pop("num_key_value_heads")
    cfg = m_config.MoshiConfig.from_hf_dict(hf)
    assert cfg.num_key_value_heads == cfg.num_attention_heads == 32


def test_from_hf_dict_handles_flat_rope_theta():
    """Some downstream pipelines emit `rope_theta` at the top level rather
    than nested under `rope_parameters`."""
    hf = _canonical_moshiko_hf_dict()
    hf.pop("rope_parameters")
    hf["rope_theta"] = 12345.0
    cfg = m_config.MoshiConfig.from_hf_dict(hf)
    assert cfg.rope_theta == 12345.0


def test_to_block_spec_canonical_shapes():
    cfg = m_config.MoshiConfig.from_hf_dict(_canonical_moshiko_hf_dict())
    spec = cfg.to_block_spec(layer_idx=0)
    # Norms.
    assert spec.attn_norm_position == types.NormPosition.PRE
    assert spec.ffn_norm_position == types.NormPosition.PRE
    assert spec.pre_attn_norm.kind == types.NormKind.RMS
    assert spec.pre_attn_norm.eps == 1e-8
    assert spec.pre_attn_norm.weight_mode == types.NormWeightMode.STANDARD_W
    # Attention.
    attn = spec.token_mixer
    assert isinstance(attn, specs.AttentionSpec)
    assert attn.kind == types.AttentionKind.STANDARD
    assert attn.qkv_layout == types.QKVLayout.SPLIT
    assert attn.mask_kind == types.MaskKind.CAUSAL    # eager path is causal
    assert attn.sliding_window is None
    assert attn.n_q_heads == 32 and attn.n_kv_heads == 32
    assert attn.head_dim == 128
    assert not (attn.q_bias or attn.k_bias or attn.v_bias or attn.o_bias)
    assert attn.qk_norm is None
    assert attn.rope.base_theta == 10000.0
    assert attn.rope.basis == types.RoPEBasis.SPLIT_HALF
    assert attn.rope.scaling == types.RoPEScaling.NONE
    # FFN — FUSED gate/up.
    ffn = spec.channel_mixer
    assert isinstance(ffn, specs.FFNSpec)
    assert ffn.intermediate_size == 11264
    assert ffn.activation == types.Activation.SILU
    assert ffn.gate_kind == types.GateKind.SWIGLU
    assert ffn.fused_gate_up is True
    assert not (ffn.gate_bias or ffn.up_bias or ffn.down_bias)


def test_intermediate_size_property():
    """The property must match ffn_dim // 2 exactly (modeling_moshi.py:376-381)."""
    cfg = m_config.MoshiConfig.from_hf_dict(_canonical_moshiko_hf_dict())
    assert cfg.intermediate_size == cfg.ffn_dim // 2
