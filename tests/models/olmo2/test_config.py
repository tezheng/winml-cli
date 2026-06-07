"""OLMo 2 config tests — adapter and block-spec correctness."""
import torch

from api import types
from models.olmo2 import config as olmo2_config


def _fake_hf_dict_1b():
    """Subset of allenai/OLMo-2-0425-1B config.json — only the keys our adapter reads."""
    return {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "rope_theta": 500000,
        "rms_norm_eps": 1e-6,
        "vocab_size": 100352,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "attention_bias": False,
    }


def test_olmo2_config_from_hf_dict():
    cfg = olmo2_config.Olmo2Config.from_hf_dict(_fake_hf_dict_1b())
    assert cfg.hidden_size == 2048
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 16
    assert cfg.head_dim == 128                  # derived hidden / heads
    assert cfg.intermediate_size == 8192
    assert cfg.num_hidden_layers == 16
    assert cfg.rope_theta == 500000.0
    assert cfg.dtype == torch.float32
    assert cfg.attention_bias is False
    assert cfg.tie_word_embeddings is False


def test_olmo2_config_explicit_head_dim_honored():
    """When head_dim is set explicitly in the HF dict, we honor it (not derived)."""
    d = _fake_hf_dict_1b()
    d["head_dim"] = 96
    cfg = olmo2_config.Olmo2Config.from_hf_dict(d)
    assert cfg.head_dim == 96


def test_olmo2_config_to_block_spec_uses_post_norm():
    """B3: the OLMo 2 block spec MUST advertise POST norm-position on both
    sublayers, and MUST NOT set pre_attn_norm / pre_ffn_norm specs.
    Source: modeling_olmo2.py:295-333.
    """
    cfg = olmo2_config.Olmo2Config.from_hf_dict(_fake_hf_dict_1b())
    spec = cfg.to_block_spec()
    assert spec.attn_norm_position == types.NormPosition.POST
    assert spec.ffn_norm_position == types.NormPosition.POST
    assert spec.pre_attn_norm is None
    assert spec.pre_ffn_norm is None
    assert spec.post_attn_norm is not None
    assert spec.post_ffn_norm is not None


def test_olmo2_config_to_block_spec_qk_norm_full_hdh():
    """B3: FULL_HDH QK-norm shape, PRE_ROPE phase.
    Source: modeling_olmo2.py:231-249.
    """
    cfg = olmo2_config.Olmo2Config.from_hf_dict(_fake_hf_dict_1b())
    spec = cfg.to_block_spec()
    assert spec.token_mixer.qk_norm is not None
    assert spec.token_mixer.qk_norm_phase == types.QKNormPhase.PRE_ROPE
    assert spec.token_mixer.qk_norm_shape == types.QKNormShape.FULL_HDH


def test_olmo2_config_to_block_spec_swiglu_no_bias():
    cfg = olmo2_config.Olmo2Config.from_hf_dict(_fake_hf_dict_1b())
    spec = cfg.to_block_spec()
    assert spec.channel_mixer.gate_kind == types.GateKind.SWIGLU
    assert spec.channel_mixer.activation == types.Activation.SILU
    assert spec.channel_mixer.gate_bias is False
    assert spec.channel_mixer.up_bias is False
    assert spec.channel_mixer.down_bias is False


def test_olmo2_config_to_block_spec_rope_split_half():
    cfg = olmo2_config.Olmo2Config.from_hf_dict(_fake_hf_dict_1b())
    spec = cfg.to_block_spec()
    assert spec.token_mixer.rope is not None
    assert spec.token_mixer.rope.basis == types.RoPEBasis.SPLIT_HALF
    assert spec.token_mixer.rope.base_theta == 500000.0
    assert spec.token_mixer.rope.scaling == types.RoPEScaling.NONE
    assert spec.token_mixer.rope.partial_rotary_factor == 1.0


def test_olmo2_config_accepts_dtype_5x_key():
    d = _fake_hf_dict_1b()
    d.pop("torch_dtype", None)
    d["dtype"] = "bfloat16"
    cfg = olmo2_config.Olmo2Config.from_hf_dict(d)
    assert cfg.dtype == torch.bfloat16
