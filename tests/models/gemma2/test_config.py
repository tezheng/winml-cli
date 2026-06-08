"""Gemma 2 config tests — HF dict adapter + per-layer block spec."""
import torch

from api import types
from models.gemma2 import config as g2c


def _fake_hf_dict_2b():
    """Subset of unsloth/gemma-2-2b config.json."""
    return {
        "hidden_size": 2304,
        "num_hidden_layers": 26,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 9216,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256000,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "sliding_window": 4096,
        "query_pre_attn_scalar": 256,
        "attn_logit_softcapping": 50.0,
        "final_logit_softcapping": 30.0,
        "attention_bias": False,
        "hidden_activation": "gelu_pytorch_tanh",
    }


def test_gemma2_config_from_hf_dict():
    cfg = g2c.Gemma2Config.from_hf_dict(_fake_hf_dict_2b())
    assert cfg.hidden_size == 2304
    assert cfg.num_hidden_layers == 26
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 256
    assert cfg.intermediate_size == 9216
    assert cfg.rope_theta == 10000.0
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.vocab_size == 256000
    assert cfg.sliding_window == 4096
    assert cfg.query_pre_attn_scalar == 256
    assert cfg.attn_logit_softcap == 50.0
    assert cfg.final_logit_softcap == 30.0
    assert cfg.dtype == torch.bfloat16


def test_gemma2_layer_type_alternation():
    """Default pattern: layer 0=SWA, 1=full, 2=SWA, ... per HF
    configuration_gemma2.py:95-98."""
    cfg = g2c.Gemma2Config.from_hf_dict(_fake_hf_dict_2b())
    assert cfg.layer_type(0) == "sliding_attention"
    assert cfg.layer_type(1) == "full_attention"
    assert cfg.layer_type(2) == "sliding_attention"
    assert cfg.layer_type(3) == "full_attention"


def test_gemma2_sliding_layer_block_spec():
    cfg = g2c.Gemma2Config.from_hf_dict(_fake_hf_dict_2b())
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert spec.token_mixer.sliding_window == 4096
    assert spec.token_mixer.logit_softcap == 50.0
    # PRE_AND_POST sandwich norm.
    assert spec.attn_norm_position == types.NormPosition.PRE_AND_POST
    assert spec.ffn_norm_position == types.NormPosition.PRE_AND_POST
    # All four norm specs are populated.
    assert spec.pre_attn_norm is not None
    assert spec.post_attn_norm is not None
    assert spec.pre_ffn_norm is not None
    assert spec.post_ffn_norm is not None
    # ONE_PLUS_W mode.
    assert spec.pre_attn_norm.weight_mode == types.NormWeightMode.ONE_PLUS_W


def test_gemma2_full_layer_block_spec():
    cfg = g2c.Gemma2Config.from_hf_dict(_fake_hf_dict_2b())
    spec = cfg.to_block_spec(layer_idx=1)
    assert spec.token_mixer.mask_kind == types.MaskKind.CAUSAL
    assert spec.token_mixer.sliding_window is None
    # Softcap applies on every layer.
    assert spec.token_mixer.logit_softcap == 50.0


def test_gemma2_attn_scale_uses_query_pre_attn_scalar():
    """Source: modeling_gemma2.py:240 — `scaling = config.query_pre_attn_scalar**-0.5`."""
    cfg = g2c.Gemma2Config.from_hf_dict(_fake_hf_dict_2b())
    spec = cfg.to_block_spec(layer_idx=0)
    expected = 256 ** -0.5
    assert abs(spec.token_mixer.attn_scale - expected) < 1e-10


def test_gemma2_block_spec_geglu_ffn():
    cfg = g2c.Gemma2Config.from_hf_dict(_fake_hf_dict_2b())
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.channel_mixer.gate_kind == types.GateKind.GEGLU
    assert spec.channel_mixer.activation == types.Activation.GELU


def test_gemma2_no_qk_norm_set():
    """Gemma 2 has NO QK-norm. Verify source citation."""
    cfg = g2c.Gemma2Config.from_hf_dict(_fake_hf_dict_2b())
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.token_mixer.qk_norm is None
    assert spec.token_mixer.qk_norm_phase == types.QKNormPhase.NONE
    assert spec.token_mixer.qk_norm_shape == types.QKNormShape.NONE
