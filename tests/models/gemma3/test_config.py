"""Gemma 3 config tests."""
import torch

from api import types
from models.gemma3 import config as g3c


def _fake_hf_dict_1b():
    """Subset of unsloth/gemma-3-1b-it config.json."""
    return {
        "hidden_size": 1152,
        "num_hidden_layers": 26,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 6912,
        "rope_theta": 1_000_000,
        "rope_local_base_freq": 10_000,
        "rms_norm_eps": 1e-6,
        "vocab_size": 262144,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "sliding_window": 512,
        "sliding_window_pattern": 6,
        "query_pre_attn_scalar": 256,
        "attn_logit_softcapping": None,
        "final_logit_softcapping": None,
        "attention_bias": False,
        "hidden_activation": "gelu_pytorch_tanh",
    }


def test_gemma3_config_from_hf_dict():
    cfg = g3c.Gemma3Config.from_hf_dict(_fake_hf_dict_1b())
    assert cfg.hidden_size == 1152
    assert cfg.num_hidden_layers == 26
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 1
    assert cfg.head_dim == 256
    assert cfg.rope_theta_local == 10_000.0
    assert cfg.rope_theta_global == 1_000_000.0
    assert cfg.sliding_window == 512
    assert cfg.sliding_window_pattern == 6
    assert cfg.query_pre_attn_scalar == 256
    assert cfg.attn_logit_softcap is None
    assert cfg.final_logit_softcap is None


def test_gemma3_config_accepts_rope_parameters_5x_layout():
    """transformers 5.x emits dual-rope under `rope_parameters`."""
    d = _fake_hf_dict_1b()
    d.pop("rope_theta")
    d.pop("rope_local_base_freq")
    d["rope_parameters"] = {
        "sliding_attention": {"rope_theta": 10_000, "rope_type": "default"},
        "full_attention": {"rope_theta": 1_000_000, "rope_type": "default"},
    }
    cfg = g3c.Gemma3Config.from_hf_dict(d)
    assert cfg.rope_theta_local == 10_000.0
    assert cfg.rope_theta_global == 1_000_000.0


def test_gemma3_layer_type_pattern_6():
    """Pattern=6 means layer indices 5, 11, 17, 23 are full; rest sliding.
    Source: configuration_gemma3.py:109-115.
    """
    cfg = g3c.Gemma3Config.from_hf_dict(_fake_hf_dict_1b())
    # Layers 0..4 sliding (i+1 % 6 != 0).
    for i in range(5):
        assert cfg.layer_type(i) == "sliding_attention", f"layer {i}"
    # Layer 5 is the FIRST FULL layer.
    assert cfg.layer_type(5) == "full_attention"
    # Layers 6..10 sliding.
    for i in range(6, 11):
        assert cfg.layer_type(i) == "sliding_attention", f"layer {i}"
    # Layer 11 is full.
    assert cfg.layer_type(11) == "full_attention"


def test_gemma3_sliding_layer_uses_local_theta():
    cfg = g3c.Gemma3Config.from_hf_dict(_fake_hf_dict_1b())
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert spec.token_mixer.sliding_window == 512
    assert spec.token_mixer.rope.base_theta == 10_000.0
    # QK-norm present.
    assert spec.token_mixer.qk_norm is not None
    assert spec.token_mixer.qk_norm_phase == types.QKNormPhase.PRE_ROPE
    assert spec.token_mixer.qk_norm_shape == types.QKNormShape.PER_HEAD_DH


def test_gemma3_full_layer_uses_global_theta():
    cfg = g3c.Gemma3Config.from_hf_dict(_fake_hf_dict_1b())
    spec = cfg.to_block_spec(layer_idx=5)        # first global layer
    assert spec.token_mixer.mask_kind == types.MaskKind.CAUSAL
    assert spec.token_mixer.sliding_window is None
    assert spec.token_mixer.rope.base_theta == 1_000_000.0
    # QK-norm is still active on full layers.
    assert spec.token_mixer.qk_norm is not None


def test_gemma3_no_softcaps():
    """Gemma 3 DROPPED both softcaps. Verify None propagates."""
    cfg = g3c.Gemma3Config.from_hf_dict(_fake_hf_dict_1b())
    spec_local = cfg.to_block_spec(layer_idx=0)
    spec_global = cfg.to_block_spec(layer_idx=5)
    assert spec_local.token_mixer.logit_softcap is None
    assert spec_global.token_mixer.logit_softcap is None
    assert cfg.final_logit_softcap is None


def test_gemma3_uses_full_rope_no_partial():
    """Gemma 3 (unlike Gemma 4) does NOT use partial RoPE."""
    cfg = g3c.Gemma3Config.from_hf_dict(_fake_hf_dict_1b())
    spec_local = cfg.to_block_spec(layer_idx=0)
    spec_global = cfg.to_block_spec(layer_idx=5)
    assert spec_local.token_mixer.rope.partial_rotary_factor == 1.0
    assert spec_global.token_mixer.rope.partial_rotary_factor == 1.0


def test_gemma3_sandwich_norm_with_one_plus_w():
    cfg = g3c.Gemma3Config.from_hf_dict(_fake_hf_dict_1b())
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.attn_norm_position == types.NormPosition.PRE_AND_POST
    assert spec.pre_attn_norm.weight_mode == types.NormWeightMode.ONE_PLUS_W


def test_gemma3_attn_scale_uses_query_pre_attn_scalar():
    cfg = g3c.Gemma3Config.from_hf_dict(_fake_hf_dict_1b())
    spec = cfg.to_block_spec(layer_idx=0)
    expected = 256 ** -0.5
    assert abs(spec.token_mixer.attn_scale - expected) < 1e-10
