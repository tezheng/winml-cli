import torch

from models.phi3_mini import config


def _phi3_mini_4k_hf():
    """microsoft/Phi-3-mini-4k-instruct config.json (verified against the HF repo)."""
    return {
        "hidden_size": 3072, "num_attention_heads": 32, "num_key_value_heads": 32,
        "intermediate_size": 8192, "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5, "vocab_size": 32064,
        "max_position_embeddings": 4096, "original_max_position_embeddings": 4096,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "rope_theta": 10000.0, "rope_scaling": None,
        "sliding_window": 2047,
        "attention_bias": False, "mlp_bias": False,
    }


def test_phi3_mini_config_from_hf_dict_4k():
    c = config.Phi3MiniConfig.from_hf_dict(_phi3_mini_4k_hf())
    assert c.hidden_size == 3072
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 32                  # MHA on 4k variant
    assert c.head_dim == 96                             # 3072/32 = 96
    assert c.rope_theta == 10_000.0
    assert c.rope_type == "default"
    assert c.partial_rotary_factor == 1.0
    assert c.sliding_window == 2047
    assert c.dtype == torch.bfloat16


def test_phi3_mini_config_to_block_spec_4k():
    c = config.Phi3MiniConfig.from_hf_dict(_phi3_mini_4k_hf())
    spec = c.to_block_spec()
    from api import types as _t
    assert spec.token_mixer.qkv_layout == _t.QKVLayout.FUSED
    assert spec.channel_mixer.fused_gate_up is True
    assert spec.token_mixer.qk_norm is None
    assert spec.token_mixer.mask_kind == _t.MaskKind.SWA
    assert spec.token_mixer.sliding_window == 2047
    assert spec.token_mixer.rope.scaling == _t.RoPEScaling.NONE
    assert spec.token_mixer.rope.partial_rotary_factor == 1.0


def test_phi3_mini_config_handles_longrope_scaling():
    """For Phi-4-mini-style configs with rope_scaling.type='longrope'."""
    hf = dict(_phi3_mini_4k_hf())
    hf["max_position_embeddings"] = 131072
    hf["original_max_position_embeddings"] = 4096
    hf["partial_rotary_factor"] = 0.75
    hf["rope_scaling"] = {
        "type": "longrope",
        "short_factor": [1.0] * 36,                     # head_dim=96 * 0.75 / 2 = 36
        "long_factor": [2.0] * 36,
    }
    c = config.Phi3MiniConfig.from_hf_dict(hf)
    assert c.rope_type == "longrope"
    assert c.longrope_short_factor == tuple([1.0] * 36)
    assert c.longrope_long_factor == tuple([2.0] * 36)
    spec = c.to_block_spec()
    from api import types as _t
    assert spec.token_mixer.rope.scaling == _t.RoPEScaling.LONGROPE
    assert spec.token_mixer.rope.partial_rotary_factor == 0.75


def test_phi3_mini_config_handles_dtype_key():
    """transformers 5.x emits 'dtype', not 'torch_dtype'."""
    hf = dict(_phi3_mini_4k_hf())
    hf.pop("torch_dtype")
    hf["dtype"] = "bfloat16"
    c = config.Phi3MiniConfig.from_hf_dict(hf)
    assert c.dtype == torch.bfloat16
