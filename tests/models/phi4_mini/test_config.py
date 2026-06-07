import torch

from models.phi4_mini import config


def _phi4_mini_hf():
    """microsoft/Phi-4-mini-instruct config.json (verified against the HF repo).

    The short_factor and long_factor are 48-element vectors (=
    head_dim * partial_rotary_factor / 2 = 128 * 0.75 / 2 = 48). We use
    representative dummy values for the test; the actual checkpoint values
    are loaded by the gate test via AutoConfig.from_pretrained.
    """
    return {
        "hidden_size": 3072, "num_attention_heads": 24, "num_key_value_heads": 8,
        "intermediate_size": 8192, "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5, "vocab_size": 200064,
        "max_position_embeddings": 131072, "original_max_position_embeddings": 4096,
        "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.75,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0] * 48,
            "long_factor": [2.0] * 48,
        },
        "sliding_window": None,
        "attention_bias": False, "mlp_bias": False,
    }


def test_phi4_mini_config_from_hf_dict():
    c = config.from_hf_dict(_phi4_mini_hf())
    assert c.hidden_size == 3072
    assert c.num_attention_heads == 24
    assert c.num_key_value_heads == 8                   # GQA
    assert c.head_dim == 128                            # 3072/24 = 128
    assert c.partial_rotary_factor == 0.75
    assert c.original_max_position_embeddings == 4096
    assert c.max_position_embeddings == 131072
    assert c.rope_type == "longrope"
    assert len(c.longrope_short_factor) == 48
    assert len(c.longrope_long_factor) == 48
    assert c.sliding_window is None
    assert c.tie_word_embeddings is True


def test_phi4_mini_config_to_block_spec():
    c = config.from_hf_dict(_phi4_mini_hf())
    spec = c.to_block_spec()
    from api import types as _t
    assert spec.token_mixer.qkv_layout == _t.QKVLayout.FUSED
    assert spec.channel_mixer.fused_gate_up is True
    assert spec.token_mixer.mask_kind == _t.MaskKind.CAUSAL   # no SWA
    assert spec.token_mixer.rope.scaling == _t.RoPEScaling.LONGROPE
    assert spec.token_mixer.rope.partial_rotary_factor == 0.75
    assert spec.token_mixer.rope.longrope_extra is not None
    # attention_factor computed from factor = 131072 / 4096 = 32 > 1, so
    # attention_factor = sqrt(1 + log(32)/log(4096)).
    import math
    expected = math.sqrt(1 + math.log(32) / math.log(4096))
    assert abs(spec.token_mixer.rope.longrope_extra.attention_factor - expected) < 1e-6
