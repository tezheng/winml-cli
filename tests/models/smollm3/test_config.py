import torch

from models.smollm3 import config


def test_smollm3_config_from_hf_dict_3b_default_nope_pattern():
    """SmolLM3-3B with explicit no_rope_layers field from HF."""
    nope = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 0]
    hf = {
        "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 4,
        "intermediate_size": 11008, "num_hidden_layers": 36,
        "rms_norm_eps": 1e-6, "vocab_size": 128256,
        "max_position_embeddings": 65536, "tie_word_embeddings": True,
        "dtype": "bfloat16",
        "rope_parameters": {"rope_theta": 5_000_000.0, "rope_type": "default"},
        "no_rope_layers": nope,
        "no_rope_layer_interval": 4,
        "use_sliding_window": False,
        "sliding_window": None,
    }
    c = config.Smollm3Config.from_hf_dict(hf)
    assert c.hidden_size == 2048
    assert c.num_attention_heads == 16
    assert c.num_key_value_heads == 4
    assert c.head_dim == 128  # derived from 2048//16
    assert c.rope_theta == 5_000_000.0
    assert len(c.no_rope_layers) == 36
    assert c.layer_uses_rope(0) is True
    assert c.layer_uses_rope(1) is True
    assert c.layer_uses_rope(2) is True
    assert c.layer_uses_rope(3) is False   # NoPE layer
    assert c.layer_uses_rope(4) is True
    assert c.layer_uses_rope(7) is False   # NoPE layer
    assert c.layer_uses_rope(35) is False  # NoPE layer (final)


def test_smollm3_config_derives_default_nope_pattern():
    """If `no_rope_layers` is absent, derive from no_rope_layer_interval=4."""
    hf = {
        "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 4,
        "intermediate_size": 11008, "num_hidden_layers": 8,
        "rms_norm_eps": 1e-6, "vocab_size": 128256,
        "max_position_embeddings": 65536, "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "rope_parameters": {"rope_theta": 5_000_000.0, "rope_type": "default"},
        "no_rope_layer_interval": 4,
    }
    c = config.Smollm3Config.from_hf_dict(hf)
    # Pattern: (i+1) % 4 != 0 → 1; so 0,1,2,4,5,6 use RoPE; 3 and 7 are NoPE.
    assert c.layer_uses_rope(0)
    assert c.layer_uses_rope(2)
    assert not c.layer_uses_rope(3)
    assert c.layer_uses_rope(4)
    assert not c.layer_uses_rope(7)


def test_smollm3_config_to_block_spec_rope_layer():
    c = config.Smollm3Config(
        hidden_size=2048, num_attention_heads=16, num_key_value_heads=4,
        head_dim=128, intermediate_size=11008, num_hidden_layers=36,
        rope_theta=5_000_000.0, rms_norm_eps=1e-6,
        vocab_size=128256, max_position_embeddings=65536,
        tie_word_embeddings=True, dtype=torch.float32,
        no_rope_layers=tuple([1] * 36),
    )
    spec = c.to_block_spec(layer_idx=0)
    from api import types as _t
    assert spec.token_mixer.rope is not None
    assert spec.token_mixer.rope.base_theta == 5_000_000.0
    assert spec.token_mixer.mask_kind == _t.MaskKind.CAUSAL
    assert spec.token_mixer.qk_norm is None
    assert spec.channel_mixer.gate_kind == _t.GateKind.SWIGLU


def test_smollm3_config_to_block_spec_nope_layer():
    """NoPE layer → spec.rope is None."""
    nope = list([1] * 36)
    nope[3] = 0
    c = config.Smollm3Config(
        hidden_size=2048, num_attention_heads=16, num_key_value_heads=4,
        head_dim=128, intermediate_size=11008, num_hidden_layers=36,
        rope_theta=5_000_000.0, rms_norm_eps=1e-6,
        vocab_size=128256, max_position_embeddings=65536,
        tie_word_embeddings=True, dtype=torch.float32,
        no_rope_layers=tuple(nope),
    )
    spec3 = c.to_block_spec(layer_idx=3)
    assert spec3.token_mixer.rope is None
    # Adjacent layers still have RoPE
    spec2 = c.to_block_spec(layer_idx=2)
    spec4 = c.to_block_spec(layer_idx=4)
    assert spec2.token_mixer.rope is not None
    assert spec4.token_mixer.rope is not None


def test_smollm3_config_handles_dtype_key():
    hf = {
        "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 4,
        "intermediate_size": 11008, "num_hidden_layers": 36,
        "rms_norm_eps": 1e-6, "vocab_size": 128256,
        "max_position_embeddings": 65536, "tie_word_embeddings": True,
        "dtype": "bfloat16",
        "rope_parameters": {"rope_theta": 5_000_000.0, "rope_type": "default"},
    }
    c = config.Smollm3Config.from_hf_dict(hf)
    assert c.dtype == torch.bfloat16
