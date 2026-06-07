import torch

from models.llama3 import config


def test_llama3_config_from_hf_dict_3p0_default_rope():
    """Llama 3.0 8B has rope_type='default'."""
    hf = {
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 14336, "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5, "vocab_size": 128256,
        "max_position_embeddings": 8192, "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    }
    c = config.Llama3Config.from_hf_dict(hf)
    assert c.hidden_size == 4096
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128
    assert c.rope_theta == 500000.0
    assert c.rope_type == "default"
    assert c.rope_factor is None
    assert c.dtype == torch.bfloat16


def test_llama3_config_from_hf_dict_3p2_llama3_scaling():
    """Llama 3.2 1B has rope_type='llama3' with factor=32 (NOT 8)."""
    hf = {
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 64, "intermediate_size": 8192, "num_hidden_layers": 16,
        "rms_norm_eps": 1e-5, "vocab_size": 128256,
        "max_position_embeddings": 131072, "tie_word_embeddings": True,
        "dtype": "bfloat16",  # 5.x key
        "rope_parameters": {
            "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3", "rope_theta": 500000.0,
        },
    }
    c = config.Llama3Config.from_hf_dict(hf)
    assert c.rope_type == "llama3"
    assert c.rope_factor == 32.0
    assert c.rope_low_freq_factor == 1.0
    assert c.rope_high_freq_factor == 4.0
    assert c.rope_original_max_position_embeddings == 8192
    assert c.rope_theta == 500000.0
    assert c.dtype == torch.bfloat16
    assert c.tie_word_embeddings is True


def test_llama3_config_handles_legacy_4x_rope_scaling():
    """Backward compat: transformers 4.x emitted `rope_scaling` (flat dict)."""
    hf = {
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 64, "intermediate_size": 8192, "num_hidden_layers": 16,
        "rms_norm_eps": 1e-5, "vocab_size": 128256,
        "max_position_embeddings": 131072, "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3",
        },
    }
    c = config.Llama3Config.from_hf_dict(hf)
    assert c.rope_type == "llama3"
    assert c.rope_factor == 32.0


def test_llama3_config_to_block_spec_default_rope():
    c = config.Llama3Config(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=14336, num_hidden_layers=32,
        rope_theta=500000.0, rope_type="default", rms_norm_eps=1e-5,
        vocab_size=128256, max_position_embeddings=8192,
        tie_word_embeddings=False, dtype=torch.float32,
    )
    spec = c.to_block_spec(layer_idx=0)
    from api import types as _t
    assert spec.token_mixer.n_q_heads == 32
    assert spec.token_mixer.n_kv_heads == 8
    assert spec.token_mixer.qk_norm is None
    assert spec.token_mixer.rope is not None
    assert spec.token_mixer.rope.base_theta == 500000.0
    assert spec.token_mixer.rope.scaling == _t.RoPEScaling.NONE
    assert spec.channel_mixer.gate_kind == _t.GateKind.SWIGLU
    assert spec.channel_mixer.activation == _t.Activation.SILU


def test_llama3_config_to_block_spec_llama3_scaling():
    c = config.Llama3Config(
        hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
        head_dim=64, intermediate_size=8192, num_hidden_layers=16,
        rope_theta=500000.0, rope_type="llama3",
        rope_factor=32.0, rope_low_freq_factor=1.0, rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
        rms_norm_eps=1e-5, vocab_size=128256,
        max_position_embeddings=131072, tie_word_embeddings=True,
        dtype=torch.float32,
    )
    spec = c.to_block_spec(layer_idx=0)
    from api import types as _t
    assert spec.token_mixer.rope.scaling == _t.RoPEScaling.LLAMA3
    extra = spec.token_mixer.rope.llama3_extra
    assert extra is not None
    assert extra.factor == 32.0
    assert extra.original_context_length == 8192
