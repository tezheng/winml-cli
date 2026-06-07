import torch

from models.tinyllama import config


def test_tinyllama_config_from_hf_dict():
    """TinyLlama-1.1B-Chat-v1.0 — GQA with n_q=32, n_kv=4, head_dim=64."""
    hf = {
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
        "head_dim": 64, "intermediate_size": 5632, "num_hidden_layers": 22,
        "rms_norm_eps": 1e-5, "vocab_size": 32000,
        "max_position_embeddings": 2048, "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "attention_bias": False, "mlp_bias": False,
    }
    c = config.TinyLlamaConfig.from_hf_dict(hf)
    assert c.hidden_size == 2048
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 4   # GQA (NOT MHA — research v3 §8.4 surprise)
    assert c.head_dim == 64
    assert c.intermediate_size == 5632
    assert c.num_hidden_layers == 22
    assert c.rope_theta == 10000.0
    assert c.dtype == torch.bfloat16


def test_tinyllama_config_to_block_spec_default_rope():
    hf = {
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
        "head_dim": 64, "intermediate_size": 5632, "num_hidden_layers": 22,
        "rms_norm_eps": 1e-5, "vocab_size": 32000,
        "max_position_embeddings": 2048, "tie_word_embeddings": False,
        "dtype": "float32",
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    }
    c = config.TinyLlamaConfig.from_hf_dict(hf)
    spec = c.to_block_spec()
    from api import types as _t
    assert spec.token_mixer.n_q_heads == 32
    assert spec.token_mixer.n_kv_heads == 4
    assert spec.token_mixer.head_dim == 64
    assert spec.token_mixer.qk_norm is None
    assert spec.token_mixer.rope.base_theta == 10000.0
    assert spec.token_mixer.rope.scaling == _t.RoPEScaling.NONE
    assert spec.channel_mixer.gate_kind == _t.GateKind.SWIGLU
