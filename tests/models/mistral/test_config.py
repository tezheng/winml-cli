import torch

from models.mistral import config


def test_mistral_config_from_hf_dict_v03():
    """Mistral 7B v0.3 — sliding_window=None (DROPPED in v0.3)."""
    hf = {
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 14336, "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5, "vocab_size": 32768,
        "max_position_embeddings": 32768, "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "sliding_window": None,
    }
    c = config.MistralConfig.from_hf_dict(hf)
    assert c.hidden_size == 4096
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128
    assert c.rope_theta == 1_000_000.0
    assert c.sliding_window is None
    assert c.dtype == torch.bfloat16


def test_mistral_config_from_hf_dict_v01_with_swa():
    """Mistral 7B v0.1 — sliding_window=4096."""
    hf = {
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 14336, "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5, "vocab_size": 32000,
        "max_position_embeddings": 32768, "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_theta": 10000.0,
        "sliding_window": 4096,
    }
    c = config.MistralConfig.from_hf_dict(hf)
    assert c.sliding_window == 4096
    assert c.rope_theta == 10000.0  # v0.1 used the original RoPE theta


def test_mistral_config_handles_dtype_key():
    """transformers 5.x emits 'dtype', not 'torch_dtype' — must handle both."""
    hf = {
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 14336, "num_hidden_layers": 32,
        "rms_norm_eps": 1e-5, "vocab_size": 32768,
        "max_position_embeddings": 32768, "tie_word_embeddings": False,
        "dtype": "bfloat16",  # 5.x key
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    }
    c = config.MistralConfig.from_hf_dict(hf)
    assert c.dtype == torch.bfloat16


def test_mistral_config_to_block_spec_no_swa():
    c = config.MistralConfig(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=14336, num_hidden_layers=32,
        rope_theta=1_000_000.0, rms_norm_eps=1e-5,
        vocab_size=32768, max_position_embeddings=32768,
        tie_word_embeddings=False, dtype=torch.float32,
        sliding_window=None,
    )
    spec = c.to_block_spec()
    from api import types as _t
    assert spec.token_mixer.n_q_heads == 32
    assert spec.token_mixer.n_kv_heads == 8
    assert spec.token_mixer.mask_kind == _t.MaskKind.CAUSAL
    assert spec.token_mixer.sliding_window is None
    assert spec.token_mixer.qk_norm is None
    assert spec.token_mixer.rope.base_theta == 1_000_000.0
    assert spec.channel_mixer.gate_kind == _t.GateKind.SWIGLU


def test_mistral_config_to_block_spec_with_swa():
    c = config.MistralConfig(
        hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, intermediate_size=14336, num_hidden_layers=32,
        rope_theta=10000.0, rms_norm_eps=1e-5,
        vocab_size=32000, max_position_embeddings=32768,
        tie_word_embeddings=False, dtype=torch.float32,
        sliding_window=4096,
    )
    spec = c.to_block_spec()
    from api import types as _t
    assert spec.token_mixer.mask_kind == _t.MaskKind.SWA
    assert spec.token_mixer.sliding_window == 4096
