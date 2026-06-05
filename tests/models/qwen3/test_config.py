import torch

from models.qwen3 import config


def test_qwen3_config_from_hf_dict():
    hf = {
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }
    c = config.Qwen3Config.from_hf_dict(hf)
    assert c.hidden_size == 1024
    assert c.num_attention_heads == 16
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128
    assert c.rope_theta == 1_000_000.0
    assert c.dtype == torch.bfloat16
    assert c.tie_word_embeddings is True


def test_qwen3_config_to_block_spec():
    c = config.Qwen3Config(
        hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, intermediate_size=3072, num_hidden_layers=28,
        rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        vocab_size=151936, max_position_embeddings=32768,
        tie_word_embeddings=True, dtype=torch.float32,
    )
    block_spec = c.to_block_spec()
    assert block_spec.token_mixer.n_q_heads == 16
    assert block_spec.token_mixer.n_kv_heads == 8
    assert block_spec.token_mixer.head_dim == 128
    assert block_spec.token_mixer.qk_norm is not None
    from api import types as _t
    assert block_spec.token_mixer.qk_norm_phase == _t.QKNormPhase.PRE_ROPE
    assert block_spec.token_mixer.qk_norm_shape == _t.QKNormShape.PER_HEAD_DH
    assert block_spec.channel_mixer.intermediate_size == 3072
    assert block_spec.channel_mixer.gate_kind == _t.GateKind.SWIGLU
