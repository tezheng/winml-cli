import torch

from models.granite import config


def _granite_31_2b_hf():
    """ibm-granite/granite-3.1-2b-base config.json (verified against the
    actual HF repo)."""
    return {
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 8192, "num_hidden_layers": 40,
        "rms_norm_eps": 1e-5, "vocab_size": 49152,
        "max_position_embeddings": 131072, "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "rope_theta": 5_000_000.0, "rope_scaling": None,
        "attention_multiplier": 0.015625,
        "embedding_multiplier": 12.0,
        "logits_scaling": 8.0,
        "residual_multiplier": 0.22,
        "attention_bias": False, "mlp_bias": False, "attention_dropout": 0.1,
    }


def test_granite_config_from_hf_dict_3_1_2b():
    c = config.GraniteConfig.from_hf_dict(_granite_31_2b_hf())
    assert c.hidden_size == 2048
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 8
    assert c.head_dim == 64                     # derived: 2048 / 32
    assert c.rope_theta == 5_000_000.0
    assert c.attention_multiplier == 0.015625
    assert c.embedding_multiplier == 12.0
    assert c.logits_scaling == 8.0
    assert c.residual_multiplier == 0.22
    assert c.tie_word_embeddings is True
    assert c.dtype == torch.bfloat16


def test_granite_config_handles_dtype_key():
    """transformers 5.x emits 'dtype', not 'torch_dtype'."""
    hf = dict(_granite_31_2b_hf())
    hf.pop("torch_dtype")
    hf["dtype"] = "bfloat16"
    c = config.GraniteConfig.from_hf_dict(hf)
    assert c.dtype == torch.bfloat16


def test_granite_config_handles_rope_parameters_dict():
    """transformers 5.x packs rope_theta into rope_parameters."""
    hf = dict(_granite_31_2b_hf())
    hf.pop("rope_theta")
    hf["rope_parameters"] = {"rope_theta": 5_000_000.0, "rope_type": "default"}
    c = config.GraniteConfig.from_hf_dict(hf)
    assert c.rope_theta == 5_000_000.0


def test_granite_config_defaults_when_multipliers_missing():
    """Older Granite checkpoints / non-Granite configs default to 1.0 — a no-op
    that makes the block behave like vanilla Llama."""
    hf = {
        "hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 8,
        "intermediate_size": 1024, "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5, "vocab_size": 1000,
        "max_position_embeddings": 2048, "tie_word_embeddings": False,
        "rope_theta": 10000.0,
    }
    c = config.GraniteConfig.from_hf_dict(hf)
    assert c.attention_multiplier == 1.0
    assert c.embedding_multiplier == 1.0
    assert c.logits_scaling == 1.0
    assert c.residual_multiplier == 1.0


def test_granite_config_to_block_spec_wires_mu_p_scalars():
    c = config.GraniteConfig.from_hf_dict(_granite_31_2b_hf())
    spec = c.to_block_spec()
    # μP attention scale (NOT 1/sqrt(Dh)).
    assert spec.token_mixer.attn_scale == 0.015625
    # μP residual scale plumbed.
    assert spec.residual_scale == 0.22
    # μP model-level scalars carried for assembly.
    assert spec.embedding_scale == 12.0
    assert spec.logits_scale == 8.0
    # Architectural sanity — Llama-shaped block.
    from api import types as _t
    assert spec.token_mixer.kind == _t.AttentionKind.STANDARD
    assert spec.token_mixer.qkv_layout == _t.QKVLayout.SPLIT
    assert spec.token_mixer.qk_norm is None
    assert spec.channel_mixer.gate_kind == _t.GateKind.SWIGLU
    assert spec.attn_norm_position == _t.NormPosition.PRE
    assert spec.ffn_norm_position == _t.NormPosition.PRE
