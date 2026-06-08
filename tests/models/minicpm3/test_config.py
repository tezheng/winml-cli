"""Config parsing test for MiniCPM-3."""
import math

import torch

from api import specs, types
from models.minicpm3 import config as m_config


def _sample_hf_dict():
    """Mirrors hf_cache/.../openbmb--MiniCPM3-4B/config.json."""
    return {
        "hidden_size": 2560,
        "num_attention_heads": 40,
        "num_hidden_layers": 62,
        "intermediate_size": 6400,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "q_lora_rank": 768,
        "kv_lora_rank": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "vocab_size": 73448,
        "max_position_embeddings": 32768,
        "rope_scaling": {
            "type": "longrope",
            "short_factor": [1.0] * 16,
            "long_factor": [2.0] * 16,
            "original_max_position_embeddings": 32768,
        },
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
    }


def test_config_parses_minicpm3_4b():
    hf = _sample_hf_dict()
    cfg = m_config.MiniCPM3Config.from_hf_dict(hf)
    assert cfg.hidden_size == 2560
    assert cfg.num_attention_heads == 40
    assert cfg.num_hidden_layers == 62
    assert cfg.q_lora_rank == 768
    assert cfg.kv_lora_rank == 256
    assert cfg.qk_nope_head_dim == 64
    assert cfg.qk_rope_head_dim == 32
    # v_head_dim derived: 2560 / 40 = 64
    assert cfg.v_head_dim == 64
    assert cfg.qk_head_dim == 96
    assert cfg.scale_emb == 12.0
    assert cfg.scale_depth == 1.4
    assert cfg.dim_model_base == 256
    assert cfg.rope_type == "longrope"
    assert cfg.longrope_short_factor is not None
    assert len(cfg.longrope_short_factor) == 16  # qk_rope_head_dim/2
    assert cfg.dtype == torch.bfloat16


def test_config_to_block_spec_mla():
    hf = _sample_hf_dict()
    cfg = m_config.MiniCPM3Config.from_hf_dict(hf)
    spec = cfg.to_block_spec(layer_idx=0)
    # Token mixer is MLA
    am = spec.token_mixer
    assert am.kind == types.AttentionKind.MLA
    assert am.qkv_layout == types.QKVLayout.MLA_LATENT
    assert am.n_q_heads == 40
    assert am.n_kv_heads == 40
    assert am.head_dim == 96
    assert am.q_lora_rank == 768
    assert am.kv_lora_rank == 256
    assert am.qk_nope_head_dim == 64
    assert am.qk_rope_head_dim == 32
    assert am.v_head_dim == 64
    # RoPE wires LongRoPE with cos/sin on the rope subspace only.
    assert am.rope.scaling == types.RoPEScaling.LONGROPE
    assert am.rope.partial_rotary_factor == 1.0
    # Residual scale = scale_depth / sqrt(L) = 1.4 / sqrt(62).
    expected = 1.4 / math.sqrt(62)
    assert math.isclose(spec.residual_scale, expected, rel_tol=1e-12)


def test_config_to_block_spec_default_rope_no_scaling():
    hf = _sample_hf_dict()
    hf.pop("rope_scaling")
    cfg = m_config.MiniCPM3Config.from_hf_dict(hf)
    spec = cfg.to_block_spec()
    assert spec.token_mixer.rope.scaling == types.RoPEScaling.NONE
