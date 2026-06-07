"""Unit tests for Llama4ScoutConfig.from_hf_dict and per-layer dispatch."""
import torch

from api import specs, types
from models.llama4_scout import config


def _hf_scout_16e_dict() -> dict:
    """Mirror the published unsloth/Llama-4-Scout-17B-16E text_config."""
    return {
        "attention_bias": False,
        "attention_chunk_size": 8192,
        "head_dim": 128,
        "hidden_size": 5120,
        "intermediate_size": 8192,
        "intermediate_size_mlp": 16384,
        "max_position_embeddings": 262144,
        "no_rope_layers": [],                # empty -> derived from interval
        "num_attention_heads": 40,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "num_local_experts": 16,
        "rms_norm_eps": 1e-5,
        "rope_scaling": {
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "rope_theta": 500000.0,
        "torch_dtype": "bfloat16",
        "use_qk_norm": True,
        "vocab_size": 202048,
        "interleave_moe_layer_step": 1,
        "tie_word_embeddings": False,
    }


def test_llama4_scout_config_basic_fields():
    c = config.Llama4ScoutConfig.from_hf_dict(_hf_scout_16e_dict())
    assert c.hidden_size == 5120
    assert c.num_attention_heads == 40
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128
    assert c.intermediate_size == 8192
    assert c.intermediate_size_mlp == 16384
    assert c.num_hidden_layers == 48
    assert c.rope_theta == 500_000.0
    assert c.dtype == torch.bfloat16
    assert c.use_qk_norm is True
    assert c.attn_temperature_tuning is True
    assert c.floor_scale == 8192
    assert c.attn_scale == 0.1
    assert c.attention_bias is False


def test_llama4_scout_config_llama3_scaling_extracted():
    c = config.Llama4ScoutConfig.from_hf_dict(_hf_scout_16e_dict())
    assert c.rope_scaling_factor == 8.0
    assert c.rope_scaling_low_freq_factor == 1.0
    assert c.rope_scaling_high_freq_factor == 4.0
    assert c.rope_scaling_original_max_pos == 8192


def test_llama4_scout_config_derives_default_no_rope_pattern():
    """Empty `no_rope_layers` list (Scout default) -> derive from interval=4."""
    c = config.Llama4ScoutConfig.from_hf_dict(_hf_scout_16e_dict())
    assert len(c.no_rope_layers) == 48
    # Pattern: (i+1) % 4 != 0 -> RoPE (1); otherwise NoPE (0).
    assert c.layer_uses_rope(0) is True
    assert c.layer_uses_rope(1) is True
    assert c.layer_uses_rope(2) is True
    assert c.layer_uses_rope(3) is False     # NoPE
    assert c.layer_uses_rope(4) is True
    assert c.layer_uses_rope(7) is False     # NoPE
    assert c.layer_uses_rope(47) is False    # NoPE
    # Count: 12 NoPE / 36 RoPE.
    assert sum(c.no_rope_layers) == 36
    assert (len(c.no_rope_layers) - sum(c.no_rope_layers)) == 12


def test_llama4_scout_config_derives_default_layer_types():
    """When no_rope_layers is derived, layer_types follows: chunked on NoPE,
    full on RoPE."""
    c = config.Llama4ScoutConfig.from_hf_dict(_hf_scout_16e_dict())
    assert len(c.layer_types) == 48
    assert c.layer_types[0] == "full_attention"
    assert c.layer_types[3] == "chunked_attention"
    assert c.layer_types[4] == "full_attention"
    assert c.layer_types[47] == "chunked_attention"


def test_llama4_scout_config_moe_layers_16e_all():
    """16E variant: interleave_moe_layer_step=1 -> moe_layers covers ALL 48."""
    c = config.Llama4ScoutConfig.from_hf_dict(_hf_scout_16e_dict())
    assert len(c.moe_layers) == 48
    assert all(c.layer_is_moe(i) for i in range(48))


def test_llama4_scout_to_attention_spec_rope_layer():
    """RoPE layer (0) should produce an INTERLEAVED-basis RoPESpec with
    LLAMA3 scaling, CAUSAL mask, no QK-norm (deferred), no biases."""
    c = config.Llama4ScoutConfig.from_hf_dict(_hf_scout_16e_dict())
    spec = c.to_attention_spec(layer_idx=0)
    assert isinstance(spec, specs.AttentionSpec)
    assert spec.mask_kind == types.MaskKind.CAUSAL
    assert spec.qk_norm is None
    assert spec.qkv_layout == types.QKVLayout.SPLIT
    assert spec.n_q_heads == 40
    assert spec.n_kv_heads == 8
    assert spec.head_dim == 128
    assert spec.q_bias is False
    assert spec.rope is not None
    assert spec.rope.base_theta == 500_000.0
    assert spec.rope.basis == types.RoPEBasis.INTERLEAVED
    assert spec.rope.scaling == types.RoPEScaling.LLAMA3
    assert spec.rope.llama3_extra is not None
    assert spec.rope.llama3_extra.factor == 8.0


def test_llama4_scout_to_attention_spec_chunked_layer_raises():
    """NoPE layer (3) is "chunked_attention" in the default Scout pattern;
    chunked masks are not supported in B4 and must raise NotImplementedError."""
    import pytest
    c = config.Llama4ScoutConfig.from_hf_dict(_hf_scout_16e_dict())
    assert c.layer_idx_has_chunked_attention(3)
    with pytest.raises(NotImplementedError):
        c.to_attention_spec(layer_idx=3)


def test_llama4_scout_to_attention_spec_nope_layer_with_full_attention():
    """If a synthetic config forces a NoPE layer to "full_attention" the
    call succeeds and ``spec.rope is None``."""
    hf = _hf_scout_16e_dict()
    # Force layer_types so the NoPE layers are NOT chunked.
    hf["layer_types"] = ["full_attention"] * 48
    c = config.Llama4ScoutConfig.from_hf_dict(hf)
    spec = c.to_attention_spec(layer_idx=3)     # NoPE layer in default pattern
    assert spec.rope is None
    assert spec.mask_kind == types.MaskKind.CAUSAL


def test_llama4_scout_handles_rope_parameters_nested_format():
    """transformers 5.x nests rope_theta + rope_scaling under
    rope_parameters."""
    hf = _hf_scout_16e_dict()
    rs = hf.pop("rope_scaling")
    rt = hf.pop("rope_theta")
    hf["rope_parameters"] = {
        "rope_theta": rt,
        "rope_type": rs["rope_type"],
        "factor": rs["factor"],
        "low_freq_factor": rs["low_freq_factor"],
        "high_freq_factor": rs["high_freq_factor"],
        "original_max_position_embeddings": rs["original_max_position_embeddings"],
    }
    c = config.Llama4ScoutConfig.from_hf_dict(hf)
    assert c.rope_theta == 500_000.0
    assert c.rope_scaling_factor == 8.0


def test_llama4_scout_handles_dtype_alias():
    """Newer transformers uses `dtype`, older uses `torch_dtype`."""
    hf = _hf_scout_16e_dict()
    del hf["torch_dtype"]
    hf["dtype"] = "bfloat16"
    c = config.Llama4ScoutConfig.from_hf_dict(hf)
    assert c.dtype == torch.bfloat16


def test_llama4_scout_handles_text_config_nesting():
    """Top-level Llama4Config nests text fields under ``text_config``."""
    hf = {
        "text_config": _hf_scout_16e_dict(),
        "vision_config": {},
        "model_type": "llama4",
    }
    c = config.Llama4ScoutConfig.from_hf_dict(hf)
    assert c.hidden_size == 5120
    assert c.num_hidden_layers == 48
