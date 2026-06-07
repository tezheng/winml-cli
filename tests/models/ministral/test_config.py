"""Unit tests for MinistralConfig from_hf_dict and per-layer dispatch."""
import torch

from api import types
from models.ministral import config


# The published `mistralai/Ministral-8B-Instruct-2410` advertises the 1:3
# pattern {full, sliding, sliding, sliding} repeated over 36 layers.
_8B_2410_LAYER_TYPES = (
    ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"] * 9
)


def _hf_8b_2410_dict() -> dict:
    return {
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 12288, "num_hidden_layers": 36,
        "rms_norm_eps": 1e-5, "vocab_size": 131072,
        "max_position_embeddings": 32768, "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "rope_theta": 100_000_000.0,
        "sliding_window": 32768,
        "layer_types": _8B_2410_LAYER_TYPES,
    }


def test_ministral_config_from_hf_dict_8b_2410():
    c = config.MinistralConfig.from_hf_dict(_hf_8b_2410_dict())
    assert c.hidden_size == 4096
    assert c.num_attention_heads == 32
    assert c.num_key_value_heads == 8
    assert c.head_dim == 128
    assert c.intermediate_size == 12288
    assert c.num_hidden_layers == 36
    assert c.rope_theta == 100_000_000.0
    assert c.sliding_window == 32768
    assert c.rms_norm_eps == 1e-5
    assert c.dtype == torch.bfloat16
    assert len(c.layer_types) == 36
    # 1:3 pattern: 0 full, 1-3 sliding, 4 full, ...
    assert c.layer_type(0) == "full_attention"
    assert c.layer_type(1) == "sliding_attention"
    assert c.layer_type(2) == "sliding_attention"
    assert c.layer_type(3) == "sliding_attention"
    assert c.layer_type(4) == "full_attention"
    assert c.layer_type(35) == "sliding_attention"


def test_ministral_config_handles_torch_dtype_alias():
    """Older HF configs use `torch_dtype` while newer ones use `dtype`."""
    hf = _hf_8b_2410_dict()
    del hf["dtype"]
    hf["torch_dtype"] = "bfloat16"
    c = config.MinistralConfig.from_hf_dict(hf)
    assert c.dtype == torch.bfloat16


def test_ministral_config_handles_rope_parameters_nested():
    """transformers 5.x nests rope_theta inside rope_parameters."""
    hf = _hf_8b_2410_dict()
    del hf["rope_theta"]
    hf["rope_parameters"] = {
        "rope_theta": 100_000_000.0, "rope_type": "default"
    }
    c = config.MinistralConfig.from_hf_dict(hf)
    assert c.rope_theta == 100_000_000.0


def test_ministral_config_default_layer_types_when_missing():
    """When `layer_types` is absent, fall back to the
    configuration_ministral.py:92-95 rule (all sliding when sliding_window is
    set, else all full)."""
    hf = _hf_8b_2410_dict()
    del hf["layer_types"]
    c = config.MinistralConfig.from_hf_dict(hf)
    assert len(c.layer_types) == 36
    assert all(lt == "sliding_attention" for lt in c.layer_types)


def test_ministral_config_default_layer_types_no_swa():
    """When sliding_window is None, default layer_types is all full."""
    hf = _hf_8b_2410_dict()
    del hf["layer_types"]
    hf["sliding_window"] = None
    c = config.MinistralConfig.from_hf_dict(hf)
    assert all(lt == "full_attention" for lt in c.layer_types)


def test_ministral_to_block_spec_full_layer():
    c = config.MinistralConfig.from_hf_dict(_hf_8b_2410_dict())
    spec = c.to_block_spec(layer_idx=0)
    assert spec.token_mixer.mask_kind == types.MaskKind.CAUSAL
    assert spec.token_mixer.sliding_window is None
    assert spec.token_mixer.rope is not None
    assert spec.token_mixer.rope.base_theta == 100_000_000.0
    assert spec.token_mixer.qk_norm is None
    assert not spec.token_mixer.q_bias
    assert spec.channel_mixer.gate_kind == types.GateKind.SWIGLU
    assert spec.attn_norm_position == types.NormPosition.PRE


def test_ministral_to_block_spec_sliding_layer():
    c = config.MinistralConfig.from_hf_dict(_hf_8b_2410_dict())
    spec = c.to_block_spec(layer_idx=1)
    assert spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert spec.token_mixer.sliding_window == 32768
    assert spec.token_mixer.rope is not None
    # Sliding layers also use the SAME rope_theta (Ministral has ONE rope
    # base, not Gemma 3's dual-theta).
    assert spec.token_mixer.rope.base_theta == 100_000_000.0


def test_ministral_to_block_spec_no_swa_when_sliding_window_unset():
    """Even on a layer marked "sliding_attention", if sliding_window is None
    the IR falls back to CAUSAL (matches modeling_ministral.py:155 where
    self.sliding_window stays None unless config.sliding_window is set)."""
    hf = _hf_8b_2410_dict()
    hf["sliding_window"] = None
    c = config.MinistralConfig.from_hf_dict(hf)
    # layer_types defaults to all "full_attention" when sliding_window is None
    # (see test_ministral_config_default_layer_types_no_swa). Force a sliding
    # entry to exercise the cfg path.
    c = config.MinistralConfig(
        **{**{k: getattr(c, k) for k in c.__dataclass_fields__},
           "layer_types": ("sliding_attention",) * c.num_hidden_layers}
    )
    spec = c.to_block_spec(layer_idx=0)
    assert spec.token_mixer.mask_kind == types.MaskKind.CAUSAL
    assert spec.token_mixer.sliding_window is None
