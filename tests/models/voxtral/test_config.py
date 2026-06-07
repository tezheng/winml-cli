"""Voxtral config tests — verifies from_hf_dict and to_block_spec mirror the
HF source-of-truth for the canonical Voxtral-Mini-3B-2507 config.
"""
import torch

from api import specs, types
from models.voxtral import config as v_config


def _canonical_voxtral_3b_text_config() -> dict:
    """Mirror of `mistralai/Voxtral-Mini-3B-2507` text_config.

    Verified via live AutoConfig fetch:
      LlamaConfig(hidden_size=3072, num_hidden_layers=30,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        intermediate_size=8192,
        rope_parameters={'rope_theta': 1e8, 'rope_type': 'default'},
        rms_norm_eps=1e-5, vocab_size=131072, max_position_embeddings=131072,
        sliding_window=None, tie_word_embeddings=False)
    """
    return dict(
        model_type="llama",
        hidden_size=3072,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=8192,
        rope_parameters={"rope_theta": 1e8, "rope_type": "default"},
        rms_norm_eps=1e-5,
        vocab_size=131072,
        max_position_embeddings=131072,
        sliding_window=None,
        tie_word_embeddings=False,
        hidden_act="silu",
    )


def _nested_voxtral_hf_dict() -> dict:
    """Top-level VoxtralConfig dict with nested `text_config`."""
    return dict(
        model_type="voxtral",
        tie_word_embeddings=True,         # top-level — connects lm_head to embed_tokens
        text_config=_canonical_voxtral_3b_text_config(),
        audio_config={"model_type": "voxtral_encoder"},
        audio_token_id=24,
        projector_hidden_act="gelu",
    )


def test_from_hf_dict_canonical_dims_flat():
    cfg = v_config.VoxtralConfig.from_hf_dict(_canonical_voxtral_3b_text_config())
    assert cfg.hidden_size == 3072
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8           # GQA 4x reduction
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 8192
    assert cfg.num_hidden_layers == 30
    assert cfg.rope_theta == 1e8
    assert cfg.rms_norm_eps == 1e-5
    assert cfg.vocab_size == 131072
    assert cfg.max_position_embeddings == 131072
    assert cfg.sliding_window is None
    assert cfg.tie_word_embeddings is False
    assert cfg.dtype == torch.float32


def test_from_hf_dict_canonical_dims_nested():
    """Voxtral top-level config with nested text_config — the adapter must
    unwrap to the same flat config."""
    cfg = v_config.VoxtralConfig.from_hf_dict(_nested_voxtral_hf_dict())
    assert cfg.hidden_size == 3072
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.intermediate_size == 8192
    assert cfg.rope_theta == 1e8


def test_to_block_spec_canonical_shapes():
    cfg = v_config.VoxtralConfig.from_hf_dict(_canonical_voxtral_3b_text_config())
    spec = cfg.to_block_spec(layer_idx=0)
    # Norms.
    assert spec.attn_norm_position == types.NormPosition.PRE
    assert spec.ffn_norm_position == types.NormPosition.PRE
    assert spec.pre_attn_norm.kind == types.NormKind.RMS
    assert spec.pre_attn_norm.eps == 1e-5
    assert spec.pre_attn_norm.weight_mode == types.NormWeightMode.STANDARD_W
    # Attention.
    attn = spec.token_mixer
    assert isinstance(attn, specs.AttentionSpec)
    assert attn.kind == types.AttentionKind.STANDARD
    assert attn.qkv_layout == types.QKVLayout.SPLIT
    assert attn.mask_kind == types.MaskKind.CAUSAL
    assert attn.sliding_window is None
    assert attn.n_q_heads == 32
    assert attn.n_kv_heads == 8
    assert attn.head_dim == 128
    assert not (attn.q_bias or attn.k_bias or attn.v_bias or attn.o_bias)
    assert attn.qk_norm is None
    assert attn.rope.base_theta == 1e8
    assert attn.rope.basis == types.RoPEBasis.SPLIT_HALF
    assert attn.rope.scaling == types.RoPEScaling.NONE
    # FFN — NOT fused (Llama-style separate gate/up).
    ffn = spec.channel_mixer
    assert isinstance(ffn, specs.FFNSpec)
    assert ffn.intermediate_size == 8192
    assert ffn.gate_kind == types.GateKind.SWIGLU
    assert ffn.fused_gate_up is False
    assert not (ffn.gate_bias or ffn.up_bias or ffn.down_bias)


def test_sliding_window_set_promotes_to_swa_mask():
    """If a hypothetical Voxtral variant set sliding_window, mask_kind
    should become SWA. (Not used by Voxtral-Mini-3B-2507, but tested for
    spec correctness.)"""
    hf = _canonical_voxtral_3b_text_config()
    hf["sliding_window"] = 4096
    cfg = v_config.VoxtralConfig.from_hf_dict(hf)
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert spec.token_mixer.sliding_window == 4096


def test_handles_missing_num_kv_heads_defaults_to_num_attention_heads():
    hf = _canonical_voxtral_3b_text_config()
    hf.pop("num_key_value_heads")
    cfg = v_config.VoxtralConfig.from_hf_dict(hf)
    assert cfg.num_key_value_heads == cfg.num_attention_heads == 32
