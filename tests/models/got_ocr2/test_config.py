"""GOT-OCR 2.0 config tests — synthetic HF dict + spec adapter."""
from __future__ import annotations

import torch

from api import specs, types
from models.got_ocr2 import config as g_config


_HF_GOT_OCR2 = {
    # GotOcr2Config top-level (mirrors stepfun-ai/GOT-OCR-2.0-hf config.json
    # but inlined to avoid a network round-trip in the unit test).
    "model_type": "got_ocr2",
    "tie_word_embeddings": True,
    "image_token_index": 151859,
    "image_seq_length": 576,
    "text_config": {
        # Qwen2 sub-config — defaults from configuration_got_ocr2.py:108-127.
        "model_type": "qwen2",
        "vocab_size": 151860,
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
        "use_sliding_window": False,
        "sliding_window": None,
    },
}


def test_from_hf_dict_parses_text_config():
    cfg = g_config.GotOcr2Config.from_hf_dict(_HF_GOT_OCR2)
    assert cfg.hidden_size == 1024
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 16
    assert cfg.head_dim == 64                         # 1024 / 16
    assert cfg.intermediate_size == 2816
    assert cfg.num_hidden_layers == 24
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.vocab_size == 151860
    assert cfg.max_position_embeddings == 32768
    assert cfg.tie_word_embeddings is True
    assert cfg.sliding_window is None                 # disabled by use_sliding_window=False
    assert cfg.image_token_index == 151859
    assert cfg.image_seq_length == 576
    assert cfg.dtype == torch.float32                 # no explicit dtype in dict


def test_to_block_spec_has_qkv_biases_no_qk_norm():
    """Qwen2-distinctive properties: QKV biases TRUE, NO QK-norm,
    SPLIT_HALF RoPE basis."""
    cfg = g_config.GotOcr2Config.from_hf_dict(_HF_GOT_OCR2)
    spec = cfg.to_block_spec(layer_idx=0)
    attn = spec.token_mixer
    assert isinstance(attn, specs.AttentionSpec)
    assert attn.kind == types.AttentionKind.STANDARD
    assert attn.qkv_layout == types.QKVLayout.SPLIT
    assert attn.q_bias is True
    assert attn.k_bias is True
    assert attn.v_bias is True
    assert attn.o_bias is False
    assert attn.qk_norm is None
    assert attn.mask_kind == types.MaskKind.CAUSAL
    assert attn.sliding_window is None
    assert attn.rope is not None
    assert attn.rope.basis == types.RoPEBasis.SPLIT_HALF
    assert attn.rope.scaling == types.RoPEScaling.NONE
    assert attn.rope.base_theta == 1_000_000.0


def test_to_block_spec_ffn_is_swiglu_no_biases():
    cfg = g_config.GotOcr2Config.from_hf_dict(_HF_GOT_OCR2)
    spec = cfg.to_block_spec(layer_idx=0)
    ffn = spec.channel_mixer
    assert isinstance(ffn, specs.FFNSpec)
    assert ffn.gate_kind == types.GateKind.SWIGLU
    assert ffn.activation == types.Activation.SILU
    assert ffn.fused_gate_up is False
    assert ffn.gate_bias is False
    assert ffn.up_bias is False
    assert ffn.down_bias is False
    assert ffn.intermediate_size == 2816


def test_to_block_spec_uses_pre_norm():
    cfg = g_config.GotOcr2Config.from_hf_dict(_HF_GOT_OCR2)
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.attn_norm_position == types.NormPosition.PRE
    assert spec.ffn_norm_position == types.NormPosition.PRE
    assert spec.pre_attn_norm is not None
    assert spec.pre_ffn_norm is not None
    assert spec.post_attn_norm is None
    assert spec.post_ffn_norm is None
