"""Isolation tests — each api primitive vs the corresponding HF OLMo 2 op,
at atol=1e-5. Run on CPU with deterministic seeds.

These tests instantiate HF modules WITHOUT downloading model weights — they
construct small configs in-process. So they always run as long as
``transformers.models.olmo2`` is importable.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
try:
    from transformers.models.olmo2 import modeling_olmo2
    from transformers.models.olmo2.configuration_olmo2 import Olmo2Config as HFOlmo2Config
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers.models.olmo2 not available — install transformers>=5.10",
        allow_module_level=True,
    )

from api import norm as _norm, ops, rope, specs, types


def _hf_cfg():
    return HFOlmo2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
        intermediate_size=128, num_hidden_layers=2,
        rms_norm_eps=1e-5, vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=False, torch_dtype="float32",
        rope_parameters={"rope_theta": 500_000.0, "rope_type": "default"},
        attention_bias=False,
    )


def test_rms_norm_matches_hf_olmo2():
    """Olmo2RMSNorm: (1) cast to fp32, (2) variance/rsqrt, (3) multiply by weight,
    (4) cast back to input dtype. Identical math to our ops.rms_norm(..., mode='standard_w').
    Source: modeling_olmo2.py:60-65.
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_norm = modeling_olmo2.Olmo2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="standard_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_freqs_match_hf_olmo2_theta_500000():
    """HF Olmo2RotaryEmbedding cos/sin = inv_freq tables tied to rope_theta=500000.
    Source: modeling_olmo2.py:71-132 (Olmo2RotaryEmbedding).
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_olmo2.Olmo2RotaryEmbedding(cfg)
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)
    rope_theta = cfg.rope_parameters["rope_theta"]
    rope_spec = specs.RoPESpec(base_theta=rope_theta,
                               basis=types.RoPEBasis.SPLIT_HALF,
                               scaling=types.RoPEScaling.NONE)
    api_rope = rope.RoPE(rope_spec, head_dim=head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_swiglu_matches_hf_olmo2_mlp():
    """Olmo2MLP is bog-standard SwiGLU (gate_proj, up_proj, down_proj).
    Source: modeling_olmo2.py:279-292.
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_mlp = modeling_olmo2.Olmo2MLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)
    g = F.linear(x, hf_mlp.gate_proj.weight)
    u = F.linear(x, hf_mlp.up_proj.weight)
    h = ops.mul(ops.silu(g), u)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_qk_norm_full_hdh_matches_hf_olmo2():
    """B3 core: FULL_HDH QK-norm shape semantics.

    HF Olmo2RMSNorm is applied to the FLATTENED [B, S, H*Dh] tensor BEFORE the
    reshape to [B, S, H, Dh]. Our api.norm.QKNorm receives the post-reshape
    [B, S, H, Dh] tensor and reshapes back to [B, S, H*Dh] inside forward.

    Both must produce identical results when the weight vector is the same.
    Source: modeling_olmo2.py:231-249.
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    H = cfg.num_attention_heads
    Dh = cfg.hidden_size // H
    hf_qn = modeling_olmo2.Olmo2RMSNorm(H * Dh, eps=cfg.rms_norm_eps)
    weight = torch.randn(H * Dh)
    hf_qn.weight.data.copy_(weight)

    x = torch.randn(1, 3, H * Dh)        # HF applies on flat [B, S, H*Dh]
    hf_out = hf_qn(x).view(1, 3, H, Dh)

    # Our QKNorm takes [B, S, H, Dh].
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=cfg.rms_norm_eps,
                               weight_mode=types.NormWeightMode.STANDARD_W)
    qkn = _norm.QKNorm(norm_spec, head_dim=Dh,
                       shape=types.QKNormShape.FULL_HDH,
                       n_heads=H, dtype=torch.float32)
    qkn.weight.data.copy_(weight)
    x_reshaped = x.view(1, 3, H, Dh)
    api_out = qkn(x_reshaped)
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_hf_olmo2_decoder_layer_has_no_input_layernorm():
    """OLMo 2 is POST-norm. Verify our source citation: the HF decoder layer
    does NOT instantiate `input_layernorm` or `pre_feedforward_layernorm`.
    Source: modeling_olmo2.py:295-303 (Olmo2DecoderLayer.__init__).
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_layer = modeling_olmo2.Olmo2DecoderLayer(cfg, layer_idx=0)
    assert not hasattr(hf_layer, "input_layernorm")
    assert not hasattr(hf_layer, "pre_feedforward_layernorm")
    # And does have the POST-norm modules.
    assert hasattr(hf_layer, "post_attention_layernorm")
    assert hasattr(hf_layer, "post_feedforward_layernorm")
