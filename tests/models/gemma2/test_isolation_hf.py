"""Isolation tests — each api primitive vs the corresponding HF Gemma 2 op,
at atol=1e-5. Run on CPU with deterministic seeds.

These tests instantiate HF modules WITHOUT downloading model weights — they
construct small configs in-process.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
try:
    from transformers.models.gemma2 import modeling_gemma2
    from transformers.models.gemma2.configuration_gemma2 import Gemma2Config as HFG2Config
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers.models.gemma2 not available — install transformers>=5.10",
        allow_module_level=True,
    )

from api import ops, rope, specs, types


def _hf_cfg():
    return HFG2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=4,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rms_norm_eps=1e-6,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=True, torch_dtype="float32",
        sliding_window=8, query_pre_attn_scalar=16,
        attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
        attention_bias=False, hidden_activation="gelu_pytorch_tanh",
    )


def test_rms_norm_matches_hf_gemma2_one_plus_w():
    """Gemma2RMSNorm: rms_norm(x.float()) * (1 + weight.float()), cast to x.dtype.
    Source: modeling_gemma2.py:49-65.
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_norm = modeling_gemma2.Gemma2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    # HF initialises with zeros; for a real test we copy random weights.
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="one_plus_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_freqs_match_hf_gemma2_default_theta():
    """HF Gemma2RotaryEmbedding cos/sin = inv_freq tables tied to rope_theta=10000.
    Source: modeling_gemma2.py:85-147.
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_gemma2.Gemma2RotaryEmbedding(cfg)
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, cfg.head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)
    rope_theta = cfg.rope_parameters["rope_theta"]
    rope_spec = specs.RoPESpec(base_theta=rope_theta,
                               basis=types.RoPEBasis.SPLIT_HALF,
                               scaling=types.RoPEScaling.NONE)
    api_rope = rope.RoPE(rope_spec, head_dim=cfg.head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_geglu_matches_hf_gemma2_mlp():
    """Gemma2MLP: down(gelu_pytorch_tanh(gate(x)) * up(x)). Source: L69-82."""
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_mlp = modeling_gemma2.Gemma2MLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)
    g = F.linear(x, hf_mlp.gate_proj.weight)
    u = F.linear(x, hf_mlp.up_proj.weight)
    h = ops.mul(ops.gelu_pytorch_tanh(g), u)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_sdpa_softcap_matches_hf_gemma2_eager():
    """B3 critical: our sdpa with logit_softcap matches HF eager attention math.
    Source: modeling_gemma2.py:212-225.
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    B, S = 1, 6
    Hq = cfg.num_attention_heads
    Hk = cfg.num_key_value_heads
    Dh = cfg.head_dim
    q = torch.randn(B, Hq, S, Dh)
    k = torch.randn(B, Hk, S, Dh)
    v = torch.randn(B, Hk, S, Dh)
    # Mock module for HF eager — only needs num_key_value_groups and head_dim.
    class _Mod:
        num_key_value_groups = Hq // Hk
        head_dim = Dh
        training = False
    causal = torch.zeros(S, S)
    causal = causal.masked_fill(
        torch.arange(S).unsqueeze(0) > torch.arange(S).unsqueeze(1),
        float("-inf"),
    )
    attn_mask = causal.unsqueeze(0).unsqueeze(0)
    hf_out, _ = modeling_gemma2.eager_attention_forward(
        _Mod(), q, k, v, attn_mask, dropout=0.0,
        scaling=cfg.query_pre_attn_scalar**-0.5, softcap=cfg.attn_logit_softcapping,
    )
    # HF returns [B, S, H, Dh] after `attn_output.transpose(1,2).contiguous()`.
    # Our sdpa returns [B, H, S, Dh].
    api_out = ops.sdpa(q, k, v, attn_mask=attn_mask,
                       scale=cfg.query_pre_attn_scalar**-0.5,
                       logit_softcap=cfg.attn_logit_softcapping)
    api_out_layout = api_out.transpose(1, 2).contiguous()       # match HF layout
    assert torch.allclose(hf_out, api_out_layout, atol=1e-5)


def test_hf_gemma2_decoder_layer_has_sandwich_norm():
    """Verify source citation: Gemma2DecoderLayer has all four norm modules.
    Source: modeling_gemma2.py:302-313.
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_layer = modeling_gemma2.Gemma2DecoderLayer(cfg, layer_idx=0)
    assert hasattr(hf_layer, "input_layernorm")
    assert hasattr(hf_layer, "post_attention_layernorm")
    assert hasattr(hf_layer, "pre_feedforward_layernorm")
    assert hasattr(hf_layer, "post_feedforward_layernorm")
