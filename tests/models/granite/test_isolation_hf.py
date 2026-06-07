"""Isolation tests — each api primitive vs the corresponding HF Granite op,
at atol=1e-5. Run on CPU with deterministic seeds.

These tests instantiate HF modules WITHOUT downloading model weights — they
construct small configs in-process. So they always run.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers.models.granite import modeling_granite
from transformers.models.granite.configuration_granite import GraniteConfig as HFGraniteConfig

from api import ops, rope, specs, types


def _hf_cfg():
    return HFGraniteConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, num_hidden_layers=2,
        rms_norm_eps=1e-5, vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=False, torch_dtype="float32",
        rope_parameters={"rope_theta": 5_000_000.0, "rope_type": "default"},
        attention_multiplier=0.5,
        residual_multiplier=0.7,
        embedding_multiplier=3.0,
        logits_scaling=2.0,
    )


def test_rms_norm_matches_hf_granite():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_norm = modeling_granite.GraniteRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="standard_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_freqs_match_hf_granite():
    """HF Granite cos/sin = inv_freq tables tied to rope_theta. Verify our
    RoPESpec(NONE scaling, theta) reproduces them exactly."""
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_granite.GraniteRotaryEmbedding(cfg)
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)
    rope_spec = specs.RoPESpec(
        base_theta=5_000_000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.NONE,
    )
    api_rope = rope.RoPE(rope_spec, head_dim=head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_swiglu_matches_hf_granite_mlp():
    """GraniteMLP is bog-standard SwiGLU."""
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_mlp = modeling_granite.GraniteMLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)
    g = F.linear(x, hf_mlp.gate_proj.weight)
    u = F.linear(x, hf_mlp.up_proj.weight)
    h = ops.mul(ops.silu(g), u)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_residual_multiplier_present_in_hf_layer():
    """Verify our source citation: Granite's decoder layer multiplies
    BOTH sublayer outputs by `residual_multiplier` before the residual add
    (modeling_granite.py:273,278). We can't easily run the HF layer in
    isolation without RoPE-bind + attention_mask plumbing, but we can prove
    the attribute is set from config — that's the contract our config wire
    must hit."""
    cfg = _hf_cfg()
    hf_layer = modeling_granite.GraniteDecoderLayer(cfg, layer_idx=0)
    assert hf_layer.residual_multiplier == 0.7
