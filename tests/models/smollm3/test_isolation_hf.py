"""Isolation tests — each api primitive vs the corresponding HF SmolLM3 op,
at atol=1e-5. Run on CPU with deterministic seeds.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers.models.smollm3 import modeling_smollm3
from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config as HFSmolLM3Config

from api import ops, rope, specs, types


def _hf_cfg():
    return HFSmolLM3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, num_hidden_layers=8,
        rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=True, torch_dtype="float32",
        rope_parameters={"rope_theta": 5_000_000.0, "rope_type": "default"},
    )


def test_rms_norm_matches_hf():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_norm = modeling_smollm3.SmolLM3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="standard_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_freqs_match_hf():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_smollm3.SmolLM3RotaryEmbedding(cfg)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
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


def test_swiglu_matches_hf_mlp():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_mlp = modeling_smollm3.SmolLM3MLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)
    g = F.linear(x, hf_mlp.gate_proj.weight)
    u = F.linear(x, hf_mlp.up_proj.weight)
    h = ops.mul(ops.silu(g), u)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)
