"""Isolation tests for TinyLlama. TinyLlama uses model_type='llama' so we
delegate to the HF Llama primitives (LlamaRMSNorm, LlamaRotaryEmbedding,
LlamaMLP) — same as the llama3 isolation suite, with TinyLlama-specific dims.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers.models.llama import modeling_llama
from transformers.models.llama.configuration_llama import LlamaConfig as HFLlamaConfig

from api import ops, rope, specs, types


def _hf_cfg():
    """Mini TinyLlama-like config (default RoPE theta=10000, GQA n_q=8/n_kv=2)."""
    return HFLlamaConfig(
        hidden_size=64, num_attention_heads=8, num_key_value_heads=2,
        head_dim=8, intermediate_size=128, num_hidden_layers=2,
        rms_norm_eps=1e-5, vocab_size=100, max_position_embeddings=64,
        tie_word_embeddings=False, torch_dtype="float32",
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


def test_rms_norm_matches_hf():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_norm = modeling_llama.LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="standard_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_freqs_match_hf_theta_10000():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_llama.LlamaRotaryEmbedding(cfg)
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, cfg.head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)
    rope_spec = specs.RoPESpec(
        base_theta=10000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.NONE,
    )
    api_rope = rope.RoPE(rope_spec, head_dim=cfg.head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_swiglu_matches_hf_mlp():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_mlp = modeling_llama.LlamaMLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)
    g = F.linear(x, hf_mlp.gate_proj.weight)
    u = F.linear(x, hf_mlp.up_proj.weight)
    h = ops.mul(ops.silu(g), u)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)
