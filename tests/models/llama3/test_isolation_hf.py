"""Isolation tests — each api primitive vs the corresponding HF Llama 3 op,
at atol=1e-5. Run on CPU with deterministic seeds.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers.models.llama import modeling_llama
from transformers.models.llama.configuration_llama import LlamaConfig as HFLlamaConfig

from api import ops, rope, specs, types


def _hf_cfg_default():
    return HFLlamaConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=2,
        rms_norm_eps=1e-5, vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=True, torch_dtype="float32",
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
    )


def _hf_cfg_llama3_scaling():
    return HFLlamaConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=2,
        rms_norm_eps=1e-5, vocab_size=100, max_position_embeddings=131072,
        tie_word_embeddings=True, torch_dtype="float32",
        rope_parameters={
            "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3", "rope_theta": 500000.0,
        },
    )


def test_rms_norm_matches_hf():
    torch.manual_seed(0)
    cfg = _hf_cfg_default()
    hf_norm = modeling_llama.LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="standard_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_default_freqs_match_hf():
    torch.manual_seed(0)
    cfg = _hf_cfg_default()
    hf_rotary = modeling_llama.LlamaRotaryEmbedding(cfg)
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, cfg.head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)
    rope_spec = specs.RoPESpec(
        base_theta=500000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.NONE,
    )
    api_rope = rope.RoPE(rope_spec, head_dim=cfg.head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_rope_llama3_scaling_freqs_match_hf():
    """Verify our LLAMA3 smooth-scaling formula matches HF's _compute_llama3_parameters."""
    torch.manual_seed(0)
    cfg = _hf_cfg_llama3_scaling()
    hf_rotary = modeling_llama.LlamaRotaryEmbedding(cfg)
    # Use a smaller max_seq to keep the test fast
    test_seq = 64
    position_ids = torch.arange(test_seq).unsqueeze(0)
    dummy = torch.zeros(1, 1, test_seq, cfg.head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)

    rope_spec = specs.RoPESpec(
        base_theta=500000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LLAMA3,
        llama3_extra=specs.Llama3RoPEParams(
            factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0,
            original_context_length=8192,
        ),
    )
    api_rope = rope.RoPE(rope_spec, head_dim=cfg.head_dim,
                         max_seq=test_seq, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_swiglu_matches_hf_mlp():
    torch.manual_seed(0)
    cfg = _hf_cfg_default()
    hf_mlp = modeling_llama.LlamaMLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)
    g = F.linear(x, hf_mlp.gate_proj.weight)
    u = F.linear(x, hf_mlp.up_proj.weight)
    h = ops.mul(ops.silu(g), u)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)
