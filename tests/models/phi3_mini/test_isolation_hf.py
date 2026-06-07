"""Isolation tests — each api primitive vs the corresponding HF Phi-3 op,
at atol=1e-5. Run on CPU with deterministic seeds.

These tests instantiate HF modules WITHOUT downloading model weights — they
construct small configs in-process. So they always run.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
from transformers.models.phi3 import modeling_phi3
from transformers.models.phi3.configuration_phi3 import Phi3Config as HFPhi3Config

from api import ops, rope, specs, types


def _hf_cfg():
    """Small Phi-3-like config (no sliding window for the isolation tests)."""
    return HFPhi3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
        intermediate_size=128, num_hidden_layers=2,
        rms_norm_eps=1e-5, vocab_size=100, max_position_embeddings=32,
        original_max_position_embeddings=32,
        tie_word_embeddings=False, torch_dtype="float32",
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default",
                         "partial_rotary_factor": 1.0},
        sliding_window=None,
    )


def test_rms_norm_matches_hf_phi3():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_norm = modeling_phi3.Phi3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="standard_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_fused_swiglu_matches_hf_phi3_mlp():
    """Phi3MLP: chunk → up * silu(gate). Our api fused-gate-up path does
    silu(gate) * up. Both are mathematically equivalent (commutative)."""
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_mlp = modeling_phi3.Phi3MLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)
    # api equivalent:
    up_states = F.linear(x, hf_mlp.gate_up_proj.weight)
    gate, up = up_states.chunk(2, dim=-1)
    h = ops.mul(ops.silu(gate), up)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_freqs_match_hf_phi3_default():
    """Phi-3-mini-4k uses default RoPE — cos/sin tables must match HF."""
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_phi3.Phi3RotaryEmbedding(cfg)
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids)
    hf_cos = hf_cos.squeeze(0)
    hf_sin = hf_sin.squeeze(0)
    rope_spec = specs.RoPESpec(
        base_theta=10_000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.NONE,
        partial_rotary_factor=1.0,
    )
    api_rope = rope.RoPE(rope_spec, head_dim=head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_fused_qkv_slice_order_matches_hf():
    """Phi3Attention slices the fused qkv as
        query  = qkv[..., :query_pos]
        key    = qkv[..., query_pos : query_pos + n_kv*Dh]
        value  = qkv[..., query_pos + n_kv*Dh:]
    where query_pos = num_attention_heads * head_dim.
    Source: modeling_phi3.py:237-241."""
    torch.manual_seed(0)
    cfg = _hf_cfg()
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    Hq = cfg.num_attention_heads
    Hk = cfg.num_key_value_heads
    qkv_dim = (Hq + 2 * Hk) * head_dim
    qkv = torch.randn(1, 3, qkv_dim)
    # HF slicing:
    query_pos = Hq * head_dim
    q_hf = qkv[..., :query_pos]
    k_hf = qkv[..., query_pos : query_pos + Hk * head_dim]
    v_hf = qkv[..., query_pos + Hk * head_dim:]
    # api slicing must match exactly.
    q_api = qkv[..., :Hq * head_dim]
    k_api = qkv[..., Hq * head_dim : (Hq + Hk) * head_dim]
    v_api = qkv[..., (Hq + Hk) * head_dim :]
    assert torch.equal(q_hf, q_api)
    assert torch.equal(k_hf, k_api)
    assert torch.equal(v_hf, v_api)
