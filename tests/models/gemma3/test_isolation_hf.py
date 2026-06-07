"""Isolation tests — each api primitive vs the corresponding HF Gemma 3 op."""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")
try:
    from transformers.models.gemma3 import modeling_gemma3
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig as HFG3TextConfig
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers.models.gemma3 not available — install transformers>=5.10",
        allow_module_level=True,
    )

from api import norm as _norm, ops, rope, specs, types


def _hf_cfg():
    return HFG3TextConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_hidden_layers=12,
        rope_parameters={
            "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
            "full_attention":    {"rope_theta": 1_000_000.0, "rope_type": "default"},
        },
        rms_norm_eps=1e-6,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=True, torch_dtype="float32",
        sliding_window=8, query_pre_attn_scalar=16,
        attn_logit_softcapping=None, final_logit_softcapping=None,
        attention_bias=False, hidden_activation="gelu_pytorch_tanh",
        sliding_window_pattern=6,
    )


def test_rms_norm_matches_hf_gemma3_one_plus_w():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_norm = modeling_gemma3.Gemma3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    weight = torch.randn(cfg.hidden_size)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, cfg.rms_norm_eps, mode="one_plus_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_rope_local_matches_hf_gemma3_sliding_theta():
    """Sliding layer's RoPE uses rope_local_base_freq = 10000."""
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_gemma3.Gemma3RotaryEmbedding(cfg)
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, cfg.head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids, layer_type="sliding_attention")
    hf_cos = hf_cos.squeeze(0); hf_sin = hf_sin.squeeze(0)
    rope_spec = specs.RoPESpec(base_theta=10_000.0,
                               basis=types.RoPEBasis.SPLIT_HALF)
    api_rope = rope.RoPE(rope_spec, head_dim=cfg.head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_rope_global_matches_hf_gemma3_full_theta():
    """Full layer's RoPE uses rope_theta = 1_000_000."""
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_rotary = modeling_gemma3.Gemma3RotaryEmbedding(cfg)
    position_ids = torch.arange(cfg.max_position_embeddings).unsqueeze(0)
    dummy = torch.zeros(1, 1, cfg.max_position_embeddings, cfg.head_dim)
    hf_cos, hf_sin = hf_rotary(dummy, position_ids, layer_type="full_attention")
    hf_cos = hf_cos.squeeze(0); hf_sin = hf_sin.squeeze(0)
    rope_spec = specs.RoPESpec(base_theta=1_000_000.0,
                               basis=types.RoPEBasis.SPLIT_HALF)
    api_rope = rope.RoPE(rope_spec, head_dim=cfg.head_dim,
                         max_seq=cfg.max_position_embeddings, dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_geglu_matches_hf_gemma3_mlp():
    torch.manual_seed(0)
    cfg = _hf_cfg()
    hf_mlp = modeling_gemma3.Gemma3MLP(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    hf_out = hf_mlp(x)
    g = F.linear(x, hf_mlp.gate_proj.weight)
    u = F.linear(x, hf_mlp.up_proj.weight)
    h = ops.mul(ops.gelu_pytorch_tanh(g), u)
    api_out = F.linear(h, hf_mlp.down_proj.weight)
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_qk_norm_per_head_dh_matches_hf_gemma3():
    """B3 (Gemma 3): QK-norm is `Gemma3RMSNorm(head_dim, eps)` — ONE_PLUS_W.
    Applied AFTER the view+transpose reshape, BEFORE apply_rotary_pos_emb.
    Source: modeling_gemma3.py:337-356.
    """
    torch.manual_seed(0)
    cfg = _hf_cfg()
    Dh = cfg.head_dim
    hf_qn = modeling_gemma3.Gemma3RMSNorm(Dh, eps=cfg.rms_norm_eps)
    weight = torch.randn(Dh)
    hf_qn.weight.data.copy_(weight)
    x = torch.randn(1, cfg.num_attention_heads, 5, Dh)        # [B, H, S, Dh] layout
    hf_out = hf_qn(x)

    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=cfg.rms_norm_eps,
                               weight_mode=types.NormWeightMode.ONE_PLUS_W)
    qkn = _norm.QKNorm(norm_spec, head_dim=Dh,
                       shape=types.QKNormShape.PER_HEAD_DH,
                       n_heads=None, dtype=torch.float32)
    qkn.weight.data.copy_(weight)
    api_out = qkn(x)
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_hf_gemma3_attention_has_qk_norm():
    """Verify Gemma 3 (unlike Gemma 2) has q_norm / k_norm modules.
    Source: modeling_gemma3.py:337-338.
    """
    cfg = _hf_cfg()
    hf_attn = modeling_gemma3.Gemma3Attention(cfg, layer_idx=0)
    assert hasattr(hf_attn, "q_norm")
    assert hasattr(hf_attn, "k_norm")
    # Both at head_dim shape — PER_HEAD_DH.
    assert hf_attn.q_norm.weight.shape == (cfg.head_dim,)
    assert hf_attn.k_norm.weight.shape == (cfg.head_dim,)


def test_hf_gemma3_first_full_layer_is_idx_5_for_pattern_6():
    """Verify the layer-type pattern at the HF config level."""
    cfg = _hf_cfg()
    # config sets layer_types via __post_init__.
    assert cfg.layer_types[0] == "sliding_attention"
    assert cfg.layer_types[5] == "full_attention"
    assert cfg.layer_types[11] == "full_attention"
