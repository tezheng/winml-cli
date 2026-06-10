"""Qwen3-Next numerical gate vs HF — Gated DeltaNet (linear-attention) layer.

Instantiates a small Qwen3NextDecoderLayer at a linear-attention index, copies
the weights into our api block (linear-attn mixer + RMSNorm + dense MLP), and
verifies numerical equivalence at atol=5e-4 on a synthetic-mini config.

We use `num_experts=0` so HF instantiates a plain `Qwen3NextMLP` (dense SwiGLU),
matching our `FFNSpec` envelope exactly. The MoE path is NOT gated — it carries
a `shared_expert_gate` sigmoid which the api MoE doesn't model.

The full-attention layer (index 3 in the default 3:1 schedule) is NOT gated —
HF's `Qwen3NextAttention` has a fused (q | gate) projection and output gating
which the IR's STANDARD attention doesn't model (see layer.md).

ATOL = 5e-4 (v7 numerical gate).
"""
import pytest
import torch

pytest.importorskip("transformers")

from api import specs, types
from models.qwen3_next import config as _c, layer as _l


ATOL = 5e-4
RTOL = 5e-4


def _hf_cfg():
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    return Qwen3NextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=100,
        max_position_embeddings=128,
        # num_experts=0 -> Qwen3NextDecoderLayer falls through to dense Qwen3NextMLP.
        # Source: modeling_qwen3_next.py:831-836.
        num_experts=0,
        num_experts_per_tok=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
    )


@pytest.fixture(scope="module")
def hf_small_model():
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextModel
    torch.manual_seed(42)
    model = Qwen3NextModel(_hf_cfg()).float().eval()
    return model


def _api_cfg_from_hf(hf_cfg):
    """Build an api Qwen3NextConfig that mirrors the HF config exactly."""
    return _c.Qwen3NextConfig(
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        moe_intermediate_size=hf_cfg.moe_intermediate_size,
        shared_expert_intermediate_size=hf_cfg.shared_expert_intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.head_dim,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        vocab_size=hf_cfg.vocab_size,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        tie_word_embeddings=False,
        attention_bias=False,
        rope_theta=10000.0,
        partial_rotary_factor=hf_cfg.partial_rotary_factor,
        dtype=torch.float32,
        layer_types=tuple(hf_cfg.layer_types),
        # num_experts=0 -> all layers are mlp_only via the is_mlp_only_layer check.
        mlp_only_layers=tuple(range(hf_cfg.num_hidden_layers)),
        linear_conv_kernel_dim=hf_cfg.linear_conv_kernel_dim,
        linear_key_head_dim=hf_cfg.linear_key_head_dim,
        linear_value_head_dim=hf_cfg.linear_value_head_dim,
        linear_num_key_heads=hf_cfg.linear_num_key_heads,
        linear_num_value_heads=hf_cfg.linear_num_value_heads,
        num_experts=hf_cfg.num_experts,
        num_experts_per_tok=hf_cfg.num_experts_per_tok,
        decoder_sparse_step=hf_cfg.decoder_sparse_step,
        norm_topk_prob=hf_cfg.norm_topk_prob,
    )


def _load_dense_mlp(blk, hf_layer, L):
    """Copy a Qwen3NextMLP (dense SwiGLU) into our api FeedForward."""
    sd = hf_layer.state_dict()
    with torch.no_grad():
        blk.feedforward.gate_proj.weight.copy_(sd["mlp.gate_proj.weight"])
        blk.feedforward.up_proj.weight.copy_(sd["mlp.up_proj.weight"])
        blk.feedforward.down_proj.weight.copy_(sd["mlp.down_proj.weight"])
        blk.pre_attn_norm.weight.copy_(sd["input_layernorm.weight"])
        blk.pre_ffn_norm.weight.copy_(sd["post_attention_layernorm.weight"])


def _load_linear_attn(blk, hf_layer):
    sd = hf_layer.linear_attn.state_dict()
    mixer = blk.attention
    with torch.no_grad():
        mixer.in_proj_qkvz.weight.copy_(sd["in_proj_qkvz.weight"])
        mixer.in_proj_ba.weight.copy_(sd["in_proj_ba.weight"])
        mixer.conv1d.weight.copy_(sd["conv1d.weight"])
        mixer.dt_bias.copy_(sd["dt_bias"])
        mixer.A_log.copy_(sd["A_log"])
        mixer.norm.weight.copy_(sd["norm.weight"])
        mixer.out_proj.weight.copy_(sd["out_proj.weight"])


def test_linear_attention_layer_0_matches_hf(hf_small_model):
    """B2: linear_attention layer 0 — full decoder block (RMSNorm + Gated
    DeltaNet + RMSNorm + Dense MLP) matches HF at atol=5e-4."""
    hf_cfg = hf_small_model.config
    api_cfg = _api_cfg_from_hf(hf_cfg)
    assert api_cfg.is_linear_attention_layer(0)

    api_blk = _l.build_qwen3_next_decoder_layer(api_cfg, layer_idx=0)
    api_blk.eval()
    hf_layer = hf_small_model.layers[0]
    _load_linear_attn(api_blk, hf_layer)
    _load_dense_mlp(api_blk, hf_layer, 0)

    torch.manual_seed(0)
    B, S = 1, 6
    x = torch.randn(B, S, hf_cfg.hidden_size)

    # HF's Qwen3NextDecoderLayer.forward requires position_embeddings as a
    # positional arg; it's unused for linear_attention layers (linear_attn
    # consumes cache_params only). Pass dummies.
    dummy_cos = torch.zeros(B, S, hf_cfg.head_dim)
    dummy_sin = torch.zeros(B, S, hf_cfg.head_dim)
    with torch.no_grad():
        hf_out = hf_layer(x, position_embeddings=(dummy_cos, dummy_sin))
        api_out = api_blk(x)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Qwen3-Next linear-attn layer-0 max_abs_diff={max_abs_diff:.6e}"
    )


def test_linear_attention_layer_2_matches_hf(hf_small_model):
    """B2: linear_attention layer 2 — exercise a different layer index."""
    hf_cfg = hf_small_model.config
    api_cfg = _api_cfg_from_hf(hf_cfg)
    assert api_cfg.is_linear_attention_layer(2)

    api_blk = _l.build_qwen3_next_decoder_layer(api_cfg, layer_idx=2)
    api_blk.eval()
    hf_layer = hf_small_model.layers[2]
    _load_linear_attn(api_blk, hf_layer)
    _load_dense_mlp(api_blk, hf_layer, 2)

    torch.manual_seed(1)
    B, S = 1, 5
    x = torch.randn(B, S, hf_cfg.hidden_size)

    # HF's Qwen3NextDecoderLayer.forward requires position_embeddings as a
    # positional arg; it's unused for linear_attention layers (linear_attn
    # consumes cache_params only). Pass dummies.
    dummy_cos = torch.zeros(B, S, hf_cfg.head_dim)
    dummy_sin = torch.zeros(B, S, hf_cfg.head_dim)
    with torch.no_grad():
        hf_out = hf_layer(x, position_embeddings=(dummy_cos, dummy_sin))
        api_out = api_blk(x)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Qwen3-Next linear-attn layer-2 max_abs_diff={max_abs_diff:.6e}"
    )


def test_load_hf_qwen3_next_linear_attn_layer_via_helper(hf_small_model):
    """Verify the public weight-loader helper round-trips correctly for the
    linear-attention path."""
    hf_cfg = hf_small_model.config
    api_cfg = _api_cfg_from_hf(hf_cfg)
    api_blk = _l.build_qwen3_next_decoder_layer(api_cfg, layer_idx=0)
    full_sd = {}
    # Re-key HF model.layers[0] into "model.layers.0.*" form.
    for k, v in hf_small_model.state_dict().items():
        full_sd[f"model.{k}"] = v

    _l.load_hf_qwen3_next_linear_attn_layer(api_blk, full_sd, api_cfg, layer_idx=0)
    api_blk.eval()
    hf_layer = hf_small_model.layers[0]
    torch.manual_seed(7)
    B, S = 1, 4
    x = torch.randn(B, S, hf_cfg.hidden_size)
    dummy_cos = torch.zeros(B, S, hf_cfg.head_dim)
    dummy_sin = torch.zeros(B, S, hf_cfg.head_dim)
    with torch.no_grad():
        api_out = api_blk(x)
        hf_out = hf_layer(x, position_embeddings=(dummy_cos, dummy_sin))
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert max_abs_diff < ATOL, f"helper-loaded max_abs_diff={max_abs_diff:.6e}"
