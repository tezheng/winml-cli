"""GPT-OSS MoE numerical gate (v6 A3).

Tests the api MoE with `expert_kind=gpt_oss_clamped_swiglu` +
`router_kind=topk_then_softmax_with_bias` against HF GptOssMLP.

Source: `transformers/models/gpt_oss/modeling_gpt_oss.py:73-151`.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig  # noqa: E402
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP  # noqa: E402

from api import feedforward, specs, types  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _build_hf_mlp(hidden_size, intermediate_size, num_experts, top_k):
    cfg = GptOssConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        head_dim=hidden_size // 4,
        vocab_size=128,
    )
    torch.manual_seed(0)
    mlp = GptOssMLP(cfg)
    # Initialize parameters (HF leaves them at torch.empty defaults).
    with torch.no_grad():
        torch.nn.init.normal_(mlp.experts.gate_up_proj, std=0.05)
        torch.nn.init.normal_(mlp.experts.gate_up_proj_bias, std=0.05)
        torch.nn.init.normal_(mlp.experts.down_proj, std=0.05)
        torch.nn.init.normal_(mlp.experts.down_proj_bias, std=0.05)
        torch.nn.init.normal_(mlp.router.weight, std=0.05)
        torch.nn.init.normal_(mlp.router.bias, std=0.05)
    mlp.eval()
    return cfg, mlp


def _build_api_moe(hidden_size, intermediate_size, num_experts, top_k):
    expert_ffn = specs.FFNSpec(
        intermediate_size=intermediate_size,
        activation=types.Activation.SILU,    # ignored for gpt_oss_clamped_swiglu
        gate_kind=types.GateKind.SWIGLU,     # ignored
        fused_gate_up=False,
        gate_bias=False, up_bias=False, down_bias=False,
    )
    moe_spec = specs.MoESpec(
        n_experts=num_experts,
        top_k=top_k,
        router_kind="topk_then_softmax_with_bias",
        router_norm=False,
        routed_scaling_factor=1.0,
        expert_ffn=expert_ffn,
        expert_kind="gpt_oss_clamped_swiglu",
        expert_bias=True,
        expert_swiglu_alpha=1.702,
        expert_clamp_limit=7.0,
    )
    return feedforward.MoE(moe_spec, hidden_size=hidden_size, dtype=torch.float32)


def _copy_weights(hf_mlp: GptOssMLP, api_moe: feedforward.MoE) -> None:
    with torch.no_grad():
        api_moe.gate.weight.copy_(hf_mlp.router.weight)
        api_moe.gate.bias.copy_(hf_mlp.router.bias)
        api_moe.experts_gate_up.copy_(hf_mlp.experts.gate_up_proj)
        api_moe.experts_gate_up_bias.copy_(hf_mlp.experts.gate_up_proj_bias)
        api_moe.experts_down.copy_(hf_mlp.experts.down_proj)
        api_moe.experts_down_bias.copy_(hf_mlp.experts.down_proj_bias)


def test_gpt_oss_moe_matches_hf():
    """End-to-end MoE block: router → top-k softmax → clamped-SwiGLU
    experts → weighted sum. Matches HF GptOssMLP at atol=5e-4."""
    hidden_size = 64
    intermediate_size = 96
    num_experts = 8
    top_k = 2

    cfg, hf_mlp = _build_hf_mlp(hidden_size, intermediate_size, num_experts, top_k)
    api_moe = _build_api_moe(hidden_size, intermediate_size, num_experts, top_k)
    _copy_weights(hf_mlp, api_moe)
    api_moe.eval()

    torch.manual_seed(1)
    B, S = 2, 6
    x = torch.randn(B, S, hidden_size)
    with torch.no_grad():
        hf_out, _ = hf_mlp(x)
        api_out = api_moe(x)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"GPT-OSS MoE max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    print(f"GPT-OSS MoE max_abs_diff={max_abs_diff:.2e}")


def test_gpt_oss_moe_topk1():
    """top_k=1 corner case — single expert per token."""
    hidden_size, intermediate_size, num_experts, top_k = 64, 96, 4, 1
    cfg, hf_mlp = _build_hf_mlp(hidden_size, intermediate_size, num_experts, top_k)
    api_moe = _build_api_moe(hidden_size, intermediate_size, num_experts, top_k)
    _copy_weights(hf_mlp, api_moe)
    api_moe.eval()
    torch.manual_seed(2)
    x = torch.randn(1, 4, hidden_size)
    with torch.no_grad():
        hf_out, _ = hf_mlp(x)
        api_out = api_moe(x)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"GPT-OSS MoE top_k=1 max_abs_diff={max_abs_diff:.6e}"
    )


def test_gpt_oss_moe_shapes():
    cfg, hf_mlp = _build_hf_mlp(64, 96, 8, 2)
    api_moe = _build_api_moe(64, 96, 8, 2)
    assert api_moe.experts_gate_up.shape == (8, 64, 192)
    assert api_moe.experts_gate_up_bias.shape == (8, 192)
    assert api_moe.experts_down.shape == (8, 96, 64)
    assert api_moe.experts_down_bias.shape == (8, 64)
    assert api_moe.gate.weight.shape == (8, 64)
    assert api_moe.gate.bias.shape == (8,)
