"""Isolation tests against HF DeepseekV3MoE.

The V3 router is the most novel part of the V3 family — sigmoid+bias for
ROUTING CHOICE only, weights gathered from the bias-FREE sigmoid output,
optional norm_topk_prob, then routed_scaling. Verify our MoE matches
HF's DeepseekV3MoE at atol=5e-4 on synthetic weights.
"""
import pytest
import torch

from api import feedforward, specs, types


ATOL = 5e-4


def _make_hf_v3_moe_config():
    from transformers.models.deepseek_v3 import configuration_deepseek_v3 as cfg_mod
    cfg = cfg_mod.DeepseekV3Config(
        vocab_size=100, hidden_size=64, intermediate_size=128,
        moe_intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=4,
        routed_scaling_factor=2.5, norm_topk_prob=True,
        n_group=4, topk_group=2,
        first_k_dense_replace=0,
        kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8,
        v_head_dim=16, q_lora_rank=24,
        max_position_embeddings=64, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        rope_interleave=False, tie_word_embeddings=False,
        attention_bias=False, attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_v3_moe_router_matches_hf_at_5e_4():
    """B5: our MoE sigmoid+bias router matches HF DeepseekV3MoE at atol=5e-4."""
    from transformers.models.deepseek_v3 import modeling_deepseek_v3 as mod
    torch.manual_seed(13)
    hf_cfg = _make_hf_v3_moe_config()
    hf_moe = mod.DeepseekV3MoE(hf_cfg)
    hf_moe.eval()

    # Build our MoE with matching spec.
    expert_ffn = specs.FFNSpec(
        intermediate_size=hf_cfg.moe_intermediate_size,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )
    moe_spec = specs.MoESpec(
        n_experts=hf_cfg.n_routed_experts,
        top_k=hf_cfg.num_experts_per_tok,
        n_shared_experts=hf_cfg.n_shared_experts,
        router_kind="sigmoid_plus_bias",
        router_norm=hf_cfg.norm_topk_prob,
        group_routing=specs.GroupRoutingSpec(
            n_groups=hf_cfg.n_group, topk_per_group=hf_cfg.topk_group,
        ),
        routed_scaling_factor=hf_cfg.routed_scaling_factor,
        expert_ffn=expert_ffn,
    )
    api_moe = feedforward.MoE(moe_spec, hidden_size=hf_cfg.hidden_size,
                              dtype=torch.float32)

    # Copy HF weights into api.
    with torch.no_grad():
        # Router gate.
        api_moe.gate.weight.copy_(hf_moe.gate.weight.data)
        api_moe.gate.e_score_correction_bias.copy_(
            hf_moe.gate.e_score_correction_bias.data.float()
        )
        # Experts.
        api_moe.experts_gate_up.copy_(hf_moe.experts.gate_up_proj.data)
        api_moe.experts_down.copy_(hf_moe.experts.down_proj.data)
        # Shared experts.
        api_moe.shared_experts.gate_proj.weight.copy_(
            hf_moe.shared_experts.gate_proj.weight.data)
        api_moe.shared_experts.up_proj.weight.copy_(
            hf_moe.shared_experts.up_proj.weight.data)
        api_moe.shared_experts.down_proj.weight.copy_(
            hf_moe.shared_experts.down_proj.weight.data)

    # Put a non-trivial value in e_score_correction_bias so it actually
    # changes which experts are chosen.
    with torch.no_grad():
        bias = torch.randn(hf_cfg.n_routed_experts) * 0.5
        hf_moe.gate.e_score_correction_bias.copy_(bias)
        api_moe.gate.e_score_correction_bias.copy_(bias.float())

    x = torch.randn(1, 5, hf_cfg.hidden_size)
    with torch.no_grad():
        hf_out = hf_moe(x)
        api_out = api_moe(x)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"V3 MoE diff = {max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=ATOL), (
        f"max_abs_diff = {max_abs_diff:.6f}"
    )
