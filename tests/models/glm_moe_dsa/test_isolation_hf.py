"""GLM-MoE-DSA numerical isolation against HF GlmMoeDsaMoE.

Verifies the sigmoid+bias router + single-group routing matches HF at
atol=5e-4 on synthetic weights.

ATOL = 5e-4 (v7 numerical gate).
"""
import pytest
import torch

pytest.importorskip("transformers")

from api import feedforward, specs, types


ATOL = 5e-4
RTOL = 5e-4


def _make_hf_cfg():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    cfg = GlmMoeDsaConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=100,
        kv_lora_rank=16, q_lora_rank=32,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        n_group=1, topk_group=1,
        index_topk=16, index_head_dim=8, index_n_heads=2,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _api_moe(hf_cfg):
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
    return feedforward.MoE(moe_spec, hidden_size=hf_cfg.hidden_size,
                           dtype=torch.float32)


def test_glm_moe_dsa_matches_hf_at_5e_4():
    """B3: GlmMoeDsaMoE numerically matches our MoE at atol=5e-4.

    Source: modeling_glm_moe_dsa.py:538-591 (sigmoid+bias + group routing,
    identical to DeepSeek-V3's MoE math).
    """
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as mod
    torch.manual_seed(13)
    hf_cfg = _make_hf_cfg()
    hf_moe = mod.GlmMoeDsaMoE(hf_cfg)
    hf_moe.eval()

    api_moe = _api_moe(hf_cfg)
    api_moe.eval()

    with torch.no_grad():
        # Router gate.
        api_moe.gate.weight.copy_(hf_moe.gate.weight.data)
        # GlmMoeDsa stores `e_score_correction_bias` on the router (not on
        # the MoE block — different from MiniMax M2 which stores it on the
        # SparseMoeBlock).
        api_moe.gate.e_score_correction_bias.copy_(
            hf_moe.gate.e_score_correction_bias.data.float()
        )
        # Experts — same layout as V2/V3.
        api_moe.experts_gate_up.copy_(hf_moe.experts.gate_up_proj.data)
        api_moe.experts_down.copy_(hf_moe.experts.down_proj.data)
        # Shared experts.
        api_moe.shared_experts.gate_proj.weight.copy_(
            hf_moe.shared_experts.gate_proj.weight.data)
        api_moe.shared_experts.up_proj.weight.copy_(
            hf_moe.shared_experts.up_proj.weight.data)
        api_moe.shared_experts.down_proj.weight.copy_(
            hf_moe.shared_experts.down_proj.weight.data)

    # Inject non-trivial score correction bias so routing isn't degenerate.
    with torch.no_grad():
        bias = torch.randn(hf_cfg.n_routed_experts) * 0.5
        hf_moe.gate.e_score_correction_bias.copy_(bias)
        api_moe.gate.e_score_correction_bias.copy_(bias.float())

    x = torch.randn(1, 5, hf_cfg.hidden_size)
    with torch.no_grad():
        hf_out = hf_moe(x)
        api_out = api_moe(x)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"GLM-MoE-DSA MoE max_abs_diff={max_abs_diff:.6e}"
    )


def test_glm_moe_dsa_single_group_topk_path():
    """Verify the single-group (n_group=1) path picks every expert eligible."""
    torch.manual_seed(0)
    hf_cfg = _make_hf_cfg()
    api_moe = _api_moe(hf_cfg)
    api_moe.eval()
    x = torch.randn(2, 4, hf_cfg.hidden_size)
    with torch.no_grad():
        # First the route-only path to inspect topk indices.
        x_flat = x.reshape(-1, hf_cfg.hidden_size)
        idx, w = api_moe._route_sigmoid_plus_bias(x_flat)
    # With n_group=1 and topk_group=1, score_mask covers ALL experts so
    # topk operates on the FULL expert axis (no -inf masking effect).
    assert idx.shape == (2 * 4, hf_cfg.num_experts_per_tok)
    assert w.shape == (2 * 4, hf_cfg.num_experts_per_tok)
    # Weights should be normalised AND scaled.
    sum_w = w.sum(dim=-1)
    expected = torch.tensor([hf_cfg.routed_scaling_factor]).expand_as(sum_w)
    assert torch.allclose(sum_w, expected, atol=1e-5)
