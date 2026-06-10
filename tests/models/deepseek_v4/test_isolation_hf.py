"""Numerical isolation tests against HF DeepseekV4SparseMoeBlock.

The V4 hash MoE is the architectural novelty for B1 — paper §2.1. Verify our
`MoE(router_kind="hash")` matches HF's `DeepseekV4SparseMoeBlock(is_hash=True)`
at atol=5e-4 on synthetic weights.

ATOL = 5e-4 (v7 numerical gate).
"""
import pytest
import torch

pytest.importorskip("transformers")

from api import feedforward, specs, types


ATOL = 5e-4
RTOL = 5e-4


def _make_hf_v4_cfg():
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    cfg = DeepseekV4Config(
        hidden_size=64,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16,
        q_lora_rank=32,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        # Numerical gate requires "sigmoid" (api _HashRouter only supports sigmoid).
        scoring_func="sigmoid",
        # Set very large so V4's expert + shared_experts clamp(max=limit) /
        # clamp(min=-limit, max=limit) become no-ops over random ~N(0,1) tensors.
        swiglu_limit=1e6,
        vocab_size=50, max_position_embeddings=64,
        mlp_layer_types=["hash_moe", "moe"],
        layer_types=["heavily_compressed_attention", "compressed_sparse_attention"],
    )
    cfg._attn_implementation = "eager"
    return cfg


def _api_hash_moe_from_hf_cfg(hf_cfg):
    expert_ffn = specs.FFNSpec(
        intermediate_size=hf_cfg.moe_intermediate_size,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )
    moe_spec = specs.MoESpec(
        n_experts=hf_cfg.n_routed_experts,
        top_k=hf_cfg.num_experts_per_tok,
        n_shared_experts=hf_cfg.n_shared_experts,
        router_kind="hash",
        router_norm=True,
        hash_vocab_size=hf_cfg.vocab_size,
        hash_score_fn="sigmoid",
        routed_scaling_factor=hf_cfg.routed_scaling_factor,
        expert_ffn=expert_ffn,
    )
    return feedforward.MoE(moe_spec, hidden_size=hf_cfg.hidden_size,
                           dtype=torch.float32)


def test_v4_hash_moe_matches_hf_at_5e_4():
    """B1: our MoE(router_kind='hash') matches HF DeepseekV4SparseMoeBlock
    (is_hash=True) at atol=5e-4 on a synthetic-mini config.

    Source: modeling_deepseek_v4.py:1050-1098 (HashRouter + SparseMoeBlock).
    """
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mod
    torch.manual_seed(42)
    hf_cfg = _make_hf_v4_cfg()

    # Build HF block for layer 0 (hash_moe).
    hf_block = mod.DeepseekV4SparseMoeBlock(hf_cfg, layer_idx=0)
    hf_block.eval()
    assert hf_block.is_hash, "expected hash_moe layer 0"

    # Build api MoE with matching spec.
    api_moe = _api_hash_moe_from_hf_cfg(hf_cfg)
    api_moe.eval()

    # Copy HF weights into api.
    with torch.no_grad():
        # Router gate.
        api_moe.gate.weight.copy_(hf_block.gate.weight.data)
        # tid2eid: HF initialises to zeros — fill with a non-trivial table so
        # routing isn't degenerate (every token routes to expert 0).
        tid2eid = torch.randint(
            0, hf_cfg.n_routed_experts,
            (hf_cfg.vocab_size, hf_cfg.num_experts_per_tok),
        )
        hf_block.gate.tid2eid.copy_(tid2eid)
        api_moe.gate.tid2eid.copy_(tid2eid)
        # Experts: HF stores gate_up_proj [E, 2I, H] and down_proj [E, H, I]
        # — same layout as api.
        api_moe.experts_gate_up.copy_(hf_block.experts.gate_up_proj.data)
        api_moe.experts_down.copy_(hf_block.experts.down_proj.data)
        # Shared experts: HF DeepseekV4MLP has gate_proj/up_proj/down_proj.
        # `intermediate_size` per attribute_map is moe_intermediate_size; we
        # built the api shared FFN with the same total (I * n_shared_experts).
        # When n_shared_experts == 1 these are identical.
        api_moe.shared_experts.gate_proj.weight.copy_(
            hf_block.shared_experts.gate_proj.weight.data)
        api_moe.shared_experts.up_proj.weight.copy_(
            hf_block.shared_experts.up_proj.weight.data)
        api_moe.shared_experts.down_proj.weight.copy_(
            hf_block.shared_experts.down_proj.weight.data)

    B, S = 1, 5
    x = torch.randn(B, S, hf_cfg.hidden_size)
    input_ids = torch.randint(0, hf_cfg.vocab_size, (B, S))

    with torch.no_grad():
        hf_out = hf_block(x, input_ids=input_ids)
        api_out = api_moe(x, input_ids=input_ids)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"V4 hash MoE max_abs_diff={max_abs_diff:.6e} (atol={ATOL})"
    )


def test_v4_hash_router_deterministic_by_input_ids():
    """Hash routing must depend ONLY on input_ids (not on hidden states).

    Verify that running with two different x but the SAME input_ids selects
    the same experts in our `_route_hash`.
    """
    torch.manual_seed(0)
    hf_cfg = _make_hf_v4_cfg()
    api_moe = _api_hash_moe_from_hf_cfg(hf_cfg)
    api_moe.eval()

    # Fill tid2eid with non-trivial routing.
    tid2eid = torch.randint(
        0, hf_cfg.n_routed_experts,
        (hf_cfg.vocab_size, hf_cfg.num_experts_per_tok),
    )
    with torch.no_grad():
        api_moe.gate.tid2eid.copy_(tid2eid)

    B, S = 1, 7
    input_ids = torch.randint(0, hf_cfg.vocab_size, (B, S))

    x1 = torch.randn(B, S, hf_cfg.hidden_size)
    x2 = torch.randn(B, S, hf_cfg.hidden_size)
    with torch.no_grad():
        idx1, _ = api_moe._route_hash(x1.reshape(-1, hf_cfg.hidden_size), input_ids)
        idx2, _ = api_moe._route_hash(x2.reshape(-1, hf_cfg.hidden_size), input_ids)
    assert torch.equal(idx1, idx2)
