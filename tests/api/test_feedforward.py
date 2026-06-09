import torch
import torch.nn.functional as F
from torch import nn

from api import feedforward, specs, types


def _swiglu_spec():
    return specs.FFNSpec(
        intermediate_size=128,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )


def test_swiglu_forward_shape():
    spec = _swiglu_spec()
    ffn = feedforward.FeedForward(spec, hidden_size=32, dtype=torch.float32)
    x = torch.randn(2, 4, 32)
    out = ffn(x)
    assert out.shape == (2, 4, 32)


def test_swiglu_matches_manual():
    """y = down(silu(gate(x)) * up(x))"""
    spec = _swiglu_spec()
    ffn = feedforward.FeedForward(spec, hidden_size=8, dtype=torch.float32)
    x = torch.randn(1, 2, 8)
    g = F.linear(x, ffn.gate_proj.weight)
    u = F.linear(x, ffn.up_proj.weight)
    expected = F.linear(F.silu(g) * u, ffn.down_proj.weight)
    out = ffn(x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_geglu_forward_shape():
    spec = specs.FFNSpec(
        intermediate_size=128,
        activation=types.Activation.GELU,
        gate_kind=types.GateKind.GEGLU,
    )
    ffn = feedforward.FeedForward(spec, hidden_size=32, dtype=torch.float32)
    x = torch.randn(2, 4, 32)
    out = ffn(x)
    assert out.shape == (2, 4, 32)


def test_geglu_uses_gelu_pytorch_tanh_for_gemma():
    """Gemma's GeGLU uses gelu_pytorch_tanh, not plain gelu."""
    spec = specs.FFNSpec(
        intermediate_size=8,
        activation=types.Activation.GELU,
        gate_kind=types.GateKind.GEGLU,
    )
    ffn = feedforward.FeedForward(spec, hidden_size=4, dtype=torch.float32)
    x = torch.randn(1, 2, 4)
    g = F.linear(x, ffn.gate_proj.weight)
    u = F.linear(x, ffn.up_proj.weight)
    expected = F.linear(F.gelu(g, approximate="tanh") * u, ffn.down_proj.weight)
    assert torch.allclose(ffn(x), expected, atol=1e-5)


def test_fused_gate_up_equivalent_to_split():
    """B2a: Phi-3 fused gate/up — one big projection of size 2*I, chunked into
    (gate, up). When the fused matrix is built by concatenating split
    gate/up along the output dim, the two paths produce identical output.
    Source: modeling_phi3.py:54-64."""
    hidden, I = 32, 64
    split = specs.FFNSpec(intermediate_size=I, activation=types.Activation.SILU,
                          gate_kind=types.GateKind.SWIGLU, fused_gate_up=False)
    fused = specs.FFNSpec(intermediate_size=I, activation=types.Activation.SILU,
                          gate_kind=types.GateKind.SWIGLU, fused_gate_up=True)
    f_s = feedforward.FeedForward(split, hidden_size=hidden, dtype=torch.float32)
    f_f = feedforward.FeedForward(fused, hidden_size=hidden, dtype=torch.float32)
    with torch.no_grad():
        # Phi-3 chunks along last dim — `gate, up = chunk(2, dim=-1)` — which
        # means rows 0..I are gate and rows I..2I are up in the fused weight.
        f_f.gate_up_proj.weight.copy_(torch.cat(
            [f_s.gate_proj.weight, f_s.up_proj.weight], dim=0
        ))
        f_f.down_proj.weight.copy_(f_s.down_proj.weight)
    x = torch.randn(1, 4, hidden)
    out_s = f_s(x)
    out_f = f_f(x)
    assert torch.allclose(out_s, out_f, atol=1e-5)


def test_fused_gate_up_shape():
    spec = specs.FFNSpec(intermediate_size=128,
                         activation=types.Activation.SILU,
                         gate_kind=types.GateKind.SWIGLU,
                         fused_gate_up=True)
    ffn = feedforward.FeedForward(spec, hidden_size=64, dtype=torch.float32)
    assert ffn.gate_up_proj.weight.shape == (2 * 128, 64)
    assert ffn.gate_proj is None
    assert ffn.up_proj is None


# ---------------------------------------------------------------------------
# B5: MoE tests
# ---------------------------------------------------------------------------


def _moe_expert_ffn(I: int = 16) -> specs.FFNSpec:
    return specs.FFNSpec(
        intermediate_size=I,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )


def test_b5_moe_softmax_router_shapes():
    spec = specs.MoESpec(
        n_experts=8, top_k=2, n_shared_experts=0,
        router_kind="softmax", expert_ffn=_moe_expert_ffn(I=16),
    )
    moe = feedforward.MoE(spec, hidden_size=12, dtype=torch.float32)
    x = torch.randn(2, 4, 12)
    out = moe(x)
    assert out.shape == (2, 4, 12)
    assert moe.experts_gate_up.shape == (8, 32, 12)
    assert moe.experts_down.shape == (8, 12, 16)


def test_b5_moe_softmax_with_shared_experts_adds_shared_output():
    """Shared experts run on the SAME residual stream as routed (line 122-130 V2),
    and the final output is routed + shared. We verify by running with the
    routed path zeroed (top-k weights = 0 by deliberately constructing
    inputs that produce uniform routing) — actually simpler: zero the
    experts_gate_up so routed = 0, then check out == shared(x).
    """
    spec = specs.MoESpec(
        n_experts=4, top_k=2, n_shared_experts=2,
        router_kind="softmax", expert_ffn=_moe_expert_ffn(I=8),
    )
    moe = feedforward.MoE(spec, hidden_size=6, dtype=torch.float32)
    with torch.no_grad():
        moe.experts_gate_up.zero_()
        moe.experts_down.zero_()
    x = torch.randn(1, 3, 6)
    out = moe(x)
    expected = moe.shared_experts(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_b5_moe_softmax_matches_v2_reference_inline():
    """B5: api MoE softmax + greedy top-k matches an inline V2-style reference.

    Verifies the softmax + topk + routed_scaling_factor pipeline against
    modeling_deepseek_v2.py:100-130 implementation:
        scores = softmax(router_logits, fp32)
        topk_w, topk_idx = topk(scores, k)
        topk_w *= routed_scaling_factor
        for e in hit: route tokens, gate_up_proj, silu*up, down_proj,
                       weight by topk_w, scatter-add.
    """
    torch.manual_seed(11)
    H = 8
    I = 16
    E = 6
    K = 2
    scale = 1.7
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=0,
        router_kind="softmax", routed_scaling_factor=scale,
        expert_ffn=_moe_expert_ffn(I=I),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    with torch.no_grad():
        nn.init.normal_(moe.gate.weight, std=0.5)
        nn.init.normal_(moe.experts_gate_up, std=0.05)
        nn.init.normal_(moe.experts_down, std=0.05)
    x = torch.randn(1, 5, H)
    out = moe(x)

    # Reference inline.
    x_flat = x.reshape(-1, H)
    logits = F.linear(x_flat.float(), moe.gate.weight.float())
    scores = logits.softmax(dim=-1, dtype=torch.float32)
    topk_w_ref, topk_idx_ref = torch.topk(scores, k=K, dim=-1, sorted=False)
    topk_w_ref = topk_w_ref * scale
    ref_out = torch.zeros_like(x_flat)
    for n in range(x_flat.shape[0]):
        for slot in range(K):
            e = int(topk_idx_ref[n, slot].item())
            w = float(topk_w_ref[n, slot].item())
            gate_up = F.linear(x_flat[n:n+1], moe.experts_gate_up[e])
            g, u = gate_up.chunk(2, dim=-1)
            inner = F.silu(g) * u
            inner = F.linear(inner, moe.experts_down[e])
            ref_out[n] += w * inner.squeeze(0)
    ref_out = ref_out.reshape(*x.shape)
    assert torch.allclose(out, ref_out, atol=1e-5), (
        f"max_abs_diff={(out - ref_out).abs().max().item():.3e}"
    )


def test_b5_moe_sigmoid_plus_bias_router_shapes():
    spec = specs.MoESpec(
        n_experts=4, top_k=2, n_shared_experts=0,
        router_kind="sigmoid_plus_bias",
        router_norm=True,
        routed_scaling_factor=2.5,
        expert_ffn=_moe_expert_ffn(I=12),
    )
    moe = feedforward.MoE(spec, hidden_size=8, dtype=torch.float32)
    # V3 router has a `gate` submodule with weight Parameter and a buffer.
    assert hasattr(moe.gate, "weight")
    assert hasattr(moe.gate, "e_score_correction_bias")
    x = torch.randn(1, 3, 8)
    out = moe(x)
    assert out.shape == (1, 3, 8)


def test_b5_moe_sigmoid_router_uses_bias_only_for_choice_not_weight():
    """B5: V3 router gathers weights from BIAS-FREE sigmoid, but uses
    bias for top-k INDEX selection. Source: modeling_deepseek_v3.py:230-232.

    We construct a router where biasing a single expert's score correction
    flips its inclusion in top-k. The gathered weight must remain the
    bias-free sigmoid value at that index.
    """
    torch.manual_seed(3)
    H, E, K = 4, 6, 2
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=0,
        router_kind="sigmoid_plus_bias", router_norm=False,
        routed_scaling_factor=1.0,
        expert_ffn=_moe_expert_ffn(I=4),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    with torch.no_grad():
        nn.init.normal_(moe.gate.weight, std=0.3)
        # Bias only expert 0 by a huge amount → it WILL be picked.
        moe.gate.e_score_correction_bias[:] = 0.0
        moe.gate.e_score_correction_bias[0] = 100.0
        moe.experts_gate_up.zero_()    # zero outputs so we can isolate weights.
        moe.experts_down.zero_()

    x_flat = torch.randn(3, H)
    idx, w = moe._route_sigmoid_plus_bias(x_flat)
    # Expert 0 must appear in topk for every token.
    assert (idx == 0).any(dim=-1).all()
    # The weight for expert 0 must equal sigmoid(logit_0), NOT sigmoid + bias.
    logits = F.linear(x_flat.float(), moe.gate.weight.float())
    expected_w0 = logits[:, 0].sigmoid()
    # Find the slot where expert 0 was selected per token.
    matches = (idx == 0).int().argmax(dim=-1)
    actual_w0 = torch.gather(w, 1, matches.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(actual_w0, expected_w0, atol=1e-6)


def test_b5_moe_softmax_group_routing_v2_max_then_topk():
    """B5: V2 group-limited greedy routing: group score = MAX (not sum),
    then top-k_per_group, then mask non-selected groups with 0, then top-k.
    Source: modeling_deepseek_v2.py:107-117.
    """
    torch.manual_seed(5)
    n_groups, top_g, K = 4, 2, 3
    E = 8
    H = 4
    gr = specs.GroupRoutingSpec(n_groups=n_groups, topk_per_group=top_g)
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=0,
        router_kind="softmax", group_routing=gr,
        routed_scaling_factor=1.0, expert_ffn=_moe_expert_ffn(I=4),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    with torch.no_grad():
        nn.init.normal_(moe.gate.weight, std=0.5)

    x_flat = torch.randn(7, H)
    idx, w = moe._route_softmax(x_flat)
    # Verify each chosen expert lives in one of the top-g groups.
    logits = F.linear(x_flat.float(), moe.gate.weight.float())
    scores = logits.softmax(dim=-1, dtype=torch.float32)
    E_per_g = E // n_groups
    group_scores = scores.view(-1, n_groups, E_per_g).max(dim=-1).values
    top_groups = torch.topk(group_scores, k=top_g, dim=-1).indices  # [N, top_g]
    # For each token, every chosen expert must map to a top-group.
    for n in range(x_flat.shape[0]):
        chosen_groups = (idx[n] // E_per_g).tolist()
        allowed = set(top_groups[n].tolist())
        for g in chosen_groups:
            assert g in allowed


def test_b5_moe_no_router_norm_v2_default():
    """B5: V2's default norm_topk_prob=False — top-k weights NOT renormed
    to sum-1. We verify by setting routed_scaling_factor=1 and confirming
    the gathered weights equal the raw softmax probs (which generally do
    NOT sum to 1 across top-k).
    """
    torch.manual_seed(0)
    H, E, K = 4, 6, 2
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=0,
        router_kind="softmax", router_norm=False,
        routed_scaling_factor=1.0, expert_ffn=_moe_expert_ffn(I=4),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    with torch.no_grad():
        nn.init.normal_(moe.gate.weight, std=1.0)
    x_flat = torch.randn(4, H)
    idx, w = moe._route_softmax(x_flat)
    logits = F.linear(x_flat.float(), moe.gate.weight.float())
    scores = logits.softmax(dim=-1, dtype=torch.float32)
    ref_w = torch.gather(scores, 1, idx)
    assert torch.allclose(w, ref_w, atol=1e-6)
    # Sanity: w does not sum to 1 across slots.
    assert not torch.allclose(w.sum(-1), torch.ones(4), atol=1e-2)


# ---------------------------------------------------------------------------
# v7 P1: DeepSeek-V4 hash routing tests
# ---------------------------------------------------------------------------


def test_v7_p1_hash_router_shapes_and_buffers():
    """v7 P1: hash router exposes Parameter `weight` + Buffer `tid2eid`.

    Source: modeling_deepseek_v4.py:1059-1067.
    """
    V, E, K, H = 32, 6, 2, 8
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=0,
        router_kind="hash", hash_vocab_size=V,
        expert_ffn=_moe_expert_ffn(I=12),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    assert hasattr(moe.gate, "weight")
    assert hasattr(moe.gate, "tid2eid")
    assert moe.gate.weight.shape == (E, H)
    assert moe.gate.tid2eid.shape == (V, K)
    assert moe.gate.tid2eid.dtype == torch.long
    # No e_score_correction_bias (the hash router has none).
    assert not hasattr(moe.gate, "e_score_correction_bias")


def test_v7_p1_hash_routing_deterministic_per_token_id():
    """v7 P1: same token id → same expert selection regardless of hidden state.

    The whole point of hash routing is that `tid2eid[input_ids]` is the WHICH;
    only the per-expert WEIGHT gathered from `sigmoid(F.linear(x, weight))`
    varies with x. Two different hidden states with the same input_ids must
    select the same expert indices.

    Source: modeling_deepseek_v4.py:1075.
    """
    torch.manual_seed(7)
    V, E, K, H = 16, 4, 2, 6
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=0,
        router_kind="hash", hash_vocab_size=V,
        expert_ffn=_moe_expert_ffn(I=8),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    with torch.no_grad():
        # Populate the tid2eid table deterministically.
        nn.init.normal_(moe.gate.weight, std=0.5)
        for t in range(V):
            moe.gate.tid2eid[t, 0] = t % E
            moe.gate.tid2eid[t, 1] = (t + 1) % E

    input_ids = torch.tensor([[0, 5, 10]], dtype=torch.long)
    x1 = torch.randn(1, 3, H)
    x2 = torch.randn(1, 3, H)
    idx1, _ = moe._route_hash(x1.reshape(-1, H), input_ids)
    idx2, _ = moe._route_hash(x2.reshape(-1, H), input_ids)
    assert torch.equal(idx1, idx2), "hash routing must depend only on input_ids"
    # And the expected table values.
    expected = torch.tensor(
        [[0 % E, 1 % E],
         [5 % E, 6 % E],
         [10 % E, 11 % E]],
        dtype=torch.long,
    )
    assert torch.equal(idx1, expected)


def test_v7_p1_hash_routing_weights_from_sigmoid_then_normed():
    """v7 P1: hash router weights = sigmoid(F.linear(x, W)).gather(idx),
    then renormed to sum-1 across the top_k axis, then scaled by
    routed_scaling_factor.

    Source: modeling_deepseek_v4.py:1073-1078.
    """
    torch.manual_seed(11)
    V, E, K, H = 8, 4, 2, 4
    scale = 1.3
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=0,
        router_kind="hash", hash_vocab_size=V,
        routed_scaling_factor=scale,
        expert_ffn=_moe_expert_ffn(I=4),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    with torch.no_grad():
        nn.init.normal_(moe.gate.weight, std=0.5)
        for t in range(V):
            moe.gate.tid2eid[t, 0] = t % E
            moe.gate.tid2eid[t, 1] = (t + 2) % E

    input_ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    x = torch.randn(1, 4, H)
    x_flat = x.reshape(-1, H)
    idx, w = moe._route_hash(x_flat, input_ids)

    # Reference inline matching HF L1073-1078.
    logits = F.linear(x_flat, moe.gate.weight)
    scores = torch.sigmoid(logits)
    expected_idx = moe.gate.tid2eid[input_ids.reshape(-1)].long()
    expected_w = scores.gather(1, expected_idx)
    expected_w = expected_w / (expected_w.sum(dim=-1, keepdim=True) + 1e-20)
    expected_w = expected_w * scale
    assert torch.equal(idx, expected_idx)
    assert torch.allclose(w, expected_w, atol=1e-6)
    # Sum-to-1 (modulo scale).
    assert torch.allclose(w.sum(dim=-1), torch.full((4,), scale), atol=1e-5)


def test_v7_p1_hash_moe_forward_runs_with_input_ids():
    """v7 P1: full MoE forward dispatches experts when input_ids supplied."""
    torch.manual_seed(13)
    V, E, K, H = 10, 4, 2, 6
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=1,
        router_kind="hash", hash_vocab_size=V,
        expert_ffn=_moe_expert_ffn(I=8),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    with torch.no_grad():
        nn.init.normal_(moe.gate.weight, std=0.5)
        nn.init.normal_(moe.experts_gate_up, std=0.05)
        nn.init.normal_(moe.experts_down, std=0.05)
        # Distribute tids across experts so every expert is hit.
        for t in range(V):
            moe.gate.tid2eid[t, 0] = t % E
            moe.gate.tid2eid[t, 1] = (t + 1) % E

    input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    x = torch.randn(1, 5, H)
    out = moe(x, input_ids=input_ids)
    assert out.shape == (1, 5, H)
    assert torch.isfinite(out).all()


def test_v7_p1_hash_router_requires_vocab_size():
    """v7 P1: spec validation — router_kind='hash' without vocab size raises."""
    spec = specs.MoESpec(
        n_experts=4, top_k=2, n_shared_experts=0,
        router_kind="hash",  # missing hash_vocab_size
        expert_ffn=_moe_expert_ffn(I=4),
    )
    try:
        feedforward.MoE(spec, hidden_size=6, dtype=torch.float32)
    except ValueError as e:
        assert "hash_vocab_size" in str(e)
    else:
        raise AssertionError("expected ValueError for missing hash_vocab_size")


def test_v7_p1_hash_router_forward_requires_input_ids():
    """v7 P1: calling MoE.forward without input_ids when router='hash' raises."""
    V, E, K, H = 8, 4, 2, 4
    spec = specs.MoESpec(
        n_experts=E, top_k=K, n_shared_experts=0,
        router_kind="hash", hash_vocab_size=V,
        expert_ffn=_moe_expert_ffn(I=4),
    )
    moe = feedforward.MoE(spec, hidden_size=H, dtype=torch.float32)
    x = torch.randn(1, 3, H)
    try:
        moe(x)  # no input_ids
    except ValueError as e:
        assert "input_ids" in str(e)
    else:
        raise AssertionError("expected ValueError for missing input_ids")
