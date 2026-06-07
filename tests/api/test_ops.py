import pytest
import torch
import torch.nn.functional as F

from api import ops


def test_silu_matches_torch():
    x = torch.randn(2, 4, 8)
    assert torch.allclose(ops.silu(x), F.silu(x), atol=1e-7)


def test_add_with_residual():
    x = torch.randn(2, 4, 8)
    r = torch.randn(2, 4, 8)
    assert torch.allclose(ops.add(x, r), x + r, atol=1e-7)


def test_add_with_scale():
    x = torch.randn(2, 4, 8)
    r = torch.randn(2, 4, 8)
    out = ops.add(x, r, scale=0.5)
    assert torch.allclose(out, x + 0.5 * r, atol=1e-7)


def test_mul_elementwise():
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 4, 8)
    assert torch.allclose(ops.mul(a, b), a * b, atol=1e-7)


def test_linear_no_quant():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    out = ops.linear(x, w)
    assert torch.allclose(out, F.linear(x, w), atol=1e-5)


def test_linear_with_bias():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    b = torch.randn(8, dtype=torch.float32)
    out = ops.linear(x, w, bias=b)
    assert torch.allclose(out, F.linear(x, w, b), atol=1e-5)


def test_rms_norm_standard_w():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(16, dtype=torch.float32)
    eps = 1e-6
    out = ops.rms_norm(x, w, eps, mode="standard_w")
    ref = F.rms_norm(x, (16,), w, eps)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rms_norm_one_plus_w():
    """Gemma-style RMSNorm where weight is (1 + w)."""
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(16, dtype=torch.float32)
    eps = 1e-6
    out = ops.rms_norm(x, w, eps, mode="one_plus_w")
    ref = F.rms_norm(x, (16,), 1.0 + w, eps)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rms_norm_preserves_input_dtype():
    """RMSNorm computes in fp32 but returns input dtype."""
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    w = torch.randn(16, dtype=torch.bfloat16)
    out = ops.rms_norm(x, w, 1e-6, mode="standard_w")
    assert out.dtype == torch.bfloat16


def test_embed_basic():
    weight = torch.randn(100, 8, dtype=torch.float32)
    ids = torch.tensor([[0, 1, 2], [99, 50, 1]])
    out = ops.embed(ids, weight)
    assert out.shape == (2, 3, 8)
    assert torch.allclose(out[0, 0], weight[0])
    assert torch.allclose(out[1, 2], weight[1])


def test_embed_with_scale():
    """Gemma scales embeddings by sqrt(hidden_size)."""
    weight = torch.randn(10, 8, dtype=torch.float32)
    ids = torch.tensor([[1, 2]])
    out = ops.embed(ids, weight, scale=2.0)
    assert torch.allclose(out[0, 0], 2.0 * weight[1])


def test_lm_head_no_softcap():
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(100, 16, dtype=torch.float32)
    out = ops.lm_head(x, w)
    assert torch.allclose(out, F.linear(x, w), atol=1e-5)


def test_lm_head_with_softcap():
    """Gemma 2 / 3 logits softcap."""
    x = torch.randn(2, 4, 16, dtype=torch.float32)
    w = torch.randn(100, 16, dtype=torch.float32)
    cap = 30.0
    out = ops.lm_head(x, w, softcap=cap)
    logits = F.linear(x, w)
    ref = cap * torch.tanh(logits / cap)
    assert torch.allclose(out, ref, atol=1e-5)


def _hf_rope_half(x, cos, sin):
    """Reference: HF's rotate_half + (cos, sin) application.

    HF Llama-style: input is [..., D] split into halves [..., :D/2] and [..., D/2:].
    rotate_half swaps and negates: [-x2, x1].
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def test_rope_apply_split_half_matches_hf():
    """rope_apply with SPLIT_HALF basis should match HF's apply_rotary_pos_emb."""
    B, S, H, Dh = 1, 4, 2, 8
    q = torch.randn(B, S, H, Dh, dtype=torch.float32)
    k = torch.randn(B, S, H, Dh, dtype=torch.float32)

    # cos / sin tables shaped [S, Dh] (broadcast over B and H)
    freqs = torch.linspace(0.1, 1.0, Dh // 2)
    positions = torch.arange(S).float().unsqueeze(-1) * freqs.unsqueeze(0)
    cos = torch.cat([positions.cos(), positions.cos()], dim=-1)   # [S, Dh]
    sin = torch.cat([positions.sin(), positions.sin()], dim=-1)

    q_rot, k_rot = ops.rope_apply(q, k, cos, sin, basis="split_half")

    cos_b = cos.view(1, S, 1, Dh)
    sin_b = sin.view(1, S, 1, Dh)
    q_ref = _hf_rope_half(q, cos_b, sin_b)
    k_ref = _hf_rope_half(k, cos_b, sin_b)

    assert torch.allclose(q_rot, q_ref, atol=1e-6)
    assert torch.allclose(k_rot, k_ref, atol=1e-6)


def test_sdpa_matches_torch_for_mha():
    """For n_q_heads == n_kv_heads, sdpa should match F.scaled_dot_product_attention."""
    B, H, S, Dh = 1, 4, 6, 8
    q = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k = torch.randn(B, H, S, Dh, dtype=torch.float32)
    v = torch.randn(B, H, S, Dh, dtype=torch.float32)

    out = ops.sdpa(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_gqa_broadcasts_kv():
    """For GQA (n_kv_heads < n_q_heads), sdpa should repeat KV to match Q."""
    B, Hq, Hk, S, Dh = 1, 8, 2, 6, 16   # Hq // Hk == 4
    q = torch.randn(B, Hq, S, Dh, dtype=torch.float32)
    k = torch.randn(B, Hk, S, Dh, dtype=torch.float32)
    v = torch.randn(B, Hk, S, Dh, dtype=torch.float32)

    out = ops.sdpa(q, k, v, is_causal=True)
    k_rep = k.repeat_interleave(Hq // Hk, dim=1)
    v_rep = v.repeat_interleave(Hq // Hk, dim=1)
    ref = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_custom_scale():
    B, H, S, Dh = 1, 2, 4, 8
    q = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k = torch.randn(B, H, S, Dh, dtype=torch.float32)
    v = torch.randn(B, H, S, Dh, dtype=torch.float32)

    custom_scale = 0.25
    out = ops.sdpa(q, k, v, is_causal=False, scale=custom_scale)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=custom_scale)
    assert torch.allclose(out, ref, atol=1e-5)


def test_layer_norm_matches_torch():
    x = torch.randn(2, 4, 8, dtype=torch.float32)
    w = torch.randn(8); b = torch.randn(8)
    out = ops.layer_norm(x, w, b, 1e-5)
    ref = F.layer_norm(x, (8,), w, b, 1e-5)
    assert torch.allclose(out, ref, atol=1e-5)


def test_softmax_matches_torch():
    x = torch.randn(2, 4, 8)
    assert torch.allclose(ops.softmax(x, dim=-1), F.softmax(x, dim=-1), atol=1e-6)


def test_top_k_returns_values_and_indices():
    x = torch.tensor([[1.0, 3.0, 2.0, 5.0, 4.0]])
    vals, idx = ops.top_k(x, k=2)
    assert torch.equal(vals, torch.tensor([[5.0, 4.0]]))
    assert torch.equal(idx, torch.tensor([[3, 4]]))


def test_gather_works():
    x = torch.arange(12).reshape(3, 4).float()
    idx = torch.tensor([[0, 2], [1, 3], [0, 1]])
    out = ops.gather(x, dim=1, index=idx)
    assert torch.equal(out, torch.tensor([[0., 2.], [5., 7.], [8., 9.]]))


def test_scatter_returns_new_tensor():
    x = torch.zeros(2, 4)
    idx = torch.tensor([[0, 2], [1, 3]])
    src = torch.tensor([[10., 20.], [30., 40.]])
    out = ops.scatter(x, dim=1, index=idx, src=src)
    assert torch.equal(out, torch.tensor([[10., 0., 20., 0.], [0., 30., 0., 40.]]))
    # Input is unchanged.
    assert torch.equal(x, torch.zeros(2, 4))


def test_conv1d_matches_torch():
    x = torch.randn(1, 4, 16)        # [B, C_in, L]
    w = torch.randn(4, 4, 3)         # [C_out, C_in / groups, K]
    out = ops.conv1d(x, w, padding=1)
    ref = F.conv1d(x, w, padding=1)
    assert torch.allclose(out, ref, atol=1e-5)


def test_selective_scan_basic_shapes():
    """B7: selective_scan returns (y, final_state) of correct shapes."""
    B, S, H, Dh, N = 1, 16, 4, 8, 4
    chunk = 8
    hs = torch.randn(B, S, H, Dh, dtype=torch.float32)
    A = -torch.rand(B, S, H, dtype=torch.float32) * 0.1  # negative (decay)
    Bm = torch.randn(B, S, H, N, dtype=torch.float32)
    Cm = torch.randn(B, S, H, N, dtype=torch.float32)
    y, state = ops.selective_scan(hs, A, Bm, Cm, chunk_size=chunk)
    assert y.shape == (B, S, H, Dh)
    assert state.shape == (B, H, Dh, N)


def test_selective_scan_pads_to_chunk_size():
    """seq_len need not be a multiple of chunk_size — internal pad+truncate."""
    B, H, Dh, N = 1, 2, 4, 4
    chunk = 8
    for S in (5, 8, 11, 16, 19):
        hs = torch.randn(B, S, H, Dh, dtype=torch.float32)
        A = -torch.rand(B, S, H, dtype=torch.float32) * 0.1
        Bm = torch.randn(B, S, H, N, dtype=torch.float32)
        Cm = torch.randn(B, S, H, N, dtype=torch.float32)
        y, state = ops.selective_scan(hs, A, Bm, Cm, chunk_size=chunk)
        assert y.shape == (B, S, H, Dh), f"S={S}: got {tuple(y.shape)}"
        assert state.shape == (B, H, Dh, N)


def test_selective_scan_zero_decay_aggregates_x():
    """When A==0 (no decay) and B==C==1 (no projection), the SSM degenerates.

    With A=0: dA = exp(A*dt) = 1, so ssm_state[t] = ssm_state[t-1] + dB*x.
    With B[t,k]=1 for all t,k and C[t,k]=1: y[t] = sum_{s<=t} x[s].

    This sanity-checks the recurrence accumulation across chunk boundaries.
    """
    B, S, H, Dh, N = 1, 12, 1, 2, 1
    chunk = 4
    hs = torch.randn(B, S, H, Dh, dtype=torch.float32)
    A = torch.zeros(B, S, H, dtype=torch.float32)
    Bm = torch.ones(B, S, H, N, dtype=torch.float32)
    Cm = torch.ones(B, S, H, N, dtype=torch.float32)
    y, _ = ops.selective_scan(hs, A, Bm, Cm, chunk_size=chunk)
    expected = torch.cumsum(hs, dim=1)  # [B, S, H, Dh]
    assert torch.allclose(y, expected, atol=1e-5), (
        f"cumsum mismatch, max_abs_diff={(y - expected).abs().max().item()}"
    )


def test_selective_scan_matches_hf_naive():
    """B7: validate selective_scan against an inlined HF-style scalar recurrence.

    For each batch/head, compute the SSM recurrence step-by-step:
        s[t] = exp(A_t) * s[t-1] + B_t * x_t
        y[t] = C_t · s[t]
    where s[t] has shape [head_dim, d_state], x_t has shape [head_dim],
    B_t and C_t are [d_state] (per-head in this test).
    """
    torch.manual_seed(7)
    B, S, H, Dh, N = 1, 17, 3, 4, 5
    chunk = 8
    hs = torch.randn(B, S, H, Dh, dtype=torch.float64) * 0.5
    A = -torch.rand(B, S, H, dtype=torch.float64) * 0.3
    Bm = torch.randn(B, S, H, N, dtype=torch.float64) * 0.5
    Cm = torch.randn(B, S, H, N, dtype=torch.float64) * 0.5
    # Reference: explicit recurrence.
    ref_y = torch.zeros(B, S, H, Dh, dtype=torch.float64)
    state = torch.zeros(B, H, Dh, N, dtype=torch.float64)
    for t in range(S):
        dA = torch.exp(A[:, t]).view(B, H, 1, 1)              # [B, H, 1, 1]
        dBx = (Bm[:, t].unsqueeze(-2)                          # [B, H, 1, N]
               * hs[:, t].unsqueeze(-1))                       # [B, H, Dh, 1]
        # dBx -> [B, H, Dh, N]
        state = state * dA + dBx
        # y[t] = C @ state  -> sum_n C[..., n] * state[..., n]
        ref_y[:, t] = (Cm[:, t].unsqueeze(-2) * state).sum(dim=-1)
    # Run the SSD op in fp64 by casting on the way in.
    y, _ = ops.selective_scan(
        hs.to(torch.float64), A.to(torch.float64),
        Bm.to(torch.float64), Cm.to(torch.float64),
        chunk_size=chunk,
    )
    max_diff = (y - ref_y).abs().max().item()
    assert max_diff < 5e-4, f"selective_scan vs scalar recurrence max_diff={max_diff}"


def test_selective_scan_respects_initial_state():
    """initial_state arg = continue an existing recurrence."""
    torch.manual_seed(11)
    B, S, H, Dh, N = 1, 8, 2, 4, 3
    chunk = 4
    hs = torch.randn(B, S, H, Dh, dtype=torch.float64) * 0.5
    A = -torch.rand(B, S, H, dtype=torch.float64) * 0.3
    Bm = torch.randn(B, S, H, N, dtype=torch.float64) * 0.5
    Cm = torch.randn(B, S, H, N, dtype=torch.float64) * 0.5
    init = torch.randn(B, H, Dh, N, dtype=torch.float64) * 0.5
    # Reference: explicit recurrence starting from init.
    ref_y = torch.zeros(B, S, H, Dh, dtype=torch.float64)
    state = init.clone()
    for t in range(S):
        dA = torch.exp(A[:, t]).view(B, H, 1, 1)
        dBx = (Bm[:, t].unsqueeze(-2) * hs[:, t].unsqueeze(-1))
        state = state * dA + dBx
        ref_y[:, t] = (Cm[:, t].unsqueeze(-2) * state).sum(dim=-1)
    y, final = ops.selective_scan(hs, A, Bm, Cm, chunk_size=chunk,
                                   initial_state=init)
    max_diff = (y - ref_y).abs().max().item()
    assert max_diff < 5e-4, (
        f"initial_state respected? max_diff={max_diff}"
    )
    state_diff = (final - state).abs().max().item()
    assert state_diff < 5e-4, f"final state diff {state_diff}"


def test_rope_apply_partial_rotary_factor_quarter():
    """Gemma 4 global RoPE: only the first 25% of head_dim channels rotate."""
    B, S, H, Dh = 1, 4, 2, 16
    q = torch.randn(B, S, H, Dh, dtype=torch.float32)
    k = torch.randn(B, S, H, Dh, dtype=torch.float32)

    # Build cos/sin for the rotated portion only (Dh_rot = 4 channels of the head)
    Dh_rot = int(Dh * 0.25)  # 4
    freqs = torch.linspace(0.1, 1.0, Dh_rot // 2)
    positions = torch.arange(S).float().unsqueeze(-1) * freqs.unsqueeze(0)
    cos_rot = torch.cat([positions.cos(), positions.cos()], dim=-1)  # [S, Dh_rot]
    sin_rot = torch.cat([positions.sin(), positions.sin()], dim=-1)

    q_rot, k_rot = ops.rope_apply_partial(q, k, cos_rot, sin_rot,
                                          partial_rotary_factor=0.25,
                                          basis="split_half")
    # Non-rotated tail (channels Dh_rot..Dh) is identical to input
    assert torch.allclose(q_rot[..., Dh_rot:], q[..., Dh_rot:], atol=1e-7)
    assert torch.allclose(k_rot[..., Dh_rot:], k[..., Dh_rot:], atol=1e-7)
    # Rotated head (first Dh_rot channels) differs from input at non-zero positions
    assert not torch.allclose(q_rot[:, 1, :, :Dh_rot], q[:, 1, :, :Dh_rot], atol=1e-3)


def test_gelu_pytorch_tanh_matches_torch():
    """Gemma 4 uses hidden_activation='gelu_pytorch_tanh'."""
    x = torch.randn(2, 4, 8)
    out = ops.gelu_pytorch_tanh(x)
    ref = F.gelu(x, approximate="tanh")
    assert torch.allclose(out, ref, atol=1e-6)


def test_sdpa_logit_softcap_matches_gemma2_eager():
    """B3: sdpa with `logit_softcap` matches HF Gemma 2 eager attention math.

    Source: modeling_gemma2.py:212-225 — `attn = matmul(q,k.T)*scale; attn =
    tanh(attn/cap)*cap; attn += mask; attn = softmax(...fp32).to(q.dtype);
    out = matmul(attn, v)`.
    """
    B, H, S, Dh = 1, 4, 6, 32
    cap = 50.0
    q = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k = torch.randn(B, H, S, Dh, dtype=torch.float32)
    v = torch.randn(B, H, S, Dh, dtype=torch.float32)
    # Build a causal mask matching what api/attention builds.
    causal = torch.zeros(S, S, dtype=q.dtype)
    causal = causal.masked_fill(
        torch.arange(S).unsqueeze(0) > torch.arange(S).unsqueeze(1),
        float("-inf"),
    )
    attn_mask = causal.unsqueeze(0).unsqueeze(0)

    scale = Dh ** -0.5
    # Reference: HF Gemma 2 eager_attention_forward inlined.
    attn_w = (q @ k.transpose(-2, -1)) * scale
    attn_w = torch.tanh(attn_w / cap) * cap
    attn_w = attn_w + attn_mask
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
    ref = attn_w @ v

    out = ops.sdpa(q, k, v, attn_mask=attn_mask, scale=scale, logit_softcap=cap)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_logit_softcap_none_matches_plain_sdpa():
    """B3: logit_softcap=None is a pass-through to F.scaled_dot_product_attention."""
    B, H, S, Dh = 1, 2, 4, 16
    q = torch.randn(B, H, S, Dh, dtype=torch.float32)
    k = torch.randn(B, H, S, Dh, dtype=torch.float32)
    v = torch.randn(B, H, S, Dh, dtype=torch.float32)
    causal = torch.zeros(S, S, dtype=q.dtype)
    causal = causal.masked_fill(
        torch.arange(S).unsqueeze(0) > torch.arange(S).unsqueeze(1),
        float("-inf"),
    )
    attn_mask = causal.unsqueeze(0).unsqueeze(0)
    out = ops.sdpa(q, k, v, attn_mask=attn_mask, logit_softcap=None)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_logit_softcap_gqa_repeats_kv():
    """B3: logit_softcap path still GQA-repeats K/V when n_q_heads > n_kv_heads."""
    B, Hq, Hk, S, Dh = 1, 8, 2, 5, 16
    q = torch.randn(B, Hq, S, Dh, dtype=torch.float32)
    k = torch.randn(B, Hk, S, Dh, dtype=torch.float32)
    v = torch.randn(B, Hk, S, Dh, dtype=torch.float32)
    causal = torch.zeros(S, S, dtype=q.dtype)
    causal = causal.masked_fill(
        torch.arange(S).unsqueeze(0) > torch.arange(S).unsqueeze(1),
        float("-inf"),
    )
    attn_mask = causal.unsqueeze(0).unsqueeze(0)
    out = ops.sdpa(q, k, v, attn_mask=attn_mask, logit_softcap=30.0)
    # Manual ref with KV repetition.
    k_rep = k.repeat_interleave(Hq // Hk, dim=1)
    v_rep = v.repeat_interleave(Hq // Hk, dim=1)
    scale = Dh ** -0.5
    attn_w = (q @ k_rep.transpose(-2, -1)) * scale
    attn_w = torch.tanh(attn_w / 30.0) * 30.0
    attn_w = attn_w + attn_mask
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
    ref = attn_w @ v_rep
    assert torch.allclose(out, ref, atol=1e-5)
