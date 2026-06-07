# DeepSeek-V3-Lite — decoder layer composition

**Source-of-truth**: `transformers/models/deepseek_v3/modeling_deepseek_v3.py`.

V3-full is 671B (out-of-scope for SLM gates). V3-Lite is the
shape-only / synthetic variant exercising V3's architectural deltas vs V2.

## Block envelope

Identical to V2 — PRE-norm, no μP residual scaling:

```
residual := x
x      := input_layernorm(x)        # RMSNorm
x      := MLA-attention(x)
x      := residual + x

residual := x
x      := post_attention_layernorm(x)
x      := MoE OR dense FFN(x)
x      := residual + x
```
Source: `modeling_deepseek_v3.py:497-526`.

## RoPE

V3 uses `apply_rotary_pos_emb` with `rotate_half` — that's
**SPLIT_HALF** basis in our api, NOT V2's INTERLEAVED.

Source: `modeling_deepseek_v3.py:250-280`. (V3 also has a
`rope_interleave` flag that switches to `apply_rotary_pos_emb_interleave`
which does a view+transpose reshape before standard rotate_half — this is
NOT modeled here; B5 V3-Lite assumes `rope_interleave=False`.)

## MoE: sigmoid + bias router (V3 delta)

The big architectural innovation in V3 is the auxiliary-loss-free routing:

```python
router_logits = F.linear(x.fp32, gate.weight.fp32)         # [B*S, E]
router_probs  = sigmoid(router_logits)                     # bias-free
probs_for_choice = router_probs + e_score_correction_bias  # ROUTING ONLY

# Group-limited routing (always on for V3).
group_scores = (
    probs_for_choice
    .view(B*S, n_groups, E_per_g)
    .topk(2, dim=-1).values             # SUM of TOP-2 per group (NOT max).
    .sum(dim=-1)
)
group_idx = topk(group_scores, k=topk_per_group).indices
group_mask = scatter(zeros, group_idx, 1)
score_mask = group_mask.unsqueeze(-1).expand(...).reshape(B*S, E)
scores_for_choice = probs_for_choice.masked_fill(~score_mask, -inf)

topk_idx = topk(scores_for_choice, k=top_k).indices
# CRITICAL: weights gathered from the BIAS-FREE sigmoid output.
topk_w   = router_probs.gather(1, topk_idx)
if norm_topk_prob:
    topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
topk_w   = topk_w * routed_scaling_factor
```

Source: `modeling_deepseek_v3.py:214-237`. The
`e_score_correction_bias` is a **non-learnable buffer** updated by
running statistics during training (not at inference). HF marks it
`_keep_in_fp32_modules_strict` — we keep it fp32 by default in our
`_SigmoidRouter`.

## V3-full production constants (for reference)

```
hidden_size=7168, num_attention_heads=128, num_hidden_layers=61
n_routed_experts=256, num_experts_per_tok=8, n_shared_experts=1
n_group=8, topk_group=4, routed_scaling_factor=2.5, norm_topk_prob=True
first_k_dense_replace=3      # layers 0,1,2 dense; 3..60 MoE
q_lora_rank=1536, kv_lora_rank=512
qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
```

## Numerical isolation

`tests/models/deepseek_v3_lite/test_isolation_hf.py` validates our MoE
against HF DeepseekV3MoE with synthetic weights and a non-trivial
`e_score_correction_bias`. Diff = 0.0 (bit-exact).

The full DecoderLayer numerical gate is deferred — V3-full weights are
671B and impractical for CI; the V2-Lite gate exercises the same MLA +
dense-FFN-vs-MoE-dispatch + block-envelope code paths at scale.
