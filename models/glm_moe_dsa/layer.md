# GLM-MoE-DSA (GLM-4.5+ / GLM-5) — decoder layer composition

## 0. At-a-glance

- **Signature:** MLA + DSA Lightning Indexer (shape-only) + V3-style sigmoid+bias MoE with per-layer dense/sparse dispatch
- **Active params:** ~37B active (GLM-5 approximate, real config pending)
- **Layer mix:** 61 attn (3 dense + 58 MoE, sigmoid+bias top-8 + 1 shared; DSA on all attn layers)
- **KV cache / token (bf16):** ~69 kB (61 layers × (512 + 64) × 2 B = 70272 B, MLA formula)

---

**Source-of-truth**: `transformers/models/glm_moe_dsa/{configuration,modeling}_glm_moe_dsa.py`
(transformers 5.10.x).

## At-a-glance

- **Token mixer**: MLA + DSA (DeepSeek Sparse Attention) Lightning Indexer.
  Identical architecture to DeepSeek-V3.2 MLA + Indexer (modeling
  comment, line 270-285).
- **Channel mixer**: per-layer dispatch between dense SwiGLU (`mlp_layer_types
  == "dense"`, first 3 layers by default) and a V3-style sigmoid+bias MoE
  with single-group routing (`mlp_layer_types == "sparse"`, remaining layers).
- RMSNorm uses standard `w * x_normed` form. Source:
  modeling_glm_moe_dsa.py:46-60.

## Block envelope

```
residual := x
x      := input_layernorm(x)         # RMSNorm
x      := MLA + DSA(x)               # SHAPE-ONLY in IR
x      := residual + x

residual := x
x      := post_attention_layernorm(x)
x      := dense_FFN(x)  OR  MoE(x)   # per mlp_layer_types[L]
x      := residual + x
```

Source: `GlmMoeDsaDecoderLayer.forward`, modeling_glm_moe_dsa.py:608-635.

## MoE: sigmoid + bias with group routing (sparse layer)

Identical to DeepSeek-V3 MoE — see `models/deepseek_v3_lite/layer.md` §
"MoE: sigmoid + bias router". The GLM defaults are `n_group=1, topk_group=1`
which is a degenerate single-group setup (functionally equivalent to no group
routing, but the code path always runs the group-routing scoring).

```python
router_logits = F.linear(x.fp32, gate.weight.fp32)       # [B*S, E]
router_probs  = sigmoid(router_logits)                   # [B*S, E]
probs_for_choice = router_probs + e_score_correction_bias
# Single-group: group_scores ≡ probs_for_choice.topk(2).sum(-1) over the
# WHOLE expert axis (since E_per_g = E).
group_idx = topk(group_scores, k=topk_group=1)
score_mask = scatter(zeros, group_idx)
scores_for_choice = probs_for_choice.masked_fill(~score_mask, -inf)
topk_idx = topk(scores_for_choice, k=top_k)
topk_w   = router_probs.gather(1, topk_idx)              # BIAS-FREE source
if norm_topk_prob:
    topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
topk_w   = topk_w * routed_scaling_factor
```

Source: `modeling_glm_moe_dsa.py:558-581`.

## MLA + DSA attention — SHAPE-ONLY

The api `AttentionKind.DSA` path allocates the MLA backbone (q_a_proj /
q_b_proj / kv_a_proj_with_mqa / kv_b_proj / o_proj) AND the indexer Q/K
projections, but `Attention.forward` raises NotImplementedError. Source:
api/attention.py (Phase A B5 reservation).

To verify shapes at construction time, see
`tests/models/glm_moe_dsa/test_layer_shape.py`. The MoE forward gate exists
independently — see `tests/models/glm_moe_dsa/test_isolation_hf.py`.

## Production constants (GLM-5)

```
hidden_size=6144, intermediate_size=12288, moe_intermediate_size=2048
num_hidden_layers=78, num_attention_heads=64, num_key_value_heads=64
n_routed_experts=256, n_shared_experts=1, num_experts_per_tok=8
routed_scaling_factor=2.5, norm_topk_prob=True
n_group=1, topk_group=1
kv_lora_rank=512, q_lora_rank=2048
qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256
index_topk=2048, index_head_dim=128, index_n_heads=32
mlp_layer_types = ["dense"]*3 + ["sparse"]*75
```

## Weight name mapping (§7 schema)

Sparse layer MoE (loaded by `load_hf_glm_moe_dsa_moe`):

| HF key                                                       | api slot                                                |
|--------------------------------------------------------------|---------------------------------------------------------|
| `model.layers.{L}.input_layernorm.weight`                    | `blk.pre_attn_norm.weight`                              |
| `model.layers.{L}.post_attention_layernorm.weight`           | `blk.pre_ffn_norm.weight`                               |
| `model.layers.{L}.mlp.gate.weight`                           | `blk.feedforward.gate.weight`                           |
| `model.layers.{L}.mlp.gate.e_score_correction_bias`          | `blk.feedforward.gate.e_score_correction_bias`          |
| `model.layers.{L}.mlp.experts.gate_up_proj`                  | `blk.feedforward.experts_gate_up`                       |
| `model.layers.{L}.mlp.experts.down_proj`                     | `blk.feedforward.experts_down`                          |
| `model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.W`  | `blk.feedforward.shared_experts.{gate,up,down}_proj.W`  |

MLA + DSA attention sub-block keys (`self_attn.{q_a_proj, q_b_proj,
kv_a_proj_with_mqa, kv_a_layernorm, q_a_layernorm, kv_b_proj, o_proj,
indexer.{wq_b, wk, k_norm, weights_proj}}`) are NOT loaded — shape-only.
