# MiniMax-M2 — decoder layer composition

## 0. At-a-glance

- **Signature:** GQA + FULL_HDH QK-norm (OLMo-2 style) + sigmoid+bias MoE (no group routing, no shared experts) — confirmed NOT Lightning Attention
- **Active params:** ~45.9B active / ~456B total (10%) for MiniMax-Text-01
- **Layer mix:** 80 attn (all MoE, sigmoid+bias top-8, 256 experts)
- **KV cache / token (bf16):** 320 kB (80 layers × 8 kv heads × 128 head_dim × 4 B)

---

**Source-of-truth**: `transformers/models/minimax_m2/{configuration,modeling}_minimax_m2.py`
(transformers 5.10.x).

## At-a-glance

- **Token mixer**: standard attention (NOT Lightning attention, the v7 Raschka
  agent confirmed at modeling_minimax_m2.py:296) with q/k norms applied to the
  FULL flattened projection (FULL_HDH) BEFORE view+transpose — that's the
  PRE-RoPE OLMo-2-style placement.
- **Channel mixer**: V3-style sigmoid+bias MoE WITHOUT group routing and
  WITHOUT shared experts. The `e_score_correction_bias` is normally zero but
  remains a learned/buffered tensor for forward-compatibility.
- **RoPE**: SPLIT_HALF basis, standard scaling, no partial rotary. The
  M2 default `rope_theta = 5_000_000`.

## Block envelope

```
residual := x
x      := input_layernorm(x)             # RMSNorm (w * x_normed)
x      := self_attn(x)                   # STANDARD with q_norm/k_norm FULL_HDH PRE_ROPE
x      := residual + x

residual := x
x      := post_attention_layernorm(x)
x      := mlp(x)                         # MoE: sigmoid+bias, no groups, no shared
x      := residual + x
```

Source: `MiniMaxM2DecoderLayer.forward`, modeling_minimax_m2.py:371-395.

## Attention

```python
q = q_proj(x)                       # [B, S, H*Dh]
k = k_proj(x)                       # [B, S, Hk*Dh]
v = v_proj(x)                       # [B, S, Hk*Dh]
q = q_norm(q).view(B, S, H, Dh).transpose(1,2)
k = k_norm(k).view(B, S, Hk, Dh).transpose(1,2)
v = v.view(B, S, Hk, Dh).transpose(1,2)
q, k = apply_rotary_pos_emb(q, k, cos, sin)        # SPLIT_HALF
attn_out = sdpa(q, k, v, scale=Dh^{-0.5}, causal=True)
return o_proj(attn_out.reshape(B, S, -1))
```

Source: modeling_minimax_m2.py:295-330.

## MoE

```python
flat = x.view(-1, H)
router_logits = F.linear(flat, gate.weight)
routing_weights = sigmoid(router_logits.float())          # in fp32
scores_for_choice = routing_weights + e_score_correction_bias
_, top_k_idx = topk(scores_for_choice, top_k, sorted=False)
top_k_w = routing_weights.gather(1, top_k_idx)           # BIAS-FREE source
top_k_w /= top_k_w.sum(-1, keepdim=True)                 # always normalised
# experts: standard chunk(2,-1) SwiGLU on the routed tokens
# No shared experts. No routed_scaling_factor.
```

Source: modeling_minimax_m2.py:46-124.

The router output is bit-equivalent to our `_route_sigmoid_plus_bias` with
`group_routing=None`, `router_norm=True`, `routed_scaling_factor=1.0`.

## Production constants (MiniMax-M2)

```
hidden_size=3072, intermediate_size=1536  (= moe_intermediate_size — single name)
num_hidden_layers=62
num_attention_heads=48, num_key_value_heads=8, head_dim=128
num_local_experts=256, num_experts_per_tok=8
rope_theta=5_000_000, max_position_embeddings=196608
vocab_size=200064
```

## Weight name mapping (§7 schema)

Loaded by `load_hf_minimax_m2_layer`:

| HF key                                              | api slot                                                 |
|-----------------------------------------------------|----------------------------------------------------------|
| `model.layers.{L}.input_layernorm.weight`           | `blk.pre_attn_norm.weight`                               |
| `model.layers.{L}.post_attention_layernorm.weight`  | `blk.pre_ffn_norm.weight`                                |
| `model.layers.{L}.self_attn.{q,k,v,o}_proj.weight`  | `blk.attention.{q,k,v,o}_proj.weight`                    |
| `model.layers.{L}.self_attn.q_norm.weight`          | `blk.attention.q_norm.weight`                            |
| `model.layers.{L}.self_attn.k_norm.weight`          | `blk.attention.k_norm.weight`                            |
| `model.layers.{L}.mlp.gate.weight`                  | `blk.feedforward.gate.weight`                            |
| `model.layers.{L}.mlp.e_score_correction_bias`*     | `blk.feedforward.gate.e_score_correction_bias`           |
| `model.layers.{L}.mlp.experts.gate_up_proj`         | `blk.feedforward.experts_gate_up`                        |
| `model.layers.{L}.mlp.experts.down_proj`            | `blk.feedforward.experts_down`                           |

\* The `e_score_correction_bias` is registered on the SparseMoeBlock in HF
(modeling_minimax_m2.py:114), NOT on the router. Our api MoE puts it under
the router (matching V3 / GLM convention). The weight loader re-keys.
