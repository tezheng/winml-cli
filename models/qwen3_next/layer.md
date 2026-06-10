# Qwen3-Next — decoder layer composition

**Source-of-truth**: `transformers/models/qwen3_next/{configuration_qwen3_next,
modeling_qwen3_next}.py` (transformers 5.10.x).

## At-a-glance

- 3:1 hybrid: every 4th layer is **full attention**, the rest are **Gated
  DeltaNet** (linear attention). Source: configuration_qwen3_next.py:121-127.
- RMSNorm uses `(1 + weight)` form (NormWeightMode.ONE_PLUS_W). Source:
  modeling_qwen3_next.py:152-166.
- Channel mixer: Qwen3NextSparseMoeBlock (softmax router + shared_expert with
  per-token `sigmoid(shared_expert_gate(x))` weighting). Some layers may opt
  into dense MLP only (`mlp_only_layers`).

## Block envelope

```
residual := x
x      := input_layernorm(x)                # RMSNorm — (1 + w)
x      := linear_attn(x)  OR  self_attn(x)  # per layer_types[L]
x      := residual + x

residual := x
x      := post_attention_layernorm(x)
x      := mlp(x)                            # MoE  OR  dense MLP
x      := residual + x
```

Source: `Qwen3NextDecoderLayer.forward`, modeling_qwen3_next.py:841-884.

## Linear-attention layer — Gated DeltaNet (numerical-gated)

The `Qwen3NextGatedDeltaNet` module is the architectural novelty (paper §
2.2). Phase A P3 (`api.ssm.GatedDeltaNetMixer`) ports the forward at
sequential per-step granularity. State-dict naming aligns 1:1 (see §7).

```
proj_qkvz = in_proj_qkvz(x)     # 2*key_dim + 2*value_dim
proj_ba   = in_proj_ba(x)       # 2*num_v_heads
q,k,v,z,b,a = fix_query_key_value_ordering(proj_qkvz, proj_ba)
mixed_qkv = cat(q.flat, k.flat, v.flat).T
mixed_qkv = silu(conv1d(mixed_qkv))[..., :S].T
q,k,v = split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
beta = sigmoid(b)
g    = -exp(A_log.float()) * softplus(a.float() + dt_bias)
# GQA: repeat_interleave(q, k, n_v_heads // n_k_heads, dim=2)
out, _ = gated_delta_rule(q, k, v, g, beta, use_qk_l2norm=True)
out = norm(out, z)                # silu(z) * rms_normed(out)
return out_proj(out.flatten)
```

Source: modeling_qwen3_next.py:499-717.

## Full-attention layer — SHAPE-ONLY

Qwen3-Next's `Qwen3NextAttention` has TWO non-standard features:

1. **Fused (q | gate) projection** (modeling_qwen3_next.py:268, 296-299):
   ```
   q_proj = Linear(hidden, num_heads * head_dim * 2, bias=False)
   q, gate = q_proj(x).chunk(2, dim=-1)
   ```
2. **Output gating** (line 326-327):
   ```
   attn_output = attn_output * sigmoid(gate)
   attn_output = o_proj(attn_output)
   ```

Neither feature is in the api `AttentionSpec`. The IR's STANDARD attention
ships a plain q_proj of size `num_heads * head_dim` with no output gating.
We mark this layer SHAPE-ONLY and do NOT gate it against HF numerically.

For the linear-attention layers, we DO gate vs HF at atol=5e-4 — see
`tests/models/qwen3_next/test_numerical_synthetic.py`.

## Channel mixer

Qwen3NextSparseMoeBlock (modeling_qwen3_next.py:797-816):

```
hidden_2d = hidden.reshape(-1, H)
shared_out = shared_expert(hidden_2d)
_, w, idx = gate(hidden_2d)                    # softmax + norm_topk_prob
expert_out = experts(hidden_2d, idx, w)
shared_out = sigmoid(shared_expert_gate(hidden_2d)) * shared_out
return (expert_out + shared_out).reshape(B, S, H)
```

The api `MoE(router_kind="softmax")` matches the routed-experts side but
does NOT model the per-token `sigmoid(shared_expert_gate(x))` multiply on
the shared expert. For numerical gating we use the dense MLP path (set
`mlp_only_layers=[L]`).

## Production constants (Qwen3-Next 80B-A3B)

```
hidden_size=2048, intermediate_size=5632, moe_intermediate_size=512
shared_expert_intermediate_size=512, num_hidden_layers=48
num_attention_heads=16, num_key_value_heads=2, head_dim=256
linear_num_v_heads=32, linear_num_k_heads=16
linear_value_head_dim=128, linear_key_head_dim=128, linear_conv_kernel_dim=4
num_experts=512, num_experts_per_tok=10
partial_rotary_factor=0.25, rope_theta=10000 (or yarn-scaled)
```

## Weight name mapping (§7 schema)

Linear-attention layer (loaded by `load_hf_qwen3_next_linear_attn_layer`):

| HF key                                                | api slot                                                |
|-------------------------------------------------------|---------------------------------------------------------|
| `model.layers.{L}.input_layernorm.weight`             | `blk.pre_attn_norm.weight`                              |
| `model.layers.{L}.post_attention_layernorm.weight`    | `blk.pre_ffn_norm.weight`                               |
| `model.layers.{L}.linear_attn.in_proj_qkvz.weight`    | `blk.attention.in_proj_qkvz.weight`                     |
| `model.layers.{L}.linear_attn.in_proj_ba.weight`      | `blk.attention.in_proj_ba.weight`                       |
| `model.layers.{L}.linear_attn.conv1d.weight`          | `blk.attention.conv1d.weight`                           |
| `model.layers.{L}.linear_attn.dt_bias`                | `blk.attention.dt_bias`                                 |
| `model.layers.{L}.linear_attn.A_log`                  | `blk.attention.A_log`                                   |
| `model.layers.{L}.linear_attn.norm.weight`            | `blk.attention.norm.weight`                             |
| `model.layers.{L}.linear_attn.out_proj.weight`        | `blk.attention.out_proj.weight`                         |
| `model.layers.{L}.mlp.{gate,up,down}_proj.weight`     | `blk.feedforward.{gate,up,down}_proj.weight` (dense MLP)|

Full-attention layer keys (`linear_attn` → `self_attn`) are NOT loaded — the
shape-only IR is missing q-gate and output-gating tensors.
