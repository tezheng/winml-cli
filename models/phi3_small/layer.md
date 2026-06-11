# Phi-3-small (microsoft/Phi-3-small-8k-instruct) — layer.md

## 0. At-a-glance

- **Signature:** GQA-32:8 + block-sparse attention (every other layer dense) + GeGELU FFN + LayerNorm + μP scalars; shape-only (numerical gate deferred)
- **Active params:** 7B total
- **Layer mix:** 32 attn (GQA-32:8), alternating block-sparse / dense (dense every 2nd layer)
- **KV cache / token (bf16):** 32 L × 8 KV × 128 Dh × 2 × 2 = 128 kB

---

SHAPE-ONLY skeleton for B2b. The numerical gate is deferred — the
BlockSparse mask construction is involved enough to warrant a separate
milestone.

## 1. Architecture summary

- 32 decoder layers, hidden_size=4096, n_q_heads=32, n_kv_heads=8 (GQA),
  head_dim=128.
- LayerNorm (NOT RMSNorm) at every position, eps=1e-5.
- GeGELU MLP with limit=20.0; ff_intermediate_size=14336.
- Half the layers are BlockSparse, half are dense — controlled by
  `dense_attention_every_n_layers=2`: layer `L` is DENSE iff
  `(L + 1) % 2 == 0`, i.e. layer_idx 1, 3, 5, ... are dense and
  layer_idx 0, 2, 4, ... are BlockSparse.
  Source: modeling_phi3_small.py:213.
- BlockSparse params: block_size=64, num_local_blocks=16, vert_stride=8,
  kernel_block_size=64, homo_head_pattern=False.
- RoPE base=1_000_000, position_scale=1.0, partial_rotary_factor=1.0.
- μP scaling: mup_use_scaling=True, mup_attn_multiplier=1.0,
  mup_width_multiplier=8.0, mup_embedding_multiplier=10.0.
  softmax_scale = `mup_attn_multiplier / head_dim` (NOT `1/sqrt(head_dim)`).
  Source: modeling_phi3_small.py:200-206.
- No biases (attention_bias=False).

## 2. Block diagram (LayerNorm sandwich, PRE-norm)

```
       x        ── residual ─────────────────┐
       |                                      |
       input_layernorm (LayerNorm, eps=1e-5)  |
       |                                      v
       self_attn (BlockSparse OR dense)       +
       |                                      ^
       └──────────────────────────────────────┘
       |
       x'       ── residual ─────────────────┐
       |                                      |
       post_attention_layernorm               |
       |                                      v
       Phi3SmallMLP (GeGELU)                  +
       |                                      ^
       └──────────────────────────────────────┘
       |
       y
```

Source: modeling_phi3_small.py:647-696.

## 3. Self-attention shape

Single fused `query_key_value` Linear of output dim `(Hq + 2*Hk) * head_dim`
= `(32 + 16) * 128 = 6144`. This is different from Phi-3-mini's
`(Hq + 2*Hk) * head_dim` flat layout — Phi-3-small ORDERS the heads
INTERLEAVED so that under tensor-parallel sharding all heads in a single
KV-group stay together:

```
[bs, sq, (q00, q01, ... q0m, k0, v0),         # KV-group 0
         (q10, q11, ... q1m, k1, v1),         # KV-group 1
         ...                                    
         (q_{n0}, ..., q_{nm}, k_n, v_n)]     # KV-group n
```

where `m = num_q_per_kv = Hq/Hk = 4` for the 8K variant.

Source: modeling_phi3_small.py:265-279.

A `dense` Linear (`h → h`) handles the output projection (== `o_proj`
in Llama-shaped models).

## 4. GeGELU MLP

```
up_proj  : Linear(h → 2 * intermediate)         # 4096 → 28672
down_proj: Linear(intermediate → h)             # 14336 → 4096
```

Forward:

```python
y = up_proj(x)                                # [..., 2*intermediate]
a_gelu   = y[..., ::2]                         # even indices
a_linear = y[..., 1::2]                        # odd indices
a_gelu   = clamp(a_gelu, max=20.0)             # gegelu_limit
a_linear = clamp(a_linear, min=-20.0, max=20.0)
out = quick_gelu(a_gelu) * (a_linear + 1)      # quick_gelu = x * sigmoid(1.702*x)
return down_proj(out)
```

This is structurally a fused-gate-up GLU variant with `quick_gelu` as the
activation and `+1` added to the linear branch (NOT the standard
`act(gate) * up` from Llama/Phi-3-mini). The `+1` shift means the FFN
identity-passes the linear contribution when `a_gelu` is far negative.

Source: modeling_phi3_small.py:88-98 (gegelu), 154-172 (MLP forward).

## 5. BlockSparse mask

For each query block, the keys attended to are:
- LOCAL: the most-recent `num_local_blocks * block_size = 16 * 64 = 1024`
  positions.
- VERTICAL: keys at strides of `vert_stride * block_size = 8 * 64 = 512`
  positions (`homo_head_pattern=False` → each head picks its own random
  vertical anchors per modeling_phi3_small.py BlockSparseAttentionLayer).
- The mask is applied per-head; head heterogeneity matters for the gate.

The reference implementation uses a Triton kernel (`BlockSparseAttentionLayer`
in `positional_embedding.py`); we do NOT port the Triton kernel for B2b.

## 6. μP scalars

| Scalar                       | HF source                                   | Where applied                          |
|------------------------------|---------------------------------------------|----------------------------------------|
| mup_embedding_multiplier=10  | modeling_phi3_small.py: embedding scale     | DecoderBlockSpec.embedding_scale       |
| mup_attn_multiplier=1.0      | modeling_phi3_small.py:200-206              | AttentionSpec.attn_scale (= 1.0/128)   |
| mup_width_multiplier=8.0     | modeling_phi3_small.py: lm_head divisor     | (B2b: not wired; future)               |

## 7. Deferred work (B2b says SHAPE-ONLY)

For the numerical gate to land:
1. Implement `LayerNorm` flavor of `api.norm.RMSNorm` (currently only RMS).
2. Implement the interleaved-fused QKV slicing in `api.attention` (B2b
   only handles the simple `[Q; K; V]` flat fused layout).
3. Implement the GeGELU forward (quick_gelu + `+1` shift) in `api.ops` and
   `api.feedforward`.
4. Implement BlockSparse mask construction. The vert_stride per-head pattern
   uses a deterministic seed per head; this must be reproduced exactly.

## 8. Weight name mapping (for future loader)

```
model.layers.{L}.input_layernorm.{weight,bias}
model.layers.{L}.post_attention_layernorm.{weight,bias}
model.layers.{L}.self_attn.query_key_value.{weight,bias}
model.layers.{L}.self_attn.dense.{weight,bias}
model.layers.{L}.mlp.up_proj.{weight,bias}
model.layers.{L}.mlp.down_proj.{weight,bias}
```

Note: Phi-3-small uses biases on LayerNorm AND on Linear (because
`attention_bias=False` actually refers only to nn.Linear; LayerNorm always
has a bias). Verified: the layer factory uses `bias=cfg.attention_bias` for
Linear and the standard LayerNorm constructor for norms (which always has
a learnable bias).
