# DeepSeek-V4 — decoder layer composition

**Source-of-truth**: `transformers/models/deepseek_v4/{configuration_deepseek_v4,
modeling_deepseek_v4}.py` (transformers 5.10.x).

V4 introduces FOUR cross-cutting architectural changes vs V3:

1. **Hash MoE bootstrap** (paper §2.1) — first 3 MoE layers use a frozen
   `tid2eid[input_ids]` lookup instead of learned top-k. The learned gate
   weight still produces per-expert scores that weight the SELECTED experts'
   activations; only WHICH experts is static.
2. **CSA + HCA attention** (paper §2.3.1 / §2.3.2) — per-layer dispatch between
   sliding-window, compressed sparse, and heavily compressed branches. CSA uses
   m=4 compression + a Lightning Indexer top-k; HCA uses m'=128 compression
   over the full compressed sequence (no indexer).
3. **Manifold-Constrained Hyper-Connections (mHC)** (paper §2.2) — `hc_mult`
   parallel residual streams mixed via Sinkhorn-Knopp projected matrices.
4. **Grouped output projection** (paper §2.3.1) — output projection over
   `o_groups` head-groups with per-group low-rank intermediate.

This package lands only (1) and (2) into the api IR. (3) and (4) are out of
scope: the mHC residual streams + grouped output projection live entirely
inside the V4 decoder layer (`DeepseekV4DecoderLayer.forward`,
modeling_deepseek_v4.py:1125-1149) and are not expressible in the
single-residual `DecoderBlock` envelope.

## Block envelope (lossy abstraction)

```
residual := x
x      := input_layernorm(x)       # RMSNorm
x      := attention(x)              # shape-only (NotImplementedError in forward)
x      := residual + x

residual := x
x      := post_attention_layernorm(x)
x      := hash_or_topk_MoE(x, input_ids=ids)
x      := residual + x
```

The HF V4 block is NOT this — it has mHC mixing in/out of `hc_mult` parallel
streams (modeling_deepseek_v4.py:1138-1149). The `layer.py` factory returns a
SINGLE-residual envelope so the hash MoE sub-block can be exercised with the
existing block/MoE plumbing.

## Per-layer MLP dispatch

- `mlp_layer_types[L] == "hash_moe"` → `MoESpec.router_kind = "hash"` +
  `hash_vocab_size = vocab_size`.
- `mlp_layer_types[L] == "moe"` → `MoESpec.router_kind = "sigmoid_plus_bias"`
  + standard top-k with `e_score_correction_bias` (V3-style).

Default V4 schedule: first 3 layers `hash_moe`, remainder `moe`. Source:
configuration_deepseek_v4.py:165 (`default_num_hash_layers = 3`), 279-281.

## Per-layer attention dispatch

- `layer_types[L] == "sliding_attention"` → `AttentionKind.STANDARD` with
  `mask_kind = SWA` and `sliding_window = config.sliding_window` (default 128).
- `layer_types[L] == "compressed_sparse_attention"` → `AttentionKind.CSA_HCA`
  carrying a `CSASpec` (compress_rate=4, indexer params from
  index_n_heads/index_head_dim/index_topk).
- `layer_types[L] == "heavily_compressed_attention"` → `AttentionKind.CSA_HCA`
  carrying an `HCASpec` (compress_rate=128).

Default V4 schedule: first 2 layers `heavily_compressed_attention`, remainder
alternating CSA/HCA. Source: configuration_deepseek_v4.py:270-276.

## RoPE

V4 uses **interleaved** RoPE (one θ per channel pair, no end-to-end
duplication — `DeepseekV4RotaryEmbedding.forward`, modeling_deepseek_v4.py
:153-168) on a partial slice: `qk_rope_head_dim` of each head_dim is rotated,
the rest pass through. With `head_dim=512` and default
`partial_rotary_factor=64/512`, the rope slice is 64.

The CSA/HCA layers use a yarn-scaled "compress" rope theta (default 160000).
We model only the main θ on the AttentionSpec; the compressor's own rope
plumbing is shape-only.

## Hash MoE forward (numerical gate)

Verified at modeling_deepseek_v4.py:1069-1098. Per-token math:

```python
flat = hidden_states.reshape(-1, hidden)
logits = F.linear(flat, gate.weight)                   # [N, E]
scores = scoring_func(logits)                          # sigmoid (default V4: sqrtsoftplus)
indices = tid2eid[input_ids.reshape(-1)]               # [N, top_k]  FROZEN
weights = scores.gather(1, indices)
weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
weights = weights * routed_scaling_factor

# Experts: chunk(2, -1) SwiGLU with optional clamping.
gate, up = F.linear(x[token_idx], experts.gate_up_proj[e]).chunk(2, dim=-1)
gate = gate.clamp(max=swiglu_limit)
up   = up.clamp(min=-swiglu_limit, max=swiglu_limit)
out_e = (silu(gate) * up) @ experts.down_proj[e].T
out  += weights[token_idx, slot, None] * out_e

# Shared experts: a single DeepseekV4MLP on the ORIGINAL residual.
# Same clamping as routed.
return routed + shared_experts(residual)
```

The api `MoE(router_kind="hash")` matches this when `scoring_func="sigmoid"`
and `swiglu_limit` is large enough that the clamps are no-ops (we set 1e6 in
the test config). The HF default `scoring_func="sqrtsoftplus"` is NOT
supported by our `_HashRouter` — only sigmoid is.

## Shape-only attention forward

The CSA / HCA attention forwards raise `NotImplementedError` (Phase A P2).
The Attention module init reserves the spec dims so the block can be
constructed and weights can be allocated, but `blk.attention(x, ...)` is not
callable. For the layer-shape test we instantiate the block and only run the
MoE sublayer end-to-end.

## Production constants (DeepSeek-V4-Flash-Base)

```
hidden_size=4096, num_attention_heads=64, num_key_value_heads=1
head_dim=512, q_lora_rank=1024, qk_rope_head_dim=64
moe_intermediate_size=2048, n_routed_experts=256, n_shared_experts=1
num_experts_per_tok=6, routed_scaling_factor=1.5, scoring_func="sqrtsoftplus"
swiglu_limit=10.0, sliding_window=128
index_n_heads=64, index_head_dim=128, index_topk=512
compress_rate_csa=4, compress_rate_hca=128
mlp_layer_types=["hash_moe"]*3 + ["moe"]*(43-3) = 3 hash + 40 moe
layer_types=["heavily_compressed_attention"]*2 + alternating CSA/HCA
hc_mult=4, hc_sinkhorn_iters=20    # NOT modeled in llm-layers
o_groups=8, o_lora_rank=1024       # NOT modeled in llm-layers
```

## Weight name mapping (§7 schema)

HF state-dict keys for the MoE sublayer of `DeepseekV4ForCausalLM`:

```
model.layers.{L}.mlp.gate.weight                           # Hash + topk both
model.layers.{L}.mlp.gate.tid2eid                          # Hash only
model.layers.{L}.mlp.gate.e_score_correction_bias          # topk only
model.layers.{L}.mlp.experts.gate_up_proj                  # [E, 2*I, hidden]
model.layers.{L}.mlp.experts.down_proj                     # [E, hidden, I]
model.layers.{L}.mlp.shared_experts.gate_proj.weight       # [I*n_shared, hidden]
model.layers.{L}.mlp.shared_experts.up_proj.weight         # [I*n_shared, hidden]
model.layers.{L}.mlp.shared_experts.down_proj.weight       # [hidden, I*n_shared]
```

These map directly to `blk.feedforward.{gate.{weight,tid2eid,e_score_correction_bias},
experts_gate_up, experts_down, shared_experts.{gate,up,down}_proj.weight}` via
`load_hf_deepseek_v4_hash_moe`.

The V4 attention sub-block's state-dict keys (`q_a_proj`, `q_b_proj`,
`kv_proj`, `o_a_proj`, `o_b_proj`, `sinks`, `compressor.*`,
`q_a_norm`, `q_b_norm`, `kv_norm`) are NOT loaded here — that side ships
shape-only.
