# Qwen3-MoE — Decoder Layer

## 0. At-a-glance

- **Signature:** Qwen3 backbone + per-layer dense/MoE dispatch (mlp_only_layers in config) + PRE-RoPE QK-norm
- **Active params:** ~3B active / ~30B total (10%) for Qwen3-30B-A3B; ~22B active / ~235B total for Qwen3-235B-A22B
- **Layer mix:** 28 attn (all layers; MoE channel unless in mlp_only_layers, 128 experts top-8)
- **KV cache / token (bf16):** 56 kB (28 layers × 4 kv heads × 128 head_dim × 4 B)

---

## 1. Identity

- **Family:** Qwen3-MoE (Alibaba). Reference checkpoints: `Qwen/Qwen3-30B-A3B`
  (30B params, 3B active per token), `Qwen/Qwen3-235B-A22B`.
- **transformers source:** `transformers/models/qwen3_moe/modeling_qwen3_moe.py`

## 2. Decoder block diagram (MoE layer)

```
        x  [B, S, D]
        |
   +----+----+
   |         |
   |     pre_attn_norm  (RMSNorm STANDARD_W, eps=1e-6)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj   (split QKV, bias=attention_bias=False)
   |    ├── q_norm(Dh), k_norm(Dh)   (PER_HEAD_DH PRE-RoPE QK-norm)
   |    ├── RoPE (SPLIT_HALF, theta = config.rope_theta)
   |    ├── KVCache write/read       (CONTIGUOUS HND; mask = CAUSAL or SWA)
   |    ├── SDPA (GQA n_q=32, n_kv=4)
   |    └── o_proj                   (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm  (RMSNorm STANDARD_W)
   |         |
   |    MoE(channel_mixer) — mlp
   |    ├── gate.weight              [num_experts=128, D]   (no bias)
   |    ├── softmax(logits, dtype=fp32) → topk(top_k=8)
   |    ├── if norm_topk_prob: renormalize sum→1
   |    ├── experts.gate_up_proj     [128, 2*moe_I, D]
   |    ├── experts.down_proj        [128, D, moe_I]
   |    └── dense scatter (no shared experts)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, D]
```

For layers in `mlp_only_layers` (or when `(layer_idx+1) % decoder_sparse_step != 0`),
the MoE block is replaced with a dense Qwen3MoeMLP using
`intermediate_size = config.intermediate_size` — NOT `moe_intermediate_size`.
Source: `modeling_qwen3_moe.py:319`.

## 3. Op trace (api.ops sequence — MoE layer)

```
x_in       = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q          = linear(x_in, q_proj.weight)
k          = linear(x_in, k_proj.weight)
v          = linear(x_in, v_proj.weight)
q          = rms_norm_per_head(view(q, ..., H_q, Dh), q_norm.weight, eps, "standard_w")
k          = rms_norm_per_head(view(k, ..., H_kv, Dh), k_norm.weight, eps, "standard_w")
q, k       = rope_apply(q, k, cos, sin, basis="split_half")
a          = sdpa(q, k_full, v_full, attn_mask=mask, scale=Dh**-0.5)
attn_out   = linear(a, o_proj.weight)
x          = add(x, attn_out)
h          = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
router_logits = linear(h_flat, gate.weight)
router_probs  = softmax(router_logits, dtype=fp32, dim=-1)
top_w, top_idx = topk(router_probs, k=top_k, dim=-1)
if norm_topk_prob:
    top_w /= top_w.sum(dim=-1, keepdim=True)
top_w     = top_w.to(router_logits.dtype)
# Dense scatter:
y          = add(x, scatter_moe(h_flat, top_idx, top_w))
```

## 4. Spec instantiation (for Qwen3-30B-A3B, MoE layer)

Typical config: hidden=2048, n_q=32, n_kv=4, head_dim=128 (override),
intermediate=6144 (dense MLP, if any), moe_intermediate=768 (per expert),
num_experts=128, top_k=8, norm_topk_prob=True, vocab=151936.

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=4, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        sliding_window=None,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        qk_norm=NormSpec(RMS, 1e-6),
        qk_norm_phase=QKNormPhase.PRE_ROPE,
        qk_norm_shape=QKNormShape.PER_HEAD_DH,
        rope=RoPESpec(base_theta=10_000.0, basis=RoPEBasis.SPLIT_HALF),
    ),
    channel_mixer=MoESpec(
        n_experts=128, top_k=8,
        n_shared_experts=0,
        router_kind="softmax",
        router_norm=True,                # norm_topk_prob = True on 30B-A3B
        score_correction_bias=False,
        group_routing=None,
        routed_scaling_factor=1.0,
        expert_ffn=FFNSpec(intermediate_size=768, ...),
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
```

## 5. Quirks

- **`norm_topk_prob` is a config flag** with default False, but Qwen3-30B-A3B
  sets it to True. Drift to watch: a misread of the default would underweight
  experts. Source: `configuration_qwen3_moe.py:106`,
  `modeling_qwen3_moe.py:268-270`.
- **Per-layer dense vs MoE.** Default `decoder_sparse_step=1`,
  `mlp_only_layers=[]` → every layer is MoE. If a checkpoint specifies
  `mlp_only_layers=[0,1]`, those layers become dense MLP at
  `intermediate_size`. Source: modeling_qwen3_moe.py:314-319.
- **Dense MLP uses ``config.intermediate_size``, NOT ``moe_intermediate_size``.**
  Source: modeling_qwen3_moe.py:319.
- **QK-norm shape is PER_HEAD_DH, not FULL_HDH** — comment at
  modeling_qwen3_moe.py:152 (`"unlike olmo, only on the head dim!"`).
- **`sliding_window` is gated by `use_sliding_window`** (default False).
  When False, `__post_init__` overrides `sliding_window` to None
  (configuration_qwen3_moe.py:115).
- **No shared experts.** `Qwen3MoeSparseMoeBlock` (modeling_qwen3_moe.py:275-286)
  has no `shared_experts` module.
- **No group routing, no routed_scaling_factor.**
- **HF aliases ``num_experts`` → ``num_local_experts``**
  (`attribute_map` at configuration_qwen3_moe.py:51-53). The internal
  reference is ``config.num_experts``.
- **Router weight init is zero** (modeling_qwen3_moe.py:261 —
  ``torch.zeros(num_experts, hidden_dim)``), differing from Mixtral's
  ``torch.empty``. This affects only initialization, not architecture; weight
  loading from a trained checkpoint overrides it.
- **Output of router weights cast back to logit dtype.** Source:
  modeling_qwen3_moe.py:270 — ``router_top_value.to(router_logits.dtype)``.
  Our MoE keeps weights as fp32 inside the dispatch loop and casts at the
  expert multiply via ``.to(inner.dtype)``; this is equivalent in fp32.

## 6. Weight-name mapping (HF → API)

| HF tensor name | API tensor slot |
|---|---|
| `model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.layers.{L}.self_attn.q_proj.weight` | `blk.attention.q_proj.weight` |
| `model.layers.{L}.self_attn.k_proj.weight` | `blk.attention.k_proj.weight` |
| `model.layers.{L}.self_attn.v_proj.weight` | `blk.attention.v_proj.weight` |
| `model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `model.layers.{L}.self_attn.q_norm.weight` | `blk.attention.q_norm.weight` |
| `model.layers.{L}.self_attn.k_norm.weight` | `blk.attention.k_norm.weight` |
| `model.layers.{L}.mlp.gate.weight` (MoE) | `blk.feedforward.gate.weight` |
| `model.layers.{L}.mlp.experts.gate_up_proj` (MoE) | `blk.feedforward.experts_gate_up` |
| `model.layers.{L}.mlp.experts.down_proj` (MoE) | `blk.feedforward.experts_down` |
| `model.layers.{L}.mlp.gate_proj.weight` (dense) | `blk.feedforward.gate_proj.weight` |
| `model.layers.{L}.mlp.up_proj.weight` (dense) | `blk.feedforward.up_proj.weight` |
| `model.layers.{L}.mlp.down_proj.weight` (dense) | `blk.feedforward.down_proj.weight` |

## 7. Source citations

- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:126-195` Qwen3MoeAttention
- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:198-211` Qwen3MoeMLP (dense)
- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:215-251` Qwen3MoeExperts
- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:254-272` Qwen3MoeTopKRouter
- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:275-286` SparseMoeBlock
- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:310-353` DecoderLayer
- `transformers/models/qwen3_moe/configuration_qwen3_moe.py:25-120` defaults
