# OLMoE 1B-7B — Decoder Layer

## 1. Identity

- **Family:** OLMoE 1B-7B (Allen Institute for AI) — 7B params total, 1B active.
- **HF model card:** `allenai/OLMoE-1B-7B-0924` (open weights, FP32 checkpoints).
- **transformers source:** `transformers/models/olmoe/modeling_olmoe.py`

## 2. Decoder block diagram

```
        x  [B, S, D=2048]
        |
   +----+----+
   |         |
   |     pre_attn_norm  (RMSNorm STANDARD_W, eps=1e-5)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj   (split QKV, no bias)
   |    ├── q_norm(H_q*Dh), k_norm(H_kv*Dh)   (FULL_HDH PRE-RoPE QK-norm)
   |    ├── [optional clip_qkv clamp — None on public checkpoints]
   |    ├── RoPE (SPLIT_HALF, theta = config.rope_theta)
   |    ├── KVCache write/read       (CONTIGUOUS HND, CAUSAL mask)
   |    ├── SDPA (MHA n_q=16, n_kv=16)
   |    └── o_proj                   (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm  (RMSNorm STANDARD_W)
   |         |
   |    MoE(channel_mixer) — mlp
   |    ├── gate.weight              [num_experts=64, D]   (no bias)
   |    ├── softmax(logits, dtype=fp32) → topk(top_k=8)
   |    ├── if norm_topk_prob: renormalize sum→1
   |    ├── experts.gate_up_proj     [64, 2*I, D]
   |    ├── experts.down_proj        [64, D, I]
   |    └── dense scatter (no shared experts)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, D]
```

## 3. Op trace

Identical to Qwen3-MoE in the FFN half; identical to OLMo 2 in the QK-norm
shape (FULL_HDH) but with PRE-norm envelope instead of POST-norm.

```
x_in       = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q          = linear(x_in, q_proj.weight)
k          = linear(x_in, k_proj.weight)
v          = linear(x_in, v_proj.weight)
q          = rms_norm_full_hdh(q, q_norm.weight, eps)     # FULL_HDH norm
k          = rms_norm_full_hdh(k, k_norm.weight, eps)     # FULL_HDH norm
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
y          = add(x, scatter_moe(h_flat, top_idx, top_w))
```

## 4. Spec instantiation (for OLMoE-1B-7B-0924)

Typical config: hidden=2048, n_q=16, n_kv=16, head_dim=128, intermediate=1024,
num_experts=64, top_k=8, norm_topk_prob=False, vocab=50304.

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=16, n_kv_heads=16, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        qk_norm=NormSpec(RMS, 1e-5),
        qk_norm_phase=QKNormPhase.PRE_ROPE,
        qk_norm_shape=QKNormShape.FULL_HDH,
        rope=RoPESpec(base_theta=10_000.0, basis=RoPEBasis.SPLIT_HALF),
    ),
    channel_mixer=MoESpec(
        n_experts=64, top_k=8,
        n_shared_experts=0,
        router_kind="softmax",
        router_norm=False,                # OLMoE-0924 ships False
        score_correction_bias=False,
        group_routing=None,
        routed_scaling_factor=1.0,
        expert_ffn=FFNSpec(intermediate_size=1024, ...),
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
```

## 5. Quirks

- **FULL_HDH QK-norm shape** — distinct from Qwen3-MoE's PER_HEAD_DH.
  Source: modeling_olmoe.py:246-249. q_norm weight shape is `[hidden_size]`,
  k_norm weight shape is `[(hidden_size / num_attention_heads) * num_key_value_heads]`.
- **QK-norm is applied to the FLAT tensor BEFORE view+transpose+clip+RoPE.**
  Source: modeling_olmoe.py:262-275.
- **`clip_qkv` is a runtime clamp on Q/K/V after norms.** The public
  OLMoE-1B-7B-0924 checkpoint has `clip_qkv = None` — we assert this in
  ``to_block_spec`` and raise NotImplementedError if non-None.
- **PRE-norm envelope** (vs OLMo 2's POST-norm).
- **No SWA.** Source: modeling_olmoe.py:490 (comment: "diff with mixtral:
  no sliding").
- **No shared experts.**
- **No group routing, no routed_scaling_factor.**
- **No per-layer dense fallback** — every layer is MoE (vs Qwen3-MoE which
  supports `decoder_sparse_step` / `mlp_only_layers`).
- **`tie_word_embeddings = False`** (configuration_olmoe.py:72).
- **Vocab 50304, pad_token_id = 1, eos_token_id = 50279.**
- **`norm_topk_prob` flag with default False** — OLMoE-1B-7B-0924 keeps
  this False.

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
| `model.layers.{L}.mlp.gate.weight` | `blk.feedforward.gate.weight` |
| `model.layers.{L}.mlp.experts.gate_up_proj` | `blk.feedforward.experts_gate_up` |
| `model.layers.{L}.mlp.experts.down_proj` | `blk.feedforward.experts_down` |

## 7. Source citations

- `transformers/models/olmoe/modeling_olmoe.py:220-298` OlmoeAttention
- `transformers/models/olmoe/modeling_olmoe.py:301-339` OlmoeExperts
- `transformers/models/olmoe/modeling_olmoe.py:341-359` OlmoeTopKRouter
- `transformers/models/olmoe/modeling_olmoe.py:362-375` SparseMoeBlock
- `transformers/models/olmoe/modeling_olmoe.py:378-416` DecoderLayer
- `transformers/models/olmoe/configuration_olmoe.py:23-89` defaults
