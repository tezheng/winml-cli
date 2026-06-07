# Mixtral 8x7B — Decoder Layer

## 1. Identity

- **Family:** Mixtral 8x7B (Mistral AI) — canonical softmax-routed MoE.
- **HF base model card:** `mistralai/Mixtral-8x7B-v0.1` (gated) /
  `mistralai/Mixtral-8x7B-Instruct-v0.1` (gated). Open mirrors at
  `unsloth/Mixtral-8x7B-Instruct-v0.1` (community-converted, BF16).
- **transformers source:** `transformers/models/mixtral/modeling_mixtral.py`

## 2. Decoder block diagram

```
        x  [B, S, 4096]
        |
   +----+----+
   |         |
   |     pre_attn_norm  (RMSNorm STANDARD_W, eps=1e-5)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj   (split QKV, no bias)
   |    ├── RoPE (SPLIT_HALF, theta=1_000_000)
   |    ├── KVCache write/read       (CONTIGUOUS HND, mask = CAUSAL)
   |    ├── SDPA (GQA n_q=32, n_kv=8)
   |    └── o_proj                   (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm  (RMSNorm STANDARD_W)
   |         |
   |    MoE(channel_mixer) — block_sparse_moe
   |    ├── gate.weight  (no bias) ── [n_experts=8, hidden]
   |    ├── softmax over experts → top-2 → ALWAYS renormalize sum→1
   |    ├── experts.gate_up_proj    [8, 2*14336, 4096]
   |    ├── experts.down_proj       [8, 4096, 14336]
   |    └── dense scatter: sum over routed experts (no shared experts)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, 4096]
```

## 3. Tensor IO trace

For Mixtral 8x7B (D=4096, n_q=32, n_kv=8, head_dim=128, I_expert=14336, E=8, top_k=2):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 4096] | bf16 |
| pre_attn_norm | x_in | [B, S, 4096] | bf16 |
| q_proj | q | [B, S, 4096] | bf16 |
| k_proj | k | [B, S, 1024] | bf16 |
| v_proj | v | [B, S, 1024] | bf16 |
| RoPE | q, k | [B, *, S, 128] | bf16 |
| cache.write | (state) | KVCache k/v [B, 8, max_seq, 128] | bf16 |
| sdpa | a | [B, 32, S, 128] | bf16 |
| o_proj | attn_out | [B, S, 4096] | bf16 |
| residual add | x | [B, S, 4096] | bf16 |
| pre_ffn_norm | h | [B, S, 4096] | bf16 |
| gate (linear) | router_logits | [B*S, 8] | bf16 |
| softmax (fp32) | router_probs | [B*S, 8] | fp32 |
| topk → renorm | top_w, top_idx | [B*S, 2] | fp32 |
| MoE scatter (per-expert) | inner | [N_e, 4096] | bf16 |
| residual add | y | [B, S, 4096] | bf16 |

## 4. Op trace (api.ops sequence)

```
x_in       = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q, k, v    = linear(x_in, {q,k,v}_proj.weight)
q, k       = rope_apply(q, k, cos, sin, basis="split_half")
a          = sdpa(q, k_full, v_full, attn_mask=mask, scale=head_dim**-0.5)
attn_out   = linear(a, o_proj.weight)
x          = add(x, attn_out)
h          = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
router_logits = linear(h_flat, gate.weight)
router_probs  = softmax(router_logits.float(), dim=-1)
top_w, top_idx = topk(router_probs, k=2, dim=-1)
top_w     /= top_w.sum(dim=-1, keepdim=True)               # always-on renorm
# Dense scatter: for each expert e selected by ≥1 token:
#   inner = silu(linear(x_e, gate_rows)) * linear(x_e, up_rows)
#   out_e = linear(inner, down_proj[e])
#   accumulator.index_add_(0, tok_idx, out_e * top_w[tok_idx, slot])
y          = add(x, accumulator)
```

## 5. Spec instantiation (for Mixtral 8x7B)

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=8, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        sliding_window=None,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=RoPESpec(base_theta=1_000_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=MoESpec(
        n_experts=8, top_k=2,
        n_shared_experts=0,
        router_kind="softmax",
        router_norm=True,                # ALWAYS-on per HF source
        score_correction_bias=False,
        group_routing=None,
        routed_scaling_factor=1.0,
        expert_ffn=FFNSpec(intermediate_size=14336,
                           activation=Activation.SILU,
                           gate_kind=GateKind.SWIGLU),
    ),
    pre_attn_norm=NormSpec(RMS, 1e-5), pre_ffn_norm=NormSpec(RMS, 1e-5),
)
```

## 6. Quirks

- **`router_norm` is always True for Mixtral.** Unlike Qwen3-MoE / OLMoE,
  there is NO `norm_topk_prob` config flag — HF unconditionally executes
  `router_top_value /= router_top_value.sum(dim=-1, keepdim=True)` at
  `modeling_mixtral.py:114`. Drift to watch for: if a downstream Mixtral
  derivative ever introduces such a flag, this assumption must change.
- **`routed_scaling_factor = 1.0`.** Mixtral does NOT scale the routed
  output (compare DeepSeek-V3 which uses 2.5).
- **No shared experts.** `MixtralSparseMoeBlock` (modeling_mixtral.py:119-135)
  has no `shared_experts` module. Set `n_shared_experts = 0`.
- **No group routing.**
- **No QK-norm.** Mixtral attention init (modeling_mixtral.py:298-311) has
  no `q_norm` / `k_norm`.
- **GQA with `head_dim = hidden_size // num_attention_heads`** unless
  `head_dim` is set explicitly. Mixtral 8x7B uses 4096/32 = 128.
- **`rope_theta = 1_000_000`** (configuration_mixtral.py:44 `default_theta`).
- **No biases anywhere** (q/k/v/o, gate, experts).
- **HF stores experts packed.** `MixtralExperts.gate_up_proj` is a single
  Parameter of shape `[E, 2I, H]`, sliced per-expert at forward time
  (modeling_mixtral.py:70-72, 92-94). Our `MoE.experts_gate_up` and
  `experts_down` match this layout exactly.
- **`sliding_window` defaults to None** on Mixtral 8x7B (config L77).
  Earlier Mistral-family used SWA; Mixtral does not.
- **Router math runs in fp32 starting at softmax** —
  `F.softmax(router_logits.float(), dim=-1)` (modeling_mixtral.py:112).
  Our `MoE._route_softmax` casts to fp32 one step earlier (at the linear).
  In FP32 mode these are identical; in BF16 mode our path is slightly
  more precise, but compatible.

## 7. Weight-name mapping (HF → API)

| HF tensor name | API tensor slot |
|---|---|
| `model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.layers.{L}.self_attn.q_proj.weight` | `blk.attention.q_proj.weight` |
| `model.layers.{L}.self_attn.k_proj.weight` | `blk.attention.k_proj.weight` |
| `model.layers.{L}.self_attn.v_proj.weight` | `blk.attention.v_proj.weight` |
| `model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `model.layers.{L}.block_sparse_moe.gate.weight` | `blk.feedforward.gate.weight` |
| `model.layers.{L}.block_sparse_moe.experts.gate_up_proj` | `blk.feedforward.experts_gate_up` |
| `model.layers.{L}.block_sparse_moe.experts.down_proj` | `blk.feedforward.experts_down` |

Older checkpoints with per-expert linears (`experts.{i}.w1.weight` etc.)
are unpacked into the same packed tensors on load (gate from `w1`, up from
`w3`, down from `w2`). See `models/mixtral/layer.py::load_hf_mixtral_layer`.

## 8. Source citations

- `transformers/models/mixtral/modeling_mixtral.py:62-98` MixtralExperts
- `transformers/models/mixtral/modeling_mixtral.py:101-116` MixtralTopKRouter
- `transformers/models/mixtral/modeling_mixtral.py:119-135` MixtralSparseMoeBlock
- `transformers/models/mixtral/modeling_mixtral.py:295-351` MixtralAttention
- `transformers/models/mixtral/modeling_mixtral.py:354-389` MixtralDecoderLayer
- `transformers/models/mixtral/configuration_mixtral.py:25-93` MixtralConfig
- Census: `research/01-model-census.v3.md` (Mixtral entry)
