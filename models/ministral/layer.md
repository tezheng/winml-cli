# Ministral 8B (and Ministral-3-3B) — Decoder Layer

## 0. At-a-glance

- **Signature:** Mistral backbone + interleaved SWA (1:3 full:sliding pattern per layer_types) + SwiGLU + RMSNorm; θ=100M
- **Active params:** 3B / 8B total
- **Layer mix:** 36 attn (8B, GQA-32:8), 1 full + 3 sliding alternating (1:3 per 4-layer period)
- **KV cache / token (bf16):** 8B: 36 L × 8 KV × 128 Dh × 2 × 2 = 144 kB

---

## 1. Identity

- **Family:** Ministral (Mistral AI, 2024 release "les Ministraux")
- **Variants in B4 scope:** `mistralai/Ministral-8B-Instruct-2410`
  (canonical, with interleaved SWA), `mistralai/Ministral-3-3B-Instruct-2512`
  (3B multimodal release, no SWA)
- **Release date:** Oct 2024 (8B), Dec 2025 (3B-2512)
- **HF base model card:** `mistralai/Ministral-8B-Instruct-2410`
- **transformers source:** `transformers/models/ministral/modeling_ministral.py`

## 2. Decoder block diagram

```
        x  [B, S, 4096]
        |
   +----+----+
   |         |
   |     pre_attn_norm (RMSNorm STANDARD_W, eps=1e-5)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj   (split QKV, no bias)
   |    ├── RoPE SPLIT_HALF, theta=100_000_000.0
   |    ├── KVCache write/read       (CONTIGUOUS HND)
   |    ├── SDPA (GQA, n_q=32, n_kv=8, head_dim=128)
   |    │      mask = SWA if layer_types[i]=="sliding_attention"
   |    │             else CAUSAL
   |    │      sliding_window = 32768 on sliding layers, None on full
   |    └── o_proj                   (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm (RMSNorm STANDARD_W)
   |         |
   |    feedforward(channel_mixer)
   |    ├── gate_proj                (SwiGLU gate)
   |    ├── up_proj                  (SwiGLU up)
   |    ├── SiLU(gate) * up
   |    └── down_proj
   |         |
   +---->add (residual 2)
        |
        y  [B, S, 4096]
```

## 3. Tensor IO trace

For Ministral-8B (D=4096, n_q=32, n_kv=8, head_dim=128, I=12288):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 4096] | bf16 |
| pre_attn_norm | x_in | [B, S, 4096] | bf16 |
| q_proj | q | [B, S, 32*128]=[B, S, 4096] | bf16 |
| k_proj | k | [B, S, 8*128]=[B, S, 1024] | bf16 |
| v_proj | v | [B, S, 8*128]=[B, S, 1024] | bf16 |
| reshape | q | [B, S, 32, 128] | bf16 |
| reshape | k, v | [B, S, 8, 128] | bf16 |
| RoPE | q, k | [B, S, *, 128] | bf16 |
| transpose | q | [B, 32, S, 128] | bf16 |
| transpose | k, v | [B, 8, S, 128] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 8, max_seq, 128] | bf16 |
| sdpa | a | [B, 32, S, 128] | bf16 |
| transpose+reshape | a | [B, S, 4096] | bf16 |
| o_proj | attn_out | [B, S, 4096] | bf16 |
| residual add | x | [B, S, 4096] | bf16 |
| pre_ffn_norm | h | [B, S, 4096] | bf16 |
| gate_proj | g | [B, S, 12288] | bf16 |
| up_proj | u | [B, S, 12288] | bf16 |
| silu+mul | h_act | [B, S, 12288] | bf16 |
| down_proj | ffn_out | [B, S, 4096] | bf16 |
| residual add | y | [B, S, 4096] | bf16 |

## 4. Op trace (api.ops sequence)

```
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q       = linear(x_in, q_proj.weight)
k       = linear(x_in, k_proj.weight)
v       = linear(x_in, v_proj.weight)
q, k    = rope_apply(q, k, cos, sin, basis="split_half")
# transpose + cache.write + cache.read
attn_out = sdpa(q, k, v, mask=causal_or_swa, scale=1/sqrt(128))
attn_out = linear(attn_out, o_proj.weight)
x       = add(x, attn_out)
h       = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
ffn_out = linear(mul(silu(g), u), down_proj.weight)
y       = add(x, ffn_out)
```

The full and sliding paths differ ONLY in the SDPA mask kind and the
`sliding_window` field. All other ops are identical and the weight tensors
are identical.

## 5. Spec instantiation

Per-layer, `MinistralConfig.to_block_spec(layer_idx)` returns a
`DecoderBlockSpec`. For ``Ministral-8B-Instruct-2410`` with its
`layer_types` field equal to `[full, sliding, sliding, sliding] * 9`:

```python
# Layer 0 (full):
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=8, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        sliding_window=None,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=RoPESpec(base_theta=100_000_000.0, basis=RoPEBasis.SPLIT_HALF),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=12288,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
# Layer 1 (sliding):  identical except mask_kind=SWA, sliding_window=32768
```

## 6. Quirks

- **Per-layer interleaved SWA — the 3rd alternation pattern.** Gemma 2 had
  1:1 sliding/full alternation, Gemma 3 has 5:1 sliding-then-full,
  Ministral-8B has **1:3 full-then-sliding** (one full layer followed by
  three sliding layers, repeated). Source for the dispatch:
  `modeling_ministral.py:142` (`self.layer_type = config.layer_types[
  layer_idx]`), `modeling_ministral.py:155` (`self.sliding_window =
  config.sliding_window if self.layer_type == "sliding_attention" else
  None`), `modeling_ministral.py:411-414` (the model.forward loop dispatching
  the mask by `causal_mask_mapping[layer_types[i]]`).

  The published `Ministral-8B-Instruct-2410` declares 36 layers with the
  pattern `["full", "sliding", "sliding", "sliding"] * 9`. The IR carries
  this tuple verbatim in `MinistralConfig.layer_types` and the per-layer
  block spec selects `mask_kind` accordingly.

- **`sliding_window=32768` equals `max_position_embeddings=32768` on the
  8B-2410.** Within the published context window, SWA and CAUSAL produce
  identical masks — the per-layer dispatch is observable only at S > W
  during inference. Numerical equivalence vs HF at S ≤ 32k therefore holds
  on full layers and sliding layers alike (and we test both).

- **HF model loading caveat.** Although the package supports per-layer
  dispatch, the published `Ministral-8B-Instruct-2410` advertises
  `architectures: ["MistralForCausalLM"]` and `model_type: "mistral"`.
  AutoModelForCausalLM therefore loads it via Mistral modeling, which
  applies ONE global mask (per `modeling_mistral.py:372`) and IGNORES the
  per-layer pattern. The IR per-layer dispatch only takes effect when the
  user explicitly constructs a MinistralForCausalLM (or this package's
  `build_ministral_decoder_layer`).

- **rope_theta=100_000_000.0** — two orders of magnitude above Mistral v0.3's
  1_000_000.0. Supports the published 32k context.

- **head_dim=128 is explicit** in the published config (also equals
  hidden_size/num_heads = 4096/32). `MinistralConfig.head_dim` honours the
  explicit value when present and falls back to the derivation otherwise.

- **rms_norm_eps=1e-5** in the published config, even though
  configuration_ministral.py:76 defaults to 1e-6. The IR reads it from the
  HF dict and does not assume the default.

- **No QK-norm. No biases. tie_word_embeddings=False.**

## 7. Weight-name mapping (HF → API)

Identical to Mistral/Llama 3 (verified at `modeling_ministral.py:220-260`,
the `MinistralDecoderLayer` is structurally identical to MistralDecoderLayer
modulo the per-layer attention-side fields).

| HF tensor name | API tensor slot |
|---|---|
| `model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.layers.{L}.self_attn.q_proj.weight` | `blk.attention.q_proj.weight` |
| `model.layers.{L}.self_attn.k_proj.weight` | `blk.attention.k_proj.weight` |
| `model.layers.{L}.self_attn.v_proj.weight` | `blk.attention.v_proj.weight` |
| `model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.layers.{L}.mlp.gate_proj.weight` | `blk.feedforward.gate_proj.weight` |
| `model.layers.{L}.mlp.up_proj.weight` | `blk.feedforward.up_proj.weight` |
| `model.layers.{L}.mlp.down_proj.weight` | `blk.feedforward.down_proj.weight` |

The per-layer dispatch (full vs sliding) is purely runtime configuration
on the AttentionSpec; the state-dict tensors are layer-type agnostic.

## 8. Source citations

- HF reference (Ministral): `transformers/models/ministral/modeling_ministral.py` (523 lines)
  - `modular_ministral.py:46-77` (MinistralConfig.__post_init__ layer_types default)
  - `modeling_ministral.py:137-196` (MinistralAttention with per-layer
    layer_type and sliding_window)
  - `modeling_ministral.py:411-414` (per-layer mask dispatch in MinistralModel.forward)
- HF reference (Mistral, for the global-mask divergence):
  `transformers/models/mistral/modeling_mistral.py:372-393`
- Census: `research/01-model-census.v3.md`
