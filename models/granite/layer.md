# Granite 3.x — Decoder Layer

## 1. Identity

- **Family:** Granite (IBM)
- **Variants in B2a scope:** Granite-3.0-2B-Base, Granite-3.1-2B-Base (gate)
- **Out of B2a scope:** Granite-4.1-Tiny (GraniteMoEHybrid: MoE + Mamba — lands
  with MoE/SSM batch).
- **Release date:** 3.0 = Oct 2024; 3.1 = Dec 2024
- **HF base model card (gate):** `ibm-granite/granite-3.1-2b-base`
- **transformers source:**
  `transformers/src/transformers/models/granite/modeling_granite.py`

## 2. Decoder block diagram

```
        x  [B, S, D]
        |
   +----+----+
   |         |
   |     pre_attn_norm (RMSNorm STANDARD_W, eps=1e-5)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj   (split QKV, no bias)
   |    ├── RoPE (SPLIT_HALF, theta=5_000_000.0, default scaling)
   |    ├── KVCache write/read       (CONTIGUOUS HND)
   |    ├── SDPA (GQA, n_q=32, n_kv=8, scale = attention_multiplier)
   |    └── o_proj                   (no bias)
   |         |
   |         * residual_multiplier   ← μP residual scale (BEFORE add)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm (RMSNorm STANDARD_W)
   |         |
   |    feedforward(channel_mixer)
   |    ├── gate_proj                (SwiGLU gate, no bias)
   |    ├── up_proj                  (SwiGLU up,  no bias)
   |    ├── SiLU(gate) * up
   |    └── down_proj                (no bias)
   |         |
   |         * residual_multiplier   ← μP residual scale (BEFORE add)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, D]
```

Model-level μP scalars (NOT inside this block):
- `embedding_multiplier` scales `embed_tokens(input_ids)` BEFORE layer 0
  (modeling_granite.py:405).
- `logits_scaling` DIVIDES the final `lm_head(hidden)` output
  (modeling_granite.py:504).

## 3. Tensor IO trace

For Granite 3.1-2B-Base (D=2048, n_q=32, n_kv=8, head_dim=64, I=8192):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 2048] | bf16 |
| pre_attn_norm | x_in | [B, S, 2048] | bf16 |
| q_proj | q | [B, S, 32*64]=[B, S, 2048] | bf16 |
| k_proj | k | [B, S, 8*64]=[B, S, 512] | bf16 |
| v_proj | v | [B, S, 8*64]=[B, S, 512] | bf16 |
| reshape | q | [B, S, 32, 64] | bf16 |
| reshape | k, v | [B, S, 8, 64] | bf16 |
| RoPE | q, k | [B, S, *, 64] | bf16 |
| transpose | q | [B, 32, S, 64] | bf16 |
| transpose | k, v | [B, 8, S, 64] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 8, max_seq, 64] | bf16 |
| sdpa | a | [B, 32, S, 64] | bf16 (fp32 accum) |
| transpose+reshape | a | [B, S, 2048] | bf16 |
| o_proj | attn_out | [B, S, 2048] | bf16 |
| **mul residual_multiplier** | attn_out | [B, S, 2048] | bf16 |
| residual add | x | [B, S, 2048] | bf16 |
| pre_ffn_norm | h | [B, S, 2048] | bf16 |
| gate_proj | g | [B, S, 8192] | bf16 |
| up_proj | u | [B, S, 8192] | bf16 |
| silu+mul | h_act | [B, S, 8192] | bf16 |
| down_proj | ffn_out | [B, S, 2048] | bf16 |
| **mul residual_multiplier** | ffn_out | [B, S, 2048] | bf16 |
| residual add | y | [B, S, 2048] | bf16 |

## 4. Op trace (api.ops sequence)

```
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q       = linear(x_in, q_proj.weight)
k       = linear(x_in, k_proj.weight)
v       = linear(x_in, v_proj.weight)
q, k    = rope_apply(q, k, cos, sin, basis="split_half")
a       = sdpa(q, k_full, v_full, attn_mask=causal,
               scale=attention_multiplier)       # μP, NOT 1/sqrt(Dh)
attn_out = linear(a, o_proj.weight)
attn_out = attn_out * residual_multiplier        # μP residual scale
x       = add(x, attn_out)

h       = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
h_act   = mul(silu(g), u)
ffn_out = linear(h_act, down_proj.weight)
ffn_out = ffn_out * residual_multiplier          # μP residual scale
y       = add(x, ffn_out)
```

## 5. Spec instantiation (for Granite 3.1-2B-Base)

See `models/granite/config.py:GraniteConfig.to_block_spec()`. Concretely:

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=8, head_dim=64,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        attn_scale=0.015625,        # = attention_multiplier
        rope=RoPESpec(base_theta=5_000_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=8192,
        activation=Activation.SILU, gate_kind=GateKind.SWIGLU,
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
    residual_scale=0.22,           # = residual_multiplier
    embedding_scale=12.0,          # = embedding_multiplier (assembly metadata)
    logits_scale=8.0,              # = logits_scaling      (assembly metadata)
)
```

## 6. Quirks

- **μP scaling (the Granite distinguisher vs Llama)** — verified against
  `modeling_granite.py`:
  - `attention_multiplier` (line 124, `self.scaling = config.attention_multiplier`)
    REPLACES 1/sqrt(head_dim) in the softmax. For 3.1-2B this is 0.015625
    (= 1/64); the default 1/sqrt(64) ≈ 0.125 would be 8× too big.
  - `residual_multiplier` (lines 273, 278) scales BOTH the attention output and
    MLP output BEFORE the residual add. For 3.1-2B this is 0.22.
  - `embedding_multiplier` (line 405) scales `inputs_embeds`. For 3.1-2B this
    is 12.0. Carried on the BlockSpec for assembly; applied by the embedding
    layer when the model is built end-to-end.
  - `logits_scaling` (line 504) DIVIDES the final lm_head output. For 3.1-2B
    this is 8.0. Carried on the BlockSpec for assembly.
- **No QK-norm** — distinct from Qwen3 / Gemma 3 / 4 (modeling_granite.py
  GraniteAttention has only q/k/v/o projections).
- **`rope_theta=5_000_000`** for 3.1-2B (large because of the 131K context).
- **GQA n_q=32 / n_kv=8 / head_dim=64**; head_dim is hidden_size/n_q.
- **tie_word_embeddings=True** for 3.1-2B-Base.
- **rope_scaling=null** in the HF config (default RoPE, no Llama3 / LongRoPE).

## 7. Weight-name mapping (HF → API)

Granite uses Llama's tensor names verbatim — the μP scalars are scalar config
fields, not tensors, so the loader is line-for-line identical to Llama 3.

| HF tensor name (Granite) | API tensor slot |
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

Model-level (NOT consumed by the decoder block):
- `model.embed_tokens.weight` (scaled by embedding_multiplier at the embed layer)
- `model.norm.weight` (the final pre-lm_head RMSNorm)
- `lm_head.weight` (tied to embed_tokens for 3.1-2B; output divided by logits_scaling)

See `models/granite/layer.py::load_hf_granite_layer` for the loader.

## 8. Source citations

- HF reference: `transformers/models/granite/modeling_granite.py`
  - `GraniteAttention.__init__` line 118-139 (q/k/v/o, scale = attention_multiplier).
  - `GraniteMLP` line 203-216 (SwiGLU, optional bias via mlp_bias).
  - `GraniteDecoderLayer.forward` line 230-280 (residual_multiplier on
    attn_out and ffn_out before add).
  - `GraniteModel.forward` line 405 (`inputs_embeds * self.embedding_multiplier`).
  - `GraniteForCausalLM.forward` line 504 (`logits / self.config.logits_scaling`).
- HF config: `transformers/models/granite/configuration_granite.py`
  (fields: embedding_multiplier, logits_scaling, residual_multiplier, attention_multiplier).
- Census: `research/01-model-census.v3.md` (Granite section).
