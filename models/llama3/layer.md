# Llama 3 / 3.1 / 3.2 — Decoder Layer

## 1. Identity

- **Family:** Llama 3 (Meta)
- **Variants in B1 scope:** 3.0 8B, 3.1 8B, 3.2 1B, 3.2 3B (all dense)
- **Release date:** 3.0 = Apr 2024; 3.1 = Jul 2024; 3.2 = Sep 2024
- **HF base model cards:** `meta-llama/Meta-Llama-3-8B`, `meta-llama/Meta-Llama-3.1-8B`,
  `unsloth/Llama-3.2-1B-Instruct` (B1 gate), `unsloth/Llama-3.2-3B-Instruct`
- **transformers source:**
  `transformers/src/transformers/models/llama/modeling_llama.py`

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
   |    ├── RoPE (SPLIT_HALF, theta=500_000.0,
   |    │   3.0: rope_type=default ; 3.1/3.2: rope_type=llama3 smooth scaling)
   |    ├── KVCache write/read       (CONTIGUOUS HND)
   |    ├── SDPA (GQA, n_q=…, n_kv=…)
   |    └── o_proj                   (no bias)
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
   +---->add (residual 2)
        |
        y  [B, S, D]
```

## 3. Tensor IO trace

For Llama 3.2 1B (D=2048, n_q=32, n_kv=8, head_dim=64, I=8192):

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
| residual add | x | [B, S, 2048] | bf16 |
| pre_ffn_norm | h | [B, S, 2048] | bf16 |
| gate_proj | g | [B, S, 8192] | bf16 |
| up_proj | u | [B, S, 8192] | bf16 |
| silu+mul | h_act | [B, S, 8192] | bf16 |
| down_proj | ffn_out | [B, S, 2048] | bf16 |
| residual add | y | [B, S, 2048] | bf16 |

## 4. Op trace (api.ops sequence)

```
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q       = linear(x_in, q_proj.weight)         # reshape to [B, S, Hq, Dh]
k       = linear(x_in, k_proj.weight)         # reshape to [B, S, Hk, Dh]
v       = linear(x_in, v_proj.weight)         # reshape to [B, S, Hk, Dh]
q, k    = rope_apply(q, k, cos, sin, basis="split_half")
# cache.write(k, v, start_pos); k_full, v_full = cache.read(start_pos + S)
a       = sdpa(q, k_full, v_full, attn_mask=causal, scale=head_dim ** -0.5)
attn_out = linear(a, o_proj.weight)
x       = add(x, attn_out)

h       = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
h_act   = mul(silu(g), u)
ffn_out = linear(h_act, down_proj.weight)
y       = add(x, ffn_out)
```

## 5. Spec instantiation (for Llama 3.2 1B)

See `models/llama3/config.py:Llama3Config.to_block_spec()`. Concretely:

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=8, head_dim=64,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=RoPESpec(
            base_theta=500_000.0,
            basis=RoPEBasis.SPLIT_HALF,
            scaling=RoPEScaling.LLAMA3,
            llama3_extra=Llama3RoPEParams(
                factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0,
                original_context_length=8192,
            ),
        ),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=8192,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
```

## 6. Quirks

- **NO QK-norm** — distinct from Qwen3 and Gemma 3 / 4. `LlamaAttention.__init__`
  (`modeling_llama.py:225-249`) has only q/k/v/o projections, no q_norm/k_norm.
- **RoPE scaling factor differs across 3.x:**
  - 3.0: `rope_type="default"`, factor unused.
  - 3.1 8B: `rope_type="llama3"`, factor=**8**, low=1, high=4, ctx=8192.
  - 3.2 1B/3B: `rope_type="llama3"`, factor=**32**, low=1, high=4, ctx=8192.
  Source: `modeling_rope_utils.py::_compute_llama3_parameters` (line 550-626).
- **`rope_theta=500_000`** across all 3.x variants — different from Qwen3 (1M)
  and Mistral v0.3 (1M).
- **`head_dim=64`** for 3.2 1B (not 64 = hidden/n_q = 2048/32 in this case;
  the explicit field happens to coincide). For 3.0/3.1 8B, head_dim=128.
- **No biases anywhere** (attention_bias=False, mlp_bias=False).
- **tie_word_embeddings:** True for 3.2 1B/3B (verified in HF config.json);
  False for 3.0/3.1 8B.
- **rope_parameters dict** in transformers 5.x consolidates what was `rope_theta`
  + `rope_scaling` in 4.x; `from_hf_dict` handles both shapes.

## 7. Weight-name mapping (HF → API)

| HF tensor name (Llama 3) | API tensor slot |
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

See `models/llama3/layer.py::load_hf_llama3_layer` for the loader.

## 8. Source citations

- HF reference: `transformers/models/llama/modeling_llama.py:LlamaDecoderLayer`
- HF config: `transformers/models/llama/configuration_llama.py:LlamaConfig`
- HF RoPE: `transformers/modeling_rope_utils.py::_compute_llama3_parameters`
- Survey: `research/02-layer-sources.v3.md` (Llama 3 section)
- Census: `research/01-model-census.v3.md` §3.1
