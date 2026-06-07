# Mistral 7B — Decoder Layer

## 1. Identity

- **Family:** Mistral 7B (Mistral AI)
- **Variants in B1 scope:** v0.3 (gate variant). v0.1 / v0.2 are layer-IR-compatible.
- **Release date:** v0.1 Sep 2023; v0.2 Mar 2024; v0.3 May 2024.
- **HF base model card:** `mistralai/Mistral-7B-v0.3`
- **transformers source:** `transformers/models/mistral/modeling_mistral.py`

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
   |    ├── RoPE (SPLIT_HALF, theta=1_000_000.0 [v0.3] / 10000 [v0.1])
   |    ├── KVCache write/read       (CONTIGUOUS HND)
   |    ├── SDPA (GQA, n_q=32, n_kv=8)
   |    │       — v0.3: CAUSAL mask only (sliding_window=None)
   |    │       — v0.1: SWA mask (sliding_window=4096)
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
        y  [B, S, 4096]
```

## 3. Tensor IO trace

For Mistral 7B v0.3 (D=4096, n_q=32, n_kv=8, head_dim=128, I=14336):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 4096] | bf16 |
| pre_attn_norm | x_in | [B, S, 4096] | bf16 |
| q_proj | q | [B, S, 32*128]=[B, S, 4096] | bf16 |
| k_proj | k | [B, S, 8*128]=[B, S, 1024] | bf16 |
| v_proj | v | [B, S, 8*128]=[B, S, 1024] | bf16 |
| reshape+transpose | q,k,v | [B,H,S,Dh] | bf16 |
| RoPE | q, k | [B, *, S, 128] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 8, max_seq, 128] | bf16 |
| sdpa | a | [B, 32, S, 128] | bf16 |
| transpose+reshape | a | [B, S, 4096] | bf16 |
| o_proj | attn_out | [B, S, 4096] | bf16 |
| residual add | x | [B, S, 4096] | bf16 |
| pre_ffn_norm | h | [B, S, 4096] | bf16 |
| gate_proj | g | [B, S, 14336] | bf16 |
| up_proj | u | [B, S, 14336] | bf16 |
| silu+mul | h_act | [B, S, 14336] | bf16 |
| down_proj | ffn_out | [B, S, 4096] | bf16 |
| residual add | y | [B, S, 4096] | bf16 |

## 4. Op trace (api.ops sequence)

Identical to Llama 3:

```
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q       = linear(x_in, q_proj.weight)
k       = linear(x_in, k_proj.weight)
v       = linear(x_in, v_proj.weight)
q, k    = rope_apply(q, k, cos, sin, basis="split_half")
a       = sdpa(q, k_full, v_full, attn_mask=mask, scale=head_dim**-0.5)
attn_out = linear(a, o_proj.weight)
x       = add(x, attn_out)
h       = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
ffn_out = linear(mul(silu(g), u), down_proj.weight)
y       = add(x, ffn_out)
```

## 5. Spec instantiation (for Mistral 7B v0.3)

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=8, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,        # v0.3 dropped SWA
        sliding_window=None,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=RoPESpec(base_theta=1_000_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=14336,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
```

## 6. Quirks

- **v0.3 dropped sliding-window attention.** The HF config has
  `sliding_window: null` (verified at
  `hf_cache/hub/models--mistralai--Mistral-7B-v0.3/snapshots/*/config.json`).
  v0.1 and v0.2 still set `sliding_window=4096`. The IR honours both shapes:
  when sliding_window is set, the block dispatches a SWA mask; otherwise plain
  causal. Source check: `modeling_mistral.py:172` —
  `sliding_window=getattr(self.config, "sliding_window", None)` — `None` is
  passed straight to the attention interface which interprets it as "no SWA".
- **`rope_theta=1_000_000`** in v0.3 (raised from v0.1's `10000` — both are
  documented officially in the Mistral release notes).
- **No QK-norm.** Like Llama 3, Mistral has no q_norm/k_norm modules.
- **No biases.** All projections are bias-free (hard-coded `bias=False` in
  MistralAttention and MistralMLP — `modeling_mistral.py:134-137, 41-43`).
- **GQA.** All v0.x ≥ v0.2 use n_q=32, n_kv=8, head_dim=128.

## 7. Weight-name mapping (HF → API)

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

See `models/mistral/layer.py::load_hf_mistral_layer`.

## 8. Source citations

- HF reference: `transformers/models/mistral/modeling_mistral.py:MistralDecoderLayer`
- HF config: `transformers/models/mistral/configuration_mistral.py:MistralConfig`
- Census: `research/01-model-census.v3.md` §3.2 (Mistral entry)
