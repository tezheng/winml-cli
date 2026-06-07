# SmolLM3-3B — Decoder Layer

## 1. Identity

- **Family:** SmolLM3 (HuggingFaceTB)
- **Variants in B1 scope:** SmolLM3-3B
- **Release date:** Jul 2025
- **HF base model card:** `HuggingFaceTB/SmolLM3-3B`
- **transformers source:** `transformers/models/smollm3/modeling_smollm3.py`

## 2. Decoder block diagram

```
        x  [B, S, 2048]
        |
   +----+----+
   |         |
   |     pre_attn_norm (RMSNorm STANDARD_W, eps=1e-6)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj   (split QKV, no bias)
   |    ├── (RoPE conditional — see §6)
   |    │      use_rope = no_rope_layers[layer_idx] == 1
   |    │      RoPE: SPLIT_HALF, theta=5_000_000.0
   |    ├── KVCache write/read       (CONTIGUOUS HND)
   |    ├── SDPA (GQA, n_q=16, n_kv=4, head_dim=128)
   |    │      CAUSAL mask (use_sliding_window=False in 3B)
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
        y  [B, S, 2048]
```

## 3. Tensor IO trace

For SmolLM3-3B (D=2048, n_q=16, n_kv=4, head_dim=128, I=11008):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 2048] | bf16 |
| pre_attn_norm | x_in | [B, S, 2048] | bf16 |
| q_proj | q | [B, S, 16*128]=[B, S, 2048] | bf16 |
| k_proj | k | [B, S, 4*128]=[B, S, 512] | bf16 |
| v_proj | v | [B, S, 4*128]=[B, S, 512] | bf16 |
| reshape | q | [B, S, 16, 128] | bf16 |
| reshape | k, v | [B, S, 4, 128] | bf16 |
| RoPE (if RoPE layer) | q, k | [B, S, *, 128] | bf16 |
| transpose | q | [B, 16, S, 128] | bf16 |
| transpose | k, v | [B, 4, S, 128] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 4, max_seq, 128] | bf16 |
| sdpa | a | [B, 16, S, 128] | bf16 |
| transpose+reshape | a | [B, S, 2048] | bf16 |
| o_proj | attn_out | [B, S, 2048] | bf16 |
| residual add | x | [B, S, 2048] | bf16 |
| pre_ffn_norm | h | [B, S, 2048] | bf16 |
| gate_proj | g | [B, S, 11008] | bf16 |
| up_proj | u | [B, S, 11008] | bf16 |
| silu+mul | h_act | [B, S, 11008] | bf16 |
| down_proj | ffn_out | [B, S, 2048] | bf16 |
| residual add | y | [B, S, 2048] | bf16 |

## 4. Op trace (api.ops sequence)

For a RoPE layer (`use_rope=True`):

```
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q       = linear(x_in, q_proj.weight)
k       = linear(x_in, k_proj.weight)
v       = linear(x_in, v_proj.weight)
q, k    = rope_apply(q, k, cos, sin, basis="split_half")    # NoPE layer: SKIPPED
# transpose + cache.write + cache.read + sdpa as Llama 3
attn_out = linear(a, o_proj.weight)
x       = add(x, attn_out)
h       = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
ffn_out = linear(mul(silu(g), u), down_proj.weight)
y       = add(x, ffn_out)
```

For a NoPE layer (`use_rope=False`), the `rope_apply` line is OMITTED — Q and
K go to SDPA un-rotated.

## 5. Spec instantiation

Per-layer, `Smollm3Config.to_block_spec(layer_idx)` returns a
`DecoderBlockSpec` whose `token_mixer.rope` is either an `RoPESpec` (RoPE
layer) or `None` (NoPE layer). All other fields are identical across the 36
layers.

```python
# RoPE layer (e.g. layer 0):
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=16, n_kv_heads=4, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=RoPESpec(base_theta=5_000_000.0, basis=RoPEBasis.SPLIT_HALF),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=11008,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
# NoPE layer (e.g. layer 3):  identical except token_mixer.rope = None
```

## 6. Quirks

- **Per-layer NoPE dispatch — the B1 IR-novel hurdle.** The HF config field
  is `no_rope_layers: list[int]` (one entry per decoder layer). A `1` means
  the layer uses RoPE; a `0` means NoPE. SmolLM3-3B disables RoPE on every
  4th layer (indices 3, 7, 11, ..., 35 — 9 NoPE layers / 27 RoPE layers
  out of 36). The default pattern is derived from `no_rope_layer_interval=4`
  if the explicit list is absent. Source: `configuration_smollm3.py:89-104`
  (post_init derivation), `modeling_smollm3.py:211` (`self.use_rope =
  config.no_rope_layers[layer_idx]`), `modeling_smollm3.py:233-235` (the
  `if self.use_rope:` guard around `apply_rotary_pos_emb`).

  In our IR this is modelled by allowing `AttentionSpec.rope` to be `None`.
  The `Attention.forward` short-circuits the RoPE step when `self.rope is
  None` (see `api/attention.py:139-140` after B1).

- **`rope_theta=5_000_000`** — distinct from Llama 3 (500K) and Mistral v0.3
  (1M). The high theta supports long-context tasks at the 64k window.
- **`head_dim` is derived** in SmolLM3-3B (not stored in HF config); the IR
  computes it as `hidden_size // num_attention_heads = 2048/16 = 128`.
- **`use_sliding_window=False`** in the 3B release; the `layer_types` field
  is filled with `"full_attention"` for all 36 layers. The IR honours the
  per-layer-type field if a future SmolLM3 turns SWA on.
- **No QK-norm. No biases. tie_word_embeddings=True.**

## 7. Weight-name mapping (HF → API)

Identical to Llama 3 (verified in `modeling_smollm3.py:298-307`):

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

NoPE layers carry the same tensor names — only the RoPE module is absent
from the block. The state-dict has no rope-cache entries (cos/sin are
non-persistent buffers).

## 8. Source citations

- HF reference: `transformers/models/smollm3/modeling_smollm3.py` (528 lines)
- HF config: `transformers/models/smollm3/configuration_smollm3.py`
- Census: `research/01-model-census.v3.md` §3.x (SmolLM3 entry)
