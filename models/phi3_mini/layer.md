# Phi-3 mini — Decoder Layer

## 1. Identity

- **Family:** Phi-3 (Microsoft)
- **Variant in B2a scope:** Phi-3-mini-4k-instruct (3.8B)
- **Release date:** Apr 2024
- **HF base model card:** `microsoft/Phi-3-mini-4k-instruct`
- **transformers source:**
  `transformers/src/transformers/models/phi3/modeling_phi3.py`

## 2. Decoder block diagram

```
        x  [B, S, D]
        |
   +----+----+
   |         |
   |     pre_attn_norm (RMSNorm STANDARD_W, eps=1e-5)
   |         |
   |    attention(token_mixer)
   |    ├── qkv_proj                 ← FUSED, output (Hq+2*Hk)*Dh
   |    │   then slice (Q | K | V) along output dim
   |    ├── RoPE (SPLIT_HALF, θ=10000.0, default scaling, full rotation)
   |    ├── KVCache write/read       (CONTIGUOUS HND)
   |    ├── SDPA (MHA, n_q=32, n_kv=32; SWA window=2047)
   |    └── o_proj                   (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm (RMSNorm STANDARD_W)
   |         |
   |    feedforward(channel_mixer)
   |    ├── gate_up_proj             ← FUSED, output 2*I
   |    │   then chunk(2, dim=-1) → (gate, up)
   |    ├── SiLU(gate) * up
   |    └── down_proj                (no bias)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, D]
```

## 3. Tensor IO trace

For Phi-3-mini-4k-instruct (D=3072, Hq=Hk=32, head_dim=96, I=8192):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 3072] | bf16 |
| pre_attn_norm | x_in | [B, S, 3072] | bf16 |
| qkv_proj | qkv | [B, S, (32+64)*96]=[B, S, 9216] | bf16 |
| slice Q | q | [B, S, 32, 96] | bf16 |
| slice K | k | [B, S, 32, 96] | bf16 |
| slice V | v | [B, S, 32, 96] | bf16 |
| RoPE | q, k | [B, S, *, 96] | bf16 |
| transpose | q, k, v | [B, 32, S, 96] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 32, max_seq, 96] | bf16 |
| sdpa (SWA mask, W=2047) | a | [B, 32, S, 96] | bf16 (fp32 accum) |
| transpose+reshape | a | [B, S, 3072] | bf16 |
| o_proj | attn_out | [B, S, 3072] | bf16 |
| residual add | x | [B, S, 3072] | bf16 |
| pre_ffn_norm | h | [B, S, 3072] | bf16 |
| gate_up_proj | up_states | [B, S, 16384] | bf16 |
| chunk(2, -1) | gate, up | [B, S, 8192] each | bf16 |
| SiLU(gate) * up | h_act | [B, S, 8192] | bf16 |
| down_proj | ffn_out | [B, S, 3072] | bf16 |
| residual add | y | [B, S, 3072] | bf16 |

## 4. Op trace (api.ops sequence)

```
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
qkv     = linear(x_in, qkv_proj.weight)
q       = qkv[..., :Hq*Dh].view(B, S, Hq, Dh)
k       = qkv[..., Hq*Dh : (Hq+Hk)*Dh].view(B, S, Hk, Dh)
v       = qkv[..., (Hq+Hk)*Dh:].view(B, S, Hk, Dh)
q, k    = rope_apply(q, k, cos, sin, basis="split_half")
a       = sdpa(q, k_full, v_full, attn_mask=SWA(W=2047), scale=head_dim ** -0.5)
attn_out = linear(a, o_proj.weight)
x       = add(x, attn_out)

h       = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
up_states = linear(h, gate_up_proj.weight)
gate, up = up_states.chunk(2, dim=-1)
h_act   = mul(silu(gate), up)
ffn_out = linear(h_act, down_proj.weight)
y       = add(x, ffn_out)
```

## 5. Spec instantiation (for Phi-3-mini-4k-instruct)

See `models/phi3_mini/config.py:Phi3MiniConfig.to_block_spec()`. Concretely:

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=32, head_dim=96,
        kind=AttentionKind.STANDARD,
        qkv_layout=QKVLayout.FUSED,            # ← Phi-3 fused QKV
        mask_kind=MaskKind.SWA, sliding_window=2047,
        rope=RoPESpec(base_theta=10000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE,
                      partial_rotary_factor=1.0),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=8192,
        activation=Activation.SILU, gate_kind=GateKind.SWIGLU,
        fused_gate_up=True,                    # ← Phi-3 fused gate_up
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
```

## 6. Quirks

- **FUSED QKV** — `qkv_proj.weight` has output dim `(Hq + 2*Hk) * Dh = 9216`
  for the 4k variant. Sliced in forward order Q | K | V along the output dim.
  Source: modeling_phi3.py:222 (op_size), 224 (qkv_proj), 237-241 (slicing).
- **FUSED gate_up** — `gate_up_proj.weight` has output dim `2 * I = 16384`.
  Phi-3 chunks along the LAST dim and returns `(gate, up)`, then computes
  `up * act(gate)` (commutative with `act(gate) * up`). Source:
  modeling_phi3.py:54 (init), 58-62 (forward chunk + silu*up).
- **MHA, not GQA** for Phi-3-mini-4k — `num_key_value_heads == num_attention_heads = 32`.
- **head_dim=96** is unusual (most models use 64 or 128); but `head_dim/2 = 48`,
  even — RoPE rotation works.
- **SWA window=2047** — applied via `mask_kind=SWA`. Phi-3 applies SWA to ALL
  layers (no alternating with global, unlike Gemma 3 family).
- **Default RoPE (`rope_scaling=null`)** for the 4k variant — Phi-4-mini and
  Phi-3-mini-128k variants use LongRoPE; that path is covered by Phi-4-mini.
- **partial_rotary_factor=1.0** — full rotation (Phi-3.5-mini lowered it to
  0.5; Phi-4-mini-instruct sets 0.75; both lands with Phi-4-mini).
- **No QK-norm** — Phi-3 attention has only qkv_proj and o_proj.
- **No biases** — `attention_bias=False`, `mlp_bias=False`.
- **`resid_pdrop=0`** at the trained checkpoint — the HF code path applies
  `Dropout(0)` (`= identity`) in eval mode; we omit dropout entirely.

## 7. Weight-name mapping (HF → API)

The fused projections come straight from HF — no reshape, no concat. Our api
slicing logic mirrors HF's at forward time.

| HF tensor name (Phi-3) | API tensor slot |
|---|---|
| `model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.layers.{L}.self_attn.qkv_proj.weight` | `blk.attention.qkv_proj.weight` |
| `model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.layers.{L}.mlp.gate_up_proj.weight` | `blk.feedforward.gate_up_proj.weight` |
| `model.layers.{L}.mlp.down_proj.weight` | `blk.feedforward.down_proj.weight` |

Notably absent (vs Llama): there is NO `self_attn.{q,k,v}_proj` and NO
`mlp.{gate,up}_proj` — those tensors are stored fused in HF.

See `models/phi3_mini/layer.py::load_hf_phi3_mini_layer` for the loader.

## 8. Source citations

- HF reference: `transformers/models/phi3/modeling_phi3.py`
  - `Phi3MLP` line 49-64 (fused gate_up + chunk(2, -1) + up*silu(gate)).
  - `Phi3Attention.__init__` line 208-224 (op_size + qkv_proj + o_proj).
  - `Phi3Attention.forward` line 226-271 (slicing Q|K|V).
  - `Phi3DecoderLayer.forward` line 295-335.
- HF config: `transformers/models/phi3/configuration_phi3.py` (rope_parameters,
  short_factor/long_factor validation lines 107-149).
- HF RoPE: `transformers/modeling_rope_utils.py::_compute_longrope_parameters`
  lines 462-547 (used by Phi-4-mini, NOT by Phi-3-mini-4k).
- Census: `research/01-model-census.v3.md` (Phi-3 section).
