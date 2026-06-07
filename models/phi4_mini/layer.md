# Phi-4-mini-instruct — Decoder Layer

## 1. Identity

- **Family:** Phi-4 (Microsoft)
- **Variant in B2a scope:** Phi-4-mini-instruct (3.84B)
- **Release date:** Feb 2025
- **HF base model card:** `microsoft/Phi-4-mini-instruct`
- **transformers source:**
  `transformers/src/transformers/models/phi3/modeling_phi3.py` (yes — Phi-4
  mini reuses the Phi3ForCausalLM code path with model_type="phi3"; the new
  rotary axis is handled entirely by config).

## 2. Decoder block diagram

Identical to Phi-3 mini in structure, with these knobs flipped:

| axis | Phi-3 mini 4k | Phi-4-mini-instruct |
|---|---|---|
| Hq | 32 | **24** |
| Hk | 32 (MHA) | **8** (GQA) |
| head_dim | 96 | **128** |
| sliding_window | 2047 | **None** (dropped) |
| partial_rotary_factor | 1.0 | **0.75** |
| rope_scaling | null | **longrope** (short_factor/long_factor) |
| original_max_position | 4096 | 4096 |
| max_position | 4096 | **131072** |
| tie_word_embeddings | False | **True** |

```
        x  [B, S, 3072]
        |
   +----+----+
   |         |
   |     pre_attn_norm (RMSNorm STANDARD_W, eps=1e-5)
   |         |
   |    attention(token_mixer)
   |    ├── qkv_proj                 ← FUSED, output (24+16)*128 = 5120
   |    │   then slice (Q | K | V) along output dim
   |    ├── RoPE (SPLIT_HALF, partial_rotary_factor=0.75 → 96 of 128 channels
   |    │         rotate; LongRoPE dispatch short/long at seq=4096)
   |    ├── KVCache write/read       (CONTIGUOUS HND)
   |    ├── SDPA (GQA, n_q=24, n_kv=8, plain CAUSAL mask)
   |    └── o_proj                   (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm (RMSNorm STANDARD_W)
   |         |
   |    feedforward(channel_mixer)
   |    ├── gate_up_proj             ← FUSED, output 2 * 8192 = 16384
   |    │   then chunk(2, dim=-1) → (gate, up)
   |    ├── SiLU(gate) * up
   |    └── down_proj                (no bias)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, 3072]
```

## 3. Tensor IO trace

For Phi-4-mini-instruct (D=3072, Hq=24, Hk=8, head_dim=128, I=8192):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 3072] | bf16 |
| pre_attn_norm | x_in | [B, S, 3072] | bf16 |
| qkv_proj | qkv | [B, S, (24+16)*128] = [B, S, 5120] | bf16 |
| slice Q | q | [B, S, 24, 128] | bf16 |
| slice K | k | [B, S, 8, 128] | bf16 |
| slice V | v | [B, S, 8, 128] | bf16 |
| RoPE (partial, 96 of 128) | q, k | [B, S, *, 128] | bf16 |
| transpose | q | [B, 24, S, 128] | bf16 |
| transpose | k, v | [B, 8, S, 128] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 8, max_seq, 128] | bf16 |
| sdpa (causal, GQA) | a | [B, 24, S, 128] | bf16 (fp32 accum) |
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

Identical to Phi-3 mini except the RoPE step uses LongRoPE dispatch:

```
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
qkv     = linear(x_in, qkv_proj.weight)
q       = qkv[..., :Hq*Dh].view(B, S, Hq, Dh)
k       = qkv[..., Hq*Dh : (Hq+Hk)*Dh].view(B, S, Hk, Dh)
v       = qkv[..., (Hq+Hk)*Dh:].view(B, S, Hk, Dh)
# RoPE uses cos_cached_long when max(pos)+1 > original_max (4096), else cos_cached.
q, k    = rope_apply(q, k, cos, sin, basis="split_half")
a       = sdpa(q, k_full, v_full, attn_mask=CAUSAL, scale=head_dim ** -0.5)
attn_out = linear(a, o_proj.weight)
x       = add(x, attn_out)

h       = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
up_states = linear(h, gate_up_proj.weight)
gate, up = up_states.chunk(2, dim=-1)
h_act   = mul(silu(gate), up)
ffn_out = linear(h_act, down_proj.weight)
y       = add(x, ffn_out)
```

## 5. Spec instantiation

See `models/phi4_mini/config.py:from_hf_dict()` (alias for
`Phi3MiniConfig.from_hf_dict`). Concretely:

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=24, n_kv_heads=8, head_dim=128,
        kind=AttentionKind.STANDARD,
        qkv_layout=QKVLayout.FUSED,
        mask_kind=MaskKind.CAUSAL,
        rope=RoPESpec(
            base_theta=10000.0, basis=RoPEBasis.SPLIT_HALF,
            scaling=RoPEScaling.LONGROPE,
            longrope_extra=LongRoPEParams(
                short_factor=(...,)*48, long_factor=(...,)*48,
                original_max_position_embeddings=4096,
                attention_factor=sqrt(1 + log(32)/log(4096)) ≈ 1.189,
            ),
            partial_rotary_factor=0.75,
        ),
    ),
    channel_mixer=FFNSpec(intermediate_size=8192, gate_kind=GateKind.SWIGLU,
                          fused_gate_up=True),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
```

## 6. Quirks

- **LongRoPE — TWO inv_freq tables.** `short_factor` is used when
  `seq_len ≤ original_max_position_embeddings (=4096)`; `long_factor`
  otherwise. Both vectors have length 48 (= head_dim_rot/2 = 96/2). cos/sin
  are scaled by `attention_factor = sqrt(1 + log(factor)/log(orig_max))`
  where `factor = max_position / orig_max = 32`. Source:
  modeling_rope_utils.py:_compute_longrope_parameters lines 462-547.
- **`partial_rotary_factor=0.75`** — only the first 96 of 128 head_dim
  channels rotate. We use the existing proportional-RoPE path (api/rope.py
  splits inv_freq into 48 real frequencies + 16 zeros); cos=1, sin=0 on the
  trailing channels, which means rotate_half STILL pairs them but with
  identity, leaving them unchanged at the matmul step. Verified at
  modeling_phi3.py:178-205 (apply_rotary_pos_emb slices q_rot/q_pass by
  rotary_dim = cos.shape[-1]).
- **GQA 24/8** with **head_dim=128** — unusual choice (most 3B models use 96
  or 128).
- **No SWA** — `sliding_window=None` (Phi-3 mini had 2047).
- **`tie_word_embeddings=True`** — lm_head shares weights with embed_tokens
  (model-level; layer is unaffected).
- **FUSED QKV / FUSED gate_up** — same as Phi-3.
- **Reuses Phi3ForCausalLM** (model_type=phi3 in config.json) — there is NO
  separate Phi4 modeling file in transformers 5.x as of this milestone; only
  phi4_multimodal is its own module.

## 7. Weight-name mapping (HF → API)

Identical to Phi-3 mini (same HF code path):

| HF tensor name | API tensor slot |
|---|---|
| `model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.layers.{L}.self_attn.qkv_proj.weight` | `blk.attention.qkv_proj.weight` |
| `model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.layers.{L}.mlp.gate_up_proj.weight` | `blk.feedforward.gate_up_proj.weight` |
| `model.layers.{L}.mlp.down_proj.weight` | `blk.feedforward.down_proj.weight` |

See `models/phi4_mini/layer.py::load_hf_phi4_mini_layer` (which delegates to
the Phi-3 mini loader).

## 8. Source citations

- HF reference: `transformers/models/phi3/modeling_phi3.py` (Phi-4-mini
  reuses Phi3ForCausalLM).
- HF config: `transformers/models/phi3/configuration_phi3.py`
  (validates short_factor / long_factor length at lines 119-149).
- HF LongRoPE: `transformers/modeling_rope_utils.py::_compute_longrope_parameters`
  lines 462-547; `dynamic_rope_update.longrope_frequency_update` lines 47-80.
- Phi-4-mini HF config.json (microsoft/Phi-4-mini-instruct):
  attention_bias=False, partial_rotary_factor=0.75, original_max_position=4096,
  rope_scaling.type="longrope".
