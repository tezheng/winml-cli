# Gemma 4 — Decoder Layer

## 1. Identity

- **Family:** Gemma 4 (Google DeepMind)
- **Variants in B0.5 scope:** E2B (5.1B total / 2.3B effective active params, edge/mobile-tuned)
- **Future B0.5 follow-ups (not part of this batch):** E4B, 12B Unified, 26B-A4B MoE, 31B
- **Release date:** 2026-04-02
- **Announcement:** https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- **HF model card:** https://huggingface.co/google/gemma-4-E2B-it (config schema verified
  against the public 12B-Unified card and the 1B/270M Gemma 3 cards used as a shape
  template for synthetic test fixtures in this milestone).
- **transformers source:** `transformers/src/transformers/models/gemma4/modeling_gemma4.py`
- **Research notes:** `research/01-model-census.v3.md` §3.5, `research/02-layer-sources.v3.md` Gemma 4 section,
  `research/issues/10-gemma4-investigation.md`.

## 2. Decoder block diagram

E2B alternates four local (sliding-window) attention layers with one global (full
causal, partial-RoPE) layer in a 4:1 repeating pattern. Per-layer type dispatch
happens in `Gemma4Config.to_block_spec(layer_idx)`. Both layer types use the same
sandwich-norm (PRE_AND_POST) structure; only the attention sub-layer's mask, head
dimension, RoPE θ, partial-rotary factor, and QK fixed-scale differ.

```
                  x  [B, S, D]
                  |
            +-----+-----+
            |           |
            |    pre_attn_norm (RMSNorm ONE_PLUS_W)
            |           |
            |    attention(token_mixer)
            |    +-- q_proj, k_proj, v_proj    (split QKV, no bias)
            |    +-- q_norm, k_norm            (QKNorm PRE-RoPE, PER_HEAD_DH,
            |    |                              ONE_PLUS_W; fixed-scale absorbed
            |    |                              into weight at load time)
            |    +-- RoPE                       (local: theta=1e4, partial=1.0
            |    |                              global: theta=1e6, partial=0.25)
            |    +-- KVCache write/read         (local: ContiguousKVCache + SWA mask
            |    |                              shared layers: SharedLayerKVCache)
            |    +-- SDPA                       (effective_scale=1.0; SWA mask on
            |    |                              local, full causal on global)
            |    +-- o_proj                     (no bias)
            |           |
            |    post_attn_sublayer_norm (RMSNorm ONE_PLUS_W)
            |           |
            +-----> add (residual 1)
                  |
            +-----+-----+
            |           |
            |    pre_ffn_norm (RMSNorm ONE_PLUS_W)
            |           |
            |    feedforward(channel_mixer)
            |    +-- gate_proj                  (GeGLU gate, no bias)
            |    +-- up_proj                    (GeGLU up,   no bias)
            |    +-- gelu_pytorch_tanh(gate) * up
            |    +-- down_proj                  (no bias)
            |           |
            |    post_ffn_sublayer_norm (RMSNorm ONE_PLUS_W)
            |           |
            +-----> add (residual 2)
                  |
            (+ per_layer_residual from PerLayerEmbedding, when PLE enabled —
             scaled by 1/sqrt(2); injected by the model-assembly loop, not
             by build_gemma4_decoder_layer)
                  |
                  y  [B, S, D]
```

## 3. Tensor IO trace

For Gemma 4 E2B (D=1536, Hq=8, Hk=1, Dh_local=256, Dh_global=512, I=8192, ple_dim=256):

### Layer 0 (local SWA layer)

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 1536] | bf16 |
| pre_attn_norm | x_in | [B, S, 1536] | bf16 (computed fp32) |
| q_proj | q | [B, S, 8*256]=[B, S, 2048] | bf16 |
| k_proj | k | [B, S, 1*256]=[B, S, 256] | bf16 |
| v_proj | v | [B, S, 1*256]=[B, S, 256] | bf16 |
| reshape | q | [B, S, 8, 256] | bf16 |
| reshape | k, v | [B, S, 1, 256] | bf16 |
| q_norm | q | [B, S, 8, 256] | bf16 (fp32 norm) |
| k_norm | k | [B, S, 1, 256] | bf16 (fp32 norm) |
| RoPE (partial=1.0, θ=1e4) | q, k | [B, S, *, 256] | bf16 |
| transpose | q | [B, 8, S, 256] | bf16 |
| transpose | k, v | [B, 1, S, 256] | bf16 |
| cache.write | KVCache.k/v | [B, 1, max_seq, 256] | bf16 |
| cache.read | k_full, v_full | [B, 1, T, 256] (T=start_pos+S) | bf16 |
| SDPA (SWA mask W=512, scale=1.0) | a | [B, 8, S, 256] | bf16 (fp32 accum) |
| transpose+reshape | a | [B, S, 2048] | bf16 |
| o_proj | attn_out | [B, S, 1536] | bf16 |
| post_attn_sublayer_norm | attn_out | [B, S, 1536] | bf16 |
| residual add | x | [B, S, 1536] | bf16 |
| pre_ffn_norm | h | [B, S, 1536] | bf16 |
| gate_proj | g | [B, S, 8192] | bf16 |
| up_proj | u | [B, S, 8192] | bf16 |
| gelu_pytorch_tanh + mul | h_act | [B, S, 8192] | bf16 |
| down_proj | ffn_out | [B, S, 1536] | bf16 |
| post_ffn_sublayer_norm | ffn_out | [B, S, 1536] | bf16 |
| residual add | x | [B, S, 1536] | bf16 |
| PLE residual add (optional) | y | [B, S, 1536] | bf16 |

### Layer 4 (global full-attention layer)

Same shape table, except:
- q_proj output: [B, S, 8*512]=[B, S, 4096]
- k_proj / v_proj output: [B, S, 1*512]=[B, S, 512]
- reshape Dh = 512 throughout the attention sublayer
- KVCache head_dim=512 (the model assembly must allocate global-layer caches separately
  from local-layer caches because head_dim differs)
- RoPE applies partial rotation: only the first floor(512*0.25)=128 channel pairs
  rotate; the remaining channels pass through unchanged
- SDPA mask: full causal (no W cap)
- attn_out shape after o_proj: [B, S, 1536] (unchanged — o_proj projects 8*512 → 1536)

## 4. Op trace (`api.ops` / `api.norm` sequence)

Local layer (index 0):
```
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "one_plus_w")
q       = linear(x_in, q_proj.weight); reshape to [B, S, 8, 256]
k       = linear(x_in, k_proj.weight); reshape to [B, S, 1, 256]
v       = linear(x_in, v_proj.weight); reshape to [B, S, 1, 256]
q       = rms_norm(q, q_norm.weight, eps, "one_plus_w")    # PER_HEAD_DH; weight pre-absorbed
k       = rms_norm(k, k_norm.weight, eps, "one_plus_w")
q, k    = rope_apply_partial(q, k, cos, sin, basis="split_half", partial=1.0)
# cache.write(k, v, start_pos); k_full, v_full = cache.read(start_pos + S)
a       = sdpa(q, k_full, v_full, attn_mask=SWA_mask(W=512), scale=1.0)
attn_out = linear(a, o_proj.weight)
attn_out = rms_norm(attn_out, post_attn_sublayer_norm.weight, eps, "one_plus_w")
x       = add(x, attn_out)

h       = rms_norm(x, pre_ffn_norm.weight, eps, "one_plus_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
h_act   = mul(gelu_pytorch_tanh(g), u)
ffn_out = linear(h_act, down_proj.weight)
ffn_out = rms_norm(ffn_out, post_ffn_sublayer_norm.weight, eps, "one_plus_w")
y       = add(x, ffn_out)
# Optional PLE injection (model-level): y = add(y, per_layer_residual)
```

Global layer (index 4) — only the attention sub-block differs:
```
# Same as local, but:
q, k    = rope_apply_partial(q, k, cos, sin, basis="split_half", partial=0.25)
a       = sdpa(q, k_full, v_full, attn_mask=full_causal_mask, scale=1.0)
```

## 5. Spec instantiation (`Gemma4Config.to_block_spec(layer_idx)`)

Layer 0 (local SWA) for E2B:
```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE_AND_POST,
    ffn_norm_position=NormPosition.PRE_AND_POST,
    token_mixer=AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=256,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.SWA, sliding_window=512,
        qk_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                         weight_mode=NormWeightMode.ONE_PLUS_W),
        qk_norm_phase=QKNormPhase.PRE_ROPE,
        qk_norm_shape=QKNormShape.PER_HEAD_DH,
        qk_norm_fixed_scale=0.9916,           # local
        attention_k_eq_v=False,
        rope=RoPESpec(base_theta=10_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      partial_rotary_factor=1.0),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=8192,
        activation=Activation.GELU,
        gate_kind=GateKind.GEGLU,
    ),
    pre_attn_norm=NormSpec(...ONE_PLUS_W...),
    post_attn_norm=NormSpec(...ONE_PLUS_W...),
    pre_ffn_norm=NormSpec(...ONE_PLUS_W...),
    post_ffn_norm=NormSpec(...ONE_PLUS_W...),
    per_layer_embedding=PLESpec(ple_dim=256, residual_scale=1/sqrt(2),
                                injection_norm=NormSpec(...ONE_PLUS_W...)),
    final_logit_softcap=30.0,
    embedding_scale=sqrt(1536),
)
```

Layer 4 (global full-attention) for E2B — diffs from layer 0:
```python
token_mixer=AttentionSpec(
    head_dim=512,                              # global_head_dim (E2B-only doubling)
    mask_kind=MaskKind.CAUSAL,
    sliding_window=None,
    qk_norm_fixed_scale=1.0228,                # global
    attention_k_eq_v=False,                    # would be True on 12B/26B/31B; E2B has split K/V
    rope=RoPESpec(base_theta=1_000_000.0,
                  basis=RoPEBasis.SPLIT_HALF,
                  partial_rotary_factor=0.25),
    ...
)
```

## 6. Quirks

- **Sandwich norm (PRE_AND_POST):** both the attention and FFN sublayers have a
  norm AFTER the sublayer output as well as before, all RMSNorm with the Gemma
  ONE_PLUS_W weight mode. There are FOUR norms per decoder block, not two.
- **ONE_PLUS_W RMSNorm baking foot-gun:** Gemma's RMSNorm stores `w` such that
  the effective gain is `(1 + w)`. Our QK-norm weight loader absorbs the
  Gemma 4 fixed scale AND `sqrt(Dh)` into the weight, so the on-disk
  `learned_w` is transformed to `w_new = (1 + learned_w) * fixed_scale * sqrt(Dh) - 1`.
  This lets the attention runtime use `effective_scale = 1.0` (no per-step
  divide by `sqrt(Dh)`) and skip the fixed-scale multiplication entirely.
  Math justification: pre-baking, attention computes
  `softmax( ((g_q * q) @ (g_k * k).T) / sqrt(Dh) )` where `g_q, g_k`
  include the fixed scales. Post-baking, the gain itself carries the
  `1/sqrt(Dh)` so we softmax `((g'_q * q) @ (g'_k * k).T)` with no extra scale.
- **Per-layer-type partial RoPE on global only:** local layers rotate the full
  head dim (partial=1.0); global layers rotate only the first 25% of channel
  pairs (partial=0.25). The remaining channels pass through unchanged.
- **Dual RoPE θ:** local θ=1e4, global θ=1e6 — a single cos/sin table cannot
  serve both layer types; the `api.rope.RoPE` module owned by each Attention
  has the layer-type-specific θ.
- **Global head_dim ≠ local head_dim on E2B (and E4B):** E2B uses Dh_local=256
  but Dh_global=512. The o_proj input dim is `n_q_heads * head_dim_eff`, so the
  global and local o_proj weight matrices have different input dimensions
  (8*256=2048 vs 8*512=4096) yet the same output dim (1536). The KV cache must
  allocate per-layer-type sizes; the model assembly cannot share a single
  cache shape across both layer types.
- **Cross-layer KV sharing (E2B/E4B only — axis A18):** the last
  `num_kv_shared_layers=20` layers (indices 15..34 on E2B) DO NOT own their
  K/V buffer. Instead the model assembly wraps the source layer's
  `ContiguousKVCache` in `SharedLayerKVCache` and passes it to the shared
  layer. The source layer index is computed by
  `Gemma4Config.kv_source_layer_idx_map()`:
  for E2B with period=5 and shared_start=15, `source[i] = ((i - 15) mod 5) + 10`,
  so layers 15..19 source from 10..14 (the last unshared block), layers 20..24
  also source from 10..14, etc. Every source layer is itself unshared. The
  shared layer still computes its own K/V tensors but `SharedLayerKVCache.write`
  is a no-op — they are discarded.
- **Per-Layer Embeddings (E2B/E4B only — axis A19):** a separate
  `PerLayerEmbedding` module owns the PLE table (`vocab_size × ple_dim=256`),
  the injection RMSNorm, and one linear projection per decoder layer that maps
  ple_dim → hidden_size. At each layer the model-assembly loop computes
  `per_layer_residual = layer_proj_l(inj_norm(ple_table[input_ids])) / sqrt(2)`
  and passes it to `DecoderBlock.forward(..., per_layer_residual=...)`. The
  block adds it to the residual stream AFTER the FFN residual add. The
  `build_gemma4_decoder_layer` factory is unaware of PLE — the PLE module
  is owned at the model level.
- **`attention_k_eq_v` is size-specific:** True on 12B Unified, 26B-A4B MoE,
  and 31B; FALSE on E2B and E4B. When True AND the layer is global,
  `api.attention.Attention` skips allocating `v_proj` and aliases V := K
  after the K projection; the weight loader correspondingly does not look
  for a `v_proj.weight` tensor. On E2B every layer (local AND global) has
  a separate V projection.
- **final_logit_softcap = 30.0 carried at block level:** Gemma 3 had dropped
  the softcap; Gemma 4 restores it. The cap applies at the LM head, not at
  attention logits (`attn_logit_softcap` is None on Gemma 4 — also restored
  from Gemma 3 which had it). The cap is propagated via
  `DecoderBlockSpec.final_logit_softcap` so the model-assembly LM head can
  apply `softcap * tanh(logits / softcap)` after the lm_head linear.
- **embedding_scale = sqrt(hidden_size):** classic Gemma signature carried
  through to Gemma 4. The model-assembly embedding lookup scales by
  `sqrt(hidden_size)` before injecting into the residual stream.
- **GeGLU with `gelu_pytorch_tanh` activation:** preserved Gemma signature.
  This is the `tanh`-approximation GELU, not the default `erf`-GELU; the
  ops module exposes both via `api.ops.gelu_pytorch_tanh`.

## 7. Weight-name mapping (HF → API)

For decoder layer L of an HF-format Gemma 4 checkpoint:

| HF tensor name (Gemma 4) | API tensor slot | Notes |
|---|---|---|
| `model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` | RMSNorm ONE_PLUS_W |
| `model.layers.{L}.self_attn.q_proj.weight` | `blk.attention.q_proj.weight` | shape [Hq*Dh_eff, D] |
| `model.layers.{L}.self_attn.k_proj.weight` | `blk.attention.k_proj.weight` | shape [Hk*Dh_eff, D] |
| `model.layers.{L}.self_attn.v_proj.weight` | `blk.attention.v_proj.weight` | absent on 12B+ global (K=V) |
| `model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` | shape [D, Hq*Dh_eff] |
| `model.layers.{L}.self_attn.q_norm.weight` | `blk.attention.q_norm.weight` | absorb: `(1+w)*fixed*sqrt(Dh) - 1` |
| `model.layers.{L}.self_attn.k_norm.weight` | `blk.attention.k_norm.weight` | absorb: `(1+w)*fixed*sqrt(Dh) - 1` |
| `model.layers.{L}.post_attention_layernorm.weight` | `blk.post_attn_sublayer_norm.weight` | post-attn (sandwich) |
| `model.layers.{L}.pre_feedforward_layernorm.weight` | `blk.pre_ffn_norm.weight` | pre-FFN |
| `model.layers.{L}.post_feedforward_layernorm.weight` | `blk.post_ffn_sublayer_norm.weight` | post-FFN (sandwich) |
| `model.layers.{L}.mlp.gate_proj.weight` | `blk.feedforward.gate_proj.weight` | GeGLU gate |
| `model.layers.{L}.mlp.up_proj.weight` | `blk.feedforward.up_proj.weight` | GeGLU up |
| `model.layers.{L}.mlp.down_proj.weight` | `blk.feedforward.down_proj.weight` | GeGLU down |

`Dh_eff = global_head_dim` on global layers (E2B: 512), else local `head_dim`
(E2B: 256). `fixed = qk_norm_global_fixed_scale` on global layers else
`qk_norm_local_fixed_scale`. The loader function `load_hf_gemma4_layer`
performs the absorb arithmetic before `copy_` on the QK-norm weights.

Model-level tensors (NOT loaded by `load_hf_gemma4_layer` — owned by the
model assembly):

| HF tensor name | API location |
|---|---|
| `model.embed_tokens.weight` | model-level Embedding (with `embedding_scale=sqrt(D)`) |
| `model.norm.weight` | model-level final RMSNorm (ONE_PLUS_W) |
| `model.per_layer_embeddings.weight` | `PerLayerEmbedding.ple_table.weight` |
| `model.per_layer_projections.{L}.weight` | `PerLayerEmbedding.layer_projs[L].weight` |
| `model.per_layer_input_norm.weight` | `PerLayerEmbedding.inj_norm.weight` |
| `lm_head.weight` | tied to `model.embed_tokens.weight` when `tie_word_embeddings=True` |

## 8. Source citations

- HF reference: `transformers/src/transformers/models/gemma4/modeling_gemma4.py:Gemma4DecoderLayer`
- HF config schema: `transformers/src/transformers/models/gemma4/configuration_gemma4.py`
- Survey: `research/02-layer-sources.v3.md` — Gemma 4 section (sandwich norm, QK fixed-scale absorb math, partial-RoPE on global)
- Census/axis matrix: `research/01-model-census.v3.md` §3.5 (Gemma 4 row; axes A1/A15/A18/A19)
- Investigation notes: `research/issues/10-gemma4-investigation.md` (per-size value table; absorb math derivation)
- KV cache + attention extensions: `research/05-kvcache-attention.v3.md` (axis A18 cross-layer sharing entry)
- Public announcement: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- HF model cards: https://huggingface.co/google/gemma-4-E2B-it (E2B), https://huggingface.co/google/gemma-3-1b-it (used as shape template for synthetic test fixtures while open-downloadable Gemma 4 E2B weights are gated)

## 9. B0.5 validation status

- Shape tests at the smallified E2B variant (local + global layers) — `test_layer_shape.py`, 2 tests
- Per-layer block-spec dispatch (local vs global vs PLE wiring) — `test_config.py`, 5 tests
- Synthetic HF weight loader round-trip (local + global) + QK-norm absorb math
  (local + global at known input w=0) — `test_weight_loader.py`, 4 tests
- Real-weight numerical equivalence vs HF — NOT in B0.5 scope (Gemma 4 E2B
  weights are gated at announcement and may not be open-downloadable in this
  environment); deferred to a later B0.5 task that exercises a public Gemma 4
  E2B checkpoint
