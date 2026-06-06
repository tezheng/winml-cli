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
            |    pre_attn_norm (RMSNorm STANDARD_W)
            |           |
            |    attention(token_mixer)
            |    +-- q_proj, k_proj, v_proj    (split QKV, no bias)
            |    +-- q_norm, k_norm            (QKNorm PRE-RoPE, PER_HEAD_DH,
            |    |                              STANDARD_W; no fixed-scale absorption
            |    |                              - HF Gemma 4 attention uses runtime
            |    |                              scaling = 1.0 directly)
            |    +-- v_norm                     (unit RMSNorm, with_scale=False —
            |    |                              non-persistent buffer; applied to V
            |    |                              after V projection, before transpose+cache)
            |    +-- RoPE                       (local: theta=1e4, partial=1.0
            |    |                              global: theta=1e6, partial=0.25)
            |    +-- KVCache write/read         (local: ContiguousKVCache + SWA mask
            |    |                              shared layers: SharedLayerKVCache)
            |    +-- SDPA                       (attn_scale=1.0 — HF runtime
            |    |                              `self.scaling = 1.0`, no /sqrt(Dh);
            |    |                              SWA mask on local, full causal on global)
            |    +-- o_proj                     (no bias)
            |           |
            |    post_attn_sublayer_norm (RMSNorm STANDARD_W)
            |           |
            +-----> add (residual 1)
                  |
            +-----+-----+
            |           |
            |    pre_ffn_norm (RMSNorm STANDARD_W)
            |           |
            |    feedforward(channel_mixer)
            |    +-- gate_proj                  (GeGLU gate, no bias)
            |    +-- up_proj                    (GeGLU up,   no bias)
            |    +-- gelu_pytorch_tanh(gate) * up
            |    +-- down_proj                  (no bias)
            |           |
            |    post_ffn_sublayer_norm (RMSNorm STANDARD_W)
            |           |
            +-----> add (residual 2)
                  |
            (B0.6 PLE-at-END injection, when PLE enabled and per_layer_input
             [B, S, ple_dim] is supplied:
                  residual_pe = x
                  x = per_layer_input_gate(x)         # Linear D → ple_dim
                  x = gelu_pytorch_tanh(x)
                  x = x * per_layer_input             # broadcast multiply
                  x = per_layer_projection(x)         # Linear ple_dim → D
                  x = post_per_layer_input_norm(x)    # RMSNorm STANDARD_W
                  x = residual_pe + x
             Model assembly is responsible for computing per_layer_input from
             the model-level PLE table + context-aware projection.)
                  |
            x = x * layer_scalar                      # B0.6 trailing per-layer
                                                      # scalar (init = 1.0)
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
| v_norm (unit RMS, with_scale=False) | v | [B, S, 1, 256] | bf16 (fp32 norm) |
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
| per_layer_input_gate (optional, PLE) | h_ple | [B, S, 256] | bf16 |
| gelu_pytorch_tanh + mul by per_layer_input | h_ple | [B, S, 256] | bf16 |
| per_layer_projection | h_ple | [B, S, 1536] | bf16 |
| post_per_layer_input_norm | h_ple | [B, S, 1536] | bf16 |
| residual + h_ple | x | [B, S, 1536] | bf16 |
| layer_scalar multiply | y | [B, S, 1536] | bf16 |

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
x_in    = rms_norm(x, pre_attn_norm.weight, eps, "standard_w")
q       = linear(x_in, q_proj.weight); reshape to [B, S, 8, 256]
k       = linear(x_in, k_proj.weight); reshape to [B, S, 1, 256]
v       = linear(x_in, v_proj.weight); reshape to [B, S, 1, 256]
q       = rms_norm(q, q_norm.weight, eps, "standard_w")    # PER_HEAD_DH
k       = rms_norm(k, k_norm.weight, eps, "standard_w")
q, k    = rope_apply_partial(q, k, cos, sin, basis="split_half", partial=1.0)
v       = rms_norm(v, v_norm_weight, eps, "standard_w")    # B0.6: unit RMSNorm, weight is frozen ones
# cache.write(k, v, start_pos); k_full, v_full = cache.read(start_pos + S)
a       = sdpa(q, k_full, v_full, attn_mask=SWA_mask(W=512), scale=1.0)
attn_out = linear(a, o_proj.weight)
attn_out = rms_norm(attn_out, post_attn_sublayer_norm.weight, eps, "standard_w")
x       = add(x, attn_out)

h       = rms_norm(x, pre_ffn_norm.weight, eps, "standard_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
h_act   = mul(gelu_pytorch_tanh(g), u)
ffn_out = linear(h_act, down_proj.weight)
ffn_out = rms_norm(ffn_out, post_ffn_sublayer_norm.weight, eps, "standard_w")
x       = add(x, ffn_out)

# B0.6 PLE-at-END injection (when per_layer_input is provided):
residual_pe = x
h_ple   = linear(x, per_layer_input_gate.weight)          # [B, S, ple_dim]
h_ple   = gelu_pytorch_tanh(h_ple)
h_ple   = mul(h_ple, per_layer_input)                      # [B, S, ple_dim]
h_ple   = linear(h_ple, per_layer_projection.weight)      # [B, S, D]
h_ple   = rms_norm(h_ple, post_per_layer_input_norm.weight, eps, "standard_w")
x       = add(residual_pe, h_ple)

# B0.6 trailing scalar — runs even when PLE is off
y       = x * layer_scalar
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
                         weight_mode=NormWeightMode.STANDARD_W),
        qk_norm_phase=QKNormPhase.PRE_ROPE,
        qk_norm_shape=QKNormShape.PER_HEAD_DH,
        qk_norm_fixed_scale=None,             # B0.6: no absorb on Gemma 4
        attn_scale=1.0,                       # B0.6: HF runtime scaling = 1.0
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
    pre_attn_norm=NormSpec(...STANDARD_W...),
    post_attn_norm=NormSpec(...STANDARD_W...),
    pre_ffn_norm=NormSpec(...STANDARD_W...),
    post_ffn_norm=NormSpec(...STANDARD_W...),
    per_layer_embedding=PLESpec(ple_dim=256, residual_scale=1/sqrt(2),
                                injection_norm=NormSpec(...STANDARD_W...)),
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
    qk_norm_fixed_scale=None,                  # B0.6: no absorb (unchanged from local)
    attn_scale=1.0,                            # B0.6 (unchanged from local)
    attention_k_eq_v=False,                    # would be True on 12B/26B/31B; E2B has split K/V
    rope=RoPESpec(base_theta=1_000_000.0,
                  basis=RoPEBasis.SPLIT_HALF,
                  partial_rotary_factor=0.25),
    ...
)
```

## 6. Quirks

- **Sandwich norm (PRE_AND_POST):** both the attention and FFN sublayers have a
  norm AFTER the sublayer output as well as before, all RMSNorm with the Gemma 4
  STANDARD_W weight mode (`y = x_normed * w`, NOT the Gemma 1/2/3 ONE_PLUS_W
  variant). Verified against `transformers/models/gemma4/modeling_gemma4.py:193-211`
  (`Gemma4RMSNorm.forward`). There are FOUR norms per decoder block, not two.
- **V norm with `with_scale=False`:** Gemma 4 attention runs a unit RMSNorm
  on V (`Gemma4RMSNorm(head_dim, with_scale=False)`) after the V projection
  and before the cache write. Because `with_scale=False`, there is no
  learnable weight — we store a frozen ones-buffer (non-persistent, NOT in
  the state dict). Verified at `modeling_gemma4.py:1215` (init) and 1265
  (forward). This applies even when `attention_k_eq_v=True`: V is aliased
  to the raw K projection, then K and V go through separate `k_norm` and
  `v_norm` paths.
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
- **Per-Layer Embeddings (E2B/E4B only — axis A19):** there are TWO sides
  to PLE. Model-level (`modeling_gemma4.py:1617-1631`): a packed PLE table
  `embed_tokens_per_layer` (`[vocab, num_layers*ple_dim]`), a Linear
  `per_layer_model_projection` (`[D, num_layers*ple_dim]`) for the
  context-aware path, and `per_layer_projection_norm` (RMSNorm on ple_dim).
  The model computes per_layer_inputs for ALL layers up front (one [B,S,L,ple_dim]
  tensor) and slices the layer-l slot before passing to the decoder block.
  Per-layer side (`modeling_gemma4.py:1387-1389`, 1446-1453): each decoder
  block owns `per_layer_input_gate` (Linear hidden→ple_dim),
  `per_layer_projection` (Linear ple_dim→hidden), and a `post_per_layer_input_norm`
  (RMSNorm). The block applies them AT END after the FFN residual add via
  the gate→act→multiply→proj→norm→residual chain (NOT as a simple residual
  add, which is what the B0.5 IR speculated).
- **`layer_scalar` trailing multiply:** every Gemma 4 decoder layer registers
  a buffer `layer_scalar` of shape [1] (`modeling_gemma4.py:1382`,
  initialised to ones) and multiplies the FINAL output by it
  (`modeling_gemma4.py:1455`). Loaded from the HF state dict.
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
| `model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` | RMSNorm STANDARD_W |
| `model.layers.{L}.self_attn.q_proj.weight` | `blk.attention.q_proj.weight` | shape [Hq*Dh_eff, D] |
| `model.layers.{L}.self_attn.k_proj.weight` | `blk.attention.k_proj.weight` | shape [Hk*Dh_eff, D] |
| `model.layers.{L}.self_attn.v_proj.weight` | `blk.attention.v_proj.weight` | absent on 12B+ global (K=V) |
| `model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` | shape [D, Hq*Dh_eff] |
| `model.layers.{L}.self_attn.q_norm.weight` | `blk.attention.q_norm.weight` | load straight (STANDARD_W RMSNorm, no absorb) |
| `model.layers.{L}.self_attn.k_norm.weight` | `blk.attention.k_norm.weight` | load straight (STANDARD_W RMSNorm, no absorb) |
| `model.layers.{L}.post_attention_layernorm.weight` | `blk.post_attn_sublayer_norm.weight` | post-attn (sandwich) |
| `model.layers.{L}.pre_feedforward_layernorm.weight` | `blk.pre_ffn_norm.weight` | pre-FFN |
| `model.layers.{L}.post_feedforward_layernorm.weight` | `blk.post_ffn_sublayer_norm.weight` | post-FFN (sandwich) |
| `model.layers.{L}.mlp.gate_proj.weight` | `blk.feedforward.gate_proj.weight` | GeGLU gate |
| `model.layers.{L}.mlp.up_proj.weight` | `blk.feedforward.up_proj.weight` | GeGLU up |
| `model.layers.{L}.mlp.down_proj.weight` | `blk.feedforward.down_proj.weight` | GeGLU down |
| `model.layers.{L}.per_layer_input_gate.weight` | `blk.per_layer_input_gate.weight` | PLE gate: `Linear(D, ple_dim)` (PLE-enabled only) |
| `model.layers.{L}.per_layer_projection.weight` | `blk.per_layer_projection.weight` | PLE proj: `Linear(ple_dim, D)` (PLE-enabled only) |
| `model.layers.{L}.post_per_layer_input_norm.weight` | `blk.post_per_layer_input_norm.weight` | RMSNorm STANDARD_W (PLE-enabled only) |
| `model.layers.{L}.layer_scalar` | `blk.layer_scalar` | registered buffer, shape [1] (HF init = ones) |

`Dh_eff = global_head_dim` on global layers (E2B: 512), else local `head_dim`
(E2B: 256). The loader function `load_hf_gemma4_layer` copies all tensors
straight from the HF state dict — there is no absorb arithmetic (per B0.6
correction; HF Gemma 4 attention has `self.scaling = 1.0` and plain
STANDARD_W QK norms with no fixed scale).

Model-level tensors (NOT loaded by `load_hf_gemma4_layer` — owned by the
model assembly; verified against `modeling_gemma4.py:1602-1631`):

| HF tensor name | API location |
|---|---|
| `model.embed_tokens.weight` | model-level Embedding (with `embedding_scale=sqrt(D)`) |
| `model.norm.weight` | model-level final RMSNorm (STANDARD_W) |
| `model.embed_tokens_per_layer.weight` | PLE table — packed `[vocab, num_layers*ple_dim]` |
| `model.per_layer_model_projection.weight` | Linear(D, num_layers*ple_dim) — context-aware path |
| `model.per_layer_projection_norm.weight` | RMSNorm on ple_dim (STANDARD_W) |
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

## 9. B0.6 validation status

- Shape tests at the smallified E2B variant (local + global layers) — `test_layer_shape.py`, 2 tests
- Per-layer block-spec dispatch (local vs global vs PLE wiring) — `test_config.py`, 5 tests
- Synthetic HF weight loader round-trip (local + global) + QK-norm
  load-straight (no absorb) — `test_weight_loader.py`, 4 tests
- Sub-op isolation vs HF Gemma 4 (RMSNorm STANDARD_W, GeGLU MLP, SWA mask,
  RoPE full + proportional partial geometry, no-fixed-scale, v_norm support)
  — `test_isolation_hf.py`, 8 tests
- T17 real-weight layer-0/4 numerical equivalence at atol=5e-4 — passes
- T18 real-weight KV-cache prefill+decode equivalence at atol=5e-4 — passes
