# Llama 4 Scout (17B-16E) — Decoder Layer

## 1. Identity

- **Family:** Llama 4 (Meta)
- **Variants in B4 scope:** `meta-llama/Llama-4-Scout-17B-16E` (gated;
  mirror `unsloth/Llama-4-Scout-17B-16E`); MoE 16-expert text-only path.
- **Release date:** April 2025
- **HF base model card (gated):** `meta-llama/Llama-4-Scout-17B-16E`
- **transformers source:** `transformers/models/llama4/modeling_llama4.py`

## 2. Decoder block diagram

For a RoPE layer (``no_rope_layers[i] == 1``):

```
        x  [B, S, 5120]
        |
   +----+----+
   |         |
   |     pre_attn_norm (RMSNorm STANDARD_W, eps=1e-5)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj    (split QKV, no bias)
   |    ├── qk_norm  (L2Norm — deferred from B4)
   |    ├── RoPE INTERLEAVED, theta=500_000.0
   |    │      + LLAMA3 scaling (factor=8, low=1, high=4, orig=8192)
   |    ├── KVCache write/read        (CONTIGUOUS HND)
   |    ├── SDPA (GQA, n_q=40, n_kv=8, head_dim=128)
   |    │      mask = CAUSAL (full layer in layer_types)
   |    └── o_proj                    (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm (RMSNorm STANDARD_W)
   |         |
   |    feed_forward (MoE in 16E variant)
   |    ├── router(hidden_states)     (Linear -> top-1 over 16 experts)
   |    ├── experts(routed)           (Llama4TextExperts BMM)
   |    └── shared_expert(hidden_states)  (Llama4TextMLP — always run)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, 5120]
```

For a NoPE layer (``no_rope_layers[i] == 0``):
- The ``apply_rotary_emb`` call is SKIPPED (modeling_llama4.py:369-372).
- The ``qk_norm`` module is NOT instantiated (modeling_llama4.py:351
  gates on ``self.config.use_qk_norm AND self.use_rope``).
- ``attn_temperature_tuning`` (modeling_llama4.py:379) is ADDED:
  ``q *= log1p(floor((pos+1)/floor_scale)) * attn_scale + 1.0``.
- ``layer_types[i] == "chunked_attention"`` (configuration_llama4.py:197-200):
  attention is chunked into non-overlapping windows of
  ``attention_chunk_size=8192``. NOT supported by B4 IR.

B4 ships the ATTENTION MODULE ONLY (no DecoderBlock). The MoE channel mixer
lands in B6 along with the temperature tuning and chunked-attention mask.

## 3. Tensor IO trace

For Scout 16E (D=5120, n_q=40, n_kv=8, head_dim=128):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 5120] | bf16 |
| pre_attn_norm (caller) | x_in | [B, S, 5120] | bf16 |
| q_proj | q | [B, S, 40*128]=[B, S, 5120] | bf16 |
| k_proj | k | [B, S, 8*128]=[B, S, 1024] | bf16 |
| v_proj | v | [B, S, 8*128]=[B, S, 1024] | bf16 |
| reshape | q | [B, S, 40, 128] | bf16 |
| reshape | k, v | [B, S, 8, 128] | bf16 |
| RoPE INTERLEAVED (RoPE layers only) | q, k | [B, S, *, 128] | bf16 |
| (qk_norm, attn_temperature_tuning — deferred) | -- | -- | -- |
| transpose | q | [B, 40, S, 128] | bf16 |
| transpose | k, v | [B, 8, S, 128] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 8, max_seq, 128] | bf16 |
| sdpa | a | [B, 40, S, 128] | bf16 |
| transpose+reshape | a | [B, S, 5120] | bf16 |
| o_proj | attn_out | [B, S, 5120] | bf16 |

## 4. Op trace (api.ops sequence) — RoPE layer

```
q       = linear(x_in, q_proj.weight)
k       = linear(x_in, k_proj.weight)
v       = linear(x_in, v_proj.weight)
q, k    = rope_apply(q, k, cos, sin, basis="interleaved")    # NoPE: SKIPPED
# transpose + cache.write + cache.read
attn_out = sdpa(q, k, v, mask=causal, scale=1/sqrt(128))
attn_out = linear(attn_out, o_proj.weight)
```

For a NoPE layer (``use_rope=False``):
- ``rope_apply`` is omitted.
- ``attn_temperature_tuning`` would scale Q by a position-dependent factor
  (DEFERRED from B4).
- ``qk_norm`` is omitted entirely (no L2Norm).

## 5. Spec instantiation

Per-layer, ``Llama4ScoutConfig.to_attention_spec(layer_idx)`` returns an
``AttentionSpec``. The DecoderBlock-level FFN spec is NOT emitted by B4
(MoE wiring is deferred).

```python
# RoPE layer (e.g. layer 0):
AttentionSpec(
    n_q_heads=40, n_kv_heads=8, head_dim=128,
    kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
    mask_kind=MaskKind.CAUSAL,
    q_bias=False, k_bias=False, v_bias=False, o_bias=False,
    rope=RoPESpec(
        base_theta=500_000.0,
        basis=RoPEBasis.INTERLEAVED,             # 3rd basis after SPLIT_HALF
        scaling=RoPEScaling.LLAMA3,
        llama3_extra=Llama3RoPEParams(
            factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
            original_context_length=8192,
        ),
    ),
)
# NoPE layer (e.g. layer 3 with default interval=4):
#   identical except rope=None.
```

## 6. Quirks

- **iRoPE per-layer dispatch — the headline B4 axis.** The HF field is
  ``no_rope_layers: list[int]`` — confusingly named, but ``1`` means USE
  RoPE and ``0`` means NoPE (skip RoPE). Source: modeling_llama4.py:338
  (``self.use_rope = config.no_rope_layers[layer_idx]``) and the default
  pattern at configuration_llama4.py:179-181
  (``(layer_idx + 1) % no_rope_layer_interval != 0`` -> 1).

  This is FUNCTIONALLY THE SAME dispatch as SmolLM3 (same field, same
  polarity, same default-formula structure), so the IR didn't need
  changes — ``AttentionSpec.rope = None`` already encodes NoPE layers.
  The novel pieces are:

- **INTERLEAVED RoPE basis (3rd basis in the corpus).** Llama 4's RoPE
  rotates consecutive (real, imag) pairs via complex multiply:
  ``view_as_complex(q.reshape(..., Dh/2, 2)) * freqs_cis``. This is
  mathematically the GPT-J INTERLEAVED basis, NOT the GPT-NeoX SPLIT_HALF
  basis used by Llama 3, Qwen3, Mistral, Gemma, etc. Added to ``api.ops``
  and ``api.rope`` as part of B4. Source: modeling_llama4.py:245-254.

- **L2Norm QK-norm only on RoPE layers.** When ``config.use_qk_norm=True``
  AND ``self.use_rope=True``, modeling_llama4.py:351 instantiates a
  ``Llama4TextL2Norm`` (modeling_llama4.py:107-119) — a unit-sphere
  projection, NOT the RMSNorm-shaped QK-norm we have in the IR for Gemma 3
  / Qwen3. **B4 does NOT plumb L2Norm through the IR.** The
  ``Llama4ScoutConfig.use_qk_norm`` field is preserved for downstream
  consumers, but ``to_attention_spec`` returns ``qk_norm=None`` so the
  shape-only path stays clean. A later milestone will add a QKNormShape
  variant for L2Norm together with the numerical gate.

- **Attention temperature tuning on NoPE layers only.** Per
  modeling_llama4.py:379-386, NoPE layers scale Q by a
  log-position-dependent factor: ``log1p(floor((pos+1)/floor_scale)) *
  attn_scale + 1.0``. This is a per-token scalar applied to Q AFTER the
  no-op RoPE step. NOT exercised in B4 (would require a Q-side scale
  field on AttentionSpec).

- **``layer_types[i] == "chunked_attention"`` on NoPE layers.** Per
  configuration_llama4.py:197-200, chunked attention masks attention into
  non-overlapping windows of ``attention_chunk_size`` (default 8192).
  This is a 3rd mask family beyond CAUSAL / SWA. B4 does not implement
  it — ``to_attention_spec`` raises ``NotImplementedError`` on chunked
  layers. The 16E variant under the default pattern places chunked on
  every 4th layer (i = 3, 7, ..., 47).

- **MoE at every layer in 16E.** ``interleave_moe_layer_step=1`` means
  ``moe_layers = list(range(48))`` — all 48 layers are MoE. There are no
  pure-dense layers in Scout 16E. MoE wiring is deferred to B6. B4
  therefore ships the per-layer ATTENTION only.

- **Weights gated AND too large to gate numerically.** The published
  ``meta-llama/Llama-4-Scout-17B-16E`` requires a gated HF token; the
  ``unsloth/Llama-4-Scout-17B-16E`` mirror is open but distributes 50
  safetensors shards totaling ~215 GB in fp32. The B4 numerical-gate
  budget cannot accommodate this. We ship a SHAPE-ONLY synthetic-weight
  test that exercises the iRoPE per-layer dispatch end-to-end on the
  attention module.

## 7. Weight-name mapping (HF → API)

ATTENTION-ONLY mapping (channel-mixer wiring lands in B6):

| HF tensor name | API tensor slot |
|---|---|
| `model.layers.{L}.self_attn.q_proj.weight` | `attn.q_proj.weight` |
| `model.layers.{L}.self_attn.k_proj.weight` | `attn.k_proj.weight` |
| `model.layers.{L}.self_attn.v_proj.weight` | `attn.v_proj.weight` |
| `model.layers.{L}.self_attn.o_proj.weight` | `attn.o_proj.weight` |

(Optional `q_proj.bias`, etc. when `attention_bias=True` — Scout config
has `attention_bias=False`.)

NoPE layers carry the same tensor names — the dispatch lives in the
attention module's spec, not in the state-dict.

The ``model.layers.{L}.self_attn.qk_norm.weight`` tensor (present on RoPE
layers when ``use_qk_norm=True``) is NOT loaded because the IR does not
materialise the L2Norm module in B4.

The ``model.layers.{L}.input_layernorm.weight`` and the channel-mixer
weights (router, experts, shared_expert) are NOT loaded — they belong to
the DecoderBlock / MoE wiring deferred to B6.

## 8. Source citations

- HF reference: `transformers/models/llama4/modeling_llama4.py` (1413 lines)
  - L122-139 RMSNorm
  - L107-119 Llama4TextL2Norm (qk_norm primitive)
  - L179-242 Llama4TextRotaryEmbedding (INTERLEAVED basis)
  - L245-254 apply_rotary_emb (view_as_complex + complex multiply)
  - L321-410 Llama4TextAttention (iRoPE branch, qk_norm gate, temperature tuning)
  - L413-460 Llama4TextDecoderLayer (MoE / dense routing)
- HF config: `transformers/models/llama4/configuration_llama4.py`
  - L139-202 Llama4TextConfig (no_rope_layers, layer_types defaults)
- Public mirror: `unsloth/Llama-4-Scout-17B-16E` (50 shards, ~215 GB fp32)
- Gated original: `meta-llama/Llama-4-Scout-17B-16E`
- Census: `research/01-model-census.v3.md`
