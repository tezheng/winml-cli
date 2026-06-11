# DeepSeek-V3.2 — DSA Lightning Indexer (shape-only)

## 0. At-a-glance

- **Signature:** MLA + DSA Lightning Indexer (indexer_dim=64, top-k scoring per query) on V3 backbone (shape-only forward)
- **Active params:** ~37B active / ~671B total (6%) (approximate, real config; no published HF weights yet)
- **Layer mix:** 61 attn (3 dense + 58 MoE, V3-identical MoE routing)
- **KV cache / token (bf16):** ~69 kB (61 layers × (512 + 64) × 2 B = 70272 B, MLA formula)

---

V3.2 = V3 + **DeepSeek Sparse Attention (DSA)**. The architectural delta
relative to V3 is a new **Lightning Indexer** head per attention layer
that scores `(query, key)` pairs cheaply via a separate Q/K projection of
smaller dimension (`indexer_dim`), then top-k's per query, then runs the
main MLA SDPA on only the top-k key positions.

There is no published HF transformers file for V3.2 as of the current
research snapshot. This package ships **shape-only**:
- `AttentionKind.DSA` composes through `to_block_spec`.
- `AttentionSpec.indexer` (an `IndexerSpec`) carries `indexer_dim`,
  `top_k`, and `warmup_tokens`.
- `api.attention.Attention.__init__` recognises `kind=DSA` and allocates:
  - The full MLA backbone (re-uses `_init_mla`).
  - Two indexer projections:
    - `indexer_q_proj: Linear(hidden, n_q_heads * indexer_dim, bias=False)`.
    - `indexer_k_proj: Linear(hidden, indexer_dim, bias=False)` —
      single-head Indexer K, analogous to MLA's k_pe MQA-style head.
- `Attention.forward` raises `NotImplementedError("DSA forward implementation
  deferred")`. The DSA runtime (indexer scoring → top-k → masked SDPA)
  is reserved for a future batch.

## Composition reference

A V3.2-Lite block at layer L ≥ first_k_dense_replace is:

```
DecoderBlockSpec(
    attn_norm_position = PRE,
    ffn_norm_position  = PRE,
    token_mixer = AttentionSpec(
        kind = AttentionKind.DSA,
        qkv_layout = QKVLayout.MLA_LATENT,
        ...MLA fields...,
        indexer = IndexerSpec(indexer_dim, top_k, warmup_tokens),
    ),
    channel_mixer = MoESpec(
        router_kind = "sigmoid_plus_bias",
        ...V3 MoE fields...,
    ),
    pre_attn_norm = RMSNorm,
    pre_ffn_norm  = RMSNorm,
)
```

## Tests

- `test_v32_block_spec_composes_with_dsa_kind`: the spec composition is
  valid and the indexer field round-trips.
- `test_v32_block_allocates_indexer_modules`: indexer Q/K linear shapes.
- `test_v32_forward_raises_not_implemented`: the deferred-forward contract.
- `test_v32_indexer_spec_rejected_without_dsa_kind`: validation gate.
