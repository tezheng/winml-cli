"""Llama 4 Scout (17B-16E) — per-layer iRoPE dispatch + interleaved MoE.

Llama 4 Scout is the first production-scale heterogeneous-attention model:
- iRoPE per-layer dispatch: alternating RoPE / NoPE layers via
  ``config.no_rope_layers``. The IR maps this to ``AttentionSpec.rope = None``
  on NoPE layers — the same lever that SmolLM3 uses.
- ``layer_types`` per-layer = "chunked_attention" on NoPE layers, else
  "full_attention". Scout uses "full_attention" everywhere on the published
  16E variant due to the empty ``no_rope_layers`` list deferring to default.
- QK-norm (L2Norm) only on RoPE layers (config.use_qk_norm AND use_rope).
- Attention temperature tuning (log-position scale on Q) on NoPE layers
  only — deferred to a later milestone, see Quirks §6 in layer.md.
- MoE at every layer in the 16E variant (interleave_moe_layer_step=1) —
  full MoE wiring lands in B6. B4 ships the per-layer ATTENTION spec only;
  layer factories build ``api.attention.Attention`` modules directly and
  the FFN-side wiring is deferred.

Numerical gate is shape-only (weights gated AND >200 GB total). The
synthetic-weight shape test exercises the iRoPE per-layer pattern
end-to-end on the attention module.
"""
