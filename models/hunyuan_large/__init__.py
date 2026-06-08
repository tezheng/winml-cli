"""Hunyuan-Large family — Cross-Layer Attention (CLA) per-layer KV pointer.

For v5-phase2 V4 we land the per-layer KV-sharing pattern used by
Hunyuan-Large (tencent/Tencent-Hunyuan-Large, 389B-A52B MoE):
- AttentionSpec.kv_source_layer_offset: when set (e.g. -1), the layer's
  K/V projections are NOT built. The layer's Attention.forward expects
  the K/V from the source layer to be passed via the `kv_shared` arg.

CLA in Hunyuan-Large uses `cla_share_factor=2`: even-indexed layers
(0, 2, 4, ...) compute and own K/V; odd-indexed layers (1, 3, 5, ...)
borrow K/V from their predecessor. This is expressed in
to_block_spec(layer_idx) by setting `kv_source_layer_offset=None` on
even layers and `kv_source_layer_offset=-1` on odd layers.

This is a SHAPE-ONLY family for v5-phase2: a numerical gate against
the full Hunyuan-Large checkpoint requires the open-weights tencent
release (not on HF transformers as of v5.10.2; the HF v1_dense and
v1_moe variants do NOT implement CLA). We exercise the shape +
forward-paths via synthetic weights and verify the borrow semantics
end-to-end.

Apple AFM 3.18B uses cross-BLOCK (not cross-LAYER) sharing — out of
scope here; would be a different axis (block_source_block_idx).

Source: Hunyuan-Large public config.json — `use_cla=true`,
`cla_share_factor=2`.
"""
