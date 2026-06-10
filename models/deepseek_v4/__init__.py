"""DeepSeek-V4 family — hash-routed MoE + per-layer CSA / HCA attention.

Architecture (verified against `transformers/models/deepseek_v4/modeling_deepseek_v4.py`,
transformers 5.10.x):

- The first `num_hash_layers` MoE layers route via a frozen `tid2eid[input_ids]`
  lookup (DeepseekV4HashRouter, modeling_deepseek_v4.py:1050-1078). Remaining
  MoE layers use the standard sigmoid-scored top-k (DeepseekV4TopKRouter,
  modeling_deepseek_v4.py:1029-1047).
- Per-layer attention dispatches on `layer_types[i]` between
  `sliding_attention`, `compressed_sparse_attention`, `heavily_compressed_attention`
  (modeling_deepseek_v4.py:751-869). The CSA and HCA branches add a compressor
  feeding extra K/V entries onto the attention axis (paper §2.3.1 / §2.3.2).
- The decoder block carries Manifold-Constrained Hyper-Connections (mHC) which
  keep `hc_mult` parallel residual streams (modeling_deepseek_v4.py:872-948,
  1101-1149) — these wrap the standard attention + MoE sublayers and are
  out-of-scope for the llm-layers IR.

v7 Phase B scope (v7-phase-b-complete):
- Hash MoE sub-block forward is FULLY numerical (v7 Phase A P1 wired
  `router_kind="hash"` + `_HashRouter`).
- CSA / HCA attention is shape-only (v7 Phase A P2 raised `NotImplementedError`).
- The mHC residual streams + grouped output projection + attention sinks are
  NOT modeled. The "layer" we expose here is therefore a STRIPPED V4 decoder
  layer: a single residual stream, no mHC, no sinks, with the V4 hash MoE
  block as channel mixer. The numerical gate verifies the hash MoE matches HF
  `DeepseekV4SparseMoeBlock(is_hash=True)` bit-for-bit; the attention sub-block
  ships shape-only.
"""
