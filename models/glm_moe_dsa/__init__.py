"""GLM-MoE-DSA family (GLM-4.5+ / GLM-5) — MLA + DSA Lightning Indexer + V3-style MoE.

Architecture (verified against `transformers/models/glm_moe_dsa/{configuration,
modeling}_glm_moe_dsa.py`, transformers 5.10.x):

- Token mixer: Multi-Latent Attention (MLA) with a DSA Lightning Indexer
  feeding a top-k-selection mask onto the attention scores. Source:
  modeling_glm_moe_dsa.py:268-459 (GlmMoeDsaAttention) and
  :104-228 (GlmMoeDsaIndexer).
- Channel mixer per layer:
    * `mlp_layer_types[i] == "dense"`  -> GlmMoeDsaMLP (standard SwiGLU)
      First 3 layers by default (configuration_glm_moe_dsa.py:127-130).
    * `mlp_layer_types[i] == "sparse"` -> GlmMoeDsaMoE (sigmoid+bias router
      with group routing + shared experts). Source:
      modeling_glm_moe_dsa.py:538-591.

v7 Phase B scope:
- MoE sub-block forward is FULL numerical equivalence vs HF (sigmoid_plus_bias
  router + group routing — the same code path the api MoE already supports
  for V3).
- MLA + DSA attention is SHAPE-ONLY (Phase A reservation:
  `AttentionKind.DSA` raises NotImplementedError in `Attention.forward`).
"""
