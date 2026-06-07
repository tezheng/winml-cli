"""MiniCPM-3 (openbmb/MiniCPM3-4B) — first MLA implementation in llm-layers.

Architecturally distinguishing features vs Llama 3:
- Multi-head Latent Attention (MLA) — q_lora_rank=768, kv_lora_rank=256,
  qk_nope_head_dim=64, qk_rope_head_dim=32 (qk_head_dim=96), v_head_dim=64.
- LongRoPE — short/long factor tables on the qk_rope_head_dim slice only.
- μP-style per-layer residual scaling: residual + sublayer * (scale_depth /
  sqrt(num_hidden_layers)) — modeling_minicpm.py:941,948 with
  scale_depth=1.4, num_hidden_layers=62.
- scale_emb=12 on input embeddings, and lm_head divides by hidden/dim_model_base
  (these are model-level scalars exposed via spec but not applied in the
  decoder block — same convention as Granite μP).
"""
