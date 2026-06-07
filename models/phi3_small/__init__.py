"""Phi-3-small (microsoft/Phi-3-small-8k-instruct) — first BlockSparse model.

SHAPE-ONLY skeleton (B2b). Forward path raises NotImplementedError for the
BlockSparse mask kind; the numerical gate is deferred to a future milestone.

Architectural facts (from config.json and modeling_phi3_small.py):
- Dense layers every 2 (`dense_attention_every_n_layers=2`) — odd layer
  indices (1, 3, 5, ...) are dense; even layer indices (0, 2, 4, ...) are
  BlockSparse.
- BlockSparse: block_size=64, vert_stride=8, num_local_blocks=16,
  homo_head_pattern=False (each head has its own vertical-stride seed).
- Hidden=4096, n_heads=32, n_kv_heads=8 (GQA), head_dim=128.
- LayerNorm (NOT RMSNorm), eps=1e-5.
- GeGELU activation with limit=20.0; FFN intermediate=14336 (chunked from
  up_proj output 2*intermediate via [::2] / [1::2] gating).
- mup_*: width_multiplier=8.0, embedding_multiplier=10.0, attn_multiplier=1.0,
  use_scaling=True. softmax_scale = mup_attn_multiplier / head_dim (not the
  usual 1/sqrt). Source: modeling_phi3_small.py:200-206.
- RoPE base=1_000_000, position_scale=1.0.
- Interleaved fused QKV: groups of (m=Hq/Hk Q heads, 1 K head, 1 V head) — a
  different fusion than Phi-3-mini's. Source: modeling_phi3_small.py:265-279.
"""
