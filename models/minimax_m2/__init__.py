"""MiniMax-M2 family — standard attention + sigmoid+bias MoE (no shared, no groups).

Architecture (verified against `transformers/models/minimax_m2/{configuration,
modeling}_minimax_m2.py`, transformers 5.10.x):

- Token mixer: STANDARD attention with q/k norms applied to the FULL
  hidden-dim projection (FULL_HDH) BEFORE view+transpose, i.e. PRE-RoPE.
  Source: modeling_minimax_m2.py:295-330 (MiniMaxM2Attention init/forward,
  q_norm and k_norm sized num_heads*head_dim and num_kv_heads*head_dim
  respectively).
- Channel mixer: sigmoid router + e_score_correction_bias top-k choice,
  with weights gathered from the BIAS-FREE sigmoid output. No group routing.
  No shared experts. Source: modeling_minimax_m2.py:46-124
  (TopKRouter + SparseMoeBlock).
- RMSNorm: standard `w * x_normed` form (modeling_minimax_m2.py:127-145).

v7 Phase B scope: FULL DECODER LAYER numerical equivalence vs HF at
atol=5e-4 on a synthetic-mini config. This family exercises the v6
QK-norm FULL_HDH + sigmoid_plus_bias MoE WITHOUT group routing — the
latter is a pre-existing capability of `_SigmoidRouter` that hadn't been
exercised by any other family.

The Raschka v7 confirmation at modeling_minimax_m2.py:296 — M2 is plain
attention, NOT lightning attention. No new building blocks are required.
"""
