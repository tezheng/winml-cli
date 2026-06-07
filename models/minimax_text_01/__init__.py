"""MiniMax-Text-01 — DEFERRED — gated weights + 456B params.

Reason: MiniMax-Text-01 (MiniMax 2025) uses the "Lightning" linear
attention variant with a 7:1 (lightning:vanilla-attention) hybrid layer
pattern + MoE. The model is GATED at ~456B parameters, well beyond what
can be tested at small synthetic scale without a substantial config
fixture.

Architecturally requires:
1. `AttentionKind.LINEAR_RETENTION` / `LIGHTNING` — already reserved in
   api/types.py.
2. A new `linear_attention(q, k, v)` op (kernel-feature softmax form).
3. MoE channel mixer — reuse api/feedforward.MoE.

Recommended for the B7+ follow-up after a smaller open MiniMax variant
ships.
"""
