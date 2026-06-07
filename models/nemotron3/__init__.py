"""Nemotron-3 Super/Ultra — DEFERRED — pending B7+ follow-up.

Reason: Nemotron-3 Super/Ultra (NVIDIA 2026 Q2) are very large
(hundreds of billions of parameters) hybrid Mamba-2 + Transformer + MoE
models. The numerical gate is achievable on a small synthetic config but
the full model wiring (per-layer MoE + Mamba dispatch + iRoPE-like
attention pattern + several tens of distinct tensor groups) requires
substantial implementation time.

Architecturally most close to Granite-4-H + Mixtral. Should land by
composing existing infrastructure:
- Mamba-2 layers: api/ssm.Mamba2Mixer (DONE in B7).
- Attention layers: standard GQA + RoPE.
- MoE channel mixer: api/feedforward.MoE (DONE in B5).

Recommended for the B7+ follow-up.
"""
