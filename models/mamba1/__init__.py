"""Mamba-1 family (state-spaces).

v6 B1 lands the Mamba-1 selective-scan mixer + per-channel A_log/D +
learned dt_proj. The canonical reference is `state-spaces/mamba-130m-hf`
(small, open, downloadable).

Architecture key differences from Mamba-2 (SSD):
- A_log shape [d_inner, d_state] (per-CHANNEL × state, not per-head).
- D shape [d_inner] (per-channel).
- Learned dt_proj: Linear(dt_rank, d_inner, bias=True) — both weight and
  bias contribute (HF inverts softplus to init the bias, so the bias is
  the dominant pre-softplus offset).
- in_proj outputs 2*d_inner (chunk into hidden_states and gate).
- Sequential time-step scan, NOT chunk-parallel SSD.
- gate applied AFTER scan: `scan_output * silu(gate)`.
- No gated norm before out_proj (unlike Mamba-2 which has MambaRMSNormGated).

Source: `transformers/models/mamba/modeling_mamba.py:58-363`.
"""
