"""Jamba (ai21labs/Jamba-v0.1) — DEFERRED — pending B7+ follow-up.

Reason: Jamba uses **Mamba-1** selective scan, NOT Mamba-2 SSD. Specifically:
- Mamba-1 has a LEARNED dt projection (`dt_proj: Linear(time_step_rank ->
  d_inner)`) instead of Mamba-2's per-head dt scalar in `in_proj`.
- A_log is per-channel `[d_inner]`, not per-head `[num_heads]`.
- D is per-channel `[d_inner]`, not per-head `[num_heads]`.
- The selective scan is the original recurrent form, NOT the SSD
  chunk-parallel form we implemented in api/ops.selective_scan.

Source: transformers/models/jamba/modeling_jamba.py:202-460 (JambaMambaMixer)
and configuration_jamba.py:75-86 (no `mamba_n_heads`/`mamba_n_groups`/
`mamba_chunk_size`/`mamba_d_head` — the structural absences confirm
Mamba-1 form).

Implementing Mamba-1 requires:
1. A separate `selective_scan_mamba1(x, A, B, C, D, dt)` op that runs the
   per-time-step recurrence (NOT the SSD chunk form).
2. A `Mamba1Mixer` class in api/ssm.py with `dt_proj`, full-d_inner A_log,
   and per-channel D.

Recommended for the B7+ follow-up. Spec: research/05-kvcache-attention.v3.md
section on Mamba-1.
"""
