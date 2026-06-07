"""Mamba-3 — DEFERRED — research scales only at HF as of B7 cutoff.

Reason: Mamba-3 introduces COMPLEX-VALUED state and MIMO decoding (Mar
2026), with no production-scale open checkpoint published on HF at the
B7 milestone date. The v3 design spec calls out `is_complex_state` and
`n_mimo_outputs` SSDSpec extensions, but no HF model file means the
numerical gate target is missing.

Reserved namespace; revisit after a production-scale Mamba-3 ships.

Source: v3 design spec §5.2.5 (Mamba3Spec), research/08-recent-releases-2026q2.md.
"""
