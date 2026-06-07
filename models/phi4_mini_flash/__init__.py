"""Phi-4-mini-flash (Samba) — DEFERRED — pending B7+ follow-up.

Reason: Samba is SEQUENTIAL Mamba-1 ‖ SWA-attention hybrid: each block
has a Mamba sublayer FOLLOWED by an attention sublayer (NOT alternating
layers). It also uses Mamba-1 form (per the original Samba paper,
Ren et al. 2024).

This requires:
1. The Mamba-1 op (see models/jamba/__init__.py deferral note).
2. A new block structure where BOTH a token mixer (SSM) AND a second
   token mixer (attention) run in sequence within ONE decoder block,
   plus the standard FFN sublayer.

We currently model `DecoderBlock` as ONE token mixer + ONE channel mixer.
Adding a sequential dual-token-mixer block requires extending
`DecoderBlockSpec` with an optional second token mixer.

Recommended for the B7+ follow-up after Mamba-1 lands (Jamba dependency).
"""
