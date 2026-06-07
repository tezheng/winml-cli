"""Hymba — DEFERRED — pending B7+ exotic-hybrid follow-up.

Reason: Hymba (NVIDIA 2024) uses PARALLEL Mamba ‖ Attention heads IN THE
SAME BLOCK — both run on the same input and their outputs are concatenated
along the head dim before o_proj. This is structurally distinct from
ALTERNATING (Granite-4-H, Jamba) and SEQUENTIAL (Samba) hybrids.

Implementing requires:
1. A new `HybridParallelMixer` building block that runs SSM + Attention
   on the SAME input.
2. Extending `DecoderBlockSpec.token_mixer` to accept a
   tuple[AttentionSpec, SSDSpec] — already reserved in v3 §5.2.5.

The v3 spec calls this `TokenMixerKind.HYBRID_PARALLEL`. We have it
enumerated but not implemented.

Recommended for the B7+ follow-up after a production-grade Hymba
checkpoint is available.
"""
