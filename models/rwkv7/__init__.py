"""RWKV-7 (RWKV/rwkv-7-world-3b) — DEFERRED — pending B7+ follow-up.

Reason: RWKV-7 is the "Goose" architecture — a learned-state-update
recurrence MORE DIFFERENT from Mamba than RecurrentGemma is. Each block
uses a Test-Time-Training-flavoured WKV recurrence where the per-token
state is itself a learnable matrix updated by an outer-product write,
NOT the SSM B*x*C contraction.

Source: RWKV-7 paper "RWKV-7 'Goose' with Expressive Dynamic State
Evolution" (Peng et al. 2025) and HF model card
`RWKV/rwkv-7-world-3b`. As of transformers cutoff, the modeling file
either does not exist or lives outside `transformers/models/` (custom
`trust_remote_code=True` model class).

Implementing requires:
1. A `wkv_recurrence` op (per-channel learned-state outer-product update).
2. Time-mix and channel-mix shifts (Conv1D with kernel=2 + token-shift).
3. Group-norm before the time-mix output projection.

Recommended for the B7+ follow-up after RecurrentGemma lands the
gated-linear-recurrence ops.
"""
