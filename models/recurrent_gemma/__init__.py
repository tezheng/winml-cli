"""RecurrentGemma (google/recurrentgemma-2b) — DEFERRED — pending B7+ follow-up.

Reason: RecurrentGemma uses the **Griffin/Hawk gated linear recurrent unit
(LRU)** — a completely DIFFERENT recurrence form from Mamba's selective
scan. Specifically:

- The "gated linear recurrence" has:
    h_t = a_t * h_{t-1} + (1 - a_t) * v_t      # forgetting gate
    y_t = q_t * h_t                            # query gate
  where a_t, v_t, q_t are all input-dependent. This is closer to the
  LRU + gating ideas of Orvieto et al. 2023 than the SSM B/C/dt formulation.
- The block ALSO contains a small 1D causal conv (kernel 4, similar to
  Mamba) and a temporal block dimension that's not the SSM hidden state.
- Some layers are LOCAL ATTENTION (sliding window), not LRU.

Source: transformers/models/recurrent_gemma/modeling_recurrent_gemma.py
(see `RGLRU`, `Conv1D`, and `RecurrentGemmaSdpaAttention`).

The 5e-4 atol gate is achievable but requires a separate `gated_linear_recurrence`
op in api/ops.py. Implementing it requires reading the entire RGLRU module.

Recommended for the B7+ follow-up. Weights are gated on HF (try
`unsloth/recurrentgemma-2b` mirror for the test).
"""
