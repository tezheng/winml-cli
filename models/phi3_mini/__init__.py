"""Phi-3-mini-4k-instruct decoder layer (Microsoft).

B2a covers the Phi-3 mini 4k variant (`microsoft/Phi-3-mini-4k-instruct`):
3.8B params, dense, 32 layers, hidden=3072, MHA n_q=n_kv=32, head_dim=96,
SWA(2047), fused QKV + fused gate_up, plain RoPE θ=10000 (NO LongRoPE — the
4k variant has rope_scaling=null in its config).

LongRoPE quirks are covered by the Phi-4-mini gate (separate package).

Verified against transformers/models/phi3/modeling_phi3.py.
"""
