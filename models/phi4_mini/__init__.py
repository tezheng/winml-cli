"""Phi-4-mini-instruct decoder layer (Microsoft).

B2a covers Phi-4-mini-instruct (3.84B params, dense), which uses the Phi3
HF code path (model_type="phi3", architecture=Phi3ForCausalLM) with a richer
RoPE/rotary configuration:

- GQA: n_q=24, n_kv=8, head_dim=128.
- partial_rotary_factor=0.75 — only the first 75% of head_dim rotates
  (head_dim=128 → 96 channels rotate, 32 pass through).
- LongRoPE with original_max_position_embeddings=4096,
  max_position_embeddings=131072. The 48-element short_factor / long_factor
  vectors live in rope_scaling.
- SWA: sliding_window=None (Phi-4-mini drops SWA).

Since the architecture is byte-identical to Phi-3 (FUSED QKV, FUSED gate_up,
no QK-norm, no biases, RMSNorm STANDARD_W), we DELEGATE to the Phi3Mini
factory + loader; the only difference is the config field set (which is
auto-handled by Phi3MiniConfig's longrope branch).

Verified against the HF Phi-4-mini-instruct config and modeling_phi3.py.
"""
