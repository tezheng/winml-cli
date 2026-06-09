"""Falcon-7B family — parallel residual flow + MQA + ungated GELU.

v5-phase2 V2 landed the PARALLEL residual block layout topology.
v6 A2 lands the FULL decoder block:
- LayerNorm (with bias) on the shared `input_layernorm`
- MQA attention (n_kv_heads=1) + RoPE SPLIT_HALF, no Linear biases
- Ungated exact-GELU FFN (`dense_h_to_4h -> GELU(approximate='none')
  -> dense_4h_to_h`)
- PARALLEL residual: `x + attention(shared) + mlp(shared)`

Source: `transformers/models/falcon/modeling_falcon.py`:
- FalconAttention.forward (lines 316-427)
- FalconMLP (lines 531-544 — ungated GELU)
- FalconDecoderLayer init (554-578) + forward (580-636) — parallel_attn=True
- FalconConfig defaults (configuration_falcon.py:66-89) for
  Falcon-7B: hidden_size=4544, num_attention_heads=71,
  multi_query=True → num_kv_heads=1, parallel_attn=True,
  new_decoder_architecture=False → num_ln_in_parallel_attn=1
"""
