"""OLMoE 1B-7B (Allen Institute for AI).

PRE-norm decoder with Llama-style envelope + FULL_HDH QK-norm (q_norm on
all H_q*Dh dims, k_norm on H_kv*Dh dims) PRE-RoPE, then a 64-expert MoE
with top-8 softmax routing and optional ``norm_topk_prob`` (default
False).

Source-grounded at:
- `transformers/models/olmoe/modeling_olmoe.py:220-298` OlmoeAttention.
- `transformers/models/olmoe/modeling_olmoe.py:301-359` MoE block + router.
- `transformers/models/olmoe/modeling_olmoe.py:378-416` decoder layer.
- `transformers/models/olmoe/configuration_olmoe.py:23-89` config defaults.
"""
