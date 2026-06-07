"""DeepSeek-V3-Lite — scaled-down V3 with sigmoid+bias router + group routing.

There is no official DeepSeek-V3-Lite release; V3-full is 671B and
out-of-scope. This package ships a SHAPE/SYNTHETIC configuration that
exercises the V3 architectural deltas vs V2:
- sigmoid+bias router (auxiliary-loss-free balancing).
- Group-limited routing (n_group, topk_group).
- routed_scaling_factor != 1.0 (V3 default 2.5).
- norm_topk_prob = True.
- RoPE: SPLIT_HALF basis (rotate_half) instead of V2's INTERLEAVED.
  Source: modeling_deepseek_v3.py:250-280.
"""
