"""DeepSeek-V3-MoE thin wrapper around the V3-Lite MoE module.

V3-Lite (in ``models/deepseek_v3_lite``) is the synthetic small-scale config
exercised in B5; this package presents the same MoE machinery in the shape
of a FULL V3 decoder layer: MLA (q_lora_rank=1536) + sigmoid+bias router +
group routing (n_group=8, topk_group=4) + shared experts (n_shared=1) +
``norm_topk_prob=True`` + ``routed_scaling_factor=2.5``.

The 671B weights of DeepSeek-V3 are not exercised numerically — the gate
ships as a synthetic-weight shape test (and a config round-trip) only.
"""
