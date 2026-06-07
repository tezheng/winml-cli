"""Mixtral 8x7B — canonical softmax-router MoE family.

Layout: GQA + SwiGLU experts + top-2 softmax routing with ALWAYS-on top-k
normalization. No shared experts, no group routing.

Source-grounded at:
- `transformers/models/mixtral/modeling_mixtral.py:101-135` (router + MoE block)
- `transformers/models/mixtral/modeling_mixtral.py:295-389` (attention + decoder layer)
- `transformers/models/mixtral/configuration_mixtral.py:25-93` (config defaults).
"""
