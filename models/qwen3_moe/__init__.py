"""Qwen3-MoE family (e.g. Qwen3-30B-A3B, Qwen3-235B-A22B).

Qwen3-style decoder (PRE-norm + QK-norm PRE-RoPE PER_HEAD_DH + SwiGLU) with
per-layer dispatch between dense MLP (using config.intermediate_size) and
SparseMoeBlock (using config.moe_intermediate_size).

Source-grounded at:
- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:126-195` Qwen3MoeAttention
- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:198-272` Qwen3MoeMLP + Experts + Router
- `transformers/models/qwen3_moe/modeling_qwen3_moe.py:275-353` SparseMoeBlock + DecoderLayer
- `transformers/models/qwen3_moe/configuration_qwen3_moe.py:25-120` config defaults.
"""
