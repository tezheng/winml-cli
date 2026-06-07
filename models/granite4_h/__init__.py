"""Granite 4 H (hybrid Mamba-2 + attention) layer implementation via api/.

Reference HF model: `ibm-granite/granite-4.0-h-micro` (open).
Architecture source: `transformers.models.granitemoehybrid.modeling_granitemoehybrid`.

Granite-4-H micro is the smallest hybrid: 40 layers in a 5:1 pattern
(`mamba×5, attention×1`), no MoE, NoPE attention, residual μP scaling.

Per-layer dispatch is driven by `layer_types[layer_idx]`.
"""
