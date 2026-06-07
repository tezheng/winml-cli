"""Ministral dense decoder layer with per-layer interleaved SWA dispatch.

Ministral is Mistral-shaped (GQA, RoPE SPLIT_HALF, RMSNorm STANDARD_W, SwiGLU,
no biases, no QK-norm) but adds a per-layer ``layer_types`` field that selects
either ``"sliding_attention"`` or ``"full_attention"`` for each decoder layer.

The canonical pattern in `mistralai/Ministral-8B-Instruct-2410` is a 1:3
alternation (one full layer followed by three sliding layers, repeated). This
is the third heterogeneous-mask alternation pattern in the corpus after
Gemma 2 (1:1) and Gemma 3 (5:1 sliding-to-full).

Per-layer dispatch lives in ``MinistralConfig.to_block_spec(layer_idx)``.
"""
