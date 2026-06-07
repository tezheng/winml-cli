"""Llama 3 / 3.1 / 3.2 dense decoder layer.

Covers the Meta Llama 3 family (3.0 8B; 3.1 8B; 3.2 1B / 3B) — all dense GQA
SwiGLU stacks differing only in the RoPE flavour:

- Llama 3.0:  vanilla RoPE (rope_type="default"), θ=500_000.
- Llama 3.1/3.2: LLAMA3 smooth scaling (rope_type="llama3") with factor,
  low/high_freq_factor, and original_max_position_embeddings.

See `models/llama3/layer.md` for the canonical architectural notes.
"""
