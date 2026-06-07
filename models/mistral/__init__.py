"""Mistral 7B dense decoder layer.

Covers the Mistral 7B family (v0.1 / v0.2 / v0.3 — all dense GQA SwiGLU).
v0.1/v0.2 use sliding window 4096; v0.3 dropped sliding-window attention
(sliding_window=None in HF config) and is otherwise architecturally identical
to a Llama 3.0-style block with rope_theta=1_000_000.
"""
