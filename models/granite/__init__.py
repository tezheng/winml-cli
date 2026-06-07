"""Granite 3.x dense decoder layer (IBM).

B2a covers the dense Granite 3.0 / 3.1 / 3.2 / 3.3 family (model_type="granite",
GraniteForCausalLM in HF). Architecturally Llama-shaped (PRE-norm RMSNorm
STANDARD_W, SPLIT QKV, GQA SwiGLU, no biases, no QK-norm), with four μP scalar
overrides that distinguish it from a vanilla Llama-3 block:

- `attention_multiplier` — overrides the default 1/sqrt(head_dim) softmax scale.
- `residual_multiplier`  — multiplies sublayer outputs BEFORE the residual add.
- `embedding_multiplier` — scales the input embedding before the decoder stack
  (model-level, lives on the Block spec for assembly metadata).
- `logits_scaling`       — divides the final lm_head output (model-level).

Granite 4.1-Tiny is GraniteMoEHybrid (MoE + Mamba) and is OUT of B2a scope —
it lands in the MoE / SSM batch. B2a focuses on dense Granite 3.x only.

Verified against transformers/models/granite/modeling_granite.py.
"""
