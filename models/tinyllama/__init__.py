"""TinyLlama 1.1B decoder layer.

TinyLlama uses the Llama architecture (HF `model_type="llama"`) and is loaded
through `transformers.models.llama.modeling_llama`. Architecturally it is a
22-layer GQA SwiGLU stack with:
- n_q=32, n_kv=4, head_dim=64 (a SURPRISE per the research notes — TinyLlama
  is GQA not MHA, despite the small size; verified against HF config.json).
- RoPE default theta=10000 (vanilla, no LLAMA3 scaling).
- RMSNorm STANDARD_W, SwiGLU, no biases.
- tie_word_embeddings=False.

Because the on-disk weights are saved under the Llama tensor-name convention,
TinyLlama re-uses `models.llama3.layer.load_hf_llama3_layer` directly. The
config still goes through a thin `TinyLlamaConfig` wrapper for clarity.
"""
