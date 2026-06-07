"""TinyLlama 1.1B config + adapter to api specs.

TinyLlama uses the Llama model_type. Verified against
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` HF config.json:
- 22 hidden layers (NOT 32 like Llama 2 7B).
- hidden_size=2048, intermediate_size=5632.
- n_q=32, n_kv=4, head_dim=64 (GQA, NOT MHA — research v3 §8.4 surprise).
- RoPE rope_type="default", theta=10000.0 (vanilla).
- RMSNorm STANDARD_W, eps=1e-5.
- SwiGLU FFN, no biases (attention_bias=False, mlp_bias=False).
- tie_word_embeddings=False.

Because TinyLlama is loaded under HF's Llama infrastructure, we delegate to
`models.llama3.config.Llama3Config` underneath, but expose a thin TinyLlamaConfig
wrapper for clarity at the call site.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from models.llama3 import config as _llama3


@dataclass(frozen=True)
class TinyLlamaConfig:
    """Thin wrapper around Llama3Config — TinyLlama uses Llama default RoPE."""
    inner: _llama3.Llama3Config

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "TinyLlamaConfig":
        return cls(inner=_llama3.Llama3Config.from_hf_dict(hf))

    # Pass-throughs for the fields the call sites care about.
    @property
    def hidden_size(self) -> int: return self.inner.hidden_size
    @property
    def num_attention_heads(self) -> int: return self.inner.num_attention_heads
    @property
    def num_key_value_heads(self) -> int: return self.inner.num_key_value_heads
    @property
    def head_dim(self) -> int: return self.inner.head_dim
    @property
    def intermediate_size(self) -> int: return self.inner.intermediate_size
    @property
    def num_hidden_layers(self) -> int: return self.inner.num_hidden_layers
    @property
    def rope_theta(self) -> float: return self.inner.rope_theta
    @property
    def rms_norm_eps(self) -> float: return self.inner.rms_norm_eps
    @property
    def vocab_size(self) -> int: return self.inner.vocab_size
    @property
    def max_position_embeddings(self) -> int: return self.inner.max_position_embeddings
    @property
    def tie_word_embeddings(self) -> bool: return self.inner.tie_word_embeddings
    @property
    def dtype(self) -> torch.dtype: return self.inner.dtype

    def to_block_spec(self, layer_idx: int = 0):
        return self.inner.to_block_spec(layer_idx=layer_idx)
