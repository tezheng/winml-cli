"""Phi-4-mini-instruct model config + adapter to api specs.

Phi-4-mini uses the Phi3ForCausalLM HF code path (model_type="phi3") with
a richer rotary configuration. Architecturally byte-identical to Phi-3
(FUSED QKV, FUSED gate_up, no QK-norm, no biases) — we delegate to
Phi3MiniConfig for the heavy lifting.

Key Phi-4-mini facts (verified against the HF Phi-4-mini-instruct config.json
and modeling_phi3.py):
- GQA: n_q=24, n_kv=8, head_dim=128 (3072/24 = 128).
- partial_rotary_factor=0.75 → 96 channels of 128 rotate.
- LongRoPE: original=4096, max=131072 (factor=32).
- sliding_window=None — Phi-4-mini drops SWA.
- tie_word_embeddings=True (lm_head shares weights with embed_tokens).
- attention_bias=False, mlp_bias=False, no QK-norm.
"""
from __future__ import annotations

from models.phi3_mini import config as _phi3_config


# Phi-4-mini reuses the Phi3MiniConfig dataclass — the from_hf_dict path
# already handles partial_rotary_factor, longrope short/long factor, and
# original_max_position_embeddings, all of which Phi-4 sets in its config.
Phi4MiniConfig = _phi3_config.Phi3MiniConfig


def from_hf_dict(hf: dict):
    """Phi-4-mini factory — delegates to Phi3MiniConfig.from_hf_dict."""
    return _phi3_config.Phi3MiniConfig.from_hf_dict(hf)
