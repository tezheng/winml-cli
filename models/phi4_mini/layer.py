"""Phi-4-mini-instruct decoder-layer factory + HF weight loader.

Architecturally identical to Phi-3 mini (same FUSED QKV, FUSED gate_up, same
HF tensor names) — we DELEGATE to the Phi3Mini factory + loader.

Verified against the HF Phi-4-mini-instruct config.json (model_type=phi3,
architecture=Phi3ForCausalLM) and modeling_phi3.py.
"""
from __future__ import annotations

from models.phi3_mini import layer as _phi3_layer


# Phi-4-mini reuses the same factory + loader as Phi-3 mini — there are no
# additional axes that differ at the per-layer level. The model-level differences
# (tie_word_embeddings, longrope) are handled by the config and the global
# build path; the decoder layer itself is byte-identical.
def build_phi4_mini_decoder_layer(cfg, layer_idx: int = 0, max_seq=None):
    return _phi3_layer.build_phi3_mini_decoder_layer(
        cfg, layer_idx=layer_idx, max_seq=max_seq,
    )


def load_hf_phi4_mini_layer(blk, hf_state_dict: dict, layer_idx: int) -> None:
    return _phi3_layer.load_hf_phi3_mini_layer(blk, hf_state_dict, layer_idx)
