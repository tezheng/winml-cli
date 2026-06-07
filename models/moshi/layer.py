"""Moshi 7B main-decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a MoshiConfig using only api/ primitives. Does
NOT import transformers.models.moshi.modeling_moshi.

Architectural parity vs Llama 3 / Mistral 7B v0.3: identical PRE-norm
RMSNorm STANDARD_W structure, SPLIT QKV, SwiGLU. The Moshi-specific
quirks are:
  1. **MHA (n_q = n_kv), NOT GQA** — verified configuration_moshi.py:134.
  2. **FUSED gate/up SwiGLU** — `MoshiGatingMLP.fc1` is a single Linear of
     output size `ffn_dim`, split via `.view(B, S, 2, -1)` to (gate, up).
     Source: modeling_moshi.py:368-390.
  3. **No QK-norm, no biases** — modeling_moshi.py:250-265 / 436-447.
  4. **RoPE SPLIT_HALF theta=10000** — modeling_moshi.py:269-365 (Llama-
     copy with rotate_half + cat((freqs,freqs),-1)).
  5. **Eager MoshiAttention does NOT apply SWA**; sliding_window is only
     forwarded into Flash Attention's mask. Our gate runs eager →
     MaskKind.CAUSAL.

HF state-dict keys: q/k/v/o projections live under `MoshiLinear`, which
wraps an inner `self.linear: nn.Linear`. So the HF keys are
`model.layers.{L}.self_attn.{q,k,v,o}_proj.linear.weight` (NOT
`...q_proj.weight`). Source: modeling_moshi.py:250-265.

The MLP keys are `model.layers.{L}.mlp.{fc1,fc2}.weight` because in the
non-flexible (LM-decoder) path MoshiGatingMLP uses plain `nn.Linear`s
directly (modeling_moshi.py:376-378), NOT MoshiLinear.
"""
from __future__ import annotations

import torch

from api import block
from models.moshi import config as _config


def build_moshi_decoder_layer(
    cfg: _config.MoshiConfig,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one Moshi main-decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_moshi_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    prefix: str = "model.layers",
) -> None:
    """Copy HF Moshi main-decoder layer weights into the api DecoderBlock.

    HF tensor names (source: modeling_moshi.py:436-447 for MoshiLinear and
    L376-378 for MoshiGatingMLP `fc1`/`fc2`):
      - self_attn.q_proj.linear.weight (Linear wrapped in MoshiLinear)
      - self_attn.k_proj.linear.weight
      - self_attn.v_proj.linear.weight
      - self_attn.o_proj.linear.weight
      - mlp.fc1.weight   shape [ffn_dim, hidden]   (== gate_up_proj.weight)
      - mlp.fc2.weight   shape [hidden, ffn_dim/2] (== down_proj.weight)
      - input_layernorm.weight
      - post_attention_layernorm.weight
    """
    L = layer_idx
    pfx = f"{prefix}.{L}"
    mapping = {
        f"{pfx}.input_layernorm.weight":          blk.pre_attn_norm.weight,
        f"{pfx}.self_attn.q_proj.linear.weight":  blk.attention.q_proj.weight,
        f"{pfx}.self_attn.k_proj.linear.weight":  blk.attention.k_proj.weight,
        f"{pfx}.self_attn.v_proj.linear.weight":  blk.attention.v_proj.weight,
        f"{pfx}.self_attn.o_proj.linear.weight":  blk.attention.o_proj.weight,
        f"{pfx}.post_attention_layernorm.weight": blk.pre_ffn_norm.weight,
        f"{pfx}.mlp.fc1.weight":                  blk.feedforward.gate_up_proj.weight,
        f"{pfx}.mlp.fc2.weight":                  blk.feedforward.down_proj.weight,
    }
    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")

    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
