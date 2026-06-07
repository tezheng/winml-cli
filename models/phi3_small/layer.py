"""Phi-3-small decoder-layer factory (SHAPE-ONLY for B2b).

Builds a placeholder layer with nn.LayerNorm + the per-Phi-3-small
fused query_key_value Linear + dense Linear + (up_proj, down_proj) MLP.

The forward is NOT implemented for B2b — BlockSparse mask construction +
the GeGELU activation gating ([::2] / [1::2] interleave) + the interleaved
QKV layout are involved enough to merit a separate milestone. The shape
test verifies the underlying nn.Linear shapes against the public config.

Verified against `modeling_phi3_small.py`:
- 154-172  Phi3SmallMLP: up_proj (h → 2*ff), down_proj (ff → h). Forward
           does `gegelu(up_proj(x), limit) * (a_linear + 1)` — i.e. the
           output of up_proj is interleave-split into a_gelu (even indices)
           and a_linear (odd indices), then `quick_gelu(a_gelu) * (a_linear + 1)`.
- 175-260  Phi3SmallSelfAttention: query_key_value Linear of out shape
           (Hq + 2*Hk) * head_dim, dense Linear of out shape h, RotaryEmbedding,
           BlockSparseAttentionLayer.
- 647-696  Phi3SmallDecoderLayer: input_layernorm + self_attn + residual;
           post_attention_layernorm + mlp + residual. Standard PRE-norm.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from models.phi3_small import config as _config


@dataclass
class Phi3SmallLayer:
    """Minimal placeholder holding the per-layer Linear/LayerNorm modules.

    NOT an nn.Module — this is a frozen shape-only skeleton intended to
    document tensor sizes without claiming forward equivalence. Numerical
    gate is deferred per B2b plan.
    """
    input_layernorm: nn.LayerNorm
    post_attention_layernorm: nn.LayerNorm
    query_key_value: nn.Linear
    dense: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear
    is_dense_layer: bool
    layer_idx: int


def build_phi3_small_decoder_layer(
    cfg: _config.Phi3SmallConfig,
    layer_idx: int = 0,
) -> Phi3SmallLayer:
    """Construct the per-layer modules at the shapes implied by `cfg`.

    The forward path is NOT implemented; see layer.md §7 for the
    deferred-work checklist.
    """
    H_q = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    Dh = cfg.head_dim
    h = cfg.hidden_size
    qkv_out = (H_q + 2 * H_kv) * Dh
    return Phi3SmallLayer(
        input_layernorm=nn.LayerNorm(h, eps=cfg.layer_norm_epsilon,
                                     dtype=cfg.dtype),
        post_attention_layernorm=nn.LayerNorm(h, eps=cfg.layer_norm_epsilon,
                                              dtype=cfg.dtype),
        query_key_value=nn.Linear(h, qkv_out,
                                  bias=cfg.attention_bias, dtype=cfg.dtype),
        dense=nn.Linear(h, h, bias=cfg.attention_bias, dtype=cfg.dtype),
        # Phi3SmallMLP: up_proj has 2*intermediate output, down_proj
        # consumes intermediate (post-gating). Source: modeling_phi3_small.py:163-164.
        up_proj=nn.Linear(h, 2 * cfg.intermediate_size,
                          bias=cfg.attention_bias, dtype=cfg.dtype),
        down_proj=nn.Linear(cfg.intermediate_size, h,
                            bias=cfg.attention_bias, dtype=cfg.dtype),
        is_dense_layer=cfg.is_dense_layer(layer_idx),
        layer_idx=layer_idx,
    )
