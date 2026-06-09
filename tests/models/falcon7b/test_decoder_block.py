"""Falcon-7B full decoder block numerical gate (v6 A2).

Tests the full PARALLEL residual Falcon-7B decoder block end-to-end
against HF's FalconDecoderLayer with parallel_attn=True,
new_decoder_architecture=False, num_ln_in_parallel_attn=1.

Source: `transformers/models/falcon/modeling_falcon.py:554-636`.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.falcon.configuration_falcon import FalconConfig  # noqa: E402
from transformers.models.falcon.modeling_falcon import (  # noqa: E402
    FalconDecoderLayer, FalconRotaryEmbedding,
)

from api import kvcache, specs, types  # noqa: E402
from models.falcon7b import config as falcon_config, layer as falcon_layer  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _build_hf_falcon7b_block(hidden_size: int, n_heads: int, max_seq: int):
    cfg = FalconConfig(
        vocab_size=256, hidden_size=hidden_size,
        num_hidden_layers=1, num_attention_heads=n_heads,
        num_kv_heads=1,                                # MQA
        max_position_embeddings=max_seq,
        bias=False, parallel_attn=True,
        new_decoder_architecture=False, multi_query=True,
        alibi=False,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        layer_norm_epsilon=1e-5,
        hidden_dropout=0,
        attention_dropout=0,
        ffn_hidden_size=hidden_size * 4,
        activation="gelu",
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    block = FalconDecoderLayer(cfg, layer_idx=0)
    block.eval()
    rot = FalconRotaryEmbedding(cfg)
    rot.eval()
    return cfg, block, rot


def _build_api_falcon7b_block(hidden_size: int, n_heads: int, max_seq: int):
    f7b = falcon_config.Falcon7BConfig(
        hidden_size=hidden_size, num_attention_heads=n_heads,
        num_kv_heads=1, num_hidden_layers=1,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=max_seq,
    )
    return falcon_layer.build_falcon7b_decoder_layer(f7b)


def _copy_weights(hf_block: FalconDecoderLayer, api_block, hidden_size: int,
                  n_heads: int) -> None:
    Hq = n_heads
    Dh = hidden_size // n_heads
    W = hf_block.self_attention.query_key_value.weight.detach().clone()
    assert W.shape == ((Hq + 2) * Dh, hidden_size)
    Wq = W[: Hq * Dh, :]
    Wk = W[Hq * Dh : (Hq + 1) * Dh, :]
    Wv = W[(Hq + 1) * Dh :, :]
    with torch.no_grad():
        api_block.pre_attn_norm.weight.copy_(hf_block.input_layernorm.weight)
        api_block.pre_attn_norm.bias.copy_(hf_block.input_layernorm.bias)
        api_block.attention.q_proj.weight.copy_(Wq)
        api_block.attention.k_proj.weight.copy_(Wk)
        api_block.attention.v_proj.weight.copy_(Wv)
        api_block.attention.o_proj.weight.copy_(hf_block.self_attention.dense.weight)
        api_block.feedforward.up_proj.weight.copy_(hf_block.mlp.dense_h_to_4h.weight)
        api_block.feedforward.down_proj.weight.copy_(hf_block.mlp.dense_4h_to_h.weight)


def test_falcon7b_decoder_block_matches_hf():
    hidden_size, n_heads, max_seq = 256, 8, 32
    cfg_hf, hf_block, rot = _build_hf_falcon7b_block(hidden_size, n_heads, max_seq)
    api_block = _build_api_falcon7b_block(hidden_size, n_heads, max_seq)
    _copy_weights(hf_block, api_block, hidden_size, n_heads)
    api_block.eval()

    torch.manual_seed(1)
    B, S = 1, 6
    x = torch.randn(B, S, hidden_size)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(x, pos)
        # HF builds an additive 4D causal mask for non-flash attention.
        causal_mask = torch.full((1, 1, S, S), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        hf_out, _ = hf_block(
            x, alibi=None, attention_mask=causal_mask,
            position_ids=pos, position_embeddings=(cos, sin),
            layer_past=None, use_cache=False,
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=1, head_dim=hidden_size // n_heads,
        max_seq=max_seq,
    )
    with torch.no_grad():
        api_out = api_block(x, position_ids=pos.squeeze(0),
                            cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Falcon-7B decoder block max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    print(f"Falcon-7B decoder-block max_abs_diff={max_abs_diff:.2e}")


def test_falcon7b_layer_norm_has_bias():
    cfg = falcon_config.Falcon7BConfig(
        hidden_size=256, num_attention_heads=8, num_kv_heads=1,
        num_hidden_layers=1, intermediate_size=1024,
        max_position_embeddings=32,
    )
    api_block = falcon_layer.build_falcon7b_decoder_layer(cfg)
    assert api_block.pre_attn_norm.bias is not None


def test_falcon7b_block_layout_parallel():
    cfg = falcon_config.Falcon7BConfig(
        hidden_size=256, num_attention_heads=8, num_kv_heads=1,
        num_hidden_layers=1, intermediate_size=1024,
        max_position_embeddings=32,
    )
    spec = cfg.to_block_spec()
    assert spec.block_layout == types.BlockLayout.PARALLEL
    assert spec.pre_ffn_norm is None
