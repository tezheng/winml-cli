"""MPT full decoder block numerical gate (v6 A1).

Tests the full MPT decoder block: LayerNorm(no-bias) + ALiBi-MHA +
LayerNorm(no-bias) + ungated GELU FFN.

Source: `transformers/models/mpt/modeling_mpt.py:158-212` (MptBlock).
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.mpt.modeling_mpt import (  # noqa: E402
    MptBlock, MptConfig as HFMptConfig, build_mpt_alibi_tensor,
)

from api import kvcache, specs, types  # noqa: E402
from models.mpt import config as mpt_config, layer as mpt_layer  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _build_hf_mpt_block(hidden_size: int, n_heads: int, max_seq: int):
    cfg = HFMptConfig(
        d_model=hidden_size,
        n_heads=n_heads,
        n_layers=1,
        expansion_ratio=4,
        max_seq_len=max_seq,
        vocab_size=64,
    )
    # Ensure dropout disabled (eval-time anyway, but harmless).
    cfg.attn_config.attn_pdrop = 0
    torch.manual_seed(0)
    block = MptBlock(cfg, layer_idx=0)
    block.eval()
    return cfg, block


def _build_api_mpt_block(hidden_size: int, n_heads: int, max_seq: int):
    cfg = mpt_config.MptConfig(
        hidden_size=hidden_size,
        num_attention_heads=n_heads,
        num_hidden_layers=1,
        max_position_embeddings=max_seq,
        vocab_size=64,
        layer_norm_epsilon=1e-5,
    )
    return mpt_layer.build_mpt_decoder_layer(cfg)


def _copy_weights(hf_block: MptBlock, api_block, hidden_size: int) -> None:
    """Copy HF MptBlock parameters into the api-assembled DecoderBlock.

    HF state-dict layout:
        norm_1.weight    (norm_1.bias = None, removed in __init__)
        attn.Wqkv.weight (fused [3*H, H])
        attn.out_proj.weight
        norm_2.weight    (norm_2.bias = None)
        ffn.up_proj.weight
        ffn.down_proj.weight
    """
    H = hidden_size
    with torch.no_grad():
        api_block.pre_attn_norm.weight.copy_(hf_block.norm_1.weight)
        api_block.pre_ffn_norm.weight.copy_(hf_block.norm_2.weight)
        Wq, Wk, Wv = hf_block.attn.Wqkv.weight.chunk(3, dim=0)
        api_block.attention.q_proj.weight.copy_(Wq)
        api_block.attention.k_proj.weight.copy_(Wk)
        api_block.attention.v_proj.weight.copy_(Wv)
        api_block.attention.o_proj.weight.copy_(hf_block.attn.out_proj.weight)
        api_block.feedforward.up_proj.weight.copy_(hf_block.ffn.up_proj.weight)
        api_block.feedforward.down_proj.weight.copy_(hf_block.ffn.down_proj.weight)


def test_mpt_decoder_block_matches_hf():
    """Full MPT decoder block end-to-end vs HF MptBlock."""
    hidden_size, n_heads, max_seq = 64, 4, 32
    cfg_hf, hf_block = _build_hf_mpt_block(hidden_size, n_heads, max_seq)
    api_block = _build_api_mpt_block(hidden_size, n_heads, max_seq)
    _copy_weights(hf_block, api_block, hidden_size)
    api_block.eval()

    torch.manual_seed(1)
    B, S = 1, 8
    x = torch.randn(B, S, hidden_size)
    alibi = build_mpt_alibi_tensor(n_heads, max_seq, alibi_bias_max=8)
    causal_mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool), diagonal=1,
    )[None, None, :, :]

    with torch.no_grad():
        hf_out, _ = hf_block(
            x, position_bias=alibi,
            attention_mask=causal_mask,
            layer_past=None, use_cache=False,
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=n_heads, head_dim=hidden_size // n_heads,
        max_seq=max_seq,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_block(x, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"MPT decoder block max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    print(f"MPT decoder-block max_abs_diff={max_abs_diff:.2e}")


def test_mpt_decoder_block_non_pow2_heads():
    """Non-power-of-2 head count."""
    hidden_size, n_heads, max_seq = 60, 6, 32
    cfg_hf, hf_block = _build_hf_mpt_block(hidden_size, n_heads, max_seq)
    api_block = _build_api_mpt_block(hidden_size, n_heads, max_seq)
    _copy_weights(hf_block, api_block, hidden_size)
    api_block.eval()

    torch.manual_seed(2)
    B, S = 1, 6
    x = torch.randn(B, S, hidden_size)
    alibi = build_mpt_alibi_tensor(n_heads, max_seq, alibi_bias_max=8)
    causal_mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool), diagonal=1,
    )[None, None, :, :]
    with torch.no_grad():
        hf_out, _ = hf_block(
            x, position_bias=alibi,
            attention_mask=causal_mask, layer_past=None, use_cache=False,
        )
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=n_heads, head_dim=hidden_size // n_heads,
        max_seq=max_seq,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_block(x, position_ids=pos, cache=cache, start_pos=0)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"MPT non-pow2 max_abs_diff={max_abs_diff:.6e}"
    )


def test_mpt_layer_norm_has_no_bias():
    """The api block should NOT allocate norm bias (MPT sets bias=None)."""
    cfg = mpt_config.MptConfig.mpt_7b()
    # Use small max_seq to avoid OOM.
    cfg_small = mpt_config.MptConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
    )
    api_block = mpt_layer.build_mpt_decoder_layer(cfg_small)
    assert api_block.pre_attn_norm.bias is None
    assert api_block.pre_ffn_norm.bias is None


def test_mpt_ffn_has_no_gate_proj():
    """GELU_ONLY FFN does NOT allocate gate_proj."""
    cfg = mpt_config.MptConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
    )
    api_block = mpt_layer.build_mpt_decoder_layer(cfg)
    assert api_block.feedforward.gate_proj is None
    assert api_block.feedforward.gate_up_proj is None
    assert api_block.feedforward.up_proj is not None
    assert api_block.feedforward.down_proj is not None
