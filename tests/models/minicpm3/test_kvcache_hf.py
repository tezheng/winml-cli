"""KV-cache equivalence gate: prefill + 1 decode step.

Compares api-block forward across two scenarios:
  (1) full prefill of S+1 tokens in one pass.
  (2) prefill of S tokens, then a single 1-token decode step appended to
      the cache.
Outputs must match at atol=5e-4.

Avoids the HF MiniCPM weights — uses random weights in an api block only.
This still verifies the MLA cache asymmetric K/V is correctly read/written.
"""
import math

import pytest
import torch

from api import kvcache, specs, types
from models.minicpm3 import config as m_config, layer as m_layer


ATOL = 5e-4
RTOL = 5e-4


def _tiny_minicpm3_cfg():
    return m_config.MiniCPM3Config(
        hidden_size=256,
        num_attention_heads=4,
        num_hidden_layers=4,
        intermediate_size=512,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        vocab_size=100,
        max_position_embeddings=64,
        original_max_position_embeddings=64,
        tie_word_embeddings=False,
        dtype=torch.float32,
        scale_emb=1.0,
        scale_depth=1.4,
        dim_model_base=64,
        rope_type="default",
        attention_bias=False,
    )


def test_kvcache_decode_step_matches_full_prefill():
    """Decode step (1 token at start_pos=S) appended to prefill must match
    full prefill of S+1 tokens for the last token output."""
    torch.manual_seed(0)
    cfg = _tiny_minicpm3_cfg()
    blk = m_layer.build_minicpm3_decoder_layer(cfg, layer_idx=0, max_seq=32)
    blk.eval()

    B, S = 1, 5
    x_full = torch.randn(B, S + 1, cfg.hidden_size)

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )

    # Full prefill of S+1 tokens.
    cache_full = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=32,
        v_head_dim=cfg.v_head_dim,
    )
    with torch.no_grad():
        out_full = blk(x_full, position_ids=torch.arange(S + 1),
                       cache=cache_full, start_pos=0)

    # Chunked: prefill S tokens then 1 decode step.
    cache_chunk = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=32,
        v_head_dim=cfg.v_head_dim,
    )
    with torch.no_grad():
        _ = blk(x_full[:, :S], position_ids=torch.arange(S),
                cache=cache_chunk, start_pos=0)
        out_decode = blk(x_full[:, S:], position_ids=torch.tensor([S]),
                         cache=cache_chunk, start_pos=S)

    max_abs_diff = (out_full[:, S:] - out_decode).abs().max().item()
    print(f"MiniCPM-3 KV-cache decode max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(out_full[:, S:], out_decode, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )


def test_kvcache_chunked_prefill_matches_full_prefill():
    """Two-chunk prefill (2 + 4) must equal full 6-token prefill."""
    torch.manual_seed(0)
    cfg = _tiny_minicpm3_cfg()
    blk = m_layer.build_minicpm3_decoder_layer(cfg, layer_idx=0, max_seq=16)
    blk.eval()

    B, S = 1, 6
    x = torch.randn(B, S, cfg.hidden_size)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache_full = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=16,
        v_head_dim=cfg.v_head_dim,
    )
    with torch.no_grad():
        out_full = blk(x, position_ids=torch.arange(S),
                       cache=cache_full, start_pos=0)
    cache_chunk = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=16,
        v_head_dim=cfg.v_head_dim,
    )
    with torch.no_grad():
        _ = blk(x[:, :2], position_ids=torch.arange(2),
                cache=cache_chunk, start_pos=0)
        out_tail = blk(x[:, 2:], position_ids=torch.arange(2, S),
                       cache=cache_chunk, start_pos=2)
    max_abs_diff = (out_full[:, 2:] - out_tail).abs().max().item()
    assert torch.allclose(out_full[:, 2:], out_tail, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
