"""MPT-style ALiBi attention numerical gate (v5-phase2 V1).

The MPT FFN (ungated GELU) and LayerNorm-without-bias are not yet
expressible in the IR — these are Phase 3 work (StarCoder / Falcon-old
LN-no-bias). For Phase 2 we exercise the ATTENTION sublayer (where ALiBi
lives) end-to-end against HF's MPT attention path:

    api.Attention(spec with alibi, qkv_layout=SPLIT, rope=None)
        vs
    transformers.models.mpt.modeling_mpt.MptAttention

with the same weights. The ALiBi position bias is the only Phase 2
extension exercised here.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.mpt.modeling_mpt import (  # noqa: E402
    MptAttention, MptConfig, build_mpt_alibi_tensor,
)

from api import attention as api_attention, kvcache, specs, types  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _build_hf_mpt_attn(hidden_size: int, n_heads: int, max_seq: int):
    cfg = MptConfig(
        d_model=hidden_size,
        n_heads=n_heads,
        n_layers=1,
        expansion_ratio=4,
        max_seq_len=max_seq,
        vocab_size=64,
    )
    torch.manual_seed(0)
    attn = MptAttention(cfg, layer_idx=0)
    attn.eval()
    return cfg, attn


def _build_api_attn(hidden_size: int, n_heads: int, max_seq: int):
    head_dim = hidden_size // n_heads
    spec = specs.AttentionSpec(
        n_q_heads=n_heads,
        n_kv_heads=n_heads,         # MPT is MHA
        head_dim=head_dim,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=None,                  # MPT has no RoPE
        alibi=specs.AliBiSpec(n_heads=n_heads, alibi_bias_max=8.0),
    )
    return api_attention.Attention(spec, hidden_size=hidden_size,
                                   max_seq=max_seq, dtype=torch.float32)


def _copy_weights(hf_attn: MptAttention, api_attn: api_attention.Attention,
                  hidden_size: int) -> None:
    """HF MPT uses fused Wqkv [3*H, H_in]; our api uses SPLIT q/k/v_proj.
    We slice Wqkv into thirds and assign."""
    Wqkv = hf_attn.Wqkv.weight.detach().clone()      # [3*hidden, hidden]
    assert Wqkv.shape == (3 * hidden_size, hidden_size)
    Wq, Wk, Wv = Wqkv.chunk(3, dim=0)                # each [hidden, hidden]
    with torch.no_grad():
        api_attn.q_proj.weight.copy_(Wq)
        api_attn.k_proj.weight.copy_(Wk)
        api_attn.v_proj.weight.copy_(Wv)
        api_attn.o_proj.weight.copy_(hf_attn.out_proj.weight.detach())


def test_alibi_attention_matches_hf_mpt():
    """Tiny MPT attention layer: api.Attention(alibi) vs MptAttention."""
    hidden_size, n_heads, max_seq = 64, 4, 32
    cfg, hf_attn = _build_hf_mpt_attn(hidden_size, n_heads, max_seq)
    api_attn = _build_api_attn(hidden_size, n_heads, max_seq)
    _copy_weights(hf_attn, api_attn, hidden_size)
    api_attn.eval()

    torch.manual_seed(1)
    B, S = 1, 8
    x = torch.randn(B, S, hidden_size)
    # HF path: build ALiBi (per-batch shape [num_heads, 1, S]) and a
    # bool causal mask (True = MASK OUT; modeling_mpt.py:124 masks where
    # mask is True). The MPT causal-mask uses an upper-triangular ones
    # matrix above the diagonal.
    alibi = build_mpt_alibi_tensor(n_heads, max_seq, alibi_bias_max=8)
    # Causal mask: bool, True above diagonal
    causal_mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool), diagonal=1,
    )[None, None, :, :]                                     # [1, 1, S, S]
    with torch.no_grad():
        hf_out, _ = hf_attn(
            x, position_bias=alibi,
            past_key_values=None, attention_mask=causal_mask,
        )

    # API path
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
        api_out = api_attn(x, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"MPT ALiBi attention max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    print(f"MPT (synthetic) ALiBi attention max_abs_diff={max_abs_diff:.2e}")


def test_alibi_attention_n_heads_non_power_of_2():
    """Verify non-power-of-2 head count (e.g. 6 heads) still works."""
    hidden_size, n_heads, max_seq = 60, 6, 32
    cfg, hf_attn = _build_hf_mpt_attn(hidden_size, n_heads, max_seq)
    api_attn = _build_api_attn(hidden_size, n_heads, max_seq)
    _copy_weights(hf_attn, api_attn, hidden_size)
    api_attn.eval()

    torch.manual_seed(2)
    B, S = 1, 6
    x = torch.randn(B, S, hidden_size)
    alibi = build_mpt_alibi_tensor(n_heads, max_seq, alibi_bias_max=8)
    causal_mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool), diagonal=1,
    )[None, None, :, :]
    with torch.no_grad():
        hf_out, _ = hf_attn(
            x, position_bias=alibi, past_key_values=None,
            attention_mask=causal_mask,
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
        api_out = api_attn(x, position_ids=pos, cache=cache, start_pos=0)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"MPT n_heads=6 max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
