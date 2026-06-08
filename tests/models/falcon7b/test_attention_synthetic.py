"""Falcon-7B attention sublayer numerical gate (v5-phase2 V2).

The Falcon-7B decoder layer's PARALLEL residual flow is exercised at
the IR level by tests/api/test_parallel_residual.py. This test
verifies the ATTENTION sublayer (MQA + RoPE, no biases) matches the HF
reference FalconAttention with parallel_attn=True semantics.

The FFN form (ungated GELU) and LayerNorm-with-bias are documented
substitutions — see models/falcon7b/__init__.py — so the end-to-end
decoder-layer gate is deferred to Phase 3.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.falcon.configuration_falcon import FalconConfig  # noqa: E402
from transformers.models.falcon.modeling_falcon import (  # noqa: E402
    FalconAttention, FalconRotaryEmbedding,
)

from api import attention as api_attention, kvcache, specs, types  # noqa: E402
from models.falcon7b import config as falcon_config  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _build_tiny_falcon_attn(hidden_size: int, n_heads: int, max_seq: int):
    cfg = FalconConfig(
        vocab_size=256, hidden_size=hidden_size,
        num_hidden_layers=1, num_attention_heads=n_heads,
        num_kv_heads=1,                                # MQA
        max_position_embeddings=max_seq,
        bias=False, parallel_attn=True,
        new_decoder_architecture=False, multi_query=True,
        alibi=False,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    attn = FalconAttention(cfg, layer_idx=0)
    attn.eval()
    rot = FalconRotaryEmbedding(cfg)
    rot.eval()
    return cfg, attn, rot


def _build_api_attn_from_cfg(cfg: FalconConfig):
    f7b = falcon_config.Falcon7BConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_kv_heads=1,
        num_hidden_layers=1,
        intermediate_size=cfg.hidden_size * 4,
        max_position_embeddings=cfg.max_position_embeddings,
    )
    spec = f7b.to_attention_spec()
    return api_attention.Attention(spec, hidden_size=cfg.hidden_size,
                                   max_seq=cfg.max_position_embeddings,
                                   dtype=torch.float32)


def _copy_weights(hf_attn: FalconAttention, api_attn) -> None:
    """HF Falcon-7B fused qkv (multi_query=True):
    `query_key_value` has output dim `(num_heads + 2) * head_dim`.
    Layout in `_split_heads`:
        fused.view(B, S, num_heads + 2, head_dim)
        → q = fused[..., :-2, :]
        → k = fused[..., [-2], :]
        → v = fused[..., [-1], :]
    So WEIGHT rows [0, num_heads*Dh) are q, then K is one head, V is one head.
    """
    Hq = hf_attn.num_heads
    Dh = hf_attn.head_dim
    W = hf_attn.query_key_value.weight.detach().clone()    # [(Hq+2)*Dh, hidden]
    assert W.shape == ((Hq + 2) * Dh, hf_attn.hidden_size)
    Wq = W[: Hq * Dh, :]
    Wk = W[Hq * Dh : (Hq + 1) * Dh, :]
    Wv = W[(Hq + 1) * Dh :, :]
    with torch.no_grad():
        api_attn.q_proj.weight.copy_(Wq)
        api_attn.k_proj.weight.copy_(Wk)
        api_attn.v_proj.weight.copy_(Wv)
        api_attn.o_proj.weight.copy_(hf_attn.dense.weight.detach())


def test_falcon7b_attention_matches_hf():
    hidden_size, n_heads, max_seq = 256, 8, 32
    cfg, hf_attn, rot = _build_tiny_falcon_attn(hidden_size, n_heads, max_seq)
    api_attn = _build_api_attn_from_cfg(cfg)
    _copy_weights(hf_attn, api_attn)
    api_attn.eval()

    torch.manual_seed(1)
    B, S = 1, 6
    x = torch.randn(B, S, hidden_size)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rot(x, pos)
        # HF builds an additive 4D causal mask for non-flash attention.
        causal_mask = torch.full((1, 1, S, S), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        hf_out, _ = hf_attn(
            x, alibi=None, attention_mask=causal_mask,
            position_ids=pos, position_embeddings=(cos, sin),
            layer_past=None,
        )

    # API
    cache = kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=B, n_kv_heads=1, head_dim=hidden_size // n_heads,
        max_seq=max_seq,
    )
    with torch.no_grad():
        api_out = api_attn(x, position_ids=pos.squeeze(0),
                           cache=cache, start_pos=0)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Falcon-7B attention max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    print(f"Falcon-7B (synthetic) attention max_abs_diff={max_abs_diff:.2e}")
