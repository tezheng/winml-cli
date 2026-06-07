"""DeepSeek-OCR-2 LM-decoder numerical gate vs HF — SYNTHETIC weights.

We synthesize a tiny DeepseekOcr2TextModel in-process (random init), copy
its weights into our DecoderBlock, then compare layer-0 forward outputs.
This avoids the 6.7 GB checkpoint download while still exercising the
exact HF forward path (eager attention, softmax router, packed-experts
MoE, shared-experts addition).

The canonical numerical gate vs the public checkpoint is
test_numerical_hf.py (gated by the network availability of the 6.7 GB
weights).
"""
from __future__ import annotations
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")
from transformers import DeepseekOcr2TextConfig  # noqa: E402
from transformers.models.deepseek_ocr2.modeling_deepseek_ocr2 import (  # noqa: E402
    DeepseekOcr2TextModel,
)

from api import kvcache, specs, types  # noqa: E402
from models.deepseek_ocr2 import config as ds_config, layer as ds_layer  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _build_tiny_text_model() -> tuple[DeepseekOcr2TextConfig, DeepseekOcr2TextModel]:
    cfg = DeepseekOcr2TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=2,
        num_experts_per_tok=2,
        topk_method="greedy",
        n_group=1, topk_group=1,
        routed_scaling_factor=1.0,
        attention_bias=False,
        mlp_bias=False,
        head_dim=16,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        mlp_layer_types=["dense", "sparse"],
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = DeepseekOcr2TextModel(cfg)
    model.eval()
    return cfg, model


def _our_cfg(hf_cfg: DeepseekOcr2TextConfig) -> ds_config.DeepseekOcr2Config:
    return ds_config.DeepseekOcr2Config.from_hf_dict(hf_cfg.to_dict())


def _forward_hf_layer(
    hf_model: DeepseekOcr2TextModel,
    layer_idx: int,
    hidden: torch.Tensor,
) -> torch.Tensor:
    B, S, _ = hidden.shape
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = hf_model.rotary_emb(hidden, pos)
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        out = hf_model.layers[layer_idx](
            hidden,
            attention_mask=attn_mask,
            position_ids=pos,
            position_embeddings=(cos, sin),
            past_key_values=None,
            use_cache=False,
        )
    return out


def _forward_api_layer(api_blk, hidden, cfg):
    B, S, _ = hidden.shape
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        return api_blk(hidden, position_ids=pos, cache=cache, start_pos=0)


def _load_into_block(api_blk, hf_model, layer_idx, our_cfg):
    """Build a state-dict in HF's `model.language_model.layers.{L}.*`
    namespace so our loader (which expects `model.language_model.layers`
    prefix by default) can find tensors. The HF text-model exposes them
    under `layers.{L}.*` — we just re-prefix."""
    src = hf_model.state_dict()
    remap = {}
    pfx = f"model.language_model.layers.{layer_idx}"
    for k, v in src.items():
        if k.startswith(f"layers.{layer_idx}."):
            remap[k.replace(f"layers.{layer_idx}.", f"{pfx}.")] = v
    ds_layer.load_hf_deepseek_ocr2_layer(api_blk, remap, layer_idx, our_cfg)


def test_dense_layer0_matches_hf_synthetic():
    """Layer 0 is dense FFN — exercises MHA + plain Qwen2-style MLP path."""
    hf_cfg, hf_model = _build_tiny_text_model()
    our_cfg = _our_cfg(hf_cfg)
    api_blk = ds_layer.build_deepseek_ocr2_decoder_layer(
        our_cfg, layer_idx=0, max_seq=our_cfg.max_position_embeddings,
    )
    _load_into_block(api_blk, hf_model, layer_idx=0, our_cfg=our_cfg)
    api_blk.eval()

    torch.manual_seed(1)
    B, S = 1, 6
    hidden = torch.randn(B, S, hf_cfg.hidden_size)

    hf_out = _forward_hf_layer(hf_model, 0, hidden)
    api_out = _forward_api_layer(api_blk, hidden, our_cfg)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"DeepSeek-OCR-2 dense layer max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"DeepSeek-OCR-2 dense layer max_abs_diff={max_abs_diff:.2e}")


def test_moe_layer1_matches_hf_synthetic():
    """Layer 1 is sparse MoE — exercises softmax router, expert dispatch,
    and shared-experts addition. This is the canonical numerical gate
    for the DeepSeek-OCR-2 family (synthetic weights, real HF code path).
    """
    hf_cfg, hf_model = _build_tiny_text_model()
    our_cfg = _our_cfg(hf_cfg)
    api_blk = ds_layer.build_deepseek_ocr2_decoder_layer(
        our_cfg, layer_idx=1, max_seq=our_cfg.max_position_embeddings,
    )
    _load_into_block(api_blk, hf_model, layer_idx=1, our_cfg=our_cfg)
    api_blk.eval()

    torch.manual_seed(2)
    B, S = 1, 8
    hidden = torch.randn(B, S, hf_cfg.hidden_size)

    hf_out = _forward_hf_layer(hf_model, 1, hidden)
    api_out = _forward_api_layer(api_blk, hidden, our_cfg)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"DeepSeek-OCR-2 MoE layer max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"DeepSeek-OCR-2 MoE layer max_abs_diff={max_abs_diff:.2e}")
