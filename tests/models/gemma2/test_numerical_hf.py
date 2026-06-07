"""B3 — Gate test: Gemma 2 layer-0 numerical equivalence vs HF.

Loads `unsloth/gemma-2-2b` (mirror of `google/gemma-2-2b` which is gated by
license acceptance). Tests both a sliding-attention layer (idx 0) and a
full-attention layer (idx 1) — separate test functions.

Tests both layer types because Gemma 2's IR-novel feature in B3 is the
`attn_logit_softcap` AND the sandwich norm. The sliding layer exercises
softcap + SWA mask + sandwich; the full layer exercises softcap + causal mask
+ sandwich.
"""
from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")

try:
    from transformers.models import gemma2  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers.models.gemma2 not available — install transformers>=5.10",
        allow_module_level=True,
    )

from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.gemma2 import config as g2c, layer as g2l


MODEL_ID = "unsloth/gemma-2-2b"
ATOL = 5e-4
RTOL = 5e-4

FIXED_INPUT = torch.tensor(
    [[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long,
)


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    try:
        _ = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        return model
    except Exception as e:
        pytest.skip(
            f"Gemma 2 weights download/load failed: {type(e).__name__}: {str(e)[:200]}. "
            "B3 numerical-gate test is in place and will execute when weights become available."
        )


def _build_and_load(hf_model, layer_idx: int):
    cfg = g2c.Gemma2Config.from_hf_dict(hf_model.config.to_dict())
    blk = g2l.build_gemma2_decoder_layer(
        cfg, layer_idx=layer_idx, max_seq=cfg.max_position_embeddings,
    )
    g2l.load_hf_gemma2_layer(blk, hf_model.state_dict(), layer_idx=layer_idx)
    blk.eval()
    return cfg, blk


def _run_hf_layer(hf_model, layer_idx: int, embed_out, S: int):
    """Run HF Gemma 2 decoder layer idx in isolation with a causal mask.

    For sliding layers, the per-layer attention reads `self.sliding_window`
    from init; HF further restricts the mask inside attention but the mask
    we provide must allow SWA to operate. We pass a standard causal mask;
    when the layer-type is sliding, the layer further narrows it via SWA.
    Actually HF passes the SWA-narrowed mask externally (from
    `create_sliding_window_causal_mask`), so we replicate that.
    """
    hf_layer = hf_model.model.layers[layer_idx]
    position_ids = torch.arange(S).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(embed_out, position_ids)

    # Build the right mask for this layer type.
    cfg = hf_model.config
    layer_type = cfg.layer_types[layer_idx]
    if layer_type == "sliding_attention":
        sw = cfg.sliding_window
        i = torch.arange(S).unsqueeze(1)
        j = torch.arange(S).unsqueeze(0)
        keep = (j <= i) & (i - j < sw)
        attn_mask = torch.zeros(S, S)
        attn_mask = attn_mask.masked_fill(~keep, float("-inf"))
    else:
        attn_mask = torch.full((S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return hf_layer(
            hidden_states=embed_out,
            position_embeddings=(cos, sin),
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
        )


@pytest.mark.gate
def test_gemma2_layer0_sliding_matches_hf(hf_model):
    """Layer 0 in Gemma 2 is a SLIDING attention layer (per the default
    pattern). Exercises softcap + SWA + sandwich norm + GeGLU.
    """
    cfg, blk = _build_and_load(hf_model, layer_idx=0)
    B, S = FIXED_INPUT.shape
    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)
    hf_out = _run_hf_layer(hf_model, 0, embed_out, S)

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"layer-0 (sliding) max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )


@pytest.mark.gate
def test_gemma2_layer1_full_matches_hf(hf_model):
    """Layer 1 in Gemma 2 is a FULL attention layer (per the default pattern).
    Exercises softcap + causal full mask + sandwich norm + GeGLU.
    """
    cfg, blk = _build_and_load(hf_model, layer_idx=1)
    B, S = FIXED_INPUT.shape
    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)
    hf_out = _run_hf_layer(hf_model, 1, embed_out, S)

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"layer-1 (full) max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
