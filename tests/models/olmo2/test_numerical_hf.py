"""B3 — Gate test: OLMo 2 layer-0 numerical equivalence vs HF.

Loads `allenai/OLMo-2-0425-1B` (ungated, 1B params, FP32 checkpoint), extracts
layer-0 weights, loads them into our DecoderBlock via `load_hf_olmo2_layer`,
runs both implementations on a fixed input, and asserts allclose at atol=5e-4.

This is the first POST-norm + FULL_HDH QK-norm gate in the codebase. The
B3 IR additions exercised: NormPosition.POST, QKNormShape.FULL_HDH.

Gating behaviour
================
Two skip paths:
  1. ``transformers.models.olmo2`` not importable → skipped at module level.
  2. HF weights not downloadable → skipped per-test with the exception string.
"""
from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")

try:
    from transformers.models import olmo2  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers.models.olmo2 not available — install transformers>=5.10",
        allow_module_level=True,
    )

from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.olmo2 import config as o2_config, layer as o2_layer


MODEL_ID = "allenai/OLMo-2-0425-1B"
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
            f"OLMo 2 weights download/load failed: {type(e).__name__}: {str(e)[:200]}. "
            "B3 numerical-gate test is in place and will execute when weights become available."
        )


@pytest.mark.gate
def test_olmo2_layer0_forward_matches_hf(hf_model):
    cfg = o2_config.Olmo2Config.from_hf_dict(hf_model.config.to_dict())

    api_blk = o2_layer.build_olmo2_decoder_layer(
        cfg, layer_idx=0, max_seq=cfg.max_position_embeddings,
    )
    o2_layer.load_hf_olmo2_layer(api_blk, hf_model.state_dict(), layer_idx=0)
    api_blk.eval()

    B, S = FIXED_INPUT.shape
    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)

    hf_layer = hf_model.model.layers[0]
    position_ids = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        # OLMo 2 uses Olmo2RotaryEmbedding which is the same default-RoPE shape
        # as Llama / Qwen3 — single (cos, sin) tuple from the model-level
        # rotary_emb. Source: modeling_olmo2.py:367, 409.
        cos, sin = hf_model.model.rotary_emb(embed_out, position_ids)
        # Standard 4D additive causal mask with -inf above the diagonal.
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        # Olmo2DecoderLayer forward signature (modeling_olmo2.py:305-333).
        hf_out = hf_layer(
            hidden_states=embed_out,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec,
        batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
