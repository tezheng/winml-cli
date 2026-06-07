"""Gate test: full Ministral-8B-Instruct-2410 decoder layer numerical
equivalence vs HF on CPU/fp32. Tests TWO layers:
  - Layer 0: FULL attention layer (1:3 pattern starts with full).
  - Layer 1: SLIDING attention layer.

At S=9 and sliding_window=32768 (the published config), the SWA and CAUSAL
masks are NUMERICALLY identical, so this gate proves the per-layer build
correctness end-to-end (RMSNorm, GQA, RoPE @ theta=1e8, SwiGLU) but does
NOT empirically distinguish the masks. The shape-only smoking-gun test in
test_layer_shape.py covers the W<S divergence case.

Loading via Ministral modeling (NOT Mistral). We override the model_type
in the config so the AutoModel router builds a MinistralForCausalLM whose
forward dispatches per layer (modeling_ministral.py:411-414). When using
MistralForCausalLM (the published architecture), HF applies the SAME mask
to all layers — which at S=9 happens to give the same output as Ministral's
per-layer dispatch (both layers see a causal-equivalent mask). For
correctness fidelity we use the Ministral path.
"""
from __future__ import annotations

import os

import pytest
import torch


pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")

try:
    from transformers.models import ministral  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers.models.ministral not available — install transformers>=5.x",
        allow_module_level=True,
    )

from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.ministral import config as mc, layer as ml


MODEL_ID = "mistralai/Ministral-8B-Instruct-2410"
ATOL = 5e-4
RTOL = 5e-4

FIXED_INPUT = torch.tensor(
    [[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long,
)


@pytest.fixture(scope="module")
def hf_model():
    """Load Ministral-8B-Instruct-2410 via Ministral modeling (not Mistral).

    The published config declares architectures=["MistralForCausalLM"]; we
    override model_type to "ministral" + architectures to ["MinistralForCausalLM"]
    so HF builds the per-layer-dispatching MinistralModel.
    """
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    try:
        cfg = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        # Re-route to Ministral modeling so layer_types is honoured. The
        # tensor names are unchanged.
        cfg.model_type = "ministral"
        cfg.architectures = ["MinistralForCausalLM"]
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            config=cfg,
            cache_dir=cache_dir,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        return model
    except Exception as e:
        pytest.skip(
            f"Ministral-8B weights download/load failed: "
            f"{type(e).__name__}: {str(e)[:200]}"
        )


def _build_and_load(hf_model, layer_idx: int):
    cfg = mc.MinistralConfig.from_hf_dict(hf_model.config.to_dict())
    blk = ml.build_ministral_decoder_layer(
        cfg, layer_idx=layer_idx, max_seq=cfg.max_position_embeddings,
    )
    ml.load_hf_ministral_layer(blk, hf_model.state_dict(), layer_idx=layer_idx)
    blk.eval()
    return cfg, blk


def _run_hf_layer(hf_model, layer_idx: int, embed_out, S: int):
    """Run HF Ministral decoder layer in isolation."""
    hf_layer = hf_model.model.layers[layer_idx]
    position_ids = torch.arange(S).unsqueeze(0)
    cos, sin = hf_model.model.rotary_emb(embed_out, position_ids)

    layer_type = hf_model.config.layer_types[layer_idx]
    if layer_type == "sliding_attention":
        sw = hf_model.config.sliding_window
        i = torch.arange(S).unsqueeze(1)
        j = torch.arange(S).unsqueeze(0)
        keep = (j <= i) & (i - j <= sw)
        attn_mask = torch.zeros(S, S)
        attn_mask = attn_mask.masked_fill(~keep, float("-inf"))
    else:
        attn_mask = torch.full((S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        out = hf_layer(
            hidden_states=embed_out,
            position_embeddings=(cos, sin),
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
        )
    # MinistralDecoderLayer.forward returns a bare tensor (modeling_ministral.py:260).
    return out


@pytest.mark.gate
def test_ministral_layer0_full_matches_hf(hf_model):
    """Layer 0 of Ministral-8B-Instruct-2410 is a FULL attention layer
    (1:3 pattern starts with full). Exercises RMSNorm STANDARD_W eps=1e-5,
    GQA (n_q=32, n_kv=8, head_dim=128), RoPE theta=1e8, SwiGLU SiLU FFN.
    """
    cfg, blk = _build_and_load(hf_model, layer_idx=0)
    # Sanity: this had better be a full layer.
    assert cfg.layer_type(0) == "full_attention", cfg.layer_type(0)

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
    print(f"Ministral-8B layer-0 (full) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"layer-0 (full) max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )


@pytest.mark.gate
def test_ministral_layer1_sliding_matches_hf(hf_model):
    """Layer 1 of Ministral-8B-Instruct-2410 is a SLIDING attention layer.
    At S=9 < sliding_window=32768 the SWA mask equals the causal mask, so
    this checks the same math as layer-0 (proving the per-layer factory
    builds the sliding layer correctly), while the shape-only smoking-gun
    test in test_layer_shape.py covers the W<S divergence path.
    """
    cfg, blk = _build_and_load(hf_model, layer_idx=1)
    assert cfg.layer_type(1) == "sliding_attention", cfg.layer_type(1)

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
    print(f"Ministral-8B layer-1 (sliding) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"layer-1 (sliding) max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
