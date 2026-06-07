"""Qwen2.5-VL LM-decoder numerical gate vs HF.

We load Qwen/Qwen2.5-VL-3B-Instruct and verify that layer 0 of the LM
decoder reproduces HF's eager forward to atol=5e-4 on a 3D position_ids
M-RoPE input.

The 3-axis position_ids tensor mirrors what HF's `get_rope_index` produces
for a mixed image+text prompt: a leading vision-token block where T, H, W
take distinct values, followed by text tokens where the three axes are
aligned.
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoConfig, AutoModelForImageTextToText  # noqa: E402

from api import kvcache, specs, types  # noqa: E402
from models.qwen2_5_vl import config as q_config, layer as q_layer  # noqa: E402


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ATOL = 5e-4
RTOL = 5e-4


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    _ = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return model


def test_layer0_forward_matches_hf(hf_model):
    """Forward LM-decoder layer 0 on synthetic 3D position_ids input."""
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = q_config.Qwen2_5_VLConfig.from_hf_dict(hf_cfg_dict)

    api_blk = q_layer.build_qwen2_5_vl_decoder_layer(
        cfg, layer_idx=0,
        # 3B Instruct lists max_position_embeddings=128_000 — but the cos/sin
        # cache is unused for M-RoPE (built on-the-fly per position_ids).
        # We pass a small max_seq to keep the buffer footprint trivial.
        max_seq=64,
    )
    full_sd = hf_model.state_dict()
    q_layer.load_hf_qwen2_5_vl_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    # Synthetic mixed-modality input: a few image tokens followed by text.
    n_vision = 6
    n_text = 5
    S = n_vision + n_text
    B = 1
    fixed_ids = torch.tensor(
        [[151655] * n_vision + [101, 1024, 4789, 38, 9]],
        dtype=torch.long,
    )

    lang_model = hf_model.model.language_model
    with torch.no_grad():
        embed_out = lang_model.embed_tokens(fixed_ids)

    # Build 3D position_ids by hand: for vision tokens we set distinct T/H/W;
    # for text tokens the three axes are aligned (text-only fallback).
    # Vision block: T=0..0, H=0..2, W=0..2 (3x2 grid pattern just to vary).
    pos = torch.zeros(3, B, S, dtype=torch.long)
    for s in range(n_vision):
        pos[0, 0, s] = 0
        pos[1, 0, s] = s // 3
        pos[2, 0, s] = s % 3
    # Text block: aligned axes resume from the max vision T-index + 1.
    text_start = int(pos[0, 0, :n_vision].max().item()) + 1
    for i in range(n_text):
        pos[0, 0, n_vision + i] = text_start + i
        pos[1, 0, n_vision + i] = text_start + i
        pos[2, 0, n_vision + i] = text_start + i

    hf_layer = lang_model.layers[0]
    with torch.no_grad():
        cos, sin = lang_model.rotary_emb(embed_out, pos)
        # Causal mask: standard upper-triangular -inf above diagonal.
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_out = hf_layer(
            embed_out,
            attention_mask=attn_mask,
            position_ids=pos,
            position_embeddings=(cos, sin),
            past_key_values=None,
            use_cache=False,
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=64,
    )
    with torch.no_grad():
        api_out = api_blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Qwen2.5-VL LM-layer max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"Qwen2.5-VL LM-layer max_abs_diff={max_abs_diff:.2e}")
