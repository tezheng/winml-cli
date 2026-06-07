"""DeepSeek-OCR-2 LM-decoder numerical gate vs the public HF checkpoint.

Skipped automatically if HF cannot supply the ~6.7 GB
`deepseek-community/DeepSeek-OCR-2` weights within the test session
(network errors, missing HF_HOME, etc.). The synthetic-weight gate in
test_numerical_synthetic.py covers the same code path and is
unconditional.
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoConfig, AutoModelForImageTextToText  # noqa: E402

from api import kvcache, specs, types  # noqa: E402
from models.deepseek_ocr2 import config as ds_config, layer as ds_layer  # noqa: E402


MODEL_ID = "deepseek-community/DeepSeek-OCR-2"
ATOL = 5e-4
RTOL = 5e-4


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    try:
        _ = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    except Exception as e:
        pytest.skip(
            f"DeepSeek-OCR-2 HF download unavailable ({e}); "
            f"see test_numerical_synthetic.py for the same gate on "
            f"locally-synthesized weights."
        )
    model.eval()
    return model


def test_layer0_dense_matches_hf(hf_model):
    """Layer 0 of canonical 12-layer checkpoint is dense (mlp_layer_types[0]='dense')."""
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = ds_config.DeepseekOcr2Config.from_hf_dict(hf_cfg_dict)
    api_blk = ds_layer.build_deepseek_ocr2_decoder_layer(
        cfg, layer_idx=0, max_seq=cfg.max_position_embeddings,
    )
    full_sd = hf_model.state_dict()
    ds_layer.load_hf_deepseek_ocr2_layer(api_blk, full_sd, layer_idx=0, cfg=cfg)
    api_blk.eval()

    B, S = 1, 6
    fixed_ids = torch.tensor([[101, 1024, 4789, 38, 9, 2]], dtype=torch.long)
    lang_model = hf_model.model.language_model
    with torch.no_grad():
        embed_out = lang_model.embed_tokens(fixed_ids)
    pos_2d = torch.arange(S).unsqueeze(0)
    hf_layer = lang_model.layers[0]
    with torch.no_grad():
        cos, sin = lang_model.rotary_emb(embed_out, pos_2d)
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_out = hf_layer(
            embed_out,
            attention_mask=attn_mask,
            position_ids=pos_2d,
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
        max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"DS-OCR-2 dense layer 0 max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"DS-OCR-2 dense layer 0 max_abs_diff={max_abs_diff:.2e}")


def test_layer1_moe_matches_hf(hf_model):
    """Layer 1 is sparse MoE — exercises full softmax+experts+shared path."""
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = ds_config.DeepseekOcr2Config.from_hf_dict(hf_cfg_dict)
    api_blk = ds_layer.build_deepseek_ocr2_decoder_layer(
        cfg, layer_idx=1, max_seq=cfg.max_position_embeddings,
    )
    full_sd = hf_model.state_dict()
    ds_layer.load_hf_deepseek_ocr2_layer(api_blk, full_sd, layer_idx=1, cfg=cfg)
    api_blk.eval()

    B, S = 1, 6
    fixed_ids = torch.tensor([[101, 1024, 4789, 38, 9, 2]], dtype=torch.long)
    lang_model = hf_model.model.language_model
    with torch.no_grad():
        embed_out = lang_model.embed_tokens(fixed_ids)
    pos_2d = torch.arange(S).unsqueeze(0)
    hf_layer = lang_model.layers[1]
    with torch.no_grad():
        cos, sin = lang_model.rotary_emb(embed_out, pos_2d)
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_out = hf_layer(
            embed_out,
            attention_mask=attn_mask,
            position_ids=pos_2d,
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
        max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"DS-OCR-2 MoE layer 1 max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"DS-OCR-2 MoE layer 1 max_abs_diff={max_abs_diff:.2e}")
