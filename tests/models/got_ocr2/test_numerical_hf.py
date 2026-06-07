"""GOT-OCR 2.0 LM-decoder numerical gate vs HF.

We load the canonical stepfun-ai/GOT-OCR-2.0-hf checkpoint and verify that
layer 0 of the LM decoder (Qwen2-0.5B backbone) reproduces HF's eager
forward to atol=5e-4 on a synthetic vision-prefix + text token sequence.

The vision adapter is OUT OF SCOPE — we treat the LM-decoder input as a
"hidden_states" tensor coming from `model.language_model.embed_tokens` and
feed it directly. Vision-fusion in HF happens at the embedding level
(image features are scatter-overwritten into specific token positions
matching image_token_index), so the LM decoder sees a single unified
[B, S, hidden] tensor regardless of fusion path. We exercise this by
sampling a representative mixed-token input (some text tokens, some
positioned at fictional vision-token slots).

Source citations:
- modeling_got_ocr2.py:546-587 (apply_multimodal_projector +
  get_placeholder_mask: image features overwrite embedding positions).
- modeling_qwen2.py:269-309 (Qwen2DecoderLayer.forward — what we mimic).
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoConfig, AutoModelForImageTextToText  # noqa: E402

from api import kvcache, specs, types  # noqa: E402
from models.got_ocr2 import config as g_config, layer as g_layer  # noqa: E402


MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
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
    """Forward LM-decoder layer 0 on synthetic concat-prefix input."""
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = g_config.GotOcr2Config.from_hf_dict(hf_cfg_dict)

    api_blk = g_layer.build_got_ocr2_decoder_layer(
        cfg, layer_idx=0, max_seq=cfg.max_position_embeddings,
    )
    full_sd = hf_model.state_dict()
    # The HF state-dict has `model.language_model.layers.{L}.*` keys.
    g_layer.load_hf_got_ocr2_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    # Build a synthetic concat-prefix input: a few image-token IDs at the
    # head followed by some text token IDs. The actual identities don't
    # matter for the LM-decoder forward — we just need both implementations
    # to see the same hidden_states tensor.
    n_vision = 8
    n_text = 6
    S = n_vision + n_text
    fixed_ids = torch.tensor(
        [[cfg.image_token_index] * n_vision + [101, 1024, 4789, 38, 9, 2]],
        dtype=torch.long,
    )
    B = 1

    lang_model = hf_model.model.language_model
    with torch.no_grad():
        embed_out = lang_model.embed_tokens(fixed_ids)
    position_ids = torch.arange(S).unsqueeze(0)
    hf_layer = lang_model.layers[0]
    with torch.no_grad():
        cos, sin = lang_model.rotary_emb(embed_out, position_ids)
        # Causal mask: [1, 1, S, S] with -inf above the diagonal.
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_out = hf_layer(
            embed_out,
            attention_mask=attn_mask,
            position_ids=position_ids,
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
        f"GOT-OCR 2.0 LM-layer max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"GOT-OCR 2.0 LM-layer max_abs_diff={max_abs_diff:.2e}")
