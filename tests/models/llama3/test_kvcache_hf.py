"""Verify our ContiguousKVCache prefill+decode produces the same layer-0 output
as HF for Llama 3.2 1B.

Runs on CPU, fp32. Drives HF's DynamicCache through one prefill step and one
decode step, then drives our ContiguousKVCache through the same two-step
sequence, and asserts both layer outputs match at atol=5e-4.
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from api import kvcache, specs, types
from models.llama3 import config as l3_config, layer as l3_layer


MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
PREFILL_IDS = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024]], dtype=torch.long)
DECODE_ID = torch.tensor([[9]], dtype=torch.long)
ATOL = 5e-4


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return model


def test_prefill_then_decode_layer0(hf_model):
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = l3_config.Llama3Config.from_hf_dict(hf_cfg_dict)

    api_blk = l3_layer.build_llama3_decoder_layer(
        cfg, layer_idx=0, max_seq=cfg.max_position_embeddings,
    )
    l3_layer.load_hf_llama3_layer(api_blk, hf_model.state_dict(), layer_idx=0)
    api_blk.eval()

    hf_layer = hf_model.model.layers[0]

    # ---- HF prefill ----
    with torch.no_grad():
        prefill_h = hf_model.model.embed_tokens(PREFILL_IDS)
        B, S_pre = PREFILL_IDS.shape
        pos_pre = torch.arange(S_pre).unsqueeze(0)
        cos_pre, sin_pre = hf_model.model.rotary_emb(prefill_h, pos_pre)
        attn_mask_pre = torch.full((1, 1, S_pre, S_pre), float("-inf"))
        attn_mask_pre = torch.triu(attn_mask_pre, diagonal=1)
        hf_cache = DynamicCache()
        hf_pre_out = hf_layer(
            prefill_h,
            attention_mask=attn_mask_pre,
            position_ids=pos_pre,
            position_embeddings=(cos_pre, sin_pre),
            past_key_values=hf_cache,
            use_cache=True,
        )

        # ---- HF decode (single token) ----
        decode_h = hf_model.model.embed_tokens(DECODE_ID)
        pos_dec = torch.tensor([[S_pre]])
        cos_dec, sin_dec = hf_model.model.rotary_emb(decode_h, pos_dec)
        hf_dec_out = hf_layer(
            decode_h,
            attention_mask=None,
            position_ids=pos_dec,
            position_embeddings=(cos_dec, sin_dec),
            past_key_values=hf_cache,
            use_cache=True,
        )

    # ---- API prefill + decode ----
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32,
        v_dtype=torch.float32,
    )
    api_cache = kvcache.ContiguousKVCache(
        cache_spec,
        batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq=cfg.max_position_embeddings,
    )
    with torch.no_grad():
        api_prefill_h = hf_model.model.embed_tokens(PREFILL_IDS)
        api_decode_h = hf_model.model.embed_tokens(DECODE_ID)
        api_pre_out = api_blk(
            api_prefill_h,
            position_ids=torch.arange(S_pre),
            cache=api_cache,
            start_pos=0,
        )
        api_dec_out = api_blk(
            api_decode_h,
            position_ids=torch.tensor([S_pre]),
            cache=api_cache,
            start_pos=S_pre,
        )

    pre_diff = (hf_pre_out - api_pre_out).abs().max().item()
    dec_diff = (hf_dec_out - api_dec_out).abs().max().item()
    print(f"Llama 3.2 1B prefill max_abs_diff={pre_diff:.3e}, decode max_abs_diff={dec_diff:.3e}")

    assert torch.allclose(hf_pre_out, api_pre_out, atol=ATOL), (
        f"prefill max_abs_diff={pre_diff:.6e}, atol={ATOL}"
    )
    assert torch.allclose(hf_dec_out, api_dec_out, atol=ATOL), (
        f"decode max_abs_diff={dec_diff:.6e}, atol={ATOL}"
    )
    assert api_cache.seq_len == S_pre + 1
