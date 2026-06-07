"""Verify our ContiguousKVCache prefill+decode produces the same layer-0 output
as HF for Granite 3.1-2B-Base.
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from api import kvcache, specs, types
from models.granite import config as g_config, layer as g_layer


MODEL_ID = "ibm-granite/granite-3.1-2b-base"
PREFILL_IDS = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024]], dtype=torch.long)
DECODE_ID = torch.tensor([[9]], dtype=torch.long)
ATOL = 5e-4


def _granite_weights_complete(cache_dir: str) -> bool:
    import glob
    for root in (cache_dir, os.path.join(cache_dir, "hub")):
        pat = os.path.join(
            root, "models--ibm-granite--granite-3.1-2b-base", "blobs", "*.incomplete"
        )
        if glob.glob(pat):
            return False
        snap = glob.glob(os.path.join(
            root, "models--ibm-granite--granite-3.1-2b-base", "snapshots", "*"
        ))
        if snap:
            return any(
                f.endswith(".safetensors") and not os.path.basename(f).startswith(".")
                for f in os.listdir(snap[0])
            )
    return False


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    if not _granite_weights_complete(cache_dir):
        pytest.skip("Granite 3.1-2B weights not fully downloaded")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    except Exception as e:
        pytest.skip(f"Granite 3.1-2B load failed: {type(e).__name__}: {str(e)[:200]}")
    model.eval()
    return model


def test_prefill_then_decode_layer0(hf_model):
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = g_config.GraniteConfig.from_hf_dict(hf_cfg_dict)

    api_blk = g_layer.build_granite_decoder_layer(
        cfg, layer_idx=0, max_seq=64,
    )
    g_layer.load_hf_granite_layer(api_blk, hf_model.state_dict(), layer_idx=0)
    api_blk.eval()

    hf_layer = hf_model.model.layers[0]
    emb_mult = hf_model.model.embedding_multiplier

    with torch.no_grad():
        prefill_h = hf_model.model.embed_tokens(PREFILL_IDS) * emb_mult
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

        decode_h = hf_model.model.embed_tokens(DECODE_ID) * emb_mult
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
        max_seq=64,
    )
    with torch.no_grad():
        api_prefill_h = hf_model.model.embed_tokens(PREFILL_IDS) * emb_mult
        api_decode_h = hf_model.model.embed_tokens(DECODE_ID) * emb_mult
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
    print(f"Granite 3.1-2B prefill max_abs_diff={pre_diff:.3e}, decode max_abs_diff={dec_diff:.3e}")

    assert torch.allclose(hf_pre_out, api_pre_out, atol=ATOL), (
        f"prefill max_abs_diff={pre_diff:.6e}, atol={ATOL}"
    )
    assert torch.allclose(hf_dec_out, api_dec_out, atol=ATOL), (
        f"decode max_abs_diff={dec_diff:.6e}, atol={ATOL}"
    )
    assert api_cache.seq_len == S_pre + 1
