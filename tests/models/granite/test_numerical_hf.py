"""Gate test: full Granite 3.1-2B-Base decoder layer numerical equivalence vs HF.

Runs on CPU, fp32. Uses ibm-granite/granite-3.1-2b-base (~5 GB).

Exercises:
- GQA n_q=32/n_kv=8 head_dim=64.
- attention_multiplier=0.015625 (μP, NOT 1/sqrt(64) ≈ 0.125 — 8× difference).
- residual_multiplier=0.22 (μP, scales BOTH sublayer outputs before residual add).
- RMSNorm STANDARD_W, SwiGLU, no biases, no QK-norm.
- Default RoPE theta=5_000_000 (no scaling).
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.granite import config as g_config, layer as g_layer


MODEL_ID = "ibm-granite/granite-3.1-2b-base"
FIXED_INPUT = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)
ATOL = 5e-4
RTOL = 5e-4


def _granite_weights_complete(cache_dir: str) -> bool:
    """Check the model is fully downloaded. The hf cache can live either at
    `<cache_dir>/hub/models--…/` (transformers default) or at
    `<cache_dir>/models--…/` (huggingface_hub snapshot_download with our cache_dir).
    """
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
        pytest.skip(
            f"Granite 3.1-2B weights not fully downloaded (cache_dir={cache_dir})."
        )
    try:
        _ = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    except Exception as e:
        pytest.skip(
            f"Granite 3.1-2B weights load failed: {type(e).__name__}: "
            f"{str(e)[:200]}"
        )
    model.eval()
    return model


def test_layer0_forward_matches_hf(hf_model):
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = g_config.GraniteConfig.from_hf_dict(hf_cfg_dict)

    api_blk = g_layer.build_granite_decoder_layer(
        cfg, layer_idx=0, max_seq=64,  # small to limit RoPE cache size
    )
    full_sd = hf_model.state_dict()
    g_layer.load_hf_granite_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)
        # NOTE: Granite scales embeddings by embedding_multiplier inside the
        # model. We compare LAYER OUTPUTS, not LOGITS. So we must feed both
        # implementations the SAME pre-layer hidden state. HF's layer-0 input
        # is `embed_out * embedding_multiplier`. We use that as the api's `x`
        # input too. (modeling_granite.py:405)
        layer_input = embed_out * hf_model.model.embedding_multiplier
    B, S = FIXED_INPUT.shape

    hf_layer = hf_model.model.layers[0]
    position_ids = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = hf_model.model.rotary_emb(layer_input, position_ids)
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_out = hf_layer(
            layer_input,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_values=None,
            use_cache=False,
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32,
        v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec,
        batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq=64,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(layer_input, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"Granite 3.1-2B layer-0 max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
