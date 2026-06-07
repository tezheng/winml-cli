"""Gate test: full Mistral 7B v0.3 decoder layer numerical equivalence vs HF.

Runs on CPU, fp32. Uses mistralai/Mistral-7B-v0.3 (~14GB download).

Exercises: GQA n_q=32/n_kv=8 head_dim=128, RoPE default theta=1_000_000.0,
RMSNorm STANDARD_W, SwiGLU, no biases. v0.3 has sliding_window=None — so
this exercises plain CAUSAL mask. (The SWA path on v0.1/v0.2 is exercised
by the api unit tests + the layer-shape parametrize.)
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.mistral import config as m_config, layer as m_layer


MODEL_ID = "mistralai/Mistral-7B-v0.3"
FIXED_INPUT = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)
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
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    except Exception as e:
        pytest.skip(
            f"Mistral 7B v0.3 weights not accessible: {type(e).__name__}: "
            f"{str(e)[:200]}. Gate test stub remains; will execute once weights "
            "are local."
        )
    model.eval()
    return model


def test_layer0_forward_matches_hf(hf_model):
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = m_config.MistralConfig.from_hf_dict(hf_cfg_dict)

    api_blk = m_layer.build_mistral_decoder_layer(
        cfg, layer_idx=0, max_seq=cfg.max_position_embeddings,
    )
    full_sd = hf_model.state_dict()
    m_layer.load_hf_mistral_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)
    B, S = FIXED_INPUT.shape

    hf_layer = hf_model.model.layers[0]
    position_ids = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = hf_model.model.rotary_emb(embed_out, position_ids)
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
        k_dtype=torch.float32,
        v_dtype=torch.float32,
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
    print(f"Mistral 7B v0.3 layer-0 max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
