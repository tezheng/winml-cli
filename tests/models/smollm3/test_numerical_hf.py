"""Gate test: full SmolLM3-3B decoder layer numerical equivalence vs HF.

Runs on CPU, fp32. Uses HuggingFaceTB/SmolLM3-3B. Tests TWO layers:
  - Layer 0: RoPE layer (no_rope_layers[0] == 1).
  - Layer 3: NoPE layer (no_rope_layers[3] == 0).

Together these exercise the per-layer NoPE dispatch — the IR-novel B1 path.
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.smollm3 import config as s_config, layer as s_layer


MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
FIXED_INPUT = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)
ATOL = 5e-4
RTOL = 5e-4


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    _ = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return model


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_layer_forward_matches_hf(hf_model, layer_idx):
    """layer_idx=0 is a RoPE layer; layer_idx=3 is a NoPE layer in SmolLM3-3B."""
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = s_config.Smollm3Config.from_hf_dict(hf_cfg_dict)

    api_blk = s_layer.build_smollm3_decoder_layer(
        cfg, layer_idx=layer_idx, max_seq=cfg.max_position_embeddings,
    )
    full_sd = hf_model.state_dict()
    s_layer.load_hf_smollm3_layer(api_blk, full_sd, layer_idx=layer_idx)
    api_blk.eval()

    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)
    B, S = FIXED_INPUT.shape

    hf_layer = hf_model.model.layers[layer_idx]
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
    use_rope = cfg.layer_uses_rope(layer_idx)
    tag = "RoPE" if use_rope else "NoPE"
    print(f"SmolLM3-3B layer-{layer_idx} ({tag}) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"layer-{layer_idx} ({tag}) max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
