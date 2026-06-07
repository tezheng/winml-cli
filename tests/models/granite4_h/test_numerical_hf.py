"""Granite 4 H numerical gate vs the HF reference.

Instantiates a small GraniteMoeHybridForCausalLM with random weights, then
compares both Mamba and Attention layer-types against our api block.

ATOL = 5e-4 (B7 numerical gate).
"""
import pytest
import torch

pytest.importorskip("transformers")

from api import kvcache, specs, types
from models.granite4_h import config as g_config, layer as g_layer


ATOL = 5e-4
RTOL = 5e-4


@pytest.fixture(scope="module")
def hf_small_model():
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM
    hf_cfg = GraniteMoeHybridConfig(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=256, shared_intermediate_size=256,
        num_hidden_layers=4, vocab_size=100,
        layer_types=["mamba", "mamba", "attention", "mamba"],
        mamba_n_heads=4, mamba_n_groups=2, mamba_d_state=16,
        mamba_d_head=64, mamba_d_conv=4, mamba_expand=2,
        mamba_chunk_size=8,
        residual_multiplier=0.22, attention_multiplier=0.25,
        embedding_multiplier=12.0, logits_scaling=8.0,
        position_embedding_type="nope",
        num_local_experts=0,
    )
    torch.manual_seed(42)
    model = GraniteMoeHybridForCausalLM(hf_cfg).to(torch.float32).eval()
    return model


def _api_cfg(hf_model):
    hf_cfg_dict = hf_model.config.to_dict()
    return g_config.Granite4HConfig.from_hf_dict(hf_cfg_dict)


def test_mamba_layer_0_matches_hf(hf_small_model):
    cfg = _api_cfg(hf_small_model)
    assert cfg.is_mamba_layer(0)

    api_blk = g_layer.build_granite4_h_decoder_layer(cfg, layer_idx=0)
    full_sd = hf_small_model.state_dict()
    g_layer.load_hf_granite4_h_layer(api_blk, full_sd, cfg, layer_idx=0)
    api_blk.eval()

    torch.manual_seed(0)
    B, S = 1, 9
    hidden = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)

    hf_layer = hf_small_model.model.layers[0]
    with torch.no_grad():
        # HF layer forward needs attention_mask=None and works for both block types.
        hf_out = hf_layer(hidden, attention_mask=None, past_key_values=None,
                          use_cache=False)

    mixer = api_blk.attention
    cache = kvcache.SSMStateCache(
        batch_size=B, conv_dim=mixer.conv_dim,
        conv_kernel=mixer.conv_kernel,
        n_heads=mixer.num_heads, head_dim=mixer.head_dim,
        d_state=mixer.d_state,
    )
    with torch.no_grad():
        api_out = api_blk(hidden, cache=cache)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Granite4H mamba layer-0 max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )


def test_attention_layer_2_matches_hf(hf_small_model):
    cfg = _api_cfg(hf_small_model)
    assert not cfg.is_mamba_layer(2)

    api_blk = g_layer.build_granite4_h_decoder_layer(cfg, layer_idx=2)
    full_sd = hf_small_model.state_dict()
    g_layer.load_hf_granite4_h_layer(api_blk, full_sd, cfg, layer_idx=2)
    api_blk.eval()

    torch.manual_seed(0)
    B, S = 1, 9
    hidden = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)

    hf_layer = hf_small_model.model.layers[2]
    # HF attention path needs an attention_mask AND position_embeddings (when
    # rotary_emb is None, position_embeddings is None → no RoPE). We need to
    # supply a causal mask shaped like (1, 1, S, S) with -inf above diagonal.
    attn_mask = torch.full((1, 1, S, S), float("-inf"))
    attn_mask = torch.triu(attn_mask, diagonal=1)
    with torch.no_grad():
        hf_out = hf_layer(
            hidden,
            attention_mask=attn_mask,
            past_key_values=None,
            use_cache=False,
            position_embeddings=None,  # NoPE — no rotary
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    kv_cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(hidden, position_ids=pos, cache=kv_cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Granite4H attention layer-2 max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
