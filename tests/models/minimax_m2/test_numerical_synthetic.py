"""MiniMax-M2 numerical gate vs HF DecoderLayer.

Instantiates a small MiniMaxM2DecoderLayer, copies weights into our api block,
verifies numerical equivalence at atol=5e-4.

ATOL = 5e-4 (v7 numerical gate).
"""
import pytest
import torch

pytest.importorskip("transformers")

from api import kvcache, specs, types
from models.minimax_m2 import config as _c, layer as _l


ATOL = 5e-4
RTOL = 5e-4


def _hf_cfg():
    from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
    cfg = MiniMaxM2Config(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=100,
        max_position_embeddings=128,
        num_local_experts=8,
        num_experts_per_tok=2,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 5_000_000.0},
        # Bring bos/eos inside the synthetic vocab to silence config warning.
        bos_token_id=0, eos_token_id=99,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _api_cfg_from_hf(hf_cfg):
    return _c.MiniMaxM2Config(
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.head_dim,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        vocab_size=hf_cfg.vocab_size,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        tie_word_embeddings=False,
        rope_theta=5_000_000.0,
        num_experts_per_tok=hf_cfg.num_experts_per_tok,
        num_local_experts=hf_cfg.num_local_experts,
        dtype=torch.float32,
    )


def _copy_hf_to_api(api_blk, hf_layer):
    """Copy HF MiniMaxM2DecoderLayer weights into our api DecoderBlock."""
    sd = hf_layer.state_dict()
    with torch.no_grad():
        api_blk.pre_attn_norm.weight.copy_(sd["input_layernorm.weight"])
        api_blk.pre_ffn_norm.weight.copy_(sd["post_attention_layernorm.weight"])
        # Attention.
        api_blk.attention.q_proj.weight.copy_(sd["self_attn.q_proj.weight"])
        api_blk.attention.k_proj.weight.copy_(sd["self_attn.k_proj.weight"])
        api_blk.attention.v_proj.weight.copy_(sd["self_attn.v_proj.weight"])
        api_blk.attention.o_proj.weight.copy_(sd["self_attn.o_proj.weight"])
        api_blk.attention.q_norm.weight.copy_(sd["self_attn.q_norm.weight"])
        api_blk.attention.k_norm.weight.copy_(sd["self_attn.k_norm.weight"])
        # MoE.
        api_blk.feedforward.gate.weight.copy_(sd["mlp.gate.weight"])
        # HF stores e_score_correction_bias on the SparseMoeBlock, not on
        # gate — see layer.md.
        api_blk.feedforward.gate.e_score_correction_bias.copy_(
            sd["mlp.e_score_correction_bias"].float()
        )
        api_blk.feedforward.experts_gate_up.copy_(sd["mlp.experts.gate_up_proj"])
        api_blk.feedforward.experts_down.copy_(sd["mlp.experts.down_proj"])


def test_layer_0_matches_hf():
    """B4: full DecoderLayer (attention + MoE) matches HF at atol=5e-4."""
    from transformers.models.minimax_m2.modeling_minimax_m2 import (
        MiniMaxM2DecoderLayer, MiniMaxM2RotaryEmbedding,
    )
    torch.manual_seed(42)
    hf_cfg = _hf_cfg()
    hf_layer = MiniMaxM2DecoderLayer(hf_cfg, layer_idx=0).float().eval()

    # Inject a non-trivial e_score_correction_bias so routing isn't degenerate
    # (HF default is zeros).
    with torch.no_grad():
        bias = torch.randn(hf_cfg.num_local_experts) * 0.3
        hf_layer.mlp.e_score_correction_bias.copy_(bias)

    api_cfg = _api_cfg_from_hf(hf_cfg)
    api_blk = _l.build_minimax_m2_decoder_layer(api_cfg, layer_idx=0)
    api_blk.eval()
    _copy_hf_to_api(api_blk, hf_layer)

    rope = MiniMaxM2RotaryEmbedding(hf_cfg)

    B, S = 1, 5
    torch.manual_seed(7)
    x = torch.randn(B, S, hf_cfg.hidden_size)
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = rope(x, pos)
    causal = torch.zeros(1, 1, S, S)
    causal = causal.masked_fill(
        torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"),
    )

    with torch.no_grad():
        hf_out = hf_layer(
            x, position_embeddings=(cos, sin), attention_mask=causal,
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    kv_cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B, n_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.head_dim, max_seq=hf_cfg.max_position_embeddings,
    )
    pos_1d = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(x, position_ids=pos_1d, cache=kv_cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"MiniMax-M2 layer-0 max_abs_diff={max_abs_diff:.6e}"
    )


def test_load_hf_via_helper():
    """Verify the public load_hf_minimax_m2_layer helper round-trips."""
    from transformers.models.minimax_m2.modeling_minimax_m2 import (
        MiniMaxM2DecoderLayer, MiniMaxM2RotaryEmbedding,
    )
    torch.manual_seed(42)
    hf_cfg = _hf_cfg()
    hf_layer = MiniMaxM2DecoderLayer(hf_cfg, layer_idx=0).float().eval()

    api_cfg = _api_cfg_from_hf(hf_cfg)
    api_blk = _l.build_minimax_m2_decoder_layer(api_cfg, layer_idx=0)

    # Wrap HF layer state dict with model.layers.0 prefix.
    full_sd = {
        f"model.layers.0.{k}": v for k, v in hf_layer.state_dict().items()
    }
    _l.load_hf_minimax_m2_layer(api_blk, full_sd, api_cfg, layer_idx=0)
    api_blk.eval()

    rope = MiniMaxM2RotaryEmbedding(hf_cfg)
    B, S = 1, 4
    x = torch.randn(B, S, hf_cfg.hidden_size)
    pos = torch.arange(S).unsqueeze(0)
    cos, sin = rope(x, pos)
    causal = torch.zeros(1, 1, S, S)
    causal = causal.masked_fill(
        torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"),
    )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    kv_cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B, n_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.head_dim, max_seq=hf_cfg.max_position_embeddings,
    )
    pos_1d = torch.arange(S)
    with torch.no_grad():
        hf_out = hf_layer(x, position_embeddings=(cos, sin), attention_mask=causal)
        api_out = api_blk(x, position_ids=pos_1d, cache=kv_cache, start_pos=0)
    diff = (hf_out - api_out).abs().max().item()
    assert diff < ATOL, f"helper-loaded max_abs_diff={diff:.6e}"
