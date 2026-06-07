"""Isolation tests against HF Qwen3MoeSparseMoeBlock + Qwen3MoeDecoderLayer.

Verify our MoE matches HF's Qwen3MoeSparseMoeBlock at atol=5e-4 on synthetic
weights, and that a full DecoderBlock matches Qwen3MoeDecoderLayer in both
the MoE branch and the dense branch (via mlp_only_layers).
"""
import pytest
import torch

from api import feedforward, kvcache, specs, types
from models.qwen3_moe import config as m_config, layer as m_layer


ATOL = 5e-4
RTOL = 5e-4


def _make_hf_qwen3_moe_config(mlp_only_layers=None):
    from transformers.models.qwen3_moe import configuration_qwen3_moe as cfg_mod
    cfg = cfg_mod.Qwen3MoeConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        num_experts_per_tok=2,
        num_experts=8,
        norm_topk_prob=True,
        mlp_only_layers=mlp_only_layers,
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_qwen3_moe_router_matches_hf_at_5e_4():
    """B6: our MoE softmax router matches HF Qwen3MoeSparseMoeBlock at atol=5e-4."""
    from transformers.models.qwen3_moe import modeling_qwen3_moe as mod
    torch.manual_seed(13)
    hf_cfg = _make_hf_qwen3_moe_config()
    hf_moe = mod.Qwen3MoeSparseMoeBlock(hf_cfg)
    hf_moe.eval()

    expert_ffn = specs.FFNSpec(
        intermediate_size=hf_cfg.moe_intermediate_size,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )
    moe_spec = specs.MoESpec(
        n_experts=hf_cfg.num_experts,
        top_k=hf_cfg.num_experts_per_tok,
        n_shared_experts=0,
        router_kind="softmax",
        router_norm=hf_cfg.norm_topk_prob,
        score_correction_bias=False,
        group_routing=None,
        routed_scaling_factor=1.0,
        expert_ffn=expert_ffn,
    )
    api_moe = feedforward.MoE(moe_spec, hidden_size=hf_cfg.hidden_size,
                              dtype=torch.float32)

    # Copy HF weights. HF Qwen3MoeTopKRouter init is `torch.zeros`, so we
    # randomize after construction to make routing meaningful.
    with torch.no_grad():
        hf_moe.gate.weight.normal_(0, 1)
        api_moe.gate.weight.copy_(hf_moe.gate.weight.data)
        api_moe.experts_gate_up.copy_(hf_moe.experts.gate_up_proj.data)
        api_moe.experts_down.copy_(hf_moe.experts.down_proj.data)

    x = torch.randn(1, 5, hf_cfg.hidden_size)
    with torch.no_grad():
        hf_out = hf_moe(x)
        api_out = api_moe(x)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"Qwen3MoE MoE diff = {max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff = {max_abs_diff:.6f}"
    )


def _small_api_cfg(hf_cfg, mlp_only_layers=()) -> m_config.Qwen3MoeConfig:
    return m_config.Qwen3MoeConfig(
        hidden_size=hf_cfg.hidden_size,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        intermediate_size=hf_cfg.intermediate_size,
        moe_intermediate_size=hf_cfg.moe_intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_experts=hf_cfg.num_experts,
        num_experts_per_tok=hf_cfg.num_experts_per_tok,
        norm_topk_prob=hf_cfg.norm_topk_prob,
        decoder_sparse_step=hf_cfg.decoder_sparse_step,
        mlp_only_layers=mlp_only_layers,
        rope_theta=hf_cfg.rope_parameters["rope_theta"],
        rms_norm_eps=hf_cfg.rms_norm_eps,
        vocab_size=hf_cfg.vocab_size,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        tie_word_embeddings=False,
        attention_bias=False,
        dtype=torch.float32,
    )


def _copy_hf_to_api_layer(hf_layer, api_blk, layer_idx, cfg, is_moe):
    sd = {}
    L = layer_idx
    sd[f"model.layers.{L}.input_layernorm.weight"] = hf_layer.input_layernorm.weight.data
    sd[f"model.layers.{L}.post_attention_layernorm.weight"] = hf_layer.post_attention_layernorm.weight.data
    sa = hf_layer.self_attn
    sd[f"model.layers.{L}.self_attn.q_proj.weight"] = sa.q_proj.weight.data
    sd[f"model.layers.{L}.self_attn.k_proj.weight"] = sa.k_proj.weight.data
    sd[f"model.layers.{L}.self_attn.v_proj.weight"] = sa.v_proj.weight.data
    sd[f"model.layers.{L}.self_attn.o_proj.weight"] = sa.o_proj.weight.data
    sd[f"model.layers.{L}.self_attn.q_norm.weight"] = sa.q_norm.weight.data
    sd[f"model.layers.{L}.self_attn.k_norm.weight"] = sa.k_norm.weight.data
    if is_moe:
        moe = hf_layer.mlp
        sd[f"model.layers.{L}.mlp.gate.weight"] = moe.gate.weight.data
        sd[f"model.layers.{L}.mlp.experts.gate_up_proj"] = moe.experts.gate_up_proj.data
        sd[f"model.layers.{L}.mlp.experts.down_proj"]   = moe.experts.down_proj.data
    else:
        sd[f"model.layers.{L}.mlp.gate_proj.weight"] = hf_layer.mlp.gate_proj.weight.data
        sd[f"model.layers.{L}.mlp.up_proj.weight"]   = hf_layer.mlp.up_proj.weight.data
        sd[f"model.layers.{L}.mlp.down_proj.weight"] = hf_layer.mlp.down_proj.weight.data
    m_layer.load_hf_qwen3_moe_layer(api_blk, sd, layer_idx=layer_idx, cfg=cfg)


def _run_hf_and_api(layer_idx, mlp_only_layers=()):
    from transformers.models.qwen3_moe import modeling_qwen3_moe as mod
    torch.manual_seed(7 + layer_idx)
    hf_cfg = _make_hf_qwen3_moe_config(mlp_only_layers=list(mlp_only_layers))
    hf_layer = mod.Qwen3MoeDecoderLayer(hf_cfg, layer_idx=layer_idx)
    hf_layer.eval()
    # Randomize gate weight (HF inits to zeros).
    if isinstance(hf_layer.mlp, mod.Qwen3MoeSparseMoeBlock):
        with torch.no_grad():
            hf_layer.mlp.gate.weight.normal_(0, 1)
    rotary = mod.Qwen3MoeRotaryEmbedding(hf_cfg)
    rotary.eval()

    B, S = 1, 5
    x = torch.randn(B, S, hf_cfg.hidden_size)
    position_ids = torch.arange(S).unsqueeze(0)
    attn_mask = torch.full((B, 1, S, S), float("-inf"))
    attn_mask = torch.triu(attn_mask, diagonal=1)

    with torch.no_grad():
        cos, sin = rotary(x, position_ids)
        hf_out = hf_layer(
            x,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

    api_cfg = _small_api_cfg(hf_cfg, mlp_only_layers=mlp_only_layers)
    api_blk = m_layer.build_qwen3_moe_decoder_layer(api_cfg, layer_idx=layer_idx,
                                                    max_seq=32)
    api_blk.eval()
    is_moe = api_cfg._is_moe_layer(layer_idx)
    _copy_hf_to_api_layer(hf_layer, api_blk, layer_idx, cfg=api_cfg, is_moe=is_moe)

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=api_cfg.num_key_value_heads,
        head_dim=api_cfg.head_dim, max_seq=32,
    )
    with torch.no_grad():
        api_out = api_blk(x, position_ids=torch.arange(S),
                          cache=cache, start_pos=0)
    return hf_out, api_out


def test_qwen3_moe_decoder_layer_matches_hf_moe_branch_at_5e_4():
    """B6: a Qwen3-MoE DecoderBlock in the MoE branch matches HF at atol=5e-4."""
    hf_out, api_out = _run_hf_and_api(layer_idx=0)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"Qwen3MoE layer (MoE) max_abs_diff = {max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )


def test_qwen3_moe_decoder_layer_matches_hf_dense_branch_at_5e_4():
    """B6: a Qwen3-MoE DecoderBlock in the DENSE branch (via mlp_only_layers)
    matches HF at atol=5e-4. Verifies the dense fallback path uses
    `intermediate_size` (not moe_intermediate_size)."""
    hf_out, api_out = _run_hf_and_api(layer_idx=0, mlp_only_layers=(0,))
    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"Qwen3MoE layer (dense) max_abs_diff = {max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
