"""Numerical equivalence vs HF DeepseekV2DecoderLayer with RANDOM weights.

Verifies BOTH branches of V2-Lite at atol=5e-4:
1. Dense (layer 0): MLA + standard SwiGLU FFN.
2. MoE  (layer 1): MLA + DeepseekV2Moe (softmax router + shared experts).
"""
import pytest
import torch

from api import kvcache, specs, types
from models.deepseek_v2_lite import config as m_config, layer as m_layer


ATOL = 5e-4
RTOL = 5e-4


def _small_hf_v2_config(rope_type: str = "default", first_k_dense_replace: int = 1):
    """Build a small HF DeepseekV2Config — V2-Lite-style.

    rope_type="default" avoids the YARN init code path so we can verify
    the rest of the stack first. A second test exercises YARN.
    """
    from transformers.models.deepseek_v2 import configuration_deepseek_v2 as cfg_mod

    rope_params = {"rope_type": "default", "rope_theta": 10000.0}
    if rope_type == "yarn":
        rope_params = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 40.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 64,
        }

    cfg = cfg_mod.DeepseekV2Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=24,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=2,
        n_group=None,
        topk_group=None,
        topk_method="greedy",
        norm_topk_prob=False,
        first_k_dense_replace=first_k_dense_replace,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=None,                # V2-Lite direct q_proj
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_parameters=rope_params,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _small_api_cfg(
    hf_cfg, rope_type: str = "default",
) -> m_config.DeepSeekV2LiteConfig:
    kwargs = dict(
        hidden_size=hf_cfg.hidden_size,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        intermediate_size=hf_cfg.intermediate_size,
        moe_intermediate_size=hf_cfg.moe_intermediate_size,
        n_routed_experts=hf_cfg.n_routed_experts,
        n_shared_experts=hf_cfg.n_shared_experts,
        num_experts_per_tok=hf_cfg.num_experts_per_tok,
        routed_scaling_factor=hf_cfg.routed_scaling_factor,
        norm_topk_prob=hf_cfg.norm_topk_prob,
        n_group=hf_cfg.n_group or 1,
        topk_group=hf_cfg.topk_group or 1,
        first_k_dense_replace=hf_cfg.first_k_dense_replace,
        kv_lora_rank=hf_cfg.kv_lora_rank,
        qk_nope_head_dim=hf_cfg.qk_nope_head_dim,
        qk_rope_head_dim=hf_cfg.qk_rope_head_dim,
        v_head_dim=hf_cfg.v_head_dim,
        q_lora_rank=hf_cfg.q_lora_rank,
        rope_theta=10000.0,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        vocab_size=hf_cfg.vocab_size,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        tie_word_embeddings=False,
        attention_bias=False,
        dtype=torch.float32,
        rope_type=rope_type,
    )
    if rope_type == "yarn":
        kwargs.update(
            yarn_factor=40.0,
            yarn_beta_fast=32,
            yarn_beta_slow=1,
            yarn_mscale=0.707,
            yarn_mscale_all_dim=0.707,
            yarn_original_max_position_embeddings=64,
        )
    return m_config.DeepSeekV2LiteConfig(**kwargs)


def _copy_hf_to_api_layer(hf_layer, api_blk, layer_idx, is_moe: bool, cfg):
    """Build a synthetic state-dict in HF naming, then call our loader."""
    sd = {}
    L = layer_idx
    sd[f"model.layers.{L}.input_layernorm.weight"] = hf_layer.input_layernorm.weight.data
    sd[f"model.layers.{L}.post_attention_layernorm.weight"] = hf_layer.post_attention_layernorm.weight.data

    sa = hf_layer.self_attn
    # V2-Lite has q_proj (q_lora_rank=null).
    sd[f"model.layers.{L}.self_attn.q_proj.weight"]            = sa.q_proj.weight.data
    sd[f"model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight"] = sa.kv_a_proj_with_mqa.weight.data
    sd[f"model.layers.{L}.self_attn.kv_a_layernorm.weight"]    = sa.kv_a_layernorm.weight.data
    sd[f"model.layers.{L}.self_attn.kv_b_proj.weight"]          = sa.kv_b_proj.weight.data
    sd[f"model.layers.{L}.self_attn.o_proj.weight"]             = sa.o_proj.weight.data

    if is_moe:
        moe = hf_layer.mlp
        sd[f"model.layers.{L}.mlp.gate.weight"] = moe.gate.weight.data
        sd[f"model.layers.{L}.mlp.experts.gate_up_proj"] = moe.experts.gate_up_proj.data
        sd[f"model.layers.{L}.mlp.experts.down_proj"]   = moe.experts.down_proj.data
        if cfg.n_shared_experts > 0:
            sd[f"model.layers.{L}.mlp.shared_experts.gate_proj.weight"] = moe.shared_experts.gate_proj.weight.data
            sd[f"model.layers.{L}.mlp.shared_experts.up_proj.weight"]   = moe.shared_experts.up_proj.weight.data
            sd[f"model.layers.{L}.mlp.shared_experts.down_proj.weight"] = moe.shared_experts.down_proj.weight.data
    else:
        sd[f"model.layers.{L}.mlp.gate_proj.weight"] = hf_layer.mlp.gate_proj.weight.data
        sd[f"model.layers.{L}.mlp.up_proj.weight"]   = hf_layer.mlp.up_proj.weight.data
        sd[f"model.layers.{L}.mlp.down_proj.weight"] = hf_layer.mlp.down_proj.weight.data

    m_layer.load_hf_deepseek_v2_lite_layer(api_blk, sd, layer_idx=layer_idx, cfg=cfg)


def _run_hf_and_api(layer_idx: int, rope_type: str):
    """Build matching HF + api blocks at layer_idx, run forward, return (hf, api)."""
    from transformers.models.deepseek_v2 import modeling_deepseek_v2 as mod
    torch.manual_seed(7 + layer_idx)
    hf_cfg = _small_hf_v2_config(rope_type=rope_type)

    hf_layer = mod.DeepseekV2DecoderLayer(hf_cfg, layer_idx=layer_idx)
    hf_layer.eval()
    # HF rotary lives on the model. We instantiate it ourselves and call it
    # to produce freqs_cis (complex tensor for the V2 RoPE).
    rotary = mod.DeepseekV2RotaryEmbedding(hf_cfg)
    rotary.eval()

    # Inputs.
    B, S = 1, 5
    x = torch.randn(B, S, hf_cfg.hidden_size)
    position_ids = torch.arange(S).unsqueeze(0)
    # Build a causal (B, 1, S, S) mask with -inf above diag.
    attn_mask = torch.full((B, 1, S, S), float("-inf"))
    attn_mask = torch.triu(attn_mask, diagonal=1)

    # Compute position_embeddings (complex tensor [B, S, dim//2]) via HF rotary.
    with torch.no_grad():
        freqs_cis = rotary(x, position_ids=position_ids)
        hf_out = hf_layer(
            x,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=freqs_cis,
        )
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

    # Build api block.
    is_moe = layer_idx >= hf_cfg.first_k_dense_replace
    api_cfg = _small_api_cfg(hf_cfg, rope_type=rope_type)
    api_blk = m_layer.build_deepseek_v2_lite_decoder_layer(
        api_cfg, layer_idx=layer_idx, max_seq=32,
    )
    api_blk.eval()
    _copy_hf_to_api_layer(hf_layer, api_blk, layer_idx, is_moe=is_moe, cfg=api_cfg)

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=api_cfg.num_attention_heads,
        head_dim=api_cfg.qk_head_dim, max_seq=32,
        v_head_dim=api_cfg.v_head_dim,
    )
    with torch.no_grad():
        api_out = api_blk(x, position_ids=torch.arange(S),
                          cache=cache, start_pos=0)
    return hf_out, api_out


def test_v2_lite_layer0_dense_numerical_equivalence_default_rope():
    """B5: layer 0 (dense FFN) matches HF DeepseekV2DecoderLayer at atol=5e-4
    with default RoPE."""
    hf_out, api_out = _run_hf_and_api(layer_idx=0, rope_type="default")
    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"V2-Lite layer-0 (dense, default RoPE) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )


def test_v2_lite_layer1_moe_numerical_equivalence_default_rope():
    """B5: layer 1 (MoE: softmax router + shared experts) matches HF
    DeepseekV2DecoderLayer at atol=5e-4 with default RoPE."""
    hf_out, api_out = _run_hf_and_api(layer_idx=1, rope_type="default")
    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"V2-Lite layer-1 (MoE, default RoPE) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )


def test_v2_lite_layer0_dense_numerical_equivalence_yarn_rope():
    """B5: layer 0 (dense) matches HF at atol=5e-4 with YARN-scaled RoPE."""
    hf_out, api_out = _run_hf_and_api(layer_idx=0, rope_type="yarn")
    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"V2-Lite layer-0 (dense, YARN) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )


def test_v2_lite_layer1_moe_numerical_equivalence_yarn_rope():
    """B5: layer 1 (MoE) matches HF at atol=5e-4 with YARN-scaled RoPE."""
    hf_out, api_out = _run_hf_and_api(layer_idx=1, rope_type="yarn")
    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"V2-Lite layer-1 (MoE, YARN) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
