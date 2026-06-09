"""Jamba per-layer numerical gate (v6 B2).

Tests two distinct layer types end-to-end against HF Jamba decoder
layers at atol=5e-4:
- attention layer (layer_idx satisfies the period/offset rule)
- mamba layer (every other index)

The HF JambaMambaMixer uses Mamba-1 selective scan with intra-mixer
LayerNorms on dt/B/C. JambaAttention has NO RoPE.

Source: `transformers/models/jamba/modeling_jamba.py:570-638`.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.jamba.configuration_jamba import JambaConfig as HFJambaConfig  # noqa: E402
from transformers.models.jamba.modeling_jamba import (  # noqa: E402
    JambaAttentionDecoderLayer, JambaMambaDecoderLayer,
)

from api import specs, types, kvcache  # noqa: E402
from models.jamba import config as jamba_config, layer as jamba_layer  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _build_hf_jamba(hidden_size=128, intermediate_size=384,
                    num_heads=4, num_kv_heads=4):
    cfg = HFJambaConfig(
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_hidden_layers=4,
        num_attention_heads=num_heads, num_key_value_heads=num_kv_heads,
        attn_layer_period=2, attn_layer_offset=1,
        # Disable MoE: set expert_layer_offset == 0 with period > num_hidden_layers
        # so only the unreachable layer index 0 mod 999 == 0 matches when L=0;
        # the JambaModel constructor reads layers_num_experts (period=999,
        # offset=0 → only L=0 matches; but our test instantiates SINGLE layers
        # at idx 0 and 1, where idx=0 is mamba — Mamba layer with num_experts=1
        # falls through to JambaMLP (num_experts > 1 check), so this is safe).
        # We actually just bypass MoE entirely by using num_experts=1.
        expert_layer_period=999, expert_layer_offset=0,
        num_experts=1,
        num_experts_per_tok=1,
        use_mamba_kernels=False,        # force slow path
        mamba_d_state=8, mamba_d_conv=4, mamba_expand=2,
        mamba_dt_rank=8,
        vocab_size=256,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _copy_attention_layer_weights(hf_block, api_block) -> None:
    with torch.no_grad():
        api_block.pre_attn_norm.weight.copy_(hf_block.input_layernorm.weight)
        api_block.pre_ffn_norm.weight.copy_(hf_block.pre_ff_layernorm.weight)
        api_block.attention.q_proj.weight.copy_(hf_block.self_attn.q_proj.weight)
        api_block.attention.k_proj.weight.copy_(hf_block.self_attn.k_proj.weight)
        api_block.attention.v_proj.weight.copy_(hf_block.self_attn.v_proj.weight)
        api_block.attention.o_proj.weight.copy_(hf_block.self_attn.o_proj.weight)
        api_block.feedforward.gate_proj.weight.copy_(hf_block.feed_forward.gate_proj.weight)
        api_block.feedforward.up_proj.weight.copy_(hf_block.feed_forward.up_proj.weight)
        api_block.feedforward.down_proj.weight.copy_(hf_block.feed_forward.down_proj.weight)


def _copy_mamba_layer_weights(hf_block, api_block) -> None:
    with torch.no_grad():
        api_block.pre_attn_norm.weight.copy_(hf_block.input_layernorm.weight)
        api_block.pre_ffn_norm.weight.copy_(hf_block.pre_ff_layernorm.weight)
        m_api = api_block.attention
        m_hf = hf_block.mamba
        m_api.in_proj.weight.copy_(m_hf.in_proj.weight)
        m_api.conv1d.weight.copy_(m_hf.conv1d.weight)
        if m_api.conv1d.bias is not None and m_hf.conv1d.bias is not None:
            m_api.conv1d.bias.copy_(m_hf.conv1d.bias)
        m_api.x_proj.weight.copy_(m_hf.x_proj.weight)
        m_api.dt_proj.weight.copy_(m_hf.dt_proj.weight)
        m_api.dt_proj.bias.copy_(m_hf.dt_proj.bias)
        m_api.A_log.copy_(m_hf.A_log)
        m_api.D.copy_(m_hf.D)
        m_api.out_proj.weight.copy_(m_hf.out_proj.weight)
        m_api.dt_layernorm.weight.copy_(m_hf.dt_layernorm.weight)
        m_api.b_layernorm.weight.copy_(m_hf.b_layernorm.weight)
        m_api.c_layernorm.weight.copy_(m_hf.c_layernorm.weight)
        # FFN
        api_block.feedforward.gate_proj.weight.copy_(hf_block.feed_forward.gate_proj.weight)
        api_block.feedforward.up_proj.weight.copy_(hf_block.feed_forward.up_proj.weight)
        api_block.feedforward.down_proj.weight.copy_(hf_block.feed_forward.down_proj.weight)


def test_jamba_attention_layer_matches_hf():
    """layer_idx=1 (attention layer under period=2, offset=1)."""
    cfg_hf = _build_hf_jamba()
    torch.manual_seed(0)
    hf_block = JambaAttentionDecoderLayer(cfg_hf, layer_idx=1)
    hf_block.eval()

    cfg = jamba_config.JambaConfig(
        hidden_size=128, intermediate_size=384, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4,
        attn_layer_period=2, attn_layer_offset=1,
        expert_layer_period=999, expert_layer_offset=0,
        num_experts=1, num_experts_per_tok=1,
        mamba_d_state=8, mamba_d_conv=4, mamba_expand=2,
        mamba_dt_rank=8, vocab_size=256, rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )
    api_block = jamba_layer.build_jamba_decoder_layer(cfg, layer_idx=1)
    _copy_attention_layer_weights(hf_block, api_block)
    api_block.eval()

    torch.manual_seed(1)
    B, S = 1, 6
    x = torch.randn(B, S, cfg.hidden_size)
    # HF expects 4D additive mask.
    causal_mask = torch.full((1, 1, S, S), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        hf_out = hf_block(
            x, attention_mask=causal_mask, position_ids=pos,
            past_key_values=None, use_cache=False,
        )

    cache = kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=B, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    with torch.no_grad():
        api_out = api_block(x, position_ids=pos.squeeze(0), cache=cache, start_pos=0)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Jamba attention layer max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    print(f"Jamba attention layer max_abs_diff={max_abs_diff:.2e}")


def test_jamba_mamba_layer_matches_hf():
    """layer_idx=0 (mamba layer)."""
    cfg_hf = _build_hf_jamba()
    torch.manual_seed(0)
    hf_block = JambaMambaDecoderLayer(cfg_hf, layer_idx=0)
    hf_block.eval()

    cfg = jamba_config.JambaConfig(
        hidden_size=128, intermediate_size=384, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4,
        attn_layer_period=2, attn_layer_offset=1,
        expert_layer_period=999, expert_layer_offset=0,
        num_experts=1, num_experts_per_tok=1,
        mamba_d_state=8, mamba_d_conv=4, mamba_expand=2,
        mamba_dt_rank=8, vocab_size=256, rms_norm_eps=1e-6,
        max_position_embeddings=64,
    )
    api_block = jamba_layer.build_jamba_decoder_layer(cfg, layer_idx=0)
    _copy_mamba_layer_weights(hf_block, api_block)
    api_block.eval()

    torch.manual_seed(2)
    B, S = 1, 8
    x = torch.randn(B, S, cfg.hidden_size)
    with torch.no_grad():
        hf_out = hf_block(
            x, attention_mask=None, position_ids=None,
            past_key_values=None,
        )
    with torch.no_grad():
        api_out = api_block(x)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Jamba mamba layer max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    print(f"Jamba mamba layer max_abs_diff={max_abs_diff:.2e}")


def test_jamba_layer_type_dispatch():
    """to_block_spec dispatches based on layer_idx."""
    cfg = jamba_config.JambaConfig.jamba_tiny_dev()
    # period=2, offset=1 → 0=mamba, 1=attn, 2=mamba, 3=attn.
    spec0 = cfg.to_block_spec(0)
    spec1 = cfg.to_block_spec(1)
    spec2 = cfg.to_block_spec(2)
    spec3 = cfg.to_block_spec(3)
    assert isinstance(spec0.token_mixer, specs.SSMSpec)
    assert isinstance(spec1.token_mixer, specs.AttentionSpec)
    assert isinstance(spec2.token_mixer, specs.SSMSpec)
    assert isinstance(spec3.token_mixer, specs.AttentionSpec)


def test_jamba_attention_has_no_rope():
    """Jamba attention spec must have rope=None."""
    cfg = jamba_config.JambaConfig.jamba_tiny_dev()
    spec = cfg.to_attention_spec()
    assert spec.rope is None
