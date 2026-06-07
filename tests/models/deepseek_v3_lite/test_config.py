"""V3-Lite synthetic config tests — verify architectural deltas vs V2."""
import torch

from api import specs, types
from models.deepseek_v3_lite import config as m_config


def _small_cfg() -> m_config.DeepSeekV3LiteConfig:
    return m_config.DeepSeekV3LiteConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=24,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=4,
        routed_scaling_factor=2.5,    # V3 default — NOT 1.0
        norm_topk_prob=True,          # V3: renorm top-k weights
        n_group=4,                    # V3: always group-routes
        topk_group=2,
        first_k_dense_replace=1,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=24,               # V3 uses q-LoRA
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        dtype=torch.float32,
    )


def test_v3_lite_layer1_moe_router_is_sigmoid_plus_bias():
    cfg = _small_cfg()
    blk = cfg.to_block_spec(layer_idx=1)
    assert isinstance(blk.channel_mixer, specs.MoESpec)
    moe = blk.channel_mixer
    assert moe.router_kind == "sigmoid_plus_bias"
    assert moe.score_correction_bias is True
    assert moe.router_norm is True
    assert moe.routed_scaling_factor == 2.5
    # V3 ALWAYS has group routing.
    assert moe.group_routing is not None
    assert moe.group_routing.n_groups == 4
    assert moe.group_routing.topk_per_group == 2


def test_v3_lite_rope_is_split_half_not_interleaved():
    """V3 RoPE uses rotate_half pattern (apply_rotary_pos_emb), NOT V2's
    complex-multiply pairing. Source: modeling_deepseek_v3.py:250-280."""
    cfg = _small_cfg()
    blk = cfg.to_block_spec(layer_idx=0)
    attn = blk.token_mixer
    assert attn.rope.basis == types.RoPEBasis.SPLIT_HALF


def test_v3_lite_attention_uses_q_lora():
    """V3 has q_lora_rank set (q-LoRA path)."""
    cfg = _small_cfg()
    blk = cfg.to_block_spec(layer_idx=1)
    attn = blk.token_mixer
    assert attn.q_lora_rank == 24
