"""Qwen3-Next config + to_block_spec tests."""
import torch

from api import specs, types
from models.qwen3_next import config as _c


def _small_cfg(**overrides):
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attention_bias=False,
        rope_theta=10000.0,
        partial_rotary_factor=0.25,
        dtype=torch.float32,
        # 8 layers, every 4th = full_attention -> idx 3, 7
        layer_types=("linear_attention",) * 3 + ("full_attention",) +
                    ("linear_attention",) * 3 + ("full_attention",),
        mlp_only_layers=(),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
    )
    base.update(overrides)
    return _c.Qwen3NextConfig(**base)


def test_per_layer_dispatch():
    cfg = _small_cfg()
    assert cfg.is_linear_attention_layer(0)
    assert cfg.is_linear_attention_layer(2)
    assert not cfg.is_linear_attention_layer(3)   # full
    assert cfg.is_linear_attention_layer(4)
    assert not cfg.is_linear_attention_layer(7)   # full


def test_linear_layer_to_block_spec_gated_deltanet():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=0)
    assert isinstance(spec.token_mixer, specs.GatedDeltaNetSpec)
    assert spec.token_mixer.num_v_heads == cfg.linear_num_value_heads
    assert spec.token_mixer.num_k_heads == cfg.linear_num_key_heads
    assert spec.token_mixer.head_v_dim == cfg.linear_value_head_dim


def test_full_layer_to_block_spec_standard_attention():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=3)
    assert isinstance(spec.token_mixer, specs.AttentionSpec)
    assert spec.token_mixer.kind == types.AttentionKind.STANDARD
    assert spec.token_mixer.qk_norm_phase == types.QKNormPhase.PRE_ROPE
    assert spec.token_mixer.qk_norm_shape == types.QKNormShape.PER_HEAD_DH
    # Qwen3-Next RoPE: partial 0.25.
    assert abs(spec.token_mixer.rope.partial_rotary_factor - 0.25) < 1e-9


def test_one_plus_w_norms():
    """Qwen3-Next RMSNorm uses (1 + w) form (ONE_PLUS_W)."""
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=0)
    assert spec.pre_attn_norm.weight_mode == types.NormWeightMode.ONE_PLUS_W
    assert spec.pre_ffn_norm.weight_mode == types.NormWeightMode.ONE_PLUS_W


def test_moe_channel_mixer_default():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=0)
    assert isinstance(spec.channel_mixer, specs.MoESpec)
    assert spec.channel_mixer.router_kind == "softmax"
    assert spec.channel_mixer.router_norm is True
    assert spec.channel_mixer.n_experts == cfg.num_experts


def test_mlp_only_layer_uses_dense_ffn():
    cfg = _small_cfg(mlp_only_layers=(2,))
    spec = cfg.to_block_spec(layer_idx=2)
    assert isinstance(spec.channel_mixer, specs.FFNSpec)
    assert spec.channel_mixer.intermediate_size == cfg.intermediate_size


def test_from_hf_dict_default_layer_types_three_to_one():
    hf = dict(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, rms_norm_eps=1e-6, vocab_size=100,
        num_experts=8, num_experts_per_tok=2,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        dtype="float32",
    )
    cfg = _c.Qwen3NextConfig.from_hf_dict(hf)
    # Default: every 4th layer (1-indexed) is full -> layer 3.
    assert cfg.layer_types == (
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    )


def test_from_hf_dict_uses_dtype_key():
    hf = dict(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, rms_norm_eps=1e-6, vocab_size=100,
        num_experts=8, num_experts_per_tok=2,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        dtype="bfloat16",
    )
    cfg = _c.Qwen3NextConfig.from_hf_dict(hf)
    assert cfg.dtype == torch.bfloat16
