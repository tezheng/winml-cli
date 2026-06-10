"""MiniMax-M2 config tests."""
import torch

from api import specs, types
from models.minimax_m2 import config as _c


def _small_cfg(**overrides):
    base = dict(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        rope_theta=5_000_000.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        dtype=torch.float32,
    )
    base.update(overrides)
    return _c.MiniMaxM2Config(**base)


def test_attention_spec_full_hdh_pre_rope_qk_norm():
    cfg = _small_cfg()
    spec = cfg.to_block_spec()
    a = spec.token_mixer
    assert a.kind == types.AttentionKind.STANDARD
    assert a.qk_norm_phase == types.QKNormPhase.PRE_ROPE
    assert a.qk_norm_shape == types.QKNormShape.FULL_HDH
    # All biases False.
    assert a.q_bias is False and a.k_bias is False
    assert a.v_bias is False and a.o_bias is False
    # GQA.
    assert a.n_q_heads == cfg.num_attention_heads
    assert a.n_kv_heads == cfg.num_key_value_heads


def test_moe_spec_sigmoid_no_groups_no_shared():
    cfg = _small_cfg()
    spec = cfg.to_block_spec()
    m = spec.channel_mixer
    assert isinstance(m, specs.MoESpec)
    assert m.router_kind == "sigmoid_plus_bias"
    assert m.group_routing is None              # NO group routing
    assert m.n_shared_experts == 0              # NO shared experts
    assert m.router_norm is True                # always normalised
    assert m.routed_scaling_factor == 1.0


def test_rope_default():
    cfg = _small_cfg()
    spec = cfg.to_block_spec()
    rope = spec.token_mixer.rope
    assert rope.basis == types.RoPEBasis.SPLIT_HALF
    assert rope.scaling == types.RoPEScaling.NONE
    assert rope.partial_rotary_factor == 1.0
    assert rope.base_theta == cfg.rope_theta


def test_pre_norm_envelope():
    cfg = _small_cfg()
    spec = cfg.to_block_spec()
    assert spec.attn_norm_position == types.NormPosition.PRE
    assert spec.ffn_norm_position == types.NormPosition.PRE
    assert spec.pre_attn_norm is not None
    assert spec.pre_ffn_norm is not None


def test_from_hf_dict_minimal():
    hf = dict(
        hidden_size=64, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=100,
        num_local_experts=8, num_experts_per_tok=2,
        torch_dtype="float32",   # legacy key
    )
    cfg = _c.MiniMaxM2Config.from_hf_dict(hf)
    assert cfg.dtype == torch.float32
    assert cfg.num_local_experts == 8
    assert cfg.rope_theta == 5_000_000.0   # default


def test_from_hf_dict_uses_dtype_key_preference():
    hf = dict(
        hidden_size=64, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=100,
        num_local_experts=8, num_experts_per_tok=2,
        dtype="bfloat16",
        torch_dtype="float16",
    )
    cfg = _c.MiniMaxM2Config.from_hf_dict(hf)
    # `dtype` should win over `torch_dtype`.
    assert cfg.dtype == torch.bfloat16
