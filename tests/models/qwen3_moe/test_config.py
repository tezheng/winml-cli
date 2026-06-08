"""Qwen3-MoE config tests — adapter and per-layer dispatch correctness."""
import torch

from api import specs, types
from models.qwen3_moe import config as m_config


def _fake_hf_dict_30b_a3b():
    """Subset of Qwen/Qwen3-30B-A3B config.json — only the keys our adapter reads."""
    return {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 48,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "torch_dtype": "bfloat16",
        "use_sliding_window": False,
        "sliding_window": 4096,        # gated by use_sliding_window=False → None.
    }


def test_qwen3_moe_config_from_hf_dict():
    cfg = m_config.Qwen3MoeConfig.from_hf_dict(_fake_hf_dict_30b_a3b())
    assert cfg.hidden_size == 2048
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 6144
    assert cfg.moe_intermediate_size == 768
    assert cfg.num_experts == 128
    assert cfg.num_experts_per_tok == 8
    assert cfg.norm_topk_prob is True
    assert cfg.decoder_sparse_step == 1
    assert cfg.mlp_only_layers == ()
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.dtype == torch.bfloat16
    assert cfg.use_sliding_window is False
    # sliding_window must be forced to None when use_sliding_window=False.
    assert cfg.sliding_window is None


def test_qwen3_moe_sliding_window_enabled():
    d = _fake_hf_dict_30b_a3b()
    d["use_sliding_window"] = True
    d["sliding_window"] = 2048
    cfg = m_config.Qwen3MoeConfig.from_hf_dict(d)
    assert cfg.use_sliding_window is True
    assert cfg.sliding_window == 2048
    blk = cfg.to_block_spec(layer_idx=0)
    assert blk.token_mixer.mask_kind == types.MaskKind.SWA
    assert blk.token_mixer.sliding_window == 2048


def test_qwen3_moe_default_layer_is_moe():
    """B6: default decoder_sparse_step=1, mlp_only_layers=[] →
    every layer is MoE. Source: modeling_qwen3_moe.py:314-319."""
    cfg = m_config.Qwen3MoeConfig.from_hf_dict(_fake_hf_dict_30b_a3b())
    for L in (0, 1, 5, 47):
        blk = cfg.to_block_spec(layer_idx=L)
        assert isinstance(blk.channel_mixer, specs.MoESpec), f"layer {L}"
        assert blk.channel_mixer.n_experts == 128
        assert blk.channel_mixer.top_k == 8
        # Each expert uses moe_intermediate_size, NOT intermediate_size.
        assert blk.channel_mixer.expert_ffn.intermediate_size == 768


def test_qwen3_moe_decoder_sparse_step_2_alternates():
    """When decoder_sparse_step=2, only EVEN layer_idx (where (idx+1)%2==1 is
    False) is MoE — actually (idx+1)%2==0 → odd layer_idx (1, 3, ...) is MoE.
    Wait: per HF, MoE when ``(layer_idx + 1) % decoder_sparse_step == 0`` →
    with step=2, layer_idx in {1, 3, 5, ...} (1-indexed-from-1 odd).
    """
    d = _fake_hf_dict_30b_a3b()
    d["decoder_sparse_step"] = 2
    cfg = m_config.Qwen3MoeConfig.from_hf_dict(d)
    # layer_idx=0 → (0+1)%2 = 1 ≠ 0 → DENSE
    blk0 = cfg.to_block_spec(layer_idx=0)
    assert isinstance(blk0.channel_mixer, specs.FFNSpec)
    assert blk0.channel_mixer.intermediate_size == 6144   # dense uses intermediate_size
    # layer_idx=1 → (1+1)%2 = 0 → MOE
    blk1 = cfg.to_block_spec(layer_idx=1)
    assert isinstance(blk1.channel_mixer, specs.MoESpec)


def test_qwen3_moe_mlp_only_layers_override():
    """`mlp_only_layers` forces specific layers to be dense even when MoE
    schedule would otherwise pick them."""
    d = _fake_hf_dict_30b_a3b()
    d["mlp_only_layers"] = [0, 5]
    cfg = m_config.Qwen3MoeConfig.from_hf_dict(d)
    assert cfg.mlp_only_layers == (0, 5)
    for L in (0, 5):
        blk = cfg.to_block_spec(layer_idx=L)
        assert isinstance(blk.channel_mixer, specs.FFNSpec)
    for L in (1, 2, 6):
        blk = cfg.to_block_spec(layer_idx=L)
        assert isinstance(blk.channel_mixer, specs.MoESpec)


def test_qwen3_moe_attention_spec_qk_norm_per_head_dh_pre_rope():
    """B6: Qwen3-MoE uses Qwen3-style QK-norm PRE_ROPE, PER_HEAD_DH shape.
    Source: modeling_qwen3_moe.py:152-153 + 167-172."""
    cfg = m_config.Qwen3MoeConfig.from_hf_dict(_fake_hf_dict_30b_a3b())
    blk = cfg.to_block_spec()
    attn = blk.token_mixer
    assert attn.qk_norm is not None
    assert attn.qk_norm_phase == types.QKNormPhase.PRE_ROPE
    assert attn.qk_norm_shape == types.QKNormShape.PER_HEAD_DH


def test_qwen3_moe_moe_spec_no_shared_no_group_no_scaling():
    cfg = m_config.Qwen3MoeConfig.from_hf_dict(_fake_hf_dict_30b_a3b())
    blk = cfg.to_block_spec()
    moe = blk.channel_mixer
    assert isinstance(moe, specs.MoESpec)
    assert moe.router_kind == "softmax"
    assert moe.n_shared_experts == 0
    assert moe.group_routing is None
    assert moe.routed_scaling_factor == 1.0
    # norm_topk_prob = True for Qwen3-30B-A3B → router_norm True.
    assert moe.router_norm is True


def test_qwen3_moe_norm_topk_prob_default_false():
    d = _fake_hf_dict_30b_a3b()
    d.pop("norm_topk_prob")
    cfg = m_config.Qwen3MoeConfig.from_hf_dict(d)
    # configuration_qwen3_moe.py:106 — default False.
    assert cfg.norm_topk_prob is False
    blk = cfg.to_block_spec()
    assert blk.channel_mixer.router_norm is False
