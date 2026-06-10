"""GLM-MoE-DSA config tests."""
import torch

from api import specs, types
from models.glm_moe_dsa import config as _c


def _small_cfg(**overrides):
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attention_bias=False,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        rope_theta=10000.0,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        index_topk=16,
        index_head_dim=8,
        index_n_heads=2,
        mlp_layer_types=("dense", "dense", "dense",
                         "sparse", "sparse", "sparse"),
        indexer_types=("full",) * 6,
        dtype=torch.float32,
    )
    base.update(overrides)
    return _c.GlmMoeDsaConfig(**base)


def test_dense_vs_sparse_dispatch():
    cfg = _small_cfg()
    assert not cfg.is_sparse_layer(0)
    assert not cfg.is_sparse_layer(2)
    assert cfg.is_sparse_layer(3)
    assert cfg.is_sparse_layer(5)


def test_dense_layer_to_block_spec_uses_ffn():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=0)
    assert isinstance(spec.channel_mixer, specs.FFNSpec)
    assert spec.channel_mixer.intermediate_size == cfg.intermediate_size


def test_sparse_layer_to_block_spec_uses_moe():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=3)
    assert isinstance(spec.channel_mixer, specs.MoESpec)
    assert spec.channel_mixer.router_kind == "sigmoid_plus_bias"
    assert spec.channel_mixer.router_norm is True
    assert spec.channel_mixer.group_routing is not None
    assert spec.channel_mixer.group_routing.n_groups == cfg.n_group
    assert spec.channel_mixer.routed_scaling_factor == cfg.routed_scaling_factor


def test_attention_spec_is_dsa_with_indexer():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=0)
    a = spec.token_mixer
    assert a.kind == types.AttentionKind.DSA
    assert a.qkv_layout == types.QKVLayout.MLA_LATENT
    assert a.indexer is not None
    assert a.indexer.top_k == cfg.index_topk
    assert a.indexer.indexer_dim == cfg.index_head_dim
    assert a.head_dim == cfg.qk_head_dim


def test_qk_head_dim_property():
    cfg = _small_cfg()
    assert cfg.qk_head_dim == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim


def test_from_hf_dict_default_mlp_layer_types():
    hf = dict(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=4,
        rms_norm_eps=1e-6, vocab_size=100,
        kv_lora_rank=16, q_lora_rank=32,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        index_topk=16, index_head_dim=8, index_n_heads=2,
        dtype="float32",
    )
    cfg = _c.GlmMoeDsaConfig.from_hf_dict(hf)
    # Default: first 3 dense, rest sparse.
    assert cfg.mlp_layer_types == ("dense",) * 3 + ("sparse",) * 2


def test_from_hf_dict_uses_dtype_key_not_torch_dtype():
    """B7 convention: prefer `dtype` over legacy `torch_dtype`."""
    hf = dict(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        rms_norm_eps=1e-6, vocab_size=50,
        kv_lora_rank=16, q_lora_rank=32,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16,
        n_routed_experts=4, num_experts_per_tok=1,
        dtype="bfloat16", torch_dtype="float16",  # `dtype` should win
    )
    cfg = _c.GlmMoeDsaConfig.from_hf_dict(hf)
    assert cfg.dtype == torch.bfloat16
