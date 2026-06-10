"""GLM-MoE-DSA layer-shape tests.

The MLA + DSA attention is shape-only (Phase A reservation); the MoE / dense
FFN sublayer is fully functional.
"""
import pytest
import torch

from api import feedforward, specs, types
from models.glm_moe_dsa import config as _c, layer as _l


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
        routed_scaling_factor=1.0,
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


def test_block_constructs_for_dense_layer():
    cfg = _small_cfg()
    blk = _l.build_glm_moe_dsa_decoder_layer(cfg, layer_idx=0)
    assert isinstance(blk.feedforward, feedforward.FeedForward)


def test_block_constructs_for_sparse_layer():
    cfg = _small_cfg()
    blk = _l.build_glm_moe_dsa_decoder_layer(cfg, layer_idx=3)
    assert isinstance(blk.feedforward, feedforward.MoE)
    assert blk.feedforward.spec.router_kind == "sigmoid_plus_bias"


def test_dsa_attention_forward_raises():
    """MLA + DSA attention forward is shape-only (Phase A reservation)."""
    cfg = _small_cfg()
    blk = _l.build_glm_moe_dsa_decoder_layer(cfg, layer_idx=0)
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4)
    with pytest.raises(NotImplementedError):
        blk.attention(x, position_ids=pos, cache=None, start_pos=0)


def test_dense_ffn_sublayer_forward():
    """The dense FFN forward runs on a dense-layer block."""
    cfg = _small_cfg()
    blk = _l.build_glm_moe_dsa_decoder_layer(cfg, layer_idx=0)
    blk.eval()
    x = torch.randn(1, 4, cfg.hidden_size)
    with torch.no_grad():
        y = blk.feedforward(x)
    assert y.shape == x.shape


def test_moe_sublayer_forward():
    """The MoE forward runs on a sparse-layer block."""
    cfg = _small_cfg()
    blk = _l.build_glm_moe_dsa_decoder_layer(cfg, layer_idx=3)
    blk.eval()
    x = torch.randn(1, 4, cfg.hidden_size)
    with torch.no_grad():
        y = blk.feedforward(x)
    assert y.shape == x.shape
