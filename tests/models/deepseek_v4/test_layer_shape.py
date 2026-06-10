"""DeepSeek-V4 layer-shape tests (no HF instantiation).

V4 attention is shape-only (Phase A P2 raises NotImplementedError on forward),
so we test:
- Block construction succeeds for all layer types (hash MoE + CSA / HCA /
  sliding attention).
- The hash-MoE channel mixer forward returns the right shape and threads
  input_ids correctly.
"""
import pytest
import torch

from api import feedforward, specs, types
from models.deepseek_v4 import config as _c, layer as _l


def _small_cfg(**overrides):
    base = dict(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, q_lora_rank=64, qk_rope_head_dim=8,
        moe_intermediate_size=64,
        num_hidden_layers=6,
        n_routed_experts=8, n_shared_experts=1,
        num_experts_per_tok=2, routed_scaling_factor=1.0,
        norm_topk_prob=True, scoring_func="sigmoid",
        swiglu_limit=1e6, sliding_window=32,
        rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=128,
        tie_word_embeddings=False, attention_bias=False,
        rope_theta=10000.0, compress_rope_theta=160000.0,
        layer_types=("heavily_compressed_attention",) * 2 +
                    ("compressed_sparse_attention", "heavily_compressed_attention",
                     "sliding_attention", "compressed_sparse_attention"),
        mlp_layer_types=("hash_moe",) * 3 + ("moe",) * 3,
        compress_rate_csa=4, compress_rate_hca=8,
        index_n_heads=2, index_head_dim=8, index_topk=4,
        dtype=torch.float32,
    )
    base.update(overrides)
    return _c.DeepSeekV4Config(**base)


def test_block_constructs_for_hash_hca_layer():
    cfg = _small_cfg()
    blk = _l.build_deepseek_v4_decoder_layer(cfg, layer_idx=0)
    assert isinstance(blk.feedforward, feedforward.MoE)
    assert blk.feedforward.spec.router_kind == "hash"
    # Attention is shape-only; forward should raise.
    assert blk.attention is not None


def test_block_constructs_for_hash_csa_layer():
    cfg = _small_cfg()
    blk = _l.build_deepseek_v4_decoder_layer(cfg, layer_idx=2)
    assert blk.feedforward.spec.router_kind == "hash"
    # CSA carrier
    assert blk.spec.token_mixer.csa is not None


def test_block_constructs_for_topk_sliding_layer():
    cfg = _small_cfg()
    blk = _l.build_deepseek_v4_decoder_layer(cfg, layer_idx=4)
    assert blk.feedforward.spec.router_kind == "sigmoid_plus_bias"
    assert blk.spec.token_mixer.kind == types.AttentionKind.STANDARD


def test_hash_moe_sublayer_forward_runs():
    """The MoE sublayer forward (hash routing) returns the right shape and
    requires input_ids."""
    cfg = _small_cfg()
    blk = _l.build_deepseek_v4_decoder_layer(cfg, layer_idx=0)
    blk.eval()
    B, S = 2, 5
    moe = blk.feedforward
    x = torch.randn(B, S, cfg.hidden_size)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        y = moe(x, input_ids=input_ids)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_csa_attention_forward_raises():
    """Phase A P2: CSA_HCA attention forward is shape-only."""
    cfg = _small_cfg()
    blk = _l.build_deepseek_v4_decoder_layer(cfg, layer_idx=2)
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4)
    with pytest.raises(NotImplementedError):
        blk.attention(x, position_ids=pos, cache=None, start_pos=0)
