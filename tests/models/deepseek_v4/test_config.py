"""DeepSeek-V4 config + to_block_spec tests."""
import torch

from api import specs, types
from models.deepseek_v4 import config as _c


def _small_cfg(**overrides) -> _c.DeepSeekV4Config:
    base = dict(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, q_lora_rank=64, qk_rope_head_dim=8,
        moe_intermediate_size=64,
        num_hidden_layers=6,
        n_routed_experts=16, n_shared_experts=1,
        num_experts_per_tok=2, routed_scaling_factor=1.5,
        norm_topk_prob=True, scoring_func="sigmoid",
        swiglu_limit=1e6, sliding_window=64,
        rms_norm_eps=1e-6, vocab_size=100, max_position_embeddings=128,
        tie_word_embeddings=False, attention_bias=False,
        rope_theta=10000.0, compress_rope_theta=160000.0,
        layer_types=("heavily_compressed_attention", "heavily_compressed_attention",
                     "compressed_sparse_attention", "heavily_compressed_attention",
                     "sliding_attention", "compressed_sparse_attention"),
        mlp_layer_types=("hash_moe", "hash_moe", "hash_moe",
                         "moe", "moe", "moe"),
        compress_rate_csa=4, compress_rate_hca=8,
        index_n_heads=2, index_head_dim=8, index_topk=4,
        dtype=torch.float32,
    )
    base.update(overrides)
    return _c.DeepSeekV4Config(**base)


def test_hash_mlp_dispatch_first_three_layers():
    cfg = _small_cfg()
    assert cfg.is_hash_layer(0) is True
    assert cfg.is_hash_layer(1) is True
    assert cfg.is_hash_layer(2) is True
    assert cfg.is_hash_layer(3) is False
    assert cfg.is_hash_layer(4) is False
    assert cfg.is_hash_layer(5) is False


def test_to_block_spec_hash_layer_router_kind():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=0)
    assert isinstance(spec.channel_mixer, specs.MoESpec)
    assert spec.channel_mixer.router_kind == "hash"
    assert spec.channel_mixer.hash_vocab_size == cfg.vocab_size
    assert spec.channel_mixer.top_k == cfg.num_experts_per_tok


def test_to_block_spec_topk_layer_router_kind():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=3)
    assert isinstance(spec.channel_mixer, specs.MoESpec)
    assert spec.channel_mixer.router_kind == "sigmoid_plus_bias"
    assert spec.channel_mixer.router_norm is True


def test_to_block_spec_attention_kinds():
    cfg = _small_cfg()
    # layer 0: HCA
    s0 = cfg.to_block_spec(layer_idx=0)
    assert s0.token_mixer.kind == types.AttentionKind.CSA_HCA
    assert s0.token_mixer.hca is not None
    assert s0.token_mixer.hca.compress_rate == 8
    # layer 2: CSA
    s2 = cfg.to_block_spec(layer_idx=2)
    assert s2.token_mixer.kind == types.AttentionKind.CSA_HCA
    assert s2.token_mixer.csa is not None
    assert s2.token_mixer.csa.compress_rate == 4
    assert s2.token_mixer.csa.indexer_topk == 4
    # layer 4: sliding
    s4 = cfg.to_block_spec(layer_idx=4)
    assert s4.token_mixer.kind == types.AttentionKind.STANDARD
    assert s4.token_mixer.mask_kind == types.MaskKind.SWA
    assert s4.token_mixer.sliding_window == cfg.sliding_window


def test_rope_partial_rotary_factor():
    cfg = _small_cfg()
    spec = cfg.to_block_spec(layer_idx=4)
    rope = spec.token_mixer.rope
    assert rope is not None
    # V4 sources use INTERLEAVED RoPE on partial slice; we model SPLIT_HALF
    # for IR composability (see config.py comment).
    assert rope.basis == types.RoPEBasis.SPLIT_HALF
    assert abs(rope.partial_rotary_factor - cfg.qk_rope_head_dim / cfg.head_dim) < 1e-9


def test_from_hf_dict_minimal():
    hf = dict(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, q_lora_rank=64,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        n_routed_experts=8, n_shared_experts=1,
        num_experts_per_tok=2, routed_scaling_factor=1.5,
        scoring_func="sigmoid",
        rms_norm_eps=1e-6, vocab_size=100,
        index_n_heads=2, index_head_dim=8, index_topk=4,
        swiglu_limit=1e6,
        torch_dtype="float32",   # legacy HF key
    )
    cfg = _c.DeepSeekV4Config.from_hf_dict(hf)
    assert cfg.dtype == torch.float32
    # Default: first 3 hash, rest moe
    assert cfg.mlp_layer_types[:3] == ("hash_moe",) * 3
    assert cfg.mlp_layer_types[3] == "moe"
    # Default attention: 2× HCA + alternating
    assert cfg.layer_types[0] == "heavily_compressed_attention"
    assert cfg.layer_types[1] == "heavily_compressed_attention"


def test_from_hf_dict_uses_dtype_key():
    """B7 convention: prefer `dtype` over legacy `torch_dtype`."""
    hf = dict(
        hidden_size=64, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, q_lora_rank=32, moe_intermediate_size=32,
        num_hidden_layers=2, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=1, scoring_func="sigmoid",
        vocab_size=50, swiglu_limit=1e6,
        dtype="bfloat16",
    )
    cfg = _c.DeepSeekV4Config.from_hf_dict(hf)
    assert cfg.dtype == torch.bfloat16
