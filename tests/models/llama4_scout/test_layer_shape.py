"""Shape-only tests for Llama 4 Scout per-layer attention.

Exercises the iRoPE per-layer dispatch end-to-end on the attention module:
build a RoPE layer (rope module present, INTERLEAVED basis) AND a NoPE
layer (rope=None), feed both with synthetic weights, and verify the
NoPE-layer output is invariant to position_ids while the RoPE-layer
output IS sensitive to it.

The Scout-default ``layer_types`` would mark NoPE layers as
``"chunked_attention"`` which is NOT in B4 scope; we therefore feed a
``layer_types = ["full_attention"] * 48`` override into the synthetic
config so NoPE layers are testable end-to-end.

For Scout-16E weights are gated AND ~215 GB in fp32 — numerical gate is
out of B4 scope. See models/llama4_scout/layer.md §6 for details.
"""
import pytest
import torch

from api import kvcache, specs, types
from models.llama4_scout import config, layer


def _synthetic_scout_cfg(num_hidden_layers=8):
    """Small synthetic config matching the iRoPE pattern with hidden=128.
    Even at this size we exercise INTERLEAVED RoPE + per-layer dispatch.

    We override ``layer_types`` to all "full_attention" so NoPE layers are
    callable (Scout-default would mark them "chunked_attention").
    """
    return config.Llama4ScoutConfig(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, intermediate_size=256, intermediate_size_mlp=512,
        num_hidden_layers=num_hidden_layers,
        rope_theta=500_000.0, rms_norm_eps=1e-5,
        vocab_size=512, max_position_embeddings=128,
        tie_word_embeddings=False, dtype=torch.float32,
        no_rope_layers=tuple(
            int((i + 1) % 4 != 0) for i in range(num_hidden_layers)
        ),
        no_rope_layer_interval=4,
        layer_types=tuple(["full_attention"] * num_hidden_layers),
        attention_chunk_size=8192,
        use_qk_norm=True,
        attn_temperature_tuning=True,
        floor_scale=8192, attn_scale=0.1,
        moe_layers=tuple(range(num_hidden_layers)),
        interleave_moe_layer_step=1,
        num_local_experts=16, num_experts_per_tok=1,
        rope_scaling_factor=8.0,
        rope_scaling_low_freq_factor=1.0,
        rope_scaling_high_freq_factor=4.0,
        rope_scaling_original_max_pos=8192,
        attention_bias=False,
    )


@pytest.mark.parametrize("layer_idx,expects_rope", [
    (0, True), (1, True), (2, True), (3, False), (4, True), (7, False),
])
def test_llama4_scout_attention_forward_shape(layer_idx, expects_rope):
    cfg = _synthetic_scout_cfg()
    assert cfg.layer_uses_rope(layer_idx) is expects_rope
    attn = layer.build_llama4_scout_attention(cfg, layer_idx=layer_idx, max_seq=64)
    if expects_rope:
        assert attn.rope is not None
    else:
        assert attn.rope is None

    B, S = 2, 6
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    out = attn(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_llama4_scout_nope_layer_invariant_to_position_ids():
    """NoPE layer output is invariant to position_ids — without RoPE there
    is no positional encoding, so feeding [0..S) vs [20..S+20) returns
    bit-identical outputs (same cache state, same input)."""
    cfg = _synthetic_scout_cfg()

    def run(layer_idx, pos):
        attn = layer.build_llama4_scout_attention(cfg, layer_idx=layer_idx, max_seq=64)
        torch.manual_seed(layer_idx)
        for p in attn.parameters():
            p.data.copy_(torch.randn_like(p) * 0.02)
        cache_spec = specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=cfg.dtype, v_dtype=cfg.dtype,
        )
        cache = kvcache.ContiguousKVCache(
            cache_spec, batch_size=1,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, max_seq=64,
        )
        torch.manual_seed(0)
        x = torch.randn(1, 4, cfg.hidden_size)
        return attn(x, position_ids=pos, cache=cache, start_pos=0)

    nope_a = run(3, torch.arange(4))
    nope_b = run(3, torch.arange(20, 24))
    # Equal to bit precision — no rotation applied either time.
    assert torch.equal(nope_a, nope_b)


def test_llama4_scout_rope_layer_sensitive_to_relative_position():
    """RoPE layer attention output is invariant to a UNIFORM positional
    shift (a fundamental property of RoPE: qk^T depends on i-j) but
    SENSITIVE to a non-uniform reordering of positions. We exercise the
    sensitivity case to prove the rotation is actually being applied.

    Compare position_ids = [0, 1, 2, 3] vs [0, 1, 5, 6]: the relative
    positions between tokens 0,1 and 2,3 change, so the rotation
    actually shifts the dot products. NoPE layers ignore this entirely
    (covered by the invariance test above)."""
    cfg = _synthetic_scout_cfg()

    def run(layer_idx, pos):
        attn = layer.build_llama4_scout_attention(cfg, layer_idx=layer_idx, max_seq=64)
        torch.manual_seed(layer_idx)
        for p in attn.parameters():
            p.data.copy_(torch.randn_like(p) * 0.02)
        cache_spec = specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=cfg.dtype, v_dtype=cfg.dtype,
        )
        cache = kvcache.ContiguousKVCache(
            cache_spec, batch_size=1,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, max_seq=64,
        )
        torch.manual_seed(0)
        x = torch.randn(1, 4, cfg.hidden_size)
        return attn(x, position_ids=pos, cache=cache, start_pos=0)

    # Uniform shift: outputs should be CLOSE for RoPE layer (relative
    # positions identical).
    rope_a = run(0, torch.arange(4))
    rope_b = run(0, torch.arange(20, 24))
    # Non-uniform reorder: relative positions change, outputs MUST differ.
    rope_c = run(0, torch.tensor([0, 1, 5, 6]))
    # The uniform-shift case is approximately invariant for RoPE (the
    # qk^T only sees relative position differences i-j). The non-uniform
    # case definitely differs.
    assert not torch.allclose(rope_a, rope_c, atol=1e-4), (
        "RoPE layer's output must change when relative positions change."
    )
    # Sanity: same-relative-pos uniform shift IS close.
    assert torch.allclose(rope_a, rope_b, atol=1e-4)


def test_llama4_scout_attention_uses_interleaved_basis():
    """The Scout attention's RoPE module should use INTERLEAVED basis
    (cos/sin tables are sized head_dim/2, not head_dim).
    """
    cfg = _synthetic_scout_cfg()
    attn = layer.build_llama4_scout_attention(cfg, layer_idx=0, max_seq=32)
    assert attn.rope is not None
    # INTERLEAVED basis: cos/sin tables span head_dim/2.
    assert attn.rope.cos_cached.shape == (32, cfg.head_dim // 2)
    assert attn.rope.sin_cached.shape == (32, cfg.head_dim // 2)


def test_llama4_scout_load_hf_weights_into_attention():
    """Synthetic state-dict load test for the per-layer attention. RoPE
    layer (0) and NoPE layer (3) both share the same q/k/v/o tensor names;
    the dispatch lives only in the spec."""
    cfg = _synthetic_scout_cfg()
    rope_attn = layer.build_llama4_scout_attention(cfg, layer_idx=0, max_seq=32)
    nope_attn = layer.build_llama4_scout_attention(cfg, layer_idx=3, max_seq=32)

    # Build a fake state dict with the right names + shapes for layers 0 and 3.
    fake_sd = {}
    for L in (0, 3):
        fake_sd[f"model.layers.{L}.self_attn.q_proj.weight"] = torch.randn(
            cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size,
        )
        fake_sd[f"model.layers.{L}.self_attn.k_proj.weight"] = torch.randn(
            cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size,
        )
        fake_sd[f"model.layers.{L}.self_attn.v_proj.weight"] = torch.randn(
            cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size,
        )
        fake_sd[f"model.layers.{L}.self_attn.o_proj.weight"] = torch.randn(
            cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim,
        )

    layer.load_hf_llama4_scout_attention(rope_attn, fake_sd, layer_idx=0, cfg=cfg)
    layer.load_hf_llama4_scout_attention(nope_attn, fake_sd, layer_idx=3, cfg=cfg)
    # Sanity: q_proj weights should equal what we loaded.
    assert torch.equal(
        rope_attn.q_proj.weight.data,
        fake_sd["model.layers.0.self_attn.q_proj.weight"].to(rope_attn.q_proj.weight.dtype),
    )
    assert torch.equal(
        nope_attn.q_proj.weight.data,
        fake_sd["model.layers.3.self_attn.q_proj.weight"].to(nope_attn.q_proj.weight.dtype),
    )


def test_llama4_scout_numerical_gate_skipped():
    """No numerical gate vs HF — Scout weights are gated AND ~215 GB.
    See models/llama4_scout/layer.md §6. This test is a marker to make the
    skip explicit and discoverable by grep."""
    pytest.skip(
        "Llama 4 Scout numerical gate deferred: weights are gated at "
        "meta-llama/Llama-4-Scout-17B-16E and ~215 GB on the open "
        "unsloth/Llama-4-Scout-17B-16E mirror. Plus MoE wiring lands "
        "in B6. Shape-only test exercises iRoPE per-layer dispatch."
    )
