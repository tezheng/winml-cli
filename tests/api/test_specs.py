import dataclasses
import pytest
import torch

from api import specs, types


def test_norm_spec_is_frozen():
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.eps = 1e-5  # type: ignore


def test_rope_spec_defaults():
    spec = specs.RoPESpec(base_theta=1_000_000.0, basis=types.RoPEBasis.SPLIT_HALF)
    assert spec.scaling == types.RoPEScaling.NONE
    assert spec.scale_factor is None


def test_attention_spec_minimum():
    spec = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    assert spec.qk_norm_phase == types.QKNormPhase.NONE


def test_ffn_spec_swiglu():
    spec = specs.FFNSpec(intermediate_size=3072,
                         activation=types.Activation.SILU,
                         gate_kind=types.GateKind.SWIGLU)
    assert not spec.fused_gate_up


def test_kvcache_spec_contiguous():
    spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.NHD,
        k_dtype=torch.bfloat16, v_dtype=torch.bfloat16,
        ownership=types.CacheOwnership.EXPLICIT_PASS,
    )
    assert spec.block_size is None  # not used for CONTIGUOUS


def test_quant_spec_awq():
    spec = specs.QuantSpec(
        qdtype=types.QDType.INT4, group_size=128, quant_axis=0,
        scale_dtype=torch.float16, has_zero_point=True,
        packing=types.PackingLayout.AWQ_INTERLEAVE,
        accumulator_dtype=torch.float32,
        role=types.QuantRole.WEIGHT,
    )
    assert spec.codebook is None


def test_new_specs_frozen_and_default_consistent():
    """The 6 newly-added specs are reserved for M2 batches but must instantiate cleanly today."""
    conv = specs.ConvSpec(kernel_size=4)
    assert conv.bias is True
    ssm = specs.SSMSpec(d_state=16, d_conv=4, d_inner=64)
    assert ssm.dt_rank == -1
    ssd = specs.SSDSpec(base=ssm)
    assert ssd.chunk_size == 256
    group = specs.GroupRoutingSpec(n_groups=8, topk_per_group=2)
    moe = specs.MoESpec(n_experts=8, top_k=2, group_routing=group)
    assert moe.router_kind == "softmax"
    ls = specs.LayerScaleSpec(
        num_q_heads_per_layer=(16,) * 28,
        num_kv_heads_per_layer=(8,) * 28,
        ffn_multipliers_per_layer=(1.0,) * 28,
    )
    assert len(ls.num_q_heads_per_layer) == 28


def test_new_specs_are_frozen():
    import dataclasses as _dc
    conv = specs.ConvSpec(kernel_size=4)
    with pytest.raises(_dc.FrozenInstanceError):
        conv.kernel_size = 6  # type: ignore


def test_decoder_block_spec_composition():
    norm = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    attn = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=3072,
                       activation=types.Activation.SILU,
                       gate_kind=types.GateKind.SWIGLU)
    block = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm, pre_ffn_norm=norm,
    )
    assert block.residual_scale is None


def test_share_scheme_enum_values_exist():
    assert types.ShareScheme.NONE
    assert types.ShareScheme.SAME_BLOCK_SHARED       # Gemma 4 E2B / E4B
    assert types.ShareScheme.CROSS_BLOCK_SHARED      # Apple AFM (deferred but enumerated)


def test_qk_norm_phase_has_pre_rope_fixed_scale():
    # Gemma 4 uses PRE_ROPE QK-norm with a fixed-scale (non-learned-absorbing-1/sqrt(Dh)).
    # We model this as the existing PRE_ROPE phase plus a fixed_scale field on AttentionSpec.
    assert types.QKNormPhase.PRE_ROPE


def test_mask_kind_has_swa_global_alt():
    # Gemma 3 / 4 alternate SWA local with full-attention global layers at 5:1.
    assert types.MaskKind.SWA_GLOBAL_ALT


def test_attention_spec_has_attention_k_eq_v():
    spec = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        attention_k_eq_v=True,
    )
    assert spec.attention_k_eq_v is True


def test_attention_spec_has_qk_norm_fixed_scale():
    spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        qk_norm_fixed_scale=0.9916,  # Gemma 4 local
    )
    assert spec.qk_norm_fixed_scale == 0.9916


def test_attention_spec_has_final_logit_softcap_marker():
    # final_logit_softcap belongs on the *model* level (lm_head op), but the block-level
    # spec carries it for the assembly path. We put it on DecoderBlockSpec.
    pass  # tested via DecoderBlockSpec below


def test_rope_spec_has_partial_rotary_factor():
    spec = specs.RoPESpec(
        base_theta=1_000_000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        partial_rotary_factor=0.25,  # Gemma 4 global
    )
    assert spec.partial_rotary_factor == 0.25


def test_rope_spec_partial_rotary_default_is_one():
    spec = specs.RoPESpec(base_theta=1_000_000.0, basis=types.RoPEBasis.SPLIT_HALF)
    assert spec.partial_rotary_factor == 1.0  # full rotation by default


def test_kvcache_spec_has_share_scheme():
    spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.bfloat16, v_dtype=torch.bfloat16,
        share_scheme=types.ShareScheme.SAME_BLOCK_SHARED,
        num_kv_shared_layers=20,
    )
    assert spec.share_scheme == types.ShareScheme.SAME_BLOCK_SHARED
    assert spec.num_kv_shared_layers == 20


def test_ple_spec_exists():
    spec = specs.PLESpec(
        ple_dim=256,
        residual_scale=1.0 / (2 ** 0.5),
        injection_norm=specs.NormSpec(
            kind=types.NormKind.RMS, eps=1e-6,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        ),
    )
    assert spec.ple_dim == 256


def test_decoder_block_spec_has_pre_ple_injection():
    norm = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.ONE_PLUS_W)
    attn = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
    )
    ffn = specs.FFNSpec(intermediate_size=8192,
                       activation=types.Activation.GELU,
                       gate_kind=types.GateKind.GEGLU)
    ple = specs.PLESpec(ple_dim=256, residual_scale=0.7071,
                       injection_norm=norm)
    block = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE_AND_POST,
        ffn_norm_position=types.NormPosition.PRE_AND_POST,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm, post_attn_norm=norm,
        pre_ffn_norm=norm, post_ffn_norm=norm,
        per_layer_embedding=ple,
        final_logit_softcap=30.0,
    )
    assert block.per_layer_embedding is ple
    assert block.final_logit_softcap == 30.0
