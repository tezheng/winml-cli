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
        input_norm=norm, pre_attn_norm=norm, pre_ffn_norm=norm,
    )
    assert block.residual_scale is None
