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
