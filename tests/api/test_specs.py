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
    assert spec.layout == types.CacheLayout.CONTIGUOUS


def test_quant_spec_awq():
    spec = specs.QuantSpec(
        qdtype=types.QDType.INT4, group_size=128, quant_axis=0,
        scale_dtype=torch.float16, has_zero_point=True,
        packing=types.PackingLayout.AWQ_INTERLEAVE,
        accumulator_dtype=torch.float32,
        role=types.QuantRole.WEIGHT,
    )
    assert spec.role == types.QuantRole.WEIGHT


def test_new_specs_frozen_and_default_consistent():
    """The 6 newly-added specs are reserved for M2 batches but must instantiate cleanly today."""
    conv = specs.ConvSpec(kernel_size=4)
    assert conv.bias is True
    ssm = specs.SSMSpec(d_state=16, d_conv=4, d_inner=64)
    assert ssm.d_state == 16
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


def test_token_mixer_kind_enum_values_exist():
    """B7: TokenMixerKind enum carries ATTENTION + the SSM variants."""
    assert types.TokenMixerKind.ATTENTION
    assert types.TokenMixerKind.SSM_MAMBA1
    assert types.TokenMixerKind.SSM_MAMBA2
    assert types.TokenMixerKind.SSM_GRIFFIN
    assert types.TokenMixerKind.SSM_RWKV


def test_ssd_spec_b7_extra_fields_default():
    """B7: SSDSpec carries n_heads, time_step_limit, layer_norm_epsilon.
    All Mamba-2 specific."""
    ssm = specs.SSMSpec(d_state=128, d_conv=4, d_inner=5120)
    assert ssm.activation == types.Activation.SILU
    ssd = specs.SSDSpec(
        base=ssm, headdim=64, ngroups=8, n_heads=80,
        chunk_size=256,
    )
    assert ssd.n_heads == 80
    assert ssd.time_step_limit_low == 0.0
    assert ssd.time_step_limit_high == float("inf")
    assert ssd.layer_norm_epsilon == 1e-5


def test_decoder_block_spec_with_ssd_token_mixer_skips_ffn():
    """B7: token_mixer can be SSDSpec; skip_ffn flag allows no FFN."""
    norm = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-5,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    ssm = specs.SSMSpec(d_state=128, d_conv=4, d_inner=128)
    ssd = specs.SSDSpec(base=ssm, headdim=64, ngroups=1, n_heads=2)
    ffn = specs.FFNSpec(intermediate_size=64,
                       activation=types.Activation.SILU,
                       gate_kind=types.GateKind.SWIGLU)
    block = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=ssd, channel_mixer=ffn,
        pre_attn_norm=norm, pre_ffn_norm=norm,
        skip_ffn=True,
    )
    assert isinstance(block.token_mixer, specs.SSDSpec)
    assert block.skip_ffn is True


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
    )
    assert block.per_layer_embedding is ple


def test_attention_spec_mla_fields_default_none():
    """B2b: MLA fields default to None for non-MLA specs."""
    spec = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
    )
    assert spec.q_lora_rank is None
    assert spec.kv_lora_rank is None
    assert spec.qk_nope_head_dim is None
    assert spec.qk_rope_head_dim is None
    assert spec.v_head_dim is None


def test_attention_spec_mla_minicpm3_shape():
    """B2b: MLA spec for MiniCPM-3.

    Source: modeling_minicpm.py:351-357 / config.json (q_lora_rank=768,
    kv_lora_rank=256, qk_nope_head_dim=64, qk_rope_head_dim=32,
    v_head_dim = hidden_size/n_heads = 2560/40 = 64).
    """
    spec = specs.AttentionSpec(
        n_q_heads=40, n_kv_heads=40, head_dim=96,  # qk_head_dim
        kind=types.AttentionKind.MLA,
        qkv_layout=types.QKVLayout.MLA_LATENT,
        mask_kind=types.MaskKind.CAUSAL,
        q_lora_rank=768,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
    )
    assert spec.kind == types.AttentionKind.MLA
    assert spec.qkv_layout == types.QKVLayout.MLA_LATENT
    assert spec.qk_nope_head_dim + spec.qk_rope_head_dim == spec.head_dim


def test_mask_kind_block_sparse_enum():
    """B2b: BlockSparse mask kind exists on the enum (Phi-3-small)."""
    assert hasattr(types.MaskKind, "BLOCK_SPARSE")


def test_activation_gegelu_enum():
    """B2b: Phi-3-small uses GeGELU activation. Enum value exists."""
    assert hasattr(types.Activation, "GEGELU")


def test_b5_attention_kind_dsa_enum():
    """B5: DeepSeek-V3.2 Lightning Indexer attention kind."""
    assert hasattr(types.AttentionKind, "DSA")


def test_b5_rope_scaling_yarn_enum():
    """B5: YARN scaling enum exists (used by DeepSeek-V2-Lite & DeepSeek-V3)."""
    assert hasattr(types.RoPEScaling, "YARN")


def test_b5_yarn_rope_params():
    """B5: YarnRoPEParams matches DeepSeek-V2-Lite production config.

    Source: deepseek-ai/DeepSeek-V2-Lite/config.json (rope_scaling block:
    factor=40, beta_fast=32, beta_slow=1, mscale=0.707, mscale_all_dim=0.707,
    original_max_position_embeddings=4096).
    """
    yarn = specs.YarnRoPEParams(
        factor=40.0,
        original_max_position_embeddings=4096,
        beta_fast=32.0, beta_slow=1.0,
        mscale=0.707, mscale_all_dim=0.707,
    )
    spec = specs.RoPESpec(
        base_theta=10000.0,
        basis=types.RoPEBasis.INTERLEAVED,
        scaling=types.RoPEScaling.YARN,
        yarn_extra=yarn,
    )
    assert spec.yarn_extra is yarn
    assert spec.scaling == types.RoPEScaling.YARN


def test_b5_moe_spec_v2_lite_shape():
    """B5: MoESpec matches DeepSeek-V2-Lite production config.

    Source: deepseek-ai/DeepSeek-V2-Lite/config.json: n_routed_experts=64,
    num_experts_per_tok=6, n_shared_experts=2, routed_scaling_factor=1.0,
    norm_topk_prob=false, scoring_func=softmax, moe_intermediate_size=1408.
    """
    expert_ffn = specs.FFNSpec(
        intermediate_size=1408,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )
    moe = specs.MoESpec(
        n_experts=64, top_k=6,
        n_shared_experts=2,
        router_kind="softmax",
        router_norm=False,
        routed_scaling_factor=1.0,
        expert_ffn=expert_ffn,
    )
    assert moe.router_kind == "softmax"
    assert moe.expert_ffn.intermediate_size == 1408


def test_b5_moe_spec_v3_lite_shape():
    """B5: MoESpec for DeepSeek-V3 (sigmoid + bias + group routing).

    Source: deepseek-ai/DeepSeek-V3 default config (n_routed_experts=256,
    num_experts_per_tok=8, n_shared_experts=1, n_group=8, topk_group=4,
    routed_scaling_factor=2.5, norm_topk_prob=true).
    """
    expert_ffn = specs.FFNSpec(
        intermediate_size=2048,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )
    moe = specs.MoESpec(
        n_experts=256, top_k=8,
        n_shared_experts=1,
        router_kind="sigmoid_plus_bias",
        router_norm=True,
        group_routing=specs.GroupRoutingSpec(n_groups=8, topk_per_group=4),
        routed_scaling_factor=2.5,
        expert_ffn=expert_ffn,
    )
    assert moe.router_kind == "sigmoid_plus_bias"
    assert moe.group_routing.n_groups == 8


def test_b5_indexer_spec():
    """B5: IndexerSpec exists for DSA composition (DeepSeek-V3.2)."""
    spec = specs.IndexerSpec(indexer_dim=64, top_k=2048)
    assert spec.indexer_dim == 64


def test_b5_attention_spec_has_indexer_field():
    """B5: AttentionSpec.indexer is plumbed for DSA composition."""
    indexer = specs.IndexerSpec(indexer_dim=64, top_k=2048)
    spec = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=16, head_dim=192,
        kind=types.AttentionKind.DSA,
        qkv_layout=types.QKVLayout.MLA_LATENT,
        mask_kind=types.MaskKind.CAUSAL,
        q_lora_rank=1536, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        indexer=indexer,
    )
    assert spec.indexer is indexer


def test_b5_decoder_block_spec_accepts_moe():
    """B5: DecoderBlockSpec.channel_mixer accepts a MoESpec (V2-Lite MoE layers)."""
    norm = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.STANDARD_W)
    attn = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=16, head_dim=192,
        kind=types.AttentionKind.MLA,
        qkv_layout=types.QKVLayout.MLA_LATENT,
        mask_kind=types.MaskKind.CAUSAL,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        kv_lora_rank=512, q_lora_rank=None,
    )
    expert_ffn = specs.FFNSpec(intermediate_size=1408,
                               activation=types.Activation.SILU,
                               gate_kind=types.GateKind.SWIGLU)
    moe = specs.MoESpec(n_experts=64, top_k=6, n_shared_experts=2,
                        routed_scaling_factor=1.0, expert_ffn=expert_ffn)
    block = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn,
        channel_mixer=moe,
        pre_attn_norm=norm,
        pre_ffn_norm=norm,
    )
    assert isinstance(block.channel_mixer, specs.MoESpec)
