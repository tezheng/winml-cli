"""SHAPE-ONLY tests for Phi-3-small.

These verify the layer-factory builds nn modules at the dims implied by the
public config. Numerical equivalence vs HF Phi3SmallDecoderLayer is
DEFERRED (skip marker at end).
"""
import pytest
import torch

from api import specs, types
from models.phi3_small import config as p_config, layer as p_layer


def _hf_config_dict():
    """Mirrors the public Phi-3-small-8k-instruct config.json."""
    return {
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "ff_intermediate_size": 14336,
        "num_hidden_layers": 32,
        "rope_embedding_base": 1000000,
        "rope_position_scale": 1.0,
        "layer_norm_epsilon": 1e-5,
        "vocab_size": 100352,
        "max_position_embeddings": 8192,
        "dense_attention_every_n_layers": 2,
        "torch_dtype": "bfloat16",
        "gegelu_limit": 20.0,
        "blocksparse_block_size": 64,
        "blocksparse_num_local_blocks": 16,
        "blocksparse_vert_stride": 8,
        "blocksparse_triton_kernel_block_size": 64,
        "blocksparse_homo_head_pattern": False,
        "mup_attn_multiplier": 1.0,
        "mup_embedding_multiplier": 10.0,
        "mup_width_multiplier": 8.0,
        "mup_use_scaling": True,
        "attention_bias": False,
    }


def test_config_parses_phi3_small_8k():
    cfg = p_config.Phi3SmallConfig.from_hf_dict(_hf_config_dict())
    assert cfg.hidden_size == 4096
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.blocksparse.block_size == 64
    assert cfg.blocksparse.vert_stride == 8
    assert cfg.blocksparse.num_local_blocks == 16
    assert cfg.blocksparse.homo_head_pattern is False
    assert cfg.dense_attention_every_n_layers == 2
    assert cfg.mup_embedding_multiplier == 10.0


def test_dense_vs_blocksparse_layer_dispatch():
    """dense_attention_every_n_layers=2 ⇒ odd 1-indexed layers (i.e.
    layer_idx 1, 3, 5, ...) are dense; layer_idx 0, 2, 4, ... are BlockSparse.
    Source: modeling_phi3_small.py:213 — `(L+1) % every == 0`."""
    cfg = p_config.Phi3SmallConfig.from_hf_dict(_hf_config_dict())
    assert cfg.is_dense_layer(layer_idx=0) is False
    assert cfg.is_dense_layer(layer_idx=1) is True
    assert cfg.is_dense_layer(layer_idx=2) is False
    assert cfg.is_dense_layer(layer_idx=3) is True


def test_block_spec_layer0_blocksparse_mask():
    cfg = p_config.Phi3SmallConfig.from_hf_dict(_hf_config_dict())
    spec = cfg.to_block_spec(layer_idx=0)
    am = spec.token_mixer
    assert am.mask_kind == types.MaskKind.BLOCK_SPARSE
    assert am.qkv_layout == types.QKVLayout.FUSED
    assert am.attn_scale == 1.0 / 128            # mup_attn / head_dim
    fm = spec.channel_mixer
    assert fm.activation == types.Activation.GEGELU
    assert fm.gate_kind == types.GateKind.GEGLU
    assert fm.fused_gate_up is True
    assert spec.pre_attn_norm.kind == types.NormKind.LAYER


def test_block_spec_layer1_dense_mask():
    cfg = p_config.Phi3SmallConfig.from_hf_dict(_hf_config_dict())
    spec = cfg.to_block_spec(layer_idx=1)
    assert spec.token_mixer.mask_kind == types.MaskKind.CAUSAL


def test_layer_factory_builds_with_correct_shapes():
    """Tiny config — verify weight shapes match config-implied dims."""
    hf = _hf_config_dict()
    hf.update({
        "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "ff_intermediate_size": 256, "num_hidden_layers": 4,
        "vocab_size": 100, "max_position_embeddings": 64,
        "torch_dtype": "float32",
    })
    cfg = p_config.Phi3SmallConfig.from_hf_dict(hf)
    L = p_layer.build_phi3_small_decoder_layer(cfg, layer_idx=0)
    # query_key_value: (Hq + 2*Hk) * head_dim = (4 + 4) * 32 = 256
    assert L.query_key_value.weight.shape == (
        (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim,
        cfg.hidden_size,
    )
    assert L.dense.weight.shape == (cfg.hidden_size, cfg.hidden_size)
    # up_proj outputs 2*intermediate (gegelu chunks via even/odd indices).
    assert L.up_proj.weight.shape == (2 * cfg.intermediate_size, cfg.hidden_size)
    assert L.down_proj.weight.shape == (cfg.hidden_size, cfg.intermediate_size)
    # LayerNorms shape == hidden.
    assert L.input_layernorm.weight.shape == (cfg.hidden_size,)
    assert L.post_attention_layernorm.weight.shape == (cfg.hidden_size,)
    # is_dense_layer derived from layer_idx 0 → False (BlockSparse).
    assert L.is_dense_layer is False


def test_layer_factory_layer1_is_dense():
    hf = _hf_config_dict()
    hf.update({
        "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "ff_intermediate_size": 256, "num_hidden_layers": 4,
        "vocab_size": 100, "max_position_embeddings": 64,
        "torch_dtype": "float32",
    })
    cfg = p_config.Phi3SmallConfig.from_hf_dict(hf)
    L1 = p_layer.build_phi3_small_decoder_layer(cfg, layer_idx=1)
    assert L1.is_dense_layer is True


@pytest.mark.skip(reason="BlockSparse forward + GeGELU + interleaved-QKV "
                  "numerical gate deferred to follow-up milestone.")
def test_numerical_gate_vs_hf():
    pass
