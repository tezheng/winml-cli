"""Numerical equivalence vs HF MiniCPMDecoderLayer with RANDOM weights.

This avoids downloading the 8GB MiniCPM3-4B weights — we instantiate the
HF MLA module, copy its random-initialized weights into our api block, and
verify the layer-0 forward outputs match at atol=5e-4.

Provides the strongest single-test numerical guarantee for MLA without
network dependency.
"""
import glob
import importlib.util
import math
import os
import sys

import pytest
import torch

from api import kvcache, specs, types
from models.minicpm3 import config as m_config, layer as m_layer


ATOL = 5e-4
RTOL = 5e-4


def _load_openbmb_minicpm():
    """Mirror the loader from test_isolation_hf.py."""
    cache = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    candidates = glob.glob(os.path.join(
        cache, "models--openbmb--MiniCPM3-4B", "snapshots", "*", "modeling_minicpm.py"
    ))
    if not candidates:
        pytest.skip("openbmb modeling_minicpm.py not in cache")
    src = candidates[0]
    src_dir = os.path.dirname(src)
    # Stub missing transformers symbols.
    from transformers.utils import import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: False
    from transformers import modeling_attn_mask_utils as _mam
    for sym in ("_prepare_4d_causal_attention_mask", "_prepare_4d_attention_mask",
                "_prepare_4d_causal_attention_mask_for_sdpa"):
        if not hasattr(_mam, sym):
            setattr(_mam, sym, lambda *a, **k: None)
    from transformers import pytorch_utils as _pu
    if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
        _pu.is_torch_greater_or_equal_than_1_13 = True

    pkg_name = "_openbmb_minicpm"
    if pkg_name not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_name, os.path.join(src_dir, "__init__.py"),
            submodule_search_locations=[src_dir],
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkg_name] = pkg
    cfg_path = os.path.join(src_dir, "configuration_minicpm.py")
    cfg_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.configuration_minicpm", cfg_path,
    )
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    sys.modules[f"{pkg_name}.configuration_minicpm"] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)
    mod_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.modeling_minicpm", src,
    )
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[f"{pkg_name}.modeling_minicpm"] = mod
    try:
        mod_spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"openbmb modeling_minicpm.py failed to import: {type(e).__name__}: {str(e)[:200]}")
    return mod, cfg_mod


def _build_small_hf_config(cfg_mod, mod):
    """Build a small MiniCPM3Config in-process — keeps the test under 1s."""
    hf_cfg = cfg_mod.MiniCPM3Config(
        vocab_size=100,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=128,
        kv_lora_rank=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling={
            "type": "longrope",
            "short_factor": [1.0 + 0.05 * i for i in range(8)],
            "long_factor": [1.0 + 0.05 * i for i in range(8)],
            "original_max_position_embeddings": 64,
        },
        scale_emb=1.0,        # We test layer-level, not embedding scale.
        scale_depth=1.4,
        dim_model_base=64,
        torch_dtype="float32",
        attention_dropout=0.0,
        attention_bias=False,
    )
    # Tell HF to use eager attention so MiniCPMAttention is selected
    # (not MiniCPMSdpaAttention which has a slightly different code path).
    hf_cfg._attn_implementation = "eager"
    return hf_cfg


def _copy_hf_layer_to_api_block(hf_layer, api_blk, layer_idx=0):
    """Take an in-process HF MiniCPMDecoderLayer and copy its weights
    into the api DecoderBlock. Builds a synthetic state-dict in the HF
    naming convention, then delegates to load_hf_minicpm3_layer.
    """
    sd = {}
    L = layer_idx
    sd[f"model.layers.{L}.input_layernorm.weight"]          = hf_layer.input_layernorm.weight.data
    sd[f"model.layers.{L}.post_attention_layernorm.weight"] = hf_layer.post_attention_layernorm.weight.data
    sd[f"model.layers.{L}.self_attn.q_a_proj.weight"]       = hf_layer.self_attn.q_a_proj.weight.data
    sd[f"model.layers.{L}.self_attn.q_a_layernorm.weight"]  = hf_layer.self_attn.q_a_layernorm.weight.data
    sd[f"model.layers.{L}.self_attn.q_b_proj.weight"]       = hf_layer.self_attn.q_b_proj.weight.data
    sd[f"model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight"] = hf_layer.self_attn.kv_a_proj_with_mqa.weight.data
    sd[f"model.layers.{L}.self_attn.kv_a_layernorm.weight"] = hf_layer.self_attn.kv_a_layernorm.weight.data
    sd[f"model.layers.{L}.self_attn.kv_b_proj.weight"]      = hf_layer.self_attn.kv_b_proj.weight.data
    sd[f"model.layers.{L}.self_attn.o_proj.weight"]         = hf_layer.self_attn.o_proj.weight.data
    sd[f"model.layers.{L}.mlp.gate_proj.weight"]            = hf_layer.mlp.gate_proj.weight.data
    sd[f"model.layers.{L}.mlp.up_proj.weight"]              = hf_layer.mlp.up_proj.weight.data
    sd[f"model.layers.{L}.mlp.down_proj.weight"]            = hf_layer.mlp.down_proj.weight.data
    m_layer.load_hf_minicpm3_layer(api_blk, sd, layer_idx=layer_idx)


def test_mla_decoder_layer_numerical_equivalence():
    """MiniCPM-3 MLA decoder layer: api forward == HF forward at atol=5e-4."""
    mod, cfg_mod = _load_openbmb_minicpm()

    torch.manual_seed(0)
    hf_cfg = _build_small_hf_config(cfg_mod, mod)

    # Build HF decoder layer at layer_idx=0.
    hf_layer = mod.MiniCPMDecoderLayer(hf_cfg, layer_idx=0)
    hf_layer.eval()

    # Build api block.
    api_cfg = m_config.MiniCPM3Config(
        hidden_size=hf_cfg.hidden_size,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        intermediate_size=hf_cfg.intermediate_size,
        q_lora_rank=hf_cfg.q_lora_rank,
        kv_lora_rank=hf_cfg.kv_lora_rank,
        qk_nope_head_dim=hf_cfg.qk_nope_head_dim,
        qk_rope_head_dim=hf_cfg.qk_rope_head_dim,
        v_head_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        rope_theta=hf_cfg.rope_theta,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        vocab_size=hf_cfg.vocab_size,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        original_max_position_embeddings=64,
        tie_word_embeddings=False,
        dtype=torch.float32,
        scale_depth=hf_cfg.scale_depth,
        scale_emb=hf_cfg.scale_emb,
        dim_model_base=hf_cfg.dim_model_base,
        rope_type="longrope",
        longrope_short_factor=tuple(hf_cfg.rope_scaling["short_factor"]),
        longrope_long_factor=tuple(hf_cfg.rope_scaling["long_factor"]),
        attention_bias=False,
    )
    api_blk = m_layer.build_minicpm3_decoder_layer(api_cfg, layer_idx=0, max_seq=32)
    _copy_hf_layer_to_api_block(hf_layer, api_blk, layer_idx=0)
    api_blk.eval()

    # Forward inputs.
    B, S = 1, 5
    x = torch.randn(B, S, hf_cfg.hidden_size)
    position_ids = torch.arange(S).unsqueeze(0)
    # Build a causal 4D mask of shape (B, 1, S, S) with -inf above diag.
    attn_mask = torch.full((B, 1, S, S), float("-inf"))
    attn_mask = torch.triu(attn_mask, diagonal=1)

    with torch.no_grad():
        hf_out = hf_layer(
            x,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=api_cfg.num_attention_heads,
        head_dim=api_cfg.qk_head_dim, max_seq=32,
        v_head_dim=api_cfg.v_head_dim,
    )
    with torch.no_grad():
        api_out = api_blk(x, position_ids=torch.arange(S), cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"MiniCPM-3 layer-0 (synthetic weights) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
