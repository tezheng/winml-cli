"""Numerical gate vs openbmb/MiniCPM3-4B with REAL weights.

Loads the openbmb pytorch_model.bin (~8GB) into the HF MiniCPMDecoderLayer
and copies weights into our api block. Skipped if the bin file is not
present in HF_HOME.

This is the canonical gate; the synthetic-weights gate in
test_numerical_synthetic.py covers the same logic without the download.
"""
import glob
import importlib.util
import os
import sys

import pytest
import torch

from api import kvcache, specs, types
from models.minicpm3 import config as m_config, layer as m_layer


MODEL_ID = "openbmb/MiniCPM3-4B"
ATOL = 5e-4
RTOL = 5e-4
FIXED_INPUT = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)


def _cache_dir() -> str:
    return os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )


def _minicpm3_weights_present(cache_dir: str) -> bool:
    snaps = glob.glob(os.path.join(
        cache_dir, "models--openbmb--MiniCPM3-4B", "snapshots", "*",
    ))
    for snap in snaps:
        # Either pytorch_model.bin or model.safetensors
        if os.path.exists(os.path.join(snap, "pytorch_model.bin")):
            return True
        if glob.glob(os.path.join(snap, "*.safetensors")):
            return True
    return False


def _load_openbmb_modeling():
    cache = _cache_dir()
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
        pytest.skip(f"openbmb modeling_minicpm.py failed: {type(e).__name__}: {str(e)[:200]}")
    return mod, cfg_mod


@pytest.fixture(scope="module")
def hf_model_and_cfg():
    cache_dir = _cache_dir()
    if not _minicpm3_weights_present(cache_dir):
        pytest.skip(f"MiniCPM3-4B weights not downloaded in {cache_dir}")
    mod, cfg_mod = _load_openbmb_modeling()

    # Read config.json directly.
    import json
    snap = glob.glob(os.path.join(
        cache_dir, "models--openbmb--MiniCPM3-4B", "snapshots", "*",
    ))[0]
    with open(os.path.join(snap, "config.json")) as f:
        cfg_dict = json.load(f)
    hf_cfg = cfg_mod.MiniCPM3Config(**{k: v for k, v in cfg_dict.items()
                                       if not k.startswith("_")})
    hf_cfg._attn_implementation = "eager"

    # Load weights.
    bin_path = os.path.join(snap, "pytorch_model.bin")
    if os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        sd_files = glob.glob(os.path.join(snap, "*.safetensors"))
        if not sd_files:
            pytest.skip("No weights file found")
        from safetensors.torch import load_file
        sd = {}
        for f in sd_files:
            sd.update(load_file(f))
    # Filter to layer 0 + embed.
    return mod, cfg_mod, hf_cfg, sd


def test_layer0_forward_matches_hf_real_weights(hf_model_and_cfg):
    mod, cfg_mod, hf_cfg, sd = hf_model_and_cfg

    # Build only layer 0 + embed (avoid materialising 62 layers).
    hf_layer = mod.MiniCPMDecoderLayer(hf_cfg, layer_idx=0)
    # Copy weights for layer 0 from the global state dict.
    layer_sd = {}
    for k, v in sd.items():
        if k.startswith("model.layers.0."):
            layer_sd[k[len("model.layers.0."):]] = v
    missing, unexpected = hf_layer.load_state_dict(layer_sd, strict=False)
    if missing:
        pytest.skip(f"missing layer-0 state dict keys: {missing[:5]}")
    hf_layer = hf_layer.to(torch.float32)
    hf_layer.eval()

    # Embed inputs via embed_tokens weight.
    embed_w = sd["model.embed_tokens.weight"].to(torch.float32)
    embed_out = torch.nn.functional.embedding(FIXED_INPUT, embed_w)
    embed_out = embed_out * float(hf_cfg.scale_emb)
    B, S = FIXED_INPUT.shape

    position_ids = torch.arange(S).unsqueeze(0)
    attn_mask = torch.full((B, 1, S, S), float("-inf"))
    attn_mask = torch.triu(attn_mask, diagonal=1)
    with torch.no_grad():
        hf_out = hf_layer(
            embed_out, attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=False,
        )
        if isinstance(hf_out, tuple):
            hf_out = hf_out[0]

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
        original_max_position_embeddings=hf_cfg.rope_scaling[
            "original_max_position_embeddings"] if hf_cfg.rope_scaling else hf_cfg.max_position_embeddings,
        tie_word_embeddings=False,
        dtype=torch.float32,
        scale_emb=float(hf_cfg.scale_emb),
        scale_depth=float(hf_cfg.scale_depth),
        dim_model_base=int(hf_cfg.dim_model_base),
        rope_type=(hf_cfg.rope_scaling["type"] if hf_cfg.rope_scaling else "default"),
        longrope_short_factor=(tuple(hf_cfg.rope_scaling["short_factor"])
                               if hf_cfg.rope_scaling else None),
        longrope_long_factor=(tuple(hf_cfg.rope_scaling["long_factor"])
                              if hf_cfg.rope_scaling else None),
        attention_bias=False,
    )
    api_blk = m_layer.build_minicpm3_decoder_layer(
        api_cfg, layer_idx=0, max_seq=64,
    )
    m_layer.load_hf_minicpm3_layer(api_blk, sd, layer_idx=0)
    api_blk.eval()

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=api_cfg.num_attention_heads,
        head_dim=api_cfg.qk_head_dim, max_seq=64,
        v_head_dim=api_cfg.v_head_dim,
    )
    with torch.no_grad():
        api_out = api_blk(embed_out, position_ids=torch.arange(S),
                          cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"MiniCPM-3-4B layer-0 (real weights) max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
