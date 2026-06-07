"""Isolation tests for MiniCPM-3 MLA sub-ops.

We import the openbmb modeling_minicpm.py from the HF cache (the model is a
trust_remote_code model — there is no transformers.models.minicpm3 native
package). The isolation tests instantiate HF primitives WITHOUT loading the
full 4B-param weights — they only need the small per-op constructors. So
they always run as long as the modeling .py file exists.
"""
import glob
import importlib.util
import os
import math

import pytest
import torch
import torch.nn.functional as F

from api import ops, rope, specs, types


def _load_modeling_minicpm():
    """Locate and import the openbmb modeling_minicpm.py file.

    The openbmb file targets transformers 4.41.0 and imports several symbols
    that were removed in newer versions (is_torch_fx_available,
    _prepare_4d_causal_attention_mask, etc.). We monkey-patch the minimal
    set of missing symbols BEFORE the module is imported. The patches are
    no-op stubs since the isolation tests only construct the per-op submodules
    (MiniCPMRMSNorm, MiniCPMRotaryEmbedding, MiniCPMLongRoPE,
    apply_rotary_pos_emb) — none of those need the masking helpers.
    """
    cache = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    candidates = glob.glob(os.path.join(
        cache, "models--openbmb--MiniCPM3-4B", "snapshots", "*", "modeling_minicpm.py"
    ))
    if not candidates:
        pytest.skip("openbmb modeling_minicpm.py not in cache")

    # Patch missing symbols in transformers.utils.import_utils and
    # transformers.modeling_attn_mask_utils.
    from transformers.utils import import_utils as _iu
    if not hasattr(_iu, "is_torch_fx_available"):
        _iu.is_torch_fx_available = lambda: False
    from transformers import modeling_attn_mask_utils as _mam
    if not hasattr(_mam, "_prepare_4d_causal_attention_mask"):
        _mam._prepare_4d_causal_attention_mask = lambda *a, **k: None
    if not hasattr(_mam, "_prepare_4d_attention_mask"):
        _mam._prepare_4d_attention_mask = lambda *a, **k: None
    if not hasattr(_mam, "_prepare_4d_causal_attention_mask_for_sdpa"):
        _mam._prepare_4d_causal_attention_mask_for_sdpa = lambda *a, **k: None
    # is_torch_greater_or_equal_than_1_13 was moved/removed in newer
    # transformers; patch a True stub onto pytorch_utils if missing.
    from transformers import pytorch_utils as _pu
    if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
        _pu.is_torch_greater_or_equal_than_1_13 = True

    src = candidates[0]
    src_dir = os.path.dirname(src)
    # Create a synthetic package so the relative import
    # `from .configuration_minicpm import MiniCPM3Config` resolves.
    import sys
    pkg_name = "_openbmb_minicpm"
    if pkg_name not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_name, os.path.join(src_dir, "__init__.py"),
            submodule_search_locations=[src_dir],
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkg_name] = pkg
    # Load configuration_minicpm first.
    cfg_path = os.path.join(src_dir, "configuration_minicpm.py")
    try:
        cfg_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.configuration_minicpm", cfg_path,
        )
        cfg_mod = importlib.util.module_from_spec(cfg_spec)
        sys.modules[f"{pkg_name}.configuration_minicpm"] = cfg_mod
        cfg_spec.loader.exec_module(cfg_mod)
    except Exception as e:
        pytest.skip(f"configuration_minicpm.py import failed: {type(e).__name__}: {str(e)[:200]}")
    # Now load modeling_minicpm.
    try:
        mod_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.modeling_minicpm", src,
        )
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules[f"{pkg_name}.modeling_minicpm"] = mod
        mod_spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"openbmb modeling_minicpm.py failed to import: {type(e).__name__}: {str(e)[:200]}")
    return mod


def test_rms_layernorm_matches_hf():
    """MiniCPMRMSNorm — same as Qwen3 RMSNorm STANDARD_W."""
    torch.manual_seed(0)
    mod = _load_modeling_minicpm()
    hidden = 64
    hf_norm = mod.MiniCPMRMSNorm(hidden, eps=1e-5)
    weight = torch.randn(hidden)
    hf_norm.weight.data.copy_(weight)
    x = torch.randn(1, 3, hidden)
    hf_out = hf_norm(x)
    api_out = ops.rms_norm(x, weight, 1e-5, mode="standard_w")
    assert torch.allclose(hf_out, api_out, atol=1e-5)


def test_apply_rotary_pos_emb_matches_hf():
    """MiniCPM's apply_rotary_pos_emb has an extra fp32 upcast wrapper but the
    underlying math is rotate_half-style SPLIT_HALF RoPE. We verify against
    api.ops.rope_apply.

    HF signature: apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)
    where cos/sin have shape [max_seq, qk_rope_head_dim] and q/k have shape
    [B, H, S, qk_rope_head_dim]. With position_ids of shape [B, S], the HF
    op indexes cos[position_ids] → [B, S, qk_rope_head_dim] then unsqueezes
    at dim=1 → [B, 1, S, qk_rope_head_dim].
    """
    torch.manual_seed(0)
    mod = _load_modeling_minicpm()
    H = 2
    S = 4
    qk_rope = 16
    max_seq = 32
    # Build cos/sin matching MiniCPMRotaryEmbedding shape [max_seq, qk_rope].
    base = 10_000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, qk_rope, 2).float() / qk_rope))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)     # [max_seq, qk_rope]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

    q = torch.randn(1, H, S, qk_rope)
    k = torch.randn(1, 1, S, qk_rope)
    position_ids = torch.arange(S).unsqueeze(0)              # [1, S]

    # HF op
    q_hf, k_hf = mod.apply_rotary_pos_emb(
        q, k, cos, sin, position_ids, unsqueeze_dim=1,
    )

    # api equivalent: rope_apply expects cos/sin shape [S, qk_rope] (after
    # position_ids indexing), q/k shape [B, S, H, Dh]. We transpose 1↔2 first.
    q_bshd = q.transpose(1, 2)
    k_bshd = k.transpose(1, 2)
    cos_idx = cos[position_ids.squeeze(0)]                  # [S, qk_rope]
    sin_idx = sin[position_ids.squeeze(0)]
    q_api, k_api = ops.rope_apply(q_bshd, k_bshd, cos_idx, sin_idx,
                                  basis="split_half")
    q_api = q_api.transpose(1, 2)
    k_api = k_api.transpose(1, 2)

    # HF version upcasts to fp32 internally, so small numerical drift is
    # possible — atol=1e-5 still holds for fp32 inputs.
    assert torch.allclose(q_hf, q_api, atol=1e-5)
    assert torch.allclose(k_hf, k_api, atol=1e-5)


def test_rope_default_cos_sin_match_hf():
    """Default MiniCPMRotaryEmbedding (no scaling) cos/sin tables match
    api.rope.RoPE built with head_dim=qk_rope_head_dim, partial_rotary=1.0.
    """
    torch.manual_seed(0)
    mod = _load_modeling_minicpm()
    qk_rope = 16
    max_seq = 32
    hf_rotary = mod.MiniCPMRotaryEmbedding(qk_rope, max_position_embeddings=max_seq,
                                           base=10000.0)
    # HF returns cos/sin of shape [max_seq, qk_rope] (after .to(dtype)).
    hf_cos = hf_rotary.cos_cached
    hf_sin = hf_rotary.sin_cached

    rope_spec = specs.RoPESpec(
        base_theta=10000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.NONE,
        partial_rotary_factor=1.0,
    )
    api_rope = rope.RoPE(rope_spec, head_dim=qk_rope, max_seq=max_seq,
                         dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5)
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5)


def test_longrope_cos_sin_match_hf_short_branch():
    """LongRoPE short-table branch (seq_len ≤ orig_max). MiniCPM-3 has
    short_factor == long_factor, so the short branch covers the practical case."""
    torch.manual_seed(0)
    mod = _load_modeling_minicpm()
    qk_rope = 16
    max_seq = 32  # ≤ orig_max → short branch
    orig_max = 64
    short_factor = [1.0 + 0.1 * i for i in range(qk_rope // 2)]
    long_factor = [2.0 + 0.1 * i for i in range(qk_rope // 2)]

    hf_rotary = mod.MiniCPMLongRoPE(
        qk_rope, max_position_embeddings=max_seq,
        base=10000.0, short_factor=short_factor, long_factor=long_factor,
        original_max_position_embeddings=orig_max,
    )
    hf_cos = hf_rotary.cos_cached
    hf_sin = hf_rotary.sin_cached

    # api spec — attention_factor follows MiniCPM-3's HF formula
    # (no scale<=1 clamp; modeling_minicpm.py:218-222).
    scale = max_seq / orig_max
    att = math.sqrt(1.0 + math.log(scale) / math.log(orig_max))
    rope_spec = specs.RoPESpec(
        base_theta=10000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        scaling=types.RoPEScaling.LONGROPE,
        longrope_extra=specs.LongRoPEParams(
            short_factor=tuple(short_factor),
            long_factor=tuple(long_factor),
            original_max_position_embeddings=orig_max,
            attention_factor=att,
        ),
        partial_rotary_factor=1.0,
    )
    api_rope = rope.RoPE(rope_spec, head_dim=qk_rope, max_seq=max_seq,
                         dtype=torch.float32)
    assert torch.allclose(api_rope.cos_cached, hf_cos, atol=1e-5), \
        f"cos diff max={(api_rope.cos_cached - hf_cos).abs().max()}"
    assert torch.allclose(api_rope.sin_cached, hf_sin, atol=1e-5), \
        f"sin diff max={(api_rope.sin_cached - hf_sin).abs().max()}"
