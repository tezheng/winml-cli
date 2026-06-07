"""Moshi main-decoder numerical gate vs HF — SYNTHETIC weights.

We synthesize a tiny MoshiModel in-process (random init) and verify that
our DecoderBlock reproduces HF's eager forward exactly. This is the
canonical B9 Moshi numerical gate: it exercises the EXACT HF code path
(MoshiAttention eager, MoshiGatingMLP fused SwiGLU split, RoPE,
RMSNorm-with-fp32-upcast) without needing the 14 GB Helium checkpoint
download.

Numerical tolerance: atol = 5e-4 (project gate).
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import MoshiConfig  # noqa: E402
from transformers.models.moshi.modeling_moshi import MoshiModel  # noqa: E402

from api import kvcache, specs, types  # noqa: E402
from models.moshi import config as m_config, layer as m_layer  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _tiny_cfg() -> MoshiConfig:
    return MoshiConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,                 # MHA
        head_dim=16,
        ffn_dim=128,                           # intermediate = 64
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        rms_norm_eps=1e-8,
        max_position_embeddings=64,
        sliding_window=64,                      # >= max_pos → equivalent to causal
        num_codebooks=8,
        audio_vocab_size=32,
        attn_implementation="eager",
    )


def _build_hf_model() -> tuple[MoshiConfig, MoshiModel]:
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    m = MoshiModel(cfg)
    m.eval()
    return cfg, m


def _our_cfg(hf_cfg: MoshiConfig) -> m_config.MoshiConfig:
    return m_config.MoshiConfig.from_hf_dict(hf_cfg.to_dict())


def _forward_hf_layer(
    hf_model: MoshiModel, layer_idx: int, hidden: torch.Tensor
) -> torch.Tensor:
    """Run a single HF MoshiDecoderLayer with causal mask + cos/sin built
    via the model-owned MoshiRotaryEmbedding inside MoshiAttention.

    Each MoshiAttention has its own `rotary_emb` instance — we let the
    layer's forward call it via `position_ids`. We pass attention_mask
    in the HF-canonical 4D additive form ([B, 1, S, S], -inf above
    diagonal).
    """
    B, S, _ = hidden.shape
    pos = torch.arange(S).unsqueeze(0)
    attn_mask = torch.full((1, 1, S, S), float("-inf"))
    attn_mask = torch.triu(attn_mask, diagonal=1)
    layer = hf_model.layers[layer_idx]
    with torch.no_grad():
        outputs = layer(
            hidden,
            attention_mask=attn_mask,
            position_ids=pos,
            past_key_values=None,
            use_cache=False,
        )
    # MoshiDecoderLayer returns a tuple (hidden,) (plus optional attn weights).
    return outputs[0]


def _forward_api_layer(api_blk, hidden, cfg: m_config.MoshiConfig):
    B, S, _ = hidden.shape
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        return api_blk(hidden, position_ids=pos, cache=cache, start_pos=0)


def _load_into_block(api_blk, hf_model: MoshiModel, layer_idx: int):
    """The HF MoshiModel state_dict has keys `layers.{L}.*`; our loader
    expects `model.layers.{L}.*`. We re-prefix the relevant subset."""
    src = hf_model.state_dict()
    remap = {}
    pfx = f"model.layers.{layer_idx}"
    for k, v in src.items():
        if k.startswith(f"layers.{layer_idx}."):
            remap[k.replace(f"layers.{layer_idx}.", f"{pfx}.")] = v
    m_layer.load_hf_moshi_layer(api_blk, remap, layer_idx)


def test_layer0_matches_hf_synthetic():
    hf_cfg, hf_model = _build_hf_model()
    our_cfg = _our_cfg(hf_cfg)
    api_blk = m_layer.build_moshi_decoder_layer(
        our_cfg, layer_idx=0, max_seq=our_cfg.max_position_embeddings,
    )
    _load_into_block(api_blk, hf_model, layer_idx=0)
    api_blk.eval()

    torch.manual_seed(1)
    B, S = 1, 6
    hidden = torch.randn(B, S, hf_cfg.hidden_size)

    hf_out = _forward_hf_layer(hf_model, 0, hidden)
    api_out = _forward_api_layer(api_blk, hidden, our_cfg)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Moshi layer-0 max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"Moshi (synthetic) layer-0 max_abs_diff={max_abs_diff:.2e}")


def test_layer1_matches_hf_synthetic():
    hf_cfg, hf_model = _build_hf_model()
    our_cfg = _our_cfg(hf_cfg)
    api_blk = m_layer.build_moshi_decoder_layer(
        our_cfg, layer_idx=1, max_seq=our_cfg.max_position_embeddings,
    )
    _load_into_block(api_blk, hf_model, layer_idx=1)
    api_blk.eval()

    torch.manual_seed(2)
    B, S = 1, 8
    hidden = torch.randn(B, S, hf_cfg.hidden_size)

    hf_out = _forward_hf_layer(hf_model, 1, hidden)
    api_out = _forward_api_layer(api_blk, hidden, our_cfg)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Moshi layer-1 max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"Moshi (synthetic) layer-1 max_abs_diff={max_abs_diff:.2e}")
