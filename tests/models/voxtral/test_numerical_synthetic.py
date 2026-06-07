"""Voxtral LM-decoder numerical gate vs HF — SYNTHETIC weights.

We synthesize a tiny VoxtralForConditionalGeneration in-process so the LM
decoder is built via the SAME `AutoModel.from_config(text_config)` codepath
that the public checkpoint uses, then verify that our DecoderBlock
reproduces HF's eager forward exactly.

This is the canonical B9 Voxtral numerical gate — atol=5e-4.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import (  # noqa: E402
    LlamaConfig,
    VoxtralConfig as HFVoxtralConfig,
    VoxtralForConditionalGeneration,
)
from transformers.models.voxtral.configuration_voxtral import (  # noqa: E402
    VoxtralEncoderConfig,
)

from api import kvcache, specs, types  # noqa: E402
from models.voxtral import config as v_config, layer as v_layer  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _tiny_text_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 1e8, "rope_type": "default"},
        tie_word_embeddings=False,
    )


def _tiny_audio_config() -> VoxtralEncoderConfig:
    """Minimal audio encoder — present only so VoxtralConfig validates.
    Not exercised by the LM-decoder gate."""
    return VoxtralEncoderConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_mel_bins=32,
        max_source_positions=100,
    )


def _build_hf_model() -> tuple[HFVoxtralConfig, VoxtralForConditionalGeneration]:
    text_cfg = _tiny_text_config()
    audio_cfg = _tiny_audio_config()
    cfg = HFVoxtralConfig(
        audio_config=audio_cfg.to_dict(),
        text_config=text_cfg.to_dict(),
        audio_token_id=24,
        projector_hidden_act="gelu",
        tie_word_embeddings=False,
    )
    # Force eager attention so we exercise the path our api gate exercises.
    cfg.text_config._attn_implementation = "eager"
    torch.manual_seed(0)
    m = VoxtralForConditionalGeneration(cfg)
    m.eval()
    return cfg, m


def _our_cfg(hf_cfg: HFVoxtralConfig) -> v_config.VoxtralConfig:
    return v_config.VoxtralConfig.from_hf_dict(hf_cfg.to_dict())


def _forward_hf_layer(
    hf_model: VoxtralForConditionalGeneration,
    layer_idx: int,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Run a single LlamaDecoderLayer (Voxtral's language_model layer) with
    the standard HF eager preamble (cos/sin via the model's rotary_emb,
    causal mask)."""
    lang_model = hf_model.model.language_model
    B, S, _ = hidden.shape
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = lang_model.rotary_emb(hidden, pos)
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        out = lang_model.layers[layer_idx](
            hidden,
            attention_mask=attn_mask,
            position_ids=pos,
            position_embeddings=(cos, sin),
            past_key_values=None,
            use_cache=False,
        )
    return out


def _forward_api_layer(api_blk, hidden, cfg: v_config.VoxtralConfig):
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


def test_layer0_matches_hf_synthetic():
    hf_cfg, hf_model = _build_hf_model()
    our_cfg = _our_cfg(hf_cfg)
    api_blk = v_layer.build_voxtral_decoder_layer(
        our_cfg, layer_idx=0, max_seq=our_cfg.max_position_embeddings,
    )
    # HF state-dict here has the canonical Voxtral prefix
    # `model.language_model.layers.{L}.*` — load_hf_voxtral_layer's default
    # prefix matches.
    v_layer.load_hf_voxtral_layer(api_blk, hf_model.state_dict(), layer_idx=0)
    api_blk.eval()

    torch.manual_seed(1)
    B, S = 1, 6
    hidden = torch.randn(B, S, our_cfg.hidden_size)

    hf_out = _forward_hf_layer(hf_model, 0, hidden)
    api_out = _forward_api_layer(api_blk, hidden, our_cfg)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Voxtral layer-0 max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"Voxtral (synthetic) layer-0 max_abs_diff={max_abs_diff:.2e}")


def test_layer1_matches_hf_synthetic():
    hf_cfg, hf_model = _build_hf_model()
    our_cfg = _our_cfg(hf_cfg)
    api_blk = v_layer.build_voxtral_decoder_layer(
        our_cfg, layer_idx=1, max_seq=our_cfg.max_position_embeddings,
    )
    v_layer.load_hf_voxtral_layer(api_blk, hf_model.state_dict(), layer_idx=1)
    api_blk.eval()

    torch.manual_seed(2)
    B, S = 1, 8
    hidden = torch.randn(B, S, our_cfg.hidden_size)

    hf_out = _forward_hf_layer(hf_model, 1, hidden)
    api_out = _forward_api_layer(api_blk, hidden, our_cfg)
    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Voxtral layer-1 max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
    print(f"Voxtral (synthetic) layer-1 max_abs_diff={max_abs_diff:.2e}")
