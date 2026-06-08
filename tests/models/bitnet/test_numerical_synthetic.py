"""BitNet b1.58 decoder-layer numerical gate — SYNTHETIC weights.

Builds a tiny BitNetForCausalLM in-process, then verifies our
DecoderBlock matches the HF eager forward.

v5-phase2 V3 — exercises:
- AttentionSpec.attn_sub_norm (RMSNorm on attn output BEFORE o_proj)
- FFNSpec.ffn_sub_norm (RMSNorm on `act(gate)*up` BEFORE down_proj)
- Activation.RELU2 (squared ReLU in the SwiGLU branch)

atol = 5e-4.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import BitNetConfig as HFBitNetConfig  # noqa: E402
from transformers.models.bitnet.modeling_bitnet import BitNetModel  # noqa: E402

from api import kvcache, specs, types  # noqa: E402
from models.bitnet import config as b_config, layer as b_layer  # noqa: E402


ATOL = 5e-4
RTOL = 5e-4


def _tiny_hf_cfg() -> HFBitNetConfig:
    return HFBitNetConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="relu2",
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        attention_bias=False,
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
    )


def _build_hf_model() -> tuple[HFBitNetConfig, BitNetModel]:
    cfg = _tiny_hf_cfg()
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = BitNetModel(cfg)
    m.eval()
    return cfg, m


def _our_cfg(hf_cfg: HFBitNetConfig) -> b_config.BitNetConfig:
    return b_config.BitNetConfig.from_hf_dict(hf_cfg.to_dict())


def _forward_hf_layer(hf_model: BitNetModel, layer_idx: int,
                      hidden: torch.Tensor) -> torch.Tensor:
    """Run one BitNetDecoderLayer with the standard HF eager preamble."""
    B, S, _ = hidden.shape
    pos = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = hf_model.rotary_emb(hidden, pos)
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        out = hf_model.layers[layer_idx](
            hidden, attention_mask=attn_mask, position_ids=pos,
            past_key_values=None, use_cache=False,
            position_embeddings=(cos, sin),
        )
    return out


def _forward_api_layer(api_blk, hidden, cfg: b_config.BitNetConfig):
    B, S, _ = hidden.shape
    cache = kvcache.ContiguousKVCache(
        specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=torch.float32, v_dtype=torch.float32,
        ),
        batch_size=B, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=cfg.max_position_embeddings,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        return api_blk(hidden, position_ids=pos, cache=cache, start_pos=0)


def test_bitnet_layer0_matches_hf_synthetic():
    hf_cfg, hf_model = _build_hf_model()
    our_cfg = _our_cfg(hf_cfg)
    api_blk = b_layer.build_bitnet_decoder_layer(
        our_cfg, layer_idx=0, max_seq=our_cfg.max_position_embeddings,
    )
    b_layer.load_hf_bitnet_layer(api_blk, hf_model.state_dict(),
                                  layer_idx=0, prefix="layers")
    api_blk.eval()

    torch.manual_seed(1)
    B, S = 1, 7
    hidden = torch.randn(B, S, our_cfg.hidden_size)
    hf_out = _forward_hf_layer(hf_model, 0, hidden)
    api_out = _forward_api_layer(api_blk, hidden, our_cfg)
    diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"BitNet layer-0 max_abs_diff={diff:.6e}, atol={ATOL}"
    )
    print(f"BitNet (synthetic) layer-0 max_abs_diff={diff:.2e}")


def test_bitnet_layer1_matches_hf_synthetic():
    hf_cfg, hf_model = _build_hf_model()
    our_cfg = _our_cfg(hf_cfg)
    api_blk = b_layer.build_bitnet_decoder_layer(
        our_cfg, layer_idx=1, max_seq=our_cfg.max_position_embeddings,
    )
    b_layer.load_hf_bitnet_layer(api_blk, hf_model.state_dict(),
                                  layer_idx=1, prefix="layers")
    api_blk.eval()
    torch.manual_seed(2)
    B, S = 1, 9
    hidden = torch.randn(B, S, our_cfg.hidden_size)
    hf_out = _forward_hf_layer(hf_model, 1, hidden)
    api_out = _forward_api_layer(api_blk, hidden, our_cfg)
    diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"BitNet layer-1 max_abs_diff={diff:.6e}, atol={ATOL}"
    )
