import pytest
import torch

from api import kvcache, specs, types
from models.smollm3 import config, layer


def _config_3b():
    nope = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 0]
    return config.Smollm3Config(
        hidden_size=2048, num_attention_heads=16, num_key_value_heads=4,
        head_dim=128, intermediate_size=11008, num_hidden_layers=36,
        rope_theta=5_000_000.0, rms_norm_eps=1e-6,
        vocab_size=128256, max_position_embeddings=65536,
        tie_word_embeddings=True, dtype=torch.float32,
        no_rope_layers=tuple(nope),
    )


@pytest.mark.parametrize("layer_idx", [0, 3, 4, 7, 35])
def test_smollm3_decoder_layer_forward_shape(layer_idx):
    """Test both RoPE (0, 4) and NoPE (3, 7, 35) layers."""
    cfg = _config_3b()
    blk = layer.build_smollm3_decoder_layer(cfg, layer_idx=layer_idx, max_seq=128)
    B, S = 1, 8
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=cfg.dtype, v_dtype=cfg.dtype,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=128,
    )
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_smollm3_nope_layer_has_no_rope_module():
    cfg = _config_3b()
    nope_blk = layer.build_smollm3_decoder_layer(cfg, layer_idx=3, max_seq=64)
    rope_blk = layer.build_smollm3_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert nope_blk.attention.rope is None
    assert rope_blk.attention.rope is not None


def test_smollm3_nope_invariant_to_position_ids():
    """A NoPE layer's output is invariant to position_ids — the attention output
    has no rotation applied. Verified by running the same layer twice with
    different position_ids (but same cache + start_pos) and asserting bitwise
    equal outputs.
    """
    cfg = _config_3b()
    torch.manual_seed(0)
    x = torch.randn(1, 4, cfg.hidden_size)

    def run(layer_idx, pos):
        blk = layer.build_smollm3_decoder_layer(cfg, layer_idx=layer_idx, max_seq=64)
        torch.manual_seed(layer_idx)
        for p in blk.parameters():
            p.data.copy_(torch.randn_like(p) * 0.02)
        cache_spec = specs.KVCacheSpec(
            layout=types.CacheLayout.CONTIGUOUS,
            memory_layout=types.MemoryLayout.HND,
            k_dtype=cfg.dtype, v_dtype=cfg.dtype,
        )
        cache = kvcache.ContiguousKVCache(
            cache_spec, batch_size=1,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, max_seq=64,
        )
        return blk(x, position_ids=pos, cache=cache, start_pos=0)

    nope_a = run(3, torch.arange(4))
    nope_b = run(3, torch.arange(20, 24))
    # NoPE layer: outputs are bit-identical regardless of position_ids.
    assert torch.equal(nope_a, nope_b)
