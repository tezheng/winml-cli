import torch

from api import kvcache, specs, types
from models.tinyllama import config, layer


def _config_1p1b():
    hf = {
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
        "head_dim": 64, "intermediate_size": 5632, "num_hidden_layers": 22,
        "rms_norm_eps": 1e-5, "vocab_size": 32000,
        "max_position_embeddings": 2048, "tie_word_embeddings": False,
        "dtype": "float32",
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    }
    return config.TinyLlamaConfig.from_hf_dict(hf)


def test_tinyllama_decoder_layer_forward_shape():
    cfg = _config_1p1b()
    blk = layer.build_tinyllama_decoder_layer(cfg, layer_idx=0, max_seq=128)
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


def test_tinyllama_decoder_layer_no_qk_norm():
    cfg = _config_1p1b()
    blk = layer.build_tinyllama_decoder_layer(cfg, layer_idx=0, max_seq=64)
    assert blk.attention.q_norm is None
    assert blk.attention.k_norm is None


def test_tinyllama_decoder_layer_gqa_shape():
    """Documentation test: TinyLlama is GQA — n_kv (4) divides n_q (32)."""
    cfg = _config_1p1b()
    assert cfg.num_key_value_heads == 4
    assert cfg.num_attention_heads == 32
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
