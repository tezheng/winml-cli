"""Shape-only test: build MiniCPM-3 decoder layer from synthetic config + verify
projection weight shapes vs the public config dims.

Does NOT load HF weights. Uses small tweaked dims (n_heads=8, hidden=512) so
the test runs in <1s.
"""
import torch

from api import kvcache, specs, types
from models.minicpm3 import config as m_config, layer as m_layer


def _tiny_minicpm3_hf_dict():
    return {
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "intermediate_size": 1024,
        "qk_nope_head_dim": 32,
        "qk_rope_head_dim": 16,            # rope/2 = 8 must match short/long factor lengths
        "q_lora_rank": 128,
        "kv_lora_rank": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "vocab_size": 1000,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "scale_emb": 2.0,
        "scale_depth": 1.4,
        "dim_model_base": 64,
    }


def test_build_layer_shapes_match():
    cfg = m_config.MiniCPM3Config.from_hf_dict(_tiny_minicpm3_hf_dict())
    blk = m_layer.build_minicpm3_decoder_layer(cfg, layer_idx=0, max_seq=32)

    H = cfg.num_attention_heads
    qk_h = cfg.qk_head_dim
    v_h = cfg.v_head_dim
    # Q LoRA path
    assert blk.attention.q_a_proj.weight.shape == (cfg.q_lora_rank, cfg.hidden_size)
    assert blk.attention.q_a_layernorm.weight.shape == (cfg.q_lora_rank,)
    assert blk.attention.q_b_proj.weight.shape == (H * qk_h, cfg.q_lora_rank)
    # KV LoRA path
    assert blk.attention.kv_a_proj_with_mqa.weight.shape == (
        cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size,
    )
    assert blk.attention.kv_a_layernorm.weight.shape == (cfg.kv_lora_rank,)
    assert blk.attention.kv_b_proj.weight.shape == (
        H * (cfg.qk_nope_head_dim + v_h), cfg.kv_lora_rank,
    )
    # Output
    assert blk.attention.o_proj.weight.shape == (cfg.hidden_size, H * v_h)
    # No standard q/k/v_proj
    assert blk.attention.q_proj is None
    assert blk.attention.k_proj is None
    assert blk.attention.v_proj is None


def test_forward_runs_with_synthetic_weights():
    """End-to-end forward (random weights) — shape + cache update."""
    cfg = m_config.MiniCPM3Config.from_hf_dict(_tiny_minicpm3_hf_dict())
    blk = m_layer.build_minicpm3_decoder_layer(cfg, layer_idx=0, max_seq=16)
    blk.eval()

    B, S = 1, 4
    x = torch.randn(B, S, cfg.hidden_size)

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=16,
        v_head_dim=cfg.v_head_dim,
    )
    with torch.no_grad():
        out = blk(x, position_ids=torch.arange(S), cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)
    assert cache.seq_len == S
