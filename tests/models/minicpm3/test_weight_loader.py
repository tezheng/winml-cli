"""Weight loader unit test: synthetic state-dict in HF naming convention
loads cleanly into the api block."""
import torch

from api import kvcache, specs, types
from models.minicpm3 import config as m_config, layer as m_layer


def _tiny_cfg():
    return m_config.MiniCPM3Config(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=4,
        intermediate_size=256,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        vocab_size=100,
        max_position_embeddings=32,
        original_max_position_embeddings=32,
        tie_word_embeddings=False,
        dtype=torch.float32,
        scale_emb=1.0, scale_depth=1.4, dim_model_base=32,
        rope_type="default",
        attention_bias=False,
    )


def test_load_synthetic_hf_state_dict():
    cfg = _tiny_cfg()
    blk = m_layer.build_minicpm3_decoder_layer(cfg, layer_idx=0, max_seq=16)
    attn = blk.attention
    H = cfg.num_attention_heads
    qk_h = cfg.qk_head_dim
    v_h = cfg.v_head_dim
    sd = {
        "model.layers.0.input_layernorm.weight": torch.randn(cfg.hidden_size),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(cfg.hidden_size),
        "model.layers.0.self_attn.q_a_proj.weight": torch.randn(cfg.q_lora_rank, cfg.hidden_size),
        "model.layers.0.self_attn.q_a_layernorm.weight": torch.randn(cfg.q_lora_rank),
        "model.layers.0.self_attn.q_b_proj.weight": torch.randn(H * qk_h, cfg.q_lora_rank),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight":
            torch.randn(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
        "model.layers.0.self_attn.kv_a_layernorm.weight": torch.randn(cfg.kv_lora_rank),
        "model.layers.0.self_attn.kv_b_proj.weight":
            torch.randn(H * (cfg.qk_nope_head_dim + v_h), cfg.kv_lora_rank),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(cfg.hidden_size, H * v_h),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(cfg.intermediate_size, cfg.hidden_size),
        "model.layers.0.mlp.up_proj.weight": torch.randn(cfg.intermediate_size, cfg.hidden_size),
        "model.layers.0.mlp.down_proj.weight": torch.randn(cfg.hidden_size, cfg.intermediate_size),
    }
    m_layer.load_hf_minicpm3_layer(blk, sd, layer_idx=0)
    # Spot-check a couple weights actually got copied (not still default ones).
    assert torch.allclose(attn.q_a_proj.weight, sd["model.layers.0.self_attn.q_a_proj.weight"])
    assert torch.allclose(attn.q_a_layernorm.weight, sd["model.layers.0.self_attn.q_a_layernorm.weight"])
    assert torch.allclose(attn.kv_b_proj.weight, sd["model.layers.0.self_attn.kv_b_proj.weight"])


def test_load_then_forward():
    cfg = _tiny_cfg()
    blk = m_layer.build_minicpm3_decoder_layer(cfg, layer_idx=0, max_seq=8)
    H = cfg.num_attention_heads
    qk_h = cfg.qk_head_dim
    v_h = cfg.v_head_dim
    sd = {
        "model.layers.0.input_layernorm.weight": torch.ones(cfg.hidden_size),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(cfg.hidden_size),
        "model.layers.0.self_attn.q_a_proj.weight": torch.zeros(cfg.q_lora_rank, cfg.hidden_size),
        "model.layers.0.self_attn.q_a_layernorm.weight": torch.ones(cfg.q_lora_rank),
        "model.layers.0.self_attn.q_b_proj.weight": torch.zeros(H * qk_h, cfg.q_lora_rank),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight":
            torch.zeros(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
        "model.layers.0.self_attn.kv_a_layernorm.weight": torch.ones(cfg.kv_lora_rank),
        "model.layers.0.self_attn.kv_b_proj.weight":
            torch.zeros(H * (cfg.qk_nope_head_dim + v_h), cfg.kv_lora_rank),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(cfg.hidden_size, H * v_h),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(cfg.intermediate_size, cfg.hidden_size),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(cfg.intermediate_size, cfg.hidden_size),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(cfg.hidden_size, cfg.intermediate_size),
    }
    m_layer.load_hf_minicpm3_layer(blk, sd, layer_idx=0)
    blk.eval()
    x = torch.randn(1, 3, cfg.hidden_size)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_attention_heads,
        head_dim=cfg.qk_head_dim, max_seq=8,
        v_head_dim=cfg.v_head_dim,
    )
    with torch.no_grad():
        out = blk(x, position_ids=torch.arange(3), cache=cache, start_pos=0)
    # With all attn weights zero, attn_out == 0, so output is identity + 0 == x
    # (scaled by residual_scale=0 ... wait, residual_scale is scale_depth/sqrt(L)
    # multiplied on the attn_out + mlp_out, but the residual itself is unscaled.
    # mlp_out also goes through silu(zero gate)*zero up = zero. So out == x.
    assert torch.allclose(out, x, atol=1e-6)
