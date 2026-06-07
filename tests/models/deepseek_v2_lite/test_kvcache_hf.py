"""KV-cache consistency vs HF on V2-Lite — prefill + 1-step decode."""
import torch

from api import kvcache, specs, types
from models.deepseek_v2_lite import config as m_config, layer as m_layer


ATOL = 5e-4


def _make_pair(layer_idx: int):
    from transformers.models.deepseek_v2 import modeling_deepseek_v2 as mod
    from transformers.models.deepseek_v2 import configuration_deepseek_v2 as cfg_mod
    torch.manual_seed(2 + layer_idx)
    hf_cfg = cfg_mod.DeepseekV2Config(
        vocab_size=100, hidden_size=48, intermediate_size=64,
        moe_intermediate_size=16, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, n_group=None, topk_group=None,
        topk_method="greedy", norm_topk_prob=False,
        first_k_dense_replace=1,
        kv_lora_rank=12, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=12,
        q_lora_rank=None,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        routed_scaling_factor=1.0,
        tie_word_embeddings=False, attention_bias=False, attention_dropout=0.0,
    )
    hf_cfg._attn_implementation = "eager"

    hf_layer = mod.DeepseekV2DecoderLayer(hf_cfg, layer_idx=layer_idx)
    hf_layer.eval()
    rotary = mod.DeepseekV2RotaryEmbedding(hf_cfg)
    rotary.eval()

    api_cfg = m_config.DeepSeekV2LiteConfig(
        hidden_size=48, num_attention_heads=4, num_hidden_layers=3,
        intermediate_size=64, moe_intermediate_size=16,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        routed_scaling_factor=1.0, norm_topk_prob=False,
        n_group=1, topk_group=1, first_k_dense_replace=1,
        kv_lora_rank=12, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=12,
        q_lora_rank=None, rope_theta=10000.0, rms_norm_eps=1e-6,
        vocab_size=100, max_position_embeddings=32,
        tie_word_embeddings=False, attention_bias=False,
        dtype=torch.float32, rope_type="default",
    )
    api_blk = m_layer.build_deepseek_v2_lite_decoder_layer(
        api_cfg, layer_idx=layer_idx, max_seq=16,
    )
    api_blk.eval()
    # Copy weights.
    is_moe = layer_idx >= 1
    sd = {}
    L = layer_idx
    sd[f"model.layers.{L}.input_layernorm.weight"] = hf_layer.input_layernorm.weight.data
    sd[f"model.layers.{L}.post_attention_layernorm.weight"] = hf_layer.post_attention_layernorm.weight.data
    sa = hf_layer.self_attn
    sd[f"model.layers.{L}.self_attn.q_proj.weight"]            = sa.q_proj.weight.data
    sd[f"model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight"] = sa.kv_a_proj_with_mqa.weight.data
    sd[f"model.layers.{L}.self_attn.kv_a_layernorm.weight"]    = sa.kv_a_layernorm.weight.data
    sd[f"model.layers.{L}.self_attn.kv_b_proj.weight"]          = sa.kv_b_proj.weight.data
    sd[f"model.layers.{L}.self_attn.o_proj.weight"]             = sa.o_proj.weight.data
    if is_moe:
        moe = hf_layer.mlp
        sd[f"model.layers.{L}.mlp.gate.weight"] = moe.gate.weight.data
        sd[f"model.layers.{L}.mlp.experts.gate_up_proj"] = moe.experts.gate_up_proj.data
        sd[f"model.layers.{L}.mlp.experts.down_proj"]   = moe.experts.down_proj.data
        sd[f"model.layers.{L}.mlp.shared_experts.gate_proj.weight"] = moe.shared_experts.gate_proj.weight.data
        sd[f"model.layers.{L}.mlp.shared_experts.up_proj.weight"]   = moe.shared_experts.up_proj.weight.data
        sd[f"model.layers.{L}.mlp.shared_experts.down_proj.weight"] = moe.shared_experts.down_proj.weight.data
    else:
        sd[f"model.layers.{L}.mlp.gate_proj.weight"] = hf_layer.mlp.gate_proj.weight.data
        sd[f"model.layers.{L}.mlp.up_proj.weight"]   = hf_layer.mlp.up_proj.weight.data
        sd[f"model.layers.{L}.mlp.down_proj.weight"] = hf_layer.mlp.down_proj.weight.data
    m_layer.load_hf_deepseek_v2_lite_layer(api_blk, sd, layer_idx=L, cfg=api_cfg)
    return hf_cfg, hf_layer, rotary, api_blk, api_cfg


def test_v2_lite_layer1_prefill_then_step_matches_hf():
    """B5: prefill S, then step 1 token. The 1-step api output (using KV
    cache) must match the HF layer when run on the full S+1 sequence."""
    hf_cfg, hf_layer, rotary, api_blk, api_cfg = _make_pair(layer_idx=1)
    B = 1
    S_pref = 4
    S_full = 5
    x_full = torch.randn(B, S_full, hf_cfg.hidden_size)
    pos_full = torch.arange(S_full).unsqueeze(0)
    attn_mask = torch.full((B, 1, S_full, S_full), float("-inf"))
    attn_mask = torch.triu(attn_mask, diagonal=1)
    with torch.no_grad():
        freqs_cis = rotary(x_full, position_ids=pos_full)
        hf_out_full = hf_layer(
            x_full, attention_mask=attn_mask, position_ids=pos_full,
            past_key_values=None, use_cache=False,
            position_embeddings=freqs_cis,
        )
        if isinstance(hf_out_full, tuple):
            hf_out_full = hf_out_full[0]

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=api_cfg.num_attention_heads,
        head_dim=api_cfg.qk_head_dim, max_seq=16,
        v_head_dim=api_cfg.v_head_dim,
    )
    with torch.no_grad():
        x_pref = x_full[:, :S_pref, :]
        api_pref = api_blk(x_pref, position_ids=torch.arange(S_pref),
                           cache=cache, start_pos=0)
        x_step = x_full[:, S_pref:S_full, :]
        api_step = api_blk(x_step,
                           position_ids=torch.arange(S_pref, S_full),
                           cache=cache, start_pos=S_pref)
    # Compare prefill region.
    diff_pref = (hf_out_full[:, :S_pref] - api_pref).abs().max().item()
    # Compare step region.
    diff_step = (hf_out_full[:, S_pref:] - api_step).abs().max().item()
    print(f"V2-Lite MoE prefill diff={diff_pref:.3e}, step diff={diff_step:.3e}")
    assert diff_pref < ATOL
    assert diff_step < ATOL
