"""DeepSeek-OCR-2 LM-decoder shape tests.

Exercises:
1. Dense layer (layer 0) assembly + forward shape.
2. MoE layer (layer 1) assembly + forward shape.
3. Block-bidirectional mask CONSTRUCTION via the v3-spec hook
   (`AttentionSpec.block_bidirectional_mask=True` + vision_token_count).
   This is the "Visual Causal Flow" mask path of the original DeepSeek-OCR
   paper; the HF reference does NOT use it, so this test is shape-only —
   we just verify the mask logic produces a forward of the correct shape
   without erroring.
"""
from __future__ import annotations

import torch

from api import kvcache, specs, types
from api import block as _block
from models.deepseek_ocr2 import config as ds_config, layer as ds_layer


_HF_DS_OCR2_TINY = {
    "model_type": "deepseek_ocr2",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "deepseek_ocr2_text",
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "attention_bias": False,
        "mlp_bias": False,
        "head_dim": 16,
        "n_group": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 2,
        "routed_scaling_factor": 1.0,
        "topk_group": 1,
        "topk_method": "greedy",
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "mlp_layer_types": ["dense", "sparse"],
    },
}


def _cfg():
    return ds_config.DeepseekOcr2Config.from_hf_dict(_HF_DS_OCR2_TINY)


def test_dense_layer_assembles_and_forwards():
    cfg = _cfg()
    blk = ds_layer.build_deepseek_ocr2_decoder_layer(cfg, layer_idx=0, max_seq=32)
    blk.eval()
    assert blk.attention.q_proj.bias is None         # attention_bias=False
    assert blk.feedforward.gate_proj.in_features == cfg.hidden_size
    assert blk.feedforward.gate_proj.out_features == cfg.intermediate_size

    B, S = 1, 7
    hidden = torch.randn(B, S, cfg.hidden_size)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=32,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        out = blk(hidden, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_moe_layer_assembles_and_forwards():
    cfg = _cfg()
    blk = ds_layer.build_deepseek_ocr2_decoder_layer(cfg, layer_idx=1, max_seq=32)
    blk.eval()
    moe = blk.feedforward
    assert moe.n_experts == 4
    assert moe.top_k == 2
    assert moe.shared_experts is not None
    # Shared expert FFN has intermediate = moe_intermediate * n_shared.
    assert moe.shared_experts.gate_proj.out_features == 32 * 2
    # Packed expert tensors at expected shapes.
    assert moe.experts_gate_up.shape == (4, 2 * 32, cfg.hidden_size)
    assert moe.experts_down.shape == (4, cfg.hidden_size, 32)

    B, S = 1, 7
    hidden = torch.randn(B, S, cfg.hidden_size)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=32,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        out = blk(hidden, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_block_bidirectional_mask_construction():
    """Shape-only exercise of MaskKind.BLOCK_BIDIRECTIONAL — the Visual
    Causal Flow mask from the original DeepSeek-OCR paper §3.2.

    Build an attention spec with `block_bidirectional_mask=True`, forward
    with a fictitious `vision_token_count`, and verify shape + that
    the mask path runs without error. The HF v5.10.2 deepseek_ocr2
    reference does NOT use this mask — this is a v3-spec hook validation.
    """
    cfg = _cfg()
    # Hand-modify spec to enable the VCF mask.
    base_spec = cfg.to_block_spec(layer_idx=0)
    attn = base_spec.token_mixer
    vcf_attn = specs.AttentionSpec(
        n_q_heads=attn.n_q_heads,
        n_kv_heads=attn.n_kv_heads,
        head_dim=attn.head_dim,
        kind=attn.kind,
        qkv_layout=attn.qkv_layout,
        mask_kind=types.MaskKind.BLOCK_BIDIRECTIONAL,
        block_bidirectional_mask=True,
        q_bias=attn.q_bias, k_bias=attn.k_bias, v_bias=attn.v_bias,
        o_bias=attn.o_bias,
        rope=attn.rope,
    )
    vcf_block_spec = specs.DecoderBlockSpec(
        attn_norm_position=base_spec.attn_norm_position,
        ffn_norm_position=base_spec.ffn_norm_position,
        token_mixer=vcf_attn,
        channel_mixer=base_spec.channel_mixer,
        pre_attn_norm=base_spec.pre_attn_norm,
        pre_ffn_norm=base_spec.pre_ffn_norm,
    )
    blk = _block.DecoderBlock(
        spec=vcf_block_spec, hidden_size=cfg.hidden_size, max_seq=32,
        dtype=cfg.dtype,
    )
    blk.eval()

    B, S = 1, 8
    vision_token_count = 4
    hidden = torch.randn(B, S, cfg.hidden_size)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq=32,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        out = blk(hidden, position_ids=pos, cache=cache, start_pos=0,
                  vision_token_count=vision_token_count)
    assert out.shape == (B, S, cfg.hidden_size)


def test_block_bidirectional_mask_actually_bidirectional_in_vision_block():
    """Algebraic check on the mask construction. Build a small attention
    forward with VCF mask, set V=3, and verify (via varying the text
    region while keeping vision tokens fixed) that the vision-block
    output is INVARIANT to text changes — proving vision queries do not
    attend to text keys."""
    cfg = _cfg()
    base_spec = cfg.to_block_spec(layer_idx=0)
    attn = base_spec.token_mixer
    vcf_attn = specs.AttentionSpec(
        n_q_heads=attn.n_q_heads,
        n_kv_heads=attn.n_kv_heads,
        head_dim=attn.head_dim,
        kind=attn.kind,
        qkv_layout=attn.qkv_layout,
        mask_kind=types.MaskKind.BLOCK_BIDIRECTIONAL,
        block_bidirectional_mask=True,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=attn.rope,
    )
    from api import attention as _attn
    a = _attn.Attention(vcf_attn, cfg.hidden_size, max_seq=32, dtype=cfg.dtype)
    a.eval()
    B = 1
    V = 3
    n_text = 4
    S = V + n_text
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )

    def _forward(x):
        cache = kvcache.ContiguousKVCache(
            cache_spec, batch_size=B,
            n_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            max_seq=32,
        )
        pos = torch.arange(S)
        with torch.no_grad():
            return a(x, position_ids=pos, cache=cache, start_pos=0,
                     vision_token_count=V)

    torch.manual_seed(0)
    base = torch.randn(B, S, cfg.hidden_size)
    out_a = _forward(base)
    # Mutate ONLY the text region (positions V..S-1) — vision-region
    # output (positions 0..V-1) must remain identical.
    mut = base.clone()
    mut[:, V:, :] = torch.randn(B, n_text, cfg.hidden_size)
    out_b = _forward(mut)
    diff_vision = (out_a[:, :V, :] - out_b[:, :V, :]).abs().max().item()
    diff_text = (out_a[:, V:, :] - out_b[:, V:, :]).abs().max().item()
    assert diff_vision < 1e-6, (
        f"Vision queries should not attend to text keys under VCF; "
        f"diff={diff_vision}"
    )
    assert diff_text > 0.0, (
        f"Text queries should see vision tokens too; diff={diff_text}"
    )
