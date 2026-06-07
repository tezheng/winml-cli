"""DeepSeek-V2-Lite decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a DeepSeekV2LiteConfig using only api/ primitives.

Verified against `transformers/models/deepseek_v2/modeling_deepseek_v2.py`:
- 287-395  DeepseekV2Attention init + forward (MLA).
- 398-438  DeepseekV2DecoderLayer init + forward
           (input_layernorm → self_attn → residual; post_attention_layernorm
            → mlp → residual; no μP scaling).
- 85-130   DeepseekV2Moe.forward (router + experts + shared experts).
- 46-82    DeepseekV2Experts.forward (gate_up_proj 3D, down_proj 3D,
           index_add scatter).
- 404      Per-layer dispatch: layer_idx >= first_k_dense_replace → MoE
           else dense MLP.
"""
from __future__ import annotations
from typing import Optional

import torch

from api import block
from models.deepseek_v2_lite import config as _config


def build_deepseek_v2_lite_decoder_layer(
    cfg: _config.DeepSeekV2LiteConfig,
    layer_idx: int = 0,
    max_seq: Optional[int] = None,
) -> block.DecoderBlock:
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_deepseek_v2_lite_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    cfg: _config.DeepSeekV2LiteConfig,
) -> None:
    """Copy HF DeepSeek-V2 layer tensors into the api DecoderBlock.

    Norm + attention tensors are the same across dense / MoE layers; the
    `mlp.*` half differs.
    """
    L = layer_idx
    prefix = f"model.layers.{L}"
    attn = blk.attention
    mapping: dict[str, torch.Tensor] = {
        f"{prefix}.input_layernorm.weight":           blk.pre_attn_norm.weight,
        f"{prefix}.post_attention_layernorm.weight":  blk.pre_ffn_norm.weight,
        # MLA — KV LoRA path identical to MiniCPM-3.
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": attn.kv_a_proj_with_mqa.weight,
        f"{prefix}.self_attn.kv_a_layernorm.weight":   attn.kv_a_layernorm.weight,
        f"{prefix}.self_attn.kv_b_proj.weight":        attn.kv_b_proj.weight,
        f"{prefix}.self_attn.o_proj.weight":           attn.o_proj.weight,
    }
    # Q path varies:
    # - V2-Lite (q_lora_rank=null): direct q_proj.
    # - V2-7B-and-up: q_a_proj / q_a_layernorm / q_b_proj.
    if cfg.q_lora_rank is None:
        mapping[f"{prefix}.self_attn.q_proj.weight"] = attn.q_proj.weight
    else:
        mapping[f"{prefix}.self_attn.q_a_proj.weight"]      = attn.q_a_proj.weight
        mapping[f"{prefix}.self_attn.q_a_layernorm.weight"] = attn.q_a_layernorm.weight
        mapping[f"{prefix}.self_attn.q_b_proj.weight"]      = attn.q_b_proj.weight

    # Optional attention biases (config.attention_bias=False on V2-Lite, but
    # we plumb them in for V2-7B compatibility).
    if cfg.attention_bias:
        for hf_name, slot in (
            (f"{prefix}.self_attn.q_a_proj.bias", getattr(attn, "q_a_proj", None)
                 and attn.q_a_proj.bias),
            (f"{prefix}.self_attn.kv_a_proj_with_mqa.bias",
                attn.kv_a_proj_with_mqa.bias),
            (f"{prefix}.self_attn.o_proj.bias", attn.o_proj.bias),
        ):
            if slot is not None and hf_name in hf_state_dict:
                mapping[hf_name] = slot

    # Channel mixer half: dense vs MoE.
    if layer_idx >= cfg.first_k_dense_replace:
        # MoE path: blk.feedforward is api.feedforward.MoE.
        moe = blk.feedforward
        # Router gate: nn.Linear weight at [n_experts, hidden].
        mapping[f"{prefix}.mlp.gate.weight"] = moe.gate.weight
        # Expert packed tensors. HF stores them as
        # `model.layers.{L}.mlp.experts.gate_up_proj` shape
        # [E, 2*I, H], and `.down_proj` shape [E, H, I]. Source:
        # modeling_deepseek_v2.py:54-55. They match our (experts_gate_up,
        # experts_down) layout exactly.
        mapping[f"{prefix}.mlp.experts.gate_up_proj"] = moe.experts_gate_up
        mapping[f"{prefix}.mlp.experts.down_proj"]    = moe.experts_down
        # Shared experts: a single DeepseekV2MLP with intermediate_size =
        # moe_intermediate_size * n_shared_experts. Source:
        # modeling_deepseek_v2.py:91-93.
        if cfg.n_shared_experts > 0:
            mapping[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = (
                moe.shared_experts.gate_proj.weight)
            mapping[f"{prefix}.mlp.shared_experts.up_proj.weight"] = (
                moe.shared_experts.up_proj.weight)
            mapping[f"{prefix}.mlp.shared_experts.down_proj.weight"] = (
                moe.shared_experts.down_proj.weight)
    else:
        # Dense FFN path.
        mapping[f"{prefix}.mlp.gate_proj.weight"] = blk.feedforward.gate_proj.weight
        mapping[f"{prefix}.mlp.up_proj.weight"]   = blk.feedforward.up_proj.weight
        mapping[f"{prefix}.mlp.down_proj.weight"] = blk.feedforward.down_proj.weight

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing[:10]}")
    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
