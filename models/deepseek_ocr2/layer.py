"""DeepSeek-OCR-2 LM-decoder-layer factory + HF weight loader.

Builds a DecoderBlock from a DeepseekOcr2Config using only api/ primitives.
Does NOT import transformers.models.deepseek_ocr2.modeling_deepseek_ocr2
at runtime.

HF source: `transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py`:
- 1073-1138 DeepseekOcr2TextAttention (STANDARD MHA — no MLA).
- 1141-1154 DeepseekOcr2TextMLP (dense SwiGLU for layer 0).
- 1158-1194 DeepseekOcr2TextExperts (packed gate_up_proj [E, 2I, H] +
  down_proj [E, H, I]).
- 1197-1242 DeepseekOcr2TextMoe (softmax router + shared_experts).
- 1266-1309 DeepseekOcr2TextDecoderLayer (PRE-norm; per-layer
  dense vs MoE via `mlp_layer_types[layer_idx]`).
- 1339-1410 DeepseekOcr2TextModel (uses `create_causal_mask` — STANDARD
  causal masking, NO block-bidirectional / VCF).
"""
from __future__ import annotations

import torch

from api import block, specs
from models.deepseek_ocr2 import config as _config


def build_deepseek_ocr2_decoder_layer(
    cfg: _config.DeepseekOcr2Config,
    layer_idx: int = 0,
    max_seq: int | None = None,
) -> block.DecoderBlock:
    """Instantiate one DeepSeek-OCR-2 LM-decoder block at `layer_idx`."""
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return block.DecoderBlock(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_deepseek_ocr2_layer(
    blk: "block.DecoderBlock",
    hf_state_dict: dict,
    layer_idx: int,
    cfg: _config.DeepseekOcr2Config,
    prefix: str = "model.language_model.layers",
) -> None:
    """Copy HF DeepSeek-OCR-2 LM-decoder layer weights into the api DecoderBlock.

    HF layout (verified by sampling a DeepseekOcr2TextModel state_dict on
    a tiny synthesised config):
    - dense layer:
        `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.down_proj.weight`.
    - sparse (MoE) layer:
        `mlp.gate.weight`             [n_experts, hidden]
        `mlp.experts.gate_up_proj`    [n_experts, 2*moe_inter, hidden]
        `mlp.experts.down_proj`       [n_experts, hidden, moe_inter]
        `mlp.shared_experts.{gate,up,down}_proj.weight`  — bare Linear.

    The standalone-text-model prefix is `model.layers.{L}.*`; the
    full DeepseekOcr2Model wraps it as `model.language_model.layers.{L}.*`.
    """
    L = layer_idx
    pfx = f"{prefix}.{L}"
    mapping: dict[str, torch.Tensor] = {
        f"{pfx}.input_layernorm.weight":             blk.pre_attn_norm.weight,
        f"{pfx}.post_attention_layernorm.weight":    blk.pre_ffn_norm.weight,
        f"{pfx}.self_attn.q_proj.weight":            blk.attention.q_proj.weight,
        f"{pfx}.self_attn.k_proj.weight":            blk.attention.k_proj.weight,
        f"{pfx}.self_attn.v_proj.weight":            blk.attention.v_proj.weight,
        f"{pfx}.self_attn.o_proj.weight":            blk.attention.o_proj.weight,
    }
    if cfg.attention_bias:
        mapping[f"{pfx}.self_attn.q_proj.bias"] = blk.attention.q_proj.bias
        mapping[f"{pfx}.self_attn.k_proj.bias"] = blk.attention.k_proj.bias
        mapping[f"{pfx}.self_attn.v_proj.bias"] = blk.attention.v_proj.bias
        mapping[f"{pfx}.self_attn.o_proj.bias"] = blk.attention.o_proj.bias

    # Channel mixer.
    spec = cfg.to_block_spec(layer_idx=layer_idx)
    if isinstance(spec.channel_mixer, specs.MoESpec):
        moe = blk.feedforward
        # Router gate — softmax router has `gate.weight` as a plain Linear.
        mapping[f"{pfx}.mlp.gate.weight"] = moe.gate.weight
        mapping[f"{pfx}.mlp.experts.gate_up_proj"] = moe.experts_gate_up
        mapping[f"{pfx}.mlp.experts.down_proj"]   = moe.experts_down
        # Shared experts (n_shared_experts > 0).
        if moe.shared_experts is not None:
            mapping[f"{pfx}.mlp.shared_experts.gate_proj.weight"] = \
                moe.shared_experts.gate_proj.weight
            mapping[f"{pfx}.mlp.shared_experts.up_proj.weight"] = \
                moe.shared_experts.up_proj.weight
            mapping[f"{pfx}.mlp.shared_experts.down_proj.weight"] = \
                moe.shared_experts.down_proj.weight
    else:
        # Dense layer (layer 0 of canonical checkpoint).
        mapping[f"{pfx}.mlp.gate_proj.weight"] = blk.feedforward.gate_proj.weight
        mapping[f"{pfx}.mlp.up_proj.weight"]   = blk.feedforward.up_proj.weight
        mapping[f"{pfx}.mlp.down_proj.weight"] = blk.feedforward.down_proj.weight

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing[:8]}")

    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
