"""Llama 4 Scout attention-layer factory.

In B4 scope we ship the per-layer ATTENTION only — the channel mixer in
Scout 16E is MoE at every layer (``interleave_moe_layer_step=1``,
``moe_layers = list(range(48))``) and full MoE wiring lands in B6. The
factory therefore returns an ``api.attention.Attention`` module rather than
a ``DecoderBlock``. The iRoPE per-layer dispatch lives in
``Llama4ScoutConfig.to_attention_spec(layer_idx)`` — the same lever that
SmolLM3's NoPE dispatch uses (``rope=None`` on NoPE layers).

Source: modeling_llama4.py:321-410 (Llama4TextAttention) — verifies that
on NoPE layers (``self.use_rope = False``) both the ``apply_rotary_emb``
call (L369-372) and the ``qk_norm`` (L351, gated on ``use_rope``) are
skipped, leaving the attention math identical to a plain GQA causal block
with no QK-norm and no RoPE.
"""
from __future__ import annotations

import torch

from api import attention as _attention
from models.llama4_scout import config as _config


def build_llama4_scout_attention(
    cfg: _config.Llama4ScoutConfig,
    layer_idx: int,
    max_seq: int | None = None,
) -> _attention.Attention:
    """Instantiate the Scout per-layer Attention module.

    Per-layer dispatch:
    - RoPE layer (``no_rope_layers[i] == 1``): carries an INTERLEAVED-basis
      RoPE with LLAMA3 scaling.
    - NoPE layer (``no_rope_layers[i] == 0``): ``spec.rope is None``;
      attention.forward skips ``rope_apply`` (verified by the SmolLM3
      precedent at ``api/attention.py:323``).

    Raises ``NotImplementedError`` on layers whose ``layer_types[i] ==
    "chunked_attention"`` — chunked masks land in a later milestone.
    """
    spec = cfg.to_attention_spec(layer_idx=layer_idx)
    seq = max_seq if max_seq is not None else cfg.max_position_embeddings
    return _attention.Attention(
        spec=spec, hidden_size=cfg.hidden_size, max_seq=seq, dtype=cfg.dtype,
    )


def load_hf_llama4_scout_attention(
    attn: _attention.Attention,
    hf_state_dict: dict,
    layer_idx: int,
    cfg: _config.Llama4ScoutConfig,
) -> None:
    """Copy HF Llama 4 Scout per-layer ATTENTION weights into the api module.

    HF Llama 4 places the attention tensors under
    ``model.layers.{L}.self_attn.{q,k,v,o}_proj`` (and optionally
    ``self_attn.qk_norm`` on RoPE layers when ``use_qk_norm=True``). The
    ``language_model.`` prefix is added by Llama4ForConditionalGeneration;
    callers loading via ``Llama4ForCausalLM`` see no extra prefix.

    For the shape-only path (synthetic state dict) the loader is a thin
    name-mapping helper. The qk_norm tensor is NOT loaded because the IR
    does not currently materialize the L2Norm head-side QK-norm; that
    lands together with the numerical gate in a later milestone.

    Source: modeling_llama4.py:324-353 (Llama4TextAttention init), :337-350
    (the four projection matrices).
    """
    L = layer_idx
    prefix = f"model.layers.{L}.self_attn"
    mapping = {
        f"{prefix}.q_proj.weight": attn.q_proj.weight,
        f"{prefix}.k_proj.weight": attn.k_proj.weight,
        f"{prefix}.v_proj.weight": attn.v_proj.weight,
        f"{prefix}.o_proj.weight": attn.o_proj.weight,
    }
    if cfg.attention_bias:
        mapping[f"{prefix}.q_proj.bias"] = attn.q_proj.bias
        mapping[f"{prefix}.k_proj.bias"] = attn.k_proj.bias
        mapping[f"{prefix}.v_proj.bias"] = attn.v_proj.bias
        mapping[f"{prefix}.o_proj.bias"] = attn.o_proj.bias

    missing = [k for k in mapping if k not in hf_state_dict]
    if missing:
        raise KeyError(f"missing tensors in state dict: {missing}")

    with torch.no_grad():
        for hf_name, slot in mapping.items():
            src = hf_state_dict[hf_name]
            if src.shape != slot.shape:
                raise ValueError(
                    f"shape mismatch for {hf_name}: src {src.shape} vs slot {slot.shape}"
                )
            slot.copy_(src.to(slot.dtype))
