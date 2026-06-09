"""Jamba (ai21labs) family — Mamba-1 + attention alternation (v6 B2).

Architectural alternation:
- attention at layer indices where `i % attn_layer_period == attn_layer_offset`
  (default period=8, offset=4 → layers 4, 12, 20, ...).
- mamba (Mamba-1) at every other layer.

Mamba layers use Mamba-1 selective scan with Jamba's intra-mixer
LayerNorms on dt/B/C (modeling_jamba.py:249-251, 324-326).

Attention layers have NO rotary embedding — Jamba relies on the
recurrent Mamba layers for positional information. The HF
`apply_rotary_pos_emb` function is defined but NEVER CALLED in
JambaAttention.forward (modeling_jamba.py:165-199).

v6 B2 lands the DENSE-only Jamba path. The MoE branch (when
`config.layers_num_experts[i] > 1`) is DEFERRED — Mamba+MoE composition
needs a Jamba-specific JambaSparseMoeBlock router (no scaling,
hidden_states dtype). See `models/jamba/__init__.py` for follow-ups.

Source: `transformers/models/jamba/modeling_jamba.py`:
- 56-74 (JambaRMSNorm)
- 122-199 (JambaAttention — no rotary)
- 202-474 (JambaMambaMixer with dt/B/C layernorms)
- 477-490 (JambaMLP — standard SwiGLU)
- 533-567 (JambaSparseMoeBlock — DEFERRED in B2)
- 570-638 (JambaAttentionDecoderLayer / JambaMambaDecoderLayer)
"""
