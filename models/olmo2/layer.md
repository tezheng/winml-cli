# OLMo 2 Reference Layer

This document describes the architectural primitives of the OLMo 2 decoder
layer as implemented by `models.olmo2.{config,layer}`, mapped to the IR types
in `api/{specs,types,block,attention,norm,rope,feedforward}.py`.

Verified against `transformers/models/olmo2/modeling_olmo2.py` (transformers
5.10.x). All line citations refer to that file unless otherwise stated.

## 1. Block structure (POST-norm)

OLMo 2 is the first model in `llm-layers` to use POST-norm. The decoder
layer has only one norm per sublayer, applied to the sublayer output BEFORE
the residual add:

```
residual = x
h, _ = self_attn(x)            # NO pre-norm on x
h = post_attention_layernorm(h)
x = residual + h               # POST-norm wraps attention output

residual = x
h = mlp(x)                     # NO pre-norm on x
h = post_feedforward_layernorm(h)
x = residual + h               # POST-norm wraps FFN output
```

Source: `Olmo2DecoderLayer.forward` L315-326. The decoder also lacks the
``input_layernorm`` and ``pre_feedforward_layernorm`` modules entirely
(verify via `Olmo2DecoderLayer.__init__` L295-303).

IR mapping: `NormPosition.POST` on both `attn_norm_position` and
`ffn_norm_position`. The block builder in `api/block.py` skips allocation
of `pre_attn_norm` / `pre_ffn_norm` when this is set and feeds the raw
residual to the sublayer.

## 2. QK normalization

OLMo 2 adds RMSNorm on the Q and K projections BEFORE RoPE, using FULL_HDH
shape — the weight tensor is sized `[num_attention_heads * head_dim]` for
Q and `[num_key_value_heads * head_dim]` for K. The norm is applied to the
flattened `[B, S, H*Dh]` tensor (which equivalently can be applied after the
`view(B, S, H, Dh)` reshape, since RMSNorm over the last dim of shape
`[H*Dh]` is mathematically identical whether the tensor was reshaped first or
not).

Source: `Olmo2Attention.__init__` L231-232 declares the norm modules; forward
L245-249 applies them:

```python
query_states = self.q_norm(self.q_proj(hidden_states))   # [B, S, Hq*Dh]
key_states   = self.k_norm(self.k_proj(hidden_states))   # [B, S, Hk*Dh]
query_states = query_states.view(hidden_shape).transpose(1, 2)
key_states   = key_states.view(hidden_shape).transpose(1, 2)
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

IR mapping: `AttentionSpec.qk_norm_shape = QKNormShape.FULL_HDH`,
`qk_norm_phase = QKNormPhase.PRE_ROPE`. The api.norm.QKNorm constructor
allocates a weight tensor of size `n_heads * head_dim` in this mode.

## 3. Attention

- GQA: `num_key_value_heads` can be less than `num_attention_heads`.
  For OLMo-2-1B both are 16 (true MHA); for OLMo-2-7B both are 32 (MHA).
- `attention_bias = False` on all public 1B/7B configs (no biases on q/k/v/o).
- Scale = `head_dim ** -0.5` (no fixed-scale absorb). Source: L215.
- Vanilla causal mask.

IR mapping: standard `AttentionKind.STANDARD`, `MaskKind.CAUSAL`, no
`attn_scale` override.

## 4. RoPE

- Basis: SPLIT_HALF (GPT-NeoX / Llama style). `rotate_half` at L198-202
  matches our `ops._rotate_half`.
- `rope_theta` from config (1B and 7B both use 500_000).
- No partial rotation — the full head_dim rotates.
- No scaling variant active in 1B / 7B (rope_type = "default", L83).

IR mapping: `RoPESpec(base_theta=..., basis=SPLIT_HALF, scaling=NONE,
partial_rotary_factor=1.0)`.

## 5. FFN

SwiGLU with `gate_proj`, `up_proj`, `down_proj`, all without biases. Activation
is `silu` (config `hidden_act="silu"`). Source: `Olmo2MLP` L279-292.

IR mapping: `FFNSpec(activation=SILU, gate_kind=SWIGLU, fused_gate_up=False)`.

## 6. Precision

OLMo 2 was trained with a BF16 residual stream (per the OLMo 2 paper); the
public weights are FP32 for 1B and BF16 for 7B (verify each model card).
Our numerical gate runs in FP32 to use the strictest atol=5e-4 contract;
runtime can configure `Olmo2Config.dtype` to BF16 or FP16 if desired.

The numerical gate `tests/models/olmo2/test_numerical_hf.py` loads
`allenai/OLMo-2-0425-1B` in FP32 (matching the FP32 checkpoint dtype) and
asserts allclose at atol=rtol=5e-4 for a layer-0 forward.

## 7. Weight-name mapping

| HF tensor name (under `model.layers.{L}.`) | API tensor slot |
|---|---|
| `self_attn.q_proj.weight`                | `blk.attention.q_proj.weight` |
| `self_attn.k_proj.weight`                | `blk.attention.k_proj.weight` |
| `self_attn.v_proj.weight`                | `blk.attention.v_proj.weight` |
| `self_attn.o_proj.weight`                | `blk.attention.o_proj.weight` |
| `self_attn.q_norm.weight`                | `blk.attention.q_norm.weight` (FULL_HDH `[Hq*Dh]`) |
| `self_attn.k_norm.weight`                | `blk.attention.k_norm.weight` (FULL_HDH `[Hk*Dh]`) |
| `post_attention_layernorm.weight`        | `blk.post_attn_sublayer_norm.weight` |
| `post_feedforward_layernorm.weight`      | `blk.post_ffn_sublayer_norm.weight` |
| `mlp.gate_proj.weight`                   | `blk.feedforward.gate_proj.weight` |
| `mlp.up_proj.weight`                     | `blk.feedforward.up_proj.weight` |
| `mlp.down_proj.weight`                   | `blk.feedforward.down_proj.weight` |

Note absence of `input_layernorm` and `pre_feedforward_layernorm` — OLMo 2
is POST-norm only.

## 8. Known good models (gate matrix)

| Model | Size | dtype | atol gate |
|---|---|---|---|
| `allenai/OLMo-2-0425-1B` | 1B | FP32 | 5e-4 (the chosen gate) |
| `allenai/OLMo-2-1124-7B` | 7B | BF16 | (not run — 13 GB download) |
