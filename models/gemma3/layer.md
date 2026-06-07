# Gemma 3 Reference Layer

This document describes the architectural primitives of the Gemma 3 decoder
layer as implemented by `models.gemma3.{config,layer}`, mapped to the IR types
in `api/`.

Verified against `transformers/models/gemma3/modeling_gemma3.py` and
`configuration_gemma3.py` (transformers 5.10.x). All line citations in this
doc refer to those two files unless otherwise stated.

## 1. Block structure (dual sandwich norm)

Identical to Gemma 2: PRE_AND_POST on both sublayers, with all four norm
modules allocated. Source: modeling_gemma3.py:385-396 (init) + L398-428
(forward).

```
residual = x; x = input_layernorm(x); x, _ = attn(x); x = post_attention_layernorm(x); x = residual + x
residual = x; x = pre_feedforward_layernorm(x); x = mlp(x); x = post_feedforward_layernorm(x); x = residual + x
```

IR mapping: `NormPosition.PRE_AND_POST`.

## 2. RMSNorm (ONE_PLUS_W)

Same as Gemma 2: `Gemma3RMSNorm.weight` initialised to zeros; forward
computes `output * (1 + weight.float())` and casts to input dtype.
Source: modeling_gemma3.py:128-145.

IR mapping: `NormWeightMode.ONE_PLUS_W`.

## 3. QK normalization (PER_HEAD_DH, PRE_ROPE)

UNLIKE Gemma 2 (which has no QK-norm) but LIKE Gemma 4: q_norm and k_norm
are `Gemma3RMSNorm(dim=head_dim, eps=rms_norm_eps)` — ONE_PLUS_W mode.
Weight shape is `[head_dim]` (PER_HEAD_DH), applied after the `view + transpose`
reshape to `[B, H, S, Dh]`. Source: modeling_gemma3.py:337-356.

```python
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
query_states = self.q_norm(query_states)        # [B, H, S, Dh]
key_states   = self.k_norm(key_states)          # [B, Hk, S, Dh]
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

NO fixed-scale absorb (unlike Gemma 4). The attention scale is plain
`query_pre_attn_scalar**-0.5`. Source: L317.

IR mapping:
- `AttentionSpec.qk_norm = NormSpec(RMS, eps, ONE_PLUS_W)`.
- `AttentionSpec.qk_norm_phase = PRE_ROPE`.
- `AttentionSpec.qk_norm_shape = PER_HEAD_DH`.
- `AttentionSpec.qk_norm_fixed_scale = None`.

## 4. 5:1 SWA:full alternation

Default rule from configuration_gemma3.py:109-115:

```python
self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)
self.layer_types = [
    "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
    for i in range(self.num_hidden_layers)
]
```

For `sliding_window_pattern=6` (Gemma 3 1B and 4B default): full layers at
i = 5, 11, 17, 23, ... — every 6th. So 5 sliding then 1 full repeated.

IR mapping: `Gemma3Config.layer_type(layer_idx)` returns the string;
`to_block_spec(layer_idx)` dispatches mask kind and RoPE theta.

## 5. Dual RoPE θ

Sliding layers use `rope_local_base_freq` (default 10_000). Full layers use
`rope_theta` (default 1_000_000). These appear in HF 5.x as a nested
`rope_parameters` dict; older configs use the flat fields.

Source: configuration_gemma3.py:127-150 (RoPE param standardization) +
modeling_gemma3.py:148-225 (per-layer-type Gemma3RotaryEmbedding).

Both layer types use FULL head_dim rotation — Gemma 3 does NOT use partial
RoPE. Source: modeling_gemma3.py:197-206 — the inv_freq denominator is
`dim = head_dim`, no partial_rotary_factor scaling.

IR mapping: `RoPESpec(base_theta=rope_theta_local|global, basis=SPLIT_HALF,
scaling=NONE, partial_rotary_factor=1.0)`.

## 6. Attention

- GQA: `num_attention_heads=4`, `num_key_value_heads=1` (1B). Heavy GQA.
- No biases. `attention_bias=False`.
- `attn_scale = query_pre_attn_scalar**-0.5`. For 1B query_pre_attn_scalar=256.
- `attn_logit_softcap = None` (Gemma 3 DROPPED both softcaps). Source:
  configuration_gemma3.py:99-100 — both default to None and 1B/4B configs
  do not override.
- `sliding_window = 512` for 1B (much smaller than Gemma 2's 4096).

IR mapping: `AttentionSpec.attn_scale = query_pre_attn_scalar**-0.5`,
`logit_softcap=None`, `mask_kind=SWA|CAUSAL`, `sliding_window=512|None`.

## 7. FFN (GeGLU)

Same as Gemma 2: `gate_proj`, `up_proj`, `down_proj` with
`gelu_pytorch_tanh`. Source: modeling_gemma3.py:112-125.

## 8. Weight-name mapping

| HF tensor name (under `model.layers.{L}.`) | API tensor slot |
|---|---|
| `input_layernorm.weight`                | `blk.pre_attn_norm.weight` |
| `post_attention_layernorm.weight`       | `blk.post_attn_sublayer_norm.weight` |
| `pre_feedforward_layernorm.weight`      | `blk.pre_ffn_norm.weight` |
| `post_feedforward_layernorm.weight`     | `blk.post_ffn_sublayer_norm.weight` |
| `self_attn.q_proj.weight`               | `blk.attention.q_proj.weight` |
| `self_attn.k_proj.weight`               | `blk.attention.k_proj.weight` |
| `self_attn.v_proj.weight`               | `blk.attention.v_proj.weight` |
| `self_attn.o_proj.weight`               | `blk.attention.o_proj.weight` |
| `self_attn.q_norm.weight`               | `blk.attention.q_norm.weight` (PER_HEAD_DH `[Dh]`) |
| `self_attn.k_norm.weight`               | `blk.attention.k_norm.weight` (PER_HEAD_DH `[Dh]`) |
| `mlp.gate_proj.weight`                  | `blk.feedforward.gate_proj.weight` |
| `mlp.up_proj.weight`                    | `blk.feedforward.up_proj.weight` |
| `mlp.down_proj.weight`                  | `blk.feedforward.down_proj.weight` |

Multimodal Gemma 3 (e.g. `google/gemma-3-4b-it`) stores the text-model state
under `model.language_model.*`; callers should strip the prefix before
passing the state dict to the loader. (Same convention as Gemma 4.)

## 9. Known good models (gate matrix)

| Model | Layers | Pattern | First-full-layer | atol gate |
|---|---|---|---|---|
| `unsloth/gemma-3-1b-it` | 26 | 6 | 5 | 5e-4 |
| `google/gemma-3-4b-pt` (gated) | 34 | 6 | 5 | (deferred) |
