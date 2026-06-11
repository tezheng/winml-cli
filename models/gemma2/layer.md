# Gemma 2 Reference Layer

## 0. At-a-glance

- **Signature:** Sandwich norm (ONE_PLUS_W) + alternating 1:1 SWA/global + GeGLU + attn_logit_softcap=50 + final_logit_softcap=30; no QK-norm
- **Active params:** 2.6B / 9B total
- **Layer mix:** 26 attn (2B, 1:1 SWA/global) / 42 attn (9B, 1:1 SWA/global)
- **KV cache / token (bf16):** 2B: 26 L × 4 KV × 256 Dh × 2 × 2 = 104 kB; 9B: 42 L × 4 KV × 256 Dh × 2 × 2 = 168 kB

---

This document describes the architectural primitives of the Gemma 2 decoder
layer as implemented by `models.gemma2.{config,layer}`, mapped to the IR types
in `api/`.

Verified against `transformers/models/gemma2/modeling_gemma2.py` (transformers
5.10.x). All line citations refer to that file unless otherwise stated.

## 1. Block structure (dual sandwich norm)

Gemma 2 uses PRE_AND_POST norm on both the attention and FFN sublayers:

```
residual = x
x = input_layernorm(x)
x, _ = self_attn(x)
x = post_attention_layernorm(x)
x = residual + x

residual = x
x = pre_feedforward_layernorm(x)
x = mlp(x)
x = post_feedforward_layernorm(x)
x = residual + x
```

Source: `Gemma2DecoderLayer.forward` L324-344. The module init
(L302-313) allocates `input_layernorm`, `post_attention_layernorm`,
`pre_feedforward_layernorm`, `post_feedforward_layernorm`.

IR mapping: `NormPosition.PRE_AND_POST` on both `attn_norm_position` and
`ffn_norm_position`. Block builder allocates four `RMSNorm` modules.

## 2. RMSNorm (ONE_PLUS_W)

`Gemma2RMSNorm.weight` is initialised to ZEROS, and the forward computes
`output * (1.0 + weight.float()).type_as(x)`:

```python
def forward(self, x):
    output = self._norm(x.float())           # rms-normalize in fp32
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)
```

Source: L49-65. This is the `weight_mode=ONE_PLUS_W` branch in
`api.ops.rms_norm`, where the loaded weight is added to 1 inside the kernel.

## 3. Attention

- GQA with `num_attention_heads=8`, `num_key_value_heads=4` (Gemma 2 2B).
- No biases on q/k/v/o projections (`attention_bias=False` everywhere).
- **Scale**: `query_pre_attn_scalar ** -0.5` — overrides the default
  `head_dim ** -0.5`. For 2B both equal 256, but the IR carries the explicit
  scale via `AttentionSpec.attn_scale`. Source: L240.
- **Logit softcap**: `attn_logit_softcapping=50.0` from config. Applied AFTER
  matmul * scale and BEFORE the mask add:
  ```python
  attn = (q @ k.T) * scale
  attn = tanh(attn / softcap) * softcap
  attn = attn + mask
  attn = softmax(...)
  ```
  Source: L212-217 (eager_attention_forward). `Gemma2Attention.forward` calls
  the attention_interface with `softcap=self.attn_logit_softcapping`
  (L293).
- Sliding window: `sliding_window=4096` when the layer type is
  `"sliding_attention"`. Source: L257 (`self.sliding_window =
  config.sliding_window if self.layer_type == "sliding_attention" else None`).

IR mapping:
- `AttentionSpec.attn_scale = query_pre_attn_scalar ** -0.5`.
- `AttentionSpec.logit_softcap = 50.0`.
- `AttentionSpec.mask_kind = SWA` (sliding layers) or `CAUSAL` (full layers).
- `AttentionSpec.sliding_window = sliding_window` (sliding only) or `None`.

NOT present: QK-norm (Gemma 2 has no q_norm / k_norm). Verify at L233-258 —
no q_norm / k_norm members.

## 4. RoPE

Single rope_theta (default 10_000), full head_dim rotation, SPLIT_HALF basis.
Source: L85-147 (Gemma2RotaryEmbedding) and L150-180 (apply_rotary_pos_emb).

IR mapping: `RoPESpec(base_theta=rope_theta, basis=SPLIT_HALF, scaling=NONE)`.

## 5. Layer-type alternation

Gemma 2 alternates SWA and full attention every other layer. Default pattern
from `configuration_gemma2.py:95-98`:

```python
self.layer_types = [
    "sliding_attention" if bool((i + 1) % 2) else "full_attention"
    for i in range(self.num_hidden_layers)
]
```

So layer 0 → SWA, layer 1 → full, layer 2 → SWA, ... The 1:1 alternation.

IR mapping: `Gemma2Config.layer_type(layer_idx)` returns the string, and
`to_block_spec(layer_idx)` dispatches on it. `to_block_spec` returns a
per-layer `DecoderBlockSpec` — the per-layer object is the only place the
SWA/full distinction lives in the IR.

## 6. FFN (GeGLU)

`Gemma2MLP` has `gate_proj`, `up_proj`, `down_proj` and `act_fn = ACT2FN[
config.hidden_activation]` where `hidden_activation = "gelu_pytorch_tanh"`.
GeGLU = `down(act(gate) * up)` with `act = gelu_pytorch_tanh`.

Source: L69-82.

IR mapping: `FFNSpec(activation=GELU, gate_kind=GEGLU, fused_gate_up=False)`.

## 7. Weight-name mapping

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
| `mlp.gate_proj.weight`                  | `blk.feedforward.gate_proj.weight` |
| `mlp.up_proj.weight`                    | `blk.feedforward.up_proj.weight` |
| `mlp.down_proj.weight`                  | `blk.feedforward.down_proj.weight` |

Note: Gemma 2 weights are STORED in the HF state dict with the ZERO-CENTERED
initialisation (`init.zeros_(module.weight)`, modular_gemma2.py:_init_weights).
Trained weights are NOT zero; the `(1 + weight)` is part of the forward.

## 8. Model-level (NOT in the per-layer spec)

- Embedding scaling: `embed_scale = sqrt(hidden_size)`. The
  `DecoderBlockSpec.embedding_scale` field carries this for the assembly
  layer.
- `final_logit_softcapping=30.0`: applied to lm_head output. The
  `DecoderBlockSpec.final_logit_softcap` field carries this for the assembly
  layer.

The per-layer DecoderBlock does NOT apply these — they are model-level scalars
the assembly layer plumbs through `embed` and `lm_head`.

## 9. Known good models (gate matrix)

| Model | Size | dtype | atol gate |
|---|---|---|---|
| `unsloth/gemma-2-2b` (mirror of google/gemma-2-2b) | 2B | BF16 | 5e-4 |
