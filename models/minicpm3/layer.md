# MiniCPM-3 (openbmb/MiniCPM3-4B) — layer.md

## 0. At-a-glance

- **Signature:** MLA (q_lora_rank=768, kv_lora_rank=256) + Llama-shape PRE-norm backbone + partial-rotary on qk_rope slice + μP residual scaling
- **Active params:** ~4B (MiniCPM3-4B, dense)
- **Layer mix:** 62 attn (dense, MLA throughout)
- **KV cache / token (bf16):** ~35 kB (62 layers × (256 + 32) × 2 B = 35712 B)

---

This document captures the decoder-layer architecture of MiniCPM-3 as
understood from the openbmb HF repo (`configuration_minicpm.py`,
`modeling_minicpm.py`, `config.json`). It serves as the source-of-truth for
the `models/minicpm3/` factory and the test-time numerical gate.

## 1. Architecture summary

- 62 decoder layers, hidden_size=2560, 40 attention heads.
- Multi-head Latent Attention (MLA), Llama-shaped envelope.
- LongRoPE with short==long factor at threshold == max_position_embeddings.
- PRE-norm RMSNorm STANDARD_W. SiLU SwiGLU FFN (intermediate_size=6400).
- μP-style per-layer residual scaling: `residual + sublayer * (scale_depth / sqrt(L))`
  with scale_depth=1.4, L=62.
- Input embedding scaled by `scale_emb=12`. lm_head output scaled by
  `dim_model_base/hidden_size = 256/2560 = 0.1` (HF divides by 10).

## 2. Block diagram

```
       x        ── residual ─────────────────────┐
       |                                          |
       input_layernorm (RMSNorm, eps=1e-5)        |
       |                                          v
       MLA self_attn  (see §3)                    +
       |                                          ^
       * scale_depth/sqrt(L)  (μP) ───────────────┘
       |
       x'       ── residual ─────────────────────┐
       |                                          |
       post_attention_layernorm                   |
       |                                          v
       SwiGLU MLP (gate_proj, up_proj, down_proj) +
       |                                          ^
       * scale_depth/sqrt(L) ─────────────────────┘
       |
       y
```

Source: `modeling_minicpm.py:928-948`.

## 3. MLA attention (the hardest piece)

Constants:
- H = 40 (num_attention_heads)
- q_lora_rank = 768, kv_lora_rank = 256
- qk_nope_head_dim = 64
- qk_rope_head_dim = 32
- qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 96
- v_head_dim = hidden_size / num_attention_heads = 2560 / 40 = 64
  (NOT in public config; derived in modeling_minicpm.py:354)

Layers:

```
q_a_proj          : Linear(2560, 768)
q_a_layernorm     : RMSNorm(768, eps=1e-5)
q_b_proj          : Linear(768, 40 * 96 = 3840)
kv_a_proj_with_mqa: Linear(2560, 256 + 32 = 288)
kv_a_layernorm    : RMSNorm(256, eps=1e-5)
kv_b_proj         : Linear(256, 40 * (64 + 64) = 5120)
o_proj            : Linear(40 * 64 = 2560, 2560)
rotary_emb        : RoPE(dim=32) at LongRoPE
```

Forward (per modeling_minicpm.py:427-526):

```
# Q path
q = q_b_proj( q_a_layernorm( q_a_proj(x) ) )           # [B, S, H * qk_h]
q = q.view(B, S, H, qk_h)
q_nope, q_pe = q[..., :64], q[..., 64:]                # split nope / rope

# KV path
comp = kv_a_proj_with_mqa(x)                            # [B, S, 288]
c_kv, k_pe = comp.split([256, 32], dim=-1)
kv = kv_b_proj( kv_a_layernorm(c_kv) ).view(B, S, H, 128)
k_nope, v = kv.split([64, 64], dim=-1)                  # nope / v

# RoPE on q_pe (per-head) and k_pe (single head, broadcasted later)
q_pe, k_pe = rope( q_pe, k_pe.view(B, S, 1, 32), position_ids )

# Q/K assembly at qk_head_dim, V at v_head_dim
q_full = cat([q_nope, q_pe], -1)        # [B, S, H, 96]
k_full = cat([k_nope, k_pe.expand(..., H, ...)], -1)
v_full = v                              # [B, S, H, 64]

# Cache (HF reference): store DECOMPRESSED K (qk_h=96) and V (v_h=64)
cache.write(k_full.transpose(1,2), v_full.transpose(1,2), start_pos=...)

# Softmax scale = qk_head_dim ** -0.5 = 96 ** -0.5
attn_out = sdpa(q_full, k_cached, v_cached,
                attn_mask=causal_mask, scale=qk_h ** -0.5)

# attn_out: [B, H, S, v_h=64] → [B, S, H * v_h = 2560]
return o_proj(attn_out)
```

## 4. RoPE detail

MiniCPM-3 uses `MiniCPMLongRoPE(dim=qk_rope_head_dim=32)`. The cos/sin tables
are computed at FULL `qk_rope_head_dim` (the rope subspace) — there is NO
partial-rotary semantic here at the head level. `rotate_half` pairs i ↔ i+16
across the rope subspace only.

The short and long factor vectors are length 16 (= qk_rope_head_dim/2).
`inv_freq` is computed as `1 / (ext_factor * base ** (2i / qk_rope_head_dim))`.
Selection of short vs long is based on
`seq_len > original_max_position_embeddings` (strict >). For MiniCPM-3-4B
these are both 32768, so the SHORT table is selected for any
seq_len ≤ 32768.

`attention_factor = sqrt(1 + log(scale)/log(orig_max))` where
`scale = max_position_embeddings / original_max_position_embeddings`. With
scale==1 this is 1.0.

In api/rope: spec uses `head_dim=qk_rope_head_dim` (i.e. RoPE module is
built with `head_dim=32`), `partial_rotary_factor=1.0`, scaling=LONGROPE.

## 5. μP scalars

| Scalar             | HF source                                | Where applied in our IR                                               |
|--------------------|------------------------------------------|-----------------------------------------------------------------------|
| scale_emb (=12)    | `modeling_minicpm.py:1163`               | embedding multiplies (model-level)                                    |
| scale_depth (=1.4) | `modeling_minicpm.py:941,948`            | DecoderBlockSpec.residual_scale = scale_depth / sqrt(L)               |
| dim_model_base/h   | `modeling_minicpm.py:1262-ish` (lm_head) | DecoderBlockSpec.logits_scale = dim_model_base / hidden_size = 0.1    |

## 6. Mask kind

CAUSAL — MLA uses standard left-triangular causal masking (no SWA).
`modeling_minicpm.py:485-505` builds the standard `(bsz, 1, q_len, kv_seq_len)`
causal mask.

## 7. KV cache (reference path)

Stores DECOMPRESSED K and V (HF reference). Asymmetric:
- K: `[B, H, S, qk_head_dim=96]`
- V: `[B, H, S, v_head_dim=64]`

The production "absorb kv_b_proj into o_proj and store compressed c_kv + k_pe"
path is deferred to a future milestone — at that point the cache becomes:
- c_kv: `[B, S, kv_lora_rank=256]` (per token; 1 head — MQA-style)
- k_pe: `[B, S, qk_rope_head_dim=32]`
giving a per-token cache footprint of 256+32=288 vs the reference 96+64=160 × H=40 = 6400.
The reference (decompressed) path here costs ~40× more KV memory.

## 8. Weight name mapping

```
model.layers.{L}.input_layernorm.weight              -> blk.pre_attn_norm.weight
model.layers.{L}.post_attention_layernorm.weight     -> blk.pre_ffn_norm.weight

model.layers.{L}.self_attn.q_a_proj.weight           -> blk.attention.q_a_proj.weight
model.layers.{L}.self_attn.q_a_layernorm.weight      -> blk.attention.q_a_layernorm.weight
model.layers.{L}.self_attn.q_b_proj.weight           -> blk.attention.q_b_proj.weight
model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight -> blk.attention.kv_a_proj_with_mqa.weight
model.layers.{L}.self_attn.kv_a_layernorm.weight     -> blk.attention.kv_a_layernorm.weight
model.layers.{L}.self_attn.kv_b_proj.weight          -> blk.attention.kv_b_proj.weight
model.layers.{L}.self_attn.o_proj.weight             -> blk.attention.o_proj.weight

model.layers.{L}.mlp.gate_proj.weight                -> blk.feedforward.gate_proj.weight
model.layers.{L}.mlp.up_proj.weight                  -> blk.feedforward.up_proj.weight
model.layers.{L}.mlp.down_proj.weight                -> blk.feedforward.down_proj.weight
```
