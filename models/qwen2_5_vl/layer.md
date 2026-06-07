# Qwen2.5-VL — LM-Decoder Layer (with M-RoPE)

## 1. Identity

- **Family:** Qwen2.5-VL (Alibaba)
- **Variants in B8 scope:** Qwen/Qwen2.5-VL-3B-Instruct (LM-decoder portion)
- **Release date:** January 2025
- **Paper / tech report:** Qwen2.5-VL Technical Report
- **HF base model card:** https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- **transformers source:** `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`

## 2. Architecture summary

- **Vision encoder:** dynamic-resolution Vision Transformer (with 2D axial
  RoPE on H/W) — OUT OF SCOPE for B8.
- **Vision connector:** PatchMerger (4× spatial mean-pool + MLP).
- **LM decoder:** Qwen2.5 backbone with **M-RoPE** (multimodal rotary position
  embedding). The decoder is otherwise Qwen2-shaped: QKV biases TRUE, NO
  QK-norm, SwiGLU FFN no biases, PRE-norm RMSNorm.
- **Canonical 3B-Instruct dims:** hidden=2048, n_q=16, n_kv=2 (GQA 8:1),
  head_dim=128, intermediate=11008, num_layers=36, vocab=151936,
  rope_theta=1_000_000, mrope_section=[16, 24, 24], rms_norm_eps=1e-6,
  tie_word_embeddings=False.

## 3. M-RoPE — multimodal rotary position embedding

This is the novel piece. Standard RoPE consumes a 1D `position_ids` of
shape [B, S]. M-RoPE consumes a 3D `position_ids` of shape [3, B, S]:
the three axes are **temporal**, **height**, **width**.

For each token, the HF processor's `get_rope_index` populates the three
axes per the modality:
- Pure text token at position p: T=p, H=p, W=p (all axes equal).
- Vision token at (t, h, w) in a video patch grid: T=t, H=h, W=w.

The rotary table is built per-axis:
```python
inv_freq = 1 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
inv_freq_expanded = inv_freq[None, None, :, None].expand(3, B, -1, 1)
position_ids_expanded = position_ids[:, :, None, :].float()
freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
emb = cat([freqs, freqs], -1)              # [3, B, S, head_dim]
cos = emb.cos() * attention_scaling
sin = emb.sin()
```

Then `apply_multimodal_rotary_pos_emb` **stitches** the three axis tables
into a single (cos, sin) of shape [B, S, head_dim] by partitioning
head_dim channels per `mrope_section` and cyclically picking axis index
`i % 3`:

```python
mrope_section = mrope_section * 2           # 6 chunks total for split-half
cos = cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, -1))], -1)
sin = cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, -1))], -1)
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

For Qwen2.5-VL 3B with `head_dim=128` and `mrope_section=[16, 24, 24]`,
the channel layout is:
```
channels [0:16)   ← T axis (first half of rotate-pair)
channels [16:40)  ← H axis
channels [40:64)  ← W axis
channels [64:80)  ← T axis (second half of rotate-pair, copy)
channels [80:104) ← H axis
channels [104:128) ← W axis
```

Source: `modeling_qwen2_5_vl.py:485-545` (Qwen2_5_VLRotaryEmbedding),
`modeling_qwen2_5_vl.py:564-606` (apply_multimodal_rotary_pos_emb).

**Bit-exact verified:** `tests/models/qwen2_5_vl/test_isolation_hf.py`
exercises `api.ops.rope_apply_mrope` against
`apply_multimodal_rotary_pos_emb` on identical inputs and obtains zero
delta.

## 4. Decoder block diagram

```
   hidden_states  [B, S, 2048]  (vision_tokens + text_tokens interleaved)
   position_ids   [3, B, S]     (T/H/W axes, equal for text-only tokens)
        |
   +----+----+
   |         |
   |     input_layernorm (RMSNorm STANDARD_W eps=1e-6)
   |         |
   |    self_attn
   |    ├── q_proj   (W [16*128, 2048], bias [2048])
   |    ├── k_proj   (W [2*128, 2048],  bias [256])
   |    ├── v_proj   (W [2*128, 2048],  bias [256])
   |    ├── M-RoPE   (SPLIT_HALF + mrope_section=[16,24,24], theta=1M)
   |    ├── KV cache (CONTIGUOUS HND)
   |    ├── SDPA     (GQA n_q=16, n_kv=2 — 8:1 grouping, causal)
   |    └── o_proj   (W [2048, 16*128], no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     post_attention_layernorm (RMSNorm STANDARD_W)
   |         |
   |    mlp (Qwen2MLP)
   |    ├── gate_proj (W [11008, 2048], no bias)
   |    ├── up_proj   (W [11008, 2048], no bias)
   |    ├── SiLU(g)*u
   |    └── down_proj (W [2048, 11008], no bias)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, 2048]
```

## 5. Spec instantiation

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=16, n_kv_heads=2, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=True, k_bias=True, v_bias=True, o_bias=False,
        rope=RoPESpec(
            base_theta=1_000_000.0,
            basis=RoPEBasis.SPLIT_HALF,
            scaling=RoPEScaling.NONE,
            mrope_section=(16, 24, 24),
        ),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=11008,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
```

## 6. Quirks

- **No QK-norm** (Qwen2.5 predates Qwen3's QK-norm).
- **QKV biases TRUE** (inherited from Qwen2; modeling_qwen2_5_vl.py:641-643).
- **No sliding window** in 3B-Instruct (use_sliding_window=False).
- **`mrope_section` invariant:** `sum(mrope_section) * 2 == head_dim`
  (the *2 comes from rotate_half pairing the two halves of head_dim).
  For 3B: 16+24+24=64, 64*2=128=head_dim.
- **3D position_ids:** the runtime auto-dispatches to M-RoPE when
  `position_ids.dim() == 3`. The 1D path is preserved for plain text
  inference (which is mathematically equivalent because all 3 axes are
  equal under aligned-position semantics).

## 7. Weight-name mapping (HF → API)

| HF tensor name (Qwen2.5-VL) | API tensor slot |
|---|---|
| `model.language_model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.language_model.layers.{L}.self_attn.q_proj.{weight,bias}` | `blk.attention.q_proj.{weight,bias}` |
| `model.language_model.layers.{L}.self_attn.k_proj.{weight,bias}` | `blk.attention.k_proj.{weight,bias}` |
| `model.language_model.layers.{L}.self_attn.v_proj.{weight,bias}` | `blk.attention.v_proj.{weight,bias}` |
| `model.language_model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `model.language_model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.language_model.layers.{L}.mlp.{gate,up,down}_proj.weight` | `blk.feedforward.{gate,up,down}_proj.weight` |

## 8. Source citations

- HF reference: `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
  (485-545 Qwen2_5_VLRotaryEmbedding, 564-606 apply_multimodal_rotary_pos_emb,
  609-696 Qwen2_5_VLAttention, 699-764 Qwen2_5_VLDecoderLayer,
  766-840 Qwen2_5_VLTextModel).

## 9. B8 validation status

- Shape tests (tiny config, 3D and 1D-via-3D position_ids) — 3 tests
- Config from_hf_dict — 3 tests
- Sub-op isolation (M-RoPE BIT-EXACT vs HF) — 3 tests
- **Full layer numerical equivalence vs HF at atol=5e-4** —
  max_abs_diff = 1.67e-6 (~299× tighter than required).
