# GOT-OCR 2.0 — LM-Decoder Layer

## 0. At-a-glance

- **Signature:** Qwen2-0.5B LM portion + linear vision connector (LM-decoder only in scope); Qwen2-style QKV biases, no QK-norm
- **Active params:** ~0.5B (LM decoder; 0.58B total including vision encoder)
- **Layer mix:** 24 attn (dense MHA 16Q/16KV, all causal)
- **KV cache / token (bf16):** 96 kB (24 layers × 16 kv heads × 64 head_dim × 4 B)

---

## 1. Identity

- **Family:** GOT-OCR 2.0 (StepFun)
- **Variants in B8 scope:** stepfun-ai/GOT-OCR-2.0-hf (LM-decoder portion)
- **Release date:** September 2024
- **Paper / tech report:** "General OCR Theory: Towards OCR-2.0 via a Unified
  End-to-end Model" — arXiv:2409.01704
- **HF base model card:** https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf
- **transformers source:** `transformers/src/transformers/models/got_ocr2/modeling_got_ocr2.py`
- **transformers backbone source:**
  `transformers/src/transformers/models/qwen2/modeling_qwen2.py` (the LM
  decoder is a vanilla `Qwen2Model`).

## 2. Architecture summary

- **Vision encoder:** SAM-style ViT (~250M params) — OUT OF SCOPE for B8.
- **Vision connector:** `nn.Linear(vision_dim, hidden_size)` projecting
  patch tokens into the text-decoder hidden space (one connector, no MLP).
- **Concat-prefix fusion:** the vision features are scatter-overwritten
  into specific embedding positions matching `image_token_index=151859`.
  After fusion the LM decoder sees a flat `[B, S_total, hidden]` tensor
  where image and text tokens are interleaved — there is **no special
  attention mask, no separate cross-attention, and no architectural
  change in the decoder**.

  Source: `modeling_got_ocr2.py:546-587` (apply_multimodal_projector +
  get_placeholder_mask).

- **LM decoder:** Qwen2-0.5B —
  hidden_size=1024, n_heads=16, n_kv_heads=16, head_dim=64,
  intermediate_size=2816, num_hidden_layers=24, vocab=151860,
  rope_theta=1_000_000, rms_norm_eps=1e-6, tie_word_embeddings=True.

## 3. Decoder block diagram

```
   hidden_states  [B, S, 1024]  (vision_tokens prepended to text_tokens by HF)
        |
   +----+----+
   |         |
   |     input_layernorm (RMSNorm STANDARD_W)
   |         |
   |    self_attn
   |    ├── q_proj  (W [16*64, 1024], bias [16*64])
   |    ├── k_proj  (W [16*64, 1024], bias [16*64])
   |    ├── v_proj  (W [16*64, 1024], bias [16*64])
   |    ├── RoPE    (SPLIT_HALF basis, theta=1_000_000)
   |    ├── KV cache (CONTIGUOUS HND)
   |    ├── SDPA    (n_q=n_kv=16, causal)
   |    └── o_proj  (W [1024, 16*64], bias=False)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     post_attention_layernorm (RMSNorm STANDARD_W)
   |         |
   |    mlp
   |    ├── gate_proj (W [2816, 1024], bias=False)
   |    ├── up_proj   (W [2816, 1024], bias=False)
   |    ├── SiLU(g) * u
   |    └── down_proj (W [1024, 2816], bias=False)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, 1024]
```

## 4. Key differences vs Qwen3

| feature | Qwen2 (GOT-OCR-2.0) | Qwen3 |
| --- | --- | --- |
| QKV biases | YES (q/k/v) | NO (no biases anywhere) |
| QK-norm | NONE | RMSNorm PRE_ROPE PER_HEAD_DH |
| head_dim | hidden / num_heads (64) | EXPLICIT, distinct (e.g. 128) |
| RoPE theta | 1e6 (also Qwen3) | 1e6 |
| FFN biases | NO | NO |

Source vs Qwen3: `modeling_qwen2.py:200-203` (`bias=True` on q/k/v),
`modeling_qwen2.py:217-222` (apply_rotary_pos_emb directly, NO QK-norm),
vs `modeling_qwen3.py:194-204` and `modeling_qwen3.py:225-228` (q_norm /
k_norm PRE-RoPE on per-head Dh).

## 5. Concat-prefix fusion details (for completeness)

HF flow:
```
inputs_embeds = embed_tokens(input_ids)                  # [B, S, 1024]
image_embeds = vision_tower(pixel_values)                # [B, n_patches, V_h]
image_embeds = multi_modal_projector(image_embeds)       # [B, n_patches, 1024]
mask = (input_ids == image_token_index)                  # [B, S]
inputs_embeds[mask] = image_embeds.reshape(-1, 1024)     # scatter overwrite
hidden_states = inputs_embeds                            # fed to layers as-is
```

Source: `modeling_got_ocr2.py:546-587`. The LM-decoder layer is unaware
of this fusion — it just sees `hidden_states`. Our B8 numerical gate
exploits this: we sample a synthetic `inputs_embeds` from
`lang_model.embed_tokens` (using mixed image-token-index + text-token
IDs) and feed it directly to layer 0.

## 6. Spec instantiation

See `models/got_ocr2/config.py:GotOcr2Config.to_block_spec()`. Concretely:

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=16, n_kv_heads=16, head_dim=64,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=True, k_bias=True, v_bias=True, o_bias=False,
        rope=RoPESpec(base_theta=1_000_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=2816,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    pre_attn_norm=NormSpec(...), pre_ffn_norm=NormSpec(...),
)
```

## 7. Weight-name mapping (HF → API)

| HF tensor name (GOT-OCR 2.0) | API tensor slot |
|---|---|
| `model.language_model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.language_model.layers.{L}.self_attn.q_proj.{weight,bias}` | `blk.attention.q_proj.{weight,bias}` |
| `model.language_model.layers.{L}.self_attn.k_proj.{weight,bias}` | `blk.attention.k_proj.{weight,bias}` |
| `model.language_model.layers.{L}.self_attn.v_proj.{weight,bias}` | `blk.attention.v_proj.{weight,bias}` |
| `model.language_model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `model.language_model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.language_model.layers.{L}.mlp.gate_proj.weight` | `blk.feedforward.gate_proj.weight` |
| `model.language_model.layers.{L}.mlp.up_proj.weight` | `blk.feedforward.up_proj.weight` |
| `model.language_model.layers.{L}.mlp.down_proj.weight` | `blk.feedforward.down_proj.weight` |

## 8. Source citations

- HF reference: `transformers/models/got_ocr2/modeling_got_ocr2.py`
  (lines 277-308 `GotOcr2PreTrainedModel`, 531-617 `GotOcr2Model`).
- HF backbone: `transformers/models/qwen2/modeling_qwen2.py:186-309`
  (Qwen2Attention init + forward, Qwen2DecoderLayer).
- Config defaults: `transformers/models/got_ocr2/configuration_got_ocr2.py:108-127`
  (default text_config is a Qwen2 with hidden=1024, n_heads=16,
  rope_theta=1_000_000, vocab=151860, tie_word_embeddings=True).

## 9. B8 validation status

- Shape tests at canonical 0.5B variant + tiny synthetic — 3 tests
- Config from_hf_dict — 4 tests
- **Numerical equivalence vs HF at atol=5e-4** — max_abs_diff = 3.81e-6
  (~131× tighter than required).
