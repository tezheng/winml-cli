# Voxtral — LM-Decoder Layer

## 0. At-a-glance

- **Signature:** Llama/Mistral-style LM decoder in audio-LM wrapper — GQA-32:8 + RoPE (θ=1e8) + SwiGLU + RMSNorm; no SWA
- **Active params:** 3B total (LM-decoder only; full Voxtral-Mini-3B-2507 includes audio frontend)
- **Layer mix:** 30 attn (GQA-32:8, all causal)
- **KV cache / token (bf16):** 30 L × 8 KV × 128 Dh × 2 × 2 = 120 kB

---

## 1. Identity

- **Family:** Voxtral (Mistral.ai)
- **Variants in B9 scope:** `mistralai/Voxtral-Mini-3B-2507` (canonical
  gate); larger Voxtral variants share the same Llama-style LM-decoder
  scaffolding with different dims.
- **Release date:** July 2025 (Mistral.ai's audio model release).
- **HF base model card:** https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
- **transformers source:**
  `transformers/src/transformers/models/voxtral/modeling_voxtral.py`
- **Architecture:** Whisper-style audio encoder + multi-modal projector +
  **Llama LM decoder**. For B9 we ship only the LM decoder.

## 2. KEY DRIFT vs prompt assumptions

The B9 plan says "LM decoder likely Mistral-style (we already have
models/mistral/)" and asks us to "verify decoder is identical to Mistral
7B v0.3 or has its own variations". The HF source resolution:

| feature | B9-plan assumption | HF source |
|---|---|---|
| backbone | Mistral-style | **Llama** (`text_config.model_type='llama'`). Verified on `mistralai/Voxtral-Mini-3B-2507` config. Architecturally indistinguishable from Mistral 7B v0.3 at the layer level (RMSNorm STANDARD_W + SwiGLU + GQA + no SWA). |
| rope_theta | (1e6 like Mistral) | **1e8** (HF `_default_text_config_kwargs` at configuration_voxtral.py:96-107). Note Mistral v0.3 uses 1e6 — Voxtral raised it ten-fold for 131k context. |
| sliding_window | (4096 like Mistral v0.1/2) | **None** — `text_config.sliding_window=None`. Same as Mistral v0.3 (which also dropped SWA). |
| GQA | (verify) | n_q=32, n_kv=8 (4× reduction). Same as Mistral 7B v0.3. |
| QK-norm | (verify) | **None.** Llama 3 / Mistral baseline. |
| biases | (verify) | **None** on q/k/v/o or gate/up/down. Llama 3 / Mistral baseline. |
| tie_word_embeddings (text_config) | (verify) | **False** (inner). The TOP-level VoxtralConfig has `tie_word_embeddings=True` — that tie connects the lm_head to the language_model's embed_tokens, NOT a per-layer detail. |

**Net result:** the LM-decoder block is identical to Mistral 7B v0.3's
block with `rope_theta=1e8`. We could literally reuse `models/mistral/`
but per the project's per-family discipline we ship a Voxtral wrapper.

## 3. Architecture summary

- **Audio encoder:** Whisper-style encoder (1280-d, 32 layers).
  OUT OF SCOPE.
- **Multi-modal projector:** `VoxtralMultiModalProjector` (a 2-Linear MLP).
  OUT OF SCOPE.
- **LM decoder:** 30-layer Llama-style block.
- **Canonical Voxtral-Mini-3B-2507 dims:** hidden=3072, num_layers=30,
  n_q=32, n_kv=8 (GQA), head_dim=128, intermediate=8192, rope_theta=1e8,
  vocab=131072, max_pos=131072, sliding_window=None, rms_norm_eps=1e-5.

## 4. Decoder block diagram

```
                hidden_states  [B, S, 3072]
                  |
        +---------+---------+
        |                   |
        | input_layernorm (RMSNorm STANDARD_W, eps=1e-5)
        |                   |
        |   self_attn  (GQA n_q=32 / n_kv=8, head_dim=128)
        |   ├── q_proj  (W [32*128, 3072], no bias)
        |   ├── k_proj  (W  [8*128, 3072], no bias)
        |   ├── v_proj  (W  [8*128, 3072], no bias)
        |   ├── RoPE    (SPLIT_HALF, theta=1e8)
        |   ├── KV cache (CONTIGUOUS HND)
        |   ├── eager softmax(qk/sqrt(Dh) + causal_mask) @ v
        |   └── o_proj  (W [3072, 32*128], no bias)
        |                   |
        +---->add (residual 1)
                  |
        +---------+---------+
        |                   |
        | post_attention_layernorm (RMSNorm STANDARD_W, eps=1e-5)
        |                   |
        |   mlp  (SwiGLU)
        |   ├── gate_proj  Linear(3072, 8192) no bias
        |   ├── up_proj    Linear(3072, 8192) no bias
        |   ├── silu(gate) * up   → [B, S, 8192]
        |   └── down_proj  Linear(8192, 3072) no bias
        |                   |
        +---->add (residual 2)
                  |
                  y  [B, S, 3072]
```

## 5. Spec instantiation

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=8, head_dim=128,
        kind=AttentionKind.STANDARD,
        qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=RoPESpec(base_theta=1e8,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=8192,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
        fused_gate_up=False,
    ),
    pre_attn_norm=NormSpec(kind=NormKind.RMS, eps=1e-5,
                            weight_mode=NormWeightMode.STANDARD_W),
    pre_ffn_norm=NormSpec(kind=NormKind.RMS, eps=1e-5,
                            weight_mode=NormWeightMode.STANDARD_W),
)
```

## 6. Quirks

- **Llama backbone, not Mistral.** `text_config.model_type='llama'` —
  but algebraically identical at the layer level (the difference is
  only at the model-level tokenizer + LM head + tie semantics).
- **rope_theta=1e8.** A 100x increase vs Llama 3 and 10x vs Mistral v0.3.
  Supports the 131k context window without YARN/Llama3 scaling tricks.
- **No sliding window.** `text_config.sliding_window=None`.
- **Audio + text tokens share the same decoder.** Audio embeddings
  produced by the encoder + projector are scattered into the
  `inputs_embeds` tensor at positions where `input_ids == audio_token_id`
  (modeling_voxtral.py:455-463). The per-layer code never sees a
  modality marker — only fp16/fp32 hidden states.

## 7. Weight-name mapping (HF → API)

| HF tensor name | API tensor slot |
|---|---|
| `model.language_model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.language_model.layers.{L}.self_attn.q_proj.weight` | `blk.attention.q_proj.weight` |
| `model.language_model.layers.{L}.self_attn.k_proj.weight` | `blk.attention.k_proj.weight` |
| `model.language_model.layers.{L}.self_attn.v_proj.weight` | `blk.attention.v_proj.weight` |
| `model.language_model.layers.{L}.self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `model.language_model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.language_model.layers.{L}.mlp.gate_proj.weight` | `blk.feedforward.gate_proj.weight` |
| `model.language_model.layers.{L}.mlp.up_proj.weight` | `blk.feedforward.up_proj.weight` |
| `model.language_model.layers.{L}.mlp.down_proj.weight` | `blk.feedforward.down_proj.weight` |

## 8. Source citations

- `transformers/models/voxtral/configuration_voxtral.py:74-131` (VoxtralConfig
  sub-configs; text_config defaults: hidden=3072, n_kv=8, head_dim=128,
  rope_theta=1e8, rms_norm_eps=1e-5).
- `transformers/models/voxtral/modeling_voxtral.py:384-481` (VoxtralModel —
  `self.language_model = AutoModel.from_config(text_config)`).
- `transformers/models/llama/modeling_llama.py` (canonical LlamaDecoderLayer
  used by the language_model).

## 9. B9 validation status

- Shape + assembly tests — `tests/models/voxtral/test_layer_shape.py`.
- Config from_hf_dict tests — `tests/models/voxtral/test_config.py`.
- **Synthetic-weight numerical gate vs HF LlamaModel code path** (no
  3B checkpoint download required) — `tests/models/voxtral/test_numerical_synthetic.py`.
- HF checkpoint numerical gate — `tests/models/voxtral/test_numerical_hf.py`
  (gated on network availability; the ~7 GB safetensors not cached by default).
