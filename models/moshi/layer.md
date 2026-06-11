# Moshi 7B — Main-Decoder Layer

## 0. At-a-glance

- **Signature:** Helium MHA + fused gate-up SwiGLU (CAUSAL mask, no QK-norm) — dual-stream text+audio handled at model level, not per-layer
- **Active params:** ~7B (main decoder dense, Helium backbone)
- **Layer mix:** 32 attn (dense GQA 32Q/8KV, all causal)
- **KV cache / token (bf16):** 128 kB (32 layers × 8 kv heads × 128 head_dim × 4 B)

---

## 1. Identity

- **Family:** Moshi (Kyutai)
- **Variants in B9 scope:** `kmhf/hf-moshiko`, `kmhf/hf-moshika` (and the
  bf16-checkpoint mirrors `kyutai/moshik{o,a}-pytorch-bf16`).
- **Release date:** September 2024 (Kyutai's full-duplex audio LM).
- **HF base model card:** https://huggingface.co/kmhf/hf-moshiko
- **transformers source:**
  `transformers/src/transformers/models/moshi/modeling_moshi.py`
- **Paper:** Kyutai-AI Moshi technical report (2024).

## 2. KEY DRIFT vs prompt assumptions

The B9 plan describes Moshi's main decoder as "Llama-like (RMSNorm, RoPE,
SwiGLU, GQA — verify)". The HF transformers source shows:

| feature | B9-plan assumption | HF source (modeling_moshi.py) |
|---|---|---|
| attention kind | GQA | **MHA** (n_q = n_kv = 32) |
| KV heads | "verify" | configuration_moshi.py:134-137: default `num_key_value_heads=None` → post_init sets it to `num_attention_heads=32`. Verified via canonical `kmhf/hf-moshiko` config: `num_key_value_heads=32`. |
| SwiGLU | "verify" | **FUSED gate/up** (MoshiGatingMLP at L368-390): single `fc1: Linear(hidden, ffn_dim)`, then `.view(B,S,2,-1)` chunks into (gate, up). NOT the split `gate_proj`/`up_proj` of Llama. |
| QK-norm | "verify presence/absence" | **NONE** — MoshiAttention has no q_norm/k_norm modules (modeling_moshi.py:406-509). |
| RoPE | "RoPE" | SPLIT_HALF basis, `cat((freqs, freqs), -1)` (Llama copy at L334-339). theta=10000, rope_type="default". |
| RMSNorm eps | "verify" | **1e-8** (vs Llama 3's 1e-5) — modeling_moshi.py:713 + configuration_moshi.py:148. |
| sliding_window | "verify SWA" | Eager `MoshiAttention.forward` (L454-509) does NOT apply SWA. `sliding_window` is only forwarded to `_flash_attention_forward` (L611). With `sliding_window == max_position_embeddings == 3000`, the eager path is pure causal. We map to `MaskKind.CAUSAL`. |
| dual-stream | "no architectural change to per-layer code" | **Confirmed.** The text + 8 audio-codebook stream merging happens at the embedding layer (8x `nn.Embedding` plus 1 text embedding, summed before layer 0) and at the depth-decoder head (per-codebook tied head). The per-layer MoshiDecoderLayer is unchanged regardless of which stream the tokens came from. Source: modeling_moshi.py:994-1011 (MoshiModel `embed_tokens` + `layers` ModuleList). |

## 3. Architecture summary

- **Audio encoder/decoder (Mimi codec):** OUT OF SCOPE.
- **Depth decoder (MoshiDepthDecoder):** 6-layer per-codebook prediction
  head with `use_flexible_linear=True` (one set of weights per codebook).
  OUT OF SCOPE.
- **Main decoder (Helium):** 32 dense MHA layers.
- **Canonical 7B dims (kmhf/hf-moshiko):** hidden=4096, n_q=n_kv=32 (MHA),
  head_dim=128, ffn_dim=22528 (intermediate=11264), num_layers=32,
  rope_theta=10000, vocab=32000, max_pos=3000, sliding_window=3000,
  rms_norm_eps=1e-8, num_codebooks=8, audio_vocab_size=2048,
  tie_word_embeddings=False.

## 4. Decoder block diagram

```
                hidden_states  [B, S, 4096]
                  |
        +---------+---------+
        |                   |
        | input_layernorm (RMSNorm STANDARD_W, eps=1e-8)
        |                   |
        |   self_attn  (STANDARD MHA, 32 heads, head_dim=128)
        |   ├── q_proj.linear  (W [32*128, 4096], no bias)
        |   ├── k_proj.linear  (W [32*128, 4096], no bias)
        |   ├── v_proj.linear  (W [32*128, 4096], no bias)
        |   ├── RoPE    (SPLIT_HALF, theta=10000)
        |   ├── KV cache (CONTIGUOUS HND)
        |   ├── eager softmax(qk/sqrt(Dh) + mask) @ v
        |   └── o_proj.linear  (W [4096, 32*128], no bias)
        |                   |
        +---->add (residual 1)
                  |
        +---------+---------+
        |                   |
        | post_attention_layernorm (RMSNorm STANDARD_W, eps=1e-8)
        |                   |
        |   mlp  (FUSED SwiGLU)
        |   ├── fc1   Linear(4096, 22528) no bias  (== gate_up_proj)
        |   ├── view(B, S, 2, 11264)  → (gate, up)
        |   ├── silu(gate) * up   → [B, S, 11264]
        |   └── fc2   Linear(11264, 4096) no bias  (== down_proj)
        |                   |
        +---->add (residual 2)
                  |
                  y  [B, S, 4096]
```

## 5. Spec instantiation

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=32, n_kv_heads=32, head_dim=128,
        kind=AttentionKind.STANDARD,
        qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=RoPESpec(base_theta=10_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=11264,                # ffn_dim // 2
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
        fused_gate_up=True,                     # MoshiGatingMLP fc1 layout
    ),
    pre_attn_norm=NormSpec(kind=NormKind.RMS, eps=1e-8,
                            weight_mode=NormWeightMode.STANDARD_W),
    pre_ffn_norm=NormSpec(kind=NormKind.RMS, eps=1e-8,
                            weight_mode=NormWeightMode.STANDARD_W),
)
```

## 6. Quirks

- **MHA, NOT GQA.** Despite the 7B size, the Helium decoder uses 32
  K/V heads (full MHA).
- **Fused SwiGLU gate/up.** Single `fc1: Linear(4096, 22528)` whose
  output is split via `.view(B, S, 2, -1)`. Algebraically equivalent to
  `chunk(2, dim=-1)` because view-then-index-0 returns the first half.
- **RMSNorm eps = 1e-8** (much smaller than Llama 3's 1e-5).
- **MoshiLinear wrapper.** q/k/v/o projections are wrapped in a thin
  `MoshiLinear` module whose `.linear` is the actual `nn.Linear`. This
  means the HF state-dict keys are `...q_proj.linear.weight` not
  `...q_proj.weight`. The MLP's fc1/fc2 are NOT wrapped (they're plain
  `nn.Linear` in the non-flexible path).
- **Dual-stream is at the model level only.** MoshiForConditionalGeneration
  has multiple `audio_embeddings` (one per codebook) summed with the text
  embedding before layer 0, but the per-layer Helium block is
  stream-agnostic.

## 7. Weight-name mapping (HF → API)

| HF tensor name | API tensor slot |
|---|---|
| `model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.layers.{L}.self_attn.q_proj.linear.weight` | `blk.attention.q_proj.weight` |
| `model.layers.{L}.self_attn.k_proj.linear.weight` | `blk.attention.k_proj.weight` |
| `model.layers.{L}.self_attn.v_proj.linear.weight` | `blk.attention.v_proj.weight` |
| `model.layers.{L}.self_attn.o_proj.linear.weight` | `blk.attention.o_proj.weight` |
| `model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.layers.{L}.mlp.fc1.weight` | `blk.feedforward.gate_up_proj.weight` |
| `model.layers.{L}.mlp.fc2.weight` | `blk.feedforward.down_proj.weight` |

## 8. Source citations

- `transformers/models/moshi/configuration_moshi.py:93-218` (MoshiConfig).
- `transformers/models/moshi/modeling_moshi.py`:
  - L192-208 MoshiRMSNorm (STANDARD_W, weight upcast then `.type_as(x)`).
  - L250-265 MoshiLinear (wraps `nn.Linear` via `.linear`).
  - L269-365 MoshiRotaryEmbedding + rotate_half + apply_rotary_pos_emb
    (Llama copy — SPLIT_HALF rotate_half, `cat((freqs,freqs),-1)`).
  - L368-390 MoshiGatingMLP (FUSED `fc1` + view(2,-1) split + `fc2`).
  - L406-509 MoshiAttention (eager — no SWA application).
  - L702-760 MoshiDecoderLayer (PRE-norm RMSNorm + self_attn + mlp).
  - L994-1101 MoshiModel (embedding + layers + final RMSNorm).

## 9. B9 validation status

- Shape tests — see `tests/models/moshi/test_layer_shape.py`.
- Config from_hf_dict tests — see `tests/models/moshi/test_config.py`.
- **Synthetic-weight numerical gate vs HF MoshiModel code path** (no
  7B checkpoint download required) — see
  `tests/models/moshi/test_numerical_synthetic.py`.
- HF checkpoint numerical gate — `tests/models/moshi/test_numerical_hf.py`
  (gated on network availability; the 14 GB safetensors are not in the
  repo's hf_cache by default).
