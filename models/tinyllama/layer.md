# TinyLlama 1.1B — Decoder Layer

## 1. Identity

- **Family:** TinyLlama (community model — Llama architecture).
- **Variants in B1 scope:** TinyLlama-1.1B-Chat-v1.0 (gate variant); base
  TinyLlama-1.1B-intermediate-step-* checkpoints are IR-compatible.
- **Release date:** Jan 2024 (Chat v1.0 final release).
- **HF base model card:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **transformers source:** `transformers/models/llama/modeling_llama.py`
  (same modeling code as Llama 3 because TinyLlama uses HF `model_type="llama"`).

## 2. Decoder block diagram

Identical to Llama 3 default-RoPE. See `models/llama3/layer.md` for the full
block diagram. TinyLlama re-uses `models.llama3.layer.build_llama3_decoder_layer`.

## 3. Tensor IO trace

For TinyLlama-1.1B (D=2048, n_q=32, n_kv=4, head_dim=64, I=5632):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 2048] | bf16 |
| q_proj | q | [B, S, 32*64]=[B, S, 2048] | bf16 |
| k_proj | k | [B, S, 4*64]=[B, S, 256] | bf16 |
| v_proj | v | [B, S, 4*64]=[B, S, 256] | bf16 |
| gate_proj | g | [B, S, 5632] | bf16 |
| down_proj | ffn_out | [B, S, 2048] | bf16 |

## 4. Op trace

Identical to Llama 3 default-RoPE (no LLAMA3 scaling, no QK-norm).

## 5. Spec instantiation

`TinyLlamaConfig.to_block_spec()` returns the Llama3Config block spec at
theta=10000.0, scaling=NONE.

## 6. Quirks

- **GQA — NOT MHA.** Despite TinyLlama's small size, the model is GQA with
  `num_key_value_heads=4` (vs `num_attention_heads=32`). This was an open
  question in research v3 §8.4 and is now confirmed source-grounded against
  TinyLlama-1.1B-Chat-v1.0/config.json: `n_q=32, n_kv=4, head_dim=64`.
- **22 hidden layers.** Half of Llama 2 7B's 32-layer depth.
- **RoPE theta=10000.0.** Vanilla RoPE (rope_type="default"); no LLAMA3 smooth
  scaling. This differs from Llama 3 (theta=500K) and Mistral v0.3 (theta=1M).
- **`tie_word_embeddings=False`** — embedding and LM head do NOT share weights
  in TinyLlama-1.1B-Chat-v1.0. (Contrast: Llama 3.2 1B has them tied.)
- **No QK-norm. No biases. RMSNorm STANDARD_W, eps=1e-5.**

## 7. Weight-name mapping (HF → API)

Identical to Llama 3 (see `models/llama3/layer.md` §7). The TinyLlama loader
forwards to `load_hf_llama3_layer`.

## 8. Source citations

- HF reference: `transformers/models/llama/modeling_llama.py` (TinyLlama uses
  `model_type='llama'` per its config).
- HF config: `TinyLlama/TinyLlama-1.1B-Chat-v1.0/config.json`.
- Census: `research/01-model-census.v3.md` §3.x (TinyLlama entry).
- Research surprise: `research/issues/08-coverage-justification.md` §TinyLlama
  notes (GQA vs MHA disambiguation).
