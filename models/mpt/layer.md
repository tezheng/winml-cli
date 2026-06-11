# MPT 7B — Decoder Layer

## 0. At-a-glance

- **Signature:** MHA + ALiBi positional bias (no RoPE) + ungated exact-GELU FFN + LayerNorm-without-bias
- **Active params:** ~7B (mosaicml/mpt-7b, dense)
- **Layer mix:** 32 attn (dense MHA 32Q/32KV, all causal + ALiBi)
- **KV cache / token (bf16):** 128 kB (32 layers × 32 kv heads × 128 head_dim × 4 B)

---

## 1. Identity

- **Family:** MPT (MosaicML / Databricks)
- **Variants in scope:** `mosaicml/mpt-7b` (32 layers, hidden=4096, 32 heads)
- **HF model card:** https://huggingface.co/mosaicml/mpt-7b
- **transformers source:** `transformers/models/mpt/modeling_mpt.py`

## 2. Decoder block diagram

```
        x  [B, S, 4096]
        |
   +----+----+
   |         |
   |     norm_1  (LayerNorm, bias=None)
   |         |
   |    attn (MHA, ALiBi bias, no RoPE)
   |    ├── Wqkv fused → q, k, v (split QKV)
   |    ├── ALiBi positional bias added to attention scores
   |    ├── KVCache write/read (CAUSAL mask)
   |    ├── SDPA (MHA 32/32, scale=1/sqrt(128))
   |    └── out_proj
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     norm_2  (LayerNorm, bias=None)
   |         |
   |    ffn (ungated GELU, 4× expansion)
   |    ├── up_proj: Linear(4096, 16384, bias=False)
   |    ├── GELU_exact activation
   |    └── down_proj: Linear(16384, 4096, bias=False)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, 4096]
```

## 3. Key architectural facts

- **No RoPE**: MPT uses ALiBi (Attention with Linear Biases) — position is encoded as a linear bias on attention scores.
- **LayerNorm without bias**: `norm_1.bias = None` and `norm_2.bias = None`. Source: `modeling_mpt.py:163-165`.
- **Ungated GELU FFN**: unlike SwiGLU (which uses a gating mechanism), MPT uses a single `up_proj` → `GELU` → `down_proj` path. No gate_proj. Source: `modeling_mpt.py:137-155`.
- **Fused QKV**: `Wqkv = Linear(hidden, 3*hidden)` then split into Q/K/V. Source: `modeling_mpt.py:65-134`.
- **MHA (not GQA)**: `n_q = n_kv = 32`, `head_dim = 128`.

## 4. Config constants (mosaicml/mpt-7b)

| Field | Value |
|---|---|
| hidden_size | 4096 |
| num_attention_heads | 32 |
| num_key_value_heads | 32 (MHA) |
| head_dim | 128 |
| intermediate_size | 16384 (4× expansion) |
| num_hidden_layers | 32 |
| max_position_embeddings | 2048 |
| alibi_bias_max | 8.0 |
| layer_norm_epsilon | 1e-5 |
| vocab_size | 50432 |
