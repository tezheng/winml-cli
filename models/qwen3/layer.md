# Qwen3 — Decoder Layer

## 1. Identity

- **Family:** Qwen3 (Alibaba)
- **Variants in M1 scope:** 0.6B, 1.7B, 4B, 8B (all dense)
- **Release date:** March / April 2025
- **Paper / tech report:** Qwen3 Technical Report — arxiv:2505.09388
- **HF base model card:** https://huggingface.co/Qwen/Qwen3-0.6B
- **transformers source:** `transformers/src/transformers/models/qwen3/modeling_qwen3.py`

## 2. Decoder block diagram

```
        x  [B, S, D]
        |
   +----+----+
   |         |
   |     input_norm (RMSNorm STANDARD_W)
   |         |
   |    attention(token_mixer)
   |    ├── q_proj, k_proj, v_proj   (split QKV, no bias)
   |    ├── q_norm, k_norm           (QKNorm PRE-RoPE, PER_HEAD_DH shape)
   |    ├── RoPE (SPLIT_HALF, theta=1_000_000)
   |    ├── KVCache write/read       (CONTIGUOUS HND layout)
   |    ├── SDPA (GQA, n_q=…, n_kv=…)
   |    └── o_proj                   (no bias)
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     post_attn_norm (RMSNorm STANDARD_W)
   |         |
   |    feedforward(channel_mixer)
   |    ├── gate_proj                (SwiGLU gate, no bias)
   |    ├── up_proj                  (SwiGLU up,  no bias)
   |    ├── SiLU(gate) * up          (api.ops.mul ∘ api.ops.silu)
   |    └── down_proj                (no bias)
   |         |
   +---->add (residual 2)
        |
        y  [B, S, D]
```

## 3. Tensor IO trace

For Qwen3-0.6B (D=1024, n_q=16, n_kv=8, head_dim=128, I=3072):

| Step | Tensor | Shape | dtype |
|---|---|---|---|
| input | x | [B, S, 1024] | bf16 |
| input_norm | x_in | [B, S, 1024] | bf16 |
| q_proj | q | [B, S, 16*128]=[B, S, 2048] | bf16 |
| k_proj | k | [B, S, 8*128]=[B, S, 1024] | bf16 |
| v_proj | v | [B, S, 8*128]=[B, S, 1024] | bf16 |
| reshape | q | [B, S, 16, 128] | bf16 |
| reshape | k, v | [B, S, 8, 128] | bf16 |
| q_norm | q | [B, S, 16, 128] | bf16 (computed in fp32) |
| k_norm | k | [B, S, 8, 128] | bf16 (computed in fp32) |
| RoPE | q, k | [B, S, *, 128] | bf16 |
| transpose | q | [B, 16, S, 128] | bf16 |
| transpose | k, v | [B, 8, S, 128] | bf16 |
| cache.write | (state) | KVCache.k/v [B, 8, max_seq, 128] | bf16 |
| cache.read | k_full, v_full | [B, 8, T, 128] where T = start_pos + S | bf16 |
| sdpa | a | [B, 16, S, 128] | bf16 (fp32 accum) |
| transpose+reshape | a | [B, S, 2048] | bf16 |
| o_proj | attn_out | [B, S, 1024] | bf16 |
| residual add | x | [B, S, 1024] | bf16 |
| post_attn_norm | h | [B, S, 1024] | bf16 |
| gate_proj | g | [B, S, 3072] | bf16 |
| up_proj | u | [B, S, 3072] | bf16 |
| silu+mul | h_act | [B, S, 3072] | bf16 |
| down_proj | ffn_out | [B, S, 1024] | bf16 |
| residual add | y | [B, S, 1024] | bf16 |

## 4. Op trace (api.ops sequence)

```
x_in    = rms_norm(x, input_norm.weight, eps, "standard_w")
q       = linear(x_in, q_proj.weight)         # reshape to [B, S, 16, 128]
k       = linear(x_in, k_proj.weight)         # reshape to [B, S, 8, 128]
v       = linear(x_in, v_proj.weight)         # reshape to [B, S, 8, 128]
q       = rms_norm(q, q_norm.weight, eps, "standard_w")   # PER_HEAD_DH
k       = rms_norm(k, k_norm.weight, eps, "standard_w")
q, k    = rope_apply(q, k, cos, sin, basis="split_half")
# cache.write(k, v, start_pos); k_full, v_full = cache.read(start_pos + S)
a       = sdpa(q, k_full, v_full, is_causal=(S > 1), scale=head_dim ** -0.5)
attn_out = linear(a, o_proj.weight)
x       = add(x, attn_out)

h       = rms_norm(x, post_attn_norm.weight, eps, "standard_w")
g       = linear(h, gate_proj.weight)
u       = linear(h, up_proj.weight)
h_act   = mul(silu(g), u)
ffn_out = linear(h_act, down_proj.weight)
y       = add(x, ffn_out)
```

## 5. Spec instantiation (for Qwen3-0.6B)

See `models/qwen3/config.py:Qwen3Config.to_block_spec()`. Concretely:

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=128,
        kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        qk_norm=NormSpec(kind=NormKind.RMS, eps=1e-6,
                         weight_mode=NormWeightMode.STANDARD_W),
        qk_norm_phase=QKNormPhase.PRE_ROPE,
        qk_norm_shape=QKNormShape.PER_HEAD_DH,
        rope=RoPESpec(base_theta=1_000_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=3072,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    input_norm=NormSpec(...), pre_attn_norm=..., pre_ffn_norm=...,
)
```

## 6. Quirks

- **QK-norm is PRE-RoPE** (verified against `modeling_qwen3.py` `Qwen3Attention.forward` —
  `q_norm` and `k_norm` are called BEFORE `apply_rotary_pos_emb`). This was incorrectly
  documented as POST-RoPE in research/05 v1.
- **QK-norm weight shape is `[head_dim]`** (per-head Dh), distinct from OLMo 2's
  `[n_heads * head_dim]` shape. Source comment in `modeling_qwen3.py`:
  *"unlike olmo, only on the head dim!"*.
- **`rope_theta=1_000_000`** for Qwen3 base context (32k), NOT 5M (v1 error).
- **`tie_word_embeddings=True`** for 0.6B / 1.7B / 4B; `False` for 8B.
- **`head_dim` is explicit** in Qwen3 (1024/16=64 in 0.6B, vs head_dim=128 — they
  differ). Don't infer head_dim from hidden_size / num_heads.
- **No biases on any projection** (q/k/v/o, gate/up/down).
- **transformers 5.10.2 API note:** `Qwen3Config.rope_theta` is no longer a direct
  attribute — it lives under `cfg.rope_parameters["rope_theta"]`. `from_hf_dict`
  handles both shapes.

## 7. Source citations

- HF reference: `transformers/src/transformers/models/qwen3/modeling_qwen3.py:Qwen3DecoderLayer`
- HF config: `transformers/src/transformers/models/qwen3/configuration_qwen3.py`
- vLLM reference: `vllm/model_executor/models/qwen3.py` (mostly aliased from Llama)
- llama.cpp reference: `convert_hf_to_gguf.py` Qwen3 conversion + ggml tensor names
- Survey: `research/02-layer-sources.v2.md` §3 (Qwen3 section)
- Cache+attention: `research/05-kvcache-attention.v2.md` §3 (Qwen3 entry), §9.1 attention arc

## 8. M1 validation status

- ✅ Shape tests at 3 size variants (`test_layer_shape.py`) — 4 tests
- ✅ Determinism (`test_layer_shape.py`)
- ✅ HF weight loader (`test_weight_loader.py`) — 2 tests
- ✅ Sub-op isolation vs HF at atol=1e-5 (`test_isolation_hf.py`) — 4 tests
- ✅ **Full layer numerical equivalence vs HF at atol=5e-4 (kickoff gate)** — max_abs_diff observed: 4.77e-7 (~1000× tighter than required)
- ✅ KV cache prefill+decode equivalence vs HF — prefill diff 4.77e-7, decode diff 2.38e-7
- ✅ AWQ W4A16 round-trip on real q_proj — median rel-err 0.157 (tolerance 0.20 documented; near-zero weights inflate rel-err)
