# DeepSeek-V3 (671B) — Decoder Layer

## 1. Identity

- **Family:** DeepSeek-V3 (DeepSeek). 671B total params, ~37B active per token.
- **HF model card:** `deepseek-ai/DeepSeek-V3` (weights ~ 700GB BF16).
- **transformers source:**
  `transformers/models/deepseek_v3/modeling_deepseek_v3.py`

## 2. Decoder block diagram

Identical IR to V3-Lite (which models the same module at synthetic dims):

```
        x  [B, S, 7168]
        |
   +----+----+
   |         |
   |     pre_attn_norm  (RMSNorm STANDARD_W, eps=1e-6)
   |         |
   |    attention(token_mixer) — MLA
   |    ├── q_a_proj → q_a_layernorm → q_b_proj   (q_lora_rank=1536)
   |    ├── kv_a_proj_with_mqa → kv_a_layernorm → kv_b_proj  (kv_lora_rank=512)
   |    ├── RoPE on qk_rope_head_dim=64 channels (SPLIT_HALF)
   |    └── SDPA (n_heads=128, qk_head=192, v_head=128) + o_proj
   |         |
   +---->add (residual 1)
        |
   +----+----+
   |         |
   |     pre_ffn_norm
   |         |
   |    Channel mixer — dispatched per layer_idx:
   |    ├── layer_idx < 3   → dense SwiGLU(intermediate=18432)
   |    └── layer_idx >= 3  → MoE:
   |         ├── gate: sigmoid+bias   (256 experts, top-8)
   |         ├── group routing: n_group=8, topk_group=4
   |         │      (per-group score = sum of top-2 entries; V3 algorithm)
   |         ├── norm_topk_prob=True → renormalize sum→1
   |         ├── routed_scaling_factor=2.5 multiply
   |         ├── 1 shared expert (intermediate = moe_intermediate * n_shared)
   |         └── experts.gate_up_proj [256, 2*2048, 7168]
   |             experts.down_proj    [256, 7168, 2048]
   |         |
   +---->add (residual 2)
        |
        y  [B, S, 7168]
```

## 3. Spec instantiation

Production dims (configuration_deepseek_v3.py:71-104):
```
hidden_size            = 7168
num_attention_heads    = 128
num_hidden_layers      = 61
intermediate_size      = 18432    # dense FFN size
moe_intermediate_size  = 2048
n_routed_experts       = 256
n_shared_experts       = 1
num_experts_per_tok    = 8
routed_scaling_factor  = 2.5
norm_topk_prob         = True
n_group                = 8
topk_group             = 4
first_k_dense_replace  = 3
kv_lora_rank           = 512
qk_nope_head_dim       = 128
qk_rope_head_dim       = 64
v_head_dim             = 128
q_lora_rank            = 1536
vocab_size             = 129280
rms_norm_eps           = 1e-6
attention_bias         = False
```

The block-spec build path is IDENTICAL to V3-Lite's
``models/deepseek_v3_lite/config.py::to_block_spec`` — only the dim
constants differ. V3-Lite's per-module numerical gate at
``tests/models/deepseek_v3_lite/test_isolation_hf.py::test_v3_moe_router_matches_hf_at_5e_4``
exercises the same module the production wrap composes.

## 4. Quirks (vs V3-Lite)

- All quirks of V3-Lite apply (sigmoid+bias router, weights gathered from
  bias-free sigmoid, group routing on top-2 per group, norm_topk_prob,
  routed_scaling_factor, MLA q-LoRA path with q_lora_rank=1536).
- **Production weights are 671B (~700GB BF16).** Numerical gate at this
  scale is impractical; the synthetic V3-Lite gate is the deterministic
  contract.
- `rope_interleave` is True by default in `configuration_deepseek_v3.py:102`.
  We mirror V3-Lite's choice of `rope_interleave=False` (SPLIT_HALF basis);
  the interleaved branch is deferred.
- `first_k_dense_replace = 3` — layers 0, 1, 2 are dense SwiGLU; layers 3+
  are MoE.

## 5. Source citations

- `transformers/models/deepseek_v3/modeling_deepseek_v3.py:139-237`  router
- `transformers/models/deepseek_v3/modeling_deepseek_v3.py:154-191`  experts
- `transformers/models/deepseek_v3/modeling_deepseek_v3.py:194-247`  MoE
- `transformers/models/deepseek_v3/modeling_deepseek_v3.py:482-526`  decoder layer
- `transformers/models/deepseek_v3/configuration_deepseek_v3.py:71-104` defaults
- Numerical gate at synthetic dims: B5
  ``tests/models/deepseek_v3_lite/test_isolation_hf.py``
