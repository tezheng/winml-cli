# DeepSeek-OCR-2 — LM-Decoder Layer

## 0. At-a-glance

- **Signature:** Standard MHA (no MLA) + softmax MoE LM portion (causal mask, not block-bidirectional) — vision encoder out of scope
- **Active params:** ~570M active / ~3B total (19%) (3B-MoE-A570M canonical checkpoint)
- **Layer mix:** 12 attn (1 dense + 11 MoE, softmax top-6 + 2 shared)
- **KV cache / token (bf16):** 60 kB (12 layers × 10 kv heads × 128 head_dim × 4 B)

---

## 1. Identity

- **Family:** DeepSeek-OCR-2 (DeepSeek-AI)
- **Variants in B8 scope:** deepseek-community/DeepSeek-OCR-2
  (LM-decoder portion — 3B-MoE-A570M canonical checkpoint).
- **Release date:** mid-2026 (per transformers v5.10.2 inclusion).
- **HF base model card:** https://huggingface.co/deepseek-community/DeepSeek-OCR-2
- **transformers source:**
  `transformers/src/transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py`
- **Original paper:** "DeepSeek-OCR: A Unified OCR Model with Vision Tokens
  and Block-Bidirectional Attention" (preprint; the "Visual Causal Flow"
  block-bidirectional mask is the paper's central architectural claim).

## 2. KEY DRIFT vs prompt assumptions

The B8 plan describes DeepSeek-OCR's LM-decoder as using **MLA attention
+ a block-bidirectional Visual Causal Flow mask + MoE channel mixer**.
The HF transformers v5.10.2 reference (`deepseek_ocr2`) IMPLEMENTS A
SUBSET — only the MoE channel mixer matches; the attention is plain
STANDARD MHA and the mask is plain CAUSAL.

| feature | original DeepSeek-OCR paper | HF deepseek_ocr2 v5.10.2 | our LM-decoder |
| --- | --- | --- | --- |
| attention kind | MLA (q/kv_lora_rank) | STANDARD MHA | STANDARD MHA |
| mask | Block-bidirectional (VCF) | CAUSAL | CAUSAL (default) + VCF hook |
| MoE router | softmax greedy | softmax greedy | softmax greedy |
| shared experts | yes | yes (n_shared=2) | yes (n_shared=2) |

Source verification:
- `modeling_deepseek_ocr2.py:1073-1138` (DeepseekOcr2TextAttention init
  + forward) — plain `nn.Linear(hidden, n_heads*head_dim, bias=...)` for
  q/k/v/o. No `q_a_proj` / `kv_a_proj_with_mqa` (MLA-style projections).
- `modeling_deepseek_ocr2.py:1339-1410` (DeepseekOcr2TextModel.forward)
  — uses `create_causal_mask(...)` exclusively. No vision/text split.
- `modeling_deepseek_ocr2.py:1212-1232` (route_tokens_to_experts) —
  softmax then top-k with greedy or group_limited_greedy. Identical to
  our `MoE._route_softmax`.

We honor source-first discipline: the LM-decoder factory and numerical
gate target the HF source exactly. The Visual Causal Flow mask is
exercised via:
- `api.types.MaskKind.BLOCK_BIDIRECTIONAL` enum
- `api.specs.AttentionSpec.block_bidirectional_mask` flag
- `api.attention.Attention.forward(vision_token_count=…)` argument

…in a shape-only test
(`test_layer_shape.py::test_block_bidirectional_mask_construction` and
`::test_block_bidirectional_mask_actually_bidirectional_in_vision_block`).

## 3. Architecture summary

- **Vision encoder:** dual-tower (SAM-style + Vision Transformer encoder).
  OUT OF SCOPE for B8.
- **Multi-modal projector:** single Linear from vision_dim → text hidden.
- **LM decoder:** 12 layers; layer 0 dense, layers 1-11 sparse MoE.
- **Canonical 3B-MoE-A570M dims:** hidden=1280, n_q=n_kv=10 (MHA),
  head_dim=128, intermediate=6848 (dense), moe_intermediate=896,
  n_routed_experts=64, n_shared_experts=2, num_experts_per_tok=6,
  rope_theta=10_000, vocab=129280, max_pos=8192, rms_norm_eps=1e-6,
  attention_bias=False, mlp_bias=False, tie_word_embeddings=False.

## 4. Decoder block diagram

```
                hidden_states  [B, S, 1280]
                  |
        +---------+---------+
        |                   |
        | input_layernorm (RMSNorm STANDARD_W)
        |                   |
        |   self_attn  (STANDARD MHA, NO MLA)
        |   ├── q_proj  (W [10*128, 1280], no bias)
        |   ├── k_proj  (W [10*128, 1280], no bias)
        |   ├── v_proj  (W [10*128, 1280], no bias)
        |   ├── RoPE    (SPLIT_HALF, theta=10000)
        |   ├── KV cache (CONTIGUOUS HND)
        |   ├── SDPA    (causal mask)
        |   └── o_proj  (W [1280, 10*128], no bias)
        |                   |
        +---->add (residual 1)
                  |
        +---------+---------+
        |                   |
        | post_attention_layernorm (RMSNorm STANDARD_W)
        |                   |
        |   if layer == 0: dense Qwen2-style SwiGLU MLP
        |       (gate_proj, up_proj, down_proj — all [in,out,no bias])
        |   else: MoE (softmax router, 64 routed + 2 shared experts)
        |       ├── gate.weight              [64, 1280]
        |       ├── softmax → top-6 selection
        |       ├── experts.gate_up_proj     [64, 2*896, 1280]
        |       ├── experts.down_proj        [64, 1280, 896]
        |       ├── shared_experts.{gate,up,down}_proj  (FFN ~1.8x moe_inter)
        |       └── out = routed_output + shared_experts(x)
        |                   |
        +---->add (residual 2)
                  |
                  y  [B, S, 1280]
```

## 5. Spec instantiation

Dense layer 0:
```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=AttentionSpec(
        n_q_heads=10, n_kv_heads=10, head_dim=128,
        kind=AttentionKind.STANDARD,
        qkv_layout=QKVLayout.SPLIT,
        mask_kind=MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        rope=RoPESpec(base_theta=10_000.0,
                      basis=RoPEBasis.SPLIT_HALF,
                      scaling=RoPEScaling.NONE),
    ),
    channel_mixer=FFNSpec(
        intermediate_size=6848,
        activation=Activation.SILU,
        gate_kind=GateKind.SWIGLU,
    ),
    ...
)
```

Sparse layer 1+:
```python
DecoderBlockSpec(
    ...
    channel_mixer=MoESpec(
        n_experts=64, top_k=6, n_shared_experts=2,
        router_kind="softmax", router_norm=False,
        score_correction_bias=False, group_routing=None,
        routed_scaling_factor=1.0,
        expert_ffn=FFNSpec(intermediate_size=896, ...),
    ),
)
```

## 6. Quirks

- **STANDARD MHA, NOT MLA.** Drift vs prompt — see §2.
- **Softmax router with topk_method='greedy'.** No sigmoid, no
  e_score_correction_bias.
- **router_norm=False.** The HF source does NOT divide top-k weights by
  their sum (`route_tokens_to_experts` at L1212-1232 multiplies by
  `routed_scaling_factor` and returns directly).
- **n_group=1 in canonical checkpoint.** Group routing reduces to plain
  greedy. The `group_limited_greedy` path is wired but inactive.
- **n_shared_experts=2.** Shared expert FFN has intermediate =
  moe_intermediate * 2 = 1792.
- **Mask is CAUSAL.** The Visual Causal Flow mask of the original paper
  is NOT in the HF source — see §2.

## 7. Weight-name mapping (HF → API)

### Dense layer (layer 0)

| HF tensor name | API tensor slot |
|---|---|
| `model.language_model.layers.0.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.language_model.layers.0.self_attn.{q,k,v,o}_proj.weight` | `blk.attention.{q,k,v,o}_proj.weight` |
| `model.language_model.layers.0.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.language_model.layers.0.mlp.{gate,up,down}_proj.weight` | `blk.feedforward.{gate,up,down}_proj.weight` |

### Sparse layer (layer 1+)

| HF tensor name | API tensor slot |
|---|---|
| `model.language_model.layers.{L}.input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `model.language_model.layers.{L}.self_attn.{q,k,v,o}_proj.weight` | `blk.attention.{q,k,v,o}_proj.weight` |
| `model.language_model.layers.{L}.post_attention_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `model.language_model.layers.{L}.mlp.gate.weight` | `blk.feedforward.gate.weight` |
| `model.language_model.layers.{L}.mlp.experts.gate_up_proj` | `blk.feedforward.experts_gate_up` |
| `model.language_model.layers.{L}.mlp.experts.down_proj` | `blk.feedforward.experts_down` |
| `model.language_model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.weight` | `blk.feedforward.shared_experts.{gate,up,down}_proj.weight` |

## 8. Source citations

- `transformers/models/deepseek_ocr2/configuration_deepseek_ocr2.py:184-264`
  (DeepseekOcr2TextConfig).
- `transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py`:
  - 1073-1138 DeepseekOcr2TextAttention (STANDARD MHA).
  - 1141-1154 DeepseekOcr2TextMLP (dense SwiGLU).
  - 1158-1194 DeepseekOcr2TextExperts (packed gate_up_proj + down_proj).
  - 1197-1242 DeepseekOcr2TextMoe (softmax router + shared_experts).
  - 1266-1309 DeepseekOcr2TextDecoderLayer (PRE-norm).
  - 1339-1410 DeepseekOcr2TextModel (create_causal_mask — STANDARD causal).

## 9. B8 validation status

- Shape tests (dense + MoE + VCF construction + VCF behavioral) — 4 tests
- Config from_hf_dict — 5 tests
- **Synthetic-weight numerical gate vs HF code path** (no checkpoint
  download required) — 2 tests; both at **1.19e-7** max_abs_diff
  (~4_200_000× tighter than required).
- **HF checkpoint numerical gate** (gated on network availability) —
  2 tests; dense layer 0 at **4.77e-7**, MoE layer 1 at **1.42e-7**.
