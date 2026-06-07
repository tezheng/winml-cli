# DeepSeek-V2-Lite — decoder layer composition

**Source-of-truth**: `transformers/models/deepseek_v2/modeling_deepseek_v2.py`
plus `deepseek-ai/DeepSeek-V2-Lite/config.json`.

V2-Lite (16B-A2.4B): an MLA SLM with dense+MoE layer alternation. The
production model has 27 decoder layers; layer 0 is a dense SwiGLU FFN and
layers 1..26 are MoE blocks.

## Layer 0 (dense) — `first_k_dense_replace == 1`

```
residual := x
x      := pre_attn_norm(x)            # RMSNorm, eps=1e-6, STANDARD_W
x      := MLA-attention(x)            # see "Attention" below
x      := residual + x

residual := x
x      := pre_ffn_norm(x)             # RMSNorm
x      := SwiGLU(x; I=10944)          # dense FFN: gate_proj, up_proj, down_proj
x      := residual + x
```

Block envelope: PRE-norm. No μP residual scaling (unlike MiniCPM-3).
Source: `modeling_deepseek_v2.py:409-438`.

## Layer 1..26 (MoE) — `layer_idx >= first_k_dense_replace`

Same envelope as layer 0 but the FFN sublayer is a DeepseekV2Moe block:

```
def DeepseekV2Moe(x):
    residuals = x
    # Router — F.linear in fp32 even when the model is bf16.
    router_logits = F.linear(x.fp32, gate.weight.fp32)  # [B*S, E=64]
    scores = softmax(router_logits, dim=-1, dtype=fp32)
    # V2-Lite has n_group=1, topk_group=1 — degenerate group routing,
    # equivalent to greedy top-k over all 64 experts.
    topk_w, topk_idx = topk(scores, k=6, sorted=False)
    # norm_topk_prob=False on V2-Lite: NO renormalization.
    topk_w = topk_w * routed_scaling_factor  # routed_scaling_factor=1.0
    # Dense scatter — iterate hit experts.
    routed = experts(x, topk_idx, topk_w)    # [B*S, H]
    routed = routed.view(B, S, H)
    # Shared experts: a single SwiGLU FFN with intermediate = I * n_shared.
    # n_shared_experts = 2, so I_shared = 1408 * 2 = 2816.
    routed = routed + shared_experts(residuals)
    return routed
```

Source: `modeling_deepseek_v2.py:85-130` (DeepseekV2Moe) and `46-82`
(DeepseekV2Experts: gate_up_proj packed `[E, 2*I, H]`, down_proj `[E, H, I]`,
index_add scatter).

## Attention (both branches) — MLA with direct q_proj

V2-Lite sets `q_lora_rank = null` → no Q LoRA. The full Q projection is
direct, of shape `[H * qk_head_dim, hidden] = [16 * 192, 2048]`.

```
q       = q_proj(x)                                       # [B, S, H*qk_h]
q       = q.view(B, S, H, qk_h).transpose(1, 2)           # [B, H, S, qk_h]
q_nope, q_pe = split(q, [qk_nope=128, qk_rope=64], -1)

compressed = kv_a_proj_with_mqa(x)                        # [B, S, kv_lora+qk_rope]
c_kv, k_pe = split(compressed, [kv_lora=512, qk_rope=64], -1)
k_pe       = k_pe.view(B, 1, S, qk_rope)                  # SINGLE rope sub-head

kv         = kv_b_proj(kv_a_layernorm(c_kv))              # [B, S, H*(qk_nope+v_h)]
kv         = kv.view(B, S, H, qk_nope+v_h).transpose(1, 2)
k_nope, v  = split(kv, [qk_nope=128, v_h=128], -1)

# RoPE — INTERLEAVED basis (V2 uses complex-multiply on (real, imag) pairs)
# with YARN scaling. Source: modeling_deepseek_v2.py:271-284.
q_pe, k_pe = apply_rope_interleaved_yarn(q_pe, k_pe, position_embeddings)
k_pe       = k_pe.expand(B, H, S, qk_rope)                # broadcast to H heads

q = concat([q_nope, q_pe], -1)                            # [B, H, S, qk_h=192]
k = concat([k_nope, k_pe], -1)                            # [B, H, S, qk_h=192]

# SDPA — scale = qk_h ** -0.5 = (192)^-0.5.
# Source: modeling_deepseek_v2.py:335 (self.scaling = qk_head_dim ** -0.5).
out = softmax(q @ k.T * scale + causal_mask) @ v          # v shape [B, H, S, v_h=128]
out = out.transpose(1, 2).reshape(B, S, H*v_h)
out = o_proj(out)                                          # [B, S, hidden]
```

`attention_bias=False` on V2-Lite — no biases on any attention linears.

## RoPE — INTERLEAVED basis + YARN

V2 RoPE is implemented as complex multiplication on `(real, imag)` pairs of
the rotated channels:

```python
freqs_cis = polar(ones, freqs)                      # complex tensor
xq_       = view_as_complex(xq.reshape(*, -1, 2))   # pairs (re, im)
xq_out    = view_as_real(xq_ * freqs_cis).flatten(-2)
```

That's EXACTLY our `RoPEBasis.INTERLEAVED` (`api/ops.py::rope_apply` with
`basis="interleaved"`).

YARN scaling (V2-Lite config):
```
factor=40, beta_fast=32, beta_slow=1, mscale=0.707, mscale_all_dim=0.707,
original_max_position_embeddings=4096.
```
With `mscale == mscale_all_dim`, the attention_factor is 1.0 — YARN affects
only the inv_freq blend, not the cos/sin amplitude.

## Module → state dict mapping

```
model.layers.{L}.input_layernorm.weight          → blk.pre_attn_norm.weight
model.layers.{L}.post_attention_layernorm.weight → blk.pre_ffn_norm.weight

# MLA attention
model.layers.{L}.self_attn.q_proj.weight              → blk.attention.q_proj.weight
model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight  → blk.attention.kv_a_proj_with_mqa.weight
model.layers.{L}.self_attn.kv_a_layernorm.weight      → blk.attention.kv_a_layernorm.weight
model.layers.{L}.self_attn.kv_b_proj.weight           → blk.attention.kv_b_proj.weight
model.layers.{L}.self_attn.o_proj.weight              → blk.attention.o_proj.weight

# Layer 0 (dense FFN)
model.layers.0.mlp.gate_proj.weight   → blk.feedforward.gate_proj.weight
model.layers.0.mlp.up_proj.weight     → blk.feedforward.up_proj.weight
model.layers.0.mlp.down_proj.weight   → blk.feedforward.down_proj.weight

# Layer 1+ (MoE)
model.layers.{L}.mlp.gate.weight                    → blk.feedforward.gate.weight
model.layers.{L}.mlp.experts.gate_up_proj           → blk.feedforward.experts_gate_up
model.layers.{L}.mlp.experts.down_proj              → blk.feedforward.experts_down
model.layers.{L}.mlp.shared_experts.gate_proj.weight → blk.feedforward.shared_experts.gate_proj.weight
model.layers.{L}.mlp.shared_experts.up_proj.weight   → blk.feedforward.shared_experts.up_proj.weight
model.layers.{L}.mlp.shared_experts.down_proj.weight → blk.feedforward.shared_experts.down_proj.weight
```

## Numerical equivalence

`tests/models/deepseek_v2_lite/test_numerical_synthetic.py`:
- layer 0 (dense, default RoPE): max_abs_diff = 2.384e-7.
- layer 1 (MoE, default RoPE):   max_abs_diff = 2.384e-7.
- layer 0 (dense, YARN RoPE):    max_abs_diff = 2.384e-7.
- layer 1 (MoE, YARN RoPE):      max_abs_diff = 2.384e-7.

All sub-3e-7 — essentially machine epsilon at fp32.
