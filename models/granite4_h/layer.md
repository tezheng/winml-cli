# Granite 4 H — one decoder layer (B7)

## 0. At-a-glance

- **Signature:** 9:1 Mamba-2 : GQA hybrid with Granite μP scalars on all layers (residual_multiplier, attention_multiplier, embedding_multiplier)
- **Active params:** ~3B (granite-4.0-h-micro dense, no MoE)
- **Layer mix:** 40 layers (36 Mamba-2 SSM + 4 attn GQA, 9:1)
- **KV cache / token (bf16):** ~8 kB (4 attn layers × 8 kv heads × 64 head_dim × 4 B)

---

**Reference HF model:** `ibm-granite/granite-4.0-h-micro` (open, ~3B).

**HF source file:** `transformers/models/granitemoehybrid/modeling_granitemoehybrid.py` (1412 LoC).

## Architecture summary

Granite 4 H is a HYBRID architecture: per-layer dispatch between Mamba-2
SSM and standard GQA attention. For `granite-4.0-h-micro`:

- 40 layers
- pattern: `mamba×5, attention×1` repeating (so 5 of every 6 layers are
  Mamba; the 6th is attention).
- NO RoPE (position_embedding_type='nope')
- num_local_experts=0 → no MoE, just a single shared SwiGLU MLP per layer.
- Granite μP scaling: embedding_multiplier=12, logits_scaling=8,
  residual_multiplier=0.22, attention_multiplier=0.015625.

## Block structure (modeling_granitemoehybrid.py:1054-1095)

```
residual = x
x = input_layernorm(x)
x = (mamba | self_attn)(x)
x = residual + x * residual_multiplier

residual = x
x = post_attention_layernorm(x)
if has_experts:                          # granite-4-h-micro: False
    x = block_sparse_moe(x) + shared_mlp(x)
else:
    x = shared_mlp(x)
x = residual + x * residual_multiplier
```

This is PRE-norm + Granite μP residual scaling — identical to the B2a
Granite reference, except the token mixer is now per-layer-dispatched.

## Mamba layer (modeling_granitemoehybrid.py:256-732)

The `GraniteMoeHybridMambaLayer` is structurally identical to
`Mamba2Mixer` (the HF docstring explicitly says so). Differences are
narrow:
- d_mlp is fixed at 0 (no `_, _,` projection slice prefix — direct
  3-way `gate, x_BC, dt`).
- A_log is initialized as `log(arange(1, num_heads+1))` (line 318-319).

Our `api/ssm.Mamba2Mixer` already supports this with d_mlp=0 (default).
Weight loading uses the same HF tensor names but under `mamba.*` instead
of `mixer.*`.

## Attention layer (modeling_granitemoehybrid.py:121-186)

Standard GQA:
- `q_proj`, `k_proj`, `v_proj`, `o_proj` with `attention_bias` flag.
- Scale = `attention_multiplier` (NOT 1/sqrt(head_dim)).
  Source: line 130, 180. We use `AttentionSpec.attn_scale =
  attention_multiplier`.
- RoPE OPTIONAL: applied only when `position_embeddings` is not None
  (line 162-164), which is gated by `position_embedding_type == "rope"`
  at the model level (line 1141). For granite-4.0-h-micro, NoPE.

## Shared MLP (modeling_granitemoehybrid.py:753-776)

```
class GraniteMoeHybridMLP:
    input_linear  : Linear(hidden, 2 * shared_intermediate, bias=False)
    output_linear : Linear(shared_intermediate, hidden, bias=False)
    forward(x):
        h = input_linear(x).chunk(2, dim=-1)    # (gate, up)
        h = silu(gate) * up
        return output_linear(h)
```

This is **fused-gate-up SwiGLU** — same shape as Phi-3's gate_up_proj.
We use `FFNSpec.fused_gate_up=True`.

Weight mapping:
```
shared_mlp.input_linear.weight  -> blk.feedforward.gate_up_proj.weight
shared_mlp.output_linear.weight -> blk.feedforward.down_proj.weight
```

## Drifts caught BEFORE coding

1. `modeling_granitemoehybrid.py:130, 180` — attention scale is
   `attention_multiplier`, not `1/sqrt(head_dim)`.
2. `modeling_granitemoehybrid.py:1141` — `rotary_emb is None` when
   `position_embedding_type != "rope"`. NoPE is real.
3. `modeling_granitemoehybrid.py:768-775` — shared_mlp is FUSED-gate-up
   SwiGLU, NOT a 3-projection (gate/up/down) like Llama. Weight tensor
   names differ accordingly.
4. `modeling_granitemoehybrid.py:1084,1094` — residual_multiplier is on
   the SUBLAYER OUTPUT, before residual add. Same as B2a Granite.
5. `modeling_granitemoehybrid.py:1167` — embed output is scaled by
   `embedding_multiplier` BEFORE flowing into the first layer (we capture
   this in DecoderBlockSpec.embedding_scale but don't apply it inside
   the block — caller's job).
6. `modeling_granitemoehybrid.py:1088-1092` — when `has_experts=False`,
   the channel mixer is JUST shared_mlp. When experts > 0, the channel
   mixer is `shared_mlp(x) + block_sparse_moe(x)`. B7 lands the
   `num_local_experts == 0` case (granite-4-h-micro).
7. `modeling_granitemoehybrid.py:304` — Granite mamba in_proj output is
   `intermediate_size + conv_dim + num_heads`, NO extra `2*d_mlp` prefix.
   Matches our Mamba2Mixer with `d_mlp = 0`.
8. `modeling_granitemoehybrid.py:1037, 1041` — `input_layernorm` is the
   PRE-attn norm; `post_attention_layernorm` is misnomered: it's the
   PRE-FFN norm (despite its name suggesting POST). HF's same naming
   convention is used in Llama, so we mirror it.

## Numerical gate

`tests/models/granite4_h/test_numerical_hf.py` constructs the HF model
from a small GraniteMoeHybridConfig (no download), loads weights into our
api block (one mamba layer + one attention layer), and compares.

Target atol = 5e-4.
