# Mamba-2 — one decoder layer (B7)

## 0. At-a-glance

- **Signature:** Pure Mamba-2 SSD selective scan, no attention, no FFN sublayer (skip_ffn=True)
- **Active params:** ~2.7B (state-spaces/mamba2-2.7b reference)
- **Layer mix:** 64 SSM-mamba2 + 0 attn
- **KV cache / token (bf16):** `n/a (SSM state, not KV)`

---

**Reference HF model:** `state-spaces/mamba2-2.7b` (variants: 130M, 370M,
780M, 1.3B, 2.7B at `state-spaces/`).

**HF source file:** `transformers/models/mamba2/modeling_mamba2.py` (931 LoC).

## Block structure (modeling_mamba2.py:617-640)

```
Mamba2Block(hidden):
    residual = hidden
    h = norm(hidden)                                # Mamba2RMSNorm
    h = mixer(h, cache)                             # Mamba2Mixer
    return residual + h                             # NO FFN — single sublayer
```

This is fundamentally DIFFERENT from attention transformer blocks: there
is no second `pre_ffn_norm + ffn` sublayer. The DecoderBlock builds with
`skip_ffn=True` for canonical Mamba-2 layers.

## Mixer pipeline (modeling_mamba2.py:398-585, `torch_forward`)

Single forward pass on `[B, S, hidden_size]`:

1. **in_proj** projects to size `2*d_mlp + d_inner + conv_dim + num_heads`.
   For canonical Mamba-2 `d_mlp = 0` so the slice is `[gate, x_BC, dt]`.
   - `gate`: `[B, S, d_inner]`
   - `x_BC`: `[B, S, conv_dim]` where `conv_dim = d_inner + 2*n_groups*d_state`
   - `dt`: `[B, S, num_heads]`

2. **conv1d** is a 1D *depthwise* causal conv with kernel `conv_kernel=4`,
   `groups=conv_dim`, `padding=conv_kernel-1`. After the conv the output is
   sliced `[..., :seq_len]` to remain causal. **SiLU** activation.

3. Split conv output into `[x, B_unrep, C_unrep]`:
   - `x`: `[B, S, d_inner]` (reshape later to `[B, S, num_heads, head_dim]`)
   - `B_unrep`: `[B, S, n_groups, d_state]`
   - `C_unrep`: `[B, S, n_groups, d_state]`

4. **dt** = `softplus(dt + dt_bias).clamp(low, high)` (per-head, in fp32).

5. Discretize:
   - `A = -exp(A_log)` (per-head, fp32)
   - `A_disc = A * dt` (`[B, S, num_heads]`)
   - `x_disc = x * dt[..., None]`

6. **B and C repeat_interleave** by `num_heads // n_groups` along the
   group axis, giving shape `[B, S, num_heads, d_state]`.

7. **selective_scan** — SSD chunk-parallel scan via `api.ops.selective_scan`.
   Returns `(y, final_ssm_state)`.

8. **D-residual**: `y += D[..., None] * x_unscaled` (per-head skip).

9. Reshape `y` to `[B, S, d_inner]`.

10. **Gated RMSNorm**: `y = w * rms(y * silu(gate))` (`MambaRMSNormGated`).

11. **out_proj** to `[B, S, hidden_size]`.

## State cache (SSMStateCache)

Two tensors per layer:
- `conv_state`: `[B, conv_dim, conv_kernel]` — rolling window into conv1d
- `ssm_state`: `[B, num_heads, head_dim, d_state]` — recurrent state at the
  END of the most recent chunk

Prefill writes both; decode (NOT implemented in B7) advances them by one step.

## Defaults — `state-spaces/mamba2-2.7b`

| field | value |
|-------|-------|
| `hidden_size` | 2560 |
| `num_heads` | 80 |
| `head_dim` | 64 |
| `state_size` (d_state) | 128 |
| `conv_kernel` | 4 |
| `expand` | 2 |
| `n_groups` | 1 |
| `chunk_size` | 256 |
| `num_hidden_layers` | 64 |
| `vocab_size` | 50288 |

Invariant: `num_heads * head_dim == hidden_size * expand == d_inner` (5120)
and `num_heads % n_groups == 0`.

## Numerical gate

`tests/models/mamba2/test_numerical_hf.py` constructs:
1. The HF `Mamba2ForCausalLM.backbone.layers[0]` reference layer.
2. Our `build_mamba2_decoder_layer(cfg)` block.
3. Loads the HF state dict into our block via `load_hf_mamba2_layer`.
4. Runs both on the same fixed input.
5. Asserts `max_abs_diff < 5e-4`.

## Drifts caught

See B7 commit message for the full list of 8 drifts caught at
modeling_mamba2.py source-reading time, before writing the spec or mixer.
