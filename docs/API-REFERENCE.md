# llm-layers — API Reference v4 (post-rollout)

**Status:** Post-rollout reference, current as of B10 (2026-06-08).
**Replaces:** `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md` (the pre-rollout design spec).
**Scope:** every public symbol exported by `api/*.py`, every per-field intent
discoverable in `models/<family>/config.py`, and the IR-drift history that
shaped them.

---

## §0  Preface

### 0.1 What this document is

This is the **post-rollout API reference** for the `api/` package — the
Intermediate Representation (IR) layer of the llm-layers project. It is
distinct from the pre-rollout design spec (v3) in three important ways:

1. **The spec v3 was aspirational.** It enumerated 38 layer-source axes, a
   16-op floor, ~18 dataclasses, and 30 target families — but most of those
   were forward-looking when written. This document covers what *actually
   landed* in `api/*.py` and `models/*/config.py` across batches M1 → B10.
2. **Every claim is line-cited.** Each architectural assertion is grounded in
   either `api/<file>:line` (the IR itself) or `modeling_*.py:line` (the
   upstream HF reference). Spec v3's prose is no longer the source of truth.
3. **It documents drift.** Where the rolled-out code diverged from the v3
   design — usually because a real-model investigation surfaced a behaviour
   the spec hadn't anticipated — §8 traces the commit, the source, and the
   verifying test.

### 0.2 Who this is for

You have read `README.md`. You understand that this project sits between two
worlds: above it, model-family code in `models/<family>/` builds
`DecoderBlockSpec` objects per layer; below it, this IR (the 16 logical ops,
the building-block `nn.Module`s, and the frozen specs that parameterise them)
implements those layers. You want to either:

- **Use** the IR: build a new layer for a new model family. Read §1–§4 plus
  the recipe in §6 closest to your architecture, then mimic the worked
  example in §7 most similar to yours.
- **Extend** the IR: add a new building block or spec field for an
  architecture that today's IR cannot express. Read §8 (recent drift) to
  understand the change pattern, then §9 (open extension points) to see what
  family is blocking what.

### 0.3 Differences from spec v3

The following extensions landed in `api/*.py` AFTER spec v3 was written. The
full chronological log lives in §8; the headline list:

| Batch | Extension | Why it landed |
|-------|-----------|---------------|
| M1 fix-pass | 7 new ops (`layer_norm`, `softmax`, `top_k`, `gather`, `scatter`, `conv1d`, `selective_scan` stub) | Close the IHV-consensus 16-op floor from research/03 v2. |
| M1 fix-pass | 6 new specs (`MoESpec`, `SSMSpec`, `SSDSpec`, `ConvSpec`, `GroupRoutingSpec`, `LayerScaleSpec`) | Close the 18-dataclass gap with spec v3. |
| B0.5 | `PLESpec`, cross-layer KV sharing (via `Gemma4Config.kv_source_layer_idx_map()` + `SharedLayerKVCache`), `partial_rotary_factor`, `attention_k_eq_v`, `qk_norm_fixed_scale` | Gemma 4 — five axes that diverged from Gemma 3. |
| B0.6 | `v_norm`, `v_norm_with_scale`, `partial_rotary_kind`, dropped fixed-scale absorption | Gemma 4 numerical-gate reading of `modeling_gemma4.py`. |
| B2a | `FUSED` QKV layout, `fused_gate_up`, `LongRoPEParams`, `partial_rotary_kind="prefix"`, μP `residual_scale` | Phi-3 / Phi-4 family + Granite. |
| B2b | MLA fields (`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`), `MLA_LATENT`, `ContiguousKVCache(v_head_dim=...)` | MiniCPM-3 + DeepSeek-V2/V3 family. |
| B3 | `POST` norm, sandwich-norm verifications, `attn_logit_softcap` in `sdpa` | OLMo 2 + Gemma 2. |
| B4 | `INTERLEAVED` RoPE basis (complex-multiply) | Llama 4 Scout. |
| B5 | `YarnRoPEParams`, `IndexerSpec`, real `MoE` module (softmax + sigmoid+bias), MLA `q_lora_rank=None` path | DeepSeek-V2-Lite, V3, V3.2. |
| B7 | `api/ssm.py` with `Mamba2Mixer` + `selective_scan`, `SSMStateCache`, `SSDSpec.token_mixer`, `skip_ffn`, `TokenMixerKind` enum | Mamba-2 + Granite-4-H hybrid. |
| B8 | `mrope_section`, `rope_apply_mrope`, `MaskKind.BLOCK_BIDIRECTIONAL`, `block_bidirectional_mask`, `vision_token_count` | Qwen2.5-VL, DeepSeek-OCR-2, GOT-OCR 2.0. |

### 0.3b The rollout trajectory in a paragraph

M1 landed the Qwen3 anchor — a single working family with a partial IR
(8 ops, 4 dataclasses, 8 building-block modules). The M1 fix-pass closed
two known gaps: the 16-op floor (7 stub ops added) and the 18-dataclass
target (6 dataclasses added for forward compatibility with B5+ MoE and
B7 SSM work). B0.5 took the IR to Gemma 4 — that family alone added
6 specs, 4 ops or op variants, 4 building blocks (PRE_AND_POST, PLE,
SharedLayerKVCache, partial-RoPE). B0.6 was a fix-pass on B0.5 that
caught 5 axes the spec hadn't matched HF for. B1 landed four
Llama-shaped families (Llama 3, Mistral, SmolLM3, TinyLlama) with
minimal IR extension (NoPE was the only one). B2a covered Granite μP +
Phi-3 FUSED/LongRoPE; B2b added MLA for MiniCPM-3. B3 added POST-norm
(OLMo 2) + softcap-in-sdpa (Gemma 2/3). B4 added INTERLEAVED RoPE
(Llama 4) + Ministral SWA. B5 was the big batch: YARN, IndexerSpec,
real MoE module (softmax + sigmoid+bias), MLA q_lora_rank=None path,
DSA shape-only. B6 added Mixtral, Qwen3-MoE, OLMoE, V3-MoE production
wrapper — all exercising the B5 MoE without further IR extension.
B7 added Mamba-2 + Granite-4-H hybrid (the first SSM landing). B8 added
M-RoPE + Visual Causal Flow + GOT-OCR 2.0 / Qwen2.5-VL / DeepSeek-OCR-2.
B9 landed Moshi 7B + Voxtral 3B (audio LMs). B10 was quantization
(GGUF Q4_K_M + FP8 E4M3 + MXFP4 + LiteRT W4A8).

### 0.4 How this doc cross-references

- Per-op signatures cite `api/ops.py:<line>` or the relevant building block.
- Per-spec field defaults are quoted from `api/specs.py:<line>`.
- Per-field "which families set this" cells point to `models/<family>/config.py:<line>`.
- Per-field "when this landed" cells cite the introducing commit by the
  shortened SHA (full SHAs are listed at the end of §8).
- Per-field "why" cells cite the source-of-truth in
  `modeling_*.py:<line>` (HF Transformers) wherever the IR mirrors HF behaviour.

---

## §1  The 16 Logical Ops (`api/ops.py`)

The op floor synthesises the IHV-consensus minimum primitives (see
`research/03-ihv-opsets.v2.md §3`). All ops are pure functions of tensors;
backends (ONNX / QNN / OpenVINO) re-implement these signatures. M1 landed the
Qwen3 subset (`silu`, `add`, `mul`, `linear`, `rms_norm`, `embed`, `lm_head`,
`rope_apply`, `sdpa`); the M1 fix-pass added the other 7 to close the 16-op
floor. Subsequent batches added `rope_apply_partial` (B0.5),
`gelu_pytorch_tanh` (B0.5), `rope_apply_mrope` (B8), and the
`selective_scan` real implementation (B7). The current count is **19** —
the 16-op floor plus `rope_apply_partial`, `rope_apply_mrope`, and
`gelu_pytorch_tanh`. (Audit 2026-06-08 corrected from "18".)

### 1.1 `silu(x)` — `api/ops.py:17`

```python
def silu(x: torch.Tensor) -> torch.Tensor: return F.silu(x)
```

- **Shape:** identity, any tensor in → same shape out.
- **Semantic:** SwiGLU's gate activation. Equivalent to `x * sigmoid(x)`.
- **Where used:** `FeedForward.forward` (SWIGLU path), inside MoE expert
  forward, inside `Mamba2Mixer` (conv-activation and gated-norm gate).
- **Subtleties:** none — pure pointwise.

### 1.2 `add(x, residual, scale=None)` — `api/ops.py:21`

```python
def add(x: torch.Tensor, residual: torch.Tensor,
        scale: Optional[float] = None) -> torch.Tensor:
```

- **Shape:** `x` and `residual` must broadcast; output is `x + scale*residual`
  (or `x + residual` when `scale is None`).
- **Semantic:** the residual-add primitive. The `scale` knob exists to absorb
  a per-sublayer multiplier *before* the add — used by Granite μP and
  MiniCPM-3 (their `residual_multiplier` / `scale_depth / sqrt(L)`).
- **Where used:** every `DecoderBlock.forward` for the post-attn and post-FFN
  residual adds (`block.py:240, 255`). The PLE injection at `block.py:265`
  also goes through this op.
- **Subtleties:** `DecoderBlock` applies the scale *inside* `self._residual_scale`
  multiplication on `attn_out` / `ffn_out` rather than via this `scale=` arg
  — the caller's `scale=` is currently UNUSED inside `api/block.py` because
  the scale lives on the sublayer output, not on the residual term.

### 1.3 `mul(a, b)` — `api/ops.py:28`

```python
def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: return a * b
```

- **Shape:** `a` and `b` must broadcast; output broadcast shape.
- **Semantic:** pointwise multiply. Used as the gate × up multiply in
  SwiGLU / GeGLU FFN, and as the per-layer-input multiply inside the PLE
  injection block (`block.py:262`).
- **Subtleties:** explicitly an op (not just `*`) so backends can lower it
  to a fused multiply when paired with the surrounding activations.

### 1.4 `linear(x, weight, bias=None)` — `api/ops.py:32`

```python
def linear(x: torch.Tensor, weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
```

- **Shape:** `x: [..., K]`, `weight: [N, K]`, `bias: [N]` or `None`; output
  `[..., N]`.
- **Semantic:** `F.linear`. Quantised linears use a wrapper module in
  `api/quant.py` — the bare op is fp32 / bf16 / fp16.
- **Where used:** every `nn.Linear` projection inside Attention / MLA /
  FeedForward / MoE / `Mamba2Mixer` ultimately lowers to this. The
  routing path in `MoE._route_softmax` / `_route_sigmoid_plus_bias`
  explicitly calls `F.linear(x_fp32, weight_fp32)` (feedforward.py:207, 248,
  365) to mirror HF's fp32-promoted router math.

### 1.5 `rms_norm(x, weight, eps, mode="standard_w")` — `api/ops.py:38`

```python
def rms_norm(x, weight, eps, mode="standard_w") -> torch.Tensor:
```

- **Shape:** `x: [..., H]`, `weight: [H]` (or compatible broadcast), output
  same shape as `x`.
- **Semantic:** root-mean-square LayerNorm without the mean-subtract.
  `mode="standard_w"` ⇒ `y = (x / rms(x)) * w`; `mode="one_plus_w"` ⇒
  `y = (x / rms(x)) * (1 + w)` (Gemma 1/2/3 RMSNorm signature).
- **Subtleties — dtype promotion:** the variance and rsqrt are computed in
  `fp32` regardless of input dtype; the final multiply and the
  `.to(orig_dtype)` cast happen in `x`'s dtype. This mirrors
  `modeling_gemma2.py:49-65` / `modeling_qwen3.py` exactly. Without this
  promotion, bf16 rsqrt drift is visible at atol 5e-4 on real model gates.
- **Where used:** `RMSNorm.forward`, `QKNorm.forward`, the v_norm path in
  `Attention.forward` (`attention.py:365`), and `_MambaRMSNormGated`
  inside SSM (`ssm.py:88-90`).

### 1.6 `layer_norm(x, weight, bias, eps)` — `api/ops.py:313`

```python
def layer_norm(x, weight, bias, eps) -> torch.Tensor:
```

- **Shape:** `x: [..., H]`, `weight: [H]`, `bias: [H]`, output same shape.
- **Semantic:** standard `F.layer_norm` over the last dim, fp32-promoted then
  cast back. Added at M1 fix-pass for spec coverage; the building-block
  `RMSNorm` in `api/norm.py` does NOT consume this — only Phi-3-small
  declares `NormSpec.kind = LAYER`, and that path is shape-only today
  (the constructor `RMSNorm.__init__` raises on `kind != RMS`,
  `norm.py:17`).
- **Where used:** none in production paths today — reserved for Phi-3-small
  / hybrid families that need full LayerNorm.

### 1.7 `softmax(x, dim=-1, dtype=None)` — `api/ops.py:325`

```python
def softmax(x, dim=-1, dtype=None) -> torch.Tensor:
```

- **Shape:** identity; reduces along `dim` to produce a probability simplex.
- **Semantic:** `F.softmax`, with optional fp32 promotion via `dtype=`. The
  fp32-promotion is critical for MoE routers and for Gemma 2 softcap
  attention (`ops.py:305` does `F.softmax(..., dtype=fp32).to(q.dtype)`).
- **Where used:** the MoE router does its own `softmax(dim=-1, dtype=fp32)`
  inline (feedforward.py:209); the `sdpa` softcap path goes through this op.

### 1.8 `top_k(x, k, dim=-1)` — `api/ops.py:331`

```python
def top_k(x, k, dim=-1) -> tuple[torch.Tensor, torch.Tensor]:
```

- **Shape:** input `[..., E]`, returns `(values, indices)` each `[..., k]`.
- **Semantic:** wraps `torch.topk`. MoE router returns top-k expert indices
  for both V2 (softmax) and V3 (sigmoid+bias) routers. Used directly inside
  `MoE._route_softmax` and `_route_sigmoid_plus_bias`
  (feedforward.py:213, 256, 282).
- **Subtleties:** `sorted=False` is passed everywhere in the MoE path —
  expert-order is irrelevant after expert dispatch.

### 1.9 `gather(x, dim, index)` — `api/ops.py:336`

```python
def gather(x, dim, index) -> torch.Tensor: return torch.gather(x, dim, index)
```

- **Shape:** `x` and `index` must agree on all dims except `dim`; output
  shape equals `index.shape`.
- **Semantic:** thin wrapper. Used by the V3 router to pull the bias-free
  router_probs for the (already-chosen) top-k indices
  (feedforward.py:287 — `router_probs.gather(1, topk_idx)`).

### 1.10 `scatter(x, dim, index, src)` — `api/ops.py:340`

```python
def scatter(x, dim, index, src) -> torch.Tensor:
    out = x.clone()
    return out.scatter(dim, index, src)
```

- **Shape:** non-in-place scatter; returns a new tensor of `x`'s shape.
- **Semantic:** floor-op for backends. Today's MoE path uses
  `tensor.scatter_` and `index_add_` directly (feedforward.py:224, 271, 322)
  rather than this op, because in-place semantics matter for the dense-scatter
  dispatch performance.
- **Status:** mostly a placeholder against the 16-op floor.

### 1.11 `conv1d(x, weight, bias, stride, padding, groups)` — `api/ops.py:347`

```python
def conv1d(x, weight, bias=None, stride=1, padding=0, groups=1)
        -> torch.Tensor:
```

- **Shape:** `x: [B, C_in, L]`, `weight: [C_out, C_in/groups, K]`, output
  `[B, C_out, L_out]` with the standard convolution length formula.
- **Semantic:** Mamba-2 uses this for the causal depthwise conv at the front
  of the SSM block. The full causal trick is done by the caller:
  `padding=conv_kernel-1` plus `[..., :seq_len]` slice (`ssm.py:244`).
- **Subtleties — the depthwise trick:** Mamba-2's `Conv1d` has
  `groups=conv_dim`, so each channel convolves independently. Source:
  `modeling_mamba2.py:155-162` (Mamba2Mixer init).

### 1.12 `selective_scan(...)` — `api/ops.py:425`

```python
def selective_scan(
    hidden_states: torch.Tensor,    # [B, S, num_heads, head_dim]
    A: torch.Tensor,                # [B, S, num_heads]
    B: torch.Tensor,                # [B, S, num_heads, d_state]
    C: torch.Tensor,                # [B, S, num_heads, d_state]
    chunk_size: int,
    initial_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

- **Shape:** see signature. Returns `(y, ssm_state)` with
  `y: [B, S, num_heads, head_dim]` and
  `ssm_state: [B, num_heads, head_dim, d_state]`.
- **Semantic:** the Mamba-2 SSD chunk-parallel selective scan. Pure PyTorch
  reference implementation; faithfully mirrors
  `modeling_mamba2.py::Mamba2Mixer.torch_forward` L503-577. Prioritises
  correctness over speed.
- **The four conceptual steps** (HF inline comments at lines 527-565):
  1. `Y_diag` — intra-chunk causal SSD via `L = exp(segment_sum(A))` and
     `G = C·B^T`.
  2. `states` per-chunk — right-term decay
     `B_decay = B * exp(A_cumsum_last - A_cumsum)`, summed over the chunk.
  3. Inter-chunk recurrence —
     `decay_chunk = exp(segment_sum(pad(A_cumsum_last, (1,0))))`.
  4. `Y_off = C @ states * exp(A_cumsum)` — left-term decay.
- **What the caller has to do first:** the caller pre-multiplies `x*dt` and
  `A_log_neg*dt` before calling — i.e. `selective_scan` consumes already-
  discretised activations.
- **What the caller has to do after:** apply the `D`-residual
  (`D[..., None] * x_unscaled`) and the gated RMSNorm (`norm(y, gate)`).
- **`_segment_sum` and `_reshape_into_chunks`** are private helpers
  (`api/ops.py:359-422`) that the SSM mixer uses indirectly through this
  op. They are not part of the public floor.

### 1.13 `embed(ids, weight, scale=None)` — `api/ops.py:60`

```python
def embed(ids, weight, scale=None) -> torch.Tensor:
```

- **Shape:** `ids: [B, S]` long, `weight: [V, H]`, output `[B, S, H]`.
- **Semantic:** vocab embedding lookup with optional output scale.
- **Subtleties:** the `scale` knob lets Gemma's `sqrt(D)` embedding scale
  ride on the op (Gemma 2/3/4 all set `embedding_scale = sqrt(hidden_size)`
  on the `DecoderBlockSpec`; the embedding layer applies it). MiniCPM-3
  passes its `scale_emb` (≈12 for 4B) through the same knob.

### 1.14 `lm_head(x, weight, scale=None, softcap=None)` — `api/ops.py:71`

```python
def lm_head(x, weight, scale=None, softcap=None) -> torch.Tensor:
```

- **Shape:** `x: [B, S, H]`, `weight: [V, H]`, output `[B, S, V]`.
- **Semantic:** projection from hidden to logits with optional pre-scale and
  optional tanh-softcap. Gemma 2/3/4 set `softcap = 30.0`; MiniCPM-3 sets
  `scale = dim_model_base / hidden_size`.
- **Subtleties:** the softcap order is `logits = scale*logits` (if `scale`)
  then `logits = cap * tanh(logits / cap)` (if `softcap`). This matches
  `modeling_gemma2.py:537-540`.

### 1.15 `rope_apply(q, k, cos, sin, basis="split_half")` — `api/ops.py:93`

```python
def rope_apply(q, k, cos, sin, basis="split_half")
        -> tuple[torch.Tensor, torch.Tensor]:
```

- **Shape:** `q: [B, S, H, Dh]`, `k: [B, S, Hk, Dh]`; cos/sin shape depends on
  basis (see below).
- **Semantic:** apply rotary position embedding.
- **`basis="split_half"`** (default — Qwen / Llama / Gemma / most modern
  SLMs): `cos`/`sin` span `[S, Dh]`. The `_rotate_half` helper pairs
  `i ↔ i+Dh/2` and the rotation is the standard
  `q*cos + rotate_half(q)*sin`.
- **`basis="interleaved"`** (Llama 4, DeepSeek-V2 with `rope_interleave=False`
  meaning complex-multiply): `cos`/`sin` span `[S, Dh/2]` — one entry per
  `(real, imag)` channel pair. The op reshapes the last dim from `Dh` to
  `(Dh/2, 2)`, treats each pair as `(re, im)`, and applies the standard
  complex multiplication
  `(re + j*im) * (cos + j*sin)`. Source mirror: `modeling_llama4.py:239-254`.
- **Subtleties — the gqa head-broadcast:** this op does NOT broadcast `k`
  across query heads — that's `sdpa`'s job. `q.shape[2] == H` and
  `k.shape[2] == Hk` are both fine.

### 1.16 `rope_apply_partial(q, k, cos, sin, partial_rotary_factor, basis)` — `api/ops.py:213`

```python
def rope_apply_partial(q, k, cos, sin, partial_rotary_factor,
                       basis="split_half") -> tuple[Tensor, Tensor]:
```

- **Shape:** `q: [B, S, H, Dh]`, `k: [B, S, Hk, Dh]`,
  `cos/sin: [S, int(Dh * pr_factor)]`.
- **Semantic:** apply RoPE to the first `int(pr_factor * Dh)` channels of
  each head; pass the remaining channels through unchanged. Used by Phi-3 /
  Phi-4 prefix-partial-RoPE.
- **Subtleties — prefix vs proportional dispatch:** the op handles the
  *prefix* case (rotate the first `Dh_rot` channels, pass through the rest).
  The *proportional* case (Gemma 4 global) handles the partial via inv_freq
  zero-padding so the full-dim `rope_apply` does the right thing — see
  `RoPE.__init__` (`rope.py:178-192`) for the geometry-difference comment.

### 1.17 `rope_apply_mrope(q, k, cos, sin, mrope_section)` — `api/ops.py:148`

```python
def rope_apply_mrope(q, k, cos, sin, mrope_section)
        -> tuple[torch.Tensor, torch.Tensor]:
```

- **Shape:** `q: [B, S, H, Dh]`, `k: [B, S, Hk, Dh]`, `cos/sin: [3, B, S, Dh]`
  (the leading "3-axis" dim is temporal / height / width).
- **Semantic:** Qwen2.5-VL's multimodal RoPE. The op stitches the 3 per-axis
  cos/sin tables into a single `[B, S, Dh]` tensor by taking, for each
  channel band, the per-axis (T/H/W) table whose row index `i % 3` matches
  the band's position. `mrope_section * 2` is the per-band channel-count
  list (the `*2` because `rotate_half` pairs the two halves of `Dh`).
- **Semantic verification:** `modeling_qwen2_5_vl.py:596-606`. The signature
  shape comment lives at `api/ops.py:170-172`.

### 1.17b The M-RoPE stitching math, line by line

The interesting bit of `rope_apply_mrope` is the stitching loop
(`ops.py:194-203`). The 3-axis (T/H/W) cos/sin tables enter as
`[3, B, S, Dh]`. The output should be `[B, S, Dh]` — i.e. the leading
"3-axis" dim must be consumed. The HF idiom (mirrored here) is:

```python
sec = list(mrope_section) * 2                  # e.g. (16,24,24) -> (16,24,24,16,24,24)
cos_chunks = cos.split(sec, dim=-1)             # 6 tensors of shape [3, B, S, sec_i]
sin_chunks = sin.split(sec, dim=-1)
cos_stitched = torch.cat(
    [m[i % 3] for i, m in enumerate(cos_chunks)], dim=-1,
)                                                # [B, S, Dh]
```

The `i % 3` picks T for `i ∈ {0, 3}`, H for `i ∈ {1, 4}`, W for `i ∈ {2, 5}`.
The duplicate (`* 2`) reflects rotate_half pairing across the two halves of
`head_dim` — the first half-band uses (T,H,W), the second half-band uses
(T,H,W) again.

The stitched cos/sin then enters the standard `_rotate_half` multiply:
```python
cos_b = cos_stitched.unsqueeze(2)      # [B, S, 1, Dh]
sin_b = sin_stitched.unsqueeze(2)
q_rot = q * cos_b + _rotate_half(q) * sin_b
```

The unsqueeze at axis 2 (where the head axis lives in `[B, S, H, Dh]`)
broadcasts cos/sin across all heads. The result is `[B, S, H, Dh]` — the
input shape preserved.

### 1.18 `gelu_pytorch_tanh(x)` — `api/ops.py:248`

```python
def gelu_pytorch_tanh(x) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")
```

- **Shape:** identity.
- **Semantic:** GeGLU's gate activation in Gemma 1/2/3/4. Equivalent to
  `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
- **Where used:** `FeedForward.forward` (GEGLU path), the PLE injection block
  (`block.py:261`).

### 1.18b `_segment_sum(input_tensor)` — `api/ops.py:390` (private helper)

```python
def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
```

- **Shape:** input `[..., chunk_size]`, output `[..., chunk_size, chunk_size]`.
- **Semantic:** numerically stable segment cumulative sum. The output is the
  strict lower-triangular cumulative sum with `-inf` above the diagonal so
  that `exp(_segment_sum(A))` yields a zero entry above the diagonal — the
  classic Mamba-2 SSD decay matrix.
- **Used by:** `selective_scan` only — used twice (once to build the
  intra-chunk decay matrix `L`, once to build the inter-chunk recurrence
  `decay_chunk`).
- **Verified against:** `modeling_mamba2.py:71-88` (`segment_sum`).

### 1.18c `_reshape_into_chunks(input, pad_size, chunk_size)` — `api/ops.py:374` (private helper)

```python
def _reshape_into_chunks(input_tensor, pad_size, chunk_size) -> torch.Tensor:
```

- Shape: pads along axis 1 by `pad_size`, then reshapes
  `[B, S+pad, ...]` to `[B, n_chunks, chunk_size, ...]`.
- Used only inside `selective_scan` (line 488-491).
- Verified against `modeling_mamba2.py:51-68`.

### 1.18d `_pad_tensor_by_size(input, pad_size)` — `api/ops.py:359` (private helper)

Sequence-axis pad helper used by `_reshape_into_chunks`. Handles rank-3
(`[B, S, H]`) and rank-4 (`[B, S, H, D]`) inputs. Mirrors
`modeling_mamba2.py:40-48`.

### 1.18e `_rotate_half(x)` — `api/ops.py:85` (private helper)

```python
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)
```

The `(x1, x2) ↦ (-x2, x1)` 90-degree rotation used by SPLIT_HALF RoPE. Used
inside `rope_apply` and `rope_apply_mrope`.

### 1.19 `sdpa(q, k, v, attn_mask, is_causal, scale, logit_softcap)` — `api/ops.py:253`

```python
def sdpa(q, k, v, attn_mask=None, is_causal=False,
         scale=None, logit_softcap=None) -> torch.Tensor:
```

- **Shape:** `q: [B, Hq, S_q, Dh]`, `k: [B, Hk, S_k, Dh]`,
  `v: [B, Hk, S_k, Dv]`, output `[B, Hq, S_q, Dv]`.
- **Semantic:** scaled dot-product attention with **built-in GQA support**.
- **GQA broadcast (the subtle bit):** when `Hq != Hk`, the op
  `repeat_interleave`s K and V to match Q heads. The repeats factor is
  `Hq // Hk`. This is done at op level rather than the caller because
  `F.scaled_dot_product_attention` does NOT support GQA — every callsite
  would otherwise duplicate this code.
- **Softcap dispatch:** when `logit_softcap is not None`, the op falls back to
  a manual `matmul → scale → tanh-softcap → mask-add → fp32-softmax →
  matmul` path. This matches the Gemma 2 eager attention order
  (`modeling_gemma2.py:212-225`). Default `scale` is `Dh**-0.5` when None.
- **Why no `is_causal=True` in production:** the attention block always
  builds an explicit additive mask because `is_causal=True` would
  mis-align when `start_pos > 0` (the diagonal lives at the top-left of
  `[S_q × S_k]`, but for chunked prefill the right region is
  `[start_pos, start_pos+S_q] × [0, start_pos+S_q]`).

---

## §2  Enums (`api/types.py`)

Every enum class lives in `api/types.py` and has no PyTorch dependency.

### 2.1 `AttentionKind` (`types.py:6`)

- `STANDARD` — MHA / GQA / MQA, distinguished by `n_kv_heads`. Used by every
  family except DeepSeek / MiniCPM and the SSM-only families.
- `MLA` — Multi-head Latent Attention (DeepSeek-V2/V3, MiniCPM-3). Triggers
  the `_init_mla` path in `api/attention.py:214`.
- `LINEAR_RETENTION`, `LINEAR_DELTANET`, `LINEAR_GLA`, `DIFFERENTIAL` —
  declared for v3-spec parity, NOT yet wired (every callsite raises
  `NotImplementedError` at `attention.py:30`).
- `DSA` — DeepSeek-V3.2 Sparse Attention with Lightning Indexer. Triggers
  the `_init_mla` path PLUS allocates the indexer Q/K projections, but
  `Attention.forward` raises (shape-only at B5).

### 2.2 `TokenMixerKind` (`types.py:16`)

Top-level dispatch enum added at B7. Currently exercised only for
documentation / typing — the actual dispatch in `DecoderBlock` is by
`isinstance(spec.token_mixer, (AttentionSpec, SSDSpec))`.

- `ATTENTION` (default — `AttentionSpec` is the token mixer for everything
  pre-B7).
- `SSM_MAMBA1` — `SSMSpec`, Mamba-1 selective scan. **Reserved**.
- `SSM_MAMBA2` — `SSDSpec`, Mamba-2 SSD form. B7 landing target.
- `SSM_GRIFFIN` — RecurrentGemma. **Reserved**.
- `SSM_RWKV` — RWKV-7 Goose. **Reserved**.

### 2.3 `QKVLayout` (`types.py:37`)

- `SPLIT` — separate `q_proj`, `k_proj`, `v_proj`. Almost everyone.
- `FUSED` — one big QKV projection of size `(Hq + 2*Hk) * Dh`, sliced at
  forward time. Phi-3 / Phi-3-small.
- `MLA_LATENT` — MLA's compressed latent layout
  (`q_a_proj → q_a_layernorm → q_b_proj`, etc.). DeepSeek / MiniCPM.

### 2.4 `MaskKind` (`types.py:43`)

- `CAUSAL` — standard lower-triangular causal mask.
- `SWA` — sliding-window-attention mask. Gemma 2/3/4 local layers, Mistral
  v0.2 (currently off in v0.3), Ministral, SmolLM3 (per-layer dispatch).
- `SWA_GLOBAL_ALT` — declared for spec parity; not used.
- `SINK` — declared; not used.
- `FULL` — declared; not used.
- `CUSTOM` — declared; not used.
- `BLOCK_SPARSE` — Phi-3-small dense/blocksparse alternation. Construction
  succeeds; forward raises (shape-only).
- `BLOCK_BIDIRECTIONAL` — B8 Visual Causal Flow mask (DeepSeek-OCR). The
  `vision_token_count` prefix is bidirectional; trailing text is causal.

### 2.5 `QKNormPhase` (`types.py:67`)

- `NONE` — no QK-norm (Llama 3, Mistral, Mixtral, Granite, OLMoE).
- `PRE_ROPE` — Qwen3, Gemma 3, Gemma 4, OLMo 2.
  Verified PRE-RoPE in `research/05` v2.
- `POST_ROPE` — declared; not used.

### 2.6 `QKNormShape` (`types.py:73`)

- `NONE` — no QK-norm.
- `PER_HEAD_DH` — weight shape `[head_dim]`. Qwen3, Gemma 3, Gemma 4. The
  norm is applied independently per head.
- `FULL_HDH` — weight shape `[n_heads * head_dim]`. OLMo 2, OLMoE. The norm
  is applied to the flattened `[B, S, H*Dh]` tensor before reshape to
  `[B, S, H, Dh]`.

### 2.7 `NormKind` (`types.py:79`)

- `RMS` — every production family today.
- `LAYER` — declared for Phi-3-small parity; the `RMSNorm` constructor
  raises on `LAYER` so this is shape-only.

### 2.8 `NormWeightMode` (`types.py:84`)

- `STANDARD_W` — `y = x_normed * w`. Llama, Qwen, Gemma 4, Mistral, OLMo 2,
  Mamba-2, ... — almost everyone.
- `ONE_PLUS_W` — `y = x_normed * (1 + w)`. Gemma 1/2/3 only. **Note:** Gemma
  4 *dropped* `ONE_PLUS_W` and uses `STANDARD_W` (B0.6 drift; see §8).

### 2.9 `NormPosition` (`types.py:89`)

- `PRE` — pre-norm (standard). Llama, Qwen, Mistral, ...
- `POST` — post-norm. OLMo 2 only.
- `PRE_AND_POST` — sandwich norm. Gemma 2 / 3 / 4.

### 2.10 `Activation` (`types.py:95`)

- `SILU` — SwiGLU's gate activation (almost everyone with SwiGLU).
- `GELU` — Gemma's GeGLU gate activation.
- `GEGELU` — Phi-3-small's GE-GELU (gated GELU with a "limit" clamp).
  Declared for spec parity; not wired.
- `RELU2` — declared; not used.

### 2.11 `GateKind` (`types.py:102`)

- `SWIGLU` — `down(silu(gate(x)) * up(x))`. Llama, Qwen, Mistral, DeepSeek,
  ...
- `GEGLU` — `down(gelu_pytorch_tanh(gate(x)) * up(x))`. Gemma family.
- `GELU_ONLY` — declared; not used.
- `RELU2_ONLY` — declared; not used.

### 2.12 `RoPEBasis` (`types.py:109`)

- `SPLIT_HALF` — GPT-NeoX style. `rotate_half` pairs `i ↔ i+Dh/2`. Default
  for Llama, Qwen, Gemma, Mistral, DeepSeek-V3 (with `rope_interleave=False`),
  Phi, etc.
- `INTERLEAVED` — GPT-J / complex-multiply style. Pairs `(i, i+1)`. Used by
  Llama 4 Scout, DeepSeek-V2 (`rope_interleave=True` complex-multiply path).

### 2.13 `RoPEScaling` (`types.py:114`)

- `NONE` — no scaling.
- `PI` — declared; not used.
- `NTK_STATIC` — declared; not used.
- `NTK_DYNAMIC` — declared; not used.
- `YARN` — DeepSeek-V2 / V3 YARN scaling. B5 implementation.
- `LLAMA3` — Llama 3 smooth-scaling.
- `LONGROPE` — Phi-3 / Phi-4 LongRoPE (two factor tables with per-position
  dispatch).

### 2.14 `CacheLayout` (`types.py:124`)

- `CONTIGUOUS` — HF baseline. Almost everyone.
- `PAGED` — vLLM-style. Declared; not used.
- `RING` — SWA wrap-around. Declared; not used.
- `MLA_LATENT` — DeepSeek MLA latent cache. Declared (`MLA_LATENT` is
  carried as a `qkv_layout`); the MLA path today stores decompressed K
  and V in a `ContiguousKVCache`.
- `SSM_STATE` — Mamba. The `SSMStateCache` class (`api/kvcache.py:91`)
  doesn't actually consume this enum — it's a separate class.

### 2.15 `MemoryLayout` (`types.py:132`)

- `HND` — `[batch, heads, seq, dim]`. The shape `sdpa` expects.
- `NHD` — `[batch, seq, heads, dim]`. Declared; not used.

### 2.16 `CacheOwnership` (`types.py:137`)

- `EXPLICIT_PASS` — PyTorch eager — caller passes cache via forward kwargs.
  Everyone today.
- `STATEFUL` — Core ML state op. Declared; not used.

### 2.17 `QDType` (`types.py:142`)

`INT4`, `INT8`, `FP8_E4M3`, `FP8_E5M2`, `FP4`, `NF4`, `MX_FP4` — see §1 of
`api/quant.py` for which roundtrip uses each. B10 landed AWQ, GGUF Q4_K_M,
FP8 E4M3 W8A8, MXFP4, and LiteRT W4A8.

### 2.18 `PackingLayout` (`types.py:152`)

- `NONE` — fp scales / no packing.
- `NIBBLE_LSB`, `NIBBLE_MSB` — declared; not used.
- `AWQ_INTERLEAVE` — `[0,2,4,6,1,3,5,7]` nibble permutation. AWQ W4A16.
- `GPTQ_INT32_PACK` — declared; not used.
- `GGUF_K` — GGUF Q4_K_M super-block.

### 2.19 `QuantRole` (`types.py:161`)

`WEIGHT`, `ACTIVATION`, `KV_K`, `KV_V`, `ATTN_INTERNAL` — declared so the
roundtrip helpers can tell apart per-weight, per-activation, and per-KV
schemes. Today's tests exercise WEIGHT and ACTIVATION only.

### 2.20 Cross-layer KV sharing (v5 Phase 1: per-layer offset model)

The v4 `ShareScheme` enum was removed in v5 Phase 1 (commit `8532879`).
The current mechanism is a per-layer pointer carried on `AttentionSpec`:

- `AttentionSpec.kv_source_layer_offset: Optional[int]` (`api/specs.py:278`)
  — `None` on owning layers (k_proj/v_proj built, layer owns K/V).
  Integer offset on borrowing layers (k_proj/v_proj NOT built; the caller
  passes K/V from layer `L + offset`). Hunyuan-Large CLA uses `-1`.
- `models/gemma4/config.py::Gemma4Config.kv_source_layer_idx_map()` —
  per-family dispatch that returns `dict[int, int]` mapping each shared
  layer's idx to the source layer's idx (Gemma 4 E2B = 20 shared layers,
  E4B = 35). The model-level assembly translates this into per-layer
  `kv_source_layer_offset` values and wires `SharedLayerKVCache`
  (`api/kvcache.py`) at runtime.

---

## §3  Spec Dataclasses (`api/specs.py`)

The 18 frozen dataclasses below are the parameter space of the IR. Anything
that varies between families is here.

### 3.1 `NormSpec` (`specs.py:14`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `kind` | `types.NormKind` | — | RMS or LAYER. Today's `RMSNorm` consumes only RMS. |
| `eps` | `float` | — | Variance epsilon. Typical 1e-5 (Llama) / 1e-6 (Qwen3, Gemma). |
| `weight_mode` | `NormWeightMode` | `STANDARD_W` | `ONE_PLUS_W` for Gemma 1/2/3. |

- **Who sets it:** every family — via `pre_attn_norm` / `pre_ffn_norm` /
  `post_attn_norm` / `post_ffn_norm` on `DecoderBlockSpec`, plus inside
  `AttentionSpec.qk_norm` / `AttentionSpec.v_norm`. The Gemma 2/3 reuse the
  same `ONE_PLUS_W` `NormSpec` for everything in the block.
- **When added:** M1 commit `4fa0498`.
- **Notable drift:** Gemma 4 was changed B0.5 → B0.6 to use `STANDARD_W` to
  match `modeling_gemma4.py:193-211` (commit `18f1bd6`).

### 3.2 `Llama3RoPEParams` (`specs.py:21`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `factor` | `float` | — | Smooth-scaling factor. 3.1 = 8.0, 3.2 = 32.0. |
| `low_freq_factor` | `float` | — | Typically 1.0. |
| `high_freq_factor` | `float` | — | Typically 4.0. |
| `original_context_length` | `int` | — | 8192 for Llama 3.x. |

- **Who sets it:** `models/llama3/config.py:144-150` for Llama 3.1 / 3.2.
  `models/llama4_scout/config.py:307-312` carries it for Scout's iRoPE
  RoPE layers.
- **When added:** M1 commit `bd5d009`.
- **Why:** `_compute_llama3_parameters` in HF `modeling_rope_utils.py`.

### 3.3 `YarnRoPEParams` (`specs.py:31`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `factor` | `float` | — | Extrapolation factor. V2-Lite = 40. |
| `original_max_position_embeddings` | `int` | — | V2-Lite = 4096. |
| `beta_fast` | `float` | `32.0` | Extrapolation boundary. |
| `beta_slow` | `float` | `1.0` | Interpolation boundary. |
| `mscale` | `float` | `1.0` | The numerator m-scale. V2-Lite = 0.707. |
| `mscale_all_dim` | `float` | `0.0` | The denominator m-scale. V2-Lite = 0.707. |
| `truncate` | `bool` | `True` | Floor/ceil the correction range. |

- **Who sets it:** `models/deepseek_v2_lite/config.py:159-166`. The V3 / V3.2
  families currently disable YARN at the synthetic-Lite level
  (`scaling=NONE`); the v3 production wrapper would set this if loaded.
- **When added:** B5 commit `5e352cf`.
- **Why:** `_compute_yarn_parameters` in `modeling_rope_utils.py:327-459`.

### 3.4 `LongRoPEParams` (`specs.py:71`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `short_factor` | `tuple[float, ...]` | — | Length `head_dim_rot/2`. |
| `long_factor` | `tuple[float, ...]` | — | Length `head_dim_rot/2`. |
| `original_max_position_embeddings` | `int` | — | Dispatch boundary. |
| `attention_factor` | `float` | `1.0` | mscale-like cos/sin multiplier. |

- **Who sets it:** `models/phi3_mini/config.py:152-159` (Phi-4-mini-instruct
  also uses this via delegation); `models/minicpm3/config.py:158-163`.
- **When added:** B2a commit `4d6b9a5`.
- **Why:** Phi-3 / Phi-4-mini have TWO inv_freq tables and the per-forward
  dispatch is `seq_len > original_max_position_embeddings`. Source:
  `modeling_rope_utils.py:462-547`.

### 3.5 `RoPESpec` (`specs.py:91`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `base_theta` | `float` | — | RoPE base. Llama 3 = 500_000. |
| `basis` | `RoPEBasis` | — | `SPLIT_HALF` (everyone) or `INTERLEAVED` (Llama 4, V2 complex). |
| `scaling` | `RoPEScaling` | `NONE` | NONE / LLAMA3 / LONGROPE / YARN. |
| `scale_factor` | `Optional[float]` | `None` | Unused — kept for spec parity. |
| `llama3_extra` | `Optional[Llama3RoPEParams]` | `None` | Required when scaling=LLAMA3. |
| `longrope_extra` | `Optional[LongRoPEParams]` | `None` | Required when scaling=LONGROPE. |
| `yarn_extra` | `Optional[YarnRoPEParams]` | `None` | Required when scaling=YARN. |
| `partial_rotary_factor` | `float` | `1.0` | Gemma 4 global = 0.25; Phi-3 legacy = 0.5; MLA = qk_rope/qk_head. |
| `partial_rotary_kind` | `str` | `"prefix"` | `"prefix"` (Phi-3/4) or `"proportional"` (Gemma 4). |
| `mrope_section` | `Optional[tuple[int, ...]]` | `None` | M-RoPE. Qwen2.5-VL = (16, 24, 24). |
| `is_2d` | `bool` | `False` | Reserved for axial-RoPE; unused. |

- **Who sets it:** every family — see the coverage matrix in §5.
- **`partial_rotary_factor`** added at B0.5 commit `cf4bd24` (Gemma 4 global).
- **`partial_rotary_kind`** added at B2a commit `ae6dc64` because Phi-3
  prefix-RoPE diverged from Gemma 4 proportional. See `specs.py:101-120`
  for the inline explanation.
- **`yarn_extra`** added at B5 commit `5e352cf`.
- **`longrope_extra`** added at B2a commit `4d6b9a5`.
- **`mrope_section`** added at B8 commit `4b8f415`.

### 3.6 `AttentionSpec` (`specs.py:136`)

The single largest spec — every attention-shape axis.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `n_q_heads` | `int` | — | Number of Q heads. |
| `n_kv_heads` | `int` | — | Number of K/V heads (=n_q_heads for MHA, MLA; <n_q_heads for GQA). |
| `head_dim` | `int` | — | Per-head dim. For MLA this is qk_nope+qk_rope. |
| `kind` | `AttentionKind` | — | STANDARD / MLA / DSA / (reserved linear). |
| `qkv_layout` | `QKVLayout` | — | SPLIT / FUSED / MLA_LATENT. |
| `mask_kind` | `MaskKind` | — | CAUSAL / SWA / BLOCK_SPARSE / BLOCK_BIDIRECTIONAL. |
| `q_bias` | `bool` | `False` | Q proj bias. Qwen2 family = True. |
| `k_bias` | `bool` | `False` | K proj bias. |
| `v_bias` | `bool` | `False` | V proj bias. |
| `o_bias` | `bool` | `False` | Output proj bias. |
| `attn_scale` | `Optional[float]` | `None` | Overrides 1/sqrt(head_dim). Gemma 2/3 use `query_pre_attn_scalar**-0.5`; Granite uses `attention_multiplier`. |
| `logit_softcap` | `Optional[float]` | `None` | Gemma 2 = 50.0; Gemma 3/4 = None. |
| `sliding_window` | `Optional[int]` | `None` | Window size when mask_kind=SWA. |
| `qk_norm` | `Optional[NormSpec]` | `None` | When set, the Q/K projections get a per-head RMSNorm. For MLA this slot ALSO carries `q_a_layernorm` / `kv_a_layernorm` eps. |
| `qk_norm_phase` | `QKNormPhase` | `NONE` | PRE_ROPE / POST_ROPE. |
| `qk_norm_shape` | `QKNormShape` | `NONE` | PER_HEAD_DH / FULL_HDH. |
| `rope` | `Optional[RoPESpec]` | `None` | When None: NoPE layer (SmolLM3 NoPE, Llama 4 NoPE, Granite-4-H attention layers with `position_embedding_type=nope`). |
| `attention_k_eq_v` | `bool` | `False` | Gemma 4 12B+ global: V := K. |
| `qk_norm_fixed_scale` | `Optional[float]` | `None` | Gemma 4 fixed-scale gain absorbing 1/sqrt(Dh) — when set, attn_scale should be 1.0. *Set to None in current Gemma 4 wiring (B0.6 drop).* |
| `v_norm` | `Optional[NormSpec]` | `None` | Gemma 4: unit RMSNorm on V before transpose+cache. |
| `v_norm_with_scale` | `bool` | `True` | False for Gemma 4 (no learnable weight). |
| `q_lora_rank` | `Optional[int]` | `None` | MLA Q-LoRA rank. None for V2-Lite (direct q_proj path). |
| `kv_lora_rank` | `Optional[int]` | `None` | MLA KV-LoRA rank. Required for MLA. |
| `qk_nope_head_dim` | `Optional[int]` | `None` | MLA nope sub-head dim. |
| `qk_rope_head_dim` | `Optional[int]` | `None` | MLA rope sub-head dim. |
| `v_head_dim` | `Optional[int]` | `None` | MLA V sub-head dim (often ≠ QK sub-head dim). |
| `indexer` | `Optional[IndexerSpec]` | `None` | DSA Lightning Indexer. Required when kind=DSA. |
| `block_bidirectional_mask` | `bool` | `False` | B8 Visual Causal Flow flag. |

- **Who sets each:** see the coverage matrix in §5. Notable:
  - `qk_norm` set by Qwen3, Gemma 3/4, OLMo 2, OLMoE, Qwen3-MoE.
  - `attention_k_eq_v` set ONLY by Gemma 4 12B+ global layers
    (`models/gemma4/config.py:124`).
  - `qkv_layout=FUSED` set by Phi-3 family + Phi-3-small + Phi-4-mini.
  - `qkv_layout=MLA_LATENT` set by MiniCPM-3, DeepSeek-V2-Lite,
    DeepSeek-V3-Lite, DeepSeek-V3-MoE, DeepSeek-V3.2.
  - `kind=DSA` set ONLY by DeepSeek-V3.2.
  - `block_bidirectional_mask` declared by the DeepSeek-OCR spec hook but
    currently unused (the HF v5.10.2 OCR-2 reference uses CAUSAL).
- **When the MLA fields were added:** B2b commit `453df69`.
- **When `attention_k_eq_v` was added:** B0.5 commit `cf4bd24`.
- **When `v_norm` was added:** B0.6 commit `496cc58` — discovered after
  reading `modeling_gemma4.py:1215, 1265` end-to-end.
- **When `indexer` was added:** B5 commit `fd672f3`.
- **When `block_bidirectional_mask` was added:** B8 commit `4b8f415`.
- **Dead fields:** `qk_norm_fixed_scale` is set to `None` in the current
  Gemma 4 wiring (B0.6 commit `50829ed` removed the absorb in favour of
  setting `attn_scale=1.0` directly). The field exists for forward
  compatibility with QAT-baked fixed-scale weights but is not exercised by
  any family today.

#### 3.6.1 The `effective_scale` resolution inside `Attention.__init__`

The attention block resolves three knobs into a single `effective_scale`:

```python
if spec.qk_norm_fixed_scale is not None:
    self.effective_scale = 1.0
elif spec.attn_scale is not None:
    self.effective_scale = spec.attn_scale
else:
    self.effective_scale = spec.head_dim ** -0.5
```

So three families' settings collide on this knob:
- **Default** (Llama, Qwen, Mistral, ...): `head_dim ** -0.5`.
- **`attn_scale` override** (Gemma 2/3, Granite, Granite-4-H, Phi-3-small,
  Gemma 4): use the explicit value.
- **`qk_norm_fixed_scale` set** (legacy Gemma 4 B0.5 path, now unused): the
  fixed-scale gain has absorbed `1/sqrt(Dh)` into the QK weight at load
  time, so the effective scale is `1.0`. As of B0.6 Gemma 4 sets
  `qk_norm_fixed_scale=None` and `attn_scale=1.0` directly, making this
  branch unused in any landed family.

#### 3.6.2 The forward mask construction (3 paths)

`Attention.forward` builds the attention mask explicitly rather than
trusting `is_causal=True`, because `is_causal` would mis-align when
`start_pos > 0`. The three paths (`attention.py:378-425`):

1. **`block_bidirectional_mask` + vision_token_count** (Visual Causal Flow,
   B8 hook):
   - `keep = vision_vis | text_text | text_vis`.
   - The mask is `[1, 1, S_q, T]` in q's dtype with `-inf` outside `keep`.
2. **`mask_kind == SWA` + `sliding_window`** (Gemma 2/3 sliding layers,
   Mistral SWA, Ministral, Phi-3, SmolLM3 sliding):
   - `keep = (k_pos <= q_pos) & (q_pos - k_pos <= W)`.
3. **Default causal** (everyone else):
   - `allowed = j <= start_pos + i`; `mask = 0` if allowed else `-inf`.

The shape is always broadcast to `[1, 1, S_q, T]` before passing to `sdpa`.

#### 3.6.3 The MLA forward dimension flow

The MLA forward (`_forward_mla`, `attention.py:437-515`) involves several
slice/reshape moves. The shapes at each step for MiniCPM-3 (H=40,
qk_nope=64, qk_rope=32, qk_head_dim=96, kv_lora_rank=256, v_head_dim=64):

```
x: [B, S, hidden=2560]
q = q_b_proj(q_a_layernorm(q_a_proj(x)))      [B, S, H*qk_head=3840]
q = q.view(B, S, H, qk_head=96)
q_nope = q[..., :64]                          [B, S, H, qk_nope=64]
q_pe   = q[..., 64:]                          [B, S, H, qk_rope=32]

compressed = kv_a_proj_with_mqa(x)            [B, S, kv_lora+qk_rope=288]
c_kv, k_pe_raw = split([256, 32])             [B, S, 256], [B, S, 32]
kv = kv_b_proj(kv_a_layernorm(c_kv))          [B, S, H*(qk_nope+v_head)=5120]
kv = kv.view(B, S, H, qk_nope+v_head=128)
k_nope, v = split([64, 64])                   [B, S, H, 64], [B, S, H, 64]

# RoPE on q_pe and k_pe_raw (shared single-head)
k_pe = k_pe_raw.view(B, S, 1, 32)
q_pe_rot, k_pe_rot = rope(q_pe, k_pe, position_ids)

# Assemble q at qk_head_dim, k at qk_head_dim, broadcast k_pe to H heads
q_full = cat([q_nope, q_pe_rot], dim=-1)      [B, S, H, qk_head=96]
k_pe_b = k_pe_rot.expand(B, S, H, 32)
k_full = cat([k_nope, k_pe_b], dim=-1)        [B, S, H, qk_head=96]

# Transpose to HND, write decompressed K (at 96) and V (at 64)
q_full = q_full.transpose(1, 2)               [B, H, S, 96]
k_full = k_full.transpose(1, 2)               [B, H, S, 96]
v      = v.transpose(1, 2)                    [B, H, S, 64]
cache.write(k_full, v, start_pos)
# Note: cache has v_head_dim=64 even though K is at 96. The cache stores
# tensors at different last-dim because ContiguousKVCache supports asymmetric
# K/V (B2b commit b4c8e6c).

# SDPA. n_heads matches between q/k/v so no GQA.
attn_out = sdpa(q_full, k_cached, v_cached, attn_mask, scale=96**-0.5)
# attn_out: [B, H, S, v_head=64]
attn_out = attn_out.transpose(1, 2).reshape(B, S, H*64=2560)
return o_proj(attn_out)                        [B, S, hidden]
```

The asymmetric K/V head_dim (96 vs 64) is the reason `ContiguousKVCache`
needs the `v_head_dim` kwarg.

### 3.7 `FFNSpec` (`specs.py:213`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `intermediate_size` | `int` | — | The `I` in down(act(gate(x)) * up(x)). |
| `activation` | `Activation` | — | SILU / GELU / GEGELU / RELU2. |
| `gate_kind` | `GateKind` | — | SWIGLU / GEGLU / GELU_ONLY / RELU2_ONLY. |
| `fused_gate_up` | `bool` | `False` | One `2*I` projection chunked at forward. Phi-3, Moshi, Granite-4-H shared MLP. |
| `gate_bias` | `bool` | `False` | Bias on gate_proj. |
| `up_bias` | `bool` | `False` | Bias on up_proj. |
| `down_bias` | `bool` | `False` | Bias on down_proj. |

- **Who sets `fused_gate_up=True`:** Phi-3 mini, Phi-3-small, Phi-4-mini,
  Granite-4-H shared MLP, Moshi FFN.
- **Who sets `gate_kind=GEGLU`:** Gemma 2 / 3 / 4 (`models/gemma{2,3,4}/config.py`).
  Everyone else uses `SWIGLU`.
- **When added:** M1 commit `4fa0498`. `fused_gate_up` added at B2a commit
  `4d6b9a5`.

### 3.8 `QuantSpec` (`specs.py:224`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `qdtype` | `QDType` | — | INT4 / INT8 / FP8_E4M3 / ... |
| `group_size` | `Optional[int]` | — | None=per-tensor, -1=per-channel, N=blockwise. |
| `quant_axis` | `int` | — | Usually 0 for weights (per-output-channel). |
| `scale_dtype` | `torch.dtype` | — | fp16 or bf16 typically. |
| `has_zero_point` | `bool` | — | True for AWQ (asymmetric); False for symmetric INT4. |
| `packing` | `PackingLayout` | — | AWQ_INTERLEAVE, GGUF_K, NONE. |
| `accumulator_dtype` | `torch.dtype` | — | fp32 for safety. |
| `role` | `QuantRole` | `WEIGHT` | WEIGHT / ACTIVATION / KV_K / ... |
| `codebook` | `Optional[object]` | `None` | Reserved for IQ2_M / AQLM. |

- **Who sets it:** the quantization round-trip tests in
  `tests/api/quant/*` and the LiteRT / GGUF / FP8 / MXFP4 round-trip
  helpers in `api/quant.py`. No `models/<family>/config.py` sets a
  `QuantSpec` directly — the quantization is a separate layer.
- **When added:** M1 commit `3b5c837` (AWQ) — incrementally extended
  through B10.

### 3.9 `KVCacheSpec` (`specs.py:237`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `layout` | `CacheLayout` | — | CONTIGUOUS in production. |
| `memory_layout` | `MemoryLayout` | — | HND. |
| `k_dtype` | `torch.dtype` | — | Often bf16. |
| `v_dtype` | `torch.dtype` | — | Often bf16. |
| `ownership` | `CacheOwnership` | `EXPLICIT_PASS` | Stateful is reserved. |
| `block_size` | `Optional[int]` | `None` | For PAGED layout; unused. |
| `k_quant` | `Optional[QuantSpec]` | `None` | KV-cache K quantization; unused. |
| `v_quant` | `Optional[QuantSpec]` | `None` | KV-cache V quantization; unused. |

- **Cross-layer KV sharing:** the v4 `share_scheme` / `num_kv_shared_layers`
  fields were removed in v5 Phase 1. Sharing is now expressed per-layer on
  `AttentionSpec.kv_source_layer_offset` (see §2.20) and the per-family
  source-layer map lives on the model config (e.g.
  `Gemma4Config.kv_source_layer_idx_map()`, `Gemma4Config.num_kv_shared_layers`).
  Runtime wrapper: `api/kvcache.py::SharedLayerKVCache`.
- **When added:** B0.5 commit `6dbe232` (enum); reshaped v5 Phase 1
  commit `8532879` (enum removed; per-layer offset model).

### 3.10 `ConvSpec` (`specs.py:253`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `kernel_size` | `int` | — | Mamba-2 default 4. |
| `bias` | `bool` | `True` | Mamba-2 default True. |
| `activation` | `Optional[Activation]` | `None` | Today applied externally in `Mamba2Mixer`. |

- **Who sets it:** today nobody — `Mamba2Mixer` reads conv params directly
  from `SSDSpec.base` (which is an `SSMSpec`). Kept for v3 spec parity.
- **When added:** M1 fix-pass commit `5aa8374`.

### 3.11 `SSMSpec` (`specs.py:269`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `d_state` | `int` | — | `state_size` in HF (Mamba-2 default 128). |
| `d_conv` | `int` | — | `conv_kernel` in HF (default 4). |
| `d_inner` | `int` | — | `intermediate_size` = hidden × expand. |
| `expand_factor` | `int` | `2` | Mamba default. |
| `dt_rank` | `int` | `-1` | Auto = hidden//16; UNUSED by Mamba-2. |
| `dt_min` | `float` | `0.001` | |
| `dt_max` | `float` | `0.1` | |
| `dt_init_floor` | `float` | `1e-4` | |
| `conv_bias` | `bool` | `True` | |
| `bias` | `bool` | `False` | in_proj / out_proj bias. |
| `use_fast_path` | `bool` | `True` | Reserved — `Mamba2Mixer` always uses the reference path. |
| `activation` | `Activation` | `SILU` | |

- **Who sets it:** `models/mamba2/config.py:115-127`,
  `models/granite4_h/config.py:204-216`.
- **When added:** M1 fix-pass commit `5aa8374`.

### 3.12 `SSDSpec` (`specs.py:297`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `base` | `SSMSpec` | — | Per-head defaults. |
| `chunk_size` | `int` | `256` | SSD chunk-parallel scan chunk. |
| `headdim` | `int` | `64` | Per-head dim. |
| `ngroups` | `int` | `1` | Number of B/C groups. |
| `n_heads` | `int` | `1` | Number of SSM heads. |
| `time_step_limit_low` | `float` | `0.0` | Lower clamp after softplus(dt+dt_bias). |
| `time_step_limit_high` | `float` | `inf` | Upper clamp. |
| `layer_norm_epsilon` | `float` | `1e-5` | For the gated RMS norm before out_proj. |
| `residual_in_fp32` | `bool` | `True` | Whether residual is upcast. |

- **Invariants:** `n_heads * headdim == base.d_inner` and
  `n_heads % ngroups == 0`. Validated at `Mamba2Mixer.__init__`
  (`ssm.py:117-126`).
- **Who sets it:** `models/mamba2/config.py:128-138`,
  `models/granite4_h/config.py:217-227`.
- **When added:** M1 fix-pass commit `5aa8374`.

### 3.13 `GroupRoutingSpec` (`specs.py:336`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `n_groups` | `int` | — | Number of expert groups. DeepSeek-V3 = 8. |
| `topk_per_group` | `int` | — | Top-k groups to keep. DeepSeek-V3 = 4. |
| `routed_expert_grouping` | `bool` | `True` | Reserved flag. |

- **Who sets it:** `models/deepseek_v3_lite/config.py:104-107`,
  `models/deepseek_v3_moe/config.py:163-166`,
  `models/deepseek_v32/config.py:105-107`. V2-Lite suppresses to None
  when `n_group==1 AND topk_group==1` (degenerate case).
- **When added:** M1 fix-pass commit `5aa8374`.

### 3.14 `MoESpec` (`specs.py:344`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `n_experts` | `int` | — | Total routed experts. |
| `top_k` | `int` | — | Per-token expert count. |
| `n_shared_experts` | `int` | `0` | Number of shared experts (DeepSeek). |
| `router_kind` | `str` | `"softmax"` | `"softmax"` (V2, Mixtral, OLMoE, Qwen3-MoE) or `"sigmoid_plus_bias"` (V3). |
| `router_norm` | `bool` | `False` | V3 `norm_topk_prob`. |
| `score_correction_bias` | `bool` | `False` | V3 e_score_correction_bias. |
| `group_routing` | `Optional[GroupRoutingSpec]` | `None` | V3 / OCR with group routing. |
| `routed_scaling_factor` | `float` | `1.0` | V3 = 2.5. |
| `expert_ffn` | `Optional[FFNSpec]` | `None` | Required. The shape of each expert. |

- **Who sets it:** Mixtral, OLMoE, Qwen3-MoE (softmax / no shared);
  DeepSeek-V2-Lite (softmax + shared + V2 group-limited); DeepSeek-V3-Lite
  / V3-MoE / V3.2 (sigmoid+bias + shared + V3 group-limited); DeepSeek-OCR-2
  (softmax + shared).
- **When added:** B5 commit `43a4471`.

#### 3.14.1 The MoE dispatch routing math, by router family

The MoE module supports two router families with distinct routing math.

**Softmax router (V2, Mixtral, OLMoE, Qwen3-MoE, DeepSeek-OCR-2)** —
`MoE._route_softmax` (feedforward.py:198-238):
1. `router_logits = F.linear(x.float(), gate.weight.float())` — fp32 promoted.
2. `scores = router_logits.softmax(dim=-1, dtype=fp32)`.
3. If no group routing: `topk_w, topk_idx = torch.topk(scores, top_k)`.
4. If group routing: per-group MAX (V2-style), top-k groups by max score,
   mask out non-selected groups, top-k within the kept groups.
5. If `router_norm`: renormalize `topk_w / sum(topk_w)`.
6. `topk_w *= routed_scaling_factor`.
7. Return `(topk_idx, topk_w)`.

**Sigmoid+bias router (V3, V3-MoE, V3.2)** —
`MoE._route_sigmoid_plus_bias` (feedforward.py:240-292):
1. `router_logits = gate(x)` — uses `_SigmoidRouter` which does the fp32
   linear.
2. `router_probs = router_logits.sigmoid()`.
3. **Routing CHOICE uses** `probs_for_choice = router_probs +
   e_score_correction_bias`. The bias breaks ties in expert selection.
4. **Routing WEIGHTS come from** the bias-free `router_probs` (this is the
   critical detail — `feedforward.py:287`).
5. Group routing (when set): per-group SUM of TOP-2 (V3-style — distinct
   from V2's per-group MAX), top-k groups, mask non-selected groups with
   `-inf`, top-k within.
6. `topk_w = router_probs.gather(1, topk_idx)`.
7. If `router_norm`: renormalize.
8. `topk_w *= routed_scaling_factor`.

The V2 vs V3 group-routing math is the source of multiple shapes that LOOK
similar but compute distinct results. The recipes (§6.5, §6.6) emphasise
the right `router_kind` flag.

### 3.15 `IndexerSpec` (`specs.py:388`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `indexer_dim` | `int` | — | Indexer Q/K projection dim. V3.2 = 64. |
| `top_k` | `int` | — | Per-query indexer top-k. V3.2 = 2048. |
| `warmup_tokens` | `int` | `0` | Pre-DSA training token count (informational). |

- **Who sets it:** `models/deepseek_v32/config.py:70-74`.
- **When added:** B5 commit `fd672f3`.
- **Status:** shape-only. The `Attention.forward` raises
  `NotImplementedError` when `_is_dsa`.

### 3.16 `LayerScaleSpec` (`specs.py:411`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `num_q_heads_per_layer` | `tuple[int, ...]` | — | OpenELM per-layer Q head count. |
| `num_kv_heads_per_layer` | `tuple[int, ...]` | — | Per-layer KV head count. |
| `ffn_multipliers_per_layer` | `tuple[float, ...]` | — | Per-layer FFN ratio. |

- **Who sets it:** today nobody — OpenELM is not in `models/`. Kept for
  v3 spec parity.
- **When added:** M1 fix-pass commit `5aa8374`.
- **Status:** dead — unused by any family.

### 3.17 `PLESpec` (`specs.py:419`)

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `ple_dim` | `int` | — | Per-Layer Embedding inner dim. Gemma 4 = 256. |
| `residual_scale` | `float` | — | Gemma 4 = 1/sqrt(2). |
| `injection_norm` | `NormSpec` | — | The RMSNorm applied after the per-layer projection. |

- **Who sets it:** `models/gemma4/config.py:181-185` for E2B/E4B variants.
- **When added:** B0.5 commit `c6d5836`.

### 3.18 `DecoderBlockSpec` (`specs.py:438`)

The top-level per-layer container.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `attn_norm_position` | `NormPosition` | — | PRE / POST / PRE_AND_POST. |
| `ffn_norm_position` | `NormPosition` | — | Same as above. |
| `token_mixer` | `Union[AttentionSpec, SSDSpec]` | — | The token-mixer spec. (SSMSpec reserved.) |
| `channel_mixer` | `Union[FFNSpec, MoESpec]` | — | Dense FFN or MoE. |
| `pre_attn_norm` | `Optional[NormSpec]` | `None` | Required for PRE / PRE_AND_POST attn. |
| `post_attn_norm` | `Optional[NormSpec]` | `None` | Required for POST / PRE_AND_POST attn. |
| `pre_ffn_norm` | `Optional[NormSpec]` | `None` | Required for PRE / PRE_AND_POST ffn. |
| `post_ffn_norm` | `Optional[NormSpec]` | `None` | Required for POST / PRE_AND_POST ffn. |
| `residual_scale` | `Optional[float]` | `None` | Granite / MiniCPM-3 μP per-sublayer multiplier. |
| `embedding_scale` | `Optional[float]` | `None` | Gemma sqrt(D); MiniCPM scale_emb; Granite embedding_multiplier. |
| `logits_scale` | `Optional[float]` | `None` | Lm_head multiplier (MiniCPM dim_model_base/hidden; Granite logits_scaling). |
| `per_layer_embedding` | `Optional[PLESpec]` | `None` | Gemma 4 E2B/E4B. |
| `final_logit_softcap` | `Optional[float]` | `None` | Gemma 2/3/4 = 30.0 (model-level). |
| `skip_ffn` | `bool` | `False` | Canonical Mamba-2 = True (no FFN sublayer). |

- **Who sets each:** see the worked examples in §7 and the coverage matrix
  in §5.
- **`token_mixer` broadened to Union** at B7 commit `7171ac5`.
- **`channel_mixer` broadened to Union** at B5 commit `5e352cf`.
- **`residual_scale`** added at B2a commit `a62551f`.
- **`per_layer_embedding`** added at B0.5 commit `cf4bd24`.
- **`skip_ffn`** added at B7 commit `7171ac5`.

---

## §4  Building Blocks (`api/{norm,rope,kvcache,quant,attention,feedforward,embedding,ssm,block}.py`)

### 4.1 `RMSNorm` (`api/norm.py:11`)

```python
RMSNorm(spec: NormSpec, hidden_size: int, dtype=fp32)
RMSNorm.forward(x: [B, S, H]) -> [B, S, H]
```

- **Constructor allocates:** `weight: nn.Parameter` of shape `[hidden_size]`,
  initialised to ones.
- **Internal:** wraps `api.ops.rms_norm` with `mode` set from
  `spec.weight_mode`.
- **Spec fields consumed:** `kind`, `eps`, `weight_mode`.
- **Conditional branches:** raises in `__init__` when `kind != RMS`. The
  `mode` dispatch in `forward` selects `"one_plus_w"` for Gemma 1/2/3.

### 4.2 `QKNorm` (`api/norm.py:29`)

```python
QKNorm(spec: NormSpec, head_dim, shape: QKNormShape, n_heads=None, dtype=fp32)
QKNorm.forward(x: [B, S, H, Dh]) -> [B, S, H, Dh]
```

- **Constructor allocates:** `weight: nn.Parameter` of shape
  `[head_dim]` (PER_HEAD_DH) or `[n_heads * head_dim]` (FULL_HDH).
- **Forward path:** for `PER_HEAD_DH`, applies `rms_norm` over the last
  dim per head independently. For `FULL_HDH`, flattens
  `[B, S, H, Dh] → [B, S, H*Dh]`, applies the norm with the full weight,
  then reshapes back. Algebraic equivalence with reshape-first then
  norm-last holds because RMSNorm is independent across the leading dims.

### 4.3 `RoPE` (`api/rope.py:106`)

```python
RoPE(spec: RoPESpec, head_dim, max_seq, dtype=fp32)
RoPE.forward(q, k, position_ids) -> (q_rot, k_rot)
```

- **Constructor allocates buffers:** `cos_cached`, `sin_cached` of shape
  `[max_seq, *]`. For LongRoPE: additionally `cos_cached_long`,
  `sin_cached_long`.
- **inv_freq construction is per-`partial_rotary_kind`:**
  - `"prefix"` (Phi-3): denominator = `dim_rot` (the rotated subset only);
    inv_freq stays at `rope_angles` entries; cos/sin live at `head_dim_rot`.
  - `"proportional"` (Gemma 4): denominator = full `head_dim`; inv_freq is
    zero-padded to `head_dim/2`; cos/sin live at full `head_dim`.
- **YARN path:** computes inv_freq via the YARN linear-ramp blend and
  multiplies cos/sin by an `attention_factor` derived from `mscale` /
  `mscale_all_dim` (`_yarn_inv_freq_and_scale` in `rope.py:17-78`).
- **LongRoPE path:** builds TWO inv_freq tables (`cos_cached` for
  `seq_len ≤ boundary`; `cos_cached_long` otherwise) plus an
  `attention_factor` cos/sin multiply.
- **Forward dispatch (4 branches):**
  1. `position_ids.dim() == 3` → M-RoPE path: rebuilds cos/sin on-the-fly,
     calls `ops.rope_apply_mrope`.
  2. `INTERLEAVED` basis → `ops.rope_apply(basis="interleaved")`.
  3. `partial_rotary_kind == "prefix" AND partial_rotary_factor < 1.0` →
     `ops.rope_apply_partial`.
  4. Default → `ops.rope_apply(basis="split_half")`.
- **Reject list (raises NotImplementedError):**
  - `INTERLEAVED` + LongRoPE.
  - `INTERLEAVED` + partial_rotary_factor != 1.0.
  - M-RoPE + partial_rotary_factor < 1.
  - YARN + proportional partial-rotary.

### 4.4 `ContiguousKVCache` (`api/kvcache.py:14`)

```python
ContiguousKVCache(spec, batch_size, n_kv_heads, head_dim, max_seq,
                  device, v_head_dim=None)
.write(k, v, start_pos)
.read(seq_len) -> (k, v)
.reset()
```

- **HND layout:** stores `k: [B, n_kv_heads, max_seq, head_dim]` and
  `v: [B, n_kv_heads, max_seq, v_head_dim]`.
- **`v_head_dim` kwarg** added at B2b commit `b4c8e6c` so MLA can store K at
  `qk_head_dim` and V at `v_head_dim` (the two are different on MiniCPM-3 /
  DeepSeek-V2-Lite: 192 vs 128, or 96 vs 64).
- **Bounds-check:** `write` validates `start_pos + S_new <= max_seq` and
  raises on dtype mismatch between `k`/`v` and the spec's `k_dtype`/`v_dtype`.

### 4.5 `SSMStateCache` (`api/kvcache.py:91`)

```python
SSMStateCache(batch_size, conv_dim, conv_kernel, n_heads, head_dim,
              d_state, dtype, device)
.update_conv_state(new_state)
.update_recurrent_state(new_state)
.reset()
```

- **Two pieces of state per layer:**
  1. `conv_state: [B, conv_dim, conv_kernel]` — rolling window of the last
     `conv_kernel-1` conv inputs.
  2. `ssm_state: [B, n_heads, head_dim, d_state]` — recurrent SSM state at
     chunk end.
- **`has_previous_state` flag** mirrors HF's `has_previous_state(layer_idx)`.
- **Status:** prefill only. Decode raises `NotImplementedError` in
  `Mamba2Mixer.forward` (`ssm.py:226-229`).

### 4.6 `SharedLayerKVCache` (`api/kvcache.py:190`)

A read-only alias to another `ContiguousKVCache`. Used by Gemma 4 E2B/E4B
layers that reuse a prior same-type layer's K/V. `write` is a no-op; `read`
delegates to the source cache.

### 4.6b The `_init_mla` constructor sequence (`api/attention.py:214-315`)

Worth its own subsection because MLA construction is the longest single
code path in the attention block. The sequence of attribute creation:

1. **`attn_bias`** — the single MLA bias flag is `spec.q_bias`. MLA uses one
   `config.attention_bias` in HF; q_a_proj, kv_a_proj_with_mqa, and o_proj
   all take it. q_b_proj and kv_b_proj are NEVER biased per
   `modeling_minicpm.py:364-378`.
2. **`qa_norm_spec`** — when `spec.qk_norm is not None`, reuse it for the
   q_a_layernorm / kv_a_layernorm eps. Otherwise default to
   `NormSpec(RMS, 1e-5, STANDARD_W)`. The qk_norm slot is OVERLOADED for
   MLA — for STANDARD attention it would mean a Qwen3-style PRE-RoPE
   QK-norm, for MLA it carries only `.eps` and `.weight_mode`.
3. **Q path** dispatch:
   - `spec.q_lora_rank is None` (V2-Lite direct path) — allocate
     `self.q_proj = nn.Linear(hidden, H * qk_h, bias=False)`. Source:
     `modeling_deepseek_v2.py:310-311`.
   - `spec.q_lora_rank is not None` (MiniCPM-3, V3) — allocate the LoRA
     stack: `q_a_proj` (bias=attn_bias) → `q_a_layernorm` (RMSNorm at
     `q_lora_rank`) → `q_b_proj` (bias=False).
4. **KV LoRA path:**
   - `kv_a_proj_with_mqa = nn.Linear(hidden, kv_lora_rank + qk_rope_head_dim,
     bias=attn_bias)`. Note the "+qk_rope_head_dim" — the K-rope sub-head
     comes OUT of this projection alongside the compressed latent, then is
     split at forward time.
   - `kv_a_layernorm = RMSNorm(qa_norm_spec, kv_lora_rank)`.
   - `kv_b_proj = nn.Linear(kv_lora_rank, H * (qk_nope_head_dim + v_head_dim),
     bias=False)`. The output is split at forward time into K-nope and V.
5. **`o_proj = nn.Linear(H * v_h, hidden, bias=attn_bias)`** — note the input
   dim is `H * v_head_dim`, NOT `H * head_dim` (head_dim here means
   qk_head_dim).
6. **`effective_scale = head_dim ** -0.5`** — note this is qk_head_dim**-0.5,
   not v_head_dim**-0.5 (the SDPA softmax scale is by the QK shape).
   Source: `modeling_minicpm.py:387`.
7. **RoPE** built at `qk_rope_head_dim` (NOT a partial within a larger head).
   MiniCPM/V2 use `MiniCPMRotaryEmbedding(dim=qk_rope_head_dim)`. We
   therefore build an `api.rope.RoPE` with `head_dim=qk_rope_head_dim` and
   `partial_rotary_factor=1.0` (full rotation over the rope slice). The
   slicing `q[..., qk_nope:]` happens at forward time.
8. The constructor explicitly sets the standard-attention attributes
   (`qkv_proj`, `k_proj`, `v_proj`, `q_norm`, `k_norm`,
   `_v_norm_eps`/`mode`/`with_scale`) to None and tags `self._is_mla = True`
   so `forward` dispatches.

### 4.7 `Attention` (`api/attention.py:21`)

```python
Attention(spec: AttentionSpec, hidden_size, max_seq, dtype=fp32)
.forward(x, position_ids, cache, start_pos, vision_token_count=None)
        -> [B, S, hidden_size]
```

- **Two construction paths** dispatched by `spec.kind`:
  - `STANDARD` → standard q/k/v projection allocation.
  - `MLA` / `DSA` → `_init_mla` allocates the LoRA-compressed projections.
- **Standard sub-modules:**
  - `q_proj`, `k_proj`, `v_proj` OR `qkv_proj` (FUSED).
  - `o_proj`.
  - `q_norm`, `k_norm` (when `qk_norm` set).
  - `v_norm_weight` (when `v_norm` set; parameter or buffer based on
    `v_norm_with_scale`).
  - `rope` (RoPE module).
- **MLA sub-modules:**
  - `q_a_proj`, `q_a_layernorm`, `q_b_proj` (when `q_lora_rank` set) OR
    `q_proj` (when `q_lora_rank == None`).
  - `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`.
  - `o_proj`.
  - `rope` built at `qk_rope_head_dim` (NOT a partial within a larger head).
- **DSA additional sub-modules:** `indexer_q_proj`, `indexer_k_proj`.
- **Standard forward conditional branches:**
  - **FUSED slice** when `qkv_layout == FUSED`.
  - **K = V alias** when `attention_k_eq_v=True`.
  - **q_norm / k_norm** when `qk_norm` set.
  - **NoPE skip** when `self.rope is None`.
  - **v_norm** when `_v_norm_mode is not None` — applied to V BEFORE
    transpose+cache.
  - **Mask construction (3 paths):**
    - `BLOCK_BIDIRECTIONAL` + `vision_token_count` not None → Visual Causal
      Flow mask.
    - `SWA` + `sliding_window` not None → sliding-window mask.
    - Otherwise → standard causal mask (constructed explicitly, NOT via
      `is_causal=True`).
  - **Softcap** dispatched inside `ops.sdpa` via `logit_softcap`.
- **MLA forward branches:** see §7.3 worked example.

### 4.8 `FeedForward` (`api/feedforward.py:35`)

```python
FeedForward(spec: FFNSpec, hidden_size, dtype=fp32)
.forward(x: [B, S, H]) -> [B, S, H]
```

- **Sub-modules:**
  - `gate_proj`, `up_proj`, `down_proj` (SPLIT path), OR
  - `gate_up_proj`, `down_proj` (FUSED path).
- **Conditional:** SwiGLU vs GeGLU selects `silu` vs `gelu_pytorch_tanh` on
  the gate.
- **Reject list:** raises NotImplementedError on activation/gate mismatch
  (e.g. SWIGLU + GELU activation).

### 4.9 `MoE` (`api/feedforward.py:87`)

```python
MoE(spec: MoESpec, hidden_size, dtype=fp32)
.forward(x: [B, S, H]) -> [B, S, H]
```

- **Sub-modules:**
  - `gate: nn.Linear` (softmax router) OR `gate: _SigmoidRouter`
    (sigmoid+bias router, includes the `e_score_correction_bias` buffer).
  - `experts_gate_up: nn.Parameter[n_experts, 2*I, hidden]` — packed gate+up.
  - `experts_down: nn.Parameter[n_experts, hidden, I]`.
  - `shared_experts: FeedForward` (when `n_shared_experts > 0`).
- **Forward (4 phases):**
  1. Flatten `x` to `[N, H]` where `N = B*S`.
  2. Route via `_route_softmax` or `_route_sigmoid_plus_bias` →
     `(topk_idx, topk_w)`.
  3. Expert dispatch in `_experts_forward` (dense-scatter loop over the
     experts that were chosen by at least one token).
  4. Add `shared_experts(x)` to the routed output if shared experts present.
- **Critical detail:** in the V3 sigmoid+bias router, the routing CHOICE
  uses `probs + bias` but the gathered WEIGHTS come from the bias-free
  `router_probs` (`feedforward.py:287`). This matches
  `modeling_deepseek_v3.py:232`.

### 4.10 `PerLayerEmbedding` (`api/embedding.py:22`)

```python
PerLayerEmbedding(spec: PLESpec, vocab_size, hidden_size,
                  num_layers, dtype=fp32)
.forward(input_ids: [B, S], layer_idx) -> [B, S, hidden]
```

- **Sub-modules:**
  - `ple_table: nn.Embedding[vocab_size, ple_dim]`.
  - `inj_norm: RMSNorm[ple_dim]`.
  - `layer_projs: nn.ModuleList[nn.Linear[ple_dim → hidden]]`.
- **Used externally:** the `DecoderBlock.forward` receives a
  `per_layer_input` tensor already projected to hidden_size — this module
  computes that tensor outside the block (one per (token, layer) pair).
- **`register_layers`** lazily allocates the per-layer projections when
  `num_layers` was unknown at construction.

### 4.11 `Mamba2Mixer` (`api/ssm.py:93`)

```python
Mamba2Mixer(spec: SSDSpec, hidden_size, dtype=fp32)
.forward(hidden_states: [B, S, H], cache: Optional[SSMStateCache])
        -> [B, S, H]
```

- **Sub-modules:**
  - `in_proj: nn.Linear[H → 2*d_mlp + d_inner + conv_dim + n_heads]`.
  - `conv1d: nn.Conv1d` depthwise, `groups=conv_dim`.
  - `dt_bias: nn.Parameter[n_heads]`.
  - `A_log: nn.Parameter[n_heads]`.
  - `D: nn.Parameter[n_heads]`.
  - `norm: _MambaRMSNormGated[d_inner]` (gated RMS, silu(gate) inside).
  - `out_proj: nn.Linear[d_inner → H]`.
- **Forward conditional:** decode path (`cache.has_previous_state == True`)
  raises NotImplementedError; prefill path runs the full chunked SSD scan
  via `ops.selective_scan`.

### 4.11b The Mamba-2 forward — line-by-line shape trace

Following one prefill pass through `Mamba2Mixer.forward` for state-spaces
Mamba-2-2.7B (hidden=2560, num_heads=80, head_dim=64, n_groups=1,
d_state=128, expand=2, conv_kernel=4, chunk_size=256):

```
hidden_states: [B, S, 2560]
projected = in_proj(hidden_states)   # in_proj_out = 2*0 + d_inner + conv_dim + n_heads
                                     # = 0 + 5120 + 5376 + 80 = 10576
_, _, gate, x_BC, dt = projected.split(
    [0, 0, 5120, 5376, 80], dim=-1)
# gate: [B, S, 5120]  x_BC: [B, S, 5376]  dt: [B, S, 80]

# Depthwise conv (groups=conv_dim=5376)
x_BC_t = x_BC.transpose(1, 2)              # [B, 5376, S]
conv_states = F.pad(x_BC_t, (4-S if S<4 else 0, 0))[..., -4:]
                                             # [B, 5376, 4]  → cache.update_conv_state
conv_out = conv1d(x_BC_t)[..., :S]          # [B, 5376, S]
x_BC_act = F.silu(conv_out).transpose(1, 2) # [B, S, 5376]

# Slice into x, B, C  (d_inner + 2*ng*ds = 5120 + 2*1*128 = 5376)
x_for_ssm, B_unrep, C_unrep = split(
    x_BC_act, [5120, 128, 128], dim=-1)

# dt: softplus + clamp
A = -torch.exp(A_log.float())              # [80]
dt_fp32 = F.softplus(dt + dt_bias)         # [B, S, 80]
dt_fp32 = torch.clamp(dt_fp32, low, high)

x_disc = x_for_ssm.reshape(B, S, 80, 64).float()  # [B, S, 80, 64]
B_ssm = B_unrep.reshape(B, S, 1, 128).float()     # [B, S, 1, 128]
C_ssm = C_unrep.reshape(B, S, 1, 128).float()
B_ssm = B_ssm.repeat_interleave(80, dim=2,
                                 output_size=80)  # [B, S, 80, 128]
C_ssm = C_ssm.repeat_interleave(80, dim=2,
                                 output_size=80)

# Discretize
x_disc = x_disc * dt_fp32[..., None]              # [B, S, 80, 64]
A_disc = A.to(dt_fp32.dtype) * dt_fp32            # [B, S, 80]

# Selective scan
y, ssm_state = selective_scan(
    x_disc, A_disc, B_ssm, C_ssm, chunk_size=256)
# y: [B, S, 80, 64]  ssm_state: [B, 80, 64, 128]

# D-residual
D_residual = D[..., None].float() * x_unscaled    # [B, S, 80, 64]
y = y + D_residual

# Reshape to [B, S, d_inner]
y = y.reshape(B, S, 5120)

# Cache and gated RMSNorm
cache.update_recurrent_state(ssm_state.to(cache.dtype))
y = self.norm(y, gate)  # silu(gate) inside fp32 path

return self.out_proj(y.to(dtype))                 # [B, S, 2560]
```

The output reshape from `[B, S, num_heads, head_dim]` back to
`[B, S, d_inner]` is justified by the invariant
`num_heads * head_dim == d_inner` (validated at `Mamba2Mixer.__init__`,
`ssm.py:117-126`).

### 4.12 `DecoderBlock` (`api/block.py:39`)

```python
DecoderBlock(spec: DecoderBlockSpec, hidden_size, max_seq, dtype=fp32)
.forward(x, position_ids=None, cache=None, start_pos=0,
         per_layer_input=None, vision_token_count=None)
        -> [B, S, hidden]
```

- **Sub-modules:**
  - `pre_attn_norm` (when PRE / PRE_AND_POST).
  - `post_attn_sublayer_norm` (when POST / PRE_AND_POST).
  - `attention` — either `Attention` or `Mamba2Mixer` based on
    `isinstance(spec.token_mixer, SSDSpec)`. Note the attribute is named
    `attention` even for SSM blocks (kept for backward compat with all
    existing weight loaders).
  - `pre_ffn_norm`, `post_ffn_sublayer_norm` (positionally).
  - `feedforward` — `FeedForward` or `MoE`. Skipped entirely when
    `spec.skip_ffn=True` (canonical Mamba-2).
  - `per_layer_input_gate`, `per_layer_projection`,
    `post_per_layer_input_norm` (when PLE).
  - `layer_scalar` buffer — Gemma 4 trailing multiply. Initialised to ones,
    applied on EVERY layer regardless of PLE state.
- **Forward sequence:** pre-norm → token mixer → optional post-norm →
  optional residual scale → residual add → pre-FFN-norm → FFN → optional
  post-FFN-norm → optional residual scale → residual add → optional PLE
  injection block → `layer_scalar` multiply.

---

## §5  Coverage Matrix

Each row is a model family that landed; each column is one of the 25
spec axes that materially varies across families. Cells use compact
abbreviations.

Abbreviations:
- Attention shape "GQA-32:8" = 32 Q heads, 8 KV heads, head_dim
  inferred. "MLA" = MLA latent. "MHA" = `n_q_heads == n_kv_heads`.
- QK norm phase / shape: "PRE/H" = PRE_ROPE + PER_HEAD_DH; "PRE/FH" =
  PRE_ROPE + FULL_HDH; "—" = NONE.
- RoPE basis: "SH" = SPLIT_HALF; "IL" = INTERLEAVED; "—" = no RoPE
  (Mamba-2 etc.).
- RoPE scaling: "—" = NONE; "LL3" = LLAMA3; "LR" = LONGROPE; "YR" = YARN.
- Norm position: "PRE" / "POST" / "P&P" (PRE_AND_POST).
- Channel mixer: "FFN" = dense FFN; "MoE-SM" = softmax MoE; "MoE-V3" =
  sigmoid+bias MoE.

| Family | attn.kind | qkv_layout | Hq:Hk | qk_norm phase/shape | partial_rotary_factor | partial_rotary_kind | attn_k_eq_v | v_norm | mask_kind | sliding_window | logit_softcap | rope.basis | rope.scaling | rope.base_theta | norm.weight_mode | attn_norm_position | residual_scale | kv share_scheme | ffn.gate_kind | ffn.fused_gate_up | channel_mixer | moe.router | moe.shared_experts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 | STANDARD | SPLIT | GQA | PRE/H | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | 1e6 | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| Qwen3-MoE | STANDARD | SPLIT | GQA | PRE/H | 1.0 | prefix | F | — | CAUSAL/SWA | optional | — | SH | — | 1e6 | STD_W | PRE | — | NONE | SWIGLU | F | FFN/MoE-SM | softmax | 0 |
| Qwen2.5-VL | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — (M-RoPE) | 1e6 | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| Llama 3 | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — / LL3 | 5e5 | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| TinyLlama | STANDARD | SPLIT | GQA-32:4 | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | 1e4 | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| Llama 4 Scout | STANDARD | SPLIT | GQA | (L2-norm reserved) | 1.0 | prefix | F | — | CAUSAL | — | — | IL | — / LL3 | varies | STD_W | PRE | — | NONE | SWIGLU | F | MoE-SM | softmax | varies |
| Mistral | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL/SWA | optional | — | SH | — | varies | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| Ministral | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL/SWA per-layer | yes | — | SH | — | varies | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| Mixtral | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | varies | STD_W | PRE | — | NONE | SWIGLU | F | MoE-SM | softmax+norm | 0 |
| SmolLM3 | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL (+SWA opt) | — | — | SH or NoPE | — | varies | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| Granite | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | varies | STD_W | PRE | yes (μP) | NONE | SWIGLU | F | FFN | — | — |
| Granite-4-H (attn) | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH or none | — | varies | STD_W | PRE | yes (μP) | NONE | SWIGLU | T (shared MLP) | FFN | — | — |
| Granite-4-H (mamba) | SSDSpec | — | — | — | — | — | — | — | — | — | — | — | — | — | STD_W | PRE | yes (μP) | NONE | SWIGLU | T | FFN | — | — |
| Gemma 2 | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL/SWA per-layer | yes | 50.0 | SH | — | 1e4 | 1+W | P&P | — | NONE | GEGLU | F | FFN | — | — |
| Gemma 3 | STANDARD | SPLIT | GQA | PRE/H | 1.0 | prefix | F | — | CAUSAL/SWA per-layer | yes | None | SH | — | dual θ | 1+W | P&P | — | NONE | GEGLU | F | FFN | — | — |
| Gemma 4 (local) | STANDARD | SPLIT | GQA | PRE/H | 1.0 | prefix | F | yes | SWA | yes | None | SH | — | dual θ | STD_W | P&P | — | E2B/E4B | GEGLU | F | FFN | — | — |
| Gemma 4 (global ≥12B) | STANDARD | SPLIT | GQA | PRE/H | 0.25 | proportional | T | yes | CAUSAL | — | None | SH | — | dual θ | STD_W | P&P | — | E2B/E4B | GEGLU | F | FFN | — | — |
| Phi-3 mini | STANDARD | FUSED | GQA | — | 0.5 or 1.0 | prefix | F | — | CAUSAL (+SWA opt) | optional | — | SH | — / LR | varies | STD_W | PRE | — | NONE | SWIGLU | T | FFN | — | — |
| Phi-3 small | STANDARD | FUSED | GQA | — | 1.0 | prefix | F | — | CAUSAL/BLOCK_SPARSE | — | — | SH | — | varies | STD_W (LAYER kind) | PRE | — | NONE | GEGLU (gegelu) | T | FFN | — | — |
| Phi-4-mini | STANDARD | FUSED | GQA | — | 0.75 | prefix | F | — | CAUSAL | — | — | SH | LR | varies | STD_W | PRE | — | NONE | SWIGLU | T | FFN | — | — |
| OLMo 2 | STANDARD | SPLIT | GQA | PRE/FH | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | 5e5 | STD_W | POST | — | NONE | SWIGLU | F | FFN | — | — |
| OLMoE | STANDARD | SPLIT | GQA | PRE/FH | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | 1e4 | STD_W | PRE | — | NONE | SWIGLU | F | MoE-SM | softmax | 0 |
| MiniCPM-3 | MLA | MLA_LATENT | MLA | (slot=eps) | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — / LR | 1e4 | STD_W | PRE | yes (μP) | NONE | SWIGLU | F | FFN | — | — |
| DeepSeek-V2-Lite | MLA | MLA_LATENT | MLA (no q-LoRA) | (slot=eps) | 1.0 | prefix | F | — | CAUSAL | — | — | IL | — / YR | 1e4 | STD_W | PRE | — | NONE | SWIGLU | F | FFN/MoE-SM | softmax | 2 |
| DeepSeek-V3-Lite | MLA | MLA_LATENT | MLA | (slot=eps) | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | varies | STD_W | PRE | — | NONE | SWIGLU | F | FFN/MoE-V3 | sigmoid+bias | 1 |
| DeepSeek-V3-MoE | MLA | MLA_LATENT | MLA (with q-LoRA) | (slot=eps) | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | 1e4 | STD_W | PRE | — | NONE | SWIGLU | F | FFN/MoE-V3 | sigmoid+bias | 1 |
| DeepSeek-V3.2 | DSA | MLA_LATENT | MLA + indexer | (slot=eps) | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | 1e4 | STD_W | PRE | — | NONE | SWIGLU | F | FFN/MoE-V3 | sigmoid+bias | 1 |
| DeepSeek-OCR-2 | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | varies | STD_W | PRE | — | NONE | SWIGLU | F | FFN/MoE-SM | softmax | 2 |
| Mamba-2 | SSDSpec | — | — | — | — | — | — | — | — | — | — | — | — | — | STD_W | PRE | — | NONE | — | — | (skip_ffn) | — | — |
| GOT-OCR-2.0 | STANDARD | SPLIT | MHA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | varies | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| Voxtral | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | 1e8 | STD_W | PRE | — | NONE | SWIGLU | F | FFN | — | — |
| Moshi | STANDARD | SPLIT | GQA | — | 1.0 | prefix | F | — | CAUSAL | — | — | SH | — | 1e4 | STD_W | PRE | — | NONE | SWIGLU | T (gating MLP) | FFN | — | — |

Rows: 32 family/layer-type combinations. Columns: 23 (after compaction; some
columns merged in narrative form).

---

## §6  Composition Recipes

For each common architectural pattern below, the recipe enumerates the
fields you set on `AttentionSpec`, `FFNSpec`, `RoPESpec`, etc. to express
the pattern. These recipes mirror but compact the worked examples in §7.

### 6.1 Vanilla GQA Llama-3 decoder

```python
AttentionSpec(
    n_q_heads=H, n_kv_heads=Hk, head_dim=Dh,
    kind=AttentionKind.STANDARD, qkv_layout=QKVLayout.SPLIT,
    mask_kind=MaskKind.CAUSAL,
    q_bias=False, k_bias=False, v_bias=False, o_bias=False,
    rope=RoPESpec(base_theta=500_000, basis=RoPEBasis.SPLIT_HALF),
)
FFNSpec(
    intermediate_size=I, activation=Activation.SILU,
    gate_kind=GateKind.SWIGLU,
)
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE,
    ffn_norm_position=NormPosition.PRE,
    token_mixer=attn, channel_mixer=ffn,
    pre_attn_norm=NormSpec(kind=NormKind.RMS, eps=1e-5),
    pre_ffn_norm=NormSpec(kind=NormKind.RMS, eps=1e-5),
)
```

For Llama 3.1 / 3.2: add `scaling=RoPEScaling.LLAMA3` and
`llama3_extra=Llama3RoPEParams(factor=8 or 32, low=1, high=4, ctx=8192)`.

### 6.2 Qwen3-style (GQA + QK-norm PRE-RoPE)

Add to the Llama 3 recipe:

```python
qk_norm=NormSpec(kind=NormKind.RMS, eps=rms_eps,
                 weight_mode=NormWeightMode.STANDARD_W),
qk_norm_phase=QKNormPhase.PRE_ROPE,
qk_norm_shape=QKNormShape.PER_HEAD_DH,
```

Qwen3 also bumps `base_theta` to 1_000_000 (NOT 5e5 as v1 incorrectly
claimed — see `models/qwen3/config.py:10`).

### 6.3 Gemma 4-style (sandwich + p-RoPE + cross-layer KV + PLE + v_norm + layer_scalar)

The full toolkit:

```python
AttentionSpec(
    ...,
    qk_norm=norm_spec, qk_norm_phase=PRE_ROPE, qk_norm_shape=PER_HEAD_DH,
    qk_norm_fixed_scale=None,                       # B0.6: no absorb
    attn_scale=1.0,                                 # explicit 1.0
    attention_k_eq_v=True,                          # 12B+ global ONLY
    v_norm=norm_spec, v_norm_with_scale=False,      # unit RMSNorm
    rope=RoPESpec(
        base_theta=rope_theta_global,
        basis=RoPEBasis.SPLIT_HALF,
        partial_rotary_factor=0.25,
        partial_rotary_kind="proportional",
    ),
    mask_kind=MaskKind.CAUSAL,                      # global; SWA for local
)
FFNSpec(intermediate_size=I, activation=GELU, gate_kind=GEGLU)
PLESpec(ple_dim=256, residual_scale=1/sqrt(2), injection_norm=norm_spec)
DecoderBlockSpec(
    attn_norm_position=NormPosition.PRE_AND_POST,
    ffn_norm_position=NormPosition.PRE_AND_POST,
    token_mixer=attn, channel_mixer=ffn,
    pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
    pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
    per_layer_embedding=ple_spec,        # E2B/E4B only
    final_logit_softcap=30.0,
    embedding_scale=sqrt(hidden_size),
)
```

Cross-layer KV sharing for Gemma 4 is NOT expressed on `KVCacheSpec` —
it lives on the model config and is wired per-layer at model assembly.
The owning `Gemma4Config` carries `num_kv_shared_layers` (E2B = 20,
E4B = 35) and a `kv_source_layer_idx_map()` method
(`models/gemma4/config.py:257-283`) which returns
`dict[int, int]` mapping each shared layer's idx to its source layer's
idx — derived from the (sliding_window_pattern + 1) period so each shared
layer wraps onto the same-type slot of the last unshared block. The model
factory translates this map into per-layer
`AttentionSpec.kv_source_layer_offset` values (None on owning layers,
integer offset on borrowing layers) and instantiates `SharedLayerKVCache`
(`api/kvcache.py`) at runtime.

### 6.4 MLA (DeepSeek / MiniCPM)

```python
AttentionSpec(
    n_q_heads=H, n_kv_heads=H,            # MLA: full heads everywhere
    head_dim=qk_nope_head_dim + qk_rope_head_dim,
    kind=AttentionKind.MLA,
    qkv_layout=QKVLayout.MLA_LATENT,
    mask_kind=MaskKind.CAUSAL,
    q_bias=attn_bias, k_bias=attn_bias, v_bias=attn_bias, o_bias=attn_bias,
    q_lora_rank=q_lora_rank,              # or None for V2-Lite
    kv_lora_rank=kv_lora_rank,
    qk_nope_head_dim=qk_nope_head_dim,
    qk_rope_head_dim=qk_rope_head_dim,
    v_head_dim=v_head_dim,                # often != qk_head_dim
    qk_norm=norm_spec,                    # slot reused for q_a/kv_a norm eps
    rope=RoPESpec(base_theta=10000, basis=SPLIT_HALF),
)
```

The `Attention` ctor builds the LoRA stack at `_init_mla` (`attention.py:214`).

### 6.5 Standard softmax MoE (Mixtral)

```python
expert_ffn = FFNSpec(intermediate_size=I, activation=SILU, gate_kind=SWIGLU)
MoESpec(
    n_experts=8, top_k=2, n_shared_experts=0,
    router_kind="softmax", router_norm=True,    # Mixtral always renorms
    score_correction_bias=False, group_routing=None,
    routed_scaling_factor=1.0, expert_ffn=expert_ffn,
)
# DecoderBlockSpec.channel_mixer = MoESpec
```

### 6.6 Sigmoid+bias MoE with group routing (DeepSeek-V3)

```python
gr = GroupRoutingSpec(n_groups=8, topk_per_group=4)
MoESpec(
    n_experts=256, top_k=8, n_shared_experts=1,
    router_kind="sigmoid_plus_bias", router_norm=True,
    score_correction_bias=True, group_routing=gr,
    routed_scaling_factor=2.5,
    expert_ffn=FFNSpec(...),
)
```

### 6.7 Mamba-2 SSD

```python
ssm = SSMSpec(d_state=128, d_conv=4, d_inner=hidden*expand,
              expand_factor=2, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
              conv_bias=True, bias=False, use_fast_path=False)
SSDSpec(
    base=ssm, chunk_size=256, headdim=64, ngroups=8, n_heads=128,
    time_step_limit_low=0.0, time_step_limit_high=float("inf"),
    layer_norm_epsilon=1e-5, residual_in_fp32=True,
)
DecoderBlockSpec(
    attn_norm_position=PRE, ffn_norm_position=PRE,
    token_mixer=ssd_spec,
    channel_mixer=placeholder_ffn,        # ignored
    pre_attn_norm=norm_spec,
    skip_ffn=True,                        # canonical Mamba-2
)
```

### 6.8 Hybrid Mamba + Attention (Granite-4-H)

Per-layer dispatch in `to_block_spec(layer_idx)`:

```python
if is_mamba_layer:
    token_mixer = SSDSpec(...)
else:
    token_mixer = AttentionSpec(
        n_q_heads=..., n_kv_heads=..., head_dim=...,
        kind=AttentionKind.STANDARD, qkv_layout=SPLIT,
        mask_kind=CAUSAL,
        attn_scale=attention_multiplier,          # μP scale
        rope=None,                                 # NoPE attention
    )
DecoderBlockSpec(
    ..., token_mixer=token_mixer,
    channel_mixer=FFNSpec(fused_gate_up=True, ...),  # shared MLP
    residual_scale=residual_multiplier,
    skip_ffn=False,                                 # mamba ALSO has FFN sublayer!
)
```

Note: Granite-4-H is the FIRST family where SSM layers ALSO have a downstream
FFN sublayer (`models/granite4_h/config.py:9-22`). Canonical Mamba-2 sets
`skip_ffn=True`; Granite-4-H sets `skip_ffn=False` even for mamba layers.

### 6.9 POST-norm (OLMo 2)

```python
DecoderBlockSpec(
    attn_norm_position=NormPosition.POST,
    ffn_norm_position=NormPosition.POST,
    token_mixer=attn, channel_mixer=ffn,
    pre_attn_norm=None, post_attn_norm=norm_spec,
    pre_ffn_norm=None,  post_ffn_norm=norm_spec,
)
```

Combined with `qk_norm_shape=FULL_HDH` (OLMo 2's distinctive head-dim norm).

### 6.10 Per-layer SWA alternation (Gemma 3)

In `to_block_spec(layer_idx)`:

```python
is_sliding = (layer_idx + 1) % sliding_window_pattern != 0
mask_kind = MaskKind.SWA if is_sliding else MaskKind.CAUSAL
sw = sliding_window if is_sliding else None
rope_theta = rope_theta_local if is_sliding else rope_theta_global
# Build AttentionSpec with these
```

Gemma 3 also uses dual θ (10k for local, 1M for global).

### 6.11 NoPE alternation (SmolLM3, Llama 4)

In `to_block_spec(layer_idx)`:

```python
if self.layer_uses_rope(layer_idx):
    rope = RoPESpec(...)
else:
    rope = None
# attention.rope is None on NoPE layers; forward skips rope_apply
```

The attention block handles `rope=None` at `attention.py:208-212` and
`356-357`.

### 6.12 Fused QKV (Phi-3)

```python
AttentionSpec(
    ...,
    qkv_layout=QKVLayout.FUSED,
    q_bias=False, k_bias=False, v_bias=False, o_bias=False,
)
FFNSpec(..., fused_gate_up=True, gate_bias=False, up_bias=False)
```

The attention block allocates a single `qkv_proj: Linear(H, (Hq+2*Hk)*Dh)`
and slices Q/K/V at forward time. Reject: `attention_k_eq_v=True` and
`fused_gate_up=True` with mixed gate/up biases.

### 6.12b Visual Causal Flow mask (B8 hook, currently unused)

When a future family wires `block_bidirectional_mask=True` plus a real
`vision_token_count` per forward, the attention block:

```python
i_idx = torch.arange(S, device=device).unsqueeze(1)   # [S, 1]
j_idx = torch.arange(T, device=device).unsqueeze(0)   # [1, T]
V = vision_token_count
q_pos = start_pos + i_idx
is_vision_q = q_pos < V
is_vision_k = j_idx < V
keep_text_text = (~is_vision_k) & (~is_vision_q) & (j_idx <= q_pos)
keep_vis_vis = is_vision_k & is_vision_q
keep_text_vis = is_vision_k & (~is_vision_q)
keep = keep_vis_vis | keep_text_text | keep_text_vis
```

The vision prefix attends bidirectionally; text-to-text is causal; text-to-
vision attends backward; vision-to-text is suppressed (vision can't see
future text). DeepSeek-OCR-2 in HF v5.10.2 actually uses CAUSAL — the
BLOCK_BIDIRECTIONAL flag is reserved for the spec hook but not exercised
numerically.

### 6.12c M-RoPE (Qwen2.5-VL)

```python
RoPESpec(
    base_theta=rope_theta,                # 1_000_000 for Qwen2.5-VL
    basis=RoPEBasis.SPLIT_HALF,
    scaling=RoPEScaling.NONE,
    mrope_section=(16, 24, 24),           # T, H, W bands per band; sums to Dh/2 = 64
)
```

At forward time, `position_ids` enters as `[3, B, S]` (T/H/W axes). The
`RoPE.forward` 3-d position_ids branch (`rope.py:324-371`) rebuilds cos/sin
against the 3-axis `position_ids` tensor on the fly (cannot precompute
because position_ids is arbitrary per forward).

The intra-head dispatch is handled by `ops.rope_apply_mrope` — see
§1.17b for the stitching math.

### 6.13 CLA cross-layer sharing (Hunyuan-Large)

The hooks live on `AttentionSpec.kv_source_layer_offset` (`api/specs.py:278`).
Hunyuan-Large (`models/hunyuan_large/`) wires CLA by setting
`kv_source_layer_offset=-1` on odd-indexed layers — those layers omit
k_proj/v_proj entirely and borrow K/V from the preceding (even-indexed)
layer at forward time. Source: Hunyuan-Large `config.json` `use_cla=True`,
`cla_share_factor=2`.

---

## §7  Worked Examples — Full `to_block_spec(layer_idx)` for 5 Families

### 7.1 Qwen3 — the M1 anchor

The simplest production family. From `models/qwen3/config.py:83-120`:

```python
def to_block_spec(self) -> specs.DecoderBlockSpec:
    norm_spec = specs.NormSpec(                                 # line 84
        kind=types.NormKind.RMS, eps=self.rms_norm_eps,
        weight_mode=types.NormWeightMode.STANDARD_W,
    )
    qk_norm_spec = norm_spec                                    # line 88
    attn_spec = specs.AttentionSpec(                            # line 89
        n_q_heads=self.num_attention_heads,                     # 90
        n_kv_heads=self.num_key_value_heads,                    # 91 (GQA)
        head_dim=self.head_dim,                                 # 92 (explicit, NOT hidden/H)
        kind=types.AttentionKind.STANDARD,                      # 93
        qkv_layout=types.QKVLayout.SPLIT,                       # 94
        mask_kind=types.MaskKind.CAUSAL,                        # 95
        q_bias=False, k_bias=False, v_bias=False, o_bias=False, # 96
        qk_norm=qk_norm_spec,                                   # 97 (Qwen3 QK-norm)
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,               # 98
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,            # 99
        rope=specs.RoPESpec(                                    # 100
            base_theta=self.rope_theta,                         # 101 (1_000_000)
            basis=types.RoPEBasis.SPLIT_HALF,                   # 102
            scaling=types.RoPEScaling.NONE,                     # 103
        ),
    )
    ffn_spec = specs.FFNSpec(                                   # 106
        intermediate_size=self.intermediate_size,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
        fused_gate_up=False,
        gate_bias=False, up_bias=False, down_bias=False,
    )
    return specs.DecoderBlockSpec(                              # 113
        attn_norm_position=types.NormPosition.PRE,              # 114
        ffn_norm_position=types.NormPosition.PRE,               # 115
        token_mixer=attn_spec,                                  # 116
        channel_mixer=ffn_spec,                                 # 117
        pre_attn_norm=norm_spec,                                # 118
        pre_ffn_norm=norm_spec,                                 # 119
    )
```

Axes touched: **STANDARD attention, GQA shape, explicit `head_dim`, QK-norm
PRE-RoPE PER_HEAD_DH, SwiGLU, PRE-norm, base_theta = 1e6**.

### 7.2 Gemma 4 — the most architecturally adventurous

From `models/gemma4/config.py:112-195`:

```python
def to_block_spec(self, layer_idx: int) -> specs.DecoderBlockSpec:
    is_global = self.is_global_layer(layer_idx)                 # 122
    head_dim_eff = (self.global_head_dim if is_global           # 123
                    else self.head_dim)
    attention_k_eq_v_eff = self.attention_k_eq_v and is_global  # 124
    rope_theta_eff = (self.rope_theta_global if is_global       # 131
                      else self.rope_theta_local)
    partial_eff = (self.partial_rotary_factor_global if is_global
                   else 1.0)                                     # 132
    mask_eff = (types.MaskKind.CAUSAL if is_global               # 133
                else types.MaskKind.SWA)
    sw_eff = None if is_global else self.sliding_window          # 134

    norm_spec = specs.NormSpec(                                  # 140
        kind=types.NormKind.RMS, eps=self.rms_norm_eps,
        weight_mode=types.NormWeightMode.STANDARD_W,             # B0.6: NOT 1+W
    )
    qk_norm_spec = norm_spec                                     # 144

    attn = specs.AttentionSpec(                                  # 145
        n_q_heads=self.num_attention_heads,
        n_kv_heads=self.num_key_value_heads,
        head_dim=head_dim_eff,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=mask_eff,
        sliding_window=sw_eff,
        qk_norm=qk_norm_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        qk_norm_fixed_scale=None,            # B0.6: drop absorb
        attn_scale=1.0,                       # explicit
        attention_k_eq_v=attention_k_eq_v_eff,  # 12B+ global = True
        v_norm=norm_spec,                     # B0.6: Gemma4 V-norm
        v_norm_with_scale=False,              # unit RMSNorm
        rope=specs.RoPESpec(
            base_theta=rope_theta_eff,
            basis=types.RoPEBasis.SPLIT_HALF,
            partial_rotary_factor=partial_eff,      # 0.25 for global
            partial_rotary_kind="proportional",     # Gemma 4 semantic
        ),
    )
    ffn = specs.FFNSpec(                                          # 174
        intermediate_size=self.intermediate_size,
        activation=types.Activation.GELU,
        gate_kind=types.GateKind.GEGLU,
    )
    ple_spec = None                                                # 179
    if self.use_per_layer_embedding:                               # 180
        ple_spec = specs.PLESpec(
            ple_dim=self.ple_dim,                                  # 256
            residual_scale=1.0 / (2 ** 0.5),
            injection_norm=norm_spec,
        )
    return specs.DecoderBlockSpec(                                 # 186
        attn_norm_position=types.NormPosition.PRE_AND_POST,        # sandwich
        ffn_norm_position=types.NormPosition.PRE_AND_POST,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
        per_layer_embedding=ple_spec,
        final_logit_softcap=self.final_logit_softcap,              # 30.0
        embedding_scale=self.hidden_size ** 0.5,                   # Gemma signature
    )
```

Axes touched: **per-layer local/global dispatch (mask + RoPE theta + partial_rotary),
attention_k_eq_v on global layers ≥12B, v_norm (unit RMSNorm), partial_rotary_kind="proportional",
sandwich norm (PRE_AND_POST), GeGLU, PLE injection on E2B/E4B, layer_scalar buffer,
final_logit_softcap = 30, embedding_scale = sqrt(D), KV share scheme = SAME_BLOCK_SHARED**.

### 7.3 MiniCPM-3 — MLA reference

From `models/minicpm3/config.py:130-228`:

```python
def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
    import math
    residual_scale = self.scale_depth / math.sqrt(                # 134
        self.num_hidden_layers)                                    # μP

    if self.rope_type == "longrope":                               # 142
        scale = self.max_position_embeddings / \
                self.original_max_position_embeddings
        attention_factor = math.sqrt(
            1.0 + math.log(scale) / math.log(self.original_max_position_embeddings))
        rope_spec = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.LONGROPE,
            longrope_extra=specs.LongRoPEParams(
                short_factor=self.longrope_short_factor,
                long_factor=self.longrope_long_factor,
                original_max_position_embeddings=self.original_max_position_embeddings,
                attention_factor=attention_factor,
            ),
            partial_rotary_factor=1.0,
            partial_rotary_kind="prefix",
        )
    else:                                                          # 167
        rope_spec = specs.RoPESpec(base_theta=self.rope_theta,
                                    basis=SPLIT_HALF,
                                    scaling=RoPEScaling.NONE)

    norm_spec = specs.NormSpec(kind=RMS, eps=self.rms_norm_eps,    # 176
                                weight_mode=STANDARD_W)

    attn_spec = specs.AttentionSpec(                                # 181
        n_q_heads=self.num_attention_heads,
        n_kv_heads=self.num_attention_heads,    # MLA: H heads everywhere
        head_dim=self.qk_head_dim,               # qk_nope + qk_rope = 96
        kind=types.AttentionKind.MLA,            # ←
        qkv_layout=types.QKVLayout.MLA_LATENT,   # ←
        mask_kind=types.MaskKind.CAUSAL,
        q_bias=self.attention_bias, k_bias=self.attention_bias,
        v_bias=self.attention_bias, o_bias=self.attention_bias,
        q_lora_rank=self.q_lora_rank,            # 768
        kv_lora_rank=self.kv_lora_rank,          # 256
        qk_nope_head_dim=self.qk_nope_head_dim,  # 64
        qk_rope_head_dim=self.qk_rope_head_dim,  # 32
        v_head_dim=self.v_head_dim,              # 64
        qk_norm=norm_spec,        # MLA slot: q_a/kv_a layernorm eps
        rope=rope_spec,
    )

    ffn_spec = specs.FFNSpec(                                       # 203
        intermediate_size=self.intermediate_size,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )

    return specs.DecoderBlockSpec(                                  # 210
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn_spec, channel_mixer=ffn_spec,
        pre_attn_norm=norm_spec, pre_ffn_norm=norm_spec,
        residual_scale=residual_scale,    # μP per-sublayer
        embedding_scale=self.scale_emb,   # μP embedding scale
        logits_scale=self.dim_model_base / self.hidden_size,  # μP logits
    )
```

Axes touched: **MLA (5 dims), LongRoPE, residual_scale (μP), embedding_scale,
logits_scale, qk_head_dim != v_head_dim**.

### 7.4 Granite-4-H — per-layer Mamba/attn alternation

From `models/granite4_h/config.py:175-263`:

```python
def to_block_spec(self, layer_idx: int) -> specs.DecoderBlockSpec:
    norm_spec = specs.NormSpec(kind=RMS, eps=self.rms_norm_eps,    # 176
                                weight_mode=STANDARD_W)

    ffn_spec = specs.FFNSpec(                                       # 189
        intermediate_size=self.shared_intermediate_size,
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
        fused_gate_up=True,                  # Granite-4-H shared MLP
        gate_bias=False, up_bias=False, down_bias=False,
    )

    is_mamba = self.is_mamba_layer(layer_idx)                       # 197
    if is_mamba:                                                    # 198
        ssm_inner = self.mamba_expand * self.hidden_size
        ssm = specs.SSMSpec(
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            d_inner=ssm_inner,
            expand_factor=self.mamba_expand,
            dt_min=self.time_step_min, dt_max=self.time_step_max,
            dt_init_floor=self.time_step_floor,
            conv_bias=self.mamba_conv_bias,
            bias=self.mamba_proj_bias,
            use_fast_path=False,
            activation=types.Activation.SILU,
        )
        token_mixer = specs.SSDSpec(
            base=ssm, chunk_size=self.mamba_chunk_size,
            headdim=self.mamba_d_head,
            ngroups=self.mamba_n_groups, n_heads=self.mamba_n_heads,
            time_step_limit_low=self.time_step_limit[0],
            time_step_limit_high=self.time_step_limit[1],
            layer_norm_epsilon=self.rms_norm_eps,
            residual_in_fp32=True,
        )
    else:                                                            # 228
        rope_spec = None
        if self.position_embedding_type == "rope":
            rope_spec = specs.RoPESpec(base_theta=self.rope_theta,
                                        basis=SPLIT_HALF)
        token_mixer = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            attn_scale=self.attention_multiplier,    # μP attention scale
            rope=rope_spec,                          # None for "nope"
        )

    return specs.DecoderBlockSpec(                                  # 251
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=token_mixer, channel_mixer=ffn_spec,
        pre_attn_norm=norm_spec, pre_ffn_norm=norm_spec,
        residual_scale=self.residual_multiplier,    # μP per-sublayer
        embedding_scale=self.embedding_multiplier,
        logits_scale=self.logits_scaling,
        skip_ffn=False,                              # SSM + FFN BOTH run
    )
```

Axes touched: **per-layer Mamba/attn dispatch, NoPE attention, μP three-scalar
stack, fused_gate_up shared MLP, mamba + FFN coexistence (skip_ffn=False)**.

### 7.5 DeepSeek-V2-Lite — MoE + MLA + YARN

From `models/deepseek_v2_lite/config.py:247-267`:

```python
def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
    norm = self._norm_spec()
    attn = self._attn_spec()
    if layer_idx >= self.first_k_dense_replace:                    # MoE layers
        channel = self._moe_spec()
    else:                                                          # dense FFN layer 0
        channel = self._dense_ffn_spec()
    return specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE,
        ffn_norm_position=types.NormPosition.PRE,
        token_mixer=attn,
        channel_mixer=channel,
        pre_attn_norm=norm,
        pre_ffn_norm=norm,
    )

# _attn_spec (line 186):
def _attn_spec(self) -> specs.AttentionSpec:
    return specs.AttentionSpec(
        n_q_heads=self.num_attention_heads,
        n_kv_heads=self.num_attention_heads,
        head_dim=self.qk_head_dim,                # qk_nope+qk_rope = 192
        kind=types.AttentionKind.MLA,
        qkv_layout=types.QKVLayout.MLA_LATENT,
        mask_kind=types.MaskKind.CAUSAL,
        q_bias=False, k_bias=False, v_bias=False, o_bias=False,
        q_lora_rank=self.q_lora_rank,            # None — direct q_proj
        kv_lora_rank=self.kv_lora_rank,          # 512
        qk_nope_head_dim=self.qk_nope_head_dim,  # 128
        qk_rope_head_dim=self.qk_rope_head_dim,  # 64
        v_head_dim=self.v_head_dim,              # 128
        qk_norm=norm,
        rope=self._rope_spec(),                   # YARN + INTERLEAVED
    )

# _rope_spec (line 154):
def _rope_spec(self) -> specs.RoPESpec:
    if self.rope_type == "yarn":
        yarn = specs.YarnRoPEParams(
            factor=self.yarn_factor,              # 40
            original_max_position_embeddings=self.yarn_original_max_position_embeddings,
            beta_fast=self.yarn_beta_fast, beta_slow=self.yarn_beta_slow,
            mscale=self.yarn_mscale,              # 0.707
            mscale_all_dim=self.yarn_mscale_all_dim,  # 0.707
        )
        return specs.RoPESpec(
            base_theta=self.rope_theta,           # 10000
            basis=types.RoPEBasis.INTERLEAVED,    # V2 complex-multiply
            scaling=types.RoPEScaling.YARN,
            yarn_extra=yarn,
        )
    return specs.RoPESpec(base_theta=self.rope_theta,
                          basis=INTERLEAVED, scaling=NONE)

# _moe_spec (line 219):
def _moe_spec(self) -> specs.MoESpec:
    expert_ffn = specs.FFNSpec(
        intermediate_size=self.moe_intermediate_size,  # 1408
        activation=types.Activation.SILU,
        gate_kind=types.GateKind.SWIGLU,
    )
    group = None  # V2-Lite has n_group=1, topk_group=1 → degenerate
    return specs.MoESpec(
        n_experts=self.n_routed_experts,         # 64
        top_k=self.num_experts_per_tok,          # 6
        n_shared_experts=self.n_shared_experts,  # 2
        router_kind="softmax",                    # V2
        router_norm=self.norm_topk_prob,          # False for V2
        score_correction_bias=False,
        group_routing=group,
        routed_scaling_factor=self.routed_scaling_factor,  # 1.0
        expert_ffn=expert_ffn,
    )
```

Axes touched: **MLA with q_lora_rank=None (direct q_proj path),
qk_head_dim=192 vs v_head_dim=128, YARN scaling, INTERLEAVED RoPE basis,
per-layer dense/MoE dispatch, softmax MoE with shared experts, degenerate group routing**.

---

## §7a The HF dispatch families — a structural overview

When a model "exports a layer" — i.e. surfaces a `to_block_spec(layer_idx)`
method that returns a `DecoderBlockSpec` — the structural overhead is
small but real. Looking at the 30 landed families, two structural
patterns dominate:

### Pattern A: Uniform layer

Every layer has the same spec. The `to_block_spec` method either takes
no `layer_idx` arg (Qwen3, Llama 3, Mistral, OLMo 2) or accepts it for
parity with other families but ignores it (MiniCPM-3, Mamba-2,
Granite, OLMoE, Mixtral, GOT-OCR 2.0, Voxtral, Moshi, Qwen2.5-VL).

The model-level loader builds N identical block specs and uses one shared
weight loader per family.

### Pattern B: Per-layer dispatch

Some axis varies per layer. The `to_block_spec(layer_idx)` reads from a
per-layer config field (e.g. `layer_types[layer_idx]`,
`layer_uses_rope(layer_idx)`, `mlp_layer_types[layer_idx]`) and returns
a different spec accordingly. The model-level loader runs the same
weight loader (the per-layer dispatch doesn't change the WEIGHT keys,
only the block's INTERNAL behaviour).

The per-layer dispatch families (§13.6) all follow Pattern B but vary in
WHAT axis they dispatch on:

- **SWA pattern** (Gemma 2/3/4, Ministral): `layer_types[layer_idx]` →
  mask kind, sliding_window, sometimes rope_theta, sometimes head_dim.
- **NoPE pattern** (SmolLM3, Llama 4): `no_rope_layers[layer_idx]` →
  `rope = None` for NoPE layers.
- **Dense/MoE pattern** (Mixtral never; Qwen3-MoE, DeepSeek-V2-Lite,
  V3-Lite, V3-MoE, V3.2, OCR-2): `first_k_dense_replace` or
  `mlp_layer_types[layer_idx]` → `channel_mixer` is `FFNSpec` for dense
  layers, `MoESpec` for MoE.
- **SSM/attn pattern** (Granite-4-H): `layer_types[layer_idx]` →
  `token_mixer` is `AttentionSpec` for attention layers, `SSDSpec` for
  mamba layers.
- **Dense/sparse pattern** (Phi-3-small): `dense_attention_every_n_layers`
  → `mask_kind = CAUSAL` on dense layers, `BLOCK_SPARSE` on sparse layers.

### Composition of dispatch axes

Gemma 4 combines THREE dispatch axes per layer:
1. is_global (CAUSAL vs SWA, plus rope_theta, plus head_dim, plus
   partial_rotary_factor).
2. is_global AND `attention_k_eq_v` config flag (=> True for 12B+ global).
3. PLE injection (Gemma 4 E2B/E4B only — uniform across all layers).

So Gemma 4 E2B layer 0 (local) has: SWA, head_dim=256, partial=1.0,
attention_k_eq_v=False, qk_norm fixed-scale local, PLE on.
Gemma 4 12B layer 5 (global) has: CAUSAL, head_dim=256, partial=0.25,
attention_k_eq_v=True, qk_norm fixed-scale global, PLE off.

This is the most axis-rich per-layer dispatch in the rolled-out set.

## §7b The HF weight-loader pattern (`load_<family>_layer`)

Each family also ships a `load_<family>_layer` function that translates HF
state-dict keys to the API attribute names. The pattern follows the
structure of the `DecoderBlock` we just built. For Qwen3:

```python
def load_qwen3_layer(blk: DecoderBlock, hf_state: dict, layer_idx: int):
    p = f"model.layers.{layer_idx}."
    # Norm
    blk.pre_attn_norm.weight.data = hf_state[p+"input_layernorm.weight"]
    blk.pre_ffn_norm.weight.data  = hf_state[p+"post_attention_layernorm.weight"]
    # Attention projections
    blk.attention.q_proj.weight.data = hf_state[p+"self_attn.q_proj.weight"]
    blk.attention.k_proj.weight.data = hf_state[p+"self_attn.k_proj.weight"]
    blk.attention.v_proj.weight.data = hf_state[p+"self_attn.v_proj.weight"]
    blk.attention.o_proj.weight.data = hf_state[p+"self_attn.o_proj.weight"]
    # QK norms
    blk.attention.q_norm.weight.data = hf_state[p+"self_attn.q_norm.weight"]
    blk.attention.k_norm.weight.data = hf_state[p+"self_attn.k_norm.weight"]
    # FFN
    blk.feedforward.gate_proj.weight.data = hf_state[p+"mlp.gate_proj.weight"]
    blk.feedforward.up_proj.weight.data   = hf_state[p+"mlp.up_proj.weight"]
    blk.feedforward.down_proj.weight.data = hf_state[p+"mlp.down_proj.weight"]
```

Subtleties to watch for in the weight loader:

- **`tie_word_embeddings`** — when true, the lm_head weight is the
  embedding weight transposed. The block-level loader doesn't usually
  handle this; the model-level loader does.
- **Sandwich norm** — for Gemma 2/3/4, the HF state-dict keys are
  `pre_feedforward_layernorm`, `post_feedforward_layernorm`,
  `input_layernorm`, `post_attention_layernorm`. The mapping to API attrs
  is `input_layernorm → pre_attn_norm`,
  `post_attention_layernorm → post_attn_sublayer_norm`,
  `pre_feedforward_layernorm → pre_ffn_norm`,
  `post_feedforward_layernorm → post_ffn_sublayer_norm`. This needs the
  block to have been constructed with `attn_norm_position=PRE_AND_POST`.
- **MLA weight names** — `self_attn.q_a_proj`, `self_attn.q_a_layernorm`,
  `self_attn.q_b_proj`, `self_attn.kv_a_proj_with_mqa`,
  `self_attn.kv_a_layernorm`, `self_attn.kv_b_proj`, `self_attn.o_proj`.
  For V2-Lite (no q-LoRA), the HF key is just `self_attn.q_proj`.
- **MoE weight names** — `mlp.gate.weight` for the router,
  `mlp.experts.<e>.gate_proj.weight` / `up_proj` / `down_proj` per expert.
  The API stores experts as packed tensors `experts_gate_up: [E, 2*I, H]`
  and `experts_down: [E, H, I]` — the loader has to STACK individual
  expert weights into these packed tensors and concatenate gate+up along
  the row dim.
- **Shared experts** — for DeepSeek MoE, the HF key is
  `mlp.shared_experts.*` and these map onto the API's `shared_experts`
  sub-module (which is a regular `FeedForward`).
- **Gemma 4 layer_scalar** — `model.layers.<L>.layer_scalar` (a
  shape-`[1]` buffer) loads directly into `blk.layer_scalar`.
- **PLE table** — `model.per_layer_embedding_tokens.weight` is the
  vocab×ple_dim table; `model.layers.<L>.per_layer_input_gate.weight`
  and `model.layers.<L>.per_layer_projection.weight` are per-layer.
- **Mamba A_log** — `backbone.layers.<L>.mixer.A_log` is a per-head bias
  in HF; loads onto `blk.attention.A_log` (note `blk.attention` is the
  attribute name even for SSM blocks, per the B7 design comment in
  `block.py:118-119`).

The loader pattern is uniform — each family's `load_*_layer` follows the
same structural shape with family-specific key suffixes.

## §8  IR Drift Log (chronological)

Source-grounded log of every spec / op / module extension that landed
after spec v3 was written. Commits are referenced by short SHA; the full
SHA list is at the end of this section.

### 8.1 M1 fix-pass — close the 16-op floor (2026-06-05 to 2026-06-06)

**Commit `31378be`** — 7 new ops added: `layer_norm`, `softmax`, `top_k`,
`gather`, `scatter`, `conv1d`, `selective_scan` (stub).
**Reason:** spec v3 §8.1 promised a 16-op floor (research/03 v2 §3); M1
landed only the 8-op Qwen3 subset. The fix-pass closed the gap.
**Test:** `tests/api/test_ops.py` (the canonical op test file — there is no separate `test_ops_floor.py`; audit 2026-06-08 corrected).

**Commit `5aa8374`** — 6 new dataclasses added: `MoESpec`, `SSMSpec`,
`SSDSpec`, `ConvSpec`, `GroupRoutingSpec`, `LayerScaleSpec`.
**Reason:** spec v3 listed ~18 dataclasses; M1 landed 4. Fix-pass closed the
gap (so subsequent batches could pick them up).

**Commit `30a4848`** — `models/qwen3/layer.md` added per the v3 §6 schema.
**Reason:** docs gap.

**Commit `12e53f6`** — drop dead `DecoderBlockSpec.input_norm`; rename
block attrs to `pre_attn_norm` / `pre_ffn_norm`.
**Reason:** the v3 design had a single `input_norm` field; reality is two
distinct norm slots (attn vs FFN).

### 8.2 B0.5 — Gemma 4 spec hooks (2026-06-06)

**Commit `cf4bd24`** — spec extensions for Gemma 4: `partial_rotary_factor`
(default 1.0), `attention_k_eq_v` (default False), `qk_norm_fixed_scale`
(default None), `PLESpec`, `share_scheme` on `KVCacheSpec`.
**Reason:** Gemma 4 has per-layer-type partial RoPE (global=0.25),
K=V on global layers for 12B+, fixed-scale QK norm, PLE, cross-layer KV.
None of these existed in spec v3.

**Commit `7ab90d1`** — ops `rope_apply_partial` and `gelu_pytorch_tanh`.
**Reason:** Gemma 4 GEGLU + partial RoPE both needed pure ops first.

**Commit `55332d6`** — `RoPE` module supports `partial_rotary_factor`.
**Reason:** consume the spec extension above.

**Commit `60a9caa`** — `Attention` supports `attention_k_eq_v` (alias
V := K, no separate v_proj), `qk_norm_fixed_scale` (effective_scale=1.0
when set), `SWA` mask.

**Commit `9f98fb7`** — `FeedForward` supports GEGLU with `gelu_pytorch_tanh`.

**Commit `6dbe232`** — `ShareScheme` enum for cross-layer KV sharing
(axis A18 from research/01 v3). Note: this enum was later removed in
v5 Phase 1 commit `8532879` in favour of the per-layer
`AttentionSpec.kv_source_layer_offset` model (see §2.20). The Gemma 4
batch (B0.5) is the historical context for why the IR needed a KV-sharing
hook at all.

**Commit `c6d5836`** — `PerLayerEmbedding` module.

**Commit `e951419`** — `SharedLayerKVCache` for Gemma 4 cross-layer KV.

**Commit `d9d1b7a`** — `DecoderBlock` supports `PRE_AND_POST` sandwich norm.

### 8.3 B0.6 — Gemma 4 numerical-gate fixes (2026-06-06)

These fixes landed after reading `modeling_gemma4.py` end-to-end revealed
the B0.5 spec didn't match HF behaviour exactly.

**Commit `18f1bd6`** — Gemma 4 RMSNorm is `STANDARD_W`, NOT `ONE_PLUS_W`.
**Reason:** verified at `modeling_gemma4.py:193-211` — Gemma4RMSNorm
forward is `normed_output * self.weight`, no `(1 + w)` term. Gemma 1/2/3
were the only `1+w` users.

**Commit `50829ed`** — drop the `qk_norm_fixed_scale` absorb on Gemma 4.
**Reason:** HF Gemma 4 attention sets `self.scaling = 1.0` and the QK norms
are plain Gemma4RMSNorm (no fixed-scale gain). The B0.5 IR had baked
`sqrt(Dh) * fixed_scale` into the QK weight at load time; this commit
removed that absorb and surfaced the spec change.

**Commit `496cc58`** — add `AttentionSpec.v_norm` (and `v_norm_with_scale`).
**Reason:** discovered `modeling_gemma4.py:1215` has
`self.v_norm = Gemma4RMSNorm(self.head_dim, with_scale=False)`. A per-head
unit RMSNorm on V before transpose+cache.write — completely missing from
spec v3.

**Commit `0bc9d0c`** — proportional RoPE geometry on partial rotation.
**Reason:** Gemma 4 uses the "proportional" rope_type — inv_freq is built at
FULL head_dim with zero padding, and `rotate_half` pairs across the full
head_dim. Spec v3 implicitly assumed Phi-3's "prefix" semantic.

**Commit `9d5daea`** — PLE injection AT-END + layer_scalar buffer.
**Reason:** verified against `modeling_gemma4.py:1382, 1384-1389, 1446-1455`
that PLE is applied AFTER the FFN residual add, then a per-layer scalar
multiply is applied to every layer (even when PLE is off).

### 8.4 B1 — Llama 3, SmolLM3, Mistral, TinyLlama (2026-06-06)

**Commit `3ea394f`** — `api/attention` allows `rope=None` for NoPE layers.
**Reason:** SmolLM3 disables RoPE on a periodic subset of layers
(`modeling_smollm3.py:211, 233-235`).

### 8.5 B2a — Phi-3 mini, Phi-4-mini, Granite (2026-06-07)

**Commit `4d6b9a5`** — api extensions for Phi-3: `FUSED` QKV layout, fused
gate/up FFN, `LongRoPEParams`, `LONGROPE` scaling enum.
**Reason:** Phi-3 has one big `qkv_proj` of size `(Hq+2*Hk)*Dh`
(`modeling_phi3.py:222-224`); one big `gate_up_proj` of size `2*I`
(`modeling_phi3.py:54`); and LongRoPE with two factor tables
(`modeling_rope_utils.py:462-547`).

**Commit `ae6dc64`** — `partial_rotary_kind="prefix"` for Phi-3/4 partial
rotation.
**Reason:** Phi-3 / Phi-4 use prefix RoPE — rotate the first `Dh_rot`
channels with internal pairing (`modeling_phi3.py:199-204`). Gemma 4 uses
proportional with zero-padded inv_freq across the full head_dim. The two
semantics are mathematically distinct on partial rotations; spec v3 did
not distinguish them.

**Commit `a62551f`** — wire `DecoderBlockSpec.residual_scale` for Granite μP.
**Reason:** Granite multiplies sublayer output by `residual_multiplier`
before the residual add (`modeling_granite.py:273, 278`). Spec v3 did not
have a per-sublayer scale.

**Commit `6c5d66c`** — Granite Config + μP scalar plumbing
(`embedding_multiplier`, `attention_multiplier`, `logits_scaling`,
`residual_multiplier`) all surfaced through `DecoderBlockSpec`.

### 8.6 B2b — MiniCPM-3 MLA (2026-06-07)

**Commit `453df69`** — `AttentionSpec` MLA fields: `q_lora_rank`,
`kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`. Plus
`QKVLayout.MLA_LATENT` and `AttentionKind.MLA`. Plus `BlockSparse` /
`GeGELU` enum entries for forward compat.
**Reason:** MiniCPM-3 attention is Multi-head Latent Attention with five
distinct dimension parameters (`modeling_minicpm.py:351-357`).

**Commit `b4c8e6c`** — `ContiguousKVCache.v_head_dim` kwarg.
**Reason:** MLA stores decompressed K at `qk_head_dim` and V at `v_head_dim`
which is often a different size (MiniCPM-3: 96 vs 64).

**Commit `a30e9aa`** — `api.attention` MLA branch.
**Reason:** wire the spec extension. The `_init_mla` path builds the LoRA
stack; `_forward_mla` runs the decompress + concat + sdpa sequence.

### 8.7 B3 — OLMo 2, Gemma 2, Gemma 3 (2026-06-07)

**Commit `f1dcd46`** — `api.block` `POST`-norm + `ops.sdpa`
`attn_logit_softcap`.
**Reason:** OLMo 2 has post-norm (`modeling_olmo2.py:295-333`) — distinct
from PRE and from sandwich. Gemma 2 attention has a tanh-softcap that the
PyTorch built-in `F.scaled_dot_product_attention` doesn't expose; B3 added
a manual matmul→tanh→softmax→matmul fallback in `ops.sdpa`.

### 8.8 B4 — Llama 4 Scout (2026-06-07)

**Commit `8ea9930`** — `INTERLEAVED` RoPE basis real implementation.
**Reason:** Llama 4 uses complex-multiply pairing
(`modeling_llama4.py:239-254`: `freqs_cis = polar(1, freqs)`,
`view_as_complex → complex-multiply → view_as_real → flatten`). Spec v3
had the enum value but no implementation.

### 8.9 B5 — DeepSeek-V2-Lite, V3, V3.2, MoE module (2026-06-07)

**Commit `5e352cf`** — specs: `YarnRoPEParams`, `IndexerSpec`; broaden
`DecoderBlockSpec.channel_mixer` to `Union[FFNSpec, MoESpec]`.

**Commit `71f601c`** — YARN RoPE scaling for V2-Lite / V3.
**Reason:** YARN is a linear-ramp blend of extrapolation and interpolation
inv_freq tables, plus a cos/sin multiply by `attention_factor` derived from
mscale (`modeling_rope_utils.py:327-459`).

**Commit `43a4471`** — `MoE` channel mixer — softmax (V2) and sigmoid+bias
(V3) routers.
**Reason:** V2 and V3 routers differ in 5 ways: softmax vs sigmoid,
group-routing algorithm (V2 max per group vs V3 sum of top-2),
score_correction_bias on V3, `norm_topk_prob` on V3, `routed_scaling_factor`
on V3. The `_route_softmax` and `_route_sigmoid_plus_bias` methods
encapsulate both.

**Commit `1804f21`** — `DecoderBlock` dispatches FFNSpec vs MoESpec.

**Commit `576f7a2`** — MLA supports `q_lora_rank=None` (V2-Lite direct
q_proj path).
**Reason:** V2-Lite has `q_lora_rank=null` in its config; the Q path is a
direct `q_proj` instead of the LoRA stack.

**Commit `fd672f3`** — DSA Lightning Indexer shape-only (`IndexerSpec` +
construction; forward raises).

### 8.10 B6 — Mixtral, Qwen3-MoE, OLMoE (2026-06-07)

No new spec/op extensions — exercised existing MoE machinery.

### 8.11 B7 — Mamba-2 + Granite-4-H (2026-06-07)

**Commit `7171ac5`** — SSM infrastructure: `api/ssm.py` with
`Mamba2Mixer`, `selective_scan` real implementation, `SSMStateCache`,
broadened `DecoderBlockSpec.token_mixer` to
`Union[AttentionSpec, SSMSpec, SSDSpec]`, added `skip_ffn` flag.
**Reason:** Mamba-2 has a totally different token mixer (chunked SSD scan).
SSD spec needs to drive `Mamba2Mixer` construction; `SSMStateCache` has a
different shape than `ContiguousKVCache`; canonical Mamba-2 has no FFN
sublayer.

**Commit `c8c97d4`** — Mamba-2 family full numerical gate BIT-EXACT vs HF
torch_forward.

**Commit `cd584eb`** — Granite-4-H — hybrid Mamba-2 + GQA (5:1) BIT-EXACT
numerical gate.
**Reason:** first hybrid family — `is_mamba_layer(layer_idx)` dispatches
between SSDSpec and AttentionSpec token mixer per layer. Plus the shared
MLP runs on BOTH mamba and attention layers (i.e. `skip_ffn=False`).

**Commit `38f512a`** — deferred SSM-family stubs with rationale for
Mamba-1, Mamba-3, RecurrentGemma, RWKV-7, Falcon-H1, Nemotron-3, Hymba,
Phi-4-mini-flash, Jamba, MiniMax.

### 8.12 B8 — Multimodal LM gates (2026-06-07/08)

**Commit `4b8f415`** — api hooks for M-RoPE + Visual Causal Flow:
`mrope_section` on `RoPESpec`, `ops.rope_apply_mrope`,
`MaskKind.BLOCK_BIDIRECTIONAL`, `block_bidirectional_mask` on
`AttentionSpec`, `vision_token_count` kwarg on
`Attention.forward`.
**Reason:** Qwen2.5-VL has the 3-axis M-RoPE
(`modeling_qwen2_5_vl.py:596-606`); the original DeepSeek-OCR paper has a
Visual Causal Flow mask where vision-prefix queries attend
bidirectionally and text suffix is causal.

**Commit `cb68b0c`** — GOT-OCR 2.0 family (Qwen2-0.5B decoder).
**Commit `e22d141`** — Qwen2.5-VL family — M-RoPE LM-decoder gate.
**Commit `47ff482`** — DeepSeek-OCR-2 family — LM-decoder gate (standard
MHA + softmax MoE).

### 8.13 B9 — Audio LM gates (2026-06-07)

**Commit `2eabcef`** — Moshi 7B family — main-decoder gate (MHA + fused
SwiGLU). No new IR extensions — `fused_gate_up` covers Moshi's
`GatingMLP.fc1` shape.

**Commit `9cd250c`** — Voxtral family — LM-decoder gate (Llama backbone,
`rope_theta=1e8`).

### 8.14 B10 — Quantization round-trips (2026-06-08)

**Commit `8d00ea1`** — GGUF Q4_K_M round-trip — k-quant super-block +
6-bit sub-scales.
**Commit `9b1d2dd`** — GGUF Q4_K_M real-weight test on Qwen3 q_proj.
**Commit `10a0fe7`** — FP8 E4M3 W8A8 round-trip — per-tensor W +
per-token A.
**Commit `551ddf3`** — MXFP4 round-trip — UE8M0 shared exp + E2M1 mantissas.
**Commit `e6449f6`** — LiteRT W4A8 — per-channel INT4 weights + per-tensor
INT8 activations.
**Commit `19b9dec`** — IQ2_M / AQLM codebook-quant stubs — reserved for M3.

### 8.15 Full SHA list (for archival)

| Batch | Tag | Short SHA | Description |
|-------|-----|-----------|-------------|
| M1 | bootstrap | `5068bf7` | project bootstrap — uv venv, pyproject |
| M1 | specs | `4fa0498` | api types + spec dataclasses (Qwen3 subset) |
| M1 | ops | `d9adf84` / `9db0b70` / `4eb3668` / `7ab90d1`* | ops batch (silu/add/mul/linear; rms/embed/lm_head; rope/sdpa) |
| M1 | modules | `bea2294` / `bd5d009` / `46a56d0` / `4e9aad9` / `9faa031` / `2b2e416` | RMSNorm / RoPE / KVCache / Attention / FFN / Block |
| M1 | quant | `3b5c837` | AWQ W4A16 grouped INT4 |
| M1 | qwen3 | `68f9042` / `34abb34` / `1d453c1` | Qwen3 Config + factory + weight loader |
| M1 fix | ops floor | `31378be` | 7 new ops |
| M1 fix | spec floor | `5aa8374` | 6 new dataclasses |
| M1 fix | layer.md | `30a4848` | layer.md per spec §6 schema |
| B0.5 | enum | `6dbe232` | ShareScheme enum (removed v5 Phase 1 `8532879`; replaced by per-layer `kv_source_layer_offset`) |
| B0.5 | spec | `cf4bd24` | Gemma 4 spec extensions |
| B0.5 | ops | `7ab90d1` | rope_apply_partial + gelu_pytorch_tanh |
| B0.5 | RoPE | `55332d6` | partial_rotary_factor support |
| B0.5 | Attn | `60a9caa` | attention_k_eq_v, qk_norm_fixed_scale, SWA |
| B0.5 | FFN | `9f98fb7` | GEGLU + gelu_pytorch_tanh |
| B0.5 | KV share | `e951419` | SharedLayerKVCache |
| B0.5 | PLE | `c6d5836` | PerLayerEmbedding module |
| B0.5 | Block | `d9d1b7a` | PRE_AND_POST sandwich norm |
| B0.6 | fix | `18f1bd6` | RMSNorm STANDARD_W (not 1+W) |
| B0.6 | fix | `50829ed` | drop qk_norm_fixed_scale absorption |
| B0.6 | fix | `496cc58` | add v_norm field |
| B0.6 | fix | `0bc9d0c` | proportional RoPE geometry |
| B0.6 | fix | `9d5daea` | PLE injection AT-END + layer_scalar |
| B1 | NoPE | `3ea394f` | rope=None for NoPE layers |
| B2a | spec | `4d6b9a5` | FUSED QKV, fused gate/up, LongRoPE |
| B2a | RoPE | `ae6dc64` | partial_rotary_kind='prefix' |
| B2a | Block | `a62551f` | residual_scale wiring |
| B2b | spec | `453df69` | MLA fields |
| B2b | KV | `b4c8e6c` | v_head_dim kwarg |
| B2b | Attn | `a30e9aa` | MLA branch |
| B3 | Block | `f1dcd46` | POST-norm + attn_logit_softcap in sdpa |
| B4 | RoPE | `8ea9930` | INTERLEAVED basis real impl |
| B5 | spec | `5e352cf` | YarnRoPEParams, IndexerSpec |
| B5 | RoPE | `71f601c` | YARN scaling |
| B5 | MoE | `43a4471` | softmax + sigmoid+bias routers |
| B5 | Block | `1804f21` | FFNSpec vs MoESpec dispatch |
| B5 | MLA | `576f7a2` | q_lora_rank=None path |
| B5 | DSA | `fd672f3` | IndexerSpec shape-only |
| B7 | SSM | `7171ac5` | api/ssm.py + selective_scan + Mamba2Mixer |
| B8 | hooks | `4b8f415` | M-RoPE + BLOCK_BIDIRECTIONAL hooks |

*The `7ab90d1` SHA covers a stack of multiple ops in one commit.

---

## §9  Open Extension Points

Today's IR cannot natively express the following architectures — each
requires NEW building blocks (not just new spec fields). The deferred
stubs in `models/<family>/__init__.py` document the rationale.

### 9.1 Mamba-1 selective scan kernel

The Mamba-1 selective scan uses a DIFFERENT discretisation than Mamba-2's
SSD form (per-channel `dt_rank` projection vs per-head `dt` parameter). A
new `selective_scan_mamba1` op would be needed. The `SSMSpec.dt_rank`
field is reserved for this future use.

### 9.2 Mamba-3 complex-state + MIMO decoding

The Mamba-3 paper introduces:
- **Complex-state** — the recurrent state is complex-valued (cos/sin pair
  per real-state slot).
- **MIMO decoding** — multiple inputs / multiple outputs per token, which
  requires a 2D scan shape distinct from SSD's `[B, S, H, D]`.

Both would require new dataclasses (`Mamba3Spec`?) and a new scan kernel.

### 9.3 Griffin / Hawk RG-LRU (RecurrentGemma)

RG-LRU is a gated linear recurrence with diagonal-only A matrix and a
per-channel learned forget gate. Distinct from SSD: no chunk-parallel scan
(uses a simpler linear recurrence). Would need:
- A `RGSSpec` dataclass.
- A `gated_linear_recurrence` op.
- A `GriffinMixer` building block.

### 9.4 RWKV-7 Goose WKV with outer-product state updates

RWKV-7's per-token state is itself a learnable matrix, updated by an
outer-product write per token (Test-Time-Training flavour). Distinct from
both SSD and RG-LRU. Would need:
- A `WKV7Spec` dataclass.
- A `wkv_outer_product_update` op.
- A `RWKV7Mixer` building block.
- Time-mix and channel-mix shift conv1d (kernel=2) with token-shift.
- Group-norm before the time-mix output projection (NOT RMS).

### 9.5 Hymba parallel HYBRID_PARALLEL block

Hymba runs attention AND mamba IN PARALLEL within the same block, then
sums their outputs. Today's `DecoderBlock` runs token mixer + channel
mixer in series. Would need:
- A `HybridParallelBlock` building block (or a new `parallel_token_mixers`
  field on `DecoderBlockSpec`).
- The `TokenMixerKind.HYBRID_PARALLEL` enum value.

### 9.6 Phi-4-mini-flash (Samba: Mamba-1 + attention sequential)

Samba runs Mamba-1 → attention sequentially within the same block. Today's
hybrid path (Granite-4-H) alternates Mamba and attention across DIFFERENT
layers, not within the same layer. Would need a new `SambaBlock` building
block or a stacked-mixers field on `DecoderBlockSpec`.

### 9.7 Falcon-H1, Nemotron-3 (compose existing but at huge scale)

These compose existing pieces (Mamba + attention + MoE + GQA) but with
NEW per-batch shape and weight loaders. Deferred for engineering bandwidth,
not IR limitation.

### 9.8 MiniMax Lightning Attention

A linear-attention variant with a different kernel function than RetNet /
GLA / DeltaNet. Would need a new `lightning_attention` op and a
`MiniMaxAttentionSpec`. The `AttentionKind.LINEAR_RETENTION` /
`LINEAR_DELTANET` / `LINEAR_GLA` enum values are reserved for this family.

### 9.9 DSA Lightning Indexer forward (composition only today)

The B5 landing of `IndexerSpec` is shape-only. To go numerical the
`Attention.forward` would need:
- An indexer Q/K projection forward (linear projections at `indexer_dim`).
- A score `Q_idx @ K_idx^T` computation.
- A `top_k` along the key axis per query.
- A scatter that masks out non-top-k keys in the main SDPA mask.

### 9.10 BlockSparse mask forward (Phi-3-small)

`MaskKind.BLOCK_SPARSE` is wired into the enum and the Phi-3-small config
sets it, but the `Attention.forward` raises NotImplementedError because
the block-sparse pattern (per-head blocksparse_*) requires a specific
mask-construction path. Deferred.

---

## §10  References

### 10.1 Predecessor specs

- **Spec v3** — `docs/superpowers/specs/2026-06-06-llm-layers-design.v3.md`
  (the pre-rollout design). Sections that this doc supersedes: §3 (axis
  taxonomy — now per-spec field in §3 here), §5 (dataclasses — now §3
  here with rolled-out fields), §8.1 (op floor — now §1 here with rolled-out
  signatures).

### 10.2 Research v3 reports

- `research/01-model-census.v3.md` — the 30-family census, organisational
  index for `models/`.
- `research/02-layer-sources.v3.md §3` — the 38-axis taxonomy. Cross-ref
  with §3 (specs) here for the axis-to-field mapping.
- `research/03-ihv-opsets.v2.md §3` — the 16-op floor synthesis. Cross-ref
  with §1 here.
- `research/04-quantization.v3.md §10` — the quant-axis enumeration.
  Cross-ref with `api/quant.py` modules per scheme.
- `research/05-kvcache-attention.v3.md §7` — attention / RoPE / cache axes.
  Cross-ref with §3.5 (RoPESpec), §3.6 (AttentionSpec), §3.9 (KVCacheSpec)
  here.

### 10.3 Per-family `layer.md` files (one-line summary each)

- `models/deepseek_ocr2/layer.md` — DeepSeek-OCR-2 LM decoder (standard MHA
  + softmax MoE; visual prefix masking deferred).
- `models/deepseek_v2_lite/layer.md` — V2-Lite: MLA (q_lora_rank=None),
  YARN+INTERLEAVED RoPE, softmax MoE with 2 shared experts.
- `models/deepseek_v3_lite/layer.md` — V3-Lite synthetic: MLA, sigmoid+bias
  router with group routing.
- `models/deepseek_v3_moe/layer.md` — V3-MoE production wrapper.
- `models/deepseek_v32/layer.md` — V3.2 DSA (shape-only).
- `models/gemma2/layer.md` — Gemma 2: sandwich norm + 1+w RMSNorm +
  attn_logit_softcap.
- `models/gemma3/layer.md` — Gemma 3: dual θ RoPE + QK-norm PER_HEAD_DH
  PRE_ROPE.
- `models/gemma4/layer.md` — Gemma 4: STANDARD_W RMSNorm + proportional
  partial RoPE + v_norm + PLE + layer_scalar + cross-layer KV.
- `models/got_ocr2/layer.md` — GOT-OCR 2.0: Qwen2-0.5B backbone (MHA + Q/K/V
  biases).
- `models/granite/layer.md` — Granite μP scalar stack.
- `models/granite4_h/layer.md` — Granite-4-H hybrid Mamba/attention with
  shared MLP on every layer.
- `models/llama3/layer.md` — Llama 3: GQA + SwiGLU + optional LLAMA3 RoPE
  scaling.
- `models/llama4_scout/layer.md` — Llama 4 Scout: INTERLEAVED RoPE +
  per-layer NoPE alternation.
- `models/mamba2/layer.md` — Mamba-2: chunked SSD scan, no FFN sublayer.
- `models/minicpm3/layer.md` — MiniCPM-3: MLA + LongRoPE + μP residual scale.
- `models/ministral/layer.md` — Ministral: per-layer SWA alternation.
- `models/mistral/layer.md` — Mistral 7B: vanilla GQA + SwiGLU.
- `models/mixtral/layer.md` — Mixtral 8x7B: softmax MoE, always-on
  top-k renormalisation.
- `models/moshi/layer.md` — Moshi 7B: fused gating MLP + GQA + standard RoPE.
- `models/olmo2/layer.md` — OLMo 2: POST-norm + FULL_HDH QK-norm PRE_ROPE.
- `models/olmoe/layer.md` — OLMoE: FULL_HDH QK-norm + softmax MoE.
- `models/phi3_mini/layer.md` — Phi-3 mini: FUSED QKV + fused gate/up.
- `models/phi3_small/layer.md` — Phi-3-small: FUSED QKV + GE-GELU
  + LayerNorm (shape-only).
- `models/phi4_mini/layer.md` — Phi-4-mini: Phi-3 family + LongRoPE.
- `models/qwen2_5_vl/layer.md` — Qwen2.5-VL: M-RoPE LM decoder.
- `models/qwen3/layer.md` — Qwen3 0.6B/1.7B/4B/8B M1 anchor.
- `models/qwen3_moe/layer.md` — Qwen3-MoE: softmax MoE + per-layer
  dense/MoE dispatch.
- `models/smollm3/layer.md` — SmolLM3: per-layer NoPE alternation.
- `models/tinyllama/layer.md` — TinyLlama: GQA-32:4 Llama backbone (delegates
  to Llama 3 factory).
- `models/voxtral/layer.md` — Voxtral 3B: Llama backbone with
  `rope_theta=1e8`.

### 10.4 Per-batch tags & commit ranges

| Batch | Date | Description | Commit range |
|-------|------|-------------|--------------|
| M1 | 2026-06-05 | Qwen3 IR + 16-op floor closure | `5068bf7..30a4848` |
| B0.5 | 2026-06-06 | Gemma 4 v3 IR seed | `6dbe232..3d84668` |
| B0.6 | 2026-06-06 | Gemma 4 numerical-gate fixes | `18f1bd6..f52efad` |
| B1 | 2026-06-06 | Llama 3 / Mistral / SmolLM3 / TinyLlama | `57057e7..b4c8eff` |
| B2a | 2026-06-07 | Granite / Phi-3 mini / Phi-4-mini | `db02f54..eee7439` |
| B2b | 2026-06-07 | MiniCPM-3 MLA / Phi-3-small shape-only | `453df69..2ac78c1` |
| B3 | 2026-06-07 | OLMo 2 / Gemma 2 / Gemma 3 | `f1dcd46..1b25e9c` |
| B4 | 2026-06-07 | Ministral / Llama 4 Scout iRoPE | `b72e116..d35c507` |
| B5 | 2026-06-07 | DeepSeek-V2-Lite / V3-Lite / V3.2 + MoE module | `5e352cf..fd672f3` |
| B6 | 2026-06-07 | Mixtral / Qwen3-MoE / OLMoE / V3-MoE wrapper | `d0bf902..a50f005` |
| B7 | 2026-06-07 | Mamba-2 + Granite-4-H + deferred SSM stubs | `7171ac5..38f512a` |
| B8 | 2026-06-07/08 | GOT-OCR 2.0 / Qwen2.5-VL / DeepSeek-OCR-2 + M-RoPE/VCF hooks | `4b8f415..ce9e799` |
| B9 | 2026-06-07/08 | Moshi 7B / Voxtral 3B | `2eabcef..9cd250c` |
| B10 | 2026-06-08 | GGUF Q4_K_M / FP8 / MXFP4 / LiteRT W4A8 | `8d00ea1..4010e74` |

---

---

## §10b  In-Depth: The `sdpa` Operation

The `sdpa` op (`api/ops.py:253`) is the most complex single op in the
floor. Its full set of behaviours:

### 10b.1 The GQA broadcast

When `q.shape[1] = Hq > Hk = k.shape[1]`, the op `repeat_interleave`s K and V
to match Q heads:
```python
if Hk != Hq:
    if Hq % Hk != 0:
        raise ValueError(...)
    repeats = Hq // Hk
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
```

Done at op level rather than callsite because `F.scaled_dot_product_attention`
does NOT support GQA natively — every caller would otherwise duplicate this
six-line block. The `repeat_interleave` is the SAME pattern HF uses
(`modeling_llama.py:repeat_kv`).

### 10b.2 The softcap fallback path

When `logit_softcap` is set (Gemma 2 = 50.0; Gemma 3/4 = None), the op
falls back from `F.scaled_dot_product_attention` to a manual eager
attention path because `F.scaled_dot_product_attention` does not expose
pre-softmax post-scale transforms.

The manual path:
```python
if scale is None:
    scale = q.shape[-1] ** -0.5     # default to head_dim**-0.5
attn = torch.matmul(q, k.transpose(-2, -1)) * scale
attn = torch.tanh(attn / logit_softcap) * logit_softcap
if attn_mask is not None:
    attn = attn + attn_mask
elif is_causal:
    # defensive — callers always supply explicit mask
    ...
attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
return torch.matmul(attn, v)
```

The order is critical and matches `modeling_gemma2.py:212-225` exactly:
1. matmul Q×K^T
2. multiply by scale
3. apply softcap (tanh then multiply)
4. add mask
5. softmax (fp32 promoted)
6. matmul × V

The mask is added AFTER the softcap so that `-inf` mask entries remain
`-inf` (tanh saturates to ±cap on bounded inputs only; `-inf` would
saturate to `-cap` which would NOT zero out the softmax output).

### 10b.3 The `is_causal=True` trap

The standard SDPA path uses `is_causal=True` when no mask is supplied.
This works only when `S_q == S_k` and the diagonal lives at the top-left.
For chunked prefill (where `start_pos > 0`), the diagonal lives at
`(start_pos, start_pos+S_q]`, so `is_causal=True` would mask the wrong
positions. The attention block therefore ALWAYS builds an explicit mask
(see §3.6.2). The `is_causal=True` codepath is defensive only.

### 10b.4 dtype promotion behaviour

`F.scaled_dot_product_attention` honours q's dtype throughout. Our softcap
fallback path explicitly promotes the softmax to fp32
(`F.softmax(..., dtype=torch.float32)`) then casts back to q's dtype.
This matches HF's eager attention behaviour. Without the promotion, bf16
softmax drift is visible at atol 5e-4 on Gemma 2 gates.

### 10b.5 Shape contract summary

| Tensor | Standard | MLA |
|--------|----------|-----|
| q | `[B, Hq, S, head_dim]` | `[B, H, S, qk_head_dim]` |
| k | `[B, Hk, T, head_dim]` | `[B, H, T, qk_head_dim]` |
| v | `[B, Hk, T, head_dim]` | `[B, H, T, v_head_dim]` |
| attn_mask | `[1, 1, S, T]` (broadcast over batch and heads) | same |
| output | `[B, Hq, S, head_dim]` (or `v_head_dim` for MLA) | `[B, H, S, v_head_dim]` |

The MLA case has `Hq == Hk == H` (no GQA), but K and V have different
last-dim. The op handles this correctly because the matmul `Q @ K^T`
contracts over `qk_head_dim` and the matmul `attn @ V` contracts over T,
leaving `v_head_dim` as the output's last dim.

---

## §11  Quantization Subsystem (`api/quant.py`)

The quantization subsystem in `api/quant.py` is a separate concern from the
spec / building-block IR: the model factories never construct a `QuantSpec`
directly. Instead, the quantization round-trips are exercised through
weight-loading tests that read a quantized checkpoint, dequantize it
through this module, and verify bit-exact recovery against the upstream
reference.

This section documents each scheme that landed in B10 plus the M1 AWQ path.

### 11.1 Where `QuantSpec` lives in the IR

`QuantSpec` (§3.8) is referenced by `KVCacheSpec.k_quant` and
`KVCacheSpec.v_quant` — both default to `None`, both raise NotImplementedError
in the current `ContiguousKVCache` constructor (`kvcache.py:47-48`). So
the spec field is present but no KV-cache quantization is exercised today.

For WEIGHT-quantization, the IR's `Linear` layers stay fp / bf16 / fp32 —
quantized weights live OUTSIDE the `api/specs.py` envelope. The flow is:
load a quantized checkpoint → dequantize through `api/quant.py` → fill the
`nn.Linear.weight` parameter at fp / bf16. This means the IR sees the
*dequantized* weight, and quantization is verified at the load step rather
than during forward.

### 11.2 AWQ W4A16 grouped INT4 — `api/quant.py:30-156`

**Scheme:** Grouped INT4 weights with asymmetric zero-point and fp16 scales.
- Layout for a weight `W: [K, N]` with group_size `G` along `K`:
  - `qweight: int32[K // 8, N]` — 8 nibbles packed per int32, AWQ
    permutation `[0,2,4,6,1,3,5,7]`.
  - `qzeros: int32[K // G, N // 8]` — 8 zero-nibbles packed per int32.
  - `scales: float16[K // G, N]` — per-group scale.
- **Permutation:** the AWQ nibble permutation `[0,2,4,6,1,3,5,7]` puts the
  evens first then the odds. The `_awq_shifts()` helper returns
  `[0, 8, 16, 24, 4, 12, 20, 28]` (the shift counts as `4*p`).
- **Dequant formula:** `w = (q_int - zeros_int) * scales` per group.

**Functions:**
- `awq_pack(int_values: torch.Tensor) -> torch.Tensor` — pack INT4 [K, N]
  to INT32 [K//8, N].
- `awq_unpack(packed, K, N) -> torch.Tensor` — inverse.
- `awq_quantize(w, spec) -> (qweight, scales, qzeros)` — full quantization
  path: compute per-group min/max, asymmetric zero-point, pack.
- `awq_dequantize(qweight, scales, qzeros, spec, K, N) -> torch.Tensor`.

**Tested against:** real Qwen3-0.6B q_proj AWQ checkpoint
(`tests/models/qwen3/test_quant_awq.py`). Bit-exact recovery in the
qweight bytes; numerical recovery against the dequantized reference.

**Landed at:** M1 commit `3b5c837`.

### 11.3 GGUF Q4_K_M — `api/quant.py:159-398`

**Scheme:** llama.cpp "K"-form 4-bit quantization with super-blocks.
- Super-block of `QK_K = 256` weights.
- 8 sub-blocks of 32 weights each per super-block.
- Per-super-block: 16 bits of `d` (fp16 max-scale) and 16 bits of `dmin`
  (fp16 max-min).
- Per-sub-block: 6-bit `sc` (scale) and 6-bit `m` (min) packed into 12
  bytes total per super-block.
- Per-element: 4-bit `q`.
- Total bytes per super-block: `4 + 12 + 128 = 144`.

**Bit-pack of `scales[12]` (for `j = 0..7`):**
```
if j < 4:
    scales[j]     = sc_j        # lower 6 bits
    scales[j + 4] = m_j         # lower 6 bits
else:
    scales[j + 4]  = (sc_j & 0x0F) | ((m_j & 0x0F) << 4)
    scales[j - 4] |= ((sc_j >> 4) << 6)   # high 2 bits of sc_j
    scales[j]     |= ((m_j  >> 4) << 6)   # high 2 bits of m_j
```

**qs packing:** per 64 weights, 32 bytes:
```
for l in 0..31: qs[l] = q[base + l] | (q[base + l + 32] << 4)
```

**Dequant:** `y = d * sc * q - dmin * m` per sub-block.

**Functions:**
- `_gguf_pack_scales_mins(sc, mn)` — pack 8 (6-bit, 6-bit) pairs to 12
  bytes per block.
- `_gguf_unpack_scales_mins(packed)` — inverse.
- `_gguf_pack_qs(quants)` — pack 256 nibbles into 128 bytes.
- `_gguf_unpack_qs(qs)` — inverse.
- `gguf_q4_k_quantize(w)` — full quantization (uses simple per-sub-block
  min/max search rather than the reference `make_qkx2_quants` weighted
  search; documented error budget ~9-13% on Gaussian rows, vs ~3-5% for
  the reference).
- `gguf_q4_k_dequantize(d, dmin, scales_packed, qs)` — full dequantization;
  bit-exact inverse of `gguf_q4_k_quantize` for any inputs.

**Reference:** `ggml/src/ggml-common.h` (`block_q4_K` struct);
`ggml/src/ggml-quants.c` (`quantize_row_q4_K_ref`, `dequantize_row_q4_K`,
`get_scale_min_k4`).

**Landed at:** B10 commits `8d00ea1`, `9b1d2dd`.

### 11.4 FP8 E4M3 W8A8 — `api/quant.py:401-472`

**Scheme:**
- Weights: per-tensor symmetric scale, cast to `torch.float8_e4m3fn`.
- Activations: per-token symmetric scale (one scale per row of
  `[tokens, hidden]`), cast to `torch.float8_e4m3fn`.
- Compute: fp32 accumulator.

**`torch.float8_e4m3fn`** carries: sign(1) + exp(4, bias=7) + mantissa(3);
dynamic range roughly `[-448, 448]`. IEEE-incompatible "fn" variant — no
infinities, NaN only at `0x7F/0xFF`.

**Functions:**
- `fp8_e4m3_quantize_per_tensor(w)` → `(w_fp8, scale_fp32)`.
- `fp8_e4m3_quantize_per_token(x)` → `(x_fp8, scales_fp32)` with
  `scales.shape == x.shape[:-1] + (1,)`.
- `fp8_e4m3_dequantize(t_fp8, scale)` → fp32 tensor.
- `fp8_e4m3_matmul(x_fp8, x_scale, w_fp8, w_scale)` → fp32 matmul with
  per-tensor `w_scale` and per-token `x_scale`.

**Where used:** the dominant H100/B200 inference format (vLLM,
TensorRT-LLM, Triton fused kernels). We expose a tensor-level round-trip
without depending on a CUDA backend.

**Landed at:** B10 commit `10a0fe7`.

### 11.5 MXFP4 — `api/quant.py:475-590` (approx)

**Scheme:** OCP "Microscaling (MX) Formats" v1.0 §5.4 + §5.5; NVIDIA
Blackwell PTX guide §14.7.
- **Block size:** 32 elements.
- **Shared scale:** UE8M0 (8-bit unsigned exponent, no mantissa, no sign,
  bias=127) — a pure power-of-two.
- **Element format:** E2M1 (1 sign + 2 exp + 1 mantissa, bias=1).

**E2M1 representable magnitudes (positive half):**
`0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0`. Max = 6.0.

**Dequant:** `y = 2**(ue8m0_exp - 127) * e2m1_value`.

**Quant:** choose the shared exponent so `abs(max_in_block) / 2**(e-127)`
fits within the E2M1 max (6.0); round each element to the nearest E2M1
code.

**Landed at:** B10 commit `551ddf3`.

### 11.6 LiteRT W4A8 — added at B10 commit `e6449f6`

**Scheme:** Gemma 4 mobile QAT shape on AI Edge / LiteRT (axes A20-A24
from `research/04-quantization.v3.md`).
- Per-output-channel symmetric INT4 weights (range `[-7, 7]`).
- Per-tensor symmetric INT8 activations.
- No zero-point on either side.

The lack of zero-points means the dequant is `scale * q_int` per channel
(weight) and per tensor (activation), with fp32 matmul accumulation.

### 11.7 IQ2_M / AQLM stubs — `api/quant.py` and `codebook.py` (reserved for M3)

**Scheme:** codebook-based quantization. The K-means-like codebook stores
N centroids; each element is replaced by its codebook index (log2(N) bits).
- IQ2_M: 2-bit-per-weight via 16-entry codebook plus auxiliary scales.
- AQLM: additive quantization with multiple codebooks.

**Status:** the `QuantSpec.codebook` field is reserved for these schemes.
Stubs landed at B10 commit `19b9dec`.

### 11.8 The quantization round-trip test pattern

All quant tests follow the same shape:

```python
# 1. Build a synthetic weight tensor (or load a real checkpoint slice).
w = ... # e.g. fp32 [K, N] Gaussian, or Qwen3-0.6B q_proj.weight

# 2. Build a QuantSpec.
spec = QuantSpec(qdtype=INT4, group_size=128, quant_axis=0,
                 scale_dtype=torch.float16, has_zero_point=True,
                 packing=PackingLayout.AWQ_INTERLEAVE,
                 accumulator_dtype=torch.float32)

# 3. Quantize.
qweight, scales, qzeros = awq_quantize(w, spec)

# 4. Dequantize.
w_recon = awq_dequantize(qweight, scales, qzeros, spec, K, N)

# 5. Assert bit-exact or close-to-bit-exact recovery.
assert (w - w_recon).abs().max() < tol
```

For real checkpoints, step 4 also asserts the BYTE-level qweight matches
the upstream packing (i.e. `awq_pack(awq_unpack(qweight_from_disk)) ==
qweight_from_disk` — verifies the packing-layout interpretation).

---

## §11b RoPE: The Six-Variant Decision Tree

The `RoPE` module supports six distinct configurations. The decision tree
that selects each:

```
spec.basis = SPLIT_HALF or INTERLEAVED?

  SPLIT_HALF (almost everyone):
    spec.partial_rotary_kind = "prefix" or "proportional"?

      "prefix" (Phi-3, Phi-4, MLA):
        spec.partial_rotary_factor < 1.0?
          yes:  apply via rope_apply_partial (B0.5 op)
          no:   apply via rope_apply (default path)
        spec.scaling = NONE, LLAMA3, LONGROPE, or YARN?
          NONE / LLAMA3: build single inv_freq table
          LONGROPE:      build TWO inv_freq tables, dispatch per forward
          YARN:          apply linear-ramp blend + attention_factor

      "proportional" (Gemma 4):
        spec.partial_rotary_factor < 1.0?
          yes:  zero-pad inv_freq to full head_dim/2; apply standard rope_apply
          no:   identical to prefix (no rotation occurs in nope channels)
        scaling restricted to NONE or LLAMA3

  INTERLEAVED (Llama 4, DeepSeek-V2 complex-mul):
    forced spec.partial_rotary_factor = 1.0
    cos/sin shape: [max_seq, head_dim/2]
    apply via rope_apply(basis="interleaved") — complex multiply
    scaling: NONE, LLAMA3, or YARN

M-RoPE branch (Qwen2.5-VL):
  spec.mrope_section set + position_ids has rank 3
  cos/sin REBUILT per forward (cannot precompute)
  apply via rope_apply_mrope
```

Edge cases the constructor REJECTS:
- INTERLEAVED + LongRoPE (the per-forward boundary dispatch + complex
  geometry don't combine).
- INTERLEAVED + partial_rotary_factor != 1.0 (no model uses this).
- M-RoPE + partial_rotary_factor < 1 (no model uses this).
- YARN + proportional partial (no model uses this).

### The proportional vs prefix geometric difference

This is the subtlest IR distinction. Both compute "RoPE on a subset of
channels with the rest passing through." But:

**Prefix** (Phi-3 style — `rope.py:178-179`):
- `inv_freq` denominator = `dim_rot` (the rotated subset only).
- inv_freq shape: `[rope_angles]`.
- cos/sin shape: `[max_seq, head_dim_rot]`.
- Apply via `rope_apply_partial`: slice `q[..., :Dh_rot]`, rotate, slice
  `q[..., Dh_rot:]`, pass through, concat.

**Proportional** (Gemma 4 style — `rope.py:180-181`):
- `inv_freq` denominator = full `head_dim`.
- inv_freq shape: `[head_dim/2]` (with zeros in trailing
  `head_dim/2 - rope_angles` slots).
- cos/sin shape: `[max_seq, head_dim]`.
- Apply via standard `rope_apply` (basis="split_half"): the full `rotate_half`
  pairs `i ↔ i + Dh/2`, but the trailing zero-inv_freq channels have
  cos=1, sin=0 so they pass through. They STILL participate in the
  `rotate_half` pairing (the pairing isn't `i ↔ i+Dh_rot/2`; it's
  `i ↔ i+Dh/2`).

The difference is observable in the channel-mixing: prefix's rotation
pairs `(i, i+Dh_rot/2)` channels within the rotated prefix; proportional's
pairs `(i, i+Dh/2)` channels — including some rotated with some
passing-through. They produce DIFFERENT outputs for the same q tensor.

Phi-3 prefix matches `modeling_phi3.py:199-204`; Gemma 4 proportional
matches `modeling_gemma4.py:787-806`. The IR distinguishes them at
`partial_rotary_kind`.

### The YARN inv_freq construction in detail

`_yarn_inv_freq_and_scale` (`rope.py:17-78`) implements
`_compute_yarn_parameters` from `modeling_rope_utils.py:327-459`:

```python
factor = extra.factor
original = extra.original_max_position_embeddings
beta_fast = extra.beta_fast      # 32 (extrapolation boundary)
beta_slow = extra.beta_slow      # 1  (interpolation boundary)

def get_mscale(scale, mscale=1.0):
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * log(scale) + 1.0

# attention_factor: mscale / mscale_all_dim ratio
if extra.mscale > 0 and extra.mscale_all_dim > 0:
    attention_factor = (
        get_mscale(factor, extra.mscale)
        / get_mscale(factor, extra.mscale_all_dim)
    )
else:
    attention_factor = get_mscale(factor)

# Two inv_freq tables:
pos_freqs = base ** (arange(0, dim_rot, 2) / dim_rot)
inv_freq_extrapolation = 1.0 / pos_freqs
inv_freq_interpolation = 1.0 / (factor * pos_freqs)

# find_correction_range identifies the channel band where the ramp lives:
low, high = find_correction_range(
    beta_fast, beta_slow, dim_rot, base, original, truncate)

# linear_ramp_factor: 0 outside [low, high], 1 above high, ramp between
ramp = linear_ramp_factor(low, high, dim_rot // 2)

# Blend extrapolation and interpolation per channel:
inv_freq_extrapolation_factor = 1.0 - ramp
inv_freq = (
    inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
    + inv_freq_extrapolation * inv_freq_extrapolation_factor
)
return inv_freq, attention_factor
```

For V2-Lite production values (factor=40, beta_fast=32, beta_slow=1,
mscale=0.707, mscale_all_dim=0.707), the `get_mscale(40, 0.707) /
get_mscale(40, 0.707) = 1.0`, so attention_factor = 1.0 — i.e. the YARN
m-scale cancels out in this canonical setup. The IR still goes through the
formula because OTHER V2 / V3 configurations have differing
mscale / mscale_all_dim.

---

## §12  The Numerical-Gate Pattern

Across all 41 families, the validation pattern that gates each batch is
the same: build the IR layer for layer 0 of a family, load its weights
from a real HF checkpoint, run a forward pass on a synthetic input, and
compare against the same input run through the HF reference impl. The
target tolerance is **atol=5e-4** for fp32, looser for bf16.

The test files live in `tests/models/<family>/`. Each family has:

- `test_isolation.py` — sub-op-level tests vs HF (RMSNorm, RoPE, QKNorm,
  FFN, etc.) at atol=1e-5 or 1e-4.
- `test_numerical_gate.py` — full layer-0 forward at atol=5e-4.
- `test_kv_cache_gate.py` — prefill+decode equivalence at atol=5e-4.

The gate-test pattern is what surfaces drift: B0.6 added five fixes
because the B0.5 Gemma 4 layer's numerical gate failed and the fix-pass
read `modeling_gemma4.py` end-to-end to find five distinct behaviours the
spec hadn't anticipated.

---

## §12b The B7 SSM design — what's the same, what's different

The B7 SSM block (Mamba-2) differs from the standard attention block in
several structural ways. This subsection summarises the asymmetries so
that a reader of the IR can predict the behaviour without re-reading
`ssm.py`.

### What's the same

- **Block envelope is `DecoderBlock`.** The same class hosts attention
  and SSM blocks. Dispatch is by `isinstance(spec.token_mixer, SSDSpec)`
  (`block.py:70`). The attribute name `blk.attention` holds the
  `Mamba2Mixer` for SSM blocks — kept for backward compat with all
  existing weight loaders (`block.py:118-119`).
- **Pre-norm before the token mixer.** Same as standard. The `pre_attn_norm`
  is consumed regardless of mixer type. (No POST or PRE_AND_POST mode for
  SSM in B7.)
- **Optional FFN sublayer.** The `skip_ffn` flag (default False) on
  `DecoderBlockSpec` controls whether the channel mixer runs. Canonical
  Mamba-2 sets `skip_ffn=True`; Granite-4-H sets `skip_ffn=False` so the
  shared MLP runs on EVERY layer (mamba and attention alike).
- **`residual_scale`** still applies. Granite-4-H multiplies the SSM
  output by `residual_multiplier` before the residual add, identically
  to attention layers.

### What's different

- **The cache type.** `SSMStateCache` holds `conv_state: [B, conv_dim,
  conv_kernel]` and `ssm_state: [B, n_heads, head_dim, d_state]`. Distinct
  from `ContiguousKVCache`. There is no `start_pos` / `seq_len` semantic;
  instead there is a `has_previous_state` flag for prefill vs decode
  dispatch.
- **No `position_ids`.** Mamba-2 is intrinsically position-aware via the
  recurrent state and does not consume external position information.
  The `DecoderBlock.forward` passes `position_ids` only when the mixer is
  an `Attention`.
- **No `vision_token_count`.** SSM blocks do not consume the B8 hook.
- **`token_mixer.forward(hidden_states, cache=...)`** — only two args.
  Attention's `forward(x, position_ids, cache, start_pos, vision_token_count)`
  takes five.
- **The mixer has its OWN final norm.** The `_MambaRMSNormGated` inside
  `Mamba2Mixer` (`ssm.py:64-90`) is a gated RMS norm with `silu(gate)`
  inside. It is INSIDE the mixer, not at block level. The block's
  `pre_attn_norm` (or sandwich norm) still runs on `hidden_states` before
  the mixer; the gated RMSNorm runs after the SSD scan on the output `y`.
- **No KV.** No K, no V, no cache.write/read by `(key, value)`. The
  recurrent state IS the analog of KV, but it's a small fixed-size
  tensor regardless of seq_len.

### What's NOT exercised in B7

- **Decode** (single-step recurrent advance). `Mamba2Mixer.forward` raises
  NotImplementedError when `cache.has_previous_state == True`
  (`ssm.py:222-229`). Prefill-only gate.
- **Mamba-1** (SSMSpec without SSDSpec). The `SSMSpec`-only path is
  reserved for Mamba-1, RecurrentGemma, etc. — landed for typing only.
- **`d_mlp > 0`**. The `Mamba2Mixer` hardcodes `self.d_mlp = 0` (`ssm.py:141`).
  Some hybrid families have a small extra MLP head off the same in_proj;
  these would need `d_mlp > 0` and an extra split.

## §13  Notable Per-Family Subtleties (cross-cutting)

These behaviours show up in multiple families but are tied to specific
axes. Useful when reading the coverage matrix in §5.

### 13.1 `attn_scale` overrides — six families

Six families override the default `1/sqrt(head_dim)`:

- **Gemma 2 / 3** — `attn_scale = query_pre_attn_scalar**-0.5`. For 2B/9B
  this often differs from `head_dim**-0.5` (the `query_pre_attn_scalar`
  field is independent of `head_dim`).
- **Gemma 4** — `attn_scale = 1.0` explicitly. The IR previously baked
  `1/sqrt(Dh)` into the QK norm weight via `qk_norm_fixed_scale`; B0.6
  dropped the absorb and made `attn_scale=1.0` explicit. The HF model has
  `self.scaling = 1.0` directly (`modeling_gemma4.py:1195`).
- **Granite, Granite-4-H** — `attn_scale = attention_multiplier` (μP per-layer
  scale instead of `1/sqrt(Dh)`). For Granite 3.1-2B this is typically a
  fixed scalar that doesn't match `head_dim**-0.5`.
- **Phi-3-small** — `attn_scale = mup_attn_multiplier / head_dim` (NOT
  `1/sqrt(head_dim)`) when `mup_use_scaling` is set. Source:
  `modeling_phi3_small.py:200-206`.

For everyone else `attn_scale=None` and the default `head_dim ** -0.5` is
applied inside `Attention.__init__` (`attention.py:151`).

### 13.2 Cross-family μP scalar stack

Three families plumb a μP scalar stack through `DecoderBlockSpec`:

| Family | `embedding_scale` | `logits_scale` | `residual_scale` |
|--------|-------------------|----------------|------------------|
| Granite (3.1) | `embedding_multiplier` | `logits_scaling` | `residual_multiplier` |
| Granite-4-H | `embedding_multiplier` | `logits_scaling` | `residual_multiplier` |
| MiniCPM-3 | `scale_emb` (~12 for 4B) | `dim_model_base / hidden_size` | `scale_depth / sqrt(L)` (≈0.1778 for 4B) |

Gemma 2/3/4 also set `embedding_scale=sqrt(hidden_size)` but not the other
two scalars.

### 13.3 Cross-family QK-norm shape distribution

- `PER_HEAD_DH` (weight shape `[head_dim]`): Qwen3, Qwen3-MoE, Gemma 3,
  Gemma 4.
- `FULL_HDH` (weight shape `[n_heads * head_dim]`): OLMo 2, OLMoE.
- `NONE`: everyone else.

The PER_HEAD_DH path is independent per head (one shared weight broadcast
across heads). The FULL_HDH path applies the norm to the flattened
`[B, S, H*Dh]` tensor — algebraically equivalent to reshape-first
norm-last when the weight is also reshaped, but the WEIGHT shape differs.

### 13.4 Cross-family RoPE scaling distribution

- `NONE`: Qwen3, Mistral, OLMo 2, Granite, Mamba-2 (no RoPE), Gemma 2/3,
  Mixtral, OLMoE, V3 wrappers, ...
- `LLAMA3`: Llama 3.1, Llama 3.2, Llama 4 Scout RoPE layers.
- `LONGROPE`: Phi-3 mini (when long-context variant), Phi-4-mini,
  MiniCPM-3.
- `YARN`: DeepSeek-V2-Lite.

### 13.5 Cross-family RoPE basis distribution

- `SPLIT_HALF`: almost everyone.
- `INTERLEAVED`: Llama 4 Scout, DeepSeek-V2-Lite (complex-multiply path).

### 13.6 Per-layer dispatch families

Families where `to_block_spec(layer_idx)` returns a DIFFERENT spec for
different layer indices:

| Family | What varies per layer |
|--------|------------------------|
| Gemma 2 | mask_kind (SWA every other layer) |
| Gemma 3 | mask_kind + rope_theta (5:1 SWA/full pattern) |
| Gemma 4 | mask_kind + head_dim + rope_theta + partial_rotary + attention_k_eq_v + sliding_window (4:1 or 5:1 SWA/full) |
| Ministral | mask_kind (per `layer_types` config) |
| SmolLM3 | rope (None on NoPE layers) + mask_kind |
| Llama 4 Scout | rope (None on NoPE layers) + channel_mixer (FFN vs MoE per `moe_layers`) |
| Qwen3-MoE | channel_mixer (FFN vs MoE per `mlp_only_layers` / `decoder_sparse_step`) |
| Granite-4-H | token_mixer (Attention vs SSDSpec per `layer_types`) |
| DeepSeek-V2-Lite | channel_mixer (dense FFN for layer 0, MoE for 1..26) |
| DeepSeek-V3-Lite | channel_mixer (dense FFN for layers 0..first_k_dense-1, MoE for rest) |
| DeepSeek-V3-MoE | same as V3-Lite |
| DeepSeek-V3.2 | same as V3 plus AttentionKind.DSA throughout |
| DeepSeek-OCR-2 | channel_mixer per `mlp_layer_types[L]` |
| Phi-3-small | mask_kind (BLOCK_SPARSE vs CAUSAL per dense_attention_every_n_layers) |

Families with uniform spec across all layers: Qwen3, TinyLlama, Llama 3,
Mistral, MiniCPM-3, Mixtral, OLMo 2, OLMoE, Mamba-2 (single SSD spec),
Granite, GOT-OCR-2.0, Voxtral, Moshi, Qwen2.5-VL.

### 13.7 Families with non-CAUSAL masks

- `SWA`: Gemma 2 (every other layer), Gemma 3 (5:1), Gemma 4 (4:1 or
  5:1 SWA/full), Ministral, Mistral v0.2 (off in v0.3), Phi-3 (optional),
  Qwen3-MoE (when `sliding_window` set), SmolLM3 (when `use_sliding_window`).
- `BLOCK_SPARSE`: Phi-3-small (shape-only).
- `BLOCK_BIDIRECTIONAL`: declared by DeepSeek-OCR-2 spec hook only;
  the production OCR-2 config uses CAUSAL.

### 13.8 Families with biased projections

Default everywhere is no bias. The exceptions:

- **Qwen2 family** (Qwen2.5-VL, GOT-OCR 2.0): `q_bias = k_bias = v_bias = True;
  o_bias = False`. Source: `modeling_qwen2.py:200-203`.
- **MLA families with `attention_bias=True`** (configured per-family):
  q_a_proj, kv_a_proj_with_mqa, o_proj all take the bias. q_b_proj and
  kv_b_proj never have bias.
- **Phi-3-small** with `attention_bias=True` (defaults False).

### 13.8b Families with biases on MLP projections

Default: no biases anywhere. The exceptions:

- **Phi-3-small** with `attention_bias=True` is the only family that
  enables biases on gate / up / down MLP projections (HF
  Phi3SmallMLP has biased projections in some configs).
- **Llama 3.1 / 3.2** has an `mlp_bias` flag (default False); the
  config can override.

### 13.8c Per-family `intermediate_size` ratios

Selected `intermediate_size / hidden_size` ratios:

- Qwen3 0.6B: hidden=1024, I=3072 (3.0×).
- Qwen3 1.7B: hidden=2048, I=6144 (3.0×).
- Llama 3 8B: hidden=4096, I=14336 (3.5×).
- Llama 3.2 1B: hidden=2048, I=8192 (4.0×).
- Mistral 7B: hidden=4096, I=14336 (3.5×).
- Phi-3 mini 4k: hidden=3072, I=8192 (2.67× — fused gate/up means the
  flat projection is `2*I = 16384`).
- Gemma 4 E2B: hidden=1536, I=6144 (4.0×).
- Mamba-2 2.7B: hidden=2560, `d_inner = hidden * expand = 5120`.
- DeepSeek-V2-Lite dense: hidden=2048, I=10944 (5.34× — the dense layer 0).
- DeepSeek-V2-Lite MoE: hidden=2048, moe_intermediate_size=1408 (0.69×).
  But shared experts run with `I * n_shared = 1408 * 2 = 2816`.
- MiniCPM-3 4B: hidden=2560, I=6912 (2.7×).
- Granite 3.1-2B: hidden=2048, I=8192 (4.0×).

### 13.9 The `tie_word_embeddings` distribution

- True: Qwen3 0.6B/1.7B/4B (NOT 8B), Llama 3.2 1B/3B (NOT 8B), Gemma 4
  E2B/E4B/12B, TinyLlama, SmolLM3, GOT-OCR 2.0, Qwen2.5-VL.
- False: everyone else.

The IR does not enforce or consume this — `tie_word_embeddings` is a
model-level concern handled in the weight loader and final lm_head
construction.

---

## §14  Frequently-Encountered IR Errors (and what they mean)

This section catalogs the explicit raises in `api/*.py` that you might hit
when authoring a new family. Each `raise` is keyed to a constraint the
IR enforces; the table below maps message → diagnosis.

| Error message | File:line | Diagnosis |
|---|---|---|
| `M1 supports CONTIGUOUS layout only` | kvcache.py:44 | Setting `KVCacheSpec.layout` to anything other than CONTIGUOUS. PAGED / RING / MLA_LATENT / SSM_STATE are reserved. |
| `M1 supports HND memory layout only` | kvcache.py:46 | Setting `KVCacheSpec.memory_layout` to NHD. |
| `M1 supports unquantized KV cache only` | kvcache.py:48 | Setting `KVCacheSpec.k_quant` or `v_quant` to a non-None QuantSpec. |
| `B5: STANDARD, MLA, and DSA only, got <kind>` | attention.py:33 | Using `AttentionKind.LINEAR_*` or DIFFERENTIAL. |
| `DSA kind requires AttentionSpec.indexer` | attention.py:40 | Setting `kind=DSA` without setting `indexer`. |
| `DSA kind requires QKVLayout.MLA_LATENT` | attention.py:43 | DSA composes on top of MLA. |
| `MLA kind requires QKVLayout.MLA_LATENT` | attention.py:65 | MLA needs the latent layout. |
| `MLA spec missing <field>` | attention.py:77 | When `kind=MLA`, all five MLA fields must be set: kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim. (q_lora_rank is allowed to be None.) |
| `MLA: spec.head_dim must equal qk_nope_head_dim + qk_rope_head_dim` | attention.py:79 | Invariant on the QK assembly dim. |
| `MLA + attention_k_eq_v unsupported` | attention.py:85 | These two are mutually exclusive. |
| `MLA + v_norm unsupported` | attention.py:92 | Gemma 4's v_norm doesn't apply to MLA. |
| `B2a: SPLIT and FUSED QKV only, got <layout>` | attention.py:96 | The fallthrough from MLA_LATENT goes through this check. |
| `B8: CAUSAL, SWA, and BLOCK_BIDIRECTIONAL masks only` | attention.py:103 | BLOCK_SPARSE is not exercised in the forward (Phi-3-small is shape-only). |
| `FUSED QKV with attention_k_eq_v is not supported` | attention.py:120 | FUSED needs identical bias config for q/k/v; the alias is incompatible. |
| `FUSED QKV requires q/k/v_bias to all match` | attention.py:124 | All three must be equal for the fused projection. |
| `M1: PRE_ROPE QK-norm only` | attention.py:155 | Setting `qk_norm_phase=POST_ROPE`. |
| `v_norm: RMS kind only` | attention.py:178 | LAYER kind for v_norm is reject. |
| `INTERLEAVED basis with partial_rotary_factor != 1.0 not supported` | rope.py:142 | Llama 4 doesn't compose with partial RoPE. |
| `INTERLEAVED basis supports NONE / LLAMA3 / YARN scaling only` | rope.py:150 | LongRoPE + INTERLEAVED not supported (the boundary dispatch + complex-multiply geometry don't combine). |
| `YARN + proportional partial-rotary not supported` | rope.py:272 | YARN composes only with prefix-mode partial RoPE. |
| `partial_rotary_kind must be 'prefix' or 'proportional'` | rope.py:155 | Typos in `partial_rotary_kind`. |
| `M-RoPE + partial_rotary_factor < 1 not supported` | rope.py:342 | Qwen2.5-VL uses full rotation. |
| `M-RoPE only supports RoPEScaling.NONE` | rope.py:346 | Qwen2.5-VL has no scaling on the LM-decoder RoPE. |
| `M-RoPE expects 3 axes (T/H/W)` | rope.py:330 | The 3D `position_ids` must have leading dim 3. |
| `B0.5: SWIGLU and GEGLU only` | feedforward.py:41 | GELU_ONLY / RELU2_ONLY are reserved. |
| `B0.5: SWIGLU requires SILU activation` | feedforward.py:44 | Mismatch between gate_kind and activation. |
| `B0.5: GEGLU requires GELU activation` | feedforward.py:48 | Same. |
| `fused_gate_up requires gate_bias == up_bias` | feedforward.py:60 | The fused projection has a single bias vector. |
| `B5: MoE experts require SwiGLU + SILU` | feedforward.py:124 | DeepSeek family expert FFN shape. |
| `B5: MoE experts must have no biases` | feedforward.py:134 | Experts are bias-free in V2/V3. |
| `n_experts must divide n_groups` | feedforward.py:139 | Group routing invariant. |
| `B3: PRE, PRE_AND_POST, and POST attn-norm only` | block.py:53 | The four enum values reduce to three usable ones. |
| `PRE_AND_POST attn norm requires pre and post norm specs` | block.py:95 | Sandwich needs both. |
| `POST attn-norm requires post_attn_norm` | block.py:101 | OLMo 2 needs post_*_norm set. |
| `POST attn-norm must not also set pre_attn_norm` | block.py:104 | Enforce single-norm-per-sublayer. |
| `B7: token_mixer must be AttentionSpec or SSDSpec` | block.py:68 | SSMSpec (Mamba-1) reserved. |
| `Mamba-2: num_heads*head_dim != d_inner` | ssm.py:118 | SSD shape invariant. |
| `Mamba-2: num_heads must be divisible by n_groups` | ssm.py:124 | `repeat_interleave` factor must be integer. |
| `Mamba-2 decode is deferred` | ssm.py:226 | B7 lands prefill only. |

Each error message points at the constraint and the file where the
constraint lives — the IR doesn't silently degrade; it raises explicitly.

---

## §14b The HF state-dict → API attribute name mapping cheat sheet

The following mappings hold across all families that use the standard
block envelope. Family-specific deviations are noted parenthetically.

### Standard attention (most families)

| HF key suffix | API attribute |
|---|---|
| `self_attn.q_proj.weight` | `blk.attention.q_proj.weight` |
| `self_attn.k_proj.weight` | `blk.attention.k_proj.weight` |
| `self_attn.v_proj.weight` | `blk.attention.v_proj.weight` |
| `self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |
| `self_attn.q_proj.bias` | `blk.attention.q_proj.bias` (when `q_bias=True`) |
| `self_attn.q_norm.weight` | `blk.attention.q_norm.weight` |
| `self_attn.k_norm.weight` | `blk.attention.k_norm.weight` |
| `self_attn.v_norm.weight` | (Gemma 4 only — when `v_norm_with_scale=True`; for Gemma 4 the v_norm has no scale so this key doesn't exist) |

### FUSED QKV (Phi-3, Phi-4-mini, Moshi-style fused FFN)

| HF key suffix | API attribute |
|---|---|
| `self_attn.qkv_proj.weight` | `blk.attention.qkv_proj.weight` |
| `mlp.gate_up_proj.weight` | `blk.feedforward.gate_up_proj.weight` |

### MLA (MiniCPM-3, DeepSeek)

| HF key suffix | API attribute |
|---|---|
| `self_attn.q_a_proj.weight` | `blk.attention.q_a_proj.weight` (when `q_lora_rank` set) |
| `self_attn.q_a_layernorm.weight` | `blk.attention.q_a_layernorm.weight` |
| `self_attn.q_b_proj.weight` | `blk.attention.q_b_proj.weight` |
| `self_attn.q_proj.weight` | `blk.attention.q_proj.weight` (V2-Lite only — `q_lora_rank=None`) |
| `self_attn.kv_a_proj_with_mqa.weight` | `blk.attention.kv_a_proj_with_mqa.weight` |
| `self_attn.kv_a_layernorm.weight` | `blk.attention.kv_a_layernorm.weight` |
| `self_attn.kv_b_proj.weight` | `blk.attention.kv_b_proj.weight` |
| `self_attn.o_proj.weight` | `blk.attention.o_proj.weight` |

### Sandwich norm (Gemma 2/3/4)

| HF key suffix | API attribute |
|---|---|
| `input_layernorm.weight` | `blk.pre_attn_norm.weight` |
| `post_attention_layernorm.weight` | `blk.post_attn_sublayer_norm.weight` |
| `pre_feedforward_layernorm.weight` | `blk.pre_ffn_norm.weight` |
| `post_feedforward_layernorm.weight` | `blk.post_ffn_sublayer_norm.weight` |

### POST-norm (OLMo 2)

| HF key suffix | API attribute |
|---|---|
| `post_attention_layernorm.weight` | `blk.post_attn_sublayer_norm.weight` |
| `post_feedforward_layernorm.weight` | `blk.post_ffn_sublayer_norm.weight` |

(No `pre_*` keys exist in OLMo 2 state dicts.)

### Dense FFN (SwiGLU / GeGLU split)

| HF key suffix | API attribute |
|---|---|
| `mlp.gate_proj.weight` | `blk.feedforward.gate_proj.weight` |
| `mlp.up_proj.weight` | `blk.feedforward.up_proj.weight` |
| `mlp.down_proj.weight` | `blk.feedforward.down_proj.weight` |

### MoE (DeepSeek V2 softmax)

| HF key suffix | API attribute |
|---|---|
| `mlp.gate.weight` | `blk.feedforward.gate.weight` |
| `mlp.experts.<e>.gate_proj.weight` | stacked into `blk.feedforward.experts_gate_up[e, :I]` |
| `mlp.experts.<e>.up_proj.weight` | stacked into `blk.feedforward.experts_gate_up[e, I:]` |
| `mlp.experts.<e>.down_proj.weight` | stacked into `blk.feedforward.experts_down[e]` |
| `mlp.shared_experts.gate_proj.weight` | `blk.feedforward.shared_experts.gate_proj.weight` |
| `mlp.shared_experts.up_proj.weight` | `blk.feedforward.shared_experts.up_proj.weight` |
| `mlp.shared_experts.down_proj.weight` | `blk.feedforward.shared_experts.down_proj.weight` |

### MoE (DeepSeek V3 sigmoid+bias)

Same as V2 plus:
| HF key suffix | API attribute |
|---|---|
| `mlp.gate.e_score_correction_bias` | `blk.feedforward.gate.e_score_correction_bias` (buffer) |

### Mamba-2 SSM (canonical Mamba-2; Granite-4-H ssm layers similar)

| HF key suffix | API attribute |
|---|---|
| `mixer.in_proj.weight` | `blk.attention.in_proj.weight` (`blk.attention` per B7 convention) |
| `mixer.conv1d.weight` | `blk.attention.conv1d.weight` |
| `mixer.conv1d.bias` | `blk.attention.conv1d.bias` |
| `mixer.A_log` | `blk.attention.A_log` |
| `mixer.D` | `blk.attention.D` |
| `mixer.dt_bias` | `blk.attention.dt_bias` |
| `mixer.norm.weight` | `blk.attention.norm.weight` |
| `mixer.out_proj.weight` | `blk.attention.out_proj.weight` |

### Per-layer scalars & buffers

| HF key suffix | API attribute |
|---|---|
| `layer_scalar` | `blk.layer_scalar` (Gemma 4 only — shape `[1]`) |
| `per_layer_input_gate.weight` | `blk.per_layer_input_gate.weight` (Gemma 4 E2B/E4B) |
| `per_layer_projection.weight` | `blk.per_layer_projection.weight` (Gemma 4 E2B/E4B) |
| `post_per_layer_input_norm.weight` | `blk.post_per_layer_input_norm.weight` |

## §15  The Decision Tree for "How Do I Add a New Family?"

The pattern that worked 30 times in a row:

1. **Read `modeling_<family>.py` end-to-end** — the HF reference for the
   family. Note every divergence from a family already in `models/`.
2. **Identify the closest existing family** in `models/`. (Llama-shaped?
   Use llama3 as template. GQA + QK-norm? Use qwen3. MLA? Use minicpm3.
   MoE? Use deepseek_v2_lite or mixtral. Sandwich? Use gemma3 or gemma4.)
3. **Author `models/<family>/config.py`** with a `from_hf_dict` classmethod
   and a `to_block_spec(layer_idx)` method. The config dataclass is
   `@dataclass(frozen=True)` and carries only what's needed to build the
   spec (NOT every field from HF config.json).
4. **Author `models/<family>/layer.py`** with `build_<family>_decoder_layer`
   factory (calls into `api.block.DecoderBlock`) and `load_<family>_layer`
   weight loader (translates HF state-dict keys to API attribute names).
5. **Author `tests/models/<family>/test_isolation.py`** — sub-op
   equivalence vs HF at atol=1e-5 to 1e-4.
6. **Author `tests/models/<family>/test_numerical_gate.py`** — full
   layer-0 forward vs HF at atol=5e-4.
7. **Author `tests/models/<family>/test_kv_cache_gate.py`** — prefill +
   decode equivalence at atol=5e-4.
8. **Author `models/<family>/layer.md`** per `models/qwen3/layer.md`'s
   8-section schema.

If steps 5-7 fail, the most common root causes are:

- **Norm mode wrong** (1+w vs standard_w; see §8.3 B0.6).
- **Attention scale wrong** (`head_dim**-0.5` vs explicit; see §13.1).
- **RoPE basis wrong** (SPLIT_HALF vs INTERLEAVED; see §13.5).
- **Partial RoPE semantic wrong** (prefix vs proportional; see §8.5 B2a).
- **QK-norm shape wrong** (PER_HEAD_DH vs FULL_HDH; see §13.3).
- **K=V alias not detected** (Gemma 4 12B+).
- **v_norm not detected** (Gemma 4 — see §8.3 B0.6).
- **Fused gate/up direction wrong** (Phi-3 chunk(2, dim=-1)).

The drift log in §8 is a useful pattern: every entry there is a case where
a fresh family forced the IR to grow. If you're hitting an axis the IR
doesn't express, you're likely on the path to a new drift-log entry.

---

## §16  Discovered Behaviours Not Documented in Spec v3

Working through B0.5 → B10 surfaced behaviours that spec v3 did not name
explicitly. Beyond the drift log in §8 (which catalogs SPEC extensions),
these are HF reference behaviours that the IR had to MATCH but spec v3
didn't anticipate. They live in the IR today but were "discovered" rather
than designed.

### 16.1 Gemma 2 fp32 softmax inside softcap

`modeling_gemma2.py:215` does:
```python
attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
```
The fp32 promotion is INSIDE the softcap eager path. The IR's `ops.sdpa`
softcap fallback honours this (`ops.py:305`). Without the promotion, bf16
softmax drift makes Gemma 2 attention fail at atol 5e-4.

### 16.2 Gemma 4 `layer_scalar` on EVERY layer (even no-PLE)

`modeling_gemma4.py:1382` registers `layer_scalar` as a buffer on every
decoder layer, not just E2B/E4B with PLE. The IR mirrors this — every
`DecoderBlock` allocates `self.layer_scalar = torch.ones(1)` regardless
of `spec.per_layer_embedding` (`block.py:194-196`). The trailing
`x = x * layer_scalar` runs on every forward.

### 16.3 The V3 router weights come from the BIAS-FREE probs

The non-obvious detail at `modeling_deepseek_v3.py:232`: the router CHOICE
is made on `probs_for_choice = sigmoid(logits) + bias`, but the WEIGHTS
used to combine experts are gathered from the bias-free
`router_probs = sigmoid(logits)`. The IR mirrors this exactly at
`feedforward.py:287`. Treating the bias-augmented probs as weights yields
slightly higher expert combination magnitudes and breaks the gate at
atol 5e-4.

### 16.4 V2-Lite `n_group=1, topk_group=1` is degenerate

The V2-Lite config has `n_group=1` and `topk_group=1`, which is
mathematically a no-op (top-1 of 1 group = greedy top-k over all experts).
The factory in `models/deepseek_v2_lite/config.py:228-234` detects this
and sets `group_routing=None` so the runtime uses the simpler greedy path.

### 16.5 Mamba-2 `seq_len % chunk_size != 0` padding

The chunked SSD scan pads the sequence to a multiple of `chunk_size`
(`ops.py:485`) and trims after the scan (`ops.py:562-563`). The pad amount
is `(chunk_size - S % chunk_size) % chunk_size`. The trailing truncation
ensures the output `y` has the right `S` regardless of padding.

### 16.6 The MLA cache stores DECOMPRESSED K/V (not the latent)

The B2b MLA implementation stores K at `qk_head_dim` and V at `v_head_dim`
in the cache. A production deployment would store the LATENT
(`c_kv: [B, S, kv_lora_rank]`) and rebuild K/V at SDPA time. The IR's
current decompressed-cache path is correct numerically but uses more
memory than a latent-cache path. See `attention.py:445-449` for the
comment. Production absorbing of `kv_b_proj` into `o_proj` is deferred.

### 16.7 Spec v3 fields that are NOT in the actual code

Working through the drift log surfaced a few v3 design points that were
silently dropped:

- **Per-axis explicit op count.** Spec v3 §8.1 cited 16 ops. The actual
  `api/ops.py` ships 18 (16 + `rope_apply_partial` and
  `rope_apply_mrope`). Plus the private helpers (`_rotate_half`,
  `_segment_sum`, `_reshape_into_chunks`, `_pad_tensor_by_size`).
- **`KVCacheSpec.k_quant` / `v_quant`** were in spec v3 §5.2.7 but the
  current `ContiguousKVCache` raises on any non-None value
  (`kvcache.py:47-48`). KV-cache quantization is reserved for future
  work.
- **`AttentionSpec.q_bias` / `k_bias` / `v_bias` / `o_bias` as four
  separate booleans.** Spec v3 §5.2.4 had a single `attention_bias` flag.
  The IR split this into four to support Qwen2's asymmetric pattern
  (Q/K/V biased, O not) and to make the MLA single-`attention_bias`
  semantic work with `spec.q_bias`.
- **`NormSpec.kind = LAYER`.** Spec v3 §5.2.1 declared this; the IR
  honours the enum value but the `RMSNorm` constructor raises on
  `LAYER` (`norm.py:17`). LayerNorm is a separate `api.ops.layer_norm`
  that no building block consumes today.
- **`DecoderBlockSpec.input_norm`** was a single field in spec v3 §5.2.8.
  The M1 refactor (`12e53f6`) split it into `pre_attn_norm` and
  `pre_ffn_norm` because the two norms have different specs in some
  families (sandwich-norm Gemma 2 has separate `pre_feedforward` and
  `input` layernorms).
- **`CacheLayout.SSM_STATE` enum value.** Declared in v3 §5.2.7 but
  `SSMStateCache` is a separate class that doesn't consume this enum.
- **`AttentionKind.LINEAR_RETENTION / LINEAR_DELTANET / LINEAR_GLA /
  DIFFERENTIAL`.** Declared in v3 §5.2.4 but every callsite raises
  `NotImplementedError`. Reserved for MiniMax Lightning Attention,
  DeltaNet, GLA, and DIFF-Transformer respectively — none of which are
  in `models/` today.
- **`TokenMixerKind.SSM_MAMBA1 / SSM_GRIFFIN / SSM_RWKV`.** Declared in
  v3 §5.2.5 but not exercised. Reserved for the deferred families.
- **`PackingLayout.NIBBLE_LSB / NIBBLE_MSB / GPTQ_INT32_PACK`.** Declared
  in v3 §5.2.6 but unused; quantization ships AWQ_INTERLEAVE, GGUF_K, and
  NONE only.
- **`QuantRole.KV_K / KV_V / ATTN_INTERNAL`.** Declared in v3 §5.2.6 but
  no tests exercise them — only `WEIGHT` and `ACTIVATION` round-trip.

### 16.8 What's INTERNAL to building blocks but NOT in the spec

Spec v3 didn't anticipate that some building-block internals would need
to be parameterized AFTER inspecting upstream code:

- **`_MambaRMSNormGated.eps`** — defaults to 1e-6 (`ssm.py:77`). HF uses
  `Mamba2Config.layer_norm_epsilon` (typically 1e-5). The mismatch is
  bridged by `SSDSpec.layer_norm_epsilon` being plumbed through
  `Mamba2Mixer.__init__` (`ssm.py:171-173`).
- **fp32 promotion in the V3 sigmoid router.** `_SigmoidRouter.forward`
  (`feedforward.py:363-365`) does `F.linear(x.float(), weight.float())`
  to mirror HF's `_keep_in_fp32_modules_strict` annotation on the V3
  router weight. The `e_score_correction_bias` buffer is also fp32 even
  when the model is bf16.
- **fp32 promotion in V2 softmax router.** `feedforward.py:207` casts
  `x.float()` AND `gate.weight.float()` before the F.linear. Without this
  the router probs differ at atol 1e-4 vs HF.
- **The bf16 residual stream cast for OLMo 2.** `models/olmo2/config.py:26`
  notes that the OLMo 2 paper recommends bf16 residual; the IR doesn't
  enforce — runtime dtype is whatever the caller passes via
  `Olmo2Config.dtype`. Tests run in fp32 for the gate.

### 16.9 Behaviours that LOOK identical but compute different results

These are the IR's "tarpits" — pairs of fields where the wrong choice
silently passes shape checks but breaks the numerical gate:

- **`partial_rotary_kind = "prefix"` vs `"proportional"`** — see §11b.
- **`router_kind = "softmax"` vs `"sigmoid_plus_bias"`** — V2 softmax over
  expert probs vs V3 sigmoid per expert.
- **`norm.weight_mode = STANDARD_W` vs `ONE_PLUS_W`** — Gemma 1/2/3 are
  the only `1+w` families.
- **`attn_scale = None` vs explicit** — default `head_dim**-0.5` vs the
  family's chosen scale (Gemma 4 = 1.0, Granite = attention_multiplier,
  Phi-3-small = mup_attn_multiplier / head_dim).
- **`qk_norm_shape = PER_HEAD_DH` vs `FULL_HDH`** — weight shape
  `[head_dim]` (broadcast across heads) vs `[n_heads * head_dim]`
  (full slab).
- **`router_norm = True` vs `False`** — Mixtral always renormalises;
  V2 typically doesn't; V3 does.
- **`attention_k_eq_v = True`** — V := K alias on Gemma 4 12B+ global
  layers only. False elsewhere (false alias → wrong V projection).
- **`v_norm` set vs not** — Gemma 4 has a UNIT RMSNorm on V; not setting
  this breaks the gate but doesn't change shape.

---

## §16b Unused / dead fields & enum values (catalog)

These are present in `api/specs.py` / `api/types.py` but consumed by no
production family today. They exist for one of three reasons:
(a) reserved for future families, (b) spec v3 parity, (c) defensive
construction that no code path actually triggers.

### Reserved for future families

- `AttentionKind.LINEAR_RETENTION` — RetNet, MiniMax Lightning Attention.
- `AttentionKind.LINEAR_DELTANET` — DeltaNet.
- `AttentionKind.LINEAR_GLA` — Gated Linear Attention.
- `AttentionKind.DIFFERENTIAL` — DIFF-Transformer.
- `TokenMixerKind.SSM_MAMBA1` — Mamba-1 selective scan.
- `TokenMixerKind.SSM_GRIFFIN` — RecurrentGemma (gated linear recurrence).
- `TokenMixerKind.SSM_RWKV` — RWKV-7 Goose.
- `CacheLayout.PAGED` — vLLM-style.
- `CacheLayout.RING` — SWA wrap-around buffer.
- `CacheLayout.SSM_STATE` — declared but the `SSMStateCache` class
  doesn't consume the enum (it has its own ctor signature).
- `CacheOwnership.STATEFUL` — Core ML state op.
- `MemoryLayout.NHD` — `[B, S, H, D]` layout.
- `QDType.FP4`, `NF4`, `MX_FP4` — variant quant dtypes.
- `MaskKind.SWA_GLOBAL_ALT`, `SINK`, `FULL`, `CUSTOM` — declared in v3 spec
  but not exercised. `CUSTOM` would be used by per-layer arbitrary mask
  patterns; SINK by attention sinks; FULL by encoder-style bidirectional.

### Spec v3 parity (declared, never wired)

- `RoPEScaling.PI`, `NTK_STATIC`, `NTK_DYNAMIC` — alternative RoPE
  long-context schemes. Today only NONE / LLAMA3 / LONGROPE / YARN are
  consumed.
- `Activation.GEGELU` — Phi-3-small uses this name but its block is
  shape-only.
- `Activation.RELU2` — squared-ReLU for some research families.
- `GateKind.GELU_ONLY`, `RELU2_ONLY` — bias-free MLPs without gating.
- `NormKind.LAYER` — Phi-3-small uses LayerNorm but its `RMSNorm`
  constructor raises on `LAYER`. The op `layer_norm` exists for backends
  to lower but no building block consumes it.
- `QKNormPhase.POST_ROPE` — declared; never wired (the `Attention`
  constructor raises on POST_ROPE per `attention.py:155`).
- `PackingLayout.NIBBLE_LSB`, `NIBBLE_MSB`, `GPTQ_INT32_PACK` — declared
  but unused.
- `QuantRole.KV_K`, `KV_V`, `ATTN_INTERNAL` — declared for KV-cache
  quantization plumbing; current `ContiguousKVCache` raises on any
  non-None `k_quant` / `v_quant`.

### Defensive / forward-compat fields

- `RoPESpec.scale_factor` — declared but unused. Was intended for a
  simpler scaling knob; replaced by per-scheme params.
- `RoPESpec.is_2d` — reserved for future axial-RoPE work; no model
  consumes it.
- `LayerScaleSpec` — reserved for OpenELM-style per-layer
  parameterisation. No `models/<family>/` builds one.
- `ConvSpec` — declared as a stand-alone spec but the `Mamba2Mixer` reads
  conv params from `SSDSpec.base` (an `SSMSpec`). `ConvSpec` is unused.
- `AttentionSpec.qk_norm_fixed_scale` — Gemma 4 B0.5 used this to absorb
  `1/sqrt(Dh)` into the QK weight; B0.6 dropped the absorb in favour of
  setting `attn_scale=1.0` directly. The field is kept for QAT-baked
  fixed-scale weights but unused by any current family.
- `SSMSpec.dt_rank` — Mamba-1's `dt_rank` projection parameter. Mamba-2
  uses per-head `dt` instead, so this field is unused (carried for
  Mamba-1 forward compat).
- `SSMSpec.use_fast_path` — flag for whether to use a CUDA-fused
  selective_scan kernel. The reference pure-PyTorch `Mamba2Mixer` always
  uses the slow path; the fast path is reserved for backends.
- `DecoderBlockSpec.final_logit_softcap` — model-level concern, carried
  on the per-layer spec for assembly convenience. The `DecoderBlock`
  doesn't consume it (the lm_head does).

## §17  Glossary

- **API** — the `api/*.py` package; the IR.
- **GQA** — Grouped-Query Attention; `n_kv_heads < n_q_heads`.
- **MLA** — Multi-head Latent Attention (DeepSeek family).
- **MoE** — Mixture-of-Experts.
- **MQA** — Multi-Query Attention (`n_kv_heads == 1`).
- **NoPE** — No Positional Encoding (Llama 4 / SmolLM3 selected layers).
- **PRE-norm** — pre-norm transformer block (norm before sublayer).
- **POST-norm** — post-norm transformer block (norm AFTER sublayer, before
  residual add). OLMo 2.
- **PLE** — Per-Layer Embedding. Gemma 4 E2B/E4B residual injection at
  every layer.
- **RoPE** — Rotary Position Embedding.
- **Sandwich norm** — `PRE_AND_POST` — both pre- and post-norm in a single
  block.
- **SDPA** — Scaled Dot-Product Attention. PyTorch built-in is
  `F.scaled_dot_product_attention`.
- **SSD** — Structured State-Space Duality (Mamba-2's chunk-parallel
  scan form).
- **SSM** — State-Space Model.
- **SWA** — Sliding-Window Attention.
- **YARN** — "Yet Another RoPE extensioN" (DeepSeek-V2/V3 long-context
  scaling).
- **softcap** — `tanh(x/cap) * cap` — saturating function applied
  pre-softmax to bound attention logits (Gemma 2 = 50.0, lm_head = 30.0).
- **QK-norm** — RMSNorm applied to Q and K before SDPA, optionally
  pre-RoPE.
- **CLA** — Cross-Layer Attention (sharing KV across layers — Hunyuan).
- **LoRA** — Low-Rank Adaptation; in MLA used as a compressed-latent
  projection.
- **AWQ** — Activation-aware Weight Quantization. INT4 grouped, asymmetric,
  AWQ_INTERLEAVE nibble permutation.
- **GGUF** — llama.cpp checkpoint format. Q4_K_M is the K-quant 4-bit
  super-block format (256 weights per super-block, 6-bit per-sub-block
  scales).
- **MXFP4** — OCP Microscaling FP4. 32-element block, UE8M0 shared
  exponent, E2M1 mantissas.
- **iRoPE** — interleaved RoPE — Llama 4 Scout's term for the
  combination of complex-multiply RoPE plus per-layer NoPE alternation.
- **DSA** — DeepSeek Sparse Attention. V3.2's Lightning Indexer routes
  queries to a per-query top-k of keys for sparse SDPA.
- **PRE-RoPE / POST-RoPE** — phase of QK-norm relative to RoPE.
- **mscale** — YARN's m-scale factor used to multiply cos/sin after the
  inv_freq blend.
- **DSL** — Down-sampling layer (not used in this doc; placeholder).
- **μP** — maximal update parametrisation; Granite / MiniCPM-3 use μP
  scalars to make layer-count-agnostic training behave.
- **HND** — `[batch, heads, seq, dim]` tensor layout.
- **NHD** — `[batch, seq, heads, dim]` tensor layout.
- **K=V alias** — Gemma 4 12B+ global layers: the V projection is
  ALIASED to the K projection's output (no separate v_proj allocated).
- **`fused_gate_up`** — single Linear of size `2*I` for SwiGLU/GeGLU,
  chunked at forward time. Phi-3, Granite-4-H shared MLP, Moshi.
- **`v_norm`** — Gemma 4's unit RMSNorm on V before transpose+cache.
- **`layer_scalar`** — Gemma 4's per-layer trailing scalar multiply.
  Buffer initialised to ones; applied even when PLE is off.
- **`first_k_dense_replace`** — DeepSeek convention: the first K layers
  use dense FFN; layers K+ use MoE.
- **`mlp_only_layers`** — Qwen3-MoE convention: list of layer indices
  that use dense FFN instead of MoE.
- **`decoder_sparse_step`** — Qwen3-MoE: MoE layer every N (1 = every
  layer; 2 = every other; etc.).
- **`layer_types`** — generic per-layer dispatch list. Gemma 2 / 3 / 4
  use "sliding_attention" / "full_attention". Granite-4-H uses "mamba" /
  "attention". Ministral uses "sliding_attention" / "full_attention".
- **`sliding_window_pattern`** — Gemma 3/4: the period of the SWA/full
  alternation (5 for 4:1, 6 for 5:1).
- **`no_rope_layers`** — Llama 4 / SmolLM3: per-layer flag for whether
  RoPE is applied. Confusingly, `1` selects RoPE; `0` selects NoPE.

---

---

## §18  The 41-Family Roster (one-paragraph each)

These are the 41 families that have landed at some level (config + factory
+ tests; some with numerical gates, some shape-only) plus the 9 deferred stubs. The descriptions
are deliberately short — for deeper detail consult `models/<family>/layer.md`
or §7 worked examples.

### 18.0 Common shape and config notes for the roster

Across all 41 families, the IR-canonical shape of one decoder layer's
input/output is `[B, S, hidden_size]`. The internal head shape is
`[B, H, S, head_dim]` (HND) inside attention. The internal SSM shape is
`[B, S, n_heads, head_dim]` (NHD-like) inside Mamba-2 — a deliberate
asymmetry because the SSD scan operates over time per head.

Family-level `hidden_size` sizes:

- Sub-billion: Qwen3-0.6B (1024), Llama 3.2-1B (2048), TinyLlama (2048),
  SmolLM3 (2048), Granite 3.1-2B (2048), MiniCPM-3 (2560), Mamba-2-2.7B
  (2560), DeepSeek-V2-Lite (2048), Voxtral (3072).
- 4-8B range: Qwen3-4B (3584), Qwen3-8B (4096), Llama 3-8B (4096),
  Mistral-7B (4096), OLMo-2-7B (4096), Phi-3-mini (3072), Phi-4-mini
  (3072), Moshi (4096), Gemma 4 4B variants.
- 12B+: Gemma 4 12B/26B/31B, Llama 4 Scout 16E (varies), DeepSeek-V3
  family (large).

### 18.1 Qwen3 (0.6B / 1.7B / 4B / 8B)

The M1 anchor. GQA + QK-norm PRE-RoPE PER_HEAD_DH + SwiGLU + RMSNorm
STANDARD_W + RoPE SPLIT_HALF + base_theta=1_000_000. Explicit `head_dim`
(not `hidden / n_heads`). `tie_word_embeddings=True` for sub-8B.

### 18.2 Llama 3 (3.0 / 3.1 / 3.2)

GQA + SwiGLU + RoPE SPLIT_HALF with optional LLAMA3 smooth scaling.
`base_theta=500_000`. No QK-norm. 3.2 1B/3B tie embeddings; 8B doesn't.

### 18.3 Mistral 7B (v0.3)

Plain GQA + SwiGLU. `sliding_window=None` (v0.2 had SWA; v0.3 dropped it).

### 18.4 Ministral

Per-layer SWA dispatch via `layer_types[layer_idx]`.

### 18.5 Mixtral 8x7B

GQA + softmax MoE (8 experts, top-k=2, always-renormalised top-k). No
shared experts, no group routing. SwiGLU experts.

### 18.6 SmolLM3 3B

Per-layer NoPE alternation via `no_rope_layers[layer_idx]`.

### 18.7 TinyLlama 1.1B

MHA Llama backbone. The factory delegates to the Llama 3 builder
(`models/tinyllama/__init__.py`).

### 18.8 Granite 3.1-2B

GQA + μP scalar stack (`embedding_multiplier`, `attention_multiplier`,
`residual_multiplier`, `logits_scaling`). `attn_scale = attention_multiplier`.

### 18.9 Granite-4-H micro

Hybrid Mamba-2 + GQA, 5:1 SSM/attention pattern. Shared MLP runs on EVERY
layer. `position_embedding_type = "nope"` (so attention has `rope=None`).

### 18.10 Phi-3 mini (4k / 128k)

FUSED QKV + fused gate_up FFN. 4k uses plain RoPE; 128k variants use
LongRoPE.

### 18.11 Phi-3-small 7B

FUSED QKV + GE-GELU + LayerNorm (`NormKind.LAYER`). Shape-only — the
`RMSNorm` constructor raises on LAYER, so the block is not built end-to-end.
Dense/blocksparse mask alternation also reserved.

### 18.12 Phi-4-mini

Phi-3-family with LongRoPE + larger `partial_rotary_factor=0.75` (rotates
96 of 128 channels — head_dim=128). Delegates to Phi-3 mini factory.

### 18.13 MiniCPM-3 4B

MLA with q_lora_rank=768, kv_lora_rank=256, qk_nope_head_dim=64,
qk_rope_head_dim=32, v_head_dim=64. LongRoPE with short_factor ==
long_factor (degenerate at production context). μP residual scale.

### 18.14 Gemma 2 (2B / 9B)

Sandwich norm + ONE_PLUS_W RMSNorm + `attn_logit_softcapping=50.0` +
`final_logit_softcapping=30.0` + SWA every other layer. GeGLU.
`embedding_scale = sqrt(hidden_size)`.

### 18.15 Gemma 3 (1B / 4B / 12B / 27B)

Like Gemma 2 plus: QK-norm PRE-RoPE PER_HEAD_DH, dual θ (10k local, 1M
global), 5:1 SWA/full pattern, NO softcap (dropped).

### 18.16 Gemma 4 (E2B / E4B / 12B / 26B / 31B)

The most adventurous landed family. STANDARD_W RMSNorm (not 1+w), p-RoPE
proportional (global = 0.25), v_norm (unit RMSNorm), K=V on 12B+ global,
cross-layer KV sharing on E2B/E4B, PLE injection on E2B/E4B,
layer_scalar on every layer, `final_logit_softcap=30.0`,
`embedding_scale = sqrt(D)`. See §7.2 for full worked example.

### 18.17 OLMo 2 (1B / 7B)

POST-norm decoder block + QK-norm PRE-RoPE FULL_HDH + STANDARD_W RMSNorm
+ SwiGLU. No biases, no SWA.

### 18.18 OLMoE 1B-7B

OLMo 2 attention shape + softmax MoE (64 experts, top-k=8). No shared
experts.

### 18.19 Qwen3-MoE

Qwen3 attention + softmax MoE + per-layer dense/MoE dispatch via
`decoder_sparse_step` and `mlp_only_layers`.

### 18.20 Mamba-2 (state-spaces/mamba2-2.7b)

Pure SSM family. `Mamba2Mixer` token mixer, `skip_ffn=True` (no FFN
sublayer). Single SSD spec across all 64 layers.

### 18.21 Llama 4 Scout (16E)

INTERLEAVED RoPE (complex-multiply) + per-layer NoPE alternation +
per-layer dense/MoE alternation. Reserved L2-norm QK-norm.

### 18.22 DeepSeek-V2-Lite 16B

MLA with `q_lora_rank=None` (direct q_proj). YARN-scaled INTERLEAVED
RoPE. Dense FFN on layer 0, softmax MoE with 2 shared experts on layers
1-26.

### 18.23 DeepSeek-V3-Lite (synthetic config)

V3 architectural family. MLA with q-LoRA, SPLIT_HALF RoPE (rope_interleave
=False path), sigmoid+bias MoE with group routing.

### 18.24 DeepSeek-V3-MoE (production-dim wrapper)

Production-shape wrapper that delegates to V3-Lite. n_routed_experts=256.

### 18.25 DeepSeek-V3.2

V3 + DSA Lightning Indexer (shape-only — `Attention.forward` raises).

### 18.26 GOT-OCR 2.0

Qwen2 backbone (0.5B). Q/K/V biases + no O bias. MHA (not GQA).

### 18.27 Qwen2.5-VL 3B

Qwen2 backbone + M-RoPE for the LM decoder. Q/K/V biases + no O bias.
`mrope_section=(16, 24, 24)`.

### 18.28 DeepSeek-OCR-2

DeepSeek-V2 family but with STANDARD attention (not MLA). Softmax MoE
with 2 shared experts. Visual Causal Flow mask reserved as alternative.

### 18.29 Moshi 7B

GQA + FUSED gate/up MLP (Moshi's "GatingMLP" fc1). `rope_theta=10000`.
`rms_norm_eps=1e-8` (unusually small).

### 18.30 Voxtral 3B

Llama-shaped LM decoder with `rope_theta=1e8` (unusually large).

### 18.30b The B9 audio families in detail

**Moshi 7B** ships an unusual fused gating MLP. The `GatingMLP.fc1`
projection produces a `2*I` tensor that is then reshaped via
`.view(B, S, 2, -1)` to extract gate and up. This is structurally a
fused gate/up (so we set `fused_gate_up=True`) but the API's chunk-2
along the last dim produces the right slices. The `intermediate_size` in
the IR is `ffn_dim // 2` because Moshi's `ffn_dim` config field is the
size of the `fc1` output (i.e. `2 * I`).

`rope_theta=10000` is the canonical Moshi value (low-θ — most modern
LLMs use higher values for longer context).

`rms_norm_eps=1e-8` is unusually small — most families use 1e-5 or 1e-6.
This shows up in the isolation tests as a stricter numerical tolerance
on the norm sub-op.

**Voxtral 3B** is a Llama 3 backbone with `rope_theta=1e8`. The 1e8 value
is set unconditionally in `configuration_voxtral.py:105` because Voxtral
inherits Mistral 7B's RoPE configuration. The IR sets up a vanilla
SPLIT_HALF RoPE at this θ — no other special handling.

### 18.30c The B8 multimodal LM decoders in detail

**GOT-OCR 2.0** uses a Qwen2-0.5B backbone. The signature difference from
Qwen3 is the Q/K/V biases: Qwen2 has `q_bias = k_bias = v_bias = True`
and `o_bias = False` (`models/got_ocr2/config.py:166`).

**Qwen2.5-VL** is Qwen2 LM decoder + M-RoPE. The M-RoPE is the IR
extension — see §1.17 and §6.12c. The bias pattern is inherited from
Qwen2.

**DeepSeek-OCR-2** is structurally a DeepSeek-V2 family but with STANDARD
attention (not MLA). It has the same softmax MoE shape (2 shared experts,
top-k=6 of 64 routed) and per-layer dense/MoE alternation. The IR's
hook for `BLOCK_BIDIRECTIONAL` (`AttentionSpec.block_bidirectional_mask`,
`MaskKind.BLOCK_BIDIRECTIONAL`) is reserved for the original DeepSeek-OCR
Visual Causal Flow but the HF v5.10.2 reference impl uses standard
CAUSAL — so the production config sets `block_bidirectional_mask=False`
and the VCF mask is not exercised by the numerical gate.

### Deferred families (stubs)

- **Mamba-1** — needs `selective_scan_mamba1` op.
- **Mamba-3** — complex-state + MIMO.
- **RecurrentGemma** (Griffin/Hawk RG-LRU).
- **RWKV-7 Goose** — outer-product WKV.
- **Hymba** — parallel hybrid block.
- **Phi-4-mini-flash** — Samba (Mamba-1 + attention sequential).
- **Falcon-H1**, **Nemotron 3** — composition + scale.
- **Jamba** — Mamba + attention + MoE.
- **MiniMax Text 01** — Lightning Attention linear-attention.

Each has a stub `models/<family>/__init__.py` documenting the reason it
was deferred. None blocks the existing IR; each requires a new op or new
building block to land.

---

*End of API Reference v4 — generated 2026-06-08 post-B10.*
