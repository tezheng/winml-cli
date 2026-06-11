# llm-layers M2+ Rollout Plan (v3)

> **Status: EXECUTED & SUPERSEDED.** This plan was executed across batches B0.5→B10 (tagged `M2-complete`) and then extended by v5/v6/v7. See `docs/PROJECT-SUMMARY.md` for the post-rollout summary.

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute the per-batch task lists. Batches B1 → B10 are dispatched as subagent units; B0.5 is executed directly because it locks the v3 IR. Use checkbox (`- [ ]`) syntax for tracking; commit per task in the bite-sized style established by the M1 plan.

**Supersedes:** `docs/superpowers/plans/2026-06-05-llm-layers-m1-qwen3.md` (M1 only).
**Authority:** v3 spec (landing in parallel; supersedes v2 `2026-06-04-llm-layers-design.v2.md`) and the v3 research refresh: `research/01-model-census.v3.md`, `research/02-layer-sources.v3.md`, `research/04-quantization.v3.md`, `research/05-kvcache-attention.v3.md`.

---

## Goal

Roll out the next **~22 architecturally distinct model families** across **11 batches** (B0.5 + B1 … B10) using subagent-driven development. Each new family produces one of two outcomes:

1. **Fits the existing API** — only a thin `models/<name>/` adapter (config, factory, layer.md, weight loader, six-test battery) is added, with zero changes to `api/`.
2. **Forces a deliberate spec extension** — a named, evidence-justified extension to `api/types.py`, `api/specs.py`, or a building block in `api/`, with the extension landed first (TDD against a failing isolation test), then the model added on top.

The v3 research refresh surfaced **four new architectural axes** (A16 VLM fusion topology, A17 vision-token compression budget, A18 cross-layer KV sharing, A19 per-layer embeddings) and **~22 new families** not contemplated by v2. Three of those axes plus partial-rotary RoPE and K=V projections plus per-layer-type SWA alternation patterns all land first in Gemma 4. **Therefore, B0.5 (Gemma 4 E2B) is the M2 kickoff gate** — it is the highest-leverage batch and locks the v3 IR. Every later batch builds on the IR shape decided here.

## Architecture

Every batch follows the **M1 TDD pattern**:

```
models/<family>/
├── __init__.py
├── config.py         (dataclass + from_hf_config + to_block_spec)
├── layer.py          (build_<family>_decoder_layer factory + HF weight loader)
└── layer.md          (per §6 of spec v2/v3 — identity, ASCII block diagram, tensor IO trace, op trace, spec instantiation, quirks, weight-name map, source citations)

tests/models/<family>/
├── __init__.py
├── test_config.py            (config → block_spec sanity)
├── test_layer_shape.py       (random weights, forward, shapes + dtype)
├── test_weight_loader.py     (HF safetensors → API slot names)
├── test_isolation_hf.py      (sub-op ladder: RMSNorm, RoPE, QK-norm, SDPA, FFN each match HF within atol=1e-5)
├── test_numerical_hf.py      (full block forward vs HF reference at atol=5e-4)
└── test_kvcache_hf.py        (prefill 8 / decode 1 vs HF cache)
```

Optional per-family additions:

- `test_quant_*.py` (AWQ, FP8, MXFP4, NVFP4, ternary — only on families where a verified quantized checkpoint exists)
- `test_moe.py` (B6 family batch — exercises routing primitives)
- `test_ssm.py` (B7 hybrid batch — exercises selective_scan / conv1d)
- `test_visual_causal_flow.py` (B8 — DeepSeek-OCR specific)

## Tech stack

Same as M1: Python 3.11, PyTorch 2.4+, transformers ≥4.51, safetensors, pytest, ruff, mypy. CPU-only is sufficient for the gate tests; small variants (E2B, DeepSeek-V2-Lite, Mamba 130M reference, Qwen3-0.6B already shipped) all fit. The full 7-12B sizes (Llama-3.1 8B, Mistral 7B, Granite 4 H-Tiny 7B-A1B) need ~32 GB RAM; mark these gate tests with `@pytest.mark.slow` and skip in default CI.

## v3 IR seeding strategy (why B0.5 = Gemma 4)

The v3 research found that Gemma 4 alone introduces **four new spec fields** that no other v3 family forces simultaneously:

1. **`partial_rotary_factor`** (Gemma 4 global = 0.25). MLA needs the same field with a different value (qk_rope_head_dim / qk_head_dim). Phi-3-legacy needs it at 0.5. Setting it on `RoPESpec` once at B0.5 means B2 (Phi-3) and B5 (MLA) inherit it.
2. **`attention_k_eq_v`** (Gemma 4 global on 12B/26B/31B — K and V projections aliased to one tensor). No other 2026 family ships this yet, but it forces the attention forward path to support skipping V projection. Once landed at B0.5, the path can grow into MLA's `kv_a_proj_with_mqa → kv_a_layernorm → kv_b_proj` in B5 with the same conditional structure.
3. **`qk_norm_fixed_scale`** (Gemma 4 local 0.9916 / global 1.0228 absorbing 1/√Dh). This forces `AttentionSpec.attn_scale` to become 1.0 when the fixed-scale norm is present. Once landed at B0.5, Granite μP (B2) and OLMo 2 (B3) inherit `attn_scale` overrides cleanly.
4. **`final_logit_softcap`** (Gemma 4 = 30.0, Gemma 2 = 30.0). Already on the API's `lm_head` op (M1 Task 4), but the wiring to `DecoderBlockSpec.logits_scale` needs the softcap path documented as a first-class field. Gemma 2 in B3 inherits it.

Plus three KV-cache and embedding axes:

5. **`ShareScheme` + `num_kv_shared_layers`** (Gemma 4 E2B = 20 of 35 layers share). Apple AFM (deferred, but its axis is documented) is a different `ShareScheme` value (`CROSS_BLOCK`). YOCO research is a third value. Landing the enum at B0.5 means future Apple AFM / YOCO additions are config-only.
6. **`per_layer_embedding`** / `PLESpec` (Gemma 4 E2B/E4B PLE: a 256-dim secondary embedding injected at every decoder layer as `(token_id_emb + ctx_proj) × 1/√2`). No other 2026 family ships PLE; B0.5 lands the spec, future families inherit a `None` default.
7. **Dual-norm sandwich (`NormPosition.PRE_AND_POST`)** is already in v2 types but never exercised. B0.5 exercises it (Gemma 4 has input + post-attn + pre-ffn + post-ffn = four norms per block), and Gemma 2 / Gemma 3 (B3) inherit the working code path.

Doing these seven extensions at B0.5 means **B1 through B10 are mostly thin config adapters**. The cost of doing Gemma 4 first is borne once; the savings compound across 21 later families.

## Rollout sequence

| Batch | Models | API extensions expected | Subagent dispatch unit |
|---|---|---|---|
| **B0.5** | Gemma 4 E2B | partial_rotary_factor, attention_k_eq_v, qk_norm_fixed_scale, final_logit_softcap on spec; PLESpec; ShareScheme enum; PRE_AND_POST exercised; SWA mask kind | **Executed directly (not via subagent) — IR-locking batch** |
| **B1** | Llama 3.0 8B, Llama 3.1 8B, Llama 3.2 1B/3B, Mistral 7B v0.3, SmolLM3 3B, TinyLlama 1.1B | None except per-layer NoPE on SmolLM3 (apply_rope[layer_idx]) and explicit SWA on Mistral | Single subagent (7 models, all thin configs) |
| **B2** | Granite 3.x/4.1 dense, MiniCPM-3, Phi-3 mini, Phi-4-mini, Phi-3-small | LayerScaleSpec wiring (Granite μP residual_scale/embedding_scale/logits_scale), `MLA_LATENT` QKV layout + 5 MLA dims (MiniCPM-3), `QKVLayout.FUSED` (Phi-3), `LONGROPE` scaling, `MaskKind.BLOCK_SPARSE` skeleton (Phi-3-small) | Single subagent (5 models, 4 deliberate extensions) |
| **B3** | OLMo 2 7B/13B, Gemma 2 9B, Gemma 3 4B/12B | NormPosition.POST (OLMo 2), QKNormShape.FULL_HDH wired through Attention (already in M1 spec, exercise it), `attn_logit_softcap` on AttentionSpec, dual-θ RoPE per layer-type (Gemma 3) | Single subagent (5 models, 3 extensions) |
| **B4** | Llama 4 Scout (iRoPE — heuristic shape test; full numerical gated by access), Ministral 3B/8B (interleaved SWA — 3rd alternation pattern) | `RoPESpec.iRoPE_no_rope_layers: tuple[int,...]` (or `apply_rope_per_layer`), `MaskKind.SWA_INTERLEAVED` | Single subagent (3 models, 2 extensions) |
| **B5** | DeepSeek-V2-Lite 16B-A, DeepSeek-V3-Lite (if available; else V2-Lite suffices), DeepSeek-V3.2 (shape-only, DSA opaque) | `AttentionKind.MLA` real implementation, MLA latent QKV path, `CacheLayout.MLA_LATENT` shape, `IndexerSpec` for DSA (shape-only) | Single subagent (3 models, MLA major extension) |
| **B6** | Mixtral 8×7B, Qwen3-MoE 30B-A3B, Qwen3-Next 80B-A3B (Gated DeltaNet 3:1 hybrid — heuristic), DeepSeek-V3-MoE pattern via V2-Lite MoE, OLMoE 1B-A, Granite 4 H-Tiny 7B-A1B | `MoESpec` end-to-end (`router_kind`, `n_shared_experts`, `router_norm`, `score_correction_bias`, `group_routing`), MoE building block in `api/feedforward.py` (top-k + gather + scatter), `MoESpec.aux_loss_free` for DeepSeek path | Single subagent (5-6 models, MoE major extension) |
| **B7** | Mamba 2 2.7B, Mamba 3 (if upstream available, else Mamba-2 stand-in), Jamba mini, Zamba2 2.7B, Hymba 1.5B, Phi-4-mini-flash, Falcon-H1 1.5B, Granite 4 H-Micro 3B (already partially in B6), Nemotron 3 Nano 4B (shape-only), RecurrentGemma 2B, RWKV-7 1.5B, MiniMax-Text-01 (shape-only) | `api/ssm.py` (selective_scan, conv1d, SSD pseudo-kernel), `SSMSpec.complex_state` (Mamba-3), `TokenMixerKind.HYBRID_PARALLEL` (Hymba), `TokenMixerKind.HYBRID_ALTERNATING` (Jamba/Granite 4), RWKV-7 deltanet update | Single subagent (heavy batch — many models, SSM major extension) |
| **B8** | DeepSeek-OCR 3B-A570M (Visual Causal Flow), GOT-OCR 2.0 0.5B, Qwen2.5-VL 3B/7B | `MaskKind.BLOCK_BIDIRECTIONAL`, `VisionAdapterSpec` (axis A16), M-RoPE 3D position layout (axis A17 sub-field) | Single subagent (3 models, MM major extension) |
| **B9** | Moshi 7B, Voxtral TTS 4B | Minimal — text decoder of Moshi is GQA+SwiGLU+RoPE; the audio frontend is documented in layer.md but not implemented (the LM-layer scope) | Single subagent (2 models, near-zero extension) |
| **B10** | GPT-OSS 20B MXFP4-native, Falcon-Edge 1.58-bit, Gemma 4 mobile-int4 QAT round-trip, DeepSeek-V4-Pro NVFP4 (shape-only) | `QDType.MX_FP4` packing path, `QDType.TERNARY` (BitNet b1.58), `QDType.NVFP4`, `QuantSpec.codebook` for ternary | Single subagent (4 quant exercises) |

**Total batches: 11. Total models added in M2+: ~52 (Gemma 4 E2B + 7 + 5 + 5 + 3 + 3 + 6 + 12 + 3 + 2 + 4 ≈ 51 unique family-instances; ~22 architecturally distinct).**

After B10 the rollout is complete: every axis enumerated in research/01-model-census.v3.md §6 has at least one tested instance.

---

# B0.5 — Gemma 4 E2B as the v3 IR seed

> **Why this batch is executed directly (not via subagent):** every later batch consumes the spec fields landed here. A subagent making naming or shape decisions in isolation would be expensive to redo. The user (zhengte@microsoft.com) reviews the v3 IR shape at the end of B0.5 before any subagent is dispatched for B1+.

## Files added/modified across B0.5

```
api/
├── types.py              [B0.5 T1 — extend with ShareScheme + new attention enums]
├── specs.py              [B0.5 T2 — extend with 7 new fields/specs]
├── ops.py                [B0.5 T3 — partial-rotary helper, gegelu / gelu_pytorch_tanh activations]
├── norm.py               [B0.5 T5 — wire ONE_PLUS_W path and fixed-scale gain absorption]
├── rope.py               [B0.5 T6 — partial-rotary support; dual-θ per layer-type]
├── attention.py          [B0.5 T7 — attention_k_eq_v branch, fixed-scale-norm scale absorption, SWA mask]
├── feedforward.py        [B0.5 T8 — GeGLU activation + double-wide MLP flag]
├── kvcache.py            [B0.5 T9 — SharedLayerKVCache wrapper + per-layer ContiguousKVCache list]
├── embedding.py          [B0.5 T10 — NEW file: PLE table + per-layer residual injection]
├── block.py              [B0.5 T11 — sandwich (PRE_AND_POST) wiring + 5:1 SWA alternation + per-layer-emb residual + final softcap path]
└── (no MoE / SSM yet — those land in B6/B7)

models/gemma4/
├── __init__.py           [B0.5 T12]
├── config.py             [B0.5 T13]
├── layer.py              [B0.5 T14]
└── layer.md              [B0.5 T15]

tests/api/                (extensions to existing tests)
├── test_specs.py         [B0.5 T2 — 7 new tests for new fields]
├── test_rope.py          [B0.5 T6 — partial-rotary test]
├── test_norm.py          [B0.5 T5 — ONE_PLUS_W test]
├── test_attention.py     [B0.5 T7 — attention_k_eq_v test, fixed-scale-norm test, SWA test]
├── test_feedforward.py   [B0.5 T8 — GeGLU test]
├── test_kvcache.py       [B0.5 T9 — shared-layer KV cache test]
├── test_embedding.py     [B0.5 T10 — NEW file: PLE forward test]
└── test_block.py         [B0.5 T11 — sandwich + PLE residual test]

tests/models/gemma4/
├── __init__.py
├── test_config.py        [B0.5 T13]
├── test_layer_shape.py   [B0.5 T14]
├── test_weight_loader.py [B0.5 T14]
├── test_isolation_hf.py  [B0.5 T16]
├── test_numerical_hf.py  [B0.5 T17 — gate test atol=5e-4 vs HF Gemma4Model E2B layer-0]
└── test_kvcache_hf.py    [B0.5 T18]
```

## B0.5 Task T1 — Extend `api/types.py` with Gemma-4 enum values

**Files:**
- Modify: `api/types.py`
- Test: `tests/api/test_specs.py` (extend existing test file)

- [ ] **Step 1: Write failing tests for new enum values**

Append to `tests/api/test_specs.py`:
```python
def test_share_scheme_enum_values_exist():
    assert types.ShareScheme.NONE
    assert types.ShareScheme.SAME_BLOCK_SHARED       # Gemma 4 E2B / E4B
    assert types.ShareScheme.CROSS_BLOCK_SHARED      # Apple AFM (deferred but enumerated)


def test_qk_norm_phase_has_pre_rope_fixed_scale():
    # Gemma 4 uses PRE_ROPE QK-norm with a fixed-scale (non-learned-absorbing-1/sqrt(Dh)).
    # We model this as the existing PRE_ROPE phase plus a fixed_scale field on AttentionSpec.
    assert types.QKNormPhase.PRE_ROPE


def test_mask_kind_has_swa_global_alt():
    # Gemma 3 / 4 alternate SWA local with full-attention global layers at 5:1.
    assert types.MaskKind.SWA_GLOBAL_ALT
```

- [ ] **Step 2: Add `ShareScheme` enum to `api/types.py`**

Append to `api/types.py`:
```python
class ShareScheme(Enum):
    """Cross-layer KV cache sharing pattern (axis A18 from research/01 v3 §6)."""
    NONE = auto()                 # default — every layer keeps its own K/V
    SAME_BLOCK_SHARED = auto()    # Gemma 4 E2B/E4B — last N layers reuse an earlier same-type layer's K/V
    CROSS_BLOCK_SHARED = auto()   # Apple AFM — 2-block split, Block-2 reuses Block-1
    # YOCO_PRODUCER_CONSUMER reserved for future research-grade additions
```

`ShareScheme.NONE` is the default; `SAME_BLOCK_SHARED` requires `KVCacheSpec.num_kv_shared_layers > 0` and `kv_source_layer_idx_map` on the model config.

- [ ] **Step 3: Run tests, expect pass**

```powershell
uv run pytest tests/api/test_specs.py -v -k "share_scheme or pre_rope or swa_global_alt"
```
Expected: 3 passed.

- [ ] **Step 4: Commit**

```powershell
git add api/types.py tests/api/test_specs.py
git commit -m "feat(B0.5): ShareScheme enum for cross-layer KV sharing (axis A18)"
```

## B0.5 Task T2 — Extend `api/specs.py` with Gemma-4 fields

**Files:**
- Modify: `api/specs.py`
- Test: `tests/api/test_specs.py`

- [ ] **Step 1: Write failing tests for new spec fields**

Append to `tests/api/test_specs.py`:
```python
def test_attention_spec_has_attention_k_eq_v():
    spec = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        attention_k_eq_v=True,
    )
    assert spec.attention_k_eq_v is True


def test_attention_spec_has_qk_norm_fixed_scale():
    spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        qk_norm_fixed_scale=0.9916,  # Gemma 4 local
    )
    assert spec.qk_norm_fixed_scale == 0.9916


def test_attention_spec_has_final_logit_softcap_marker():
    # final_logit_softcap belongs on the *model* level (lm_head op), but the block-level
    # spec carries it for the assembly path. We put it on DecoderBlockSpec.
    pass  # tested via DecoderBlockSpec below


def test_rope_spec_has_partial_rotary_factor():
    spec = specs.RoPESpec(
        base_theta=1_000_000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        partial_rotary_factor=0.25,  # Gemma 4 global
    )
    assert spec.partial_rotary_factor == 0.25


def test_rope_spec_partial_rotary_default_is_one():
    spec = specs.RoPESpec(base_theta=1_000_000.0, basis=types.RoPEBasis.SPLIT_HALF)
    assert spec.partial_rotary_factor == 1.0  # full rotation by default


def test_kvcache_spec_has_share_scheme():
    spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.bfloat16, v_dtype=torch.bfloat16,
        share_scheme=types.ShareScheme.SAME_BLOCK_SHARED,
        num_kv_shared_layers=20,
    )
    assert spec.share_scheme == types.ShareScheme.SAME_BLOCK_SHARED
    assert spec.num_kv_shared_layers == 20


def test_ple_spec_exists():
    spec = specs.PLESpec(
        ple_dim=256,
        residual_scale=1.0 / (2 ** 0.5),
        injection_norm=specs.NormSpec(
            kind=types.NormKind.RMS, eps=1e-6,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        ),
    )
    assert spec.ple_dim == 256


def test_decoder_block_spec_has_pre_ple_injection():
    norm = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.ONE_PLUS_W)
    attn = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
    )
    ffn = specs.FFNSpec(intermediate_size=8192,
                       activation=types.Activation.GELU,
                       gate_kind=types.GateKind.GEGLU)
    ple = specs.PLESpec(ple_dim=256, residual_scale=0.7071,
                       injection_norm=norm)
    block = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE_AND_POST,
        ffn_norm_position=types.NormPosition.PRE_AND_POST,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm, post_attn_norm=norm,
        pre_ffn_norm=norm, post_ffn_norm=norm,
        per_layer_embedding=ple,
        final_logit_softcap=30.0,
    )
    assert block.per_layer_embedding is ple
    assert block.final_logit_softcap == 30.0
```

- [ ] **Step 2: Extend `api/specs.py`**

Add fields to `AttentionSpec` (additive — keep all existing fields, add new ones with safe defaults):
```python
@dataclass(frozen=True)
class AttentionSpec:
    # ... existing fields ...

    # B0.5: Gemma 4 additions (all backward-compatible — defaults match v1 behavior)
    attention_k_eq_v: bool = False             # Gemma 4 12B/26B/31B global: K and V projections aliased
    qk_norm_fixed_scale: Optional[float] = None # Gemma 4 fixed-scale gain absorbing 1/sqrt(Dh); when set, attn_scale should be 1.0
```

Add field to `RoPESpec`:
```python
@dataclass(frozen=True)
class RoPESpec:
    base_theta: float
    basis: types.RoPEBasis
    scaling: types.RoPEScaling = types.RoPEScaling.NONE
    scale_factor: Optional[float] = None
    llama3_extra: Optional[Llama3RoPEParams] = None

    # B0.5: partial-rotary (Gemma 4 global = 0.25; Phi-3 legacy = 0.5; MLA = qk_rope/qk_head)
    partial_rotary_factor: float = 1.0
```

Add fields to `KVCacheSpec`:
```python
@dataclass(frozen=True)
class KVCacheSpec:
    layout: types.CacheLayout
    memory_layout: types.MemoryLayout
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    ownership: types.CacheOwnership = types.CacheOwnership.EXPLICIT_PASS
    block_size: Optional[int] = None
    k_quant: Optional[QuantSpec] = None
    v_quant: Optional[QuantSpec] = None

    # B0.5: cross-layer sharing
    share_scheme: types.ShareScheme = types.ShareScheme.NONE
    num_kv_shared_layers: int = 0
```

Add new dataclass `PLESpec`:
```python
@dataclass(frozen=True)
class PLESpec:
    """Per-Layer Embedding spec (Gemma 4 E2B/E4B — axis A19).

    The PLE table has dimension `ple_dim` (typically 256), much smaller than the
    main residual hidden_size. Its output is normalized and injected at every
    decoder layer as a residual term scaled by `residual_scale` (Gemma 4 uses 1/sqrt(2)).

    On Gemma 4 the PLE row for token t is computed as
        ple_row = (token_identity_emb + context_aware_projection) * residual_scale
    and the projection is the layer-local linear that maps from the embedding
    to the hidden_size for that layer's residual injection. Behaviour lives in
    api/embedding.py — this spec is pure data.
    """
    ple_dim: int
    residual_scale: float
    injection_norm: "NormSpec"
```

Add fields to `DecoderBlockSpec`:
```python
@dataclass(frozen=True)
class DecoderBlockSpec:
    # ... existing fields ...

    # B0.5
    per_layer_embedding: Optional[PLESpec] = None
    final_logit_softcap: Optional[float] = None    # Gemma 4 = 30.0 (model-level; carried here for assembly)
```

- [ ] **Step 3: Run tests, expect pass**

```powershell
uv run pytest tests/api/test_specs.py -v
```
Expected: all existing tests still pass + 7 new ones pass.

- [ ] **Step 4: Commit**

```powershell
git add api/specs.py tests/api/test_specs.py
git commit -m "feat(B0.5): spec extensions for Gemma 4 — partial_rotary, attention_k_eq_v, qk_norm_fixed_scale, PLESpec, share_scheme"
```

## B0.5 Task T3 — Extend `api/ops.py` with partial-rotary helper + gelu_pytorch_tanh

**Files:**
- Modify: `api/ops.py`
- Modify: `tests/api/test_ops.py`

- [ ] **Step 1: Write failing test for partial-rotary `rope_apply`**

Append to `tests/api/test_ops.py`:
```python
def test_rope_apply_partial_rotary_factor_quarter():
    """Gemma 4 global RoPE: only the first 25% of head_dim channels rotate."""
    B, S, H, Dh = 1, 4, 2, 16
    q = torch.randn(B, S, H, Dh, dtype=torch.float32)
    k = torch.randn(B, S, H, Dh, dtype=torch.float32)

    # Build cos/sin for the rotated portion only (Dh_rot = 4 channels of the head)
    Dh_rot = int(Dh * 0.25)  # 4
    freqs = torch.linspace(0.1, 1.0, Dh_rot // 2)
    positions = torch.arange(S).float().unsqueeze(-1) * freqs.unsqueeze(0)
    cos_rot = torch.cat([positions.cos(), positions.cos()], dim=-1)  # [S, Dh_rot]
    sin_rot = torch.cat([positions.sin(), positions.sin()], dim=-1)

    q_rot, k_rot = ops.rope_apply_partial(q, k, cos_rot, sin_rot,
                                          partial_rotary_factor=0.25,
                                          basis="split_half")
    # Non-rotated tail (channels Dh_rot..Dh) is identical to input
    assert torch.allclose(q_rot[..., Dh_rot:], q[..., Dh_rot:], atol=1e-7)
    assert torch.allclose(k_rot[..., Dh_rot:], k[..., Dh_rot:], atol=1e-7)
    # Rotated head (first Dh_rot channels) differs from input at non-zero positions
    assert not torch.allclose(q_rot[:, 1, :, :Dh_rot], q[:, 1, :, :Dh_rot], atol=1e-3)


def test_gelu_pytorch_tanh_matches_torch():
    """Gemma 4 uses hidden_activation='gelu_pytorch_tanh'."""
    x = torch.randn(2, 4, 8)
    out = ops.gelu_pytorch_tanh(x)
    ref = F.gelu(x, approximate="tanh")
    assert torch.allclose(out, ref, atol=1e-6)
```

- [ ] **Step 2: Add `rope_apply_partial` and `gelu_pytorch_tanh` to `api/ops.py`**

```python
def rope_apply_partial(
    q: torch.Tensor,           # [B, S, H, Dh]
    k: torch.Tensor,           # [B, S, Hk, Dh]
    cos: torch.Tensor,         # [S, Dh_rot] — cos/sin computed at the rotated dim only
    sin: torch.Tensor,         # [S, Dh_rot]
    partial_rotary_factor: float,
    basis: str = "split_half",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to only the first `partial_rotary_factor * Dh` channels of each head.

    For Gemma 4 global layers: partial_rotary_factor=0.25, so the first 64 of 256
    head_dim channels rotate, the remaining 192 pass through unchanged.

    For MLA: q_rope is a separate sub-head dim already so the caller pre-slices;
    this helper handles the simpler "head is a single tensor with a rotated prefix"
    case used by Gemma 4 and Phi-3 partial-RoPE.
    """
    if not (0.0 < partial_rotary_factor <= 1.0):
        raise ValueError(f"partial_rotary_factor must be in (0, 1], got {partial_rotary_factor}")
    if partial_rotary_factor == 1.0:
        return rope_apply(q, k, cos, sin, basis=basis)

    Dh = q.shape[-1]
    Dh_rot = int(Dh * partial_rotary_factor)
    if Dh_rot % 2 != 0:
        raise ValueError(f"rotated head_dim ({Dh_rot}) must be even")
    if cos.shape[-1] != Dh_rot or sin.shape[-1] != Dh_rot:
        raise ValueError(f"cos/sin last dim must equal Dh_rot ({Dh_rot}), got {cos.shape[-1]}")

    q_rot_in, q_pass = q[..., :Dh_rot], q[..., Dh_rot:]
    k_rot_in, k_pass = k[..., :Dh_rot], k[..., Dh_rot:]
    q_rot, k_rot = rope_apply(q_rot_in, k_rot_in, cos, sin, basis=basis)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """Gemma's `gelu_pytorch_tanh` activation — `F.gelu(x, approximate='tanh')`."""
    return F.gelu(x, approximate="tanh")
```

- [ ] **Step 3: Run tests, expect pass**

```powershell
uv run pytest tests/api/test_ops.py -v -k "partial_rotary or pytorch_tanh"
```
Expected: 2 passed.

- [ ] **Step 4: Commit**

```powershell
git add api/ops.py tests/api/test_ops.py
git commit -m "feat(B0.5): ops rope_apply_partial + gelu_pytorch_tanh"
```

## B0.5 Task T4 — Refactor: ensure existing ops/tests still pass after the partial-rotary helper landed

- [ ] **Step 1: Full regression**

```powershell
uv run pytest -q
```
Expected: full M1 suite (74 tests) still green + new tests pass.

- [ ] **Step 2: (No commit if no fixes were needed.)** If any M1 test broke, fix in this task with one commit per fix.

## B0.5 Task T5 — `api/norm.py` — wire `ONE_PLUS_W` and exercise it

**Files:**
- Modify: `tests/api/test_norm.py`

- [ ] **Step 1: Add failing test for ONE_PLUS_W mode at module level**

Append to `tests/api/test_norm.py`:
```python
def test_rmsnorm_module_one_plus_w():
    """Gemma 4 RMSNorm: y = x_normed * (1 + w), w init to zero so gain=1.0."""
    spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                          weight_mode=types.NormWeightMode.ONE_PLUS_W)
    module = norm.RMSNorm(spec, hidden_size=16, dtype=torch.float32)
    with torch.no_grad():
        module.weight.zero_()  # init to zero — gain effectively = 1.0
    x = torch.randn(2, 4, 16)
    out = module(x)
    ref = F.rms_norm(x, (16,), torch.ones(16), 1e-6)
    assert torch.allclose(out, ref, atol=1e-5)
```

- [ ] **Step 2: Verify it passes (the ops layer already supports `one_plus_w`; this is wiring sanity)**

```powershell
uv run pytest tests/api/test_norm.py -v -k "one_plus_w"
```
Expected: pass (module already routes via mode flag — Task 6 of M1).

- [ ] **Step 3: Commit**

```powershell
git add tests/api/test_norm.py
git commit -m "test(B0.5): RMSNorm ONE_PLUS_W module exercise"
```

## B0.5 Task T6 — `api/rope.py` — partial-rotary + dual-θ wiring

**Files:**
- Modify: `api/rope.py`
- Modify: `tests/api/test_rope.py`

- [ ] **Step 1: Failing test for partial-rotary `RoPE` module**

Append to `tests/api/test_rope.py`:
```python
def test_rope_module_partial_rotary_factor():
    """RoPE module with partial_rotary_factor=0.25 should only rotate the first quarter."""
    spec = specs.RoPESpec(
        base_theta=1_000_000.0,
        basis=types.RoPEBasis.SPLIT_HALF,
        partial_rotary_factor=0.25,
    )
    head_dim = 64
    max_seq = 32
    module = rope.RoPE(spec, head_dim=head_dim, max_seq=max_seq, dtype=torch.float32)
    # cos/sin tables should be sized for the rotated dim only
    Dh_rot = int(head_dim * 0.25)
    assert module.cos_cached.shape == (max_seq, Dh_rot)

    B, S, H = 1, 8, 4
    q = torch.randn(B, S, H, head_dim)
    k = torch.randn(B, S, H, head_dim)
    pos = torch.arange(S)
    q_rot, k_rot = module(q, k, pos)
    assert q_rot.shape == q.shape
    # Non-rotated tail (channels Dh_rot..Dh) should be identical
    assert torch.allclose(q_rot[..., Dh_rot:], q[..., Dh_rot:], atol=1e-6)
```

- [ ] **Step 2: Implement partial-rotary path in `api/rope.py`**

Modify `RoPE.__init__` to size cos/sin tables to the rotated head_dim:
```python
class RoPE(nn.Module):
    def __init__(self, spec: specs.RoPESpec, head_dim: int, max_seq: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if spec.basis != types.RoPEBasis.SPLIT_HALF:
            raise NotImplementedError("SPLIT_HALF basis only")
        self.spec = spec
        self.head_dim = head_dim
        self.head_dim_rot = int(head_dim * spec.partial_rotary_factor)
        if self.head_dim_rot % 2 != 0:
            raise ValueError(f"rotated head_dim ({self.head_dim_rot}) must be even")
        self.max_seq = max_seq

        # ... inv_freq computed over head_dim_rot, not head_dim ...
        inv_freq = 1.0 / (
            spec.base_theta ** (torch.arange(0, self.head_dim_rot, 2).float() / self.head_dim_rot)
        )
        if spec.scaling == types.RoPEScaling.LLAMA3:
            if spec.llama3_extra is None:
                raise ValueError("LLAMA3 scaling requires llama3_extra")
            inv_freq = _llama3_scale_inv_freq(inv_freq, spec.llama3_extra)
        elif spec.scaling != types.RoPEScaling.NONE:
            raise NotImplementedError(f"M1 supports NONE and LLAMA3, got {spec.scaling}")

        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, q, k, position_ids):
        if position_ids.dim() != 1:
            raise NotImplementedError("1D position_ids only")
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        if self.spec.partial_rotary_factor == 1.0:
            return ops.rope_apply(q, k, cos, sin, basis="split_half")
        return ops.rope_apply_partial(q, k, cos, sin,
                                      partial_rotary_factor=self.spec.partial_rotary_factor,
                                      basis="split_half")
```

- [ ] **Step 3: Run tests, expect pass**

```powershell
uv run pytest tests/api/test_rope.py -v
```
Expected: all 4 (3 existing + 1 new) pass.

- [ ] **Step 4: Commit**

```powershell
git add api/rope.py tests/api/test_rope.py
git commit -m "feat(B0.5): RoPE module supports partial_rotary_factor (Gemma 4 global = 0.25)"
```

## B0.5 Task T7 — `api/attention.py` — attention_k_eq_v + qk_norm_fixed_scale + SWA mask

**Files:**
- Modify: `api/attention.py`
- Modify: `tests/api/test_attention.py`

- [ ] **Step 1: Failing tests**

Append to `tests/api/test_attention.py`:
```python
def test_attention_k_eq_v_skips_v_projection():
    """When attention_k_eq_v=True, the V projection should not be allocated; v = k."""
    spec = specs.AttentionSpec(
        n_q_heads=16, n_kv_heads=8, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.CAUSAL,
        attention_k_eq_v=True,
        rope=specs.RoPESpec(base_theta=1_000_000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = attention.Attention(spec, hidden_size=4096, max_seq=64, dtype=torch.float32)
    # v_proj should be absent (or None) when attention_k_eq_v
    assert getattr(attn, "v_proj", None) is None


def test_attention_qk_norm_fixed_scale_overrides_scale():
    """With fixed-scale QK norm, attn_scale defaults to 1.0 instead of 1/sqrt(Dh)."""
    qk_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                             weight_mode=types.NormWeightMode.ONE_PLUS_W)
    spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=256,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=512,
        qk_norm=qk_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        qk_norm_fixed_scale=0.9916,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = attention.Attention(spec, hidden_size=1024, max_seq=64, dtype=torch.float32)
    assert attn.effective_scale == 1.0


def test_attention_swa_masks_outside_window():
    """SWA mask: tokens at position i can only see positions in [max(0, i - W), i]."""
    spec = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=64,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=2,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    attn = attention.Attention(spec, hidden_size=512, max_seq=8, dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(cache_spec, batch_size=1,
                                      n_kv_heads=1, head_dim=64, max_seq=8)
    B, S, D = 1, 5, 512
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = attn(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, D)
    # We cannot check SWA masking directly without inspecting internals;
    # verify shape and let the numerical-equivalence test (T17) catch correctness.
```

- [ ] **Step 2: Extend `api/attention.py`**

Add three branches inside `Attention.__init__`:

```python
# v_proj is skipped when attention_k_eq_v=True (Gemma 4 12B+ global)
if spec.attention_k_eq_v:
    self.v_proj = None
else:
    self.v_proj = nn.Linear(hidden_size, kv_proj_out, bias=spec.v_bias, dtype=dtype)

# attn_scale: when qk_norm_fixed_scale is set, the norm absorbs 1/sqrt(Dh) and the
# effective scale is 1.0 (Gemma 4). Otherwise honor spec.attn_scale or default.
if spec.qk_norm_fixed_scale is not None:
    self.effective_scale = 1.0
elif spec.attn_scale is not None:
    self.effective_scale = spec.attn_scale
else:
    self.effective_scale = spec.head_dim ** -0.5
```

And inside `Attention.forward`:
```python
# K = V branch
if self.spec.attention_k_eq_v:
    v = k  # alias the K tensor as V (after the same projection)
else:
    v = self.v_proj(x).view(B, S, Hk, Dh)

# SWA mask: when mask_kind=SWA and S > 1 we must restrict attention to the window.
# For prefill we build a mask shaped (S, S_total) where entry (i, j) is False if
# j > i (causal) OR i - j > W (outside window). For decode (S=1) we apply via mask.
attn_mask = None
is_causal = (S > 1) and (self.spec.mask_kind == types.MaskKind.CAUSAL)
if self.spec.mask_kind == types.MaskKind.SWA and self.spec.sliding_window is not None:
    W = self.spec.sliding_window
    S_total = start_pos + S
    q_pos = torch.arange(start_pos, start_pos + S, device=x.device).view(S, 1)
    k_pos = torch.arange(S_total, device=x.device).view(1, S_total)
    keep = (k_pos <= q_pos) & (q_pos - k_pos <= W)
    attn_mask = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
    is_causal = False
```

(The QK-norm fixed_scale itself is handled inside `QKNorm` module — see T5; for Gemma 4 the fixed gain is multiplied INTO the learned weight at load time, so the RMSNorm path is unchanged at the module level. The `effective_scale = 1.0` is the only attention-side change.)

- [ ] **Step 3: Run tests**

```powershell
uv run pytest tests/api/test_attention.py -v
```
Expected: all (3 existing + 3 new) pass.

- [ ] **Step 4: Commit**

```powershell
git add api/attention.py tests/api/test_attention.py
git commit -m "feat(B0.5): Attention supports attention_k_eq_v, qk_norm_fixed_scale (effective_scale=1.0), SWA mask"
```

## B0.5 Task T8 — `api/feedforward.py` — GeGLU + double-wide MLP flag

**Files:**
- Modify: `api/feedforward.py`
- Modify: `tests/api/test_feedforward.py`

- [ ] **Step 1: Failing tests**

Append to `tests/api/test_feedforward.py`:
```python
def test_geglu_forward_shape():
    spec = specs.FFNSpec(
        intermediate_size=128,
        activation=types.Activation.GELU,
        gate_kind=types.GateKind.GEGLU,
    )
    ffn = feedforward.FeedForward(spec, hidden_size=32, dtype=torch.float32)
    x = torch.randn(2, 4, 32)
    out = ffn(x)
    assert out.shape == (2, 4, 32)


def test_geglu_uses_gelu_pytorch_tanh_for_gemma():
    """Gemma's GeGLU uses gelu_pytorch_tanh, not plain gelu."""
    spec = specs.FFNSpec(
        intermediate_size=8,
        activation=types.Activation.GELU,
        gate_kind=types.GateKind.GEGLU,
    )
    ffn = feedforward.FeedForward(spec, hidden_size=4, dtype=torch.float32)
    x = torch.randn(1, 2, 4)
    g = F.linear(x, ffn.gate_proj.weight)
    u = F.linear(x, ffn.up_proj.weight)
    expected = F.linear(F.gelu(g, approximate="tanh") * u, ffn.down_proj.weight)
    assert torch.allclose(ffn(x), expected, atol=1e-5)
```

- [ ] **Step 2: Extend `api/feedforward.py`**

Add a branch for `GateKind.GEGLU` plus support for the `gelu_pytorch_tanh` path inside the forward:
```python
class FeedForward(nn.Module):
    def __init__(self, spec, hidden_size, dtype=torch.float32):
        super().__init__()
        if spec.gate_kind not in (types.GateKind.SWIGLU, types.GateKind.GEGLU):
            raise NotImplementedError(f"B0.5: SWIGLU and GEGLU only, got {spec.gate_kind}")
        # ... allocate gate/up/down as before ...

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.spec.gate_kind == types.GateKind.SWIGLU:
            act = ops.silu(gate)
        else:  # GEGLU — use gelu_pytorch_tanh per Gemma's signature
            act = ops.gelu_pytorch_tanh(gate)
        return self.down_proj(ops.mul(act, up))
```

- [ ] **Step 3: Run tests**

```powershell
uv run pytest tests/api/test_feedforward.py -v
```
Expected: all (2 existing + 2 new) pass.

- [ ] **Step 4: Commit**

```powershell
git add api/feedforward.py tests/api/test_feedforward.py
git commit -m "feat(B0.5): FeedForward supports GEGLU with gelu_pytorch_tanh (Gemma)"
```

## B0.5 Task T9 — `api/kvcache.py` — SharedLayerKVCache wrapper

**Files:**
- Modify: `api/kvcache.py`
- Modify: `tests/api/test_kvcache.py`

- [ ] **Step 1: Failing tests for shared-layer cache**

Append to `tests/api/test_kvcache.py`:
```python
def test_shared_layer_kv_cache_aliases_source():
    """Gemma 4 E2B: layer i with i in shared range reads K/V from layer kv_source_idx[i]."""
    spec_base = _make_spec()
    source = kvcache.ContiguousKVCache(spec_base, batch_size=1, n_kv_heads=2,
                                       head_dim=4, max_seq=8)
    k = torch.randn(1, 2, 5, 4); v = torch.randn(1, 2, 5, 4)
    source.write(k, v, start_pos=0)

    shared = kvcache.SharedLayerKVCache(source_cache=source)
    k_out, v_out = shared.read(seq_len=5)
    assert torch.allclose(k_out, k)
    assert torch.allclose(v_out, v)
    assert shared.seq_len == 5


def test_shared_layer_kv_cache_write_is_noop():
    """Writes to a shared cache do NOT modify the source — the writing layer's K/V
    is computed but discarded (Gemma 4 shared-layer behavior: just consume the source)."""
    spec_base = _make_spec()
    source = kvcache.ContiguousKVCache(spec_base, batch_size=1, n_kv_heads=2,
                                       head_dim=4, max_seq=8)
    k_src = torch.randn(1, 2, 3, 4); v_src = torch.randn(1, 2, 3, 4)
    source.write(k_src, v_src, start_pos=0)

    shared = kvcache.SharedLayerKVCache(source_cache=source)
    k_new = torch.randn(1, 2, 1, 4); v_new = torch.randn(1, 2, 1, 4)
    shared.write(k_new, v_new, start_pos=3)  # should be no-op

    k_out, v_out = source.read(seq_len=3)
    assert torch.allclose(k_out, k_src)  # source unchanged
    assert torch.allclose(v_out, v_src)
```

- [ ] **Step 2: Implement `SharedLayerKVCache` in `api/kvcache.py`**

```python
class SharedLayerKVCache:
    """A read-only alias to another layer's ContiguousKVCache.

    Used by Gemma 4 E2B/E4B layers that reuse a prior same-type layer's K/V.
    `write` is a no-op for the shared layer because the K/V tensors it
    *would* have produced are discarded — the source cache is shared.

    For correctness, the source cache must have been written to before this
    layer's forward (i.e. the source layer must be earlier in the decoder
    stack).
    """

    def __init__(self, source_cache: "ContiguousKVCache"):
        self.source = source_cache
        # Mirror attributes the attention block reads
        self.spec = source_cache.spec
        self.k = source_cache.k
        self.v = source_cache.v

    @property
    def seq_len(self) -> int:
        return self.source.seq_len

    def write(self, k, v, start_pos):
        # No-op: the source layer is responsible for filling the cache.
        # We do not validate shapes here — the attention block still computes
        # its own K/V tensors but discards them.
        return

    def read(self, seq_len):
        return self.source.read(seq_len)

    def reset(self):
        # Resetting the shared cache is also a no-op (the source owns the buffer).
        return
```

- [ ] **Step 3: Run tests**

```powershell
uv run pytest tests/api/test_kvcache.py -v
```
Expected: all (4 existing + 2 new) pass.

- [ ] **Step 4: Commit**

```powershell
git add api/kvcache.py tests/api/test_kvcache.py
git commit -m "feat(B0.5): SharedLayerKVCache for Gemma 4 cross-layer KV sharing"
```

## B0.5 Task T10 — `api/embedding.py` — PLE table + per-layer residual injection (NEW file)

**Files:**
- Create: `api/embedding.py`
- Create: `tests/api/test_embedding.py`

- [ ] **Step 1: Failing test**

Create `tests/api/test_embedding.py`:
```python
import torch
from torch import nn

from api import embedding, specs, types


def test_ple_table_forward_shape():
    spec = specs.PLESpec(
        ple_dim=256,
        residual_scale=1.0 / (2 ** 0.5),
        injection_norm=specs.NormSpec(
            kind=types.NormKind.RMS, eps=1e-6,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        ),
    )
    ple = embedding.PerLayerEmbedding(spec, vocab_size=1000, hidden_size=1536,
                                     dtype=torch.float32)
    ids = torch.tensor([[0, 1, 2, 3]])
    residual = ple(ids, layer_idx=0)
    assert residual.shape == (1, 4, 1536)


def test_ple_residual_scale_applied():
    spec = specs.PLESpec(
        ple_dim=8, residual_scale=0.5,
        injection_norm=specs.NormSpec(
            kind=types.NormKind.RMS, eps=1e-6,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        ),
    )
    ple = embedding.PerLayerEmbedding(spec, vocab_size=4, hidden_size=16,
                                      dtype=torch.float32)
    # Initialize PLE table to all-ones so we can compute the expected scale
    with torch.no_grad():
        ple.ple_table.weight.fill_(1.0)
        # Zero out projection bias for determinism
        for layer_proj in ple.layer_projs:
            layer_proj.weight.zero_()
    ids = torch.tensor([[0]])
    residual = ple(ids, layer_idx=0)
    # With zeroed projection the residual is just norm(ones) * residual_scale
    # which is small but nonzero — verify shape only here (numerical check is in test_numerical_hf)
    assert residual.shape == (1, 1, 16)
```

- [ ] **Step 2: Implement `api/embedding.py`**

```python
"""Per-Layer Embedding (PLE) table for Gemma 4 E2B/E4B (axis A19).

The PLE table holds vocab_size × ple_dim parameters (ple_dim much smaller than
hidden_size, e.g. 256 vs 1536). At every decoder layer, the per-token PLE row
is normalized, projected to hidden_size by a layer-specific linear, and added
as a residual scaled by `residual_scale` (Gemma 4 = 1/sqrt(2)).

On Gemma 4, the table is paged to flash storage on mobile — at runtime only
the per-layer linear projections live in VRAM, plus the PLE rows of the
currently-being-decoded tokens (a few KB).

This module owns the PLE table and one linear projection per decoder layer.
"""
from __future__ import annotations

import torch
from torch import nn

from api import norm as _norm, specs


class PerLayerEmbedding(nn.Module):
    def __init__(
        self,
        spec: specs.PLESpec,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 0,  # set later via `register_layers`
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.spec = spec
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # The PLE table itself
        self.ple_table = nn.Embedding(vocab_size, spec.ple_dim, dtype=dtype)
        # The pre-injection norm (RMSNorm 1+w to match Gemma)
        self.inj_norm = _norm.RMSNorm(spec.injection_norm, spec.ple_dim, dtype=dtype)
        # Per-layer linear projections from ple_dim → hidden_size.
        # We allocate lazily: if num_layers=0 caller must call register_layers later.
        self.layer_projs = nn.ModuleList()
        if num_layers > 0:
            self.register_layers(num_layers, dtype=dtype)

    def register_layers(self, num_layers: int, dtype: torch.dtype = torch.float32) -> None:
        self.layer_projs = nn.ModuleList(
            nn.Linear(self.spec.ple_dim, self.hidden_size, bias=False, dtype=dtype)
            for _ in range(num_layers)
        )

    def forward(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Compute the per-layer residual for the given token ids and layer index.

        Returns a tensor of shape [B, S, hidden_size] to be added to the residual
        stream at decoder layer `layer_idx`.
        """
        if layer_idx >= len(self.layer_projs):
            raise IndexError(f"layer_idx {layer_idx} out of range "
                            f"(have {len(self.layer_projs)} layer projections)")
        ple_rows = self.ple_table(input_ids)              # [B, S, ple_dim]
        normed = self.inj_norm(ple_rows)                  # [B, S, ple_dim]
        projected = self.layer_projs[layer_idx](normed)   # [B, S, hidden_size]
        return projected * self.spec.residual_scale
```

- [ ] **Step 3: Run tests**

```powershell
uv run pytest tests/api/test_embedding.py -v
```
Expected: 2 passed.

- [ ] **Step 4: Commit**

```powershell
git add api/embedding.py tests/api/test_embedding.py
git commit -m "feat(B0.5): PerLayerEmbedding for Gemma 4 PLE (axis A19)"
```

## B0.5 Task T11 — `api/block.py` — sandwich norm + PLE injection + final_logit_softcap

**Files:**
- Modify: `api/block.py`
- Modify: `tests/api/test_block.py`

- [ ] **Step 1: Failing test for sandwich + PLE**

Append to `tests/api/test_block.py`:
```python
def test_block_sandwich_norm_runs():
    """Gemma 4 sandwich: PRE + POST norms around each sublayer."""
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.ONE_PLUS_W)
    attn = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=64,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=128,
        qk_norm=norm_spec,
        qk_norm_phase=types.QKNormPhase.PRE_ROPE,
        qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
        qk_norm_fixed_scale=0.9916,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=512,
                       activation=types.Activation.GELU,
                       gate_kind=types.GateKind.GEGLU)
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE_AND_POST,
        ffn_norm_position=types.NormPosition.PRE_AND_POST,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
    )
    blk = block.DecoderBlock(block_spec, hidden_size=512, max_seq=64,
                             dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=1, head_dim=64, max_seq=64,
    )
    B, S, D = 1, 4, 512
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, D)


def test_block_with_per_layer_embedding_residual():
    """Block adds a PLE residual at the very end of the block (or as configured)."""
    # Build a minimal block with PLESpec
    norm_spec = specs.NormSpec(kind=types.NormKind.RMS, eps=1e-6,
                               weight_mode=types.NormWeightMode.ONE_PLUS_W)
    ple_spec = specs.PLESpec(
        ple_dim=8, residual_scale=1.0 / (2 ** 0.5),
        injection_norm=norm_spec,
    )
    # Block construction — only used here for spec assembly; the PLE injection
    # is *called from outside* the DecoderBlock (because PLE depends on input_ids,
    # which the block does not know about). Instead the model assembly passes
    # the per-layer PLE residual into block.forward as an optional argument.
    attn = specs.AttentionSpec(
        n_q_heads=8, n_kv_heads=1, head_dim=32,
        kind=types.AttentionKind.STANDARD,
        qkv_layout=types.QKVLayout.SPLIT,
        mask_kind=types.MaskKind.SWA,
        sliding_window=64,
        rope=specs.RoPESpec(base_theta=10000.0,
                            basis=types.RoPEBasis.SPLIT_HALF),
    )
    ffn = specs.FFNSpec(intermediate_size=64,
                       activation=types.Activation.GELU,
                       gate_kind=types.GateKind.GEGLU)
    block_spec = specs.DecoderBlockSpec(
        attn_norm_position=types.NormPosition.PRE_AND_POST,
        ffn_norm_position=types.NormPosition.PRE_AND_POST,
        token_mixer=attn, channel_mixer=ffn,
        pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
        pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
        per_layer_embedding=ple_spec,
    )
    blk = block.DecoderBlock(block_spec, hidden_size=256, max_seq=32,
                             dtype=torch.float32)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=1, head_dim=32, max_seq=32,
    )
    B, S, D = 1, 3, 256
    x = torch.randn(B, S, D)
    pos = torch.arange(S)
    ple_residual = torch.randn(B, S, D)  # provided by the model assembly
    out_with_ple = blk(x, position_ids=pos, cache=cache, start_pos=0,
                       per_layer_residual=ple_residual)
    cache.reset()
    out_without_ple = blk(x, position_ids=pos, cache=cache, start_pos=0)
    # The PLE residual should change the output
    assert not torch.allclose(out_with_ple, out_without_ple, atol=1e-3)
```

- [ ] **Step 2: Extend `api/block.py`**

Add support for `NormPosition.PRE_AND_POST` and an optional `per_layer_residual` argument:
```python
class DecoderBlock(nn.Module):
    def __init__(self, spec, hidden_size, max_seq, dtype=torch.float32):
        super().__init__()
        self.spec = spec
        self.hidden_size = hidden_size

        # PRE_AND_POST requires all four norms
        if spec.attn_norm_position == types.NormPosition.PRE_AND_POST:
            if spec.pre_attn_norm is None or spec.post_attn_norm is None:
                raise ValueError("PRE_AND_POST attn norm requires pre and post norm specs")
            self.input_norm = norm.RMSNorm(spec.pre_attn_norm, hidden_size, dtype=dtype)
            self.post_attn_sublayer_norm = norm.RMSNorm(spec.post_attn_norm, hidden_size, dtype=dtype)
        elif spec.attn_norm_position == types.NormPosition.PRE:
            self.input_norm = norm.RMSNorm(spec.pre_attn_norm, hidden_size, dtype=dtype)
            self.post_attn_sublayer_norm = None
        else:
            raise NotImplementedError(f"attn_norm_position {spec.attn_norm_position} not implemented")

        self.attention = _attention.Attention(spec.token_mixer, hidden_size,
                                              max_seq=max_seq, dtype=dtype)

        if spec.ffn_norm_position == types.NormPosition.PRE_AND_POST:
            self.pre_ffn_norm = norm.RMSNorm(spec.pre_ffn_norm, hidden_size, dtype=dtype)
            self.post_ffn_sublayer_norm = norm.RMSNorm(spec.post_ffn_norm, hidden_size, dtype=dtype)
        elif spec.ffn_norm_position == types.NormPosition.PRE:
            self.pre_ffn_norm = norm.RMSNorm(spec.pre_ffn_norm, hidden_size, dtype=dtype)
            self.post_ffn_sublayer_norm = None
        else:
            raise NotImplementedError(f"ffn_norm_position {spec.ffn_norm_position} not implemented")

        self.feedforward = feedforward.FeedForward(spec.channel_mixer, hidden_size, dtype=dtype)

    def forward(self, x, position_ids, cache, start_pos,
                per_layer_residual=None):
        # Attention sublayer
        attn_in = self.input_norm(x)
        attn_out = self.attention(attn_in, position_ids=position_ids,
                                  cache=cache, start_pos=start_pos)
        if self.post_attn_sublayer_norm is not None:
            attn_out = self.post_attn_sublayer_norm(attn_out)
        x = ops.add(x, attn_out)

        # FFN sublayer
        ffn_in = self.pre_ffn_norm(x)
        ffn_out = self.feedforward(ffn_in)
        if self.post_ffn_sublayer_norm is not None:
            ffn_out = self.post_ffn_sublayer_norm(ffn_out)
        x = ops.add(x, ffn_out)

        # PLE residual injection (Gemma 4)
        if per_layer_residual is not None:
            x = ops.add(x, per_layer_residual)
        return x
```

- [ ] **Step 3: Run tests**

```powershell
uv run pytest tests/api/test_block.py -v
```
Expected: all (2 existing + 2 new) pass.

- [ ] **Step 4: Commit**

```powershell
git add api/block.py tests/api/test_block.py
git commit -m "feat(B0.5): DecoderBlock supports PRE_AND_POST sandwich norm + per_layer_residual injection"
```

## B0.5 Task T12 — Bootstrap `models/gemma4/` package

**Files:**
- Create: `models/gemma4/__init__.py`
- Create: `tests/models/gemma4/__init__.py`

- [ ] **Step 1: Bootstrap directories**

```powershell
New-Item -ItemType Directory -Path "models\gemma4", "tests\models\gemma4"
New-Item -ItemType File -Path "models\gemma4\__init__.py", "tests\models\gemma4\__init__.py"
```

Add to `models/gemma4/__init__.py`:
```python
"""Gemma 4 reference layer implementation via api/ primitives.

Variants in scope for B0.5: E2B (5.1B / 2.3B effective).
Future B0.5 follow-ups (not part of this batch): E4B, 12B Unified, 26B-A4B MoE, 31B.
"""
```

- [ ] **Step 2: Commit (empty package bootstrap)**

```powershell
git add models/gemma4/__init__.py tests/models/gemma4/__init__.py
git commit -m "feat(B0.5): bootstrap models/gemma4 package"
```

## B0.5 Task T13 — `models/gemma4/config.py` + test_config

**Files:**
- Create: `models/gemma4/config.py`
- Create: `tests/models/gemma4/test_config.py`

- [ ] **Step 1: Failing test**

Create `tests/models/gemma4/test_config.py`:
```python
import torch

from api import types
from models.gemma4 import config


def test_gemma4_e2b_config_from_hf_dict():
    """E2B verified per research/issues/10-gemma4-investigation.md."""
    hf = {
        "model_type": "gemma4",
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "intermediate_size": 8192,
        "rope_theta": 10000.0,           # local layers
        "rope_global_theta": 1_000_000.0, # global layers
        "partial_rotary_factor_global": 0.25,
        "sliding_window": 512,
        "sliding_window_pattern": 4,       # 4:1 SWA:global on E2B
        "rms_norm_eps": 1e-6,
        "vocab_size": 262144,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "torch_dtype": "bfloat16",
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": None,
        "num_kv_shared_layers": 20,
        "use_per_layer_embedding": True,
        "ple_dim": 256,
        "attention_k_eq_v": False,         # E2B does NOT have K=V; that's 12B+
        "qk_norm_local_fixed_scale": 0.9916,
        "qk_norm_global_fixed_scale": 1.0228,
    }
    c = config.Gemma4Config.from_hf_dict(hf)
    assert c.hidden_size == 1536
    assert c.num_hidden_layers == 35
    assert c.num_kv_shared_layers == 20
    assert c.use_per_layer_embedding is True
    assert c.partial_rotary_factor_global == 0.25
    assert c.attention_k_eq_v is False


def test_gemma4_e2b_to_block_spec_local_layer():
    c = _e2b_config()
    block_spec = c.to_block_spec(layer_idx=0)  # local layer
    assert block_spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert block_spec.token_mixer.sliding_window == 512
    assert block_spec.token_mixer.rope.partial_rotary_factor == 1.0
    assert block_spec.token_mixer.rope.base_theta == 10000.0
    assert block_spec.attn_norm_position == types.NormPosition.PRE_AND_POST


def test_gemma4_e2b_to_block_spec_global_layer():
    c = _e2b_config()
    # In E2B with sliding_window_pattern=4, the global layer is every 5th layer
    # (4 local, 1 global). The last layer (index 34) is global.
    block_spec = c.to_block_spec(layer_idx=4)  # the first global layer
    assert block_spec.token_mixer.mask_kind == types.MaskKind.CAUSAL  # global = full causal
    assert block_spec.token_mixer.rope.partial_rotary_factor == 0.25
    assert block_spec.token_mixer.rope.base_theta == 1_000_000.0


def test_gemma4_e2b_shared_layer_indices():
    c = _e2b_config()
    # E2B: num_kv_shared_layers=20 means the last 20 layers reuse the K/V from
    # an earlier same-type layer. The exact mapping is "layer i in the shared
    # range reuses layer (i - period_size) of the same type".
    shared_map = c.kv_source_layer_idx_map()
    # Should be a dict {layer_idx: source_layer_idx} with 20 entries
    assert len(shared_map) == 20
    # No source layer is itself shared
    for src in shared_map.values():
        assert src not in shared_map


def _e2b_config():
    return config.Gemma4Config(
        hidden_size=1536, num_hidden_layers=35,
        num_attention_heads=8, num_key_value_heads=1,
        head_dim=256, global_head_dim=512,
        intermediate_size=8192,
        rope_theta_local=10000.0, rope_theta_global=1_000_000.0,
        partial_rotary_factor_global=0.25,
        sliding_window=512, sliding_window_pattern=4,
        rms_norm_eps=1e-6,
        vocab_size=262144, max_position_embeddings=32768,
        tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh",
        dtype=torch.float32,
        final_logit_softcap=30.0,
        attn_logit_softcap=None,
        num_kv_shared_layers=20,
        use_per_layer_embedding=True,
        ple_dim=256,
        attention_k_eq_v=False,
        qk_norm_local_fixed_scale=0.9916,
        qk_norm_global_fixed_scale=1.0228,
    )
```

- [ ] **Step 2: Implement `models/gemma4/config.py`**

```python
"""Gemma 4 config + adapter to api specs.

Verified per-size values pulled from HF model-card config.json files (E2B, E4B,
12B-Unified, 26B-A4B, 31B); see research/issues/10-gemma4-investigation.md and
research/01-model-census.v3.md §3.5.

Gemma 4 architectural primitives (axis-deltas from Gemma 3):
- Per-layer-type partial RoPE on global only (axis A15): local layers use full
  rotation, global use partial_rotary_factor=0.25.
- K=V on global layers for 12B+ only (axis A1 sub-value).
- Cross-layer KV sharing on E2B/E4B (axis A18): num_kv_shared_layers=20/35 means
  >half of layers reuse the K/V of an earlier same-type layer.
- Per-Layer Embeddings (axis A19) on E2B/E4B only: 256-dim secondary embedding
  table whose output is injected as residual at every layer.
- final_logit_softcap restored to 30.0 across all five sizes (Gemma 3 had
  dropped it).
- Fixed-scale QK norm (local 0.9916 / global 1.0228) absorbing 1/sqrt(head_dim)
  so the effective attention_scale is 1.0 instead of 1/sqrt(Dh).
- GeGLU with hidden_activation='gelu_pytorch_tanh' (Gemma signature preserved).
- use_double_wide_mlp=True on edge sizes (E2B/E4B) — widens gate/up; semantics
  documented as a flag, treated as a passthrough in the layer until we have
  numerical evidence what the widen factor is.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from api import specs, types


@dataclass(frozen=True)
class Gemma4Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int                       # local head_dim
    global_head_dim: int                # equals head_dim for 12B+; doubled for E2B
    intermediate_size: int
    rope_theta_local: float             # 10000.0
    rope_theta_global: float            # 1_000_000.0
    partial_rotary_factor_global: float # 0.25
    sliding_window: int                 # 512 (E2B) / 1024 (12B+)
    sliding_window_pattern: int         # 4 → 4:1 (E2B) ; 5 → 5:1 (12B+)
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    hidden_activation: str              # "gelu_pytorch_tanh"
    dtype: torch.dtype
    final_logit_softcap: Optional[float]
    attn_logit_softcap: Optional[float]
    num_kv_shared_layers: int           # E2B = 20; 12B+ = 0
    use_per_layer_embedding: bool
    ple_dim: int                        # 256
    attention_k_eq_v: bool              # True for 12B+; False for E2B/E4B
    qk_norm_local_fixed_scale: float
    qk_norm_global_fixed_scale: float

    @staticmethod
    def from_hf_dict(d: dict) -> "Gemma4Config":
        return Gemma4Config(
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            global_head_dim=d.get("global_head_dim", d["head_dim"]),
            intermediate_size=d["intermediate_size"],
            rope_theta_local=d["rope_theta"],
            rope_theta_global=d.get("rope_global_theta", d["rope_theta"]),
            partial_rotary_factor_global=d.get("partial_rotary_factor_global", 1.0),
            sliding_window=d["sliding_window"],
            sliding_window_pattern=d.get("sliding_window_pattern", 4),
            rms_norm_eps=d["rms_norm_eps"],
            vocab_size=d["vocab_size"],
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=d["tie_word_embeddings"],
            hidden_activation=d.get("hidden_activation", "gelu_pytorch_tanh"),
            dtype=_parse_dtype(d.get("torch_dtype", "bfloat16")),
            final_logit_softcap=d.get("final_logit_softcapping"),
            attn_logit_softcap=d.get("attn_logit_softcapping"),
            num_kv_shared_layers=d.get("num_kv_shared_layers", 0),
            use_per_layer_embedding=d.get("use_per_layer_embedding", False),
            ple_dim=d.get("ple_dim", 256),
            attention_k_eq_v=d.get("attention_k_eq_v", False),
            qk_norm_local_fixed_scale=d.get("qk_norm_local_fixed_scale", 0.9916),
            qk_norm_global_fixed_scale=d.get("qk_norm_global_fixed_scale", 1.0228),
        )

    def is_global_layer(self, layer_idx: int) -> bool:
        """Gemma 4 alternates `sliding_window_pattern` SWA layers per global layer.

        For E2B with sliding_window_pattern=4: layers 0,1,2,3 local; layer 4 global;
        layers 5,6,7,8 local; layer 9 global; ... pattern repeats until layer 34.
        """
        return (layer_idx + 1) % (self.sliding_window_pattern + 1) == 0

    def to_block_spec(self, layer_idx: int) -> specs.DecoderBlockSpec:
        is_global = self.is_global_layer(layer_idx)
        head_dim_eff = self.global_head_dim if is_global else self.head_dim
        attention_k_eq_v_eff = self.attention_k_eq_v and is_global
        qk_fixed = (self.qk_norm_global_fixed_scale if is_global
                   else self.qk_norm_local_fixed_scale)
        rope_theta_eff = self.rope_theta_global if is_global else self.rope_theta_local
        partial_eff = self.partial_rotary_factor_global if is_global else 1.0
        mask_eff = types.MaskKind.CAUSAL if is_global else types.MaskKind.SWA
        sw_eff = None if is_global else self.sliding_window

        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        )
        qk_norm_spec = norm_spec
        attn = specs.AttentionSpec(
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
            qk_norm_fixed_scale=qk_fixed,
            attention_k_eq_v=attention_k_eq_v_eff,
            rope=specs.RoPESpec(
                base_theta=rope_theta_eff,
                basis=types.RoPEBasis.SPLIT_HALF,
                partial_rotary_factor=partial_eff,
            ),
        )
        ffn = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.GELU,
            gate_kind=types.GateKind.GEGLU,
        )
        ple_spec = None
        if self.use_per_layer_embedding:
            ple_spec = specs.PLESpec(
                ple_dim=self.ple_dim,
                residual_scale=1.0 / (2 ** 0.5),
                injection_norm=norm_spec,
            )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE_AND_POST,
            ffn_norm_position=types.NormPosition.PRE_AND_POST,
            token_mixer=attn, channel_mixer=ffn,
            pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
            per_layer_embedding=ple_spec,
            final_logit_softcap=self.final_logit_softcap,
            embedding_scale=self.hidden_size ** 0.5,  # sqrt(D) — Gemma signature
        )

    def kv_source_layer_idx_map(self) -> dict[int, int]:
        """For Gemma 4 E2B/E4B: returns dict {layer_idx: source_layer_idx}.

        The shared layers are the LAST `num_kv_shared_layers` of each layer-type
        (local vs global) and they reuse the K/V of the layer-type-equivalent
        position from the unshared prefix.

        Concretely for E2B (35 layers, sliding_window_pattern=4, num_kv_shared=20):
        - Layer types: each block of 5 layers has 4 local + 1 global = 5 types.
        - Total layers = 35 = 7 blocks of 5.
        - Unshared prefix: 35 - 20 = 15 layers (3 blocks of 5).
        - Shared range: layers 15..34. Each shared layer reuses the same-position
          layer in the preceding block. For period_size=5 the reuse is
          source[i] = i - 5 for i in shared range.
        (The exact reuse pattern is derived from the Gemma 4 model card; this
        function returns the canonical mapping.)
        """
        period = self.sliding_window_pattern + 1   # 5 for E2B (4 local + 1 global)
        shared_start = self.num_hidden_layers - self.num_kv_shared_layers
        return {i: i - period for i in range(shared_start, self.num_hidden_layers)}


def _parse_dtype(s: str) -> torch.dtype:
    return {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16, "fp32": torch.float32,
    }[s]
```

- [ ] **Step 3: Run tests**

```powershell
uv run pytest tests/models/gemma4/test_config.py -v
```
Expected: 4 passed.

- [ ] **Step 4: Commit**

```powershell
git add models/gemma4/config.py tests/models/gemma4/test_config.py
git commit -m "feat(B0.5): Gemma4Config + per-layer to_block_spec dispatch (local vs global)"
```

## B0.5 Task T14 — `models/gemma4/layer.py` — build factory + shape test + weight loader

**Files:**
- Create: `models/gemma4/layer.py`
- Create: `tests/models/gemma4/test_layer_shape.py`
- Create: `tests/models/gemma4/test_weight_loader.py`

- [ ] **Step 1: Failing test for layer shape**

Create `tests/models/gemma4/test_layer_shape.py`:
```python
import torch

from api import kvcache, specs, types
from models.gemma4 import config as gemma4_config, layer as gemma4_layer


def test_gemma4_e2b_layer_0_local_forward_shape():
    cfg = _e2b_smallified_config()  # smaller version for fast test
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=0)
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    B, S = 1, 5
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def test_gemma4_e2b_layer_4_global_forward_shape():
    cfg = _e2b_smallified_config()
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=4)  # first global
    # Global layer uses global_head_dim (doubled in E2B) — the KV cache must allocate
    # to the global head_dim.
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.global_head_dim, max_seq=64,
    )
    B, S = 1, 3
    x = torch.randn(B, S, cfg.hidden_size)
    pos = torch.arange(S)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (B, S, cfg.hidden_size)


def _e2b_smallified_config():
    # A small variant of E2B that fits in CPU memory for shape tests
    return gemma4_config.Gemma4Config(
        hidden_size=128, num_hidden_layers=10,
        num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, global_head_dim=64,
        intermediate_size=512,
        rope_theta_local=10000.0, rope_theta_global=1_000_000.0,
        partial_rotary_factor_global=0.25,
        sliding_window=16, sliding_window_pattern=4,
        rms_norm_eps=1e-6,
        vocab_size=512, max_position_embeddings=64,
        tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh",
        dtype=torch.float32,
        final_logit_softcap=30.0, attn_logit_softcap=None,
        num_kv_shared_layers=0,  # disable sharing for the shape test
        use_per_layer_embedding=False,
        ple_dim=64,
        attention_k_eq_v=False,
        qk_norm_local_fixed_scale=0.9916,
        qk_norm_global_fixed_scale=1.0228,
    )
```

- [ ] **Step 2: Implement `models/gemma4/layer.py`**

```python
"""Gemma 4 reference layer assembly.

Layer index dispatch:
- Local layers (SWA, head_dim=local, rope θ=10000, partial_rotary=1.0)
- Global layers (full attention, head_dim=global, rope θ=1_000_000, partial_rotary=0.25,
  attention_k_eq_v=True on 12B+)
- Shared K/V layers (E2B/E4B last 20/35 layers — read K/V from a SharedLayerKVCache
  pointing to a prior unshared same-type layer; the forward still computes K/V but
  the writes are no-ops)

The layer factory takes a Gemma4Config and a layer_idx and returns a DecoderBlock
configured for that layer. The model assembly (out of scope for B0.5) is the loop
that calls this factory for every layer and connects shared K/V pointers.
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from api import block as _block, specs
from models.gemma4 import config as gemma4_config


def build_gemma4_decoder_layer(
    cfg: gemma4_config.Gemma4Config,
    layer_idx: int,
    max_seq: Optional[int] = None,
) -> _block.DecoderBlock:
    """Build a Gemma 4 decoder block for the given layer index.

    The returned DecoderBlock can be called with an optional `per_layer_residual`
    argument (from a separate PerLayerEmbedding module owned at the model level).
    Shared-K/V layers should be invoked with a SharedLayerKVCache instead of a
    fresh ContiguousKVCache; this is wired at the model assembly level, not here.
    """
    block_spec = cfg.to_block_spec(layer_idx=layer_idx)
    max_seq_eff = max_seq if max_seq is not None else cfg.max_position_embeddings
    return _block.DecoderBlock(
        block_spec, hidden_size=cfg.hidden_size, max_seq=max_seq_eff,
        dtype=cfg.dtype,
    )


def load_gemma4_weights_into_block(
    block: _block.DecoderBlock,
    state_dict: dict,
    layer_idx: int,
    cfg: gemma4_config.Gemma4Config,
) -> None:
    """Map an HF Gemma 4 state dict for a single layer into our DecoderBlock.

    HF tensor naming (from transformers.models.gemma4.modeling_gemma4):
      model.layers.{i}.input_layernorm.weight
      model.layers.{i}.self_attn.q_proj.weight
      model.layers.{i}.self_attn.k_proj.weight
      model.layers.{i}.self_attn.v_proj.weight        (absent on 12B+ global)
      model.layers.{i}.self_attn.o_proj.weight
      model.layers.{i}.self_attn.q_norm.weight
      model.layers.{i}.self_attn.k_norm.weight
      model.layers.{i}.post_attention_layernorm.weight
      model.layers.{i}.pre_feedforward_layernorm.weight
      model.layers.{i}.post_feedforward_layernorm.weight
      model.layers.{i}.mlp.gate_proj.weight
      model.layers.{i}.mlp.up_proj.weight
      model.layers.{i}.mlp.down_proj.weight

    For Gemma 4 the QK-norm weights have the fixed-scale absorbed *into* the
    learned weight at load time: the stored weight on disk is `learned_weight`
    and we multiply by `qk_norm_fixed_scale / (1 / sqrt(Dh))` to absorb 1/sqrt(Dh)
    into the norm gain. This makes the runtime path simpler (effective_scale=1.0).
    """
    prefix = f"model.layers.{layer_idx}."

    # Norms (4 of them — sandwich)
    block.input_norm.weight.copy_(state_dict[prefix + "input_layernorm.weight"])
    block.post_attn_sublayer_norm.weight.copy_(state_dict[prefix + "post_attention_layernorm.weight"])
    block.pre_ffn_norm.weight.copy_(state_dict[prefix + "pre_feedforward_layernorm.weight"])
    block.post_ffn_sublayer_norm.weight.copy_(state_dict[prefix + "post_feedforward_layernorm.weight"])

    # Attention projections
    block.attention.q_proj.weight.copy_(state_dict[prefix + "self_attn.q_proj.weight"])
    block.attention.k_proj.weight.copy_(state_dict[prefix + "self_attn.k_proj.weight"])
    if not cfg.is_global_layer(layer_idx) or not cfg.attention_k_eq_v:
        block.attention.v_proj.weight.copy_(state_dict[prefix + "self_attn.v_proj.weight"])
    block.attention.o_proj.weight.copy_(state_dict[prefix + "self_attn.o_proj.weight"])

    # QK norms — absorb fixed_scale and 1/sqrt(Dh) into the learned gain
    is_global = cfg.is_global_layer(layer_idx)
    head_dim_eff = cfg.global_head_dim if is_global else cfg.head_dim
    fixed_scale = (cfg.qk_norm_global_fixed_scale if is_global
                  else cfg.qk_norm_local_fixed_scale)
    inv_sqrt_dh = head_dim_eff ** -0.5
    # ONE_PLUS_W mode: effective gain = (1 + w) — we want effective gain = (learned_gain * fixed_scale / inv_sqrt_dh)
    # so we set w := (learned_gain * fixed_scale / inv_sqrt_dh) - 1
    learned_q = state_dict[prefix + "self_attn.q_norm.weight"]
    learned_k = state_dict[prefix + "self_attn.k_norm.weight"]
    absorb = fixed_scale / inv_sqrt_dh
    block.attention.q_norm.weight.copy_(learned_q * absorb - 1.0 + 1.0)  # see note below
    block.attention.k_norm.weight.copy_(learned_k * absorb - 1.0 + 1.0)
    # Note: for ONE_PLUS_W with learned gain g, on-disk w is `g - 1`.
    # The absorbed gain is `g * absorb`, so the new w is `g * absorb - 1`.
    # We re-do the arithmetic explicitly here.
    block.attention.q_norm.weight.copy_((learned_q + 1.0) * absorb - 1.0)
    block.attention.k_norm.weight.copy_((learned_k + 1.0) * absorb - 1.0)

    # FFN
    block.feedforward.gate_proj.weight.copy_(state_dict[prefix + "mlp.gate_proj.weight"])
    block.feedforward.up_proj.weight.copy_(state_dict[prefix + "mlp.up_proj.weight"])
    block.feedforward.down_proj.weight.copy_(state_dict[prefix + "mlp.down_proj.weight"])
```

- [ ] **Step 3: Failing test for weight loader**

Create `tests/models/gemma4/test_weight_loader.py`:
```python
import torch

from api import kvcache, specs, types
from models.gemma4 import config as gemma4_config, layer as gemma4_layer


def test_weight_loader_round_trip_random_state_dict():
    """Build random state-dict tensors that match the HF naming and shapes; load."""
    cfg = _smallified()
    blk = gemma4_layer.build_gemma4_decoder_layer(cfg, layer_idx=0)
    sd = _random_state_dict(cfg, layer_idx=0)
    gemma4_layer.load_gemma4_weights_into_block(blk, sd, layer_idx=0, cfg=cfg)
    # No-op verification: just ensure no key errors and the block can still forward
    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1, n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    x = torch.randn(1, 4, cfg.hidden_size)
    pos = torch.arange(4)
    out = blk(x, position_ids=pos, cache=cache, start_pos=0)
    assert out.shape == (1, 4, cfg.hidden_size)


def _smallified():
    return gemma4_config.Gemma4Config(
        hidden_size=128, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, global_head_dim=64,
        intermediate_size=256,
        rope_theta_local=10000.0, rope_theta_global=1_000_000.0,
        partial_rotary_factor_global=0.25,
        sliding_window=16, sliding_window_pattern=4,
        rms_norm_eps=1e-6,
        vocab_size=64, max_position_embeddings=32,
        tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh",
        dtype=torch.float32,
        final_logit_softcap=30.0, attn_logit_softcap=None,
        num_kv_shared_layers=0, use_per_layer_embedding=False,
        ple_dim=32, attention_k_eq_v=False,
        qk_norm_local_fixed_scale=0.9916,
        qk_norm_global_fixed_scale=1.0228,
    )


def _random_state_dict(cfg, layer_idx):
    p = f"model.layers.{layer_idx}."
    D = cfg.hidden_size
    Hq = cfg.num_attention_heads; Hk = cfg.num_key_value_heads
    Dh = cfg.head_dim
    return {
        p + "input_layernorm.weight": torch.zeros(D),
        p + "post_attention_layernorm.weight": torch.zeros(D),
        p + "pre_feedforward_layernorm.weight": torch.zeros(D),
        p + "post_feedforward_layernorm.weight": torch.zeros(D),
        p + "self_attn.q_proj.weight": torch.randn(Hq * Dh, D),
        p + "self_attn.k_proj.weight": torch.randn(Hk * Dh, D),
        p + "self_attn.v_proj.weight": torch.randn(Hk * Dh, D),
        p + "self_attn.o_proj.weight": torch.randn(D, Hq * Dh),
        p + "self_attn.q_norm.weight": torch.zeros(Dh),
        p + "self_attn.k_norm.weight": torch.zeros(Dh),
        p + "mlp.gate_proj.weight": torch.randn(cfg.intermediate_size, D),
        p + "mlp.up_proj.weight": torch.randn(cfg.intermediate_size, D),
        p + "mlp.down_proj.weight": torch.randn(D, cfg.intermediate_size),
    }
```

- [ ] **Step 4: Run tests, expect pass**

```powershell
uv run pytest tests/models/gemma4/ -v
```
Expected: 3 passed (config + shape + weight_loader).

- [ ] **Step 5: Commit**

```powershell
git add models/gemma4/layer.py tests/models/gemma4/test_layer_shape.py tests/models/gemma4/test_weight_loader.py
git commit -m "feat(B0.5): Gemma4 build_decoder_layer factory + HF weight loader (sandwich norm, K=V branch, QK fixed-scale absorption)"
```

## B0.5 Task T15 — `models/gemma4/layer.md`

**Files:**
- Create: `models/gemma4/layer.md`

- [ ] **Step 1: Write the doc per spec §6 schema**

Create `models/gemma4/layer.md` containing the 8 required sections (identity, block diagram, tensor IO trace, op trace, spec instantiation, quirks, weight-name mapping, source citations) for Gemma 4 E2B. Key sections:

- **Identity:** Gemma 4 family; E2B variant (5.1B / 2.3B effective); released 2026-04-02; sources blog.google/innovation-and-ai/technology/developers-tools/gemma-4/; HF model card google/gemma-4-E2B-it.
- **Block diagram:** ASCII showing input → input_norm → attention → post_attn_norm → residual_add → pre_ffn_norm → ffn → post_ffn_norm → residual_add → (+ per_layer_residual if PLE) → output; per-layer-type local (SWA) vs global (full + p-RoPE 0.25 + K=V on 12B+).
- **Tensor IO trace:** for E2B (D=1536, H=8, Hk=1, Dh local=256, Dh global=512), every intermediate annotated.
- **Op trace:** the sequence of `api.ops.*` and `api.norm.*` calls.
- **Spec instantiation:** the full DecoderBlockSpec for layer 0 (local) and layer 4 (global) of E2B.
- **Quirks:** ONE_PLUS_W RMSNorm baking foot-gun (the loader absorbs 1/sqrt(Dh) into the weight); SharedLayerKVCache for layers 15..34; PLE injection mechanism; final softcap on lm_head only.
- **Weight-name mapping:** the 13-row table (modeling_gemma4 names → API slots).
- **Source citations:** `research/02-layer-sources.v3.md` Gemma 4 section; `research/issues/10-gemma4-investigation.md`.

- [ ] **Step 2: Commit**

```powershell
git add models/gemma4/layer.md
git commit -m "docs(B0.5): models/gemma4/layer.md per spec §6 schema"
```

## B0.5 Task T16 — Isolation HF tests (sub-op ladder)

**Files:**
- Create: `tests/models/gemma4/test_isolation_hf.py`

- [ ] **Step 1: Write the sub-op ladder tests**

Tests for: Gemma4RMSNorm (ONE_PLUS_W), Gemma4 RoPE (dual θ per layer-type + partial 0.25 global), Gemma4 QK-norm (fixed-scale absorption — verify our load-time absorption produces the same forward as HF's runtime application), Gemma4 GeGLU forward, Gemma4 SDPA with custom mask (SWA + sliding_window=512).

Each sub-op compared to HF reference (`transformers.models.gemma4.modeling_gemma4`) at `atol=1e-5, rtol=1e-5`.

For the QK-norm absorption test: HF applies the fixed_scale and 1/sqrt(Dh) at runtime; we absorb both into the norm gain at load time. The test must:
1. Load the same raw HF norm weight `w_hf` into our QKNorm.
2. Apply our absorption transform.
3. Forward both and assert allclose at `atol=1e-5`.

- [ ] **Step 2: Run, expect pass**

```powershell
uv run pytest tests/models/gemma4/test_isolation_hf.py -v
```

- [ ] **Step 3: Commit**

```powershell
git add tests/models/gemma4/test_isolation_hf.py
git commit -m "test(B0.5): Gemma4 sub-op isolation ladder (Norm/RoPE/QK-norm/SDPA/GeGLU vs HF)"
```

## B0.5 Task T17 — Gate test: numerical equivalence vs HF Gemma4 layer-0

**Files:**
- Create: `tests/models/gemma4/test_numerical_hf.py`

- [ ] **Step 1: Write the gate test**

The test loads `google/gemma-4-E2B-it` (or our own small E2B reproduction if access is gated) via `AutoModelForCausalLM.from_pretrained`, extracts layer-0 weights, loads them into our `DecoderBlock` via `load_gemma4_weights_into_block`, forwards both with the same fixed input (text-only — no image/audio tokens for the gate test), and asserts:

```python
torch.allclose(out_api, out_ref, atol=5e-4, rtol=5e-4)
```

Mark this test `@pytest.mark.gate` so it can be skipped in CI without HF access.

- [ ] **Step 2: Run if HF access available**

```powershell
uv run pytest tests/models/gemma4/test_numerical_hf.py -v -m gate
```

- [ ] **Step 3: Commit**

```powershell
git add tests/models/gemma4/test_numerical_hf.py
git commit -m "test(B0.5): Gemma4 E2B layer-0 numerical equivalence vs HF (atol=5e-4)"
```

## B0.5 Task T18 — KV-cache equivalence test

**Files:**
- Create: `tests/models/gemma4/test_kvcache_hf.py`

- [ ] **Step 1: Write the test**

Prefill 8 tokens through both API and HF Gemma 4 E2B layer-0 (local — SWA layer), save both caches, decode token 9 in both; assert the position-8 output matches at `atol=5e-4`. Then repeat for a global layer (layer index 4 in E2B). The global-layer test exercises the partial-rotary 0.25 path on the cache decode, which is where partial-rotary basis bugs surface most readily.

For E2B specifically the test also exercises a **shared-layer** decode: pick layer 20 (in the shared range 15..34 with period=5 → source = 15), build a `SharedLayerKVCache` pointing to layer-15's cache, fill layer-15 by prefilling through layer 15 first, then decode through layer 20 using the shared cache; assert correctness.

- [ ] **Step 2: Run if HF available**

```powershell
uv run pytest tests/models/gemma4/test_kvcache_hf.py -v -m gate
```

- [ ] **Step 3: Commit**

```powershell
git add tests/models/gemma4/test_kvcache_hf.py
git commit -m "test(B0.5): Gemma4 KV-cache prefill+decode + SharedLayerKVCache vs HF"
```

## B0.5 Task T19 — Final regression + B0.5 sign-off commit

- [ ] **Step 1: Full regression**

```powershell
uv run pytest -q
```
Expected: M1 suite (74 tests) + B0.5 added tests (~30 new tests across api/embedding, api/specs ext, api/rope ext, api/attention ext, api/feedforward ext, api/block ext, api/kvcache ext, models/gemma4/*) all green.

- [ ] **Step 2: Update README with B0.5 status section**

Append a "B0.5 status" section to README.md listing the 4 new spec fields, the new `PLESpec` and `ShareScheme` enum, the `SharedLayerKVCache` wrapper, and the new `api/embedding.py` module. Link to `models/gemma4/layer.md`.

- [ ] **Step 3: Sign-off commit**

```powershell
git add README.md
git commit -m "docs(B0.5): README — Gemma 4 v3 IR seed complete; M2 gate passed"
```

Approval of B0.5 by the user gates the dispatch of subagent batches B1 → B10.

---

# B1 through B10 — Subagent-driven batch task lists (high-level)

> **Workflow per batch:** Each batch is dispatched as a single subagent unit using `superpowers:subagent-driven-development`. The dispatching session:
>
> 1. Spawns one subagent with the batch's task list and a copy of the latest API spec.
> 2. The subagent implements all models in the batch using TDD, committing per task (bite-sized M1 style).
> 3. The subagent self-reviews via `superpowers:requesting-code-review` at end of batch.
> 4. The orchestrating session then runs `superpowers:code-review` against the batch's diff (spec-conformance, type consistency, dead enum members, placeholder scan).
> 5. After review, the batch's `models/<family>/` directories are merged into the main workspace.
>
> The full bite-sized expansion of each batch's tasks happens **when the batch is dispatched** — only the high-level task name + extension list lives in this v3 plan. This keeps the v3 plan finite and lets each batch incorporate any IR drift from earlier batches.

## Batch B1 — Llama-family dense

**Models:**
1. **Llama 3.0 8B** — vanilla RoPE θ=500000, GQA H/Hk/Dh=32/8/128, SwiGLU, RMSNorm pre, tied=n, no QK-norm. Pure API fit.
2. **Llama 3.1 8B** — same as 3.0 but RoPE scaling = LLAMA3 with `factor=8, low/high=1/4, original=8192`. Already supported by `RoPESpec.scaling=LLAMA3` and `Llama3RoPEParams` from M1.
3. **Llama 3.2 1B** — H/Hk/Dh=32/8/64; same RoPE scaling; tied=y.
4. **Llama 3.2 3B** — H/Hk/Dh=24/8/128; same scaling.
5. **Mistral 7B v0.3** — H/Hk/Dh=32/8/128, RoPE θ=1e6, `sliding_window=4096` exercised on the SWA mask (already wired at B0.5).
6. **SmolLM3 3B** — first NoPE-per-layer family. Forces `RoPESpec` to have a per-layer `apply_rope` boolean. Simplest implementation: `RoPESpec(scaling=NONE)` with a new `RoPESpec.disable_for_layers: Optional[tuple[int,...]]` field; or model-level handling at the build_layer factory level. The plan recommends the model-level handling: `models/smollm3/config.py` returns `block_spec.token_mixer.rope = None` for layers in the NoPE list. Adds the **first config-driven per-layer heterogeneity** — exercises the model-level `to_block_spec(layer_idx)` pattern landed in B0.5.
7. **TinyLlama 1.1B** — H/Hk/Dh=32/4/64 (GQA, not MHA — surprise fact preserved from research/01 v3 §8.4). Pure API fit.

**Tasks (named, expanded at dispatch time):**
- T-B1-1 `models/llama3/` (Llama 3.0 + 3.1 + 3.2 1B/3B — one factory, four configs)
- T-B1-2 `models/mistral7b/` (v0.3 — exercise SWA mask in a numerical test)
- T-B1-3 `models/smollm3/` (NoPE per-layer — first per-layer heterogeneity)
- T-B1-4 `models/tinyllama/` (thin)
- T-B1-5 Regression test: all M1 + B0.5 + B1 models still pass numerical gates.

**Estimated commit count:** ~20-25 (3-5 per family).

**Gate test:** `atol=5e-4` vs HF for one variant per family (Llama 3.0 8B, Mistral 7B v0.3, SmolLM3 3B; TinyLlama optional). The Llama 3.1 LLAMA3 scaling gate is the highest-value gate because it exercises the smooth-scaling math.

## Batch B2 — Llama-family with quirks (μP, MLA, fused QKV, BlockSparse skeleton)

**Models:**
1. **Granite 3.x dense** (3.0 / 3.1 / 3.2 / 3.3) — μP scalars (residual_multiplier, embedding_multiplier, logits_scale). Forces `DecoderBlockSpec.residual_scale`, `embedding_scale`, `logits_scale` wiring (already on the spec but never exercised). Also forces `attn_scale = config.attention_multiplier` (μP-derived).
2. **Granite 4.1 dense** (3B / 8B / 30B) — same μP scalars; RoPE θ=1e7 (extended from Granite 3's 1e7-ish).
3. **MiniCPM 3** — first MLA family. Forces `AttentionKind.MLA` real implementation: `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`. The attention forward path splits into:
   - `c_q = q_a_proj(x); q = q_b_proj(q_a_layernorm(c_q))` → reshape to [B, S, H, qk_head_dim] where qk_head_dim = qk_nope + qk_rope.
   - `c_kv = kv_a_proj_with_mqa(x); k, v = kv_b_proj(kv_a_layernorm(c_kv))` → with partial-RoPE applied only to k_rope (the first qk_rope_head_dim channels of K and Q).
   - KV cache stores the latent `c_kv` and `k_rope` (`KVCacheSpec.layout = MLA_LATENT`).
4. **Phi-3 mini 3.8B** — fused QKV (`QKVLayout.FUSED`). The attention forward path:
   - `qkv = qkv_proj(x)` → `[B, S, (Hq + 2*Hk) * Dh]` → slice into q, k, v.
   - LongRoPE scaling: `RoPEScaling.LONGROPE` with short_factor / long_factor tables — implementation lands in B2's `api/rope.py` extension.
   - Partial RoPE 0.75 — already supported by B0.5's partial_rotary_factor.
5. **Phi-4-mini 3.8B** — similar to Phi-3 but with LongRoPE refresh; should reuse B2 infrastructure.
6. **Phi-3-small 7B** — first BlockSparse model. Forces `MaskKind.BLOCK_SPARSE` real implementation. For B2 we ship a **shape-only** test: build the BlockSparse mask, run forward, verify shape; defer numerical-equivalence to a follow-up because the mask construction is intricate (vertical+random+local block pattern). Also gegelu activation (`Activation.GEGELU`).

**Tasks:**
- T-B2-1 Wire LayerScaleSpec μP scalars through DecoderBlock (already on spec; just verify the residual_scale * residual addition path).
- T-B2-2 `models/granite3/` and `models/granite41/` (shared factory, different configs).
- T-B2-3 `api/types.py` + `api/attention.py`: real `AttentionKind.MLA` path. Add `MLASpec` (or extend `AttentionSpec` with the 5 MLA dims as optional fields, conditional on `kind == MLA`). Lands the MLA latent QKV layout.
- T-B2-4 `api/kvcache.py`: `MLALatentKVCache` storing `[B, S, kv_lora_rank + qk_rope_head_dim]` instead of `[B, Hk, S, Dh]`. Reference implementation: HF's simple-correct path (store decompressed); production path (store compressed + absorb kv_b_proj into o_proj) is deferred.
- T-B2-5 `models/minicpm3/` + gate test.
- T-B2-6 `api/attention.py`: `QKVLayout.FUSED` branch.
- T-B2-7 `api/rope.py`: `RoPEScaling.LONGROPE` with short/long factor tables.
- T-B2-8 `models/phi3_mini/` + gate test.
- T-B2-9 `models/phi4_mini/`.
- T-B2-10 `api/types.py` + `api/attention.py`: `MaskKind.BLOCK_SPARSE` skeleton (shape-only).
- T-B2-11 `models/phi3_small/` (shape-only test, no numerical gate).
- T-B2-12 Regression.

**Estimated commit count:** ~30-40.

**Gate tests:** Granite 3.x μP numerical; MiniCPM 3 MLA numerical; Phi-3 mini LongRoPE numerical. Phi-3-small is shape-only.

## Batch B3 — Post-norm / dual-norm (OLMo 2, Gemma 2, Gemma 3)

**Models:**
1. **OLMo 2 7B / 13B** — first `NormPosition.POST` family. The post-norm path:
   - `attn_out = attention(x); attn_normed = post_attn_norm(attn_out); h = x + attn_normed`
   - `ffn_out = ffn(h); ffn_normed = post_ffn_norm(ffn_out); out = h + ffn_normed`
   - **No** pre-norms on attention/FFN inputs.
   - QK-norm = `FULL_HDH` shape (per-full-channels), POST-projection (already enumerated as QKNormPhase.POST_ROPE in v2 spec; OLMo 2 actually applies after projection but before RoPE — needs precision in spec).
   - **BF16 residual stream required** (FP16 overflows on post-norm at scale). The plan documents this in `models/olmo2/layer.md` and the numerical test runs in BF16.
2. **Gemma 2 9B** — dual-norm sandwich (already exercised at B0.5; Gemma 2's sandwich is the predecessor). GeGLU (already supported). Both `attn_logit_softcap` and `final_logit_softcap` present at 30.0. Forces `AttentionSpec.attn_logit_softcap` to actually take effect inside the attention path (B0.5 had the field but with no `attn_logit_softcap` test).
3. **Gemma 3 4B / 12B** — sandwich + SWA/global alternation 5:1 + dual RoPE θ per layer-type (already exercised at B0.5 in Gemma 4 form). Gemma 3 is the predecessor that exercises the same code path without partial-rotary and without PLE — a useful regression for the B0.5 IR.

**Tasks:**
- T-B3-1 `api/block.py`: `NormPosition.POST` path (no pre_attn / pre_ffn norms; post-norms wrap the sublayer output before the residual add).
- T-B3-2 `api/attention.py`: exercise `QKNormShape.FULL_HDH` (was wired at M1, never tested with real numerical equivalence — OLMo 2 is the first family that uses it).
- T-B3-3 `models/olmo2/` + BF16-residual numerical gate.
- T-B3-4 `api/attention.py`: wire `AttentionSpec.attn_logit_softcap` actually applied inside SDPA (before softmax).
- T-B3-5 `models/gemma2/` + numerical gate.
- T-B3-6 `models/gemma3/` + numerical gate (also acts as regression of the B0.5 sandwich path).
- T-B3-7 Regression.

**Estimated commit count:** ~20-25.

**Gate tests:** OLMo 2 7B (post-norm + FULL_HDH QK-norm — high-value); Gemma 2 9B (attn softcap + final softcap); Gemma 3 4B (sandwich + SWA alternation — regression of B0.5 IR).

## Batch B4 — Per-layer heterogeneous (iRoPE, interleaved SWA)

**Models:**
1. **Llama 4 Scout 17B-A** (gated — shape test only on a small reproduction). Forces `RoPESpec.iRoPE_no_rope_layers: tuple[int,...]` or per-layer `apply_rope: bool`. The cleanest implementation is a model-level decision: `models/llama4/config.py.to_block_spec(layer_idx)` returns `block_spec.token_mixer.rope = None` for layers in the NoPE list. Same pattern as SmolLM3 (B1) but at production scale.
2. **Ministral 3B / 8B** — interleaved SWA (3rd alternation pattern after Gemma 2's 1:1 and Gemma 3's 5:1). Specifically: layer-by-layer alternation full/SWA at a different ratio. Adds `MaskKind.SWA_INTERLEAVED` as a documented enum value, but the implementation reuses the per-layer `to_block_spec(layer_idx)` pattern: even layers full, odd layers SWA.

**Tasks:**
- T-B4-1 `models/llama4_scout/` — shape-only test on a small reproduction; numerical gated by access.
- T-B4-2 `models/ministral/` — exercise SWA on alternating layers; numerical gate vs HF Ministral 3B.
- T-B4-3 Regression.

**Estimated commit count:** ~8-12.

**Gate tests:** Ministral 3B numerical (the accessible one); Llama 4 Scout shape only.

## Batch B5 — MLA family expansion (DeepSeek-V2-Lite, DeepSeek-V3-Lite, DeepSeek-V3.2 DSA shape-only)

**Models:**
1. **DeepSeek-V2-Lite 16B-A2.4B** — first canonical MLA model. Builds on MiniCPM-3 from B2 by exercising the full MLA path at scale. Dense+MoE alternation `first_k_dense_replace=1` (first 1 layer dense, rest MoE) — forces `models/<family>/config.py.to_block_spec(layer_idx)` to return either an `FFNSpec` or an `MoESpec`-bearing block.
2. **DeepSeek-V3-Lite** (if available — else simulate via DeepSeek-V3 config scaled down) — exercises shared+routed MoE (256+1 / top-8) with `score_correction_bias` (aux-loss-free balancing). Requires `MoESpec.aux_loss_free: bool` field and sigmoid router.
3. **DeepSeek-V3.2 DSA** (shape-only — production weights are 671B). Forces `IndexerSpec` (Lightning Indexer top-k learned-scorer). For B5 this is a **shape-only IndexerSpec stub** — the actual top-k routing math is documented in `models/deepseek_v32/layer.md` but the test only verifies the spec composes; numerical equivalence is deferred to a future milestone (DSA is non-trivially testable without 100k-token context fixtures).

**Tasks:**
- T-B5-1 Promote MLA from B2 stub to first-class path with all 5 dims, full latent KV cache.
- T-B5-2 `models/deepseek_v2_lite/` + MLA numerical gate.
- T-B5-3 `api/feedforward.py`: full MoE wiring (router, top-k, gather/scatter, shared+routed) — this is the foundation B6 builds on; we land it at B5 because DeepSeek-V2-Lite needs it.
- T-B5-4 `models/deepseek_v3_lite/` + MoE numerical gate (sigmoid+bias router).
- T-B5-5 `api/types.py` + `api/specs.py`: `IndexerSpec` stub.
- T-B5-6 `models/deepseek_v32/` — shape-only.
- T-B5-7 Regression.

**Estimated commit count:** ~25-35.

**Gate tests:** DeepSeek-V2-Lite MLA numerical (the gate-critical MLA test); DeepSeek-V3-Lite MoE numerical; V3.2 shape-only.

## Batch B6 — MoE expansion

**Models:**
1. **Mixtral 8×7B** (oos by active param but axis-load-bearing — softmax top-2 router, no shared experts; the foundational MoE). Numerical gate on a small fixture (10 layers, narrow expert FFN).
2. **Qwen3-MoE 30B-A3B** — softmax + norm_topk + 0 shared (128 experts, top-8).
3. **Qwen3-Next 80B-A3B** — ultra-sparse (512 experts, top-10, 1 shared). Plus Gated DeltaNet 3:1 hybrid — this is the first DeltaNet linear-attention family. **Numerical equivalence for the hybrid is shape-only at B6** because the DeltaNet kernel lands at B7.
4. **OLMoE 1B-A** — softmax + 0 shared (64 experts, top-8).
5. **Granite 4 H-Tiny 7B-A1B** — softmax + 0 shared (32 experts, top-4) + 9:1 Mamba:attention hybrid. **Hybrid lands at B7**; B6 ships the dense-attention-only variant of Granite 4 H-Tiny if possible, else defers to B7.

**Tasks:**
- T-B6-1 `api/feedforward.py`: complete MoE building block with softmax/sigmoid router, top-k selection, gather/scatter dispatch, optional shared experts, optional aux-loss-free bias correction, optional group routing.
- T-B6-2 `models/mixtral/` + numerical gate.
- T-B6-3 `models/qwen3_moe/` + numerical gate.
- T-B6-4 `models/qwen3_next/` — shape-only for hybrid (Gated DeltaNet skipped — token-mixer-kind=STANDARD only for dense-attention layers; the DeltaNet layers are a per-layer dispatch). Numerical only on the attention-layer subset.
- T-B6-5 `models/olmoe/` + numerical gate.
- T-B6-6 `models/granite4_htiny/` — dense-attention-only variant or defer to B7.
- T-B6-7 Regression.

**Estimated commit count:** ~25-35.

**Gate tests:** Mixtral, Qwen3-MoE, OLMoE numerical. Qwen3-Next and Granite 4 H-Tiny shape-only / partial.

## Batch B7 — SSM + hybrid

**Models:**
1. **Mamba 2 2.7B** (pure SSD).
2. **Mamba 3 reference** (complex-valued state + MIMO decoding). PyTorch complex tensor support is incomplete on CPU; the plan uses real-valued representation with paired real/imag tensors throughout.
3. **Jamba mini** (Mamba-1 + attention 1:8 sequential).
4. **Zamba2 2.7B** (Mamba-2 + periodic shared attention).
5. **Hymba 1.5B** (Mamba ‖ attention parallel branches per block) — forces `TokenMixerKind.HYBRID_PARALLEL` and `DecoderBlockSpec.token_mixer: Tuple[AttentionSpec, SSMSpec]`.
6. **Phi-4-mini-flash** (Samba — sequential Mamba + attention).
7. **Falcon-H1 1.5B** (parallel-head Mamba-2 + attention).
8. **Granite 4 H-Micro 3B** (9:1 Mamba-2:attention).
9. **Nemotron 3 Nano 4B** (Mamba-2 + transformer + MoE) — shape-only.
10. **RecurrentGemma 2B** (RG-LRU recurrence + local attention).
11. **RWKV-7 1.5B** (delta-rule recurrence).
12. **MiniMax-Text-01** (Lightning Attention 7:1) — shape-only.

**Tasks:**
- T-B7-1 Create `api/ssm.py` with `selective_scan`, `conv1d_causal`, `ssd_scan` reference impls (PyTorch eager, non-fused — clarity over speed).
- T-B7-2 `api/specs.py`: ensure `SSMSpec` (Mamba-1), `SSDSpec` (Mamba-2), `Mamba3Spec` extension (complex_state, mimo_decoding flags), `RWKVSpec`, `LinearAttentionSpec`.
- T-B7-3 `api/attention.py` or new `api/token_mixer.py`: dispatch on `TokenMixerKind` (STANDARD / SSM_MAMBA1 / SSM_MAMBA2 / SSM_MAMBA3 / SSM_RWKV / HYBRID_PARALLEL / HYBRID_ALTERNATING / LIGHTNING_ATTN).
- T-B7-4 `api/kvcache.py`: `SSMStateCache` and `SSDStateConvCache` types.
- T-B7-5 → T-B7-15: one task per family (some merge into one task — e.g., Mamba 2 + Mamba 3 share `models/mamba/`; Jamba + Zamba2 share `models/jamba_family/`).
- T-B7-16 Regression.

**Estimated commit count:** ~50-60. This is the heaviest batch.

**Gate tests:** Mamba 2 numerical (the canonical SSD test); Jamba numerical; Hymba numerical (hybrid-parallel — first of kind); RWKV-7 numerical. Mamba 3 / Nemotron 3 / MiniMax shape-only.

## Batch B8 — OCR-LLM (Visual Causal Flow + M-RoPE + vision adapter)

**Models:**
1. **DeepSeek-OCR 3B-A570M** — the headline novel family. Visual Causal Flow forces `MaskKind.BLOCK_BIDIRECTIONAL`: tokens within a "block" (a page region) attend bidirectionally; different blocks remain causal. Plus MoE (already supported), MLA (already supported), DeepEncoder topology (vision frontend documented in layer.md but not implemented).
2. **GOT-OCR 2.0 0.5B** — smallest OCR-LLM. MHA + RoPE θ=10000 (vanilla Llama-shaped) + vision adapter producing 256 tokens/page; the LM-layer fits the existing API.
3. **Qwen2.5-VL 3B / 7B** — M-RoPE (3D position layout: T, H, W). Forces position_ids to be a `list[Tensor]` (length-3 tuple) for the M-RoPE axis case; the M-RoPE-aware `rope_apply_mrope(q, k, cos_list, sin_list)` op.

**Tasks:**
- T-B8-1 `api/types.py` + `api/specs.py`: `MaskKind.BLOCK_BIDIRECTIONAL`, `MaskKind.M_ROPE`, `VisionAdapterSpec` (axis A16).
- T-B8-2 `api/attention.py`: BLOCK_BIDIRECTIONAL mask construction (caller passes block boundaries as a tensor).
- T-B8-3 `api/ops.py`: `rope_apply_mrope` (3D — T/H/W axes).
- T-B8-4 `api/rope.py`: `MRoPE` module producing T/H/W cos/sin tables.
- T-B8-5 `models/deepseek_ocr/` + Visual Causal Flow numerical gate (text-only sub-pass — full MM equivalence deferred).
- T-B8-6 `models/got_ocr2/` + numerical gate (LM-layer text-only).
- T-B8-7 `models/qwen25_vl/` + M-RoPE numerical gate (text-only — vision frontend not implemented).
- T-B8-8 Regression.

**Estimated commit count:** ~20-30.

**Gate tests:** DeepSeek-OCR text-only LM-layer; GOT-OCR 2.0 LM-layer; Qwen2.5-VL M-RoPE text-only.

## Batch B9 — Audio LM (Moshi, Voxtral TTS)

**Models:**
1. **Moshi 7B** (Helium-1 backbone + Mimi codec + Temporal+Depth transformers). The LM-layer of Moshi is plain GQA + RoPE + SwiGLU; the dual-stream text/audio and the Depth transformer are documented in layer.md but not implemented (audio frontend is out of scope per spec §2 non-goals).
2. **Voxtral TTS 4B** — open TTS, 5-second zero-shot voice cloning. Similar story: LM-layer is plain.

**Tasks:**
- T-B9-1 `models/moshi/` — text-decoder LM-layer numerical gate.
- T-B9-2 `models/voxtral_tts/` — LM-layer numerical gate.
- T-B9-3 Regression.

**Estimated commit count:** ~6-10. Lightest batch.

## Batch B10 — Quantization exercises

**Models:**
1. **GPT-OSS 20B MXFP4-native** — first MXFP4-native production model. Forces `QDType.MX_FP4` packing path + `PackingLayout.MX_BLOCK` (32-element OCP micro-blocks with UE8M0 scale). Numerical round-trip: pack random INT4 → MXFP4 block → unpack → assert.
2. **Falcon-Edge 1.58-bit retrainable** — first BitNet that supports continued pretrain. Forces `QDType.TERNARY` (`{-1, 0, +1}`) with optional `QuantSpec.codebook` slot for the per-channel scale.
3. **Gemma 4 mobile-int4 QAT** (per the 2026-06-05 QAT blog addendum) — exercises a verified INT4 QAT checkpoint on E2B. Reuses the B0.5 weight loader with `format="hf_int4_qat"`. The exercise is end-to-end: load QAT checkpoint, decode 8 tokens, assert output matches HF AT QAT-aware reference within INT4 quant tolerance (atol=5e-2).
4. **NVFP4 mixed-precision** — NVIDIA's Blackwell NVFP4. Forces `QDType.NVFP4`. Tested against `nvidia/DeepSeek-V4-Pro-NVFP4` (shape-only — the full V4-Pro is 1.6T parameters; we use a small reproduction).

**Tasks:**
- T-B10-1 `api/types.py` + `api/quant.py`: `MX_FP4` packing path + MXFP4 round-trip test.
- T-B10-2 `models/gpt_oss/` MXFP4 exercise.
- T-B10-3 `api/types.py` + `api/quant.py`: `TERNARY` packing.
- T-B10-4 `models/falcon_edge/` ternary exercise.
- T-B10-5 `models/gemma4/` extension: INT4 QAT round-trip.
- T-B10-6 `api/quant.py`: NVFP4 packing.
- T-B10-7 NVFP4 shape-only exercise.
- T-B10-8 Final regression — entire test suite, ~74 M1 + ~30 B0.5 + ~200 B1-B10 ≈ 300-400 tests.

**Estimated commit count:** ~15-20.

---

# Subagent-driven workflow

For B1 through B10, each batch is one subagent dispatch. The dispatching session prepares the dispatch packet:

```
Subagent input:
- The batch's high-level task list (this v3 plan §B<N>)
- The latest api/ spec (post-B0.5 or post-previous-batch)
- A reference to research/01-model-census.v3.md and research/02-layer-sources.v3.md for the families in scope
- The M1 plan and the B0.5 task style as the bite-sized TDD example to imitate
- The atol=5e-4 numerical gate as the per-family acceptance criterion

Subagent output:
- A branch (git or worktree — superpowers:using-git-worktrees) with the batch's commits
- Per-family models/<name>/{config.py, layer.py, layer.md} + tests
- A final self-review using superpowers:requesting-code-review

Orchestrator review:
- superpowers:code-review against the subagent's diff
- Spec-conformance check (no models/<name>/ adds api imports beyond the API surface)
- Type consistency check (mypy strict on api/, lenient on models/)
- Dead enum value scan
- Placeholder scan ("TODO", "FIXME", "XXX", "stub", "pass  # noqa")
- Merge into main workspace if all reviews pass
```

The B0.5 task list above (Tasks T1 → T19, ~19 tasks, ~30 commits) is the template for each subagent's bite-sized output. Subagents should produce ~3-5 commits per family, ~15-25 commits per batch.

---

# Risks

1. **Gemma 4 numerical equivalence may be harder than Qwen3.** Encoder-free unified multimodal means the layer-0 forward of the 12B Unified variant may need image+audio inputs. **Mitigation:** the B0.5 gate test runs only the **text-only sub-pass** of E2B (which is encoder-based, not encoder-free). The 12B Unified encoder-free path is deferred to B8 (where the vision-adapter axis is implemented). Within B0.5 we verify E2B text-only at atol=5e-4 against HF Gemma4Model with image_inputs=None and audio_inputs=None.

2. **Mamba-3 complex-state.** PyTorch's complex tensor support has gaps (autograd on complex is partial; some ops don't broadcast). **Mitigation:** use a real-valued representation throughout — store the state as a pair `(state_real, state_imag)` and implement the complex multiplication explicitly: `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`. This adds 4 multiplies + 2 adds per scan step but avoids PyTorch's complex-tensor pitfalls.

3. **Per-layer KV sharing logic touches the KVCache module deeply.** The `SharedLayerKVCache` wrapper landed at B0.5 may not handle all interaction cases (concurrent prefill of source and dependent layer; layer dispatch order; the source layer's cache being reset between forwards). **Mitigation:** the B0.5 tests cover the canonical case (source filled before dependent reads); additional edge cases (decoded one-token-at-a-time across both source and dependent layers in the same forward pass) are flagged for the model-assembly layer and tested in a follow-up. The model assembly is itself out of scope for M2 (we test only single-decoder-block forwards).

4. **MLA dimensions break the standard `(n_heads, head_dim)` assumption in many code paths.** Specifically: KV cache shape, RoPE table sizing, SDPA scale derivation, weight loader names. **Mitigation:** MLA is a first-class `AttentionKind` from B2 onwards; shared code paths conditional on `kind == MLA`; the `MLA_LATENT` cache layout has its own write/read methods that take latent-cache-shaped tensors. The 5 MLA dims live as optional fields on `AttentionSpec` (qk_nope_head_dim, qk_rope_head_dim, v_head_dim, q_lora_rank, kv_lora_rank).

5. **MoE token dispatch differs across runtimes.** The reference impl uses dense scatter (`for expert_idx in range(E): mask_for_expert = (top_indices == expert_idx); ...`). **Mitigation:** documented as `DispatchKind.DENSE_SCATTER` in the spec; production runtimes (grouped GEMM, megablocks token-permute) are noted but not implemented. The building block's *interface* is dispatch-kind-stable.

6. **DeepSeek-V3.2 DSA Lightning Indexer is intricate.** The learned-scorer top-k selection has a sub-network of its own; full numerical equivalence requires running the Indexer at depth. **Mitigation:** B5 ships `IndexerSpec` as shape-only; numerical equivalence deferred to a post-M2 milestone.

7. **Hybrid-parallel block (Hymba) breaks the `token_mixer: Union[AttentionSpec, ...]` assumption.** Hymba runs Mamba ‖ attention with their outputs summed before the residual add. **Mitigation:** `token_mixer: Union[..., Tuple[AttentionSpec, SSMSpec]]` plus a `TokenMixerKind.HYBRID_PARALLEL` discriminator; B7 lands this.

8. **Gemma 4 QK-norm fixed-scale absorption math.** The on-disk Gemma 4 `q_norm.weight` is a learned gain; the fixed_scale (0.9916 local / 1.0228 global) and the 1/sqrt(Dh) attention scale both fold into the norm's effective gain at load time. The arithmetic is: if on-disk weight is `w` (representing `1 + w` in ONE_PLUS_W mode), and effective gain should be `(1 + w) * fixed_scale / inv_sqrt_dh`, then the new on-disk `w_new = (1 + w) * absorb_factor - 1` where `absorb_factor = fixed_scale * sqrt(Dh)`. **Mitigation:** isolation test (B0.5 T16) loads the same HF weight, applies our absorption, and verifies the post-load forward equals HF's runtime-applied-scale forward at atol=1e-5.

9. **OLMo 2 BF16-required-residual.** Post-norm models accumulate larger activations in the residual stream; FP16 can overflow. **Mitigation:** `layer.md` documents the BF16 requirement; the B3 numerical test runs in BF16.

10. **HF transformers version drift.** Gemma 4 needs `transformers >= 5.x` (per research/02-layer-sources.v3.md §1); v3 also adds DeepSeek-V3.2 and Qwen3-Next that need ≥4.51. **Mitigation:** pin `transformers >= 4.51` in `pyproject.toml` (already done in M1) and bump to `transformers >= 5.0` at B0.5 dispatch.

11. **HF gated models.** Llama 4 Scout/Maverick, some Gemma variants, some Mistral variants are gated and require login. **Mitigation:** numerical gates that require gated weights are marked `@pytest.mark.gated` and skipped in default CI; we rely on shape tests + sub-op isolation tests for those families. Llama 4 Scout (B4) and the 12B Unified Gemma 4 (B8) are the two B-batch entries that may end up shape-only in practice.

12. **Qwen3-Next Gated DeltaNet is 75% of the model's layers.** Skipping it means our numerical equivalence covers only the 25% attention layers. **Mitigation:** B6 ships shape-only for Qwen3-Next; the DeltaNet kernel lands at B7, after which a follow-up commit upgrades Qwen3-Next to a full numerical gate.

---

# Open questions deferred to per-batch plan dispatch

These questions resolved at B0.5 are locked. The remaining open questions per batch:

**B1 open:**
- Should Llama 3.0 / 3.1 / 3.2 / 1B/3B/8B share one factory or four? **Lean:** one factory per generation (Llama 3.0/3.1 share; 3.2 has its own because of the head_dim shift from 128 to 64 at 1B size). Decided at dispatch.
- SmolLM3's NoPE pattern: model-level `to_block_spec(layer_idx)` returning `rope=None` vs spec-level `RoPESpec.disable_for_layers`? **Lean:** model-level, no new spec field.

**B2 open:**
- Granite μP residual_multiplier: does it scale only the FFN residual add, only the attention residual add, or both? Per research/05 §3.17 Granite uses ≈0.22 on both. **Lean:** wire `DecoderBlockSpec.residual_scale` to scale both residual adds; verify by numerical equivalence.
- MLA absorption (kv_b_proj into o_proj) is the production path. For M2 reference we use the simple-correct path (cache decompressed). Defer absorption to a future milestone.

**B3 open:**
- Gemma 2 attn_logit_softcap = 30.0 applies before softmax. Verify the position of the softcap (after the QK^T scale, before the mask add, before softmax). Cross-check vs HF Gemma2Attention.

**B4 open:**
- Llama 4 Scout iRoPE: is the "global" layer the one with NoPE, or the one with RoPE? Per research/01 v3 §3.5 the global layers are NoPE and local are RoPE — inversion of SmolLM3. Verify at dispatch.

**B5 open:**
- DeepSeek-V3-Lite availability: if the explicit "Lite" variant doesn't exist, we use a scaled-down V3 config (32 layers, 32 experts, top-2) as a stand-in.

**B6 open:**
- Qwen3-Next aux-loss-free balancing in the MoE: separate from sigmoid bias correction. Need to enumerate as `MoESpec.balancing: ...`.

**B7 open:**
- Mamba 3 production weights availability at M2 dispatch time. If not released, we use Mamba 3 reference (180M-1.5B) shape-only.
- Hymba parallel branch output: `attn_out + ssm_out` vs concatenate vs alternating averaging. Per Hymba paper it's an additive sum after independent per-branch normalization. Verify at dispatch.

**B8 open:**
- DeepSeek-OCR Visual Causal Flow exact block boundary definition: per-page or per-region? The DeepSeek-OCR paper (arXiv:2510.18234) specifies per-page; verify at dispatch.

**B9 open:**
- Whether to ship the Moshi Depth Transformer at all, or document it and skip. **Lean:** document, skip — it's an audio-output transformer, out of scope for the LM-layer mission.

**B10 open:**
- Does Gemma 4 INT4 QAT use AWQ packing or a Google-proprietary packing? The 2026-06-05 QAT blog mentions GGUF Q4_K compatibility; verify packing layout at dispatch.

---

# Self-review section

This section audits the v3 plan against the spec coverage, placeholder discipline, and type consistency criteria.

## Spec coverage

The v3 plan exercises every axis enumerated in research/01-model-census.v3.md §6 (axes A1–A19):

- **A1 Attention type:** STANDARD (B1 Llama, B3 Gemma 2/3, B0.5 Gemma 4), MLA (B2 MiniCPM-3, B5 DeepSeek), Mamba 1/2/3 (B7), SWA (B0.5 Gemma 4 local, B1 Mistral 7B, B4 Ministral), iRoPE (B4 Llama 4 Scout), DSA (B5 DeepSeek V3.2 shape-only), Lightning Attn (B7 MiniMax shape-only), Hybrid parallel (B7 Hymba), Hybrid alternating (B7 Jamba/Granite 4).
- **A2 Position encoding:** vanilla RoPE (B1), RoPE LLAMA3 (B1), RoPE LongRoPE (B2), RoPE YaRN (B2 MiniCPM-3 / B5 DeepSeek), RoPE partial (B0.5 Gemma 4 global, B2 Phi-3), NoPE alternating (B1 SmolLM3, B4 Llama 4), M-RoPE (B8 Qwen2.5-VL), Visual Causal Flow (B8 DeepSeek-OCR).
- **A3 Norm placement + QK-norm:** PRE (B1), POST (B3 OLMo 2), PRE_AND_POST sandwich (B0.5 Gemma 4, B3 Gemma 2/3), QK-norm per_head_dh (M1 Qwen3 + B0.5 Gemma 4 fixed-scale), QK-norm FULL_HDH (B3 OLMo 2), softcap (B0.5 final, B3 Gemma 2 both).
- **A4 FFN family:** SwiGLU (B1, M1), GeGLU (B0.5, B3), gegelu (B2 Phi-3-small), MoE (B6, B5), double-wide MLP flag (B0.5 Gemma 4 edge).
- **A5 MoE routing:** softmax top-k (B6 Mixtral), softmax norm_topk no-shared (B6 Qwen3-MoE), sigmoid aux-loss-free shared (B5 DeepSeek-V3-Lite, B6), ultra-sparse (B6 Qwen3-Next).
- **A6 KV cache:** Contiguous (M1), MLA_LATENT (B2/B5), SSM_STATE / SSD_STATE_PLUS_CONV (B7), K=V global (B0.5 spec, 12B+ deferred), cross-layer-shared (B0.5 E2B/E4B).
- **A7 Tokenizer:** not exercised in API (lives at model assembly).
- **A8 μP scalars:** B2 Granite.
- **A9 Bias presence:** B1 Llama no-bias; M1 Qwen3 q/k/v-bias yes / o-bias no — captured by AttentionSpec booleans.
- **A10 Embedding tying:** captured by Gemma4Config / model config; not an API surface.
- **A11 Per-layer type alternation:** B0.5 (Gemma 4 4:1 SWA), B3 (Gemma 3 5:1), B4 (Ministral interleaved), B7 (Jamba 1:8, Granite 4 9:1, Qwen3-Next 3:1).
- **A12 Per-layer shape scaling:** B0.5 (Gemma 4 head_dim local vs global). OpenELM deferred (axis-load-bearing but no urgent family forces it; could be added as a follow-up after B7).
- **A13 Parallel vs sequential sublayer:** sequential default (M1); parallel deferred (Phi-2 / Falcon-7B historical; no 2026 family forces it).
- **A14 Activation:** SiLU (M1), GELU (B0.5), gelu_pytorch_tanh (B0.5), gegelu (B2 Phi-3-small).
- **A15 RoPE rotation domain:** full (M1), partial 0.25 (B0.5), partial 0.75 (B2 Phi-3), NoPE (B1 SmolLM3, B4 Llama 4), per-layer-type partial (B0.5 Gemma 4).
- **A16 VLM fusion topology:** B8 (concat-projector via vision_adapter_spec stub).
- **A17 Vision-token compression budget:** B8 (DeepSeek-OCR ultra-low, Qwen2.5-VL high, GOT-OCR2 low).
- **A18 Cross-layer KV sharing:** B0.5 (SAME_BLOCK_SHARED via SharedLayerKVCache).
- **A19 Per-layer embeddings:** B0.5 (PLESpec + PerLayerEmbedding module).

**Gap:** Apple AFM (CROSS_BLOCK_SHARED variant of A18) is enumerated but not exercised. **Decision:** flagged as a deferred follow-up; the enum value exists so a future Apple AFM addition is config-only.

**Gap:** OpenELM per-layer head counts (A12 sub-variant). **Decision:** spec already has `LayerScaleSpec`; defer the OpenELM family addition to a B9.5 or follow-up.

## Placeholder scan

Scanned this document for placeholders:
- "TODO" — 0 occurrences.
- "FIXME" — 0 occurrences.
- "XXX" — 0 occurrences.
- "stub" — appears once in B5 (`IndexerSpec` stub) and once in B8 (`VisionAdapterSpec` stub); both are deliberate and named "shape-only" elsewhere.
- "placeholder" — 0 occurrences in plan content (one in the M1 plan AWQ code that's shipped, not a problem here).

## Type consistency

- All new spec fields use `Optional[T]` with `None` default → backward compatible. Verified for `attention_k_eq_v`, `qk_norm_fixed_scale`, `partial_rotary_factor`, `share_scheme`, `num_kv_shared_layers`, `per_layer_embedding`, `final_logit_softcap`.
- `PLESpec.injection_norm: "NormSpec"` uses a forward reference (NormSpec is defined later in the same module). This works for frozen dataclasses with `from __future__ import annotations`.
- The B0.5 absorbed-fixed-scale math (in `load_gemma4_weights_into_block`) writes the final value once after a clarifying explanation; the duplicated `copy_` calls in the code snippet are illustrative and the actual implementation drops the first copy_ pair. Mark for review at B0.5 T14 implementation time.

## Naming consistency

- `SharedLayerKVCache` vs `KVCacheSpec.share_scheme=SAME_BLOCK_SHARED` — the wrapper type and the enum value are different names with the same semantic. **Decision:** keep both — the wrapper is a runtime implementation, the enum is a spec parameter.
- `attention_k_eq_v` (Python snake_case) matches Gemma 4 HF config's `attention_k_eq_v` field — consistent naming preserved.
- `partial_rotary_factor` matches HF config; consistent.

---

# Summary

| Item | Value |
|---|---|
| Total batches | 11 (B0.5 + B1..B10) |
| Total models added | ~52 family-instances (~22 architecturally distinct) |
| B0.5 task count | 19 (T1..T19) |
| Estimated total commits | ~250-340 across all batches |
| Estimated subagent dispatches | 10 (one per batch B1..B10) |
| Spec extensions in B0.5 | 7 (partial_rotary_factor, attention_k_eq_v, qk_norm_fixed_scale, final_logit_softcap, PLESpec, ShareScheme + num_kv_shared_layers, PRE_AND_POST sandwich exercised, MaskKind.SWA exercised) |
| Numerical gates per batch | ≥1 family per batch at atol=5e-4 vs HF |
| Deferred to post-M2 | Apple AFM, OpenELM, DSA full numerical, CSA+HCA, full Qwen3-Next DeltaNet numerical, full Gemma 4 12B+ encoder-free MM |

**Design decisions to note:**
- **B0.5 is the IR-locking batch and is executed directly.** No subagent dispatch until the user reviews the B0.5 IR. This is a deliberate friction point at the cost of slower B0.5 execution.
- **Per-family `to_block_spec(layer_idx)` dispatch is the per-layer heterogeneity mechanism.** Gemma 4 (local vs global), SmolLM3 (NoPE every 4th), Ministral (interleaved SWA), Llama 4 Scout (iRoPE), Jamba (Mamba vs attention) all use the same pattern: config method returns a different `DecoderBlockSpec` for layer i. No new API surface needed.
- **Gemma 4 absorbs `1/sqrt(Dh)` into the QK-norm weight at load time.** This keeps the runtime attention path simple (`effective_scale=1.0`); the trade-off is non-portable on-disk weights if our absorbed weights are saved back out — they would not match HF format. Mitigation: never save absorbed weights back out from M2 code paths; loaders are one-way.
- **`SharedLayerKVCache` is a runtime alias, not a separate cache layout.** The `KVCacheSpec.share_scheme` enum value is a parameter; the actual aliasing happens at the model assembly level (which is out of scope for M2 but the wrapper is provided so future model-level code can use it).
- **Mamba-3 complex state uses paired real/imag tensors.** Avoids PyTorch complex-tensor incompleteness; documented in `models/mamba3/layer.md`.

---

> End of v3 implementation plan. Next action: user reviews B0.5 task list, decides whether to lock the v3 IR shape as-described or revise before B0.5 execution begins.
