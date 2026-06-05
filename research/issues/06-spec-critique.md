# Spec Critique — `2026-06-04-llm-layers-design.md`

**Reviewer stance:** SKEPTICAL technical reviewer applying a survey-paper bar.
**Target:** `docs/superpowers/specs/2026-06-04-llm-layers-design.md` v1 (draft).
**Evidence base:** the five research reports referenced by the spec
(`research/01..05`).

The spec is structurally sound and ambitious; it captures most of what the
research surfaces. The flaws below are not "rewrite the whole thing" flaws —
they are gaps and unspoken assumptions that, left in, will leak through into
the M1 implementation, force ad-hoc patches in M2+, and weaken the claim that
"the API is the IR." This critique is written so each finding maps to a
concrete change in v2.

---

## Section 1 — API completeness issues (logical-op floor)

The "9 logical ops" claim is the spec's load-bearing thesis (§5.1). The
research (`research/03-ihv-opsets.md`, §"Synthesis") explicitly says nine ops
cover **"≥95% of every decoder block"** — a deliberately weak claim. The spec
elevates that to a *floor*, which is a stronger claim and is wrong as stated.
The following ops are needed by models *already in the spec's rollout
(Section 9)* and are silently missing from §5.1.

**Gap 1.1 — `softmax` is not in the op list.** SDPA is fused (`sdpa(...)`), so
softmax is implicit inside attention; but the MoE router (`MoESpec.router_kind
∈ {SOFTMAX, SIGMOID_PLUS_BIAS}`) requires a standalone softmax/sigmoid on
gate logits. The spec has no op for it. Either expand the op list, or
explicitly state that the MoE building block hides this primitive inside
itself.

**Gap 1.2 — `top_k` is not in the op list.** MoE routing (`top_k: int` on
`MoESpec`) is a graph-level operation. TRT-LLM exposes `topkLastDimPlugin` as
a named plugin (research/03, §"TRT-LLM plugin list"); it is *not* free. The
spec hides it inside the `MoE` building block but never says so. For a model
like DeepSeek-V3 with `group_routing`, top-k is applied *twice* (once on
groups, once on experts) — an unstated op.

**Gap 1.3 — `gather` / `scatter` (token routing).** MoE dispatch sends a
subset of tokens to each expert and recombines. This is a `gather`/`scatter`
or grouped-GEMM pattern. The spec's Risk table mentions "expert ffn
abstraction may leak" but doesn't surface the underlying op. ONNX has
`GatherBlockQuantized`; QAIRT, MIGraphX, MLX all expose gather/scatter as
primitives. Either add them, or document that the reference impl uses dense
all-to-all (research/03 hints at this).

**Gap 1.4 — `conv1d` (depthwise, causal, k=4) is not in the op list.** Mamba
is in Batch B7. `research/02-layer-sources.md` line 1027 shows the depthwise
causal `Conv1d(E_int, E_int, kernel_size=4, groups=E_int, padding=3)` Mamba
needs. `research/03-ihv-opsets.md` line 300 names TRT-LLM's
`mambaConv1dPlugin` as a *primitive* (i.e., not decomposable into
linear+sdpa). The spec lists `SSMSpec` at §5.4 line 289 but never enumerates
its op surface. Without `conv1d`, B7 cannot land.

**Gap 1.5 — `selective_scan` / `parallel_scan` / `cumsum` are absent.** Same
source: TRT-LLM ships `selectiveScanPlugin` and `cumsumLastDimPlugin` as
*primitives*. SSMs (Mamba 1/2, RWKV-7) require at least one of these. The
spec gestures at "SSMSpec" but provides no dataclass definition or op trace.

**Gap 1.6 — `lru` / state-update for RecurrentGemma / Griffin / RWKV-7.**
TRT-LLM has `lruPlugin` for exactly this. The spec mentions RWKV-7 in B7 but
this is a different recurrence family (WKV update). One op slot or a clear
"out of scope" stance is needed.

**Gap 1.7 — `cast` / dtype conversion is not an op.** Quantized linear emits
in `accumulator_dtype` and must cast back to `compute_dtype`. The spec hides
this inside `linear`, which is fine, but the *op trace doc* (§6 `layer.md`
schema) needs to show casts explicitly, or the doc is misleading.

**Gap 1.8 — No `concat` / `split`.** MLA splits the projected head into
`[d_nope, d_rope]`. Partial-RoPE (GPT-J, Phi) splits/concat halves. The
research (research/05 §"MLA forces partial RoPE") says this is structural,
not optional. `sdpa` doesn't perform the split; the building block must.
Either expose `split`/`concat` as ops or document them as
`api/attention.py` internals.

**Gap 1.9 — No `causal_mask_builder` / `mask_compose`.** Phi-3-small uses
**BlockSparse** attention (research/01, §"Phi-3-small"), Gemma 3 alternates
SWA/global per layer, hybrid models alternate Mamba/attention. The spec
treats this with `MaskKind` enum on `AttentionSpec` — but the *construction*
of a custom 2D mask (block-sparse, document-causal, packed-sample) is itself
graph work. Either add a mask-build op or fully enumerate the supported
shapes.

**Gap 1.10 — `embed`'s `scale?` is binary, not general.** §5.1 row "embed"
says "optional Gemma `sqrt(D)` scale". Granite has `embedding_multiplier`
(research/01 line 156); MiniCPM-3 has `scale_emb`; μP variants vary. The op
should accept a generic scalar, and the *spec* (not the op) should encode
which value. Same for `lm_head` `scale?`.

**Gap 1.11 — Codebook embedding / `gather_with_dequant`.** ONNX
`GatherBlockQuantized` (research/03 line 579) is the canonical op for
embedding-table dequantization. IQ-quants (research/04 §3.2) use a *global
codebook*; the embedding lookup is `codebook[indices[token_id]]`. The spec's
`embed` is fp-only. Either widen the signature or restrict scope.

**Op gaps not addressed by the spec but not strictly required by B1–B7:**
top-k routing nuance (group + within-group), early-exit / layer-skipping
(out of scope is fine but should be stated), cross-attention
(encoder-decoder; Florence-2 in research/01 is encoder-decoder — but the
spec scopes to decoder-only, which is a defensible cut if stated).

**Counted API gaps (mandatory for declared scope):** 11.
- Hard (blocks B7): 1.4, 1.5, 1.6
- Hard (blocks B6 MoE done right): 1.1, 1.2, 1.3
- Hard (blocks Phi-3-small / Gemma 3 alternation): 1.9
- Soft (documentation / op-trace fidelity): 1.7, 1.8, 1.10, 1.11

---

## Section 2 — Spec dataclass gaps

### 2.1 `AttentionSpec` (§5.2.1)

- **`kind` enum is `{STANDARD, MLA, DIFFERENTIAL, LINEAR}` — too narrow.**
  `research/05` §"13 attention variants" enumerates: MHA, GQA, MQA, MLA,
  SWA, SWA+global, sink, linear/retention, Mamba, Mamba-2, CLA-sharing,
  Differential, ALiBi-bias. The spec collapses MHA/GQA/MQA into `STANDARD`
  (defensible: parameterized by `n_kv_heads`), but SWA and SWA+global and
  sink are *masking* kinds, already handled via `mask_kind` — good. Where
  it fails: `LINEAR` is one bucket for retention, RWKV-style linear
  attention, and "linear attention" generically — three different recurrences
  with three state shapes. Make `kind ∈ {STANDARD, MLA, DIFFERENTIAL,
  LINEAR_RETENTION, LINEAR_RWKV, LINEAR_GLA}` or hoist linear-attention
  into the `SSMSpec` family alongside Mamba (which is what the IHV ops list
  suggests — TRT-LLM groups them).

- **No `apply_rope: bool` per layer.** SmolLM3 NoPE pattern (research/01
  line 152, research/05 §1.7) skips RoPE every 4th layer. The spec has
  `rope: Optional[RoPESpec]` — setting it to `None` per layer works, but
  the *DecoderBlockSpec* must explicitly support this per-layer override.
  Confirm in §5.4 doc that a `list[DecoderBlockSpec]` is canonical.

- **No `attn_logit_softcap` distinct from final-logit softcap.** §5.2.1
  has `logit_softcap` on attention; §5.4 has `logits_scale` on decoder
  block. Gemma 2 has BOTH a per-attention softcap (cap=50) and a
  final-logits softcap (cap=30). The spec needs to be explicit that these
  are different fields (research/05 line 119). Currently they live in
  different specs but the naming is confusing.

- **`shares_kv_with: Optional[int]` is too thin for CLA/YOCO.** YOCO has a
  single global cache shared by *many* downstream layers and a different
  producer for prefill vs decode. CLA shares pairwise. `shares_kv_with`
  encodes only pairwise share-with-one-producer; it doesn't say whether the
  current layer *also* writes (producer/consumer/both). Add
  `cache_role ∈ {PRODUCER, CONSUMER, PRODUCER_CONSUMER}`.

- **`mask_kind` doesn't include `BLOCK_SPARSE`.** Phi-3-small block-sparse
  attention (research/01 surfaces this in passing) is a distinct mask
  family. Either add `BLOCK_SPARSE` with its own param block, or document
  Phi-3-small as out of scope.

### 2.2 `RoPESpec` (§5.2.2)

- **Missing `PI` (Position Interpolation, Chen et al. 2023).** Spec lists
  `{NONE, NTK, YARN, LLAMA3, LONGROPE}`. PI is the original 2023 method
  used by Llama-2-Long, CodeLlama, Vicuna-16k. Even if no current SLM uses
  pure PI (Llama-3 supersedes it), the survey-paper framing demands its
  presence.

- **Missing `DYNAMIC_NTK`.** Several HF models (Qwen2 7B, some Mistral
  fine-tunes) ship `rope_scaling.type = "dynamic"` — NTK with a per-call
  recomputed scale based on actual `seq_len`. Not the same as NTK static.

- **Missing `NoPE`.** SmolLM3 (research/01 line 152). Currently you'd
  encode by setting `rope=None` on a layer — but then the *RoPESpec*
  itself has no `NONE` value distinguished from "rope absent." That's
  fine as long as `DecoderBlockSpec` clearly supports `rope=None` per
  layer; document it.

- **Missing `ALiBi`.** Research/05 §1.8: MPT/BLOOM/Falcon use ALiBi. The
  spec excludes ALiBi by scoping RoPE only, which is defensible if the
  excluded-models list is documented. Today, the spec just elides ALiBi.

- **No `partial_rope` axis on `RoPESpec` itself.** Partial RoPE belongs on
  `RoPESpec` because the rotated channel dim is a property of the RoPE
  configuration. The spec puts `rope_partial_dim` on `AttentionSpec` (line
  154). Defensible but inconsistent — MLA's `qk_rope_head_dim` lives on
  AttentionSpec while non-MLA partial RoPE (GPT-J) lives… also on
  AttentionSpec. Pick a home.

### 2.3 `NormSpec` (§5.2.3)

- **`kind ∈ {RMS, LAYER}` is too narrow.** Missing **ScaleNorm** (T5,
  some research models), **DeepNorm** (post-norm variant with scaled
  residual). DeepNorm is relevant if you take OLMo 2's post-norm
  seriously — many post-norm models stabilize with DeepNorm-style
  residual scaling. Either add or out-of-scope explicitly.

- **No `bias` flag.** LayerNorm has bias; RMSNorm doesn't. The spec hides
  this in `kind` (RMS implies no bias). Fine if documented.

- **`WeightMode = {STANDARD_W, ONE_PLUS_W}` is Gemma-specific. Add
  `NONE` for unscaled RMS (some research) and possibly `LEARNED_PER_HEAD`
  for QK-norm-per-head.** QK-norm uses `NormSpec` (line 148), but QK-norm
  shape `PER_HEAD_DH` vs `FULL_HDH` lives on AttentionSpec
  (`qk_norm_shape`). Splitting the responsibility between two dataclasses
  for the same norm is a smell. Push `shape` into `NormSpec` or keep all
  QK-norm fields on AttentionSpec.

### 2.4 `FFNSpec` + `MoESpec` (§5.2.4)

- **`group_routing: Optional[GroupRoutingSpec]` is referenced but not
  defined.** The spec mentions DeepSeek-V3 "node-limited routing" via
  `group_routing` but doesn't show the `GroupRoutingSpec` dataclass. v2
  must define: `n_groups`, `top_k_groups`, `routed_scaling_factor`,
  whether group score is `sum_top_k_in_group` or `max`. This is a
  semantically subtle DeepSeek-V3 detail.

- **No `expert_pruning` / `n_routed_experts vs n_experts`.** DeepSeek-V3
  has `n_routed_experts=160` + `n_shared_experts=2`; you route across
  the 160 with top-k=6, but the *physical* tensor packs all 162. The
  spec has `n_experts` and `n_shared_experts` — clarify which one is the
  "routed pool" count.

- **`expert_ffn: FFNSpec`** — assumes all experts share shape. DeepSeek
  models do. But Mixtral *also* allows non-uniform — confirm or simplify.

- **`router_kind` missing `LINEAR_TOP_K_NO_SOFTMAX`.** Some open MoE
  papers use plain top-k without softmax/sigmoid. Probably OOS for SLMs;
  state so.

### 2.5 `KVCacheSpec` (§5.2.5)

- **No PagedAttention v2 / vAttention / YOCO global cache distinction.**
  Spec lists `{CONTIGUOUS, PAGED, RING, MLA_LATENT, SSM_STATE}`. vAttention
  (research/05 §"Cache layouts") uses CUDA VM mapping — same logical
  layout as PAGED for the API. YOCO has *one* cache shared by many
  layers; logically encoded by `share_group`, but the cache itself is
  per-block-of-layers, not per-layer. Add a comment: "PAGED covers
  vAttention and PagedAttention v1/v2 — they differ only in memory
  allocator." Make the share-group story explicit for YOCO.

- **`memory_layout: {HND, NHD}` — missing BNHD vs BHND ordering.** vLLM
  uses `[N_blocks, block_size, H, D]`. TRT-LLM has its own. MLX has
  Python lists. The two values may not be sufficient.

- **No `dtype_promotion_path` for quantized cache.** When K is INT8 per
  channel and SDPA wants FP16, the dequant-on-read path needs to be a
  property of the cache, not the SDPA op. The spec has `k_dtype` and
  `k_quant` but doesn't say where dequant happens.

- **No cache-side scaling/zero-point lifetime.** Quantized KV cache
  stores scale tensors alongside values; their dtype, layout, and
  granularity are not in `KVCacheSpec`. They are implicit in `k_quant:
  QuantSpec`. Confirm `QuantSpec.scale_dtype` and `scale_axis` propagate
  through; the spec doesn't trace this.

### 2.6 `QuantSpec` (§5.2.6) — the highest-stakes dataclass

The recursive `scale_quant: Optional['QuantSpec']` is the right idea, but
it does not cleanly handle the three "support only 3" baseline schemes the
spec itself names (line 258–263). Tracing:

- **GGUF Q4_K_M** (research/04 §3.1): 256-element super-block, **6-bit
  sub-block scale + 6-bit sub-block min**, plus a `fp16` super-block
  scale that scales the sub-block scales. So you have `qdtype=INT4`,
  `group_size=32` (sub-block), wrapped in a super-block of 8 sub-blocks
  (256 elements), and the sub-block scales themselves are 6-bit
  quantized. Does the recursive `scale_quant=QuantSpec(qdtype=Q6,
  group_size=256, ...)` express this? It does *if* you allow `Q6` as a
  qdtype and if you accept that the outer block size (256) and inner
  group (32) co-exist. The spec's enum `QDType ∈ {INT4, INT8, FP8_E4M3,
  FP8_E5M2, FP4, NF4, MX_FP4, ...}` does not list a 6-bit type. Add
  `INT6_SCALE_GGUF` or document that GGUF k-quants need a `block_layout`
  enum value rather than a clean recursive structure.

- **AWQ W4A16 g=128** (research/04 §1.1): values are INT4 grouped at 128,
  scales are FP16, zero-points are INT4 packed. AWQ also uses an
  **interleaved storage layout** `[0,2,4,6,1,3,5,7]` for the MMA-friendly
  GEMM kernel — *not* the same as GPTQ's pack order. The spec
  acknowledges `packing: AWQ_INTERLEAVE` but does not show the
  unpack→dequant→matmul lowering. M1's `api/quant.py` line item ("AWQ
  W4A16 (g=128) packed-int weight load + dequant + matmul path") is the
  hardest single deliverable in M1 and the spec gives it one bullet.

- **AWQ ≡ GPTQ ≡ HQQ "storage-identical"** (§5.2.6 last paragraph) is
  **not quite true.** Per research/04 §1.1–1.3: AWQ uses interleaved
  pack `[0,2,4,6,1,3,5,7]`; AutoGPTQ packs `W` as int32 with column-major
  pack-factor; HQQ matches AWQ kernel but with a slightly different
  scale-storage convention. *Numerically* the three produce different
  dequantized weights even at "same bit width, same group" because the
  *calibration produces different scales/zero-points* — AWQ is
  activation-aware, GPTQ uses Hessians, HQQ uses no calibration. The
  storage layout can be unified after re-packing, but the values are
  not interchangeable. The spec sentence implies they are equivalent,
  which is false. Soften to: "*storage layout is unifiable; scales are
  computed by different procedures and are not interchangeable.*"

- **FP8 W8A8** (research/04 §2.4): per-tensor W, per-token A. Does
  `QuantSpec` say where the activation scale lives? It must:
  per-token activation scale is computed at inference time, not stored
  with weights. The spec has `role: QuantRole ∈ {WEIGHT, ACTIVATION,
  KV_K, KV_V, ATTN_INTERNAL}` — good. But there's no field for "scale
  computed dynamically vs loaded static," which matters for FP8 SmoothQuant
  vs FP8 static. Add `scale_source: STATIC | DYNAMIC_PER_TOKEN |
  DYNAMIC_PER_TENSOR`.

- **NF4** (research/04 §1.4): 4-bit *codebook*, double-quantized scales.
  Spec has `codebook: Optional[CodebookSpec]` and recursive `scale_quant`
  — covers it. Verify in v2 that `CodebookSpec` includes a `codebook_kind
  ∈ {NF4, FP4, IQ_LATTICE_E8, IQ_LATTICE_NL}` and the 16 NF4 codepoints.

- **NVFP4** (research/04 §1.6): two-level scales — FP8 inner scale per
  16-element block + FP32 outer scale per tensor. Recursive `scale_quant`
  handles this *if* `qdtype=FP8` is a valid inner scale type, which it
  is. Good — verify in implementation.

- **AWQ does NOT cleanly fit "support only 3 exercises every QuantSpec
  axis" as claimed in §5.2.6.** The three baselines (GGUF Q4_K_M, AWQ
  W4A16, FP8 W8A8) do not exercise:
  - `codebook` axis (NF4, IQ — neither GGUF Q4_K nor AWQ uses a codebook;
    only IQ-quants and NF4 do)
  - `pre_transform: Hadamard` (QuaRot/SpinQuant — research/04 line 274;
    not used by any of the three)
  - `block_layout: OCP_MX` (only MXFP4/6/8 uses it)

  Either add a fourth scheme (NF4 or MXFP4) to exercise these, or scope
  them out explicitly.

**Counted dataclass gaps:** 18 (across the six dataclasses).
- `AttentionSpec`: 5
- `RoPESpec`: 4
- `NormSpec`: 2
- `MoESpec`: 3
- `KVCacheSpec`: 3
- `QuantSpec`: 1 *substantive* (the false-equivalence claim) + several
  documentation tightening items

---

## Section 3 — Hidden assumptions

The "survey paper bar" demands that the spec name its assumptions so a
careful reader can verify them. Currently the spec carries several baked-in
assumptions that are not labelled as such.

**A. Pre-norm is the default; post-norm is "supported."** §5.4
`DecoderBlockSpec` allows `attn_norm_position: {PRE, POST, PRE_AND_POST}`
— good. But §5.1's `rms_norm(x, weight, eps, mode)` signature does not say
what `x` is normalized *into* (the sublayer input or the residual stream).
For pre-norm: norm(residual)→sublayer. For post-norm: sublayer(residual)
→norm. The op signature is the same, but the *graph topology* differs.
OLMo 2 (research/01 line 341) is post-norm + a high-magnitude residual
stream that may overflow FP16; OLMo 2 uses BF16 throughout. The spec
should state: "post-norm implies BF16/FP32 residual stream; FP16 not
supported for post-norm models."

**B. Single token-mixer per block.** §5.4: `token_mixer: Union[AttentionSpec,
SSMSpec]`. **Hymba** (NVIDIA, late 2024) runs Mamba + Attention in
*parallel within one block* — both consume the same input and their
outputs are added. The spec's `Union` precludes this. If you want to be
the floor under "any mainstream SLM," Hymba is borderline; if you want to
cover it, `token_mixer: list[TokenMixer]` with an additive combine. State
the position. Otherwise document Hymba as out of scope.

**C. Single channel-mixer per block.** §5.4: `channel_mixer:
Union[FFNSpec, MoESpec]`. DeepSeek-V3 interleaves dense FFN layers and
MoE layers — handled by `list[DecoderBlockSpec]` with different
`channel_mixer` per layer. Good. But: DeepSeek-V3 *also* has
**shared experts** within an MoE layer that run *alongside* the routed
experts (`n_shared_experts`). The MoE building block must compose
shared-FFN + routed-FFN, summing their outputs. `MoESpec` has
`n_shared_experts: int` — confirm the building block sums correctly. Not
a spec gap, but worth a one-line explanation in §5.3.

**D. Single residual stream.** Differential Transformer (research/05 §1.5)
uses TWO residual streams: `(λ · attn_1 − attn_2)` form. The spec has
`kind=DIFFERENTIAL` on AttentionSpec, with `diff_lambda_init: Optional[float]`.
But the residual *path* is unchanged in `DecoderBlockSpec`. Confirm: does
the differential block remain a single-stream block where the "two
attentions" are internal to the building block? If yes (the typical
implementation), state so. If no, the spec needs dual residual paths.

**E. The 1e-3 numerical tolerance** (§8 line 366: "*within 1e-3*") is
**too permissive for FP16 single-block forward.** A typical HF
`Qwen3DecoderLayer` forward at FP16 with random inputs at length=64
produces values O(1); a 1e-3 absolute tolerance hides accumulator-order
bugs and silent RoPE-basis mismatches. The norm should be specified
(L∞? mean abs? relative?). A defensible bar: `max(|y − y_ref|) < 1e-3
AND mean(|y − y_ref|) < 1e-4` at BF16, with FP32 reference. State the
test-input distribution too.

**F. Single sequence axis.** RoPESpec has `mrope_section` for M-RoPE (VL),
implying multi-axis positions. But the API ops (`rope_apply`, `sdpa`)
assume `seq` is a single dimension. Multi-axis positions need the mask
and the RoPE table indexed differently. Either explicitly scope out VL
(reasonable for SLM-text), or carry the cost.

**G. The `kind` discriminator implicitly assumes mutually-exclusive
families.** A model that is "MLA + Differential" or "SWA + Linear" is not
expressible. Probably no SLM is hybrid in this sense yet. State that
assumption.

**H. KV cache is per-layer.** YOCO violates this (one global cache served
by a *block* of layers). The spec's `share_group: Optional[int]` covers it
weakly; the cache *ownership lifecycle* is a per-block-of-layers concern,
not per-layer.

---

## Section 4 — Validation realism (M1 Qwen3 sample)

The Qwen3 kickoff plan in §8 is concrete but underspecified in several
operational details. The M1 "definition of done" must be tight or the
review gate will block on disagreements about what "done" means.

**Issue 4.1 — Weight mapping is unspecified.** The table row "AWQ W4A16
(g=128) packed-int weight load + dequant + matmul path" needs a weight
mapping function: HF tensor names (`model.layers.0.self_attn.q_proj.weight`,
`...q_proj.qweight`, `...q_proj.scales`, `...q_proj.qzeros`) →
API tensor slots in `Attention(spec=...)`. This is non-trivial and
non-portable across HF format families (AWQ, GPTQ, bitsandbytes all
differ). The spec is silent on whether this mapping is part of M1 or
hidden in test code. *Make it explicit*: M1 ships
`models/qwen3/weight_loader.py::load_awq_into_layer(safetensors_path,
layer: DecoderBlock) -> None` or equivalent.

**Issue 4.2 — Numerical-equivalence test input is unspecified.** The spec
says "*on a known prompt*" (line 366). One prompt? What length? Batch
size? Token IDs from where? Use the prompt distribution that minimizes
test fragility — e.g., a fixed 32-token integer sequence (deterministic,
no tokenizer) over `batch_size=2`, `seq_len=32`. Specify in v2.

**Issue 4.3 — Numerical-equivalence reference scope.** The test compares
**a single decoder layer**, not the full model. Position embeddings
(RoPE freq tables) depend on `position_ids` — does the test pass
`position_ids=[0..31]` or simulate prefill from position N? Both have
been shown to surface RoPE-basis bugs; spec the choice.

**Issue 4.4 — AWQ unpacking path is the hardest M1 deliverable.** AWQ
storage is `qweight ∈ INT32[K/8, N]` with the 8-int4 lane reorder
`[0,2,4,6,1,3,5,7]`, plus `qzeros ∈ INT32[K/group/8, N/8]` with the same
reorder, plus `scales ∈ FP16[K/group, N]`. The dequant path is:
1. Unpack 8 INT4 lanes per INT32 word, undo the interleave.
2. Subtract zero-point (unpacked from qzeros similarly).
3. Multiply by scale (broadcast by group along K).
4. Cast to compute dtype.
5. GEMM in compute dtype.

The spec mentions packing: `AWQ_INTERLEAVE` (line 244) but does not
trace this lowering. M1's success depends on getting bit-exact
unpacking; this is research/04 §1.1 territory. Fleshing out the lowering
in `api/quant.py` doc is mandatory for M1.

**Issue 4.5 — QK-norm placement.** Qwen3 uses QK-norm
*after* RoPE (research/05 line 111), but the literature says before-RoPE
preserves rotation. The spec has `QKNormPhase ∈ {NONE, PRE_ROPE,
POST_ROPE}`. Confirm in M1 that POST_ROPE is the Qwen3 setting and that
test_layer.py covers it. (Spec already does this; verify.)

**Issue 4.6 — The "numerical-equivalence test for Qwen3 fails due to
RoPE basis bug" risk (§10) is real and the mitigation is correct.** Add a
*sub-op* equivalence ladder to M1: (a) rms_norm vs HF, (b) rope vs HF,
(c) sdpa vs HF, (d) ffn vs HF, before (e) block-level. The spec already
hints at this (§10 Risks), but it should be a deliverable in M1, not a
contingent plan.

---

## Section 5 — Rollout / scope issues

**Issue 5.1 — Model count.** §9 claims ~25, lists ~23 in the batch table
+ Qwen3 in M1 = 22 actually named. The spec hand-waves "1-2 more if a
B-batch surfaces a new axis." Survey-paper bar: name the actual model
list or commit to a stable count. Recommend: lock at 24 (the 23 + Qwen3)
and remove the wiggle room.

**Issue 5.2 — Batch boundaries.** B1 is "Llama-family dense" but
includes SmolLM3 — which has the NoPE pattern (research/01 line 152).
That's an API extension (per-layer `apply_rope`), so SmolLM3 belongs in
B2 or B4. Re-shuffle so the batch promises hold ("no API extensions" is
a strong claim that B1 currently violates).

**Issue 5.3 — B7 SSM-hybrid as a single batch is too aggressive.** Mamba
2.8B (pure SSM, no attention), Jamba-mini (interleaved Mamba+attention
+MoE), Zamba2 (interleaved + shared global attention), RWKV-7 (WKV
recurrence, *not* Mamba) are four different recurrence families. RWKV-7
needs `lru`-style ops; Mamba needs `selective_scan` + `conv1d`. These
should be two batches with one model each, or B7 should drop to one
model (Mamba) and RWKV/Jamba spread later.

**Issue 5.4 — B8 "quant variants on existing models" is fine** but the
spec should name *which* existing models. E.g., "*FP8 W8A8 exercised on
Llama-3 8B; GGUF Q4_K_M exercised on Mistral 7B*." Currently B8 is
unbound and could be wished away.

**Issue 5.5 — OOS items.** Three were enumerated (TP, spec decoding,
LoRA). Research/04 §"LoRA mergeability" *is* raised in the quant survey
— specifically the question "can a LoRA adapter be merged into a
quantized weight without dequantizing?" — answer is "only for low-rank
adapter * dequantized weight then re-quantize, which is lossy." This is
small (one paragraph in the spec) but the spec's flat "out of scope" is
under-justified. Either explain why (no SLM-side LoRA convergence
yet) or re-include LoRA at the `QuantSpec` level (a `merged_lora` field
or a load-time merge function).

---

## Section 6 — Contradictions and consistency

**C-6.1 — §5.2.6 "AWQ ≡ GPTQ ≡ HQQ — storage-identical" vs §8 "AWQ for
Qwen3 + GGUF Q4_K_M + FP8."** If they are storage-identical, why
exercise AWQ separately (one of the three "support only 3" set) instead
of subsuming under GPTQ? The contradiction resolves by tightening the
claim: AWQ/GPTQ/HQQ share *kernel layout* but differ in *calibration
procedure*, so they produce different scales. Doing AWQ exercises
"loading INT4 grouped weights with a specific pack/scale convention" —
not GPTQ's convention. State this clearly.

**C-6.2 — "Three exercises every QuantSpec axis" (§5.2.6 last line) is
false** as written. The three (GGUF Q4_K_M, AWQ W4A16, FP8 W8A8) do
not exercise: `codebook` (no codebook in any of the three),
`pre_transform` (Hadamard not used), nor `block_layout=OCP_MX` (no
MXFP). The recursive `scale_quant` axis is exercised only by GGUF (with
super-block scales) — if you accept 6-bit as a valid `QDType`. Tighten:
either add NF4 to the baseline (covers codebook + double-quant cleanly),
or state that codebook/Hadamard/MX axes are designed-in but unverified.

**C-6.3 — §5.1 "9 logical ops" vs §5.4 `SSMSpec`/`TokenMixerKind`
mentions (line 392 in B7).** Neither `SSMSpec` nor `TokenMixerKind` is
defined in §5.2. The spec promises in §5.4 line 289 that `token_mixer:
Union[AttentionSpec, SSMSpec]` — but `SSMSpec` is referenced and never
defined. This is the biggest single hole in the dataclass definitions
section.

**C-6.4 — §5.2.1 has `kind: AttentionKind ∈ {STANDARD, MLA,
DIFFERENTIAL, LINEAR}`**, but §9 B6 names "Linear attention" only
within hybrid (B7) — never in pure attention. So `LINEAR` is either
dead in B1–B6 or it's meant for RetNet, which is not in the rollout.
Drop `LINEAR` from `AttentionKind` or add a model that uses it.

**C-6.5 — §5.2.5 `KVCacheSpec.layout` lists `MLA_LATENT` and
`SSM_STATE`** as cache layout values. But MLA's cache *also* needs the
qk_rope channel — it's `[c_kv: kv_lora_rank, k_rope: qk_rope_head_dim]`
per the spec comment. The layout enum value `MLA_LATENT` hides this.
Either rename to `MLA_LATENT_PLUS_KROPE` or document the two-tensor
structure in §5.2.5.

**C-6.6 — §6 layer.md schema requires "Op trace" but §5.1's 9 ops are
not sufficient to express MoE or Mamba — see Section 1 gaps.** The
schema will produce inconsistent doc quality across batches until the
op list is firmed up.

---

## Section 7 — Missing sections

These are sections that a survey-paper-grade design spec should have but
this draft lacks.

**7.1 — Glossary of enum values.** `NormPosition`, `AttentionKind`,
`MaskKind`, `QKVLayout`, `QKNormPhase`, `QKNormShape`, `WeightMode`,
`GateKind`, `Activation`, `RouterKind`, `CacheLayout`, `MemoryLayout`,
`CacheOwnership`, `QDType`, `PackingLayout`, `BlockLayout`,
`PreTransform`, `QuantRole`, `RoPEScaling`, `RoPEBasis`,
`TokenMixerKind` (mentioned, not defined). Each value needs (a) a
one-line semantic, (b) which model uses it. Without this the
dataclasses are unverifiable.

**7.2 — Weight-loader story.** §6 says "load real HF weights" for Qwen3.
The spec is silent on the loader API. Add a §6.5 "Weight loading"
covering: (a) safetensors → API tensor slots (mapping schema), (b)
GGUF → API tensor slots (deferred to a later milestone, fine, but
named), (c) AWQ/GPTQ packed-int unpack (covered by `api/quant.py` but
named here), (d) the unit of work: per-layer? per-model? per-shard?

**7.3 — Performance / budget targets.** PyTorch eager is the spec
choice, but even there: what's the latency ceiling for the M1 numerical
test? What memory? Some bound prevents the test from regressing into
"30-minute single forward pass." Recommend: "*Qwen3 single decoder
layer forward at seq_len=32, batch=2, FP16, on CPU: < 1 s.*"

**7.4 — Test fixture provenance.** Where do Qwen3 weights come from?
HF hub `Qwen/Qwen3-1.7B`? License (Tongyi Qianwen)? Cached locally?
Pinned to a specific revision SHA? Without this, the numerical
equivalence test will silently drift when upstream weights update.

**7.5 — Code style / lint / type-check.** Survey-paper-grade requires
**ruff + mypy + pytest --strict-markers** at minimum. Spec mentions
`pyproject.toml` with deps but no quality bar. Recommend: ruff,
mypy `--strict` on `api/`, mypy non-strict on `models/`, pytest with
`--tb=short`.

**7.6 — Contribution process for a new model.** When the rollout is
running and a new SLM family ships (e.g., Llama 5 in M3), what's the
recipe? Recommend §6.6: "How to add a model" — 5 steps from upstream
config to merged `layer.md` + `layer.py` + `test_layer.py`.

**7.7 — API versioning / stability.** `@dataclass(frozen=True)` is
*structural* immutability, but API evolution is a separate concern. The
spec doesn't say whether v2 will break v1. Recommend: "*the API is
unstable through M9; stability commitment begins at M10 (post-rollout
freeze).*"

**7.8 — Evolution preamble (survey-paper framing).** The spec is a clean
design doc but is *ahistorical*. The user wants this project to feel
like a survey paper. The spec should open with one paragraph: "*The
decoder-only transformer has evolved along ~11 named axes since GPT-3
(2020). Each axis carries a representative model and a paper. This API
collapses those axes into N (=8?) building blocks plus 6 dataclasses;
new axes (Mamba 2024, MLA 2024, NoPE 2025) are captured as enum
extensions, not new types.*" Add as §0 "Principles" or a preamble to §5.

---

## Section 8 — Recommended fixes for v2

In priority order, with severity tags.

**v2-fix-1 (CRITICAL) — Define `SSMSpec` and `TokenMixerKind`.** Currently
referenced but undefined (§5.4 line 289, §9 line 392). v2 must include
dataclass with: `kind ∈ {MAMBA1, MAMBA2, RWKV7, LINEAR_RETENTION}`,
`d_state`, `d_inner`, `dt_rank`, `conv_kernel_size`. List the op surface:
`conv1d`, `selective_scan` (Mamba), `wkv_update` (RWKV), `chunked_scan`
(Mamba-2). If SSM is deferred past M2-M6 in practice, name the
deferral. Don't leave a hole.

**v2-fix-2 (CRITICAL) — Expand the op list to 12-14 ops.** Add:
`softmax`, `top_k`, `gather`/`scatter` (or `dispatch`), `conv1d`,
`selective_scan` (or document as out of scope for M-x). Without these,
the "API is the IR" claim is false for B6 and B7.

**v2-fix-3 (HIGH) — Fix the "AWQ ≡ GPTQ ≡ HQQ" claim** and explicitly
list which QuantSpec axes the 3-baseline exercises. Add NF4 (or
MXFP4) as a fourth baseline to cover `codebook` and `block_layout`
axes, or state these are designed-in but unverified.

**v2-fix-4 (HIGH) — Tighten the numerical-equivalence test.** Specify:
input shape, position_ids, tolerance norm (L∞ + mean abs), reference
dtype (FP32 reference vs BF16 candidate), and the sub-op ladder
(rms_norm, rope, sdpa, ffn, block).

**v2-fix-5 (HIGH) — Add weight-loader API to M1.** Explicit
`weight_loader.py` for Qwen3 and a contract for future models.
Otherwise M1's numerical test is undefined.

**v2-fix-6 (HIGH) — Glossary of all enum values.** Without this, the
dataclasses are unverifiable.

**v2-fix-7 (MEDIUM) — Define `GroupRoutingSpec`** (DeepSeek-V3
node-limited routing). Currently a forward reference with no shape.

**v2-fix-8 (MEDIUM) — Add `cache_role` to `AttentionSpec`** for CLA/YOCO
producer/consumer semantics. Current `shares_kv_with` is too weak.

**v2-fix-9 (MEDIUM) — Refactor norm placement.** Move `qk_norm_shape`
into `NormSpec` (it's a norm property, not an attention property) or
move QK-norm fields fully onto AttentionSpec. Pick one home.

**v2-fix-10 (MEDIUM) — Fix B1 batch boundary.** SmolLM3 has NoPE; it
belongs in a batch that admits per-layer RoPE skipping, not in
"no API extensions" B1.

**v2-fix-11 (MEDIUM) — Add §7.x missing sections:** Glossary (7.1),
Weight-loader story (7.2), perf budget (7.3), fixture provenance
(7.4), code-quality bar (7.5), contribution process (7.6), API
versioning (7.7).

**v2-fix-12 (MEDIUM) — Add survey-paper preamble (§0).** One paragraph
on axis-evolution since GPT-3 and the framing of this project as a
"primitives census." Without it the API looks unmotivated; with it,
the survey framing is explicit.

**v2-fix-13 (LOW) — Drop `AttentionKind.LINEAR`** from the enum or add
a model that uses it (RetNet, RWKV-7 in attention form). Currently
dead.

**v2-fix-14 (LOW) — Clarify MLA cache layout name.** `MLA_LATENT` is
two tensors (`c_kv` and `k_rope`); the name elides this. Rename or
add a note.

**v2-fix-15 (LOW) — Hybrid block support.** Either explicitly scope
Hymba (parallel attention + Mamba in one block) out, or extend
DecoderBlockSpec to `token_mixer: list[TokenMixer]` with a combine
op.

**v2-fix-16 (LOW) — Restate "9 logical ops" claim.** It comes from
research/03's "*≥95% of every decoder block*" — keep the original
weak claim, don't strengthen it to a "floor."

**v2-fix-17 (LOW) — Lock model count.** Drop "1-2 more if needed" —
commit to 24 (23 + Qwen3) or extend to a fixed 25 with a 24th-25th
slot prefilled.

---

## Summary of countable findings

- **API gaps (Section 1):** 11 (4 critical, 3 medium, 4 documentation).
- **Dataclass gaps (Section 2):** 18 across six dataclasses
  (AttentionSpec 5, RoPESpec 4, NormSpec 2, MoESpec 3, KVCacheSpec 3,
  QuantSpec 1 substantive + several tightenings).
- **Hidden assumptions (Section 3):** 8 (A–H).
- **Validation issues (Section 4):** 6.
- **Rollout issues (Section 5):** 5.
- **Contradictions (Section 6):** 6.
- **Missing sections (Section 7):** 8.
- **Recommended v2 fixes (Section 8):** 17 (5 CRITICAL/HIGH, 5 MEDIUM,
  7 LOW).

The spec is on the right track. The two CRITICAL fixes (`SSMSpec`/op
expansion, NumEq tightening) are blockers for M1 success. The HIGH
fixes (false equivalence, weight loader, glossary) are blockers for the
"survey-paper bar" framing. The MEDIUM and LOW fixes are quality work
for v2 to be coherent.
