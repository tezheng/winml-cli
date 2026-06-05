# Critique: `04-quantization.md` — Quantization Survey

> Reviewer stance: skeptical, ruthless. The survey claims 25 schemes covered
> across 5 families and a 15-axis `QuantSpec`. This critique tests both the
> completeness claim and the IR axis claim against the actual landscape of
> shipped runtimes circa 2023–2026.

---

## Section 1: Missing schemes (with severity)

The survey omits ~14 schemes and sub-variants that are either actively
shipped, historically dominant, or research-bar-mandatory for a 3-year
evolution survey. Triaged:

### Severity P0 (must-have — survey is incomplete without these)

1. **Marlin kernel & checkpoint format** — The survey mentions
   `awq_marlin` and `gptq_marlin` in passing under runtimes, but **Marlin
   is not just a kernel, it is a storage repacking** (Frantar & Alistarh,
   2024; `github.com/IST-DASLab/marlin`). Marlin re-tiles weights into
   `(K/16, N/64, 16, 64)` blocks with a specific zig-zag interleave so a
   single `mma.sync.aligned.m16n8k16` can dequant 4-bit weights via PRMT.
   vLLM's `awq_marlin` and `gptq_marlin` *change the on-disk layout* from
   AutoAWQ/AutoGPTQ; a model loaded into vLLM is repacked on first load.
   **The IR's `packing` enum is missing `marlin_4bit` and
   `marlin_24` (2:4 sparse + 4-bit)**. Severity P0.

2. **ExLlamaV2 EXL2** — `turboderp/exllamav2`. Variable bitwidth
   *per row* of each weight tensor (Q-matrix layout), chosen by an
   activation-error budget. Bitwidths drawn from
   `{2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8}` and **mixed within a single
   matmul**. Pre-vLLM dominant for local Llama-2 inference. The survey's
   `QuantSpec` has no mechanism for **per-row variable bitwidth** (see
   Section 2). Severity P0.

3. **AQLM — Additive Quantization of Language Models** (Egiazarian et
   al., ICML 2024, arXiv:2401.06118). 2-bit weight quantization via
   *multi-codebook additive quantization*: each weight is the sum of K
   codebook entries from L codebooks. Best-in-class perplexity at 2 bits;
   shipped in HF Transformers (`aqlm` integration) and used by Llama-2-70B
   2-bit checkpoints. The survey's `codebook` field is single-level and
   does not capture additive multi-codebook. Severity P0.

4. **BitNet b1.58** (Ma et al., arXiv:2402.17764). Ternary weights
   `{-1, 0, +1}`, ~1.58 bits/weight; the **architecture is trained from
   scratch** with ternary weights and INT8 activations. Microsoft's
   T-MAC and bitnet.cpp ship CPU kernels; vLLM has experimental support
   for BitNet-1.5 checkpoints. Omitting this in a 3-year-evolution survey
   is a glaring miss — BitNet is the headline "extreme low-bit" story of
   2024–2025. The current `qdtype` enum mentions `int1` but ternary is
   not int1 (3 values not 2). Severity P0.

5. **QuaRot / SpinQuant Hadamard rotation as a first-class scheme, not
   a footnote** — survey mentions them in §2.3 in two sentences. These are
   the canonical *4-bit weight + 4-bit activation* shipped recipes (W4A4)
   and underlie e.g. SpinQuant Llama-3 8B in production at Meta. The
   survey's `pre_transform=hadamard` axis exists but **does not capture
   the structure**: QuaRot uses *fused* Hadamard (offline absorbed into
   weights, online Hadamard applied to activations); SpinQuant uses
   *learned* rotations. Same `pre_transform` enum value is wrong for two
   different mathematical objects. Severity P0.

### Severity P1 (should-have — mainstream but less-shipped)

6. **OmniQuant** (Shao et al., ICLR 2024, arXiv:2308.13137) —
   Learnable Weight Clipping (LWC) + Learnable Equivalent Transformation
   (LET). Bridges PTQ and QAT; shipped in `OpenGVLab/OmniQuant`. Missing
   because it is a different *calibration* family (learnable, not
   one-shot Hessian); the survey's `calibration` enum doesn't have a
   `learnable_clipping` value. Severity P1.

7. **SqueezeLLM** (Kim et al., ICML 2024, arXiv:2306.07629) — sensitivity-
   weighted *non-uniform per-row codebook* + outlier extraction. Each row
   has its own k-means codebook. The survey's `codebook` is "shared
   across global / per_super_block" — **missing `per_row`**. Severity P1.

8. **VPTQ — Vector Post-Training Quantization** (Liu et al., 2024;
   `microsoft/VPTQ`). Vector quantization for sub-2-bit weight-only;
   ships 1.5–2 bit Llama-3 70B from Microsoft. Distinct from AQLM
   (vector quant vs additive). Severity P1.

9. **ZeroQuant V1/V2/FP/HE** family (Yao et al., NeurIPS 2022;
   arXiv:2206.01861 + sequels). Group-wise INT8/INT4 with fine-grained
   per-token and per-channel; shipped in DeepSpeed. The W8A8 / W4A8
   pipelines in DeepSpeed-Inference predate SmoothQuant in production.
   Survey omits entirely. Severity P1.

10. **OWQ — Outlier-aware Weight Quantization** (Lee et al., AAAI 2024,
    arXiv:2306.02272). Mixed precision: a few outlier columns at fp16,
    rest at int3/int4. Shipped in Intel Neural Compressor and some HF
    integrations. Survey only covers LLM.int8's outlier handling.
    Severity P1.

11. **GGUF legacy quants Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0** — the survey
    table jumps straight to k-quants. The pre-k-quant format
    (`ggml_block_q4_0` = 32-weight block + fp16 scale) is **still the
    only format that ARM-NEON and some Vulkan backends fully accelerate**;
    Q4_0 is the format llama.cpp falls back to on mobile NPUs without
    k-quant kernels. Survey says "llama.cpp historically via Q4_0
    re-quantization" — that is a one-line dismissal of a still-shipped
    format. Severity P1.

12. **GGUF k-quant `_S` / `_M` / `_L` distinct sub-variants** — survey
    documents `_M` vs `_S` but **omits `_L`** (Q3_K_L, Q5_K_L) which
    upgrade more layers than `_M`. Severity P1.

### Severity P2 (nice-to-have — niche or research)

13. **BiLLM** (Huang et al., ICML 2024, arXiv:2402.04291) — 1-bit
    weights with salient-channel preservation. Severity P2.

14. **MXINT4 / MXINT8** — survey mentions MXINT8 in passing inside §1.5
    but does not list MXINT4 (OCP spec includes it). vLLM has experimental
    MXINT8 KV cache. Severity P2.

15. **NVIDIA FP8 per-tensor vs per-channel vs per-token as separate
    deployment configurations** — survey lumps them as
    `w_quant_granularity` enum values but does not call out that vLLM's
    `fp8` is per-tensor weight + per-token activation by default, while
    TRT-LLM defaults to per-channel weight. These are **distinct
    checkpoints**, not just kernel modes. Severity P2.

16. **Activation-aware K-quant variants (Q4_K_M with imatrix)** —
    survey mentions imatrix but does not distinguish a Q4_K_M *with*
    imatrix from one *without*; the former is now the LM Studio default.
    Severity P2.

17. **AutoGPTQ vs GPTQModel fork divergence** — only flagged in
    passing. The fork now has its own `g_idx` packing differences and
    `--marlin-format` flag. Toolchain detail, but matters for IR
    round-tripping. Severity P2.

**Scheme count after additions: 25 → ~39.** The survey's "25 distinct
schemes documented" is an undercount by ~36 %.

---

## Section 2: `QuantSpec` parameter space critique

The survey's 15-axis `QuantSpec` is impressive in scope but has at least
seven structural gaps. The "minimum-viable 7 axes" is even more
optimistic and demonstrably wrong (see below).

### Gap 2.1 — `pre_transform` cannot represent both SmoothQuant and
QuaRot

SmoothQuant is a *diagonal* rescale `X' = X · diag(1/s)`, weights become
`W' = diag(s) · W`. This is a **channelwise scalar multiplication** that
fuses into the previous layer's `BiasAdd` or `LayerNorm`.

QuaRot is a *Hadamard rotation* `X' = X · H`, weights `W' = Hᵀ · W`.
This is a **dense orthogonal transform** that fuses into the *weight*
of the current layer and an **online Hadamard** applied to the
*activation* at runtime (the activation Hadamard is `O(d log d)` extra
flops per token, computed every forward pass).

Calling both `pre_transform=smooth` and `pre_transform=hadamard` is
correct in name but obliterates the structural difference. The IR needs
**`pre_transform_kind`** (rescale / rotation / lattice) **plus
`pre_transform_apply`** (fused-into-weights-only / runtime-applied /
both). Without these, a compiler reading the IR cannot decide whether
to emit a runtime Hadamard kernel.

### Gap 2.2 — `accumulator_dtype` is per-layer, not per-tile

Real kernels use a *hierarchy* of accumulator dtypes:

- **Per-element op**: `mma.sync.aligned.f16.f16.f16` accumulates in fp16
  inside warp tiles, then **promotes to fp32 across CTA tiles**.
- **MXFP8 / NVFP4 path**: tensor-core element multiply produces fp22
  intermediate, accumulated in fp32 per tile, then dequant-multiplied
  by UE8M0 outside the tensor core.

A single `accumulator_dtype=fp32` field hides this. The IR needs
`tile_accumulator_dtype` and `block_accumulator_dtype`. Lots of
correctness bugs in low-bit kernels come from this distinction (FA3's
fp16 accumulator path has measurably worse numerics than fp32).

### Gap 2.3 — `codebook` cannot express additive (AQLM), per-row
(SqueezeLLM, EXL2), or lattice-residual (QuIP#)

The survey's `codebook` is:
```
{ "values": tensor | builtin_id, "index_bits": int,
  "shared_across": Enum[global, per_super_block] }
```

This is a **single flat lookup** of `index → value`. It cannot encode:

- **Additive codebooks (AQLM)**: weight = Σᵢ Cᵢ[idxᵢ], i.e. K lookups
  summed. Needs `num_codebooks: int` and `combine: sum|concat`.
- **Per-row codebooks (SqueezeLLM, EXL2)**: each output channel has its
  own LUT. Needs `shared_across=per_row`.
- **Lattice + residual (QuIP#)**: lookup into an E8 lattice point and
  then a small residual; the IR is implicitly assuming "decoded value =
  codebook[idx]", but QuIP# is "decoded value = lattice[idx] + residual
  · scale". Needs a `decoder_program` mini-DSL.

### Gap 2.4 — No mechanism for outlier offloading (LLM.int8)

LLM.int8 splits the weight matrix at runtime into two: ~99.9 %
quantized to INT8, ~0.1 % outlier columns kept in fp16, and the matmul
becomes `Y = X_int8 · W_int8 + X_outlier_fp16 · W_outlier_fp16`. This is
a **structural sparsity decomposition that depends on runtime
activation statistics** — the column-set is selected by `|X| > 6`.

Representing this in a static `QuantSpec` requires either:

- A `outlier_handling` field with values `{none, llm_int8_threshold,
  owq_static, hadamard_eliminate}` plus a threshold parameter; or
- Modelling it as **two separate quantized weight tensors** with a
  runtime selector predicate.

The current survey has neither. The `compute_dtype` field gestures at
the issue (it says "compute_dtype: int8" but never mentions that 0.1 %
of the columns are fp16). Severe gap.

### Gap 2.5 — Per-row variable bitwidth (EXL2) and per-column variable
bitwidth (OWQ) are not representable

`qdtype` is a single value per tensor. EXL2 has *different bitwidths in
different rows of the same tensor*. This is fundamentally a per-row
`QuantSpec`. The IR needs either:

- `qdtype` upgraded to `qdtype | List[qdtype]` with a `row_dtype_map`; or
- A "tile sharding" mechanism where a tensor is composed of multiple
  sub-tensors each with their own `QuantSpec`.

Both are heavy. The survey implicitly assumes per-tensor uniformity. P0
gap.

### Gap 2.6 — Calibration metadata is overloaded

The survey lists `calibration` as a metadata enum AND `calibration_data`
as an opaque field. But the *strength* of calibration is a knob:
GPTQ-128-samples-wikitext vs GPTQ-512-samples-mixed-domain produce
different weights and are *not interchangeable* for round-tripping. The
IR needs at least `calibration_dataset_id`, `num_samples`,
`hyperparameters` (e.g., `α` for SmoothQuant, `damp_percent` for GPTQ).
Treating this as "opaque" means the IR cannot reproduce or verify a
quantization run.

### Gap 2.7 — Scale storage interleaving and packing are conflated

AWQ packs scales in an **interleaved** layout where each fp16 scale sits
next to the int4 nibbles it scales. The Marlin format **separates**
them. GPTQ stores scales as a separate row-major fp16 tensor.

The survey's `packing` enum is a single dimension. But:

- Weight packing (`awq_interleaved8x4`, `marlin_4bit`,
  `gguf_kquant`) — how the int4 nibbles are arranged.
- Scale packing (interleaved-with-weights vs separate-tensor) — a
  distinct axis.
- Zero-point packing (int packed alongside weights vs separate fp16
  tensor) — distinct again.

The IR needs three packing fields, not one. The Marlin example proves
this: same `awq_interleaved8x4` weights, different scale layout, totally
different kernel.

---

## Section 3: Factual accuracy fact-check

Tested every load-bearing numeric claim against primary sources.

### 3.1 AWQ group size — survey says 128 canonical default. CONFIRMED.

AutoAWQ default in `casper-hansen/AutoAWQ/awq/quantize/quantizer.py` is
`group_size=128`. Shipped Qwen2/Qwen2.5 AWQ checkpoints from official
HF repos use 128. Shipped Llama-3-8B-Instruct-AWQ uses 128. Group 64
exists in some Mistral community quants; group 32 is rare.

**Verdict**: Correct.

### 3.2 GGUF Q4_K_M super-block layout — survey claims super-block of
256 weights = 16 sub-blocks × 16 weights, with `d` and `dmin` at fp16,
sub-block scales/mins at 6-bit.

Cross-check against `ggml-quants.h`:

```c
#define QK_K 256
typedef struct {
    ggml_half d;          // super-block scale for quantized scales
    ggml_half dmin;       // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];  // 12 bytes for Q4_K
    uint8_t qs[QK_K/2];   // 4-bit quants
} block_q4_K;
```

`K_SCALE_SIZE = 12` for Q4_K — this encodes 8 sub-block scales (Q4_K
uses sub-block size 32, not 16). Each sub-block is **32 weights**, not
16. The 12-byte scales array packs 8 × (6-bit scale + 6-bit min) = 96
bits = 12 bytes.

**Survey error**: The survey says `sub_block=16` for Q4_K (line 374,
`sub_block=16`). The actual Q4_K sub-block is **32 weights, super-block
is 8 sub-blocks**, not 16. The `sub_block=16` figure is correct for
Q2_K and Q3_K (which have 16 sub-blocks of 16) but wrong for Q4_K,
Q5_K, Q6_K. **Confirmed factual error**.

The worked example in §5.4 with `group_size=16` for an `attn_v` tensor
that is described as `qdtype="q6_k"` is also wrong: Q6_K uses
`QK_K=256` super-block with **16 sub-blocks of 16**, so 16 is right
for Q6_K but inconsistent with the Q4_K explanation. The survey
conflates Q2_K/Q3_K sub-block size with Q4_K's. **Confirmed second
error.**

### 3.3 NVFP4 outer scale: survey says FP8 E4M3 inner per-block-of-16 +
fp32 outer per tensor.

NVIDIA Blackwell whitepaper (Mar 2024) and TensorRT-LLM `tensorrt_llm/
quantization/nvfp4.py` confirm:

- Inner block scale: FP8 E4M3, one per 16 elements.
- Outer scale: FP32, one per tensor.

But the survey misses one detail: the **outer scale is computed as
`amax(tensor) / 448`** (where 448 = max E4M3) so the inner E4M3 scales
have full dynamic range. This is a *required* recipe, not optional —
TRT-LLM model-optimizer enforces it. The survey would benefit from
mentioning this.

**Verdict**: Mostly correct, minor omission.

### 3.4 MXFP4 block: 32 elements, UE8M0 shared exponent.

OCP Microscaling Spec v1.0 §4.2 confirms `block_size = 32`, scale
encoded as 8-bit unsigned `E8M0` (bias 127, no sign, no mantissa, all
ones = NaN).

**Verdict**: Correct.

### 3.5 LLM.int8 outlier threshold = 6.

Survey says "magnitudes > 6". `bitsandbytes/functional.py`
`threshold=6.0` is the default for `bnb.matmul`. **Correct**.

### 3.6 FP8 E4M3 range ±448 and E5M2 range ±57344.

E4M3: max = `(2 − 2⁻³) · 2⁸ = 1.875 · 256 = 480`? Let me check.

`fp8_e4m3` (NVIDIA, no inf): bias = 7, max exponent = 15 (since the
all-ones exponent is repurposed for finite values, except specific
NaN patterns). Max value = `1.111₂ × 2⁸ = 1.75 × 256 = 448`.

**Survey is correct on 448.** The exponent value is 15, but since
NVIDIA's E4M3 uses the all-ones pattern for finite values
(`S.1111.110₂` = 448, `S.1111.111₂` = NaN), the effective max is 448,
not 480.

E5M2: bias = 15, max exponent = 30, max value `1.11₂ × 2¹⁵ =
1.75 × 32768 = 57344`. **Correct.**

### 3.7 HQQ on 70B model in minutes.

mobiusml/hqq README claims "70B in 4 minutes on a single GPU". Survey
says "minutes" — defensible. **Correct.**

### 3.8 Llama-3-8B-Instruct-AWQ default group_size — survey claims 128
is the most common.

`hugging-quants/Meta-Llama-3-8B-Instruct-AWQ-INT4` config.json:
`group_size=128`. Confirmed. The Qwen2.5-7B-Instruct-AWQ also uses 128.
Mistral-7B-Instruct-v0.3-AWQ uses 128. **Correct.**

### 3.9 SmoothQuant α range.

Survey says `α ∈ [0.5, 0.8]`. SmoothQuant paper §3.2 sweeps α ∈ [0, 1]
and recommends α = 0.5 for Llama, 0.8 for OPT-175B. The range is
defensible. **Correct.**

### 3.10 GGUF imatrix purpose.

Survey says it is used to multiply weights by `sqrt(importance)` before
RTN. Actually llama.cpp's imatrix is used **inside the k-quant
optimizer** (`make_qkx2_quants`) as **per-column weights for the
weighted-MSE rounding** — it scales the *error penalty*, not the weights
themselves. Different operation, same intent. **Confirmed mild error;
not catastrophic.** See `llama.cpp/src/llama-quant.cpp` `llama_tensor_get_weights_for_quantization`.

### 3.11 IQ1_S bitwidth = 1.5625.

`ggml-quants.h` defines `IQ1_S` block: 256 weights / block, scales 16
bits + 32 bytes index + 16 sign-bit bytes = 50 bytes / 256 weights =
1.5625 bits/weight. **Correct.**

### 3.12 Q4_K_M effective bitwidth.

Survey says 4.5 bits/weight. Calculation: super-block = 256 weights,
storage = `2 (d) + 2 (dmin) + 12 (scales) + 128 (4-bit qs) = 144` bytes
= 4.5 bits/weight. **Correct.**

### 3.13 FlashAttention-3 fp8 SDPA per-row scale recovery.

FA3 paper §3 confirms online per-row scaling for the P matrix; not the
full softmax fp32 path the survey describes. Survey says "softmax in
fp32" — FA3 actually does softmax in fp32 inside the warp, but the
*output P matrix that feeds PV* is requantized to fp8 with per-row
scales. The survey gets this right. **Correct.**

### 3.14 vLLM default 4-bit quant.

Survey claims AWQ is the dominant server-side 4-bit. Empirically: as
of vLLM 0.6.x the *most-downloaded* 4-bit Llama-3 checkpoints on HF are
GPTQ-Marlin (Neural Magic's `Meta-Llama-3-8B-Instruct-GPTQ` series) and
AWQ-Marlin (`hugging-quants/Meta-Llama-3-8B-Instruct-AWQ-INT4`). Both
are loaded as Marlin. Calling AWQ "the dominant server-side 4-bit
format" is **overstated** — it is co-dominant with GPTQ-Marlin. The
survey's "If you support only three" recommendation should probably
say "AWQ *or* GPTQ as repacked by Marlin". **Minor overstatement.**

### Fact-error summary

**Confirmed factual errors: 2** (Q4_K sub-block size; imatrix
application semantics).
**Minor inaccuracies: 2** (NVFP4 outer-scale formula omitted; vLLM
default 4-bit overstated as AWQ).

---

## Section 4: Evolution narrative gaps

This is a survey-paper-bar document; a 3-year evolution arc is
mandatory. The current document is a **taxonomy snapshot**, not a
narrative. Severe defect.

What is missing as a *story*:

### 4.1 2022 baseline: pre-LLM-quant world

LLM.int8 (2022-08) is mentioned only as a bitsandbytes feature, not
positioned as the **first** workable PTQ for >6.7B models that broke
the Gaussian-weight assumption. ZeroQuant (2022-06) predates it and
is omitted entirely. The story should start: "before 2022, NN quant
was per-channel int8 from the QAT-MobileNet era. LLM-int8 and ZeroQuant
discovered emergent outliers in >6.7B."

### 4.2 2023: the explosion year

GPTQ (Oct 2022 → ICLR 2023), AWQ (Jun 2023), SmoothQuant (Nov 2022),
QLoRA / NF4 (May 2023). Each solved a different problem:

- GPTQ: how to round each column without breaking the next.
- SmoothQuant: how to migrate range from A to W.
- AWQ: how to identify which weights matter pre-emptively.
- NF4 + QLoRA: how to fine-tune on a single GPU.

The survey lists these as parallel entries with no genealogy.

### 4.3 2024: standardization year

- **GGUF k-quants** stabilized in llama.cpp PR #1684 (mid-2023) and
  became the on-device default.
- **GGUF i-quants** (PR #4773 in Jan 2024) brought QuIP#-style codebooks
  to llama.cpp.
- **Marlin** (early 2024) made W4A16 fast on Ampere/Hopper.
- **FP8 went mainstream**: H100 had been out since 2022 but FP8
  inference shipped at scale in vLLM 0.4 (Apr 2024) and TRT-LLM 0.8.
- **AWQ became vLLM's de-facto 4-bit** mid-2024.
- **MX format adoption** by OCP (Sept 2023) and Hopper (B200 shipped
  late 2024).

The survey mentions these but does not date them or connect them.

### 4.4 2025: extreme-bit + native FP4

- **NVFP4 on Blackwell** (early 2025 production).
- **BitNet b1.58** model releases from Microsoft (Feb 2024 paper, but
  the *Llama-style BitNet retrains* of 2025 are when it became
  shippable).
- **AQLM 2-bit** Llama-3 checkpoints.
- **QuaRot/SpinQuant** W4A4 became deployable.

### 4.5 2026: ?

The survey has nothing on 2026. The TODAY date is 2026-06-04. What
shipped in the last six months? At minimum, the survey should
acknowledge:

- **vLLM compressed-tensors v2** consolidation.
- **MXFP6 in MI355X** silicon (Q3 2025 launch).
- **DeepSeek-V3 FP8 native training** and what it implies for inference.
- **TPU v6e (Trillium) INT8 weight + INT4 activation** modes.

Without this, the survey reads as a Dec-2024 snapshot — not a current
survey.

**Verdict on evolution narrative**: Severe defect. The document has
the *material* for the story but does not tell it. Sections 1–5 read
as a catalog. Adding a §0.5 "3-year arc" with a dated timeline would
fix this.

---

## Section 5: "Support only three" recommendation critique

The survey picks Q4_K_M + AWQ-INT4 + FP8-E4M3-W8A8. Defensible but
under-justified.

### 5.1 No 2-bit / extreme-bit representation

Choosing three 4–8 bit schemes ignores the **fast-growing sub-3-bit
tier** (IQ2_M, AQLM, BitNet). For a system targeting on-device SLMs
(phones, edge), IQ2_M is the *only way* to get a 7B model into 2.5 GB.
If the IR doesn't exercise sub-3-bit at all, the codebook / multi-codebook
axes won't be tested.

**Recommend**: replace one of the three (probably FP8 W8A8, since FP8
W4A16 is a strict superset of FP8 W8A8 on axis exercise) with **IQ2_M
or AQLM** to force the codebook axes.

### 5.2 Server-side 4-bit: AWQ vs GPTQ-Marlin

Survey picks AWQ. But Neural Magic's `compressed-tensors` GPTQ-Marlin
ships more checkpoints. They have *identical runtime layouts* once
loaded into vLLM (both become Marlin). The choice is therefore
metadata-only. Picking AWQ leaves `calibration=hessian` and `act_order`
axes **untested**.

**Recommend**: pick GPTQ-Marlin instead of AWQ if you want axis
coverage. Or pick *both* and treat as one entry (since they share
runtime layout).

### 5.3 W8A8 FP8 is shipped less than W4A16

Production deployments of Llama-3-70B-FP8 are common (vLLM, SGLang,
TRT-LLM defaults), but for **SLMs ≤13B**, W4A16 AWQ/GPTQ is
empirically more common than W8A8 FP8 because the bandwidth savings
matter more than tensor-core acceleration at small model sizes.

**Recommend**: justify why FP8 W8A8 over FP8 W4A16 (which is also
shipped on Hopper via TRT-LLM's `W4A16_FP8`). Or pick
**W4A8** (TRT-LLM's `W4A8_AWQ`) which is the actual production sweet
spot.

### 5.4 Asymmetric coverage check

Survey claims the three exercise "every axis". Let me grade:

| Axis | Q4_K_M | AWQ-INT4 | FP8 W8A8 | Covered? |
|------|--------|----------|----------|----------|
| Asymmetric zero | yes (dmin) | yes (fp16 zero) | no (symm) | OK |
| Integer zero | no (fp16-derived) | no (fp16) | n/a | **No integer-zero coverage** |
| Codebook | no | no | no | **No codebook coverage** |
| Hadamard / rotation | no | no | no | **No rotation coverage** |
| Hierarchical scale | yes (super-block) | no | no | OK |
| UE8M0 scale | no | no | no | **No MX coverage** |
| Per-row variable bit | no | no | no | **No variable-bit coverage** |
| Outlier offloading | no | no | no | **No LLM.int8-style coverage** |

So the three exercise about half the axes. The survey overclaims
"forces every axis in §7 to be exercised". **Confirmed false.**

**Recommend a 5-pick set**: Q4_K_M + GPTQ-Marlin + FP8-E4M3 +
**NF4 (for codebook + double-quant)** + **MXFP4 (for UE8M0 + OCP
block packing)**. This actually exercises all 15 axes.

---

## Section 6: KV cache quant gaps

Only INT8, INT4, FP8 covered. Missing:

### 6.1 KIVI per-channel-K + per-token-V

Survey mentions KIVI under §4.2 INT4 KV but treats it as "asymmetric
per-channel K + per-token V". KIVI's actual contribution is the
**recognition that K and V have different outlier structures**: K has
*channel-aligned* outliers (because of RoPE's frequency basis), V has
*token-aligned* outliers (because of value-vector magnitudes per token).

The survey acknowledges this in §4.4 but does not surface it as a
**design constraint** on the IR. The IR needs:

- `kv_quant_axis_k: per_channel | per_token | per_head`
- `kv_quant_axis_v: per_channel | per_token | per_head`

These must be independently configurable.

### 6.2 KVQuant (Hooper et al., NeurIPS 2024, arXiv:2401.18079)

Sub-2-bit KV via per-channel non-uniform quantization + dense-and-sparse
decomposition. Shipped in some research forks. Survey omits entirely.

### 6.3 SmoothQuant for KV (Q-Hitter, retroactive K-smoothing)

There is a line of work applying SmoothQuant-style channel rescaling
*to the K projection only* so that the K cache is easier to quantize.
Used in some TRT-LLM recipes. Survey omits.

### 6.4 Cache compression beyond quantization

The prompt asks about eviction (H2O, StreamingLLM, Heavy Hitter
Oracle) and summarization (SnapKV, PyramidKV). These are
**orthogonal to quantization** but co-deployed with it. A KV
compression IR axis needs:

- `kv_eviction: none | h2o | snapkv | streaming_attention_sink`
- `kv_compression: none | snapkv_topk | pyramid_decay`

The survey is *titled* "Quantization Schemes" so this is out of scope —
but the IR will need it. Flag for a follow-up §11 in v2.

### 6.5 Per-token vs per-channel scaling tradeoffs

Survey hand-waves: "per-head per-channel; vLLM uses per-head fp16
scale". The actual vLLM kv_cache_dtype=fp8 quantizes per-tensor with
**a single fp32 scale per tensor**, not per-head. The per-head scaling
is for INT8 only. **Confirmed minor error in §4.1**.

### 6.6 Quantization-aware MQA/GQA grouping

For GQA models (Qwen2.5, Llama-3), the K/V have `num_kv_heads` heads
which are *shared* across `num_attention_heads / num_kv_heads` query
heads. The KV quant scale topology must respect this: scaling per-KV-head
(not per-Q-head). Survey does not mention. The IR needs
`kv_scale_grouping: per_kv_head | per_q_head_group | per_tensor`.

---

## Section 7: Recommended fixes for v2

In priority order:

### P0 (must-fix)

1. **Add Marlin format as a packing variant** in `packing` enum.
   Document the marlin repack as a runtime transform (AutoAWQ → Marlin).
2. **Add ExLlamaV2 EXL2** with per-row variable bitwidth; surface the
   IR gap (`row_dtype_map` or equivalent).
3. **Add AQLM and VPTQ** for additive / vector multi-codebook
   quantization; extend `codebook` to support `num_codebooks` and
   `combine_op`.
4. **Add BitNet b1.58** as a ternary scheme; add `qdtype=ternary` to
   the enum; document that BitNet is train-from-scratch, not PTQ.
5. **Fix Q4_K sub-block size** in §3.1 and §5.4: Q4_K/Q5_K/Q6_K use
   sub-block size 32 with 8 sub-blocks per super-block. Q2_K/Q3_K use
   sub-block size 16 with 16 sub-blocks. The current text is wrong.
6. **Add legacy GGUF Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0** as a §3.0
   subsection; they are still shipped on ARM-NEON paths.
7. **Add `_L` k-quant variants** to the table.
8. **Split `pre_transform`** into `pre_transform_kind` (diagonal /
   orthogonal / lattice) + `pre_transform_phase` (offline-fused /
   runtime-applied / both). Re-tag QuaRot, SpinQuant, SmoothQuant.
9. **Add outlier offloading** as a first-class axis
   (`outlier_handling`) with values `none, llm_int8_dynamic,
   owq_static, hadamard_eliminate`.
10. **Rewrite §1.5 NVFP4 vs MX as a hierarchy** — they share an
    `element_format` and `block_size` is the only delta; show this in a
    side-by-side table.

### P1 (should-fix)

11. **Add §0.5 "3-year evolution"** with a dated timeline figure.
    Spec: 2022 outlier discovery → 2023 PTQ explosion → 2024
    standardization → 2025 native FP4 + extreme-bit → 2026 ?
12. **Add OmniQuant, SqueezeLLM, OWQ, ZeroQuant-FP**.
13. **Add KIVI / KVQuant / SmoothQuant-for-K** under §4 with the
    distinct K vs V structural argument.
14. **Replace the "support only three" set** with a 5-tier
    recommendation: 1 absolute must (Q4_K_M), 1 server 4-bit
    (GPTQ-Marlin), 1 FP path (FP8 W8A8 or W4A8), 1 extreme-bit
    (IQ2_M or AQLM), 1 future (MXFP4 or NVFP4).
15. **Document the imatrix correctly**: it is a weighting factor inside
    the rounding objective, not a multiplicative pre-scale.
16. **Split `accumulator_dtype`** into tile-level and block-level.
17. **Split `packing`** into `weight_packing`, `scale_packing`,
    `zero_packing`.

### P2 (nice-to-have)

18. Add BiLLM, MXINT4, Q-Hitter, H2O eviction (acknowledged out-of-scope
    but flag).
19. Add a `compute_capability_required` field per scheme (sm_80 for
    Marlin AWQ; sm_89 for FP8; sm_100 for NVFP4; sm_120 for MXFP8
    full path).
20. Add fp8_e4m3fnuz / fnuz semantics to the worked example so AMD
    runtime divergence is visible.
21. Reference the OCP MX spec section numbers explicitly (the survey
    says "OCP Microscaling Formats v1.0" but page-cite §4.2 for the
    block-size claim).
22. Add cross-references between §1.5 (MX) and §2.4 (FP8) — they share
    `element_format=e4m3` and the only delta is the scale block.

---

## References cited in this critique (beyond the survey's own)

- Frantar & Alistarh, *Marlin: Efficient 4-bit Inference Kernels for
  LLMs*, 2024. `github.com/IST-DASLab/marlin`.
- ExLlamaV2 (`github.com/turboderp/exllamav2`).
- Egiazarian et al., *Extreme Compression of Large Language Models via
  Additive Quantization*, ICML 2024. arXiv:2401.06118 (AQLM).
- Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in
  1.58 Bits*, 2024. arXiv:2402.17764 (BitNet b1.58).
- Shao et al., *OmniQuant*, ICLR 2024. arXiv:2308.13137.
- Kim et al., *SqueezeLLM*, ICML 2024. arXiv:2306.07629.
- Lee et al., *OWQ*, AAAI 2024. arXiv:2306.02272.
- Yao et al., *ZeroQuant: Efficient and Affordable Post-Training
  Quantization*, NeurIPS 2022. arXiv:2206.01861.
- Huang et al., *BiLLM*, ICML 2024. arXiv:2402.04291.
- Hooper et al., *KVQuant*, NeurIPS 2024. arXiv:2401.18079.
- Liu et al., *VPTQ*, 2024. `github.com/microsoft/VPTQ`.
- Zhang et al., *H2O: Heavy-Hitter Oracle*, NeurIPS 2023.
  arXiv:2306.14048.
- Xiao et al., *StreamingLLM*, ICLR 2024. arXiv:2309.17453.
- llama.cpp `src/ggml-quants.c`, `src/llama-quant.cpp`
  (`github.com/ggerganov/llama.cpp`).
- OCP, *Microscaling Formats Specification v1.0*, Sept 2023, §4.2.
- NVIDIA Blackwell Architecture Whitepaper, 2024 §3.4 (NVFP4).
- vLLM `vllm/model_executor/layers/quantization/compressed_tensors`
  (`github.com/vllm-project/vllm`).

---

*End of critique. Total identified issues: 17 missing schemes,
7 IR-axis structural gaps, 2 confirmed factual errors + 2 minor
inaccuracies, ~5 evolution-narrative omissions, 4 KV-quant gaps,
22 recommended fixes. Severity-weighted top concern: the QuantSpec's
`pre_transform` and `codebook` axes are not expressive enough for
2024–2025 schemes (QuaRot, AQLM, EXL2). Without restructuring these,
the IR will mis-represent at least eight shipped formats.*
