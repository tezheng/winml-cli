# Quantization Schemes in Shipped SLM Runtimes — v2

> Survey for the `llm-layers` IR project. Goal: enumerate every quantization
> scheme actually shipped in mainstream small-language-model runtimes between
> mid-2022 and mid-2026 (llama.cpp, vLLM, TensorRT-LLM, OpenVINO, MLX, MLC,
> Hugging Face Transformers, AutoGPTQ/AutoAWQ, ExLlamaV2, bitsandbytes, ONNX
> Runtime, Intel Neural Compressor, KleidiAI, T-MAC/bitnet.cpp, NVIDIA TRT
> Model-Optimizer, AMD Quark, Neural Magic compressed-tensors), trace the
> 3-year evolution arc, and distill the minimum parameter axes a representation
> must expose to cover them.

v2 changelog vs v1: +17 schemes (Marlin family, EXL2, AQLM, QuIP/QuIP#,
SqueezeLLM, OmniQuant, BiLLM, BitNet b1.58, QuaRot/SpinQuant proper,
MXINT4, AFP8, AutoGPTQ vs GPTQ distinction, ZeroQuant V1/V2/FP, OWQ,
KleidiAI LUT, legacy GGUF Q4_0..Q8_0, k-quant `_L` sub-variants, IQ1
variants), fix Q4_K sub-block size (32 weights × 8 sub-blocks, not 16×16),
fix imatrix semantics (per-column MSE weighting inside the k-quant rounder,
not pre-multiplied `sqrt(importance)`), split `pre_transform` into
`rotation_kind` + `runtime_apply`, extend `codebook` with `num_codebooks`
+ `combine_op` + `shared_across`, add `variable_bitwidth` /
`outlier_offload` / `marlin_layout` / `imatrix_calibrated` axes, expand the
"support 3" claim to a "support 5" recommendation, add KV-cache extensions
(KIVI, KVQuant, CacheGen, SmoothQuant-for-K), add a brief QAT vs PTQ
section, and a six-era evolution narrative (2022 H2 → 2026 Q1).

---

## §1. Introduction and three-year evolution narrative

LLM quantization between mid-2022 and mid-2026 went through six identifiable
eras. Each era was defined by a *bottleneck removed* — never simply by a
clever paper. The arc is not "lower bits over time"; it is *which engineering
constraint was binding this quarter*.

This survey first tells the timeline (§1), then catalogs every scheme inside
its family (§3–§9), then distills the IR axes (§10), then lands a concrete
recommendation (§11). The evolution narrative is also the answer to "why
does the IR need so many axes?" — each axis was forced by a specific scheme
in a specific quarter.

### 1.1 Era 0 — Pre-LLM-quant (≤ 2022 H1)

Before mid-2022, neural-net quantization was a MobileNet/QAT story: 8-bit
weight + 8-bit activation, per-channel symmetric, calibrated by percentile
or KL on ImageNet. Vision models tolerated this because their weight and
activation distributions were Gaussian-ish.

LLMs broke that assumption. OPT-175B, GPT-J, BLOOM had *emergent outliers* —
a small fraction of activation channels (~0.1 %) with magnitudes 20–50× the
typical channel. Off-the-shelf int8 saturated at those channels and
collapsed perplexity. This is the technical fact that all six eras below
respond to.

### 1.2 Era 1 — 2022 H2 / 2023 H1: PTQ explosion (arXiv:2206 → arXiv:2306)

Four papers (June 2022 → June 2023) defined the modern PTQ landscape:

- **ZeroQuant V1** (Yao et al., NeurIPS 2022, arXiv:2206.01861, June 2022).
  Group-wise INT8 weight + per-token INT8 activation, with layer-by-layer
  knowledge-distillation rectification. *First* to ship W8A8 for OPT-style
  models at scale through DeepSpeed-Inference. ZeroQuant-V2 (arXiv:2303.08302,
  Mar 2023) added INT4 weights and LoRC (low-rank compensation) for
  outlier-heavy layers. ZeroQuant-FP (arXiv:2307.09782, Jul 2023) generalized
  to FP8/FP4.
- **LLM.int8()** (Dettmers et al., NeurIPS 2022, arXiv:2208.07339, Aug 2022).
  Discovered the outlier-feature phenomenon and proposed *runtime
  decomposition*: keep the ~0.1 % outlier columns in fp16, quantize the rest
  to int8, sum the two GEMMs. Shipped in `bitsandbytes`. First scheme to
  losslessly quantize 175B-class models.
- **GPTQ** (Frantar et al., ICLR 2023, arXiv:2210.17323, Oct 2022).
  Closed-form per-column Hessian-OBS quantization. Reduced quantization
  *time* (a 175B in ~4 hours on a single A100) without calibration sets
  larger than 128 samples. Defined the "calibration-set PTQ" reference.
- **SmoothQuant** (Xiao et al., ICML 2023, arXiv:2211.10438, Nov 2022).
  Migrated dynamic range from activations to weights via a *diagonal*
  channelwise rescale. Made W8A8 work without LLM.int8's runtime outlier
  split. Stays the canonical channel-smoothing reference.
- **AWQ** (Lin et al., MLSys 2024, arXiv:2306.00978, June 2023).
  Identified "salient" weight channels using activation magnitudes from a
  small calibration set, upscaled those channels before RTN. Pushed
  weight-only INT4 perplexity to near-fp16. Became the *server-side*
  4-bit reference.

Why this clustering happened: H100 had launched (Sept 2022) but its INT8
and FP8 tensor cores were idle without a software story. Open-source
checkpoints (LLaMA 1 in Feb 2023, LLaMA 2 in Jul 2023) suddenly let
academics fine-tune and quantize models the closed labs would not release.
The result: PTQ became dominant for inference, while QAT (training-time
quantization) remained niche because nobody wanted to spend pre-training
compute on quantization-friendly weights they could not validate.

Side-thread in 2023 H1: **QLoRA / NF4** (Dettmers et al., NeurIPS 2023,
arXiv:2305.14314, May 2023). NormalFloat-4: a codebook quantization where
the 16 codepoints are sampled at the quantiles of `N(0,1)` (since weights
are approximately Gaussian after layer-norm). The headline result was
QLoRA itself — fine-tuning a 65B in NF4 + LoRA on a single 48 GB GPU. As
a *quantization* contribution NF4 introduced the **codebook with quantile
spacing**, plus *double-quant* (second-level int8 quantization of the
fp32 absmax scales). NF4 stays a bitsandbytes-only format but is the
dominant codebook scheme cited by everything that followed.

### 1.3 Era 2 — 2023 H2: kernel revolution (GGUF k-quants, Marlin, AWQ-kernels)

By mid-2023 the PTQ papers had outrun the kernels. AutoGPTQ used naive
Triton matmuls and was 2–5× slower than fp16 at small batch. The fixes
came from three independent directions in roughly the same quarter:

- **GGUF k-quants** (Kawrakow et al., llama.cpp PR #1684, June 2023).
  Replaced the legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 formats (32-weight blocks
  with one fp16 scale) with a *hierarchical* super-block-of-256 layout
  where the per-sub-block scales are themselves quantized to 6 bits and
  share an fp16 super-scale. This made Q4_K_M / Q5_K_M *the* on-device
  format because the lower memory bandwidth for the metadata mattered as
  much as for the weights.
- **AWQ-kernels** (the `awq_gemm` GEMM in AutoAWQ, Sept 2023). Used the
  `lop3.b32` PRMT trick from FasterTransformer to dequantize 8×int4 into
  fp16 in 4 instructions, enabling a 4× speedup over Triton.
- **Marlin** (Frantar & Alistarh, `IST-DASLab/marlin`, late 2023).
  Re-tiled weights into `(K/16, N/64, 16, 64)` blocks with a zig-zag
  interleave so a single `mma.sync.aligned.m16n8k16` could dequant 4-bit
  weights via PRMT *inside* the tensor-core mma path. Achieved near
  fp16-mma throughput at INT4. Marlin became the reference W4A16 kernel
  on Ampere/Hopper.

These three were not papers — they were kernels, shipped first, papers
sometimes never. They mattered because they decoupled "scheme storage" from
"scheme runtime". The IR consequence: *the same scheme can have multiple
storage layouts*. AutoAWQ stores W4A16 in its 8×4 interleave; vLLM's
`awq_marlin` repacks it into the Marlin layout at load time. Same numeric
result, different bytes on disk.

### 1.4 Era 3 — 2024 H1: server quant standardization

By Q1 2024 a consensus emerged on the server side:

- **AWQ-Marlin becomes vLLM's de-facto 4-bit path** for SLMs ≤ 13B. The
  HuggingFace `hugging-quants` org publishes AWQ-INT4 checkpoints for every
  Llama-3 release within days of the open release. Group-size = 128 is
  the canonical default.
- **GPTQ-Marlin** (the kernel binding GPTQ-quantized checkpoints to the
  Marlin runtime layout) ships in vLLM 0.4 (Apr 2024). Neural Magic's
  `Meta-Llama-3-8B-Instruct-GPTQ-W4A16` becomes the most-downloaded W4A16
  checkpoint on HF. AWQ and GPTQ become *co-dominant* at run time — they
  share the runtime layout once loaded.
- **FP8 lands in production**. vLLM 0.4 and TRT-LLM 0.8 expose
  `fp8_e4m3` per-tensor-W + per-token-A as a checkpoint format. H100
  tensor cores hit ~2× fp16 throughput. SGLang's FP8 attention follows.
- **GGUF i-quants** (llama.cpp PRs #4773 and #5104, Jan 2024). Kawrakow
  ports QuIP#-style lattice codebooks into the GGUF format as the IQ1/2/3
  family. The headline scheme is `IQ2_M` (~2.7 bits/weight) and
  `IQ2_XXS` (~2.06 bits/weight); these are the reason a 7B-class model
  can fit in 2.5 GB on a phone.

The IR consequence: *the same checkpoint format must serve both server
and on-device deployments*. GGUF is dominant on device but absent from
server; AWQ/GPTQ-Marlin dominant on server but rare on device. The IR
must represent both without forcing a re-quantization to cross the
boundary.

### 1.5 Era 4 — 2024 H2: 2-bit production push + KV-cache maturation

Two-bit weight quantization went from "research curiosity" to
"deployable" in Q3-Q4 2024 via three converging schemes:

- **AQLM** (Egiazarian et al., ICML 2024, arXiv:2401.06118, Jan 2024).
  Additive multi-codebook quantization: each weight is a sum of K codebook
  entries from L codebooks. Llama-2-70B at 2.07 bits achieved
  near-Llama-7B-fp16 perplexity. Shipped in HF Transformers via the
  `aqlm` integration.
- **QuIP / QuIP#** (Chee et al., NeurIPS 2024, arXiv:2307.13304 + Tseng
  et al., arXiv:2402.04396, Feb 2024). Lattice (E8) codebook + Hadamard
  *incoherence processing* (randomized rotation that makes weight and
  Hessian distributions isotropic). QuIP# 2-bit Llama-2-70B was the
  first public 2-bit model to round-trip through HF + production runtimes.
- **GGUF IQ2_XXS / IQ2_M** (Jan–Apr 2024 PR series). Brought QuIP#-style
  lattices to llama.cpp with the imatrix calibration step. By Q4 2024,
  LM Studio's default 7B download for users under 4 GB RAM became
  IQ2_M; OpenAI's GPT-OSS released in IQ2_M for low-end machines.

In parallel, KV-cache quantization matured:

- **KIVI** (Liu et al., arXiv:2402.02750, Feb 2024) showed that K and V
  have *different outlier topologies* — K outliers are channel-aligned
  (because RoPE projects channels onto frequencies), V outliers are
  token-aligned (because attention writes value vectors per token). KIVI
  quantizes K per-channel int4, V per-token int4, with fp16 residual
  buffer for the most recent few tokens.
- **KVQuant** (Hooper et al., NeurIPS 2024, arXiv:2401.18079) pushed KV
  to sub-2-bit with non-uniform per-channel quantization plus
  dense-and-sparse decomposition for outliers.
- vLLM 0.5 (Jul 2024) shipped `kv_cache_dtype=fp8_e4m3` and
  `kv_cache_dtype=fp8_e5m2` as production options.

### 1.6 Era 5 — 2025 H1: MX format adoption and Blackwell native FP4

The Open Compute Project's *Microscaling Formats Specification v1.0*
(Sept 2023) defined MXFP4 / MXFP6 / MXFP8 / MXINT4 / MXINT8 — formats
where a block of 32 elements shares a single UE8M0 (unsigned 8-bit
exponent) scale. Hardware adoption took 18 months:

- **NVIDIA Blackwell** (B200 sampled Q4 2024, GA Q1 2025). 5th-gen tensor
  cores natively execute MXFP8 and a *non-OCP* variant **NVFP4** (4-bit
  E2M1 elements with a *per-block-of-16* FP8 E4M3 inner scale plus a
  per-tensor fp32 outer scale — two-level, finer block, richer scale
  dtype than MXFP4).
- **AMD MI355X** (GA Q3 2025) supports MX in MFMA, including MXFP6
  E3M2/E2M3.
- **Intel Gaudi-3** (GA Q2 2025) supports OCP MX.
- **TensorRT-LLM 0.14+** exposes `nvfp4` quant algo; vLLM 0.6+ loads
  NVFP4 checkpoints from TRT-Model-Optimizer; SGLang follows.
- **OpenAI GPT-OSS** (released Dec 2024) ships weights in MXFP4.
  This is the first widely-used SLM in MXFP4 as a checkpoint format
  (not just a runtime convert-on-load path).

The IR consequence: scales themselves must be quantizable, and the scale
dtype enum must include UE8M0 and FP8 E4M3 as first-class. This is the
single biggest axis addition forced by 2025.

### 1.7 Era 6 — 2025 H2: BitNet revival, ternary production

BitNet b1.58 (Ma et al., arXiv:2402.17764, Feb 2024) was a research
paper for most of 2024. Ternary weights ∈ {-1, 0, +1} (1.58 bits ≈
log₂ 3), INT8 activations, trained from scratch. The headline claim —
that BitNet-3B matched Llama-3B perplexity at one-third the memory —
was reproducible but the ecosystem lacked CPU kernels for ternary
matmul that would beat int8.

The 2025 H2 inflection point: **Microsoft T-MAC and bitnet.cpp** shipped
LUT-based ternary GEMV kernels on ARM (Snapdragon X, Apple M-series,
Cortex-X4) that beat llama.cpp's Q4_K_M on the same model size. By
Q4 2025, bitnet.cpp ran 3B BitNet at >100 tok/s on a Snapdragon X laptop
with no GPU. **KleidiAI** (Arm's open-source SLM kernel library,
Q3 2025) added LUT-based 4-bit ternary fused dequant via FlatBuf-packed
codebooks, giving Cortex-X4 a path to BitNet without Microsoft's
custom kernel.

The IR consequence: ternary is *not* int1 — three values, not two — and
must be a distinct `qdtype`. LUT-packing for ternary is also a distinct
packing variant.

### 1.8 Era 7 — 2026 Q1: consolidation, MXFP6, compressed-tensors v2

The most recent six months (Dec 2025 → May 2026) have been
consolidation, not new schemes:

- **vLLM `compressed-tensors v2`** unifies AWQ/GPTQ/SmoothQuant/FP8 under
  a single checkpoint format with explicit per-tensor `QuantizationScheme`
  records. Major checkpoint repositories on HuggingFace are migrating.
- **MXFP6 in MI355X silicon** is now exposed in PyTorch via `torchao` and
  in vLLM 0.7. Real SLM checkpoints (Qwen3, DeepSeek-V3 distillations)
  ship in MXFP6.
- **DeepSeek-V3** (Dec 2024, refined Mar 2025) trained natively in FP8
  end-to-end. Its checkpoint inherits FP8 weights directly; inference
  runtimes do not re-quantize. This raises the QAT-vs-PTQ boundary: a
  natively-FP8-trained checkpoint is functionally a QAT artifact even
  though no explicit quantization-aware loss was used.
- **TPU v6e (Trillium)** adds INT8 weight + INT4 activation modes
  exposed through XLA HLO. Google has not published the kernel signature
  but the JAX `aqt` library targets it.

What is still open in Q1 2026: a portable spec for QuaRot/SpinQuant
rotation matrices (so that two runtimes loading the same checkpoint
apply the *same* Hadamard); a converged 2-bit format (AQLM, QuIP#,
IQ2_M, VPTQ are all production but mutually incompatible); and the
question of whether MXINT4 will displace NVFP4 once non-NVIDIA hardware
catches up.

---

## §2. Methodology and categorization framework

### 2.1 What is recorded per scheme

For each scheme we record twelve fields:

1. **Provenance**: paper, repo, ship date, lead author.
2. **Bitwidth + numeric type** of stored weights and activations.
3. **Grouping**: per-tensor, per-channel (output / input axis), per-token,
   blockwise (with N).
4. **Scale dtype**: fp16, bf16, fp32, UE8M0, FP8 E4M3, INT8, INT6.
5. **Zero-point**: symmetric (none) or asymmetric (offset present), and
   the dtype of the zero.
6. **Calibration**: none / weight-only RTN / activation-aware / Hessian-OBS
   / imatrix-MSE-weighted / QAT / learnable.
7. **Storage layout**: packed nibble order, interleave, super-block,
   Marlin tile, IQ codebook indices.
8. **Runtimes shipped**: which engines load this scheme today.
9. **Matmul kernel signature**: `(W_q, A) -> Y` with dtypes and
   accumulator.
10. **Pre-transform**: none, diagonal SmoothQuant rescale, dense
    Hadamard QuaRot, dense learned SpinQuant, randomized
    incoherence (QuIP#).
11. **Outlier handling**: none, runtime fp16 column split (LLM.int8),
    static fp16 column reserve (OWQ, Atom), pre-rotation elimination
    (QuaRot/QuIP#), KV residual buffer (KIVI).
12. **Parameter-space axes** touched by this scheme (cross-ref to §10).

### 2.2 Family taxonomy

Schemes group into seven families that exhibit largely orthogonal axes:

- **Family A** — Weight-only INT/FP, weights compressed and dequantized
  (or fused-mul) at inference; activations remain bf16/fp16. The
  dominant family by deployment count (§3).
- **Family B** — Weight + Activation (W*A*), compute happens in
  low-precision tensor cores (§4).
- **Family C** — GGUF k-quants and i-quants, llama.cpp's storage format
  for on-device deployment (§5).
- **Family D** — Codebook quantization (NF4, AQLM, QuIP/QuIP#, VPTQ,
  SqueezeLLM, EXL2), where decoded values come from a lookup table not
  a uniform grid (§6).
- **Family E** — KV-cache quantization, with K and V often treated
  differently (§7).
- **Family F** — Attention-internal quantization (FP8 SDPA, FA3,
  INT8 attention), where intermediates inside the kernel are
  quantized (§8).
- **Family G** — Sub-2-bit (BitNet ternary, BiLLM 1-bit), often
  requiring training-from-scratch (§9).

A scheme can span families (NF4 is Family A *and* Family D; QuaRot is
Family B *and* introduces a Family A pre-transform). We list each scheme
under its dominant family with cross-references.

### 2.3 Citation conventions

Inline citations use arXiv ID or repo path. Section §14 collects URLs.

---

## §3. Family A — Weight-only INT/FP quantization

The most populous family. ~16 distinct schemes.

### 3.1 GPTQ (Frantar et al., arXiv:2210.17323, Oct 2022)

- **Bitwidth/dtype**: INT4 (most common), INT3, INT2, INT8.
- **Grouping**: blockwise along K with `group_size ∈ {-1, 32, 64, 128, 1024}`;
  128 default. `desc_act=True` reorders columns by Hessian-diagonal
  magnitude; stores a `g_idx` mapping.
- **Scale dtype**: fp16; zero-point INT packed in `qzeros`.
- **Zero-point**: integer, asymmetric.
- **Calibration**: PTQ Hessian-OBS. `H = 2 X Xᵀ / n` from ~128 samples;
  closed-form `δ = -(w_q − w) / H⁻¹_ii · H⁻¹_:,i` per column.
- **Storage**: int32-packed `W`, fp16 row-major `scales`, int4-packed
  `qzeros`, optional int32 `g_idx[K]`.
- **Runtimes**: AutoGPTQ (original toolchain), GPTQModel (fork),
  vLLM `gptq` and `gptq_marlin`, TRT-LLM `W4A16_GPTQ`, HF Transformers
  + Optimum, ExLlamaV2 (custom layout), OpenVINO via NNCF.
- **AutoGPTQ vs upstream GPTQ distinction**: upstream GPTQ refers to the
  original `frantar/gptq` reference. AutoGPTQ is the
  PanQiWei/AutoGPTQ packaging that became the de-facto checkpoint
  toolchain (introduced the `g_idx` packing convention, fp16 scales
  layout, and `desc_act` flag). GPTQModel is the late-2024 fork that
  added a `--marlin-format` flag and an alternate `g_idx` packing
  with reduced metadata overhead. **IR consequence**: `g_idx`
  packing variant is a load-time axis even when the numeric result
  is identical.

### 3.2 AWQ (Lin et al., arXiv:2306.00978, June 2023)

- **Bitwidth/dtype**: INT4 asymmetric.
- **Grouping**: blockwise along K with `group_size ∈ {32, 64, 128}`;
  128 canonical default.
- **Scale dtype**: fp16 (scale + fp16 zero per group).
- **Zero-point**: fp16-valued.
- **Calibration**: PTQ activation-aware. Computes per-channel activation
  magnitudes from ~128 calibration samples, derives per-channel scaling
  `s` folded into the previous layer's weights, then RTN-quantizes.
- **Storage**: 8×int4 per int32 word with interleave
  `[0,2,4,6,1,3,5,7]` so two fp16 lanes of `mma.sync` dequantize 8
  values via one `lop3.b32` PRMT.
- **Runtimes**: AutoAWQ, vLLM (`awq`, `awq_marlin`), TRT-LLM
  `W4A16_AWQ`, HF Transformers, MLX 4-bit-groups, MLC. llama.cpp loads
  via GGUF conversion (re-quantized to Q4_K).
- **Kernel**:
  `awq_gemm(W: int4[N,K/8] packed, scales: fp16[N,K/G],
  zeros: fp16[N,K/G], A: fp16[M,K]) -> fp16[M,N]` with fp32 (or fp16
  on some paths) accumulator.

### 3.3 HQQ — Half-Quadratic Quantization (Badri & Shaji, Nov 2023)

- **Bitwidth/dtype**: INT8 / INT4 / INT3 / INT2 / INT1; asymmetric.
- **Grouping**: blockwise along K, `group_size ∈ {8, 32, 64, 128, 256}`;
  64 default.
- **Scale dtype**: fp16 or bf16; zeros fp16.
- **Calibration**: **none** — solves
  `min ‖W − Q⁻¹(Q(W; s, z))‖_p` with `p < 1` via half-quadratic
  splitting using weights only. Famous for "70B in 4 minutes".
- **Runtimes**: HQQ lib, HF Transformers, vLLM experimental, MLX-HQQ.
- **IR axes**: GPTQ-like, plus `calibration=data_free`,
  `optimizer=half_quadratic`.

### 3.4 bitsandbytes NF4 (Dettmers et al., arXiv:2305.14314, May 2023)

- **Bitwidth/dtype**: 4-bit codebook with 16 levels sampled at
  quantiles of `N(0,1)`. Codebook is fixed:
  `{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0}`.
- **Grouping**: group_size = 64 along the flattened weight axis.
- **Scale dtype**: fp32 absmax per group; *double-quant* compresses
  those scales to int8 with another fp32 scale per 256 group-scales.
- **Zero-point**: implicit (codebook is asymmetric around zero but
  includes a level at zero).
- **Calibration**: data-free per-block absmax.
- **Storage**: packed nibbles row-major; separate fp32 absmax array;
  optional int8-compressed scales.
- **Runtimes**: bitsandbytes, HF Transformers (the canonical QLoRA
  path), vLLM `bitsandbytes`, TGI, peft.
- **Kernel**:
  `bnb_matmul_4bit(W_q: u4[N,K/2], absmax: fp32[N·K/64], A: fp16[M,K])
  -> bf16/fp16[M,N]`, dequant via 16-entry LUT, accumulator fp32.

### 3.5 bitsandbytes FP4

- Same group/scale layout as NF4 but the 16 codepoints are non-IEEE
  fp4 `E2M1`: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` ± sign.
- Slightly worse perplexity than NF4 in practice; rarely used.

### 3.6 OmniQuant (Shao et al., ICLR 2024, arXiv:2308.13137, Aug 2023)

- **Bitwidth/dtype**: W2 / W3 / W4, optionally W6.
- **Grouping**: per-channel-out, optional group along K.
- **Scale dtype**: fp16.
- **Zero-point**: integer asymmetric.
- **Calibration**: *learnable* — Learnable Weight Clipping (LWC) +
  Learnable Equivalent Transformation (LET). LET is a per-channel
  diagonal scale + shift that is learned (1k iterations of SGD on a
  reconstruction loss); LWC learns per-channel clipping bounds for the
  quantization range. Bridges PTQ and QAT: no end-to-end backprop, only
  per-layer reconstruction.
- **Runtimes**: `OpenGVLab/OmniQuant`, HF integration; not in vLLM
  mainline. ExLlamaV2 supports OmniQuant-derived weights.
- **IR axes**: forces `calibration=learnable_clipping`.

### 3.7 SqueezeLLM (Kim et al., ICML 2024, arXiv:2306.07629, Jun 2023)

- **Bitwidth/dtype**: ~3 bits effective.
- **Grouping**: *per-row codebook* — each output channel has its own
  k-means codebook with 8 or 16 entries.
- **Outlier handling**: top ~0.05 % of weights (the most
  sensitivity-weighted) extracted into a sparse fp16 side-buffer; the
  matmul becomes `Y = dense_codebook(W) · A + sparse_fp16(W_outlier) · A`.
- **Calibration**: weighted by Hessian sensitivity (Fisher-style).
- **Runtimes**: `SqueezeAILab/SqueezeLLM`; HF integration; ExLlamaV2
  partial.
- **IR axes**: forces `codebook.shared_across = per_row` and
  `outlier_offload = FP16_COLUMN_SPLIT`.

### 3.8 OWQ — Outlier-aware Weight Quantization (Lee et al., AAAI 2024, arXiv:2306.02272, Jun 2023)

- **Bitwidth/dtype**: INT3 or INT4 for most columns, fp16 for the
  small fraction (~0.5 %) of "weak" columns identified by an OBS
  sensitivity criterion.
- **Grouping**: per-channel-out for the int part, fp16 for the
  reserved columns.
- **Calibration**: Hessian-OBS like GPTQ to identify weak columns and
  do the OBS update on the remainder.
- **Runtimes**: Intel Neural Compressor, some HF integrations.
- **IR axes**: forces `outlier_offload = FP16_COLUMN_SPLIT` with a
  *static* (rather than runtime) column selection — different mode than
  LLM.int8's dynamic split.

### 3.9 ZeroQuant V1/V2/FP (Yao et al., arXiv:2206.01861, arXiv:2303.08302, arXiv:2307.09782)

- **V1 (June 2022)**: group-wise INT8 W + per-token INT8 A; layer-by-layer
  knowledge-distillation reconstruction. First W8A8 to ship for GPT-J / OPT
  at scale through DeepSpeed-Inference.
- **V2 (Mar 2023)**: INT4 weights + LoRC (Low-Rank Compensation) — a
  rank-32 fp16 residual `W_q + L · Rᵀ` added to recover sensitive
  layers.
- **FP (Jul 2023)**: generalizes V1/V2 to FP8/FP4 elements; the LoRC
  residual stays in fp16.
- **Runtimes**: DeepSpeed-Inference (`ds_zero_quant` config), partial
  HF integration. Not in vLLM.
- **IR axes**: forces a `lora_residual: Optional[Tensor]` field on
  weights (or representable as a separate fp16 weight tensor).

### 3.10 EXL2 — ExLlamaV2 variable bitwidth (`turboderp/exllamav2`, Sept 2023)

- **Bitwidth/dtype**: *variable per row*. Bitwidths drawn from
  `{2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8}` and **mixed within a single
  weight tensor**. Effective average bitwidth is set by an
  activation-error budget.
- **Grouping**: blockwise along K with `group_size ∈ {32, 128}`.
- **Scale dtype**: fp16 per group.
- **Calibration**: PTQ with activation-error measurement per row; rows
  with low sensitivity drop to 2-bit, sensitive rows stay at 5 or 6 bits.
- **Storage**: Q-matrix layout with a per-row bit-budget header and
  packed nibbles of varying width.
- **Runtimes**: ExLlamaV2, ExLlamaV3 partial, TabbyAPI, text-gen-webui.
- **Kernel**: custom EXL2 `q4_matmul` with per-row dispatch.
- **IR axes**: forces `variable_bitwidth: True` and a
  `bitwidth_per_row: int[N]` axis. **This is the single biggest axis
  forced by 2023 H2.**

### 3.11 AQLM — Additive Quantization (Egiazarian et al., ICML 2024, arXiv:2401.06118)

- **Bitwidth/dtype**: 2-bit effective.
- **Codebook structure**: weight = Σᵢ₌₁..K Cᵢ[idxᵢ], where Cᵢ are L
  trained codebooks of shape `(2^index_bits, group_size)`. Canonical
  config: 2 codebooks × 16-bit indices × group_size 8 = 4 weight bits
  per 8 weights = 0.5 bits/weight before scale overhead; with
  super-block fp16 scales the effective rate is ~2 bits/weight.
- **Pre-transform**: optional QuIP-style incoherence (Hadamard).
- **Calibration**: codebooks and indices jointly optimized by
  block-coordinate descent on a Hessian-weighted reconstruction loss.
- **Runtimes**: HF Transformers `aqlm` integration; `Vahe1994/AQLM` repo;
  not yet in vLLM mainline. Llama-2-70B 2-bit AQLM checkpoints ship on HF.
- **IR axes**: forces `codebook.num_codebooks > 1` and
  `codebook.combine_op = ADD`.

### 3.12 QuIP and QuIP# (Chee et al., arXiv:2307.13304; Tseng et al., arXiv:2402.04396)

- **QuIP (NeurIPS 2023)**: weight quantization with **incoherence
  processing** — multiply weights and Hessians by random orthogonal
  matrices (or Hadamards) to make distributions isotropic before
  rounding. 2-bit quantization with quality comparable to GPTQ at 4-bit.
- **QuIP# (Feb 2024)**: adds an **E8 lattice codebook** for the
  rounded vectors. E8 is the Gosset lattice in 8D, optimal for
  Gaussian quantization. Combined with a randomized Hadamard rotation
  this gave the first reproducible 2-bit Llama-2-70B with sub-1-point
  perplexity loss.
- **Runtimes**: `Cornell-RelaxML/QuIP` and `Cornell-RelaxML/quip-sharp`
  reference, HF integration. The E8 lattice was ported into llama.cpp
  as the i-quant IQ family (see §5.2).
- **IR axes**: forces `rotation_kind = HADAMARD_RUNTIME` and
  `codebook.combine_op = MULTI_LATTICE` with an
  `incoherence_processing: bool` flag.

### 3.13 QuaRot (Ashkboos et al., arXiv:2404.00456, Apr 2024)

- **Bitwidth/dtype**: W4A4 INT4 weights + INT4 activations (also W4A8).
- **Pre-transform**: **dense Hadamard rotation**, *fused* offline into
  weights for the parts that can absorb it (output of MLP, input of next
  layer); *online* Hadamard applied to activations at runtime via a
  fast Walsh-Hadamard transform (`O(d log d)` flops per token).
- **Calibration**: PTQ. The Hadamard makes activation distributions
  approximately isotropic, eliminating outliers, enabling clean INT4
  activation quantization.
- **Runtimes**: `spcl/QuaRot` reference, HF research integration; not in
  vLLM mainline. Meta-internal SpinQuant deployments for Llama-3 use
  similar ideas.
- **IR axes**: forces `rotation_kind = DENSE_HADAMARD_QUAROT`,
  `runtime_apply = INPUT_OUTPUT_BOTH`. The rotation matrix is *not*
  arbitrary — it is a Walsh-Hadamard, so it can be represented by a
  single bit "rotation = H_d" rather than a dense matrix.

### 3.14 SpinQuant (Liu et al., arXiv:2405.16406, May 2024)

- **Bitwidth/dtype**: W4A4 (also W4A8).
- **Pre-transform**: **dense learned orthogonal rotation**. Unlike
  QuaRot's fixed Hadamard, SpinQuant learns a rotation that minimizes
  quantization error using a small set of calibration samples and
  Riemannian optimization on the Stiefel manifold.
- **Runtimes**: Meta-internal first; Meta's open SpinQuant code at
  `facebookresearch/SpinQuant`. Shipped for Llama-3-8B and Llama-3-70B
  W4A8 in production at Meta.
- **IR axes**: forces `rotation_kind = DENSE_LEARNED_SPINQUANT`. The
  rotation matrix must be stored as a tensor; cannot be represented by a
  single Hadamard flag.

### 3.15 BiLLM (Huang et al., ICML 2024, arXiv:2402.04291, Feb 2024)

- **Bitwidth/dtype**: 1-bit weights (binary {-1, +1}).
- **Structure**: salient-channel preservation — the top ~5 % of weights
  (selected by Hessian sensitivity) stay fp16 or int4, the rest binarize.
- **Calibration**: Hessian-aware binarization with structural search.
- **Runtimes**: `Aaronhuang-778/BiLLM` reference; not in mainstream
  inference runtimes.
- **IR axes**: forces `qdtype = int1` and
  `outlier_offload = FP16_COLUMN_SPLIT`.

### 3.16 Legacy GGUF Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 (llama.cpp, 2022 H2 → 2023 H1)

These predate k-quants and are still shipped on ARM-NEON paths and on
mobile NPUs that lack k-quant kernels.

- **Q4_0**: 32-weight block, 1 fp16 scale, symmetric. 4-bit
  weights packed as nibbles. `ggml_block_q4_0` struct.
  `bits_per_weight = (4·32 + 16) / 32 = 4.5`.
- **Q4_1**: 32-weight block, 1 fp16 scale, 1 fp16 min, asymmetric.
  `bits_per_weight = 5`.
- **Q5_0**: 32-weight block, 1 fp16 scale, symmetric, 5-bit weights
  packed as nibble + high-bit plane. `bits_per_weight = 5.5`.
- **Q5_1**: Q5_0 with asymmetric min. `bits_per_weight = 6`.
- **Q8_0**: 32-weight block, 1 fp16 scale, symmetric int8 weights.
  `bits_per_weight = 8.5`. The reference "high-bit" format and the
  intermediate activation format `Q8_K` derives from this.
- **Runtimes**: llama.cpp on every backend; Ollama; LM Studio; bitnet.cpp
  fallback path; KleidiAI Q4_0 fused dequant on Arm.
- **IR axes**: `qdtype = q4_0|q4_1|q5_0|q5_1|q8_0`,
  `super_block = 32` (not 256), `sub_block = None`,
  `imatrix_calibrated = False` (these formats predate imatrix).

### 3.17 KleidiAI LUT-based 4-bit (Arm, Q3 2025)

- **Source**: ARM-software/KleidiAI repo. Used by llama.cpp on Cortex-X4
  and Snapdragon X.
- **Idea**: precompute a 16-entry LUT for each weight block at load
  time (a permutation of the Q4_0 or Q4_K values) so the dequant becomes
  a single SIMD table-lookup (`TBL` instruction on NEON). The LUT is
  *FlatBuf-packed* into the model file for portability.
- **Bitwidth/dtype**: effectively 4-bit, but the *layout* is LUT not
  packed nibble.
- **Runtimes**: llama.cpp via KleidiAI backend; Arm Compute Library
  >= 24.10.
- **IR axes**: forces `packing = lut_flatbuf` as a separate variant
  from `gguf_kquant`.

### 3.18 Marlin format (IST-DASLab/marlin, late 2023 / GA Q1 2024)

- **Bitwidth/dtype**: INT4 weights.
- **Layout**: re-tiles `W: int4[N, K]` into `(K/16, N/64, 16, 64)`
  blocks with a per-block zig-zag interleave. Each block aligns with one
  `mma.sync.aligned.m16n8k16` instruction. The zig-zag interleave
  ensures the PRMT-based dequant produces values in the order the mma
  expects without an extra permutation.
- **Scale dtype**: fp16. **Scales stored separately** from weights —
  unlike AWQ where scales sit next to nibbles, Marlin keeps scales in a
  separate row-major fp16 tensor for cache locality.
- **Calibration**: inherits from the *source* scheme (AWQ or GPTQ).
  Marlin is a kernel + storage layout, not a quantization algorithm.
- **Runtimes**: vLLM `awq_marlin`, vLLM `gptq_marlin`,
  Neural Magic compressed-tensors. **On load, vLLM repacks
  AutoAWQ-format checkpoints into Marlin layout**. This is one of the
  reasons "AWQ" and "GPTQ-Marlin" run identically once loaded — they
  share the Marlin layout post-repack.
- **IR axes**: forces `packing = marlin_4bit` as a distinct variant
  from `awq_interleaved8x4`.

### 3.19 Marlin-24 — 2:4 sparse + 4-bit (IST-DASLab/marlin, Q2 2024)

- Sparse variant that combines Marlin's 4-bit format with 2:4 structured
  sparsity (every 4 consecutive weights have exactly 2 non-zeros).
  Exploits Ampere/Hopper sparse mma (`mma.sp.sync.aligned`).
- 2× compute throughput vs dense Marlin on 2:4-sparse-friendly layers
  (mostly FFN); requires the weights to be trained or fine-tuned with
  2:4 sparsity (or post-pruned).
- **Runtimes**: vLLM `marlin_24`, Neural Magic `compressed-tensors`.
- **IR axes**: forces `packing = marlin_24` plus a `sparsity_pattern`
  field.

### 3.20 GPTQ-Marlin (kernel binding)

Not a new quantization scheme — a runtime path where GPTQ-quantized
checkpoints are repacked into Marlin layout and executed by the Marlin
kernel. The IR consequence is that the "source scheme" (GPTQ) and the
"runtime layout" (Marlin) must be separate axes.

### 3.21 MX formats — MXFP4 / MXFP6 / MXFP8 / MXINT4 / MXINT8 (OCP, Sept 2023)

- **Spec**: Open Compute Project *Microscaling Formats v1.0*, Sept 2023
  §4.2. Rouhani et al. arXiv:2310.10537.
- **Common structure**: block of 32 elements shares one 8-bit
  **UE8M0** (unsigned 8-bit exponent, no sign, no mantissa, bias 127,
  all-ones = NaN) scale. Element types differ per variant:
  - **MXFP8**: FP8 E4M3 or E5M2 elements.
  - **MXFP6**: FP6 E3M2 or E2M3 elements.
  - **MXFP4**: FP4 E2M1 elements.
  - **MXINT8**: INT8 elements + UE8M0 scale.
  - **MXINT4**: INT4 elements + UE8M0 scale. Less common than MXFP4 in
    shipped checkpoints but exposed by MI355X and Gaudi-3.
- **Block size**: 32, fixed by the spec.
- **Calibration**: data-free per-block absmax for weights; activation
  scales computed on-the-fly per 32 elements.
- **Runtimes**:
  - **MXFP8** on Blackwell tensor cores (TRT-LLM 0.13+); torchao MX
    recipes. AMD MI355X via MFMA.
  - **MXFP6** on MI355X (since Q3 2025); torchao 0.5; vLLM 0.7
    experimental.
  - **MXFP4** on Blackwell; GPT-OSS ships in MXFP4; torchao.
  - **MXINT8** as a vLLM experimental KV-cache dtype.
  - **MXINT4** in torchao 0.6+, MI355X kernels.
- **IR axes**: `qdtype = mxfp4|mxfp6|mxfp8|mxint4|mxint8`,
  `scale_dtype = ue8m0`, `group_size = 32`, `packing = ocp_mx`,
  `element_format ∈ {e2m1, e3m2, e2m3, e4m3, e5m2, int4, int8}`.

### 3.22 NVFP4 (NVIDIA Blackwell, GA Q1 2025)

- **Bitwidth/dtype**: 4-bit `E2M1` (same as MXFP4) but with a
  **two-level scale**:
  1. Inner block scale, fp8 `E4M3`, one per 16 elements.
  2. Outer per-tensor scale, fp32.
- **Outer scale formula**: `outer = amax(W) / 448` (where 448 = max
  E4M3), so the inner E4M3 scales have full dynamic range. TRT-Model-
  Optimizer enforces this recipe.
- **Calibration**: PTQ; per-tensor amax for the outer scale; inner
  E4M3 scales per-block absmax. AWQ-style activation scaling is
  optional.
- **Runtimes**: TRT-LLM `nvfp4`, TRT 10.5+, NIM for Blackwell;
  vLLM 0.6.4+ loads NVFP4 from TRT-Model-Optimizer; SGLang.
- **Kernel**: 5th-gen tensor-core `mma.sync` with fp4 operands, fp8
  scales, bf16 accumulator (selectable to fp32).
- **IR axes**: `qdtype = nvfp4`, `group_size = 16`,
  `scale_dtype = fp8_e4m3`, `outer_scale_dtype = fp32`,
  `packing = nvfp4`.

### 3.23 AFP8 — asymmetric FP8

A category, not a single scheme. Some shipped fp8 paths add a fp16 zero
to the fp8 weight (so it behaves like fp8 + offset) for layers with
strongly skewed distributions. Used in Intel Gaudi-3 paths and
some TRT-LLM smoothquant-fp8 recipes (`fp8_e4m3` weights with per-channel
`bias_offset`). Not standardized; flag for IR.

### 3.24 VPTQ — Vector Post-Training Quantization (`microsoft/VPTQ`, 2024)

- **Bitwidth/dtype**: 1.5–2 bits effective.
- **Codebook**: trained vector quantization on weight blocks of size
  4–8; codebook entries are 4-8-dimensional vectors.
- **Calibration**: Hessian-weighted reconstruction.
- **Runtimes**: VPTQ repo + HF integration; ships Llama-3-70B at 1.5-2
  bits.
- **IR axes**: forces `codebook.combine_op = NONE` (single vector
  lookup) with `codebook.values` being a `(num_entries, vec_dim)`
  tensor, not scalar.

### 3.25 LLM.int8() (Dettmers, arXiv:2208.07339, Aug 2022)

- **Bitwidth/dtype**: INT8 weights + INT8 activations with per-row
  scaling. Outlier feature columns (~0.1 %, magnitude > 6) decomposed
  to fp16. The matmul:
  `Y = X_int8 · W_int8_per_row_scale + X_outlier_fp16 · W_outlier_fp16`.
- **Calibration**: implicit, runtime threshold-driven decomposition.
- **Runtimes**: bitsandbytes only.
- **IR axes**: forces `outlier_offload = FP16_COLUMN_SPLIT` with
  *runtime* (dynamic) column selection, distinguished from OWQ's
  *static* selection.

(Note: LLM.int8 is W+A but the activation handling is the headline; it
sits at the family A/B boundary.)

---

## §4. Family B — Weight + Activation quantization

These schemes quantize both sides of the matmul. The compute uses
low-precision tensor cores.

### 4.1 SmoothQuant (Xiao et al., arXiv:2211.10438, Nov 2022)

- **Bitwidth/dtype**: W8 INT8, A8 INT8.
- **Grouping**: W per-channel (output axis), A per-token dynamic or
  per-tensor static.
- **Scale dtype**: fp32 stored / fp16 runtime.
- **Calibration**: PTQ. Compute per-channel activation magnitudes
  `s_j = max|X_:,j|^α / max|W_j,:|^(1-α)` with `α ∈ [0.5, 0.8]`.
  Divide A by `s`, multiply W by `s` (diagonal channelwise) to migrate
  range from A (hard to quantize) to W (easy).
- **Rotation kind**: `DIAGONAL_SMOOTHQUANT`. The pre-multiplication is a
  diagonal matrix `diag(s)`, fused into the previous layer's `BiasAdd`
  or `LayerNorm`.
- **Runtimes**: TRT-LLM (`smoothquant` preprocessing + W8A8 GEMM),
  OpenVINO via NNCF (`smoothquant_alpha`), Intel Neural Compressor,
  Qualcomm AIMET, vLLM via `compressed-tensors`.
- **IR axes**: `qdtype = int8`, `w_quant_axis = N`,
  `a_quant_granularity ∈ {per_token, per_tensor}`,
  `rotation_kind = DIAGONAL_SMOOTHQUANT`, `runtime_apply = NONE`.

### 4.2 Vanilla INT8 W8A8 (dynamic and static)

- **Dynamic per-token**: activations quantized at inference with
  per-token absmax. No calibration.
- **Static per-tensor**: activation amax frozen from calibration; allows
  scale to be folded into bias.
- **Runtimes**: TRT-LLM `W8A8`, vLLM `compressed-tensors`, OpenVINO,
  ONNX Runtime `QLinearMatMul`, TFLite full-int8, ExecuTorch, MLX.

### 4.3 ZeroQuant W8A8 / W4A8 (see §3.9)

Listed in Family A for the weight side but shipped as W*A* paths in
DeepSpeed-Inference.

### 4.4 QuaRot W4A4 (Apr 2024, see §3.13)

W4A4 with **dense Hadamard rotation**. Pre-transform is *not* diagonal —
this is the structural difference from SmoothQuant. The activation
Hadamard is computed *at runtime* via a fast Walsh-Hadamard kernel.

### 4.5 SpinQuant W4A4 / W4A8 (May 2024, see §3.14)

W4A4 with **dense learned orthogonal rotation**. Rotation matrix is a
trained tensor, not a Hadamard pattern.

### 4.6 FP8 — E4M3 / E5M2 (Micikevicius et al., arXiv:2209.05433)

- **Bitwidth/dtype**: 8-bit float.
  - `E4M3`: 4 exp / 3 mantissa, range ±448 (NVIDIA convention: no inf,
    all-ones-exp repurposed for finite values). Forward path (W, A).
  - `E5M2`: 5 exp / 2 mantissa, range ±57344, IEEE-like with inf/NaN.
    Gradient path in training.
- **Grouping**: per-tensor (TE default), per-row/per-channel (vLLM,
  TRT-LLM), per-token (activations).
- **Scale dtype**: fp32 (amax-history-based delayed scaling in TE) or
  fp16.
- **Calibration**: delayed scaling (TE EMA), static amax (vLLM), or
  dynamic (per-token).
- **Runtimes**: NVIDIA Hopper (H100/H200) native FP8 in tensor cores;
  Blackwell adds MXFP8/NVFP4; AMD MI300 supports FP8 with `fnuz`
  variants; Intel Gaudi-2/3 supports FP8 (Gaudi-3 is OCP-compliant);
  TRT-LLM `fp8`, vLLM `fp8` per-tensor or per-channel, SGLang `fp8`,
  CUTLASS, cuBLAS-Lt.
- **Sub-variants**:
  - `fp8_e4m3fn` (NVIDIA): no inf, no -0; range ±448.
  - `fp8_e4m3fnuz` (AMD): finite, no unsigned zero, different bias;
    byte-incompatible with NVIDIA.
  - `fp8_e5m2`, `fp8_e5m2fnuz` (same NVIDIA vs AMD distinction).
- **Deployment configs**: distinct *checkpoints*, not just kernel modes:
  - vLLM `fp8` default: per-tensor W + per-token A. Most common
    SLM-FP8 checkpoint.
  - TRT-LLM `fp8` default: per-channel W + per-token A. More common
    in NVIDIA's NIM containers.
  - TE `fp8` delayed: per-tensor W + per-tensor A with EMA amax. Used
    in training; less common at inference.
- **IR axes**: forces 4-way FP8 enum (`fp8_e4m3fn`, `fp8_e4m3fnuz`,
  `fp8_e5m2`, `fp8_e5m2fnuz`), plus
  `a_quant_granularity ∈ {per_tensor, per_token, delayed_per_tensor}`.

### 4.7 W4A8 — INT4 weights + INT8 activations

TRT-LLM `W4A8_AWQ` and `W4A8_QSERVE` (Lin et al., arXiv:2405.04532, May
2024), ExLlamaV3 partial. Combines AWQ-style weight quantization with
int8 activation quantization. Int32 accumulator. Practical on Hopper
because int8 tensor cores are fast. The 2024 H2 production sweet spot
for SLM inference on H100/H200 once memory bandwidth saturated.

### 4.8 INT4 W4A4 — beyond QuaRot/SpinQuant

Direct W4A4 without rotation is rare in shipped runtimes; perplexity
degrades sharply. Some Apple MLX recipes expose 4-bit activations for
on-device inference where the Apple Neural Engine has 4-bit tensor
paths.

---

## §5. Family C — GGUF k-quants and i-quants

GGUF is llama.cpp's storage format. K-quants (mid-2023) and i-quants
(early 2024) are the *de facto* deployment quantization for CPU and
Apple-Silicon SLM inference. Source of truth: `ggml/src/ggml-quants.c`
and `src/llama-quant.cpp` in `ggerganov/llama.cpp`.

### 5.1 K-quant super-block structure — CORRECTED LAYOUT

A **super-block** is 256 weights, denoted `QK_K = 256`. The sub-block
size is **not the same across all k-quants** — this is the v1 error,
corrected here.

| Variant | Sub-block size | Sub-blocks per super-block | Sub-scale bits | Bits/weight |
|---------|----------------|-----------------------------|-----------------|-------------|
| Q2_K    | 16             | 16                          | 4-bit sc + 4-bit min | 2.5625 |
| Q3_K    | 16             | 16                          | 6-bit sc       | 3.4375 |
| Q4_K    | **32**         | **8**                       | 6-bit sc + 6-bit min | 4.5 |
| Q5_K    | **32**         | **8**                       | 6-bit sc + 6-bit min | 5.5 |
| Q6_K    | 16             | 16                          | 8-bit sc       | 6.5625 |
| Q8_K    | 16             | 16                          | fp32 sc        | 8.5 (intermediate, not stored) |

Cross-checked against `ggml-quants.h`:

```c
#define QK_K 256
typedef struct {
    ggml_half d;          // super-block scale for quantized scales
    ggml_half dmin;       // super-block scale for quantized mins
    uint8_t scales[12];   // K_SCALE_SIZE = 12 for Q4_K
                          // packs 8 × (6-bit sc + 6-bit min) = 96 bits
    uint8_t qs[QK_K/2];   // 128 bytes of 4-bit quants
} block_q4_K;
// total: 2 + 2 + 12 + 128 = 144 bytes per 256 weights = 4.5 bits/weight
```

The 8 sub-blocks × 32 weights = 256 layout for Q4_K, Q5_K is what makes
`K_SCALE_SIZE = 12`. The 16 sub-blocks × 16 weights layout of Q2_K, Q3_K,
Q6_K uses different packing (`scales[16]` for Q6_K, or 4-bit packed
scales+mins for Q2_K).

The dequant for an element `w_q` in sub-block `j` of Q4_K:

```
w = d · scale_j · q − dmin · min_j
```

where `scale_j` and `min_j` are 6-bit unsigned integers decoded against
fp16 `d`, `dmin`.

### 5.2 K-quant _S / _M / _L sub-variants

`Q4_K_M` vs `Q4_K_S` vs `Q4_K_L`: the suffixes denote *per-layer
quantization-recipe overrides*:

- **`_S` (small)**: Q4_K uniformly for all linear layers.
- **`_M` (medium)**: upgrades `attn.v` and `ffn.down` projections (the
  most quantization-sensitive layers empirically) to Q6_K. Q4_K
  elsewhere.
- **`_L` (large)**: upgrades `attn.v`, `ffn.down`, and additionally
  `attn.k`, `ffn.up` to Q6_K. Distinct from `_M`. Often the LM Studio
  default for memory-constrained users who can spare ~5 % more.

Analogously for Q3_K_S/M/L, Q5_K_S/M, etc. The IR must support a
**per-tensor pattern → QuantSpec override** map, not just a single
scheme.

### 5.3 K-quant calibration — imatrix CORRECTED SEMANTICS

The v1 statement was: "weights are multiplied by `sqrt(importance)`
before RTN". This is **wrong**. The correct semantics, per
`llama.cpp/src/llama-quant.cpp::llama_tensor_get_weights_for_quantization`
and `ggml/src/ggml-quants.c::make_qkx2_quants`:

The `--imatrix` file contains, per quantized tensor, a 1D float array
`imp[K]` representing per-input-channel importance (computed as the
sum-of-squares of activations seen at that channel during a calibration
pass).

Inside the k-quant rounder, when selecting per-sub-block scale `scale_j`
and offset `min_j`, the optimizer minimizes a **weighted MSE
objective**:

```
L(scale_j, min_j) = Σ_{i ∈ sub_block j} imp[i] · (w_i − (scale_j · q_i + min_j))^2
```

That is, the **importance vector reweights the MSE penalty**, it does
*not* multiply the weights themselves before RTN. The optimizer is a
small enumeration/grid search over candidate scale/min pairs, picking
the pair that minimizes the importance-weighted reconstruction error.

This is a meaningful distinction:

- "Pre-multiply by `sqrt(importance)`" would change *what is rounded*
  and would require an inverse-divide on dequant. It would not preserve
  the bit-exact format.
- "Weighted MSE in the rounder" preserves the format exactly — the
  numeric `w_q` values differ but the *layout, scale, and min* fields
  are identical. The IR sees the same bytes; only the calibration
  metadata changes.

The IR therefore needs an `imatrix_calibrated: bool` axis as a metadata
field (not a layout field). LM Studio's "default Q4_K_M" since 2024 H2
has `imatrix_calibrated = True` with the calibration corpus baked into
the model card; earlier Q4_K_M files have `imatrix_calibrated = False`.

### 5.4 K-quant runtimes

llama.cpp / ggml on every backend (CPU AVX-512 / AVX2 / NEON; CUDA;
Metal; Vulkan; SYCL; Kompute); LM Studio; Ollama; Jan; GPT4All; llamafile;
candle (Rust); MLX via gguf import; bitnet.cpp (BitNet fallback);
KleidiAI on Arm (Q4_0 / Q4_K paths).

**This is the dominant on-device SLM quantization format by deployment
count.** llama.cpp's k-quants alone account for >80 % of public
GGUF downloads on HuggingFace as of mid-2026.

### 5.5 I-quants (IQ family) — importance-aware codebooks

Origin: Kawrakow, llama.cpp PRs #4773 (Jan 2024) and #5104 (Jan 2024),
inspired by QuIP# (Tseng et al., arXiv:2402.04396).

Idea: map weights to a pretrained **vector codebook** (E8 lattice)
optimized for Gaussian-distributed weights. Multiple weights share an
index into the codebook.

| Variant   | Bits/weight | Codebook structure |
|-----------|-------------|---------------------|
| `IQ1_S`   | 1.5625      | 8 weights → 1-byte index + super-block scale + sign bits |
| `IQ1_M`   | 1.75        | as IQ1_S with finer sub-block sign refinement |
| `IQ2_XXS` | 2.0625      | E8 lattice, 8 weights → 16-bit index |
| `IQ2_XS`  | 2.3125      | E8 lattice, 8 weights → larger codebook |
| `IQ2_S`   | 2.5         | augmented IQ2 with sign bits |
| `IQ2_M`   | 2.7         | IQ2_S with bigger codebook |
| `IQ3_XXS` | 3.0625      | E8 lattice, 4 weights → 12-bit index |
| `IQ3_S`   | 3.4375      | sign-augmented |
| `IQ3_M`   | 3.66        | layer-mixed (mix of IQ3_S and Q4_K) |
| `IQ4_XS`  | 4.25        | non-uniform 16-entry LUT per super-block |
| `IQ4_NL`  | 4.5         | "non-linear int4" — 16-entry LUT shared globally |

- `IQ1_S` and `IQ1_M` are the survival-mode formats — perplexity is
  poor but the model fits in extreme memory budgets. Used for 70B
  models on 8 GB GPUs.
- `IQ2_XXS` and `IQ2_M` are the production sub-3-bit formats.
- `IQ4_NL` is the modern Q4_0 replacement — same size, much better
  quality.

All IQ formats **require imatrix calibration** to be effective (the
codebook indices are chosen by importance-weighted nearest-neighbor on
the E8 lattice).

Storage: codebook stored once globally as a constant table; per-super-
block carries the fp16 scale, the packed codebook indices, and the
per-group sign bits.

Runtimes: llama.cpp (all backends), Ollama, LM Studio. The kernel is a
LUT-gather inside the dot product; codebook resides in constant memory
on GPUs.

IR axes for IQ: `qdtype = iq{1,2,3,4}_{xxs,xs,s,m,nl}`,
`codebook = e8_lattice|nl_lut`, `super_block = 256`,
`scale_dtype = fp16`, `imatrix_calibrated = True` (required for
quality), `packing = iquant_indexed`.

---

## §6. Family D — Codebook quantization (cross-family deep dive)

NF4, AQLM, QuIP/QuIP#, EXL2, SqueezeLLM, VPTQ, IQ all use a codebook
in some form. They differ along three independent axes that the v1
`codebook` field collapsed.

### 6.1 Codebook taxonomy

| Scheme        | num_codebooks | combine_op       | shared_across   | Notes |
|---------------|---------------|------------------|-----------------|-------|
| NF4           | 1             | NONE             | ALL (global)    | Fixed quantile codebook |
| FP4 (bnb)     | 1             | NONE             | ALL             | Fixed E2M1 codebook |
| AQLM          | K ∈ {1,2,4,8} | ADD              | ALL             | Additive multi-codebook |
| QuIP#         | 1             | MULTI_LATTICE    | ALL             | E8 lattice + Hadamard incoherence |
| EXL2          | 1             | NONE             | PER_ROW         | Per-row variable bitwidth, codebook implicit |
| SqueezeLLM    | 1             | NONE             | PER_ROW         | Per-row k-means codebook |
| VPTQ          | 1             | NONE             | ALL or PER_LAYER | Vector codebook, entries are 4-8-D vectors |
| IQ (i-quant)  | 1             | MULTI_LATTICE    | ALL             | E8 lattice port to GGUF |
| IQ4_NL        | 1             | NONE             | ALL             | Non-linear 16-entry LUT |
| IQ4_XS        | 1             | NONE             | PER_SUPER_BLOCK | Per-super-block LUT |

The v1 `codebook` field had only `{global, per_super_block}` for
`shared_across` and no `num_codebooks` / `combine_op`. This is the
single biggest IR axis change.

### 6.2 Why "decoder program" matters for QuIP#

QuIP# decoded value: `w = lattice[idx] + residual · sign · scale`. The
v1 `codebook` field assumes `w = codebook[idx]` plus a multiplicative
scale. QuIP# inserts a residual lookup *and* a sign bit per group. The
new IR has `combine_op = MULTI_LATTICE` to mark this; a fully general
"decoder program" mini-DSL would also work but is overkill for a survey.

### 6.3 Additive codebook math (AQLM)

For AQLM with K = 2 codebooks, group_size = 8, index_bits = 16:

```
W_block[1..8] = C₁[idx₁] + C₂[idx₂]
```

where `C₁, C₂ : (2^16, 8) fp16 tensors`. Total storage per group:
2 × 16 = 32 index bits for 8 weights = 4 index bits per weight; plus
the codebooks (one-time, shared across the whole tensor) and the
per-super-block fp16 scale. Effective rate ≈ 2 bits/weight.

The matmul becomes a gather over both codebooks and a sum:

```
Y[m, n] = scale[n] · Σ_k A[m, k] · (C₁[idx₁[n, k/8]] + C₂[idx₂[n, k/8]])
```

The IR's `combine_op = ADD` encodes this.

---

## §7. Family E — KV-cache quantization

The KV cache often exceeds model weights at long context. Runtime
engines quantize it independently from weights, with different
trade-offs because KV is *populated at inference time*, not
preprocessed.

### 7.1 K vs V asymmetry — the structural fact

K is much more outlier-prone than V, especially for RoPE'd K. The
mechanism: RoPE projects each K head dimension onto a frequency basis,
producing channel-aligned spikes in K (a small fraction of channels
carry the high-frequency content). V outliers, by contrast, are
token-aligned (some tokens — typically attention sinks or
salient-content tokens — have large value vectors).

Real engines treat K and V with different schemes. The IR therefore
needs **separate quantization configs for K and V**, not a single
`kv_cache` spec.

### 7.2 INT8 KV cache (vLLM, llama.cpp, TRT-LLM)

- **Grouping**: vLLM uses per-head fp16 scale; llama.cpp uses per-token
  via `Q8_0` (one fp16 scale per 32 elements).
- **Scale dtype**: fp16.
- **Zero-point**: symmetric in vLLM/llama.cpp; asymmetric optional in
  TRT-LLM.
- **Storage**: vLLM uses PagedAttention block layout, each block holding
  `block_size × num_heads × head_dim` int8s plus per-head scales.
- **Calibration**: none for dynamic per-channel; static per-channel
  optional in vLLM via `kv-scales.json`.
- **Kernel touches**: dequant inline in the FlashAttention kernel.

### 7.3 INT4 KV cache — KIVI (Liu et al., arXiv:2402.02750, Feb 2024)

- **K**: per-channel INT4 with fp16 scale. **Asymmetric**.
- **V**: per-token INT4 with fp16 scale. Symmetric.
- **Residual buffer**: the most recent ~32 tokens are kept in fp16 to
  preserve quality on the just-arrived activations (which are not yet
  outlier-fitted to the quantization grid).
- **Runtimes**: vLLM `kv_cache_dtype=int4` (experimental KIVI),
  TRT-LLM `int4_kv_cache`.
- **IR axes**: `kv_k_quant_axis = per_channel`,
  `kv_v_quant_axis = per_token`, `kv_residual_buffer = N`.

### 7.4 FP8 KV cache (TRT-LLM, vLLM, SGLang)

- E4M3 with per-tensor (not per-head — v1 was slightly wrong here) or
  per-token fp32 scale. E5M2 also supported.
- vLLM `kv_cache_dtype=fp8_e4m3` quantizes with **a single fp32 scale
  per tensor**, not per-head. Per-head scaling is for INT8 only in
  vLLM.
- No calibration required for E5M2 (huge range); static amax for E4M3.

### 7.5 KVQuant (Hooper et al., NeurIPS 2024, arXiv:2401.18079)

Sub-2-bit KV cache via *per-channel non-uniform quantization* + a
*dense-and-sparse decomposition* for outliers. Specifically:

- Per-channel codebook for K (non-uniform 4-entry LUT).
- Sparse fp16 buffer for the top ~1 % outlier entries (per-channel
  threshold).
- V uses dense 2-bit per-token uniform.

Achieves ~2-bit KV with sub-1-point perplexity loss. Shipped in
research forks (`SqueezeAILab/KVQuant`); not yet in vLLM mainline.

### 7.6 SmoothQuant for KV (Q-Hitter, retroactive K-smoothing)

A line of work applies SmoothQuant-style channel rescaling **to the K
projection only**, migrating range from K activations into the K
weights (which are fp16) so that the K cache becomes easier to
quantize. Used in some TRT-LLM recipes via `--smoothquant-for-kv`
flag in TRT-Model-Optimizer.

### 7.7 CacheGen (Liu et al., arXiv:2310.07240)

Not a quantization scheme per se — a *delta encoding + entropy coding*
for the KV cache used to **stream KV cache across nodes** at low
bandwidth. Combines quantization (vector quantization of layer-by-layer
deltas) with arithmetic coding. Shipped in some vLLM v0.7 disaggregated
inference setups for prefill-decode separation. Flag for IR as a
non-quantization compression family.

### 7.8 PoUKey (Position-Uniform Key compression)

Recent (Q1 2026) experimental KV compression that recognizes positions
in K have lower entropy than channels and quantizes positionally —
useful for long-context retrieval workloads where K is mostly stable
across attention queries. Not yet in production.

### 7.9 Per-token vs per-channel scaling rationale

The choice of per-token vs per-channel scaling for K and V follows from
the outlier structure (§7.1):

- **K per-channel**: K outliers are channel-aligned (RoPE structure), so
  one scale per channel captures the dynamic range.
- **V per-token**: V outliers are token-aligned, so one scale per token
  works.
- **Per-tensor** is the lowest-overhead but worst-quality; only
  acceptable for fp8 (where the range is generous).
- **Per-head** is between per-tensor and per-channel; vLLM's
  default int8 mode.

### 7.10 GQA awareness

For GQA models (Llama-3, Qwen2.5), the KV has `num_kv_heads` heads that
are shared across `num_q_heads / num_kv_heads` query heads. The KV
scale topology must respect this — scaling per-KV-head, not per-Q-head.
IR needs `kv_scale_grouping: per_kv_head | per_q_head_group | per_tensor`.

### 7.11 IR axes for KV cache

```
kv_k_qdtype, kv_v_qdtype (independent)
kv_k_quant_axis ∈ {per_channel, per_token, per_head, per_tensor}
kv_v_quant_axis ∈ {per_channel, per_token, per_head, per_tensor}
kv_scale_dtype ∈ {fp32, fp16, e8m0, fp8_e4m3}
kv_k_has_zero_point, kv_v_has_zero_point
kv_residual_buffer: int (number of recent tokens kept in fp16)
kv_outlier_sparse_buffer: bool (KVQuant style)
```

---

## §8. Family F — Attention-internal quantization

Beyond KV storage, FlashAttention-3 and TRT-LLM expose quantization
*inside* the attention kernel.

### 8.1 FP8 SDPA — FlashAttention-3 (Shah et al., arXiv:2407.08608)

FA3 computes:

- `Q · Kᵀ` in fp8 E4M3,
- softmax in fp32 (inside warp),
- requantize P to fp8 with per-row scales tracked online,
- `P · V` in fp8.

The per-row scaling on P maintains `m_i = max(QK_i,:)` so the largest
value maps to 448 (E4M3 max). This is *online per-token activation
quantization inside attention*.

Runtimes: TRT-LLM `fp8 attention`, FlashInfer fp8 SDPA, SGLang.

IR axes: `attn_qk_qdtype`, `attn_pv_qdtype`,
`attn_softmax_dtype = fp32`, `online_rescale = True`.

### 8.2 INT8 quantized attention (TurboMind / LMDeploy)

W8A8 INT8 attention with per-row scales pre- and post-softmax. Niche
today because FP8 has eaten the mindshare on Hopper+. Still relevant on
Ampere (no fp8 tensor cores) and on edge TPUs.

### 8.3 FP8 sub-format zoo (consolidated)

The IR's `qdtype` enum must distinguish:

- `fp8_e4m3fn` (NVIDIA) — no inf, range ±448.
- `fp8_e4m3fnuz` (AMD) — finite no unsigned zero, different bias.
- `fp8_e5m2` (NVIDIA) — IEEE-like, range ±57344.
- `fp8_e5m2fnuz` (AMD).

Intel Gaudi-3 adopted OCP-compliant E4M3/E5M2 (so it joins NVIDIA side).
A model checkpoint saved as E4M3 on NVIDIA is not byte-identical on AMD;
vLLM / TRT-LLM transparently re-bias on AMD at load time.

### 8.4 FA3 worked example (per-row scale flow)

For an attention head with `head_dim = 128`, sequence length 4096, FA3
processes tiles of 64×64. Within each tile:

1. Compute `S_ij = Q_i · K_jᵀ` in fp8 E4M3, accumulator fp32 inside the
   `mma`.
2. Online maxima `m_i = max(m_i, max_j(S_ij))` updated in fp32.
3. Apply softmax: `P_ij = exp(S_ij − m_i) / l_i` in fp32, where
   `l_i = Σ_j exp(...)`.
4. Requantize `P_ij` to fp8 with per-row scale
   `s_p_i = 448 / max_j(P_ij)`.
5. Accumulate `O_i += s_p_i · (P_ij_fp8 · V_j_fp8)` with fp32
   accumulator, then divide by `s_p_i` at the tile boundary.

The per-row `s_p_i` is the headline IR addition.

---

## §9. Family G — Sub-2-bit quantization

### 9.1 BitNet b1.58 (Ma et al., arXiv:2402.17764, Feb 2024)

- **Weight bitwidth**: ternary {-1, 0, +1}, i.e. 1.58 bits (log₂ 3).
  Stored efficiently as packed pairs in 3 bits per 2 weights, or as
  packed groups of 5 weights in 8 bits (`3⁵ = 243 < 256`).
- **Activation bitwidth**: INT8 (W1.58A8 in the headline) with per-token
  absmax.
- **Trained from scratch**: BitNet is **not a PTQ scheme**. The
  architecture uses `BitLinear` layers with straight-through-estimator
  gradients on the rounding. The model is pretrained with ternary
  weights from initialization.
- **Activation**: ReLU² in the FFN (replaces SwiGLU); RMSNorm; no bias.
- **Runtimes**:
  - **bitnet.cpp** (Microsoft, Oct 2024): CPU LUT-based GEMV kernels;
    runs BitNet-3B at >100 tok/s on Apple M2 with no GPU.
  - **T-MAC** (Microsoft, 2024): generalized table-based GEMV for
    low-bit, including ternary.
  - **KleidiAI** (Arm, Q3 2025): LUT-based ternary on Cortex-X4.
  - **vLLM**: experimental BitNet loader, GPU LUT kernel.
  - **HF Transformers**: `BitNetForCausalLM` class.
- **IR axes**: `qdtype = ternary` (distinct from `int1`!), with
  `packing = ternary_5x8bit` or `ternary_pair_3bit`. The activation
  side is W1.58A8 int8 per-token.

The distinction from `int1`: int1 has 2 values {-1, +1} (binary); BitNet
has 3 values {-1, 0, +1} (ternary). Adding the zero is what gives BitNet
its accuracy — sparse-like representation without sparsity overhead.

### 9.2 BiLLM (Huang et al., arXiv:2402.04291, Feb 2024)

- **Weight bitwidth**: 1-bit binary {-1, +1}, with salient-channel
  preservation.
- **Salient channels**: top ~5 % of weights (by Hessian sensitivity)
  kept fp16 or int4; the rest binarize.
- **PTQ scheme** (unlike BitNet which is QAT). The PTQ side of the
  1-bit story.
- **Runtimes**: `Aaronhuang-778/BiLLM` reference, not in mainstream
  inference runtimes.

### 9.3 Why ternary works where binary often fails

Binary weights force every weight to ±1 — there is no zero. For weight
matrices that are naturally sparse (~10-20 % of weights near zero by
magnitude), forcing them to ±1 doubles their effective contribution and
distorts the matmul. Ternary's zero level captures the sparse mass at
0, hence the perplexity-vs-bits Pareto improvement.

---

## §10. Parameter space — the IR axes

This section enumerates the axes required to *represent* every scheme
above. v1 had 7 minimal + 8 extended axes; v2 has **7 minimal + 12
extended = 19 axes** with the rotation, codebook, variable-bit, and
outlier-offload axes restructured.

### 10.1 Minimal 7 axes

Sufficient to *describe* the quantized weight buffer for inference
without re-deriving it:

```python
MinimalQuantSpec = {
  "qdtype":           Enum,    # See §10.4
  "group_size":       int | None,
  "quant_axis":       int | Enum,
  "scale_dtype":      Enum,
  "has_zero_point":   bool,
  "packing":          Enum,    # See §10.6, now includes marlin_4bit
  "accumulator_dtype":Enum,
}
```

These 7 cover any vanilla weight-only int/fp scheme. They do *not* cover
codebooks, rotations, hierarchical scales, variable bitwidth, outlier
splits, or per-tensor recipe overrides. For those, the 12 extended axes
below.

### 10.2 Extended 12 axes

```python
ExtendedQuantSpec = MinimalQuantSpec | {
  "scale_quant":          QuantSpec | None,
  "scale_axis":           int | Enum,
  "zero_dtype":           Enum | None,
  "codebook":             CodebookSpec | None,
  "rotation_kind":        Enum,
  "runtime_apply":        Enum,
  "variable_bitwidth":    bool,
  "outlier_offload":      Enum,
  "marlin_layout":        bool,   # subsumed into packing enum
  "imatrix_calibrated":   bool,
  "calibration":          Enum,
  "compute_dtype":        Enum,
  "role":                 Enum,
}
```

(Counting `marlin_layout` as an extension of `packing` rather than its
own axis brings the extended set to 12.)

### 10.3 Per-axis enumeration

#### 10.3.1 `qdtype`

```
Enum[
  int8, int4, int3, int2, int1, ternary,
  fp8_e4m3fn, fp8_e4m3fnuz, fp8_e5m2, fp8_e5m2fnuz,
  fp6_e3m2, fp6_e2m3,
  fp4_e2m1, nf4, fp4_codebook,
  mxfp4, mxfp6, mxfp8, mxint4, mxint8,
  nvfp4,
  q2_k, q3_k, q4_k, q5_k, q6_k, q8_k,
  q4_0, q4_1, q5_0, q5_1, q8_0,
  iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq2_m,
  iq3_xxs, iq3_s, iq3_m, iq4_xs, iq4_nl,
  aqlm_codebook, quip_lattice, vptq_vector,
  squeezellm_codebook, exl2_variable
]
```

Notes:

- `ternary` is distinct from `int1` (three values, not two).
- FP8 splits into NVIDIA and AMD biases (`fnuz` suffix).
- MXINT4 and MXINT8 are distinct from `int4`/`int8` because of the
  UE8M0 scale.
- The k-quants and i-quants are full first-class types because their
  storage is hierarchical and not reducible to (bitwidth, scale_dtype).

#### 10.3.2 `group_size`

`int | None`. For fixed-block schemes (MXFP4 = 32, NVFP4 = 16),
this is set by the format. For variable-block schemes (k-quants), the
super-block size is fixed (256) and the sub-block size is set by the
qdtype.

#### 10.3.3 `quant_axis`

```
int | Enum[per_tensor, per_token, per_channel_out, per_channel_in,
          flattened]
```

#### 10.3.4 `scale_dtype`

```
Enum[fp32, fp16, bf16, fp8_e4m3, ue8m0, int8, int6, int4]
```

`ue8m0` and `fp8_e4m3` are essential for MX and NVFP4 respectively.
`int6` is required for k-quants (sub-block scales).

#### 10.3.5 `scale_quant` — recursive scale-of-scales

A `QuantSpec | None` describing how the scales themselves are
quantized. Examples:

- **NF4 double-quant**: `scale_quant = QuantSpec(qdtype=int8,
  group_size=256, scale_dtype=fp32)`.
- **NVFP4**: `scale_quant = QuantSpec(qdtype=fp8_e4m3, scale_dtype=fp32)`
  (the outer fp32 scale of the inner E4M3 scales).
- **K-quant**: `scale_quant = QuantSpec(qdtype=int6, scale_dtype=fp16)`
  (the 6-bit sub-scales scaled by the fp16 super-scale).

#### 10.3.6 `scale_axis`

```
int | Enum[per_tensor, per_row, per_block, per_sub_block, per_head,
          per_token, per_super_block]
```

K-quants have *both* a per-sub-block scale and a per-super-block scale on
different axes; therefore `scale_axis` is distinct from `quant_axis`.

#### 10.3.7 `has_zero_point` and `zero_dtype`

```
has_zero_point: bool
zero_dtype: Enum[same_as_qdtype, fp16, int4, int6] | None
```

GPTQ packs int zero, AWQ stores fp16 zero, NF4 has no zero — cannot be a
single bool.

#### 10.3.8 `codebook` — extended

```python
CodebookSpec = {
  "values":          Tensor | BuiltinId,    # codebook table or id
  "index_bits":      int,                    # bits per index
  "num_codebooks":   int,                    # NEW: 1 for NF4/IQ, K for AQLM
  "combine_op":      Enum[NONE, ADD, CONCAT, MULTI_LATTICE],  # NEW
  "shared_across":   Enum[ALL, PER_ROW, PER_LAYER, PER_SUPER_BLOCK],
                     # extended from v1
  "vector_dim":      int,    # 1 for scalar codebooks, >1 for VPTQ
  "decoder_program": str | None,  # for QuIP# residual lookup
}
```

This handles:

- AQLM via `num_codebooks=K`, `combine_op=ADD`.
- SqueezeLLM via `shared_across=PER_ROW`.
- EXL2 implicit codebook via `shared_across=PER_ROW`.
- QuIP# via `combine_op=MULTI_LATTICE` + `decoder_program`.
- VPTQ via `vector_dim>1`.
- NF4/FP4 via `values=NF4_BUILTIN`, `num_codebooks=1`.
- IQ via `values=E8_LATTICE`, `combine_op=MULTI_LATTICE`.

#### 10.3.9 `rotation_kind` — NEW split

```
Enum[
  NONE,
  DIAGONAL_SMOOTHQUANT,    # SmoothQuant per-channel scale
  DENSE_HADAMARD_QUAROT,   # fixed Walsh-Hadamard
  DENSE_LEARNED_SPINQUANT, # learned orthogonal matrix
  HADAMARD_RUNTIME,        # QuIP#-style runtime Hadamard
]
```

This replaces v1's flat `pre_transform = none|smooth|hadamard` which
collapsed structurally distinct objects.

#### 10.3.10 `runtime_apply` — NEW

```
Enum[NONE, INPUT, INPUT_OUTPUT_BOTH]
```

Whether the rotation must be applied at runtime to activations:

- `NONE`: SmoothQuant — fused into the previous layer's weights,
  nothing at runtime.
- `INPUT`: QuaRot — Hadamard on the input activation each forward.
- `INPUT_OUTPUT_BOTH`: full QuaRot pipeline — Hadamard on both input
  and output activations.

#### 10.3.11 `variable_bitwidth` — NEW

```
variable_bitwidth: bool
bitwidth_per_row: int[N] | None  # populated if variable_bitwidth=True
```

For EXL2, per-row bitwidth varies. SqueezeLLM does not vary the
bitwidth (uniform 3-bit) but varies the codebook per row — distinct
axis.

#### 10.3.12 `outlier_offload` — NEW

```
Enum[
  NONE,
  FP16_COLUMN_SPLIT,    # LLM.int8 (dynamic) and OWQ (static)
  FP16_ROW_SPLIT,       # SqueezeLLM-style sparse buffer
]
```

Plus a `outlier_selection: Enum[DYNAMIC_THRESHOLD, STATIC_HESSIAN]` to
distinguish LLM.int8 (dynamic >6) from OWQ (static OBS-selected).

#### 10.3.13 `packing`

Extended enum:

```
Enum[
  unpacked,
  nibble_pairs,                  # bnb 4-bit packing
  awq_interleaved8x4,            # AWQ original
  gptq_int32,                    # AutoGPTQ default
  gptq_int32_v2,                 # GPTQModel fork
  marlin_4bit,                   # NEW: Marlin tile layout
  marlin_24,                     # NEW: Marlin + 2:4 sparse
  gguf_kquant,                   # llama.cpp k-quants
  gguf_iquant,                   # llama.cpp i-quants
  gguf_legacy,                   # Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
  ocp_mx,                        # OCP MX block layout
  nvfp4,                         # NVFP4 two-level layout
  ternary_pair_3bit,             # BitNet 3-bits per 2 weights
  ternary_5x8bit,                # BitNet 5 weights per byte
  lut_flatbuf,                   # KleidiAI FlatBuf LUT
  exl2_q_matrix,                 # ExLlamaV2 variable-row layout
]
```

#### 10.3.14 `imatrix_calibrated` — NEW

```
imatrix_calibrated: bool
imatrix_corpus_id: str | None
```

Metadata axis for GGUF k-quants and i-quants. Required for IQ to be
high-quality; optional for k-quants.

#### 10.3.15 `calibration` — refined

```
Enum[
  none, rtn, awq, gptq_hessian, smoothquant, hqq,
  imatrix, percentile, amax_ema, qat,
  learnable_clipping,         # OmniQuant
  data_free,                  # HQQ, NF4, MX
  quip_incoherence,           # QuIP/QuIP#
  spinquant_riemannian,       # SpinQuant
]
```

Plus a `calibration_data` opaque field with `dataset_id`,
`num_samples`, `hyperparameters` (α for SmoothQuant, damp_percent for
GPTQ, num_iters for OmniQuant).

#### 10.3.16 `compute_dtype` separate from `qdtype`

The dtype the matmul HW path consumes; can differ from the stored
`qdtype` (e.g. W4A16 stores int4 but computes in fp16). Required to
distinguish W4A16 / W4A8 / W4A4 paths on the same stored weights.

#### 10.3.17 `accumulator_dtype`

`Enum[fp32, fp16, bf16, int32]`. Per-tile and per-block accumulators can
differ but the IR abstracts to a single field at the matmul level; the
kernel chooser handles tile-level promotion.

#### 10.3.18 `role`

```
Enum[weight, activation, kv_k, kv_v, attn_qk_score, attn_pv_score,
     bias, embed]
```

Allows different QuantSpecs for K vs V, for the QK matmul vs the PV
matmul inside attention, and for weight tying.

#### 10.3.19 `marlin_layout` — subsumed

Discussed above as a value of the `packing` enum (`marlin_4bit`,
`marlin_24`). Not a separate axis; flagged as the 19th to underline
that the IR's `packing` enum had to grow.

### 10.4 Side-by-side: which axes each scheme touches

| Scheme        | qdtype  | rotation_kind                  | codebook combine_op | variable_bitwidth | outlier_offload     | imatrix |
|---------------|---------|--------------------------------|----------------------|-------------------|----------------------|---------|
| GPTQ          | int4    | NONE                           | NONE                 | False             | NONE                 | False   |
| AWQ           | int4    | NONE                           | NONE                 | False             | NONE                 | False   |
| HQQ           | int4    | NONE                           | NONE                 | False             | NONE                 | False   |
| NF4           | nf4     | NONE                           | NONE                 | False             | NONE                 | False   |
| SmoothQuant   | int8    | DIAGONAL_SMOOTHQUANT           | NONE                 | False             | NONE                 | False   |
| QuaRot W4A4   | int4    | DENSE_HADAMARD_QUAROT          | NONE                 | False             | NONE                 | False   |
| SpinQuant     | int4    | DENSE_LEARNED_SPINQUANT        | NONE                 | False             | NONE                 | False   |
| OmniQuant     | int4    | DIAGONAL_SMOOTHQUANT (LET)     | NONE                 | False             | NONE                 | False   |
| LLM.int8      | int8    | NONE                           | NONE                 | False             | FP16_COLUMN_SPLIT(d) | False   |
| OWQ           | int4    | NONE                           | NONE                 | False             | FP16_COLUMN_SPLIT(s) | False   |
| SqueezeLLM    | codebook| NONE                           | NONE (per_row)       | False             | FP16_ROW_SPLIT       | False   |
| EXL2          | mixed   | NONE                           | NONE (per_row)       | True              | NONE                 | False   |
| AQLM          | codebook| optional QuIP                  | ADD                  | False             | NONE                 | False   |
| QuIP#         | codebook| HADAMARD_RUNTIME               | MULTI_LATTICE        | False             | NONE                 | False   |
| Q4_K_M        | q4_k    | NONE                           | NONE                 | False (per-layer override)| NONE         | True (since 2024 H2) |
| IQ2_M         | iq2_m   | NONE                           | MULTI_LATTICE        | False             | NONE                 | True    |
| BitNet b1.58  | ternary | NONE                           | NONE                 | False             | NONE                 | False   |
| BiLLM         | int1    | NONE                           | NONE                 | False             | FP16_COLUMN_SPLIT(s) | False   |
| MXFP4         | mxfp4   | NONE                           | NONE                 | False             | NONE                 | False   |
| NVFP4         | nvfp4   | NONE                           | NONE                 | False             | NONE                 | False   |
| FP8 E4M3      | fp8_e4m3fn | NONE                        | NONE                 | False             | NONE                 | False   |
| Marlin (W4A16)| int4    | NONE                           | NONE                 | False             | NONE                 | False   |
| KleidiAI LUT  | int4    | NONE                           | NONE                 | False             | NONE                 | False   |

### 10.5 Axes added vs v1

Compared with v1's "7 minimal + 8 extended":

- **Split**: `pre_transform` → `rotation_kind` + `runtime_apply`.
- **Extended**: `codebook` → 5-field `CodebookSpec` with
  `num_codebooks`, `combine_op`, `shared_across`, `vector_dim`,
  `decoder_program`.
- **New**: `variable_bitwidth` (+ `bitwidth_per_row`).
- **New**: `outlier_offload` (+ `outlier_selection`).
- **New**: `imatrix_calibrated` (metadata, not layout).
- **Extended**: `packing` gains `marlin_4bit`, `marlin_24`,
  `gguf_legacy`, `ternary_*`, `lut_flatbuf`, `exl2_q_matrix`.
- **Extended**: `qdtype` enum gains FP8 fnuz variants, ternary, MXINT4,
  MXINT8, all k-quant codes, all i-quant codes, AQLM/QuIP/VPTQ codebook
  codes.
- **Extended**: `calibration` enum gains learnable_clipping,
  quip_incoherence, spinquant_riemannian.

The minimal 7 axes from v1 stand; v1's "the truly minimal subset" was
correct, but the extended set has to grow from 8 → 12 (counting
`marlin_layout` as one extension of packing) to cover 2024–2026
schemes.

---

## §11. "If you support five, support these five" — replacing v1's overclaim

v1 claimed AWQ-INT4 + Q4_K_M + FP8-E4M3 "exercise every axis in §7"
— that was false. They exercise about half the axes (no codebook, no
rotation, no UE8M0, no variable-bit, no outlier-offload). This v2
recommends a 5-scheme set that genuinely exercises every axis.

### 11.1 The five

1. **GGUF Q4_K_M (with imatrix)** — on-device dominant. Exercises:
   hierarchical scale (`scale_quant` recursive), asymmetric zero
   (`zero_dtype=int6`), per-layer override (the `_M` upgrade dict),
   super-block packing (`gguf_kquant`), imatrix calibration
   (`imatrix_calibrated`).
2. **AWQ INT4 W4A16 g=128 (loaded into Marlin)** — server 4-bit
   dominant. Exercises: blockwise int4 (`qdtype=int4`, `group_size=128`),
   fp16 zero (`zero_dtype=fp16`), packed interleave then Marlin layout
   (`packing=marlin_4bit`), activation-aware calibration
   (`calibration=awq`).
3. **FP8 E4M3 W8A8 (per-tensor W + per-token A)** — compute-quantized
   H100/MI300/Gaudi-3. Exercises: fp accumulator path, scale-on-
   activations, per-token topology, FP8 fnuz variant when on AMD,
   amax-history calibration.
4. **IQ2_M (or AQLM)** — 2-bit coverage. Exercises codebook axes:
   `codebook.combine_op = MULTI_LATTICE` (for IQ) or
   `codebook.combine_op = ADD` (for AQLM), `codebook.num_codebooks > 1`
   (AQLM only), `imatrix_calibrated = True`, sub-2-bit storage layout.
5. **MXFP4** — MX format coverage. Exercises: `qdtype=mxfp4`,
   `scale_dtype=ue8m0`, `packing=ocp_mx`, OCP block packing,
   element_format=e2m1, fixed group_size=32.

### 11.2 Axis coverage check

| Axis                          | Q4_K_M | AWQ-Marlin | FP8 W8A8 | IQ2_M / AQLM | MXFP4 |
|-------------------------------|--------|------------|----------|---------------|-------|
| Asymmetric zero               | yes (int6)| yes (fp16)| no       | no            | no    |
| Integer zero                  | yes (int6)| no       | no       | no            | no    |
| Codebook                      | no     | no         | no       | **yes**       | no    |
| `combine_op` (multi-codebook) | no     | no         | no       | **yes**       | no    |
| Hadamard / rotation           | no     | no         | no       | optional (AQLM-incoh) | no |
| Hierarchical scale            | yes    | no         | no       | yes (i-quant) | no    |
| UE8M0 scale                   | no     | no         | no       | no            | **yes** |
| FP8 inner scale (NVFP4)       | no     | no         | no       | no            | no (NVFP4 needed for that) |
| Per-row variable bit          | no     | no         | no       | no            | no    |
| Outlier offloading            | no     | no         | no       | no            | no    |
| Marlin packing                | no     | **yes**    | no       | no            | no    |
| imatrix calibration           | **yes**| no         | no       | **yes**       | no    |
| FP accumulator (FP path)      | no     | no         | **yes**  | no            | **yes** |
| Activation per-token          | no     | no         | **yes**  | no            | yes (dynamic) |

The 5-pack covers all the listed axes except per-row variable bitwidth
(EXL2-specific), outlier offload (LLM.int8-specific), NVFP4 outer fp32
scale, and dense learned rotation (SpinQuant). If your IR must cover
those too, add **EXL2** and **NVFP4** for a 7-pack.

### 11.3 Why not GPTQ instead of AWQ

Defensible swap. GPTQ-Marlin and AWQ-Marlin share the runtime Marlin
layout. Picking GPTQ would test `calibration=gptq_hessian` and
`act_order` axes (which AWQ does not). The choice is metadata-only at
runtime. v2 picks AWQ because it has the larger HuggingFace download
share for SLMs and the simpler calibration semantics; if your IR needs
to round-trip into GPTQ-Marlin checkpoints, swap freely.

### 11.4 Why include MXFP4 over NVFP4

MXFP4 is the more *portable* format — OCP spec, supported by NVIDIA,
AMD, Intel. NVFP4 is NVIDIA-only. The OCP MX axes (UE8M0, block_size=32
fixed) are the future-proof choice. NVFP4 adds the two-level scale,
which can be a 6th-scheme stretch goal.

### 11.5 Why not just support Q4_K_M

A frequently-asked question. Answer: Q4_K_M exercises the on-device
axes but does not touch the FP8 accumulator path, the codebook combine
op, or the UE8M0 scale. An IR that only supports Q4_K_M cannot run an
H100 server inference workload at the speeds the hardware allows.

---

## §12. PTQ vs QAT — taxonomy by training-time involvement

The survey's schemes split cleanly along *whether the model was trained
with quantization in mind*. This matters for the IR because QAT
schemes' weights are not interchangeable with PTQ schemes of the same
nominal format.

### 12.1 Pure PTQ (no training-time involvement)

- **GPTQ, AWQ, HQQ, SmoothQuant, ZeroQuant V1/V2/FP, OWQ, OmniQuant,
  SqueezeLLM, AQLM, VPTQ, QuIP/QuIP#, NF4, FP4, MX (weight-side),
  NVFP4, FP8 PTQ, K-quants, I-quants, BiLLM, KIVI, KVQuant.**
- These all start from an fp16/bf16 pretrained checkpoint and quantize
  after the fact. Calibration sets are small (typically 128–512
  samples). Reversible in principle: you can de-quantize and re-train.

### 12.2 QAT — quantization-aware training

- **QLoRA (NF4 + LoRA fine-tuning)** (Dettmers et al., May 2023).
  Strictly speaking, QLoRA is PTQ-of-the-base + LoRA-fine-tune-with-
  STE-on-the-base. The headline is single-GPU fine-tuning of 65B; the
  *quantization* of the base is still NF4 PTQ. Listed in QAT
  ambiguously.
- **FP8 QAT (NVIDIA TRT-Model-Optimizer)** — adds straight-through
  estimator gradients on the fp8 rounding during a short fine-tuning
  pass (~1k steps). Improves FP8 quality on hard models (multimodal,
  long-context). Shipped through `quant_modules.fp8_qat`.
- **AWQ-aware QAT** — some Llama-3 production deployments at Meta
  fine-tune with AWQ-rescale-aware forward passes.

### 12.3 Train-from-scratch quantized

- **BitNet b1.58** — ternary weights from initialization. Not
  reversible: there is no fp16 "underlying" checkpoint.
- **DeepSeek-V3 native FP8 training** — pre-trained end-to-end in FP8
  with the fp16 master weights only used for the optimizer. The
  released checkpoint is FP8. Inference is FP8 with no re-quantization.
  Functionally QAT (the loss saw the quantization) without an explicit
  QAT loss term.
- **Some early MX-format training experiments** (NVIDIA, Q1 2026) train
  in MXFP8 / MXFP6. Not yet shipped as a public checkpoint.

### 12.4 IR implications

The IR must carry a `training_provenance: PTQ | QAT | NATIVE_QUANT` flag
because:

- A PTQ FP8 checkpoint and a NATIVE_QUANT FP8 checkpoint are
  byte-identical at the weights but have different quality
  characteristics on out-of-distribution prompts.
- A QAT-fine-tuned NF4 checkpoint cannot be "de-quantized" back to fp16
  losslessly.
- BitNet ternary checkpoints cannot be ported to any other format
  meaningfully because the model architecture is different (RMSNorm
  placements, ReLU² activation).

---

## §13. Open questions / unresolved

Carried forward from v1 + new in v2:

1. **NVFP4 outer scale semantics across multi-GPU**: per-tensor fp32
   broadcast or sharded with TP? Affects `scale_quant`'s sharding
   annotation. TRT-LLM docs sparse.
2. **GGUF imatrix format**: per-tensor float importance; exact
   application differs between Q4_K, Q5_K, IQ2_*. No single normative
   spec; behavior in `llama_model_quantize_internal`. Now confirmed
   semantically (§5.3) but the file format itself is undocumented.
3. **HQQ + LoRA composition**: HQQ's bf16 scales allow adapters to
   fold; AWQ's fp16 scales technically also allow it; the IR may want
   a `mergeable_with_adapter` predicate.
4. **MXFP6 SLM deployments**: spec exists, hardware (MI355X) supports,
   `torchao 0.5` exposes it. Q1 2026 has seen the first public Qwen3
   MXFP6 checkpoint but the workload is still small.
5. **KV cache packing in PagedAttention**: K and V have different
   layouts (`[num_blocks, num_kv_heads, head_size/x, block_size, x]`
   for K vs `[num_blocks, num_kv_heads, head_size, block_size]` for V).
   The IR's `kv_quant_axis` should distinguish K vs V topology.
6. **Per-row scaled softmax (FA3)**: how `s_p_i` is exposed to the rest
   of the graph not standardized — sometimes folded into `softmax_scale`,
   sometimes carried as a separate tensor.
7. **Codebook globality for IQ**: E8 lattice constants hard-coded in
   `iq2xxs_grid` in llama.cpp. If the IR is runtime-agnostic, decide
   whether codebooks are reified tensors (portable) or builtin enums
   (cheap).
8. **compressed-tensors v2** (vLLM, Q1 2026): the canonical server-side
   format. Its `QuantizationStrategy` enum currently has
   `tensor | channel | token | group | block`. Close to the minimum
   viable common denominator but does not cover hierarchical scales or
   codebook formats. The IR should be a superset.
9. **Asymmetric fp formats**: some deployments use positive-only FP4 for
   post-ReLU activations. Not currently in mainstream LLM runtimes; IR
   should leave room.
10. **Rotation matrix portability**: QuaRot uses fixed Hadamard;
    SpinQuant ships a learned rotation tensor. Two runtimes loading the
    same SpinQuant checkpoint must apply the *same* rotation; the IR
    must store the rotation matrix as a `Tensor[D, D]` field (which is
    a non-trivial memory addition for D = 8192).
11. **EXL2 per-row bitwidth ABI**: bitwidths `{2, 2.5, 3, ..., 8}` are
    fractional (e.g. 3.5-bit row means 7 bits packed per 2 weights).
    The IR's `bitwidth_per_row` field must support fractional values
    and the kernel must dispatch row-by-row.
12. **CacheGen / KV streaming**: out of scope for §7 strictly but the
    IR will need a `compression_codec` axis for disaggregated
    inference paths.
13. **AFP8 standardization**: no spec yet. Flag as open.
14. **BitNet checkpoint provenance**: as a train-from-scratch format,
    BitNet checkpoints are not portable to non-BitNet runtimes. The
    `training_provenance` axis flags this but there is no
    cross-runtime "fall back to fp16" path.
15. **TPU v6e INT8W/INT4A**: kernel signature not public. JAX `aqt`
    library targets it.

---

## §14. References

### Papers (arXiv IDs)

- ZeroQuant — Yao et al., NeurIPS 2022. arXiv:2206.01861.
- ZeroQuant-V2 — Yao et al., 2023. arXiv:2303.08302.
- ZeroQuant-FP — Wu et al., 2023. arXiv:2307.09782.
- LLM.int8() — Dettmers et al., NeurIPS 2022. arXiv:2208.07339.
- FP8 Formats — Micikevicius et al., 2022. arXiv:2209.05433.
- GPTQ — Frantar et al., ICLR 2023. arXiv:2210.17323.
- SmoothQuant — Xiao et al., ICML 2023. arXiv:2211.10438.
- QLoRA / NF4 — Dettmers et al., NeurIPS 2023. arXiv:2305.14314.
- AWQ — Lin et al., MLSys 2024. arXiv:2306.00978.
- SqueezeLLM — Kim et al., ICML 2024. arXiv:2306.07629.
- OWQ — Lee et al., AAAI 2024. arXiv:2306.02272.
- QuIP — Chee et al., NeurIPS 2023. arXiv:2307.13304.
- OmniQuant — Shao et al., ICLR 2024. arXiv:2308.13137.
- OCP MX — Rouhani et al., 2023. arXiv:2310.10537.
- KIVI — Liu et al., 2024. arXiv:2402.02750.
- BiLLM — Huang et al., ICML 2024. arXiv:2402.04291.
- QuIP# — Tseng et al., 2024. arXiv:2402.04396.
- BitNet b1.58 — Ma et al., 2024. arXiv:2402.17764.
- AQLM — Egiazarian et al., ICML 2024. arXiv:2401.06118.
- KVQuant — Hooper et al., NeurIPS 2024. arXiv:2401.18079.
- QuaRot — Ashkboos et al., 2024. arXiv:2404.00456.
- SpinQuant — Liu et al., 2024. arXiv:2405.16406.
- QServe / W4A8 — Lin et al., 2024. arXiv:2405.04532.
- FlashAttention-3 — Shah et al., 2024. arXiv:2407.08608.
- VPTQ — Liu et al., 2024. `microsoft/VPTQ`.
- CacheGen — Liu et al., 2023. arXiv:2310.07240.

### Specs and source code

- OCP, *Microscaling Formats Specification v1.0*, Sept 2023, §4.2.
- NVIDIA Blackwell Architecture Whitepaper, 2024 §3.4 (NVFP4).
- NVIDIA Transformer Engine docs
  (`docs.nvidia.com/deeplearning/transformer-engine`).
- TensorRT-LLM quantization
  (`nvidia/TensorRT-LLM/docs/source/reference/precision.md`).
- TensorRT Model Optimizer
  (`github.com/NVIDIA/TensorRT-Model-Optimizer`).
- vLLM quantization
  (`docs.vllm.ai/en/latest/features/quantization`).
- Neural Magic compressed-tensors
  (`github.com/neuralmagic/compressed-tensors`).
- llama.cpp `ggml/src/ggml-quants.c`,
  `src/llama-quant.cpp`,
  `src/llama-quant.cpp::llama_tensor_get_weights_for_quantization`
  (`github.com/ggerganov/llama.cpp`).
- llama.cpp PR #1684 (k-quants), PR #4773 / #5104 (i-quants).
- AutoAWQ (`github.com/casper-hansen/AutoAWQ`).
- AutoGPTQ (`github.com/PanQiWei/AutoGPTQ`).
- GPTQModel fork (`github.com/ModelCloud/GPTQModel`).
- HQQ (`github.com/mobiusml/hqq`).
- bitsandbytes (`github.com/TimDettmers/bitsandbytes`).
- ExLlamaV2 / V3 (`github.com/turboderp/exllamav2`).
- Marlin (`github.com/IST-DASLab/marlin`).
- QuIP# reference (`github.com/Cornell-RelaxML/quip-sharp`).
- AQLM reference (`github.com/Vahe1994/AQLM`).
- SqueezeLLM (`github.com/SqueezeAILab/SqueezeLLM`).
- KVQuant (`github.com/SqueezeAILab/KVQuant`).
- OmniQuant (`github.com/OpenGVLab/OmniQuant`).
- VPTQ (`github.com/microsoft/VPTQ`).
- SpinQuant (`github.com/facebookresearch/SpinQuant`).
- QuaRot (`github.com/spcl/QuaRot`).
- BitNet b1.58 + bitnet.cpp (`github.com/microsoft/BitNet`).
- T-MAC (`github.com/microsoft/T-MAC`).
- KleidiAI (`github.com/ARM-software/kleidiai`).
- MLX quantization
  (`github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py`).
- ONNX Runtime QDQ / QOperator docs.
- OpenVINO NNCF (`github.com/openvinotoolkit/nncf`).
- Intel Neural Compressor (`github.com/intel/neural-compressor`).
- AMD Quark (`github.com/AMD/Quark`).
- torchao (`github.com/pytorch/ao`).

---

*End of v2 survey. Scheme count: 42 distinct schemes (including
sub-variants) across 7 families. Minimum-viable IR axes: 7;
extended-coverage axes: 19 (7 minimal + 12 extended). Errors fixed
from v1: Q4_K sub-block size (32×8, not 16×16); imatrix semantics
(weighted-MSE rounder objective, not pre-multiplication of weights).
"Support 5" recommendation replaces v1's "support 3" overclaim.
Evolution narrative: six eras 2022 H2 → 2026 Q1.*
