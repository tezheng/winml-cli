# Quantization Schemes in Shipped SLM Runtimes — v3

> Survey for the `llm-layers` IR project. Goal: enumerate every quantization
> scheme actually shipped in mainstream small-language-model runtimes between
> mid-2022 and mid-2026 (llama.cpp, vLLM, TensorRT-LLM, OpenVINO, MLX, MLC,
> Hugging Face Transformers, AutoGPTQ/AutoAWQ, ExLlamaV2, bitsandbytes, ONNX
> Runtime, Intel Neural Compressor, KleidiAI, T-MAC/bitnet.cpp, NVIDIA TRT
> Model-Optimizer, AMD Quark, Neural Magic compressed-tensors, Apple
> Foundation Models toolchain, Google AI Edge / LiteRT, onebitllms), trace
> the 3-year evolution arc, and distill the minimum parameter axes a
> representation must expose to cover them.

v3 changelog vs v2 (lighter refresh, focused on 2025-H2 → 2026 additions):
**+6 new scheme entries** — (a) **MXFP4-native training** (GPT-OSS, separated
from MXFP4 PTQ), (b) **Falcon-Edge 1.58-bit retrainable** (TII, May 2025),
(c) **Gemma 4 mobile-INT4 QAT** (Google, June 2026), (d) **NVFP4
training-mode** (NVIDIA, Blackwell B200 era; promoted from PTQ-only),
(e) **DeepSeek-V3 native FP8 training** (promoted from v2 note to full
scheme), (f) **Apple AFM 2-bpw QAT with embed-INT4 / KV-INT8 split**
(promoted from v2 closed-weights note to full scheme). **Two new axes** added
to QuantSpec: `training_native: bool` (distinguishes MXFP4-native and
BitNet-trained from PTQ artifacts of the same format) and `retrainable:
bool` (Falcon-Edge's distinction from BitNet b1.58). Promoted v2's
"support 5" recommendation to **"support 6"** — the sixth slot is the new
mobile-INT4 QAT recipe. Evolution narrative gains Era 6 (2025-H2: MXFP4
native, Falcon-Edge retrainable) and Era 7 (2026-H1: Gemma 4 mobile QAT,
NVFP4 native training, DeepSeek FP8 maturation). Total schemes: **48
distinct schemes** (v2 had 42).

---

## §1. Introduction and three-year evolution narrative

LLM quantization between mid-2022 and mid-2026 has gone through seven
identifiable eras. Each era was defined by a *bottleneck removed* — never
simply by a clever paper. The arc is not "lower bits over time"; it is
*which engineering constraint was binding this quarter*. v3 extends v2's
six-era narrative with a seventh (Era 7, 2026-H1) and substantially
rewrites Era 6 (2025-H2) to reflect MXFP4-native training and Falcon-Edge
retrainable BitNet, both of which arrived after v2's cutoff.

This survey first tells the timeline (§1), then catalogs every scheme
inside its family (§3–§9), then distills the IR axes (§10), then lands a
concrete recommendation (§11). The evolution narrative is also the answer
to "why does the IR need so many axes?" — each axis was forced by a
specific scheme in a specific quarter.

### 1.1 Era 0 — Pre-LLM-quant (≤ 2022 H1)

Before mid-2022, neural-net quantization was a MobileNet/QAT story:
8-bit weight + 8-bit activation, per-channel symmetric, calibrated by
percentile or KL on ImageNet. Vision models tolerated this because their
weight and activation distributions were Gaussian-ish.

LLMs broke that assumption. OPT-175B, GPT-J, BLOOM had *emergent
outliers* — a small fraction of activation channels (~0.1 %) with
magnitudes 20–50× the typical channel. Off-the-shelf int8 saturated at
those channels and collapsed perplexity. This is the technical fact that
all seven eras below respond to.

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

### 1.6 Era 5 — 2025 H1: MX format adoption and Blackwell native FP4 (PTQ side)

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

The IR consequence (carried forward from v2): scales themselves must be
quantizable, and the scale dtype enum must include UE8M0 and FP8 E4M3 as
first-class. Era 5 in v2 covered only PTQ deployment of MXFP4 / NVFP4;
the *training-native* story is split out into Era 6 / Era 7 in v3.

### 1.7 Era 6 — 2025 H2: MXFP4-native training and Falcon-Edge retrainable BitNet (NEW IN v3)

Two distinct 2025-H2 events redefined the QAT-vs-PTQ-vs-native-training
boundary in different directions:

**Event 1 — GPT-OSS shipping in MXFP4 (Aug 2025).** OpenAI's first open
weights since GPT-2. The model card states that "the models were
post-trained with MXFP4 quantization of the MoE weights, making
gpt-oss-120b run on a single 80GB GPU and gpt-oss-20b within 16GB". The
post-training language matters: this is *not* a fp16 checkpoint that
got PTQ'd via AWQ-MXFP4 at the end; it is a checkpoint that went through
its *post-training phase* — supervised fine-tune, RLHF, distillation —
with MoE expert weights already quantized to MXFP4 and the quantization
re-applied each step. Two consequences:

1. The MoE expert linear weights live natively in MXFP4 (E2M1 elements,
   UE8M0 group scale, block 32). They are *not* dequantized to BF16 on
   load. The released checkpoint's expert tensors are byte-for-byte the
   tensor-core operands.
2. Attention QKV and gate linears stay in BF16, as does the router. Only
   the MoE expert weights are MXFP4. The IR must therefore allow
   different `QuantSpec`s per *layer role* and the load-time path must
   recognize that the MXFP4 tensors do not need a re-quant step.

This is the first widely-used SLM where the *shipped checkpoint format
matches the training-time numeric format* for a substantial fraction of
the weights, without the model being trained from scratch in low
precision. It sits between PTQ (post-hoc) and BitNet (train-from-scratch).
The new axis `training_native: bool` is set True for the MoE expert
weights of gpt-oss-20b and False for everything else.

**Event 2 — Falcon-Edge 1.58-bit retrainable BitNet (May 2025).** TII Abu
Dhabi released the first BitNet-style ternary family with a *retrainable*
checkpoint path. Three model variants from a single training process:
the native BitNet (ternary {-1, 0, +1} weights, ~635 MB for 1B params),
the **pre-quantized** revision (BF16 weights *scaled* such that loading
them and applying the BitNet-Linear conversion reproduces the ternary
result — meaning fine-tuning can continue on standard hardware before
re-quantizing), and a plain BF16 fallback. This is qualitatively
different from BitNet b1.58:

- BitNet b1.58 (Microsoft, Feb 2024) is train-from-scratch. Once
  released, the checkpoint is inference-only; there is no way to
  continue pretraining or domain-fine-tune on the ternary weights
  themselves because the gradient cannot flow through the ternarizer
  without specialized infrastructure.
- Falcon-Edge ships the *pre-quantized* BF16 weights expressly so a user
  can `replace_linear_with_bitnet_linear(model)`, run a TRL `SFTTrainer`
  for fine-tuning, and re-quantize at the end. The package
  `onebitllms` (TII + community) provides the BitnetLinear layer and
  Triton kernels for training/inference.

The IR consequence: a new axis `retrainable: bool`. Falcon-Edge is
`training_native=True, retrainable=True`; BitNet b1.58 is
`training_native=True, retrainable=False`; AWQ-INT4 is
`training_native=False, retrainable=False`.

Sizes shipped: **1B base + instruct, 3B base + instruct.** Memory at
1.58-bit: 1B → 635 MB; 3B → ~999 MB. Runtimes: bitnet.cpp PR #268,
llama.cpp community-tested, mlx-lm for Apple Silicon, HF Transformers
for the BF16 variant.

In parallel, **Microsoft T-MAC and bitnet.cpp** continued maturing on
ARM (Snapdragon X, Apple M-series, Cortex-X4) with LUT-based ternary
GEMV kernels that beat llama.cpp's Q4_K_M on the same model size. By
Q4 2025, bitnet.cpp ran 3B BitNet at >100 tok/s on a Snapdragon X laptop
with no GPU. **KleidiAI** (Arm's open-source SLM kernel library, Q3
2025) added LUT-based 4-bit ternary fused dequant via FlatBuf-packed
codebooks, giving Cortex-X4 a path to BitNet without Microsoft's custom
kernel.

The IR consequence: ternary is *not* int1 — three values, not two — and
must be a distinct `qdtype`. LUT-packing for ternary is also a distinct
packing variant. Additionally, the `retrainable: bool` flag determines
whether the checkpoint ships a `prequantized` BF16 sibling revision
(Falcon-Edge) or only the ternary final form (BitNet b1.58).

### 1.8 Era 7 — 2026 H1: training-native FP4 + mobile-INT4 QAT consolidation (NEW IN v3)

Three distinct 2026-H1 events finished consolidating the "training-time
quantization" landscape:

**Event 1 — NVFP4 training-mode validation (NVIDIA, Q1-Q2 2026).**
NVIDIA's Sept 2025 / 2026-Q1 blog *"NVFP4 trains with the precision of
16-bit and the speed and efficiency of 4-bit"* documented mixed-precision
*pretraining* in NVFP4: a 12B hybrid Mamba-Transformer was pretrained
end-to-end on 10 trillion tokens with weights, activations, and
forward-pass gradients all represented in NVFP4 (E2M1 elements, FP8 E4M3
per-16 inner scale, FP32 per-tensor outer scale), while a small set of
"selective high-precision layers" (the first/last attention projection,
LayerNorms, embeddings, the LM head) stayed in BF16. Stochastic rounding
on the FP4 forward pass eliminates the systematic bias of
round-to-nearest at low precision. **Random Hadamard transforms (RHT)**
applied at each linear bound block-level outliers — bringing the
QuaRot/SpinQuant idea (Era 5 PTQ side) into the training loop itself.
Final result matched the FP8 baseline on MMLU Pro, code, math across six
domains. **Nemotron 3 Super** (NVIDIA, Mar 2026) was the first publicly-
released NVFP4-trained checkpoint of substantial scale.

The IR consequence: the new `training_native: bool` axis is set True for
checkpoints emitted by NVFP4 pretraining. The semantic difference from
PTQ NVFP4 is significant: a PTQ NVFP4 checkpoint exists in a BF16
"underlying" world (you can dequantize and retrain in BF16); a
NVFP4-pretrained checkpoint does not — the BF16 master weights existed
only during training and have been discarded.

**Event 2 — Gemma 4 mobile-INT4 QAT (Google, June 2026 addendum).**
The Gemma 4 family (April 2026 launch) shipped GGUF Q4_K_M conversions
within hours, but for the on-device E2B target Google followed up in
**June 2026** with a dedicated **INT4 QAT** track. The headline number:
**E2B compresses from ~2 GB BF16 to ~1 GB INT4** with sub-1% MMLU
regression, *with the QAT loss applied during a final fine-tuning pass*
(not just post-training). This is distinct from prior Gemma QAT
(Gemma 2 / Gemma 3 had INT8 QAT for the LiteRT path; v3 documents
Gemma 4's INT4 QAT as a new format). The mobile target is LiteRT
(formerly TFLite) on Android and MediaPipe; the format is per-channel
INT4 with INT8 dynamic activations and a fp16 per-row scale, packed for
the Hexagon NPU / ANE / Mali-DSP families. Gemma 4 E4B also has an
INT4-QAT checkpoint at ~2 GB.

The IR consequence: the existing `calibration = qat` value gets a
mobile-targeted variant; the new `training_native: bool` is False here
(QAT is fine-tune-time, not pretrain-time); the `target_runtime` axis
must accommodate LiteRT and MediaPipe alongside the existing server and
on-device options.

**Event 3 — Apple AFM 3.18B 2-bpw QAT going production.** Apple's
mid-2025 tech report (arXiv:2507.13575) documented the on-device 3.18B
AFM as 2-bpw via learnable weight clipping QAT, with the **embedding
table at 4-bit QAT** and the **KV cache at INT8**. The decoder splits
into Block-1 (62.5% of layers) and Block-2 (37.5% of layers), with all
of Block-2's KV caches *directly shared* with Block-1's final-layer KV
— reducing KV memory by 37.5%. This format went production in iOS 18
late 2024 and remains the production format for the 3.18B on-device
model through 2026-H1; the iOS 19 / iOS 19.1 updates (May 2026) refined
the calibration corpus but kept the 2-bpw + INT4 embed + INT8 KV split.
The "novel learnable weight clipping and weight initialization" QAT
recipe is distinct from prior 2-bit PTQ approaches (AQLM, QuIP#) in that
the *clipping bounds are learned per-channel during a short fine-tuning
pass*, not chosen by Hessian analysis after the fact.

The IR consequence: AFM's `2_bpw` weights + `int4` embed + `int8` KV
combination forces a per-tensor-role QuantSpec map (not a single
`weight_qspec`), and the 2-bpw QAT semantics differ from Apple's iOS
17-era INT4 QAT for AFM-2024.

**Event 4 — DeepSeek-V3 / V3.2 native FP8 training maturation.**
DeepSeek-V3 (Dec 2024 → refined Mar 2025 → V3.2 Sept 2025 → V3.2 stable
Dec 2025) was pre-trained end-to-end in FP8: the forward pass quantizes
each linear's weights to per-tile FP8 E4M3 with FP32 master weights
maintained for the optimizer, gradients computed in BF16, and the
per-tile scale recomputed each step. The released checkpoint inherits
FP8 weights directly; inference runtimes do not re-quantize. This was
mentioned in v2 as an end-of-Era-5 note; v3 promotes it to a full scheme
entry (§3.27). The strategy: **per-tile (128×128) scaling for the
weights, per-row scaling for the activations**, with a `BF16_BIAS`
correction tensor folded into the LayerNorm. DeepSeek-V3.2 added DSA
(Lightning Indexer sparse attention) on top, which is orthogonal to the
quantization but compounds the memory savings.

The IR consequence: the existing `qdtype=fp8_e4m3fn` value must be
paired with a `training_native: bool` flag distinguishing PTQ FP8 from
native-trained FP8. The byte-level format is identical; the *quality
characteristics* differ on out-of-distribution prompts.

### 1.9 What is still open going into 2026 Q3

- A portable spec for QuaRot/SpinQuant rotation matrices (so that two
  runtimes loading the same checkpoint apply the *same* Hadamard).
- A converged 2-bit format (AQLM, QuIP#, IQ2_M, VPTQ, AFM 2-bpw are all
  production but mutually incompatible).
- Whether MXINT4 will displace NVFP4 once non-NVIDIA hardware catches up.
- Whether MXFP4-native training will displace NVFP4 training (the OCP
  vs NVIDIA standards battle, replayed at the pretraining layer).
- Whether Falcon-Edge's retrainable BitNet workflow gets adopted by
  Google / Meta / Apple for their next-generation on-device families.

---

## §2. Methodology and categorization framework

### 2.1 What is recorded per scheme

For each scheme we record fourteen fields (v2 had twelve; v3 adds two):

1. **Provenance**: paper, repo, ship date, lead author.
2. **Bitwidth + numeric type** of stored weights and activations.
3. **Grouping**: per-tensor, per-channel (output / input axis), per-token,
   blockwise (with N).
4. **Scale dtype**: fp16, bf16, fp32, UE8M0, FP8 E4M3, INT8, INT6.
5. **Zero-point**: symmetric (none) or asymmetric (offset present), and
   the dtype of the zero.
6. **Calibration**: none / weight-only RTN / activation-aware / Hessian-OBS
   / imatrix-MSE-weighted / QAT / learnable / training-native.
7. **Storage layout**: packed nibble order, interleave, super-block,
   Marlin tile, IQ codebook indices.
8. **Runtimes shipped**: which engines load this scheme today.
9. **Matmul kernel signature**: `(W_q, A) -> Y` with dtypes and
   accumulator.
10. **Pre-transform**: none, diagonal SmoothQuant rescale, dense
    Hadamard QuaRot, dense learned SpinQuant, randomized incoherence
    (QuIP#), Random Hadamard Transform (NVFP4 training).
11. **Outlier handling**: none, runtime fp16 column split (LLM.int8),
    static fp16 column reserve (OWQ, Atom), pre-rotation elimination
    (QuaRot/QuIP#), KV residual buffer (KIVI).
12. **Parameter-space axes** touched by this scheme (cross-ref to §10).
13. *NEW in v3:* **Training-native flag** — was the model's training
    loop aware of this quantization? (Distinguishes PTQ from QAT from
    train-from-scratch.)
14. *NEW in v3:* **Retrainable flag** — can the quantized checkpoint be
    further fine-tuned or continue-pretrained? (Distinguishes
    Falcon-Edge from BitNet b1.58 and from all PTQ schemes.)

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
- **Family G** — Sub-2-bit (BitNet ternary, Falcon-Edge retrainable
  ternary, BiLLM 1-bit, AFM 2-bpw), often requiring training-from-
  scratch or QAT (§9).

A scheme can span families (NF4 is Family A *and* Family D; QuaRot is
Family B *and* introduces a Family A pre-transform). We list each scheme
under its dominant family with cross-references.

### 2.3 New v3 dimension: training-time involvement

Orthogonal to the seven families, v3 introduces a three-axis
classification along training involvement:

- **Pure PTQ**: model trained in BF16, quantized after the fact. No
  gradients flow through the quantizer.
- **QAT**: model pre-trained in BF16, then a short (~1k-100k steps)
  fine-tune with the quantizer in the forward pass (STE for gradients).
- **Native-trained**: the *pretraining* loop itself uses the quantized
  representation (with FP32/BF16 master weights for the optimizer).

This dimension is recorded in `training_native` (True for the last
category) and orthogonally in `calibration` (which has `qat` as one
value for QAT). §12 expands the table.

### 2.4 Citation conventions

Inline citations use arXiv ID or repo path. Section §14 collects URLs.

---

## §3. Family A — Weight-only INT/FP quantization

The most populous family. ~20 distinct schemes in v3 (v2 had 16; v3
adds MXFP4-native, NVFP4-trained, DeepSeek FP8-native, and clarifies
Marlin's role).

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
  original `frantar/gptq` reference. AutoGPTQ is the PanQiWei/AutoGPTQ
  packaging that became the de-facto checkpoint toolchain (introduced
  the `g_idx` packing convention, fp16 scales layout, and `desc_act`
  flag). GPTQModel is the late-2024 fork that added a `--marlin-format`
  flag and an alternate `g_idx` packing with reduced metadata overhead.
- `training_native = False`, `retrainable = False`.

### 3.2 AWQ (Lin et al., arXiv:2306.00978, June 2023)

- **Bitwidth/dtype**: INT4 asymmetric.
- **Grouping**: blockwise along K with `group_size ∈ {32, 64, 128}`;
  128 canonical default.
- **Scale dtype**: fp16 (scale + fp16 zero per group).
- **Zero-point**: fp16-valued.
- **Calibration**: PTQ activation-aware. Computes per-channel activation
  magnitudes from ~128 calibration samples, derives per-channel scaling
  `s` folded into the previous layer's weights, then RTN-quantizes.
- **Storage**: 8×int4 per int32 word with interleave `[0,2,4,6,1,3,5,7]`
  so two fp16 lanes of `mma.sync` dequantize 8 values via one
  `lop3.b32` PRMT.
- **Runtimes**: AutoAWQ, vLLM (`awq`, `awq_marlin`), TRT-LLM `W4A16_AWQ`,
  HF Transformers, MLX 4-bit-groups, MLC. llama.cpp loads via GGUF
  conversion (re-quantized to Q4_K).
- `training_native = False`, `retrainable = False`.

### 3.3 HQQ — Half-Quadratic Quantization (Badri & Shaji, Nov 2023)

- **Bitwidth/dtype**: INT8 / INT4 / INT3 / INT2 / INT1; asymmetric.
- **Grouping**: blockwise along K, `group_size ∈ {8, 32, 64, 128, 256}`;
  64 default.
- **Scale dtype**: fp16 or bf16; zeros fp16.
- **Calibration**: **none** — solves
  `min ‖W − Q⁻¹(Q(W; s, z))‖_p` with `p < 1` via half-quadratic
  splitting using weights only. Famous for "70B in 4 minutes".
- **Runtimes**: HQQ lib, HF Transformers, vLLM experimental, MLX-HQQ.
- `training_native = False`, `retrainable = False`.

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
- `training_native = False`, `retrainable = True` (QLoRA's LoRA adapter
  layer trains on top of frozen NF4 weights).

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
  Learnable Equivalent Transformation (LET).
- **Runtimes**: `OpenGVLab/OmniQuant`, HF integration; not in vLLM
  mainline. ExLlamaV2 supports OmniQuant-derived weights.

### 3.7 SqueezeLLM (Kim et al., ICML 2024, arXiv:2306.07629, Jun 2023)

- **Bitwidth/dtype**: ~3 bits effective.
- **Grouping**: *per-row codebook* — each output channel has its own
  k-means codebook with 8 or 16 entries.
- **Outlier handling**: top ~0.05 % of weights extracted into a sparse
  fp16 side-buffer.
- **Calibration**: weighted by Hessian sensitivity (Fisher-style).
- **Runtimes**: `SqueezeAILab/SqueezeLLM`; HF integration; ExLlamaV2
  partial.

### 3.8 OWQ — Outlier-aware Weight Quantization (Lee et al., AAAI 2024, arXiv:2306.02272, Jun 2023)

- **Bitwidth/dtype**: INT3 or INT4 for most columns, fp16 for the
  small fraction (~0.5 %) of "weak" columns identified by an OBS
  sensitivity criterion.
- **Grouping**: per-channel-out for the int part, fp16 for the
  reserved columns.
- **Calibration**: Hessian-OBS like GPTQ to identify weak columns and
  do the OBS update on the remainder.
- **Runtimes**: Intel Neural Compressor, some HF integrations.

### 3.9 ZeroQuant V1/V2/FP (Yao et al., arXiv:2206.01861, arXiv:2303.08302, arXiv:2307.09782)

- **V1 (June 2022)**: group-wise INT8 W + per-token INT8 A;
  layer-by-layer knowledge-distillation reconstruction.
- **V2 (Mar 2023)**: INT4 weights + LoRC (Low-Rank Compensation).
- **FP (Jul 2023)**: generalizes V1/V2 to FP8/FP4 elements.
- **Runtimes**: DeepSpeed-Inference, partial HF integration.

### 3.10 EXL2 — ExLlamaV2 variable bitwidth (`turboderp/exllamav2`, Sept 2023)

- **Bitwidth/dtype**: *variable per row*. Bitwidths drawn from
  `{2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8}` and **mixed within a single
  weight tensor**.
- **Grouping**: blockwise along K with `group_size ∈ {32, 128}`.
- **Scale dtype**: fp16 per group.
- **Calibration**: PTQ with activation-error measurement per row.
- **Storage**: Q-matrix layout with a per-row bit-budget header and
  packed nibbles of varying width.
- **Runtimes**: ExLlamaV2, ExLlamaV3 partial, TabbyAPI.

### 3.11 AQLM — Additive Quantization (Egiazarian et al., ICML 2024, arXiv:2401.06118)

- **Bitwidth/dtype**: 2-bit effective.
- **Codebook structure**: weight = Σᵢ₌₁..K Cᵢ[idxᵢ], where Cᵢ are L
  trained codebooks.
- **Pre-transform**: optional QuIP-style incoherence (Hadamard).
- **Calibration**: codebooks and indices jointly optimized by
  block-coordinate descent on a Hessian-weighted reconstruction loss.
- **Runtimes**: HF Transformers `aqlm` integration; not yet in vLLM
  mainline.

### 3.12 QuIP and QuIP# (Chee et al., arXiv:2307.13304; Tseng et al., arXiv:2402.04396)

- **QuIP (NeurIPS 2023)**: weight quantization with **incoherence
  processing** — multiply weights and Hessians by random orthogonal
  matrices.
- **QuIP# (Feb 2024)**: adds an **E8 lattice codebook**.
- **Runtimes**: `Cornell-RelaxML/QuIP` and `Cornell-RelaxML/quip-sharp`
  reference, HF integration.

### 3.13 QuaRot (Ashkboos et al., arXiv:2404.00456, Apr 2024)

- **Bitwidth/dtype**: W4A4 INT4 weights + INT4 activations (also W4A8).
- **Pre-transform**: **dense Hadamard rotation**, *fused* offline into
  weights for the parts that can absorb it; *online* Hadamard applied
  to activations at runtime via a fast Walsh-Hadamard transform.
- **Calibration**: PTQ.
- **Runtimes**: `spcl/QuaRot` reference, HF research integration.

### 3.14 SpinQuant (Liu et al., arXiv:2405.16406, May 2024)

- **Bitwidth/dtype**: W4A4 (also W4A8).
- **Pre-transform**: **dense learned orthogonal rotation** via
  Riemannian optimization on the Stiefel manifold.
- **Runtimes**: Meta-internal first; `facebookresearch/SpinQuant`.

### 3.15 BiLLM (Huang et al., ICML 2024, arXiv:2402.04291, Feb 2024)

- **Bitwidth/dtype**: 1-bit weights (binary {-1, +1}).
- **Structure**: salient-channel preservation — the top ~5 % of weights
  stay fp16 or int4, the rest binarize.
- **Calibration**: Hessian-aware binarization with structural search.

### 3.16 Legacy GGUF Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 (llama.cpp, 2022 H2 → 2023 H1)

These predate k-quants and are still shipped on ARM-NEON paths and on
mobile NPUs that lack k-quant kernels. See v2 table for the per-format
bit-budget arithmetic; unchanged in v3.

### 3.17 KleidiAI LUT-based 4-bit (Arm, Q3 2025)

- **Source**: ARM-software/KleidiAI repo. Used by llama.cpp on Cortex-X4
  and Snapdragon X.
- **Idea**: precompute a 16-entry LUT for each weight block at load
  time so the dequant becomes a single SIMD table-lookup (`TBL`
  instruction on NEON).

### 3.18 Marlin format (IST-DASLab/marlin, late 2023 / GA Q1 2024)

- **Bitwidth/dtype**: INT4 weights.
- **Layout**: re-tiles `W: int4[N, K]` into `(K/16, N/64, 16, 64)`
  blocks with a per-block zig-zag interleave.
- **Scale dtype**: fp16, stored separately from weights.
- **Runtimes**: vLLM `awq_marlin`, vLLM `gptq_marlin`, Neural Magic
  compressed-tensors.

### 3.19 Marlin-24 — 2:4 sparse + 4-bit (IST-DASLab/marlin, Q2 2024)

- Sparse variant combining Marlin's 4-bit format with 2:4 structured
  sparsity. Exploits Ampere/Hopper sparse mma.

### 3.20 GPTQ-Marlin (kernel binding)

Not a new quantization scheme — a runtime path where GPTQ-quantized
checkpoints are repacked into Marlin layout.

### 3.21 MX formats — MXFP4 / MXFP6 / MXFP8 / MXINT4 / MXINT8 (OCP, Sept 2023)

- **Spec**: OCP *Microscaling Formats v1.0*, Sept 2023 §4.2. Rouhani
  et al. arXiv:2310.10537.
- **Common structure**: block of 32 elements shares one 8-bit **UE8M0**
  scale.
- **Block size**: 32, fixed by the spec.
- **Calibration (PTQ path)**: data-free per-block absmax for weights;
  activation scales computed on-the-fly per 32 elements.
- **Runtimes**: MXFP8 on Blackwell (TRT-LLM 0.13+); MXFP6 on MI355X
  (Q3 2025); MXFP4 on Blackwell.
- `training_native = False` for this entry; the training-native variant
  is split out into §3.22 (MXFP4-native).

### 3.22 MXFP4-native training (NEW IN v3 — GPT-OSS, Aug 2025)

- **Provenance**: OpenAI gpt-oss-20b and gpt-oss-120b, released
  2025-08-07. arXiv:2508.10925. Model card on `huggingface.co/openai/gpt-oss-20b`.
- **Bitwidth/dtype**: 4-bit MXFP4 (E2M1 elements, UE8M0 group scale,
  block size 32) — but applied at *training time*, not as PTQ.
- **Training workflow**: the MoE expert linear weights are kept in MXFP4
  representation during the **post-training phase** (SFT + RLHF +
  distillation). The model card explicitly states: *"the models were
  post-trained with MXFP4 quantization of the MoE weights, making
  gpt-oss-120b run on a single 80GB GPU (like NVIDIA H100 or AMD MI300X)
  and the gpt-oss-20b model run within 16GB of memory."* The
  pre-training pass before post-training was BF16; the *post-training*
  applied the quantization round-trip each step (forward = quantize
  then matmul; gradient = STE through the quantizer; FP32 master
  weights maintained for the optimizer).
- **Scope** (which tensors are MXFP4):
  - **MoE expert FFN linear weights**: MXFP4 (E2M1 + UE8M0 block-32).
  - **Router (gating) projection**: BF16.
  - **Attention QKV, output projection, dense MLPs (in the dense
    layers)**: BF16.
  - **Embeddings, LM head, RMSNorm parameters**: BF16.
  - In practice for gpt-oss-20b (~21B total, ~3.6B active per token),
    the MoE expert weights are >80% of total parameter count, so the
    storage savings dominate.
- **Storage layout on disk**: SafeTensors with mixed `bfloat16` and
  `uint8` (the latter holding packed MXFP4 nibbles). A sibling
  `uint8` tensor holds the UE8M0 scales (one per 32 elements).
- **Runtime path**: Blackwell (B100/B200) native MXFP4 tensor-core
  mma; H100 falls back to dequant-to-BF16 then BF16 mma (slower); AMD
  MI300X via Quark MXFP4 kernels; vLLM 0.7+ recognizes the format and
  dispatches accordingly. The model card warns: on hardware without
  MXFP4 mma, expect a 1.5-2× slowdown vs Blackwell.
- **IR axes touched**: `qdtype = mxfp4`, `training_native = True`,
  `retrainable = False` (you cannot continue-pretrain in MXFP4 without
  the original FP32 master weights, which were discarded; you can
  fine-tune via LoRA on top, similar to QLoRA).
- **Distinction from MXFP4 PTQ (§3.21)**: byte-identical storage
  format. The difference is the *provenance flag*: MXFP4-native
  checkpoints have been through MXFP4-aware post-training, so the
  weight distributions are calibrated to the quantization grid.
  MXFP4 PTQ checkpoints have been through a calibration set after the
  fact and may show worse OOD perplexity. The IR must distinguish
  these because `training_native = True` implies you should not apply
  further re-quantization (it would compound rounding error against
  weights already at the quantization grid).

### 3.23 NVFP4 (NVIDIA Blackwell, GA Q1 2025)

- **Bitwidth/dtype**: 4-bit `E2M1` (same as MXFP4) but with a
  **two-level scale**:
  1. Inner block scale, fp8 `E4M3`, one per **16** elements.
  2. Outer per-tensor scale, fp32.
- **Outer scale formula**: `outer = amax(W) / 448`.
- **Calibration (PTQ path)**: per-tensor amax for the outer scale;
  inner E4M3 scales per-block absmax.
- **Runtimes**: TRT-LLM `nvfp4`, TRT 10.5+, NIM for Blackwell;
  vLLM 0.6.4+ loads NVFP4 from TRT-Model-Optimizer; SGLang.
- **Kernel**: 5th-gen tensor-core `mma.sync` with fp4 operands, fp8
  scales, bf16 accumulator.
- `training_native = False` for this PTQ entry; the
  training-native variant is split out into §3.24.

### 3.24 NVFP4 training-mode (NEW IN v3 — NVIDIA, Q1-Q2 2026)

- **Provenance**: NVIDIA Blackwell whitepaper §3.4; *"NVFP4 trains with
  the precision of 16-bit and the speed and efficiency of 4-bit"*
  developer blog (Q1 2026); arXiv:2509.25149 (NVFP4 training paper).
  First publicly-released NVFP4-trained checkpoint: **NVIDIA Nemotron
  3 Super** (March 2026), 120B total / 12B active hybrid Mamba-
  Transformer MoE.
- **Bitwidth/dtype**: same FP4 E2M1 + 16-block FP8 E4M3 inner + FP32
  per-tensor outer as the PTQ NVFP4, but applied to **forward-pass
  weights, activations, and gradients** during *pretraining*.
- **Training workflow**:
  - Each linear's weights are stored in NVFP4 in the forward pass.
    FP32 master weights are maintained for the optimizer (same idea
    as bf16 mixed precision training, but the rounding step targets
    NVFP4).
  - **Stochastic rounding** on the FP4 quantization: probability of
    rounding up is proportional to the distance to the upper FP4
    grid point. Eliminates the systematic bias of round-to-nearest
    at low precision.
  - **Random Hadamard Transforms (RHT)** at each linear: a randomized
    Hadamard rotation is applied to the activation, the linear is
    computed in NVFP4, then the inverse rotation is applied. This
    bounds block-level outliers — bringing QuaRot's idea (Era 5 PTQ
    side) into the training loop. The Hadamard matrix is re-randomized
    each step so the inverse is bit-exact and the rotation doesn't
    have to be stored.
  - **Selective high-precision layers** stay in BF16: the
    first/last attention projection in each block, LayerNorms,
    embeddings, the LM head. Empirically these layers are
    quantization-sensitive enough that the gain from NVFP4 is
    swamped by the perplexity hit.
  - **Two-dimensional quantization**: both forward and backward pass
    use NVFP4 representations; the same block-16 FP8 E4M3 inner-scale
    grid is reused on both sides for numerical consistency.
- **Validation**: 12B hybrid Mamba-Transformer pretrained on 10
  trillion tokens; final validation loss matched a BF16 baseline
  within 0.1%; downstream accuracy on MMLU Pro, code, math, reasoning
  matched across six benchmark domains.
- **Master-weight storage**: FP32 master weights are kept on optimizer
  state only; they are discarded at the end of training. Released
  checkpoint inherits only the NVFP4 weights.
- **Runtimes**: TRT-LLM `nvfp4` and Nemotron-specific paths; vLLM 0.7
  via the same path as PTQ NVFP4 (checkpoint format is identical);
  inference does not re-quantize.
- **IR axes touched**: `qdtype = nvfp4`, `training_native = True`,
  `retrainable = False` (the FP32 master weights are not shipped, so
  you cannot continue-pretrain; LoRA on top is supported).
- **Distinction from NVFP4 PTQ (§3.23)**: byte-identical storage. The
  *flag* difference matters: NVFP4-trained checkpoints exhibit
  better OOD perplexity because the training loss converged with the
  forward-pass quantization in place; NVFP4 PTQ checkpoints, while
  often close, exhibit a 0.1-0.3 perplexity gap on hard prompts.

### 3.25 AFP8 — asymmetric FP8

A category, not a single scheme. Some shipped fp8 paths add a fp16 zero
to the fp8 weight for layers with strongly skewed distributions. Used
in Intel Gaudi-3 paths and some TRT-LLM smoothquant-fp8 recipes. Not
standardized.

### 3.26 VPTQ — Vector Post-Training Quantization (`microsoft/VPTQ`, 2024)

- **Bitwidth/dtype**: 1.5–2 bits effective.
- **Codebook**: trained vector quantization on weight blocks of size
  4–8; codebook entries are 4-8-dimensional vectors.
- **Calibration**: Hessian-weighted reconstruction.
- **Runtimes**: VPTQ repo + HF integration.

### 3.27 LLM.int8() (Dettmers, arXiv:2208.07339, Aug 2022)

- **Bitwidth/dtype**: INT8 weights + INT8 activations with per-row
  scaling. Outlier feature columns (~0.1 %, magnitude > 6) decomposed
  to fp16.
- **Calibration**: implicit, runtime threshold-driven decomposition.
- **Runtimes**: bitsandbytes only.

### 3.28 DeepSeek-V3 native FP8 pretraining (NEW IN v3 — promoted from v2 note)

- **Provenance**: DeepSeek-V3 Technical Report (Dec 2024 → refined Mar
  2025 → DeepSeek-V3.2-Exp Sept 2025 → DeepSeek-V3.2 stable Dec 2025);
  `github.com/deepseek-ai/DeepSeek-V3`.
- **Bitwidth/dtype**: FP8 E4M3 (NVIDIA `e4m3fn` semantics) throughout
  the forward pass.
- **Training workflow** — *"per-tile / per-row" two-axis scaling*:
  - **Weights**: tiled into 128×128 blocks along the (N, K) axes; each
    tile has its own FP32 scale. Inside the tile, weights are FP8
    E4M3. This is the *per-tile weight scaling*.
  - **Activations**: scaled per-row (per token-position). Each row of
    the activation matrix has its own FP32 scale recomputed each
    forward pass.
  - **FP32 master weights** maintained for the optimizer; gradients
    computed in BF16 then accumulated to FP32; weight updates applied
    in FP32 then re-quantized to FP8 for the next forward pass.
  - A **`BF16_BIAS` correction tensor** is folded into the LayerNorm
    output to compensate for systematic FP8 quantization bias —
    similar in spirit to the per-channel scale folding in SmoothQuant
    but learned during training.
  - **Per-tile scale recomputed each step** (not delayed/EMA, unlike
    NVIDIA TE's `fp8_delayed` recipe). This is more expensive but
    yields better quality on the 671B-class scale.
- **Released checkpoint**: FP8 E4M3 weights + the FP32 per-tile
  scales + the BF16 bias correction. Inference loaders consume the FP8
  weights directly; no re-quantization.
- **Runtimes**: SGLang (canonical for DeepSeek-V3 inference); vLLM
  0.6+; TRT-LLM via converted checkpoint; HF Transformers via a
  DeepSeek-V3 custom path. Most inference is per-tile FP8 weight +
  per-token FP8 activation, with the FP32 outer scales fused into the
  output projection.
- **IR axes touched**: `qdtype = fp8_e4m3fn`, `training_native = True`,
  `scale_axis = per_tile(128, 128)`, `bias_correction = True`,
  `retrainable = False` (the per-tile scales are recomputed when you
  continue-pretrain, but the bias correction would have to be
  re-trained; in practice most fine-tuning of V3 happens in BF16).
- **Distinction from PTQ FP8 (§4.6)**: byte-level format is similar but
  the per-tile (128,128) scale grid and the BF16 bias correction are
  DeepSeek-specific. A generic FP8 PTQ loader cannot read DeepSeek-V3
  checkpoints without the bias correction codepath. Conversely, the
  DeepSeek-V3 native-FP8 format is *not* directly loadable as a
  vanilla vLLM FP8 checkpoint without a conversion step.

---

## §4. Family B — Weight + Activation quantization

### 4.1 SmoothQuant (Xiao et al., arXiv:2211.10438, Nov 2022)

- **Bitwidth/dtype**: W8 INT8, A8 INT8.
- **Grouping**: W per-channel (output axis), A per-token dynamic or
  per-tensor static.
- **Calibration**: PTQ. Compute per-channel activation magnitudes
  `s_j = max|X_:,j|^α / max|W_j,:|^(1-α)` with `α ∈ [0.5, 0.8]`.
  Divide A by `s`, multiply W by `s` (diagonal channelwise).
- **Runtimes**: TRT-LLM, OpenVINO via NNCF, Intel Neural Compressor,
  Qualcomm AIMET, vLLM via `compressed-tensors`.

### 4.2 Vanilla INT8 W8A8 (dynamic and static)

- **Dynamic per-token**: activations quantized at inference with
  per-token absmax. No calibration.
- **Static per-tensor**: activation amax frozen from calibration.
- **Runtimes**: TRT-LLM, vLLM, OpenVINO, ONNX Runtime `QLinearMatMul`,
  TFLite full-int8, ExecuTorch, MLX.

### 4.3 ZeroQuant W8A8 / W4A8 (see §3.9)

### 4.4 QuaRot W4A4 (Apr 2024, see §3.13)

### 4.5 SpinQuant W4A4 / W4A8 (May 2024, see §3.14)

### 4.6 FP8 — E4M3 / E5M2 PTQ (Micikevicius et al., arXiv:2209.05433)

- **Bitwidth/dtype**: 8-bit float.
  - `E4M3`: 4 exp / 3 mantissa, range ±448. Forward path.
  - `E5M2`: 5 exp / 2 mantissa, range ±57344. Gradient path in training.
- **Grouping**: per-tensor (TE default), per-row/per-channel (vLLM,
  TRT-LLM), per-token (activations).
- **Scale dtype**: fp32 or fp16.
- **Calibration**: delayed scaling (TE EMA), static amax (vLLM), or
  dynamic (per-token).
- **Runtimes**: NVIDIA Hopper / Blackwell native FP8; AMD MI300 FP8 with
  `fnuz` variants; Intel Gaudi-2/3 FP8 (Gaudi-3 OCP-compliant);
  TRT-LLM `fp8`, vLLM `fp8`, SGLang `fp8`, CUTLASS, cuBLAS-Lt.
- **Sub-variants**:
  - `fp8_e4m3fn` (NVIDIA): no inf, no -0; range ±448.
  - `fp8_e4m3fnuz` (AMD): finite, no unsigned zero, different bias.
  - `fp8_e5m2`, `fp8_e5m2fnuz` (same NVIDIA vs AMD distinction).
- `training_native = False` for this entry (PTQ). The training-native
  FP8 variant is DeepSeek-V3's per-tile recipe at §3.28.

### 4.7 W4A8 — INT4 weights + INT8 activations

TRT-LLM `W4A8_AWQ` and `W4A8_QSERVE` (Lin et al., arXiv:2405.04532, May
2024), ExLlamaV3 partial.

### 4.8 INT4 W4A4 — beyond QuaRot/SpinQuant

Direct W4A4 without rotation is rare in shipped runtimes; perplexity
degrades sharply.

---

## §5. Family C — GGUF k-quants and i-quants

GGUF is llama.cpp's storage format. K-quants (mid-2023) and i-quants
(early 2024) are the *de facto* deployment quantization for CPU and
Apple-Silicon SLM inference.

### 5.1 K-quant super-block structure

A **super-block** is 256 weights, denoted `QK_K = 256`. The sub-block
size is **not the same across all k-quants**:

| Variant | Sub-block size | Sub-blocks per super-block | Sub-scale bits | Bits/weight |
|---------|----------------|-----------------------------|-----------------|-------------|
| Q2_K    | 16             | 16                          | 4-bit sc + 4-bit min | 2.5625 |
| Q3_K    | 16             | 16                          | 6-bit sc       | 3.4375 |
| Q4_K    | **32**         | **8**                       | 6-bit sc + 6-bit min | 4.5 |
| Q5_K    | **32**         | **8**                       | 6-bit sc + 6-bit min | 5.5 |
| Q6_K    | 16             | 16                          | 8-bit sc       | 6.5625 |
| Q8_K    | 16             | 16                          | fp32 sc        | 8.5 (intermediate, not stored) |

(For full code-level layout, struct definitions, and dequant arithmetic
see v2 §5.1; unchanged.)

### 5.2 K-quant _S / _M / _L sub-variants

`Q4_K_M` upgrades `attn.v` and `ffn.down` to Q6_K; `Q4_K_L` additionally
upgrades `attn.k`, `ffn.up`. Analogously for Q3_K_S/M/L, Q5_K_S/M.
Unchanged from v2.

### 5.3 K-quant calibration — imatrix weighted-MSE objective

(Unchanged from v2.) The `--imatrix` file contains per-tensor 1D float
arrays reweighting the MSE inside the k-quant rounder; it does not
pre-multiply weights.

### 5.4 K-quant runtimes

llama.cpp / ggml on every backend; LM Studio; Ollama; Jan; GPT4All;
llamafile; candle (Rust); MLX via gguf import; bitnet.cpp (BitNet
fallback); KleidiAI on Arm.

**This is the dominant on-device SLM quantization format by deployment
count.** llama.cpp's k-quants alone account for >80 % of public GGUF
downloads on HuggingFace as of mid-2026.

### 5.5 I-quants (IQ family)

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
| `IQ3_M`   | 3.66        | layer-mixed |
| `IQ4_XS`  | 4.25        | non-uniform 16-entry LUT per super-block |
| `IQ4_NL`  | 4.5         | "non-linear int4" — 16-entry LUT shared globally |

All IQ formats require imatrix calibration to be effective.

### 5.6 GGUF as the canonical export for QAT checkpoints (NEW IN v3)

A 2025-H2 / 2026-H1 pattern: vendor QAT checkpoints (Gemma 4 INT4 QAT,
Falcon-Edge BitNet i2_s, GPT-OSS MXFP4) are now routinely exported to
GGUF for community on-device inference. The GGUF format does not
"know" about QAT vs PTQ provenance — it merely stores the quantized
weights. The IR's `training_native` flag must therefore be carried as
a metadata field in the GGUF metadata KV store (a new key
`quantization.training_native: bool`), not inferred from the qdtype.

---

## §6. Family D — Codebook quantization (cross-family deep dive)

NF4, AQLM, QuIP/QuIP#, EXL2, SqueezeLLM, VPTQ, IQ all use a codebook
in some form. (Unchanged from v2 in structure.)

### 6.1 Codebook taxonomy

| Scheme        | num_codebooks | combine_op       | shared_across   | Notes |
|---------------|---------------|------------------|-----------------|-------|
| NF4           | 1             | NONE             | ALL (global)    | Fixed quantile codebook |
| FP4 (bnb)     | 1             | NONE             | ALL             | Fixed E2M1 codebook |
| AQLM          | K ∈ {1,2,4,8} | ADD              | ALL             | Additive multi-codebook |
| QuIP#         | 1             | MULTI_LATTICE    | ALL             | E8 lattice + Hadamard incoherence |
| EXL2          | 1             | NONE             | PER_ROW         | Per-row variable bitwidth |
| SqueezeLLM    | 1             | NONE             | PER_ROW         | Per-row k-means codebook |
| VPTQ          | 1             | NONE             | ALL or PER_LAYER | Vector codebook |
| IQ (i-quant)  | 1             | MULTI_LATTICE    | ALL             | E8 lattice port to GGUF |
| IQ4_NL        | 1             | NONE             | ALL             | Non-linear 16-entry LUT |
| IQ4_XS        | 1             | NONE             | PER_SUPER_BLOCK | Per-super-block LUT |

### 6.2 Additive codebook math (AQLM)

```
W_block[1..8] = C₁[idx₁] + C₂[idx₂]
```

(Detailed example unchanged from v2 §6.3.)

---

## §7. Family E — KV-cache quantization

(Unchanged from v2 in structure; v3 adds the Apple AFM cross-block KV
sharing as §7.10.)

### 7.1 K vs V asymmetry

K is much more outlier-prone than V, especially for RoPE'd K. Real
engines treat K and V with different schemes.

### 7.2 INT8 KV cache (vLLM, llama.cpp, TRT-LLM)

(See v2 §7.2.)

### 7.3 INT4 KV cache — KIVI (Liu et al., arXiv:2402.02750, Feb 2024)

(See v2 §7.3.)

### 7.4 FP8 KV cache (TRT-LLM, vLLM, SGLang)

(See v2 §7.4.)

### 7.5 KVQuant (Hooper et al., NeurIPS 2024, arXiv:2401.18079)

(See v2 §7.5.)

### 7.6 SmoothQuant for KV

(See v2 §7.6.)

### 7.7 CacheGen (Liu et al., arXiv:2310.07240)

(See v2 §7.7.)

### 7.8 PoUKey (Q1 2026)

(See v2 §7.8.)

### 7.9 Per-token vs per-channel scaling rationale

(See v2 §7.9.)

### 7.10 Apple AFM cross-block KV sharing (NEW IN v3)

- **Provenance**: Apple Foundation Models 2025 Updates (machinelearning.apple.com);
  arXiv:2507.13575.
- **Idea**: the on-device 3.18B decoder is split into Block-1 (62.5%
  of layers) and Block-2 (37.5% of layers). All Block-2 KV caches are
  *directly shared* — pointer-identical — with the **final layer of
  Block-1**'s KV. KV memory is reduced by 37.5%.
- **Quantization**: the KV cache is INT8 (per-token symmetric, fp16
  scale per token). When Block-2 reads the shared KV, it dequantizes
  on-the-fly to BF16 for the attention compute.
- **IR consequence**: the KV-cache `role` axis must carry a
  `shared_from_layer_idx: Optional[int]` reference, distinct from
  per-layer-independent KV. This is a new axis specifically for
  cross-block KV sharing.
- `training_native = True` (the model was trained from scratch with
  the sharing in place).

### 7.11 GQA awareness

(Unchanged from v2 §7.10.)

### 7.12 IR axes for KV cache

```
kv_k_qdtype, kv_v_qdtype (independent)
kv_k_quant_axis ∈ {per_channel, per_token, per_head, per_tensor}
kv_v_quant_axis ∈ {per_channel, per_token, per_head, per_tensor}
kv_scale_dtype ∈ {fp32, fp16, e8m0, fp8_e4m3}
kv_k_has_zero_point, kv_v_has_zero_point
kv_residual_buffer: int
kv_outlier_sparse_buffer: bool
kv_shared_from_layer_idx: Optional[int]  # NEW in v3 (AFM)
```

---

## §8. Family F — Attention-internal quantization

### 8.1 FP8 SDPA — FlashAttention-3 (Shah et al., arXiv:2407.08608)

(Unchanged from v2; per-row scaling on P maintains numerical sanity at
FP8 dynamic range.)

### 8.2 INT8 quantized attention (TurboMind / LMDeploy)

(Unchanged from v2.)

### 8.3 FP8 sub-format zoo

(Unchanged from v2.)

### 8.4 FA3 worked example (per-row scale flow)

(Unchanged from v2.)

---

## §9. Family G — Sub-2-bit quantization

### 9.1 BitNet b1.58 (Ma et al., arXiv:2402.17764, Feb 2024)

- **Weight bitwidth**: ternary {-1, 0, +1}, i.e. 1.58 bits.
- **Activation bitwidth**: INT8 with per-token absmax.
- **Trained from scratch**: BitNet is **not a PTQ scheme**. The
  architecture uses `BitLinear` layers with straight-through-estimator
  gradients on the rounding. The model is pretrained with ternary
  weights from initialization.
- **Activation**: ReLU² in the FFN; RMSNorm; no bias.
- **Runtimes**:
  - **bitnet.cpp** (Microsoft, Oct 2024): CPU LUT-based GEMV kernels.
  - **T-MAC** (Microsoft, 2024).
  - **KleidiAI** (Arm, Q3 2025): LUT-based ternary on Cortex-X4.
  - **vLLM**: experimental BitNet loader, GPU LUT kernel.
  - **HF Transformers**: `BitNetForCausalLM` class.
- `training_native = True`, `retrainable = False`.

### 9.2 Falcon-Edge 1.58-bit retrainable BitNet (NEW IN v3 — TII Abu Dhabi, May 2025)

- **Provenance**: TII Abu Dhabi `tiiuae/Falcon-E-1B-Base`,
  `Falcon-E-3B-Base`, plus the instruct variants. Released
  **2025-05-15**. Blog post `huggingface.co/blog/tiiuae/falcon-edge`.
  Companion package: `onebitllms` on PyPI / GitHub.
- **Weight bitwidth**: ternary {-1, 0, +1} (same as BitNet b1.58 in
  inference), but the *training and release workflow* are different.
- **Training paradigm** — *"three models from one process"*:
  - Single end-to-end pretraining pass on ~1.5 trillion tokens with a
    WSD (warmup-stable-decay) learning rate schedule.
  - At the end, the same training run produces three checkpoint
    variants on HuggingFace:
    1. **Native BitNet** (default revision): packed ternary weights,
       GGUF `i2_s` format, ~635 MB for 1B params, ~999 MB for 3B
       params. Loadable by bitnet.cpp / llama.cpp / mlx-lm.
    2. **Pre-quantized** (`revision="prequantized"`): BF16 weights
       *scaled such that* loading them and applying the
       `replace_linear_with_bitnet_linear(model)` conversion
       reproduces the ternary forward pass bit-exactly. This is the
       checkpoint a user fine-tunes from.
    3. **bfloat16 non-BitNet** (`revision="bfloat16"`): full BF16
       weights, ~2 GB for 1B params. Useful as a baseline.
- **Retrain workflow** (the headline difference vs BitNet b1.58):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from onebitllms import (
    replace_linear_with_bitnet_linear,
    quantize_to_1bit,
)

model_id = "tiiuae/Falcon-E-1B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="prequantized")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    revision="prequantized",
)
model = replace_linear_with_bitnet_linear(model)  # in-place
trainer = SFTTrainer(model=model, ...)
trainer.train()
quantize_to_1bit(output_directory)  # final ternary
```

  The pre-quantized BF16 weights are *not* the original BF16 weights
  of a regular model that was then ternarized; they are a special
  representation in which the BF16 scale per channel exactly compensates
  for the BitnetLinear's runtime quantization. This means the BF16
  forward and the ternary forward produce **bit-exact** outputs after
  the BitnetLinear conversion. Continued pretraining or fine-tuning
  therefore behaves as if you were training in BF16 mixed precision,
  while inference uses the ternary form.
- **Storage formats**:
  - Native BitNet: GGUF `i2_s` (Microsoft BitNet PR #268 layout).
    Packed pairs of ternary values in 3 bits per 2 weights, or
    5 weights per 8 bits.
  - Pre-quantized: SafeTensors BF16.
  - bfloat16: SafeTensors BF16.
- **Runtimes**:
  - **bitnet.cpp** with PR #268 (community-merged late 2025).
  - **llama.cpp**: i2_s loader, community-validated.
  - **mlx-lm**: Apple Silicon native via
    `mlx_lm.generate --model tiiuae/Falcon-E-1B-Base`.
  - **HF Transformers**: BF16 variant via standard `AutoModelForCausalLM`.
  - **vLLM / SGLang**: via OpenAI-compatible API serving the BF16
    weights.
  - **T-MAC, KleidiAI**: not explicitly documented as of Q4 2025;
    expected to work via the bitnet.cpp PR #268 layout.
- **Performance**: TII reports <1% gap between ternary and bfloat16
  variants on HF Open LLM Leaderboard v2.
- **IR axes touched**: `qdtype = ternary`, `training_native = True`,
  `retrainable = True` (the headline new axis), `packing =
  ternary_pair_3bit` or `ternary_5x8bit`, plus a sibling
  `prequantized_revision: str` field pointing to the BF16
  retrainable form.
- **Distinction from BitNet b1.58**: Falcon-Edge is the first BitNet-
  style model with a *documented and supported* retrain workflow on
  the released checkpoint. BitNet b1.58 checkpoints are inference-
  only; if you want to fine-tune them you need to start from
  Microsoft's research-only training infrastructure. This is the
  reason for the `retrainable: bool` axis in v3.

### 9.3 BiLLM (Huang et al., arXiv:2402.04291, Feb 2024)

(Unchanged from v2 §9.2.)

### 9.4 Apple AFM 2-bpw QAT (NEW IN v3 — Apple, mid-2025 → production 2026)

- **Provenance**: Apple Intelligence Foundation Language Models Tech
  Report 2025 (machinelearning.apple.com); arXiv:2507.13575. Apple's
  on-device 3.18B model has been in production since iOS 18 (late
  2024); the 2025 tech report and 2026-H1 minor updates documented
  the quantization format publicly.
- **Bitwidth/dtype per tensor role**:
  - **Decoder linear weights**: **2 bits per weight (2-bpw)** via
    QAT. Stored with per-channel learnable clipping bounds.
  - **Embedding table**: **4-bit** via QAT.
  - **KV cache**: **INT8** (per-token, per-head).
  - **LM head**: BF16 (shared with the embedding table is *not* the
    case — the LM head is full-precision).
- **QAT recipe — *"novel learnable weight clipping and weight
  initialization"***:
  - Each linear has a learnable clipping range `[-c, +c]` per output
    channel. The forward pass quantizes weights to 2 bits within
    that range; the backward pass updates both the weights and the
    clipping bound via STE.
  - Weight initialization is a structured Gaussian-mixture variant
    that places 2-bit values closer to the natural weight
    distribution than naive uniform initialization.
  - QAT runs as a *fine-tuning pass* after the main BF16 pretraining,
    typically 50-100k steps with reduced LR.
- **Distinct from prior Apple QAT (AFM-2024)**: AFM-2024 used a 4-bit
  QAT with per-channel scales; the 2-bpw recipe is genuinely new in
  2025 and ships in iOS 18/19. The embedding-table-INT4 and KV-INT8
  splits are unchanged across the two generations.
- **Quality impact**: ~4.6% MGSM regression, +1.5% MMLU vs the BF16
  baseline.
- **Runtimes**: closed Apple Foundation Models toolchain; runs on
  Apple Neural Engine, Apple GPU (Metal), and Apple Silicon CPU
  fallback. No public runtime exposes the format.
- **IR axes touched**: `qdtype = int2_qat` (a new qdtype value, with
  per-channel learnable clipping metadata), `embed_qdtype = int4`,
  `kv_qdtype = int8`, `training_native = False` (QAT is fine-tune-
  time), `retrainable = False` (in principle yes, but Apple's
  toolchain is closed).

### 9.5 Gemma 4 mobile INT4 QAT (NEW IN v3 — Google, June 2026)

- **Provenance**: Google's Gemma 4 family launched 2026-04-02
  (E2B/E4B/12B/26B-A4B/31B); the **June 2026 QAT addendum**
  released dedicated INT4-QAT checkpoints for the on-device E2B and
  E4B variants. Reference: `huggingface.co/blog/gemma4`, Google AI
  Edge docs, LiteRT 1.1 release notes.
- **Bitwidth/dtype per tensor role**:
  - **Decoder linear weights**: **INT4 per-channel symmetric** via
    QAT with a fp16 per-row scale.
  - **Activations**: INT8 dynamic per-token at inference (the
    activations *during QAT* are also INT8 per-token).
  - **Embeddings**: INT8.
  - **LM head**: shared with embedding (Gemma family convention),
    therefore INT8.
- **Headline compression**: **Gemma 4 E2B (~2 GB BF16) → ~1 GB INT4
  QAT**. Gemma 4 E4B (~4 GB BF16) → ~2 GB INT4 QAT. Sub-1% MMLU
  regression vs BF16 baseline.
- **QAT recipe**:
  - Runs as a fine-tuning pass after the main BF16 pretraining and
    multimodal alignment, typically 100k steps on a curated mixed
    text+image+audio dataset.
  - Per-channel INT4 symmetric quantization with the scale **learned
    via STE** during the QAT pass, rather than fixed at the absmax.
  - The Gemma 4 "Elastic" variants (E2B, E4B) retain their elastic
    parameter routing — the INT4 QAT is applied per-shared-activation
    group, not per-monolithic-layer.
- **Distinct from Gemma 3 QAT**: Gemma 3 had only INT8 QAT for the
  LiteRT path. Gemma 4 is Google's first INT4 QAT for an
  on-device-deployable Gemma.
- **Storage format**: a custom TFLite-style flatbuffer with
  per-channel INT4 weights packed as nibbles, fp16 row scales, and
  the model graph in LiteRT 1.1 format. Also exported as GGUF
  `Q4_K_M` for the llama.cpp ecosystem, but the bit-exact behavior
  on Android Hexagon NPU / Apple Neural Engine / Mali GPU is via
  the LiteRT path.
- **Runtimes**:
  - **Google AI Edge / LiteRT 1.1** (Android, iOS).
  - **MediaPipe Tasks**.
  - **HF Transformers** via the `gemma-4-e2b-it-int4-qat` checkpoint.
  - **GGUF Q4_K_M** export for llama.cpp / Ollama / LM Studio.
- **IR axes touched**: `qdtype = int4`, `training_native = False`
  (QAT not pretrain), `calibration = qat`, `target_runtime ∈
  {litert, mediapipe}` (new value), `retrainable = False` (you can
  LoRA on top of the BF16 underlying, then re-run QAT, but not
  continue-pretrain directly on the QAT weights).

### 9.6 Why ternary works where binary often fails

Binary forces every weight to ±1 — there is no zero. Ternary's zero
level captures the sparse mass at 0, hence the perplexity-vs-bits
Pareto improvement. This is why both BitNet b1.58 and Falcon-Edge
chose ternary not binary. BiLLM compensates for binary's limitations
via outlier offload.

---

## §10. Parameter space — the IR axes

This section enumerates the axes required to *represent* every scheme
above. v2 had 7 minimal + 12 extended = 19 axes; v3 has **7 minimal +
14 extended = 21 axes** (adds `training_native` and `retrainable`).

### 10.1 Minimal 7 axes

```python
MinimalQuantSpec = {
  "qdtype":           Enum,
  "group_size":       int | None,
  "quant_axis":       int | Enum,
  "scale_dtype":      Enum,
  "has_zero_point":   bool,
  "packing":          Enum,
  "accumulator_dtype":Enum,
}
```

### 10.2 Extended 14 axes (v3)

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
  "imatrix_calibrated":   bool,
  "calibration":          Enum,
  "compute_dtype":        Enum,
  "role":                 Enum,
  "training_native":      bool,        # NEW in v3
  "retrainable":          bool,        # NEW in v3
}
```

### 10.3 Per-axis enumeration

#### 10.3.1 `qdtype` (v3 additions in **bold**)

```
Enum[
  int8, int4, int3, int2, int1, ternary,
  int2_qat,                                # NEW in v3 (AFM)
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
  squeezellm_codebook, exl2_variable,
]
```

#### 10.3.2 `training_native: bool` (NEW)

True iff the model's **pretraining** loop used the quantized
representation (not just QAT fine-tuning). True for:

- GPT-OSS MoE expert weights (MXFP4-native post-training; debatable
  edge case — v3 sets True because the post-training is the dominant
  training-time exposure).
- DeepSeek-V3 / V3.2 FP8 native pretraining.
- Nemotron-3 family NVFP4 native training.
- BitNet b1.58 (ternary from initialization).
- Falcon-Edge 1.58-bit (ternary from initialization).
- Some early MX-format training experiments.

False for all PTQ schemes. False for QAT (QAT is fine-tune-time).

The distinction matters for two reasons:

1. *Re-quantization safety*: a `training_native = True` checkpoint
   should not be re-quantized — its weights are already at the
   quantization grid, and additional rounding compounds error against
   weights that were trained to that grid.
2. *OOD perplexity prediction*: empirically,
   `training_native = True` checkpoints have flatter OOD perplexity
   curves than PTQ at the same nominal bitwidth.

#### 10.3.3 `retrainable: bool` (NEW)

True iff the released checkpoint supports continued pretraining or
domain fine-tuning *on the quantized representation itself*. True for:

- Falcon-Edge 1.58-bit (via the `prequantized` BF16 revision and the
  `onebitllms` workflow).
- QLoRA / NF4 (LoRA adapters on frozen NF4 weights; partial
  retrainability — adapter only, not base).

False for all current native-trained checkpoints other than
Falcon-Edge (BitNet b1.58, GPT-OSS MXFP4, DeepSeek-V3 FP8,
Nemotron-3 NVFP4) because the FP32 master weights from training were
not released.

False for all PTQ schemes by default — though in principle you can
dequantize-and-retrain any PTQ checkpoint, the IR sets this to False
to indicate "no first-class retrain workflow shipped with the
checkpoint".

#### 10.3.4 `group_size`

`int | None`. For fixed-block schemes (MXFP4 = 32, NVFP4 = 16),
set by the format. For k-quants, super-block 256 and sub-block per
qdtype.

#### 10.3.5 `quant_axis`

```
int | Enum[per_tensor, per_token, per_channel_out, per_channel_in,
          per_tile_2d, flattened]
```

`per_tile_2d` is **new in v3** for DeepSeek-V3's 128×128 weight tiles.

#### 10.3.6 `scale_dtype`

```
Enum[fp32, fp16, bf16, fp8_e4m3, ue8m0, int8, int6, int4]
```

#### 10.3.7 `scale_quant` — recursive

```
QuantSpec | None
```

Examples: NF4 double-quant, NVFP4 outer fp32 / inner E4M3, K-quant
6-bit sub-scales scaled by fp16 super-scale.

#### 10.3.8 `scale_axis`

```
int | Enum[per_tensor, per_row, per_block, per_sub_block, per_head,
          per_token, per_super_block, per_tile_2d]
```

#### 10.3.9 `has_zero_point` and `zero_dtype`

```
has_zero_point: bool
zero_dtype: Enum[same_as_qdtype, fp16, int4, int6] | None
```

#### 10.3.10 `codebook` — extended

(See v2 §10.3.8; unchanged in v3.)

#### 10.3.11 `rotation_kind`

```
Enum[
  NONE,
  DIAGONAL_SMOOTHQUANT,
  DENSE_HADAMARD_QUAROT,
  DENSE_LEARNED_SPINQUANT,
  HADAMARD_RUNTIME,
  RANDOM_HADAMARD_TRAINING,   # NEW in v3 (NVFP4 training)
]
```

The new `RANDOM_HADAMARD_TRAINING` value captures NVFP4 training's
per-step randomized Hadamard, which differs from QuIP#'s
randomized-but-stored Hadamard because in NVFP4 training the
Hadamard is re-randomized each step and never stored (the inverse is
applied immediately after the matmul; round-trip bit-exact by
construction).

#### 10.3.12 `runtime_apply`

```
Enum[NONE, INPUT, INPUT_OUTPUT_BOTH, RING_AROUND_MATMUL]
```

The new `RING_AROUND_MATMUL` value is NVFP4 training (Hadamard
applied to input, matmul, then inverse Hadamard immediately after —
all inside a single fused kernel).

#### 10.3.13 `variable_bitwidth`

```
variable_bitwidth: bool
bitwidth_per_row: int[N] | None
```

#### 10.3.14 `outlier_offload`

```
Enum[
  NONE,
  FP16_COLUMN_SPLIT,
  FP16_ROW_SPLIT,
]
```

Plus `outlier_selection`.

#### 10.3.15 `packing`

(See v2; unchanged.)

#### 10.3.16 `imatrix_calibrated`

(See v2.)

#### 10.3.17 `calibration` — refined

```
Enum[
  none, rtn, awq, gptq_hessian, smoothquant, hqq,
  imatrix, percentile, amax_ema, qat,
  qat_learnable_clipping,    # NEW in v3 (AFM 2-bpw)
  qat_mobile_int4,           # NEW in v3 (Gemma 4 mobile)
  learnable_clipping,        # OmniQuant
  data_free,
  quip_incoherence,
  spinquant_riemannian,
  training_native_mxfp4,     # NEW in v3 (GPT-OSS post-train)
  training_native_nvfp4,     # NEW in v3 (Nemotron-3)
  training_native_fp8_tile,  # NEW in v3 (DeepSeek-V3 per-tile)
  training_native_bitnet,    # NEW in v3 (BitNet, Falcon-Edge)
]
```

#### 10.3.18 `compute_dtype` separate from `qdtype`

(See v2.)

#### 10.3.19 `accumulator_dtype`

`Enum[fp32, fp16, bf16, int32]`.

#### 10.3.20 `role`

```
Enum[weight, activation, kv_k, kv_v, attn_qk_score, attn_pv_score,
     bias, embed, lm_head, moe_expert_weight, moe_router_weight]
```

The `moe_expert_weight` and `moe_router_weight` values are **new in
v3** to support GPT-OSS's per-role MXFP4-vs-BF16 split.

#### 10.3.21 `target_runtime` (NEW)

```
Enum[
  generic,
  vllm, sglang, trt_llm,
  llama_cpp, mlx, ollama,
  litert, mediapipe,           # NEW in v3 (Gemma 4 mobile)
  apple_foundation_models,     # AFM
  bitnet_cpp, t_mac, kleidiai,
]
```

Documents which runtime the checkpoint was packed for. Marlin
repacking, KleidiAI LUT precomputation, and LiteRT flatbuffer all
require this hint.

### 10.4 Side-by-side: which axes each scheme touches (v3 extension)

| Scheme            | qdtype  | training_native | retrainable | calibration                  | rotation_kind            | role split |
|-------------------|---------|------------------|-------------|------------------------------|--------------------------|------------|
| GPTQ              | int4    | False            | False       | gptq_hessian                 | NONE                     | weight     |
| AWQ               | int4    | False            | False       | awq                          | NONE                     | weight     |
| NF4               | nf4     | False            | True (LoRA) | data_free                    | NONE                     | weight     |
| SmoothQuant       | int8    | False            | False       | smoothquant                  | DIAGONAL_SMOOTHQUANT     | weight+act |
| QuaRot W4A4       | int4    | False            | False       | quip_incoherence             | DENSE_HADAMARD_QUAROT    | weight+act |
| SpinQuant         | int4    | False            | False       | spinquant_riemannian         | DENSE_LEARNED_SPINQUANT  | weight+act |
| OmniQuant         | int4    | False            | False       | learnable_clipping           | DIAGONAL_SMOOTHQUANT     | weight     |
| Q4_K_M            | q4_k    | False            | False       | imatrix                      | NONE                     | weight     |
| IQ2_M             | iq2_m   | False            | False       | imatrix                      | NONE                     | weight     |
| BitNet b1.58      | ternary | True             | False       | training_native_bitnet       | NONE                     | weight+act |
| **Falcon-Edge**   | ternary | **True**         | **True**    | training_native_bitnet       | NONE                     | weight+act |
| BiLLM             | int1    | False            | False       | gptq_hessian (variant)       | NONE                     | weight     |
| MXFP4 PTQ         | mxfp4   | False            | False       | data_free                    | NONE                     | weight     |
| **MXFP4-native**  | mxfp4   | **True**         | False       | training_native_mxfp4        | NONE                     | moe_expert_weight |
| NVFP4 PTQ         | nvfp4   | False            | False       | rtn / awq                    | NONE                     | weight     |
| **NVFP4-trained** | nvfp4   | **True**         | False       | training_native_nvfp4        | RANDOM_HADAMARD_TRAINING | weight+act |
| FP8 PTQ           | fp8_e4m3fn | False         | False       | amax_ema or static           | NONE                     | weight+act |
| **DeepSeek FP8**  | fp8_e4m3fn | **True**      | False       | training_native_fp8_tile     | NONE                     | weight (per_tile_2d) |
| **AFM 2-bpw**     | int2_qat| False            | False       | qat_learnable_clipping       | NONE                     | weight+embed+kv |
| **Gemma 4 INT4 QAT** | int4 | False            | False       | qat_mobile_int4              | NONE                     | weight |
| Marlin (W4A16)    | int4    | False            | False       | inherits from source         | NONE                     | weight     |
| KleidiAI LUT      | int4    | False            | False       | inherits from source         | NONE                     | weight     |

The four bold rows in the `training_native` column are the v3
additions; the `retrainable` column has one True row (Falcon-Edge).

### 10.5 Axes added vs v2

- **NEW**: `training_native` (bool) — distinguishes
  pretraining-time-quantized from PTQ/QAT.
- **NEW**: `retrainable` (bool) — distinguishes Falcon-Edge's
  documented retrain workflow from all other native-trained
  checkpoints.
- **NEW**: `target_runtime` (enum) — distinguishes Gemma 4 mobile QAT
  (LiteRT target) from server quant paths.
- **Extended**: `qdtype` gains `int2_qat`.
- **Extended**: `calibration` gains `qat_learnable_clipping`,
  `qat_mobile_int4`, and four `training_native_*` values.
- **Extended**: `rotation_kind` gains `RANDOM_HADAMARD_TRAINING`.
- **Extended**: `runtime_apply` gains `RING_AROUND_MATMUL`.
- **Extended**: `quant_axis` and `scale_axis` gain `per_tile_2d`.
- **Extended**: `role` gains `moe_expert_weight`, `moe_router_weight`,
  `lm_head`.
- **Extended**: KV-cache spec gains `kv_shared_from_layer_idx`.

The total axis count is now 7 minimal + 14 extended = 21 axes.

---

## §11. "If you support six, support these six" — extended from v2's five

v2 recommended a 5-scheme set that exercised every IR axis present in
2025-Q1. v3 promotes the recommendation to **6 schemes** to cover (a)
training-native MXFP4 as distinct from PTQ MXFP4, and (b) the
mobile-INT4 QAT recipe that is now Google's standard for on-device
deployment.

### 11.1 The six

1. **GGUF Q4_K_M (with imatrix)** — on-device dominant. Exercises:
   hierarchical scale, asymmetric zero, per-layer override, super-block
   packing, imatrix calibration. **Target runtime**: llama.cpp / Ollama /
   LM Studio. `training_native=False, retrainable=False`.
2. **AWQ INT4 W4A16 g=128 (loaded into Marlin)** — server 4-bit
   dominant. Exercises: blockwise int4, fp16 zero, packed interleave then
   Marlin layout, activation-aware calibration. **Target runtime**:
   vLLM / TRT-LLM / SGLang. `training_native=False, retrainable=False`.
3. **FP8 E4M3 W8A8 (per-tensor W + per-token A)** — compute-quantized
   H100/MI300/Gaudi-3. Exercises: fp accumulator path, FP8 fnuz variant
   on AMD, amax-history calibration. Covers both the PTQ entry (§4.6)
   and the DeepSeek-V3 native-trained entry (§3.28) by toggling
   `training_native`. **Target runtime**: vLLM / TRT-LLM / SGLang.
4. **IQ2_M (or AQLM)** — 2-bit coverage. Exercises codebook axes,
   imatrix calibration, sub-2-bit storage layout. **Target runtime**:
   llama.cpp. `training_native=False, retrainable=False`.
5. **MXFP4** — covering both PTQ (§3.21) and MXFP4-native (§3.22) by
   the `training_native` flag. Exercises: `qdtype=mxfp4`,
   `scale_dtype=ue8m0`, `packing=ocp_mx`, OCP block packing,
   `element_format=e2m1`, fixed `group_size=32`. The native-training
   variant adds `role=moe_expert_weight` and the per-role split with
   BF16 attention. **Target runtime**: Blackwell / MI300X with Quark
   MXFP4 kernels; vLLM 0.7+. The PTQ vs native split is
   architecturally significant because vLLM's load path must skip
   re-quantization on the native variant.
6. **Gemma 4 mobile INT4 QAT (NEW)** — on-device QAT recipe.
   Exercises: `qdtype=int4`, `calibration=qat_mobile_int4`,
   `target_runtime=litert`, per-channel symmetric INT4 with fp16
   row scale, INT8 activations / embed / lm_head. The headline on
   the format: 2 GB BF16 → 1 GB INT4 for Gemma 4 E2B. **Target
   runtime**: LiteRT / MediaPipe (Android), GGUF Q4_K_M export for
   llama.cpp parity. `training_native=False, retrainable=False`.

### 11.2 Axis coverage check (extended for v3)

| Axis                          | Q4_K_M | AWQ-Marlin | FP8 W8A8 | IQ2_M / AQLM | MXFP4 (both) | Gemma 4 mobile-INT4 QAT |
|-------------------------------|--------|------------|----------|---------------|--------------|-------------------------|
| Asymmetric zero               | yes    | yes        | no       | no            | no           | no (symmetric)          |
| Integer zero                  | yes    | no         | no       | no            | no           | no                      |
| Codebook                      | no     | no         | no       | yes           | no           | no                      |
| `combine_op` (multi-codebook) | no     | no         | no       | yes           | no           | no                      |
| Hadamard / rotation           | no     | no         | no       | optional      | no (PTQ); yes (NVFP4 training, by extension) | no |
| Hierarchical scale            | yes    | no         | no       | yes           | no           | no                      |
| UE8M0 scale                   | no     | no         | no       | no            | **yes**      | no                      |
| FP8 inner scale (NVFP4)       | no     | no         | no       | no            | no (NVFP4 needed) | no                |
| Per-row variable bit          | no     | no         | no       | no            | no           | no                      |
| Outlier offloading            | no     | no         | no       | no            | no           | no                      |
| Marlin packing                | no     | **yes**    | no       | no            | no           | no                      |
| imatrix calibration           | **yes**| no         | no       | **yes**       | no           | no                      |
| QAT calibration               | no     | no         | no       | no            | no           | **yes**                 |
| FP accumulator                | no     | no         | **yes**  | no            | **yes**      | no                      |
| Activation per-token          | no     | no         | **yes**  | no            | yes (dynamic)| **yes (INT8 dyn)**      |
| `training_native = True`      | no     | no         | yes (DeepSeek subcase) | no | yes (GPT-OSS subcase) | no              |
| `target_runtime = litert`     | no     | no         | no       | no            | no           | **yes**                 |
| Per-role quant split          | no     | no         | no       | no            | yes (MoE-only)| yes (embed/LM head)     |

The 6-pack covers all listed axes except per-row variable bitwidth
(EXL2-specific), outlier offload (LLM.int8-specific), dense learned
rotation (SpinQuant), and the `retrainable=True` axis (Falcon-Edge).
If your IR must cover those, add **EXL2**, **SpinQuant**, and
**Falcon-Edge** for a 9-pack.

### 11.3 Why MXFP4-native is split from MXFP4 PTQ even though byte-format is the same

In the IR, the two share `qdtype`, `group_size`, `packing`, and
`scale_dtype` — all the storage-level axes are identical. They differ
only in:

- `training_native` (False vs True),
- `calibration` (`data_free` for PTQ vs `training_native_mxfp4` for
  GPT-OSS),
- `role` (uniform-weight for PTQ vs `moe_expert_weight`-only for
  GPT-OSS).

This is the strongest argument for keeping `training_native` as a
separate axis rather than encoding it inside `calibration`: the
storage-level axes are identical, so a generic runtime can dispatch the
same kernel; but the *load-time* behavior must differ (do not
re-quantize the native variant) and the *quality expectations* differ
(better OOD on the native variant). A flag at the QuantSpec level is
the cleanest way to express this.

### 11.4 Why include MXFP4 over NVFP4 for the recommendation

(Carried from v2.) MXFP4 is the more *portable* format — OCP spec,
supported by NVIDIA, AMD, Intel. NVFP4 is NVIDIA-only. v3 also notes
that the **training-native** distinction is currently more clearly
articulated for MXFP4 (GPT-OSS post-trained MoE) than for NVFP4
(Nemotron-3 native pretraining); both exist, but the MXFP4 ecosystem
has the larger and more diverse model count.

### 11.5 Why include mobile-INT4 QAT instead of just supporting GGUF Q4_K_M

A frequently-asked question for v3. Answer: Q4_K_M is the *runtime
format* for on-device inference, but Gemma 4 mobile-INT4 QAT is the
*authoring recipe* that produces a checkpoint with materially better
quality than a Q4_K_M PTQ of the same BF16 base. A llama.cpp loader
sees only the Q4_K_M bytes and cannot tell that they came from a QAT
process. The IR's `calibration = qat_mobile_int4` flag carries this
provenance even when the storage is Q4_K_M, and vendor model cards
should set it to True to communicate quality expectations.

### 11.6 Where retrainable BitNet fits

Falcon-Edge is the 7th-position scheme: if your IR must support a
retrainable on-device family, include it. The `prequantized` BF16
revision is the headline storage detail; the `retrainable=True` axis
is the headline IR detail. Most production IRs in 2026-H1 do not yet
support this — Falcon-Edge is the *only* checkpoint family with the
workflow shipped.

---

## §12. PTQ vs QAT vs training-native — taxonomy by training-time involvement (v3 with three-column split)

The survey's schemes split cleanly along *whether the model was trained
with quantization in mind*. This matters for the IR because the three
classes have different operational semantics.

### 12.1 Pure PTQ (no training-time involvement)

GPTQ, AWQ, HQQ, SmoothQuant, ZeroQuant V1/V2/FP, OWQ, OmniQuant,
SqueezeLLM, AQLM, VPTQ, QuIP/QuIP#, NF4, FP4, MX (weight-side, PTQ
path), NVFP4 PTQ, FP8 PTQ, K-quants, I-quants, BiLLM, KIVI, KVQuant.

These start from an fp16/bf16 pretrained checkpoint and quantize after
the fact. Calibration sets are small (128–512 samples). Reversible:
you can dequantize and re-train.

`training_native = False`, `retrainable = False` (the original BF16 is
not the released artifact, but you could in principle dequantize).

### 12.2 QAT — quantization-aware training (fine-tune-time only)

- **QLoRA (NF4 + LoRA fine-tuning)** (Dettmers et al., May 2023).
  PTQ-of-the-base + LoRA-fine-tune-with-STE-on-the-base.
- **FP8 QAT (NVIDIA TRT-Model-Optimizer)** — adds straight-through
  estimator gradients on the fp8 rounding during a short fine-tuning
  pass (~1k steps).
- **AWQ-aware QAT** — some Llama-3 production deployments at Meta
  fine-tune with AWQ-rescale-aware forward passes.
- **AFM 2-bpw QAT (Apple, 2025-H2 → production 2026)** — learnable
  weight clipping QAT with the embedding-INT4 / KV-INT8 split.
- **Gemma 4 mobile INT4 QAT (Google, June 2026)** — per-channel
  symmetric INT4 with learned scale, applied as a fine-tuning pass
  on the multimodal-aligned BF16 checkpoint.

`training_native = False, retrainable = False` in all QAT cases.

### 12.3 Train-from-scratch native quantized (training_native = True)

- **BitNet b1.58** — ternary weights from initialization.
  `retrainable = False` (Microsoft's training infra not public).
- **Falcon-Edge 1.58-bit** — ternary weights from initialization, but
  with a documented retrain workflow via the `prequantized` revision.
  `retrainable = True`.
- **DeepSeek-V3 / V3.2 native FP8 pretraining** — per-tile FP8 weights,
  per-row activations, BF16 bias correction. `retrainable = False` (the
  per-tile scales and bias correction would need to be re-trained;
  continued pretraining is technically possible but not documented).
- **NVIDIA NVFP4 mixed-precision pretraining (Nemotron-3, March
  2026)** — full forward+backward in NVFP4 with stochastic rounding +
  RHT. `retrainable = False`.
- **GPT-OSS MXFP4-native post-training (August 2025)** — MoE expert
  weights quantized in MXFP4 during SFT+RLHF+distillation, with the
  FP32 master weights discarded at the end. Edge case: technically
  the *pretraining* was BF16 and only post-training used MXFP4 in
  the forward pass; v3 sets `training_native=True` because the
  released checkpoint's MoE expert weights are at the MXFP4 grid by
  training-time exposure, not by post-hoc rounding. `retrainable =
  False`.

### 12.4 IR implications

The IR must carry a `training_native: bool` flag and a `retrainable:
bool` flag. The reasoning:

- A PTQ FP8 checkpoint and a `training_native=True` FP8 checkpoint
  (DeepSeek-V3) are *byte-identical* at the weights but have different
  quality characteristics on out-of-distribution prompts. The
  load-time path must skip re-quantization on the native variant; the
  IR is the right place to communicate this.
- A QAT-fine-tuned NF4 checkpoint cannot be "de-quantized" back to
  fp16 losslessly because the QAT loss converged with the quantizer in
  the loop. The IR's `calibration = qat_*` flag should warn loaders.
- BitNet ternary checkpoints cannot be ported to any other format
  meaningfully because the model architecture is different (RMSNorm
  placements, ReLU² activation). The IR must carry this in a
  `requires_architecture` field.
- Falcon-Edge ternary checkpoints *can* be retrained — and the IR's
  `retrainable: True + prequantized_revision: str` flag tells the
  loader where the BF16 retrainable form lives.
- MXFP4-native MoE expert weights coexist with BF16 attention weights
  in the same checkpoint. The IR's per-role QuantSpec map is required
  to express this; a single QuantSpec at the model level cannot.

### 12.5 Three-column summary

| Class               | Pretraining quantization aware? | Retrainable post-release? | Examples |
|---------------------|---------------------------------|---------------------------|----------|
| PTQ                 | No                              | No (in practice)          | GPTQ, AWQ, NF4, FP8 PTQ, IQ2_M, MXFP4 PTQ, NVFP4 PTQ, Q4_K_M |
| QAT                 | No (BF16 pretrain)              | No                        | QLoRA, FP8-QAT, AFM 2-bpw, Gemma 4 mobile INT4 QAT |
| Native-trained      | Yes (pretrain or post-train)    | Generally no              | BitNet b1.58, DeepSeek-V3 FP8, NVFP4-trained (Nemotron-3), GPT-OSS MXFP4-native |
| Native + retrainable| Yes                             | **Yes**                   | **Falcon-Edge 1.58-bit** |

The fourth row currently has exactly one member. v3 predicts it will
remain a small club through 2026-H2 but expects Google, Meta, or Apple
to ship the next entry by 2026-Q4.

---

## §13. Open questions / unresolved (v3 carries forward + adds)

Carried from v2:

1. **NVFP4 outer scale semantics across multi-GPU**: per-tensor fp32
   broadcast or sharded with TP?
2. **GGUF imatrix format**: per-tensor float importance; exact
   application differs between Q4_K, Q5_K, IQ2_*.
3. **HQQ + LoRA composition**: `mergeable_with_adapter` predicate
   wanted.
4. **MXFP6 SLM deployments**: hardware exposed but model count small.
5. **KV cache packing in PagedAttention**: K and V have different
   layouts.
6. **Per-row scaled softmax (FA3)**: how `s_p_i` is exposed.
7. **Codebook globality for IQ**.
8. **compressed-tensors v2** (vLLM, Q1 2026): canonical server-side
   format.
9. **Asymmetric fp formats**.
10. **Rotation matrix portability**: QuaRot fixed Hadamard vs
    SpinQuant learned matrix vs NVFP4 training random Hadamard.
11. **EXL2 per-row bitwidth ABI**.
12. **CacheGen / KV streaming**.
13. **AFP8 standardization**.
14. **BitNet checkpoint provenance**: as a train-from-scratch format,
    BitNet checkpoints are not portable to non-BitNet runtimes.
15. **TPU v6e INT8W/INT4A**: kernel signature not public.

New in v3:

16. **MXFP4-native vs MXFP4 PTQ checkpoint differentiation in HF
    metadata**: there is no standard tag. Two checkpoints in the same
    qdtype with different `training_native` flags are not
    distinguished in HuggingFace's standard `config.json` schema.
    Both `gpt-oss-20b` (native) and a hypothetical
    `awq-mxfp4-llama-3-8b` (PTQ) would look identical to a generic
    loader. The IR should propose a metadata key
    `"quantization_provenance": "native_post_training" | "ptq" | "qat" |
    "native_pretraining"`.
17. **Falcon-Edge `prequantized` revision provenance**: the
    `prequantized` BF16 revision is *not* the original BF16 checkpoint
    — it is a *scaled* BF16 such that the BitnetLinear conversion
    reproduces the ternary forward pass bit-exactly. A user
    re-downloading from `revision="bfloat16"` instead would get
    different weights and break the retrain workflow silently. The IR
    needs a `prequantized_revision_id` field that loaders consult.
18. **NVFP4 training: random Hadamard storage**: the Hadamard matrix
    is re-randomized each step and never stored. If the training
    pipeline is reproduced by a third party, exact reproducibility
    requires the random seed schedule. This is undocumented in NVIDIA
    public material as of 2026-Q2.
19. **DeepSeek-V3 per-tile scale alignment with TP/PP**: the 128×128
    weight tile size interacts with tensor parallelism shard sizes. If
    a model is sharded such that a tile crosses a shard boundary, the
    per-tile scale must be split. SGLang's DeepSeek path handles this;
    vLLM 0.7's experimental DeepSeek loader was reported broken on
    this case in March 2026.
20. **AFM 2-bpw cross-block KV sharing portability**: the
    `kv_shared_from_layer_idx` field needs an interaction model with
    PagedAttention. If a future open runtime tries to load an
    AFM-style sharing pattern, the PagedAttention block table must
    point to the same logical block for the two layer indices.
    Untested in any open runtime as of 2026-Q2.
21. **Gemma 4 mobile-INT4 QAT vs GGUF Q4_K_M parity**: the LiteRT
    format and the GGUF Q4_K_M export of the same QAT checkpoint
    differ in symmetry (LiteRT symmetric, Q4_K_M asymmetric). The
    rounding error after conversion is small but non-zero; the IR
    should flag this as a `lossy_export` between target_runtimes.

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
- **GPT-OSS report — OpenAI, 2025. arXiv:2508.10925.**
- **Apple Foundation Models 2025 Tech Report — Apple, 2025. arXiv:2507.13575.**
- **NVFP4 training — NVIDIA, 2025-Q4 / 2026-Q1. arXiv:2509.25149.**
- **DeepSeek-V3 Technical Report — DeepSeek-AI, 2024. arXiv:2412.19437.**
- **DeepSeek-V3.2 — DeepSeek-AI, 2025. arXiv:2512.02556.**

### Specs and source code

- OCP, *Microscaling Formats Specification v1.0*, Sept 2023, §4.2.
- NVIDIA Blackwell Architecture Whitepaper, 2024 §3.4 (NVFP4).
- NVIDIA *"NVFP4 trains with the precision of 16-bit and the speed and
  efficiency of 4-bit"* developer blog, Q1 2026.
- NVIDIA Transformer Engine docs.
- TensorRT-LLM quantization docs.
- TensorRT Model Optimizer
  (`github.com/NVIDIA/TensorRT-Model-Optimizer`).
- vLLM quantization (`docs.vllm.ai/en/latest/features/quantization`).
- Neural Magic compressed-tensors
  (`github.com/neuralmagic/compressed-tensors`).
- llama.cpp `ggml/src/ggml-quants.c`, `src/llama-quant.cpp`
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
- bitnet.cpp PR #268 (Falcon-Edge support).
- T-MAC (`github.com/microsoft/T-MAC`).
- KleidiAI (`github.com/ARM-software/kleidiai`).
- MLX quantization
  (`github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py`).
- MLX-LM Falcon-Edge integration.
- ONNX Runtime QDQ / QOperator docs.
- OpenVINO NNCF (`github.com/openvinotoolkit/nncf`).
- Intel Neural Compressor (`github.com/intel/neural-compressor`).
- AMD Quark (`github.com/AMD/Quark`).
- torchao (`github.com/pytorch/ao`).
- **OpenAI gpt-oss** (`github.com/openai/gpt-oss`,
  `huggingface.co/openai/gpt-oss-20b`,
  `huggingface.co/openai/gpt-oss-120b`).
- **TII Falcon-Edge** (`huggingface.co/tiiuae/Falcon-E-1B-Base`,
  `huggingface.co/tiiuae/Falcon-E-3B-Base`,
  `huggingface.co/blog/tiiuae/falcon-edge`).
- **onebitllms** (the PyPI / GitHub package for Falcon-Edge retrain
  workflow).
- **Google Gemma 4** (`huggingface.co/blog/gemma4`, Google AI Edge
  docs, LiteRT 1.1 release notes).
- **Google LiteRT / MediaPipe** quantization docs.
- **DeepSeek-V3 / V3.2 official** (`github.com/deepseek-ai/DeepSeek-V3`,
  `github.com/deepseek-ai/DeepSeek-V3.2-Exp`).
- **Apple Foundation Models 2025 Updates** blog
  (`machinelearning.apple.com/research/apple-foundation-models-2025-updates`).
- **NVIDIA Nemotron 3 Super / Ultra**
  (`developer.nvidia.com/blog/introducing-nemotron-3-super`).

---

*End of v3 survey. Scheme count: **48 distinct schemes** (including
sub-variants) across 7 families (v2 had 42). Minimum-viable IR axes: 7;
extended-coverage axes: 14 (7 minimal + 14 extended = 21 total; v2 had
19). New schemes vs v2: (a) MXFP4-native post-training (GPT-OSS,
§3.22); (b) NVFP4 mixed-precision training (Nemotron-3, §3.24); (c)
DeepSeek-V3 native FP8 pretraining (§3.28, promoted from v2 note);
(d) Apple AFM 2-bpw QAT (§9.4, promoted from v2 closed-weights note);
(e) Falcon-Edge 1.58-bit retrainable (§9.2); (f) Gemma 4 mobile INT4
QAT (§9.5). New axes: `training_native: bool`, `retrainable: bool`.
"Support 6" recommendation replaces v2's "Support 5"; the new sixth
slot is Gemma 4 mobile-INT4 QAT. Evolution narrative: seven eras
2022 H2 → 2026 H1, with Era 6 substantially rewritten and Era 7
added.*
