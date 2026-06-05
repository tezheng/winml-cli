# Quantization Schemes in Shipped SLM Runtimes

> Survey for the `llm-layers` IR project. Goal: enumerate every quantization
> scheme actually shipped in mainstream small-language-model runtimes
> (llama.cpp, vLLM, TensorRT-LLM, OpenVINO, MLX, Hugging Face Transformers,
> AutoGPTQ/AutoAWQ, bitsandbytes, ONNX Runtime, Intel Neural Compressor) and
> distill the minimum parameter axes a representation must expose to cover
> them.

---

## 0. Reading the survey

For each scheme we record nine fields:

1. **Bitwidth + numeric type** of stored weights/activations.
2. **Grouping** — per-tensor, per-channel (output / input axis), per-token,
   blockwise (with N).
3. **Scale dtype** — fp16, bf16, fp32, e8m0 (UE8M0 power-of-two), or implicit.
4. **Zero-point** — symmetric (none) or asymmetric (offset present).
5. **Calibration** — none / weight-only, PTQ with calibration set, GPTQ-style
   Hessian, QAT.
6. **Storage layout** — packed nibble order, interleave, super-block layout.
7. **Runtimes** — actual shipped support today.
8. **Matmul kernel signature** — `(W_q, A) -> Y` with dtypes and accumulator.
9. **Parameter axes the IR must expose** to encode this scheme exactly.

Citations are inline. Anchor references appear in section 10.

---

## 1. Weight-only INT/FP quantization

These schemes compress the linear-layer weights and dequantize (or fused-mul)
at inference; activations remain fp16/bf16. The dominant deployment family
for ≤13B SLMs.

### 1.1 AWQ — Activation-aware Weight Quantization

- **Paper**: Lin et al., *AWQ: Activation-aware Weight Quantization for
  LLM Compression and Acceleration*, MLSys 2024
  (arXiv:2306.00978).
- **Bitwidth/dtype**: weights INT4, asymmetric (signed or unsigned int4).
- **Grouping**: blockwise along the **input** axis (K dimension) with
  `group_size ∈ {32, 64, 128}`; 128 is the canonical default.
- **Scale dtype**: fp16 (one fp16 scale + one fp16 zero per group).
- **Zero-point**: yes, fp16-valued zero (stored as fp16, not int).
- **Calibration**: PTQ. Requires a small calibration set (typically ~128–512
  samples from Pile / wikitext / domain text) to compute per-channel
  **activation magnitudes**; the algorithm then derives a per-channel scaling
  factor `s` that is **folded into the previous layer** so that important
  weight channels are upscaled before quantization. Final weights are RTN
  quantized in groups.
- **Storage layout**: weights packed 8 × int4 per int32 word with a specific
  permutation used by the AutoAWQ GEMM kernel (`awq_gemm`) — the int4
  values are interleaved `[0,2,4,6,1,3,5,7]` so two fp16 lanes of `mma.sync`
  can dequantize 8 values via a single `lop3.b32` PRMT trick (the
  "fast int4 → fp16 dequant" idiom from FasterTransformer).
- **Runtimes**: AutoAWQ (reference), vLLM (`awq`, `awq_marlin`), TensorRT-LLM
  (`W4A16_AWQ` plugin), HuggingFace Transformers via `awq` integration,
  MLX (`mlx.nn.quantize` supports 4-bit groups), llama.cpp has indirect
  support via GGUF conversion (re-quantized to Q4_K).
- **Kernel signature**:
  `awq_gemm(W: int4[N, K/8] packed, scales: fp16[N, K/G], zeros: fp16[N, K/G], A: fp16[M, K]) -> Y: fp16[M, N]`
  with **fp32 accumulator** inside `mma.sync.f16.f16.f16` warp tiles (fp16
  accum in some cuda paths).
- **IR axes touched**: `qdtype=int4`, `group_size=128`, `quant_axis=K`,
  `scale_dtype=fp16`, `has_zero_point=True`, `zero_dtype=fp16`,
  `packing=interleaved8x4`, `accumulator_dtype=fp32`,
  `calibration=activation_aware`.

### 1.2 GPTQ — Optimal Brain Quantization with Hessians

- **Paper**: Frantar et al., *GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers*, ICLR 2023 (arXiv:2210.17323).
- **Bitwidth/dtype**: weights INT4 (most common), INT3, INT2, or INT8.
- **Grouping**: blockwise along K with `group_size ∈ {-1 (per-channel), 32,
  64, 128, 1024}`; 128 is default. Optional `desc_act=True` reorders columns
  by Hessian diagonal magnitude so larger-magnitude activations align with
  earlier columns — this changes the storage layout (a `g_idx` mapping is
  stored).
- **Scale dtype**: fp16 (scales) + int4 zero-point packed alongside.
- **Zero-point**: yes, **integer** zero packed in int4 (asymmetric).
- **Calibration**: PTQ. Uses Hessian `H = 2 X Xᵀ / n` from ~128 calibration
  samples; weights are quantized column-by-column with closed-form OBS
  updates `δ = -(w_q − w) / H⁻¹_ii · H⁻¹_:,i` to compensate residual error.
- **Storage layout**: AutoGPTQ packs `W` as int32 with `pack_factor = 32/bits`
  values per word; `qzeros` similarly packed; `scales` fp16 row-major.
  `g_idx` is an int32 vector of size K mapping column → group index when
  `desc_act` is on.
- **Runtimes**: AutoGPTQ, GPTQModel (fork), vLLM (`gptq`, `gptq_marlin`),
  TensorRT-LLM (`W4A16_GPTQ`), HF Transformers + Optimum, ExLlamaV2 (custom
  Q-matrix layout), llama.cpp historically via `Q4_0` re-quantization,
  OpenVINO via NNCF.
- **Kernel signature**: identical to AWQ's `W4A16` GEMM but with **int**
  zero-point and optional g_idx gather.
  `gptq_gemm(W: int4 packed[K/8, N], scales: fp16[K/G, N], qzeros: int4 packed[K/G, N/8], g_idx: int32[K]?, A: fp16[M, K]) -> Y: fp16[M, N]` with fp32 accumulator.
- **IR axes**: `qdtype=int4|int3|int2|int8`, `group_size`, `quant_axis=K`,
  `scale_dtype=fp16`, `has_zero_point=True`, `zero_dtype=int`,
  `packing=int32_packed`, `act_order=bool`, `accumulator_dtype=fp32`,
  `calibration=hessian`.

### 1.3 HQQ — Half-Quadratic Quantization

- **Source**: Badri & Shaji, mobius-labs blog (Nov 2023), `mobiusml/hqq` repo.
- **Bitwidth/dtype**: INT8, INT4, INT3, INT2, INT1; asymmetric.
- **Grouping**: blockwise along K with `group_size ∈ {8, 32, 64, 128, 256}`;
  64 is the typical default.
- **Scale dtype**: fp16 or bf16; zeros fp16.
- **Zero-point**: yes (fp16-valued).
- **Calibration**: **None**. HQQ solves a sparsity-promoting optimization
  `min ‖W − Q⁻¹(Q(W; s, z))‖_p` with `p < 1` via half-quadratic splitting,
  using only the weights — no calibration data. This is its headline feature
  (fast to quantize: a 70B model in minutes).
- **Storage layout**: weights packed 2/4/8 per byte; meta stored as fp16
  scales+zeros per group. Supports an optional second-level quantization
  of the scales themselves (double-quant) borrowing the bnb idea.
- **Runtimes**: HQQ library (`hqq.core.quantize`), HF Transformers
  (`hqq` integration), vLLM experimental, MLX via `mlx-hqq`, Apple's
  `mlx.nn.quantize` shares the philosophy.
- **Kernel signature**: same `W4A16` shape as AWQ/GPTQ; HQQ uses Marlin or
  triton kernels for compute; on CPU uses torch dequant + matmul.
- **IR axes**: same as GPTQ minus `act_order`, plus
  `calibration=data_free` and `optimizer=half_quadratic` if the IR is to
  *describe how* to produce the quantization (vs only execute it).

### 1.4 bitsandbytes — NF4 / FP4 / LLM.int8()

- **Papers**: Dettmers et al., *QLoRA* (arXiv:2305.14314) introduces
  **NF4**; *LLM.int8()* (arXiv:2208.07339) introduces vector-wise int8 with
  outlier decomposition.
- **NF4 (NormalFloat-4)**:
  - 4-bit codebook values sampled at the quantiles of `N(0,1)`. Codebook is
    fixed: `{-1.0, -0.6962, -0.5251, ..., 0.0, ..., 0.7229, 1.0}` (16 levels,
    asymmetric around zero but with a level at 0).
  - **Group size = 64** (default) along the flattened weight axis.
  - **Scale dtype**: fp32 absmax per group; *double-quant* re-quantizes
    those 64-element scales themselves as 8-bit with another fp32 scale per
    256 group-scales.
  - **Zero-point**: implicit (codebook is asymmetric, no explicit zero
    point).
  - **Calibration**: none, data-free per-block absmax.
  - **Storage**: packed nibbles row-major; the per-group fp32 absmax stored
    separately; with `bnb_4bit_use_double_quant=True`, a second-level
    int8 codebook compresses the absmax array.
  - **Runtimes**: bitsandbytes (CUDA), HF Transformers (the canonical
    QLoRA path), vLLM (`bitsandbytes` quant method), TGI, peft.
  - **Kernel**: `bnb_matmul_4bit(W_q: u4[N, K/2], absmax: fp32[N*K/64], A: fp16[M, K]) -> Y: fp16/bf16[M, N]`; dequant is a 16-entry LUT lookup; accumulator fp32.
- **FP4**:
  - Same group/scale layout as NF4 but the 16 codepoints are a non-IEEE
    fp4 format `E2M1` (with values
    `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}` and signed variants).
  - Slightly worse perplexity than NF4 in practice; rarely used in
    production.
- **LLM.int8() (W8A8 mixed)**:
  - INT8 weights + INT8 activations with **per-row** scaling.
  - Outlier feature columns (~0.1% of channels with magnitudes > 6) are
    decomposed and computed in fp16; remaining columns done in int8.
  - **Calibration**: implicit, runtime decomposition driven by a threshold.
  - **Runtimes**: bitsandbytes only.
- **IR axes for NF4**: `qdtype=nf4`, `codebook=nf4_quantile`, `group_size=64`,
  `quant_axis=flattened`, `scale_dtype=fp32`, `has_zero_point=False`,
  `double_quant=bool`, `packing=nibble`.

### 1.5 MX formats — MXFP4 / MXFP6 / MXFP8 (OCP Microscaling)

- **Spec**: Open Compute Project *Microscaling Formats v1.0* (Sept 2023);
  Rouhani et al. *Microscaling Data Formats for Deep Learning*
  (arXiv:2310.10537).
- **Common structure**: a **block of 32 elements** shares a single 8-bit
  **shared exponent in UE8M0 format** (unsigned, 8-bit, magnitude-only —
  a power-of-two scale). Element types differ per variant:
  - **MXFP8**: elements are IEEE-like FP8, either `E4M3` or `E5M2`.
  - **MXFP6**: elements are FP6 `E3M2` or `E2M3`.
  - **MXFP4**: elements are FP4 `E2M1`.
  - (Also MXINT8: int8 elements with the same e8m0 shared exponent.)
- **Bitwidth/dtype**: as above; effective bitwidth is
  `element_bits + 8/32 = element_bits + 0.25`.
- **Grouping**: hard-coded **block_size = 32 along the contraction axis**
  (K). Not negotiable in the OCP spec.
- **Scale dtype**: **UE8M0** (8-bit unsigned exponent, value = 2^(e-127), the
  same biased convention as float32 exponent).
- **Zero-point**: none; symmetric.
- **Calibration**: data-free per-block absmax rounding for weights; for
  activations the shared exponent is computed on the fly per 32 elements.
- **Storage**: 32 element bytes (packed: MXFP4 = 16 B, MXFP6 = 24 B,
  MXFP8 = 32 B) + 1 byte scale per block.
- **Runtimes**: NVIDIA Hopper (B200/H200) hardware **natively executes
  MXFP8** in tensor cores; TensorRT-LLM 0.13+ exposes MXFP8 GEMM;
  vLLM 0.6+ exposes `mxfp4` for Mixtral/GPT-OSS weight loading; AMD MI355
  supports MX in MFMA; Intel Gaudi-3 supports MX; PyTorch
  `torchao` has `mx_format` recipes. GPT-OSS (OpenAI's 2024 open-weight
  models) ships weights in **MXFP4**.
- **Kernel signature**:
  `mx_gemm(W: mxfp4[N, K] = (W_e2m1[N, K], W_scale_ue8m0[N, K/32]), A: mxfp4[M, K] = (A_e2m1[M, K], A_scale_ue8m0[M, K/32])) -> Y: bf16/fp32[M, N]` with **fp32 accumulator**. The HW automatically multiplies element pairs and scales by `2^(W_e + A_e − 2·127)`.
- **IR axes**: `qdtype=mxfp4|mxfp6|mxfp8|mxint8`, `group_size=32` (fixed),
  `quant_axis=K`, `scale_dtype=ue8m0`, `has_zero_point=False`,
  `element_format=e2m1|e3m2|e2m3|e4m3|e5m2`, `packing=ocp_mx`.

### 1.6 NVFP4 — NVIDIA Blackwell native FP4

- **Source**: NVIDIA Blackwell architecture whitepaper (2024); TRT-LLM
  release notes 0.14+.
- **Bitwidth/dtype**: 4-bit elements in `E2M1` (same as MXFP4) BUT with a
  **two-level scale**:
  1. **Inner block scale**: per-block-of-16 elements, dtype **fp8 E4M3**
     (not UE8M0!).
  2. **Outer per-tensor scale**: a single fp32 scale per tensor that scales
     the fp8 inner scales.
- **Grouping**: block_size = **16** (not 32), along K.
- **Scale dtype**: **fp8 E4M3** inner + fp32 outer.
- **Zero-point**: none.
- **Calibration**: PTQ; TensorRT-Model-Optimizer computes the
  per-tensor amax for the outer scale from a calibration set; inner E4M3
  scales are per-block absmax. AWQ-style activation scaling is optional.
- **Storage**: 16 elements packed in 8 bytes + 1 fp8 byte per block; one
  fp32 per tensor.
- **Runtimes**: TensorRT-LLM (`nvfp4` quant_algo), TensorRT 10.5+, NIM
  for Blackwell. SGLang and vLLM (0.6.4+) load NVFP4 checkpoints from
  TensorRT-Model-Optimizer.
- **Kernel signature**: 5th-gen tensor-core `mma.sync` with fp4 operands and
  fp8 scales, bf16 accumulator (selectable to fp32).
- **IR axes**: `qdtype=nvfp4`, `group_size=16`, `quant_axis=K`,
  `scale_dtype=fp8_e4m3`, `outer_scale_dtype=fp32`,
  `has_zero_point=False`, `packing=nvfp4`, `element_format=e2m1`.

> Key axis surfaced by MX/NVFP4: **scales can themselves be quantized**, and
> the scale dtype is not always fp16/fp32 — UE8M0 and fp8 are first-class.
> The IR therefore needs a `scale_dtype` enum richer than {fp16, bf16, fp32}
> and a recursive `scale_quant` field (a scale of scales).

---

## 2. Weight + Activation (W*A*) quantization

These schemes quantize **both** sides of the matmul. The compute happens
in low-precision tensor cores, not just memory bandwidth saved.

### 2.1 SmoothQuant — W8A8 with channel smoothing

- **Paper**: Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training
  Quantization for LLMs*, ICML 2023 (arXiv:2211.10438).
- **Bitwidth/dtype**: W8 INT8, A8 INT8.
- **Grouping**: W per-channel (output axis N), A per-token
  (dynamic) or per-tensor (static).
- **Scale dtype**: fp32 (stored) / fp16 at runtime.
- **Zero-point**: usually symmetric (zero=0) for both.
- **Calibration**: PTQ. Compute per-channel activation magnitudes
  `s_j = max|X_:,j|^α / max|W_j,:|^(1-α)` with `α ∈ [0.5, 0.8]`,
  divide A and multiply W to migrate the dynamic range from activations
  (hard to quantize) to weights (easy). Then quantize both to int8.
- **Storage**: int8 weights row-major plus per-channel fp16 scale vector.
- **Runtimes**: TRT-LLM (`smoothquant` preprocessing + W8A8 GEMM),
  OpenVINO (`smoothquant_alpha` in NNCF), Intel Neural Compressor,
  Qualcomm AIMET, vLLM via `compressed-tensors` (the unified format from
  Neural Magic).
- **Kernel signature**:
  `int8_gemm(W: int8[N, K], W_scale: fp16[N], A: int8[M, K], A_scale: fp16[M] (per-token) or fp16 (per-tensor)) -> Y: fp16/bf16[M, N]` with **int32 accumulator**, then dequant to fp16.
- **IR axes**: `qdtype=int8`, `w_quant_axis=N`, `a_quant_granularity=per_token|per_tensor`, `scale_dtype=fp16`, `has_zero_point=False`, `accumulator_dtype=int32`, `pre_scale=channelwise_smooth`.

### 2.2 INT8 W8A8 — vanilla dynamic / static

- **Dynamic (per-token)**: activations quantized on the fly with per-token
  absmax. No calibration set needed.
- **Static (per-tensor)**: activation amax computed from calibration set
  and frozen. Required for kernels that want to fold scales into bias.
- **Storage / kernel**: same shapes as SmoothQuant.
- **Runtimes**: TRT-LLM `W8A8`, vLLM `compressed-tensors`, OpenVINO,
  ONNX Runtime QLinearMatMul, TFLite full-int8, executorch, MLX.
- **IR axes**: same as SmoothQuant minus the pre-scale.

### 2.3 INT4 W4A4 — rare in shipped runtimes

- Used in **QuaRot** (arXiv:2404.00456) and **SpinQuant**
  (arXiv:2405.16406), which use **Hadamard rotations** to eliminate
  activation outliers before W4A4 quantization. Per-token A4 scales fp16.
- **Runtimes**: TRT-LLM has experimental W4A8 (`int4` weight + `int8`
  activation); pure W4A4 not in mainstream production yet. Some MLX recipes
  for Apple Silicon expose 4-bit activations.
- **IR axes**: `qdtype=int4` on both, `pre_rotation=hadamard`,
  `a_quant_granularity=per_token`.

### 2.4 FP8 — E4M3 / E5M2

- **Spec**: Micikevicius et al., *FP8 Formats for Deep Learning*
  (arXiv:2209.05433); NVIDIA Transformer Engine.
- **Bitwidth/dtype**: 8-bit float. **E4M3** has 4 exponent / 3 mantissa
  bits (range ±448), **E5M2** has 5 exponent / 2 mantissa (range ±57344,
  IEEE-like with inf/NaN). Convention: **E4M3 for forward (weights and
  activations), E5M2 for gradients in training**.
- **Grouping**: per-tensor scale (NVIDIA TE default), or per-row
  (vLLM/TRT-LLM "fp8_e4m3 per_channel"), or per-token for activations.
- **Scale dtype**: fp32 (amax-history-based delayed scaling in TE) or fp16.
- **Zero-point**: none; both formats include signed zero already.
- **Calibration**: **delayed scaling** — TE tracks amax over an EMA window
  and updates the scale lazily; vLLM uses static per-tensor amax from a
  calibration pass (`fp8 dynamic` exists for activations).
- **Runtimes**: NVIDIA Hopper (H100/H200) hardware FP8 in tensor cores,
  Blackwell adds MXFP8/NVFP4; AMD MI300 supports FP8; Intel Gaudi-2/3
  supports FP8 (slightly different exponent biases); TRT-LLM `fp8` quant
  algo; vLLM `fp8` (per-tensor or per-channel); SGLang `fp8`; CUTLASS,
  cuBLAS-Lt; MLX has fp8 on M-series (limited).
- **Kernel signature**:
  `fp8_gemm(W: e4m3[N, K], W_scale: fp32, A: e4m3[M, K], A_scale: fp32[M] or fp32) -> Y: bf16/fp16[M, N]` with **fp32 accumulator**.
- **IR axes**: `qdtype=fp8_e4m3|fp8_e5m2`, `w_quant_granularity=per_tensor|per_channel`, `a_quant_granularity=per_tensor|per_token|delayed_per_tensor`, `scale_dtype=fp32`, `has_zero_point=False`, `accumulator_dtype=fp32`.

### 2.5 W4A8 — INT4 weights + INT8 activations

- TRT-LLM `W4A8_AWQ` and `W4A8_QSERVE`, ExLlamaV3 partial. Combines AWQ-style
  weight quantization with int8 activation quantization. Dot-product accum
  is int32. Practical on Hopper because the int8 tensor cores are fast.

---

## 3. GGUF k-quants and i-quants

GGUF is llama.cpp's storage format. K-quants (2023) and i-quants
(IQ family, 2024) are the *de facto* deployment quantization for CPU and
Apple-Silicon SLM inference. Source of truth: `ggml-quants.h` /
`ggml-quants.c` in ggerganov/llama.cpp.

### 3.1 Super-block structure (k-quants)

A **super-block** is 256 weights. It is subdivided into 16 **sub-blocks** of
16 weights each. The super-block carries **two top-level scales**
(`d` and `dmin`, both fp16), and each sub-block has its own 6-bit `scale`
and 6-bit `min` (for asymmetric variants), encoded so the dequant of an
element `w_q` in sub-block `j` is:

```
w = (d * scale_j) * q − (dmin * min_j)
```

This compresses 16 fp16 scales into ~12 bytes per super-block.

#### Variants

| Variant   | Bits/weight (effective) | Notes |
|-----------|-------------------------|-------|
| `Q2_K`    | 2.5625                  | 2-bit weights, super-block of 256, 4-bit per-sub-block scale + min |
| `Q3_K`    | 3.4375                  | 3-bit weights, sub-block scale 6-bit (packed in 12 bytes) |
| `Q4_K`    | 4.5                     | 4-bit weights, 6-bit sub-block scale + 6-bit sub-block min |
| `Q5_K`    | 5.5                     | 5-bit weights (4-bit + 1 high-bit plane) |
| `Q6_K`    | 6.5625                  | 6-bit weights, 8-bit sub-block scale |
| `Q8_K`    | 8.5                     | int8 weights with sub-block fp32 scales (intermediate, not for storage) |

#### `_M` / `_S` suffixes (mixed-bit)

`Q4_K_M` vs `Q4_K_S`: the **M ("medium")** variant uses a higher-bit
k-quant (e.g. Q6_K) for the `attn.v` and `feed_forward.down` projections —
empirically the most quantization-sensitive layers — and Q4_K for the rest.
`S ("small")` uses Q4_K uniformly. The IR therefore must support
**per-tensor (per-layer) quantization recipe override**.

- **Scale dtype**: `d`, `dmin` are fp16. Sub-block scales/mins are
  6-bit unsigned integers, decoded against `d/dmin`.
- **Zero-point**: implicit through `dmin * min` term — asymmetric.
- **Calibration**: weight-only RTN per super-block; **importance
  weighting** since ~2024: a per-channel "importance matrix"
  (`--imatrix`) is computed from a calibration corpus and weights are
  multiplied by `sqrt(importance)` before RTN — this is the
  GPTQ-without-the-Hessian shortcut llama.cpp ships.
- **Storage**: contiguous super-blocks, row-major over (output_channel,
  input_block_index). Highly cache-friendly for AVX-512 / NEON.
- **Runtimes**: llama.cpp / ggml (CPU + CUDA + Metal + Vulkan + SYCL +
  Kompute backends), LM Studio, Ollama, Jan, GPT4All, llamafile,
  candle (Rust), MLX via gguf import. **The dominant on-device SLM
  quantization format by deployment count.**
- **Kernel signature**:
  `ggml_vec_dot_q4_K_q8_K(W: Q4_K[K_blocks], A: Q8_K[K_blocks]) -> fp32` —
  activations are dynamically quantized to **Q8_K** per super-block; the
  dot product accumulates in int32 then converts via the fp16 scales.
- **IR axes**: `qdtype=q4_k|q5_k|q6_k|q2_k|q3_k`, `super_block=256`,
  `sub_block=16`, `scale_dtype=fp16`, `sub_scale_bits=6`,
  `has_zero_point=True`, `zero_storage=sub_block_min`,
  `packing=gguf_kquant`, `per_layer_override=dict`,
  `calibration=imatrix|none`.

### 3.2 I-quants (IQ family) — importance-aware codebooks

- **Origin**: Kawrakow et al., llama.cpp PRs #4773 and #5104 (Jan 2024),
  inspired by QuIP# (Tseng et al., *QuIP#: Even Better LLM Quantization with
  Hadamard Incoherence and Lattice Codebooks*, arXiv:2402.04396).
- **Idea**: instead of mapping weights to a uniform grid, map them to a
  pretrained **vector codebook** (lattice) optimized for Gaussian-distributed
  weights. Multiple weights share an index into the codebook.
- **Bitwidth/dtype**: effective 2–4 bits/weight.

| Variant   | Bits/weight | Codebook shape |
|-----------|-------------|----------------|
| `IQ1_S`   | 1.5625      | 8 weights → 1 byte codebook index + super-block scale |
| `IQ1_M`   | 1.75        | as IQ1_S with finer subblock signs |
| `IQ2_XXS` | 2.0625      | E8 lattice, 8 weights → 16-bit index |
| `IQ2_XS`  | 2.3125      | E8 lattice, 8 weights → larger codebook |
| `IQ2_S`   | 2.5         | augmented IQ2 with sign bits |
| `IQ2_M`   | 2.7         | IQ2_S with bigger codebook |
| `IQ3_XXS` | 3.0625      | E8 lattice |
| `IQ3_S`   | 3.4375      | sign-augmented |
| `IQ3_M`   | 3.66        | layer-mixed |
| `IQ4_XS`  | 4.25        | non-uniform 16-entry LUT per super-block |
| `IQ4_NL`  | 4.5         | "non-linear int4" — 16-entry LUT shared globally |

- **Scale dtype**: fp16 per super-block (same super-block-of-256 layout).
- **Zero-point**: none in IQ (codebook handles the asymmetry implicitly).
- **Calibration**: requires `imatrix` (importance matrix from activations)
  for the lattice selection to be effective; without it IQ2/IQ3 perplexity
  degrades sharply.
- **Storage**: codebook stored once globally; per-super-block: scale +
  packed codebook indices + per-group sign bits.
- **Runtimes**: llama.cpp (all backends), Ollama, LM Studio — these IQ
  formats are the reason 7B SLMs can run on 4 GB phones.
- **Kernel signature**: `ggml_vec_dot_iqN_xxs_q8_K`: a LUT-gather inside
  the dot product. Codebook resides in constant memory on GPUs.
- **IR axes**: `qdtype=iq{1,2,3,4}_{xxs,xs,s,m,nl}`, `codebook=e8_lattice|nl_lut`, `super_block=256`, `scale_dtype=fp16`, `has_zero_point=False`, `requires_imatrix=True`, `packing=iquant_indexed`.

---

## 4. KV-cache quantization

The KV cache often exceeds model weights in long contexts. Runtime engines
quantize it independently from weights, with different trade-offs because
KV is **populated at inference time**, not preprocessed.

### 4.1 INT8 KV cache (vLLM, llama.cpp)

- **Grouping**: per-head per-channel; vLLM `kv_cache_dtype=int8` uses
  per-head fp16 scale; llama.cpp uses per-token absmax via `Q8_0`
  (one fp16 scale per 32 elements, no zero).
- **Scale dtype**: fp16.
- **Zero-point**: symmetric in vLLM/llama.cpp.
- **Storage**: PagedAttention block layout (vLLM) — each block holds
  `block_size × num_heads × head_dim` int8s plus per-head scales.
- **Calibration**: none for symmetric/dynamic; static per-channel optional
  (vLLM `kv-scales.json`).
- **Kernel touches**: dequant inline in the attention kernel (FlashAttention
  fork) — the QK and SV matmuls accept int8 K/V with fp16 scales.
- **IR axes**: `kv_qdtype`, `kv_quant_granularity=per_head|per_channel|per_token`, `kv_scale_dtype`, `kv_has_zero_point`.

### 4.2 INT4 KV cache

- **KIVI** (Liu et al., arXiv:2402.02750) introduced **asymmetric per-channel
  K + per-token V** int4 with fp16 scales, residual buffer for outliers.
- **Runtimes**: vLLM `kv_cache_dtype=int4` (experimental, KIVI-style),
  TRT-LLM `int4_kv_cache`. Real-world adoption growing for 128 k-context.
- **IR axes**: `kv_qdtype=int4`, `k_quant_axis=per_channel`, `v_quant_axis=per_token`, plus optional `residual_buffer=bool`.

### 4.3 FP8 KV cache (TRT-LLM, vLLM)

- E4M3 with per-tensor or per-head fp32 scale.
- **Runtimes**: TRT-LLM `fp8_kv_cache`, vLLM `kv_cache_dtype=fp8_e4m3` and
  `fp8_e5m2`, SGLang.
- **No calibration** for E5M2 (range is huge), static amax for E4M3.
- **IR axes**: `kv_qdtype=fp8_e4m3|fp8_e5m2`, `kv_quant_granularity`,
  `kv_scale_dtype=fp32`.

### 4.4 K vs V asymmetry

K is much more "outlier-prone" than V (especially in RoPE'd K). Real
engines treat them with different schemes — e.g., K = INT4 per-channel,
V = INT4 per-token (KIVI); or K = fp8, V = int8. The IR therefore needs
**separate quantization configs for K and V**, not a single `kv_cache`
spec.

---

## 5. Attention-internal quantization specifics

Beyond KV storage, FlashAttention-3 and TRT-LLM expose intermediate
quantization *inside* the kernel.

### 5.1 FP8 SDPA (Scaled Dot-Product Attention)

- FlashAttention-3 (Shah et al., arXiv:2407.08608) computes
  `Q · Kᵀ` in fp8 E4M3, then softmax in fp32, then `P · V` in fp8 again,
  with per-row scales tracked online to prevent softmax overflow.
- **Per-row scaled softmax**: maintain `m_i = max(QK_i,:)` and rescale the
  fp8 P matrix per row so the largest value maps to 448. This is essentially
  *online per-token activation quantization* inside attention.
- **Runtimes**: TRT-LLM `fp8 attention`, FlashInfer's fp8 SDPA, SGLang.
- **IR axes**: `attn_qk_qdtype`, `attn_pv_qdtype`,
  `attn_softmax_dtype=fp32` (almost always), `online_rescale=True`.

### 5.2 INT8 quantized attention

- TurboMind (LMDeploy) uses W8A8 INT8 attention. Saturating int8 needs
  per-row scales pre-softmax and post-softmax. Niche today; FP8 has eaten
  the mindshare on Hopper+.

### 5.3 FP8 sub-format zoo

It is worth pinning down because the IR must distinguish them:

- **NVIDIA E4M3** (range ±448): no infinity encoding (the all-ones-exponent
  pattern is repurposed for extra finite values); used by TE / TRT-LLM /
  CUTLASS.
- **NVIDIA E5M2** (range ±57344): IEEE-754-style with inf/NaN; used for
  gradients.
- **AMD MI300 OCP-FP8** (`fp8_e4m3fnuz`, `fp8_e5m2fnuz`): different bias
  and **no negative zero** ("fnuz" = finite, no unsigned zero). PyTorch
  exposes both `torch.float8_e4m3fn` (NVIDIA) and `torch.float8_e4m3fnuz`
  (AMD). The bit pattern of a given fp16 value differs by a few units
  between the two encodings. **vLLM / TRT-LLM transparently re-bias on
  AMD**, but a model checkpoint saved as E4M3 on NVIDIA is not
  byte-identical on AMD.
- **Intel Gaudi-2/3** historically used a custom FP8 with configurable
  bias; Gaudi-3 adopts OCP-compliant E4M3/E5M2.

The IR's `qdtype` enum must therefore distinguish `fp8_e4m3fn` (NVIDIA),
`fp8_e4m3fnuz` (AMD), and `fp8_e5m2` / `fp8_e5m2fnuz`. Four FP8 codes,
not two.

### 5.4 Worked example: a Llama-3-8B Q4_K_M tensor

To make the parameter axes concrete, here is how a single
`blk.0.attn_v.weight` tensor from a Q4_K_M GGUF would populate the IR:

```python
QuantSpec(
  qdtype           = "q6_k",          # _M upgrade for attn_v
  element_signed   = True,
  quant_axis       = -1,              # last axis = K
  group_size       = 16,              # sub-block
  block_layout     = "gguf_super_block_256",
  scale_dtype      = "int6",          # per-sub-block 6-bit scale
  scale_axis       = "per_sub_block",
  scale_quant      = QuantSpec(       # the super-scale of the scales
       qdtype     = "fp16",
       quant_axis = "per_super_block",
       scale_dtype= "fp16",
       has_zero_point=False,
       packing    = "unpacked",
  ),
  has_zero_point   = True,
  zero_dtype       = "int6",
  zero_axis        = "per_sub_block",
  codebook         = None,
  packing          = "gguf_kquant",
  storage_order    = "blocked",
  accumulator_dtype= "int32",          # int8 ⊗ int8 dot product
  compute_dtype    = "int8",           # activations Q8_K-converted
  dequant_fused    = True,
  calibration      = "imatrix",
  pre_transform    = None,
  role             = "weight",
)
```

The same model's `blk.0.attn_q.weight` would have `qdtype="q4_k"` and the
identical surrounding structure — confirming that the IR's "Q4_K_M
recipe" is just a **dict from tensor-name pattern → QuantSpec override**,
not a new format.

---

## 6. Where schemes overlap, where they are orthogonal

### Overlap clusters

- **AWQ ≡ GPTQ ≡ HQQ (storage-wise)**: all three produce a `W4A16` tensor
  with grouped fp16 scales (± zero) and INT4 packed nibbles. They differ
  only in **how the quantized values were chosen** (activation-aware
  rescale vs Hessian-OBS vs half-quadratic). A unified IR can represent
  the *runtime artifact* identically and surface the calibration algorithm
  as **metadata**, not a different layout.
- **MXFP4 ⊂ NVFP4 generalization**: both are E2M1 + block scale. Differ
  in block size (32 vs 16) and scale dtype (UE8M0 vs FP8 E4M3) and outer
  scale. A `block_quant_fp` op parameterized by `(elem_fmt, block_size,
  scale_dtype, outer_scale_dtype?)` covers both.
- **GGUF k-quants ≡ generic "super-block + sub-block" hierarchical scale**:
  Q4_K, Q5_K, Q6_K are the same algorithm with different bitwidths. The
  super-block layout is a recurring pattern — the IR should express it
  once.
- **FP8 E4M3 vs E5M2**: same op, different format flag.
- **SmoothQuant vs vanilla W8A8**: SmoothQuant is W8A8 with a
  **pre-multiplied scale folded into the previous layer's weights**.
  Representable as a fused "scale-then-quantize" prepass.

### Genuinely orthogonal axes

- **Weight quantization vs activation quantization vs KV-cache
  quantization**: independent configs per tensor role.
- **Element format (int / fp / codebook-LUT)**: int4 / fp4 / NF4 / IQ
  codebook are not reducible to one another.
- **Scale dtype**: fp16 vs UE8M0 vs fp8 vs fp32 must be a first-class enum.
- **Scale topology** (per-tensor / per-row / per-token / blockwise /
  hierarchical super-block): not collapsible.
- **Zero-point presence**: orthogonal to all of the above.
- **Calibration data dependency**: data-free (HQQ, NF4, MX) vs
  activation-aware (AWQ, SmoothQuant, NVFP4) vs Hessian (GPTQ) vs
  importance (i-quants, k-quants `_M`).
- **Outlier handling**: none / LLM.int8() decomposition / Hadamard rotation
  / KIVI residual buffer.

---

## 7. Quantization Parameter Space — the minimal IR axes

Distilling the survey, an IR aiming to *represent* every shipped scheme
needs the following axes (per quantized tensor):

```python
QuantSpec = {
  # --- element representation ---
  "qdtype":           Enum[ int8, int4, int3, int2, int1,
                             fp8_e4m3, fp8_e5m2,
                             fp6_e3m2, fp6_e2m3,
                             fp4_e2m1, nf4, fp4_codebook,
                             iq_codebook ],
  "element_signed":   bool,          # implicit for fp*, explicit for int*

  # --- grouping / scale topology ---
  "quant_axis":       int | Enum[per_tensor, per_token, per_channel_out,
                                 per_channel_in, flattened],
  "group_size":       int | None,    # None = no blocking
  "block_layout":     Enum[contiguous, ocp_mx, nvfp4_inner,
                            gguf_super_block_256, hqq],

  # --- scale ---
  "scale_dtype":      Enum[fp32, fp16, bf16, fp8_e4m3, ue8m0, int8, int6, int4],
  "scale_axis":       int | Enum[per_tensor, per_row, per_block,
                                  per_sub_block, per_head, per_token],
  "scale_quant":      QuantSpec | None,   # recursive (double-quant / NVFP4
                                          # outer fp32 scale / k-quant fp16
                                          # super-scale of 6-bit sub-scales)

  # --- zero point ---
  "has_zero_point":   bool,
  "zero_dtype":       Enum[same_as_qdtype, fp16, int4, int6] | None,
  "zero_axis":        same as scale_axis,

  # --- codebook (for NF4, IQ*, NL) ---
  "codebook":         { "values": tensor | builtin_id, "index_bits": int,
                         "shared_across": Enum[global, per_super_block] } | None,

  # --- packing / storage ---
  "packing":          Enum[unpacked, nibble_pairs, awq_interleaved8x4,
                            gptq_int32, gguf_kquant, gguf_iquant,
                            ocp_mx, nvfp4],
  "storage_order":    Enum[KN, NK, blocked],

  # --- compute / kernel contract ---
  "accumulator_dtype":Enum[fp32, fp16, bf16, int32],
  "compute_dtype":    Enum[fp32, fp16, bf16, fp8_e4m3, int8, int4_native],
                       # what the matmul HW path consumes
  "dequant_fused":    bool,          # weight-only path dequants inline?

  # --- calibration / origin (metadata, not runtime) ---
  "calibration":      Enum[none, rtn, awq, gptq_hessian, smoothquant,
                            hqq, imatrix, percentile, amax_ema,
                            qat, hadamard_rotation],
  "calibration_data": opaque,        # corpus id / token count / α
  "pre_transform":    Enum[none, channelwise_smooth, hadamard, quip_lattice]
                       | None,

  # --- tensor role (for cross-role policies) ---
  "role":             Enum[weight, activation, kv_k, kv_v,
                            attn_qk_score, attn_pv_score, bias],
}
```

The **truly minimal** subset for an IR that just wants to *describe* the
quantized weight buffer for inference (without re-deriving it) is:

```
{ qdtype, group_size, quant_axis, scale_dtype, has_zero_point,
  packing, accumulator_dtype }
```

— seven axes. Everything else (codebook, scale_quant, pre_transform,
calibration) is needed only when the IR must also *produce* the
quantization, do mixed-precision per-layer overrides (Q4_K_M), or surface
hardware-specific kernel selection (MX / NVFP4 / fp8 SDPA).

### Axes the survey *added* beyond the seed list

The prompt's example listed seven axes. The survey forces these
additions:

1. **`scale_quant` (recursive scale spec)** — needed for NF4 double-quant,
   NVFP4 outer fp32, MX UE8M0, k-quant fp16-super × int6-sub.
2. **`codebook`** — needed for NF4 and the IQ family. Not collapsible into
   `qdtype`.
3. **`scale_axis`** distinct from `quant_axis` — k-quants have a per-block
   scale plus a per-super-block scale on *different* axes.
4. **`compute_dtype`** separate from storage `qdtype` — fp8 hardware path
   exists for stored-as-fp8 tensors; W4A16 paths stored-int4 compute-fp16.
5. **`role`** — KV/attention/weight quantization decisions are independent;
   the IR must allow different specs by role.
6. **`pre_transform`** — Hadamard / SmoothQuant rescale precedes
   quantization and must be representable.
7. **`zero_dtype`** — GPTQ packs int zero, AWQ stores fp16 zero, NF4 has
   no zero; cannot be a single bool.
8. **`block_layout`** — packing variants (OCP MX, GGUF k-quant, AWQ
   interleave) are not just bit-widths; the byte layout matters for
   kernel selection.

---

## 8. "If you support only three, support these three"

Ranked by **deployment surface × architectural distinctness**:

1. **GGUF Q4_K_M (k-quant, mixed)** — the dominant on-device SLM format.
   llama.cpp / Ollama / LM Studio / MLX / candle all consume it. Covers
   CPU, Apple Silicon, mobile NPU paths. If you only support one
   quantization, it is this. Exercises: hierarchical scale, asymmetric
   zero, per-layer override, super-block packing, imatrix calibration.
2. **AWQ INT4 W4A16 (group_size = 128, fp16 scale + fp16 zero, AWQ
   interleave)** — the dominant *server-side* 4-bit format for SLMs on
   NVIDIA / AMD GPUs. Supported by vLLM, TRT-LLM, SGLang, HF, MLX. The
   AWQ-Marlin kernel is the de-facto W4A16 GEMM. Exercises: blockwise +
   fp16 scale + zero, packed-interleaved int4 storage, fp16 compute path.
3. **FP8 E4M3 W8A8 with per-tensor (W) + per-token (A) scales** — the
   dominant *compute-quantized* (not memory-only) format on H100/H200,
   MI300, Gaudi-3 and Blackwell. Required for FP8 KV-cache and FP8 SDPA.
   Exercises: fp accumulator path, scale-on-activations, per-token
   topology, sub-byte-but-not-int element format, KV-cache role
   parameterization.

These three are *maximally orthogonal*:

- Element families: int-uniform (AWQ), int-hierarchical-codebook (Q4_K),
  fp (FP8).
- Scale topologies: blockwise-uniform, super-block hierarchical,
  per-tensor + per-token.
- Calibration: activation-aware, weight-only RTN with imatrix,
  amax-history.
- Targets: GPU server, on-device CPU, GPU tensor-core compute.

Implementing them forces every axis in §7 to be exercised.

**Stretch candidates if four–five are feasible**: GPTQ (validates Hessian
calibration path and `act_order` storage variant), MXFP4 (validates UE8M0
and OCP block packing; future-proofs for B200/MI355), NVFP4 (validates
recursive scale-of-scales).

---

## 9. Open questions / unresolved details

1. **NVFP4 outer scale semantics across multi-GPU**: is the per-tensor fp32
   scale broadcast or sharded with TP? TRT-LLM documentation is sparse.
   Affects whether the IR's `scale_quant` field needs a sharding
   annotation.
2. **GGUF imatrix format**: the `--imatrix` file is per-tensor float
   importance; its exact application differs between Q4_K, Q5_K, IQ2_*.
   No single normative spec; behavior is in `llama_model_quantize_internal`.
3. **HQQ + LoRA composition**: HQQ's bf16 scales allow adapters to fold;
   AWQ's fp16 scales technically also allow it; the IR may want a
   `mergeable_with_adapter` predicate.
4. **MXFP6 deployment**: spec exists, hardware support is currently
   B200-tensor-core (E3M2 supported), MI355 (E2M3). No public SLM
   shipped in MXFP6 as of mid-2026; survey based on hardware ISA docs.
5. **KV cache packing in PagedAttention**: vLLM stores K and V in
   different layouts (`[num_blocks, num_kv_heads, head_size/x, block_size, x]`
   for K; `[num_blocks, num_kv_heads, head_size, block_size]` for V).
   The IR's `kv_quant_axis` should probably distinguish K vs V topology,
   not just "kv".
6. **Per-row scaled softmax (FH3)**: how the per-row scale is *exposed* to
   the rest of the graph is not standardized — sometimes folded into
   `softmax_scale`, sometimes carried as a separate tensor. The IR may
   need a `attn_intermediate_scales: maybe[tensor]` field if it wants to
   represent FA3's online rescale.
7. **Codebook globality for IQ**: the E8 lattice constants are hard-coded
   in llama.cpp's `iq2xxs_grid` table. If the IR is to be runtime-agnostic,
   we need to decide whether codebooks are reified tensors (portable) or
   builtin enums (cheap).
8. **Compressed-tensors format (Neural Magic)**: vLLM is consolidating
   schemes under `compressed-tensors`. Its `QuantizationStrategy` enum is
   `tensor | channel | token | group | block`. This is close to a "minimum
   viable common denominator" but does not cover hierarchical scales
   (k-quant) or codebook formats (NF4 / IQ). The IR should be a
   superset.
9. **Asymmetric vs symmetric in fp formats**: fp8/fp4 have signed zero but
   some deployments use unsigned subsets (e.g., positive-only FP4 for
   post-ReLU activations). Not currently shipped in mainstream LLM
   runtimes but the IR should leave room.

---

## 10. References

### Papers

- AWQ — Lin et al., MLSys 2024. arXiv:2306.00978.
- GPTQ — Frantar et al., ICLR 2023. arXiv:2210.17323.
- SmoothQuant — Xiao et al., ICML 2023. arXiv:2211.10438.
- LLM.int8() — Dettmers et al., NeurIPS 2022. arXiv:2208.07339.
- QLoRA / NF4 — Dettmers et al., NeurIPS 2023. arXiv:2305.14314.
- FP8 Formats for Deep Learning — Micikevicius et al., 2022.
  arXiv:2209.05433.
- OCP Microscaling Formats — Rouhani et al., 2023. arXiv:2310.10537.
- QuIP# — Tseng et al., 2024. arXiv:2402.04396.
- KIVI — Liu et al., 2024. arXiv:2402.02750.
- QuaRot — Ashkboos et al., 2024. arXiv:2404.00456.
- SpinQuant — Liu et al., 2024. arXiv:2405.16406.
- FlashAttention-3 — Shah et al., 2024. arXiv:2407.08608.

### Specs / blogs / source

- Open Compute Project, *Microscaling Formats Specification v1.0*, Sept
  2023.
- NVIDIA Blackwell Architecture Whitepaper, 2024.
- NVIDIA Transformer Engine documentation
  (`docs.nvidia.com/deeplearning/transformer-engine`).
- TensorRT-LLM quantization docs
  (`nvidia/TensorRT-LLM/docs/source/reference/precision.md`).
- vLLM quantization guide (`docs.vllm.ai/en/latest/features/quantization`).
- Neural Magic compressed-tensors
  (`github.com/neuralmagic/compressed-tensors`).
- llama.cpp `ggml-quants.h`, `ggml-quants.c`
  (`github.com/ggerganov/llama.cpp`).
- llama.cpp PR #1684 (k-quants), PR #4773 / #5104 (i-quants).
- AutoAWQ (`github.com/casper-hansen/AutoAWQ`).
- AutoGPTQ / GPTQModel (`github.com/ModelCloud/GPTQModel`).
- HQQ (`github.com/mobiusml/hqq`); mobius-labs blog
  "Half-Quadratic Quantization", Nov 2023.
- bitsandbytes (`github.com/TimDettmers/bitsandbytes`).
- ExLlamaV2 / V3 (`github.com/turboderp/exllamav2`).
- Marlin GEMM (`github.com/IST-DASLab/marlin`).
- TensorRT Model Optimizer
  (`github.com/NVIDIA/TensorRT-Model-Optimizer`).
- MLX quantization (`github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py`).
- ONNX Runtime QDQ / QOperator docs.
- OpenVINO NNCF (`github.com/openvinotoolkit/nncf`).

---

*End of survey. Scheme count: 25 distinct schemes documented across five
families. Minimum-viable IR axes: seven; full-coverage axes: ~fifteen
(see §7).*
