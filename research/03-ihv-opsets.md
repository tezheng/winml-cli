# 03 — IHV / Runtime Op-Sets for Transformer Decoder Blocks

> Stream 3 of the `llm-layers` research effort. Goal: extract the operator vocabulary that each independent hardware vendor (IHV) or runtime IR has converged on to express a single LLM decoder block. Where consensus exists across vendors, the `llm-layers` minimal API must cover it. Where vendors diverge (especially in *attention fusion granularity* and *KV-cache representation*) the minimal API must be parameterized.

The decoder block we are tracking, as a baseline, is a Llama / Qwen / Phi-style block:

```
x         = input_norm(h)                       # RMSNorm
q,k,v     = x @ Wq, x @ Wk, x @ Wv              # or fused QKV
q,k       = RoPE(q, k, cos, sin)
k_cache, v_cache = update_kv_cache(k, v)        # write
y         = SDPA(q, k_cache, v_cache, mask)     # core attention
y         = y @ Wo                              # output proj
h         = h + y                               # residual
x2        = ffn_norm(h)                         # RMSNorm
g         = SiLU(x2 @ Wgate)
u         = x2 @ Wup
z         = (g * u) @ Wdown                     # SwiGLU
h2        = h + z                               # residual
```

Plus the final RMSNorm and the LM head matmul on top of the stack.

---

## 1. Intel OpenVINO IR (opset1 → opset16, 2025.x)

### Opsets that matter for LLMs

OpenVINO's published opsets are at `docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets.html`. The LLM-critical additions land in **opset13** (ScaledDotProductAttention), **opset14** (RoPE infrastructure via Symbolic transformations), and a **PagedAttention extension op** that is not numbered into a standard opset but lives in `openvino::op::PagedAttentionExtension` and ships via the runtime.

### Ops actually used to express one Llama-style decoder block

| Logical step | OpenVINO op |
|---|---|
| input RMSNorm | composite: `ReduceMean(x²) → Add(eps) → Sqrt → Divide → Multiply(gamma)`, then *fused* by the `RMSFusion` pass into an internal `RMS` op for CPU/GPU plugins |
| QKV proj | `MatMul` (one per projection, or one fused MatMul when QKV are stacked) |
| RoPE | composite: `Split → Multiply(cos/sin) → Add/Subtract → Concat`, fused by `RoPEFusion`, `RoPEFusionGPTJ`, `RoPEFusionLlama`, `RoPEFusionChatGLM` into an internal `RoPE` op |
| KV cache | `ReadValue` + `Assign` (stateful Variables, opset3+) for the default stateful path; `PagedAttentionExtension` carries `key_cache` and `value_cache` as inputs for continuous-batching backends |
| SDPA | `opset13::ScaledDotProductAttention(q, k, v, attention_mask?, scale?)` with `is_causal` attribute; **sink input added in 2025.4** for attention-sink/streaming-LLM variants |
| Paged SDPA | `PagedAttentionExtension` (28-input mega-op, see below) |
| Output proj | `MatMul` |
| Residual | `Add` |
| FFN gate/up/down | `MatMul`, `MatMul`, `MatMul` |
| SiLU | `Swish` (opset4) — Swish with beta=1 ≡ SiLU |
| gate*up | `Multiply` |
| Final norm | RMSFusion'd RMS |
| LM head | `MatMul` |

### Attention fusion granularity

Two coexisting paths:

* **Stateful path (default for OpenVINO GenAI)**: single incremental KV cache stored in `ReadValue`/`Assign` Variables; the attention itself is **either** decomposed into `MatMul → Add(mask) → Softmax → MatMul` for older opsets, **or** lifted to the single fused **`ScaledDotProductAttention`** op (opset13) by the `SDPAFusion` pass. So the IR may be either primitive or single-op depending on the stage of compilation.
* **Paged path (for continuous batching)**: `ScaledDotProductAttention → PagedAttentionExtension` conversion happens during compilation. This op is a **massive 28-input fused operator** (see §1.4) that internally subsumes the KV-cache update, the masked SDPA, sliding window, ALiBi, attention sinks, KV-cache rotation for eviction, and (in newer builds) xattention and adaptive-RKV.

### PagedAttentionExtension input list (from `openvino/op/paged_attention.hpp`)

Inputs, in order: `query, key, value, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins, scale, sliding_window, alibi_slopes, max_context_len, score_aggregation_window, rotated_block_indices, rotation_deltas, rotation_trig_lut, xattention_threshold, xattention_block_size, xattention_stride, sinks, adaptive_rkv_start_size, adaptive_rkv_evictable_sizes, adaptive_rkv_diversity_block_set_indices, adaptive_rkv_diversity_block_set_indices_begins, token_type_ids, qq_bias, qq_bias_begins`.

This is by far the largest "single op" in any of the runtimes surveyed. It is essentially "the vLLM PagedAttention kernel, lifted into an IR op".

### Quantization

Weights and activations are expressed via the **`FakeQuantize`** (opset1) op or via **`Convert`** chains. NNCF / `ov::pass::low_precision` rewrites those into native-precision MatMuls. Supported native schemes (2025.x):

* W8A8 INT8 symmetric/asymmetric per-channel.
* **W4A16** grouped INT4 (group sizes 32/64/128) — `compressed_weight = u4`, decomposed at compile time as `Convert(u4→f16) → Multiply(scale) → Subtract(zero_point)`.
* W8A16 INT8 weights with FP16 activations.
* **NF4** (4-bit normal float) similar to BnB.
* FP16 / BF16 mixed precision throughout.
* **FP8** (E4M3, E5M2) added in 2025.x via `FakeConvert` op.

KV cache quantization: `KV_CACHE_PRECISION` plugin property accepts `u8`, `u4`, `f16`. The PagedAttention op stores `key_cache` / `value_cache` already-quantized; dequant happens inside the op.

### Surprising / proprietary ops

* `PagedAttentionExtension` — far broader than vLLM's; one IR op that internally fuses RoPE-on-cache rotation, ALiBi, attention sinks, sliding window, score aggregation, xattention sparsity, and adaptive Retentive-KV.
* `RoPE` and `RMS` are *not* in the public opset; they are *internal fusion targets*. The IR you serialize is decomposed; the IR you execute is fused.
* `Swish` doubles as SiLU.

### References

* `https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html`
* `https://github.com/openvinotoolkit/openvino/blob/master/src/core/include/openvino/op/paged_attention.hpp`
* `https://github.com/openvinotoolkit/openvino/pull/30339` (RoPEFusionGPTJ symbolic rewrite)
* `https://blog.openvino.ai/blog-posts/q125-technology-update---low-precision-and-model-optimization`
* OpenVINO GenAI LLM pipeline: `https://deepwiki.com/openvinotoolkit/openvino.genai/2.1-llm-pipeline`

---

## 2. Qualcomm QAIRT / QNN

QAIRT (formerly the QNN SDK) is the canonical execution path for Snapdragon / Hexagon HTP NPUs. Ops are declared in `QnnOpDef.h` (`docs.qualcomm.com/doc/80-63442-10/topic/api-rst_program_listing_file_include_QNN_QnnOpDef_h.html`). The header is the authoritative list; backends (HTP, GPU, CPU) declare which subset they accelerate. Genie is the LLM-specific extension on top of QNN.

### Ops actually used to express one decoder block (HTP path)

| Logical step | QNN op (macro from `QnnOpDef.h`) |
|---|---|
| input RMSNorm | `QNN_OP_RMS_NORM` |
| QKV proj | `QNN_OP_MAT_MUL` or `QNN_OP_FULLY_CONNECTED` |
| RoPE | `QNN_OP_ROTARY_EMBEDDING` (a16w16 sin/cos tables, uint16 inputs, 8-bit downstream) |
| KV write | `QNN_OP_KV_CACHE` (an explicit *named op* — unusual; most other vendors use state or buffer semantics) |
| KV read | the same `QNN_OP_KV_CACHE` op reads back, or KV is wired into MHA inputs |
| SDPA | Either composed: `QNN_OP_MAT_MUL → QNN_OP_ELEMENT_WISE_ADD(mask) → QNN_OP_SOFTMAX → QNN_OP_MAT_MUL`. Genie / newer QNN backends expose **`QNN_OP_SCALED_DOT_PRODUCT_ATTENTION`** (segment-KV-cache aware) |
| Output proj | `QNN_OP_MAT_MUL` / `QNN_OP_FULLY_CONNECTED` |
| Residual | `QNN_OP_ELEMENT_WISE_ADD` |
| FFN | `QNN_OP_MAT_MUL` × 3 |
| SiLU | `QNN_OP_SILU` |
| gate*up | `QNN_OP_ELEMENT_WISE_MULTIPLY` |
| Final norm | `QNN_OP_RMS_NORM` |
| LM head | `QNN_OP_MAT_MUL` |
| Plumbing | `QNN_OP_RESHAPE`, `QNN_OP_TRANSPOSE`, `QNN_OP_CONCAT`, `QNN_OP_GATHER`, `QNN_OP_CAST`, `QNN_OP_QUANTIZE`, `QNN_OP_DEQUANTIZE` |

### Attention fusion granularity

Mixed. The HTP backend prefers **fused SDPA** (Genie's segment-KV-cache SDPA kernel) where available because softmax-MatMul fusion is essential for Hexagon's tile budget. On older HTP firmware you ship decomposed primitives and the converter fuses; on current QAIRT, an explicit `SCALED_DOT_PRODUCT_ATTENTION` exists.

### KV cache representation

`QNN_OP_KV_CACHE` is uniquely an **op rather than state** in QNN. The op receives the new K/V slice and the existing cache tensor reference and returns the appended cache. The cache tensor itself is allocated by the runtime and ranges over user-controlled positional indices. This is conceptually a **named in-place write op** rather than (a) PyTorch-style state, (b) ONNX past/present pair, or (c) vLLM block tables.

### Quantization

HTP is **quantized-only**. Floating-point activations and weights *must* be quantized down to 8-bit or 16-bit integer (or some BF16 paths on Hexagon v75+). Native schemes:

* `a8w8`, `a16w8`, `a16w16` (activation / weight bit width)
* `int4` weights with per-channel or per-block scales
* mixed: empirically RMSNorm is *promoted to a16w16* even when surrounding MatMuls run a8w8, and RoPE sin/cos LUTs use uint16, with output requantized back to uint8 (see ExecuTorch QNN backend issue #16352).
* Per-axis and per-block (group) scales available.

### Mixed-precision conventions

QAIRT has an explicit "promote sensitive ops" pattern: RMSNorm, RoPE LUTs, softmax, and the final LM-head MatMul are commonly executed at a16w16 even when most MatMuls run a8w8. KV cache typically stored u8 with per-token or per-head scales.

### Surprising / proprietary

* `QNN_OP_KV_CACHE` as a first-class op.
* The "promote-sensitive-ops" mixed-precision discipline is *more aggressive* than other runtimes — driven by HTP's lack of FP support.
* Genie LLM pipeline runs **prefill and decode as two separate compiled graphs** with shared KV-cache tensors, not one graph; this affects how the minimal API must allow a "decode-1" graph vs a "prefill-N" graph from the same building blocks.

### References

* `https://docs.qualcomm.com/doc/80-63442-10/topic/api-rst_program_listing_file_include_QNN_QnnOpDef_h.html`
* `https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html`
* `https://github.com/pytorch/executorch/issues/16352` (Quantization recipes for RoPE/RMSNorm in QNN backend)
* `https://xumengwei.github.io/files/ASPLOS25-NPU.pdf` (Fast On-device LLM Inference with NPUs, ASPLOS 2025)

---

## 3. AMD — FastFlowLM (Ryzen AI NPU) and MIGraphX (GPU)

### 3a. FastFlowLM

FastFlowLM is a Ollama-style runtime built directly on the **XDNA2 NPU** in Ryzen AI (Strix / Strix Halo / Kraken / Gorgon Point). Its kernels are **proprietary binaries** (shipped as compiled IRON / AIE-MLIR microcode); the runtime is a thin 16 MB driver-side host that streams tiles.

What's publicly extractable about its op model:

* Prefill and decode are split into "tile-aligned workloads"; both reuse the same set of decoder-block kernels.
* KV state is **kept on-chip** and streamed through the 2D tiled mesh; this means no externally visible KV "buffer op" — the cache lives in NPU column SRAM and the host sees only token-level inputs and logits.
* Context window up to 256k claimed.
* Built on AMD's IRON / AIE-MLIR stack — so logically the ops are AIE-MLIR `aie.tile` micro-kernels covering MatMul, SoftMax, SiLU, RMSNorm, RoPE-apply, residual-Add. There is no published external op-set; the runtime is opaque to the user.

So for `llm-layers` purposes, FastFlowLM is evidence that **a working SLM runtime can hide its operator vocabulary entirely** and expose only the model-level API. The kernels it must contain are clearly the standard nine (RMSNorm, QKV-MatMul, RoPE, KV-write, SDPA, output-MatMul, gate/up/down-MatMul, SiLU, residual-Add) but they are not user-callable.

### 3b. AMD MIGraphX

MIGraphX is AMD's graph inference engine (ROCm). It imports ONNX/TF and lowers to MIOpen, rocBLAS, and HIP kernels. It explicitly supports the **ONNX Runtime contrib operator set** for LLMs, so its IR-level op names for a decoder are essentially ORT contrib names:

| Logical step | MIGraphX op (ONNX-importable) |
|---|---|
| input RMSNorm | `SimplifiedLayerNormalization` / `SkipSimplifiedLayerNormalization` / `RMSNormalization` (ONNX opset 23) |
| QKV proj | `MatMul` / `MatMulNBits` (for INT4 weights) / `Gemm` |
| RoPE | `RotaryEmbedding` (ORT contrib op) |
| KV cache | past/present I/O ports on `Attention` / `GroupQueryAttention` |
| SDPA | `Attention` (contrib) or **`GroupQueryAttention`** (contrib) — single fused op |
| MHA variant | `MultiHeadAttention` (contrib) |
| Output proj | `MatMul` |
| Residual | `Add` or fused into `SkipLayerNormalization` |
| FFN | `MatMul` / `MatMulNBits` |
| activations | `QuickGelu`, `BiasGelu`, `BiasSplitGelu`, `FastGelu`, `Sigmoid`, ORT-`Mul` for SwiGLU |
| Final norm | `SimplifiedLayerNormalization` |
| LM head | `MatMul` |

The CHANGELOG explicitly mentions "Added GroupQueryAttention support for LLMs" and `MatMulNBits` support, so a Llama-style block is **expressed as one `GroupQueryAttention` op + standard MatMul + RMSNorm + RoPE**.

### Attention fusion granularity

**One fused op.** MIGraphX matches the ORT contrib pattern: `GroupQueryAttention` is a single node that swallows the QKV split, RoPE, KV read/write (via past/present), and the masked softmax-MatMul.

### KV cache

ONNX-style **past/present pairs**: `past_key` / `past_value` are inputs, `present_key` / `present_value` are outputs, shape `(B, H, T_past, D)`. There is also `seqlens_k` and `total_sequence_length` for GQA. No block tables at this IR level.

### Quantization

* INT8 SmoothQuant via MIGraphX's quantization passes.
* **`MatMulNBits`** (ORT contrib) — INT4 grouped weights, FP16 activations (W4A16).
* FP8 (E4M3) on MI300X / MI325X via AITER kernels.
* MX-block FP4 on MI350 via Transformer Engine integration.

### Mixed precision

Default FP16/BF16 compute with INT4/INT8 weight compression. Norm and softmax kept FP32 where possible (consistent with other vendors).

### Surprising / proprietary

* MIGraphX deliberately **adopts ORT contrib op names** rather than inventing its own. This is the most "consensus-compatible" of any runtime surveyed.
* No proprietary attention op — it inherits the ORT GQA contract directly.

### References

* `https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/dev/onnx_operators.html`
* `https://github.com/ROCm/AMDMIGraphX/blob/develop/CHANGELOG.md`
* `https://github.com/FastFlowLM/FastFlowLM`
* `https://fastflowlm.com/how-it-works/`
* `https://rocm.blogs.amd.com/software-tools-optimization/jax-aiter/README.html`

---

## 4. NVIDIA TensorRT-LLM

TRT-LLM is the most opinionated runtime in the survey: a Python builder API plus **a sprawling C++ plugin set**. The full plugin set is enumerated by `cpp/tensorrt_llm/plugins/CMakeLists.txt`:

```
bertAttentionPlugin, cpSplitPlugin, fusedLayernormPlugin, gptAttentionCommon,
gptAttentionPlugin, identityPlugin, gemmPlugin, gemmSwigluPlugin,
fp8RowwiseGemmPlugin, smoothQuantGemmPlugin, fp4GemmPlugin,
quantizePerTokenPlugin, quantizeTensorPlugin, quantizeToFP4Plugin,
layernormQuantizationPlugin, rmsnormQuantizationPlugin,
weightOnlyGroupwiseQuantMatmulPlugin, weightOnlyQuantMatmulPlugin,
lookupPlugin, loraPlugin, doraPlugin, mixtureOfExperts,
selectiveScanPlugin, mambaConv1dPlugin, lruPlugin,
cumsumLastDimPlugin, topkLastDimPlugin,
lowLatencyGemmPlugin, eaglePlugin, lowLatencyGemmSwigluPlugin,
qserveGemmPlugin, cudaStreamPlugin, gemmAllReducePlugin,
ncclPlugin (conditional)
```

### Ops actually used to express one Llama decoder block

| Logical step | TRT-LLM plugin / op |
|---|---|
| input RMSNorm | `rmsnormQuantizationPlugin` (fused norm + quantize), or builder-level `rms_norm` lowering to a TRT layer |
| QKV proj | `GemmPlugin` (FP16/BF16), `WeightOnlyQuantMatMulPlugin` (W4A16/W8A16), `WeightOnlyGroupwiseQuantMatMulPlugin` (W4A16 grouped, GPTQ/AWQ), `SmoothQuantGemmPlugin` (W8A8), `Fp8RowwiseGemmPlugin`, `Fp4GemmPlugin` (NVFP4), `QServeGemmPlugin` (W4A8) |
| RoPE | folded **into** `GptAttentionPlugin` (`position_embedding_type` attribute — `learned_absolute`, `rope_gpt_neox`, `rope_gptj`, `alibi`, `relative`, `chatglm`, `long_rope`, `yarn`) |
| KV cache | folded **into** `GptAttentionPlugin` (`paged_kv_cache: bool`, `kv_cache_quant_mode`, `tokens_per_block` — block table managed by runtime) |
| SDPA | folded **into** `GptAttentionPlugin` (context-FMHA + generation MMHA fused inside the plugin) |
| Output proj | `GemmPlugin` / quant variants |
| Residual | tensor `+` (TRT IElementWise) or fused into the next `rmsnormQuantizationPlugin` |
| FFN gate+up | `GemmSwigluPlugin` (single kernel fusing gate-MatMul, up-MatMul, SwiGLU activation, optionally FP8) or `LowLatencyGemmSwigluPlugin` |
| FFN down | `GemmPlugin` / quant variants |
| SiLU | inside `GemmSwigluPlugin`, never standalone |
| Final norm | `rmsnormQuantizationPlugin` or plain `rms_norm` |
| LM head | `GemmPlugin` / `LowLatencyGemmPlugin` |
| All-reduce (TP) | `NcclPlugin`, `GemmAllReducePlugin` (fused GEMM+AR) |
| MoE routing+experts | `MixtureOfExpertsPlugin` |
| LoRA / DoRA | `LoraPlugin`, `DoraPlugin` |
| Embedding lookup | `LookupPlugin` |

### Attention fusion granularity

**Maximally fused.** A single `GptAttentionPlugin` swallows QKV split, RoPE (or ALiBi), KV-cache read/write, masked SDPA (context-FMHA for prefill + masked MMHA for decode), output bias, and quantization scaling. The plugin's `paged_kv_cache` flag toggles paged vs. contiguous KV. This is the strongest "one mega-op per attention" stance of any runtime here.

`BertAttentionPlugin` is the encoder counterpart, used for embedding / re-ranking models.

### KV cache representation

Two modes, both configured on `GptAttentionPlugin`:
* **Contiguous** — single flat buffer per layer.
* **Paged** — block table managed by the TRT-LLM runtime; the plugin receives `kv_cache_block_offsets` describing which physical blocks back the logical sequence.

`KvCacheConfig(dtype='fp8'|'nvfp4'|'int8')` configures KV-cache element type at runtime.

### Quantization

The richest quantization matrix of any runtime surveyed (from `tensorrt_llm.quantization.mode`):

* Weight-only: `W8A16`, `W4A16`, `W4A16_AWQ`, `W8A16_GPTQ`, `W4A16_GPTQ`
* Weight-activation int: `W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN`, `W8A8_SQ_PER_TENSOR_PLUGIN`, etc.
* FP8: `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`
* **NVFP4** (FP4 block): `NVFP4`
* W4A8 hybrid: `W4A8_AWQ`, `W4A8_QSERVE_PER_GROUP`, `W4A8_QSERVE_PER_CHANNEL`, `W4A8_NVFP4_FP8`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`
* MX-FP4: `W4A16_MXFP4`
* KV cache: `INT8`, `FP8`, `NVFP4`
* Mixed-precision: `MIXED_PRECISION`

### Mixed precision

* Activations: BF16 default; FP8 / NVFP4 on Hopper / Blackwell.
* Weights: typically W4A16 or W4A8.
* KV cache: FP8 most common in 2025.
* Norms / softmax: FP32 internal accumulators.
* `GemmAllReducePlugin` fuses TP all-reduce in FP16/BF16 with the preceding GEMM to hide latency.

### Surprising / proprietary

* `GemmSwigluPlugin` and `LowLatencyGemmSwigluPlugin` — fusion of *two GEMMs + a SwiGLU* into one kernel (specifically FP8 on Hopper) is a vendor-specific fusion not seen elsewhere except OpenVINO (where it stays decomposed) and CoreML (no such fusion exposed).
* `QServeGemmPlugin` — W4A8 with per-group scales, a research-grade scheme productized into a plugin.
* `EaglePlugin`, `selectiveScanPlugin`, `mambaConv1dPlugin`, `lruPlugin` — non-transformer ops in the same plugin family, hinting that the minimal API should leave room for SSM / linear attention.
* `cumsumLastDimPlugin`, `topkLastDimPlugin` — sampling helpers as plugins.

### References

* `https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/plugin/plugin.py`
* `https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/plugins/CMakeLists.txt`
* `https://nvidia.github.io/TensorRT-LLM/features/quantization.html`
* `https://nvidia.github.io/TensorRT-LLM/reference/precision.html`
* `https://nvidia.github.io/TensorRT-LLM/_modules/tensorrt_llm/quantization/mode.html`

---

## 5. Apple Core ML (MLProgram, iOS 17 / iOS 18)

### Ops actually used to express one decoder block

| Logical step | Core ML MIL op |
|---|---|
| input RMSNorm | composed from `reduce_mean(square) + add(eps) + rsqrt + mul`; iOS18 has no canonical `rms_norm` op but `layer_norm` is available |
| QKV proj | `linear` or `matmul` |
| RoPE | composed via `split / mul / concat` — no dedicated op |
| KV cache read | `read_state` (iOS17+) returning the pre-allocated stateful buffer |
| KV cache write | `coreml_update_state` (iOS17+) — *in-place* update of the stateful buffer |
| SDPA | **`scaled_dot_product_attention`** (iOS18+), single fused op with `q, k, v, attn_mask?, is_causal` |
| Output proj | `linear` / `matmul` |
| Residual | `add` |
| FFN | `linear` / `matmul` × 3 |
| Activation | `silu` (Swish) |
| gate*up | `mul` |
| Final norm | composed |
| LM head | `linear` |

### Attention fusion granularity

**One fused op since iOS 18** — `scaled_dot_product_attention`. Before iOS 18, attention was decomposed into matmul/softmax/matmul and relied on the Apple Neural Engine compiler to refuse.

### KV cache representation

Uniquely, Core ML expresses KV cache as **a Stateful Buffer**: a model-level pre-allocated tensor that's the input to `read_state` and the target of `coreml_update_state`. The state lives between calls to the model. There is no past/present I/O; instead, the buffer is mutated in place.

This is the most "OS-integrated" of all KV representations and works well with the Neural Engine's memory model. WWDC'24 reported ~1.6× speedup vs. concatenation-based caches.

### Quantization

Strong iOS 18 additions, all via **`constexpr_*`** prefix ops that look like constants at graph time but decompress at runtime/load time:

* `constexpr_blockwise_shift_scale` — block-wise INT4 / INT8 dequantization with shift+scale (block sizes 32/64). This is the W4A16 / W8A16 path.
* `constexpr_lut_to_dense` — palettization: weights stored as small indices into a lookup table; 1–8 bit indices; 4–8-bit-precision LUTs.
* `constexpr_sparse_blockwise_shift_scale` — joint sparsity + block quantization.
* `constexpr_affine_dequantize` (iOS16 legacy).

### Mixed precision

FP16 default on Neural Engine; FP32 accumulators in attention; INT4/INT8 weight-only on the joint compression paths. No FP8 support yet (no Apple hardware exposes FP8).

### Surprising / proprietary

* **Stateful buffers** — only runtime in survey where KV cache is "state" rather than (op, plugin, or block table).
* **LUT-based weight compression as a first-class op** — `constexpr_lut_to_dense` is essentially a vendor-specific lookup-table-quantized matmul building block (paired with regular `matmul`).
* No RoPE op — has to be composed.
* No RMSNorm op — composed from primitives.

### References

* `https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html`
* `https://developer.apple.com/videos/play/wwdc2024/10161/` (Deploy ML models on-device with Core ML)
* `https://huggingface.co/blog/mistral-coreml` (Mistral 7B on Core ML w/ stateful KV cache)
* `https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html`

---

## 6. Apple MLX

MLX is Apple's array library for Apple Silicon; the LLM-relevant surface lives in `mlx.core.fast.*` (custom Metal kernels) and `mlx.core.quantized_matmul`.

### Ops actually used

| Logical step | MLX op |
|---|---|
| input RMSNorm | `mlx.fast.rms_norm` (custom Metal kernel) |
| QKV proj | `mlx.core.matmul` / `nn.Linear` — or `mlx.core.quantized_matmul` for quantized weights |
| RoPE | `mlx.fast.rope(q, dims, traditional, base, scale, offset)` (custom Metal kernel) |
| KV cache | Python-level: `KVCache` class wraps `mx.array`s grown via `mx.concat`; no IR op |
| SDPA | `mlx.core.fast.scaled_dot_product_attention(q, k, v, scale, mask, sinks)` — single fused Metal kernel, supports MHA / GQA / MQA, `mask` accepts `"causal"`, **attention sinks input**, GQA via shape inference (k/v have fewer heads) |
| Quantized SDPA | community `qsdpa` kernel; in-tree `mlx.fast.scaled_dot_product_attention` with quantized KV is in-flight (issue #3404) |
| Output proj | `matmul` / `quantized_matmul` |
| Residual | `mx.add` |
| FFN | `matmul` / `quantized_matmul` × 3 |
| SiLU | `mlx.nn.silu` or `mx.sigmoid * x` |
| gate*up | `mx.multiply` |
| Final norm | `mlx.fast.rms_norm` or `mlx.fast.layer_norm` |
| LM head | `matmul` / `quantized_matmul` |

### Attention fusion granularity

One fused `fast.scaled_dot_product_attention` kernel. KV cache is **not** an op — it's Python state passed in as q/k/v arrays.

### KV cache representation

External arrays managed in Python. No state, no plugin, no block table at the kernel layer. The `KVCache` Python class is convention, not IR.

### Quantization

* `quantized_matmul(x, w, scales, biases, transpose, group_size, bits)` — groupwise weight-only with `bits ∈ {2,3,4,5,6,8}`, `group_size ∈ {32, 64, 128}`.
* `bits=4, group_size=64` is the workhorse W4A16 path.
* No FP8 hardware on Apple silicon, so no FP8 path.

### Mixed precision

FP16/BF16 activations, INT4 quantized weights, FP32 softmax (the docs note "softmax computation always uses float32 precision regardless of input type" — important convention).

### Surprising / proprietary

* `sinks` parameter in `fast.scaled_dot_product_attention` — first-class attention-sink / streaming-LLM support inside the SDPA kernel.
* The runtime is in-process; no compile / serialize step. The "IR" is the Python call graph.

### References

* `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html`
* `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html`
* `https://github.com/ml-explore/mlx/issues/3404` (quantized KV in fast SDPA — TurboQuant)
* `https://machinelearning.apple.com/research/exploring-llms-mlx-m5`

---

## 7. ARM KleidiAI / ExecuTorch ARM backend

ExecuTorch is the PyTorch-side runtime. On ARM CPUs, it delegates LLM workloads to **KleidiAI** micro-kernels (per-feature INT4/INT8 GEMMs) via XNNPACK and an **ARM-VGF / ARM-Ethos** path. For Cortex-X / ARM Neon / SVE / SME2 the path is XNNPACK + KleidiAI.

### Ops actually used (CPU export)

ExecuTorch's exported Llama program operates at the level of PyTorch ATen ops with custom ExecuTorch ops layered on top:

| Logical step | ExecuTorch op |
|---|---|
| input RMSNorm | `executorch.exir.passes.rms_norm` (a custom op since core ATen had no RMSNorm until recently); or `aten.rms_norm` if recent enough |
| QKV proj | `aten.linear` → lowered to **`executorch.llama.linear_int4`** / `xnnpack.dynamic_qd8_linear_qc4w` (KleidiAI W4A8 kernel) |
| RoPE | composed: `aten.split / mul / cat` |
| KV cache + SDPA | **`llama.sdpa_with_kv_cache`** — a single ExecuTorch custom op that does in-place KV update *and* the SDPA, based on Flash Attention 2 for CPU. ~3× faster than decomposed SDPA + default static KV cache; ~2.5× over default. |
| Output proj | `aten.linear` → KleidiAI INT4/INT8 |
| Residual | `aten.add` |
| FFN | `aten.linear` × 3 |
| SiLU | `aten.silu` |
| gate*up | `aten.mul` |
| Final norm | `rms_norm` custom |
| LM head | `aten.linear` |

### Attention fusion granularity

**One fused op for SDPA + KV write** (`sdpa_with_kv_cache`). Notably this op also owns the KV buffer reference, so KV state is *inside the op*'s invocation.

### KV cache representation

External buffer passed in as a tensor input + updated in place by `sdpa_with_kv_cache`. Position offset is passed as an integer. No block tables.

### Quantization

KleidiAI micro-kernels enumerated in `kleidiai/kai/ukernels/matmul/`:

* `matmul_clamp_f32_qai8dxp_qsi4cxp` — FP32 output, INT8-dynamic-quantized activations, INT4-per-channel-symmetric weights (this is the **W4A8 dynamic** path used by Llama 3.2 1B/3B on mobile).
* `matmul_qsi4cxp` family — INT4 grouped weights with BF16 scales per block (block size = group size on the K dimension).
* `matmul_qai8dxp` — INT8 dynamic activation × INT8 weight.

The INT4 SIMD path uses **lookup tables** keyed on each byte (256-entry LUT mapping byte → pair of unpacked nibbles) to dodge branchy sign-extension on Neon.

### Mixed precision

* Activations: dynamically per-row-quantized INT8.
* Weights: INT4 per-channel or per-block symmetric.
* Norms / softmax: FP32.
* Accumulators: INT32 → FP32 dequant.
* SDPA Q/K/V: FP32 inside the custom op (the KV cache itself is also typically FP32 or FP16).

### Surprising / proprietary

* **Lookup-table-based INT4 unpack** as a micro-architectural pattern (LUT-driven nibble unpack, not LUT-quantized weights — different from Core ML's `constexpr_lut_to_dense`).
* `sdpa_with_kv_cache` as a single CPU op (not a plugin) — different fusion model from CUDA plugins.
* SME2 (Scalable Matrix Extension v2) ops are exposed through KleidiAI's `qai8dxp_qsi4c32p_sme` variants for ARMv9 Cortex-X / Apple M4 SME — a new high-throughput INT8×INT4 matmul that, like NVFP4, is block-format-aware.

### References

* `https://github.com/ARM-software/kleidiai/blob/main/docs/matmul_qsi4cx/README.md`
* `https://pytorch.org/blog/unleashing-ai-mobile/`
* `https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/executorch-1-0-is-here-and-with-sme2-optimizations-through-kleidiai`
* `https://docs.pytorch.org/executorch/stable/llm/export-llm.html`
* `https://learn.arm.com/learning-paths/cross-platform/kleidiai-explainer/page3/`

---

## 8. Huawei MindIE / Ascend CANN (ATB)

MindIE is the LLM-serving layer on top of Ascend CANN. Operators are exposed via **ATB (Ascend Transformer Boost)** — a graph-of-fused-ops abstraction. Public details are limited; most authoritative documentation is in Mandarin.

### What's known

* `MindIE-LLM` builds models from **ATB common op builders** (`/usr/local/Ascend/atb-models/atb_llm/common_op_builders/`), each backed by `_libatb_torch._BaseOperation`.
* ATB ships fused transformer ops including: `FlashAttention`, `PagedAttention`, `RmsNorm`, `RopeOperation`, `LinearOperation`, `MatMul`, `SwiGluOperation`, `SoftMax`, `LayerNorm`.
* MindIE-LLM uses ATB op graphs (DAGs of ATB ops) to express a layer — one graph per layer, parameterized.

### Ops actually used (decoder block, ATB)

| Logical step | ATB op |
|---|---|
| input RMSNorm | `RmsNormOperation` |
| QKV proj | `LinearOperation` (W4A16/W8A8 paths via param) |
| RoPE | `RopeOperation` |
| KV write | folded into `PagedAttentionOperation` or `FlashAttentionOperation` |
| SDPA prefill | `FlashAttentionOperation` |
| SDPA decode | `PagedAttentionOperation` |
| Output proj | `LinearOperation` |
| Residual | `ElewiseAddOperation` |
| FFN | `LinearOperation` × 3, or fused `SwiGluOperation` for gate+up+SiLU |
| SiLU | inside `SwiGluOperation` |
| Final norm | `RmsNormOperation` |
| LM head | `LinearOperation` |

### Attention fusion granularity

**One fused op per stage**, like TRT-LLM: separate FlashAttention (prefill) and PagedAttention (decode) ops. KV cache buffers are inputs to both.

### KV cache representation

Block-table-style for `PagedAttentionOperation` (vLLM-aligned), buffers + sequence length tensors for `FlashAttentionOperation`. Cache lives on HBM; Ascend NPU expects block size 16 or 128 depending on chip generation.

### Quantization

W8A8 (per-channel), W4A16 grouped, FP16/BF16, plus Ascend-native HiFloat8 (HF8) on 910B+. KV cache supports INT8.

### Surprising / proprietary

* **SwiGLU is a first-class op** (`SwiGluOperation`) — only OpenVINO/NVIDIA also fuse this hard.
* Two distinct attention ops by phase (FlashAttention vs PagedAttention) — most other runtimes parameterize one op.
* HF8 — Huawei's FP8 variant, exposed only on Ascend.

### Caveats / access

CANN's official op documentation is gated behind Huawei's developer portal and is partially Mandarin-only. Public summaries come from CSDN, Zhihu, and Tencent Cloud blog posts citing internal docs. Treat the list as approximate.

### References

* `https://blog.csdn.net/2501_92602966/article/details/148937014` (MindIE-LLM ATB inference flow)
* `https://zhuanlan.zhihu.com/p/1921257063859884944`
* `https://www.huawei.com/en/news/2025/9/hc-shengten-opensource` (Ascend open-source announcement)
* `https://www.hiascend.com/document/detail/en/canncommercial/850/index/index.html`

---

## 9. ONNX Runtime Contrib Operators (the de-facto reference set)

This is the *lingua franca*: ORT contrib ops are imported by MIGraphX, OpenVINO's ONNX frontend, and the Microsoft ML Engine. The full reference is `https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md`.

### LLM-relevant ops

**Attention family** (fused):
* `Attention` — bidirectional/unidirectional MHA with optional RoPE *inside the op*, optional past/present KV.
* `MultiHeadAttention` — separate Q/K/V inputs, no built-in RoPE, optional past/present.
* `GroupQueryAttention` — GQA, optional RoPE, paged or contiguous KV via past/present + `seqlens_k`/`total_sequence_length`.
* `PagedAttention` — block table + `kv_cache_block_offsets`.
* `SparseAttention` — block-sparse mask.
* `PackedAttention` / `PackedMultiHeadAttention` — variable-length sequence packing.
* `DecoderMaskedMultiHeadAttention` / `DecoderMaskedSelfAttention` — single-token-input decode-path attention; supports beam search via cache indirection.
* `QAttention` / `QOrderedAttention` — quantized attention; `QOrdered*` uses cublasLt ordering.

**Positional embedding**:
* `RotaryEmbedding` — apply rotary positions to Q and K.
* `GemmaRotaryEmbedding` — Gemma-style.

**Norm + fusion**:
* `LayerNormalization`, `SimplifiedLayerNormalization` (= RMSNorm), `SkipLayerNormalization`, `SkipSimplifiedLayerNormalization` (= residual+RMSNorm fused).

**Activations / fused biases**:
* `BiasGelu`, `FastGelu`, `QuickGelu`, `BiasSplitGelu`, `BiasAdd`.

**Quantization**:
* `MatMulNBits` — INT4/INT2 grouped weights, FP16/FP32 activations (W4A16).
* `MatMulBnb4` — BitsAndBytes 4-bit (NF4/FP4).
* `GemmFloat8` — FP8 (E4M3 / E5M2).
* `QGemm`, `DynamicQuantizeMatMul`, `MatMulIntegerToFloat`, `FusedMatMul`.
* `GatherBlockQuantized` — gather + dequant, used for embedding tables.

**Decoding helpers**:
* `BeamSearch`, `GreedySearch`, `Sampling` — these encapsulate the entire autoregressive loop including a sub-graph for the model body.

### KV cache representation

`past_key`, `past_value` → `present_key`, `present_value` paired ports on attention ops. PagedAttention adds block tables. No state semantics.

### Reference

* `https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md`

---

## 10. Cross-Runtime Synthesis Table

Rows: logical operation in the decoder block. Columns: each runtime. Cell: op name in that runtime, or "fused into X" if the runtime swallows it into a larger op.

| Logical op | ONNX RT contrib | OpenVINO | QAIRT/QNN | MIGraphX | TensorRT-LLM | Core ML (iOS18) | MLX | KleidiAI/ExecuTorch | MindIE/ATB |
|---|---|---|---|---|---|---|---|---|---|
| input RMSNorm | `SimplifiedLayerNormalization` | `RMS` (post-fusion) | `QNN_OP_RMS_NORM` | `RMSNormalization` | `rmsnormQuantizationPlugin` | composed (no op) | `mlx.fast.rms_norm` | custom `rms_norm` | `RmsNormOperation` |
| QKV proj | `MatMul` / `MatMulNBits` | `MatMul` | `MAT_MUL` / `FULLY_CONNECTED` | `MatMul` / `MatMulNBits` | `GemmPlugin` (+ quant variants) | `linear` | `matmul` / `quantized_matmul` | `linear_int4` / xnnpack | `LinearOperation` |
| RoPE | `RotaryEmbedding` | `RoPE` (fused) | `ROTARY_EMBEDDING` | `RotaryEmbedding` | fused into `GptAttentionPlugin` | composed | `mlx.fast.rope` | composed | `RopeOperation` |
| KV write | inside Attention (past→present) | `Assign` (stateful) **or** `key_cache` input on `PagedAttentionExtension` | `KV_CACHE` op | inside `GroupQueryAttention` past/present | inside `GptAttentionPlugin` (paged or contiguous) | `coreml_update_state` | Python `KVCache` (no op) | inside `sdpa_with_kv_cache` | inside `PagedAttention` / `FlashAttention` |
| KV read | `ReadValue` / past-input | `ReadValue` / `key_cache` input | `KV_CACHE` op | past-input | inside plugin | `read_state` | Python array | inside custom op | inside ATB op |
| SDPA core | `GroupQueryAttention` / `MultiHeadAttention` / `Attention` | `ScaledDotProductAttention` / `PagedAttentionExtension` | `SCALED_DOT_PRODUCT_ATTENTION` (or MatMul+Softmax+MatMul) | `GroupQueryAttention` | fused in `GptAttentionPlugin` | `scaled_dot_product_attention` | `mlx.fast.scaled_dot_product_attention` | `sdpa_with_kv_cache` | `FlashAttention` / `PagedAttention` |
| Output proj | `MatMul` | `MatMul` | `MAT_MUL` | `MatMul` | `GemmPlugin` | `linear` | `matmul` | `linear` | `LinearOperation` |
| Attn residual | `Add` (often fused into `SkipLayerNormalization`) | `Add` (often fused into next RMSNorm) | `ELEMENT_WISE_ADD` | `Add` / `SkipSimplifiedLayerNormalization` | tensor `+` | `add` | `mx.add` | `aten.add` | `ElewiseAddOperation` |
| FFN gate proj | `MatMul` / `MatMulNBits` | `MatMul` | `MAT_MUL` | `MatMul` | `GemmSwigluPlugin` (fused) | `linear` | `matmul` | `linear` | `SwiGluOperation` (fused) |
| FFN up proj | `MatMul` | `MatMul` | `MAT_MUL` | `MatMul` | inside `GemmSwigluPlugin` | `linear` | `matmul` | `linear` | inside `SwiGluOperation` |
| FFN down proj | `MatMul` | `MatMul` | `MAT_MUL` | `MatMul` | `GemmPlugin` | `linear` | `matmul` | `linear` | `LinearOperation` |
| SiLU activation | `Mul(Sigmoid(x), x)` or `BiasSplitGelu` for GeLU | `Swish` | `SILU` | `Sigmoid`+`Mul` or `BiasSplitGelu` | inside `GemmSwigluPlugin` | `silu` | `mlx.nn.silu` | `aten.silu` | inside `SwiGluOperation` |
| gate*up | `Mul` | `Multiply` | `ELEMENT_WISE_MULTIPLY` | `Mul` | inside `GemmSwigluPlugin` | `mul` | `mx.multiply` | `aten.mul` | inside `SwiGluOperation` |
| FFN residual | `Add` / `SkipSimplifiedLayerNormalization` | `Add` | `ELEMENT_WISE_ADD` | `Add` / skip-norm | tensor `+` | `add` | `mx.add` | `aten.add` | `ElewiseAddOperation` |
| Final RMSNorm | `SimplifiedLayerNormalization` | `RMS` | `RMS_NORM` | `RMSNormalization` | `rmsnormQuantizationPlugin` | composed | `mlx.fast.rms_norm` | custom | `RmsNormOperation` |
| LM head | `MatMul` | `MatMul` | `MAT_MUL` | `MatMul` | `GemmPlugin` / `LowLatencyGemmPlugin` | `linear` | `matmul` / `quantized_matmul` | `linear` | `LinearOperation` |

---

## 11. Consensus, Divergence, and Implications for `llm-layers`

### Where consensus exists (op-class appears in ≥ 7 of 9 runtimes)

* **RMSNorm** — explicit op in 7/9 (ONNX contrib, QNN, MIGraphX, TRT-LLM, MLX, MindIE; *fused* in OpenVINO; *composed* in Core ML and ExecuTorch but trivially so). RMSNorm is so ubiquitous it must be a first-class API in `llm-layers`.
* **MatMul / Linear** — 9/9. Trivial.
* **RoPE** — explicit op in 6/9 (ONNX contrib, QNN, MIGraphX, MLX, MindIE; folded into attention in TRT-LLM, OpenVINO post-fusion). Should be first-class.
* **SDPA as a single fused op** — 9/9. Every runtime has converged on a fused attention. The *shape* of the fusion differs (see divergence below).
* **SiLU / Swish** — 8/9 explicit; only TRT-LLM hides it inside `GemmSwigluPlugin`. First-class.
* **Element-wise Add (residual)** — 9/9.
* **Mul (gate × up)** — 9/9, sometimes inside SwiGLU fusion.

### Where divergence exists

#### Attention fusion granularity (the *defining* divergence)

* **One mega-op** owning QKV-split, RoPE, KV r/w, masked SDPA, output bias, quant scales: **TRT-LLM** (`GptAttentionPlugin`).
* **Mega-op for SDPA only, RoPE+QKV separate**: **MIGraphX/ORT** (`GroupQueryAttention` does RoPE+KV+SDPA but takes pre-projected Q/K/V), **MindIE** (`FlashAttention`/`PagedAttention`).
* **SDPA fused but KV is external state**: **Core ML** (`scaled_dot_product_attention` + `read_state`/`coreml_update_state`), **MLX** (`fast.scaled_dot_product_attention` + Python KV), **OpenVINO stateful path**.
* **SDPA + KV in one custom op**: **ExecuTorch** (`sdpa_with_kv_cache`).
* **PagedAttention as a colossal multi-input op**: **OpenVINO** (`PagedAttentionExtension`, 28 inputs).
* **Two SDPA ops by phase**: **MindIE** (FlashAttention prefill, PagedAttention decode).

**Implication**: `llm-layers` must allow `attention()` to be invoked with parameters that select fusion shape, *not* assume a single fusion. The parameter space is roughly:
- `qkv_layout`: pre-projected vs fused QKV
- `rope_apply`: pre-call vs in-attention
- `kv_kind`: state | past_present | block_table | inline | external_buffer
- `phase`: prefill | decode | unified
- `mask_kind`: causal | custom | sliding_window | none
- `sinks`: none | attention-sinks-tensor

#### KV cache representation (six distinct models found)

1. **past/present I/O ports** — ONNX/ORT, MIGraphX.
2. **Stateful Variables** — OpenVINO (default), Core ML (`read_state`/`update_state`).
3. **Block tables (paged)** — TRT-LLM, OpenVINO PagedAttention, MindIE PagedAttention, ORT PagedAttention.
4. **Op-as-cache** — QNN `KV_CACHE` op.
5. **External buffer threaded into a fused custom op** — ExecuTorch `sdpa_with_kv_cache`.
6. **Pure Python state, no IR** — MLX.

These are *not* interchangeable from a runtime standpoint, but they share a common abstract data structure: `(layer, head, seq_pos, head_dim)` with possibly a permutation for paging. The minimal API should express the *abstract* cache and let the lowering choose the representation.

#### Quantization schemes

* **W4A16 grouped INT4** — *universal*. All 9/9 runtimes support it under some name (`MatMulNBits`, `WeightOnlyGroupwiseQuantMatMulPlugin`, `constexpr_blockwise_shift_scale`, `quantized_matmul`, `matmul_qsi4cxp`, ATB `linear` W4A16, OpenVINO `Convert(u4)+Multiply`, QNN `int4` weight type). Group sizes 32/64/128 are standard.
* **W8A8 INT8** — 8/9 (every runtime except MLX, which has no hardware FP8/INT8 GEMM advantage on Apple silicon).
* **FP8** — 4/9 (TRT-LLM, OpenVINO 2025.x, MIGraphX MI300+, MindIE HF8). Not on QNN (no FP), Core ML, MLX, ExecuTorch CPU.
* **NVFP4 / MXFP4** — only TRT-LLM (NVIDIA) and AMD MI350 via MIGraphX. Forward-looking but not portable.
* **LUT palettization** — only Core ML (`constexpr_lut_to_dense`) as a quantization scheme; ExecuTorch uses LUTs *inside* the INT4 unpack micro-kernel but not as a weight encoding.

### Surprising patterns and proprietary ops worth modeling

* **OpenVINO `PagedAttentionExtension`** is the broadest single op in the survey (28 inputs). It hints that the *paged attention* abstract op might need to grow optional inputs for ALiBi, sinks, sliding window, eviction-rotation LUTs, xattention sparsity, and adaptive Retentive-KV. `llm-layers` should *not* model these as separate ops but as optional parameters.
* **TRT-LLM `GemmSwigluPlugin` / `LowLatencyGemmSwigluPlugin`** — fusing two GEMMs + SwiGLU is a hardware-specific win on Hopper FP8. The minimal API should allow `ffn_swiglu()` to be one logical op with implementation-defined fusion boundary.
* **QNN's `KV_CACHE` op** — only runtime treating the cache update as an *operator* rather than state / buffer / port. Worth flagging because some target backends will *demand* this shape.
* **Core ML `coreml_update_state`** — only runtime with in-place stateful mutation as an explicit semantic. The minimal API needs an abstract "consume + produce updated state" arrow.
* **MLX `sinks` parameter** and **OpenVINO 2025.4 `sink` input on SDPA** — attention sinks crossed the threshold from research to first-class API in 2025. `llm-layers` should expose `sinks` as an SDPA parameter.
* **TRT-LLM `position_embedding_type` enum** — 8 RoPE variants (gptj, gpt_neox, gemma, chatglm, longrope, yarn, alibi, learned_absolute) inside one plugin. This is the most thorough enumeration of RoPE variants in any runtime and suggests the right enum for the minimal API.
* **`SkipLayerNormalization` / `SkipSimplifiedLayerNormalization`** in ORT — residual+norm fusion is so important it gets its own op. TRT-LLM does the same via the `rmsnormQuantizationPlugin` taking a `residual` input. `llm-layers` should bake "residual fed into next norm" as a first-class fused op.
* **MX-FP4 and W4A8 hybrids** — TRT-LLM has 12+ flavors of W4A8 (`W4A8_NVFP4_FP8`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`, `W4A8_QSERVE_PER_GROUP`, ...). The minimal API needs a *quantization spec* (act-bits, weight-bits, scale-format, group-size, kv-cache-bits) not a fixed enum.

### Couldn't access / partial data

* **Huawei MindIE / Ascend CANN** — primary documentation gated behind Huawei developer portal; English material is sparse. The op list in §8 is reconstructed from third-party blog posts (CSDN, Zhihu, Tencent Cloud) citing the ATB common-op-builder directory. Treat as approximate.
* **FastFlowLM** — kernel set is proprietary; only the *runtime* surface is documented. The operator inventory in §3a is inferred from "what kernels must exist to run Llama on XDNA2" — not from public op definitions.
* **Qualcomm `QnnOpDef.h` full enumeration** — the public docs page renders only via doxygen; I was able to extract LLM-relevant macros but not every op in the header. The QNN backend supported-op-list also varies by HTP firmware version (e.g., `SCALED_DOT_PRODUCT_ATTENTION` only on newer firmware).
* **Apple Core ML** — no public op for RMSNorm or RoPE; the canonical decomposition pattern is described only in WWDC session videos and the `mistral-coreml` blog post. I have not verified that recent coremltools releases haven't added them.

---

## 12. Take-aways for the `llm-layers` minimal API design

1. **Nine logical ops cover ≥95% of every decoder block** across all 9 runtimes: `rms_norm`, `linear` (with quant spec), `rope`, `attention(q,k,v,kv_state,mask,sinks)`, `residual_add`, `swiglu_ffn`, `silu`, `mul`, `lm_head`. The minimal API should expose exactly these nine plus an explicit `kv_state` abstraction.
2. **Attention must be parameterized, not fixed.** The right shape is `attention(q, k, v, kv: KVState, mask: MaskSpec, rope: Option<RoPESpec>, sinks: Option<Tensor>, sliding_window: Option<int>) -> (output, kv')`. Lowering decides whether to call `GptAttentionPlugin`, `PagedAttentionExtension`, `sdpa_with_kv_cache`, or a primitive matmul+softmax cascade.
3. **KV cache must be a first-class typed abstraction with multiple lowerings.** The abstract type is `KVState[layer, head, pos, dim]`; lowerings include `Stateful`, `PastPresentPorts`, `BlockTable(paged_block_size)`, `ExternalBuffer`, `OpAsCache`. The API caller never picks one.
4. **Quantization is a `QuantSpec` not an enum.** Parameters: `act_dtype` (bf16/fp16/fp8e4m3/fp8e5m2/int8/nvfp4/mxfp4), `weight_dtype` (int4/int8/fp8/fp4/nf4), `scale_format` (per-tensor/per-channel/per-group/per-block), `group_size`, `kv_dtype`. This subsumes every scheme in §11.
5. **Fused-residual-norm is a real op.** Almost every runtime has `SkipNorm`-style fusion. Expose `residual_rms_norm(prev, residual, gamma) -> (h, new_h)` so that lowerings to TRT-LLM, ORT contrib, Core ML, ATB can all hit a fused path.
6. **SwiGLU-as-one-op** matches the most aggressive fusion targets (TRT-LLM, MindIE) without losing the ability to decompose for runtimes that don't fuse it (CoreML, MLX, OpenVINO).
7. **The minimal API should *not* leak Q/K/V splitting as separate user-facing ops** — every fused-attention runtime accepts merged QKV input. Exposing `qkv_proj(x, Wqkv) -> (q, k, v)` as one logical op (with optional split-then-pack lowerings) is portable.
8. **Sinks and sliding windows are 2025 table-stakes**, no longer research. Both must be supported as optional inputs to `attention`.
9. **Two-phase compilation** (prefill graph vs decode graph) is the norm in QNN/Genie and TensorRT-LLM. The minimal API should produce *abstract* layer descriptions that can be specialized into either phase rather than baking phase into the IR.
