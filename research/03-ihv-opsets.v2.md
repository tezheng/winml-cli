# 03 — IHV / Runtime Op-Sets for Transformer Decoder Blocks (v2)

> Stream 3 of the `llm-layers` research effort. **v2 (2026-06-04)** addresses the issues raised in `research/issues/03-ihv-critique.md` and the master issues list. Major changes vs. v1: (a) +4 P0 runtimes added (ggml, AWS Neuron NKI, Google TPU XLA/Pallas, Microsoft DirectML/Windows ML); (b) +5 P1 runtimes added (TVM-Relax/MLC-LLM, TFLite/LiteRT GenAI, OneDNN Graph SDPA, Meta MTIA, Modular MAX/Mojo); (c) the unverified `QNN_OP_KV_CACHE` claim is **withdrawn** based on a direct fetch of the Qualcomm doxygen listing — see §3.2; (d) every runtime now carries pinned versions; (e) the synthesis table grows from 9 columns × 16 rows to 18 columns × 18 rows; (f) the KV-cache representation count is corrected from 6 to 11 distinct shapes; (g) the attention-fusion taxonomy is grown from 5 to 9 shapes; (h) §8 is a new five-era evolution narrative (2022 → 2026).
>
> **Status of central claims after re-verification:** 38 claims verified against primary sources (header file, release notes, source tree); 7 claims could not be verified and are explicitly marked `[unverified]`; the previously-load-bearing "QNN treats KV as an op" finding is **falsified** and the QAIRT row is recategorised. See §10 for the full ledger.

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

Plus the final RMSNorm, the LM head matmul, plus (for MoE families) a router + top-k expert dispatch step replacing the SwiGLU, and (for SSM-hybrid families) a `conv1d` + `selective_scan` token mixer replacing attention. The v2 minimal-API conversation must accommodate all three.

---

## 1. Methodology and runtime selection

### 1.1 Methodology

For each runtime we extract: **(a)** identity and last-pinned release version observed in 2025–2026; **(b)** the LLM-relevant op set declared in primary source (header file, IR spec, public docs); **(c)** the KV-cache representation; **(d)** attention fusion granularity; **(e)** quantization scheme coverage; **(f)** evolution markers (when major LLM features landed). Where a claim cannot be directly verified against primary source — typically header files, release notes, or merged PR descriptions — the claim is explicitly marked `[unverified]`. Where the v1 claim turned out to be **wrong** on re-verification, the v2 retraction is called out with the substring `**RETRACTED in v2**`.

Primary verification channels:

* **ggml** — `https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h` (master, 2026-Q1).
* **ORT contrib** — `https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md` (main, fetched 2026-06-04). v1's claim that `SimplifiedLayerNormalization` and `RMSNormalization` appear in this document is **partially false**: `SimplifiedLayerNormalization` and `RMSNormalization` are **not** in the index but `SkipSimplifiedLayerNormalization` **is**. `SimplifiedLayerNormalization` is implemented in `contrib_ops/cpu/layer_norm.cc` (C++ only, no markdown spec). `RMSNormalization` is shipping as an **ONNX standard opset-23 op** rather than a contrib op (see `microsoft/onnxruntime#23560`, `#21925`).
* **TensorRT-LLM** — `cpp/tensorrt_llm/plugins/CMakeLists.txt` and `tensorrt_llm/quantization/mode.py` at v0.x→1.x range.
* **OpenVINO** — `src/core/include/openvino/op/paged_attention.hpp` at 2025.4 release branch. v1's URL 404s on master; the file path is correct but the master branch has continued moving. **The 28-input list in v1 mixes shipped 2025.4 fields with feature-branch fields** (see §2.1 — input count correction).
* **Qualcomm QAIRT** — `QnnOpDef.h` Doxygen listing at `https://docs.qualcomm.com/doc/80-63442-10/topic/api-rst_program_listing_file_include_QNN_QnnOpDef_h.html`.
* **Core ML** — `coremltools.converters.mil.mil.ops.defs` at coremltools 8.x.
* **MLX** — `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html` at MLX 0.31.x.
* **ExecuTorch** — `examples/models/llama2/` at ExecuTorch 0.4 → 1.0 range.

### 1.2 Runtime selection criteria

We surveyed runtimes meeting **at least two** of the following: (a) ships in a production LLM serving stack (vLLM, llama.cpp, Ollama, TensorRT-LLM, MindIE); (b) shipped by an IHV as the canonical path to LLMs on that IHV's silicon (CoreML for Apple silicon, NKI for Trainium, etc.); (c) defines its own LLM op vocabulary distinct from the others. Runtimes meeting none of these (e.g., raw ROCm libs absent MIGraphX, isolated academic kernel collections) were excluded.

### 1.3 The 18 covered runtimes

P0 IHV/runtime stacks (each gets a full §2 section):

1. **OpenVINO 2025.4** (Intel CPU/GPU/NPU)
2. **Qualcomm QAIRT 2.36** (Snapdragon / Hexagon HTP NPU)
3. **AMD MIGraphX 2.13** + **FastFlowLM** (Ryzen AI / ROCm)
4. **NVIDIA TensorRT-LLM 1.x** (Hopper / Blackwell GPU)
5. **Apple Core ML / coremltools 8.x** (Apple silicon, iOS 17 / iOS 18 / iOS 19)
6. **Apple MLX 0.31** (Apple silicon)
7. **ARM KleidiAI / ExecuTorch 0.4 → 1.0** (ARM Cortex / Neon / SVE / SME2)
8. **Huawei MindIE / Ascend ATB** (Ascend 910B / 910C)
9. **ONNX Runtime contrib** 1.21 / 1.22 / 1.23 (lingua franca)
10. **ggml** (master, 2026-Q1) — dominant on-device CPU/Metal LLM runtime
11. **AWS Neuron NKI 0.3.0 / Neuron SDK 2.29** (Trn1/Trn2, Inf2)
12. **Google TPU — XLA HLO + Pallas / Mosaic** (TPUv4/v5/v6/Trillium, MaxText, vLLM-TPU)
13. **Microsoft DirectML 1.15.4 + Windows ML 1.x** (DX12 GPU, NPU EP)

P1 runtimes (each gets a full §2 section but slightly tighter):

14. **TVM-Relax / MLC-LLM** (browser, Android, iOS, Vulkan)
15. **TensorFlow Lite / LiteRT GenAI / LiteRT-LM** (Android, mobile)
16. **OneDNN Graph SDPA** (Intel CPU/GPU primitive layer underneath OpenVINO)
17. **Meta MTIA v1 / v2** (PyTorch eager + Triton + TorchInductor)
18. **Modular MAX 25.6 / Mojo 1.0** (Apple silicon, NVIDIA Blackwell, AMD MI355X)

In addition, we briefly touch on three "closed-vocabulary IHVs" — Hailo-10H, Groq LPU, Cerebras WSE — at the end of §2 because they form a recurring pattern of *hiding the op vocabulary entirely*, and on three "PyTorch-native attention-kernel layers" — vLLM custom ops, SGLang RadixAttention, cuDNN fused SDPA — because they shape the consumer side of how IHV op-sets are exercised.

This brings the total covered surface to **18 first-class runtimes plus 6 acknowledged closed/derivative** = 24 distinct points in the op-set design space (vs. v1's 9). The earlier "9 runtimes converged" universality claim is replaced with a more cautious "across 18 surveyed runtimes, the following ops are universal vs. divergent" (§4 and §9).

---

## 2. Per-runtime op-set analysis

### 2.1 Intel OpenVINO (2024.4 → 2025.4)

**Identity.** OpenVINO is Intel's cross-architecture inference runtime targeting CPU, integrated GPU (Xe / Battlemage), discrete GPU (Arc, Data Center GPU Max), and the NPU shipped on Meteor Lake / Lunar Lake / Arrow Lake / Panther Lake. Its IR uses numbered opsets (`opset1` … `opset16` in current docs) plus extension ops that live outside the numbered series.

**Version pinning.** Last verified against the **2025.4 release** (Intel article ID `release-notes/openvino/2025-4.html`, dated late 2025). Important version anchors:

| Version | LLM-relevant addition |
|---|---|
| 2024.4 | ContinuousBatching pipeline; KV cache compression to `u8` |
| 2024.5 | PagedAttention path lit up across CPU/GPU/NPU; first SDPA → PagedAttention conversion pass shipped |
| 2024.6 | `RoPEFusionGPTJ`/`RoPEFusionLlama` patterns broadened; FP8 (`FakeConvert`) added [unverified for exact .x] |
| 2025.0 | NPU LLM pipeline general availability |
| 2025.1–2025.3 | SDPA → PagedAttention robustness, GPU PagedAttention rotation kernels, ARM PagedAttention enablement (#27841) |
| 2025.4 | **`sink` input on PagedAttention** (GPT-OSS-20B support), **XAttention** as preview feature, hybrid attention with linear state (CausalConv1d, GatedDeltaNet), SDPA + PA backend for VLMs (Qwen3.5, Gemma 4) |

**Ops actually used to express one Llama-style decoder block.**

| Logical step | OpenVINO op | Notes |
|---|---|---|
| input RMSNorm | composite `(ReduceMean(x²) → Add(eps) → Sqrt → Divide → Multiply(gamma))` rewritten by `RMSFusion` (in `src/common/transformations/`) into an internal `RMS` op | Serialized IR remains composite; in-memory IR after compile is fused. |
| QKV proj | `MatMul` (×3 or fused) | |
| RoPE | composite Split/Mul/Add/Concat, rewritten by `RoPEFusion`, `RoPEFusionGPTJ`, `RoPEFusionLlama`, `RoPEFusionChatGLM` into internal `RoPE` op | Internal-only — not in numbered opset. |
| KV cache (stateful) | `opset3::ReadValue` + `opset3::Assign` (Variables) | OpenVINO GenAI default path. |
| KV cache (paged) | `key_cache` + `value_cache` inputs to `PagedAttentionExtension` | Block table managed by runtime. |
| SDPA (stateful) | `opset13::ScaledDotProductAttention(q, k, v, attention_mask?, scale?)` with `is_causal` attribute; **sink input added on the PagedAttention path in 2025.4** | An additional `opset14::ScaledDotProductAttention` overload exists with adjusted attribute set. |
| SDPA (paged) | `PagedAttentionExtension` (see input count correction below) | The compiled IR for a continuous-batching pipeline uses this exclusively. |
| Output proj | `MatMul` | |
| Residual | `Add` (often fused into next-RMSNorm) | |
| FFN gate/up/down | `MatMul` ×3 | No SwiGLU-fused op in opset; remains decomposed. |
| SiLU | `opset4::Swish` (β=1) | |
| gate*up | `Multiply` | |
| Final norm | RMSFusion'd `RMS` | |
| LM head | `MatMul` | |
| MoE | `MoE` extension op since 2025.x; previously decomposed `Softmax → TopK → Gather → MatMul` per expert | "Improved performance for Mixture-of-Experts subgraphs" cited in 2025.4 notes. |
| Embedding lookup | `Gather` | |
| All-reduce (TP) | `Allreduce` extension (multi-device GenAI) | |
| Sampling | external — handled by GenAI pipeline | |
| Mask construction | `is_causal` attribute on SDPA; explicit mask via `attention_mask` input | |
| Quant-dequant | `FakeQuantize`, `FakeConvert` (FP8), `Convert(u4→f16) → Multiply → Subtract` | |

**PagedAttention input count correction.** v1 asserted 28 inputs for `PagedAttentionExtension`. The header file `src/core/include/openvino/op/paged_attention.hpp` could not be fetched on the master branch (the file path changed across the 2025.x series; recent PRs #27204, #28232, #28493, #28815 indicate active churn on the operator signature). The conservative count that we can defend against the **2025.4 release branch** is ~22 mandatory inputs covering `query, key, value, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins, scale, sliding_window, alibi_slopes, max_context_len, score_aggregation_window, rotated_block_indices, rotation_deltas, rotation_trig_lut, sinks` plus 4 sparse-attention optional inputs (`xattention_threshold, xattention_block_size, xattention_stride, score_threshold`). The `adaptive_rkv_*` and `qq_bias_*` fields listed in v1 are **feature-branch only** as of 2025.4 and should not appear in the synthesis. We mark them `[unverified for 2025.4 release]`.

**Attention fusion granularity.** Two paths coexist:

* **Stateful path** (OpenVINO GenAI default for single-stream pipelines): the IR holds `ScaledDotProductAttention` plus `ReadValue`/`Assign` for KV. The SDPA op is **monolithic per call** — it accepts `q,k,v` *post*-RoPE.
* **Paged path** (continuous-batching, vLLM-equivalent throughput): `PagedAttentionExtension` is a **super-fused op** subsuming KV-cache update, masked SDPA, sliding window, ALiBi, attention sinks, cache rotation, and optional XAttention sparsity.

**Quantization.**

* `FakeQuantize` (opset1) and `Convert` chains expressed at the IR level; `ov::pass::low_precision` lowers them to native MatMul kernels.
* W8A8, W4A16 (group 32/64/128), W8A16, NF4, FP16/BF16, **FP8 (E4M3/E5M2) via `FakeConvert`** (added 2024.x — exact .x unverified).
* KV-cache: `KV_CACHE_PRECISION` plugin property — verified to accept `u8` and `f16`; `u4` support added in later 2025.x [unverified for exact release; v1 claim retained with a footnote rather than asserted].

**Doc URLs (pinned).**

* SDPA spec: `https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html`
* 2025.4 release notes (sink input, XAttention, MoE perf): `https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-4.html`
* Source tree: `https://github.com/openvinotoolkit/openvino/tree/releases/2025/4`
* PagedAttention PR series (2024 → 2025): #26975, #27204, #27841, #28017, #28232, #28493, #28645, #28815, #29383

### 2.2 Qualcomm QAIRT / QNN (QAIRT 2.27 → 2.36, formerly QNN SDK)

**Identity.** QAIRT is Qualcomm's runtime/execution layer for Hexagon HTP NPUs (Snapdragon 8 Gen 3 / 8 Elite / X Elite / X2 Elite, automotive Ride). Genie is the LLM-specific extension. Ops are macros in `QnnOpDef.h`.

**Version pinning.** AIMET-ONNX (the quant pipeline that feeds QAIRT) is at 2.27 → 2.29 across 2025–early 2026 per AI Hub release notes. The SDK itself ships as QAIRT 2.x; the Doxygen-rendered `QnnOpDef.h` we cite below is from the 80-63442-10 documentation bundle (`docs.qualcomm.com`, fetched 2026-06-04).

**`QNN_OP_KV_CACHE` claim — RETRACTED in v2.** The single most consequential proprietary-op claim in v1 was that QNN treats KV cache as a first-class **named op** via `QNN_OP_KV_CACHE`. On direct verification against the public listing of `QnnOpDef.h` (URL above, fetched 2026-06-04), **no macro named `QNN_OP_KV_CACHE` exists.** The macros that *do* exist and that we could verify by direct quote include `QNN_OP_ROTARY_EMBEDDING` ("RotaryEmbedding"), `QNN_OP_RMS_NORM` ("RmsNorm"), `QNN_OP_MAT_MUL` ("MatMul"), `QNN_OP_FULLY_CONNECTED` ("FullyConnected"), and `QNN_OP_ELEMENT_WISE_ADD` ("ElementWiseAdd"). Also **not** present in the public listing: `QNN_OP_KV_CACHE`, `QNN_OP_SCALED_DOT_PRODUCT_ATTENTION`, `QNN_OP_SILU`. v1's confident assertion of a Genie-level `SCALED_DOT_PRODUCT_ATTENTION` op rested on a paper aside that we cannot corroborate against `QnnOpDef.h` directly. Both are downgraded to `[unverified, likely Genie-internal]`.

**Implication for the taxonomy.** With `QNN_OP_KV_CACHE` retracted, the "op-as-cache" category in the v1 §11 KV-representation list **collapses**. QAIRT's actual KV mechanism, based on the public ops, is **(a)** at the QNN graph level: ordinary tensor I/O with KV tensors threaded as graph inputs/outputs (functionally equivalent to ORT past/present); **(b)** at the Genie pipeline level: prefill and decode are *two separately compiled graphs* that share KV tensors host-side via the Genie runtime (a non-IR convention, not an op). We re-categorise QAIRT as **`PrefillDecodeSplit + ExternalBuffer`** in the §5 KV taxonomy.

**Ops actually used to express one decoder block (HTP path).**

| Logical step | QNN op (macro from `QnnOpDef.h` listing 80-63442-10) | Verification |
|---|---|---|
| input RMSNorm | `QNN_OP_RMS_NORM` = `"RmsNorm"` | **verified** |
| QKV proj | `QNN_OP_MAT_MUL` = `"MatMul"` or `QNN_OP_FULLY_CONNECTED` = `"FullyConnected"` | **verified** |
| RoPE | `QNN_OP_ROTARY_EMBEDDING` = `"RotaryEmbedding"` | **verified** |
| KV r/w | threaded as graph I/O tensors; **not a named op** | retracted from v1 |
| SDPA | decomposed `MatMul → ElementWiseAdd(mask) → Softmax → MatMul` on portable QNN; a fused SDPA *may* exist in Genie/HTP-internal op tables but is not in the public `QnnOpDef.h` listing | `[unverified]` |
| Output proj | `QNN_OP_MAT_MUL` | **verified** |
| Residual | `QNN_OP_ELEMENT_WISE_ADD` | **verified** |
| FFN | `QNN_OP_MAT_MUL` × 3 | |
| SiLU | composed `Sigmoid + ElementWiseMultiply`; no `QNN_OP_SILU` in public listing | retracted from v1 |
| gate*up | `QNN_OP_ELEMENT_WISE_MULTIPLY` | |
| Final norm | `QNN_OP_RMS_NORM` | **verified** |
| LM head | `QNN_OP_MAT_MUL` | |
| Quant-dequant | `QNN_OP_QUANTIZE` / `QNN_OP_DEQUANTIZE` | **verified** |
| Plumbing | `QNN_OP_RESHAPE`, `QNN_OP_TRANSPOSE`, `QNN_OP_CONCAT`, `QNN_OP_GATHER`, `QNN_OP_CAST` | **verified** |

**Attention fusion granularity.** Without a verifiable `SCALED_DOT_PRODUCT_ATTENTION` macro, the conservative reading is **primitive decomposition** (`MatMul → ElementWiseAdd → Softmax → MatMul`) at the public QNN graph layer, with HTP-internal kernel fusion happening below the QNN API (and therefore invisible to the model author). This places QAIRT in the "primitive-decomposition with backend kernel fusion" cell of the fusion taxonomy (§5), not the "single fused SDPA op" cell where v1 placed it.

**Quantization.**

* `a8w8`, `a16w8`, `a16w16`, `int4` weights, per-axis and per-block scales.
* The mixed-precision discipline ("promote sensitive ops": RMSNorm/RoPE/softmax/LM-head MatMul at a16w16; rest at a8w8) is established practice but largely a convention enforced at conversion time, not an op-level construct.

**Surprising / proprietary.**

* RoPE has a first-class op (`RotaryEmbedding`) — confirms convergence with ORT contrib, TRT-LLM, MLX.
* RMSNorm has a first-class op (`RmsNorm`) — also convergent.
* Genie ships **prefill and decode as separate compiled graphs**, sharing KV tensors via runtime convention. The minimal API must keep room for "phase-specialised graphs from the same logical layer".

**Doc URLs (pinned).**

* `QnnOpDef.h` listing: `https://docs.qualcomm.com/doc/80-63442-10/topic/api-rst_program_listing_file_include_QNN_QnnOpDef_h.html`
* `QNN_OP_DEQUANTIZE` macro (specimen): `https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/define_QnnOpDef_8h_1a97a823ed1f60d9b1304461de39cf203d.html`
* QAIRT overview: `https://docs.qualcomm.com/nav/home/QNN_general_overview.html`
* ASPLOS 2025 NPU paper (Genie phase split confirmation): `https://xumengwei.github.io/files/ASPLOS25-NPU.pdf`

### 2.3 AMD MIGraphX (ROCm 6.x → 7.x) and FastFlowLM (Ryzen AI XDNA2)

**MIGraphX 2.13** (ROCm 7.0 era, late 2025). MIGraphX's importer accepts ORT contrib ops directly, which makes it the most "consensus-compatible" runtime: a Llama block becomes one `GroupQueryAttention` op + standard MatMul + RMSNorm + RoPE at the importable ONNX level.

The op list per logical step is identical to ORT contrib in v1 §3b and stable in v2. The newly relevant detail is the **AITER kernel family** for AMD MI300X/MI325X/MI350: FP8 (E4M3) via the AITER FlashAttention path (verified via ROCm blogs), MX-FP4 on MI350 via Transformer Engine. Pin: AITER 0.x → 1.x, ROCm 7.0+.

**FastFlowLM.** XDNA2 NPU runtime, kernels are proprietary IRON / AIE-MLIR microcode. We treat it as a "closed-vocabulary IHV" instance (joining Hailo, Groq, Cerebras in this pattern). For minimal-API purposes, FastFlowLM is evidence that the *logical* API surface can be hidden entirely; the kernels it contains must be the standard nine but they are not user-callable. Pin: FastFlowLM 0.x as of 2026-Q1.

### 2.4 NVIDIA TensorRT-LLM (0.5 → 1.x, with TRTLLM-Gen)

**Identity.** TRT-LLM is the most opinionated runtime in the survey: Python builder API + a sprawling C++ plugin set. Verified plugin list (from `cpp/tensorrt_llm/plugins/CMakeLists.txt`, fetched 2026-Q1):

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

**Version pinning.**

| Phase | TRT-LLM era | Key shift |
|---|---|---|
| 2023 | 0.5 → 0.7 | First public release; `GptAttentionPlugin` introduced as monolithic attention plugin; FP16/BF16 only. |
| 2024 H1 | 0.8 → 0.10 | Paged KV cache becomes default; FP8 (Hopper) GEMMs; W4A16 GPTQ/AWQ. |
| 2024 H2 | 0.11 → 0.14 | NVFP4 (Blackwell) support; `GemmSwigluPlugin` lands; `qserveGemmPlugin` (W4A8). |
| 2025 H1 | 0.15 → 0.20 | TRTLLM-Gen FP4 GEMM lands — the **single-binary inference** path. Plugin builds are no longer mandatory for many models; the runtime picks kernels at runtime. |
| 2025 H2 | 0.x → 1.0 | Stabilisation of TRTLLM-Gen path; FP8 MLA for Hopper and Blackwell; chunked-prefill performance work. |
| 2026 Q1 | 1.x | Mixed Plugin+Gen path is canonical. |

The **TRTLLM-Gen** path was the major architectural shift of late 2025 and is absent from v1. It replaces the "build a per-model engine of plugins" workflow with a runtime that selects fused kernels by attribute. Effectively this is a step from "plugin = op" to "plugin = library entry, kernel = op", aligning TRT-LLM closer to the cuDNN-fused-SDPA / FlashInfer model.

**Ops table.** Identical to v1 §4 with two additions:

* `mixtureOfExperts` plugin — first-class MoE op, used by Mixtral, DeepSeek-V3, Qwen3-MoE.
* `cumsumLastDimPlugin` + `topkLastDimPlugin` — sampling helpers (added to synthesis table in §3).

**Attention fusion granularity.** `GptAttentionPlugin` swallows: QKV split (or accepts pre-projected Q/K/V depending on the model), RoPE (or ALiBi), KV r/w (paged via `paged_kv_cache: bool`), masked SDPA (context-FMHA prefill + masked-MMHA decode), output bias, quant scaling. Position-embedding-type enum: `learned_absolute, rope_gpt_neox, rope_gptj, alibi, relative, chatglm, long_rope, yarn` — 8 variants (verified). **In TRTLLM-Gen builds, this plugin can be bypassed entirely** by the runtime selecting trtllm-gen kernels at dispatch time.

**KV cache.** Two modes: contiguous (single flat buffer per layer) and paged (block table via `kv_cache_block_offsets`). `KvCacheConfig(dtype=...)` accepts `int8|fp8|nvfp4`.

**Quantization matrix.** v1 list is correct and verified against `tensorrt_llm/quantization/mode.py`. The 12+ W4A8 variants (`W4A8_NVFP4_FP8`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`, `W4A8_QSERVE_PER_GROUP`, etc.) remain the richest quantization matrix in the survey.

**Doc URLs (pinned).**

* `https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0/cpp/tensorrt_llm/plugins/CMakeLists.txt` (use the v1.0 tag for the verified plugin list)
* `https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0/tensorrt_llm/plugin/plugin.py`
* `https://nvidia.github.io/TensorRT-LLM/release-notes.html`
* `https://nvidia.github.io/TensorRT-LLM/features/quantization.html`

### 2.5 Apple Core ML (coremltools 7.x → 8.x, iOS 17 → iOS 19)

**Version pinning.**

| iOS / coremltools | LLM-relevant change |
|---|---|
| iOS 16 / coremltools 6 | No state; KV cache implemented via concatenation; `constexpr_affine_dequantize` for legacy quant. |
| **iOS 17 / coremltools 7.2** | **Stateful KV cache lands.** `read_state` and `coreml_update_state` MIL ops. WWDC '24 reports ~1.6× speedup vs. concatenation on Mistral-7B / M3 Max. |
| **iOS 18 / coremltools 8.0** | **`scaled_dot_product_attention` MIL op.** Single fused SDPA replacing the previous matmul/softmax/matmul decomposition. New `constexpr_blockwise_shift_scale`, `constexpr_lut_to_dense`, `constexpr_sparse_blockwise_shift_scale` for blockwise / LUT / joint sparse compression. |
| iOS 18.1 / coremltools 8.1 | Bug fixes around view + transpose on state (issue #2275); BFloat16 broadening. |
| iOS 18.x / coremltools 8.2 / 8.3 | Stateful + SDPA integration ergonomics. |
| iOS 19 / coremltools 9 (expected) | Apple Foundation Models / Apple Intelligence on-device LLM uses a closed surface *above* MIL — distinct from the MIL ops surveyed here. |

**Ops table.** Identical to v1 §5 except for the iOS-version annotations above. No native `rms_norm` or `rope` MIL op exists — both remain composed from primitives. `layer_norm` exists (used by Mistral-style models).

**Attention fusion granularity.** "SDPA fused, KV is external state" — `scaled_dot_product_attention(q,k,v, attn_mask?, is_causal)` after iOS 18, plus `read_state` / `coreml_update_state` for KV. The state and SDPA remain **separate MIL ops**; we have no evidence of a single MIL block that fuses them.

**Doc URLs (pinned).**

* `https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html`
* `https://apple.github.io/coremltools/docs-guides/source/stateful-models.html`
* `https://developer.apple.com/videos/play/wwdc2024/10161/`
* `https://developer.apple.com/videos/play/wwdc2024/10159/`
* `https://machinelearning.apple.com/research/core-ml-on-device-llama`

### 2.6 Apple MLX (0.20 → 0.31)

**Version pinning.**

| MLX | LLM-relevant change |
|---|---|
| 0.20 / 2024 H2 | `fast.scaled_dot_product_attention(q, k, v, scale, mask)` — single fused Metal kernel for SDPA. |
| 0.24–0.25 / 2025 H1 | GQA / MQA via shape inference on k/v heads; quantized matmul mask support broadened. |
| 0.27–0.29 / 2025 H2 | `sinks` parameter added to `fast.scaled_dot_product_attention` for attention-sink / streaming-LLM models (GPT-OSS, Mistral-Small-3.1). Exact MLX release tag that introduced `sinks` is `[unverified]` — doc page on 0.31.1 shows the parameter as present but does not give the changelog entry. |
| 0.31 / 2026 Q1 | Doc page `mlx.core.fast.scaled_dot_product_attention` confirms `q, k, v, scale, mask, sinks` signature. |
| (in-flight) | Quantized-KV in fast SDPA tracked in issue #3404 (TurboQuant); FlashAttention-style integration in issue #2955. |

**Ops table.** Identical to v1 §6.

**Attention fusion granularity.** "SDPA fused, KV is pure Python state". The `KVCache` Python class is convention, not IR — there is no compile / serialize step in MLX, so the runtime *is* the IR.

**Quantization.** `quantized_matmul(x, w, scales, biases, transpose, group_size, bits)` with `bits ∈ {2,3,4,5,6,8}`, `group_size ∈ {32,64,128}`. `bits=4, group_size=64` is the canonical W4A16 path. Verified in MLX 0.31.

**Doc URLs (pinned).**

* `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html`
* `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html`
* `https://github.com/ml-explore/mlx/issues/3404` (TurboQuant)
* `https://github.com/ml-explore/mlx/issues/2955` (FlashAttention proposal)

### 2.7 ARM KleidiAI / ExecuTorch (0.4 → 1.0)

**Version pinning.**

| ExecuTorch | KleidiAI | LLM-relevant change |
|---|---|---|
| 0.2 / 2024 H1 | early KleidiAI | First Llama-2 export path on Android; basic KV cache. |
| 0.4 / 2024 H2 | KleidiAI 1.x | `sdpa_with_kv_cache` custom op shipped; ~3× over decomposed SDPA + 2.5× over default static KV cache for CPU. |
| 0.5–0.7 / 2025 | KleidiAI 1.x | XNNPACK + KleidiAI W4A8 `dynamic_qd8_linear_qc4w` becomes default for Llama 3.2 1B/3B on mobile. |
| **1.0 / late 2025** | KleidiAI ≥ 1.x with SME2 | SME2 path live via `qai8dxp_qsi4c32p_sme` variants on ARMv9 Cortex-X / Apple M4 SME. |

**Ops table.** Identical to v1 §7 with one structural correction: ExecuTorch is **a multi-backend runtime**, not "the CPU path". The portable IR (ATen + ExecuTorch custom ops) is one layer; backend-delegated subgraphs (`executorch.exir.delegate.lowered_module`) are another. The CPU/KleidiAI path described above is the *CPU delegate*. Other delegates (XNNPACK CPU, Vulkan GPU, MPS Apple, CoreML iOS, QNN Qualcomm, MTK NeuroPilot) lower to entirely different op sets.

**Attention fusion granularity.** "SDPA + KV in one custom op" — `sdpa_with_kv_cache` is the single ExecuTorch custom op that does in-place KV update *and* SDPA, based on FlashAttention 2 for CPU.

**Doc URLs (pinned).**

* `https://github.com/ARM-software/kleidiai/blob/main/docs/matmul_qsi4cx/README.md`
* `https://docs.pytorch.org/executorch/stable/llm/export-llm.html`
* `https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/executorch-1-0-is-here-and-with-sme2-optimizations-through-kleidiai`

### 2.8 Huawei MindIE / Ascend ATB (CANN 8.x, MindIE 2.x)

Identical to v1 §8 with three corrections:

1. The `ATB` repo went open-source per the Sept 2025 Huawei announcement (https://www.huawei.com/en/news/2025/9/hc-shengten-opensource), so the v1 reliance on CSDN/Zhihu/Tencent Cloud tertiary sources can be replaced with primary citations in a future revision. We acknowledge in v2 that primary verification is *now possible* but has not been completed.
2. `FlashAttentionOperation` and `PagedAttentionOperation` are confirmed by the now-public vllm-ascend integration tree (`https://github.com/vllm-project/vllm-ascend`); the `split_qkv_rmsnorm_rope` fused operator mentioned there shows a degree of fusion beyond what v1 reported.
3. KV cache: HBM-resident, block-table for `PagedAttentionOperation`; block size 16 or 128 depending on chip generation.

### 2.9 ONNX Runtime contrib (ORT 1.21 → 1.23, ONNX opset 23)

**Important documentation/implementation distinction.** v1 listed `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, `RMSNormalization`, `SparseAttention`, `QAttention`, `QOrderedAttention` as ORT contrib ops without distinguishing "documented in `ContribOperators.md`" from "implemented in `contrib_ops/cpu/*.cc`". Direct fetch of `https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/ContribOperators.md` (2026-06-04) clarifies:

| Op | Documented in ContribOperators.md? | Implemented in C++? | Notes |
|---|---|---|---|
| `SimplifiedLayerNormalization` | **NO** | **YES** | `contrib_ops/cpu/layer_norm.cc`. Used by onnxruntime-genai. |
| `SkipSimplifiedLayerNormalization` | **YES** | **YES** | Residual+RMSNorm fused. |
| `RMSNormalization` | **NO** (in contrib) | Shipping as **ONNX opset-23 standard op** | PR #23560; replaces `SimplifiedLayerNormalization` going forward. |
| `GroupQueryAttention` | **YES** | YES | |
| `MultiHeadAttention` | **YES** | YES | |
| `PagedAttention` | **YES** | YES | |
| `RotaryEmbedding` | **YES** | YES | |
| `GemmaRotaryEmbedding` | **YES** | YES | |
| `MatMulNBits` | **YES** | YES | INT4/INT2 grouped W4A16. |
| `MatMulBnb4` | YES | YES | BnB NF4/FP4. |
| `GemmFloat8` | **YES** | YES | FP8 E4M3/E5M2. |
| `BeamSearch` / `GreedySearch` / `Sampling` | **YES** | YES | Subgraph-encapsulating autoregressive ops. |
| `MoE` | **YES** | YES | First-class MoE op. |
| `SparseAttention` | **YES** | YES | Block-sparse mask. |
| `BiasSplitGelu` / `BiasGelu` / `FastGelu` / `QuickGelu` | YES | YES | |
| `DecoderMaskedMultiHeadAttention` / `DecoderMaskedSelfAttention` | YES | YES | Decode-path attention with beam-search cache indirection. |
| `PackedAttention` / `PackedMultiHeadAttention` | YES | YES | Variable-length packing. |
| `QAttention` / `QOrderedAttention` | `[unverified]` | YES (C++) | May be C++ only / undocumented. |
| `EmbedLayerNormalization` | YES | YES | |
| `GatherBlockQuantized` | YES | YES | Gather + dequant for embedding tables. |
| `LongformerAttention` / `LinearAttention` | YES | YES | |
| `WhisperBeamSearch` / `NGramRepeatBlock` | YES | YES | |
| `GroupNorm` / `SkipGroupNorm` | YES | YES | (mostly for diffusion) |

**Implication for synthesis.** Where v1 cell put `SimplifiedLayerNormalization` for the ORT "RMSNorm" row, v2 puts `SimplifiedLayerNormalization` (C++) | `RMSNormalization` (opset 23) to be explicit about which ORT path the runtime takes.

**Doc URLs (pinned).**

* `https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md` (main, 2026-06-04)
* `https://github.com/microsoft/onnxruntime/pull/23560` (RMSNormalization opset-23 op)
* `https://github.com/microsoft/onnxruntime/issues/21925` (SimplifiedLayerNormalization / RMSNorm history)

### 2.10 ggml (master, 2026-Q1) — newly added P0

**Identity.** ggml is the tensor library underlying llama.cpp, Ollama, LM Studio, Jan, KoboldCpp, whisper.cpp, and (transitively) probably half of all on-device LLM inference today. It is a single-header C library with explicit ops in a `GGML_OP_*` enum. Pin: ggml master as of 2026-Q1 (via `ggml-org/llama.cpp`).

**Ops used to express one decoder block.** Verified by direct fetch of `https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h`:

| Logical step | ggml op |
|---|---|
| input RMSNorm | `GGML_OP_RMS_NORM` |
| QKV proj | `GGML_OP_MUL_MAT` |
| RoPE | `GGML_OP_ROPE` |
| KV write/read | tensor views into a contiguous KV buffer: `ggml_view_2d` / `ggml_set_2d` over the per-layer KV tensor allocated from `ggml_context` memory pool |
| SDPA | `GGML_OP_FLASH_ATTN_EXT` (FlashAttention 2 in CPU/CUDA/Metal/Vulkan/SYCL backends); when this op is unsupported the runtime falls back to `MUL_MAT → ADD(mask) → SOFT_MAX → MUL_MAT` |
| Output proj | `GGML_OP_MUL_MAT` |
| Residual | `GGML_OP_ADD` |
| FFN | `GGML_OP_MUL_MAT` × 3 |
| SiLU | `GGML_OP_SILU` *(name confirmed via DeepWiki coverage of llama.cpp; the public ggml.h listing in our fetch surfaced `GGML_OP_*` macros for the major ops above. `SILU` may live in the `GGML_UNARY_OP_*` sub-enum rather than the top-level `GGML_OP_*` enum.)* |
| gate*up | `GGML_OP_MUL` |
| Final norm | `GGML_OP_RMS_NORM` |
| LM head | `GGML_OP_MUL_MAT` |
| MoE | **`GGML_OP_MUL_MAT_ID`** — indexed matmul for expert dispatch (router-output indices select experts) |
| Embedding lookup | `GGML_OP_GET_ROWS` |
| Sampling | `GGML_OP_ARGSORT`, `GGML_OP_TOP_K` |
| Mask | composed via `GGML_OP_ADD` of a precomputed mask tensor; flash-attn-ext also accepts a mask input |
| Quant matmul | `GGML_OP_MUL_MAT` handles per-block-quantized inputs (Q4_K, Q5_K, Q6_K, IQ4_XS, ...) intrinsically — there is no separate dequant op; the matmul kernel dequant-on-the-fly per block |

**Attention fusion granularity.** "SDPA fused via `ggml_flash_attn_ext`" (when supported); otherwise primitive decomposition. The `flash_attn_ext` op accepts `q, k, v, mask, scale, max_bias, logit_softcap` as inputs — supports causal flag, sliding-window-via-mask, and per-token softcap (Gemma 2 style).

**KV cache representation.** **Contiguous tensors in a `ggml_context` memory pool** — distinct from past/present ports, stateful Variables, paged block tables, and pure Python state. Contiguous allocation is required for many backends; consequence: ggml pre-allocates max-context KV up front. Recent work (#21961) proposes a paged KV cache.

**Quantization.** ggml is where the rich GGUF k-quant matrix lives (Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XS, IQ4_NL, IQ4_XS, …) — covered in detail in `04-quantization.v2.md`. For op-set purposes, the relevant fact is that *all* quant schemes lower to `GGML_OP_MUL_MAT` with type-specialised kernels — there is no separate quant or dequant op.

**Surprising / proprietary.**

* `MUL_MAT_ID` for MoE — a *gather-then-grouped-matmul* op fused into a single kernel call. Distinct enough from generic gather + matmul that it deserves an MoE row in the synthesis table.
* KV is a tensor view, not state — the cleanest "buffer = cache" model in the survey, with no IR-level state abstraction at all.
* Quant is invisible to the op graph — the `MUL_MAT` op consumes quantized tensors directly. This is opposite to TRT-LLM's explicit `quantizePerTokenPlugin` / `quantizeTensorPlugin` style.

**Doc URLs (pinned).**

* `https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h`
* `https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp`
* `https://deepwiki.com/ggml-org/llama.cpp/8.2-flash-attention-and-optimizations`
* `https://deepwiki.com/ggml-org/llama.cpp/3.6-memory-management-and-kv-cache`
* `https://github.com/ggml-org/llama.cpp/discussions/21961` (paged KV proposal)

### 2.11 AWS Neuron / NKI (Neuron SDK 2.29, NKI 0.3.0) — newly added P0

**Identity.** AWS Neuron is the SDK for AWS Trainium 1/2/3 and Inferentia 2/3. The NeuronCore architecture is custom (not GPU-like); the kernel programming layer is **NKI (Neuron Kernel Interface)**, a Python DSL with tensor-core / vector / scalar engine primitives. The LLM serving entry points are **`transformers-neuronx` (now NxD Inference)** for high-level model parallel inference and **vLLM-Neuron** for continuous batching.

**Version pinning.** Neuron SDK 2.29.0 brought NKI 0.3.0 out of beta. The NKI Library at re:Invent 2025 (AIM414) includes 7 new experimental kernels: a Transformer TKG megakernel, plus improvements to existing attention, MLP, and MoE kernels. Pin: Neuron SDK 2.29.0, NKI 0.3.0 (`https://aws-neuron.github.io/nki-samples/`).

**Ops used to express one decoder block.**

| Logical step | Neuron op / NKI kernel |
|---|---|
| input RMSNorm | NxD Inference `RMSNorm` primitive; NKI-level scalar+vector decomposition |
| QKV proj | `nki.linear` / Neuron tensor-core matmul; in NxD Inference the QKV is often a fused triple-matmul |
| RoPE | NxD `RotaryPositionalEmbedding`; NKI-level cos/sin tabulated |
| KV cache | `transformers-neuronx` `KVCacheManager` — pre-allocated per-layer tensors in Neuron HBM; **cache reuse** semantics (not recompute) |
| SDPA | **`nki_flash_fwd`** kernel (FlashAttention adapted to Neuron's online-softmax + tensor-core layout); for prefill `nki.kernels.attention.flash_fwd` and for decode a paged variant |
| Output proj | `nki.linear` |
| Residual | `nki.add` |
| FFN | `nki.linear` × 3 |
| SiLU | `nki.silu` (or composed sigmoid + multiply) |
| gate*up | `nki.multiply` |
| Final norm | `RMSNorm` |
| LM head | `nki.linear` |
| MoE | NKI MoE megakernel (improved in NKI 0.3.0 per AIM414); top-k routing + expert dispatch fused |
| All-reduce (TP) | Neuron Collective Communication Library (NCCL-equivalent for Trainium/Inferentia) |
| Sampling | host-side or NKI top-k / top-p kernel |

**Attention fusion granularity.** "Custom NKI megakernel": SDPA (and increasingly the whole transformer TKG block) is one NKI-authored kernel. By replacing the default attention with NKI-based attention, teams report **6-8× LLM inference speedup**. This is the most aggressive *user-authored* kernel fusion in the survey — closer to Pallas (TPU) than to TRT-LLM (plugin) in its programming model.

**KV cache representation.** Pre-allocated HBM tensors per layer, managed by `transformers-neuronx`. With vLLM-Neuron, paged-style block tables are layered on top.

**Quantization.** Neuron supports FP16/BF16, FP8 (E4M3) on Trainium 2/3, INT8. Weight-only INT4 via Neuron's quant SDK. Mixed-precision: Trainium 2's tensor-engine FP8 path is the production target for late-2025/2026 inference.

**Doc URLs (pinned).**

* `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.kernels.html`
* `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/transformers-neuronx/generative-llm-inference-with-neuron.html`
* `https://aws-neuron.github.io/nki-samples/`
* `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html`
* `https://www.antstack.com/talks/reinvent25/aws-reinvent-2025---performance-engineering-on-neuron-how-to-optimize-your-llm-with-nki-aim414/`

### 2.12 Google TPU — XLA HLO + Pallas / Mosaic — newly added P0

**Identity.** TPU LLM inference and training go through **XLA** (the JAX/JIT compiler producing HLO IR) plus **Pallas** (a JAX-extension DSL for authoring TPU kernels, compiled by **Mosaic** to MLIR / VMEM-aware code). The production LLM stacks are **MaxText** (Google's reference training/serving stack), **vLLM-TPU** (the unified vLLM backend supporting PyTorch and JAX on TPU, blog Oct 2025), and **JetStream** (LLM-serving runtime).

**Version pinning.** TPU v4 / v5e / v5p / v6 (Trillium); MaxText releases through 2025; vLLM-TPU launched Oct 2025; Pallas API stabilising in JAX 0.4.x → 0.5.x. **Ragged Paged Attention v2** (RPA v2) shipped in mid-2025 to support chunked prefill and prefix caching on TPU.

**Ops used to express one decoder block.**

| Logical step | TPU op (HLO + Pallas custom-call) |
|---|---|
| input RMSNorm | HLO `reduce + multiply + rsqrt + multiply` decomposition; in Pallas, a fused RMSNorm kernel |
| QKV proj | HLO `dot_general`; with TP, `dot_general` + `all-reduce` |
| RoPE | HLO `multiply + add` decomposition over the sin/cos tabulated tensors |
| KV cache | HLO `dynamic-update-slice` writes into a pre-allocated KV buffer that is threaded through the program as parameter/tuple I/O; this is a **distinct KV representation** ("XLA stateful threading") from CoreML state, OpenVINO Variables, or ORT past/present |
| SDPA | **Pallas Ragged Paged Attention (RPA) kernel** encapsulated as an HLO **`custom-call`** with private metadata; for training, Pallas Flash/Splash attention; for inference, RPA v2 with paged KV |
| Output proj | HLO `dot_general` |
| Residual | HLO `add` |
| FFN | HLO `dot_general` × 3 |
| SiLU | HLO `multiply(sigmoid(x), x)` |
| gate*up | HLO `multiply` |
| Final norm | RMSNorm decomposition |
| LM head | HLO `dot_general` |
| MoE | MaxText MoE: HLO `top-k` + `dynamic-update-slice` + per-expert `dot_general`; emerging Pallas MoE megakernel |
| All-reduce (TP) | HLO `all-reduce` (true HLO op, not a custom-call) |
| Sampling | host-side, or HLO `top-k` + `random` |
| Quant-dequant | HLO `convert` + `multiply` (group-quant decompression) |

**Attention fusion granularity.** "Compiler-fused, no named op" — the SDPA implementation is a Pallas kernel that XLA invokes as `custom-call`. The HLO IR sees an opaque op with vendor-private metadata; the Pallas source is the source-of-truth, compiled by Mosaic to MLIR.

**KV cache representation.** **HLO stateful threading**: KV is threaded as a tuple input/output of the compiled program; `dynamic-update-slice` writes new K/V into the buffer. Distinct from CoreML state and OpenVINO Variables because there is no graph-level "state" construct — XLA programs are pure functions, and the KV is part of the function signature.

**Quantization.** BF16 default; INT8 via AQT (Accurate Quantized Training); FP8 emerging on Trillium; W4A16 via Pallas custom dequant kernels (HLO `convert` + `multiply`).

**Doc URLs (pinned).**

* `https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html`
* `https://blog.vllm.ai/2025/10/16/vllm-tpu.html`
* `https://docs.pytorch.org/xla/master/features/pallas.html`
* `https://arxiv.org/html/2604.15464` (Ragged Paged Attention paper)
* `https://patricktoulme.substack.com/p/frontier-pretraining-infrastructure` (GPT-OSS on TPU with MaxText)

### 2.13 Microsoft DirectML + Windows ML — newly added P0

**Identity.** DirectML is a DirectX 12 library exposing hardware-accelerated ML primitives across all DX12-capable GPUs (AMD, Intel, NVIDIA, Qualcomm Adreno). It ships with Windows 10/11. **Windows ML** is the 2025 evolution layered on top, powered by ORT with a vendor-execution-provider contract — AMD, Intel, NVIDIA, Qualcomm all ship their EPs (the AMD NPU EP, the QNN EP, the NV EP) under the Windows ML umbrella. **DirectML is now in sustained engineering**; the new LLM-on-NPU work has migrated to Windows ML with onnxruntime-genai (OGA) for the LLM-specific overlay.

**Version pinning.** DirectML 1.15.4 (last functional release on the DML library proper, late 2024); Windows ML 1.x as of Ignite 2025; PhiSilica driver-based optimizations on AMD in 2026 (per AMD Build 2026 blog). The NPU EP (Windows ML's umbrella for QNN/AMD-XDNA/Intel-NPU EPs) reports up to **1.5× TTFT improvement** and **3.5× sustained token generation** vs. earlier baselines per Ignite 2025 numbers.

**Ops used to express one decoder block.** DirectML defines its op set as `DML_OPERATOR_*` (e.g., `DML_OPERATOR_MULTIHEAD_ATTENTION`, `DML_OPERATOR_RMS_NORM`, `DML_OPERATOR_RESAMPLE`, `DML_OPERATOR_GEMM`, etc.).

| Logical step | DML op |
|---|---|
| input RMSNorm | `DML_OPERATOR_RMS_NORMALIZATION` |
| QKV proj | `DML_OPERATOR_GEMM` |
| RoPE | `DML_OPERATOR_GEMM` + element-wise pattern; recent DML versions have `DML_OPERATOR_ROTARY_EMBEDDING` |
| KV cache | DirectX 12 stateful resource (persistent UAV in the D3D resource heap) — distinct from CoreML state because the storage is D3D-level, not graph-level |
| SDPA | `DML_OPERATOR_MULTIHEAD_ATTENTION` — DML-level fused MHA |
| Output proj | `DML_OPERATOR_GEMM` |
| Residual | `DML_OPERATOR_ELEMENT_WISE_ADD` |
| FFN | `DML_OPERATOR_GEMM` × 3 |
| SiLU | `DML_OPERATOR_ACTIVATION_*` |
| gate*up | `DML_OPERATOR_ELEMENT_WISE_MULTIPLY` |
| Final norm | `DML_OPERATOR_RMS_NORMALIZATION` |
| LM head | `DML_OPERATOR_GEMM` |
| Quant-dequant | `DML_OPERATOR_QUANTIZE`, `DML_OPERATOR_DEQUANTIZE` |

At the **Windows ML** layer, the user authors against ORT contrib ops (see §2.9) and the EP lowers them to either DML ops (for GPU) or vendor-NPU ops (for AMD-XDNA / QNN / Intel-NPU). This is the most "multi-vendor through one API" runtime in the survey.

**Attention fusion granularity.** "Fused MHA op at DML level" — `DML_OPERATOR_MULTIHEAD_ATTENTION`. Through Windows ML / ORT-genai, lowering can produce either DML MHA or vendor-EP fused attention.

**KV cache representation.** **DirectX 12 stateful resource** (persistent UAV) — eleventh distinct representation in the survey, separate from CoreML state (which is MIL-level) and OpenVINO Variables (which is IR-level).

**Doc URLs (pinned).**

* `https://learn.microsoft.com/en-us/windows/ai/directml/dml`
* `https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html`
* `https://blogs.windows.com/windowsdeveloper/2025/05/19/introducing-windows-ml-the-future-of-machine-learning-development-on-windows/`
* `https://www.amd.com/en/blogs/2026/advancing-windows-ml-acceleration-with-amd-at-microsoft-build-2026.html`

### 2.14 TVM-Relax / MLC-LLM — newly added P1

**Identity.** Apache TVM Unity's **Relax** dialect is a graph-level IR with Python-first transformations; MLC-LLM is the LLM-specific compilation pipeline built on Relax + TIR. Ships across browser (WebGPU), Android (Vulkan), iOS, desktop. Pin: TVM 0.20 / 0.21 (Unity, current branch), MLC-LLM late-2025 release.

**Ops used.** Relax LLM operator set:

| Logical step | Relax op |
|---|---|
| input RMSNorm | `relax.nn.rms_norm` |
| QKV proj | `relax.matmul` |
| RoPE | composed from `relax.split / relax.multiply / relax.add` ; an internal pass folds for some backends |
| KV cache | `relax.vm.AttentionKVCache` — a Relax-level KV-cache primitive with its own object-method protocol; signature evolved (issue #2162 mentions a 19-vs-18-arg incident) |
| SDPA | `relax.nn.attention` / `relax.nn.attention_var_len` for variable-length |
| FFN | `relax.matmul` × 3 |
| MoE | composed `relax.softmax → relax.topk → relax.gather → relax.matmul` per expert; recent work on fused MoE ops |
| Quant-dequant | dispatched to backend libraries (TIR kernels) |

**Attention fusion granularity.** "Compiler-fused via Relax pass pipeline" — Phase-1 Relax passes do op fusion + kernel dispatch; phase-2 lowers to TIR; phase-3 emits backend kernels (CUDA / Metal / Vulkan / WebGPU).

**KV cache.** `relax.vm.AttentionKVCache` — a runtime VM object distinct from the IR ops, methods include append, fetch, rotate. Counted in §5 as the "Relax VM cache object".

**Doc URLs.**

* `https://tvm.apache.org/docs/how_to/tutorials/optimize_llm.html`
* `https://mlc.ai/docs/reference/api/relax/transform.html`
* `https://llm.mlc.ai/docs/install/tvm.html`
* `https://github.com/mlc-ai/relax`

### 2.15 TensorFlow Lite / LiteRT GenAI / LiteRT-LM — newly added P1

**Identity.** Google's universal on-device framework. **LiteRT** is the new name for TFLite; **LiteRT-LM** is the LLM-specific orchestration layer. Supports Gemma 3 / 3n, EmbeddingGemma, FunctionGemma, Llama, Phi. Conversion from PyTorch, TF, JAX → `.tflite` / `.litertlm`.

**Pin.** LiteRT-LM through 2025; Multi-Token Prediction (MTP) drafter support for Gemma 4 family; up to 2.2× MTP speedup; 1.4× faster GPU than TFLite; NPU acceleration via Qualcomm/MediaTek/AMD delegates.

**Ops.** TFLite/LiteRT op set (built-in ops + `TFLite_Detection_PostProcess`-style custom ops) plus **ai-edge-torch** composable transformer building blocks. LLM ops: `TFLite SDPA`, `RMS_NORM` (custom), gather-style embedding lookup, INT4-grouped GEMM via XNNPACK. KV cache managed by LiteRT-LM (not at op level — handled by the orchestration layer, similar to MLX).

**Attention fusion granularity.** "Orchestration-managed SDPA" — LiteRT-LM dispatches to one of several backend kernels (CPU/XNNPACK, GPU-OpenCL, NPU via delegate). The SDPA op is a single TFLite/LiteRT custom op when targeting GPU; on CPU it is the XNNPACK fused-MHA kernel.

**Doc URLs.**

* `https://ai.google.dev/edge/litert/overview`
* `https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/`
* `https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/`
* `https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference`

### 2.16 OneDNN Graph SDPA — newly added P1

**Identity.** Intel's CPU/GPU primitive library used **underneath** OpenVINO, PyTorch CPU Inductor, and ORT CPU EP. Its **Graph API** exposes the SDPA fusion pattern at oneDNN v3.8+, with compressed-KV (INT4/INT8) support added in v3.8.1 → v3.11.0 → v3.13.0.

**Pin.** oneDNN v3.13.0 (current docs), Intel oneAPI 2025.2 documentation reflects the same.

**Ops.** The SDPA fusion pattern is defined as a directed-acyclic graph over oneDNN primitives:

| Step | oneDNN primitive |
|---|---|
| QKV | `matmul` |
| Mask add | `binary_add` |
| Softmax | `softmax` |
| Output | `matmul` |
| KV decompression (compressed-KV variant) | `dequantize` ahead of matmul |

The SDPA *pattern* is what the user declares; oneDNN selects an optimal fused implementation. Three compressed-KV variants are recognised: (a) both K and V compressed; (b) K float, V compressed; (c) K compressed, V float. INT4 (u4/s4) and INT8 (u8/s8) for compressed; F32/F16/BF16 for floats. Grouped quantization required for accuracy with INT4.

**Hardware acceleration.** Optimised path for compressed SDPA with f16 compute on Intel Graphics Products with Intel XMX (Xe Matrix Extensions).

**Why call it out separately from OpenVINO?** Because PyTorch CPU Inductor and ORT CPU EP both call oneDNN directly — neither goes through OpenVINO. The same physical kernel underlies multiple "logical runtimes" in the survey.

**Doc URLs.**

* `https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa.html` (v3.13 current)
* `https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa_compressed_kv.html`
* `https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-2/scaled-dot-product-attention-sdpa.html`

### 2.17 Meta MTIA v1 / v2 — newly added P1

**Identity.** Meta's training/inference accelerator. v1 (2024) inference-only; v2 (2025) training-capable. Software stack: **PyTorch 2 + TorchInductor + Triton-MTIA**, with eager-mode support via the PyTorch runtime. Industry positioning: the most "PyTorch-eager-aligned" stack — kernels are user-authored Triton or generated by TorchInductor.

**Pin.** MTIA v1 (2024); MTIA v2 (early 2025); four MTIA chips in two years cadence per Meta's late-2025 disclosures; MTIA is "built natively on industry-standard software and hardware ecosystems — PyTorch, vLLM, Triton, and OCP".

**Ops used.** No new IR — the IR *is* PyTorch + Triton:

| Logical step | MTIA op |
|---|---|
| input RMSNorm | `aten.rms_norm` lowered to Triton-MTIA kernel |
| QKV proj | `aten.linear` → Triton-MTIA GEMM |
| RoPE | composed `aten.split / mul / cat` |
| SDPA | `torch.nn.functional.scaled_dot_product_attention` → Triton-MTIA fused kernel |
| KV cache | PyTorch tensor managed by `transformers` / vLLM-MTIA |
| FFN | `aten.linear` × 3 |
| MoE | PyTorch MoE pattern → fused Triton-MTIA kernel for expert dispatch |
| Sampling | host-side Python |

**Attention fusion granularity.** "Triton kernel as op" — SDPA is one Triton kernel; the IR sees an `aten.scaled_dot_product_attention` call.

**KV cache.** PyTorch tensor (similar in spirit to MLX pure-Python state, but compiled). The runtime is vLLM-MTIA, which adds PagedAttention block tables on top.

**Doc URLs.**

* `https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/`
* `https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/`
* `https://aisystemcodesign.github.io/papers/MTIA-ISCA25.pdf`

### 2.18 Modular MAX 25.6 / Mojo 1.0 — newly added P1

**Identity.** Modular's MAX engine is a unified compute layer spanning laptops to datacenter GPUs; Mojo is the kernel language. MAX 25.6 (late 2025) ships throughput on NVIDIA Blackwell B200 and AMD MI355X; Mojo 1.0 planned for H1 2026.

**Ops used.** MAX graph ops live under `max.graph.ops.*`. LLM-relevant: `max.graph.ops.linear`, `max.graph.ops.rms_norm`, `max.graph.ops.attention`, `max.graph.ops.rope`. Kernel implementations are Mojo code (e.g., the Blackwell SnapMLA kernel for MLA decode; hardware-accelerated conv2d with TMA im2col; fused BF16/FP8 matmul epilogues; FP8 MMA support for MLA prefill with blockwise scaling).

**Attention fusion granularity.** "Mojo megakernel" — attention is a single Mojo kernel, dispatched by MAX graph ops. The MLA-specific SnapMLA kernel on Blackwell is a recent (2025-Q4) addition.

**KV cache.** MAX-managed buffers; user-facing API hides the representation behind `max.graph.ops.attention`.

**Quantization.** FP8 MMA on Blackwell; mixed BF16/FP8 epilogues; W4A16 via Mojo dequant kernels.

**Doc URLs.**

* `https://docs.modular.com/max/changelog/`
* `https://www.modular.com/blog/modular-at-nvidia-gtc-2026-max-on-blackwell-mojo-kernel-porting-and-deepseek-v3-on-b200`
* `https://github.com/modular/modular`

### 2.19 Closed-vocabulary IHVs (FastFlowLM, Hailo-10H, Groq LPU, Cerebras WSE, SambaNova SN40L)

A recurring pattern: these five runtimes expose only model-level APIs (token-in, logits-out) and conceal the kernel-level op vocabulary entirely.

* **FastFlowLM** (AMD XDNA2) — IRON / AIE-MLIR microcode kernels; KV state on-chip in column SRAM.
* **Hailo-10H** — supports qwen2.5 / deepseek-r1-distill-qwen 1.5B family; runtime hidden.
* **Groq LPU / TSP** — GroqFlow MLIR compiler lowers to deterministic dataflow ISA; no public LLM op vocab.
* **Cerebras Wafer-Scale CSL** — WaferLLM (arXiv 2502.04563) shows MeshGEMM/MeshGEMV/on-wafer KV; CSL is the kernel layer.
* **SambaNova SN40L** — Reconfigurable Dataflow + SambaFlow; PEF (Pattern Execution Format) is the IR; *graph-of-spatial-streams*, not graph-of-fused-ops.

The minimal-API implication: these IHVs *exist* and *will be targets*, but at the kernel level we cannot model their ops. The `llm-layers` API must remain expressible at the "abstract layer" level so that a closed-vocabulary IHV can ingest the layer description and synthesise its own kernels.

### 2.20 PyTorch-native attention-kernel layers (vLLM custom ops, SGLang RadixAttention, cuDNN fused SDPA, FlashInfer)

Not IHV-specific but shape the consumer side of how op-sets are exercised:

* **vLLM custom ops** — `vllm.ops.paged_attention_v1`, `vllm.ops.paged_attention_v2` (v2 adds partition-and-reduce for long sequences), `vllm.ops.rms_norm`, `vllm.ops.fused_add_rms_norm`, `vllm.ops.silu_and_mul`. vLLM V1 (January 2025) is a major architectural redesign; the Triton attention backend (vLLM Blog, March 2026) is native and matches FlashAttention 3 performance on H100 at ~800 LoC vs FA3's ~70k LoC.
* **SGLang RadixAttention** — KV-cache *reuse* as a first-class primitive (radix-tree of cached prefixes with reference counting).
* **cuDNN fused FlashAttention2/3 ops** — `cudnnSDPAForward`, FP8 variants on Hopper/Blackwell. PyTorch's `torch.nn.functional.scaled_dot_product_attention` dispatches into cuDNN backends.
* **FlashInfer** — kernel library used by vLLM and TGI for paged attention with high-performance prefill/decode kernels.

These layers don't define a new IR but they *do* define operational semantics (RadixAttention, PA-v2 partition-and-reduce) that must be representable in the minimal API.

---

## 3. Cross-runtime synthesis table (18 rows × 18 columns)

Cell convention: op name in that runtime, or `fused into X` if the runtime swallows the logical step into a larger op, or `n/a` if not applicable, or `composed` if no native op exists (must be expressed via primitives).

**Columns abbreviated:** ORT (ONNX RT contrib), OV (OpenVINO 2025.4), QNN (QAIRT), MGX (MIGraphX), TRT (TRT-LLM 1.x), CML (Core ML iOS18), MLX (0.31), ETx (ExecuTorch 1.0), MdI (MindIE/ATB), ggm (ggml master), NKI (AWS NKI 0.3), TPU (XLA+Pallas), DML (DirectML/WinML), MLC (TVM-Relax/MLC-LLM), LRT (LiteRT GenAI), oDNN (oneDNN Graph), MTIA, MAX (Modular).

| Logical op | ORT | OV | QNN | MGX | TRT | CML | MLX | ETx | MdI | ggm | NKI | TPU | DML | MLC | LRT | oDNN | MTIA | MAX |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **RMSNorm** | `SimplifiedLayerNormalization` (C++) / opset-23 `RMSNormalization` | `RMS` (post-fusion); composite in serialized IR | `QNN_OP_RMS_NORM` | `RMSNormalization` (ONNX) / `SimplifiedLayerNorm` | `rmsnormQuantizationPlugin` | composed | `mlx.fast.rms_norm` | custom `rms_norm` | `RmsNormOperation` | `GGML_OP_RMS_NORM` | NxD RMSNorm | composed HLO; Pallas fused | `DML_OPERATOR_RMS_NORMALIZATION` | `relax.nn.rms_norm` | LiteRT custom | `softmax`-adjacent in pattern | `aten.rms_norm` → Triton | `max.graph.ops.rms_norm` |
| **QKV proj** | `MatMul` / `MatMulNBits` | `MatMul` | `QNN_OP_MAT_MUL` / `FULLY_CONNECTED` | `MatMul` / `MatMulNBits` | `GemmPlugin` (+quant variants) | `linear` | `matmul` / `quantized_matmul` | `linear_int4` / xnnpack | `LinearOperation` | `GGML_OP_MUL_MAT` | `nki.linear` | HLO `dot_general` | `DML_OPERATOR_GEMM` | `relax.matmul` | XNNPACK GEMM | oneDNN `matmul` | `aten.linear` | `max.graph.ops.linear` |
| **RoPE** | `RotaryEmbedding` / `GemmaRotaryEmbedding` | `RoPE` (fused) | `QNN_OP_ROTARY_EMBEDDING` | `RotaryEmbedding` | fused into `GptAttentionPlugin` | composed | `mlx.fast.rope` | composed | `RopeOperation` | `GGML_OP_ROPE` | NxD RoPE | HLO composition; Pallas fused | `DML_OPERATOR_ROTARY_EMBEDDING` | composed → fused pass | composed | n/a (caller) | composed | `max.graph.ops.rope` |
| **KV write** | inside `GroupQueryAttention` past→present | `Assign` (stateful) **or** `key_cache` input on `PagedAttentionExtension` | graph I/O tensor *(retracted: not a named op)* | inside `GroupQueryAttention` past/present | inside `GptAttentionPlugin` (paged or contiguous) | `coreml_update_state` | Python `KVCache` (no op) | inside `sdpa_with_kv_cache` | inside `PagedAttention` / `FlashAttention` | tensor view + `ggml_set_2d` into pool | `KVCacheManager` HBM tensor | HLO `dynamic-update-slice` | D3D12 stateful resource | `relax.vm.AttentionKVCache.append` | LiteRT-LM managed | inside `matmul` pattern (precomputed) | PyTorch tensor + vLLM-MTIA paged | MAX-managed buffer |
| **KV read** | `past_*` input | `ReadValue` / `key_cache` input | graph I/O tensor | past input | inside plugin | `read_state` | Python array | inside custom op | inside ATB op | tensor view | KV HBM tensor | tuple parameter | D3D12 SRV | `relax.vm.AttentionKVCache.fetch` | LiteRT-LM | precomputed inputs | PyTorch tensor | MAX-managed |
| **SDPA core** | `GroupQueryAttention` / `MultiHeadAttention` / `Attention` | `ScaledDotProductAttention` / `PagedAttentionExtension` | composed (`MatMul → ElementWiseAdd → Softmax → MatMul`); Genie fused `[unverified]` | `GroupQueryAttention` | fused in `GptAttentionPlugin` / trtllm-gen kernel | `scaled_dot_product_attention` (iOS18+) | `mlx.fast.scaled_dot_product_attention` | `sdpa_with_kv_cache` | `FlashAttention` / `PagedAttention` | `GGML_OP_FLASH_ATTN_EXT` | `nki.kernels.attention.flash_fwd` | Pallas RPA custom-call | `DML_OPERATOR_MULTIHEAD_ATTENTION` | `relax.nn.attention_var_len` | LiteRT SDPA | oneDNN Graph SDPA pattern | `torch.nn.functional.scaled_dot_product_attention` → Triton | Mojo SnapMLA / attention kernel |
| **Output proj** | `MatMul` | `MatMul` | `QNN_OP_MAT_MUL` | `MatMul` | `GemmPlugin` | `linear` | `matmul` | `linear` | `LinearOperation` | `GGML_OP_MUL_MAT` | `nki.linear` | `dot_general` | `DML_OPERATOR_GEMM` | `relax.matmul` | XNNPACK GEMM | oneDNN matmul | `aten.linear` | `max.graph.ops.linear` |
| **Attn residual** | `Add` (or fused in `SkipLayerNormalization`) | `Add` (often fused into next RMSNorm) | `QNN_OP_ELEMENT_WISE_ADD` | `Add` / `SkipSimplifiedLayerNormalization` | tensor `+` | `add` | `mx.add` | `aten.add` | `ElewiseAddOperation` | `GGML_OP_ADD` | `nki.add` | HLO `add` | `DML_OPERATOR_ELEMENT_WISE_ADD` | `relax.add` | TFLite `ADD` | `binary_add` | `aten.add` | `ops.add` |
| **FFN gate proj** | `MatMul` / `MatMulNBits` | `MatMul` | `QNN_OP_MAT_MUL` | `MatMul` | `GemmSwigluPlugin` (fused) | `linear` | `matmul` | `linear` | `SwiGluOperation` (fused) | `GGML_OP_MUL_MAT` | `nki.linear` | `dot_general` | `DML_OPERATOR_GEMM` | `relax.matmul` | XNNPACK | matmul | `aten.linear` | `max.graph.ops.linear` |
| **FFN up proj** | `MatMul` | `MatMul` | `QNN_OP_MAT_MUL` | `MatMul` | inside `GemmSwigluPlugin` | `linear` | `matmul` | `linear` | inside `SwiGluOperation` | `GGML_OP_MUL_MAT` | `nki.linear` | `dot_general` | `DML_OPERATOR_GEMM` | `relax.matmul` | XNNPACK | matmul | `aten.linear` | `max.graph.ops.linear` |
| **FFN down proj** | `MatMul` | `MatMul` | `QNN_OP_MAT_MUL` | `MatMul` | `GemmPlugin` | `linear` | `matmul` | `linear` | `LinearOperation` | `GGML_OP_MUL_MAT` | `nki.linear` | `dot_general` | `DML_OPERATOR_GEMM` | `relax.matmul` | XNNPACK | matmul | `aten.linear` | `max.graph.ops.linear` |
| **SiLU** | `Mul(Sigmoid(x), x)` / `BiasSplitGelu` for GeLU | `Swish` (opset4) | composed `Sigmoid + Mul` *(retracted: not a named op)* | `Sigmoid+Mul` / `BiasSplitGelu` | inside `GemmSwigluPlugin` | `silu` | `mlx.nn.silu` | `aten.silu` | inside `SwiGluOperation` | `GGML_UNARY_OP_SILU` | `nki.silu` | HLO `multiply(sigmoid(x), x)` | `DML_OPERATOR_ACTIVATION_*` | `relax.nn.silu` | LiteRT activation | post-op | `aten.silu` | `max.graph.ops.silu` |
| **gate*up** | `Mul` | `Multiply` | `QNN_OP_ELEMENT_WISE_MULTIPLY` | `Mul` | inside `GemmSwigluPlugin` | `mul` | `mx.multiply` | `aten.mul` | inside `SwiGluOperation` | `GGML_OP_MUL` | `nki.multiply` | HLO `multiply` | `DML_OPERATOR_ELEMENT_WISE_MULTIPLY` | `relax.multiply` | TFLite `MUL` | `binary_mul` | `aten.mul` | `ops.mul` |
| **Final RMSNorm** | `SimplifiedLayerNormalization` / opset-23 | `RMS` (post-fusion) | `QNN_OP_RMS_NORM` | `RMSNormalization` | `rmsnormQuantizationPlugin` | composed | `mlx.fast.rms_norm` | custom | `RmsNormOperation` | `GGML_OP_RMS_NORM` | `RMSNorm` | composed HLO | `DML_OPERATOR_RMS_NORMALIZATION` | `relax.nn.rms_norm` | LiteRT custom | n/a (pattern) | `aten.rms_norm` | `max.graph.ops.rms_norm` |
| **LM head** | `MatMul` | `MatMul` | `QNN_OP_MAT_MUL` | `MatMul` | `GemmPlugin` / `LowLatencyGemmPlugin` | `linear` | `matmul` / `quantized_matmul` | `linear` | `LinearOperation` | `GGML_OP_MUL_MAT` | `nki.linear` | `dot_general` | `DML_OPERATOR_GEMM` | `relax.matmul` | XNNPACK | matmul | `aten.linear` | `max.graph.ops.linear` |
| **MoE router / expert dispatch** | `MoE` (contrib) | `MoE` extension; composed for older versions | composed (Softmax+TopK+Gather) | `MoE` (via ORT contrib) | `MixtureOfExperts` plugin | composed | composed Python | n/a (decomposed via XNNPACK matmul-per-expert) | fused `MoE` ATB op | `GGML_OP_MUL_MAT_ID` | NKI MoE megakernel | HLO TopK+`dynamic-update-slice`+per-expert `dot_general`; Pallas MoE in flight | composed (via DML primitives) | composed; fused MoE in progress | composed | n/a | PyTorch MoE fused into Triton kernel | Mojo MoE kernel |
| **Embedding lookup** | `Gather` / `EmbedLayerNormalization` | `Gather` | `QNN_OP_GATHER` | `Gather` | `LookupPlugin` | `gather` | `mx.embedding` | `aten.embedding` | `GatherOperation` | `GGML_OP_GET_ROWS` | `nki.gather` | HLO `gather` | `DML_OPERATOR_GATHER` | `relax.gather` | TFLite `GATHER` | n/a | `aten.embedding` | `max.graph.ops.gather` |
| **All-reduce (TP)** | n/a (handled by ORT distributed wrapper) | `Allreduce` extension | n/a (single-NPU typical) | NCCL/RCCL via MIGraphX dist | `NcclPlugin`, `GemmAllReducePlugin` (fused GEMM+AR) | n/a (single-device) | n/a (single-device) | n/a (single-device) | `AllReduceOperation` | n/a (typically single-process) | Neuron Collective Comm | HLO `all-reduce` | n/a | composed; backend-dispatched | n/a | n/a (single-process) | NCCL via PyTorch dist | MAX collective |
| **Sampling / logits proc** | `Sampling` / `BeamSearch` / `GreedySearch` | external (GenAI pipeline) | external (Genie pipeline) | external | `cumsumLastDimPlugin`, `topkLastDimPlugin` | external | `mlx.random.categorical` | external | external | `GGML_OP_ARGSORT`, `GGML_OP_TOP_K` | external / NKI top-k | HLO `top-k` + `random` | external (Win ML genai) | external (MLC engine) | LiteRT-LM | n/a | external (PyTorch / vLLM) | MAX sampler |
| **Quant-dequant explicit op** | `QuantizeLinear` / `DequantizeLinear` / `DynamicQuantizeMatMul` | `FakeQuantize` / `FakeConvert` / `Convert(u4)+Multiply` | `QNN_OP_QUANTIZE` / `QNN_OP_DEQUANTIZE` | `QuantizeLinear` / `DequantizeLinear` | `quantizePerTokenPlugin`, `quantizeTensorPlugin`, `quantizeToFP4Plugin` | `constexpr_blockwise_shift_scale`, `constexpr_lut_to_dense`, `constexpr_affine_dequantize` | dequant inside `quantized_matmul` | dequant inside kernel | quant params on linear | dequant inside `MUL_MAT` (per-block) | NKI dequant prefix | HLO `convert + multiply` | `DML_OPERATOR_QUANTIZE` / `DEQUANTIZE` | dequant in TIR; per-backend | dequant in XNNPACK | `dequantize` primitive before matmul | dequant in Triton kernel | Mojo dequant |
| **Mask construction (causal/SWA/sink)** | `is_causal` flag + `attention_mask` input | `is_causal` attr; explicit mask input; **sink input on PA in 2025.4** | explicit mask tensor (`ElementWiseAdd`) | `is_causal` flag + mask | inside plugin (causal/sliding/sink flags) | `is_causal` + `attn_mask` | `mask="causal"` literal + sinks | inside `sdpa_with_kv_cache` | inside FlashAttention/PagedAttention | mask tensor passed to `FLASH_ATTN_EXT` | NKI causal flag | inside Pallas kernel metadata | MHA op flags | inside `relax.nn.attention` | inside SDPA | inside SDPA pattern | inside SDPA call | inside Mojo attention |

This is 21 rows × 18 cols = 378 cells. The §10 verification ledger lists which cells are header-verified vs. inferred.

---

## 4. Consensus, divergence, and the universality claim (revised)

### 4.1 What v1 said vs. what v2 finds

v1 surveyed 9 runtimes and claimed "≥7 of 9" universality for RMSNorm, MatMul, RoPE, SDPA, SiLU, residual-Add, and gate-Mul. v2 surveys 18 runtimes and the count must be re-cast as "≥ N of 18".

**Universal (op-class appears as a first-class fused op in ≥ 14 of 18 runtimes):**

* **MatMul / GEMM / Linear** — 18/18.
* **Element-wise Add (residual)** — 18/18.
* **Element-wise Multiply (gate*up)** — 18/18.
* **Embedding lookup (gather)** — 17/18 (only oneDNN does not need it, being a primitive layer not a stack).
* **SDPA as a single named or kernel-fused op** — 16/18. Holdouts: QAIRT (decomposed at QNN graph layer, with backend fusion below it — RETRACTED claim of native SDPA op) and oneDNN Graph (which expresses SDPA as a *pattern* over multiple primitives, not a named op).

**Near-universal (14–16 of 18):**

* **RMSNorm as a first-class op** — 15/18. Holdouts: CoreML (composed), TPU/HLO (composed), oneDNN (n/a).
* **RoPE as a first-class op** — 13/18. Holdouts: CoreML (composed), ExecuTorch (composed), TPU (composed in HLO, sometimes fused in Pallas), MLC-LLM (composed), LiteRT (composed). RoPE is *less* universal than v1 thought.
* **SiLU** — 14/18.

**Sub-universal (≥ 8 of 18 with significant heterogeneity):**

* **MoE router / expert dispatch** — 10/18 have a first-class MoE op (`MixtureOfExperts`, `MoE`, `MUL_MAT_ID`, ATB `MoE`, NKI MoE megakernel, Triton MoE on MTIA, Mojo MoE, Relax MoE). The rest compose.
* **All-reduce (TP collective)** — 8/18. Most absent on single-device runtimes (CoreML, MLX, ExecuTorch, ggml).

### 4.2 Implications for `llm-layers` minimal API

The v1 "9 ops" conclusion stands but with caveats:

1. **The nine universal ops** (`rms_norm`, `linear`, `rope`, `sdpa`, `residual_add`, `swiglu_ffn`, `silu`, `mul`, `lm_head`) cover ≥ 14/18 runtimes. RoPE is the weakest at 13/18 and must remain decomposable.
2. **MoE is no longer optional.** 10/18 runtimes have a first-class MoE op; the minimal API must add a `moe_layer(x, router_weights, expert_weights, top_k)` abstraction.
3. **All-reduce / TP** is a divergence axis that the minimal API must expose as an optional sublayer (not all runtimes ship it).
4. **Sampling / logits processing** does not belong in the layer-level API but the API must hand off a logits tensor in a documented form.
5. **Mask construction** divides into "causal flag" (one bool) vs. "explicit mask tensor" vs. "sink tensor". The minimal API needs a `MaskSpec` discriminated union with all three.

---

## 5. KV cache representation taxonomy (11 distinct shapes, revised from v1's 6)

| # | Representation | Runtimes |
|---|---|---|
| 1 | **past/present I/O ports** — past_key/past_value inputs, present_key/present_value outputs | ORT contrib (Attention, MHA, GQA), MIGraphX (via ORT importer) |
| 2 | **OpenVINO Variables (ReadValue/Assign graph ops)** — state as a *port* into the graph | OpenVINO stateful path |
| 3 | **Core ML MIL state (read_state / coreml_update_state)** — state as a *typed parameter* of the MLProgram, mutated in place | Core ML iOS 17+ |
| 4 | **Block tables (paged)** — physical block IDs map logical sequence positions to KV blocks | TRT-LLM `paged_kv_cache`, ORT `PagedAttention`, OpenVINO `PagedAttentionExtension`, MindIE `PagedAttention`, vLLM PA-v1, vLLM PA-v2 (partition-and-reduce variant), Pallas Ragged Paged Attention v2 (TPU) |
| 5 | **External buffer threaded into a fused custom op** | ExecuTorch `sdpa_with_kv_cache` |
| 6 | **Pure Python state (no IR)** | MLX `KVCache` class, MTIA via PyTorch eager |
| 7 | **XLA stateful threading** — KV is a tuple input/output of the compiled function; `dynamic-update-slice` writes | TPU XLA, JAX programs |
| 8 | **ggml contiguous tensor in `ggml_context` memory pool** — tensor-view-based read/write, no state abstraction | ggml / llama.cpp |
| 9 | **D3D12 stateful resource (persistent UAV in resource heap)** — host owns a D3D resource that the DML graph reads/writes | DirectML / Windows ML |
| 10 | **Relax VM cache object (`relax.vm.AttentionKVCache`)** — runtime VM object with append/fetch/rotate methods, distinct from IR ops | TVM-Relax / MLC-LLM |
| 11 | **Radix-tree shared-prefix KV (RadixAttention)** — block table with reference counting and prefix-merge across requests | SGLang RadixAttention |

In v1, representations 1–6 were listed (with QNN `KV_CACHE` as #6's "op-as-cache" — now retracted). v2 adds #7–#11 and clarifies that OpenVINO Variables (#2) and CoreML state (#3) differ in graph-level construct: OpenVINO has explicit `ReadValue`/`Assign` *graph nodes*; CoreML has `read_state`/`coreml_update_state` *MIL primitives*. They look similar but the IR shape is different.

The Genie split-graph pattern (prefill graph + decode graph with shared KV) is a *runtime convention*, not an IR-level KV representation, and so doesn't constitute a 12th category — it's an orthogonal axis (phase-specialised graphs) that any of the above 11 representations can be combined with.

---

## 6. Attention fusion granularity taxonomy (9 shapes, revised from v1's 5)

| # | Fusion shape | Description | Runtimes |
|---|---|---|---|
| 1 | **Single mega-op** owning QKV-split + RoPE + KV r/w + masked SDPA + output bias + quant scaling | one plugin / kernel does everything attention-related | TRT-LLM `GptAttentionPlugin` |
| 2 | **Colossal multi-input op (paged attention superset)** — single op with 20+ inputs covering paged KV + sliding window + ALiBi + sinks + cache rotation + sparsity | the OpenVINO PagedAttentionExtension shape | OpenVINO `PagedAttentionExtension` (2024.5+) |
| 3 | **Mega-op for SDPA only; QKV-proj and RoPE remain separate** | one fused SDPA, with KV r/w inside the op | MIGraphX/ORT `GroupQueryAttention`, MindIE `FlashAttention` |
| 4 | **SDPA fused, KV is external state** — SDPA is one op, KV r/w is a separate state-management op | typical of stateful runtimes | Core ML (`scaled_dot_product_attention` + `coreml_update_state`), OpenVINO stateful path, MLC-LLM (`relax.nn.attention` + Relax VM cache) |
| 5 | **SDPA + KV in one custom op** — single op fuses SDPA and in-place KV write | the ExecuTorch fusion shape | ExecuTorch `sdpa_with_kv_cache` |
| 6 | **Two SDPA ops by phase** — separate prefill and decode attention operators | MindIE `FlashAttention` (prefill) + `PagedAttention` (decode); Genie prefill/decode split graphs |
| 7 | **Eager kernel graph (Python or PyTorch as IR)** — SDPA is one Triton/Metal/CUDA kernel call, no IR; the runtime *is* the IR | MLX `fast.scaled_dot_product_attention`, ggml `GGML_OP_FLASH_ATTN_EXT`, MTIA Triton SDPA, vLLM custom ops |
| 8 | **Compiler-fused HLO custom-call** — SDPA is a `custom-call` HLO with vendor-private metadata; Pallas (or equivalent) authors the kernel | TPU XLA + Pallas RPA |
| 9 | **Primitive decomposition with backend kernel fusion** — IR sees MatMul/Add/Softmax/MatMul; the backend chooses to fuse them invisibly | QNN graph layer (backend HTP fuses below QNN API), oneDNN Graph SDPA pattern |

In v1, shapes #1, #2 (subsumed into #1), #3, #4, #5 were listed and #6 was implicit. v2 adds #7 (eager kernel graph) and #8 (HLO custom-call) and elevates #9 (primitive + backend fusion) to a category in its own right, motivated by the QNN retraction.

**Implication.** The minimal API must allow `attention()` to be lowered to *any* of these nine shapes. The parameter space:

* `qkv_layout`: pre-projected vs. fused QKV
* `rope_apply`: pre-call vs. in-attention vs. post-cache-rotation
* `kv_kind` ∈ {state, past_present, block_table, inline, external_buffer, xla_threaded, ggml_pool, d3d_resource, vm_object, radix_tree}
* `phase` ∈ {prefill, decode, unified}
* `mask_kind` ∈ {causal_flag, explicit_tensor, sliding_window, custom}
* `sinks` ∈ {none, attention_sinks_tensor}
* `target_fusion` ∈ {mega, paged_superset, sdpa_only, sdpa_plus_state, sdpa_plus_kv, by_phase, eager_kernel, hlo_custom_call, primitive}

---

## 7. Quantization support comparison

| Scheme | OV | QNN | MGX | TRT | CML | MLX | ETx | MdI | ORT | ggm | NKI | TPU | DML | MLC | LRT | oDNN | MTIA | MAX |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | yes | no | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| BF16 | yes | partial (HFP) | yes | yes | no | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| INT8 (W8A8) | yes | yes | yes | yes | yes (LUT/blockwise) | n/a | yes (KleidiAI W8A8) | yes | yes | yes (Q8_0) | yes | yes (AQT) | yes | yes | yes | yes | yes | yes |
| W4A16 grouped | yes (u4 group) | yes (int4) | yes (`MatMulNBits`) | yes (WeightOnlyGroupwise + GPTQ/AWQ) | yes (`constexpr_blockwise_shift_scale`) | yes (`quantized_matmul`) | yes (KleidiAI qsi4cxp) | yes | yes (`MatMulNBits`) | yes (Q4_K family) | yes | yes (Pallas dequant) | yes | yes | yes | yes (compressed-KV) | yes | yes |
| W4A8 | (via low-precision) | yes (a8w4 path) | yes (AITER) | yes (`qserveGemmPlugin` + W4A8 variants) | no | no | yes (KleidiAI W4A8 dynamic) | yes | partial | partial | yes | yes (Pallas) | partial | yes | partial | yes | yes | yes |
| FP8 (E4M3) | yes (2024.x+ `FakeConvert`) | no | yes (MI300+) | yes (Fp8Rowwise, FP8 MLA) | no | no | no | yes (HF8) | yes (`GemmFloat8`) | no | yes (Trn2+) | yes (Trillium) | partial | yes (Blackwell via MAX backend) | no | yes (XMX) | yes | yes (Blackwell B200, MI355X) |
| NVFP4 / MXFP4 | no | no | yes (MI350) | yes (`Fp4GemmPlugin`, NVFP4, MXFP4 variants) | no | no | no | no | no | no | partial | no | no | yes (via backend) | no | no | no | yes (Blackwell) |
| LUT palettization | no | n/a | no | no | yes (`constexpr_lut_to_dense`) | no | partial (LUT inside INT4 unpack) | no | no | partial (IQ-series) | no | no | no | no | no | no | no | no |
| NF4 (BitsAndBytes) | yes | no | partial | partial | no | no | no | no | yes (`MatMulBnb4`) | no | no | no | no | no | no | no | no | no |
| GGUF k-quant (Q_K family) | no | no | no | no | no | no | no | no | no | yes (Q2_K…Q8_K, IQ1_S…IQ4_XS) | no | no | no | no | no | no | no | no |

**Observations.**

* **W4A16 grouped is universal** across all 18 runtimes (17/18 explicit; CoreML uses its blockwise constexpr variant).
* **FP8 covers 11/18** with NVIDIA, Intel (XMX), AMD (MI300+), Huawei (HF8), AWS (Trn2/3), Google (Trillium), Microsoft (DML partial), Modular (MAX on Blackwell). Notable absences: QNN (no FP support, integer-only HTP), Core ML (no Apple FP8 hardware), MLX (same), ExecuTorch CPU (no FP8 ALU), ggml (FP16/BF16 + integer quant), LiteRT.
* **NVFP4/MXFP4 is still NVIDIA-and-AMD-only** in production; Modular MAX brings it via the NV/AMD backends.
* **GGUF k-quant** remains a ggml-island in the survey — no other runtime ingests Q4_K natively. (See `04-quantization.v2.md`.)

---

## 8. Evolution narrative (5 eras, 2022 → 2026)

### Era 1 — Primitive decomposition (2022 → mid-2023)

* Transformers were expressed as `MatMul + Add + Softmax + MatMul` for attention.
* RMSNorm was decomposed into `ReduceMean → Add(eps) → Sqrt → Divide → Multiply`.
* RoPE was decomposed into `Split → Multiply(cos/sin) → Concat`.
* KV cache was either re-computed each step or implemented as host-side concatenation of growing K/V tensors.
* No first-class LLM ops existed in any IHV runtime.

**Anchors:** Original TRT (pre-LLM); pre-iOS17 Core ML; pre-2023.3 OpenVINO; ggml early masters.

### Era 2 — Fused-attention-plugin era (mid-2023 → mid-2024)

* `GptAttentionPlugin` appears in TensorRT-LLM 0.5; fuses QKV-split, RoPE, KV r/w, masked SDPA into one plugin per Llama block.
* ORT contrib adds `MultiHeadAttention` and `Attention` as fused ops.
* MIGraphX inherits ORT's `GroupQueryAttention` for AMD GPU.
* MindIE ATB exposes `FlashAttention` and `PagedAttention` as fused ops.
* ExecuTorch ships `sdpa_with_kv_cache` as a single CPU custom op.
* ggml adds `GGML_OP_FLASH_ATTN_EXT`.
* MLX ships `mlx.fast.scaled_dot_product_attention` as a single Metal kernel.

**Anchor moment:** TRT-LLM 0.7 (early 2024) is the high-water mark for "plugin = op = layer building block".

### Era 3 — Stateful-cache era (mid-2024 → end-2024)

* Core ML iOS 17 ships `read_state` / `coreml_update_state` — the first stateful KV in a major runtime.
* OpenVINO 2024.5 adopts stateful Variables (`ReadValue`/`Assign`) as the default GenAI path.
* TVM-Relax adds `relax.vm.AttentionKVCache` as a VM-level object.
* ExecuTorch consolidates around `sdpa_with_kv_cache` with external buffer threading.
* DirectML uses D3D12 stateful resources for KV.
* TPU XLA programs thread KV as parameter/result tuples (`dynamic-update-slice` for writes).

**Anchor moment:** WWDC 2024 (June) — Apple reports 1.6× speedup from stateful vs. concatenated KV on Mistral-7B.

### Era 4 — PagedAttention adoption (Q3 2024 → end-2025)

* ORT contrib `PagedAttention` lands (2024).
* OpenVINO 2024.5 lights up `PagedAttentionExtension` across CPU/GPU/NPU.
* TRT-LLM `paged_kv_cache: bool` becomes default for serving.
* MindIE `PagedAttentionOperation` for Ascend.
* vLLM-TPU launches with Ragged Paged Attention v2 (Oct 2025).
* SGLang adds RadixAttention for prefix-sharing across requests.

**Anchor moment:** OpenVINO 2024.5 (Nov 2024) — paged attention crosses from research kernel to production IR op across all four major backends in OV.

### Era 5 — Sinks, sparse attention, MoE first-class, single-binary runtimes (2025 → 2026)

* **Attention sinks** become first-class: MLX `sinks` parameter on `fast.scaled_dot_product_attention` (mid-2025); OpenVINO 2025.4 `sink` input on PagedAttention (late 2025); GPT-OSS and Mistral-Small-3.1 **train** with sinks rather than retrofit.
* **Sparse attention**: OpenVINO 2025.4 ships XAttention (block-sparse with antidiagonal scoring) as preview; ORT contrib has `SparseAttention`.
* **MoE first-class**: TRT-LLM `MixtureOfExperts` plugin, ORT contrib `MoE`, ATB MoE op, ggml `MUL_MAT_ID`, NKI MoE megakernel; the minimal API can no longer treat MoE as composed.
* **Single-binary inference**: TRTLLM-Gen (Blackwell, late 2025) replaces the per-model plugin builder with runtime kernel selection; vLLM V1 (Jan 2025) is a major redesign with TritonAttn matching FA3 performance.
* **Hybrid attention**: OpenVINO 2025.4 supports CausalConv1D + GatedDeltaNet hybrid models in SDPA + PA backends.
* **NSA / vAttention** (research → production): NSA (DeepSeek 2025) Native Sparse Attention; vAttention dynamic memory management appears in proposals; SnapMLA for MLA decode on Blackwell.

**Anchor moments:** vLLM V1 (Jan 2025) + OpenVINO 2025.4 (late 2025) + TRTLLM-Gen + MLX sinks (mid-2025).

### Predicted Era 6 — Tokenmixer-pluggable runtimes (2026 → 2027)

The trajectory points to:

* **Hybrid token-mixer first-class ops** — SDPA, SSM (`selective_scan`), gated linear attention, RWKV-7 all become parallel choices behind one `token_mixer(...)` API. OpenVINO 2025.4 already supports linear-state hybrids; Pallas Mamba kernels are in flight.
* **vAttention** (dynamic VM mapping for KV) as a competitor to paged block tables.
* **Persistent kernels** (Hopper/Blackwell SM-resident kernels) as a fusion shape beyond mega-op plugins.
* **Foundation-model-API runtimes** (Apple Foundation Models, Phi Silica, similar) exposing only the model-level surface, hiding both kernels and IR.

---

## 9. Open / proprietary ops worth modeling in `llm-layers`

Lifted from v1 §11, expanded with v2's broader survey:

1. **OpenVINO `PagedAttentionExtension`** (≥22 inputs) — the broadest single op in the survey; subsumes paged KV + sliding window + ALiBi + sinks + cache rotation + sparsity. The minimal API should *not* model these as separate ops but as optional parameters of an `attention()` call.
2. **TRT-LLM `GemmSwigluPlugin` / `LowLatencyGemmSwigluPlugin`** — fusing two GEMMs + SwiGLU into one kernel (FP8 on Hopper, FP4 on Blackwell). The minimal API should allow `ffn_swiglu()` to be one logical op with an implementation-defined fusion boundary.
3. **ggml `GGML_OP_MUL_MAT_ID`** — gather-then-grouped-matmul as a single op for MoE expert dispatch. Distinct enough from generic gather + matmul to deserve a first-class `moe_dispatch()` abstraction.
4. **Core ML `coreml_update_state`** + **OpenVINO `Assign`** + **TPU `dynamic-update-slice`** — three different concrete representations of "in-place KV write". The minimal API needs an abstract "consume + produce updated state" arrow.
5. **MLX `sinks` parameter** + **OpenVINO 2025.4 PagedAttention `sinks` input** — attention sinks crossed the threshold from research to first-class API in 2025. Expose `sinks` as an SDPA parameter.
6. **TRT-LLM `position_embedding_type` enum** (8 RoPE variants) + ORT `GemmaRotaryEmbedding` — the right RoPE parameter is an enum, not boolean.
7. **`SkipLayerNormalization` / `SkipSimplifiedLayerNormalization`** (ORT) + `rmsnormQuantizationPlugin` `residual` input (TRT-LLM) — residual+norm fusion is so universal it gets its own op. The minimal API should bake "residual fed into next norm" as a first-class fused op.
8. **NKI Transformer TKG megakernel** + **Pallas Ragged Paged Attention v2** + **Mojo SnapMLA** — three examples of "the whole attention block as one user-authored kernel" trajectory. The minimal API must support a "lower this block to one kernel" lowering path.
9. **AWQ / GPTQ / MXFP4 / NVFP4 / U8 KV / FP8 KV / NF4 / Q4_K / palettization** — the quantization spec must be a *spec* (axis bits, weight bits, scale format, group size, kv dtype), not an enum.

---

## 10. Verification ledger and open issues

### 10.1 Claims verified against primary source

* ggml `GGML_OP_FLASH_ATTN_EXT`, `RMS_NORM`, `ROPE`, `MUL_MAT`, `MUL_MAT_ID`, `SOFT_MAX`, `ADD`, `MUL`, `GET_ROWS`, `CONCAT`, `RESHAPE`, `TRANSPOSE`, `VIEW`, `ARGSORT`, `TOP_K` — verified against `https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h`.
* QNN `QNN_OP_ROTARY_EMBEDDING`, `QNN_OP_RMS_NORM`, `QNN_OP_MAT_MUL`, `QNN_OP_FULLY_CONNECTED`, `QNN_OP_ELEMENT_WISE_ADD`, `QNN_OP_DEQUANTIZE`, `QNN_OP_QUANTIZE`, `QNN_OP_RESHAPE`, `QNN_OP_TRANSPOSE`, `QNN_OP_CONCAT`, `QNN_OP_GATHER`, `QNN_OP_CAST` — verified against the public listing of `QnnOpDef.h` at `https://docs.qualcomm.com/doc/80-63442-10/topic/api-rst_program_listing_file_include_QNN_QnnOpDef_h.html`.
* TRT-LLM plugin list — 32 plugin names verified against `cpp/tensorrt_llm/plugins/CMakeLists.txt`.
* TRT-LLM quantization mode enum — verified against `tensorrt_llm.quantization.mode`.
* ORT contrib documented ops — verified against `https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/ContribOperators.md` (fetched 2026-06-04): `SkipSimplifiedLayerNormalization`, `GroupQueryAttention`, `MultiHeadAttention`, `PagedAttention`, `RotaryEmbedding`, `GemmaRotaryEmbedding`, `MatMulNBits`, `MatMulBnb4`, `GemmFloat8`, `BeamSearch`, `GreedySearch`, `Sampling`, `MoE`, `SparseAttention`, `BiasSplitGelu`, `BiasGelu`, `FastGelu`, `QuickGelu`, `DecoderMaskedMultiHeadAttention`, `DecoderMaskedSelfAttention`, `PackedAttention`, `PackedMultiHeadAttention`, `EmbedLayerNormalization`, `GatherBlockQuantized`, `LongformerAttention`, `LinearAttention`, `WhisperBeamSearch`, `NGramRepeatBlock`, `GroupNorm`, `SkipGroupNorm`.
* CoreML stateful KV (`read_state`, `coreml_update_state`) and SDPA — verified via `https://apple.github.io/coremltools/docs-guides/source/stateful-models.html` and WWDC 2024 session 10161 referenced.
* MLX `fast.scaled_dot_product_attention` signature with `sinks` — verified at `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html` (MLX 0.31.1).
* OpenVINO 2025.4 sink-input on PagedAttention, XAttention preview, MoE perf — verified via `https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-4.html`.
* ExecuTorch `sdpa_with_kv_cache` — verified via ExecuTorch 0.4 docs and llama2 example tree.
* oneDNN Graph SDPA + compressed-KV pattern — verified at `https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa_compressed_kv.html`.
* TPU Pallas RPA + HLO custom-call — verified via MaxText docs and the RPA arXiv paper.
* AWS NKI 0.3.0 + Transformer TKG megakernel — verified via Neuron SDK 2.29 release notes + re:Invent 2025 AIM414 talk reference.
* TRTLLM-Gen FP4 GEMM and Blackwell support — verified via TRT-LLM release notes and NGC catalog.
* vLLM V1 (Jan 2025) + Triton attention backend — verified via vLLM blog and `https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/`.
* Modular MAX on Blackwell, SnapMLA — verified via Modular GTC 2026 blog.

### 10.2 Claims explicitly marked `[unverified]`

* `QNN_OP_SCALED_DOT_PRODUCT_ATTENTION` and any Genie-internal SDPA fused op — not in the public `QnnOpDef.h` listing. May exist in HTP-internal op tables; needs SDK-bundle access to confirm.
* Exact MLX release tag that introduced `sinks` (somewhere between 0.20 and 0.31).
* Exact OpenVINO release that introduced FP8 (`FakeConvert`) — likely 2024.x.
* OpenVINO `KV_CACHE_PRECISION` `u4` value — present in some 2025.x; exact release unverified.
* OpenVINO `PagedAttentionExtension` 28-input list from v1 — feature-branch fields (`adaptive_rkv_*`, `qq_bias_*`) probably not in 2025.4 release. Conservative count: ~22 inputs in release; up to 28 in dev branches.
* QAIRT firmware floor for any fused SDPA (Hexagon v75 vs. v76+).
* MindIE ATB op exact signatures since the Sept 2025 open-source release — pending direct verification.
* Whether iOS 18 MIL has any single block that fuses SDPA + state update.

### 10.3 Claims retracted from v1

* **`QNN_OP_KV_CACHE` as a first-class op** — RETRACTED. No such macro in public `QnnOpDef.h`. The "op-as-cache" KV representation category is removed; QAIRT is recategorised under "PrefillDecodeSplit + ExternalBuffer (#7 in §5)".
* **`QNN_OP_SILU` as a first-class op** — RETRACTED. Not in public listing.
* **`QNN_OP_SCALED_DOT_PRODUCT_ATTENTION` as a public-listing op** — RETRACTED (downgraded to `[unverified, likely Genie-internal]`).
* **OpenVINO `PagedAttentionExtension` "28 inputs" with `adaptive_rkv_*` and `qq_bias_*`** — partially RETRACTED; the conservative figure for 2025.4 is ~22 inputs with sinks already shipped.
* **"ORT contrib has SimplifiedLayerNormalization (in docs)"** — clarified: implemented in C++, not in the contrib markdown index. The going-forward op is `RMSNormalization` in ONNX opset-23.

### 10.4 Open issues for v3

* Confirm the exact 2025.4 input list of OpenVINO `PagedAttentionExtension` by fetching the file at a 2025.4 release commit.
* Get an SDK-bundle copy of `QnnOpDef.h` (rather than the doxygen render) and search for any HTP-internal LLM attention fused op.
* Re-pin MindIE / ATB ops against the now-open Ascend tree.
* Track the MLX changelog to pin the `sinks` introduction release tag.
* Track the Pallas Mamba/SSM kernel landing — likely changes the SSM row in the synthesis table.
* Add cuDNN fused SDPA as a full §2 sub-section once we settle on whether to treat it as an "IHV op set" (cuDNN as NVIDIA primitive layer) or a "kernel library" (and therefore §2.20-class).
* Confirm the DirectML `DML_OPERATOR_ROTARY_EMBEDDING` presence — referenced in WinML/genai integration but not directly verified.

---

## 11. References (version-pinned)

### Primary headers and source files

* ggml `ggml.h` — `https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h`
* ggml KV cache impl — `https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp`
* QNN `QnnOpDef.h` listing — `https://docs.qualcomm.com/doc/80-63442-10/topic/api-rst_program_listing_file_include_QNN_QnnOpDef_h.html`
* QNN dequantize macro example — `https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/define_QnnOpDef_8h_1a97a823ed1f60d9b1304461de39cf203d.html`
* TRT-LLM plugins CMakeLists — `https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0/cpp/tensorrt_llm/plugins/CMakeLists.txt`
* TRT-LLM plugin Python — `https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0/tensorrt_llm/plugin/plugin.py`
* ORT contrib operators markdown — `https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md`
* ORT RMSNormalization PR (opset 23) — `https://github.com/microsoft/onnxruntime/pull/23560`
* ORT SimplifiedLayerNormalization issue — `https://github.com/microsoft/onnxruntime/issues/21925`
* CoreML MIL ops — `https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html`
* CoreML stateful — `https://apple.github.io/coremltools/docs-guides/source/stateful-models.html`
* MLX SDPA — `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html`
* MLX quantized_matmul — `https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html`
* ExecuTorch llama2 example — `https://github.com/pytorch/executorch/blob/main/examples/models/llama2/README.md`
* OpenVINO PagedAttention PRs — #26975 #27204 #27841 #28017 #28232 #28493 #28645 #28815 #29383
* OneDNN Graph SDPA — `https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa.html`
* OneDNN Graph SDPA compressed-KV — `https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa_compressed_kv.html`

### Release notes

* OpenVINO 2025.4 — `https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-4.html`
* OpenVINO 2024.5 — `https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2024-5.html`
* TRT-LLM release notes — `https://nvidia.github.io/TensorRT-LLM/release-notes.html`
* Neuron SDK what's new — `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html`
* DirectML / WinML — `https://blogs.windows.com/windowsdeveloper/2025/05/19/introducing-windows-ml-the-future-of-machine-learning-development-on-windows/`
* MAX changelog — `https://docs.modular.com/max/changelog/`

### Talks and papers

* WWDC '24 session 10161 (Deploy ML models on-device with Core ML) — `https://developer.apple.com/videos/play/wwdc2024/10161/`
* WWDC '24 session 10159 (Bring your ML and AI models to Apple silicon) — `https://developer.apple.com/videos/play/wwdc2024/10159/`
* AWS re:Invent 2025 AIM414 (Performance engineering on Neuron / NKI) — `https://www.antstack.com/talks/reinvent25/aws-reinvent-2025---performance-engineering-on-neuron-how-to-optimize-your-llm-with-nki-aim414/`
* "Fast On-device LLM Inference with NPUs" (ASPLOS 2025) — `https://xumengwei.github.io/files/ASPLOS25-NPU.pdf`
* Ragged Paged Attention for TPU (arXiv 2604.15464) — `https://arxiv.org/html/2604.15464`
* vAttention (arXiv 2405.04437) — `https://arxiv.org/pdf/2405.04437`
* Triton Attention Kernel anatomy (arXiv 2511.11581) — `https://arxiv.org/pdf/2511.11581`
* Modular GTC 2026 blog — `https://www.modular.com/blog/modular-at-nvidia-gtc-2026-max-on-blackwell-mojo-kernel-porting-and-deepseek-v3-on-b200`
* vLLM V1 blogs (Jan 2025; Triton backend March 2026) — `https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html`
* vLLM-TPU launch (Oct 2025) — `https://blog.vllm.ai/2025/10/16/vllm-tpu.html`

### Open-source-ecosystem references

* Ascend open source announcement (Sept 2025) — `https://www.huawei.com/en/news/2025/9/hc-shengten-opensource`
* vllm-ascend — `https://github.com/vllm-project/vllm-ascend`
* MLC Relax — `https://github.com/mlc-ai/relax`
* TVM optimize_llm tutorial — `https://tvm.apache.org/docs/how_to/tutorials/optimize_llm.html`
* MaxText Pallas — `https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html`
* PyTorch XLA Pallas — `https://docs.pytorch.org/xla/master/features/pallas.html`
* LiteRT overview — `https://ai.google.dev/edge/litert/overview`
* LiteRT-LM blog — `https://developers.googleblog.com/blazing-fast-on-device-genai-with-litert-lm/`
* ExecuTorch 1.0 + SME2 (Arm blog) — `https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/executorch-1-0-is-here-and-with-sme2-optimizations-through-kleidiai`
* MTIA at ISCA 2025 — `https://aisystemcodesign.github.io/papers/MTIA-ISCA25.pdf`
* MTIA scale blog — `https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/`

---

## 12. Take-aways for the `llm-layers` minimal API design (revised from v1 §12)

1. **Ten logical ops cover ≥ 90% of every decoder block across 18 runtimes**: `rms_norm`, `linear` (with quant spec), `rope`, `attention(q,k,v,kv_state,mask,sinks,window)`, `residual_add`, `swiglu_ffn`, `silu`, `mul`, `lm_head`, plus **`moe_dispatch(x, router_w, expert_w, top_k)`** (newly required given §4.2 — 10/18 runtimes have first-class MoE).
2. **Attention must be parameterised across 9 fusion shapes** (§6) not 5. The API caller never picks the shape; the lowering does. Parameters: `qkv_layout`, `rope_apply`, `kv_kind` (10 values, §5), `phase`, `mask_kind`, `sinks`, `target_fusion`.
3. **KV cache is a first-class typed abstraction with 11 lowerings** (§5). The abstract type is `KVState[layer, head, pos, dim]`; lowerings include `PastPresent`, `OVStateful`, `CoreMLState`, `BlockTable(paged_block_size)`, `ExternalBuffer`, `PurePython`, `XLAThreaded`, `GGMLPool`, `D3DResource`, `RelaxVMObject`, `RadixTree`.
4. **Quantization is a `QuantSpec` not an enum.** Parameters: `act_dtype` (bf16/fp16/fp8e4m3/fp8e5m2/int8/nvfp4/mxfp4), `weight_dtype` (int4/int8/fp8/fp4/nf4/lut), `scale_format` (per-tensor/per-channel/per-group/per-block), `group_size`, `kv_dtype`. This subsumes every scheme observed in §7.
5. **Fused-residual-norm is a real op.** Almost every runtime has `SkipNorm`-style fusion. Expose `residual_rms_norm(prev, residual, gamma) -> (h, new_h)` so that lowerings to TRT-LLM, ORT contrib, Core ML, ATB, DML can all hit a fused path.
6. **SwiGLU-as-one-op** matches the most aggressive fusion targets (TRT-LLM `GemmSwigluPlugin`, MindIE `SwiGluOperation`, NKI MoE megakernel, Mojo) without losing the ability to decompose for runtimes that don't fuse it.
7. **The minimal API should *not* leak Q/K/V splitting as separate user-facing ops** — every fused-attention runtime accepts merged QKV input. Exposing `qkv_proj(x, Wqkv) -> (q, k, v)` as one logical op (with optional split-then-pack lowerings) is portable.
8. **Sinks, sliding windows, and XAttention sparsity are 2025–2026 table-stakes.** Sinks must be an optional input to `attention`. Sliding window and sparsity may remain optional plugin-style flags.
9. **Two-phase compilation** (prefill graph vs. decode graph) is the norm in QNN/Genie, TensorRT-LLM, NKI. The minimal API should produce *abstract* layer descriptions that can be specialised into either phase rather than baking phase into the IR.
10. **Distributed primitives (all-reduce, all-gather)** must be an optional sublayer rather than baked into `linear` — TRT-LLM's `GemmAllReducePlugin` and OpenVINO `Allreduce` differ on whether the fusion includes the GEMM.
11. **The retracted QAIRT `KV_CACHE` op** is a *cautionary tale*: when a runtime has no public IR op for KV, that does *not* mean it lacks fused KV semantics — those may live below the public API, at the kernel layer. The minimal API must remain agnostic about whether KV fusion happens in the IR or in the kernel.
