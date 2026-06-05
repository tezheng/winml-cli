# Critique of `03-ihv-opsets.md` — IHV / Runtime Op-Sets Survey

> Reviewer posture: skeptical. The report is genuinely useful as a starting taxonomy, but it conflates "runtime I know how to look up" with "runtime that matters", lacks version pinning, and leans heavily on a few unverifiable secondary sources. Below is what needs to be fixed before this can support a minimal-API design proposal.

---

## Section 1 — Missing runtimes / IHV stacks

The report surveys **9** runtimes (OpenVINO, QNN/QAIRT, AMD FastFlowLM+MIGraphX, TensorRT-LLM, Core ML, MLX, KleidiAI/ExecuTorch, MindIE/ATB, ONNX Runtime contrib). For a paper that purports to extract "what every IHV does for a decoder block", the omission list is large and several omissions are *load-bearing* for the minimal-API claims.

### Severity-ranked omissions

**Critical (P0) — break the central argument of the paper**

1. **AWS Neuron SDK / NKI (Inferentia 2/3, Trainium 1/2/3).** Neuron has its own NeuronISA, its own kernel language (NKI), and ships its own LLM building blocks via NxD Inference and the Neuron Kernel Library. It exposes a *different fusion granularity* (NKI kernels are user-authored, vLLM is the production entry point on Trainium, with EP / disaggregation / spec-decode native). Missing AWS Neuron is the single largest gap because it is the only top-five hyperscaler accelerator stack absent from the survey, and it pioneered the "open-kernel, vLLM-native" model that increasingly competes with TRT-LLM. The "12 IHVs converged" claim cannot stand without Neuron.
2. **Google TPU XLA HLO / Pallas / JAX-attention.** The report never addresses TPU. XLA's `dot_general` + `reduce` + a custom-call HLO for paged attention (Pallas / Mosaic) is *the* op-set behind Gemini and Gemma serving. Pallas-based Ragged Paged Attention on TPU (arXiv 2604.15464) is a documented kernel encapsulated as a custom HLO op. Without HLO, the survey cannot claim to describe "every IHV op-set"; it describes the non-Google subset.
3. **DirectML / Windows ML / Phi Silica path.** Microsoft ships DirectML 2.0 (2025) with explicit NPU LLM acceleration across Qualcomm, Intel, and AMD XDNA2 — and this is the *consumer Windows on-device* path. DirectML defines its own op set (`DML_OPERATOR_*`), and Phi Silica is the canonical NPU-tuned LLM. ORT contrib is mentioned, but the Microsoft side of the NPU story (Windows ML EP, DirectML ops, Phi Silica's NPU tuning) is missing.
4. **ggml itself.** The survey discusses ExecuTorch and ORT contrib but never the ggml op set (`GGML_OP_*` enum in `ggml/include/ggml.h`). llama.cpp, Ollama, LM Studio, Jan, KoboldCpp, and probably half of all on-device LLM inference in 2025 dispatch through `GGML_OP_FLASH_ATTN_EXT`, `GGML_OP_ROPE`, `GGML_OP_RMS_NORM`, `GGML_OP_MUL_MAT_ID` (MoE), `GGML_OP_SOFT_MAX`. There is no "llama.cpp" row in the synthesis table; this is a critical hole. ggml is *the* op-set most LLM end-users actually interact with.

**High (P1) — change the cross-runtime synthesis materially**

5. **TVM / Relax / MLC-LLM.** MLC-LLM is a shipped runtime (browser, Android, iOS, Vulkan) built on TVM Unity with the Relax dialect. Its op surface (`relax.matmul`, `relax.nn.attention_var_len`, `relax.nn.rms_norm`) and its WebGPU lowering are unique. Missing because the report seems to assume MLC is "TVM" and that TVM is academic. It is a shipped production runtime.
6. **TFLite / LiteRT GenAI.** Google's "Universal Framework for On-Device AI" supports Gemma 3/3n via the LiteRT op set, ai-edge-torch composes transformer building blocks, and the LiteRT QNN accelerator delegates 90 LiteRT ops onto Qualcomm NPUs. None of this is captured. KleidiAI/ExecuTorch is *Apple/Android via PyTorch*; LiteRT is the *Android via TensorFlow/JAX* counterpart and must be surveyed alongside.
7. **OneDNN graph SDPA / GQA / SDPA-with-compressed-KV.** Intel's CPU stack also exposes an SDPA *fusion pattern* via the oneDNN Graph API (covered for v3.8 → v3.13, 2025). This is distinct from OpenVINO and is what PyTorch CPU inductor and OnnxRuntime CPU EP use for attention. The report only treats OpenVINO and never the OneDNN primitive layer underneath, even though OneDNN's SDPA documentation explicitly enumerates compressed-KV (int4/int8 K and V) fused into the SDPA pattern in 2025.
8. **MTIA (Meta).** PyTorch-native, Triton-based, used for ranking and increasingly for GenAI per Meta's 2025 disclosures. The fusion model (PyTorch 2.0 + TorchInductor + Triton) is *the* most PyTorch-aligned stack and predicts where ExecuTorch/Inductor go in two years. Omitting MTIA means missing the "PyTorch eager kernel = IR" position.
9. **Modular MAX engine / Mojo kernels.** MAX 24.6 → 26.2 (Apple silicon GPU, NVIDIA Blackwell, AMD RDNA, Jetson Thor) is shipping in 2025–2026. Mojo kernels are an alternative to CUDA at the kernel level, and MAX ships its own LLM serving primitives (`max.graph.ops.*`). Hard to argue a 2025/2026-vintage survey is complete without MAX.

**Medium (P2) — niche but distinct architectures**

10. **Hailo-10H.** Edge AI accelerator that the *Hailo-8 era* (CNN-only) did not support; the Hailo-10H ships with a documented transformer decoder runtime, with the qwen2.5/deepseek-r1-distill-qwen 1.5B family as the supported set. The op model is closed but the *runtime contract* (token-in, logits-out, hidden KV) is a distinct point in the design space and matches the FastFlowLM "hidden cache" pattern. Worth a paragraph.
11. **Groq LPU / TSP.** GroqFlow is an MLIR-based compiler that lowers to a deterministic dataflow ISA (MEM / COMP / NET / CTRL). There is no public LLM op vocabulary, but the kernel-level abstraction is unique and worth documenting as evidence that *some IHVs hide the op vocabulary entirely* (compare FastFlowLM §3a).
12. **Cerebras Wafer-Scale CSL.** WaferLLM (arXiv 2502.04563) details a CSL-based LLM kernel set (MeshGEMM, MeshGEMV, on-wafer KV). Same "hidden op vocabulary" pattern.
13. **SambaNova SN40L / SambaFlow.** Reconfigurable dataflow with a tiled CGRA; the op model is fundamentally different (graph-of-spatial-streams). SambaFlow's PEF (Pattern Execution Format) is the IR, and *not having a graph-of-fused-ops* is itself an important data point.
14. **Edge TPU / Coral NPU.** Original Edge TPU is dead (no updates 2021→2025), but the new **Coral NPU** (IREE/MLIR-based, transformer-capable, Synaptics Torq production) is being co-designed with the Gemma team. Worth a "next-gen open NPU" entry.
15. **NXP eIQ / i.MX.** Industrial / automotive edge LLM target.
16. **cuDNN fused FlashAttention2/3 ops.** Distinct from TRT-LLM; cuDNN exposes its own fused SDPA primitives (`cudnnSDPAForward`, fp8 variants on Hopper/Blackwell) that PyTorch's `torch.nn.functional.scaled_dot_product_attention` increasingly dispatches into. Counts as a separate IHV op surface.
17. **vLLM custom ops / SGLang RadixAttention.** The report treats "vLLM" as just a consumer of TRT-LLM/Triton kernels. In fact vLLM defines its *own* op surface (`vllm.ops.paged_attention_v1/v2`, `vllm.ops.rms_norm`, `vllm.ops.fused_add_rms_norm`, `vllm.ops.silu_and_mul`). SGLang's `radix_attention` is novel — KV-cache *reuse* as a first-class primitive — not just paged-attention.

### Verdict for Section 1

**Count of missing runtimes / IHV stacks: 17** (4 P0 critical, 5 P1 high, 8 P2 medium). The 9-runtime survey roughly doubles to ~18 once these gaps are closed. The P0 gaps (AWS Neuron, TPU/HLO, DirectML, ggml) directly invalidate the report's universality claims and must be addressed before v2.

---

## Section 2 — Depth issues per runtime (version drift, missing doc URLs)

The report is consistently weak on **version pinning** and **doc-URL verification**. Below, runtime-by-runtime, what's wrong.

### 2.1 OpenVINO

- **Opset version claims are partially unverifiable.** The report says LLM-critical ops land in opset13 (SDPA) and opset14 (RoPE infrastructure via Symbolic transformations). The OpenVINO opset specs page I fetched lists opset1–opset4 explicitly in the navigation and does not surface opset16. The Python API docs confirm `opset13.scaled_dot_product_attention` and `opset14.scaled_dot_product_attention` exist (the 2024 docs show `opset14.scaled_dot_product_attention` as the canonical Python entry, not opset13). The report should state: SDPA was *introduced* in opset13; opset14 adds an *additional* `scaled_dot_product_attention` overload (typically attribute changes). Verify against `ov::op::v13::ScaledDotProductAttention` (which is documented as the C++ class).
- **The "sink input added in 2025.4" claim** is corroborated by external OpenVINO summaries describing the `sink` tensor as shape `[N, ..., 1]` numpy-broadcastable to batch dims, added to SDPA in 2025.4. The report should cite the OpenVINO 2025.4 release notes URL directly, not just leave it as a parenthetical.
- **`PagedAttentionExtension` 28-input claim — partially verified.** I could not retrieve `openvino/op/paged_attention.hpp` directly (404 on the master branch URL given in the report; the file path may have moved to `src/core/include/openvino/op/paged_attention.hpp` or to an extension subdirectory). The report cites the inputs list with confidence; it must point at the *exact* file path + commit SHA + line range. The 28-input list in §1.4 includes `adaptive_rkv_diversity_block_set_indices` and `adaptive_rkv_diversity_block_set_indices_begins`, which together with `token_type_ids`, `qq_bias`, `qq_bias_begins` look like very recent additions; if these are not in `master` yet they may be in a feature branch. Mark as "as of 2025.x development branch" or pin a commit.
- **`RMSFusion`, `RoPEFusion`, `RoPEFusionGPTJ`, `RoPEFusionLlama`, `RoPEFusionChatGLM` pass names** are credible but not cited to specific files in the OpenVINO repo. Provide `src/common/transformations/*` paths.
- **Quantization claim "FP8 (E4M3, E5M2) added in 2025.x via `FakeConvert` op"** — needs the exact 2025.x version (was it 2024.4? 2025.0? 2025.2?). Important because vendor decks often confuse "compile-time supported" vs "kernel-shipped".
- **`KV_CACHE_PRECISION` plugin property** — verify against GenAI 2024/2025 docs; some versions only allow `u8` and `f16`, not `u4`.

**Severity: medium.** The OpenVINO section is the most detailed but also the most prone to "developer-branch creep" — citing op signatures that exist in master but not in the latest released artifact.

### 2.2 Qualcomm QAIRT / QNN

- **`QNN_OP_KV_CACHE` as a named op — UNVERIFIED.** I attempted to fetch `QnnOpDef.h` from Qualcomm docs (`docs.qualcomm.com/doc/80-63442-10/topic/api-rst_program_listing_file_include_QNN_QnnOpDef_h.html`). The page exists but is doxygen-rendered and the search tool could not extract the macro. The report's confident claim that `QNN_OP_KV_CACHE` is a first-class op deserves a direct quote from the header (`#define QNN_OP_KV_CACHE "KvCache"` or similar) — otherwise this is hearsay. The QAIRT op catalog *does* include attention-family ops but the published op list I can locate is `QNN_OP_MAT_MUL`, `QNN_OP_SOFTMAX`, `QNN_OP_DEQUANTIZE` (links extracted in search), not `QNN_OP_KV_CACHE`. The author should re-verify with a *literal grep* of the SDK header.
- **`QNN_OP_SCALED_DOT_PRODUCT_ATTENTION` Genie claim** — similarly not directly verifiable from public docs; the ExecuTorch QNN backend issue #16352 is cited but only documents quantization recipes, not the op definition. Cite the exact `QnnOpDef.h` line.
- **"a16w16 / a8w8 / int4 weights" claim** — accurate per QAIRT docs but should pin the QNN SDK version (2.x — there are major op-set additions between 2.18 and 2.27, e.g., GQA and SDPA only appeared on certain HTP firmwares).
- **HTP firmware-dependent op availability** is glossed over with a single sentence. This is the most important footnote for any cross-runtime planner. Spell it out: "SCALED_DOT_PRODUCT_ATTENTION requires HTP firmware ≥ v75 (Snapdragon 8 Gen 3 / X Elite) and QAIRT ≥ 2.24." If the actual version cutoffs are unknown, mark them TBD instead of asserting.
- **Genie pipeline (prefill + decode = two compiled graphs)** is correct in spirit but unsourced. Cite Qualcomm's Genie SDK documentation, not just the ASPLOS'25 paper.

**Severity: high.** The single most consequential proprietary-op claim in the entire report (`QNN_OP_KV_CACHE`) cannot be confirmed from any link I could fetch. This must be quoted from the header file in v2.

### 2.3 AMD MIGraphX and FastFlowLM

- **MIGraphX → ORT contrib op adoption** is well-sourced (CHANGELOG cited). Good section.
- **`MatMulNBits` claim** — accurate; ORT contrib confirms.
- **`GroupQueryAttention` as the canonical fused op** — accurate; matches ORT contrib.
- **FastFlowLM section is inherently speculative** because the runtime is closed; the report acknowledges this. Fine.
- **Missing**: MIGraphX FP8/MXFP4 paths on MI300X/MI325X/MI350 should cite *which AITER kernels* and *which ROCm version*. The Transformer Engine integration claim for MX-FP4 on MI350 needs a ROCm version number.

**Severity: low.**

### 2.4 NVIDIA TensorRT-LLM

- **Plugin list is verified** — I fetched `cpp/tensorrt_llm/plugins/CMakeLists.txt` and confirmed all 32 plugin entries match what the report quotes (`gptAttentionPlugin`, `gemmSwigluPlugin`, `qserveGemmPlugin`, `selectiveScanPlugin`, `eaglePlugin`, etc.). Strong.
- **`GptAttentionPlugin` fusion scope claim** (QKV split + RoPE + KV r/w + masked SDPA + output bias + quant) is consistent with public docs but no direct line-citation from `gptAttentionPlugin.cpp` is given. Add one.
- **`position_embedding_type` enum (gptj, gpt_neox, gemma, chatglm, longrope, yarn, alibi, learned_absolute)** — these match the `PositionEmbeddingType` enum in TensorRT-LLM. Good.
- **Quantization matrix (12+ W4A8 variants, NVFP4, MXFP4, W4A8_NVFP4_FP8, W4A8_MXFP4_FP8, W4A8_MXFP4_MXFP8)** — accurate per `tensorrt_llm.quantization.mode`. Verified.
- **MISSING: TRTLLM-Gen.** The user's hint asked about the transition from plugin-based attention to TRTLLM-Gen (Blackwell-era single-binary path). The report never mentions TRTLLM-Gen, which is the *defining* recent evolution of TRT-LLM. Public summaries confirm TRTLLM-Gen exists in late-2025 releases. v2 must add this.
- **`KvCacheConfig(dtype=...)` API** — should be cited from the Python builder docs (`tensorrt_llm.runtime.KvCacheConfig` or similar) — currently asserted.

**Severity: medium.** Solid section overall, but missing the TRTLLM-Gen architectural shift undermines the "evolution" narrative.

### 2.5 Apple Core ML

- **iOS17 stateful KV (`read_state`, `coreml_update_state`) claim** is verified by the Apple coremltools Stateful Models guide (confirmed via search). Good.
- **iOS18 `scaled_dot_product_attention` op claim** is verified by coremltools converter discussion and HuggingFace mistral-coreml post. Good.
- **`constexpr_*` quant ops** — verified via the coremltools palettization overview URL cited.
- **MISSING**: When did stateful KV cache *exactly* land — iOS17.0 or 17.2? When did SDPA exactly land in MIL — coremltools 7.2 or 8.0? Pin versions.
- **MISSING**: No mention of `coreml_unified_attention` or the iOS18 *MLProgram* variant that fuses SDPA + stateful update into one block, which Apple discussed at WWDC24 session 10161.
- **MISSING**: No coverage of Apple Foundation Models / Apple Intelligence on-device LLM, which uses a different (closed) op surface above MIL.

**Severity: low.**

### 2.6 Apple MLX

- **`mlx.fast.scaled_dot_product_attention(q, k, v, scale, mask, sinks)` signature** — the search result confirms an MLX 0.31.1 doc page exists for this function. Verify whether `sinks` is actually a *positional* or *keyword-only* argument in 0.31.1, and which MLX version first shipped `sinks` (the report says "2025 attention sinks" but does not pin the MLX release).
- **`quantized_matmul` bits ∈ {2,3,4,5,6,8}** — needs verification against current MLX docs; older versions may only support {4, 8}. Pin version.
- **Issue #3404 (TurboQuant / quantized KV in fast SDPA)** — the report flags this as "in-flight". As of 2025/2026 this may have shipped; recheck.

**Severity: low.**

### 2.7 ARM KleidiAI / ExecuTorch

- **`sdpa_with_kv_cache` custom op claim** — verified by ExecuTorch + PyTorch mobile blog (cited URL is correct).
- **KleidiAI micro-kernel naming (`matmul_clamp_f32_qai8dxp_qsi4cxp`)** — verified.
- **SME2 path** — verified by ARM blog.
- **MISSING**: ExecuTorch backend matrix is much wider than CPU (Vulkan, MPS, XNNPACK, CoreML delegate, QNN delegate, MTK NeuroPilot delegate). The report flattens this to "the CPU path". A v2 should distinguish the *ExecuTorch IR* (PortableModule with ATen + custom ops) from the *backend-delegated subgraph* (`executorch.exir.delegate.lowered_module`).
- **MISSING**: `LinearInt4` is a real ExecuTorch op (`executorch.llama.linear_int4`), but the report should cite its actual `.cpp` location.

**Severity: medium** because the backend taxonomy collapse is a real conceptual error.

### 2.8 Huawei MindIE / Ascend ATB

- **Section is explicitly hedged** ("CANN's official op documentation is gated behind Huawei's developer portal and is partially Mandarin-only"). Good intellectual honesty.
- **However**, the report cites CSDN/Zhihu/Tencent Cloud blogs as primary sources. These are tertiary at best. v2 should at least cite the open-source CANN op definition headers from the [Ascend open-source release](https://www.huawei.com/en/news/2025/9/hc-shengten-opensource) announced Sep 2025.
- **Ops claimed (`FlashAttentionOperation`, `PagedAttentionOperation`, `SwiGluOperation`, `RmsNormOperation`, `RopeOperation`)** — credible but unverified against the now-open-source repo. Re-pin to commits.
- **HiFloat8 / HF8** — verify against Ascend 910B spec.

**Severity: high** because the Mandarin-source dependency is fixable now that Ascend has open-sourced (Sept 2025). v2 has no excuse to keep tertiary citations.

### 2.9 ONNX Runtime contrib

- **Op presence verified directly.** I fetched the raw `ContribOperators.md` (274KB, 6887 lines) and confirmed presence of: `PagedAttention`, `GroupQueryAttention`, `MultiHeadAttention`, `Attention`, `RotaryEmbedding`, `GemmaRotaryEmbedding`, `MatMulNBits`, `MatMulBnb4`, `GemmFloat8`, `BeamSearch`, `GreedySearch`, `Sampling`, `BiasGelu`, `FastGelu`, `QuickGelu`, `BiasSplitGelu`, `BiasAdd`, `GatherBlockQuantized`, `DecoderMaskedMultiHeadAttention`, `DecoderMaskedSelfAttention`, `PackedAttention`, `PackedMultiHeadAttention`.
- **NOT found** in my fetched copy of `ContribOperators.md`: `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`, `RMSNormalization`, `SparseAttention`, `QAttention`, `QOrderedAttention`. The model fetch indicates these are not in the contrib operators *markdown index*. Multiple secondary sources confirm `SimplifiedLayerNormalization` is implemented in ORT's `contrib_ops/cpu/layer_norm.cc` but the report's claim that it appears in the contrib *documentation* may be a documentation bug rather than a missing op. ORT also has `RMSNormalization` only via the *ONNX standard* opset 23 (a PR in flight on the ORT side per Issue #21925). This means the report **conflates "ORT contrib op exists in C++" with "ORT contrib op is in the documented contrib list."** This is a real defect: `SimplifiedLayerNormalization`, `SparseAttention`, `QAttention`, `QOrderedAttention` need separate sourcing.
- **`PagedAttention` inputs** — the report does not give the input list. It must, especially since PagedAttention's contrib spec is one of the report's anchor points for the "block tables (paged)" KV representation. The summary search confirmed PagedAttention "takes in packed input, i.e., only the real tokens without padding" but did not enumerate inputs. v2 must enumerate inputs and outputs and cite the markdown section header.

**Severity: high.** This is the report's "lingua franca" section and contains the most directly verifiable claims — yet several op names appear unverifiable in the current public markdown. Conflation of "implemented" vs "documented" must be fixed.

---

## Section 3 — Synthesis table errors / gaps

The synthesis table in §10 is the most useful artifact in the report. It has *real* errors and *real* gaps.

### Errors

1. **ORT row for "input RMSNorm"** says `SimplifiedLayerNormalization`. As noted in §2.9, this op may not be in the contrib markdown. Either it should be cited from `contrib_ops/cpu/layer_norm.cc` (C++ source) or the cell should say `LayerNormalization` (the documented op). Mixing "documented contrib" with "C++-only contrib" without footnotes misleads readers.
2. **OpenVINO row for "RMSNorm"** says `RMS (post-fusion)`. This is technically correct but misleading: `RMS` is an *internal pass output*, not a public IR op. End users serializing OpenVINO IR will see the composite (`ReduceMean → Add → Sqrt → ...`). The table should annotate: "`RMS` only post-`RMSFusion`; serialized IR is composite."
3. **MIGraphX row for "RoPE"** says `RotaryEmbedding`. The MIGraphX importer accepts the ONNX op name, but MIGraphX's internal IR may use a different op (e.g., `migraphx::op::rope` or decompose it). The cell elides this; v2 should distinguish "importable name" from "internal op".
4. **MLX row for "KV write/read"** says "Python `KVCache` (no op)". Correct, but the cell incorrectly implies this contradicts §11's KV representation count; in fact this *is* one of the six representations (pure Python state). Re-cross-link.
5. **Core ML row for "Final norm"** says `composed`. But Core ML MIL has `layer_norm` (mentioned in §5) — for non-RMS LLMs (e.g., Mistral-style models that use LN, not RMSNorm) this matters. The cell should distinguish RMSNorm-composed vs LN-direct.
6. **QNN row for "SDPA core"** says `SCALED_DOT_PRODUCT_ATTENTION (or MatMul+Softmax+MatMul)`. This is fine semantically but the op-name claim is the *single most consequential unverified claim* in the report (see §2.2). Until verified against `QnnOpDef.h`, mark as `[unverified]`.
7. **TRT-LLM row for "RoPE"** says "fused into `GptAttentionPlugin`" — correct. But the row for "QKV proj" still lists separate GEMMs. In practice, with `GptAttentionPlugin`, the QKV GEMM is *also* often inside the plugin or its predecessor. Clarify.

### Gaps

8. **No row for "embedding lookup".** Every decoder block runs after embedding. ORT has `Embedding` / `Gather`, OpenVINO has `Gather`, QNN has `QNN_OP_GATHER`, TRT-LLM has `LookupPlugin`. Missing.
9. **No row for "MoE routing"** even though MoE is now mainstream (Mixtral, DeepSeek-V3, Qwen3-MoE, Llama-4-Scout). TRT-LLM has `MixtureOfExperts`, MindIE has fused MoE ops, ORT contrib has `MoE`, ggml has `GGML_OP_MUL_MAT_ID`. The minimal-API design will *fail* without an MoE row.
10. **No row for "all-reduce / TP collective".** TRT-LLM has `NcclPlugin`, `GemmAllReducePlugin`; OpenVINO has `Allreduce`; ATB has `AllReduce`. For multi-GPU/multi-NPU inference this is essential.
11. **No row for "speculative decoding helpers"** (TRT-LLM `EaglePlugin`, MLX speculative sampler, ORT `BeamSearch`/`GreedySearch`/`Sampling`).
12. **No row for "sampling / logits processing".** TRT-LLM has `cumsumLastDimPlugin`, `topkLastDimPlugin`. ORT has `Sampling`. ggml has `GGML_OP_ARGSORT`, `GGML_OP_TOP_K`.
13. **No row for "quant-dequant"**, even though every runtime has it as a distinct op (ORT `QuantizeLinear`/`DequantizeLinear`, OpenVINO `FakeQuantize`/`Convert`, QNN `QUANTIZE`/`DEQUANTIZE`).
14. **No row for "attention mask construction"** (causal mask materialization vs implicit `is_causal=true`).

### Verdict

The 9-row × 9-column synthesis table is too narrow and is missing 6+ rows that are load-bearing for 2025-era LLM inference. The "≥7 of 9" consensus claim in §11 is computed over a non-representative row set, so the consensus conclusions are arguably overconfident.

---

## Section 4 — Op-fusion claim verification

### 4.1 TRT-LLM `GptAttentionPlugin` fusion scope

The report claims `GptAttentionPlugin` fuses: QKV split + RoPE/ALiBi + KV r/w + masked SDPA (context-FMHA prefill + masked MMHA decode) + output bias + quant scaling, with `paged_kv_cache` toggling paged vs contiguous KV.

- **Plugin list verified** — `gptAttentionPlugin` is in `CMakeLists.txt`.
- **`position_embedding_type` enum (8 variants)** — matches public TRT-LLM source.
- **`paged_kv_cache: bool`** — matches `gpt_attention_plugin.cpp` public source.
- **Quantization parameters** — match `tensorrt_llm.quantization.mode`.
- **NOT verified**: that the plugin *itself* handles the output bias. In some TRT-LLM versions output bias is a separate `Elementwise` add fused at TRT IR level. v2 must clarify.

### 4.2 OpenVINO `PagedAttentionExtension` 28-input claim

- The 28-input list is enumerated by the report but the cited header URL returns 404 from my fetcher. The list includes very recent fields (`adaptive_rkv_*`, `qq_bias`, `xattention_*`). These read as a "speculative or feature-branch" inventory, not a stable release. Verify against an *exact* commit + line, and tag inputs that are master-only vs released.
- The "absorbs xattention sparsity, attention sinks, sliding window, ALiBi, KV-cache rotation, adaptive RKV" claim is the broadest fusion claim in any runtime in the entire survey. If verified, it is the single most important data point in the report. v2 must provide a literal copy of the input list with file path and SHA.

### 4.3 Core ML SDPA + state

- `scaled_dot_product_attention` op verified to exist in MIL (iOS18, coremltools 8.x).
- `coreml_update_state` and `read_state` verified via Apple coremltools Stateful Models guide.
- **NOT verified**: whether the iOS18 SDPA fuses *with* `coreml_update_state` into a single MIL block (the report implies they remain separate ops; this is consistent with the WWDC video but should be footnoted to a specific MLProgram example).

### 4.4 QNN MHA primitive

- `QNN_OP_SCALED_DOT_PRODUCT_ATTENTION` claim is the second most consequential unverified claim in the report (after `QNN_OP_KV_CACHE`). I could not extract the macro from the public `QnnOpDef.h` doxygen page. v2 must literally grep the SDK header.

### 4.5 "5 fusion shapes" claim

The report enumerates 5 fusion shapes in §11:
1. One mega-op (TRT-LLM)
2. Mega-op for SDPA only, RoPE+QKV separate (MIGraphX/ORT, MindIE)
3. SDPA fused, KV is external state (Core ML, MLX, OpenVINO stateful)
4. SDPA + KV in one custom op (ExecuTorch `sdpa_with_kv_cache`)
5. PagedAttention as colossal multi-input op (OpenVINO `PagedAttentionExtension`)
6. (sixth implicitly: two SDPA ops by phase — MindIE)

This is actually 6, not 5. v2 should either renumber or merge categories.

Also missing as a distinct fusion shape:
- **"Eager kernel graph"** (MLX, ggml, vLLM custom ops, PyTorch/CUDA): SDPA is one Triton/Metal/CUDA kernel call, no IR — the *runtime* is the IR.
- **"Compiler-fused, no named op"** (HLO + Pallas): SDPA is custom-call HLO with vendor-private metadata.

So the true count is closer to **8 fusion shapes**, not 5 or 6.

---

## Section 5 — KV-cache representation completeness

The report claims 6 representations:

1. past/present I/O ports (ONNX/ORT, MIGraphX)
2. Stateful Variables (OpenVINO, Core ML)
3. Block tables (TRT-LLM, OpenVINO PagedAttention, MindIE, ORT PagedAttention)
4. Op-as-cache (QNN `KV_CACHE`)
5. External buffer in custom op (ExecuTorch `sdpa_with_kv_cache`)
6. Pure Python state (MLX)

### Missing representations

7. **XLA stateful input / output (HLO `dynamic-update-slice` + `parameter`/`tuple` threading).** TPU LLM serving in PAX / MaxText / vLLM-TPU uses HLO-level state threading where KV is just another HLO input/output, distinct from "Stateful Variables" (which is IR-level state) and from "external buffer" (which is host-allocated).
8. **Radix-tree KV reuse (SGLang RadixAttention).** Not just a cache, a *shared* prefix tree across requests. This is conceptually different from "block tables" because it adds reference counting and prefix-merge semantics, and it is a real op-set point (SGLang's `radix_attention` kernel).
9. **vLLM PagedAttention v1 vs v2.** Both are distinct from ORT/TRT-LLM PagedAttention in operational semantics (v2 adds partition-and-reduce for long sequences). Should be enumerated.
10. **ggml KV cache: contiguous per-layer F16/F32 with explicit "memory pool" view ops** (`ggml_view_2d` + `ggml_set_2d` patterns). This is a 7th distinct representation.
11. **DirectML stateful resource (HLSL-side persistent UAV)** — distinct from CoreML stateful buffer because DML state lives in D3D resource heap.

### Are the 6 listed actually all distinct?

- "Stateful Variables (OpenVINO, Core ML)" lumps two representations that look similar but differ: OpenVINO uses `ReadValue`/`Assign` *graph ops* (the state is a *port* into the graph), Core ML uses `read_state`/`coreml_update_state` *MIL primitives* (the state is a *typed parameter* of the program). These are operationally different and should be two rows.
- "External buffer in custom op" (ExecuTorch) and "Op-as-cache" (QNN) differ only by *who allocates*: in ExecuTorch the caller does, in QNN the runtime does. Worth distinguishing more sharply.

### Verdict

The actual count of distinct KV representations is **9–10**, not 6. The "6 representations" line is a memorable hook but undercount.

---

## Section 6 — Evolution narrative gaps

The report has no consistent **evolution narrative**. It snapshots 2025.x state without showing the *trajectory*.

### Trajectories the report should but doesn't trace

- **OpenVINO**: pre-2024 explicit ops (`MatMul + Softmax + MatMul` SDPA) → 2024 stateful (`ReadValue` / `Assign` Variables for KV) → 2025 PagedAttention (28-input mega-op for continuous batching). The report mentions all three states but not in temporal order.
- **TRT-LLM**: plugin-per-fusion (`GptAttentionPlugin` + `GemmSwigluPlugin` + `RmsnormQuantizationPlugin`) → TRTLLM-Gen single-binary inference (Blackwell-era, late 2025). **Not mentioned at all** — major omission.
- **Core ML**: pre-iOS17 (no state, KV as concatenation) → iOS17 (stateful buffers) → iOS18 (`scaled_dot_product_attention` MIL op). Report mentions iOS17/iOS18 but not the pre-17 baseline.
- **ORT contrib**: `Attention` (encoder-style) → `MultiHeadAttention` → `GroupQueryAttention` (GQA support) → `PagedAttention` → `SparseAttention`. Each is a layer of LLM-specific accommodation. Report lists them but does not show that each was a year-over-year addition.
- **ggml**: `GGML_OP_ATTN_FLASH` → `GGML_OP_FLASH_ATTN_EXT` (causal flag added) → backend-specific fast-paths. **Section missing entirely.**
- **MLX**: SDPA without sinks → SDPA with sinks (2025). **Year and version not pinned.**
- **QNN**: composed SDPA → fused SDPA → KV_CACHE op. Each step is HTP firmware + QAIRT version dependent. **Versions not pinned.**

A v2 should add a per-runtime *timeline* (year column + op-additions column). Without it, the "evolution of LLM layers over 3 years" framing of the parent paper is not supported.

### Specific evolution claims that need verification

- **Core ML stateful KV "WWDC'24 reported ~1.6× speedup vs. concatenation-based caches"** — verify against WWDC24 session 10161 transcript.
- **OpenVINO 2025.4 sink input** — verify against 2025.4 release notes.
- **MLX sinks ship date** — pin to the MLX release tag.
- **QAIRT SDPA firmware floor** — pin to the QNN SDK release tag.

---

## Section 7 — Recommended fixes for v2

### Must-fix (blockers for publication)

1. **Add AWS Neuron, TPU/HLO, DirectML, and ggml as full sections.** These are P0 omissions that invalidate the universality claim.
2. **Verify QNN `KV_CACHE` and `SCALED_DOT_PRODUCT_ATTENTION` against `QnnOpDef.h`.** Quote the macro definitions verbatim. Without this, the "QNN treats KV as an op" claim is unsupported and the QNN row of the synthesis table is unsafe.
3. **Pin every version claim.** Replace "OpenVINO 2025.x" with "OpenVINO 2025.4 release notes — link". Replace "iOS18" with "coremltools 8.x, iOS18.0 SDK build N — link". Replace "QAIRT" with "QAIRT 2.27 — link".
4. **Replace 404 links.** The `openvino/op/paged_attention.hpp` URL is dead. Provide a commit-pinned URL and verify the 28-input list. If only 22 inputs are in the released branch and 6 are in a feature branch, say so.
5. **Resolve ORT contrib documentation conflation.** Either the report cites the `ContribOperators.md` index (in which case `SimplifiedLayerNormalization`, `SparseAttention`, `QAttention`, `QOrderedAttention` are not in the index and need separate sourcing), or it cites the C++ source (in which case the table should label cells as "C++ contrib, undocumented").
6. **Add MoE, embedding lookup, all-reduce, sampling, and quant-dequant rows to the synthesis table.** Without these the consensus analysis is incomplete.
7. **Recount KV representations: 9–10, not 6.** Add XLA threading, RadixAttention, vLLM-PA v1/v2, ggml-pool, DirectML state.
8. **Recount fusion shapes: 8, not 5.** Add eager-kernel-graph and HLO-custom-call.

### Should-fix (significantly strengthens the report)

9. **Add a per-runtime evolution timeline.** Year × op additions × release tags.
10. **Add TRTLLM-Gen and Core ML pre-17 baseline as evolution points.**
11. **Add ggml as the dominant on-device CPU runtime.** Without ggml, "what runs Llama on a laptop today" is not represented.
12. **Add Cerebras, Groq, SambaNova as a "closed-op-vocabulary IHV" category** to formalize the FastFlowLM pattern.
13. **Replace CSDN / Zhihu / Tencent Cloud citations for MindIE with the Ascend open-source repo** (publicly available since Sep 2025).
14. **Distinguish OpenVINO `ReadValue`/`Assign` state from Core ML `read_state`/`coreml_update_state` state** in the KV representation taxonomy — they look similar but are not the same.
15. **Distinguish "documented contrib" from "C++ implementation"** for every ORT row.
16. **Make explicit the difference between `attention_mask` materialization and `is_causal=true` implicit masks** — this is a real fusion-boundary divergence.
17. **Add a "missing-op" column to the synthesis table** for each runtime, listing ops that *should* exist for full LLM coverage but don't (e.g., Core ML lacks first-class RMSNorm and RoPE).

### Nice-to-fix (polish)

18. **Per-runtime header file paths and line ranges** for every named op.
19. **Per-claim citation density**: at minimum one URL per non-trivial op claim.
20. **Standardize op-name capitalization** (`SCALED_DOT_PRODUCT_ATTENTION` vs `ScaledDotProductAttention` vs `scaled_dot_product_attention` are used inconsistently across sections — pick a convention).
21. **Tabulate quantization schemes** in a separate matrix rather than nested in each runtime section — currently it's hard to compare W4A16 across runtimes.
22. **Explicit "not yet checked" markers** for unverifiable Mandarin-source claims, rather than asserting them.

---

## Closing assessment

The report is the strongest cross-runtime LLM op-set survey I've seen in any single document, and the synthesis-table approach is exactly right. But it has three structural problems:

1. **Universality is overclaimed.** Missing AWS Neuron, TPU/HLO, DirectML, ggml, and MAX/Modular leaves the "what every IHV does" framing unsupported. The actual surveyed set is "what every non-Google, non-AWS, non-llama.cpp runtime does", which is a much narrower claim.
2. **Verification is uneven.** TRT-LLM and ORT contrib sections are well-sourced. Qualcomm and Huawei sections rely on unverifiable tertiary sources. The single biggest "novel op" claim in the report (`QNN_OP_KV_CACHE`) is exactly the one that cannot be confirmed from any public link the report cites.
3. **Versioning is missing.** Every claim about an op needs a *runtime version tag*. Op-sets are not stable; "OpenVINO supports FP8" without a release number is not a useful design input. Pin every claim.

If these three are fixed and the synthesis table expanded with MoE/lookup/allreduce/sampling rows, the report becomes the canonical reference for LLM IHV op-sets in 2025–2026. As it stands it is an excellent draft that needs a verification pass and ~40% more coverage.
