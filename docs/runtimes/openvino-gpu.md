# OpenVINO on Intel GPU — verified port of llm-layers IR

**Status:** Vertical slice verified end-to-end on **Intel Arc 140V iGPU (16 GB, Lunar Lake / Xe2)**, Windows 11. OpenVINO **2026.2.0**.
**Date:** 2026-06-15.

This doc covers what was empirically validated, what the audit predicts (with source citations), and what could not be tested due to hardware constraints.

---

## 1. TL;DR — what works today

| Layer | Status | Evidence |
|---|---|---|
| Install (openvino, openvino-genai, optimum-intel) | ✅ verified | `uv sync --extra openvino` succeeds; `pyproject.toml` adds `openvino` optional-dependency group |
| GPU device discovery | ✅ verified | `ov.Core().available_devices == ['CPU', 'GPU', 'NPU']`; `FULL_DEVICE_NAME = 'Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)'` |
| Qwen3-0.6B export → IR → GPU compile | ✅ verified | `examples/openvino_qwen3_gpu.py` |
| `EXECUTION_DEVICES = ['GPU.0']` (no silent CPU fallback) | ✅ verified | smoke run output |
| Generation quality (greedy 30 tokens) | ✅ verified | "The Qwen3 architecture uses a large-scale neural network with a large number of layers and a large number of parameters, but it is not the same as the previous generation of models" |
| Throughput | ✅ measured | **49.7 tok/s** fp16 on Arc 140V; warm-up 69 ms; cold IR reload 1.4 s |
| Numerical equivalence vs torch CPU fp32 | ✅ verified | `tests/openvino/test_qwen3_gpu_numerical.py` — argmax matches, top-5 Jaccard = 1.00, max logit diff 3.04e-2 |

## 2. Reproduce

```powershell
# from C:\Users\zhengte\external\llm-layers
uv sync --extra openvino --extra test --extra dev

# vertical slice (downloads Qwen3-0.6B on first run; caches to ov_models/qwen3-0.6b/)
.venv\Scripts\python.exe examples\openvino_qwen3_gpu.py

# numerical-equivalence gate (depends on the cached IR above)
.venv\Scripts\python.exe -m pytest tests\openvino -m gate -v
```

First-run cost: ~36 s export + IR save. Subsequent runs reload in ~1.4 s.

## 3. Op coverage — 24 logical ops × OpenVINO 2026.x GPU plugin

Source-grounded against `docs.openvino.ai/2025/.../operation-sets/` and `openvinotoolkit/openvino` master. See `research/03-ihv-opsets.v2.md:83-140` for the broader OpenVINO survey.

| Bucket | Count | Examples | OV mapping |
|---|---|---|---|
| 🟢 Trivial 1:1 | 12 | `silu`, `gelu_pytorch_tanh`, `gelu_exact`, `add`, `mul`, `linear`, `layer_norm`, `embed`, `lm_head`, `softmax`, `top_k`, `gather`, `conv1d` | direct opsetN translator in `src/frontends/pytorch/src/op_table.cpp`; native GPU kernels |
| 🟢 Decomposes then fused | 2 | `rms_norm` → `RMSFusion` → `RMS` GPU kernel; `rope_apply` → `RoPEFusionLlama` → `RoPE` GPU kernel | `src/plugins/intel_gpu/src/graph/impls/ocl/rms.cpp` |
| 🟢 Decomposes cleanly | 4 | `relu2`, `l2norm`, `build_alibi_slopes` (constant-folded), `apply_alibi` | core eltwise ops; all GPU-supported |
| 🟡 SDPA variants | 3 | `sdpa`, `sdpa + attn_bias` (ALiBi), `sdpa + SWA` | `opset13::ScaledDotProductAttention` accepts additive `attention_mask`; ALiBi and SWA both flow through as masks |
| 🔴 SDPA + sinks | 1 | GPT-OSS attention sinks | `opset14::SDPA` adds `sink` input, but **PyTorch frontend translator emits v13 only** (`src/frontends/pytorch/src/op/scaled_dot_product_attention.cpp:199`). Workaround: route via `PagedAttentionExtension.sinks` (slot 20) using `optimum-intel`. Untested on this hardware. |
| 🟡 MRoPE / partial RoPE | 2 | `rope_apply_mrope`, `rope_apply_partial` | decomposes via Split/Gather/Concat; all sub-ops GPU-supported |
| 🟡 Gated DeltaNet | 1 | `gated_delta_step` (Qwen3-Next, Qwen3.5/3.6) | OV 2025.4 added a **fused GatedDeltaNet GPU kernel** pattern-matched via `optimum-intel`. Raw `ov.convert_model` falls back to `Loop-5`. Untested on this hardware. |
| 🔴 SSM | 2 | `selective_scan_mamba1`, `selective_scan` (Mamba-2 SSD) | PyTorch `for i in range(S)` either unrolls or routes to `Loop-5` — no fusion on GPU. Needs custom op or `device="HETERO:GPU,CPU"`. Untested. |

**Net:** 21/24 ops port cleanly through `ov.convert_model` + the OV common-transformations pipeline. 1 op (`gated_delta_step`) requires `optimum-intel` for the fused kernel path. 2 ops (SSM) are blockers without custom ops or HETERO.

## 4. Family classification — 41 families

| Color | Count | Families | Verified on this hardware? |
|---|---|---|---|
| 🟢 **GREEN** — port cleanly | 24 | qwen3, qwen3_moe, llama3, mistral, mixtral, gemma2/3/4, granite, phi3_mini/small, phi4_mini, smollm3, tinyllama, olmo2, olmoe, ministral, mpt (ALiBi), falcon7b, bitnet, hunyuan_large, glm_moe_dsa, got_ocr2, qwen2_5_vl, nemotron3 | **qwen3** ✅; others extrapolated from same op-set |
| 🟡 **YELLOW** — minor workaround | 6 | qwen3_next, qwen3_5_moe, qwen3_6 family, gpt_oss, voxtral, moshi, minimax_m2 | **Not validated** — smallest YELLOW candidates (gpt-oss-20b ≈ 40 GB, qwen3-next-80b ≈ 160 GB) exceed Arc 140V 16 GB iGPU memory |
| 🔴 **RED** — needs custom op or HETERO carve-out | 11 | mamba1, mamba2, mamba3, granite4_h, jamba, falcon_h1, hymba, recurrent_gemma, rwkv7, minicpm3, deepseek_v2/v3/v3.2/v4 (MLA + hash MoE) | Not attempted |

## 5. Hardware constraints — what we could not test

This validation ran on an **Intel Arc 140V iGPU with 16 GB shared memory**, the standard Lunar Lake configuration. Three YELLOW/RED items would have been valuable to validate empirically but exceed this footprint:

| Untested item | Smallest published | fp16 footprint | Verdict |
|---|---|---|---|
| GPT-OSS sinks via `PagedAttentionExtension` | gpt-oss-20b | ~40 GB | Needs ≥24 GB dGPU |
| Qwen3-Next fused GatedDeltaNet kernel | Qwen3-Next-80B-A3B | ~160 GB | Needs ≥80 GB or HETERO |
| MLA absorbed-matmul (DeepSeek-V2-Lite) | DeepSeek-V2-Lite-Chat (15.7B, ~31 GB fp16) | ~31 GB | Needs ≥40 GB dGPU |

These remain audit-classified (source-grounded against OpenVINO 2025.4 release notes and `openvinotoolkit/openvino` source) but unvalidated on this box.

## 6. Choices we made (and why)

- **`optimum.intel.OVModelForCausalLM` over hand-written `ov.convert_model`.** The PyTorch frontend handles ~22/24 ops automatically, and the `RMSFusion`/`RoPEFusionLlama` common transformations recover dedicated GPU kernels even from our manually-written `x * rsqrt(mean(x²))`. Stateful KV-cache (`opset3::ReadValue` + `opset3::Assign`) is wired up by `optimum-intel`'s `MakeStateful` transformation pass — we don't hand-emit it.
- **fp16 over fp32.** Xe2 iGPU is fp16-native; fp32 paths exist but waste throughput. Numerical-equivalence still holds — argmax and top-5 are bit-exact vs torch fp32 reference.
- **Kernel cache on disk** (`ov_cache/`). First compile takes ~5 s; subsequent runs reuse OpenCL kernel binaries from this dir. Already in `.gitignore`.
- **NOT pinning `transformers` tightly.** optimum-intel 2.0.0 pulled `transformers` down from 5.10.2 to 5.0.0. Verified that Qwen3 / Gemma3 / Mistral modeling classes still load; the existing R-class CPU gates still pass.

## 7. Known surprises

- **Tied embeddings:** the HF→OV exporter does not tie `lm_head.weight` with `model.embed_tokens.weight` even when the source model has `tie_word_embeddings=True`. The IR carries both copies. Future work: investigate `optimum-intel`'s tied-weight handling to halve weight memory on small models.
- **PyTorch tracer warnings during export:** transformers 5.x's `cache_utils.py` and `masking_utils.py` raise `TracerWarning` about Python booleans from tensors. Cosmetic — runtime output is correct.
- **Throughput room to grow:** 49.7 tok/s is at the low end of the expected band. `openvino_genai.LLMPipeline` (Plan B in the porting notes) typically wins 1.5-2× over `optimum-intel` by avoiding per-token Python overhead. The cached IR at `ov_models/qwen3-0.6b/` is directly consumable by it — future optimization.
- **HF symlink warning on Windows** when not in Developer Mode. Harmless duplicate files in `hf_cache/`.

## 8. Pointers

| Artifact | Path |
|---|---|
| Runnable sample | `examples/openvino_qwen3_gpu.py` |
| Numerical gate | `tests/openvino/test_qwen3_gpu_numerical.py` |
| Op audit (deeper) | `research/03-ihv-opsets.v2.md:39-140` |
| OpenVINO 2025.4 release notes | https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-4.html |
| OpenVINO 2026.2.0 release | https://github.com/openvinotoolkit/openvino/releases |
| PyTorch frontend op table | https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/pytorch/src/op_table.cpp |
| Intel GPU plugin SDPA | https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/plugin/ops/scaled_dot_product_attention.cpp |
| SDPA opset spec | https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/sequence/scaled-dot-product-attention.html |
