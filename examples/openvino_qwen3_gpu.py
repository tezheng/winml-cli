"""Qwen3-0.6B on the Intel OpenVINO GPU plugin (Arc 140V iGPU).

Vertical slice that:
  1. Exports Qwen/Qwen3-0.6B to OpenVINO IR (cached on disk after first run).
  2. Compiles the IR for the GPU plugin (device="GPU") with a kernel cache.
  3. Runs a smoke generation (>= 10 tokens, greedy) on the GPU.
  4. Verifies the GPU was actually engaged (not a silent CPU fallback) by
     querying compiled-model properties and reporting EXECUTION_DEVICES.
  5. Prints latency stats and the decoded continuation.

Plan A: optimum.intel.OVModelForCausalLM (high-level, stateful KV cache).

Run from the project root:
    .venv\\Scripts\\python.exe examples\\openvino_qwen3_gpu.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# --- repo-local paths --------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = REPO_ROOT / "hf_cache"
OV_MODEL_DIR = REPO_ROOT / "ov_models" / "qwen3-0.6b"
OV_KERNEL_CACHE = REPO_ROOT / "ov_cache"

# Point HF at the project-local cache (where Qwen3-0.6B is already downloaded).
os.environ.setdefault("HF_HOME", str(HF_CACHE))

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "GPU"  # OpenVINO Arc 140V iGPU
PROMPT = "The Qwen3 architecture uses"
MAX_NEW_TOKENS = 30


def _has_exported_ir(path: Path) -> bool:
    """Return True if a previous export to OV IR already lives at `path`."""
    return (path / "openvino_model.xml").exists() and (
        path / "openvino_model.bin"
    ).exists()


def main() -> int:
    import openvino as ov
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    OV_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    OV_KERNEL_CACHE.mkdir(parents=True, exist_ok=True)

    # ---- 0. Show OpenVINO build + available devices -------------------------
    core = ov.Core()
    print(f"OpenVINO version: {ov.__version__}")
    print(f"Available devices: {core.available_devices}")
    if DEVICE not in core.available_devices:
        print(f"FATAL: {DEVICE} not in available_devices; aborting.", file=sys.stderr)
        return 2
    print(f"GPU FULL_DEVICE_NAME: {core.get_property(DEVICE, 'FULL_DEVICE_NAME')}")

    # ---- 1. Export to OV IR (cached) ----------------------------------------
    ov_config = {
        "PERFORMANCE_HINT": "LATENCY",
        "CACHE_DIR": str(OV_KERNEL_CACHE),
    }

    if _has_exported_ir(OV_MODEL_DIR):
        print(f"\n[load] reusing cached IR at {OV_MODEL_DIR}")
        t_load0 = time.perf_counter()
        model = OVModelForCausalLM.from_pretrained(
            str(OV_MODEL_DIR),
            device=DEVICE,
            ov_config=ov_config,
        )
        t_load = time.perf_counter() - t_load0
    else:
        print(f"\n[export] no cached IR found; exporting {MODEL_ID} -> {OV_MODEL_DIR}")
        print("        (first run is slow: 1-3 min for the 0.6B model)")
        t_export0 = time.perf_counter()
        model = OVModelForCausalLM.from_pretrained(
            MODEL_ID,
            export=True,
            device=DEVICE,
            ov_config=ov_config,
            cache_dir=str(HF_CACHE),
        )
        t_export = time.perf_counter() - t_export0
        print(f"[export] done in {t_export:.1f}s -- saving IR to disk")
        model.save_pretrained(str(OV_MODEL_DIR))
        t_load = t_export

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
    print(f"[load] model+tokenizer ready in {t_load:.1f}s")

    # ---- 2. Verify GPU is actually engaged ----------------------------------
    # optimum-intel wraps the OpenVINO CompiledModel as `model.request` (the
    # InferRequest) or `model.model` (the ov.Model). The CompiledModel itself
    # is reachable via `model.request.get_compiled_model()` in newer optimum-
    # intel; fall back to introspecting attributes that DO exist.
    exec_devices: str = "<unknown>"
    inference_precision: str = "<unknown>"
    perf_hint: str = "<unknown>"
    compiled = None
    try:
        compiled = model.request.get_compiled_model()  # type: ignore[attr-defined]
    except Exception:
        # Older API: optimum stores the CompiledModel directly.
        compiled = getattr(model, "compiled_model", None)
    if compiled is not None:
        try:
            exec_devices = str(compiled.get_property("EXECUTION_DEVICES"))
        except Exception as e:
            exec_devices = f"<error: {e!r}>"
        try:
            inference_precision = str(
                compiled.get_property("INFERENCE_PRECISION_HINT")
            )
        except Exception:
            pass
        try:
            perf_hint = str(compiled.get_property("PERFORMANCE_HINT"))
        except Exception:
            pass

    print("\n=== Compiled-model properties (GPU verification) ===")
    print(f"  EXECUTION_DEVICES         : {exec_devices}")
    print(f"  INFERENCE_PRECISION_HINT  : {inference_precision}")
    print(f"  PERFORMANCE_HINT          : {perf_hint}")
    if "GPU" not in exec_devices:
        print(
            "WARNING: EXECUTION_DEVICES does not contain 'GPU'.\n"
            "         OpenVINO may have silently fallen back to CPU.",
            file=sys.stderr,
        )

    # ---- 3. Smoke generation ------------------------------------------------
    inputs = tokenizer(PROMPT, return_tensors="pt")
    n_prompt = int(inputs.input_ids.shape[1])
    print(f"\n[gen] prompt='{PROMPT}'  (prompt_tokens={n_prompt})")

    # Warm-up: first inference triggers kernel compilation on GPU. Do a short
    # warm-up so the reported latency reflects steady-state.
    print("[gen] warm-up forward...")
    t_warm0 = time.perf_counter()
    _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    t_warm = time.perf_counter() - t_warm0
    print(f"[gen] warm-up done in {t_warm * 1000:.0f} ms")

    print(f"[gen] generating {MAX_NEW_TOKENS} tokens (greedy)...")
    t_gen0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    t_gen = time.perf_counter() - t_gen0
    n_new = int(out.shape[1]) - n_prompt
    tps = n_new / t_gen if t_gen > 0 else float("nan")

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Cap the printed text at the first ~50 generated tokens to stay terse.
    first50_ids = out[0, : n_prompt + min(50, n_new)]
    first50_text = tokenizer.decode(first50_ids, skip_special_tokens=True)

    # ---- 4. Report ----------------------------------------------------------
    print("\n=== Latency ===")
    print(f"  generated_tokens : {n_new}")
    print(f"  wall_time        : {t_gen * 1000:.0f} ms")
    print(f"  throughput       : {tps:.1f} tok/s")

    print("\n=== Output (first 50 tokens incl. prompt) ===")
    print(first50_text)

    print("\n=== Full decoded ===")
    print(text)

    # Exit non-zero if the GPU was clearly NOT used, so callers can detect it.
    if "GPU" not in exec_devices:
        return 3
    if n_new < 10:
        print(f"FATAL: only {n_new} tokens generated (<10).", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
