"""OpenVINO GPU vs torch CPU numerical-equivalence gate for Qwen3-0.6B.

Loads HF's Qwen3-0.6B twice:
  1. As an OpenVINO GPU-compiled model (the IR cached by examples/openvino_qwen3_gpu.py).
  2. As the reference torch CPU model in fp32.

Same fixed input through both, compare next-token predictions and logits.

Tolerances:
  - argmax of last-position logits MUST match exactly (strict gate)
  - top-5 Jaccard overlap >= 0.6 (informational)
  - max abs logit diff reported (informational; fp16 GPU vs fp32 CPU typically ~5e-2 .. 1e-1)

Marked ``@pytest.mark.gate`` — requires Qwen3-0.6B HF weights + Intel GPU. Skip
gracefully if either is absent.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("openvino")
pytest.importorskip("optimum")

import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
HF_CACHE = REPO_ROOT / "hf_cache"
OV_MODEL_DIR = REPO_ROOT / "ov_models" / "qwen3-0.6b"
OV_KERNEL_CACHE = REPO_ROOT / "ov_cache"

MODEL_ID = "Qwen/Qwen3-0.6B"
FIXED_INPUT = torch.tensor(
    [[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long
)

# fp16-on-GPU vs fp32-on-CPU: argmax is the strict invariant; raw logit diffs
# are large in absolute terms but the rank-order is what's semantically meaningful.
LOGIT_ATOL_INFORMATIONAL = 5e-2  # reported, not asserted as a gate
TOP5_JACCARD_MIN = 0.6


def _gpu_available() -> bool:
    try:
        return "GPU" in ov.Core().available_devices
    except Exception:
        return False


def _ir_cached() -> bool:
    return (OV_MODEL_DIR / "openvino_model.xml").exists() and (
        OV_MODEL_DIR / "openvino_model.bin"
    ).exists()


pytestmark = [
    pytest.mark.gate,
    pytest.mark.skipif(not _gpu_available(), reason="OpenVINO GPU device not available"),
    pytest.mark.skipif(
        not _ir_cached(),
        reason=(
            "Cached OV IR not found at ov_models/qwen3-0.6b/. "
            "Run examples/openvino_qwen3_gpu.py once to populate it."
        ),
    ),
]


@pytest.fixture(scope="module")
def ov_model():
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    return OVModelForCausalLM.from_pretrained(
        str(OV_MODEL_DIR),
        device="GPU",
        ov_config={
            "PERFORMANCE_HINT": "LATENCY",
            "CACHE_DIR": str(OV_KERNEL_CACHE),
        },
    )


@pytest.fixture(scope="module")
def torch_cpu_model():
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(HF_CACHE),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return model


@torch.no_grad()
def test_qwen3_gpu_vs_cpu_argmax_matches(ov_model, torch_cpu_model):
    """Last-position argmax must be identical between OV-GPU-fp16 and torch-CPU-fp32."""
    out_ov = ov_model(input_ids=FIXED_INPUT)
    out_pt = torch_cpu_model(input_ids=FIXED_INPUT)

    logits_ov = (
        torch.from_numpy(out_ov.logits)
        if hasattr(out_ov.logits, "shape") and not isinstance(out_ov.logits, torch.Tensor)
        else out_ov.logits
    )
    logits_pt = out_pt.logits

    assert logits_ov.shape == logits_pt.shape, (
        f"Shape mismatch: OV {logits_ov.shape} vs torch {logits_pt.shape}"
    )

    last_ov = logits_ov[0, -1].float()
    last_pt = logits_pt[0, -1].float()

    argmax_ov = int(last_ov.argmax())
    argmax_pt = int(last_pt.argmax())
    assert argmax_ov == argmax_pt, (
        f"Argmax mismatch on last-position logits: "
        f"OV-GPU={argmax_ov}, torch-CPU={argmax_pt}"
    )

    diff = (last_ov - last_pt).abs()
    max_abs = float(diff.max())
    median_abs = float(diff.median())
    print(
        f"\n[info] last-pos logit diff: max={max_abs:.4e}, median={median_abs:.4e}, "
        f"shared_argmax={argmax_ov}",
        file=sys.stderr,
    )


@torch.no_grad()
def test_qwen3_gpu_vs_cpu_top5_overlap(ov_model, torch_cpu_model):
    """Top-5 predicted tokens at the last position must share most of the set."""
    out_ov = ov_model(input_ids=FIXED_INPUT)
    out_pt = torch_cpu_model(input_ids=FIXED_INPUT)

    last_ov = (
        torch.from_numpy(out_ov.logits)
        if hasattr(out_ov.logits, "shape") and not isinstance(out_ov.logits, torch.Tensor)
        else out_ov.logits
    )[0, -1].float()
    last_pt = out_pt.logits[0, -1].float()

    top5_ov = set(int(i) for i in last_ov.topk(5).indices.tolist())
    top5_pt = set(int(i) for i in last_pt.topk(5).indices.tolist())

    intersection = top5_ov & top5_pt
    union = top5_ov | top5_pt
    jaccard = len(intersection) / max(len(union), 1)

    print(
        f"\n[info] top-5 OV={sorted(top5_ov)} torch={sorted(top5_pt)} jaccard={jaccard:.2f}",
        file=sys.stderr,
    )
    assert jaccard >= TOP5_JACCARD_MIN, (
        f"Top-5 Jaccard {jaccard:.2f} below {TOP5_JACCARD_MIN}: "
        f"OV={sorted(top5_ov)}, torch={sorted(top5_pt)}"
    )
