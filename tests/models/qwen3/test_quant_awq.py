"""AWQ round-trip on a real Qwen3 projection weight.

Verifies that quant -> dequant -> matmul stays within quantization tolerance vs
the full-precision reference. This proves the QuantSpec realization works on
production-shaped weights (Qwen3-0.6B layer-0 q_proj).
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoModelForCausalLM

from api import quant, specs, types


MODEL_ID = "Qwen/Qwen3-0.6B"
# Empirical INT4-G128 round-trip on Qwen3-0.6B layer-0 q_proj:
#   median |w| = 0.019, weight median rel-err = 0.157, output median rel-err = 0.160.
# Near-zero weights dominate the relative-error tail. 0.20 gives ~25% margin
# above observed and is consistent with reported AWQ W4A16 elementwise behavior.
QUANT_REL_TOL = 0.20


@pytest.fixture(scope="module")
def hf_state_dict():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )
    return m.state_dict()


def test_awq_round_trip_on_qproj(hf_state_dict):
    w_fp16 = hf_state_dict["model.layers.0.self_attn.q_proj.weight"].to(torch.float16)
    # HF stores Linear weights as [out, in] = [n_q_heads * head_dim, hidden_size].
    # AWQ packs along the contraction (input/hidden) axis, so we transpose to
    # [K, N] = [hidden_size, n_q_heads * head_dim].
    w_KN = w_fp16.t().contiguous()
    K, N = w_KN.shape
    G = 128
    assert K % G == 0, f"K={K} must be divisible by group_size={G}"
    assert N % 8 == 0, f"N={N} must be divisible by 8 (AWQ N-axis packing)"

    spec = specs.QuantSpec(
        qdtype=types.QDType.INT4,
        group_size=G,
        quant_axis=0,
        scale_dtype=torch.float16,
        has_zero_point=True,
        packing=types.PackingLayout.AWQ_INTERLEAVE,
        accumulator_dtype=torch.float32,
        role=types.QuantRole.WEIGHT,
    )
    packed, scales, qzeros = quant.awq_quantize(w_KN, spec)
    w_recovered = quant.awq_dequantize(packed, scales, qzeros, spec, K, N)

    # Median relative weight error -- robust to a few high-magnitude outliers in
    # any single 128-element group.
    rel = (w_recovered.float() - w_KN.float()).abs() / (w_KN.float().abs() + 1e-6)
    median_w = rel.median().item()
    print(f"weight median rel-err={median_w:.4f} (tol={QUANT_REL_TOL})")
    assert median_w < QUANT_REL_TOL, (
        f"median rel-err {median_w:.4f} >= tol {QUANT_REL_TOL}"
    )

    # Downstream impact: matmul output should also be close.
    torch.manual_seed(0)
    x = torch.randn(2, K, dtype=torch.float16)
    y_full = (x.float() @ w_KN.float()).to(torch.float16)
    y_quant = (x.float() @ w_recovered.float()).to(torch.float16)
    rel_y = (y_full.float() - y_quant.float()).abs() / (y_full.float().abs() + 1e-6)
    median_y = rel_y.median().item()
    print(f"output median rel-err={median_y:.4f} (tol={QUANT_REL_TOL})")
    assert median_y < QUANT_REL_TOL, (
        f"output median rel-err {median_y:.4f} >= tol {QUANT_REL_TOL}"
    )
