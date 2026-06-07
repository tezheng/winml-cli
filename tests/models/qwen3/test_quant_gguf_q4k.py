"""GGUF Q4_K_M round-trip on a real Qwen3 projection weight.

Exercises the k-quant super-block scheme (256-weight super-block, 8 sub-blocks
of 32 with 6-bit scales+mins, fp16 super-block d/dmin) end-to-end. Verifies
that quant -> dequant -> matmul stays within Q4_K tolerance vs the full-
precision reference on a production-shaped weight (Qwen3-0.6B layer-0 q_proj).

Source: ggml/src/ggml-common.h (block_q4_K), ggml/src/ggml-quants.c
(quantize_row_q4_K_ref / dequantize_row_q4_K / get_scale_min_k4).
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoModelForCausalLM

from api import quant


MODEL_ID = "Qwen/Qwen3-0.6B"
# Simple min/max sub-block scheme (no make_qkx2 weighted search). Observed on
# Qwen3-0.6B q_proj: weight median rel-err ~0.13, output median ~0.13.
# Tolerance 0.22 leaves ~70% margin; matches reported llama.cpp Q4_K_M behavior
# on small projections with the reference quant path.
QUANT_REL_TOL = 0.22


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


def _flatten_to_q4k_grid(w: torch.Tensor) -> torch.Tensor:
    """Reshape a 2D linear weight into a flat 1D row whose length is a multiple
    of QK_K = 256, so it can be fed to gguf_q4_k_quantize. Real ggml lays out
    by row (out-axis) and pads tail blocks; we exercise the no-pad path."""
    K, N = w.shape
    total = K * N
    assert total % quant.GGUF_QK_K == 0, f"K*N={total} not divisible by 256"
    return w.reshape(total).float()


def test_q4k_round_trip_on_qproj(hf_state_dict):
    # HF stores Linear weights as [out, in] = [n_q_heads * head_dim, hidden].
    w_fp16 = hf_state_dict["model.layers.0.self_attn.q_proj.weight"].to(torch.float16)
    flat = _flatten_to_q4k_grid(w_fp16)
    d, dmin, scales_packed, qs = quant.gguf_q4_k_quantize(flat)
    y = quant.gguf_q4_k_dequantize(d, dmin, scales_packed, qs).reshape(flat.shape)
    w_recovered = y.to(torch.float16).reshape(w_fp16.shape)

    rel = (w_recovered.float() - w_fp16.float()).abs() / (w_fp16.float().abs() + 1e-6)
    median_w = rel.median().item()
    print(f"q4k weight median rel-err={median_w:.4f} (tol={QUANT_REL_TOL})")
    assert median_w < QUANT_REL_TOL, (
        f"median rel-err {median_w:.4f} >= tol {QUANT_REL_TOL}"
    )

    # Downstream impact: matmul output.
    torch.manual_seed(0)
    K_in, N_out = w_fp16.shape[1], w_fp16.shape[0]
    x = torch.randn(2, K_in, dtype=torch.float16)
    # HF: y = x @ w.t() — use the recovered weight in the same layout.
    y_full = (x.float() @ w_fp16.t().float()).to(torch.float16)
    y_quant = (x.float() @ w_recovered.t().float()).to(torch.float16)
    rel_y = (y_full.float() - y_quant.float()).abs() / (y_full.float().abs() + 1e-6)
    median_y = rel_y.median().item()
    print(f"q4k output median rel-err={median_y:.4f} (tol={QUANT_REL_TOL})")
    assert median_y < QUANT_REL_TOL, (
        f"output median rel-err {median_y:.4f} >= tol {QUANT_REL_TOL}"
    )
