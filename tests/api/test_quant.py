"""Round-trip tests for every quant scheme implemented in api/quant.py.

Scheme coverage:
  - AWQ W4A16   (INT4 grouped, asymmetric, AWQ_INTERLEAVE)
  - GGUF Q4_K_M (k-quant; super-block 256, 6-bit sub-scales/mins)
"""
import torch

from api import quant, specs, types


# ============================================================================
# AWQ W4A16
# ============================================================================


def _make_awq_spec():
    return specs.QuantSpec(
        qdtype=types.QDType.INT4,
        group_size=64,
        quant_axis=0,
        scale_dtype=torch.float16,
        has_zero_point=True,
        packing=types.PackingLayout.AWQ_INTERLEAVE,
        accumulator_dtype=torch.float32,
        role=types.QuantRole.WEIGHT,
    )


def test_awq_pack_unpack_round_trip():
    torch.manual_seed(0)
    K, N = 64, 16
    int_values = torch.randint(0, 16, (K, N), dtype=torch.int32)
    packed = quant.awq_pack(int_values)
    assert packed.dtype == torch.int32
    assert packed.shape == (K // 8, N)
    unpacked = quant.awq_unpack(packed, K, N)
    assert unpacked.shape == (K, N)
    assert torch.equal(unpacked, int_values)


def test_awq_dequant_recovers_within_quant_error():
    torch.manual_seed(0)
    K, N = 64, 8
    w_fp16 = torch.randn(K, N, dtype=torch.float16)
    spec = _make_awq_spec()
    packed, scales, zeros = quant.awq_quantize(w_fp16, spec)
    assert packed.shape == (K // 8, N)
    assert scales.shape == (K // 64, N)
    assert zeros.shape == (K // 64, N // 8)
    w_recovered = quant.awq_dequantize(packed, scales, zeros, spec, K, N)
    assert w_recovered.shape == (K, N)
    assert w_recovered.dtype == torch.float16
    rel_err = (w_recovered.float() - w_fp16.float()).abs() / (w_fp16.float().abs() + 1e-6)
    assert rel_err.median() < 0.15


def test_awq_dequant_then_linear_matches_full_precision_within_quant_tolerance():
    torch.manual_seed(0)
    K, N = 64, 8
    w_fp16 = torch.randn(K, N, dtype=torch.float16)
    spec = _make_awq_spec()
    packed, scales, zeros = quant.awq_quantize(w_fp16, spec)
    w_recovered = quant.awq_dequantize(packed, scales, zeros, spec, K, N)
    x = torch.randn(2, K, dtype=torch.float16)
    y_full = (x.float() @ w_fp16.float()).to(torch.float16)
    y_quant = (x.float() @ w_recovered.float()).to(torch.float16)
    rel = (y_full.float() - y_quant.float()).abs() / (y_full.float().abs() + 1e-6)
    assert rel.median() < 0.10


# ============================================================================
# GGUF Q4_K_M
# ============================================================================


def test_gguf_q4k_pack_unpack_scales_mins_round_trip():
    # Random 6-bit sc/min values for 4 blocks; verify the bit-pack into 12
    # bytes round-trips exactly through get_scale_min_k4.
    torch.manual_seed(1)
    B = 4
    sc = torch.randint(0, 64, (B, 8), dtype=torch.int32)
    mn = torch.randint(0, 64, (B, 8), dtype=torch.int32)
    packed = quant._gguf_pack_scales_mins(sc, mn)
    assert packed.shape == (B, 12)
    assert packed.dtype == torch.uint8
    sc2, mn2 = quant._gguf_unpack_scales_mins(packed)
    assert torch.equal(sc2, sc)
    assert torch.equal(mn2, mn)


def test_gguf_q4k_pack_unpack_qs_round_trip():
    torch.manual_seed(2)
    B = 3
    q = torch.randint(0, 16, (B, 256), dtype=torch.int32)
    qs = quant._gguf_pack_qs(q)
    assert qs.shape == (B, 128)
    assert qs.dtype == torch.uint8
    q2 = quant._gguf_unpack_qs(qs)
    assert torch.equal(q2, q)


def test_gguf_q4k_block_sizes_match_spec():
    # block_q4_K = 2 + 2 + 12 + 128 = 144 bytes per 256 elements.
    assert quant.GGUF_QK_K == 256
    assert quant.GGUF_K_SCALE_SIZE == 12
    assert quant.GGUF_BYTES_PER_BLOCK == 144


def test_gguf_q4k_synthetic_round_trip_within_tolerance():
    torch.manual_seed(3)
    N = 256 * 8  # 8 super-blocks
    w = torch.randn(N, dtype=torch.float32)
    d, dmin, scales_packed, qs = quant.gguf_q4_k_quantize(w)
    assert d.shape == (8,)
    assert dmin.shape == (8,)
    assert d.dtype == torch.float16
    assert dmin.dtype == torch.float16
    assert scales_packed.shape == (8, 12)
    assert qs.shape == (8, 128)

    y = quant.gguf_q4_k_dequantize(d, dmin, scales_packed, qs).reshape(N)
    # Per-element abs error bounded by sub-block step (~ d * 1).
    abs_err = (y - w).abs()
    rel_err = abs_err / (w.abs() + 1e-6)
    median_rel = rel_err.median().item()
    # The simple min/max scheme (no weighted make_qkx2 search) yields
    # ~9-10% median rel-err on Gaussian rows. The reference make_qkx2_quants
    # search does better (~3-5%); we stay above its bar and document it.
    assert median_rel < 0.12, f"median rel-err {median_rel:.4f} >= 0.12"
    # Max absolute error stays within ~ max sub-block step. The largest
    # sub-block step is (max_scale) since sc_q is in [0, 63] and d=max_scale/63.
    # Per-element abs err is bounded above by 0.5 * sub-block-step.
    max_scale = (d.float() * 63.0).max().item()
    assert abs_err.max().item() < max_scale, (
        f"max abs err {abs_err.max().item()} >= max sub-block step {max_scale}"
    )


def test_gguf_q4k_zeroed_block_round_trip_exact():
    # A block of all zeros must dequantize back to all zeros.
    w = torch.zeros(256, dtype=torch.float32)
    d, dmin, scales_packed, qs = quant.gguf_q4_k_quantize(w)
    y = quant.gguf_q4_k_dequantize(d, dmin, scales_packed, qs).reshape(256)
    assert torch.all(y.abs() < 1e-6)
