"""Round-trip tests for every quant scheme implemented in api/quant.py.

Scheme coverage:
  - AWQ W4A16   (INT4 grouped, asymmetric, AWQ_INTERLEAVE)
  - GGUF Q4_K_M (k-quant; super-block 256, 6-bit sub-scales/mins)
  - FP8 E4M3 W8A8 (per-tensor W + per-token A, fp32 accumulate)
  - MXFP4 (UE8M0 shared exp + E2M1 mantissas, block 32)
  - IQ2_M / AQLM   (stub guard rails — must raise NotImplementedError)
"""
import pytest
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


# ============================================================================
# FP8 E4M3 W8A8
# ============================================================================


def test_fp8_e4m3_per_tensor_round_trip_weight():
    torch.manual_seed(4)
    w = torch.randn(64, 64, dtype=torch.float32)
    w_fp8, scale = quant.fp8_e4m3_quantize_per_tensor(w)
    assert w_fp8.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    w_back = quant.fp8_e4m3_dequantize(w_fp8, scale)
    rel = (w_back - w).abs() / (w.abs() + 1e-6)
    median_rel = rel.median().item()
    # E4M3 has ~3 mantissa bits => quant step ~ 2^-3 = 0.125 of magnitude.
    assert median_rel < 0.10, f"per-tensor median rel-err {median_rel} too large"


def test_fp8_e4m3_per_token_round_trip_activation():
    torch.manual_seed(5)
    x = torch.randn(4, 128, dtype=torch.float32)
    x_fp8, scale = quant.fp8_e4m3_quantize_per_token(x)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (4, 1)
    x_back = quant.fp8_e4m3_dequantize(x_fp8, scale)
    rel = (x_back - x).abs() / (x.abs() + 1e-6)
    assert rel.median().item() < 0.10


def test_fp8_e4m3_w8a8_matmul_within_tolerance():
    torch.manual_seed(6)
    B, K, N = 2, 128, 64
    x = torch.randn(B, K, dtype=torch.float32)
    w = torch.randn(K, N, dtype=torch.float32) * 0.5

    x_fp8, x_scale = quant.fp8_e4m3_quantize_per_token(x)
    w_fp8, w_scale = quant.fp8_e4m3_quantize_per_tensor(w)
    y_quant = quant.fp8_e4m3_matmul(x_fp8, x_scale, w_fp8, w_scale)
    y_full = x @ w

    rel = (y_quant - y_full).abs() / (y_full.abs() + 1e-3)
    median_rel = rel.median().item()
    # Sum of K ~= 128 FP8 products: noise averages down, so median rel ~ a few %.
    assert median_rel < 0.10, f"W8A8 matmul median rel-err {median_rel:.4f} too large"


def test_fp8_e4m3_clamp_against_overflow():
    # Values larger than FP8 E4M3 max (~448) must be clamped by the scale.
    big = torch.tensor([1000.0, -1000.0, 0.5, -0.5], dtype=torch.float32)
    w_fp8, scale = quant.fp8_e4m3_quantize_per_tensor(big)
    w_back = quant.fp8_e4m3_dequantize(w_fp8, scale)
    # Big values recoverable to within ~1 step of scale * FP8_E4M3_MAX
    assert torch.isfinite(w_back).all()
    assert (w_back[0] > 800.0) and (w_back[1] < -800.0)


# ============================================================================
# MXFP4
# ============================================================================


def test_mxfp4_e2m1_codes_round_trip_exactly_on_grid():
    # Every representable magnitude with both signs must round-trip exactly.
    pos = quant._MXFP4_E2M1_MAGS  # [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    vals = torch.cat([pos, -pos])
    codes = quant._mxfp4_encode_e2m1(vals)
    back = quant._mxfp4_decode_e2m1(codes)
    # +0 and -0 both map to magnitude 0; sign of decoded zero is sign-bit.
    pos_back = back[: pos.numel()]
    neg_back = back[pos.numel() :]
    assert torch.allclose(pos_back.abs(), pos, atol=0)
    assert torch.allclose(neg_back.abs(), pos, atol=0)


def test_mxfp4_synthetic_round_trip_within_tolerance():
    torch.manual_seed(7)
    N = 32 * 16  # 16 blocks
    w = torch.randn(N, dtype=torch.float32)
    codes, exps = quant.mxfp4_quantize(w)
    assert codes.shape == (16, 32)
    assert exps.shape == (16,)
    assert exps.dtype == torch.uint8

    y = quant.mxfp4_dequantize(codes, exps)
    rel = (y - w).abs() / (w.abs() + 1e-6)
    median_rel = rel.median().item()
    # E2M1 has only 8 magnitudes per block; expect ~10-15% median rel-err on
    # Gaussian rows with block 32.
    assert median_rel < 0.20, f"MXFP4 median rel-err {median_rel:.4f} too large"


def test_mxfp4_zero_block_round_trip_exact():
    w = torch.zeros(64, dtype=torch.float32)
    codes, exps = quant.mxfp4_quantize(w)
    y = quant.mxfp4_dequantize(codes, exps)
    assert torch.all(y.abs() < 1e-6)


def test_mxfp4_handles_large_dynamic_range():
    # A block with one large element and many tiny ones — the shared
    # exponent should track the large element, the tiny ones round to 0.
    w = torch.zeros(32, dtype=torch.float32)
    w[0] = 1000.0
    w[1] = 0.001
    codes, exps = quant.mxfp4_quantize(w)
    y = quant.mxfp4_dequantize(codes, exps)
    assert y.shape == (32,)
    # Recovered max-element is within E2M1 grid step of 1000.
    assert abs(y[0].item() - 1000.0) / 1000.0 < 0.20


# ============================================================================
# Codebook-quant stubs (IQ2_M / AQLM)
# ============================================================================


def test_iq2_m_dequantize_is_reserved():
    with pytest.raises(NotImplementedError, match="IQ2_M"):
        quant.iq2_m_dequantize(None)


def test_aqlm_dequantize_is_reserved():
    with pytest.raises(NotImplementedError, match="AQLM"):
        quant.aqlm_dequantize(None)
