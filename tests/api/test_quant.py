import torch

from api import quant, specs, types


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
