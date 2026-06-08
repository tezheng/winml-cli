"""MPT family (MosaicML) — ALiBi position encoding.

For v5-phase2 V1 we land ONLY the ALiBi attention sublayer:
- AttentionSpec.alibi: build slopes; bias added pre-softmax via ops.apply_alibi.

The MPT FFN (ungated GELU — `up_proj` → GELU → `down_proj` with no
gating) and LayerNorm-without-bias (norm_1.bias = None, norm_2.bias =
None) are NOT yet expressible in the api/ IR. Both are Phase 3 work:
- `GateKind.NO_GATING` / pure-FFN form (StarCoder 2, MPT, Pythia)
- LayerNorm-with-optional-bias

Until those land, MPT decoder-block assembly is shape-only. The
canonical numerical gate exercises the ATTENTION sublayer alone (see
tests/models/mpt/test_alibi_attention.py).

Source: `transformers/models/mpt/modeling_mpt.py:65-134` (MptAttention,
ALiBi math at lines 42-62 and 121).
"""
