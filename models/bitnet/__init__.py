"""BitNet b1.58 family — sub-norms + ReLU² activation.

For v5-phase2 V3 we land the layer-level BitNet b1.58 distinctives:
- AttentionSpec.attn_sub_norm: RMSNorm on attn output BEFORE o_proj
- FFNSpec.ffn_sub_norm: RMSNorm on `act(gate)*up` BEFORE down_proj
- Activation.RELU2 (squared ReLU) wired into FeedForward (SWIGLU +
  RELU2 combination)

Ternary weight quantization (the BitNet b1.58 weight kind) is OUT OF
SCOPE for v5-phase2; that would require a new `QDType.TERNARY` (or
extending `QuantSpec.codebook`) plus a `_t_linear` op. The public
microsoft/bitnet-b1.58-2B-4T checkpoint ships fp16 weights so per-
layer math is exact at the {-1, 0, +1} layer once weights are loaded;
we don't exercise the actual quantization here.

Source: `transformers/models/bitnet/modeling_bitnet.py` (BitNet 1.58).
"""
