"""MPT family (MosaicML).

v5-phase2 V1 landed the ALiBi attention sublayer.
v6 A1 lands the FULL decoder block:
- LayerNorm(eps, has_bias=False) on both norm_1/norm_2
- ALiBi-MHA attention
- Ungated GELU (exact) FFN (`up_proj -> nn.GELU(approximate='none') -> down_proj`)

Wired via:
- types.NormKind.LAYER + NormSpec.has_bias
- types.GateKind.GELU_ONLY + types.Activation.GELU_EXACT
- api/norm.py::LayerNorm and api/norm.py::build_norm dispatcher
- api/feedforward.py::FeedForward GELU_ONLY path (up_proj -> gelu -> down_proj)

Source:
  `transformers/models/mpt/modeling_mpt.py`
   - 42-62 (build_mpt_alibi_tensor — slopes)
   - 65-134 (MptAttention — Wqkv fused, position_bias add)
   - 137-155 (MptMLP — `nn.GELU(approximate='none')`, up_proj+down_proj only)
   - 158-212 (MptBlock — norm_1 -> attn -> residual, norm_2 -> ffn -> residual,
     with `norm_1.bias = None` / `norm_2.bias = None`)
"""
