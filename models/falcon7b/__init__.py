"""Falcon-7B family — parallel residual flow + MQA.

For v5-phase2 V2 we land the PARALLEL residual block layout
(`DecoderBlockSpec.block_layout = BlockLayout.PARALLEL`) used by
Falcon-7B (`config.parallel_attn=True, new_decoder_architecture=False,
num_ln_in_parallel_attn=1`).

The Falcon-7B attention is canonical MQA (n_kv_heads=1) with RoPE
SPLIT_HALF basis, no biases. The FFN is `dense_h_to_4h` (GELU) ->
`dense_4h_to_h` — an UNGATED FFN. The ungated form is NOT yet in the
IR (GateKind.GELU_ONLY is enum-reserved but FeedForward rejects it),
so this family's block assembly substitutes SwiGLU + SILU for the FFN
form. The block-level PARALLEL test (tests/api/test_parallel_residual.py)
proves the residual topology; an attention-only numerical gate against
HF Falcon-7B is also provided.

LayerNorm-with-bias (Falcon-7B uses LN, not RMSNorm) is also out of
scope — same Phase 3 work as MPT. The to_block_spec adapter therefore
substitutes RMSNorm. This is a SHAPE-ONLY family for the FFN/norm
sublayers; the ATTENTION sublayer carries the canonical Falcon-7B
config and is exercised by tests/models/falcon7b/.

Source: `transformers/models/falcon/modeling_falcon.py` —
- `FalconAttention.forward` (lines 316-427)
- `FalconDecoderLayer.forward` (lines 580-636) — parallel_attn branch
- `FalconConfig` defaults (configuration_falcon.py:66-89) for
  Falcon-7B: hidden_size=4544, num_attention_heads=71,
  multi_query=True → num_kv_heads=1, parallel_attn=True
"""
