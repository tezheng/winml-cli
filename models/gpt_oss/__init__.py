"""GPT-OSS 20B family — trained attention sinks.

For v5-phase2 V5 we land trained sinks: a learnable per-head scalar
appended as one extra logit column to the attention scores BEFORE
softmax, then dropped post-softmax. Each attention head spends some
softmax mass on a "trained sink" slot.

Wired via:
- AttentionSpec.n_sink_tokens = 1
- AttentionSpec.mask_kind = MaskKind.SINK
- Attention.__init__ allocates `self.sinks` as an nn.Parameter
- ops.sdpa(sinks=...) handles the append/softmax/drop math

This is a SHAPE-ONLY family for the full decoder block. GPT-OSS 20B
also requires:
- Per-layer alternation: sliding_attention (W=128) vs full_attention
  (already supported via per-layer to_block_spec dispatch)
- Top-4 MoE with 128 experts (MoESpec supports this, but the GPT-OSS
  experts use a different forward — `experts.gate_up_proj_bias` /
  `down_proj_bias` plus a different routing/gating math — not landed)
- MXFP4 weight quantization on experts (B10 has MXFP4 round-trip but
  not a `mxfp4_linear`-into-MoE-experts wiring)
- A different RoPE variant — GPT-OSS uses YARN with channel-split
  rotary equivalent to INTERLEAVED math but on the SPLIT halves of
  the channel axis (rope_apply equivalent at the math level but
  weight layout differs)

The SINK forward is the V5 contribution and is exercised by
tests/api/test_attention_sinks.py end-to-end vs the HF eager
attention reference (hand-rolled — the test does not depend on the
GPT-OSS modeling module).

Source: `transformers/models/gpt_oss/modeling_gpt_oss.py`:
- 251-279 (eager_attention_forward with sinks math)
- 283-351 (GptOssAttention.__init__ and forward — `self.sinks` Parameter)
- configuration_gpt_oss.py:44-78 (canonical GPT-OSS 20B defaults)
"""
