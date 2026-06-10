"""Qwen3-Next family — 3:1 Gated DeltaNet : standard-attention hybrid.

Architecture (verified against
`transformers/models/qwen3_next/{configuration_qwen3_next,modeling_qwen3_next}.py`,
transformers 5.10.x):

- Per-layer dispatch on `layer_types[i]`:
    * "linear_attention"  -> Qwen3NextGatedDeltaNet  (modeling_qwen3_next.py:499-717)
    * "full_attention"    -> Qwen3NextAttention      (modeling_qwen3_next.py:255-330)
  Default schedule: `(i + 1) % 4 != 0` ⇒ linear, else full — so 3 linear : 1 full.
  Source: configuration_qwen3_next.py:121-127.
- Channel mixer: Qwen3NextSparseMoeBlock (softmax router + shared_expert with
  its own learned gate sigmoid). Source: modeling_qwen3_next.py:797-816.

v7 Phase B scope:
- Gated DeltaNet token mixer has a FULL numerical forward (Phase A P3, see
  `api.ssm.GatedDeltaNetMixer`). Per-layer factory ships these layers with
  numerical equivalence to HF.
- Full-attention layers carry SHAPE-ONLY semantics in the IR: Qwen3-Next's
  attention has a fused (q | gate) projection that doubles q_proj's output
  dim and gates the attention output via `attn_output * sigmoid(gate)`
  (modeling_qwen3_next.py:296-299, 326-327). The IR's STANDARD attention
  has no output-gating; we ship the layer as a plain STANDARD attention with
  q/k norms PRE-RoPE, accepting that its numerical output WILL diverge from
  HF on the full-attention layers. The MoE side similarly skips the
  per-token `shared_expert_gate` sigmoid.

Where to look:
- `config.Qwen3NextConfig` (api adapter)
- `to_block_spec(layer_idx)` dispatches per-layer
- `layer.build_qwen3_next_decoder_layer` returns a DecoderBlock
- `tests/models/qwen3_next/test_numerical_synthetic.py` gates linear-attention
  layers against HF at atol=5e-4
"""
