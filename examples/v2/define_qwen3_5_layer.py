"""Define a Qwen3.5-35B-A3B "full attention" layer (the 1-in-4 in the hybrid).

Demonstrates M1 extension fields on `GQASpec` and `MRoPEInterleavedSpec` plus
`NormZeroCenteredSpec`:

  - `NormZeroCenteredSpec` (Qwen3.5 uses (1 + w) * x_normed form).
    Source: modeling_qwen3_5.py:820  (`output * (1.0 + self.weight.float())`)
  - `GQASpec(output_gate="sigmoid")` (Qwen3.5/3.6 per-head sigmoid output gate).
    Source: modeling_qwen3_5.py:785  (`attn_output * torch.sigmoid(gate)`)
  - `MRoPEInterleavedSpec(theta, sections, partial_rotary_factor)`.
    Source: modeling_qwen3_5.py:192  (`mrope_section = [11, 11, 10]`)
            modeling_qwen3_5.py:214-216 (partial_rotary_factor=0.25)

Note: M0 cannot execute this layer — running it through `torch_port.run_block`
would raise NotImplementedError because the runner only dispatches `RoPESpec`
(not MRoPE) and plain `GQASpec` (no `output_gate`, no `NormZeroCenteredSpec`).
This file is for spec construction + inspection only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from components import (
    GQASpec,
    MRoPEInterleavedSpec,
    NormStandardSpec,  # unused but illustrates the contrast
    NormZeroCenteredSpec,
    SwiGLUSpec,
    pre_norm_block,
)


# ---- Qwen3.5-35B-A3B "full_attention" layer dimensions ---------------------
# Defaults from configuration_qwen3_5.py (Qwen3.5TextConfig):
HIDDEN_SIZE = 4096
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
INTERMEDIATE_SIZE = 12288
RMS_EPS = 1e-6
ROPE_THETA = 1e7  # Qwen3.5 family — set in published config; class default in HF
MROPE_SECTIONS = (11, 11, 10)  # Source: modeling_qwen3_5.py:192
PARTIAL_ROTARY_FACTOR = 0.25   # Source: modeling_qwen3_5.py:214-216 / config


def build_qwen3_5_full_attention_layer():
    # Qwen3.5 RMSNorm is zero-centered: (1 + w) * x_normed. Apply to all norms.
    block_norm = NormZeroCenteredSpec(eps=RMS_EPS)
    post_attn_norm = NormZeroCenteredSpec(eps=RMS_EPS)
    # qk_norm is head-dim-only RMSNorm (also zero-centered in Qwen3.5).
    # Source: modeling_qwen3_5.py:737-738.
    qk_norm = NormZeroCenteredSpec(eps=RMS_EPS)

    attention = GQASpec(
        num_heads=NUM_ATTN_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        qk_norm=qk_norm,
        output_gate="sigmoid",  # Qwen3.5/3.6 sigmoid output gate
    )

    # MRoPE-Interleaved: 3-axis (T, H, W) sections, partial rotary on 1/4 of head_dim.
    pos_enc = MRoPEInterleavedSpec(
        theta=ROPE_THETA,
        sections=MROPE_SECTIONS,
        partial_rotary_factor=PARTIAL_ROTARY_FACTOR,
        partial_rotary_kind="prefix",  # HF apply_rotary_pos_emb consumes prefix
    )

    # Standard SwiGLU MLP. (Note: the "linear_attention" layers in the hybrid
    # use Qwen3.5GatedDeltaNet instead — out of M0 scope.)
    ffn = SwiGLUSpec(intermediate_size=INTERMEDIATE_SIZE, fused_gate_up=False)

    return pre_norm_block(
        block_norm=block_norm,
        attention=attention,
        positional_encoding=pos_enc,
        post_attention_norm=post_attn_norm,
        ffn=ffn,
    )


def main() -> None:
    layer = build_qwen3_5_full_attention_layer()

    print("Qwen3.5 'full_attention' decoder layer (1 of every 4 in the hybrid):")
    print(f"  hidden_size       = {HIDDEN_SIZE}")
    print(f"  Hybrid pattern    : 3x linear_attention + 1x full_attention")
    print()
    print("BlockGraph topology:")
    for n in layer.nodes:
        print(f"  {n.name:8s} <- {str(n.inputs):24s} :: {type(n.spec).__name__}")
    print(f"  output: {layer.output}")
    print()

    attn = next(n.spec for n in layer.nodes if isinstance(n.spec, GQASpec))
    rope = next(n.spec for n in layer.nodes if isinstance(n.spec, MRoPEInterleavedSpec))
    n0 = next(n.spec for n in layer.nodes if isinstance(n.spec, NormZeroCenteredSpec))

    print("Key spec fields (M1 extensions in bold):")
    print(f"  block_norm.kind         = {n0.kind}                # <- (1+w)*x_normed form")
    print(f"  attn.num_heads          = {attn.num_heads}")
    print(f"  attn.num_kv_heads       = {attn.num_kv_heads}")
    print(f"  attn.head_dim           = {attn.head_dim}")
    print(f"  attn.qk_norm.kind       = {attn.qk_norm.kind}")
    print(f"  attn.output_gate        = {attn.output_gate!r}            # <- M1 sigmoid output gate")
    print(f"  rope.kind               = {rope.kind}")
    print(f"  rope.theta              = {rope.theta:g}")
    print(f"  rope.sections           = {rope.sections}")
    print(f"  rope.partial_rotary_factor = {rope.partial_rotary_factor}")
    print()
    print("Runtime status:")
    print("  M0 torch_port DOES NOT execute this layer.")
    print("  Spec inspection only — runtime support is M1 work.")


if __name__ == "__main__":
    main()


# Expected output:
#
# Qwen3.5 'full_attention' decoder layer (1 of every 4 in the hybrid):
#   hidden_size       = 4096
#   Hybrid pattern    : 3x linear_attention + 1x full_attention
#
# BlockGraph topology:
#   n0       <- ('input',)              :: NormZeroCenteredSpec
#   rope     <- ('n0',)                 :: MRoPEInterleavedSpec
#   attn     <- ('rope',)               :: GQASpec
#   r0       <- ('input', 'attn')       :: AddSpec
#   n1       <- ('r0',)                 :: NormZeroCenteredSpec
#   ffn      <- ('n1',)                 :: SwiGLUSpec
#   r1       <- ('r0', 'ffn')           :: AddSpec
#   output: r1
#
# Key spec fields (M1 extensions in bold):
#   block_norm.kind         = ZeroCentered                # <- (1+w)*x_normed form
#   attn.num_heads          = 16
#   attn.num_kv_heads       = 4
#   attn.head_dim           = 256
#   attn.qk_norm.kind       = ZeroCentered
#   attn.output_gate        = 'sigmoid'            # <- M1 sigmoid output gate
#   rope.kind               = MRoPE-Interleaved
#   rope.theta              = 1e+07
#   rope.sections           = (11, 11, 10)
#   rope.partial_rotary_factor = 0.25
#
# Runtime status:
#   M0 torch_port DOES NOT execute this layer.
#   Spec inspection only — runtime support is M1 work.
