"""Define a Gemma 4 E2B "local" (sliding-window) layer using sandwich-norm.

Demonstrates:
  - `sandwich_block(...)` topology — 4 norms surrounding the two sublayers.
    Source: modeling_gemma3.py:418-438 (input_layernorm, post_attention_layernorm,
    pre_feedforward_layernorm, post_feedforward_layernorm — same form in Gemma 4).
  - `NormZeroCenteredSpec` — Gemma RMSNorm uses (1 + w) * x_normed.
    Source: modeling_gemma3.py:472 ("initialize with 0s to be 1 centered as the
    RMSNorm here does (1 + weight)")
  - `GQASpec` extended fields: `v_norm`, `qk_norm_fixed_scale`, `sliding_window`.
    `v_norm` and `qk_norm_fixed_scale` are M1 Gemma-4-only fields (Gemma 3 has
    `q_norm`/`k_norm` only; Gemma 4 absorbs 1/sqrt(Dh) into the qk_norm scale
    and adds a value-side norm).
  - `RoPESpec` with `partial_rotary_factor=0.25` and
    `partial_rotary_kind="proportional"` — Gemma-4-specific partial-rotary
    layout (Gemma 3 used full rotation).
  - `GeGLUSpec` instead of SwiGLU — Gemma family uses GeLU-gated MLP
    (`hidden_activation="gelu_pytorch_tanh"`).
    Source: modeling_gemma3.py:125-128 (act_fn applied to gate_proj only).

Spec inspection only — Gemma 4 / sandwich-norm execution is M1 scope.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from components import (
    GeGLUSpec,
    GQASpec,
    NormZeroCenteredSpec,
    RoPESpec,
    sandwich_block,
)


# ---- Gemma 4 E2B "local" (sliding) layer dimensions ------------------------
# E2B nominally has ~2B effective params; values aligned with Gemma 3-2B-like
# scale, which is what the Gemma 4 E2B technical report extends.
HIDDEN_SIZE = 2304
NUM_ATTN_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 256
INTERMEDIATE_SIZE = 9216
RMS_EPS = 1e-6
SLIDING_WINDOW = 4096                      # local-layer window size

# Local (sliding) layers use the smaller local rope base; global layers use the
# larger global rope base. Both values cited from Gemma3TextConfig.default_theta.
ROPE_THETA_LOCAL = 10_000.0

# Gemma 4 E2B-specific: partial RoPE on 1/4 of head_dim, proportional layout
# (rotated and pass-through dimensions are interleaved across head_dim, not
# split into a prefix block).
PARTIAL_ROTARY_FACTOR = 0.25

# Gemma 4 absorbs the 1/sqrt(head_dim) factor into qk_norm:
QK_NORM_FIXED_SCALE = 1.0 / (HEAD_DIM ** 0.5)


def build_gemma4_local_layer():
    # Four sandwich-norm sites; all use the Gemma zero-centered RMSNorm form.
    pre_attn = NormZeroCenteredSpec(eps=RMS_EPS)
    post_attn = NormZeroCenteredSpec(eps=RMS_EPS)
    pre_ffn = NormZeroCenteredSpec(eps=RMS_EPS)
    post_ffn = NormZeroCenteredSpec(eps=RMS_EPS)

    # Attention-internal norms:
    # - qk_norm:  head-dim RMSNorm on Q and K before RoPE (Gemma 3+).
    # - v_norm:   value-side norm without learned scale (Gemma 4 only).
    qk_norm = NormZeroCenteredSpec(eps=RMS_EPS)
    v_norm = NormZeroCenteredSpec(eps=RMS_EPS)

    attention = GQASpec(
        num_heads=NUM_ATTN_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        qk_norm=qk_norm,
        v_norm=v_norm,
        qk_norm_fixed_scale=QK_NORM_FIXED_SCALE,
        sliding_window=SLIDING_WINDOW,
    )

    pos_enc = RoPESpec(
        theta=ROPE_THETA_LOCAL,
        partial_rotary_factor=PARTIAL_ROTARY_FACTOR,
        partial_rotary_kind="proportional",
    )

    ffn = GeGLUSpec(
        intermediate_size=INTERMEDIATE_SIZE,
        activation="gelu_tanh",  # = "gelu_pytorch_tanh" in HF
    )

    return sandwich_block(
        pre_attention_norm=pre_attn,
        attention=attention,
        positional_encoding=pos_enc,
        post_attention_norm=post_attn,
        pre_ffn_norm=pre_ffn,
        ffn=ffn,
        post_ffn_norm=post_ffn,
    )


def main() -> None:
    layer = build_gemma4_local_layer()

    print("Gemma 4 E2B 'local' (sliding-window) decoder layer:")
    print(f"  hidden_size    = {HIDDEN_SIZE}")
    print(f"  sliding_window = {SLIDING_WINDOW}")
    print(f"  topology       = sandwich-norm (4 norms surrounding attn + ffn)")
    print()
    print("BlockGraph topology:")
    for n in layer.nodes:
        print(f"  {n.name:10s} <- {str(n.inputs):24s} :: {type(n.spec).__name__}")
    print(f"  output: {layer.output}")
    print()

    attn = next(n.spec for n in layer.nodes if isinstance(n.spec, GQASpec))
    rope = next(n.spec for n in layer.nodes if isinstance(n.spec, RoPESpec))
    ffn = next(n.spec for n in layer.nodes if isinstance(n.spec, GeGLUSpec))

    print("Key spec fields:")
    print(f"  attn.num_heads             = {attn.num_heads}")
    print(f"  attn.num_kv_heads          = {attn.num_kv_heads}")
    print(f"  attn.head_dim              = {attn.head_dim}")
    print(f"  attn.sliding_window        = {attn.sliding_window}")
    print(f"  attn.qk_norm.kind          = {attn.qk_norm.kind}")
    print(f"  attn.v_norm.kind           = {attn.v_norm.kind}              # <- Gemma-4-only")
    print(f"  attn.qk_norm_fixed_scale   = {attn.qk_norm_fixed_scale:.6f}       "
          f"# 1/sqrt({attn.head_dim}) absorbed")
    print(f"  rope.theta                 = {rope.theta:g}")
    print(f"  rope.partial_rotary_factor = {rope.partial_rotary_factor}")
    print(f"  rope.partial_rotary_kind   = {rope.partial_rotary_kind!r}    # Gemma-4 'proportional'")
    print(f"  ffn.kind                   = {ffn.kind}")
    print(f"  ffn.activation             = {ffn.activation!r}")
    print(f"  ffn.intermediate_size      = {ffn.intermediate_size}")


if __name__ == "__main__":
    main()


# Expected output (the 9-node sandwich graph + 4 NormZeroCenteredSpec sites):
#
# Gemma 4 E2B 'local' (sliding-window) decoder layer:
#   hidden_size    = 2304
#   sliding_window = 4096
#   topology       = sandwich-norm (4 norms surrounding attn + ffn)
#
# BlockGraph topology:
#   n_pre_a    <- ('input',)              :: NormZeroCenteredSpec
#   rope       <- ('n_pre_a',)            :: RoPESpec
#   attn       <- ('rope',)               :: GQASpec
#   n_post_a   <- ('attn',)               :: NormZeroCenteredSpec
#   r0         <- ('input', 'n_post_a')   :: AddSpec
#   n_pre_f    <- ('r0',)                 :: NormZeroCenteredSpec
#   ffn        <- ('n_pre_f',)            :: GeGLUSpec
#   n_post_f   <- ('ffn',)                :: NormZeroCenteredSpec
#   r1         <- ('r0', 'n_post_f')      :: AddSpec
#   output: r1
#
# Key spec fields:
#   attn.num_heads             = 8
#   attn.num_kv_heads          = 4
#   attn.head_dim              = 256
#   attn.sliding_window        = 4096
#   attn.qk_norm.kind          = ZeroCentered
#   attn.v_norm.kind           = ZeroCentered              # <- Gemma-4-only
#   attn.qk_norm_fixed_scale   = 0.062500       # 1/sqrt(256) absorbed
#   rope.theta                 = 10000
#   rope.partial_rotary_factor = 0.25
#   rope.partial_rotary_kind   = 'proportional'    # Gemma-4 'proportional'
#   ffn.kind                   = GeGLU
#   ffn.activation             = 'gelu_tanh'
#   ffn.intermediate_size      = 9216
