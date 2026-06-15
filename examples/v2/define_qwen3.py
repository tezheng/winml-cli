"""Define a Qwen3-0.6B decoder layer using only component specs. No torch.

Demonstrates: the same `pre_norm_block` factory + `GQASpec`/`RoPESpec`/
`SwiGLUSpec`/`NormStandardSpec` primitives express a real production layer with
nothing but data. No model-specific Python code.

Values pulled from the cached HF config
(``hf_cache/.../Qwen3-0.6B/config.json``):

    hidden_size = 1024            # NOTE: != num_heads * head_dim (2048).
                                  # Qwen3 unties d_model from head_dim*H_q.
    num_hidden_layers = 28
    num_attention_heads = 16
    num_key_value_heads = 8       # GQA, 2x query repetition
    head_dim = 128
    intermediate_size = 3072
    rms_norm_eps = 1e-6
    rope_theta = 1_000_000        # Qwen3-specific (Llama uses 1e4/5e5)
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from components import (
    GQASpec,
    NormStandardSpec,
    RoPESpec,
    SwiGLUSpec,
    pre_norm_block,
)


# ---- Qwen3-0.6B dimensions -------------------------------------------------
HIDDEN_SIZE = 1024
NUM_HIDDEN_LAYERS = 28
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 3072
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0  # Source: modeling_qwen3.py:124 (config.rope_parameters["rope_theta"])


def build_qwen3_layer():
    # Qwen3 uses RMSNorm on the head_dim of Q and K, applied BEFORE RoPE.
    # Source: modeling_qwen3.py:248-249 (self.q_norm / self.k_norm) and
    # modeling_qwen3.py:264-269 (norm -> RoPE order).
    qk_norm = NormStandardSpec(eps=RMS_EPS)

    block_norm = NormStandardSpec(eps=RMS_EPS)
    post_attn_norm = NormStandardSpec(eps=RMS_EPS)

    attention = GQASpec(
        num_heads=NUM_ATTN_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        qk_norm=qk_norm,
    )

    # Plain RoPE; full rotation; no scaling configured in Qwen3-0.6B
    # (config.rope_scaling is None).
    pos_enc = RoPESpec(theta=ROPE_THETA)

    # SwiGLU; not fused (Qwen3 has separate gate_proj and up_proj).
    ffn = SwiGLUSpec(intermediate_size=INTERMEDIATE_SIZE, fused_gate_up=False)

    return pre_norm_block(
        block_norm=block_norm,
        attention=attention,
        positional_encoding=pos_enc,
        post_attention_norm=post_attn_norm,
        ffn=ffn,
    )


def main() -> None:
    layer = build_qwen3_layer()

    print("Qwen3-0.6B decoder layer (Spec-only definition):")
    print(f"  hidden_size       = {HIDDEN_SIZE}")
    print(f"  num_hidden_layers = {NUM_HIDDEN_LAYERS} (a stack of these)")
    print()
    print("BlockGraph topology:")
    for n in layer.nodes:
        print(f"  {n.name:8s} <- {str(n.inputs):24s} :: {type(n.spec).__name__}")
    print(f"  output: {layer.output}")
    print()

    attn = next(n.spec for n in layer.nodes if isinstance(n.spec, GQASpec))
    rope = next(n.spec for n in layer.nodes if isinstance(n.spec, RoPESpec))
    ffn = next(n.spec for n in layer.nodes if isinstance(n.spec, SwiGLUSpec))

    print("Key spec fields:")
    print(f"  attn.num_heads        = {attn.num_heads}")
    print(f"  attn.num_kv_heads     = {attn.num_kv_heads}  (GQA repeat={attn.num_heads // attn.num_kv_heads}x)")
    print(f"  attn.head_dim         = {attn.head_dim}")
    print(f"  attn.qk_norm          = {type(attn.qk_norm).__name__ if attn.qk_norm else None}"
          " (head-dim only, pre-RoPE — Qwen3-specific)")
    print(f"  attn.scale            = {attn.scale:.6f}")
    print(f"  rope.theta            = {rope.theta:g}")
    print(f"  ffn.intermediate_size = {ffn.intermediate_size}")
    print(f"  ffn.fused_gate_up     = {ffn.fused_gate_up}")


if __name__ == "__main__":
    main()


# Expected output (the topology line ordering is deterministic):
#
# Qwen3-0.6B decoder layer (Spec-only definition):
#   hidden_size       = 1024
#   num_hidden_layers = 28 (a stack of these)
#
# BlockGraph topology:
#   n0       <- ('input',)              :: NormStandardSpec
#   rope     <- ('n0',)                 :: RoPESpec
#   attn     <- ('rope',)               :: GQASpec
#   r0       <- ('input', 'attn')       :: AddSpec
#   n1       <- ('r0',)                 :: NormStandardSpec
#   ffn      <- ('n1',)                 :: SwiGLUSpec
#   r1       <- ('r0', 'ffn')           :: AddSpec
#   output: r1
#
# Key spec fields:
#   attn.num_heads        = 16
#   attn.num_kv_heads     = 8  (GQA repeat=2x)
#   attn.head_dim         = 128
#   attn.qk_norm          = NormStandardSpec (head-dim only, pre-RoPE — Qwen3-specific)
#   attn.scale            = 0.088388
#   rope.theta            = 1e+06
#   ffn.intermediate_size = 3072
#   ffn.fused_gate_up     = False
