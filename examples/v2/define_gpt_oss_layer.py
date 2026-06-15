"""Define a GPT-OSS-20B sliding-window attention layer with attention sinks.

Demonstrates:
  - `GQASinksSpec` — GQA with a learned per-head additive sink logit in the
    softmax denominator.
    Source: modeling_gpt_oss.py:303 (`self.sinks = nn.Parameter(...)`)
            modeling_gpt_oss.py:261-269 (sinks concat into combined_logits,
                                          softmax, then sink dropped before V).
  - `sinks` lives in the **weights** slot, not the runtime inputs slot.
    Source: components/attention.py GQA+Sinks ApiSchema — `sinks` is bound
    once at model build, not per call. See _GQA_SINKS_WEIGHTS entry.
  - GPT-OSS alternates layer types per layer
    (`"sliding_attention" if (i+1)%2 else "full_attention"`).
    Source: configuration_gpt_oss.py:103-104.

What this sample DOES NOT show: the MoE wrapping.
  GPT-OSS replaces the dense FFN with a sigmoid-routed MoE
  (`GptOssMLP` -> `GptOssTopKRouter` + `GptOssExperts`, 128 experts top-4).
  Source: modeling_gpt_oss.py:120-145.
  components/ has no `MoESpec` yet (stub-only), so we substitute the per-expert
  inner activation — a `ClampedSwiGLUSpec(clamp=7.0, alpha=1.702)` — to
  illustrate the spec for the activation each expert applies internally.
  The MoE router / dispatch wrapping is M1+ work.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from components import (
    ClampedSwiGLUSpec,
    GQASinksSpec,
    NormStandardSpec,
    RoPESpec,
    pre_norm_block,
)


# ---- GPT-OSS-20B dimensions (defaults from configuration_gpt_oss.py) -------
HIDDEN_SIZE = 2880
NUM_ATTN_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 64
INTERMEDIATE_SIZE = 2880        # per-expert intermediate (also the MoE inner)
NUM_HIDDEN_LAYERS = 36
NUM_LOCAL_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 4
SLIDING_WINDOW = 128            # GPT-OSS uses a very short window for half its layers
RMS_NORM_EPS = 1e-5

# Source: modeling_gpt_oss.py:79  (`self.alpha = 1.702`)
#         modeling_gpt_oss.py:84-86 (gate.clamp(max=limit), up.clamp(min=-lim, max=lim))
MOE_INNER_CLAMP = 7.0
MOE_INNER_ALPHA = 1.702


def build_gpt_oss_sliding_layer():
    block_norm = NormStandardSpec(eps=RMS_NORM_EPS)
    post_attn_norm = NormStandardSpec(eps=RMS_NORM_EPS)

    # GPT-OSS attention: GQA + per-head learned sink logit in the denominator.
    # softmax_scale_log_base=2: the sink-logit scaling base (component default
    # for GPT-OSS). Source: components/attention.py GQASinksSpec docstring.
    attention = GQASinksSpec(
        num_heads=NUM_ATTN_HEADS,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        sliding_window=SLIDING_WINDOW,
        softmax_scale_log_base=2,
    )

    # GPT-OSS uses YaRN-scaled RoPE; for this minimal sketch we use plain RoPE
    # with the same theta — a full YaRN sample would use RoPEYaRNSpec instead.
    pos_enc = RoPESpec(theta=10_000.0)

    # NOTE: not the full FFN of the layer. The real layer wraps a sigmoid-routed
    # MoE around 128 ClampedSwiGLU experts. This single ClampedSwiGLUSpec models
    # ONE expert's inner activation only — useful for spec inspection.
    moe_inner_activation = ClampedSwiGLUSpec(
        intermediate_size=INTERMEDIATE_SIZE,
        clamp=MOE_INNER_CLAMP,
        alpha=MOE_INNER_ALPHA,
    )

    return pre_norm_block(
        block_norm=block_norm,
        attention=attention,
        positional_encoding=pos_enc,
        post_attention_norm=post_attn_norm,
        ffn=moe_inner_activation,
    )


def _show_weights_slot(attn_spec: GQASinksSpec) -> None:
    """Show which slots `sinks` lives in (weights vs runtime inputs)."""
    from components import get_component
    comp = get_component("attention")
    variant = next(v for v in comp.variants if isinstance(v.spec, GQASinksSpec))
    api = variant.api
    print("  Attention API surface (GQA+Sinks variant):")
    print("    Inputs:  " + ", ".join(f.name for f in api.inputs))
    print("    Weights: " + ", ".join(f.name for f in api.weights))
    sinks_in_weights = any(w.name == "sinks" for w in api.weights)
    print(f"    'sinks' present in weights: {sinks_in_weights}  "
          "<- learned param, bound at model-build time")


def main() -> None:
    layer = build_gpt_oss_sliding_layer()

    print("GPT-OSS-20B 'sliding_attention' layer (every odd-index layer in the stack):")
    print(f"  hidden_size       = {HIDDEN_SIZE}")
    print(f"  num_hidden_layers = {NUM_HIDDEN_LAYERS} (alternating sliding/full)")
    print(f"  MoE: num_local_experts={NUM_LOCAL_EXPERTS}, "
          f"num_experts_per_tok={NUM_EXPERTS_PER_TOK}")
    print()
    print("BlockGraph topology:")
    for n in layer.nodes:
        print(f"  {n.name:8s} <- {str(n.inputs):24s} :: {type(n.spec).__name__}")
    print(f"  output: {layer.output}")
    print()

    attn = next(n.spec for n in layer.nodes if isinstance(n.spec, GQASinksSpec))
    rope = next(n.spec for n in layer.nodes if isinstance(n.spec, RoPESpec))
    ffn = next(n.spec for n in layer.nodes if isinstance(n.spec, ClampedSwiGLUSpec))

    print("Key spec fields:")
    print(f"  attn.kind                   = {attn.kind}")
    print(f"  attn.num_heads              = {attn.num_heads}")
    print(f"  attn.num_kv_heads           = {attn.num_kv_heads}")
    print(f"  attn.head_dim               = {attn.head_dim}")
    print(f"  attn.sliding_window         = {attn.sliding_window}     "
          "# alternates with full per layer")
    print(f"  attn.softmax_scale_log_base = {attn.softmax_scale_log_base}")
    print(f"  rope.theta                  = {rope.theta:g}")
    print(f"  ffn.kind                    = {ffn.kind}    "
          "# one expert's inner activation; MoE wrap not modelled")
    print(f"  ffn.clamp                   = {ffn.clamp}")
    print(f"  ffn.alpha                   = {ffn.alpha}")
    print()
    print("Where does the `sinks` learned parameter live?")
    _show_weights_slot(attn)


if __name__ == "__main__":
    main()


# Expected output (abbreviated):
#
# GPT-OSS-20B 'sliding_attention' layer (every odd-index layer in the stack):
#   hidden_size       = 2880
#   num_hidden_layers = 36 (alternating sliding/full)
#   MoE: num_local_experts=128, num_experts_per_tok=4
#
# BlockGraph topology:
#   n0       <- ('input',)              :: NormStandardSpec
#   rope     <- ('n0',)                 :: RoPESpec
#   attn     <- ('rope',)               :: GQASinksSpec
#   r0       <- ('input', 'attn')       :: AddSpec
#   n1       <- ('r0',)                 :: NormStandardSpec
#   ffn      <- ('n1',)                 :: ClampedSwiGLUSpec
#   r1       <- ('r0', 'ffn')           :: AddSpec
#   output: r1
#
# Key spec fields:
#   attn.kind                   = GQA+Sinks
#   attn.num_heads              = 64
#   attn.num_kv_heads           = 8
#   attn.head_dim               = 64
#   attn.sliding_window         = 128
#   attn.softmax_scale_log_base = 2
#   rope.theta                  = 10000
#   ffn.kind                    = Clamped-SwiGLU
#   ffn.clamp                   = 7.0
#   ffn.alpha                   = 1.702
#
# Where does the `sinks` learned parameter live?
#   Attention API surface (GQA+Sinks variant):
#     Inputs:  query, key, value, sinks, mask, kv_cache_in
#     Weights: W_q, W_k, W_v, W_o, sinks
#     'sinks' present in weights: True  <- learned param, bound at model-build time
