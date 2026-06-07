"""DeepSeek-OCR-2 model config — LM-decoder portion only.

Verified against `transformers/models/deepseek_ocr2/configuration_deepseek_ocr2.py`
(DeepseekOcr2TextConfig at L184-264) and the publicly-loaded
`deepseek-community/DeepSeek-OCR-2` config.json.

=== KEY DRIFT vs prompt assumptions ===

The B8 plan describes DeepSeek-OCR as using MLA attention and a
block-bidirectional "Visual Causal Flow" mask. The HF transformers v5.10.2
reference (modeling_deepseek_ocr2.py:1073-1310) DISAGREES:
  * `DeepseekOcr2TextAttention` uses STANDARD MHA (n_q == n_kv == 10
    heads on the canonical checkpoint) with `q_proj/k_proj/v_proj/o_proj`
    Linear layers — NO MLA, NO q_lora_rank or kv_lora_rank.
  * The mask is built by `create_causal_mask` — NO block-bidirectional,
    NO Visual Causal Flow.
  * MoE router is **softmax** with `topk_method='greedy'` (or
    `group_limited_greedy` if `n_group > 1`), n_shared_experts=2 dense
    FFN added to the routed output. NOT sigmoid+bias.

We honor source-first discipline: the LM-decoder we ship MIRRORS the HF
source exactly. The `MaskKind.BLOCK_BIDIRECTIONAL` + the
`block_bidirectional_mask` spec field are wired into api/attention.py for
v3-spec hooks and exercised by a shape-only test
(test_layer_shape.py::test_block_bidirectional_mask_construction) — but
the canonical numerical gate uses CAUSAL.

DeepSeek-OCR-2 LM-decoder summary (canonical 3B-MoE-A570M):
- 12 layers; layer 0 dense (`mlp_layer_types[0]='dense'`), layers 1-11
  sparse MoE.
- Standard MHA: hidden=1280, n_q=n_kv=10, head_dim=128, attention_bias=False.
- RoPE: SPLIT_HALF basis, theta=10_000.
- MoE: n_routed_experts=64, n_shared_experts=2, num_experts_per_tok=6,
  moe_intermediate_size=896, routed_scaling_factor=1.0, n_group=1
  (group routing reduces to plain greedy when n_group=1), topk_method='greedy'.
- RMSNorm STANDARD_W eps=1e-6, SwiGLU.

Source citations:
- transformers/models/deepseek_ocr2/configuration_deepseek_ocr2.py:184-264
  (DeepseekOcr2TextConfig).
- transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py:1073-1138
  (DeepseekOcr2TextAttention — STANDARD MHA).
- transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py:1197-1242
  (DeepseekOcr2TextMoe — softmax router, greedy / group_limited_greedy).
- transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py:1266-1309
  (DeepseekOcr2TextDecoderLayer — PRE-norm; dense vs MoE per
  `mlp_layer_types[layer_idx]`).
- transformers/models/deepseek_ocr2/modeling_deepseek_ocr2.py:1339-1410
  (DeepseekOcr2TextModel — create_causal_mask, no VCF).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class DeepseekOcr2Config:
    """LM-decoder config for DeepSeek-OCR-2 (mirrors HF text_config sub-config)."""
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int                 # dense FFN size (layer 0)
    moe_intermediate_size: int             # per-expert FFN size
    num_hidden_layers: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    n_group: int
    topk_group: int
    topk_method: str                       # 'greedy' or 'group_limited_greedy'
    attention_bias: bool
    mlp_bias: bool
    mlp_layer_types: tuple[str, ...]       # length == num_hidden_layers

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "DeepseekOcr2Config":
        if "text_config" in hf and isinstance(hf["text_config"], dict):
            top = hf
            inner = hf["text_config"]
        else:
            top = hf
            inner = hf
        dt_raw = inner.get("dtype", inner.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        elif dt_raw is None:
            dtype = torch.float32
        else:
            dtype = _TORCH_DTYPE_MAP.get(str(dt_raw), torch.float32)

        rp = inner.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 10000.0))
        elif "rope_theta" in inner:
            rope_theta = float(inner["rope_theta"])
        else:
            rope_theta = 10000.0

        head_dim = inner.get("head_dim")
        if head_dim is None:
            head_dim = inner["hidden_size"] // inner["num_attention_heads"]

        n_layers = int(inner["num_hidden_layers"])
        mlp_types_raw = inner.get("mlp_layer_types")
        if mlp_types_raw is None:
            mlp_layer_types = tuple(["dense"] + ["sparse"] * (n_layers - 1))
        else:
            mlp_layer_types = tuple(mlp_types_raw)

        return cls(
            hidden_size=int(inner["hidden_size"]),
            num_attention_heads=int(inner["num_attention_heads"]),
            num_key_value_heads=int(
                inner.get("num_key_value_heads") or inner["num_attention_heads"]
            ),
            head_dim=int(head_dim),
            intermediate_size=int(inner["intermediate_size"]),
            moe_intermediate_size=int(inner.get("moe_intermediate_size", 896)),
            num_hidden_layers=n_layers,
            rope_theta=rope_theta,
            rms_norm_eps=float(inner.get("rms_norm_eps", 1e-6)),
            vocab_size=int(inner["vocab_size"]),
            max_position_embeddings=int(inner["max_position_embeddings"]),
            tie_word_embeddings=bool(
                top.get("tie_word_embeddings",
                        inner.get("tie_word_embeddings", False))
            ),
            dtype=dtype,
            n_routed_experts=int(inner.get("n_routed_experts", 64)),
            n_shared_experts=int(inner.get("n_shared_experts", 2)),
            num_experts_per_tok=int(inner.get("num_experts_per_tok", 6)),
            routed_scaling_factor=float(inner.get("routed_scaling_factor", 1.0)),
            n_group=int(inner.get("n_group") or 1),
            topk_group=int(inner.get("topk_group") or 1),
            topk_method=str(inner.get("topk_method", "greedy") or "greedy"),
            attention_bias=bool(inner.get("attention_bias", False)),
            mlp_bias=bool(inner.get("mlp_bias", False)),
            mlp_layer_types=mlp_layer_types,
        )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # All layers use STANDARD causal masking per HF source — no VCF.
        attn_spec = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            sliding_window=None,
            q_bias=self.attention_bias, k_bias=self.attention_bias,
            v_bias=self.attention_bias, o_bias=self.attention_bias,
            rope=specs.RoPESpec(
                base_theta=self.rope_theta,
                basis=types.RoPEBasis.SPLIT_HALF,
                scaling=types.RoPEScaling.NONE,
            ),
        )
        # Channel mixer: dense for mlp_layer_types[L]=='dense', MoE otherwise.
        if self.mlp_layer_types[layer_idx] == "dense":
            channel = specs.FFNSpec(
                intermediate_size=self.intermediate_size,
                activation=types.Activation.SILU,
                gate_kind=types.GateKind.SWIGLU,
                fused_gate_up=False,
                gate_bias=self.mlp_bias,
                up_bias=self.mlp_bias,
                down_bias=self.mlp_bias,
            )
        else:
            expert_ffn = specs.FFNSpec(
                intermediate_size=self.moe_intermediate_size,
                activation=types.Activation.SILU,
                gate_kind=types.GateKind.SWIGLU,
                fused_gate_up=False,
                gate_bias=False, up_bias=False, down_bias=False,
            )
            # n_group == 1 collapses group routing to plain greedy. We still
            # carry the GroupRoutingSpec when n_group > 1 (matching HF
            # `topk_method == 'group_limited_greedy'`).
            group_routing: Optional[specs.GroupRoutingSpec]
            if self.topk_method == "group_limited_greedy" and self.n_group > 1:
                group_routing = specs.GroupRoutingSpec(
                    n_groups=self.n_group,
                    topk_per_group=self.topk_group,
                )
            else:
                group_routing = None
            channel = specs.MoESpec(
                n_experts=self.n_routed_experts,
                top_k=self.num_experts_per_tok,
                n_shared_experts=self.n_shared_experts,
                router_kind="softmax",        # HF: F.softmax + greedy top-k.
                router_norm=False,            # HF source does NOT renorm topk.
                score_correction_bias=False,
                group_routing=group_routing,
                routed_scaling_factor=self.routed_scaling_factor,
                expert_ffn=expert_ffn,
            )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn_spec,
            channel_mixer=channel,
            pre_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec,
        )
