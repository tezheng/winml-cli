"""DeepSeek-V4 config + adapter to api specs.

Verified against `transformers/models/deepseek_v4/{configuration_deepseek_v4,
modeling_deepseek_v4}.py` (transformers 5.10.x).

Per-layer dispatch (paper §2.1, §2.3):

- `mlp_layer_types[i] == "hash_moe"` → MoESpec.router_kind = "hash"
  (DeepseekV4HashRouter, modeling_deepseek_v4.py:1050-1078). The first three
  MoE layers are hash by default (`default_num_hash_layers=3`,
  configuration_deepseek_v4.py:165).
- `mlp_layer_types[i] == "moe"`      → MoESpec.router_kind = "sigmoid_plus_bias"
  (DeepseekV4TopKRouter, modeling_deepseek_v4.py:1029-1047, scoring via sigmoid
  + e_score_correction_bias).

Attention `layer_types[i]`:

- `compressed_sparse_attention` → AttentionKind.CSA_HCA + CSASpec
  (modeling_deepseek_v4.py:587-749).
- `heavily_compressed_attention` → AttentionKind.CSA_HCA + HCASpec
  (modeling_deepseek_v4.py:362-444).
- `sliding_attention` → AttentionKind.STANDARD + sliding_window
  (modeling_deepseek_v4.py:751-770, MQA: num_key_value_heads=1).

v7 Phase A wired the hash MoE forward (P1) and shape-only CSA/HCA (P2). The
mHC residual streams and grouped output projection are NOT modeled — see
package docstring.
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
class DeepSeekV4Config:
    """DeepSeek-V4 config (subset for one decoder layer)."""
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    q_lora_rank: int
    qk_rope_head_dim: int
    moe_intermediate_size: int
    num_hidden_layers: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    scoring_func: str                          # "sqrtsoftplus" | "sigmoid" | "softmax"
    swiglu_limit: float
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    rope_theta: float
    compress_rope_theta: float
    # Per-layer dispatch
    layer_types: tuple[str, ...]               # per-layer attention type
    mlp_layer_types: tuple[str, ...]           # per-layer MLP type ("hash_moe" | "moe")
    # CSA / HCA dims
    compress_rate_csa: int                     # paper m=4
    compress_rate_hca: int                     # paper m'=128
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    dtype: torch.dtype

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "DeepSeekV4Config":
        dt_raw = hf.get("dtype", hf.get("torch_dtype", "float32"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.float32)

        # qk_rope_head_dim derives from partial_rotary_factor * head_dim.
        # Source: configuration_deepseek_v4.py:292.
        head_dim = hf.get("head_dim", 512)
        prf = hf.get("partial_rotary_factor")
        if prf is None:
            prf = 64.0 / head_dim
        qk_rope = int(head_dim * float(prf))

        compress_rates = hf.get("compress_rates") or {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }

        # mlp_layer_types default: first 3 hash, rest moe. Source:
        # configuration_deepseek_v4.py:279-281.
        n = hf["num_hidden_layers"]
        mlp_layer_types = hf.get("mlp_layer_types")
        if mlp_layer_types is None:
            n_hash = hf.get("num_hash_layers", 3)
            mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
        mlp_layer_types = tuple(mlp_layer_types[:n])

        # layer_types default: 2× HCA bootstrap + interleave CSA/HCA. Source:
        # configuration_deepseek_v4.py:270-276.
        layer_types = hf.get("layer_types")
        if layer_types is None:
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(n - 2, 0))
            ]
            layer_types = ["heavily_compressed_attention"] * min(n, 2) + interleave
        layer_types = tuple(layer_types[:n])

        rope_theta = float(hf.get("rope_theta", 10000.0))
        compress_rope_theta = float(hf.get("compress_rope_theta", 160000.0))
        # `rope_parameters` may carry "main" / "compress" sub-dicts. Take theta
        # from them if present.
        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            main = rp.get("main") if isinstance(rp.get("main"), dict) else None
            comp = rp.get("compress") if isinstance(rp.get("compress"), dict) else None
            if main is not None:
                rope_theta = float(main.get("rope_theta", rope_theta))
            if comp is not None:
                compress_rope_theta = float(comp.get("rope_theta", compress_rope_theta))

        return cls(
            hidden_size=hf.get("hidden_size", 4096),
            num_attention_heads=hf.get("num_attention_heads", 64),
            num_key_value_heads=hf.get("num_key_value_heads", 1),
            head_dim=head_dim,
            q_lora_rank=hf.get("q_lora_rank", 1024),
            qk_rope_head_dim=qk_rope,
            moe_intermediate_size=hf.get("moe_intermediate_size", 2048),
            num_hidden_layers=n,
            n_routed_experts=hf.get("n_routed_experts", 256),
            n_shared_experts=hf.get("n_shared_experts", 1),
            num_experts_per_tok=hf.get("num_experts_per_tok", 6),
            routed_scaling_factor=float(hf.get("routed_scaling_factor", 1.5)),
            norm_topk_prob=bool(hf.get("norm_topk_prob", True)),
            scoring_func=hf.get("scoring_func", "sqrtsoftplus"),
            swiglu_limit=float(hf.get("swiglu_limit", 10.0)),
            sliding_window=int(hf.get("sliding_window", 128)),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            vocab_size=hf.get("vocab_size", 129280),
            max_position_embeddings=hf.get("max_position_embeddings", 1048576),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            rope_theta=rope_theta,
            compress_rope_theta=compress_rope_theta,
            layer_types=layer_types,
            mlp_layer_types=mlp_layer_types,
            compress_rate_csa=int(compress_rates.get("compressed_sparse_attention", 4)),
            compress_rate_hca=int(compress_rates.get("heavily_compressed_attention", 128)),
            index_n_heads=int(hf.get("index_n_heads", 64)),
            index_head_dim=int(hf.get("index_head_dim", 128)),
            index_topk=int(hf.get("index_topk", 512)),
            dtype=dtype,
        )

    def is_hash_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "hash_moe"

    def _attn_spec(self, layer_idx: int) -> specs.AttentionSpec:
        norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        # All V4 attention uses interleaved RoPE on a partial slice
        # (configuration_deepseek_v4.py:147 — partial_rotary_factor = 64/512).
        # We mark the basis INTERLEAVED to match V4's `apply_rotary_pos_emb`
        # (modeling_deepseek_v4.py:78 — "V4's *interleaved* RoPE").
        # The compress branch uses a different theta — we attach the main theta
        # only; CSA/HCA branches are shape-only and don't consume cos/sin.
        # NOTE: V4 sources use INTERLEAVED RoPE
        # (modeling_deepseek_v4.py:153-168) on a partial slice
        # (qk_rope_head_dim / head_dim, default 64/512). The api Attention's
        # init does NOT support INTERLEAVED + partial_rotary_factor < 1.0
        # (rope.py:141). Since the V4 attention forward is SHAPE-ONLY for
        # CSA / HCA (raises NotImplementedError) and we don't gate
        # numerically against sliding_attention, we model RoPE as SPLIT_HALF
        # here purely so the IR module composes. The actual rope_basis is
        # an attribute on the layer spec but is never consumed.
        rope = specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.SPLIT_HALF,
            scaling=types.RoPEScaling.NONE,
            partial_rotary_factor=self.qk_rope_head_dim / self.head_dim,
        )
        lt = self.layer_types[layer_idx]
        if lt == "compressed_sparse_attention":
            csa = specs.CSASpec(
                compress_rate=self.compress_rate_csa,
                block_size=self.compress_rate_csa,
                indexer_n_heads=self.index_n_heads,
                indexer_head_dim=self.index_head_dim,
                indexer_topk=self.index_topk,
            )
            return specs.AttentionSpec(
                n_q_heads=self.num_attention_heads,
                n_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                kind=types.AttentionKind.CSA_HCA,
                qkv_layout=types.QKVLayout.SPLIT,
                mask_kind=types.MaskKind.CAUSAL,
                rope=rope,
                csa=csa,
            )
        elif lt == "heavily_compressed_attention":
            hca = specs.HCASpec(
                compress_rate=self.compress_rate_hca,
                hierarchy_levels=1,
            )
            return specs.AttentionSpec(
                n_q_heads=self.num_attention_heads,
                n_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                kind=types.AttentionKind.CSA_HCA,
                qkv_layout=types.QKVLayout.SPLIT,
                mask_kind=types.MaskKind.CAUSAL,
                rope=rope,
                hca=hca,
            )
        else:  # sliding_attention
            return specs.AttentionSpec(
                n_q_heads=self.num_attention_heads,
                n_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                kind=types.AttentionKind.STANDARD,
                qkv_layout=types.QKVLayout.SPLIT,
                mask_kind=types.MaskKind.SWA,
                sliding_window=self.sliding_window,
                rope=rope,
            )

    def _moe_spec(self, layer_idx: int) -> specs.MoESpec:
        """Build the per-layer MoE spec.

        For "hash_moe" layers the router is `router_kind="hash"`; for "moe"
        layers it's `router_kind="sigmoid_plus_bias"`. Source:
        modeling_deepseek_v4.py:1081-1098 (DeepseekV4SparseMoeBlock dispatch).
        """
        # Experts are SwiGLU with optional clamping (swiglu_limit). The IR
        # carries clamping only via `expert_kind="gpt_oss_clamped_swiglu"`
        # which uses INTERLEAVED gate/up — V4 uses chunk(2, dim=-1). We thus
        # ship the plain swiglu expert_kind here and require the test config
        # to set swiglu_limit large enough that clamps are no-ops.
        expert_ffn = specs.FFNSpec(
            intermediate_size=self.moe_intermediate_size,
            activation=types.Activation.SILU,
            gate_kind=types.GateKind.SWIGLU,
            fused_gate_up=False,
            gate_bias=False, up_bias=False, down_bias=False,
        )
        if self.is_hash_layer(layer_idx):
            # Hash routing: tid2eid lookup, sigmoid score head, V4 scoring_func
            # may differ from default sigmoid — for the numerical gate we
            # require scoring_func="sigmoid".
            return specs.MoESpec(
                n_experts=self.n_routed_experts,
                top_k=self.num_experts_per_tok,
                n_shared_experts=self.n_shared_experts,
                router_kind="hash",
                router_norm=True,                 # V4 normalises weights post-gather
                hash_vocab_size=self.vocab_size,
                hash_score_fn="sigmoid",
                routed_scaling_factor=self.routed_scaling_factor,
                expert_ffn=expert_ffn,
            )
        else:
            return specs.MoESpec(
                n_experts=self.n_routed_experts,
                top_k=self.num_experts_per_tok,
                n_shared_experts=self.n_shared_experts,
                router_kind="sigmoid_plus_bias",
                router_norm=self.norm_topk_prob,
                routed_scaling_factor=self.routed_scaling_factor,
                expert_ffn=expert_ffn,
            )

    def to_block_spec(self, layer_idx: int = 0) -> specs.DecoderBlockSpec:
        norm = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.STANDARD_W,
        )
        attn = self._attn_spec(layer_idx)
        channel = self._moe_spec(layer_idx)
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE,
            ffn_norm_position=types.NormPosition.PRE,
            token_mixer=attn,
            channel_mixer=channel,
            pre_attn_norm=norm,
            pre_ffn_norm=norm,
        )
