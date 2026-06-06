"""Gemma 4 config + adapter to api specs.

Verified per-size values pulled from HF model-card config.json files (E2B, E4B,
12B-Unified, 26B-A4B, 31B); see research/issues/10-gemma4-investigation.md and
research/01-model-census.v3.md §3.5.

Gemma 4 architectural primitives (axis-deltas from Gemma 3):
- Per-layer-type partial RoPE on global only (axis A15): local layers use full
  rotation, global use partial_rotary_factor=0.25.
- K=V on global layers for 12B+ only (axis A1 sub-value).
- Cross-layer KV sharing on E2B/E4B (axis A18): num_kv_shared_layers=20/35 means
  >half of layers reuse the K/V of an earlier same-type layer.
- Per-Layer Embeddings (axis A19) on E2B/E4B only: 256-dim secondary embedding
  table whose output is injected as residual at every decoder layer.
- final_logit_softcap restored to 30.0 across all five sizes (Gemma 3 had
  dropped it).
- Fixed-scale QK norm (local 0.9916 / global 1.0228) absorbing 1/sqrt(head_dim)
  so the effective attention_scale is 1.0 instead of 1/sqrt(Dh).
- GeGLU with hidden_activation='gelu_pytorch_tanh' (Gemma signature preserved).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from api import specs, types


_TORCH_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class Gemma4Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int                       # local head_dim
    global_head_dim: int                # equals head_dim for 12B+; doubled for E2B
    intermediate_size: int
    rope_theta_local: float             # 10000.0
    rope_theta_global: float            # 1_000_000.0
    partial_rotary_factor_global: float # 0.25
    sliding_window: int                 # 512 (E2B) / 1024 (12B+)
    sliding_window_pattern: int         # 4 -> 4:1 (E2B) ; 5 -> 5:1 (12B+)
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    hidden_activation: str              # "gelu_pytorch_tanh"
    dtype: torch.dtype
    final_logit_softcap: Optional[float]
    attn_logit_softcap: Optional[float]
    num_kv_shared_layers: int           # E2B = 20; 12B+ = 0
    use_per_layer_embedding: bool
    ple_dim: int                        # 256
    attention_k_eq_v: bool              # True for 12B+; False for E2B/E4B
    qk_norm_local_fixed_scale: float
    qk_norm_global_fixed_scale: float

    @staticmethod
    def from_hf_dict(d: dict) -> "Gemma4Config":
        # transformers 5.x emits `dtype`; older 4.x emits `torch_dtype`. Accept either.
        dt_raw = d.get("dtype", d.get("torch_dtype", "bfloat16"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.bfloat16)

        return Gemma4Config(
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            global_head_dim=d.get("global_head_dim", d["head_dim"]),
            intermediate_size=d["intermediate_size"],
            rope_theta_local=float(d["rope_theta"]),
            rope_theta_global=float(d.get("rope_global_theta", d["rope_theta"])),
            partial_rotary_factor_global=float(d.get("partial_rotary_factor_global", 1.0)),
            sliding_window=d["sliding_window"],
            sliding_window_pattern=d.get("sliding_window_pattern", 4),
            rms_norm_eps=float(d["rms_norm_eps"]),
            vocab_size=d["vocab_size"],
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=bool(d["tie_word_embeddings"]),
            hidden_activation=d.get("hidden_activation", "gelu_pytorch_tanh"),
            dtype=dtype,
            final_logit_softcap=d.get("final_logit_softcapping"),
            attn_logit_softcap=d.get("attn_logit_softcapping"),
            num_kv_shared_layers=d.get("num_kv_shared_layers", 0),
            use_per_layer_embedding=bool(d.get("use_per_layer_embedding", False)),
            ple_dim=d.get("ple_dim", 256),
            attention_k_eq_v=bool(d.get("attention_k_eq_v", False)),
            qk_norm_local_fixed_scale=float(d.get("qk_norm_local_fixed_scale", 0.9916)),
            qk_norm_global_fixed_scale=float(d.get("qk_norm_global_fixed_scale", 1.0228)),
        )

    def is_global_layer(self, layer_idx: int) -> bool:
        """Gemma 4 alternates `sliding_window_pattern` SWA layers per global layer.

        For E2B with sliding_window_pattern=4: layers 0,1,2,3 local; layer 4 global;
        layers 5,6,7,8 local; layer 9 global; ... pattern repeats until layer 34.
        """
        return (layer_idx + 1) % (self.sliding_window_pattern + 1) == 0

    def to_block_spec(self, layer_idx: int) -> specs.DecoderBlockSpec:
        """Per-layer DecoderBlockSpec, dispatching on local vs global layer type.

        Local layers: SWA mask, local head_dim, theta=rope_theta_local,
            partial_rotary_factor=1.0, qk_norm fixed-scale = local.
        Global layers: full causal mask, global head_dim, theta=rope_theta_global,
            partial_rotary_factor=partial_rotary_factor_global (0.25 for E2B),
            qk_norm fixed-scale = global, attention_k_eq_v active only when
            the config flag is on (12B+).
        """
        is_global = self.is_global_layer(layer_idx)
        head_dim_eff = self.global_head_dim if is_global else self.head_dim
        attention_k_eq_v_eff = self.attention_k_eq_v and is_global
        qk_fixed = (self.qk_norm_global_fixed_scale if is_global
                    else self.qk_norm_local_fixed_scale)
        rope_theta_eff = self.rope_theta_global if is_global else self.rope_theta_local
        partial_eff = self.partial_rotary_factor_global if is_global else 1.0
        mask_eff = types.MaskKind.CAUSAL if is_global else types.MaskKind.SWA
        sw_eff = None if is_global else self.sliding_window

        norm_spec = specs.NormSpec(
            kind=types.NormKind.RMS, eps=self.rms_norm_eps,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        )
        qk_norm_spec = norm_spec
        attn = specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_key_value_heads,
            head_dim=head_dim_eff,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=mask_eff,
            sliding_window=sw_eff,
            qk_norm=qk_norm_spec,
            qk_norm_phase=types.QKNormPhase.PRE_ROPE,
            qk_norm_shape=types.QKNormShape.PER_HEAD_DH,
            qk_norm_fixed_scale=qk_fixed,
            attention_k_eq_v=attention_k_eq_v_eff,
            rope=specs.RoPESpec(
                base_theta=rope_theta_eff,
                basis=types.RoPEBasis.SPLIT_HALF,
                partial_rotary_factor=partial_eff,
            ),
        )
        ffn = specs.FFNSpec(
            intermediate_size=self.intermediate_size,
            activation=types.Activation.GELU,
            gate_kind=types.GateKind.GEGLU,
        )
        ple_spec = None
        if self.use_per_layer_embedding:
            ple_spec = specs.PLESpec(
                ple_dim=self.ple_dim,
                residual_scale=1.0 / (2 ** 0.5),
                injection_norm=norm_spec,
            )
        return specs.DecoderBlockSpec(
            attn_norm_position=types.NormPosition.PRE_AND_POST,
            ffn_norm_position=types.NormPosition.PRE_AND_POST,
            token_mixer=attn, channel_mixer=ffn,
            pre_attn_norm=norm_spec, post_attn_norm=norm_spec,
            pre_ffn_norm=norm_spec, post_ffn_norm=norm_spec,
            per_layer_embedding=ple_spec,
            final_logit_softcap=self.final_logit_softcap,
            embedding_scale=self.hidden_size ** 0.5,  # sqrt(D) - Gemma signature
        )

    def kv_source_layer_idx_map(self) -> dict[int, int]:
        """For Gemma 4 E2B/E4B: returns dict {layer_idx: source_layer_idx}.

        The shared layers are the LAST `num_kv_shared_layers` of each layer-type
        (local vs global) and they reuse the K/V of the layer-type-equivalent
        position from the LAST UNSHARED block. Concretely each shared layer
        sources from the layer with matching type-offset in the final unshared
        period block; no source is itself a shared layer.

        For E2B (35 layers, sliding_window_pattern=4, num_kv_shared=20):
        - Period = sliding_window_pattern + 1 = 5 (4 local + 1 global types).
        - Total layers = 35 = 7 blocks of 5.
        - Unshared prefix: 35 - 20 = 15 layers (3 blocks of 5).
        - Shared range: layers 15..34. Each shared layer i sources from the
          same-type slot in the last unshared block (layers 10..14), i.e.
          source[i] = ((i - shared_start) mod period) + (shared_start - period).
          - layer 15 -> source 10 (same type offset 0 within block)
          - layer 16 -> source 11, ..., layer 19 -> source 14
          - layer 20 -> source 10 (wraps), ..., layer 34 -> source 14.
        """
        period = self.sliding_window_pattern + 1   # 5 for E2B (4 local + 1 global)
        shared_start = self.num_hidden_layers - self.num_kv_shared_layers
        base = shared_start - period               # 10 for E2B
        return {
            i: ((i - shared_start) % period) + base
            for i in range(shared_start, self.num_hidden_layers)
        }
