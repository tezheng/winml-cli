"""Llama 4 Scout model config + adapter to api specs (per-layer iRoPE dispatch).

Verified against `transformers.Llama4TextConfig` and the public config of
`unsloth/Llama-4-Scout-17B-16E` (mirror of gated meta-llama/Llama-4-Scout-*).

Key Llama 4 Scout architectural facts (verified at modeling_llama4.py and
configuration_llama4.py):

- 48 decoder layers, hidden=5120, head_dim=128, n_q=40, n_kv=8 (GQA).
- RMSNorm STANDARD_W (modeling_llama4.py:122-139). eps=1e-5.
- iRoPE per-layer dispatch:
  * ``no_rope_layers[i] == 1`` -> RoPE layer (use_rope=True).
  * ``no_rope_layers[i] == 0`` -> NoPE layer (use_rope=False, RoPE skipped).
  * Default pattern (configuration_llama4.py:179-181):
      no_rope_layers[i] = int((i+1) % no_rope_layer_interval != 0)
    with default ``no_rope_layer_interval=4``. So for the 16E variant the
    NoPE layers fall at i = 3, 7, 11, ..., 47 (12 NoPE / 36 RoPE).
  * Source: modeling_llama4.py:338 (self.use_rope = no_rope_layers[layer_idx])
    and modeling_llama4.py:369-372 (the `if self.use_rope:` guard around
    apply_rotary_emb).

- INTERLEAVED RoPE basis (NOT split-half!). Source: modeling_llama4.py:245-254
  uses ``view_as_complex(xq.reshape(..., Dh/2, 2))`` and complex multiply by
  ``freqs_cis = polar(1, freqs)`` (line 239). This is the 3rd RoPE basis in
  the corpus after SPLIT_HALF (Llama/Qwen/Mistral/Gemma) and is handled by
  ``api.types.RoPEBasis.INTERLEAVED``.

- RoPE scaling: 16E variant uses LLAMA3 scaling with factor=8, low=1, high=4,
  original_context_length=8192 (config.json rope_scaling).

- QK-norm only on RoPE layers (modeling_llama4.py:351):
    self.config.use_qk_norm and self.use_rope -> instantiate self.qk_norm
  The qk_norm is ``Llama4TextL2Norm`` (modeling_llama4.py:107-119) — L2
  normalize the unit-sphere projection of the head. This is DISTINCT from
  the RMSNorm-based QK-norm in Gemma 3 / Qwen3 / OLMo 2. **The numerical
  gate is shape-only (weights gated AND too large to download) and we
  surface ``use_qk_norm`` on the config but do NOT plumb the L2-norm into
  the IR yet — the QK-norm shape we expose is None.** Surfacing L2-norm as
  a distinct QKNormShape lands together with the gated numerical gate.

- Attention temperature tuning on NoPE layers only (modeling_llama4.py:379-386):
    if attn_temperature_tuning and not self.use_rope:
        attn_scales = log1p(floor((pos+1)/floor_scale)) * attn_scale + 1.0
        query_states *= attn_scales
  This is also deferred — the shape test does not exercise it; the
  numerical gate will need it. Config field is preserved.

- ``layer_types[i]``: ``"chunked_attention"`` when ``no_rope_layers[i] == 0``,
  else ``"full_attention"`` (configuration_llama4.py:197-200). The chunked
  variant masks attention into non-overlapping chunks of size
  ``attention_chunk_size`` (default 8192). This is a 3rd mask family
  beyond CAUSAL / SWA — not supported in B4. The config preserves the field
  but ``to_attention_spec`` raises on layers with chunked attention. Full
  layers (``no_rope_layers[i] == 1``) use the standard causal mask path.

- MoE: ``moe_layers`` defaults to ``range(0, num_layers, interleave_moe_layer_step)``.
  For 16E (interleave_moe_layer_step=1) every layer is MoE. For 128E
  (Maverick, step=2) every other layer is MoE. **MoE wiring is deferred
  to B6.** B4 ships ``to_attention_spec(layer_idx)`` returning only the
  AttentionSpec — the channel mixer is not assembled.
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
class Llama4ScoutConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int           # MoE expert FFN dim
    intermediate_size_mlp: int       # dense MLP dim (when a layer is NOT MoE)
    num_hidden_layers: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: torch.dtype
    # iRoPE per-layer dispatch (see module docstring).
    no_rope_layers: tuple[int, ...] = ()
    no_rope_layer_interval: int = 4
    # Per-layer mask family from configuration_llama4.py:197-200.
    layer_types: tuple[str, ...] = ()
    # Chunk attention size for "chunked_attention" layers (NOT supported in B4).
    attention_chunk_size: Optional[int] = 8192
    # QK-norm flag (L2Norm on RoPE layers). Plumbed to IR in a later milestone.
    use_qk_norm: bool = True
    # Attention temperature tuning on NoPE layers.
    attn_temperature_tuning: bool = True
    floor_scale: int = 8192
    attn_scale: float = 0.1
    # MoE — surfaced for completeness; wiring lands in B6.
    moe_layers: tuple[int, ...] = ()
    interleave_moe_layer_step: int = 1
    num_local_experts: int = 16
    num_experts_per_tok: int = 1
    # Llama 3 RoPE smooth scaling parameters (16E variant uses these).
    rope_scaling_factor: Optional[float] = None
    rope_scaling_low_freq_factor: Optional[float] = None
    rope_scaling_high_freq_factor: Optional[float] = None
    rope_scaling_original_max_pos: Optional[int] = None
    attention_bias: bool = False

    @classmethod
    def from_hf_dict(cls, hf: dict) -> "Llama4ScoutConfig":
        # Llama 4 config nests text-side fields under ``text_config``.
        if "text_config" in hf and isinstance(hf["text_config"], dict):
            hf = hf["text_config"]

        dt_raw = hf.get("dtype", hf.get("torch_dtype", "bfloat16"))
        if isinstance(dt_raw, torch.dtype):
            dtype = dt_raw
        else:
            dtype = _TORCH_DTYPE_MAP.get(dt_raw, torch.bfloat16)

        # transformers 5.x nests rope_theta under ``rope_parameters``; older
        # configs use the flat ``rope_theta`` + ``rope_scaling``.
        rp = hf.get("rope_parameters")
        rs = hf.get("rope_scaling")
        if isinstance(rp, dict):
            rope_theta = float(rp.get("rope_theta", 500_000.0))
            rope_type = rp.get("rope_type", "default")
            if rope_type == "llama3":
                rope_scaling_factor = float(rp.get("factor", 8.0))
                rope_scaling_low_freq_factor = float(rp.get("low_freq_factor", 1.0))
                rope_scaling_high_freq_factor = float(rp.get("high_freq_factor", 4.0))
                rope_scaling_original_max_pos = int(
                    rp.get("original_max_position_embeddings", 8192)
                )
            else:
                rope_scaling_factor = None
                rope_scaling_low_freq_factor = None
                rope_scaling_high_freq_factor = None
                rope_scaling_original_max_pos = None
        elif isinstance(rs, dict):
            rope_theta = float(hf.get("rope_theta", 500_000.0))
            if rs.get("rope_type") == "llama3":
                rope_scaling_factor = float(rs.get("factor", 8.0))
                rope_scaling_low_freq_factor = float(rs.get("low_freq_factor", 1.0))
                rope_scaling_high_freq_factor = float(rs.get("high_freq_factor", 4.0))
                rope_scaling_original_max_pos = int(
                    rs.get("original_max_position_embeddings", 8192)
                )
            else:
                rope_scaling_factor = None
                rope_scaling_low_freq_factor = None
                rope_scaling_high_freq_factor = None
                rope_scaling_original_max_pos = None
        else:
            rope_theta = float(hf.get("rope_theta", 500_000.0))
            rope_scaling_factor = None
            rope_scaling_low_freq_factor = None
            rope_scaling_high_freq_factor = None
            rope_scaling_original_max_pos = None

        head_dim = hf.get("head_dim")
        if head_dim is None:
            head_dim = hf["hidden_size"] // hf["num_attention_heads"]

        num_layers = hf["num_hidden_layers"]
        no_rope_interval = int(hf.get("no_rope_layer_interval", 4))
        nrl = hf.get("no_rope_layers")
        # An EMPTY list also means "use default pattern" — matches the
        # published Scout config where ``no_rope_layers: []``.
        if nrl is None or (isinstance(nrl, list) and len(nrl) == 0):
            no_rope_layers = [
                int((i + 1) % no_rope_interval != 0) for i in range(num_layers)
            ]
        else:
            no_rope_layers = list(nrl)

        layer_types = hf.get("layer_types")
        if not layer_types:
            # configuration_llama4.py:197-200: chunked on NoPE layers, full on RoPE.
            layer_types = [
                "chunked_attention" if r == 0 else "full_attention"
                for r in no_rope_layers
            ]
        else:
            layer_types = list(layer_types)

        moe_layers = hf.get("moe_layers")
        moe_step = int(hf.get("interleave_moe_layer_step", 1))
        if not moe_layers:
            moe_layers = list(range(moe_step - 1, num_layers, moe_step))
        else:
            moe_layers = list(moe_layers)

        return cls(
            hidden_size=hf["hidden_size"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf.get(
                "num_key_value_heads", hf["num_attention_heads"]
            ),
            head_dim=head_dim,
            intermediate_size=hf.get("intermediate_size", 8192),
            intermediate_size_mlp=hf.get("intermediate_size_mlp", 16384),
            num_hidden_layers=num_layers,
            rope_theta=rope_theta,
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
            vocab_size=hf["vocab_size"],
            max_position_embeddings=hf["max_position_embeddings"],
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=dtype,
            no_rope_layers=tuple(no_rope_layers),
            no_rope_layer_interval=no_rope_interval,
            layer_types=tuple(layer_types),
            attention_chunk_size=hf.get("attention_chunk_size", 8192),
            use_qk_norm=bool(hf.get("use_qk_norm", True)),
            attn_temperature_tuning=bool(hf.get("attn_temperature_tuning", True)),
            floor_scale=int(hf.get("floor_scale", 8192)),
            attn_scale=float(hf.get("attn_scale", 0.1)),
            moe_layers=tuple(moe_layers),
            interleave_moe_layer_step=moe_step,
            num_local_experts=int(hf.get("num_local_experts", 16)),
            num_experts_per_tok=int(hf.get("num_experts_per_tok", 1)),
            rope_scaling_factor=rope_scaling_factor,
            rope_scaling_low_freq_factor=rope_scaling_low_freq_factor,
            rope_scaling_high_freq_factor=rope_scaling_high_freq_factor,
            rope_scaling_original_max_pos=rope_scaling_original_max_pos,
            attention_bias=bool(hf.get("attention_bias", False)),
        )

    def layer_uses_rope(self, layer_idx: int) -> bool:
        """``no_rope_layers[i] == 1`` means RoPE; ``0`` means NoPE.

        Source: modeling_llama4.py:338 (``self.use_rope =
        config.no_rope_layers[layer_idx]``). The field name is HF-misleading
        — a `1` selects RoPE, not NoPE.
        """
        if layer_idx >= len(self.no_rope_layers):
            return True
        return self.no_rope_layers[layer_idx] == 1

    def layer_is_moe(self, layer_idx: int) -> bool:
        """``True`` if layer_idx appears in ``moe_layers``. For the 16E
        Scout variant this is True for ALL 48 layers.
        """
        return layer_idx in self.moe_layers

    def to_attention_spec(self, layer_idx: int) -> specs.AttentionSpec:
        """Per-layer AttentionSpec — encodes the iRoPE dispatch.

        RoPE layers: INTERLEAVED basis, theta + LLAMA3 scaling carried from
        config, NoPE flag off (rope=RoPESpec(...)).
        NoPE layers: ``rope=None`` so attention.forward skips ``apply_rotary_emb``.

        Mask kind on RoPE layers is CAUSAL. NoPE layers in the published
        Scout variant carry ``layer_types[i] == "chunked_attention"`` —
        chunked attention is NOT in B4 scope and the call raises
        ``NotImplementedError``. (For a synthetic config that forces NoPE
        layers to ``"full_attention"`` the call succeeds.)
        """
        if self.layer_idx_has_chunked_attention(layer_idx):
            raise NotImplementedError(
                "Llama 4 'chunked_attention' mask is not supported in B4 — "
                "lands in a later milestone (mask families beyond CAUSAL / SWA)."
            )
        if self.layer_uses_rope(layer_idx):
            rope_spec: Optional[specs.RoPESpec] = self._build_rope_spec()
        else:
            rope_spec = None

        # QK-norm: Llama 4 uses L2Norm on RoPE layers only. The IR's QKNorm
        # primitive is RMSNorm-based; L2Norm lands with the gated numerical
        # gate. Until then we emit qk_norm=None to keep the spec consistent
        # with the build-only (no numerical) test path. The QK-norm field on
        # the config is preserved so downstream callers can inspect intent.
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
            rope=rope_spec,
        )
        return attn_spec

    def layer_idx_has_chunked_attention(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.layer_types):
            return False
        return self.layer_types[layer_idx] == "chunked_attention"

    def _build_rope_spec(self) -> specs.RoPESpec:
        if (
            self.rope_scaling_factor is not None
            and self.rope_scaling_original_max_pos is not None
        ):
            llama3_extra = specs.Llama3RoPEParams(
                factor=self.rope_scaling_factor,
                low_freq_factor=self.rope_scaling_low_freq_factor or 1.0,
                high_freq_factor=self.rope_scaling_high_freq_factor or 4.0,
                original_context_length=self.rope_scaling_original_max_pos,
            )
            scaling = types.RoPEScaling.LLAMA3
        else:
            llama3_extra = None
            scaling = types.RoPEScaling.NONE
        return specs.RoPESpec(
            base_theta=self.rope_theta,
            basis=types.RoPEBasis.INTERLEAVED,
            scaling=scaling,
            llama3_extra=llama3_extra,
        )
