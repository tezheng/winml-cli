"""DecoderBlock — assembles token mixer + channel mixer with residual structure.

M1 supports PRE-norm (Qwen3 / Llama default). B0.5 adds PRE_AND_POST (sandwich
norm — Gemma 2/3/4). B3 adds POST (OLMo 2): no pre-norm, post-norm on the
sublayer output before the residual add.

B0.6 adds the Gemma 4 PLE-at-END injection: when `spec.per_layer_embedding`
is set, after the FFN residual add we apply:
    residual_pe = x
    x = per_layer_input_gate(x)         # Linear hidden → ple_dim
    x = gelu_pytorch_tanh(x)
    x = x * per_layer_input             # broadcast multiply
    x = per_layer_projection(x)         # Linear ple_dim → hidden
    x = post_per_layer_input_norm(x)    # RMSNorm
    x = residual_pe + x
Finally we apply a per-layer scalar buffer `layer_scalar` (initialised to 1.0,
loaded from state dict): `x = x * layer_scalar`.

Verified against `modeling_gemma4.py:1382` (layer_scalar buffer),
`modeling_gemma4.py:1384-1389` (PLE module init), and
`modeling_gemma4.py:1446-1455` (forward PLE+layer_scalar block).

B3 POST-norm structure (OLMo 2):
    h = x + post_attn_norm(attn(x))
    out = h + post_ffn_norm(ffn(h))
No `pre_attn_norm` / `pre_ffn_norm` modules are allocated; attention is fed
the raw residual. Source: `modeling_olmo2.py:295-333` (Olmo2DecoderLayer init
and forward).
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import nn

from api import attention as _attention, feedforward, kvcache, norm, ops, specs, types


class DecoderBlock(nn.Module):
    def __init__(
        self,
        spec: specs.DecoderBlockSpec,
        hidden_size: int,
        max_seq: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if spec.attn_norm_position not in (
            types.NormPosition.PRE, types.NormPosition.PRE_AND_POST,
            types.NormPosition.POST,
        ):
            raise NotImplementedError(
                f"B3: PRE, PRE_AND_POST, and POST attn-norm only, got {spec.attn_norm_position}"
            )
        if spec.ffn_norm_position not in (
            types.NormPosition.PRE, types.NormPosition.PRE_AND_POST,
            types.NormPosition.POST,
        ):
            raise NotImplementedError(
                f"B3: PRE, PRE_AND_POST, and POST ffn-norm only, got {spec.ffn_norm_position}"
            )
        if not isinstance(spec.token_mixer, specs.AttentionSpec):
            raise NotImplementedError("M1: AttentionSpec token mixer only")
        # B5: channel_mixer can be FFNSpec (dense) OR MoESpec (DeepSeek-V2/V3
        # MoE layers). The DecoderBlock dispatches between FeedForward and
        # MoE based on type. Source: modeling_deepseek_v2.py:404
        # (self.mlp = DeepseekV2Moe(config) if layer_idx >= first_k_dense_replace
        # else DeepseekV2MLP(config)).
        if not isinstance(spec.channel_mixer, (specs.FFNSpec, specs.MoESpec)):
            raise NotImplementedError(
                f"B5: channel_mixer must be FFNSpec or MoESpec, "
                f"got {type(spec.channel_mixer).__name__}"
            )
        # B2a: residual_scale enabled — Granite μP scalar (residual_multiplier).
        # When set, each sublayer output is multiplied by residual_scale BEFORE
        # the residual add. Source: modeling_granite.py:273,278 (Granite multiplies
        # `hidden_states * self.residual_multiplier` before `residual + ...`).
        self.spec = spec
        self.hidden_size = hidden_size
        self._residual_scale = spec.residual_scale

        # Attention sublayer norms
        if spec.attn_norm_position == types.NormPosition.PRE_AND_POST:
            if spec.pre_attn_norm is None or spec.post_attn_norm is None:
                raise ValueError("PRE_AND_POST attn norm requires pre and post norm specs")
            self.pre_attn_norm = norm.RMSNorm(spec.pre_attn_norm, hidden_size, dtype=dtype)
            self.post_attn_sublayer_norm = norm.RMSNorm(spec.post_attn_norm, hidden_size, dtype=dtype)
        elif spec.attn_norm_position == types.NormPosition.POST:
            # B3 POST (OLMo 2): NO pre-norm; post-norm wraps the attention output
            # before the residual add. Source: modeling_olmo2.py:315-326.
            if spec.post_attn_norm is None:
                raise ValueError("POST attn-norm requires post_attn_norm")
            if spec.pre_attn_norm is not None:
                raise ValueError(
                    "POST attn-norm must not also set pre_attn_norm "
                    "(OLMo 2 has only one norm per sublayer, post-side)"
                )
            self.pre_attn_norm = None
            self.post_attn_sublayer_norm = norm.RMSNorm(spec.post_attn_norm, hidden_size, dtype=dtype)
        else:  # PRE
            if spec.pre_attn_norm is None:
                raise ValueError("PRE attn-norm requires pre_attn_norm")
            self.pre_attn_norm = norm.RMSNorm(spec.pre_attn_norm, hidden_size, dtype=dtype)
            self.post_attn_sublayer_norm = None

        self.attention = _attention.Attention(spec.token_mixer, hidden_size,
                                              max_seq=max_seq, dtype=dtype)

        # FFN sublayer norms
        if spec.ffn_norm_position == types.NormPosition.PRE_AND_POST:
            if spec.pre_ffn_norm is None or spec.post_ffn_norm is None:
                raise ValueError("PRE_AND_POST ffn norm requires pre and post norm specs")
            self.pre_ffn_norm = norm.RMSNorm(spec.pre_ffn_norm, hidden_size, dtype=dtype)
            self.post_ffn_sublayer_norm = norm.RMSNorm(spec.post_ffn_norm, hidden_size, dtype=dtype)
        elif spec.ffn_norm_position == types.NormPosition.POST:
            # B3 POST (OLMo 2): NO pre-norm on FFN. Source: modeling_olmo2.py:329-332.
            if spec.post_ffn_norm is None:
                raise ValueError("POST ffn-norm requires post_ffn_norm")
            if spec.pre_ffn_norm is not None:
                raise ValueError(
                    "POST ffn-norm must not also set pre_ffn_norm"
                )
            self.pre_ffn_norm = None
            self.post_ffn_sublayer_norm = norm.RMSNorm(spec.post_ffn_norm, hidden_size, dtype=dtype)
        else:  # PRE
            if spec.pre_ffn_norm is None:
                raise ValueError("PRE ffn-norm requires pre_ffn_norm")
            self.pre_ffn_norm = norm.RMSNorm(spec.pre_ffn_norm, hidden_size, dtype=dtype)
            self.post_ffn_sublayer_norm = None

        # B5: dense FFN vs MoE dispatch. The attribute name `feedforward`
        # is kept for backward compat (all existing model factories and
        # weight loaders index it as `blk.feedforward`); for MoE this name
        # holds the MoE module — DeepSeek loaders use it identically.
        if isinstance(spec.channel_mixer, specs.MoESpec):
            self.feedforward = feedforward.MoE(spec.channel_mixer, hidden_size,
                                               dtype=dtype)
        else:
            self.feedforward = feedforward.FeedForward(spec.channel_mixer, hidden_size,
                                                       dtype=dtype)

        # B0.6: Per-Layer Embedding AT-END injection (Gemma 4 E2B/E4B).
        # When `spec.per_layer_embedding` is set, the block owns 3 extra tensors:
        # - per_layer_input_gate: Linear(hidden → ple_dim)
        # - per_layer_projection: Linear(ple_dim → hidden)
        # - post_per_layer_input_norm: RMSNorm(hidden)
        # plus a `layer_scalar` buffer of shape [1], multiplied as the final
        # output regardless of whether PLE is on or off (per HF — it's
        # initialised to ones and loaded from the state dict).
        self.per_layer_input_gate: Optional[nn.Linear] = None
        self.per_layer_projection: Optional[nn.Linear] = None
        self.post_per_layer_input_norm: Optional[norm.RMSNorm] = None
        if spec.per_layer_embedding is not None:
            ple = spec.per_layer_embedding
            self.per_layer_input_gate = nn.Linear(
                hidden_size, ple.ple_dim, bias=False, dtype=dtype,
            )
            self.per_layer_projection = nn.Linear(
                ple.ple_dim, hidden_size, bias=False, dtype=dtype,
            )
            self.post_per_layer_input_norm = norm.RMSNorm(
                ple.injection_norm, hidden_size, dtype=dtype,
            )
        # layer_scalar: HF registers it on EVERY Gemma 4 decoder layer
        # (modeling_gemma4.py:1382), even when PLE is disabled. Default 1.
        self.register_buffer(
            "layer_scalar", torch.ones(1, dtype=dtype), persistent=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: kvcache.ContiguousKVCache,
        start_pos: int,
        per_layer_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: [B, S, hidden_size]
            per_layer_input: optional [B, S, ple_dim] tensor. When the block
                was built with `spec.per_layer_embedding` set, this is the
                PLE table lookup for the current layer; it is REQUIRED to be
                non-None (HF accepts None, with zero-fill semantics in our
                tests handled at the call site).
        """
        # Attention sublayer
        # B3 POST (OLMo 2): no pre-norm; feed raw residual to attention.
        if self.pre_attn_norm is not None:
            attn_in = self.pre_attn_norm(x)
        else:
            attn_in = x
        attn_out = self.attention(attn_in, position_ids=position_ids,
                                  cache=cache, start_pos=start_pos)
        if self.post_attn_sublayer_norm is not None:
            attn_out = self.post_attn_sublayer_norm(attn_out)
        # B2a: Granite μP residual scaling — sublayer output is multiplied by
        # `residual_multiplier` BEFORE the residual add (modeling_granite.py:273).
        if self._residual_scale is not None:
            attn_out = attn_out * self._residual_scale
        x = ops.add(x, attn_out)

        # FFN sublayer
        if self.pre_ffn_norm is not None:
            ffn_in = self.pre_ffn_norm(x)
        else:
            ffn_in = x
        ffn_out = self.feedforward(ffn_in)
        if self.post_ffn_sublayer_norm is not None:
            ffn_out = self.post_ffn_sublayer_norm(ffn_out)
        if self._residual_scale is not None:
            ffn_out = ffn_out * self._residual_scale  # modeling_granite.py:278
        x = ops.add(x, ffn_out)

        # B0.6: PLE injection AT END (Gemma 4 only). Mirror modeling_gemma4.py:1446-1453.
        if self.per_layer_input_gate is not None and per_layer_input is not None:
            residual = x
            gated = self.per_layer_input_gate(x)
            gated = ops.gelu_pytorch_tanh(gated)
            gated = ops.mul(gated, per_layer_input)
            gated = self.per_layer_projection(gated)
            gated = self.post_per_layer_input_norm(gated)
            x = ops.add(residual, gated)

        # B0.6: trailing per-layer scalar multiply (HF modeling_gemma4.py:1455).
        # Applied to EVERY layer regardless of PLE.
        x = x * self.layer_scalar
        return x
