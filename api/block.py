"""DecoderBlock — assembles token mixer + channel mixer with residual structure.

M1 supports PRE-norm (Qwen3 / Llama default). B0.5 adds PRE_AND_POST (sandwich
norm — Gemma 2/3/4) and an optional `per_layer_residual` argument carrying the
PLE injection at each layer.
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
            types.NormPosition.PRE, types.NormPosition.PRE_AND_POST
        ):
            raise NotImplementedError(
                f"B0.5: PRE and PRE_AND_POST attn-norm only, got {spec.attn_norm_position}"
            )
        if spec.ffn_norm_position not in (
            types.NormPosition.PRE, types.NormPosition.PRE_AND_POST
        ):
            raise NotImplementedError(
                f"B0.5: PRE and PRE_AND_POST ffn-norm only, got {spec.ffn_norm_position}"
            )
        if not isinstance(spec.token_mixer, specs.AttentionSpec):
            raise NotImplementedError("M1: AttentionSpec token mixer only")
        if not isinstance(spec.channel_mixer, specs.FFNSpec):
            raise NotImplementedError("M1: FFNSpec channel mixer only")
        if spec.residual_scale is not None:
            raise NotImplementedError("M1: no residual scaling")
        self.spec = spec
        self.hidden_size = hidden_size

        # Attention sublayer norms
        if spec.attn_norm_position == types.NormPosition.PRE_AND_POST:
            if spec.pre_attn_norm is None or spec.post_attn_norm is None:
                raise ValueError("PRE_AND_POST attn norm requires pre and post norm specs")
            self.pre_attn_norm = norm.RMSNorm(spec.pre_attn_norm, hidden_size, dtype=dtype)
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
        else:  # PRE
            if spec.pre_ffn_norm is None:
                raise ValueError("PRE ffn-norm requires pre_ffn_norm")
            self.pre_ffn_norm = norm.RMSNorm(spec.pre_ffn_norm, hidden_size, dtype=dtype)
            self.post_ffn_sublayer_norm = None

        self.feedforward = feedforward.FeedForward(spec.channel_mixer, hidden_size,
                                                   dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: kvcache.ContiguousKVCache,
        start_pos: int,
        per_layer_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention sublayer
        attn_in = self.pre_attn_norm(x)
        attn_out = self.attention(attn_in, position_ids=position_ids,
                                  cache=cache, start_pos=start_pos)
        if self.post_attn_sublayer_norm is not None:
            attn_out = self.post_attn_sublayer_norm(attn_out)
        x = ops.add(x, attn_out)

        # FFN sublayer
        ffn_in = self.pre_ffn_norm(x)
        ffn_out = self.feedforward(ffn_in)
        if self.post_ffn_sublayer_norm is not None:
            ffn_out = self.post_ffn_sublayer_norm(ffn_out)
        x = ops.add(x, ffn_out)

        # PLE residual injection (Gemma 4) — supplied by the model assembly
        if per_layer_residual is not None:
            x = ops.add(x, per_layer_residual)
        return x
