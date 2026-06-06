"""DecoderBlock — assembles token mixer + channel mixer with residual structure.

M1 supports PRE-norm only (Qwen3 / Llama default). POST / PRE_AND_POST (OLMo 2,
Gemma) land in later milestones.
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
        if spec.attn_norm_position != types.NormPosition.PRE:
            raise NotImplementedError("M1: PRE attn-norm only")
        if spec.ffn_norm_position != types.NormPosition.PRE:
            raise NotImplementedError("M1: PRE ffn-norm only")
        if not isinstance(spec.token_mixer, specs.AttentionSpec):
            raise NotImplementedError("M1: AttentionSpec token mixer only")
        if not isinstance(spec.channel_mixer, specs.FFNSpec):
            raise NotImplementedError("M1: FFNSpec channel mixer only")
        if spec.residual_scale is not None:
            raise NotImplementedError("M1: no residual scaling")
        self.spec = spec
        self.hidden_size = hidden_size

        if spec.pre_attn_norm is None:
            raise ValueError("PRE attn-norm requires pre_attn_norm")
        if spec.pre_ffn_norm is None:
            raise ValueError("PRE ffn-norm requires pre_ffn_norm")

        self.pre_attn_norm = norm.RMSNorm(spec.pre_attn_norm, hidden_size, dtype=dtype)
        self.attention = _attention.Attention(spec.token_mixer, hidden_size,
                                              max_seq=max_seq, dtype=dtype)
        self.pre_ffn_norm = norm.RMSNorm(spec.pre_ffn_norm, hidden_size, dtype=dtype)
        self.feedforward = feedforward.FeedForward(spec.channel_mixer, hidden_size,
                                                   dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cache: kvcache.ContiguousKVCache,
        start_pos: int,
    ) -> torch.Tensor:
        attn_in = self.pre_attn_norm(x)
        attn_out = self.attention(attn_in, position_ids=position_ids,
                                  cache=cache, start_pos=start_pos)
        x = ops.add(x, attn_out)

        ffn_in = self.pre_ffn_norm(x)
        ffn_out = self.feedforward(ffn_in)
        return ops.add(x, ffn_out)
