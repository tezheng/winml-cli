"""MPT config + attention spec adapter.

For v5-phase2 V1 we expose `MptConfig.to_attention_spec()` only.
`to_block_spec()` is intentionally not provided — see __init__.py for
the rationale (MPT's ungated GELU FFN and LayerNorm-no-bias are not
yet in the IR).

Verified against `mosaicml/mpt-7b` config + `transformers.MptConfig`.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from api import specs, types


@dataclass(frozen=True)
class MptConfig:
    hidden_size: int                # d_model (4096 for MPT 7B)
    num_attention_heads: int        # n_heads (32 for MPT 7B)
    num_hidden_layers: int          # n_layers (32 for MPT 7B)
    max_position_embeddings: int    # max_seq_len (2048 for MPT 7B)
    alibi_bias_max: float = 8.0     # MPT default
    layer_norm_epsilon: float = 1e-5
    vocab_size: int = 50432
    dtype: torch.dtype = torch.float32

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def mpt_7b(cls) -> "MptConfig":
        """Canonical mosaicml/mpt-7b config."""
        return cls(
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=32,
            max_position_embeddings=2048,
        )

    def to_attention_spec(self) -> specs.AttentionSpec:
        """The MPT attention sublayer — ALiBi-only, MHA, fused-then-split
        Q/K/V on the IR side. Source: modeling_mpt.py:65-134."""
        return specs.AttentionSpec(
            n_q_heads=self.num_attention_heads,
            n_kv_heads=self.num_attention_heads,           # MPT is MHA
            head_dim=self.head_dim,
            kind=types.AttentionKind.STANDARD,
            qkv_layout=types.QKVLayout.SPLIT,
            mask_kind=types.MaskKind.CAUSAL,
            q_bias=False, k_bias=False, v_bias=False, o_bias=False,
            rope=None,                                     # MPT has no RoPE
            alibi=specs.AliBiSpec(
                n_heads=self.num_attention_heads,
                alibi_bias_max=self.alibi_bias_max,
            ),
        )
