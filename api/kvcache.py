"""KV cache primitives.

M1 implements ContiguousKVCache with EXPLICIT_PASS ownership in HND layout
(the HF / Qwen3 default). Paged / ring / MLA-latent / SSM-state come in
later milestones.
"""
from __future__ import annotations

import torch

from api import specs, types


class ContiguousKVCache:
    """A pre-allocated K/V buffer.

    HND layout: tensors are [B, n_kv_heads, max_seq, head_dim].
    This is the shape attention SDPA expects.
    """

    def __init__(
        self,
        spec: specs.KVCacheSpec,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq: int,
        device: torch.device | str = "cpu",
    ):
        if spec.layout != types.CacheLayout.CONTIGUOUS:
            raise NotImplementedError("M1 supports CONTIGUOUS layout only")
        if spec.memory_layout != types.MemoryLayout.HND:
            raise NotImplementedError("M1 supports HND memory layout only")
        if spec.k_quant is not None or spec.v_quant is not None:
            raise NotImplementedError("M1 supports unquantized KV cache only")
        self.spec = spec
        self.batch_size = batch_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq = max_seq
        self.k = torch.zeros(batch_size, n_kv_heads, max_seq, head_dim,
                             dtype=spec.k_dtype, device=device)
        self.v = torch.zeros(batch_size, n_kv_heads, max_seq, head_dim,
                             dtype=spec.v_dtype, device=device)
        self.seq_len = 0

    def write(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> None:
        """Write k, v at positions [start_pos, start_pos + k.shape[2]).

        k, v shape: [B, n_kv_heads, S_new, head_dim]
        """
        if k.dtype != self.spec.k_dtype:
            raise ValueError(
                f"k dtype mismatch: {k.dtype} vs spec k_dtype {self.spec.k_dtype}"
            )
        if v.dtype != self.spec.v_dtype:
            raise ValueError(
                f"v dtype mismatch: {v.dtype} vs spec v_dtype {self.spec.v_dtype}"
            )
        s_new = k.shape[2]
        end_pos = start_pos + s_new
        if end_pos > self.max_seq:
            raise ValueError(
                f"start_pos+S_new ({end_pos}) exceeds max_seq ({self.max_seq})"
            )
        self.k[:, :, start_pos:end_pos] = k
        self.v[:, :, start_pos:end_pos] = v
        self.seq_len = max(self.seq_len, end_pos)

    def read(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[:, :, :seq_len], self.v[:, :, :seq_len]

    def reset(self) -> None:
        self.seq_len = 0
