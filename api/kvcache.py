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
        v_head_dim: int | None = None,
    ):
        """Pre-allocated K/V buffer.

        Args:
            head_dim: per-head K dim. For standard attention, this is also V's
                head dim. For MLA (B2b — MiniCPM-3), V has a different head dim
                given via ``v_head_dim``.
            v_head_dim: optional override for V's per-head dim. Defaults to
                ``head_dim``. MiniCPM-3 reference path stores
                decompressed K at ``qk_head_dim`` (=96) and V at ``v_head_dim``
                (=64). Source: modeling_minicpm.py:457 (kv view uses
                qk_nope_head_dim + v_head_dim), then split at 461-463.
        """
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
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.max_seq = max_seq
        self.k = torch.zeros(batch_size, n_kv_heads, max_seq, head_dim,
                             dtype=spec.k_dtype, device=device)
        self.v = torch.zeros(batch_size, n_kv_heads, max_seq, self.v_head_dim,
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


class SharedLayerKVCache:
    """A read-only alias to another layer's ContiguousKVCache.

    Used by Gemma 4 E2B/E4B layers that reuse a prior same-type layer's K/V.
    `write` is a no-op for the shared layer because the K/V tensors it
    *would* have produced are discarded — the source cache is shared.

    For correctness, the source cache must have been written to before this
    layer's forward (i.e. the source layer must be earlier in the decoder
    stack).
    """

    def __init__(self, source_cache: "ContiguousKVCache"):
        self.source = source_cache
        # Mirror attributes the attention block reads
        self.spec = source_cache.spec
        self.k = source_cache.k
        self.v = source_cache.v

    @property
    def seq_len(self) -> int:
        return self.source.seq_len

    def write(self, k, v, start_pos):
        # No-op: the source layer is responsible for filling the cache.
        # We do not validate shapes here — the attention block still computes
        # its own K/V tensors but discards them.
        return

    def read(self, seq_len):
        return self.source.read(seq_len)

    def reset(self):
        # Resetting the shared cache is also a no-op (the source owns the buffer).
        return
