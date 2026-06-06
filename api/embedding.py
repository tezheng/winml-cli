"""Per-Layer Embedding (PLE) table for Gemma 4 E2B/E4B (axis A19).

The PLE table holds vocab_size × ple_dim parameters (ple_dim much smaller than
hidden_size, e.g. 256 vs 1536). At every decoder layer, the per-token PLE row
is normalized, projected to hidden_size by a layer-specific linear, and added
as a residual scaled by `residual_scale` (Gemma 4 = 1/sqrt(2)).

On Gemma 4, the table is paged to flash storage on mobile — at runtime only
the per-layer linear projections live in VRAM, plus the PLE rows of the
currently-being-decoded tokens (a few KB).

This module owns the PLE table and one linear projection per decoder layer.
"""
from __future__ import annotations

import torch
from torch import nn

from api import norm as _norm, specs


class PerLayerEmbedding(nn.Module):
    def __init__(
        self,
        spec: specs.PLESpec,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 0,  # set later via `register_layers`
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.spec = spec
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # The PLE table itself
        self.ple_table = nn.Embedding(vocab_size, spec.ple_dim, dtype=dtype)
        # The pre-injection norm (RMSNorm 1+w to match Gemma)
        self.inj_norm = _norm.RMSNorm(spec.injection_norm, spec.ple_dim, dtype=dtype)
        # Per-layer linear projections from ple_dim -> hidden_size.
        # We allocate lazily: if num_layers=0 caller must call register_layers later.
        self.layer_projs = nn.ModuleList()
        if num_layers > 0:
            self.register_layers(num_layers, dtype=dtype)

    def register_layers(self, num_layers: int, dtype: torch.dtype = torch.float32) -> None:
        self.layer_projs = nn.ModuleList(
            nn.Linear(self.spec.ple_dim, self.hidden_size, bias=False, dtype=dtype)
            for _ in range(num_layers)
        )

    def forward(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Compute the per-layer residual for the given token ids and layer index.

        Returns a tensor of shape [B, S, hidden_size] to be added to the residual
        stream at decoder layer `layer_idx`.
        """
        if layer_idx >= len(self.layer_projs):
            raise IndexError(f"layer_idx {layer_idx} out of range "
                            f"(have {len(self.layer_projs)} layer projections)")
        ple_rows = self.ple_table(input_ids)              # [B, S, ple_dim]
        normed = self.inj_norm(ple_rows)                  # [B, S, ple_dim]
        projected = self.layer_projs[layer_idx](normed)   # [B, S, hidden_size]
        return projected * self.spec.residual_scale
