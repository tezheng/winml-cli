"""Graph-primitive specs — Add, Mul.

These are primitives used as nodes inside `BlockGraph`. They are NOT
components in the registry sense (no docs page, no Variant catalog) — they
exist purely to wire residuals and elementwise combines into the block DAG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._types import Spec


@dataclass(frozen=True)
class AddSpec(Spec):
    """N-way elementwise add. Used for residual paths. No tunable attributes;
    arity comes from the surrounding Node's `inputs` length."""
    kind: Literal["Add"] = "Add"


@dataclass(frozen=True)
class MulSpec(Spec):
    """Elementwise multiply (Hadamard product)."""
    kind: Literal["Mul"] = "Mul"


__all__ = ["AddSpec", "MulSpec"]
