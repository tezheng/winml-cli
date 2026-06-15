"""BlockGraph — DAG topology of component nodes inside a decoder block.

A block is NOT a fixed `kind=PreNorm` enum tag; it's an explicit DAG. Different
wiring patterns (pre-norm, post-norm-reordered, parallel residual, sandwich)
are different graphs sharing the same node vocabulary.

The reserved input name `"input"` denotes the block's external input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._types import Spec
from .attention import AttentionSpec
from .ffn import FFNSpec
from .misc import AddSpec
from .norm import NormSpec
from .positional_encoding import PositionalEncodingSpec


_BLOCK_INPUT = "input"


@dataclass(frozen=True)
class Node:
    """One node in a BlockGraph DAG."""
    name: str                              # local label; "input" is reserved
    spec: Spec                             # any Spec subclass
    inputs: tuple[str, ...]                # source node names

    def __post_init__(self) -> None:
        if isinstance(self.inputs, list):
            raise TypeError("Node.inputs must be a tuple, not list.")
        if self.name == _BLOCK_INPUT:
            raise ValueError(f"Node name '{_BLOCK_INPUT}' is reserved for the block input.")


@dataclass(frozen=True)
class BlockGraph:
    """A DAG of component nodes forming a decoder block.

    Invariants enforced in __post_init__:
      - No duplicate node names.
      - Every referenced input is either `"input"` or a prior node's name.
      - `output` names a node that exists in the graph.
      - The implicit DAG (input -> ... -> output) is acyclic.
    """
    nodes: tuple[Node, ...]
    output: str

    def __post_init__(self) -> None:
        if isinstance(self.nodes, list):
            raise TypeError("BlockGraph.nodes must be a tuple, not list.")

        seen: set[str] = set()
        for n in self.nodes:
            if n.name in seen:
                raise ValueError(f"BlockGraph has duplicate node name: '{n.name}'.")
            seen.add(n.name)

        # Every input must reference a known earlier node or the block input.
        defined: set[str] = {_BLOCK_INPUT}
        for n in self.nodes:
            for inp in n.inputs:
                if inp not in defined:
                    raise ValueError(
                        f"Node '{n.name}' references unknown input '{inp}'. "
                        f"Inputs must reference '{_BLOCK_INPUT}' or a prior node."
                    )
            defined.add(n.name)

        if self.output not in seen:
            raise ValueError(
                f"BlockGraph.output '{self.output}' is not a defined node name."
            )


# ---- Factory helpers --------------------------------------------------------

def pre_norm_block(
    *,
    block_norm: NormSpec,
    attention: AttentionSpec,
    positional_encoding: Optional[PositionalEncodingSpec],
    post_attention_norm: NormSpec,
    ffn: FFNSpec,
) -> BlockGraph:
    """Standard pre-norm:
        x -> norm -> [rope] -> attn -> +x -> norm -> ffn -> +x

    For M0 RoPE lives outside the attention op (`rope_applied_inside=False`).
    Pass `positional_encoding=None` (or NoPESpec) to drop the rope node.
    """
    nodes: list[Node] = []

    # x -> norm
    nodes.append(Node("n0", block_norm, (_BLOCK_INPUT,)))

    # Optional rope between norm and attn (when applied outside attention)
    attn_input = "n0"
    if positional_encoding is not None and not getattr(attention, "rope_applied_inside", False):
        nodes.append(Node("rope", positional_encoding, ("n0",)))
        attn_input = "rope"

    # attn
    nodes.append(Node("attn", attention, (attn_input,)))

    # residual add: x + attn(...)
    nodes.append(Node("r0", AddSpec(), (_BLOCK_INPUT, "attn")))

    # post-attention norm
    nodes.append(Node("n1", post_attention_norm, ("r0",)))

    # ffn
    nodes.append(Node("ffn", ffn, ("n1",)))

    # residual add: r0 + ffn(n1)
    nodes.append(Node("r1", AddSpec(), ("r0", "ffn")))

    return BlockGraph(nodes=tuple(nodes), output="r1")


def post_norm_reordered_block(
    *,
    block_norm: NormSpec,
    attention: AttentionSpec,
    positional_encoding: Optional[PositionalEncodingSpec],
    post_attention_norm: NormSpec,
    ffn: FFNSpec,
) -> BlockGraph:
    """OLMo-2 reordered post-norm:
        x -> [rope] -> attn -> norm -> +x -> ffn -> norm -> +x

    The norm sits inside the residual branch (after the sublayer), unlike
    pre-norm where it sits before the sublayer.
    """
    nodes: list[Node] = []

    attn_input = _BLOCK_INPUT
    if positional_encoding is not None and not getattr(attention, "rope_applied_inside", False):
        nodes.append(Node("rope", positional_encoding, (_BLOCK_INPUT,)))
        attn_input = "rope"

    nodes.append(Node("attn", attention, (attn_input,)))
    nodes.append(Node("n0", block_norm, ("attn",)))
    nodes.append(Node("r0", AddSpec(), (_BLOCK_INPUT, "n0")))
    nodes.append(Node("ffn", ffn, ("r0",)))
    nodes.append(Node("n1", post_attention_norm, ("ffn",)))
    nodes.append(Node("r1", AddSpec(), ("r0", "n1")))

    return BlockGraph(nodes=tuple(nodes), output="r1")


def parallel_block(
    *,
    block_norm: NormSpec,
    attention: AttentionSpec,
    positional_encoding: Optional[PositionalEncodingSpec],
    ffn: FFNSpec,
) -> BlockGraph:
    """Falcon-7B / Cohere parallel-residual:
        x -> norm -> {attn, ffn} -> 3-way add(x, attn, ffn)

    One pre-norm feeds both sublayers in parallel; their outputs join via a
    3-way residual add at the block tail.
    """
    nodes: list[Node] = []
    nodes.append(Node("n0", block_norm, (_BLOCK_INPUT,)))

    attn_input = "n0"
    if positional_encoding is not None and not getattr(attention, "rope_applied_inside", False):
        nodes.append(Node("rope", positional_encoding, ("n0",)))
        attn_input = "rope"

    nodes.append(Node("attn", attention, (attn_input,)))
    nodes.append(Node("ffn", ffn, ("n0",)))
    nodes.append(Node("r0", AddSpec(), (_BLOCK_INPUT, "attn", "ffn")))

    return BlockGraph(nodes=tuple(nodes), output="r0")


def sandwich_block(
    *,
    pre_attention_norm: NormSpec,
    attention: AttentionSpec,
    positional_encoding: Optional[PositionalEncodingSpec],
    post_attention_norm: NormSpec,
    pre_ffn_norm: NormSpec,
    ffn: FFNSpec,
    post_ffn_norm: NormSpec,
) -> BlockGraph:
    """Gemma 2/3/4 sandwich: 4 norms surrounding the two sublayers.
        x -> n_pre_a -> [rope] -> attn -> n_post_a -> +x ->
              n_pre_f -> ffn -> n_post_f -> +
    """
    nodes: list[Node] = []
    nodes.append(Node("n_pre_a", pre_attention_norm, (_BLOCK_INPUT,)))

    attn_input = "n_pre_a"
    if positional_encoding is not None and not getattr(attention, "rope_applied_inside", False):
        nodes.append(Node("rope", positional_encoding, ("n_pre_a",)))
        attn_input = "rope"

    nodes.append(Node("attn", attention, (attn_input,)))
    nodes.append(Node("n_post_a", post_attention_norm, ("attn",)))
    nodes.append(Node("r0", AddSpec(), (_BLOCK_INPUT, "n_post_a")))
    nodes.append(Node("n_pre_f", pre_ffn_norm, ("r0",)))
    nodes.append(Node("ffn", ffn, ("n_pre_f",)))
    nodes.append(Node("n_post_f", post_ffn_norm, ("ffn",)))
    nodes.append(Node("r1", AddSpec(), ("r0", "n_post_f")))

    return BlockGraph(nodes=tuple(nodes), output="r1")


__all__ = [
    "Node", "BlockGraph",
    "pre_norm_block", "post_norm_reordered_block",
    "parallel_block", "sandwich_block",
]
