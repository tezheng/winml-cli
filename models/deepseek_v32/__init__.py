"""DeepSeek-V3.2 — DSA Lightning Indexer (shape-only).

V3.2 = V3 architecture + DeepSeek Sparse Attention (DSA). DSA adds a
lightweight 'indexer' head that scores (query, key) pairs cheaply via a
separate Q/K projection of `indexer_dim < head_dim`, top-k's along the
key axis, and runs the main SDPA on only the top-k keys per query.

Production weights are 671B; B5 ships SHAPE-ONLY support:
- IndexerSpec wired into AttentionSpec.
- AttentionKind.DSA composition path that allocates the indexer module
  but raises NotImplementedError at forward time (deferred to a future
  DSA-runtime batch).
"""
