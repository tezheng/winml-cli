"""Falcon-H1 — DEFERRED — pending B7+ follow-up.

Reason: Falcon-H1 (TII 2025) is a Mamba-2 + Transformer hybrid similar
in spirit to Granite-4-H but with a different layer pattern and a
production-scale checkpoint that requires substantial RAM to load even
the layer-0 weight subset for the numerical gate. Its modeling file is
also less mature in transformers (some glue still in progress at the
B7 cutoff).

Architecturally close to Granite-4-H — the Mamba-2 sublayer reuses our
api/ssm.Mamba2Mixer infrastructure. Implementation is mostly a config
adapter + weight loader.

Recommended for the B7+ follow-up. Should land in <200 LoC once tested
against a small synthetic Falcon-H1 config (similar workflow to
Granite-4-H).
"""
