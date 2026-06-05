"""llm-layers API — minimal primitives for SLM decoder blocks.

The public surface lives in submodules:
- api.types: enums and small types
- api.specs: frozen dataclasses (parameter spaces)
- api.ops: pure functional primitives
- api.norm, api.rope, api.kvcache, api.quant, api.attention,
  api.feedforward, api.block: nn.Module building blocks
"""
