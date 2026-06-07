"""SmolLM3-3B dense decoder layer with per-layer NoPE dispatch.

SmolLM3 is a Llama-3-shaped decoder that disables RoPE on every 4th layer
(`no_rope_layers[i] == 0`). The IR exposes per-layer NoPE via
`Smollm3Config.to_block_spec(layer_idx)` returning a `DecoderBlockSpec` whose
`token_mixer.rope` is `None` on NoPE layers.
"""
