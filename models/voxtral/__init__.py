"""Voxtral (Mistral.ai) audio-LM text-decoder portion.

Voxtral is Mistral's audio model (Whisper-style encoder + multi-modal
projector + LM decoder). For B9 we ship only the LM-decoder portion.
Audio encoder + projector are OUT OF SCOPE.

The canonical gate model `mistralai/Voxtral-Mini-3B-2507` ships with
`text_config.model_type='llama'` — i.e. the LM decoder is a plain
Llama-style block (PRE-norm RMSNorm STANDARD_W, SPLIT QKV, SwiGLU, no
biases, no QK-norm, SPLIT_HALF RoPE). Architecturally identical to
Mistral 7B v0.3 (no sliding window, theta=1e8 instead of 1e6,
rms_eps=1e-5, head_dim=128, GQA n_kv=8).

We delegate the per-layer assembly to a thin local config + factory.
The LM-decoder block IS identical to models/mistral's, so we re-use
api/block.DecoderBlock directly without inventing a new module.
"""
