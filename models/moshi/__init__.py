"""Moshi 7B (Kyutai) main decoder — LM-decoder portion only.

Moshi is a 7B full-duplex audio LM. Architecturally, the "main" (Helium)
decoder is the LM-decoder portion. Text and audio (Mimi codec) token
streams pass through the SAME decoder — the dual-stream design lives at
the model level (different embeddings + multiple LM heads) and does NOT
change the per-layer code.

For B9 we ship only the per-layer LM decoder. The Mimi audio
encoder/decoder and the MoshiDepthDecoder (a smaller per-codebook
prediction head) are OUT OF SCOPE.

Verified against transformers/models/moshi/modeling_moshi.py
(MoshiDecoderLayer at L702-760, MoshiAttention at L406-509, MoshiGatingMLP
at L368-390, MoshiRMSNorm at L192-208).
"""
