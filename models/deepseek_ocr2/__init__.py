"""DeepSeek-OCR-2 LM-decoder portion (DeepseekV3-style MoE backbone).

NOTE: this package targets the HF transformers `deepseek_ocr2` reference
implementation. The HF source uses **standard MHA + softmax-router MoE
+ standard causal mask** — there is NO MLA and NO block-bidirectional
'Visual Causal Flow' mask in the HF v5.10.2 source (verified at
modeling_deepseek_ocr2.py:1073-1310). The Visual Causal Flow mask
described in the original DeepSeek-OCR paper §3.2 is reserved in our
api.types.MaskKind.BLOCK_BIDIRECTIONAL and exercised by a separate
shape-only test, NOT by the canonical numerical gate.
"""
