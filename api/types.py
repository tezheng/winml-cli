"""Enums and small leaf types. No PyTorch dependency."""
from __future__ import annotations
from enum import Enum, auto


class AttentionKind(Enum):
    STANDARD = auto()       # MHA / GQA / MQA — distinguished by n_kv_heads
    MLA = auto()            # Multi-head Latent Attention (DeepSeek)
    LINEAR_RETENTION = auto()
    LINEAR_DELTANET = auto()
    LINEAR_GLA = auto()
    DIFFERENTIAL = auto()
    DSA = auto()            # DeepSeek V3.2 Sparse Attention (Lightning Indexer)


class TokenMixerKind(Enum):
    """B7: top-level dispatch between attention and state-space token mixers.

    The DecoderBlockSpec.token_mixer is a Union[AttentionSpec, SSMSpec, SSDSpec].
    For pure attention models (everything pre-B7), the token_mixer is always
    AttentionSpec — `ATTENTION` is the implicit default and existing model
    factories don't have to set this field. For SSM and hybrid families, the
    factory chooses the appropriate kind per layer index in `to_block_spec`.

    Source-grounded: v3 design spec §5.2.5 lists `SSM_MAMBA1, SSM_MAMBA2,
    SSM_GRIFFIN, SSM_RWKV, HYBRID_PARALLEL` etc. For B7 we land
    `ATTENTION`, `SSM_MAMBA2`, and reserve `SSM_GRIFFIN` / `SSM_RWKV` for
    the deferred-family stubs.
    """
    ATTENTION = auto()      # default — AttentionSpec
    SSM_MAMBA1 = auto()     # SSMSpec — Mamba-1 selective scan (reserved)
    SSM_MAMBA2 = auto()     # SSDSpec — Mamba-2 SSD form (B7 lands this)
    SSM_GRIFFIN = auto()    # RecurrentGemma — reserved (deferred stub)
    SSM_RWKV = auto()       # RWKV-7 — reserved (deferred stub)


class QKVLayout(Enum):
    SPLIT = auto()          # separate q/k/v projections
    FUSED = auto()          # one big QKV projection (Phi-3)
    MLA_LATENT = auto()     # MLA-specific


class MaskKind(Enum):
    CAUSAL = auto()
    SWA = auto()
    SWA_GLOBAL_ALT = auto()
    SINK = auto()
    FULL = auto()
    CUSTOM = auto()
    BLOCK_SPARSE = auto()
    # B8: Visual Causal Flow / block-bidirectional mask. Used by
    # the original DeepSeek-OCR architecture (and reserved as a v3
    # spec hook): a leading "vision_token_count" prefix of keys/queries
    # attends bidirectionally (every vision token sees every other vision
    # token), and the trailing text region uses standard causal masking.
    # The cross-block keep rule is asymmetric: text queries CAN attend to
    # all preceding vision tokens, vision queries CANNOT attend to text
    # tokens (since they come first in the sequence anyway under typical
    # OCR-LLM prefix-fusion). NOTE: the HF v5.10.2 reference impl for
    # `deepseek_ocr2` uses STANDARD causal masking — block-bidirectional
    # is exercised by the B8 deepseek_ocr2 family ONLY as an alternative
    # mask op (the canonical numerical gate uses CAUSAL).
    # Source: original DeepSeek-OCR paper §3.2 (Visual Causal Flow).
    BLOCK_BIDIRECTIONAL = auto()


class QKNormPhase(Enum):
    NONE = auto()
    PRE_ROPE = auto()   # Qwen3, Gemma3 — verified PRE-RoPE in research/05 v2
    POST_ROPE = auto()


class QKNormShape(Enum):
    NONE = auto()
    PER_HEAD_DH = auto()    # Qwen3, Gemma3 — weight shape [head_dim]
    FULL_HDH = auto()       # OLMo 2 — weight shape [n_heads * head_dim]


class NormKind(Enum):
    RMS = auto()
    LAYER = auto()


class NormWeightMode(Enum):
    STANDARD_W = auto()     # y = x_normed * w
    ONE_PLUS_W = auto()     # y = x_normed * (1 + w) — Gemma


class NormPosition(Enum):
    PRE = auto()
    POST = auto()           # OLMo 2
    PRE_AND_POST = auto()   # Gemma 2 / 3


class Activation(Enum):
    SILU = auto()
    GELU = auto()
    GEGELU = auto()
    RELU2 = auto()


class GateKind(Enum):
    SWIGLU = auto()         # Llama, Qwen
    GEGLU = auto()          # Gemma
    GELU_ONLY = auto()      # no gating
    RELU2_ONLY = auto()


class RoPEBasis(Enum):
    INTERLEAVED = auto()    # GPT-J style
    SPLIT_HALF = auto()     # GPT-NeoX / Llama / Qwen style


class RoPEScaling(Enum):
    NONE = auto()
    PI = auto()
    NTK_STATIC = auto()
    NTK_DYNAMIC = auto()
    YARN = auto()           # DeepSeek-V2 / V3 — NTK + linear ramp + mscale
    LLAMA3 = auto()         # Llama 3 smooth scaling
    LONGROPE = auto()       # Phi-3 short/long


class CacheLayout(Enum):
    CONTIGUOUS = auto()     # HF baseline
    PAGED = auto()          # vLLM
    RING = auto()           # SWA wrap-around
    MLA_LATENT = auto()     # DeepSeek-V2/V3
    SSM_STATE = auto()      # Mamba


class MemoryLayout(Enum):
    HND = auto()            # [batch, heads, seq, dim]
    NHD = auto()            # [batch, seq, heads, dim]


class CacheOwnership(Enum):
    EXPLICIT_PASS = auto()  # PyTorch eager
    STATEFUL = auto()       # Core ML state op


class QDType(Enum):
    INT4 = auto()
    INT8 = auto()
    FP8_E4M3 = auto()
    FP8_E5M2 = auto()
    FP4 = auto()
    NF4 = auto()
    MX_FP4 = auto()


class PackingLayout(Enum):
    NONE = auto()
    NIBBLE_LSB = auto()
    NIBBLE_MSB = auto()
    AWQ_INTERLEAVE = auto()     # [0,2,4,6,1,3,5,7] nibble permutation
    GPTQ_INT32_PACK = auto()
    GGUF_K = auto()


class QuantRole(Enum):
    WEIGHT = auto()
    ACTIVATION = auto()
    KV_K = auto()
    KV_V = auto()
    ATTN_INTERNAL = auto()


class ShareScheme(Enum):
    """Cross-layer KV cache sharing pattern (axis A18 from research/01 v3 §6)."""
    NONE = auto()                 # default — every layer keeps its own K/V
    SAME_BLOCK_SHARED = auto()    # Gemma 4 E2B/E4B — last N layers reuse an earlier same-type layer's K/V
    CROSS_BLOCK_SHARED = auto()   # Apple AFM — 2-block split, Block-2 reuses Block-1
    # YOCO_PRODUCER_CONSUMER reserved for future research-grade additions
