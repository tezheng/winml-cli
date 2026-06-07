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
