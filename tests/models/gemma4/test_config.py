import torch

from api import types
from models.gemma4 import config


def test_gemma4_e2b_config_from_hf_dict():
    """E2B verified per research/issues/10-gemma4-investigation.md."""
    hf = {
        "model_type": "gemma4",
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "intermediate_size": 8192,
        "rope_theta": 10000.0,           # local layers
        "rope_global_theta": 1_000_000.0, # global layers
        "partial_rotary_factor_global": 0.25,
        "sliding_window": 512,
        "sliding_window_pattern": 4,       # 4:1 SWA:global on E2B
        "rms_norm_eps": 1e-6,
        "vocab_size": 262144,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "torch_dtype": "bfloat16",
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": None,
        "num_kv_shared_layers": 20,
        "use_per_layer_embedding": True,
        "ple_dim": 256,
        "attention_k_eq_v": False,         # E2B does NOT have K=V; that's 12B+
        "qk_norm_local_fixed_scale": 0.9916,
        "qk_norm_global_fixed_scale": 1.0228,
    }
    c = config.Gemma4Config.from_hf_dict(hf)
    assert c.hidden_size == 1536
    assert c.num_hidden_layers == 35
    assert c.num_kv_shared_layers == 20
    assert c.use_per_layer_embedding is True
    assert c.partial_rotary_factor_global == 0.25
    assert c.attention_k_eq_v is False


def test_gemma4_e2b_config_handles_dtype_key():
    """transformers 5.x emits 'dtype', not 'torch_dtype' — must handle both."""
    hf = _e2b_minimal_hf_dict()
    hf["dtype"] = "bfloat16"  # transformers 5.x key
    hf.pop("torch_dtype", None)
    c = config.Gemma4Config.from_hf_dict(hf)
    assert c.dtype == torch.bfloat16


def test_gemma4_e2b_to_block_spec_local_layer():
    c = _e2b_config()
    block_spec = c.to_block_spec(layer_idx=0)  # local layer
    assert block_spec.token_mixer.mask_kind == types.MaskKind.SWA
    assert block_spec.token_mixer.sliding_window == 512
    assert block_spec.token_mixer.rope.partial_rotary_factor == 1.0
    assert block_spec.token_mixer.rope.base_theta == 10000.0
    assert block_spec.attn_norm_position == types.NormPosition.PRE_AND_POST


def test_gemma4_e2b_to_block_spec_global_layer():
    c = _e2b_config()
    # In E2B with sliding_window_pattern=4, the global layer is every 5th layer
    # (4 local, 1 global). The last layer (index 34) is global.
    block_spec = c.to_block_spec(layer_idx=4)  # the first global layer
    assert block_spec.token_mixer.mask_kind == types.MaskKind.CAUSAL  # global = full causal
    assert block_spec.token_mixer.rope.partial_rotary_factor == 0.25
    assert block_spec.token_mixer.rope.base_theta == 1_000_000.0


def test_gemma4_e2b_shared_layer_indices():
    c = _e2b_config()
    # E2B: num_kv_shared_layers=20 means the last 20 layers reuse the K/V from
    # an earlier same-type layer.
    shared_map = c.kv_source_layer_idx_map()
    # Should be a dict {layer_idx: source_layer_idx} with 20 entries
    assert len(shared_map) == 20
    # No source layer is itself shared
    for src in shared_map.values():
        assert src not in shared_map


def _e2b_minimal_hf_dict():
    return {
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "intermediate_size": 8192,
        "rope_theta": 10000.0,
        "rope_global_theta": 1_000_000.0,
        "partial_rotary_factor_global": 0.25,
        "sliding_window": 512,
        "sliding_window_pattern": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 262144,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "torch_dtype": "bfloat16",
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": None,
        "num_kv_shared_layers": 20,
        "use_per_layer_embedding": True,
        "ple_dim": 256,
        "attention_k_eq_v": False,
        "qk_norm_local_fixed_scale": 0.9916,
        "qk_norm_global_fixed_scale": 1.0228,
    }


def _e2b_config():
    return config.Gemma4Config(
        hidden_size=1536, num_hidden_layers=35,
        num_attention_heads=8, num_key_value_heads=1,
        head_dim=256, global_head_dim=512,
        intermediate_size=8192,
        rope_theta_local=10000.0, rope_theta_global=1_000_000.0,
        partial_rotary_factor_global=0.25,
        sliding_window=512, sliding_window_pattern=4,
        rms_norm_eps=1e-6,
        vocab_size=262144, max_position_embeddings=32768,
        tie_word_embeddings=True,
        hidden_activation="gelu_pytorch_tanh",
        dtype=torch.float32,
        final_logit_softcap=30.0,
        attn_logit_softcap=None,
        num_kv_shared_layers=20,
        use_per_layer_embedding=True,
        ple_dim=256,
        attention_k_eq_v=False,
        qk_norm_local_fixed_scale=0.9916,
        qk_norm_global_fixed_scale=1.0228,
    )
