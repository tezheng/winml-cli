"""Mamba-2 numerical gate vs the HF reference.

We use a SMALL config (no download — instantiate HF Mamba2ForCausalLM with
random weights), then copy the state dict into our api-assembled block and
compare layer-0 forward outputs.

This isolates the SSD chunk-parallel scan against HF's `torch_forward`
(modeling_mamba2.py:398-585) which is the SAME numerical reference. So
the test is genuinely "did our refactor preserve the math" rather than
testing on noisy production weights.

The reference HF model used is the canonical
`state-spaces/mamba2-2.7b` config shape (downscaled to 64-dim for speed).

ATOL = 5e-4 (B7 numerical gate).
"""
import pytest
import torch

pytest.importorskip("transformers")

from api import kvcache
from models.mamba2 import config as m_config, layer as m_layer


ATOL = 5e-4
RTOL = 5e-4


def _small_mamba2_hf_config_dict():
    """Build a small Mamba2Config-compatible dict for fast tests."""
    return {
        "hidden_size": 64,
        "num_heads": 4,
        "head_dim": 32,        # 4 * 32 = 128 = 64 * 2 (expand)
        "state_size": 16,
        "conv_kernel": 4,
        "expand": 2,
        "n_groups": 2,
        "num_hidden_layers": 2,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "residual_in_fp32": True,
        "chunk_size": 8,
        "vocab_size": 100,
        "tie_word_embeddings": False,
    }


@pytest.fixture(scope="module")
def hf_small_model():
    """Construct a small HF Mamba2 with random init — no download."""
    from transformers import Mamba2Config, Mamba2ForCausalLM
    hf_cfg = Mamba2Config(**_small_mamba2_hf_config_dict())
    torch.manual_seed(42)
    model = Mamba2ForCausalLM(hf_cfg)
    model.eval()
    # Force fp32 throughout for numerical clarity.
    model.to(torch.float32)
    return model


def test_layer0_forward_matches_hf_small(hf_small_model):
    """Compare api/Mamba2Mixer + DecoderBlock against HF Mamba2Block forward."""
    hf_cfg_dict = hf_small_model.config.to_dict()
    cfg = m_config.Mamba2Config.from_hf_dict(hf_cfg_dict)

    api_blk = m_layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    full_sd = hf_small_model.state_dict()
    m_layer.load_hf_mamba2_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    # Run a fixed input through both.
    torch.manual_seed(0)
    B, S = 1, 9   # exercise pad path (9 % chunk_size=8 != 0)
    hidden = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)

    hf_layer = hf_small_model.backbone.layers[0]
    with torch.no_grad():
        # HF block takes the raw hidden_states and returns the post-residual.
        hf_out = hf_layer(hidden, cache_params=None, attention_mask=None)

    with torch.no_grad():
        api_out = api_blk(hidden)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Mamba2 layer-0 max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )


def test_layer0_forward_matches_hf_small_with_cache(hf_small_model):
    """Same comparison but with an explicit SSMStateCache on our side.

    Note: HF's Mamba2Block with `cache_params=None` does NOT touch a cache.
    On our side, passing a cache with `has_previous_state=False` still
    routes through the prefill path, so the outputs MUST match.
    """
    hf_cfg_dict = hf_small_model.config.to_dict()
    cfg = m_config.Mamba2Config.from_hf_dict(hf_cfg_dict)

    api_blk = m_layer.build_mamba2_decoder_layer(cfg, layer_idx=0)
    full_sd = hf_small_model.state_dict()
    m_layer.load_hf_mamba2_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    torch.manual_seed(0)
    B, S = 1, 16  # exact multiple of chunk_size=8
    hidden = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)

    hf_layer = hf_small_model.backbone.layers[0]
    with torch.no_grad():
        hf_out = hf_layer(hidden, cache_params=None, attention_mask=None)

    cache = kvcache.SSMStateCache(
        batch_size=B,
        conv_dim=api_blk.attention.conv_dim,
        conv_kernel=api_blk.attention.conv_kernel,
        n_heads=api_blk.attention.num_heads,
        head_dim=api_blk.attention.head_dim,
        d_state=api_blk.attention.d_state,
    )
    with torch.no_grad():
        api_out = api_blk(hidden, cache=cache)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"Mamba2 layer-0 (with cache) max_abs_diff={max_abs_diff:.6e}, atol={ATOL}"
    )
    # Cache should now be populated.
    assert cache.has_previous_state
