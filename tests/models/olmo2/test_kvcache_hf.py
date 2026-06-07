"""OLMo 2 KV-cache prefill-then-decode test.

Loads `allenai/OLMo-2-0425-1B` for layer-0 weights, runs an N-token prefill,
then a 1-token decode at start_pos=N. Asserts cache.seq_len progression and
output stability against the same N+1-token prefill output position N.
"""
from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")

try:
    from transformers.models import olmo2  # noqa: F401
except ImportError:  # pragma: no cover
    pytest.skip(
        "transformers.models.olmo2 not available — install transformers>=5.10",
        allow_module_level=True,
    )

from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.olmo2 import config as o2_config, layer as o2_layer


MODEL_ID = "allenai/OLMo-2-0425-1B"
ATOL = 5e-4
RTOL = 5e-4


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    try:
        _ = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        return model
    except Exception as e:
        pytest.skip(
            f"OLMo 2 weights download/load failed: {type(e).__name__}: {str(e)[:200]}"
        )


def test_prefill_then_decode_olmo2_layer0(hf_model):
    cfg = o2_config.Olmo2Config.from_hf_dict(hf_model.config.to_dict())
    api_blk = o2_layer.build_olmo2_decoder_layer(
        cfg, layer_idx=0, max_seq=64,
    )
    o2_layer.load_hf_olmo2_layer(api_blk, hf_model.state_dict(), layer_idx=0)
    api_blk.eval()

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )

    # Reference: prefill N+1 tokens in one pass.
    N = 5
    x_full = torch.randn(1, N + 1, cfg.hidden_size)
    pos_full = torch.arange(N + 1)
    cache_full = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    with torch.no_grad():
        out_full = api_blk(x_full, position_ids=pos_full,
                           cache=cache_full, start_pos=0)
    assert cache_full.seq_len == N + 1

    # Prefill-then-decode: same N+1 inputs but in two passes.
    cache_pd = kvcache.ContiguousKVCache(
        cache_spec, batch_size=1,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    pos_prefill = torch.arange(N)
    pos_decode = torch.tensor([N])
    with torch.no_grad():
        out_prefill = api_blk(x_full[:, :N, :], position_ids=pos_prefill,
                              cache=cache_pd, start_pos=0)
        out_decode = api_blk(x_full[:, N:N+1, :], position_ids=pos_decode,
                             cache=cache_pd, start_pos=N)
    assert cache_pd.seq_len == N + 1
    # The decode-step output must match the full-prefill output at position N.
    assert torch.allclose(out_decode[0, 0], out_full[0, N], atol=ATOL, rtol=RTOL), (
        f"prefill-then-decode max_abs_diff="
        f"{(out_decode[0, 0] - out_full[0, N]).abs().max().item():.3e}"
    )
