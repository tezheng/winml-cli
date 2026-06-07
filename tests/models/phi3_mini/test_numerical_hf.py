"""Gate test: Phi-3-mini-4k-instruct decoder layer numerical equivalence vs HF.

Runs on CPU, fp32. Uses microsoft/Phi-3-mini-4k-instruct (~7.6 GB).

Exercises:
- FUSED QKV projection (Hq*Dh + 2*Hk*Dh = 9216 output dim).
- FUSED gate_up projection (2*I = 16384 output dim, chunk(2,-1) for gate/up).
- MHA n_q=n_kv=32, head_dim=96.
- SWA mask (sliding_window=2047), but with S=9 < 2047 the SWA mask equals
  the plain causal mask, so this test pass also gates the no-SWA path.
- Default RoPE θ=10000, partial_rotary_factor=1.0.
- No QK-norm, no biases.
"""
import os
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
from transformers import AutoConfig, AutoModelForCausalLM

from api import kvcache, specs, types
from models.phi3_mini import config as p_config, layer as p_layer


MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
FIXED_INPUT = torch.tensor([[101, 1024, 4789, 38, 9, 2, 1, 1024, 9]], dtype=torch.long)
ATOL = 5e-4
RTOL = 5e-4


def _phi3_weights_complete(cache_dir: str) -> bool:
    import glob
    for root in (cache_dir, os.path.join(cache_dir, "hub")):
        pat = os.path.join(
            root, "models--microsoft--Phi-3-mini-4k-instruct", "blobs", "*.incomplete"
        )
        if glob.glob(pat):
            return False
        snap = glob.glob(os.path.join(
            root, "models--microsoft--Phi-3-mini-4k-instruct", "snapshots", "*"
        ))
        if snap:
            return any(
                f.endswith(".safetensors") and not os.path.basename(f).startswith(".")
                for f in os.listdir(snap[0])
            )
    return False


@pytest.fixture(scope="module")
def hf_model():
    cache_dir = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "hf_cache"),
    )
    if not _phi3_weights_complete(cache_dir):
        pytest.skip(f"Phi-3-mini-4k weights not fully downloaded ({cache_dir})")
    try:
        _ = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            trust_remote_code=False,
        )
    except Exception as e:
        pytest.skip(
            f"Phi-3-mini-4k weights load failed: {type(e).__name__}: {str(e)[:200]}"
        )
    model.eval()
    return model


def test_layer0_forward_matches_hf(hf_model):
    hf_cfg_dict = hf_model.config.to_dict()
    cfg = p_config.Phi3MiniConfig.from_hf_dict(hf_cfg_dict)

    api_blk = p_layer.build_phi3_mini_decoder_layer(
        cfg, layer_idx=0, max_seq=64,
    )
    full_sd = hf_model.state_dict()
    p_layer.load_hf_phi3_mini_layer(api_blk, full_sd, layer_idx=0)
    api_blk.eval()

    with torch.no_grad():
        embed_out = hf_model.model.embed_tokens(FIXED_INPUT)
    B, S = FIXED_INPUT.shape

    hf_layer = hf_model.model.layers[0]
    position_ids = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        cos, sin = hf_model.model.rotary_emb(embed_out, position_ids)
        # 4D causal mask. With S=9 < sliding_window=2047, SWA == causal.
        attn_mask = torch.full((1, 1, S, S), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        hf_out = hf_layer(
            embed_out,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_values=None,
            use_cache=False,
        )

    cache_spec = specs.KVCacheSpec(
        layout=types.CacheLayout.CONTIGUOUS,
        memory_layout=types.MemoryLayout.HND,
        k_dtype=torch.float32, v_dtype=torch.float32,
    )
    cache = kvcache.ContiguousKVCache(
        cache_spec, batch_size=B,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, max_seq=64,
    )
    pos = torch.arange(S)
    with torch.no_grad():
        api_out = api_blk(embed_out, position_ids=pos, cache=cache, start_pos=0)

    max_abs_diff = (hf_out - api_out).abs().max().item()
    print(f"Phi-3-mini-4k layer-0 max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(hf_out, api_out, atol=ATOL, rtol=RTOL), (
        f"max_abs_diff={max_abs_diff:.6f}, atol={ATOL}"
    )
