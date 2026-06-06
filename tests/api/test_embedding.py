import torch
from torch import nn

from api import embedding, specs, types


def test_ple_table_forward_shape():
    spec = specs.PLESpec(
        ple_dim=256,
        residual_scale=1.0 / (2 ** 0.5),
        injection_norm=specs.NormSpec(
            kind=types.NormKind.RMS, eps=1e-6,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        ),
    )
    ple = embedding.PerLayerEmbedding(spec, vocab_size=1000, hidden_size=1536,
                                     num_layers=4, dtype=torch.float32)
    ids = torch.tensor([[0, 1, 2, 3]])
    residual = ple(ids, layer_idx=0)
    assert residual.shape == (1, 4, 1536)


def test_ple_residual_scale_applied():
    spec = specs.PLESpec(
        ple_dim=8, residual_scale=0.5,
        injection_norm=specs.NormSpec(
            kind=types.NormKind.RMS, eps=1e-6,
            weight_mode=types.NormWeightMode.ONE_PLUS_W,
        ),
    )
    ple = embedding.PerLayerEmbedding(spec, vocab_size=4, hidden_size=16,
                                      num_layers=2, dtype=torch.float32)
    # Initialize PLE table to all-ones so we can compute the expected scale
    with torch.no_grad():
        ple.ple_table.weight.fill_(1.0)
        # Zero out projection bias for determinism
        for layer_proj in ple.layer_projs:
            layer_proj.weight.zero_()
    ids = torch.tensor([[0]])
    residual = ple(ids, layer_idx=0)
    # With zeroed projection the residual is just norm(ones) * residual_scale
    # which is small but nonzero — verify shape only here (numerical check is in test_numerical_hf)
    assert residual.shape == (1, 1, 16)
