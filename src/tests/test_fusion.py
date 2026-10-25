import math

import pytest
import torch

from av_contradiction.models.tensor_fusion import LowRankTensorFusion


@pytest.mark.parametrize("dims,rank", [([8, 8, 8, 8], 4), ([5, 7], 1), ([16], 3)])
def test_low_rank_fusion_shapes_and_params(dims, rank):
    out_dim = 6
    f = LowRankTensorFusion(dims, out_dim, rank)
    y = f([torch.randn(3, d) for d in dims])
    assert y.shape == (3, out_dim)
    expected = rank * sum(d + 1 for d in dims) * out_dim + rank + out_dim
    norms = sum(2 * d for d in dims) + 2 * out_dim  # LayerNorm affine params
    assert f.num_parameters() == expected + norms
    assert f.full_tensor_weight_count() == math.prod(d + 1 for d in dims) * out_dim


def test_low_rank_fusion_captures_multiplicative_interaction():
    """Rank-1 fusion of two scalars can represent x*y exactly (outer-product term)."""
    f = LowRankTensorFusion([1, 1], 1, rank=1, dropout=0.0)
    f.norms = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    f.out_norm = torch.nn.Identity()
    with torch.no_grad():
        f.factors[0].copy_(torch.tensor([[[1.0], [0.0]]]))
        f.factors[1].copy_(torch.tensor([[[1.0], [0.0]]]))
        f.rank_weights.fill_(1.0)
        f.bias.zero_()
    x, y = torch.tensor([[2.0]]), torch.tensor([[3.0]])
    assert f([x, y]).item() == pytest.approx(6.0)
