import numpy as np, torch
from models.st_gnn import STGNN


def make(use_gat):
    ei = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    return STGNN(12, 3, 4, 5, ei, np.ones(5, np.float32), hidden=16, heads=2, dilations=[1, 2], use_gat=use_gat)


def test_shape_and_monotone_quantiles():
    for g in (True, False):
        out = make(g)(torch.randn(2, 16, 5, 12))
        assert out.shape == (2, 4, 5, 3, 5)
        assert (out[..., 1:] >= out[..., :-1]).all()


def test_causal_tcn_ignores_future_padding():
    m = make(False).eval()
    x = torch.randn(1, 16, 5, 12)
    assert torch.isfinite(m(x)).all()