import torch
from models.losses import pinball_loss


def test_median_pinball():
    p, t = torch.zeros(4, 1), torch.ones(4)
    assert abs(pinball_loss(p, t, [0.5], torch.ones(4, dtype=torch.bool)).item() - 0.5) < 1e-6


def test_mask_ignores_invalid():
    p, t = torch.zeros(2, 1), torch.tensor([1.0, 100.0])
    assert abs(pinball_loss(p, t, [0.5], torch.tensor([True, False])).item() - 0.5) < 1e-6