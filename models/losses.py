import torch


def pinball_loss(pred, target, quantiles, mask):
    """pred (...,Q), target (...), mask (...) bool. Mean pinball loss over valid targets & quantiles."""
    q = torch.as_tensor(quantiles, device=pred.device, dtype=pred.dtype)
    err = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * err, (q - 1) * err) * mask.unsqueeze(-1)
    return loss.sum() / (mask.sum() * len(quantiles)).clamp(min=1)