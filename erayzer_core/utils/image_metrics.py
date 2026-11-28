"""Image metrics the evaluation reports, written in torch so the container needs no extra package.

The container image has neither torchmetrics nor skimage, and the project may not grow new packages,
so SSIM lives here. It follows Wang et al. 2004 with the usual settings of both libraries: an 11 by 11
Gaussian window with sigma 1.5, C1 = (0.01 L)^2 and C2 = (0.03 L)^2 for a data range L of 1, and the
statistics of every colour channel on its own. The window is applied with reflection padding, so the
map keeps the size of the image and a mask of the pixels a camera can render applies to it directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32) -> torch.Tensor:
    """Normalised separable Gaussian window, [window_size, window_size]."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    line = torch.exp(-(coords**2) / (2.0 * sigma**2))
    line = line / line.sum()
    return line[:, None] * line[None, :]


def ssim_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Per pixel SSIM of two image batches [n, c, h, w], same shape as the inputs.

    Half precision inputs, which the autocast of the renderer produces, are read as float32; float64
    stays float64, so a test can check the formula without the rounding of float32 in the way.
    """
    if pred.dtype not in (torch.float32, torch.float64):
        pred = pred.float()
    if target.dtype != pred.dtype:
        target = target.to(pred.dtype)
    channels = pred.shape[1]
    window = gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), window, groups=channels)

    mu_pred, mu_target = blur(pred), blur(target)
    mu_pred_sq, mu_target_sq, mu_cross = mu_pred * mu_pred, mu_target * mu_target, mu_pred * mu_target
    var_pred = blur(pred * pred) - mu_pred_sq
    var_target = blur(target * target) - mu_target_sq
    covariance = blur(pred * target) - mu_cross

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2.0 * mu_cross + c1) * (2.0 * covariance + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (var_pred + var_target + c2)
    )


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """SSIM per image, [n]. mask [n, 1, h, w] restricts the average to the pixels it marks."""
    smap = ssim_map(pred, target, window_size, sigma, data_range)
    if mask is None:
        return smap.flatten(1).mean(1)
    weights = mask.float().expand_as(smap)
    return (smap * weights).flatten(1).sum(1) / weights.flatten(1).sum(1).clamp_min(1.0)
