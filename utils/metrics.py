"""PSNR / SSIM metrics (Section 4.1).

PSNR and SSIM are computed on images in [0, 1]. SSIM uses an 11x11 Gaussian
window (sigma=1.5) following Wang et al., matching the convention used by
common deraining benchmarks.
"""

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    mse = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)
    mse = mse.clamp(min=1e-12)
    return (10.0 * torch.log10((max_val ** 2) / mse)).mean()


def _gaussian_window(window_size: int, sigma: float, channels: int,
                     device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    window_2d = (g @ g.t()).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    C = pred.shape[1]
    window = _gaussian_window(window_size, sigma, C, pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()
