"""PSNR / SSIM following the evaluation protocol of the Rain13K benchmark
(MPRNet, `evaluate_PSNR_SSIM.m`), which produced the baseline numbers in Table 2:

  1. the restored image is clipped to [0, 1] and quantised to 8 bits;
  2. both images are converted to the Y channel of YCbCr (ITU-R BT.601, as
     MATLAB's rgb2ycbcr on uint8 data: Y = 16 + (65.481 R + 128.553 G
     + 24.966 B) / 255, rounded);
  3. PSNR with peak 255; SSIM (Wang et al., 2004) with an 11x11 Gaussian window
     (sigma 1.5), K1 = 0.01, K2 = 0.03, L = 255, averaged over the 'valid' region.

For final paper numbers the official MATLAB script can also be run on the
images written by test.py --save-dir; this module is a faithful port.
RGB-space metrics are available with space='rgb' for monitoring only.
"""

import torch
import torch.nn.functional as F


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """[0,1] float -> rounded 0..255 values (kept as float tensor)."""
    return torch.round(img.clamp(0, 1) * 255.0)


def rgb_to_y(img255: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in 0..255 -> (B,1,H,W) Y channel, rounded like MATLAB uint8 rgb2ycbcr."""
    r, g, b = img255[:, 0:1], img255[:, 1:2], img255[:, 2:3]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return torch.round(y).clamp(0, 255)


def _gaussian_window(size=11, sigma=1.5, device=None, dtype=torch.float64):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def psnr(pred: torch.Tensor, target: torch.Tensor, space: str = 'y') -> torch.Tensor:
    """Mean PSNR over the batch. pred/target: (B,3,H,W) in [0,1]."""
    p, t = to_uint8(pred).double(), to_uint8(target).double()
    if space == 'y':
        p, t = rgb_to_y(p), rgb_to_y(t)
    mse = ((p - t) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10.0 * torch.log10(255.0 ** 2 / mse)).mean()


def ssim(pred: torch.Tensor, target: torch.Tensor, space: str = 'y') -> torch.Tensor:
    """Mean SSIM over the batch (valid-region averaging, as in ssim_index.m)."""
    p, t = to_uint8(pred).double(), to_uint8(target).double()
    if space == 'y':
        p, t = rgb_to_y(p), rgb_to_y(t)
    c = p.shape[1]
    w = _gaussian_window(device=p.device).expand(c, 1, 11, 11)
    mu1, mu2 = F.conv2d(p, w, groups=c), F.conv2d(t, w, groups=c)
    s11 = F.conv2d(p * p, w, groups=c) - mu1 ** 2
    s22 = F.conv2d(t * t, w, groups=c) - mu2 ** 2
    s12 = F.conv2d(p * t, w, groups=c) - mu1 * mu2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return m.flatten(1).mean(1).mean()
