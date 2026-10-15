"""
Shared building blocks.

These follow the official NeRD-Rain implementation (Chen, Pan & Dong, CVPR 2024,
github.com/cschenxiang/NeRD-Rain) so that ablation row M0 reproduces NeRD-Rain
exactly (see tools/verify_nerd_rain.py). All convolutions are bias-free unless
stated otherwise, as in NeRD-Rain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm with bias for (B, C, H, W) tensors.

    Numerically identical to NeRD-Rain's 'WithBias' LayerNorm (eps = 1e-5),
    applied over the channel dimension at every pixel.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class OverlapPatchEmbed(nn.Module):
    """3x3 convolution mapping the RGB input of one pyramid level to `dim` channels."""

    def __init__(self, in_c: int = 3, dim: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(in_c, dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """3x3 conv (C -> C/2) + PixelUnshuffle(2): halves resolution, doubles width."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """3x3 conv (C -> 2C) + PixelShuffle(2): doubles resolution, halves width."""

    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def resize(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Nearest-neighbour rescaling, exactly as used throughout NeRD-Rain
    (F.interpolate with its default mode) for the image pyramid, the
    coarse-to-fine image injection and the cross-scale feature transfers."""
    return F.interpolate(x, scale_factor=scale)
