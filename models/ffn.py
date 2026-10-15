"""
Feed-forward networks.

GDFN  : NeRD-Rain's gated-Dconv FFN (from Restormer), used in rows M0-M2.
        hidden h = int(2.66 C); 1x1 (C -> 2h), depthwise 3x3 on 2h channels,
        gate GELU(x1) * x2, 1x1 (h -> C).
        Multiply-adds per pixel: 3hC + 18h  (= 7.98 C^2 + 47.9 C).

DSFFN : LMS-TRD's depthwise-separable FFN (Sec. 3.2), used from row M3.
        1x1 (C -> rC), depthwise 3x3 on rC channels, GELU, 1x1 (rC -> C), r = 4.
        Multiply-adds per pixel: 2 r C^2 + 9 r C  (= 8 C^2 + 36 C for r = 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class DSFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, bias: bool = False):
        super().__init__()
        hidden = dim * expansion
        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1,
                                groups=hidden, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_out(F.gelu(self.dwconv(self.project_in(x))))
