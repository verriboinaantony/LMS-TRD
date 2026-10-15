"""
Rain Refinement Block (RRB, Sec. 3.5, row M8) and ECA channel attention.

The RRB is applied to the final full-resolution decoder feature F (C channels),
after the coarse-to-fine decoder skip and before the output convolution:

    R     = W_pw( ECA( DWConv5x5_dil2( GELU( DWConv3x3(F) ) ) ) )
    F_ref = F + R

DWConv3x3     : depthwise 3x3 (fine, < 3 px streak residues)
DWConv5x5_dil2: depthwise 5x5 with dilation 2 (9 px receptive field; halos)
ECA           : efficient channel attention (Wang et al., 2020): global average
                pooling, 1-D convolution across channels with adaptive kernel
                k = |(log2 C + 1) / 2|_odd (k = 3 for C = 32), sigmoid gating
W_pw          : pointwise 1x1 projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.k = k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(2, 3)).unsqueeze(1)                 # (B, 1, C)
        y = torch.sigmoid(self.conv(y)).squeeze(1)          # (B, C)
        return x * y[:, :, None, None]


class RRB(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5_d2 = nn.Conv2d(channels, channels, 5, padding=4, dilation=2,
                                groups=channels, bias=False)
        self.eca = ECA(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.dw5_d2(F.gelu(self.dw3(x)))
        return x + self.pw(self.eca(r))
