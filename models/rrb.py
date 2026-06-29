"""
Rain Refinement Block / RRB (Section 3.5) and ECA channel attention [13].

The RRB is a lightweight post-decoder residual module that suppresses the two
characteristic artifacts of lightweight transformer decoders:
  (a) fine-scale isolated streak pixels  (<3 px)
  (b) coarse-scale halo artifacts        (5-15 px)

It chains, in order:
    1. depthwise 3x3 conv                 (fine streaks)
    2. dilated depthwise 5x5 conv, d=2    (effective RF 9 px; halo artifacts)
    3. ECA channel attention              (amplify artifact-correlated channels)
    4. 1x1 pointwise projection
The block predicts only a residual that is added to the decoder features.
"""

import math
import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention [13]: GAP -> 1D conv (adaptive k) -> sigmoid."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = x.mean(dim=(2, 3))                 # GAP -> (B, C)
        y = y.unsqueeze(1)                     # (B, 1, C)
        y = self.conv(y)                       # 1D conv across channels
        y = self.sigmoid(y).squeeze(1)         # (B, C)
        return x * y.unsqueeze(-1).unsqueeze(-1)


class RainRefinementBlock(nn.Module):
    """RRB: residual artifact corrector (Section 3.5)."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                             groups=channels, bias=True)
        self.dw5_dil = nn.Conv2d(channels, channels, kernel_size=5, padding=4,
                                 dilation=2, groups=channels, bias=True)
        self.eca = ECA(channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.dw3(x)
        r = self.act(r)
        r = self.dw5_dil(r)
        r = self.eca(r)
        r = self.proj(r)
        return x + r                            # residual correction
