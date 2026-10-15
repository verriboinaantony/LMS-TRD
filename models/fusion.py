"""
Cross-scale feedback fusion.

NeRD-Rain fuses latent features of successive cascaded stages with its
bidirectional feedback propagation unit (BFPU, class `Fusion` in the official
code). Three such fusions exist (two at the full-resolution level, one at the
middle level); their outputs are fed back to the next-coarser level, which -- together
with the coarse-to-fine INR injection -- gives the bidirectional information flow.

BFPU (rows M0-M5): two full 3x3 convolutions (C -> C) form a sigmoid co-attention
     map, and two 3x3 convolutions (2C -> 2) predict per-pixel mixing weights.

eBFU (rows M6-M8, Sec. 3.4): gated additive fusion

        z     = sigmoid( GConv1x1( [F_a ; F_b] ) )
        F_out = F_a + z * F_b

     where GConv1x1 is a grouped 1x1 convolution with C groups acting on the
     channel-interleaved concatenation (a_1, b_1, a_2, b_2, ...): each output
     channel sees only the pair (a_i, b_i), i.e. a depthwise operation on the
     two scales. Parameters: 3C (2C weights + C biases), i.e. O(C) instead of O(C^2).
"""

import torch
import torch.nn as nn


class BFPU(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        attention = torch.sigmoid(self.query_conv(x) * self.key_conv(y))
        ax, ay = x * attention, y * attention
        xg = self.gamma1(torch.cat((x, ax), dim=1))
        yg = self.gamma2(torch.cat((y, ay), dim=1))
        x_out = x * xg[:, [0]] + ax * xg[:, [1]]
        y_out = y * yg[:, [0]] + ay * yg[:, [1]]
        return x_out + y_out


class EBFU(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.gate = nn.Conv2d(2 * in_dim, in_dim, kernel_size=1, groups=in_dim, bias=True)

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fa.shape
        pairs = torch.stack([fa, fb], dim=2).reshape(b, 2 * c, h, w)   # (a1, b1, a2, b2, ...)
        return fa + torch.sigmoid(self.gate(pairs)) * fb
