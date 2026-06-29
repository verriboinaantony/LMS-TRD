"""
Efficient Bidirectional Feedback Unit / eBFU (Section 3.4).

Replaces NeRD-Rain's BFPU (a 3x3 conv over concatenated two-scale features,
O(C^2) per fusion) with a gated additive formulation (Eq. 4):

    z      = sigmoid( DWConv( [F_a ; F_b] ) )
    F_out  = F_a + z (elementwise) F_b

To keep the gate strictly O(C) while producing a C-channel gate from the
2C-channel concatenation, DWConv is a grouped 1x1 convolution with
groups = C (each group sees the {a, b} pair of one channel -> 1 output).
Parameter count = C * 2 = O(C), never O(C^2).
"""

import torch
import torch.nn as nn


class GatedAdditiveFusion(nn.Module):
    """Single-direction Eq. 4: F_out = F_a + sigmoid(DWConv([F_a;F_b])) * F_b."""

    def __init__(self, channels: int):
        super().__init__()
        # grouped 1x1: in=2C, out=C, groups=C  ->  2C params (O(C))
        self.gate = nn.Conv2d(2 * channels, channels, kernel_size=1,
                              groups=channels, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        # interleave a/b channels so each conv group covers one (a_i, b_i) pair
        B, C, H, W = fa.shape
        stacked = torch.stack([fa, fb], dim=2).reshape(B, 2 * C, H, W)
        z = self.act(self.gate(stacked))
        return fa + z * fb


class eBFU(nn.Module):
    """Bidirectional feedback between two same-shape scale features.

    Produces updated versions of BOTH inputs (matching the two arrows in
    Fig. 1c), each with its own gate:
        F_a' = F_a + sigma(DWConv([F_a;F_b])) * F_b
        F_b' = F_b + sigma(DWConv([F_b;F_a])) * F_a
    """

    def __init__(self, channels: int):
        super().__init__()
        self.a_from_b = GatedAdditiveFusion(channels)
        self.b_from_a = GatedAdditiveFusion(channels)

    def forward(self, fa: torch.Tensor, fb: torch.Tensor):
        fa_new = self.a_from_b(fa, fb)
        fb_new = self.b_from_a(fb, fa)
        return fa_new, fb_new
