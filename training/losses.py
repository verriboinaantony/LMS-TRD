"""
Training objective (Sec. 3.6, Eq. 5-8).

The network returns restored images at the three pyramid levels
(I_hat_X, I_hat_M, I_hat_S) and the two INR proxies (P_M, P_S). With the
ground-truth pyramid G_X = I_GT, G_M = pyrdown(G_X), G_S = pyrdown(G_M)
(5x5 Gaussian blur + 2x bilinear downsampling, as in NeRD-Rain):

    L = sum_{l in {X,M,S}} [ L_char(I_hat_l, G_l) + lambda1 L_edge(I_hat_l, G_l)
                             + lambda2 L_freq(I_hat_l, G_l) ]
        + lambda3 ( ||P_M - G_M||_1 + ||P_S - G_S||_1 )

    lambda1 = 0.05, lambda2 = 0.01, lambda3 = 0.1   (the weights of NeRD-Rain)

L_char : mean sqrt((I_hat - G)^2 + eps^2), eps = 1e-3                    (Eq. 6)
L_edge : mean | Sobel(I_hat) - Sobel(G) |, Sobel x and y, per channel   (Eq. 7)
L_freq : mean | |FFT2(I_hat)| - |FFT2(G)| |  (unnormalised 2-D DFT)      (Eq. 8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pyr_down(x: torch.Tensor) -> torch.Tensor:
    """5x5 Gaussian blur (binomial [1,4,6,4,1]/16, reflect padding) + 2x bilinear downsampling."""
    k1 = torch.tensor([1., 4., 6., 4., 1.], device=x.device, dtype=x.dtype) / 16.0
    k2 = (k1[:, None] * k1[None, :]).expand(x.shape[1], 1, 5, 5)
    xb = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), k2, groups=x.shape[1])
    return F.interpolate(xb, size=(x.shape[-2] // 2, x.shape[-1] // 2), mode='bilinear',
                         align_corners=False)


def gt_pyramid(gt: torch.Tensor):
    g_m = pyr_down(gt)
    return gt, g_m, pyr_down(g_m)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer('kernel', torch.stack([kx, kx.t()]).unsqueeze(1))   # (2,1,3,3)

    def _grad(self, x):
        c = x.shape[1]
        w = self.kernel.to(x.dtype).repeat(c, 1, 1, 1)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), w, groups=c)

    def forward(self, pred, target):
        return F.l1_loss(self._grad(pred), self._grad(target))


class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        fp = torch.fft.fft2(pred.float())
        ft = torch.fft.fft2(target.float())
        return (fp.abs() - ft.abs()).abs().mean()


class LMSTRDLoss(nn.Module):
    def __init__(self, lambda_edge: float = 0.05, lambda_freq: float = 0.01,
                 lambda_inr: float = 0.1, char_eps: float = 1e-3):
        super().__init__()
        self.char = CharbonnierLoss(char_eps)
        self.edge = SobelEdgeLoss()
        self.freq = FrequencyLoss()
        self.l_edge, self.l_freq, self.l_inr = lambda_edge, lambda_freq, lambda_inr

    def forward(self, outputs, gt):
        out_x, out_m, out_s, proxy_m, proxy_s = outputs
        g_x, g_m, g_s = gt_pyramid(gt)
        char = edge = freq = 0.0
        for pred, g in ((out_x, g_x), (out_m, g_m), (out_s, g_s)):
            char = char + self.char(pred, g)
            edge = edge + self.edge(pred, g)
            freq = freq + self.freq(pred, g)
        inr = F.l1_loss(proxy_m, g_m) + F.l1_loss(proxy_s, g_s)
        total = char + self.l_edge * edge + self.l_freq * freq + self.l_inr * inr
        logs = {'loss': float(total.detach()), 'char': float(char.detach()),
                'edge': float(edge.detach()), 'freq': float(freq.detach()),
                'inr': float(inr.detach())}
        return total, logs
