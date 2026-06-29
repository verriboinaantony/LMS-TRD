"""
Composite training objective (Section 3.6):

    L = L_char + lambda1 * L_edge + lambda2 * L_freq + lambda3 * ||I_INR - I_GT||_1
    lambda1 = 0.05, lambda2 = 0.01, lambda3 = 0.1

Terms:
  * Charbonnier (Eq. 6):  mean sqrt((pred-gt)^2 + eps^2), eps = 1e-3
                          (the additive constant is eps^2 = 1e-6).
  * Edge (Eq. 7):         L1 between channel-wise Sobel gradients.
  * Frequency (Eq. 8):    L1 between |FFT2(pred)| and |FFT2(gt)|.
  * INR auxiliary:        L1 between the INR proxy image and the GT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class EdgeLoss(nn.Module):
    """L1 of channel-wise Sobel gradients (Sobel-x and Sobel-y)."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]])
        ky = kx.t().contiguous()
        # (2, 1, 3, 3): [Gx; Gy]
        kernel = torch.stack([kx, ky], dim=0).unsqueeze(1)
        self.register_buffer("kernel", kernel)

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        # apply both Sobel kernels to every channel independently
        w = self.kernel.repeat(C, 1, 1, 1)          # (2C, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(x, w, groups=C)             # (B, 2C, H, W) -> [Gx,Gy] per channel

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._grad(pred), self._grad(target))


class FrequencyLoss(nn.Module):
    """L1 between FFT magnitude spectra (Eq. 8)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = torch.fft.fft2(pred, norm='ortho')
        ft = torch.fft.fft2(target, norm='ortho')
        return F.l1_loss(fp.abs(), ft.abs())


class LMSTRDLoss(nn.Module):
    def __init__(self, lambda_edge: float = 0.05, lambda_freq: float = 0.01,
                 lambda_inr: float = 0.1, char_eps: float = 1e-3):
        super().__init__()
        self.char = CharbonnierLoss(eps=char_eps)
        self.edge = EdgeLoss()
        self.freq = FrequencyLoss()
        self.lambda_edge = lambda_edge
        self.lambda_freq = lambda_freq
        self.lambda_inr = lambda_inr

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                inr_pred: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        l_char = self.char(pred, target)
        l_edge = self.edge(pred, target)
        l_freq = self.freq(pred, target)
        total = l_char + self.lambda_edge * l_edge + self.lambda_freq * l_freq

        l_inr = pred.new_zeros(())
        if inr_pred is not None:
            l_inr = F.l1_loss(inr_pred, target)
            total = total + self.lambda_inr * l_inr

        logs = {
            "loss": total.detach().item(),
            "char": l_char.detach().item(),
            "edge": l_edge.detach().item(),
            "freq": l_freq.detach().item(),
            "inr": float(l_inr.detach().item()),
        }
        return total, logs
