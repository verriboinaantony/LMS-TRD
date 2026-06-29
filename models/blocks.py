"""
Core lightweight building blocks (Section 3.2). UNCHANGED from the faithful
reference implementation — only relocated here so both the flat and the U-Net
SST backbones can share them.

  * LayerNorm2d : channel-wise LayerNorm for (B, C, H, W) tensors.
  * DWMSA       : linear-complexity O(n) multi-head self-attention with
                  l2 feature normalisation and low-rank Q/K projections [16].
  * DSFFN       : depthwise-separable feed-forward network, expansion r=4 (Eq. 2).
  * SSTBlock    : pre-norm transformer block = DW-MSA + DS-FFN, both residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an (B, C, H, W) tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class DWMSA(nn.Module):
    """Depthwise Multi-head Self-Attention with linear O(n) complexity.

    Replaces softmax(QK^T)V with the linear-attention factorisation of
    Katharopoulos et al. [16], with low-rank Q/K projections and l2 feature
    normalisation. The kernel feature map is phi(.) = elu(l2norm(.)) + 1.
    Complexity is O(n * d_k * d_v) per head, linear in the number of tokens.
    """

    def __init__(self, dim: int, num_heads: int = 4, low_rank_ratio: int = 2,
                 eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.d_k = max(1, self.head_dim // low_rank_ratio)
        self.eps = eps

        self.q_proj = nn.Conv2d(dim, num_heads * self.d_k, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, num_heads * self.d_k, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        h, dk, dv = self.num_heads, self.d_k, self.head_dim

        q = self.q_proj(x).reshape(B, h, dk, N)
        k = self.k_proj(x).reshape(B, h, dk, N)
        v = self.v_proj(x).reshape(B, h, dv, N)

        q = F.elu(F.normalize(q, dim=2)) + 1.0
        k = F.elu(F.normalize(k, dim=2)) + 1.0

        kv = torch.einsum('bhdn,bhen->bhde', k, v)
        num = torch.einsum('bhdn,bhde->bhen', q, kv)
        k_sum = k.sum(dim=-1)
        den = torch.einsum('bhdn,bhd->bhn', q, k_sum)
        den = den.clamp(min=self.eps).unsqueeze(2)

        out = num / den
        out = out.reshape(B, C, H, W)
        return self.out_proj(out)


class DSFFN(nn.Module):
    """Depthwise-Separable Feed-Forward Network (Section 3.2, Eq. 2)."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.pw_expand = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1,
                            groups=hidden, bias=True)
        self.pw_project = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.dw(x)
        x = self.pw_project(x)
        return x


class SSTBlock(nn.Module):
    """One pre-norm transformer block: DW-MSA then DS-FFN, both residual."""

    def __init__(self, dim: int, num_heads: int = 4, expansion: int = 4,
                 low_rank_ratio: int = 2):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = DWMSA(dim, num_heads=num_heads, low_rank_ratio=low_rank_ratio)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = DSFFN(dim, expansion=expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
