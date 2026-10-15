"""
Token mixers.

MDTA  : NeRD-Rain's attention (the "multi-Dconv head transposed attention" of
        Restormer). It attends across CHANNELS: per head it forms a
        (C/h x C/h) map, so its cost is O(n * C^2 / h) -- already linear in the
        number of pixels n. Used in ablation rows M0-M3.

DWMSA : LMS-TRD's Depthwise Multi-head Self-Attention (Sec. 3.2), used from M4.
        It attends across SPATIAL positions with kernelised linear attention
        (Katharopoulos et al., 2020):

            q = DW3x3(W_q x),  k = DW3x3(W_k x),  v = DW3x3(W_v x)
            phi(u) = elu( u / ||u||_2 ) + 1          (l2-normalised feature map)
            y_i = W_o [ sum_j phi(q_i)^T phi(k_j) v_j  /  sum_j phi(q_i)^T phi(k_j) ]

        Q/K are low-rank: per head d_k = d_v / low_rank_ratio, d_v = C / h.
        Cost: O(n * h * d_k * d_v) for the attention product, linear in n.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    """NeRD-Rain / Restormer transposed (channel-wise) attention."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class DWMSA(nn.Module):
    """LMS-TRD depthwise multi-head self-attention with linear spatial attention."""

    def __init__(self, dim: int, num_heads: int, low_rank_ratio: int = 2,
                 bias: bool = False, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_v = dim // num_heads
        self.d_k = max(1, self.d_v // low_rank_ratio)
        self.eps = eps
        qk = num_heads * self.d_k
        self.split = (qk, qk, dim)
        # W_q, W_k, W_v as one 1x1 conv, followed by the depthwise 3x3 ("DW") conv
        self.qkv = nn.Conv2d(dim, 2 * qk + dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(2 * qk + dim, 2 * qk + dim, kernel_size=3, stride=1,
                                    padding=1, groups=2 * qk + dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        q, k, v = torch.split(self.qkv_dwconv(self.qkv(x)), self.split, dim=1)
        # the attention product is computed in float32 (sums over n tokens can
        # overflow float16 under mixed precision)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = q.float().reshape(b, self.num_heads, self.d_k, n)
            k = k.float().reshape(b, self.num_heads, self.d_k, n)
            v = v.float().reshape(b, self.num_heads, self.d_v, n)
            q = F.elu(F.normalize(q, dim=2)) + 1.0
            k = F.elu(F.normalize(k, dim=2)) + 1.0
            # (1/n) scaling of numerator and denominator cancels; it only keeps
            # the running sums O(1) in magnitude
            kv = torch.einsum('bhdn,bhen->bhde', k, v) / n          # (b, h, d_k, d_v)
            k_mean = k.mean(dim=-1)                                  # (b, h, d_k)
            z = torch.einsum('bhdn,bhd->bhn', q, k_mean).clamp_min(self.eps)
            out = torch.einsum('bhdn,bhde->bhen', q, kv) / z.unsqueeze(2)
        out = out.to(x.dtype).reshape(b, c, h, w)
        return self.project_out(out)
