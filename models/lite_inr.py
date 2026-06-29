"""
Lite-INR branch (Section 3.3).

Replaces NeRD-Rain's coordinate-MLP implicit representation with a two-level
multiresolution hash-grid encoder H_theta [12]. For a pixel coordinate
x in [-1, 1]^2 and a local backbone feature embedding f:

    I_hat(x) = M_phi( [ f || H_theta(x) ] )                         (Eq. 3)

where M_phi is a 3-layer MLP with hidden width 64. The penultimate MLP
feature is injected back into the SST backbone, and I_hat is supervised by
the auxiliary L1 loss ||I_INR - I_GT||_1.

The hash grid is implemented from scratch (dense indexing for coarse levels,
spatial hashing for fine levels) with bilinear interpolation, so there is no
dependency on tiny-cuda-nn / external CUDA kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Large primes for spatial hashing (Mueller et al. [12], 2D subset).
_PRIME_X = 1
_PRIME_Y = 2654435761


class HashGrid2D(nn.Module):
    """Two-level (configurable) 2D multiresolution hash-grid encoder."""

    def __init__(self, num_levels: int = 2, base_resolution: int = 16,
                 level_scale: float = 2.0, log2_table_size: int = 14,
                 feat_dim: int = 2):
        super().__init__()
        self.num_levels = num_levels
        self.feat_dim = feat_dim
        self.table_size = 2 ** log2_table_size

        resolutions, dense = [], []
        for lvl in range(num_levels):
            res = int(round(base_resolution * (level_scale ** lvl)))
            resolutions.append(res)
            # if the full vertex grid fits in the table, index densely
            dense.append((res + 1) * (res + 1) <= self.table_size)
        self.resolutions = resolutions
        self.dense = dense

        # one embedding table per level
        self.tables = nn.ParameterList([
            nn.Parameter(torch.empty(self.table_size, feat_dim).uniform_(-1e-4, 1e-4))
            for _ in range(num_levels)
        ])

    @property
    def out_dim(self) -> int:
        return self.num_levels * self.feat_dim

    def _hash(self, ix: torch.Tensor, iy: torch.Tensor, level: int) -> torch.Tensor:
        res = self.resolutions[level]
        if self.dense[level]:
            return (iy * (res + 1) + ix) % self.table_size
        h = (ix * _PRIME_X) ^ (iy * _PRIME_Y)
        return h % self.table_size

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (B, N, 2) in [-1, 1]. Returns (B, N, num_levels*feat_dim)."""
        B, N, _ = coords.shape
        feats = []
        for lvl in range(self.num_levels):
            res = self.resolutions[lvl]
            u = (coords * 0.5 + 0.5) * res            # [0, res]
            u = u.clamp(0, res - 1e-5)
            ux, uy = u[..., 0], u[..., 1]
            ix0 = torch.floor(ux).long()
            iy0 = torch.floor(uy).long()
            wx = (ux - ix0.float()).unsqueeze(-1)
            wy = (uy - iy0.float()).unsqueeze(-1)
            ix1 = (ix0 + 1).clamp(max=res)
            iy1 = (iy0 + 1).clamp(max=res)

            table = self.tables[lvl]
            f00 = table[self._hash(ix0, iy0, lvl)]
            f10 = table[self._hash(ix1, iy0, lvl)]
            f01 = table[self._hash(ix0, iy1, lvl)]
            f11 = table[self._hash(ix1, iy1, lvl)]

            # bilinear interpolation
            f0 = f00 * (1 - wx) + f10 * wx
            f1 = f01 * (1 - wx) + f11 * wx
            feats.append(f0 * (1 - wy) + f1 * wy)
        return torch.cat(feats, dim=-1)               # (B, N, out_dim)


class LiteINR(nn.Module):
    """Hash-grid INR branch: predicts a rain-free RGB proxy and an
    injection feature for the SST backbone."""

    def __init__(self, feat_channels: int = 128, embed_dim: int = 32,
                 hidden: int = 64, inject_channels: int = 128,
                 num_levels: int = 2, log2_table_size: int = 14):
        super().__init__()
        self.hashgrid = HashGrid2D(num_levels=num_levels,
                                   log2_table_size=log2_table_size)
        self.feat_embed = nn.Conv2d(feat_channels, embed_dim, kernel_size=1)

        in_dim = embed_dim + self.hashgrid.out_dim
        # 3-layer MLP M_phi implemented as 1x1 convs (per-pixel)
        self.fc1 = nn.Conv2d(in_dim, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.head = nn.Conv2d(hidden, 3, kernel_size=1)          # predicted RGB
        self.act = nn.ReLU(inplace=True)

        # projection of the MLP feature back into the backbone width
        self.inject = nn.Conv2d(hidden, inject_channels, kernel_size=1)

    @staticmethod
    def _coord_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2)
        return grid.expand(B, -1, -1)

    def forward(self, feat: torch.Tensor):
        """feat: (B, feat_channels, H, W) backbone embedding f (full res).

        Returns (inr_img: (B,3,H,W), inject_feat: (B,inject_channels,H,W))."""
        B, _, H, W = feat.shape
        coords = self._coord_grid(B, H, W, feat.device, feat.dtype)
        hashfeat = self.hashgrid(coords)                          # (B, N, hd)
        hashfeat = hashfeat.permute(0, 2, 1).reshape(B, -1, H, W) # (B, hd, H, W)

        f = self.feat_embed(feat)
        x = torch.cat([f, hashfeat], dim=1)                       # [ f || H(x) ]
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        inr_img = self.head(x)
        inject_feat = self.inject(x)
        return inr_img, inject_feat
