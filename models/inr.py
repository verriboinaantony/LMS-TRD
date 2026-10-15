"""
Implicit neural representation (INR) branches.

NeRD-Rain places one INR after the coarse (1/4) level and one after the middle
(1/2) level. Each INR receives the upsampled latent feature F of that level
(C channels, same resolution as the level's image), predicts an RGB image that
is added to the level's input to form a rain-free proxy, and that proxy is
injected (upsampled x2) into the input of the next finer level. The proxies are
supervised with an auxiliary L1 loss. LMS-TRD keeps this branch and its data
flow unchanged and only replaces what happens inside it.

LIIFINR : NeRD-Rain's INR (rows M0-M4), reproduced from the official code.
          Input per pixel = [3x3-unfolded feature (9C), relative coordinate +
          positional encoding (18), cell size (2)] -> MLP 256-256-256 -> RGB.
          NOTE (verified in tools/verify_nerd_rain.py): because the INR is
          queried on its own feature grid, the relative coordinates are
          identically zero and the cell term is constant, and the four
          "local-ensemble" queries are identical. We keep the four evaluations
          so that M0 has exactly the cost of the official implementation.

LiteINR : LMS-TRD's Lite-INR (Sec. 3.3, rows M5-M8):

              I_hat(x) = M_phi( [ f(x) || H_theta(x) ] )

          f(x)       : 3x3-unfolded local feature of F at pixel x (9C values),
          H_theta(x) : two-level multiresolution hash-grid encoding of the
                       pixel-centre coordinate x in [-1, 1]^2 (Mueller et al.,
                       2022): L = 2 levels, F = 2 features per level,
                       grid resolutions N_l = 16 * 4^l (16, 64), table size
                       T = 2^12; a level whose (N_l+1)^2 vertices fit in T is
                       stored densely (Instant-NGP convention), otherwise it is
                       indexed with the spatial hash (x * 1) XOR (y * 2654435761)
                       mod T; bilinear interpolation of the 4 cell corners,
          M_phi      : 3-layer MLP, hidden width 64, ReLU, output RGB.
          The first MLP layer acting on the unfolded feature is implemented as a
          3x3 convolution, which is mathematically identical.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord(shape, flatten: bool = True, device=None) -> torch.Tensor:
    """Pixel-centre coordinates in (-1, 1), as in NeRD-Rain / LIIF.
    Returns (h*w, 2) if flatten else (h, w, 2); [..., 0] is the row axis."""
    seqs = []
    for n in shape:
        r = 1.0 / n
        seqs.append(-1 + r + (2 * r) * torch.arange(n, device=device).float())
    ret = torch.stack(torch.meshgrid(*seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


# --------------------------------------------------------------------------- #
#  NeRD-Rain INR (official behaviour)
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_list):
        super().__init__()
        layers, last = [], in_dim
        for hdim in hidden_list:
            layers += [nn.Linear(last, hdim), nn.ReLU()]
            last = hdim
        layers.append(nn.Linear(last, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        return self.layers(x.view(-1, x.shape[-1])).view(*shape, -1)


class LIIFINR(nn.Module):
    L = 4                      # positional-encoding frequencies
    HIDDEN = (256, 256, 256)   # MLP widths

    def __init__(self, dim: int, ensemble_evals: int = 4):
        super().__init__()
        in_dim = dim * 9 + (2 + 4 * self.L) + 2
        self.imnet = _MLP(in_dim, 3, self.HIDDEN)
        self.ensemble_evals = ensemble_evals

    def _pe(self, x: torch.Tensor) -> torch.Tensor:
        freq = (2 ** torch.arange(self.L, dtype=torch.float32, device=x.device)) * math.pi
        spec = x[..., None] * freq
        enc = torch.stack([spec.sin(), spec.cos()], dim=-2)
        return enc.view(*x.shape[:-1], -1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        coord = make_coord((h, w), device=feat.device).unsqueeze(0).repeat(b, 1, 1)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / h
        cell[..., 1] *= 2 / w
        coord = torch.cat([coord, self._pe(coord)], dim=-1)

        f = F.unfold(feat, 3, padding=1).view(b, c * 9, h, w)
        q_feat = f.view(b, c * 9, -1).permute(0, 2, 1)
        q_coord = make_coord((h, w), flatten=False, device=feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w).reshape(b, 2, -1).permute(0, 2, 1)
        q_coord = torch.cat([q_coord, self._pe(q_coord)], dim=-1)
        rel_coord = coord - q_coord                       # identically zero (see note)
        rel_coord[..., 0] *= h
        rel_coord[..., 1] *= w
        rel_cell = cell.clone()
        rel_cell[..., 0] *= h
        rel_cell[..., 1] *= w
        inp = torch.cat([q_feat, rel_coord, rel_cell], dim=-1)

        preds, areas = [], []
        for _ in range(self.ensemble_evals):
            preds.append(self.imnet(inp.view(b * h * w, -1)).view(b, h * w, -1))
            areas.append(torch.abs(rel_coord[..., 0] * rel_coord[..., 1]) + 1e-9)
        tot = torch.stack(areas).sum(dim=0)
        if self.ensemble_evals == 4:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]
        ret = 0
        for p, a in zip(preds, areas):
            ret = ret + p * (a / tot).unsqueeze(-1)
        return ret.view(b, h, w, -1).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------- #
#  LMS-TRD Lite-INR
# --------------------------------------------------------------------------- #
_PRIME_Y = 2654435761


class HashGrid2D(nn.Module):
    def __init__(self, num_levels: int = 2, base_resolution: int = 16, growth: int = 4,
                 log2_table_size: int = 12, feat_dim: int = 2):
        super().__init__()
        self.table_size = 2 ** log2_table_size
        self.resolutions = [base_resolution * growth ** l for l in range(num_levels)]
        self.dense = [(n + 1) ** 2 <= self.table_size for n in self.resolutions]
        sizes = [min(self.table_size, (n + 1) ** 2) for n in self.resolutions]
        self.tables = nn.ParameterList([
            nn.Parameter(torch.empty(s, feat_dim).uniform_(-1e-4, 1e-4)) for s in sizes])
        self.out_dim = num_levels * feat_dim

    def _index(self, ix: torch.Tensor, iy: torch.Tensor, level: int) -> torch.Tensor:
        n = self.resolutions[level]
        if self.dense[level]:
            return iy * (n + 1) + ix
        return torch.bitwise_xor(ix, iy * _PRIME_Y) % self.table_size

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (P, 2) in (-1, 1)  ->  (P, num_levels * feat_dim)."""
        out = []
        for lvl, n in enumerate(self.resolutions):
            u = (coords + 1.0) * 0.5 * n
            uy, ux = u[:, 0], u[:, 1]
            iy0 = torch.floor(uy).long().clamp(0, n - 1)
            ix0 = torch.floor(ux).long().clamp(0, n - 1)
            wy = (uy - iy0.to(uy.dtype)).unsqueeze(-1)
            wx = (ux - ix0.to(ux.dtype)).unsqueeze(-1)
            t = self.tables[lvl]
            f00 = t[self._index(ix0, iy0, lvl)]
            f10 = t[self._index(ix0 + 1, iy0, lvl)]
            f01 = t[self._index(ix0, iy0 + 1, lvl)]
            f11 = t[self._index(ix0 + 1, iy0 + 1, lvl)]
            out.append((f00 * (1 - wx) + f10 * wx) * (1 - wy) + (f01 * (1 - wx) + f11 * wx) * wy)
        return torch.cat(out, dim=-1)


class LiteINR(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, num_levels: int = 2,
                 base_resolution: int = 16, growth: int = 4, log2_table_size: int = 12,
                 feat_dim: int = 2):
        super().__init__()
        self.hashgrid = HashGrid2D(num_levels, base_resolution, growth, log2_table_size, feat_dim)
        self.fc1_feat = nn.Conv2d(dim, hidden, kernel_size=3, padding=1, bias=True)
        self.fc1_pos = nn.Conv2d(self.hashgrid.out_dim, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden, hidden, kernel_size=1, bias=True)
        self.fc3 = nn.Conv2d(hidden, 3, kernel_size=1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        coords = make_coord((h, w), device=feat.device)
        pos = self.hashgrid(coords).t().reshape(1, -1, h, w).to(feat.dtype)
        x = F.relu(self.fc1_feat(feat) + self.fc1_pos(pos))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
