"""
U-Net Shared-Split Transformer backbone (Section 3.2).

This restores the per-scale ENCODER-DECODER U-Net structure that Section 3.2
attributes to the backbone ("NeRD-Rain ... maintains independent encoder-decoder
U-Nets per scale; LMS-TRD shares kernel weights across all three pyramid
levels"). A flat per-scale stack (no internal downsampling) cannot reconcile the
reported 7.87 M parameters with the reported 64.22 G FLOPs, because parameters
running at full resolution are far too expensive in FLOPs. By contrast, an
encoder-decoder places the wide (parameter-heavy) blocks at DOWNSAMPLED
resolution, where they are cheap in FLOPs — which is exactly what makes
7.87 M params fit inside ~64 G FLOPs. See ../RECONCILIATION.md for the analysis.

Cross-scale weight sharing — the paper's headline contribution — is PRESERVED:
there is one physical U-Net whose weights are reused at every pyramid scale, with
only light per-scale 1x1 in/out projections and a scale-alignment LayerNorm.

Channel schedule notation
-------------------------
`native_dims = (32, 64, 128)` are the per-pyramid-scale stem widths from §3.2.
The shared U-Net runs at a canonical schedule `base * mult`, e.g.
base=40, mult=(1,2,4,8) -> widths (40, 80, 160, 320) over four resolution levels.
Per-scale in/out projections map native_dim <-> base.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import SSTBlock, LayerNorm2d


class Down(nn.Module):
    """Strided 3x3 conv downsampler (halves spatial, changes width)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.op = nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Up(nn.Module):
    """Bilinear upsample to a target size + 1x1 width projection."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.op = nn.Conv2d(cin, cout, kernel_size=1)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.op(x)


class UNetSST(nn.Module):
    """Shared encoder-decoder SST backbone applied at every pyramid scale.

    Args:
        native_dims    : per-scale stem widths (paper §3.2: 32, 64, 128).
        base           : canonical base width of the shared U-Net.
        mult           : per-level width multipliers (low res -> wide).
        blocks         : SST blocks per encoder/decoder level. int or per-level list.
        bott_blocks    : SST blocks at the bottleneck (lowest resolution).
        share_weights  : if True (default) ONE U-Net is reused at all scales (the
                         SST mechanism); if False, an independent U-Net per scale
                         (diagnostic only — NOT the paper's design).
    """

    def __init__(self, native_dims=(32, 64, 128), base: int = 40,
                 mult=(1, 2, 4, 8), blocks=(1, 2, 2, 3), bott_blocks: int = 2,
                 num_heads: int = 4, expansion: int = 4, low_rank_ratio: int = 2,
                 share_weights: bool = True):
        super().__init__()
        self.native_dims = tuple(native_dims)
        self.widths = [base * m for m in mult]
        self.share_weights = share_weights
        nlev = len(self.widths)
        blk = list(blocks) if isinstance(blocks, (list, tuple)) else [blocks] * nlev
        assert len(blk) == nlev, "blocks list must match number of levels"

        def stack(w, n):
            return nn.ModuleList([
                SSTBlock(w, num_heads=num_heads, expansion=expansion,
                         low_rank_ratio=low_rank_ratio)
                for _ in range(n)
            ])

        def make_unet():
            return nn.ModuleDict(dict(
                enc=nn.ModuleList([stack(self.widths[i], blk[i]) for i in range(nlev)]),
                downs=nn.ModuleList([Down(self.widths[i], self.widths[i + 1])
                                     for i in range(nlev - 1)]),
                bott=stack(self.widths[-1], bott_blocks),
                ups=nn.ModuleList([Up(self.widths[i + 1], self.widths[i])
                                   for i in reversed(range(nlev - 1))]),
                dec=nn.ModuleList([stack(self.widths[i], blk[i])
                                   for i in reversed(range(nlev - 1))]),
                fuse=nn.ModuleList([nn.Conv2d(2 * self.widths[i], self.widths[i], 1)
                                    for i in reversed(range(nlev - 1))]),
            ))

        if share_weights:
            self.unet = make_unet()                                   # THE shared U-Net
        else:
            self.unets = nn.ModuleList([make_unet() for _ in native_dims])

        # per-scale (NOT shared) light projections + scale-alignment norms
        self.in_proj = nn.ModuleList([nn.Conv2d(d, base, 1, bias=False) for d in native_dims])
        self.align = nn.ModuleList([LayerNorm2d(base) for _ in native_dims])
        self.out_proj = nn.ModuleList([nn.Conv2d(base, d, 1, bias=False) for d in native_dims])

    def _unet_for(self, scale_idx: int):
        return self.unet if self.share_weights else self.unets[scale_idx]

    def forward(self, feat: torch.Tensor, scale_idx: int) -> torch.Tensor:
        u = self._unet_for(scale_idx)
        x = self.align[scale_idx](self.in_proj[scale_idx](feat))      # native -> base, aligned

        skips = []
        for i, level_blocks in enumerate(u['enc']):
            for b in level_blocks:
                x = b(x)
            skips.append(x)
            if i < len(u['downs']):
                x = u['downs'][i](x)

        for b in u['bott']:                                           # bottleneck (lowest res)
            x = b(x)

        for j, (up, dec_blocks, fuse) in enumerate(zip(u['ups'], u['dec'], u['fuse'])):
            skip = skips[-(j + 2)]
            x = up(x, skip.shape[-2:])
            x = fuse(torch.cat([x, skip], dim=1))                     # skip connection
            for b in dec_blocks:
                x = b(x)

        return self.out_proj[scale_idx](x)                           # base -> native
