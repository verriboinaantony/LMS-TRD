"""
Flat Shared-Split Transformer backbone (literal §3.2 prose interpretation).

This is the original faithful reconstruction: a single stack of SST blocks at a
canonical width, reused at every pyramid scale, with NO internal downsampling.
It is kept for reference and for the `configs/lms_trd_prose.yaml` configuration.

NOTE: built strictly to the prose (channels 32/64/128, depth [2,2,2]) this
backbone measures well under 1 M parameters and ~25 G FLOPs — far below the
reported 7.87 M / 64.22 G. It is the right implementation of the *method as
written*, but does not reproduce the reported complexity. Use `unet_sst.UNetSST`
(the default `configs/lms_trd.yaml`) to match the reported numbers.
"""

import torch
import torch.nn as nn

from .blocks import SSTBlock, LayerNorm2d


class FlatSharedSST(nn.Module):
    def __init__(self, native_dims=(32, 64, 128), shared_dim: int = 64,
                 depth: int = 2, num_heads: int = 4, expansion: int = 4,
                 low_rank_ratio: int = 2, share_weights: bool = True):
        super().__init__()
        self.native_dims = tuple(native_dims)
        self.shared_dim = shared_dim
        self.share_weights = share_weights

        def make_stack():
            return nn.ModuleList([
                SSTBlock(shared_dim, num_heads=num_heads, expansion=expansion,
                         low_rank_ratio=low_rank_ratio)
                for _ in range(depth)
            ])

        if share_weights:
            self.blocks = make_stack()
        else:
            self.per_scale_blocks = nn.ModuleList([make_stack() for _ in native_dims])

        self.in_proj = nn.ModuleList([nn.Conv2d(d, shared_dim, 1, bias=False) for d in native_dims])
        self.align_norm = nn.ModuleList([LayerNorm2d(shared_dim) for _ in native_dims])
        self.out_proj = nn.ModuleList([nn.Conv2d(shared_dim, d, 1, bias=False) for d in native_dims])

    def _blocks_for(self, scale_idx: int):
        return self.blocks if self.share_weights else self.per_scale_blocks[scale_idx]

    def forward(self, feat: torch.Tensor, scale_idx: int) -> torch.Tensor:
        x = self.in_proj[scale_idx](feat)
        x = self.align_norm[scale_idx](x)
        for blk in self._blocks_for(scale_idx):
            x = blk(x)
        x = self.out_proj[scale_idx](x)
        return x
