"""
LMS-TRD full model (Section 3.1 pipeline).

  rainy image I
      |-- 3-level pyramid {S1 (1/4), S2 (1/2), S3 (full)}
      |-- per-scale shallow 3x3 conv -> native widths (32, 64, 128)
      |-- shared SST backbone applied at every scale (cross-scale weight sharing)
      |     backbone = 'unet' (encoder-decoder; matches reported complexity)  [default]
      |              or 'flat' (literal §3.2 prose; ~0.25 M params)
      |-- Lite-INR branch on the full-res feature (rain-free RGB proxy + injection)
      |-- bidirectional eBFU fusion across adjacent scales
      |-- Rain Refinement Block (residual artifact suppression)
      |-- reconstruction conv -> residual image; shallow skip: out = I + residual

forward() returns (derained_image, inr_image). inr_image feeds the auxiliary L1
INR loss during training and is ignored at inference (None when use_inr=False).

ABLATION TOGGLES (Section 4.5, Table 5): share_weights, use_inr, use_ebfu, use_rrb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_sst import UNetSST
from .sst_flat import FlatSharedSST
from .lite_inr import LiteINR
from .ebfu import eBFU
from .rrb import RainRefinementBlock


class Align(nn.Module):
    """1x1 channel projection + resize to a target spatial size."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-2:] != tuple(size):
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class LMSTRD(nn.Module):
    def __init__(self, native_dims=(32, 64, 128), backbone: str = 'unet',
                 # --- U-Net backbone params (backbone='unet') ---
                 base: int = 40, mult=(1, 2, 4, 8), blocks=(1, 2, 2, 3),
                 bott_blocks: int = 2,
                 # --- flat backbone params (backbone='flat') ---
                 shared_dim: int = 64, depth: int = 2,
                 # --- shared block params ---
                 num_heads: int = 4, expansion: int = 4, low_rank_ratio: int = 2,
                 inr_table_log2: int = 14,
                 # --- co-design toggles ---
                 share_weights: bool = True, use_inr: bool = True,
                 use_ebfu: bool = True, use_rrb: bool = True):
        super().__init__()
        self.native_dims = tuple(native_dims)
        self.backbone_kind = backbone
        self.use_inr = use_inr
        self.use_ebfu = use_ebfu
        self.use_rrb = use_rrb
        c1, c2, c3 = native_dims

        # per-scale shallow feature extraction (3x3 conv)
        self.shallow = nn.ModuleList([
            nn.Conv2d(3, c1, 3, padding=1),
            nn.Conv2d(3, c2, 3, padding=1),
            nn.Conv2d(3, c3, 3, padding=1),
        ])

        # shared transformer backbone (cross-scale weight sharing preserved)
        if backbone == 'unet':
            self.backbone = UNetSST(
                native_dims=native_dims, base=base, mult=mult, blocks=blocks,
                bott_blocks=bott_blocks, num_heads=num_heads, expansion=expansion,
                low_rank_ratio=low_rank_ratio, share_weights=share_weights)
        elif backbone == 'flat':
            self.backbone = FlatSharedSST(
                native_dims=native_dims, shared_dim=shared_dim, depth=depth,
                num_heads=num_heads, expansion=expansion,
                low_rank_ratio=low_rank_ratio, share_weights=share_weights)
        else:
            raise ValueError(f"backbone must be 'unet' or 'flat', got {backbone!r}")

        # Lite-INR branch (operates on full-res feature, width c3)
        if use_inr:
            self.lite_inr = LiteINR(feat_channels=c3, inject_channels=c3,
                                    log2_table_size=inr_table_log2)

        # cross-scale fusion (gated additive eBFU, or plain add if disabled)
        self.align_12 = Align(c1, c2)
        self.align_23 = Align(c2, c3)
        if use_ebfu:
            self.ebfu_12 = eBFU(c2)
            self.ebfu_23 = eBFU(c3)

        # refinement + reconstruction
        if use_rrb:
            self.rrb = RainRefinementBlock(c3)
        self.recon = nn.Conv2d(c3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        # ---- 3-level pyramid of the INPUT image ----
        s3 = x
        s2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        s1 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

        # ---- per-scale shallow conv + SHARED backbone ----
        f1 = self.backbone(self.shallow[0](s1), 0)
        f2 = self.backbone(self.shallow[1](s2), 1)
        f3 = self.backbone(self.shallow[2](s3), 2)

        # ---- Lite-INR branch + feature injection ----
        inr_img = None
        if self.use_inr:
            inr_img, inject_feat = self.lite_inr(f3)
            f3 = f3 + inject_feat

        # ---- bidirectional eBFU fusion (S1<->S2, then S2<->S3) ----
        f1_up = self.align_12(f1, f2.shape[-2:])
        if self.use_ebfu:
            _, f2 = self.ebfu_12(f1_up, f2)
        else:
            f2 = f2 + f1_up

        f2_up = self.align_23(f2, f3.shape[-2:])
        if self.use_ebfu:
            _, f3 = self.ebfu_23(f2_up, f3)
        else:
            f3 = f3 + f2_up

        # ---- refinement + reconstruction + shallow skip ----
        fused = self.rrb(f3) if self.use_rrb else f3
        residual = self.recon(fused)
        out = x + residual
        return out, inr_img


def build_model(cfg: dict | None = None) -> LMSTRD:
    """Factory from the 'model' sub-dict of a config (see configs/lms_trd.yaml)."""
    cfg = cfg or {}
    return LMSTRD(
        native_dims=tuple(cfg.get('native_dims', (32, 64, 128))),
        backbone=cfg.get('backbone', 'unet'),
        base=cfg.get('base', 40),
        mult=tuple(cfg.get('mult', (1, 2, 4, 8))),
        blocks=tuple(cfg.get('blocks', (1, 2, 2, 3))),
        bott_blocks=cfg.get('bott_blocks', 2),
        shared_dim=cfg.get('shared_dim', 64),
        depth=cfg.get('depth', 2),
        num_heads=cfg.get('num_heads', 4),
        expansion=cfg.get('expansion', 4),
        low_rank_ratio=cfg.get('low_rank_ratio', 2),
        inr_table_log2=cfg.get('inr_table_log2', 14),
        share_weights=cfg.get('share_weights', True),
        use_inr=cfg.get('use_inr', True),
        use_ebfu=cfg.get('use_ebfu', True),
        use_rrb=cfg.get('use_rrb', True),
    )
