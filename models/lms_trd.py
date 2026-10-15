"""
LMS-TRD: Lightweight Multi-Scale Transformer with Rain-Refinement (Sec. 3),
implemented as a sequence of modifications of NeRD-Rain (Chen, Pan & Dong,
CVPR 2024). Every row of the ablation study (Sec. 4.5, Table 4) is a preset of
the same class, so each row can be built, profiled and trained:

    M0  NeRD-Rain                         dim 48, blocks [2,3,3], MDTA, GDFN,
                                          LIIF-INR, BFPU, independent U-Nets
    M1  + channel reduction 48 -> 32      (identical to the official NeRD-Rain-S)
    M2  + depth pruning [2,3,3] -> [2,2,2]
    M3  + DS-FFN            (replaces GDFN)
    M4  + DW-MSA            (linear spatial attention, replaces MDTA)
    M5  + Lite-INR          (hash-grid INR, replaces LIIF-INR)
    M6  + eBFU              (gated additive fusion, replaces BFPU)
    M7  + cross-scale weight sharing (Shared-Split Transformer, SST)
    M8  + Rain Refinement Block  = LMS-TRD

Overall pipeline (identical data flow to NeRD-Rain)
---------------------------------------------------
The input I is decomposed into a three-level pyramid by nearest-neighbour
resizing: S (1/4), M (1/2) and X (full resolution). Each level is processed by
cascaded three-level Transformer U-Nets ("stages", widths C, 2C, 4C):
level S runs stage 1; level M runs stages 1-2; level X runs stages 1-3.

  * Coarse-to-fine injection: the latent of the first stage at S (resp. M) is
    upsampled x4 to that level's resolution and decoded by the INR branch into
    an RGB residual; residual + level input = rain-free proxy, which is
    upsampled x2 and added to the input of the next finer level.
  * Fine-to-coarse feedback: latents of successive stages at X (stage 1-2 and
    stage 2-3) and at M (stage 1-2) are fused (BFPU / eBFU), downsampled x2,
    and added to the latent of the next coarser level before decoding.
  * Each level ends with a 3x3 output convolution plus a global skip to its own
    input; the decoder feature of a coarser level is upsampled and added to the
    final decoder feature of the next finer level.
  * Outputs: restored images at X, M, S and the INR proxies at M and S
    (all used by the training objective, Sec. 3.6).

Cross-scale weight sharing (Sec. 3.2, rows M7-M8)
--------------------------------------------------
NeRD-Rain keeps six independent U-Nets (S1, M1, M2, X1, X2, X3). In the
Shared-Split Transformer (SST) the U-Net of cascade stage k is SHARED by every
pyramid level that runs stage k: stage 1 by S, M and X; stage 2 by M and X;
stage 3 by X. This is possible because all levels use the same widths
(C, 2C, 4C) = (32, 64, 128). What stays SPLIT (level-specific) are the patch
embeddings, the context and output convolutions, the INR branch, the fusions,
and a scale-alignment LayerNorm placed before every shared stage (one per
level and stage), which re-normalises each level's features before the shared
weights are applied.

Implementation note: in the official NeRD-Rain code the last full-resolution
decoder re-uses `reduce_chan_level1_max2` and leaves `reduce_chan_level1_max3`
unused. Here every stage owns its 1x1 reduction (same parameter count); the
verification script copies the official weights accordingly and checks that
the outputs are identical.
"""

from dataclasses import dataclass, asdict, fields
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .common import LayerNorm2d, OverlapPatchEmbed, Downsample, Upsample, resize
from .attention import MDTA, DWMSA
from .ffn import GDFN, DSFFN
from .inr import LIIFINR, LiteINR
from .fusion import BFPU, EBFU
from .rrb import RRB


@dataclass
class LMSTRDConfig:
    dim: int = 32
    num_blocks: Tuple[int, int, int] = (2, 2, 2)
    heads: Tuple[int, int, int] = (1, 2, 4)
    attn: str = 'dwmsa'            # 'mdta' (NeRD-Rain) | 'dwmsa' (LMS-TRD)
    ffn: str = 'dsffn'             # 'gdfn' (NeRD-Rain) | 'dsffn' (LMS-TRD)
    inr: str = 'lite'              # 'liif' (NeRD-Rain) | 'lite'  (LMS-TRD)
    fusion: str = 'ebfu'           # 'bfpu' (NeRD-Rain) | 'ebfu'  (LMS-TRD)
    share_weights: bool = True     # cross-scale weight sharing (SST)
    use_rrb: bool = True           # Rain Refinement Block
    ffn_expansion_factor: float = 2.66   # GDFN
    dsffn_expansion: int = 4             # DS-FFN expansion ratio r
    low_rank_ratio: int = 2              # DW-MSA: d_k = d_v / low_rank_ratio
    inr_hidden: int = 64
    hash_levels: int = 2
    hash_base_resolution: int = 16
    hash_growth: int = 4
    hash_log2_table_size: int = 12
    hash_feat_dim: int = 2

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "LMSTRDConfig":
        d = dict(d or {})
        preset = d.pop('preset', None)
        base = dict(PRESETS[preset]) if preset else {}
        base.update(d)
        names = {f.name for f in fields(cls)}
        unknown = set(base) - names
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        for key in ('num_blocks', 'heads'):
            if key in base:
                base[key] = tuple(base[key])
        return cls(**base)


_M0 = dict(dim=48, num_blocks=(2, 3, 3), attn='mdta', ffn='gdfn', inr='liif',
           fusion='bfpu', share_weights=False, use_rrb=False)
PRESETS = {'M0': _M0}
PRESETS['M1'] = {**PRESETS['M0'], 'dim': 32}
PRESETS['M2'] = {**PRESETS['M1'], 'num_blocks': (2, 2, 2)}
PRESETS['M3'] = {**PRESETS['M2'], 'ffn': 'dsffn'}
PRESETS['M4'] = {**PRESETS['M3'], 'attn': 'dwmsa'}
PRESETS['M5'] = {**PRESETS['M4'], 'inr': 'lite'}
PRESETS['M6'] = {**PRESETS['M5'], 'fusion': 'ebfu'}
PRESETS['M7'] = {**PRESETS['M6'], 'share_weights': True}
PRESETS['M8'] = {**PRESETS['M7'], 'use_rrb': True}
PRESET_NAMES = {
    'M0': 'NeRD-Rain (baseline)',
    'M1': '+ channel reduction 48->32',
    'M2': '+ depth pruning [2,2,2]',
    'M3': '+ DS-FFN',
    'M4': '+ DW-MSA linear attention',
    'M5': '+ Lite-INR (hash grid)',
    'M6': '+ eBFU (gated additive)',
    'M7': '+ cross-scale sharing (SST)',
    'M8': '+ RRB (LMS-TRD)',
}


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, dim: int, num_heads: int, cfg: LMSTRDConfig):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = (MDTA(dim, num_heads) if cfg.attn == 'mdta'
                     else DWMSA(dim, num_heads, low_rank_ratio=cfg.low_rank_ratio))
        self.norm2 = LayerNorm2d(dim)
        self.ffn = (GDFN(dim, cfg.ffn_expansion_factor) if cfg.ffn == 'gdfn'
                    else DSFFN(dim, cfg.dsffn_expansion))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class BlockSequence(nn.Sequential):
    """Sequence of Transformer blocks with optional activation checkpointing.

    Checkpointing (training only) recomputes each block's activations in the
    backward pass instead of storing them. It changes memory and time, not the
    computed values or gradients.
    """

    grad_checkpointing = False

    def forward(self, x):
        for blk in self:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


class UNetStage(nn.Module):
    """Three-level encoder-decoder Transformer U-Net with widths (C, 2C, 4C)."""

    def __init__(self, cfg: LMSTRDConfig):
        super().__init__()
        c, nb, hd = cfg.dim, cfg.num_blocks, cfg.heads

        def blocks(d, n, h):
            return BlockSequence(*[TransformerBlock(d, h, cfg) for _ in range(n)])

        self.enc1 = blocks(c, nb[0], hd[0])
        self.down12 = Downsample(c)
        self.enc2 = blocks(2 * c, nb[1], hd[1])
        self.down23 = Downsample(2 * c)
        self.latent = blocks(4 * c, nb[2], hd[2])
        self.up32 = Upsample(4 * c)
        self.reduce2 = nn.Conv2d(4 * c, 2 * c, kernel_size=1, bias=False)
        self.dec2 = blocks(2 * c, nb[1], hd[1])
        self.up21 = Upsample(2 * c)
        self.reduce1 = nn.Conv2d(2 * c, c, kernel_size=1, bias=False)
        self.dec1 = blocks(c, nb[0], hd[0])

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down12(e1))
        return e1, e2, self.latent(self.down23(e2))

    def decode(self, lat, e1, e2):
        d2 = self.dec2(self.reduce2(torch.cat([self.up32(lat), e2], dim=1)))
        return self.dec1(self.reduce1(torch.cat([self.up21(d2), e1], dim=1)))


class LMSTRD(nn.Module):
    LEVEL_STAGES = (('S', 1), ('M', 1), ('M', 2), ('X', 1), ('X', 2), ('X', 3))

    def __init__(self, cfg: Optional[LMSTRDConfig] = None):
        super().__init__()
        cfg = cfg or LMSTRDConfig()
        self.cfg = cfg
        c = cfg.dim

        # ---- level-specific ("split") modules ----
        self.embed = nn.ModuleDict({lv: OverlapPatchEmbed(3, c) for lv in 'SMX'})
        self.ctx = nn.ModuleDict({k: nn.Conv2d(c, c, 3, padding=1, bias=False)
                                  for k in ('M1', 'X1', 'X2')})
        self.head = nn.ModuleDict({lv: nn.Conv2d(c, 3, 3, padding=1, bias=False)
                                   for lv in 'SMX'})
        self.inr_up = nn.ModuleDict({lv: nn.Sequential(Upsample(4 * c), Upsample(2 * c))
                                     for lv in 'SM'})
        if cfg.inr == 'liif':
            self.inr = nn.ModuleDict({lv: LIIFINR(c) for lv in 'SM'})
        else:
            self.inr = nn.ModuleDict({lv: LiteINR(
                c, hidden=cfg.inr_hidden, num_levels=cfg.hash_levels,
                base_resolution=cfg.hash_base_resolution, growth=cfg.hash_growth,
                log2_table_size=cfg.hash_log2_table_size, feat_dim=cfg.hash_feat_dim)
                for lv in 'SM'})
        fuse_cls = BFPU if cfg.fusion == 'bfpu' else EBFU
        self.fuse = nn.ModuleList([fuse_cls(4 * c) for _ in range(3)])

        # ---- U-Net stages: independent (NeRD-Rain) or shared across levels (SST) ----
        if cfg.share_weights:
            self.stages = nn.ModuleDict({str(k): UNetStage(cfg) for k in (1, 2, 3)})
            self.align = nn.ModuleDict({f'{lv}{k}': LayerNorm2d(c)
                                        for lv, k in self.LEVEL_STAGES if k < 3})
        else:
            self.stages = nn.ModuleDict({f'{lv}{k}': UNetStage(cfg)
                                         for lv, k in self.LEVEL_STAGES})
            self.align = None

        self.rrb = RRB(c) if cfg.use_rrb else None

    # ------------------------------------------------------------------ #
    def set_grad_checkpointing(self, enabled: bool = True) -> "LMSTRD":
        for m in self.modules():
            if isinstance(m, BlockSequence):
                m.grad_checkpointing = enabled
        return self

    def stage(self, level: str, k: int) -> UNetStage:
        return self.stages[str(k)] if self.cfg.share_weights else self.stages[f'{level}{k}']

    def aligned(self, level: str, k: int, x: torch.Tensor) -> torch.Tensor:
        key = f'{level}{k}'
        if self.align is not None and key in self.align:
            return self.align[key](x)
        return x

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor):
        """x: (B, 3, H, W) with H, W divisible by 16.

        Returns (out_X, out_M, out_S, proxy_M, proxy_S); out_X is the derained image.
        """
        x_m, x_s = resize(x, 0.5), resize(x, 0.25)

        # level S (1/4): stage-1 encoder, INR proxy injected into level M
        e1s, e2s, lat_s = self.stage('S', 1).encode(self.aligned('S', 1, self.embed['S'](x_s)))
        proxy_s = self.inr['S'](self.inr_up['S'](lat_s)) + x_s
        in_m = x_m + resize(proxy_s, 2)

        # level M (1/2): stage-1 encoder, INR proxy injected into level X
        e1m, e2m, lat_m1 = self.stage('M', 1).encode(self.aligned('M', 1, self.embed['M'](in_m)))
        proxy_m = self.inr['M'](self.inr_up['M'](lat_m1)) + in_m
        in_x = x + resize(proxy_m, 2)

        # level X (full): stages 1, 2 and the encoder of stage 3
        e1x, e2x, lat_x1 = self.stage('X', 1).encode(self.aligned('X', 1, self.embed['X'](in_x)))
        d_x = self.stage('X', 1).decode(lat_x1, e1x, e2x)
        e1x, e2x, lat_x2 = self.stage('X', 2).encode(self.aligned('X', 2, self.ctx['X1'](d_x)))
        d_x = self.stage('X', 2).decode(lat_x2, e1x, e2x)
        e1x3, e2x3, lat_x3 = self.stage('X', 3).encode(self.ctx['X2'](d_x))

        # fine-to-coarse feedback from level X to level M
        fb1 = resize(self.fuse[0](lat_x1, lat_x2), 0.5)
        fb2 = resize(self.fuse[1](lat_x2, lat_x3), 0.5)

        # level M: stage-1 decoder (with feedback), stage-2 encoder
        d_m = self.stage('M', 1).decode(lat_m1 + fb1, e1m, e2m)
        e1m2, e2m2, lat_m2 = self.stage('M', 2).encode(self.aligned('M', 2, self.ctx['M1'](d_m)))

        # fine-to-coarse feedback from level M to level S
        fb3 = resize(self.fuse[2](lat_m1, lat_m2), 0.5)

        # decoders, coarse to fine
        d_s = self.stage('S', 1).decode(lat_s + fb3, e1s, e2s)
        out_s = self.head['S'](d_s) + x_s

        d_m2 = self.stage('M', 2).decode(lat_m2 + fb2, e1m2, e2m2) + resize(d_s, 2)
        out_m = self.head['M'](d_m2) + x_m

        d_x3 = self.stage('X', 3).decode(lat_x3, e1x3, e2x3) + resize(d_m2, 2)
        if self.rrb is not None:
            d_x3 = self.rrb(d_x3)
        out_x = self.head['X'](d_x3) + x

        return out_x, out_m, out_s, proxy_m, proxy_s


def build_model(cfg: Optional[dict] = None) -> LMSTRD:
    """Build from the `model` section of a YAML config.

    Accepts `preset: M0..M8` plus any LMSTRDConfig field as override.
    An empty config builds the definitive LMS-TRD (= preset M8).
    """
    return LMSTRD(LMSTRDConfig.from_dict(cfg))


def config_dict(model: LMSTRD) -> dict:
    return asdict(model.cfg)
