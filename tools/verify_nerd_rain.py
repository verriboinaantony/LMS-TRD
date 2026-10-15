#!/usr/bin/env python3
"""Check that ablation row M0 (resp. M1) is the official NeRD-Rain (resp. NeRD-Rain-S).

Steps
  1. Import the official model from a clone of github.com/cschenxiang/NeRD-Rain
     (CUDA calls in the official code are redirected to the CPU if needed).
  2. Copy the official weights into our M0 / M1 using an explicit name map.
  3. Compare parameter counts and all five outputs on random inputs.
  4. Confirm that the official INR's four local-ensemble queries coincide.

Usage
    git clone https://github.com/cschenxiang/NeRD-Rain.git ../NeRD-Rain
    python tools/verify_nerd_rain.py --nerd-rain ../NeRD-Rain [--size 128]
"""

import argparse
import importlib
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model  # noqa: E402


def _patch_cuda():
    if torch.cuda.is_available():
        return
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self


STAGE_MAP = {
    # (official suffix pattern) -> our stage key
    'small': 'S1', 'mid1': 'M1', 'mid2': 'M2', 'max1': 'X1', 'max2': 'X2', 'max3': 'X3',
}


def official_to_ours(name: str):
    """Map one official parameter name to a list of our names (usually one)."""
    n = name
    n = n.replace('.body.weight', '.weight').replace('.body.bias', '.bias') \
        if re.search(r'norm[12]\.body\.', n) else n
    n = n.replace('.main.0.', '.')                       # BasicConv wrapper
    rules = [
        (r'^patch_embed_small\.', 'embed.S.'), (r'^patch_embed_mid\.', 'embed.M.'),
        (r'^patch_embed_max\.', 'embed.X.'),
        (r'^output_small\.', 'head.S.'), (r'^output_mid\.', 'head.M.'), (r'^output_max\.', 'head.X.'),
        (r'^output_mid_context\.', 'ctx.M1.'), (r'^output_max_context1\.', 'ctx.X1.'),
        (r'^output_max_context2\.', 'ctx.X2.'),
        (r'^INR\.', 'inr.S.'), (r'^INR2\.', 'inr.M.'),
        (r'^BF1\.', 'fuse.0.'), (r'^BF2\.', 'fuse.1.'), (r'^BF3\.', 'fuse.2.'),
        (r'^upsmall2mid1\.', 'inr_up.S.0.'), (r'^upsmall2mid2\.', 'inr_up.S.1.'),
        (r'^upmid2max1\.', 'inr_up.M.0.'), (r'^upmid2max2\.', 'inr_up.M.1.'),
    ]
    for pat, rep in rules:
        if re.match(pat, n):
            return [re.sub(pat, rep, n)]

    # U-Net parts: <part>_<level><idx>.  e.g. encoder_level1_mid1, down1_2_max3,
    # up3_2_small, reduce_chan_level2_max1, decoder_level1_mid2, latent_max2
    m = re.match(r'^(encoder_level1|encoder_level2|latent|decoder_level2|decoder_level1|'
                 r'down1_2|down2_3|up3_2|up2_1|reduce_chan_level2|reduce_chan_level1)'
                 r'_(small|mid|max)(\d?)\.(.*)$', n)
    if not m:
        raise KeyError(f'unmapped official parameter: {name}')
    part, level, idx, rest = m.groups()
    part_map = {'encoder_level1': 'enc1', 'encoder_level2': 'enc2', 'latent': 'latent',
                'decoder_level2': 'dec2', 'decoder_level1': 'dec1', 'down1_2': 'down12',
                'down2_3': 'down23', 'up3_2': 'up32', 'up2_1': 'up21',
                'reduce_chan_level2': 'reduce2', 'reduce_chan_level1': 'reduce1'}
    if level == 'small':
        stage = 'S1'
    else:
        stage = {'mid': 'M', 'max': 'X'}[level] + (idx or '1')
    ours = f'stages.{stage}.{part_map[part]}.{rest}'
    # Official quirk: the third max-scale decoder re-uses reduce_chan_level1_max2,
    # and reduce_chan_level1_max3 is never used.
    if part == 'reduce_chan_level1' and level == 'max' and idx == '2':
        return [ours, ours.replace('stages.X2.', 'stages.X3.')]
    if part == 'reduce_chan_level1' and level == 'max' and idx == '3':
        return []
    return [ours]


def check(official_cls, preset: str, size: int, seed: int = 0):
    torch.manual_seed(seed)
    off = official_cls().eval()
    ours = build_model({'preset': preset}).eval()
    n_off = sum(p.numel() for p in off.parameters())
    n_ours = sum(p.numel() for p in ours.parameters())

    sd_off = off.state_dict()
    sd_new = {}
    for k, v in sd_off.items():
        for t in official_to_ours(k):
            sd_new[t] = v.clone()
    ours.load_state_dict(sd_new, strict=True)
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        o = off(x)            # [out_max, out_mid, out_small, mid_proxy, mid/2, small_proxy, small]
        n = ours(x)           # (out_X, out_M, out_S, proxy_M, proxy_S)
    pairs = [('out_X', o[0], n[0]), ('out_M', o[1], n[1]), ('out_S', o[2], n[2]),
             ('proxy_M', o[3], n[3]), ('proxy_S', o[5], n[4])]
    print(f'[{preset}] params official={n_off:,}  ours={n_ours:,}  equal={n_off == n_ours}')
    worst = 0.0
    for name, a, b in pairs:
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f'    {name:8s} max|diff| = {d:.3e}')
    return n_off == n_ours and worst < 1e-4


def check_liif_ensemble(official_inr_cls, dim=32, size=32):
    """The official INR evaluates its MLP four times on identical inputs."""
    torch.manual_seed(0)
    inr = official_inr_cls(dim).eval()
    feat = torch.rand(1, dim, size, size)
    captured = []
    h = inr.imnet.register_forward_hook(lambda m, i, o: captured.append(i[0].detach().clone()))
    with torch.no_grad():
        inr(feat)
    h.remove()
    same = all(torch.equal(captured[0], c) for c in captured[1:])
    coord_part = captured[0][:, dim * 9: dim * 9 + 18]
    print(f'[INR] MLP calls={len(captured)}  identical inputs={same}  '
          f'max|relative-coordinate input|={coord_part.abs().max().item():.1e}')
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nerd-rain', required=True, help='path to a clone of the official NeRD-Rain repo')
    ap.add_argument('--size', type=int, default=128)
    args = ap.parse_args()
    _patch_cuda()
    sys.path.insert(0, os.path.abspath(args.nerd_rain))
    model_mod = importlib.import_module('model')
    model_s_mod = importlib.import_module('model_S')
    mlp_mod = importlib.import_module('mlp')
    ok0 = check(model_mod.MultiscaleNet, 'M0', args.size)
    ok1 = check(model_s_mod.MultiscaleNet, 'M1', args.size)
    ok2 = check_liif_ensemble(mlp_mod.INR)
    print('\nRESULT:', 'PASS' if (ok0 and ok1 and ok2) else 'FAIL')
    sys.exit(0 if (ok0 and ok1 and ok2) else 1)


if __name__ == '__main__':
    main()
