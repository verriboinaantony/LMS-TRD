#!/usr/bin/env python3
"""Evaluate a trained model on the five Rain13K test sets (Sec. 4.1, Table 2).

    python test.py --checkpoint runs/lms_trd_s42/last.pth \
        --test-root Datasets/test --datasets Test100 Rain100H Rain100L Test2800 Test1200 \
        --save-dir results/images/lms_trd_s42 --json results/eval_lms_trd_s42.json

Protocol: full-resolution images, reflect-padded to a multiple of 16 and cropped
back, output clipped and quantised to 8 bits, PSNR/SSIM on the Y channel
(utils/metrics.py). No test-time augmentation.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from datasets import DerainDataset
from models import load_model
from utils import psnr, ssim, pad_to_multiple, describe_environment

DEFAULT_SETS = ['Test100', 'Rain100H', 'Rain100L', 'Test2800', 'Test1200']


@torch.no_grad()
def evaluate(model, root, device, save_dir=None):
    ds = DerainDataset(root, train=False)
    per_image = []
    for i in range(len(ds)):
        item = ds[i]
        x, (h, w) = pad_to_multiple(item['rainy'][None].to(device), 16)
        out = model(x)[0][..., :h, :w].clamp(0, 1)
        gt = item['clean'][None].to(device)
        p, s = psnr(out, gt).item(), ssim(out, gt).item()
        per_image.append({'name': item['name'], 'psnr_y': p, 'ssim_y': s})
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(save_dir, os.path.splitext(item['name'])[0] + '.png'))
    return {'images': len(per_image),
            'psnr_y': float(np.mean([r['psnr_y'] for r in per_image])),
            'ssim_y': float(np.mean([r['ssim_y'] for r in per_image])),
            'per_image': per_image}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--test-root', required=True, help='folder containing one sub-folder per test set')
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_SETS)
    ap.add_argument('--save-dir', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    model = load_model(args.checkpoint, device)

    report = {'checkpoint': args.checkpoint, 'environment': describe_environment(device), 'sets': {}}
    for name in args.datasets:
        sd = os.path.join(args.save_dir, name) if args.save_dir else None
        res = evaluate(model, os.path.join(args.test_root, name), device, sd)
        report['sets'][name] = res
        print(f'{name:10s} images={res["images"]:5d}  PSNR(Y)={res["psnr_y"]:.2f} dB  SSIM(Y)={res["ssim_y"]:.4f}',
              flush=True)
    sets = report['sets'].values()
    report['average'] = {'psnr_y': float(np.mean([r['psnr_y'] for r in sets])),
                         'ssim_y': float(np.mean([r['ssim_y'] for r in sets]))}
    print(f'{"Average":10s}                PSNR(Y)={report["average"]["psnr_y"]:.2f} dB  '
          f'SSIM(Y)={report["average"]["ssim_y"]:.4f}')
    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
