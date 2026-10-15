#!/usr/bin/env python3
"""Derain one image or a folder of images.

    python inference.py --checkpoint runs/lms_trd_s42/last.pth --input rainy.png --output derained.png
    python inference.py --checkpoint runs/lms_trd_s42/last.pth --input rainy_dir/ --output out_dir/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets import load_rgb
from datasets.derain_dataset import IMG_EXT
from models import load_model
from utils import pad_to_multiple


@torch.no_grad()
def derain(model, path, device):
    x, (h, w) = pad_to_multiple(load_rgb(path)[None].to(device), 16)
    out = model(x)[0][..., :h, :w].clamp(0, 1)
    return (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    model = load_model(args.checkpoint, device)
    src = Path(args.input)
    if src.is_dir():
        os.makedirs(args.output, exist_ok=True)
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                Image.fromarray(derain(model, p, device)).save(os.path.join(args.output, p.stem + '.png'))
                print('saved', os.path.join(args.output, p.stem + '.png'))
    else:
        Image.fromarray(derain(model, src, device)).save(args.output)
        print('saved', args.output)


if __name__ == '__main__':
    main()
