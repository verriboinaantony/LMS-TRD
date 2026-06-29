#!/usr/bin/env python3
"""Run LMS-TRD on a single rainy image (or a folder of images).

Example:
    python inference.py --checkpoint checkpoints/lms_trd_best.pth \
        --input rainy.png --output derained.png
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import build_model
from utils import set_seed
from utils.config import load_config

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_image(path: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str):
    arr = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="LMS-TRD single-image inference")
    ap.add_argument("--config", default="configs/lms_trd.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--output", required=True, help="output file or folder")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("train", {}).get("seed", 42))
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    model = build_model(cfg.get("model", {})).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    in_path = Path(args.input)
    if in_path.is_dir():
        os.makedirs(args.output, exist_ok=True)
        files = [p for p in sorted(in_path.iterdir()) if p.suffix.lower() in _IMG_EXT]
        for p in files:
            x = load_image(str(p)).to(device)
            out, _ = model(x)
            save_image(out, os.path.join(args.output, p.name))
            print(f"saved {os.path.join(args.output, p.name)}")
    else:
        x = load_image(str(in_path)).to(device)
        out, _ = model(x)
        save_image(out, args.output)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
