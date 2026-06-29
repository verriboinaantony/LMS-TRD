#!/usr/bin/env python3
"""Evaluate LMS-TRD on a paired test set and report MEASURED PSNR/SSIM.

This script computes metrics from the model's actual outputs. It does not
contain or compare against any hard-coded value from the paper.

Example:
    python test.py --config configs/lms_trd.yaml \
        --checkpoint checkpoints/lms_trd_best.pth \
        --test-root data/test/Rain100H
"""

import argparse
import torch
from torch.utils.data import DataLoader

from models import build_model
from datasets import DerainDataset
from utils import psnr, ssim, set_seed, complexity_report
from utils.config import load_config


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate LMS-TRD")
    ap.add_argument("--config", default="configs/lms_trd.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--rainy-dir", default="rainy")
    ap.add_argument("--clean-dir", default="clean")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-complexity", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("train", {}).get("seed", 42))
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    model = build_model(cfg.get("model", {})).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    if args.report_complexity:
        rep = complexity_report(model, input_size=(1, 3, 256, 256), device=device)
        print(f"[complexity] params(total)={rep['params_total_M']:.3f} M  "
              f"params(unique)={rep['params_unique_M']:.3f} M  "
              f"MACs={rep['macs_G']:.2f} GMac  FLOPs(2xMAC)={rep['flops_G']:.2f} G  @256x256")

    test_set = DerainDataset(args.test_root, args.rainy_dir, args.clean_dir, train=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    tot_p, tot_s, n = 0.0, 0.0, 0
    for batch in loader:
        rainy = batch["rainy"].to(device)
        clean = batch["clean"].to(device)
        pred, _ = model(rainy)
        pred = pred.clamp(0, 1)
        tot_p += psnr(pred, clean).item()
        tot_s += ssim(pred, clean).item()
        n += 1
    print(f"[{args.test_root}] images={n}  PSNR={tot_p/max(n,1):.3f} dB  "
          f"SSIM={tot_s/max(n,1):.4f}")


if __name__ == "__main__":
    main()
