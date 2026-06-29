#!/usr/bin/env python3
"""Train LMS-TRD.

Example:
    python train.py --config configs/lms_trd.yaml \
        --train-root data/train --val-root data/val
"""

import argparse

from models import build_model
from datasets import DerainDataset
from training import Trainer
from utils import set_seed
from utils.config import load_config, flatten_train_cfg


def parse_args():
    ap = argparse.ArgumentParser(description="Train LMS-TRD")
    ap.add_argument("--config", default="configs/lms_trd.yaml")
    ap.add_argument("--train-root", default=None)
    ap.add_argument("--val-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    train_root = args.train_root or data_cfg.get("train_root")
    val_root = args.val_root or data_cfg.get("val_root")
    rainy_dir = data_cfg.get("rainy_dir", "rainy")
    clean_dir = data_cfg.get("clean_dir", "clean")

    train_cfg = flatten_train_cfg(cfg)
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    set_seed(train_cfg.get("seed", 42))

    patch = cfg.get("train", {}).get("patch_size", 256)
    train_set = DerainDataset(train_root, rainy_dir, clean_dir,
                              patch_size=patch, train=True)
    val_set = None
    if val_root:
        val_set = DerainDataset(val_root, rainy_dir, clean_dir, train=False)

    model = build_model(cfg.get("model", {}))
    trainer = Trainer(model, train_set, val_set, cfg=train_cfg, device=args.device)
    if args.resume:
        trainer.load(args.resume, resume=True)
    trainer.train()


if __name__ == "__main__":
    main()
