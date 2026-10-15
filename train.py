#!/usr/bin/env python3
"""Train LMS-TRD (or any ablation row M0-M8).

    python train.py --config configs/lms_trd.yaml --train-root Datasets/train/Rain13K \
        --seed 42 --run-dir runs/lms_trd_s42

Ablation rows:  --config configs/ablation/M5.yaml
Command-line values override the config.
"""

import argparse

from datasets import DerainDataset
from models import build_model
from training import Trainer
from utils import set_seed, load_config


def main():
    ap = argparse.ArgumentParser(description='Train LMS-TRD')
    ap.add_argument('--config', default='configs/lms_trd.yaml')
    ap.add_argument('--train-root', default=None)
    ap.add_argument('--val-root', default=None)
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--accum-steps', type=int, default=None,
                    help='gradient accumulation steps (GPU batch = batch_size / accum_steps)')
    ap.add_argument('--amp', action='store_true', help='bfloat16 autocast (not used for the paper)')
    ap.add_argument('--no-grad-checkpointing', action='store_true')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = {**cfg.get('train', {}), **cfg.get('loss', {})}
    for key, val in (('epochs', args.epochs), ('batch_size', args.batch_size), ('lr', args.lr),
                     ('seed', args.seed), ('num_workers', args.workers),
                     ('accum_steps', args.accum_steps)):
        if val is not None:
            tcfg[key] = val
    if args.amp:
        tcfg['amp'] = True
    if args.no_grad_checkpointing:
        tcfg['grad_checkpointing'] = False
    set_seed(tcfg.get('seed', 42))

    data = cfg.get('data', {})
    train_root = args.train_root or data.get('train_root')
    val_root = args.val_root or data.get('val_root')
    train_set = DerainDataset(train_root, train=True, patch_size=tcfg.get('patch_size', 256))
    val_set = DerainDataset(val_root, train=False) if val_root else None

    model = build_model(cfg.get('model', {}))
    run_dir = args.run_dir or f"runs/{cfg.get('name', 'lms_trd')}_s{tcfg.get('seed', 42)}"
    trainer = Trainer(model, train_set, val_set, cfg=tcfg, device=args.device, run_dir=run_dir)
    if args.resume:
        trainer.load(args.resume, resume=True)
    trainer.train()


if __name__ == '__main__':
    main()
