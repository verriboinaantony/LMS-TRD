"""
Training loop (Sec. 4.2).

  optimiser : Adam (beta1 = 0.9, beta2 = 0.999)
  lr        : 2e-4, halved every 50 epochs (StepLR, gamma = 0.5)
  epochs    : 200; batch 16; random 256x256 crops, flips, 90-degree rotations
  memory    : activation checkpointing of the Transformer blocks (default on)
              so that batch 16 fits on a 24 GB GPU; it does not change results
  precision : float32 by default; --amp enables bfloat16 autocast

Every epoch appends a line to <run_dir>/log.csv (losses, lr, time, optional
validation PSNR/SSIM) and the run configuration and environment are stored in
<run_dir>/run.json, so that every reported number can be traced to a run.
"""

import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from utils import psnr, ssim, seed_worker, pad_to_multiple, describe_environment
from .losses import LMSTRDLoss


class Trainer:
    def __init__(self, model, train_set, val_set=None, cfg=None, device='cuda', run_dir='runs/lms_trd'):
        cfg = cfg or {}
        self.cfg = cfg
        self.device = device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu'
        self.model = model.to(self.device)
        self.model.set_grad_checkpointing(cfg.get('grad_checkpointing', True))
        self.epochs = cfg.get('epochs', 200)
        self.amp = cfg.get('amp', False) and self.device.startswith('cuda')
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self.accum = max(1, int(cfg.get('accum_steps', 1)))
        batch = cfg.get('batch_size', 16)
        assert batch % self.accum == 0, 'batch_size must be divisible by accum_steps'
        g = torch.Generator()
        g.manual_seed(cfg.get('seed', 42))
        self.train_loader = DataLoader(
            train_set, batch_size=batch // self.accum, shuffle=True,
            num_workers=cfg.get('num_workers', 8), drop_last=True,
            worker_init_fn=seed_worker, generator=g, pin_memory=self.device.startswith('cuda'),
            persistent_workers=cfg.get('num_workers', 8) > 0)
        self.val_loader = None
        if val_set is not None:
            self.val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

        self.criterion = LMSTRDLoss(cfg.get('lambda_edge', 0.05), cfg.get('lambda_freq', 0.01),
                                    cfg.get('lambda_inr', 0.1), cfg.get('char_eps', 1e-3)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.get('lr', 2e-4),
                                          betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg.get('lr_step', 50), gamma=cfg.get('lr_gamma', 0.5))
        self.start_epoch, self.best_psnr = 0, -1.0

        with open(os.path.join(run_dir, 'run.json'), 'w') as f:
            json.dump({'train_cfg': cfg, 'model_cfg': model.cfg.__dict__,
                       'n_train': len(train_set), 'n_val': len(val_set) if val_set else 0,
                       'environment': describe_environment(self.device)}, f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    def train(self):
        log_path = os.path.join(self.run_dir, 'log.csv')
        new = not os.path.exists(log_path) or self.start_epoch == 0
        with open(log_path, 'w' if new else 'a', newline='') as fh:
            writer = csv.writer(fh)
            if new:
                writer.writerow(['epoch', 'lr', 'loss', 'char', 'edge', 'freq', 'inr',
                                 'epoch_time_s', 'val_psnr_y', 'val_ssim_y'])
            for epoch in range(self.start_epoch, self.epochs):
                lr = self.optimizer.param_groups[0]['lr']
                t0 = time.time()
                logs = self._train_one_epoch(epoch)
                self.scheduler.step()
                vp, vs = (self.validate() if self.val_loader is not None else (float('nan'), float('nan')))
                dt = time.time() - t0
                writer.writerow([epoch + 1, lr, *(round(logs[k], 6) for k in
                                 ('loss', 'char', 'edge', 'freq', 'inr')), round(dt, 1),
                                 round(vp, 4), round(vs, 5)])
                fh.flush()
                print(f'[epoch {epoch + 1:3d}/{self.epochs}] loss={logs["loss"]:.4f} lr={lr:.2e} '
                      f'time={dt:.0f}s val_psnr_y={vp:.3f} val_ssim_y={vs:.4f}', flush=True)
                if self.val_loader is not None and vp > self.best_psnr:
                    self.best_psnr = vp
                    self.save(epoch, 'best')
                self.save(epoch, 'last')

    def _train_one_epoch(self, epoch):
        self.model.train()
        sums, n = {}, 0
        self.optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(self.train_loader):
            rainy = batch['rainy'].to(self.device, non_blocking=True)
            clean = batch['clean'].to(self.device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp):
                outputs = self.model(rainy)
            outputs = tuple(o.float() for o in outputs)
            loss, logs = self.criterion(outputs, clean)
            (loss / self.accum).backward()        # the model has no batch-dependent layers, so
            if (it + 1) % self.accum == 0:        # accumulation equals one large batch
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            if (it + 1) % self.cfg.get('log_interval', 100) == 0:
                print(f'  epoch {epoch + 1} it {it + 1}/{len(self.train_loader)} loss={logs["loss"]:.4f}',
                      flush=True)
        return {k: v / max(n, 1) for k, v in sums.items()}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        ps, ss, n = 0.0, 0.0, 0
        for batch in self.val_loader:
            x, (h, w) = pad_to_multiple(batch['rainy'].to(self.device), 16)
            out = self.model(x)[0][..., :h, :w]
            gt = batch['clean'].to(self.device)
            ps += psnr(out, gt).item()
            ss += ssim(out, gt).item()
            n += 1
        return ps / max(n, 1), ss / max(n, 1)

    # ------------------------------------------------------------------ #
    def save(self, epoch, tag):
        torch.save({'epoch': epoch, 'model': self.model.state_dict(),
                    'model_cfg': self.model.cfg.__dict__,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(), 'best_psnr': self.best_psnr},
                   os.path.join(self.run_dir, f'{tag}.pth'))

    def load(self, path, resume=True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if resume:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.start_epoch = ckpt.get('epoch', -1) + 1
            self.best_psnr = ckpt.get('best_psnr', -1.0)
