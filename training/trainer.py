"""
Training loop (Section 4.2).

  optimizer : Adam (beta1=0.9, beta2=0.999)
  lr        : 2e-4, halved every 50 epochs (StepLR, gamma=0.5)
  epochs    : 200
  batch     : 16 (set in the config / CLI)
  no test-time augmentation anywhere.
"""

import os
import time
import torch
from torch.utils.data import DataLoader

from utils import psnr, ssim, seed_worker
from .losses import LMSTRDLoss


class Trainer:
    def __init__(self, model, train_set, val_set=None, cfg=None, device="cuda"):
        cfg = cfg or {}
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = model.to(self.device)

        self.epochs = cfg.get("epochs", 200)
        self.batch_size = cfg.get("batch_size", 16)
        self.lr = cfg.get("lr", 2e-4)
        self.lr_step = cfg.get("lr_step", 50)
        self.lr_gamma = cfg.get("lr_gamma", 0.5)
        self.ckpt_dir = cfg.get("ckpt_dir", "checkpoints")
        self.log_interval = cfg.get("log_interval", 50)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        g = torch.Generator()
        g.manual_seed(cfg.get("seed", 42))
        self.train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=cfg.get("num_workers", 4), drop_last=True,
            worker_init_fn=seed_worker, generator=g, pin_memory=True)
        self.val_loader = None
        if val_set is not None:
            self.val_loader = DataLoader(
                val_set, batch_size=1, shuffle=False,
                num_workers=cfg.get("num_workers", 2), pin_memory=True)

        self.criterion = LMSTRDLoss(
            lambda_edge=cfg.get("lambda_edge", 0.05),
            lambda_freq=cfg.get("lambda_freq", 0.01),
            lambda_inr=cfg.get("lambda_inr", 0.1),
            char_eps=cfg.get("char_eps", 1e-3),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
        self.start_epoch = 0
        self.best_psnr = 0.0

    # ------------------------------------------------------------------ #
    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self._train_one_epoch(epoch)
            self.scheduler.step()
            if self.val_loader is not None:
                p, s = self.validate()
                print(f"[epoch {epoch+1:3d}] val PSNR={p:.3f} dB  SSIM={s:.4f}  "
                      f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
                if p > self.best_psnr:
                    self.best_psnr = p
                    self.save(epoch, tag="best")
            self.save(epoch, tag="last")

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        t0 = time.time()
        running = 0.0
        for it, batch in enumerate(self.train_loader):
            rainy = batch["rainy"].to(self.device, non_blocking=True)
            clean = batch["clean"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            pred, inr_img = self.model(rainy)
            loss, logs = self.criterion(pred, clean, inr_img)
            loss.backward()
            self.optimizer.step()

            running += logs["loss"]
            if (it + 1) % self.log_interval == 0:
                avg = running / self.log_interval
                running = 0.0
                print(f"  epoch {epoch+1:3d} it {it+1:4d}/{len(self.train_loader)} "
                      f"loss={avg:.4f} (char={logs['char']:.4f} edge={logs['edge']:.4f} "
                      f"freq={logs['freq']:.4f} inr={logs['inr']:.4f})")
        print(f"  epoch {epoch+1} done in {time.time()-t0:.1f}s")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        ps, ss, n = 0.0, 0.0, 0
        for batch in self.val_loader:
            rainy = batch["rainy"].to(self.device)
            clean = batch["clean"].to(self.device)
            pred, _ = self.model(rainy)
            pred = pred.clamp(0, 1)
            ps += psnr(pred, clean).item()
            ss += ssim(pred, clean).item()
            n += 1
        return ps / max(n, 1), ss / max(n, 1)

    # ------------------------------------------------------------------ #
    def save(self, epoch: int, tag: str = "last"):
        path = os.path.join(self.ckpt_dir, f"lms_trd_{tag}.pth")
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
        }, path)

    def load(self, path: str, resume: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if resume:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_psnr = ckpt.get("best_psnr", 0.0)
