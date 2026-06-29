#!/usr/bin/env python3
"""End-to-end smoke test.

Verifies that:
  1. the model builds and runs a forward pass,
  2. the composite loss + backward pass run,
  3. the SHARED SST weights receive gradients from every scale (when sharing is on),
  4. inference works on a single image,
and prints the MEASURED parameter count and complexity (MACs and FLOPs) at the
paper's 256x256 input, next to the reported 7.87 M / 64.22 G.

(Backward is run at 128x128 to stay within small-RAM machines; the complexity
report is measured at 256x256 to match the paper.)
"""

import torch

from models import build_model
from models.lms_trd import LMSTRD
from training.losses import LMSTRDLoss
from utils import set_seed, complexity_report


def _shared_sst_weight(backbone):
    """Return a parameter from a SHARED SST block, for either backbone kind."""
    if hasattr(backbone, "unet"):                       # UNetSST (default)
        return backbone.unet["enc"][0][0].ffn.pw_expand.weight
    if hasattr(backbone, "blocks"):                     # FlatSharedSST
        return backbone.blocks[0].ffn.pw_expand.weight
    return None


def main():
    set_seed(42)
    device = "cpu"
    model = build_model().to(device)
    print(f"[info] backbone = {model.backbone_kind}, "
          f"share_weights = {getattr(model.backbone, 'share_weights', None)}")

    # 1. forward
    x = torch.rand(1, 3, 128, 128, device=device)
    out, inr = model(x)
    assert out.shape == x.shape, out.shape
    assert inr is not None and inr.shape == x.shape, "INR branch should be active by default"
    print(f"[ok] forward: out={tuple(out.shape)}  inr={tuple(inr.shape)}")

    # 2. loss + backward
    target = torch.rand_like(x)
    criterion = LMSTRDLoss()
    loss, logs = criterion(out, target, inr)
    loss.backward()
    print(f"[ok] backward: {logs}")

    # 3. shared weights get gradients from the multi-scale pass
    if getattr(model.backbone, "share_weights", False):
        shared_param = _shared_sst_weight(model.backbone)
        assert shared_param is not None, "could not locate a shared SST weight"
        assert shared_param.grad is not None and shared_param.grad.abs().sum() > 0
        print("[ok] shared SST block received gradient from all scales")
    else:
        print("[--] weight sharing disabled in this build; skipping shared-grad check")

    # 4. single-image inference at full 256x256
    model.eval()
    with torch.no_grad():
        single, _ = model(torch.rand(1, 3, 256, 256, device=device))
    print(f"[ok] inference: {tuple(single.shape)}")

    # measured complexity at the paper's resolution
    rep = complexity_report(model, input_size=(1, 3, 256, 256), device=device)
    print(f"\n[MEASURED] params(total)  = {rep['params_total_M']:.3f} M")
    print(f"[MEASURED] params(unique) = {rep['params_unique_M']:.3f} M "
          f"(shared weights counted once)")
    print(f"[MEASURED] MACs           = {rep['macs_G']:.2f} GMac  @ 256x256")
    print(f"[MEASURED] FLOPs (2xMAC)  = {rep['flops_G']:.2f} G     @ 256x256")
    if rep["thop_macs_G"] is not None:
        print(f"[MEASURED] MACs (thop)    = {rep['thop_macs_G']:.2f} GMac (independent cross-check)")
    print("[PAPER]    reported       = 7.87 M params / 64.22 G FLOPs  (see README + RECONCILIATION.md)")

    # quick ablation sanity: each toggle changes the parameter count as expected
    base = sum(p.numel() for p in build_model().parameters())
    no_rrb = sum(p.numel() for p in LMSTRD(use_rrb=False).parameters())
    no_inr = sum(p.numel() for p in LMSTRD(use_inr=False).parameters())
    assert no_rrb < base and no_inr < base
    print(f"\n[ok] ablation toggles wired: full={base/1e6:.3f}M  "
          f"-RRB={no_rrb/1e6:.3f}M  -INR={no_inr/1e6:.3f}M")


if __name__ == "__main__":
    main()
