"""Complexity measurement: exact parameter count and FLOP/MAC profiling
(Section 4.4).

These functions report the MEASURED complexity of THIS implementation. They do
not assume or hard-code any value from the paper -- the whole point is to let a
reviewer (and the authors) compare the measured numbers against the reported
7.87 M params / 64.22 G FLOPs.

UNITS / CONVENTION (read this before quoting a number)
------------------------------------------------------
Restoration papers are inconsistent about "FLOPs": some report multiply-add
*counts* (MACs), others report 2 x MACs. To avoid ambiguity this profiler
returns BOTH:
    macs_G   : multiply-accumulate operations, in units of 1e9   (a.k.a. "GMacs")
    flops_G  : 2 x macs_G                                          (the 2xMAC convention)
The reported per-module table (paper Table 4) is closest in scale to the MAC
count. Whichever convention the manuscript intends, state it explicitly when
reporting; do not mix the two.
"""

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in params)


def count_shared_unique_parameters(model: nn.Module) -> int:
    """Count unique parameter tensors (shared weights counted once)."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


@torch.no_grad()
def estimate_macs(model: nn.Module, input_size=(1, 3, 256, 256),
                  device: str = 'cpu') -> float:
    """Multiply-accumulate ops for Conv1d/Conv2d/Linear via forward hooks.

    Returns MACs for one forward pass at `input_size`. Attention is dominated by
    its Conv2d projections, which are counted; the linear-attention einsum mixing
    adds < 1 GMac at 256x256 (a negligible correction reported in the README).
    """
    macs = [0]
    handles = []

    def conv_hook(module, inp, out):
        out_elems = out.numel()
        k = 1
        for ks in module.kernel_size:
            k *= ks
        in_per_group = module.in_channels // module.groups
        macs[0] += out_elems * in_per_group * k

    def linear_hook(module, inp, out):
        macs[0] += out.numel() * module.in_features

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    x = torch.randn(*input_size, device=device)
    model(x)
    if was_training:
        model.train()
    for h in handles:
        h.remove()
    return float(macs[0])


# Backwards-compatible alias (older scripts import estimate_flops; it returns MACs).
def estimate_flops(model: nn.Module, input_size=(1, 3, 256, 256),
                   device: str = 'cpu') -> float:
    return estimate_macs(model, input_size=input_size, device=device)


@torch.no_grad()
def thop_macs(model: nn.Module, input_size=(1, 3, 256, 256),
              device: str = 'cpu'):
    """Independent MAC cross-check via thop, if installed. Returns MACs or None.

    Runs on a deep copy so thop's bookkeeping buffers never touch the caller's
    model (some thop versions attach total_ops/total_params that can break a
    subsequent real forward pass).
    """
    try:
        import copy
        from thop import profile
        probe = copy.deepcopy(model).to(device).eval()
        x = torch.randn(*input_size, device=device)
        macs, _ = profile(probe, inputs=(x,), verbose=False)
        return float(macs)
    except Exception:
        return None


def complexity_report(model: nn.Module, input_size=(1, 3, 256, 256),
                      device: str = 'cpu') -> dict:
    total = count_parameters(model, trainable_only=False)
    unique = count_shared_unique_parameters(model)
    macs = estimate_macs(model, input_size=input_size, device=device)
    tmacs = thop_macs(model, input_size=input_size, device=device)
    return {
        "params_total_M": total / 1e6,
        "params_unique_M": unique / 1e6,
        "macs_G": macs / 1e9,
        "flops_G": 2.0 * macs / 1e9,           # 2 x MAC convention
        "thop_macs_G": (tmacs / 1e9) if tmacs is not None else None,
        "input_size": input_size,
    }
