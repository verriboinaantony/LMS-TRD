"""Measured model complexity (Sec. 4.4).

Convention used everywhere in the paper and in this repository
--------------------------------------------------------------
* Parameters: number of unique trainable tensors' elements (a weight shared by
  several pyramid levels is counted once).
* MACs: multiply-accumulate operations of all convolutions, linear layers and
  matrix products (including the attention products), for ONE forward pass of
  a 1 x 3 x 256 x 256 input. Measured with PyTorch's own operator-level counter
  (torch.utils.flop_counter.FlopCounterMode), which reports FLOPs = 2 x MACs.
  Element-wise operations (normalisation, activations, resizing, additions)
  are not counted, as is standard.
* FLOPs = 2 x MACs. Tables in the paper report GMACs; FLOPs are given in the
  text where needed.

A cross-check with ptflops (the profiler used by the official NeRD-Rain code,
which counts convolution/linear layers only) is available in
tools/profile_complexity.py.
"""

from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(model: nn.Module) -> int:
    """Unique parameters (shared tensors counted once)."""
    return sum(p.numel() for p in model.parameters())


def module_parameters(model: nn.Module, prefix: str) -> int:
    seen, total = set(), 0
    for name, p in model.named_parameters():
        if (name == prefix or name.startswith(prefix + '.')) and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


@torch.no_grad()
def measure_flops(model: nn.Module, input_size=(1, 3, 256, 256), device: str = 'cpu',
                  per_module: bool = False):
    """Return total FLOPs (2 x MACs) and optionally per-module FLOPs."""
    was_training = model.training
    model.eval()
    x = torch.rand(*input_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    if was_training:
        model.train()
    total = counter.get_total_flops()
    if not per_module:
        return total, None
    per = OrderedDict()
    for name, ops in counter.get_flop_counts().items():
        per[name] = sum(ops.values())
    return total, per


def group_flops(per_module: Dict[str, int], root: str, groups: Dict[str, tuple]) -> Dict[str, int]:
    """Sum per-module FLOPs into named groups of top-level children of `root`.

    groups: {'Backbone': ('stages', 'embed', ...), ...}
    """
    # Containers such as ModuleDict are never called themselves, so their FQN may
    # be absent; count the top-most CALLED modules below each prefix, skipping
    # modules whose parent is already counted (avoids double counting).
    out = OrderedDict((g, 0) for g in groups)
    for g, children in groups.items():
        for ch in children:
            prefix = f'{root}.{ch}'
            for key, val in per_module.items():
                if not (key == prefix or key.startswith(prefix + '.')):
                    continue
                parent, counted_above = key, False
                while '.' in parent:
                    parent = parent.rsplit('.', 1)[0]
                    if parent in per_module and (parent == prefix or parent.startswith(prefix + '.')):
                        counted_above = True
                        break
                if not counted_above:
                    out[g] += val
    return out


def complexity_report(model: nn.Module, input_size=(1, 3, 256, 256),
                      device: str = 'cpu') -> dict:
    flops, _ = measure_flops(model, input_size, device)
    return {
        'params_M': count_parameters(model) / 1e6,
        'gmacs': flops / 2e9,
        'gflops': flops / 1e9,
        'input_size': tuple(input_size),
    }


def ptflops_macs(model: nn.Module, input_res=(3, 256, 256)) -> Optional[float]:
    """GMACs as reported by ptflops (conv/linear layers only), or None if unavailable."""
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return None
    import copy
    probe = copy.deepcopy(model).eval()
    with torch.no_grad():
        macs, _ = get_model_complexity_info(probe, input_res, as_strings=False,
                                            print_per_layer_stat=False, verbose=False,
                                            backend='pytorch')
    return macs / 1e9
