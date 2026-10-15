#!/usr/bin/env python3
"""Measure parameters and MACs for every ablation row (M0-M8) and the
per-module breakdown of NeRD-Rain (M0) and LMS-TRD (M8) at 256 x 256.

Writes results/complexity.json and results/complexity.md.

    python tools/profile_complexity.py            # all rows, 256x256
    python tools/profile_complexity.py --size 512 # other resolution
"""

import argparse
import json
import os
import platform
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model, PRESETS, PRESET_NAMES  # noqa: E402
from utils.complexity import (count_parameters, measure_flops, module_parameters,  # noqa: E402
                              group_flops)

GROUPS = {
    'Transformer U-Net stages': ('stages', 'align'),
    'Embeddings, context and output convs': ('embed', 'ctx', 'head'),
    'INR branch (latent upsampling + INR)': ('inr_up', 'inr'),
    'Cross-scale fusion (BFPU / eBFU)': ('fuse',),
    'Rain Refinement Block': ('rrb',),
}


def _ptflops_subprocess(preset, size):
    """ptflops in a separate process (it deep-copies the model; keeps peak RAM low)."""
    code = ('import sys; sys.path.insert(0, %r); from models import build_model; '
            'from utils.complexity import ptflops_macs; '
            'print(ptflops_macs(build_model({"preset": %r}).eval(), (3, %d, %d)))'
            % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), preset, size, size))
    try:
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        val = out.stdout.strip().splitlines()[-1]
        return None if val == 'None' else round(float(val), 3)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--out', default='results')
    ap.add_argument('--no-ptflops', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(0)
    size = (1, 3, args.size, args.size)

    rows = []
    breakdown = {}
    for key in PRESETS:
        model = build_model({'preset': key}).eval()
        params = count_parameters(model)
        flops, per = measure_flops(model, size, per_module=True)
        row = {'id': key, 'config': PRESET_NAMES[key], 'params': params,
               'macs': int(flops // 2), 'params_M': params / 1e6, 'gmacs': flops / 2e9,
               'gflops': flops / 1e9}
        if not args.no_ptflops and key in ('M0', 'M8'):
            row['ptflops_gmacs_conv_linear_only'] = _ptflops_subprocess(key, args.size)
        rows.append(row)
        if key in ('M0', 'M8'):
            root = type(model).__name__
            g_flops = group_flops(per, root, GROUPS)
            breakdown[key] = {
                g: {'gmacs': v / 2e9,
                    'params_M': sum(module_parameters(model, c) for c in GROUPS[g]) / 1e6}
                for g, v in g_flops.items()}
        print(f"{key:3s} {PRESET_NAMES[key]:32s} params={params/1e6:8.4f} M  "
              f"MACs={flops/2e9:8.3f} G  FLOPs={flops/1e9:8.3f} G"
              + (f"  ptflops(conv/linear only)={row['ptflops_gmacs_conv_linear_only']} G"
                 if 'ptflops_gmacs_conv_linear_only' in row else ''))

    env = {'torch': torch.__version__, 'python': platform.python_version(),
           'counter': 'torch.utils.flop_counter.FlopCounterMode (FLOPs = 2 x MACs)',
           'input': list(size)}
    os.makedirs(args.out, exist_ok=True)
    tag = '' if args.size == 256 else f'_{args.size}'
    with open(os.path.join(args.out, f'complexity{tag}.json'), 'w') as f:
        json.dump({'environment': env, 'ablation': rows, 'breakdown': breakdown}, f, indent=2)

    m0, m8 = rows[0], rows[-1]
    lines = [f'# Measured complexity ({args.size}x{args.size}, batch 1)', '',
             f"Counter: {env['counter']}; PyTorch {env['torch']}.", '',
             '| ID | Configuration | Params (M) | MACs (G) | FLOPs (G) |',
             '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['config']} | {r['params_M']:.3f} | {r['gmacs']:.2f} | {r['gflops']:.2f} |")
    lines += ['', f"LMS-TRD vs NeRD-Rain: parameters -{100 * (1 - m8['params'] / m0['params']):.1f}%, "
                  f"MACs -{100 * (1 - m8['gmacs'] / m0['gmacs']):.1f}%.", '']
    if m0.get('ptflops_gmacs_conv_linear_only') is not None:
        lines += [f"Cross-check with ptflops (convolution and linear layers only, the profiler of the "
                  f"official NeRD-Rain code): NeRD-Rain {m0['ptflops_gmacs_conv_linear_only']:.2f} GMACs, "
                  f"LMS-TRD {m8['ptflops_gmacs_conv_linear_only']:.2f} GMACs.", '']
    lines += [
              '## Per-module breakdown', '',
              '| Module | NeRD-Rain MACs (G) | LMS-TRD MACs (G) | NeRD-Rain params (M) | LMS-TRD params (M) |',
              '|---|---|---|---|---|']
    for g in GROUPS:
        a, b = breakdown['M0'][g], breakdown['M8'][g]
        lines.append(f"| {g} | {a['gmacs']:.2f} | {b['gmacs']:.2f} | {a['params_M']:.3f} | {b['params_M']:.3f} |")
    with open(os.path.join(args.out, f'complexity{tag}.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[-(len(GROUPS) + 4):]))


if __name__ == '__main__':
    main()
