#!/usr/bin/env python3
"""Turn evaluation JSON files (written by test.py --json) into the numbers of the paper.

Main results (Table 2) - several seeds of the same model:
    python tools/aggregate_results.py main results/eval_lms_trd_s42.json \
        results/eval_lms_trd_s123.json results/eval_lms_trd_s456.json

Ablation (Table 4) - one JSON per row (evaluated on Rain100H) + results/complexity.json:
    python tools/aggregate_results.py ablation --complexity results/complexity.json \
        results/eval_ablation_M0.json ... results/eval_ablation_M8.json

Prints mean +- std per test set and LaTeX-ready table rows.
"""

import argparse
import json
import statistics as st

ORDER = ['Test100', 'Rain100L', 'Rain100H', 'Test1200', 'Test2800']   # column order of Table 2


def load(p):
    with open(p) as f:
        return json.load(f)


def main_table(paths):
    runs = [load(p) for p in paths]
    cells = []
    print(f'{len(runs)} run(s)')
    for name in ORDER + ['Average']:
        if name == 'Average':
            ps = [r['average']['psnr_y'] for r in runs]
            ss = [r['average']['ssim_y'] for r in runs]
        else:
            ps = [r['sets'][name]['psnr_y'] for r in runs]
            ss = [r['sets'][name]['ssim_y'] for r in runs]
        sd_p = st.pstdev(ps) if len(ps) > 1 else 0.0
        sd_s = st.pstdev(ss) if len(ss) > 1 else 0.0
        print(f'{name:9s} PSNR {st.mean(ps):.2f} +- {sd_p:.2f}   SSIM {st.mean(ss):.3f} +- {sd_s:.3f}')
        cells += [f'{st.mean(ps):.2f}', f'{st.mean(ss):.3f}']
    print('\nLaTeX row (mean over runs):\nLMS-TRD (ours) & ' + ' & '.join(cells) + r' \\')


def ablation_table(paths, complexity):
    comp = {r['id']: r for r in load(complexity)['ablation']}
    print('LaTeX rows (Rain100H):')
    for p in paths:
        r = load(p)
        key = next((k for k in comp if f'_{k}' in p or f'{k}.' in p or f'/{k}' in p), None)
        if key is None:
            raise SystemExit(f'cannot infer ablation id (M0..M8) from file name {p}')
        res = r['sets']['Rain100H']
        c = comp[key]
        print(f"{key} & {c['config']} & {c['params_M']:.2f} & {c['gmacs']:.2f} & "
              f"{res['psnr_y']:.2f} & {res['ssim_y']:.3f} \\\\")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('main')
    a.add_argument('jsons', nargs='+')
    b = sub.add_parser('ablation')
    b.add_argument('--complexity', default='results/complexity.json')
    b.add_argument('jsons', nargs='+')
    args = ap.parse_args()
    if args.cmd == 'main':
        main_table(args.jsons)
    else:
        ablation_table(args.jsons, args.complexity)
