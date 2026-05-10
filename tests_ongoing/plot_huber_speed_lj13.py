"""Plot the LJ-13 huber speed benchmark.

Two subplots: median wall (ms) vs δ, and GK eval points vs δ. Huber and
pseudo-Huber overlaid; pvre and pvre² as horizontal references.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_IN = '/pscratch/sd/e/ericyuan/temp/popcornn_huber_speed_lj13_n8d4.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=DEFAULT_IN)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    data = json.load(open(args.input))
    rows = data['rows']

    deltas, huber_ms, huber_pts = [], [], []
    pseudo_ms, pseudo_pts = [], []
    pvre_ms = pvre_sq_ms = None
    pvre_pts = pvre_sq_pts = None
    for r in rows:
        if r['label'] == 'pvre':
            pvre_ms = r['med_s'] * 1e3
            pvre_pts = r['eval_pts']
        elif r['label'] == 'pvre_squared':
            pvre_sq_ms = r['med_s'] * 1e3
            pvre_sq_pts = r['eval_pts']
        elif r['label'].startswith('pvre_huber'):
            d = float(r['label'].split('=')[1])
            deltas.append(d) if d not in deltas else None
            huber_ms.append((d, r['med_s'] * 1e3))
            huber_pts.append((d, r['eval_pts']))
        elif r['label'].startswith('pseudo_huber'):
            d = float(r['label'].split('=')[1])
            pseudo_ms.append((d, r['med_s'] * 1e3))
            pseudo_pts.append((d, r['eval_pts']))

    huber_ms.sort(); huber_pts.sort()
    pseudo_ms.sort(); pseudo_pts.sort()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogx([d for d,_ in huber_ms], [v for _,v in huber_ms],
                'o-', color='C0', label='pvre_huber')
    ax.semilogx([d for d,_ in pseudo_ms], [v for _,v in pseudo_ms],
                's-', color='C1', label='pvre_pseudo_huber')
    ax.axhline(pvre_ms, color='k', ls='--', lw=1.0, label=f'pvre = {pvre_ms:.0f} ms')
    ax.axhline(pvre_sq_ms, color='gray', ls=':', lw=1.0, label=f'pvre² = {pvre_sq_ms:.0f} ms')
    ax.set_xlabel('δ')
    ax.set_ylabel('median wall per integrate_path call (ms)')
    ax.set_title('LJ-13 per-call cost vs δ')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    ax.semilogx([d for d,_ in huber_pts], [v for _,v in huber_pts],
                'o-', color='C0', label='pvre_huber')
    ax.semilogx([d for d,_ in pseudo_pts], [v for _,v in pseudo_pts],
                's-', color='C1', label='pvre_pseudo_huber')
    ax.axhline(pvre_pts, color='k', ls='--', lw=1.0, label=f'pvre = {pvre_pts} pts')
    ax.axhline(pvre_sq_pts, color='gray', ls=':', lw=1.0, label=f'pvre² = {pvre_sq_pts} pts')
    ax.set_xlabel('δ')
    ax.set_ylabel('GK eval points per call')
    ax.set_title('LJ-13 GK refinement vs δ')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'LJ-13 huber/pseudo speed (warmup={data["warmup_iters"]} pvre² steps, '
                 f'lr={data["warmup_lr"]:.0e}, rtol={data["rtol"]:.0e}, gk21)',
                 y=1.02)
    fig.tight_layout()

    out = args.out or args.input.replace('.json', '.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'plot: {out}')


if __name__ == '__main__':
    main()
