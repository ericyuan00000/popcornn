"""Plot wall time and integration eval points per loss kernel.

Reads the JSON from eval_huber_speed.py and renders two horizontal bar
panels: ms/call (median + IQR) and integration eval points. Bars are
grouped by family (pvre, pvre², pvre_huber, pvre_pseudo_huber) so the
δ ordering inside each family is visible at a glance.

Usage:
  python tests_ongoing/plot_huber_speed.py [--in <path.json>] [--out <path.png>]
"""
import argparse
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_IN = '/pscratch/sd/e/ericyuan/temp/popcornn_huber_speed_n8d4.json'

FAMILY_COLOR = {
    'pvre': 'C0',
    'pvre_squared': 'C2',
    'pvre_huber': 'C1',
    'pvre_pseudo_huber': 'C3',
}


def _classify(label):
    """Return (family, delta_or_None) tuple from a row label."""
    if label == 'pvre':
        return 'pvre', None
    if label == 'pvre_squared':
        return 'pvre_squared', None
    m = re.match(r'(pvre_huber|pseudo_huber)\s+δ=([0-9.eE+-]+)', label)
    if m:
        family = 'pvre_huber' if m.group(1) == 'pvre_huber' else 'pvre_pseudo_huber'
        return family, float(m.group(2))
    return 'unknown', None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=DEFAULT_IN)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)

    rows = data['rows']
    parsed = []
    for r in rows:
        family, delta = _classify(r['label'])
        parsed.append({
            'label': r['label'], 'family': family, 'delta': delta,
            'med_ms': r['med_s'] * 1e3,
            'p25_ms': r['p25_s'] * 1e3,
            'p75_ms': r['p75_s'] * 1e3,
            'eval_pts': r['eval_pts'],
        })

    family_order = ['pvre', 'pvre_huber', 'pvre_pseudo_huber', 'pvre_squared']
    sorted_rows = []
    for fam in family_order:
        group = [r for r in parsed if r['family'] == fam]
        group.sort(key=lambda r: (r['delta'] if r['delta'] is not None else 0.0))
        sorted_rows.extend(group)

    labels = [r['label'] for r in sorted_rows]
    meds = np.array([r['med_ms'] for r in sorted_rows])
    p25s = np.array([r['p25_ms'] for r in sorted_rows])
    p75s = np.array([r['p75_ms'] for r in sorted_rows])
    evs = np.array([r['eval_pts'] if r['eval_pts'] is not None else 0
                    for r in sorted_rows])
    colors = [FAMILY_COLOR.get(r['family'], 'gray') for r in sorted_rows]

    n = len(sorted_rows)
    y = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * n + 1.5), sharey=True)

    ax = axes[0]
    ax.barh(y, meds, color=colors, edgecolor='k', linewidth=0.4)
    ax.errorbar(meds, y, xerr=[meds - p25s, p75s - meds],
                fmt='none', ecolor='k', capsize=3, lw=0.8)
    ax.set_xlabel('integrate_path wall (ms, median; bars = IQR)')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis='x', which='both', alpha=0.3)

    ax = axes[1]
    ax.barh(y, evs, color=colors, edgecolor='k', linewidth=0.4)
    ax.set_xlabel('GK eval points (= n_intervals × 21)')
    ax.invert_yaxis()
    ax.grid(True, axis='x', which='both', alpha=0.3)
    xmax = float(evs.max()) if evs.size else 1.0
    for yi, v in zip(y, evs):
        ax.text(v + 0.01 * xmax, yi, str(v), va='center', fontsize=8)

    mlp = data['mlp']
    fig.suptitle(
        f'integrand kernel cost on partly-trained MB '
        f'(MLP n_embed={mlp["n_embed"]} depth={mlp["depth"]}, '
        f'warmup={data["warmup_iters"]} pvre² steps, rtol={data["rtol"]:.0e}, '
        f'gk21, {data["n_repeats"]} repeats)',
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = args.out or os.path.splitext(args.inp)[0] + '.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'plot: {out}')


if __name__ == '__main__':
    main()
