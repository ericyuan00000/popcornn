"""Plot the MB sampling diagnostic trace.

Three stacked panels sharing iter on x:
  1. F_TS @ {g101, g1001, g10001, parab} — semilogy, with F_TS=1 target line.
     If g101 rebounds while denser/refined ones don't → 101-grid sampling.
  2. saddle t-position for each resolution — shows drift / hopping.
  3. loss and |g|_∞ on twin axes — what other trigger candidates would do.

Usage:
    python tests_ongoing/plot_mb_sampling_artifact.py \\
        [--trace <trace.json>] [--out <png>]

Defaults to the path the diag script writes to.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_TRACE = '/pscratch/sd/e/ericyuan/temp/popcornn_sampling_diag/lr1e-3_pseudo_d1e+2_s0/trace.json'

LABEL_STYLE = {
    'g101':   ('C0', '-',  1.2, 'g=101'),
    'g1001':  ('C1', '-',  1.2, 'g=1001'),
    'g10001': ('C2', '-',  1.2, 'g=10001'),
    'parab':  ('C3', '--', 1.4, 'parabolic refine'),
}


def _arr(xs):
    return np.array([float('nan') if v is None else v for v in xs], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', default=DEFAULT_TRACE)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.trace) as f:
        tr = json.load(f)

    n = tr['n_iter']
    iters = np.arange(n)
    f_ts = tr['f_inf_ts']
    ts_t = tr['ts_t']
    loss = _arr(tr['loss'])
    ginf = _arr(tr['ginf'])

    summary = []
    for lab in ('g101', 'g1001', 'g10001', 'parab'):
        a = _arr(f_ts[lab])
        finite = np.isfinite(a)
        if finite.any():
            idx = int(np.nanargmin(a))
            summary.append(f'{lab}: argmin@{idx}  min={a[idx]:.3e}  final={a[-1]:.3e}')
        else:
            summary.append(f'{lab}: all-nan')

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    ax = axes[0]
    for lab, (c, ls, lw, name) in LABEL_STYLE.items():
        a = _arr(f_ts[lab])
        ax.semilogy(iters, a, color=c, ls=ls, lw=lw, label=name)
    ax.axhline(1.0, color='k', ls=':', lw=1.0, alpha=0.6, label='F_TS = 1 target')
    ax.set_ylabel(r'$\|F\|_\infty$  at TS frame')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    integrand = tr.get('integrand', 'pvre_pseudo_huber')
    delta = tr.get('delta')
    delta_str = f' δ={delta:.0e}' if delta is not None else ''
    ax.set_title(f'MB  lr={tr["lr"]:.0e}  {integrand}{delta_str}  '
                 f'seed={tr["seed"]}  ({n} iters)')

    ax = axes[1]
    for lab, (c, ls, lw, name) in LABEL_STYLE.items():
        a = _arr(ts_t[lab])
        ax.plot(iters, a, color=c, ls=ls, lw=lw, label=name)
    ax.set_ylabel('t at argmax-E (saddle)')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[2]
    ax.semilogy(iters, loss, color='C4', lw=1.0, label='loss = ∫L dt')
    ax.set_ylabel('loss', color='C4')
    ax.tick_params(axis='y', labelcolor='C4')
    ax2 = ax.twinx()
    ax2.semilogy(iters, ginf, color='C5', lw=1.0, label='|g|_∞')
    ax2.set_ylabel(r'$\|\int \nabla_\theta L\,dt\|_\infty$', color='C5')
    ax2.tick_params(axis='y', labelcolor='C5')
    ax.set_xlabel('Adam iteration')
    ax.grid(True, which='both', alpha=0.3)

    txt = '\n'.join(summary)
    fig.text(0.01, 0.005, txt, fontsize=8, family='monospace', va='bottom')

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out = args.out or os.path.join(os.path.dirname(args.trace), 'diag.png')
    fig.savefig(out, dpi=120)
    print(f'plot: {out}')
    print('summary:')
    for line in summary:
        print(f'  {line}')


if __name__ == '__main__':
    main()
