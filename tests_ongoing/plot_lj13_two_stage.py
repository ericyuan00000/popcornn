"""Plot the LJ-13 two-stage diagnostic trace.

Three stacked panels sharing iter on x (concatenated across stages,
with a vertical guide at the stage boundary):
  1. F_TS @ g10001 + F_TS @ parab — semilogy, with the F_TS=0.05 target
     and F_TS=1.0 reference lines.
  2. barrier (E_max - E[0]) on g10001.
  3. loss and |g|_∞ on twin axes; stage thresholds drawn as
     dotted horizontals over their respective stage spans.

Usage:
    python tests_ongoing/plot_lj13_two_stage.py \\
        [--trace <trace.json>] [--out <png>]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_TRACE = '/pscratch/sd/e/ericyuan/temp/popcornn_lj13_diag/run0_seed0/trace.json'


def _arr(xs):
    return np.array([float('nan') if v is None else v for v in xs], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', default=DEFAULT_TRACE)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.trace) as f:
        tr = json.load(f)

    stages = tr['stages']
    seed = tr.get('seed')

    # concat across stages
    iters_concat, fg, fp = [], [], []
    barrier, loss, ginf = [], [], []
    boundaries = []
    stage_spans = []
    offset = 0
    for s in stages:
        n = s['n_iter']
        iters_concat.extend(range(offset, offset + n))
        fg.extend(s['fmax_ts_g'])
        fp.extend(s['fmax_ts_par'])
        barrier.extend(s['barrier'])
        loss.extend(s['loss'])
        ginf.extend(s['ginf'])
        stage_spans.append((s, offset, offset + n))
        offset += n
        boundaries.append(offset)

    iters_concat = np.asarray(iters_concat)
    fg, fp = _arr(fg), _arr(fp)
    barrier = _arr(barrier)
    loss = _arr(loss)
    ginf = _arr(ginf)

    # summary
    summary = []
    for s in stages:
        a_g = _arr(s['fmax_ts_g'])
        a_p = _arr(s['fmax_ts_par'])
        i_g = int(np.nanargmin(a_g)) if np.isfinite(a_g).any() else -1
        i_p = int(np.nanargmin(a_p)) if np.isfinite(a_p).any() else -1
        summary.append(
            f"stage {s['stage']} {s['integrand']:<20s} "
            f"n={s['n_iter']:>3d}  conv@{s['converged_at']}  "
            f"wall={s['elapsed_s']:.1f}s  "
            f"fmax@g argmin@{i_g}={a_g[i_g]:.3e} final={a_g[-1]:.3e}  "
            f"fmax@parab argmin@{i_p}={a_p[i_p]:.3e} final={a_p[-1]:.3e}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    boundary_xs = boundaries[:-1]

    ax = axes[0]
    ax.semilogy(iters_concat, fg, color='C2', lw=1.2, label='g=10001')
    ax.semilogy(iters_concat, fp, color='C3', ls='--', lw=1.4, label='parabolic refine')
    ax.axhline(0.05, color='k', ls=':', lw=1.2, alpha=0.8, label='fmax = 0.05 target')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5, label='fmax = 1.0')
    for x in boundary_xs:
        ax.axvline(x, color='k', alpha=0.25, lw=0.8)
    ax.set_ylabel(r'fmax = $\max_i \|F_i\|_2$ at TS frame')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    n_stages = len(stages)
    stage_word = 'single-stage' if n_stages == 1 else f'{n_stages}-stage'
    title = (f'LJ-13 {stage_word}  ' +
             ' → '.join(s['integrand'] for s in stages) +
             f'  seed={seed}  ({iters_concat.size} iters total)')
    ax.set_title(title)

    ax = axes[1]
    ax.plot(iters_concat, barrier, color='C0', lw=1.0, label='barrier (E_max − E[0]) @ g10001')
    for x in boundary_xs:
        ax.axvline(x, color='k', alpha=0.25, lw=0.8)
    ax.set_ylabel('barrier')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[2]
    ax.semilogy(iters_concat, np.abs(loss), color='C4', lw=1.0, label='|loss| = |∫L dt|')
    ax.set_ylabel('|loss|', color='C4')
    ax.tick_params(axis='y', labelcolor='C4')
    ax2 = ax.twinx()
    ax2.semilogy(iters_concat, ginf, color='C5', lw=1.0, label='|g|_∞')
    # draw stage thresholds on the |g|_∞ axis (distinct colour from the trace)
    for (s, lo, hi) in stage_spans:
        thr = s.get('threshold')
        if thr is not None and thr > 0:
            ax2.hlines(thr, lo, hi - 1, colors='k', linestyles=':',
                       lw=1.2, alpha=0.6)
    ax2.set_ylabel(r'$\|\int \nabla_\theta L\,dt\|_\infty$', color='C5')
    ax2.tick_params(axis='y', labelcolor='C5')
    for x in boundary_xs:
        ax.axvline(x, color='k', alpha=0.25, lw=0.8)
    ax.set_xlabel('Adam iteration (concatenated across stages)')
    ax.grid(True, which='both', alpha=0.3)

    # stage-boundary annotations on top axis
    for i, (_, lo, hi) in enumerate(stage_spans):
        axes[0].text((lo + hi) / 2, axes[0].get_ylim()[1] * 0.6,
                     f'stage {i}', ha='center', va='top', fontsize=9, alpha=0.6)

    txt = '\n'.join(summary)
    fig.text(0.01, 0.005, txt, fontsize=8, family='monospace', va='bottom')

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = args.out or os.path.join(os.path.dirname(args.trace), 'diag.png')
    fig.savefig(out, dpi=120)
    print(f'plot: {out}')
    print('summary:')
    for line in summary:
        print(f'  {line}')


if __name__ == '__main__':
    main()
