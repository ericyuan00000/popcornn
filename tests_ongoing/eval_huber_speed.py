"""Per-call cost of pvre / pvre_squared / pvre_huber(δ) / pvre_pseudo_huber(δ).

Build a partly-trained Müller-Brown path (50 iters of pvre_squared so
``s = v·F`` crosses both 0 and ±δ for every δ in the sweep), then on
that *frozen* path time a single ``integrate_path`` call for each
loss. Records:

  - wall time per call (median over N_REPEATS runs to dampen GPU noise)
  - integration point count (= N_intervals × 21 for ``gk21``)

The path is held fixed across all losses so the only variable is the
loss kernel itself — direct apples-to-apples cost comparison.

Pseudo-Huber tests the hypothesis that Huber's leftover cost vs pvre²
comes from the slope corner at ``|s|=δ`` (C¹ but not C²): pseudo-Huber
shares Huber's bounded-gradient behaviour but is C^∞, so adaptive GK
should not subdivide locally there.
"""
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator


WARMUP_ITERS = 50      # warm path so s(t) crosses zero a few times
N_REPEATS = 20         # per-loss timing repeats; report median + IQR
DELTAS = [1.0e-3, 1.0e-2, 1.0e-1, 1.0e+0, 1.0e+1, 1.0e+2]


def time_integration(integ, path):
    """One integrate_path call; returns (wall_s, eval_pts)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    integ.integrate_path(path)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    out = integ.integral_output
    # full_output=True populates .t with shape [N_intervals, K]; K=21 for gk21.
    if out.t is None:
        eval_pts = None
    else:
        eval_pts = int(out.t.numel())
    return elapsed, eval_pts


def bench(label, path, kwargs):
    integ = PathIntegrator(method='gk21', rtol=1e-2, atol=1e-7,
                           full_output=True,
                           device=path.device, dtype=path.dtype,
                           **kwargs)
    times, evals = [], []
    for _ in range(N_REPEATS):
        t, n = time_integration(integ, path)
        times.append(t); evals.append(n)
    times.sort(); evals.sort()
    n = len(times)
    med_t = times[n // 2]
    p25_t = times[n // 4]; p75_t = times[(3 * n) // 4]
    med_n = evals[n // 2] if evals[0] is not None else None
    return label, med_t, p25_t, p75_t, med_n


def main():
    torch.manual_seed(0)
    init = {
        'images': [[-0.558, 1.442], [0.623, 0.028]],
        'path_params': {'name': 'mlp', 'n_embed': 8, 'depth': 4, 'activation': 'gelu'},
    }
    mep = Popcornn(**init)
    pot = get_potential(images=mep.images, name='muller_brown',
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)

    # Warm-up with pvre_squared, lr=1e-2 — same recipe as the smoothness audit
    integ = PathIntegrator(method='gk21', path_integrand_names='pvre_squared',
                           rtol=1e-1, atol=1e-7,
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path,
                         optimizer={'name': 'adam', 'lr': 1.0e-2},
                         threshold=0.0,
                         device=mep.device, dtype=mep.dtype)
    for _ in range(WARMUP_ITERS):
        optr.optimization_step(mep.path, integ)

    rows = []
    rows.append(bench('pvre', mep.path,
                      {'path_integrand_names': 'pvre'}))
    rows.append(bench('pvre_squared', mep.path,
                      {'path_integrand_names': 'pvre_squared'}))
    for delta in DELTAS:
        rows.append(bench(f'pvre_huber δ={delta:.0e}', mep.path,
                          {'path_integrand_names': 'pvre_huber',
                           'path_integrand_kwargs': {'pvre_huber': {'delta': float(delta)}}}))
    for delta in DELTAS:
        rows.append(bench(f'pseudo_huber δ={delta:.0e}', mep.path,
                          {'path_integrand_names': 'pvre_pseudo_huber',
                           'path_integrand_kwargs': {'pvre_pseudo_huber': {'delta': float(delta)}}}))

    pvre_t = next(r[1] for r in rows if r[0] == 'pvre')
    pvre_sq_t = next(r[1] for r in rows if r[0] == 'pvre_squared')

    print(f'\n=== integration cost on partly-trained MB path '
          f'(warmup={WARMUP_ITERS} pvre_squared steps, rtol=1e-2, gk21) ===')
    print(f'{"loss":<24s} {"med_ms":>9s} {"IQR_ms":>14s} {"eval_pts":>10s} '
          f'{"vs_pvre":>9s} {"vs_pvre_sq":>11s}')
    for label, med, p25, p75, n_eval in rows:
        evs = f'{n_eval:>10d}' if n_eval is not None else '       n/a'
        iqr = f'[{p25*1e3:>5.1f},{p75*1e3:>5.1f}]'
        print(f'{label:<24s} {med*1e3:>9.2f} {iqr:>14s} {evs} '
              f'{med/pvre_t:>9.3f} {med/pvre_sq_t:>11.3f}')

    print(f'\nReport: huber vs pseudo-huber wall vs pvre at each δ '
          f'(< 1.0 means faster than pvre):')
    for label, med, _, _, _ in rows:
        if 'huber' in label.lower():
            print(f'  {label:<24s}: {med/pvre_t:.3f}x pvre  '
                  f'{med/pvre_sq_t:.3f}x pvre² ({pvre_t/med:.2f}x faster than pvre)')


if __name__ == '__main__':
    main()
