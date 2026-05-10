"""Per-call cost of pvre / pvre_squared / pvre_huber(δ) / pvre_pseudo_huber(δ) on LJ-13.

LJ-13 analog of tests_ongoing/eval_huber_speed.py. Build a partly-trained
LJ-13 path (50 iters of pvre² at lr=1e-3 — matches the warm-up stage of
examples/configs/lj13.yaml), then on that *frozen* path time a single
``integrate_path`` call for each loss. Records:

  - wall time per call (median over N_REPEATS, dampens GPU noise)
  - integration point count (= N_intervals × 21 for ``gk21``)

The path is held fixed across all losses so the only variable is the
loss kernel itself — direct apples-to-apples cost comparison.

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclude=nid001208 --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python /abs/path/Popcornn/tests_ongoing/eval_huber_speed_lj13.py"
"""
import json
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


WARMUP_ITERS = int(os.environ.get('WARMUP_ITERS', '50'))
N_REPEATS = 20
DELTAS = [float(x) for x in os.environ.get(
    'DELTAS', '1e-5,1e-4,1e-3,1e-2,1e-1,1e0').split(',')]
LJ13_XYZ = os.path.join(REPO_ROOT, 'examples/configs/lj13.xyz')


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
    if out.t is None:
        eval_pts = None
    else:
        eval_pts = int(out.t.numel())
    return elapsed, eval_pts


RTOL = float(os.environ.get('RTOL', '1.0e-2'))
ATOL = float(os.environ.get('ATOL', '1.0e-7'))


def bench(label, path, kwargs):
    integ = PathIntegrator(method='gk21', rtol=RTOL, atol=ATOL,
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
        'images': LJ13_XYZ,
        'path_params': {'name': 'mlp', 'n_embed': 8, 'depth': 4, 'activation': 'gelu'},
        'num_record_points': 101,
        'device': 'cuda',
        'seed': 0,
    }
    mep = Popcornn(**init)
    pot = get_potential(images=mep.images, name='lennard_jones',
                        epsilon=1.0, sigma=1.0, cutoff=3.0,
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)

    # Warm-up with pvre² at lr=1e-3 (same as lj13.yaml stage 1).
    integ = PathIntegrator(method='gk21', path_integrand_names='pvre_squared',
                           rtol=1e-1, atol=1e-7,
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path,
                         optimizer={'name': 'adam', 'lr': 1.0e-3},
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

    print(f'\n=== integration cost on LJ-13 path '
          f'(warmup={WARMUP_ITERS} pvre² steps, lr=1e-3, '
          f'rtol={RTOL:.0e}, atol={ATOL:.0e}, gk21) ===')
    print(f'{"loss":<24s} {"med_ms":>9s} {"IQR_ms":>14s} {"eval_pts":>10s} '
          f'{"vs_pvre":>9s} {"vs_pvre_sq":>11s}')
    for label, med, p25, p75, n_eval in rows:
        evs = f'{n_eval:>10d}' if n_eval is not None else '       n/a'
        iqr = f'[{p25*1e3:>5.1f},{p75*1e3:>5.1f}]'
        print(f'{label:<24s} {med*1e3:>9.2f} {iqr:>14s} {evs} '
              f'{med/pvre_t:>9.3f} {med/pvre_sq_t:>11.3f}')

    print(f'\nReport: huber/pseudo wall vs pvre at each δ '
          f'(< 1.0 means faster than pvre):')
    for label, med, _, _, _ in rows:
        if 'huber' in label.lower():
            print(f'  {label:<24s}: {med/pvre_t:.3f}x pvre  '
                  f'{med/pvre_sq_t:.3f}x pvre² ({pvre_t/med:.2f}x faster than pvre)')

    out_path = os.environ.get(
        'EVAL_OUT',
        '/pscratch/sd/e/ericyuan/temp/popcornn_huber_speed_lj13_n8d4.json',
    )
    payload = {
        'system': 'lj13',
        'mlp': {'n_embed': 8, 'depth': 4, 'activation': 'gelu'},
        'warmup_iters': WARMUP_ITERS,
        'warmup_lr': 1.0e-3,
        'n_repeats': N_REPEATS,
        'rtol': RTOL, 'atol': ATOL, 'method': 'gk21',
        'rows': [
            {'label': label, 'med_s': med, 'p25_s': p25, 'p75_s': p75,
             'eval_pts': n_eval}
            for label, med, p25, p75, n_eval in rows
        ],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nresults saved: {out_path}')


if __name__ == '__main__':
    main()
