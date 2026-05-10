"""Sweep (rtol, atol) per loss on MB / n1d2 / lr=1e-2 / thr=0 / patience=1.

Goal: for each loss in {pvre, pseudo δ∈{1, 0.1, 0.01}}, find the loosest
(rtol, atol) pair that drives |F|_2 at parabolic-refined TS below 1.0 in
the smallest wall time.

Per-iter we evaluate (wall, |F|_2, |F|_∞, barrier, |g|_∞) on a 201-point
dense grid + parabolic refine — cheap on n1d2 — and break on |F|_2 < 1
(``patience=1`` style: first hit wins).

Threshold is set to 0.0 (per user spec); functionally equivalent to None
on this code path because grad_norm is non-negative and ``< 0`` never
fires, so patience=1 never triggers either.

Usage:
    srun -A m2834 -q interactive -C gpu --exclude=nid001208 --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python /global/u2/e/ericyuan/GitHub/Popcornn/tests_ongoing/sweep_mb_n1d2_tol.py"

Output:
    /pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/results.json
    /pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/logs/<config>.json
"""
import json
import os
import time as time_mod

import numpy as np
import torch

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator


OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol'
LOG_DIR = os.path.join(OUT_DIR, 'logs')

LOSSES = [
    ('pvre',          'pvre',                {}),
    ('pseudo_d1.0',   'pvre_pseudo_huber',   {'delta': 1.0}),
    ('pseudo_d0.1',   'pvre_pseudo_huber',   {'delta': 0.1}),
    ('pseudo_d0.01',  'pvre_pseudo_huber',   {'delta': 0.01}),
]
TOLS = [1.0, 1e-1, 1e-2, 1e-3]
SEED = 0
LR = 1e-2
N_EMBED = 1
DEPTH = 2
TARGET_F2 = 1.0
MAX_ITER = 500
WALL_CAP = 120.0
QUALITY_GRID = 201
IMAGES = [[-0.558, 1.442], [0.623, 0.028]]


def quality_at_ts(mep, n_grid=QUALITY_GRID):
    """Parabolic-refined argmax-E. Returns (barrier, |F|_2, |F|_inf)."""
    t_init, t_final = mep.path.t_init.item(), mep.path.t_final.item()
    tg = torch.linspace(t_init, t_final, n_grid,
                        device=mep.device, dtype=mep.dtype)
    po = mep.path(tg, return_velocities=False,
                  return_energies=True, return_forces=True)
    e = po.energies.detach().cpu().numpy().reshape(-1)
    f = po.forces.detach().cpu().numpy()
    if f.ndim == 3:
        f = f.reshape(f.shape[0], -1)
    barrier = float(e.max() - e[0])
    ts = int(e.argmax())
    n = tg.numel()
    if 0 < ts < n - 1:
        t0_, t1_, t2_ = float(tg[ts - 1]), float(tg[ts]), float(tg[ts + 1])
        e0, e1, e2 = float(e[ts - 1]), float(e[ts]), float(e[ts + 1])
        denom = e2 - 2.0 * e1 + e0
        if denom < 0.0:
            h = t1_ - t0_
            t_star = t1_ - 0.5 * h * (e2 - e0) / denom
            t_star = max(min(t_star, t2_), t0_)
            t_eval = torch.tensor([t_star], device=mep.device, dtype=mep.dtype)
            po2 = mep.path(t_eval, return_velocities=False,
                           return_energies=True, return_forces=True)
            f_star = po2.forces.detach().cpu().numpy().reshape(-1)
            return barrier, float(np.linalg.norm(f_star)), float(np.max(np.abs(f_star)))
    return barrier, float(np.linalg.norm(f[ts])), float(np.max(np.abs(f[ts])))


def run_one(loss_tag, integrand_name, integrand_kwargs, rtol, atol):
    init = {
        'images': IMAGES,
        'path_params': {
            'name': 'mlp',
            'n_embed': N_EMBED,
            'depth': DEPTH,
            'activation': 'gelu',
        },
        'device': 'cuda',
        'seed': SEED,
    }
    mep = Popcornn(**init)

    pot = get_potential(images=mep.images, name='muller_brown',
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)

    integ_kwargs = {
        'path_integrand_names': integrand_name,
        'rtol': float(rtol),
        'atol': float(atol),
    }
    if integrand_kwargs:
        integ_kwargs['path_integrand_kwargs'] = {integrand_name: integrand_kwargs}
    integ = PathIntegrator(**integ_kwargs, device=mep.device, dtype=mep.dtype)

    optr = PathOptimizer(
        path=mep.path,
        optimizer={'name': 'adam', 'lr': LR},
        threshold=0.0,
        patience=1,
        device=mep.device, dtype=mep.dtype,
    )

    log = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time_mod.perf_counter()
    target_iter = None
    target_wall = None
    final_iter = MAX_ITER
    for step in range(MAX_ITER):
        out = optr.optimization_step(mep.path, integ)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time_mod.perf_counter() - t0
        barrier, f2, finf = quality_at_ts(mep)
        gnorm = None
        try:
            gnorm = float(out.grad_norm.item()) if hasattr(out, 'grad_norm') else None
        except Exception:
            gnorm = None
        log.append({
            'iter': step + 1,
            'wall_s': wall,
            'barrier': barrier,
            'F_TS_2': f2,
            'F_TS_inf': finf,
            'grad_norm_inf': gnorm,
        })
        if (target_iter is None) and (f2 < TARGET_F2):
            target_iter = step + 1
            target_wall = wall
            final_iter = step + 1
            break
        if wall > WALL_CAP:
            final_iter = step + 1
            break
    else:
        final_iter = MAX_ITER

    result = {
        'loss': loss_tag,
        'rtol': float(rtol),
        'atol': float(atol),
        'reached_target': target_iter is not None,
        'target_iter': target_iter,
        'target_wall_s': target_wall,
        'final_iter': final_iter,
        'final_wall_s': log[-1]['wall_s'] if log else None,
        'final_F_TS_2': log[-1]['F_TS_2'] if log else None,
        'final_F_TS_inf': log[-1]['F_TS_inf'] if log else None,
        'final_barrier': log[-1]['barrier'] if log else None,
    }
    return result, log


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    rows = []
    print(f'{"loss":<14s} {"rtol":>8s} {"atol":>8s} '
          f'{"hit":>3s} {"itr":>5s} {"wall_s":>8s} '
          f'{"finF2":>9s} {"finFinf":>9s} {"barr":>8s}', flush=True)
    print('-' * 90, flush=True)
    for loss_tag, integrand_name, integrand_kwargs in LOSSES:
        for rtol in TOLS:
            for atol in TOLS:
                tag = f'{loss_tag}__rtol{rtol:.0e}__atol{atol:.0e}'
                try:
                    r, log = run_one(loss_tag, integrand_name, integrand_kwargs,
                                     rtol, atol)
                except Exception as e:
                    r = {
                        'loss': loss_tag, 'rtol': float(rtol), 'atol': float(atol),
                        'reached_target': False, 'error': str(e),
                    }
                    log = []
                rows.append(r)
                with open(os.path.join(LOG_DIR, tag + '.json'), 'w') as f:
                    json.dump({'result': r, 'log': log}, f)
                hit = 'Y' if r.get('reached_target') else 'N'
                itr = r.get('target_iter') or r.get('final_iter') or -1
                wall = r.get('target_wall_s') or r.get('final_wall_s') or -1.0
                fF2 = r.get('final_F_TS_2')
                fFi = r.get('final_F_TS_inf')
                barr = r.get('final_barrier')
                print(f'{loss_tag:<14s} {rtol:>8.0e} {atol:>8.0e} '
                      f'{hit:>3s} {itr:>5d} {wall:>8.2f} '
                      f'{(fF2 if fF2 is not None else float("nan")):>9.2e} '
                      f'{(fFi if fFi is not None else float("nan")):>9.2e} '
                      f'{(barr if barr is not None else float("nan")):>8.2f}',
                      flush=True)
                # free CUDA cache between runs to avoid drift
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    out_json = os.path.join(OUT_DIR, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=2)

    print('\n' + '=' * 90)
    print('Per-loss winner (loosest tols that hit |F|_2 < 1, min wall):')
    print('-' * 90)
    by_loss = {}
    for r in rows:
        if r.get('reached_target'):
            by_loss.setdefault(r['loss'], []).append(r)
    for loss_tag in [t[0] for t in LOSSES]:
        candidates = by_loss.get(loss_tag, [])
        if not candidates:
            print(f'{loss_tag:<14s}  no config reached |F|_2 < 1 within '
                  f'{MAX_ITER} iters / {WALL_CAP}s wall')
            continue
        winner = min(candidates, key=lambda r: r['target_wall_s'])
        print(f'{loss_tag:<14s}  rtol={winner["rtol"]:.0e} '
              f'atol={winner["atol"]:.0e}  '
              f'iter={winner["target_iter"]:>4d}  '
              f'wall={winner["target_wall_s"]:6.2f}s')
    print(f'\nresults: {out_json}')


if __name__ == '__main__':
    main()
