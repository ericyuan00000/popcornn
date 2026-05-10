"""Compare pvre at lr=1e-3 (smooth-trajectory regime) vs lr=1e-2 winner.
Tests n4d2 (lr=1e-2 winner) and n8d4 (existing prod ref) at lr=1e-3 with
my rtol=1, atol=1e-1 (apples-to-apples integrator cost) across 3 seeds.

Expected: smoother |g|_inf trajectory, monotone descent → patience=1 +
round-decade threshold can fire reliably at F_2<1.

Wall budget: ~10x more iters than lr=1e-2 (max_iter=2000 cap), ~6s per
run = ~1 min total. We track full per-iter (|g|_inf, F_2, F_inf, wall)
so the threshold can be picked post-hoc from the trajectory.
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


OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/lr1em3_logs'

CAPACITIES = [(2, 2), (4, 2), (8, 4), (1, 2)]
SEEDS = [0, 1, 2]
LR = 1e-3
RTOL = 1.0
ATOL = 1e-1
MAX_ITER = 2000
QUALITY_GRID = 201
IMAGES = [[-0.558, 1.442], [0.623, 0.028]]


def quality_at_ts(mep, n_grid=QUALITY_GRID):
    t_init, t_final = mep.path.t_init.item(), mep.path.t_final.item()
    tg = torch.linspace(t_init, t_final, n_grid, device=mep.device, dtype=mep.dtype)
    po = mep.path(tg, return_velocities=False, return_energies=True, return_forces=True)
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


def run_one(n_embed, depth, seed):
    init = {
        'images': IMAGES,
        'path_params': {'name': 'mlp', 'n_embed': n_embed, 'depth': depth,
                        'activation': 'gelu'},
        'device': 'cuda', 'seed': seed,
    }
    mep = Popcornn(**init)
    pot = get_potential(images=mep.images, name='muller_brown',
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(path_integrand_names='pvre',
                           rtol=RTOL, atol=ATOL,
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(
        path=mep.path, optimizer={'name': 'adam', 'lr': LR},
        threshold=0.0, patience=1, device=mep.device, dtype=mep.dtype,
    )
    log = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time_mod.perf_counter()
    for step in range(MAX_ITER):
        out = optr.optimization_step(mep.path, integ)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time_mod.perf_counter() - t0
        barrier, f2, finf = quality_at_ts(mep)
        gnorm = float(out.grad_norm.item()) if hasattr(out, 'grad_norm') else None
        log.append({'iter': step + 1, 'wall_s': wall, 'barrier': barrier,
                    'F_TS_2': f2, 'F_TS_inf': finf, 'grad_norm_inf': gnorm})
    return log


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'{"cap":<6s} {"seed":>4s} {"hit_it":>7s} {"hit_wall":>9s} '
          f'{"finF2":>9s} {"tail_min":>9s} {"tail_max":>9s}', flush=True)
    print('-' * 75, flush=True)
    rows = []
    for n_embed, depth in CAPACITIES:
        for seed in SEEDS:
            tag = f'n{n_embed}d{depth}__seed{seed}'
            log = run_one(n_embed, depth, seed)
            with open(os.path.join(OUT_DIR, tag + '.json'), 'w') as f:
                json.dump({'meta': {'n_embed': n_embed, 'depth': depth,
                                    'seed': seed, 'lr': LR,
                                    'rtol': RTOL, 'atol': ATOL}, 'log': log}, f)
            hit = next((e for e in log if e['F_TS_2'] < 1.0), None)
            tail = log[-100:]
            tail_f2 = [e['F_TS_2'] for e in tail]
            row = {
                'cap': f'n{n_embed}d{depth}',
                'seed': seed,
                'hit_iter': hit['iter'] if hit else None,
                'hit_wall_s': hit['wall_s'] if hit else None,
                'final_F_2': log[-1]['F_TS_2'],
                'final_wall_s': log[-1]['wall_s'],
                'tail_F_2_min': min(tail_f2),
                'tail_F_2_max': max(tail_f2),
            }
            rows.append(row)
            hi = hit['iter'] if hit else -1
            hw = hit['wall_s'] if hit else -1.0
            print(f'n{n_embed}d{depth:<3d} {seed:>4d} {hi:>7d} {hw:>9.2f} '
                  f'{log[-1]["F_TS_2"]:>9.2e} {min(tail_f2):>9.2e} {max(tail_f2):>9.2e}',
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, '../lr1em3_results.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nlogs: {OUT_DIR}')


if __name__ == '__main__':
    main()
