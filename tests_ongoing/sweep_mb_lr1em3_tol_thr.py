"""Joint (rtol, atol) sweep at pvre + n4d2 + lr=1e-3, with post-hoc thr
simulation. Goal: find the loosest (rtol, atol, round-decade thr) where
patience=1 reliably fires at F_2<1 across 3 seeds, with smallest wall.

Strategy: per "noise floor ≈ max(rtol·|g_typical|, atol)" rule, both rtol
and atol must be ≲ thr/10 for the trigger to fire. Tested combos cover
that frontier.

Output:
    /pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/lr1em3_tol_thr_logs/
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


OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/lr1em3_tol_thr_logs'

# (rtol, atol) — covers the integrator-noise rule frontier
TOL_PAIRS = [
    (1e-1, 1e-1),
    (1e-1, 1e-2),
    (1e-2, 1e-1),
    (1e-2, 1e-2),
    (1e-2, 1e-3),
    (1e-3, 1e-3),
]
SEEDS = [0, 1, 2]
N_EMBED = 4
DEPTH = 2
LR = 1e-3
MAX_ITER = 1500
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


def run_one(rtol, atol, seed):
    init = {
        'images': IMAGES,
        'path_params': {'name': 'mlp', 'n_embed': N_EMBED, 'depth': DEPTH,
                        'activation': 'gelu'},
        'device': 'cuda', 'seed': seed,
    }
    mep = Popcornn(**init)
    pot = get_potential(images=mep.images, name='muller_brown',
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(path_integrand_names='pvre',
                           rtol=float(rtol), atol=float(atol),
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
    print(f'{"rtol":>8s} {"atol":>8s} {"seed":>5s} {"final_F2":>10s} '
          f'{"final_wall":>12s}', flush=True)
    print('-' * 60, flush=True)
    for rtol, atol in TOL_PAIRS:
        for seed in SEEDS:
            tag = f'rtol{rtol:.0e}__atol{atol:.0e}__seed{seed}'
            log = run_one(rtol, atol, seed)
            with open(os.path.join(OUT_DIR, tag + '.json'), 'w') as f:
                json.dump({'meta': {'rtol': rtol, 'atol': atol, 'seed': seed,
                                    'lr': LR, 'n_embed': N_EMBED,
                                    'depth': DEPTH}, 'log': log}, f)
            print(f'{rtol:>8.0e} {atol:>8.0e} {seed:>5d} '
                  f'{log[-1]["F_TS_2"]:>10.2e} {log[-1]["wall_s"]:>12.2f}',
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f'\nlogs: {OUT_DIR}')


if __name__ == '__main__':
    main()
