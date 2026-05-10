"""LJ13 rtol test (revised): at atol=1e-3 (the cheap variant), test whether
tightening rtol from 1e-1 to 1e-2 lets thr=1e-3 fire reliably. Per the
rule, floor = max(rtol·|g|, atol) = max(1e-2·7e-3, 1e-3) = 1e-3
(atol-dominated) — but the empirical |g|_∞ floor at atol=1e-3 was ~7e-3
on the original rtol=1e-1 sweep (Adam jitter + integration noise on top
of atol bound). Goal: confirm or refute that rtol tightening alone
brings the empirical floor down enough for thr=1e-3 to fire.

3 seeds × 1000 iter (enough to see the |g|_∞ plateau settle past the
F_2<0.05 first crossing; thr=0 in run, post-process simulates thr=1e-3,
1e-4)."""
import json
import os
import time as time_mod

import numpy as np
import torch

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator


OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_lj13_n4d2_lr1em3_rtol_test'
LOG_DIR = os.path.join(OUT_DIR, 'logs')

SEEDS = [0, 1, 2]
N_EMBED = 4
DEPTH = 2
LR = 1e-3
ATOL = 1e-3   # cheap variant atol; testing if rtol=1e-2 makes thr=1e-3 fire
RTOL = 1e-2   # test value (baseline rtol=1e-1 was in popcornn_lj13_n4d2_lr1em3 logs)
MAX_ITER = 1000
QUALITY_GRID = 201
IMAGES = 'examples/configs/lj13.xyz'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def run_one(seed):
    init = {
        'images': os.path.join(REPO_ROOT, IMAGES),
        'path_params': {'name': 'mlp', 'n_embed': N_EMBED, 'depth': DEPTH,
                        'activation': 'gelu'},
        'device': 'cuda', 'seed': seed,
    }
    mep = Popcornn(**init)
    pot = get_potential(images=mep.images, name='lennard_jones',
                        epsilon=1.0, sigma=1.0, cutoff=3.0,
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(path_integrand_names='pvre',
                           rtol=RTOL, atol=ATOL,
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(
        path=mep.path, optimizer={'name': 'adam', 'lr': LR},
        threshold=0.0, patience=1, find_ts=False,
        device=mep.device, dtype=mep.dtype,
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
        if (step + 1) % 50 == 0 or step == 0:
            print(f'  iter {step+1:>5d}  wall={wall:6.1f}s  '
                  f'|g|_inf={gnorm:.3e}  F_2={f2:.3e}  '
                  f'F_inf={finf:.3e}  barrier={barrier:.4f}', flush=True)
    return log


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f'rtol={RTOL:.0e}, atol={ATOL:.0e}, n_embed={N_EMBED}, depth={DEPTH}', flush=True)
    print(f'{"seed":>5s} {"final_iter":>10s} {"final_wall":>12s} '
          f'{"final_F_2":>11s} {"final_F_inf":>13s} {"final_barrier":>14s}',
          flush=True)
    print('-' * 75, flush=True)
    for seed in SEEDS:
        tag = f'seed{seed}'
        log = run_one(seed)
        with open(os.path.join(LOG_DIR, tag + '.json'), 'w') as f:
            json.dump({'meta': {'seed': seed, 'lr': LR, 'rtol': RTOL,
                                'atol': ATOL, 'n_embed': N_EMBED,
                                'depth': DEPTH}, 'log': log}, f)
        print(f'{seed:>5d} {log[-1]["iter"]:>10d} {log[-1]["wall_s"]:>12.2f} '
              f'{log[-1]["F_TS_2"]:>11.3e} {log[-1]["F_TS_inf"]:>13.3e} '
              f'{log[-1]["barrier"]:>14.4f}', flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'\nlogs: {LOG_DIR}')


if __name__ == '__main__':
    main()
