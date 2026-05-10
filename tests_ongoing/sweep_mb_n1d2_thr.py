"""Re-run the 4 winners from sweep_mb_n1d2_tol.py across seeds {0,1,2}
WITHOUT the F_2<1 short-circuit — capture full per-iter (|g|_inf, F_2)
trajectories so we can pick a threshold that auto-stops near/after the
F_2<1 milestone.

Output:
    /pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/thr_logs/
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


OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/thr_logs'

WINNERS = [
    ('pvre',         'pvre',                {},               1.0,  1e-1),
    ('pseudo_d1.0',  'pvre_pseudo_huber',   {'delta': 1.0},   1.0,  1.0),
    ('pseudo_d0.1',  'pvre_pseudo_huber',   {'delta': 0.1},   1.0,  1e-1),
    ('pseudo_d0.01', 'pvre_pseudo_huber',   {'delta': 0.01},  1.0,  1e-2),
]
SEEDS = [0, 1, 2]
LR = 1e-2
N_EMBED = 1
DEPTH = 2
MAX_ITER = 350
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


def run_one(loss_tag, integrand_name, integrand_kwargs, rtol, atol, seed):
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
    integ_kwargs = {'path_integrand_names': integrand_name,
                    'rtol': float(rtol), 'atol': float(atol)}
    if integrand_kwargs:
        integ_kwargs['path_integrand_kwargs'] = {integrand_name: integrand_kwargs}
    integ = PathIntegrator(**integ_kwargs, device=mep.device, dtype=mep.dtype)
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
    for loss_tag, integrand_name, integrand_kwargs, rtol, atol in WINNERS:
        for seed in SEEDS:
            tag = f'{loss_tag}__seed{seed}'
            print(f'\n=== {tag}  rtol={rtol:.0e} atol={atol:.0e} ===', flush=True)
            log = run_one(loss_tag, integrand_name, integrand_kwargs, rtol, atol, seed)
            with open(os.path.join(OUT_DIR, tag + '.json'), 'w') as f:
                json.dump({'meta': {'loss': loss_tag, 'rtol': rtol, 'atol': atol,
                                    'seed': seed}, 'log': log}, f)
            # Find F_2<1 milestone
            hit = next((e for e in log if e['F_TS_2'] < 1.0), None)
            tail = log[-30:]  # last 30 iters
            tail_g = [e['grad_norm_inf'] for e in tail if e['grad_norm_inf'] is not None]
            tail_f2 = [e['F_TS_2'] for e in tail]
            print(f'  hit_iter={hit["iter"] if hit else "NEVER"}  '
                  f'final F_2={log[-1]["F_TS_2"]:.3e}  '
                  f'final wall={log[-1]["wall_s"]:.2f}s', flush=True)
            if tail_g:
                print(f'  last-30 |g|_inf min={min(tail_g):.3e}  '
                      f'max={max(tail_g):.3e}  median={sorted(tail_g)[len(tail_g)//2]:.3e}',
                      flush=True)
            print(f'  last-30 F_2     min={min(tail_f2):.3e}  '
                  f'max={max(tail_f2):.3e}', flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f'\nlogs: {OUT_DIR}')


if __name__ == '__main__':
    main()
