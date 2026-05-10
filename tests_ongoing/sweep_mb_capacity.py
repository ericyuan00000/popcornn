"""Find the smallest stable MLP capacity for each loss on MB / lr=1e-2.

Stable := all 3 seeds end with F_2 < 1 AND last-30 F_2 max < 2 (no rebound).
Each run is thr=0, patience=1, max_iter=350, rtol/atol from the prior winners.

Capacity grid (n_embed × depth):
    (1,2) (2,2) (4,2) (2,3) (4,3) (8,3)
plus the existing production reference (8,4) as a sanity check.

Output:
    /pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/capacity_logs/<tag>.json
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


OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_mb_n1d2_tol/capacity_logs'

# (loss_tag, integrand_name, integrand_kwargs, rtol, atol)
LOSSES = [
    ('pvre',         'pvre',                {},               1.0,  1e-1),
    ('pseudo_d1.0',  'pvre_pseudo_huber',   {'delta': 1.0},   1.0,  1.0),
    ('pseudo_d0.1',  'pvre_pseudo_huber',   {'delta': 0.1},   1.0,  1e-1),
    ('pseudo_d0.01', 'pvre_pseudo_huber',   {'delta': 0.01},  1.0,  1e-2),
]
CAPACITIES = [(1, 2), (2, 2), (4, 2), (2, 3), (4, 3), (8, 3), (8, 4)]
SEEDS = [0, 1, 2]
LR = 1e-2
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


def run_one(loss_tag, integrand_name, integrand_kwargs, rtol, atol,
            n_embed, depth, seed):
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


def summarize_run(log):
    """Return per-run summary metrics."""
    f2_traj = [e['F_TS_2'] for e in log]
    hit = next((e for e in log if e['F_TS_2'] < 1.0), None)
    tail = log[-30:]
    tail_f2 = [e['F_TS_2'] for e in tail]
    return {
        'hit_iter': hit['iter'] if hit else None,
        'hit_wall_s': hit['wall_s'] if hit else None,
        'final_iter': log[-1]['iter'],
        'final_F_2': log[-1]['F_TS_2'],
        'final_wall_s': log[-1]['wall_s'],
        'min_F_2': min(f2_traj),
        'tail_F_2_min': min(tail_f2),
        'tail_F_2_max': max(tail_f2),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    print(f'{"loss":<13s} {"n_emb":>5s} {"depth":>5s} {"seed":>4s} '
          f'{"hit_it":>6s} {"hit_wall":>9s} '
          f'{"finF2":>9s} {"tailF2max":>10s}', flush=True)
    print('-' * 80, flush=True)

    for loss_tag, integrand_name, integrand_kwargs, rtol, atol in LOSSES:
        for n_embed, depth in CAPACITIES:
            for seed in SEEDS:
                tag = f'{loss_tag}__n{n_embed}d{depth}__seed{seed}'
                log = run_one(loss_tag, integrand_name, integrand_kwargs,
                              rtol, atol, n_embed, depth, seed)
                summary = summarize_run(log)
                row = {'loss': loss_tag, 'n_embed': n_embed, 'depth': depth,
                       'seed': seed, 'rtol': rtol, 'atol': atol, **summary}
                rows.append(row)
                with open(os.path.join(OUT_DIR, tag + '.json'), 'w') as f:
                    json.dump({'meta': row, 'log': log}, f)
                hit_iter = summary['hit_iter'] or -1
                hit_wall = summary['hit_wall_s'] or -1.0
                print(f'{loss_tag:<13s} {n_embed:>5d} {depth:>5d} {seed:>4d} '
                      f'{hit_iter:>6d} {hit_wall:>9.2f} '
                      f'{summary["final_F_2"]:>9.2e} '
                      f'{summary["tail_F_2_max"]:>10.2e}',
                      flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    out_json = os.path.join(OUT_DIR, '../capacity_results.json')
    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nresults: {out_json}')

    # Per (loss, capacity), check stability across seeds
    print('\n' + '=' * 90)
    print('Stability summary  (Stable = all 3 seeds final F_2<1 AND tail F_2 max<2):')
    print('-' * 90)
    print(f'{"loss":<13s} {"cap":>8s} {"seeds_hit":>10s} {"finF2 max":>10s} '
          f'{"tail max":>10s}  stable?  hit-wall mean')
    print('-' * 90)

    by_lc = {}
    for r in rows:
        key = (r['loss'], (r['n_embed'], r['depth']))
        by_lc.setdefault(key, []).append(r)

    stable_picks = {}
    for (loss, cap), runs in sorted(by_lc.items()):
        hits = sum(1 for r in runs if r['hit_iter'] is not None)
        finals = [r['final_F_2'] for r in runs]
        tail_max = [r['tail_F_2_max'] for r in runs]
        all_final_below = all(f < 1.0 for f in finals)
        all_tail_below = all(t < 2.0 for t in tail_max)
        stable = all_final_below and all_tail_below
        hits_str = f'{hits}/3'
        cap_str = f'n{cap[0]}d{cap[1]}'
        if stable:
            walls = [r['hit_wall_s'] for r in runs if r['hit_wall_s'] is not None]
            hw_mean = float(np.mean(walls)) if walls else float('nan')
            stable_picks.setdefault(loss, []).append(((cap, hw_mean, runs)))
        else:
            hw_mean = float('nan')
        print(f'{loss:<13s} {cap_str:>8s} {hits_str:>10s} '
              f'{max(finals):>10.2e} {max(tail_max):>10.2e}  '
              f'{("Y" if stable else "N"):^7s}  '
              f'{hw_mean:.2f}' if not np.isnan(hw_mean) else
              f'{loss:<13s} {cap_str:>8s} {hits_str:>10s} '
              f'{max(finals):>10.2e} {max(tail_max):>10.2e}  '
              f'{("Y" if stable else "N"):^7s}    -')

    print('\n' + '-' * 90)
    print('Smallest stable capacity per loss (by n_embed*depth, then n_embed):')
    for loss in [t[0] for t in LOSSES]:
        cands = stable_picks.get(loss, [])
        if not cands:
            print(f'  {loss}: NO STABLE CAPACITY in tested grid')
            continue
        cands.sort(key=lambda c: (c[0][0] * c[0][1], c[0][0]))
        cap, hw_mean, _ = cands[0]
        print(f'  {loss}: n{cap[0]}d{cap[1]}  (mean wall to F_2<1 = {hw_mean:.2f}s)')


if __name__ == '__main__':
    main()
