"""Find the convergence threshold that lands |F|_TS ≤ 1 on Müller-Brown.

For pvre and pvre_pseudo_huber at δ ∈ {1e-1, 1, 1e+1, 1e+2}, run max_iter=1000
with the |g|_∞ early-stop trigger DISABLED, recording per-iter |g|_∞, |F|_TS,
barrier, and wall time. Post-hoc, for each candidate power-of-10 threshold T,
simulate a patience=PATIENCE early-stop and check whether the exit iter has
|F|_TS ≤ 1; report the loosest T that satisfies this and the wall time to that
exit point.

Single-seed pilot (seed=0). Multi-seed validation deferred.

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python tests_ongoing/sweep_threshold_ts_target_mb.py"

Output:
    /pscratch/sd/e/ericyuan/temp/popcornn_thr_pilot/{tag}/trace.json
    /pscratch/sd/e/ericyuan/temp/popcornn_thr_pilot/summary.json
"""
import copy
import json
import math
import os
import time as time_mod

import numpy as np
import torch
import yaml

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown_huber.yaml')
OUT_BASE = os.environ.get('THR_PILOT_OUT', '/pscratch/sd/e/ericyuan/temp/popcornn_thr_pilot')

CONFIGS = [
    {'tag': 'pvre',         'integrand': 'pvre',              'kwargs': None},
    {'tag': 'pseudo_d1e-1', 'integrand': 'pvre_pseudo_huber', 'kwargs': {'delta': 1.0e-1}},
    {'tag': 'pseudo_d1e+0', 'integrand': 'pvre_pseudo_huber', 'kwargs': {'delta': 1.0e+0}},
    {'tag': 'pseudo_d1e+1', 'integrand': 'pvre_pseudo_huber', 'kwargs': {'delta': 1.0e+1}},
    {'tag': 'pseudo_d1e+2', 'integrand': 'pvre_pseudo_huber', 'kwargs': {'delta': 1.0e+2}},
]
LR = float(os.environ.get('THR_PILOT_LR', '1.0e-2'))
N_EMBED = 2
DEPTH = 2
SEEDS = [int(s) for s in os.environ.get('THR_PILOT_SEEDS', '0').split(',')]
MAX_ITER = 1000
F_TS_TARGET = 1.0
PATIENCE = 10
# Power-of-10 candidates from 10^-6 .. 10^+5 — sweep is post-hoc on a single trace.
THR_CANDIDATES = [10.0 ** k for k in range(-6, 6)]


def quality(mep, time_grid):
    po = mep.path(time_grid, return_velocities=False,
                  return_energies=True, return_forces=True)
    e = po.energies.detach().cpu().numpy().reshape(-1)
    f = po.forces.detach().cpu().numpy()
    pos = po.positions.detach().cpu().numpy()
    if pos.ndim == 3:
        pos = pos.reshape(pos.shape[0], -1)
    if f.ndim == 3:
        f = f.reshape(f.shape[0], -1)
    ts = int(e.argmax())
    t = np.zeros_like(pos)
    t[1:-1] = pos[2:] - pos[:-2]
    t[0] = pos[1] - pos[0]
    t[-1] = pos[-1] - pos[-2]
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    t_hat = t / norms
    f_along = (f * t_hat).sum(axis=1, keepdims=True) * t_hat
    f_perp = f - f_along
    return {
        'barrier': float(e.max() - e[0]),
        'f_inf_ts': float(np.max(np.abs(f[ts]))),
        'fperp_inf_ts': float(np.max(np.abs(f_perp[ts]))),
    }


def build_cfg(base, c, seed):
    cfg = copy.deepcopy(base)
    cfg['initialization_params']['path_params']['n_embed'] = N_EMBED
    cfg['initialization_params']['path_params']['depth'] = DEPTH
    cfg['initialization_params']['device'] = 'cuda'
    cfg['initialization_params']['seed'] = seed
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['path_integrand_names'] = c['integrand']
    if c['kwargs'] is not None:
        leg['integrator_params']['path_integrand_kwargs'] = {c['integrand']: c['kwargs']}
    else:
        leg['integrator_params'].pop('path_integrand_kwargs', None)
    leg['integrator_params']['rtol'] = 1.0e-2
    leg['integrator_params']['track_loss'] = True
    leg['optimizer_params']['optimizer'] = {'name': 'adam', 'lr': LR}
    leg['optimizer_params']['threshold'] = 0.0   # disable early stop
    leg['optimizer_params']['patience'] = PATIENCE
    leg['num_optimizer_iterations'] = MAX_ITER
    return cfg


def run_one(c, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = yaml.safe_load(open(BASE_CFG))
    cfg = build_cfg(base, c, seed)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    init_params = cfg['initialization_params']
    init_params.pop('output_dir', None)
    mep = Popcornn(**init_params)
    time_grid = torch.linspace(mep.path.t_init.item(), mep.path.t_final.item(),
                               mep.num_record_points, device=mep.device, dtype=mep.dtype)
    leg = cfg['optimization_params'][0]
    pot = get_potential(images=mep.images, **leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(**leg['integrator_params'],
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)

    n_iter = leg['num_optimizer_iterations']
    print(f'\n=== {c["tag"]} seed={seed}: integrand={c["integrand"]} kwargs={c["kwargs"]} '
          f'lr={LR} iters={n_iter} ===', flush=True)
    print(f'{"iter":>6s} {"loss":>11s} {"|g|_inf":>11s} {"barrier":>9s} {"|F|_TS":>10s}', flush=True)

    losses, ginfs = [], []
    barriers, f_ts, fperp_ts = [], [], []
    walls = []
    t0 = time_mod.perf_counter()
    for step in range(n_iter):
        s0 = time_mod.perf_counter()
        out = optr.optimization_step(mep.path, integ)
        flat = out.grad_integral.detach()
        loss = float(out.loss[0].item()) if getattr(out, 'loss', None) is not None else None
        ginf = float(flat.abs().max().item())
        q = quality(mep, time_grid)
        s1 = time_mod.perf_counter()
        losses.append(loss); ginfs.append(ginf)
        barriers.append(q['barrier']); f_ts.append(q['f_inf_ts']); fperp_ts.append(q['fperp_inf_ts'])
        walls.append(s1 - s0)
        if step in (0, 5, 25) or (step + 1) % 100 == 0:
            print(f'{step:>6d} {loss if loss is None else f"{loss:11.4e}":>11s} '
                  f'{ginf:>11.4e} {q["barrier"]:>9.4f} {q["f_inf_ts"]:>10.4e}', flush=True)
        if not math.isfinite(ginf) or (loss is not None and not math.isfinite(loss)):
            print(f'  → non-finite at step {step}; aborting', flush=True)
            break
    elapsed = time_mod.perf_counter() - t0
    print(f'  total elapsed: {elapsed:.1f}s ({len(walls)} iters)', flush=True)

    trace = {
        'tag': c['tag'], 'integrand': c['integrand'], 'kwargs': c['kwargs'],
        'lr': LR, 'n_embed': N_EMBED, 'depth': DEPTH, 'seed': seed,
        'n_iter': len(walls), 'elapsed_s': elapsed,
        'loss': losses, 'ginf': ginfs,
        'barrier': barriers, 'f_inf_ts': f_ts, 'fperp_inf_ts': fperp_ts,
        'wall_per_step': walls,
    }
    with open(os.path.join(out_dir, 'trace.json'), 'w') as f:
        json.dump(trace, f)
    return trace


def find_threshold(trace, target=F_TS_TARGET, patience=PATIENCE):
    """Loosest power-of-10 threshold T such that a `patience` consecutive run
    of |g|_∞ < T exits at an iter with |F|_TS ≤ target.

    Returns dict with chosen threshold, exit iter, |F|_TS at exit, |g|_∞ at
    exit, cumulative wall time to exit (sum of wall_per_step[:exit+1]).
    For each candidate T (descending), simulate the patience counter and pick
    the largest T whose exit iter has |F|_TS ≤ target.
    """
    ginf = trace['ginf']; fts = trace['f_inf_ts']; walls = trace['wall_per_step']
    n = len(ginf)
    results_per_T = []
    for T in sorted(THR_CANDIDATES, reverse=True):
        cnt, exit_iter = 0, None
        for k in range(n):
            if ginf[k] < T:
                cnt += 1
                if cnt >= patience:
                    exit_iter = k
                    break
            else:
                cnt = 0
        if exit_iter is None:
            results_per_T.append({'T': T, 'exit_iter': None, 'reason': 'never_triggered'})
            continue
        f_at_exit = fts[exit_iter]
        wall_to_exit = float(sum(walls[:exit_iter + 1]))
        results_per_T.append({
            'T': T, 'exit_iter': exit_iter,
            'f_inf_ts_at_exit': f_at_exit,
            'ginf_at_exit': ginf[exit_iter],
            'wall_s_to_exit': wall_to_exit,
            'meets_target': f_at_exit <= target,
        })
    chosen = None
    for r in results_per_T:
        if r.get('meets_target'):
            chosen = r
            break
    return {'chosen': chosen, 'per_T': results_per_T,
            'best_f_ts': float(min(fts)) if fts else None,
            'best_iter': int(np.argmin(fts)) if fts else None}


def aggregate(rows):
    """Per-config aggregate across seeds.

    rows: list of per-seed entries with `chosen` (dict|None) and `best_f_ts_in_run`.

    Returns dict with:
      n_meet     : how many seeds got a power-of-10 T meeting target
      Ts         : per-seed chosen T (None if not met)
      exit_iters : per-seed exit iter
      f_at_exit  : per-seed |F|_TS at exit
      wall_s     : per-seed wall to exit
      best_f_ts  : per-seed best |F|_TS in run
    """
    Ts = [r['chosen']['T'] if r['chosen'] else None for r in rows]
    exit_iters = [r['chosen']['exit_iter'] if r['chosen'] else None for r in rows]
    f_at_exit = [r['chosen']['f_inf_ts_at_exit'] if r['chosen'] else None for r in rows]
    walls = [r['chosen']['wall_s_to_exit'] if r['chosen'] else None for r in rows]
    bests = [r['best_f_ts_in_run'] for r in rows]
    n_meet = sum(1 for c in rows if c['chosen'] is not None)
    return {
        'n_meet': n_meet, 'n_seeds': len(rows),
        'Ts': Ts, 'exit_iters': exit_iters, 'f_at_exit': f_at_exit,
        'wall_s': walls, 'best_f_ts': bests,
    }


def _stats(xs):
    """mean, std for non-None floats; return None for both if empty."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.std(xs))


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f'lr={LR}  seeds={SEEDS}  max_iter={MAX_ITER}  patience={PATIENCE}  '
          f'target |F|_TS ≤ {F_TS_TARGET}', flush=True)
    summary = []  # flat list, one row per (tag, seed)
    by_tag = {c['tag']: [] for c in CONFIGS}
    for c in CONFIGS:
        for seed in SEEDS:
            tag_seed = f'{c["tag"]}_s{seed}'
            out_dir = os.path.join(OUT_BASE, tag_seed)
            trace = run_one(c, seed, out_dir)
            thr = find_threshold(trace)
            row = {'tag': c['tag'], 'seed': seed,
                   'integrand': c['integrand'], 'kwargs': c['kwargs'],
                   'elapsed_s': trace['elapsed_s'],
                   'best_f_ts_in_run': thr['best_f_ts'],
                   'best_iter_in_run': thr['best_iter'],
                   'chosen': thr['chosen'], 'per_T': thr['per_T']}
            summary.append(row)
            by_tag[c['tag']].append(row)
            with open(os.path.join(OUT_BASE, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

    aggregates = {tag: aggregate(rows) for tag, rows in by_tag.items()}
    with open(os.path.join(OUT_BASE, 'aggregate.json'), 'w') as f:
        json.dump(aggregates, f, indent=2)

    print('\n' + '=' * 110)
    print(f'Multi-seed aggregate (lr={LR}, {len(SEEDS)} seeds, max_iter={MAX_ITER}, '
          f'patience={PATIENCE}, target |F|_TS ≤ {F_TS_TARGET})')
    print('=' * 110)
    hdr = (f'{"tag":<14s} {"meet/n":>7s} {"chosen_T":>20s} '
           f'{"exit_iter mean±std":>20s} {"|F|@exit mean±std":>22s} '
           f'{"wall_s mean±std":>18s} {"best_F_TS mean±std":>22s}')
    print(hdr)
    print('-' * len(hdr))
    for c in CONFIGS:
        tag = c['tag']; agg = aggregates[tag]
        Ts_str = ','.join('--' if t is None else f'{t:.0e}' for t in agg['Ts'])
        ei_m, ei_s = _stats(agg['exit_iters'])
        fe_m, fe_s = _stats(agg['f_at_exit'])
        w_m, w_s = _stats(agg['wall_s'])
        b_m, b_s = _stats(agg['best_f_ts'])
        ei_str = '--' if ei_m is None else f'{ei_m:7.1f} ± {ei_s:6.1f}'
        fe_str = '--' if fe_m is None else f'{fe_m:8.2e} ± {fe_s:7.2e}'
        w_str = '--' if w_m is None else f'{w_m:6.1f} ± {w_s:5.1f}s'
        b_str = '--' if b_m is None else f'{b_m:8.2e} ± {b_s:7.2e}'
        print(f'{tag:<14s} {agg["n_meet"]:>3d}/{agg["n_seeds"]:<3d} '
              f'{Ts_str:>20s} {ei_str:>20s} {fe_str:>22s} {w_str:>18s} {b_str:>22s}')
    print(f'\nper-seed: {os.path.join(OUT_BASE, "summary.json")}')
    print(f'aggregate: {os.path.join(OUT_BASE, "aggregate.json")}')


if __name__ == '__main__':
    main()
