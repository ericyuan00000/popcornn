"""Per-iter quality diag for MB single-stage configs.

Drives every stage of the YAML and per-iter records:
  loss, |g|_inf, |F|_2 @ TS @ g10001, |F|_2 @ TS @ parab, barrier@g10001, ts_t.

|F|_2 = sqrt(Σ_i F_i²) at the saddle frame — the global L2 norm of the
force vector at TS. For MB this is the magnitude of the gradient at the
saddle (the natural correctness signal on a 2D potential).

Parabolic refinement uses three points around the g10001 argmax-energy
index to locate the saddle off-grid, then re-evaluates forces there.

Stops a stage on |g|_inf threshold (whatever the YAML sets) or runs the
full max_iter cap.

Usage:
    python tests_ongoing/diag_mb_single_stage.py \\
        --config <cfg.yaml> --out <out_dir> [--seed N]

Output: <out>/trace.json
"""
import argparse
import copy
import json
import math
import os
import time as time_mod

import numpy as np
import torch

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config


GRID_DENSE = 10001


def _quality(mep, dense_grid):
    """Per-iter |F|_2 @ TS on g10001 + parabolic refine."""
    po = mep.path(dense_grid, return_velocities=False,
                  return_energies=True, return_forces=True)
    e = po.energies.detach().cpu().numpy().reshape(-1)
    f = po.forces.detach().cpu().numpy()
    if f.ndim == 3:
        f = f.reshape(f.shape[0], -1)

    ts = int(e.argmax())
    n = dense_grid.numel()

    out = {
        'barrier': float(e.max() - e[0]),
        'ts_idx': ts,
        'ts_t_g': float(dense_grid[ts].item()),
        'f2_ts_g': float(np.linalg.norm(f[ts])),
        'parab_ts_t': None,
        'parab_e_ts': None,
        'parab_f2_ts': None,
    }

    if 0 < ts < n - 1:
        t0_, t1_, t2_ = (float(dense_grid[ts - 1]),
                         float(dense_grid[ts]),
                         float(dense_grid[ts + 1]))
        e0, e1, e2 = float(e[ts - 1]), float(e[ts]), float(e[ts + 1])
        denom = e2 - 2.0 * e1 + e0
        if denom < 0.0:
            h = t1_ - t0_
            t_star = t1_ - 0.5 * h * (e2 - e0) / denom
            t_star = max(min(t_star, t2_), t0_)
            t_eval = torch.tensor([t_star], device=dense_grid.device,
                                  dtype=dense_grid.dtype)
            po2 = mep.path(t_eval, return_velocities=False,
                           return_energies=True, return_forces=True)
            f_star = po2.forces.detach().cpu().numpy().reshape(-1)
            out['parab_ts_t'] = float(t_star)
            out['parab_e_ts'] = float(po2.energies.detach().cpu().item())
            out['parab_f2_ts'] = float(np.linalg.norm(f_star))

    return out


def run_stage(mep, leg, stage_idx, dense_grid):
    leg = copy.deepcopy(leg)
    leg['integrator_params']['track_loss'] = True

    pot = get_potential(images=mep.images, **leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = PathIntegrator(**leg['integrator_params'],
                           device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)

    n_iter = leg.get('num_optimizer_iterations', 0)
    n_params = sum(p.numel() for p in mep.path.parameters())
    integrand = leg['integrator_params'].get('path_integrand_names')
    lr = leg['optimizer_params']['optimizer'].get('lr')
    thr = leg['optimizer_params'].get('threshold')
    print(f'\n=== stage {stage_idx}: integrand={integrand} lr={lr} '
          f'threshold={thr} max_iter={n_iter} D={n_params} ===', flush=True)
    print(f'{"iter":>5s} {"loss":>11s} {"|g|_inf":>11s} '
          f'{"barrier":>9s} {"|F|2@g":>10s} {"|F|2@par":>10s}', flush=True)

    losses, ginfs, walls = [], [], []
    barriers = []
    f2_g, f2_par = [], []
    ts_t_g, ts_t_par = [], []

    t0 = time_mod.perf_counter()
    converged_at = None
    diverged = False
    for step in range(n_iter):
        # Pre-step divergence guard: check previous iter's metrics before
        # entering another (potentially OOM-retry-deadlocked) optimization_step.
        if step > 0:
            prev_barrier = barriers[-1]
            prev_loss = losses[-1]
            prev_ginf = ginfs[-1]
            if not math.isfinite(prev_barrier):
                print(f'  → barrier non-finite at step {step-1}; aborting', flush=True)
                diverged = True
                break
            if prev_loss is not None and not math.isfinite(prev_loss):
                print(f'  → loss non-finite at step {step-1}; aborting', flush=True)
                diverged = True
                break
            if not math.isfinite(prev_ginf) or prev_ginf > 1e+4:
                print(f'  → |g|_inf={prev_ginf:.3e} at step {step-1}; aborting', flush=True)
                diverged = True
                break
        s0 = time_mod.perf_counter()
        out = optr.optimization_step(mep.path, integ)
        flat = out.grad_integral.detach()
        loss = float(out.loss[0].item()) if getattr(out, 'loss', None) is not None else None
        ginf = float(flat.abs().max().item())
        q = _quality(mep, dense_grid)
        s1 = time_mod.perf_counter()

        losses.append(loss)
        ginfs.append(ginf)
        walls.append(s1 - s0)
        barriers.append(q['barrier'])
        f2_g.append(q['f2_ts_g'])
        ts_t_g.append(q['ts_t_g'])
        f2_par.append(q['parab_f2_ts'])
        ts_t_par.append(q['parab_ts_t'])

        if (step in (0, 5, 10, 25, 50, 75, 100, 150, 200, 250)
                or step % 50 == 0
                or step == n_iter - 1):
            par_f = q['parab_f2_ts']
            print(f'{step:>5d} '
                  f'{loss if loss is None else f"{loss:11.4e}":>11s} '
                  f'{ginf:>11.4e} {q["barrier"]:>9.4f} '
                  f'{q["f2_ts_g"]:>10.3e} '
                  f'{(par_f if par_f is not None else float("nan")):>10.3e}',
                  flush=True)

        if not math.isfinite(ginf) or (loss is not None and not math.isfinite(loss)):
            print(f'  → non-finite at step {step}; aborting stage', flush=True)
            break
        if optr.converged:
            converged_at = step
            print(f'  → converged at step {step} (|g|_inf threshold)', flush=True)
            break

    elapsed = time_mod.perf_counter() - t0
    n_done = len(walls)
    print(f'stage {stage_idx} elapsed: {elapsed:.1f}s '
          f'({n_done} iters, {1000 * elapsed / max(1, n_done):.1f} ms/iter)',
          flush=True)

    def _summ(label, arr):
        a = [v for v in arr if v is not None]
        if not a:
            print(f'  {label}: all None', flush=True)
            return
        a = np.asarray(a)
        idx = int(a.argmin())
        print(f'  {label}: argmin@{idx}  min={a[idx]:.3e}  '
              f'final={arr[-1] if arr[-1] is not None else float("nan"):.3e}',
              flush=True)

    _summ('|F|2@g10001', f2_g)
    _summ('|F|2@parab ', f2_par)

    return {
        'stage': stage_idx,
        'integrand': integrand,
        'lr': lr,
        'threshold': thr,
        'n_params': n_params,
        'max_iter': n_iter,
        'n_iter': n_done,
        'converged_at': converged_at,
        'elapsed_s': elapsed,
        'loss': losses,
        'ginf': ginfs,
        'wall_per_step': walls,
        'barrier': barriers,
        'f2_ts_g': f2_g,
        'f2_ts_par': f2_par,
        'ts_t_g': ts_t_g,
        'ts_t_par': ts_t_par,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--grid-size', type=int, default=GRID_DENSE)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = import_run_config(args.config)

    init_params = cfg.get('initialization_params', {})
    init_params.pop('output_dir', None)
    if args.seed is not None:
        init_params['seed'] = args.seed
    mep = Popcornn(**init_params)

    dense = torch.linspace(mep.path.t_init.item(), mep.path.t_final.item(),
                           args.grid_size, device=mep.device, dtype=mep.dtype)
    seed_val = init_params.get('seed', None)
    print(f'config={args.config}  seed={seed_val}  grid={args.grid_size}',
          flush=True)
    print(f'out={args.out}', flush=True)

    stages = []
    for i, leg in enumerate(cfg.get('optimization_params', [])):
        stages.append(run_stage(mep, leg, i, dense))

    out_json = os.path.join(args.out, 'trace.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': args.config,
            'system': 'mb',
            'seed': seed_val,
            'grid_size': args.grid_size,
            'stages': stages,
        }, f)
    print(f'trace: {out_json}', flush=True)


if __name__ == '__main__':
    main()
