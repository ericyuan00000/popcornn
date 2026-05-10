"""Per-iter quality diag for the two-stage LJ-13 config.

Drives every stage of the YAML and per-iter records:
  loss, |g|_inf, fmax@TS@g10001, fmax@TS@parab, barrier@g10001, ts_t.

fmax = max-over-atoms of the per-atom force magnitude
     = max_i ‖F_i‖_2  with  F shape (N_atom, 3)
the standard geometry-optimisation convergence metric (a.k.a. ASE's
``fmax``). NOT the element-wise L_inf over the flattened force vector.

Parabolic refinement uses three points around the g10001 argmax-energy
index to locate the saddle off-grid; we then re-evaluate forces at that
fractional t. The g10001 grid is the densest the analogous MB diag used
for the "real geometry vs sampling artifact" check.

Stops a stage on |g|_inf threshold (whatever the YAML sets) just like
production runs; runs the full max_iter cap if no threshold fires.

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclude=nid001208 --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python /abs/path/Popcornn/tests_ongoing/diag_lj13_two_stage.py \\
                       --config /abs/path/Popcornn/examples/configs/lj13.yaml \\
                       --out /pscratch/sd/e/ericyuan/temp/popcornn_lj13_diag/run0"

Output: <out>/trace.json with stage-by-stage arrays + final XYZ.
"""
import argparse
import copy
import json
import math
import os
import time as time_mod

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config, output_to_atoms


GRID_DENSE = 10001


def _quality(mep, dense_grid):
    """Per-iter path-quality metrics on g10001 + parabolic refine.

    Stays outside torch.no_grad because autograd-derived potentials use
    torch.autograd.grad inside the forward and silently fail under
    no_grad (see run_lj13_traced.py for the same pattern).
    """
    po = mep.path(dense_grid, return_velocities=False,
                  return_energies=True, return_forces=True)
    e = po.energies.detach().cpu().numpy().reshape(-1)
    f = po.forces.detach().cpu().numpy()
    # ensure (n_t, n_atom, 3); LJ-13 forces arrive as (n_t, 13, 3) but
    # fall back to reshaping a flat (n_t, 39) just in case.
    if f.ndim == 2:
        f = f.reshape(f.shape[0], -1, 3)

    ts = int(e.argmax())
    n = dense_grid.numel()

    out = {
        'barrier': float(e.max() - e[0]),
        'ts_idx': ts,
        'ts_t_g': float(dense_grid[ts].item()),
        'fmax_ts_g': float(np.linalg.norm(f[ts], axis=-1).max()),
        'parab_ts_t': None,
        'parab_e_ts': None,
        'parab_fmax_ts': None,
    }

    if 0 < ts < n - 1:
        t0_, t1_, t2_ = (float(dense_grid[ts - 1]),
                         float(dense_grid[ts]),
                         float(dense_grid[ts + 1]))
        e0, e1, e2 = float(e[ts - 1]), float(e[ts]), float(e[ts + 1])
        denom = e2 - 2.0 * e1 + e0
        if denom < 0.0:  # concave-down → valid maximum
            h = t1_ - t0_
            t_star = t1_ - 0.5 * h * (e2 - e0) / denom
            t_star = max(min(t_star, t2_), t0_)
            t_eval = torch.tensor([t_star], device=dense_grid.device,
                                  dtype=dense_grid.dtype)
            po2 = mep.path(t_eval, return_velocities=False,
                           return_energies=True, return_forces=True)
            f_star = po2.forces.detach().cpu().numpy()
            if f_star.ndim == 3:
                f_star = f_star[0]
            else:
                f_star = f_star.reshape(-1, 3)

            out['parab_ts_t'] = float(t_star)
            out['parab_e_ts'] = float(po2.energies.detach().cpu().item())
            out['parab_fmax_ts'] = float(np.linalg.norm(f_star, axis=-1).max())

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
          f'{"barrier":>9s} {"fmax@g":>10s} {"fmax@par":>10s}', flush=True)

    losses, ginfs, walls = [], [], []
    barriers = []
    fmax_ts_g, fmax_ts_par = [], []
    ts_t_g, ts_t_par = [], []

    t0 = time_mod.perf_counter()
    converged_at = None
    for step in range(n_iter):
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
        fmax_ts_g.append(q['fmax_ts_g'])
        ts_t_g.append(q['ts_t_g'])
        fmax_ts_par.append(q['parab_fmax_ts'])
        ts_t_par.append(q['parab_ts_t'])

        if (step in (0, 5, 10, 25, 50, 75, 100, 150, 200, 250)
                or step % 50 == 0
                or step == n_iter - 1):
            par_f = q['parab_fmax_ts']
            print(f'{step:>5d} '
                  f'{loss if loss is None else f"{loss:11.4e}":>11s} '
                  f'{ginf:>11.4e} {q["barrier"]:>9.4f} '
                  f'{q["fmax_ts_g"]:>10.3e} '
                  f'{(par_f if par_f is not None else float("nan")):>10.3e}',
                  flush=True)

        if not math.isfinite(ginf) or (loss is not None and not math.isfinite(loss)):
            print(f'  → non-finite at step {step}; aborting stage', flush=True)
            break
        if optr.converged:
            converged_at = step
            print(f'  → converged at step {step} (|g|_inf threshold)',
                  flush=True)
            break

    elapsed = time_mod.perf_counter() - t0
    n_done = len(walls)
    print(f'stage {stage_idx} elapsed: {elapsed:.1f}s '
          f'({n_done} iters, {1000 * elapsed / max(1, n_done):.1f} ms/iter)',
          flush=True)

    # report argmin / final on parab when available, else g10001.
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

    _summ('fmax@g10001', fmax_ts_g)
    _summ('fmax@parab ', fmax_ts_par)

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
        'fmax_ts_g': fmax_ts_g,
        'fmax_ts_par': fmax_ts_par,
        'ts_t_g': ts_t_g,
        'ts_t_par': ts_t_par,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None,
                    help='Override initialization_params.seed in the YAML.')
    ap.add_argument('--grid-size', type=int, default=GRID_DENSE,
                    help='Dense uniform grid size for per-iter F_TS (default 10001).')
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
    record_grid = torch.linspace(mep.path.t_init.item(), mep.path.t_final.item(),
                                 mep.num_record_points,
                                 device=mep.device, dtype=mep.dtype)
    seed_val = init_params.get('seed', None)
    print(f'config={args.config}  seed={seed_val}  grid={args.grid_size}',
          flush=True)
    print(f'out={args.out}', flush=True)

    stages = []
    for i, leg in enumerate(cfg.get('optimization_params', [])):
        stages.append(run_stage(mep, leg, i, dense))

    # final XYZ on the record grid
    path_output = mep.path(record_grid, return_velocities=True,
                           return_energies=True, return_forces=True)
    final_xyz = os.path.join(args.out, 'popcornn.xyz')
    if hasattr(mep.images, 'image_type') and issubclass(mep.images.image_type, Atoms):
        images = output_to_atoms(path_output, mep.images)
        write(final_xyz, images)
        print(f'final path: {final_xyz}', flush=True)

    out_json = os.path.join(args.out, 'trace.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': args.config,
            'seed': seed_val,
            'grid_size': args.grid_size,
            'n_record_points': mep.num_record_points,
            'stages': stages,
        }, f)
    print(f'trace: {out_json}', flush=True)


if __name__ == '__main__':
    main()
