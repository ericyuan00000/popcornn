"""Instrumented Popcornn driver — bypasses broken per-iter JSON logger.

The on-main `popcornn._optimize` writes per-iteration records that
require `integral_output.t`, but the integrator currently calls
torchpathint without `full_output=True` so `t=None` and the logger
crashes. Until the upstream one-line fix lands, this script reproduces
the equivalent traces from outside the library — same pattern as
`tests_ongoing/plot_mlp_sweep.py:42-79`.

Per-iter, per-stage we record:
  loss          ∫ L dt    (track_loss=True)
  g2            ‖∫∇L dt‖_2
  ginf          ‖∫∇L dt‖_∞   (== popcornn's `integral_output.grad_norm`)

Every `--monitor-every` iters we additionally evaluate the path on the
record grid and record path-intrinsic quality metrics:
  barrier       max(E) - E[0] over the path
  ts_idx        argmax(E) — frame index of the saddle
  f_inf_ts      ‖F‖_∞ at the saddle frame
  fperp_inf_ts  ‖F − (F·t̂)t̂‖_∞ at the saddle (the MEP-quality metric)

These let us see when path quality settles, independent of the |g|_∞
convergence trigger.

Usage:
  python run_lj13_traced.py --config <cfg.yaml> --out <out_dir>
                            [--seed N] [--monitor-every K]

Output: <out_dir>/trace.json with stage-by-stage arrays + final XYZ.
"""
import argparse
import copy
import json
import os
import time as time_mod

import numpy as np
from ase import Atoms
from ase.io import write

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config, output_to_atoms
import torch


def _quality(mep, time_grid):
    """Path-intrinsic quality metrics for the current path."""
    with torch.no_grad():
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
        'ts_idx': ts,
        'f_inf_ts': float(np.max(np.abs(f[ts]))),
        'fperp_inf_ts': float(np.max(np.abs(f_perp[ts]))),
    }


def run_stage(mep, leg, stage_idx, time_grid, monitor_every=1):
    """Drive a single optimization leg, return per-iter trace + final state."""
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
    print(f'\n=== stage {stage_idx}: integrand={integrand} lr={lr} '
          f'iters={n_iter} D={n_params} monitor_every={monitor_every} ===')
    print(f'{"iter":>6s} {"loss":>11s} {"|g|_inf":>11s} {"barrier":>9s} {"f_inf_TS":>10s} {"fperp_TS":>10s}')

    losses, g2s, ginfs = [], [], []
    q_iters, barriers, f_inf_ts, fperp_inf_ts = [], [], [], []
    t0 = time_mod.perf_counter()
    converged_at = None
    for step in range(n_iter):
        out = optr.optimization_step(mep.path, integ)
        flat = out.grad_integral.detach()
        loss = float(out.loss[0].item()) if getattr(out, 'loss', None) is not None else None
        g2 = float(flat.norm().item())
        ginf = float(flat.abs().max().item())
        losses.append(loss)
        g2s.append(g2)
        ginfs.append(ginf)
        sample_q = (step % monitor_every == 0) or (step == n_iter - 1)
        if sample_q:
            q = _quality(mep, time_grid)
            q_iters.append(step)
            barriers.append(q['barrier'])
            f_inf_ts.append(q['f_inf_ts'])
            fperp_inf_ts.append(q['fperp_inf_ts'])
        if step in (0, 5, 10, 25, 50, 75, 100, 150, 200, 250, n_iter - 1) or step % 50 == 0:
            if sample_q:
                print(f'{step:>6d} {loss if loss is None else f"{loss:11.4e}":>11s} {ginf:>11.4e} {q["barrier"]:>9.4f} {q["f_inf_ts"]:>10.4e} {q["fperp_inf_ts"]:>10.4e}')
            else:
                print(f'{step:>6d} {loss if loss is None else f"{loss:11.4e}":>11s} {ginf:>11.4e}')
        if optr.converged:
            converged_at = step
            print(f'  → converged at step {step} (threshold trigger) — exiting stage')
            break
    elapsed = time_mod.perf_counter() - t0
    print(f'stage {stage_idx} elapsed: {elapsed:.1f}s ({elapsed / max(1, n_iter) * 1000:.1f} ms/step)')

    return {
        'stage': stage_idx,
        'integrand': integrand,
        'lr': lr,
        'n_params': n_params,
        'n_iter': n_iter,
        'converged_at': converged_at,
        'loss': losses,
        'g2': g2s,
        'ginf': ginfs,
        'q_iter': q_iters,
        'barrier': barriers,
        'f_inf_ts': f_inf_ts,
        'fperp_inf_ts': fperp_inf_ts,
        'elapsed_s': elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None,
                    help='Override initialization_params.seed in the YAML.')
    ap.add_argument('--monitor-every', type=int, default=1,
                    help='Sample path-quality metrics every K iters (default 1).')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = import_run_config(args.config)

    init_params = cfg.get('initialization_params', {})
    init_params.pop('output_dir', None)   # bypass broken built-in logger
    if args.seed is not None:
        init_params['seed'] = args.seed
    mep = Popcornn(**init_params)

    time_grid = torch.linspace(mep.path.t_init.item(), mep.path.t_final.item(),
                               mep.num_record_points, device=mep.device, dtype=mep.dtype)

    stages = []
    for i, leg in enumerate(cfg.get('optimization_params', [])):
        stages.append(run_stage(mep, leg, i, time_grid, args.monitor_every))

    # Save final path
    path_output = mep.path(time_grid, return_velocities=True,
                           return_energies=True, return_forces=True)
    final_xyz = os.path.join(args.out, 'popcornn.xyz')
    if hasattr(mep.images, 'image_type') and issubclass(mep.images.image_type, Atoms):
        images = output_to_atoms(path_output, mep.images)
        write(final_xyz, images)
        print(f'final path: {final_xyz}')
    final_energies = path_output.energies.detach().cpu().reshape(-1).tolist()
    final_forces = path_output.forces.detach().cpu().tolist()

    out_json = os.path.join(args.out, 'trace.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': args.config,
            'n_record_points': mep.num_record_points,
            'final_energies': final_energies,
            'final_forces': final_forces,
            'stages': stages,
        }, f)
    print(f'trace: {out_json}')


if __name__ == '__main__':
    main()
