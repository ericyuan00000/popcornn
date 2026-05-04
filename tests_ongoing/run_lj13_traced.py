"""Instrumented Popcornn driver — bypasses broken per-iter JSON logger.

The on-main `popcornn._optimize` writes per-iteration records that
require `path_integral.t`, but the integrator currently calls
torchpathint without `full_output=True` so `t=None` and the logger
crashes. Until the upstream one-line fix lands, this script reproduces
the equivalent traces from outside the library — same pattern as
`tests_ongoing/plot_mlp_sweep.py:42-79`.

Per-iter, per-stage we record:
  loss          ∫ L dt    (track_loss=True)
  g2            ‖∫∇L dt‖_2
  ginf          ‖∫∇L dt‖_∞   (== popcornn's `path_integral.loss`)
  integral_norm same as g2 — explicit alias

Usage:
  python run_lj13_traced.py --config <cfg.yaml> --out <out_dir>

Output: <out_dir>/trace.json with stage-by-stage arrays + final XYZ.
"""
import argparse
import copy
import json
import os
import time as time_mod

from ase import Atoms
from ase.io import write

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator, import_run_config, output_to_atoms
import torch


def run_stage(mep, leg, stage_idx):
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
          f'iters={n_iter} D={n_params} ===')
    print(f'{"iter":>6s} {"loss":>11s} {"|g|_2":>11s} {"|g|_inf":>11s}')

    losses, g2s, ginfs = [], [], []
    t0 = time_mod.perf_counter()
    converged_at = None
    for step in range(n_iter):
        out = optr.optimization_step(mep.path, integ)
        flat = out.integral.detach()
        loss = float(out.loss_integral[0].item()) if hasattr(out, 'loss_integral') and out.loss_integral is not None else None
        g2 = float(flat.norm().item())
        ginf = float(flat.abs().max().item())
        losses.append(loss)
        g2s.append(g2)
        ginfs.append(ginf)
        if step in (0, 5, 10, 25, 50, 75, 100, 150, 200, 250, n_iter - 1) or step % 50 == 0:
            print(f'{step:>6d} {loss if loss is None else f"{loss:11.4e}":>11s} {g2:>11.4e} {ginf:>11.4e}')
        if optr.converged and converged_at is None:
            converged_at = step
            print(f'  → converged at step {step} (threshold trigger)')
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
        'elapsed_s': elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None,
                    help='Override initialization_params.seed in the YAML.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = import_run_config(args.config)

    init_params = cfg.get('initialization_params', {})
    init_params.pop('output_dir', None)   # bypass broken built-in logger
    if args.seed is not None:
        init_params['seed'] = args.seed
    mep = Popcornn(**init_params)

    stages = []
    for i, leg in enumerate(cfg.get('optimization_params', [])):
        stages.append(run_stage(mep, leg, i))

    # Save final path
    time_grid = torch.linspace(mep.path.t_init.item(), mep.path.t_final.item(),
                               mep.num_record_points, device=mep.device, dtype=mep.dtype)
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
