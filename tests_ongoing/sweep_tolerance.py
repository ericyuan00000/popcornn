"""Sweep (rtol, atol) on Müller-Brown to characterize gradient noise.

Picks the loosest tolerance for each stage by running the actual optimizer
and watching what the integrated gradient norm does. The script does not
recommend numbers — it produces the data the human reads to choose them.

What "minimal" means in practice:
  - Rough stage: loosest (rtol, atol) where ‖grad‖ still trends down
    monotonically over the first ~tens of steps. Anything looser and the
    quadrature noise dominates the descent direction.
  - Tight stage: loosest (rtol, atol) whose late-iteration ‖grad‖ floor sits
    below the convergence threshold you want to use for that stage. The
    floor is set by the integration error, not the optimizer.

Run from the Popcornn repo root:

    python tests_ongoing/sweep_tolerance.py

CPU is fine; Müller-Brown is 2D.
"""
import copy
import json
import os
import time as time_mod
from itertools import product

from popcornn import Popcornn
from popcornn.optimization.path_optimizer import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import ODEintegrator, import_run_config


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO_ROOT, 'examples', 'configs', 'muller_brown.yaml')

RTOLS = [1e-1, 1e-2, 1e-3, 1e-5]
ATOLS = [1e-1, 1e-2, 1e-3, 1e-5]
N_STEPS = 50


def run_one(rtol, atol, base_cfg, track_loss=False):
    cfg = copy.deepcopy(base_cfg)
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['rtol'] = rtol
    leg['integrator_params']['atol'] = atol

    mep = Popcornn(**cfg.get('initialization_params', {}))
    pot = get_potential(images=mep.images, **leg['potential_params'],
                        device=mep.device, dtype=mep.dtype)
    mep.path.set_potential(pot)
    integ = ODEintegrator(**leg['integrator_params'], track_loss=track_loss,
                          device=mep.device, dtype=mep.dtype)
    optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                         device=mep.device, dtype=mep.dtype)

    grad_norms, n_nodes, wall = [], [], []
    loss_integrals = [] if track_loss else None
    for _ in range(N_STEPS):
        t0 = time_mod.perf_counter()
        out = optr.optimization_step(mep.path, integ)
        wall.append(time_mod.perf_counter() - t0)
        grad_norms.append(out.loss.item())
        n_nodes.append(int(out.t.shape[0]))
        if track_loss:
            loss_integrals.append(out.loss_integral[0].item())
    return {
        'rtol': rtol, 'atol': atol,
        'grad_norms': grad_norms,
        'n_nodes': n_nodes,
        'wall_per_step': wall,
        'loss_integrals': loss_integrals,
    }


def monotonic_fraction(xs):
    if len(xs) < 2:
        return 1.0
    drops = sum(1 for a, b in zip(xs[:-1], xs[1:]) if b <= a)
    return drops / (len(xs) - 1)


def main():
    base_cfg = import_run_config(CONFIG)
    print(f'config: {CONFIG}')
    print(f'steps per cell: {N_STEPS}')
    print()
    header = f'{"rtol":>8s} {"atol":>8s} {"||g||₀":>10s} {"||g||_final":>12s} {"||g||_min":>10s} {"mono%":>7s} {"nodes_med":>10s} {"sec/step":>9s}'
    print(header)
    print('-' * len(header))

    rows = []
    for rtol, atol in product(RTOLS, ATOLS):
        try:
            r = run_one(rtol, atol, base_cfg)
        except Exception as exc:
            print(f'{rtol:>8.0e} {atol:>8.0e}  failed: {exc}')
            continue
        gn = r['grad_norms']
        nodes_med = sorted(r['n_nodes'])[len(r['n_nodes']) // 2]
        sec = sum(r['wall_per_step']) / len(r['wall_per_step'])
        print(f'{rtol:>8.0e} {atol:>8.0e} {gn[0]:>10.3e} {gn[-1]:>12.3e} '
              f'{min(gn):>10.3e} {monotonic_fraction(gn):>6.0%} '
              f'{nodes_med:>10d} {sec:>8.4f}s')
        rows.append(r)

    out_json = os.path.join(REPO_ROOT, 'tests_ongoing', 'sweep_tolerance.json')
    with open(out_json, 'w') as f:
        json.dump(rows, f)
    print(f'\nfull trajectories: {out_json}')


if __name__ == '__main__':
    main()
