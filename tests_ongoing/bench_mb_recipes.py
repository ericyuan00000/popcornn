"""Benchmark MB recipes (pseudo-Huber single-stage vs pvre² → pvre two-stage)
across 3 seeds.

Each run goes to its threshold trigger (or num_optimizer_iterations cap)
and reports per-stage wall time, per-stage iters-to-trigger, and final
path quality (parab F_TS).

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclude=nid001208 --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python /global/u2/e/ericyuan/GitHub/Popcornn/tests_ongoing/bench_mb_recipes.py"

Output:
    /pscratch/sd/e/ericyuan/temp/popcornn_recipe_bench/result.json
"""
import copy
import json
import os
import time as time_mod

import numpy as np
import torch
import yaml

from popcornn import Popcornn
from popcornn.optimization import PathOptimizer
from popcornn.potentials import get_potential
from popcornn.tools import PathIntegrator


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = [
    os.path.join(REPO_ROOT, 'examples/configs/muller_brown_pseudo.yaml'),
    os.path.join(REPO_ROOT, 'examples/configs/muller_brown_pvre.yaml'),
    os.path.join(REPO_ROOT, 'examples/configs/muller_brown_two_stage.yaml'),
    os.path.join(REPO_ROOT, 'examples/configs/muller_brown.yaml'),
]
SEEDS = [0, 1, 2]
OUT_DIR = '/pscratch/sd/e/ericyuan/temp/popcornn_recipe_bench'
DENSE_GRID = 1001


def quality_parab(mep):
    """Final path-quality readout: argmax-E over a 1001-point uniform t-grid
    + parabolic refine around the dense-grid argmax. Returns (barrier,
    f_inf_ts_parab)."""
    t_init, t_final = mep.path.t_init.item(), mep.path.t_final.item()
    tg = torch.linspace(t_init, t_final, DENSE_GRID,
                        device=mep.device, dtype=mep.dtype)
    po = mep.path(tg, return_velocities=False,
                  return_energies=True, return_forces=True)
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
            return barrier, float(np.max(np.abs(f_star)))
    return barrier, float(np.max(np.abs(f[ts])))


def run_recipe(config_path, seed):
    """Run one recipe at one seed, timing each stage. Returns dict with
    per-stage wall_s + iters_to_trigger + final F_TS."""
    cfg = yaml.safe_load(open(config_path))
    init = copy.deepcopy(cfg.get('initialization_params', {}))
    init['device'] = 'cuda'
    init['seed'] = seed
    init.pop('output_dir', None)
    mep = Popcornn(**init)

    stage_walls, stage_iters, stage_converged = [], [], []
    for stage_idx, leg in enumerate(cfg.get('optimization_params', [])):
        leg = copy.deepcopy(leg)
        pot = get_potential(images=mep.images, **leg['potential_params'],
                            device=mep.device, dtype=mep.dtype)
        mep.path.set_potential(pot)
        integ = PathIntegrator(**leg['integrator_params'],
                               device=mep.device, dtype=mep.dtype)
        optr = PathOptimizer(path=mep.path, **leg['optimizer_params'],
                             device=mep.device, dtype=mep.dtype)

        n_iter = leg.get('num_optimizer_iterations', 1000)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time_mod.perf_counter()
        converged_at = None
        for step in range(n_iter):
            optr.optimization_step(mep.path, integ)
            if optr.converged:
                converged_at = step
                break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time_mod.perf_counter() - t0

        stage_walls.append(wall)
        stage_iters.append(step + 1)
        stage_converged.append(converged_at is not None)
        print(f'  stage {stage_idx}: {step + 1} iters in {wall:.2f}s '
              f'(converged={converged_at is not None})', flush=True)

    barrier, f_ts = quality_parab(mep)
    return {
        'config': os.path.basename(config_path),
        'seed': seed,
        'stage_walls_s': stage_walls,
        'stage_iters': stage_iters,
        'stage_converged': stage_converged,
        'total_wall_s': float(sum(stage_walls)),
        'total_iters': int(sum(stage_iters)),
        'F_TS_parab': f_ts,
        'barrier': barrier,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for cfg_path in CONFIGS:
        for seed in SEEDS:
            tag = os.path.basename(cfg_path).replace('.yaml', '')
            print(f'\n=== {tag} seed={seed} ===', flush=True)
            r = run_recipe(cfg_path, seed)
            print(f'  TOTAL: {r["total_wall_s"]:7.2f} s   '
                  f'iters={r["total_iters"]:>5d}   '
                  f'F_TS_parab={r["F_TS_parab"]:.3e}   '
                  f'barrier={r["barrier"]:.4f}', flush=True)
            results.append(r)

    out_json = os.path.join(OUT_DIR, 'result.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 90)
    print(f'{"recipe":<30s} {"wall (s) mean±std":>22s} {"iters mean±std":>20s} '
          f'{"F_TS mean±std":>20s}')
    print('-' * 90)
    by_cfg = {}
    for r in results:
        by_cfg.setdefault(r['config'], []).append(r)
    for cfg_name, rows in by_cfg.items():
        walls = [r['total_wall_s'] for r in rows]
        iters = [r['total_iters'] for r in rows]
        fts = [r['F_TS_parab'] for r in rows]
        wm, ws = float(np.mean(walls)), float(np.std(walls))
        im, is_ = float(np.mean(iters)), float(np.std(iters))
        fm, fs = float(np.mean(fts)), float(np.std(fts))
        tag = cfg_name.replace('.yaml', '').replace('muller_brown_', '')
        print(f'{tag:<30s} {wm:8.2f} ± {ws:6.2f}     '
              f'{im:7.0f} ± {is_:5.0f}   '
              f'{fm:8.3e} ± {fs:7.2e}')
    print(f'\nresults: {out_json}')


if __name__ == '__main__':
    main()
