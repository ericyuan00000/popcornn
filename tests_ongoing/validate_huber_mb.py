"""Multi-seed validation of the top Müller-Brown Huber candidates.

Two candidates from sweep_huber_mb_hp.py (1-seed exploration):
  A. quality:  lr=1e-3, (n_embed=16, depth=4) — best path quality
                                                  (f⊥_TS=0.020 @ iter 230)
  B. speed:    lr=1e-2, (n_embed=32, depth=6) — fastest to converge
                                                  (f⊥_TS=0.050 @ iter 110)

Plus the shipped two-stage baseline for reference. Three seeds each
to confirm 1-seed exploration is seed-robust before locking in.
threshold=0 so each run shows its own convergence trajectory; we
read the recommended threshold off the multi-seed |g|@best.

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python tests_ongoing/validate_huber_mb.py"

Output: /pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_validate/...
"""
import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = '/pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_validate'
RUNNER = os.path.join(REPO_ROOT, 'tests_ongoing/run_lj13_traced.py')
HUBER_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown_huber.yaml')
BASELINE_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown.yaml')

CANDIDATES = [
    {'tag': 'A_quality', 'lr': 1.0e-3, 'n_embed': 16, 'depth': 4, 'n_iter': 400},
    {'tag': 'B_speed',   'lr': 1.0e-2, 'n_embed': 32, 'depth': 6, 'n_iter': 200},
]
SEEDS = [0, 1, 2]


def write_huber_cfg(base, c, dst):
    cfg = copy.deepcopy(base)
    cfg['initialization_params']['path_params']['n_embed'] = c['n_embed']
    cfg['initialization_params']['path_params']['depth'] = c['depth']
    cfg['initialization_params']['device'] = 'cuda'
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['path_integrand_kwargs'] = {
        'pvre_huber': {'delta': 1.0},
    }
    leg['optimizer_params']['optimizer']['lr'] = c['lr']
    leg['optimizer_params']['threshold'] = 0.0
    leg['num_optimizer_iterations'] = c['n_iter']
    with open(dst, 'w') as f:
        yaml.dump(cfg, f)


def run_one(cfg_path, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, RUNNER,
           '--config', cfg_path,
           '--out', out_dir,
           '--seed', str(seed),
           '--monitor-every', '5']
    print(f'\n>>> {os.path.basename(out_dir)}', flush=True)
    rc = subprocess.run(cmd).returncode
    return rc == 0


def analyze(out_dir):
    tr = json.load(open(os.path.join(out_dir, 'trace.json')))
    qiter, fperp, ginf, bar = [], [], [], []
    offset = 0
    for s in tr['stages']:
        for it, f, b in zip(s['q_iter'], s['fperp_inf_ts'], s['barrier']):
            qiter.append(offset + it); fperp.append(f); bar.append(b)
            ginf.append(s['ginf'][it])
        offset += s['n_iter']
    qiter = np.array(qiter); fperp = np.array(fperp); ginf = np.array(ginf); bar = np.array(bar)
    bidx = int(np.argmin(fperp))
    return {
        'best_fperp': float(fperp[bidx]),
        'best_iter': int(qiter[bidx]),
        'best_ginf': float(ginf[bidx]),
        'best_barrier': float(bar[bidx]),
        'final_fperp': float(fperp[-1]),
        'final_barrier': float(bar[-1]),
        'wall_s': float(sum(s['elapsed_s'] for s in tr['stages'])),
    }


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    results = []

    # Baseline (two-stage).
    for seed in SEEDS:
        out_dir = os.path.join(OUT_BASE, f'baseline_s{seed}')
        if run_one(BASELINE_CFG, seed, out_dir):
            r = analyze(out_dir)
            r.update({'config': 'baseline_2stage', 'seed': seed})
            results.append(r)
            with open(os.path.join(OUT_BASE, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)

    # Huber candidates.
    huber_base = yaml.safe_load(open(HUBER_CFG))
    for c in CANDIDATES:
        for seed in SEEDS:
            tag = f'{c["tag"]}_s{seed}'
            out_dir = os.path.join(OUT_BASE, tag)
            cfg_path = os.path.join(out_dir, 'config.yaml')
            os.makedirs(out_dir, exist_ok=True)
            write_huber_cfg(huber_base, c, cfg_path)
            if run_one(cfg_path, seed, out_dir):
                r = analyze(out_dir)
                r.update({'config': c['tag'], 'seed': seed,
                          'lr': c['lr'], 'n_embed': c['n_embed'],
                          'depth': c['depth'], 'n_iter': c['n_iter']})
                results.append(r)
                with open(os.path.join(OUT_BASE, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)

    # Aggregate.
    print(f'\n=== mb_validate aggregate (3 seeds) ===', flush=True)
    print(f'{"config":<18s} {"best_fp_mean":>13s} {"best_fp_std":>12s} '
          f'{"best_iter_mean":>15s} {"|g|_mean":>10s} {"wall_mean":>10s}')
    by_cfg = {}
    for r in results:
        by_cfg.setdefault(r['config'], []).append(r)
    for cfg, rs in by_cfg.items():
        bps = np.array([r['best_fperp'] for r in rs])
        bis = np.array([r['best_iter'] for r in rs])
        bgs = np.array([r['best_ginf'] for r in rs])
        ws = np.array([r['wall_s'] for r in rs])
        print(f'{cfg:<18s} {bps.mean():>13.4e} {bps.std():>12.4e} '
              f'{bis.mean():>15.1f} {bgs.mean():>10.4e} {ws.mean():>10.1f}')


if __name__ == '__main__':
    main()
