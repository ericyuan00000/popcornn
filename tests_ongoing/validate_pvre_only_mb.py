"""Single-stage pvre baseline on Müller-Brown.

The existing baseline is two-stage: pvre_squared warm-up (lr=1e-2)
then pvre fine-tune (lr=1e-4). This script runs pvre alone — same
MLP shape (n_embed=8, depth=4), same total budget (200 iters), same
patience as the recent validation — across lr ∈ {1e-2, 1e-3, 1e-4}
(round powers of 10) on 3 seeds. Lets us see what `pvre` does from a
cold start without the smooth warm-up, as a third reference point
alongside the two-stage and the Huber.

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python tests_ongoing/validate_pvre_only_mb.py"
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = '/pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_pvre_only'
RUNNER = os.path.join(REPO_ROOT, 'tests_ongoing/run_lj13_traced.py')
BASELINE_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown.yaml')

LRS = [1.0e-2, 1.0e-3, 1.0e-4]
SEEDS = [0, 1, 2]
N_ITER = 200
THRESHOLD = 1.0
PATIENCE = 10


def write_pvre_cfg(base, lr, dst):
    """Single-stage pvre. Inherit MLP shape from the shipped baseline."""
    cfg = copy.deepcopy(base)
    cfg['initialization_params'].setdefault('device', 'cuda')
    leg0 = cfg['optimization_params'][0]
    new_leg = {
        'potential_params': leg0['potential_params'],
        'integrator_params': {
            'path_integrand_names': 'pvre',
            'rtol': leg0['integrator_params'].get('rtol', 1.0e-2),
        },
        'optimizer_params': {
            'optimizer': {'name': 'adam', 'lr': float(lr)},
            'threshold': THRESHOLD,
            'patience': PATIENCE,
        },
        'num_optimizer_iterations': N_ITER,
    }
    cfg['optimization_params'] = [new_leg]
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
    qiter, fts, fperp, ginf = [], [], [], []
    offset = 0
    converged_at = None
    for s in tr['stages']:
        for it, f, fp in zip(s['q_iter'], s['f_inf_ts'], s['fperp_inf_ts']):
            qiter.append(offset + it); fts.append(f); fperp.append(fp)
            ginf.append(s['ginf'][it])
        if s.get('converged_at') is not None and converged_at is None:
            converged_at = offset + s['converged_at']
        offset += s['n_iter']
    qiter = np.array(qiter); fts = np.array(fts); fperp = np.array(fperp); ginf = np.array(ginf)
    bidx = int(np.argmin(fts))
    wall = float(sum(s['elapsed_s'] for s in tr['stages']))
    return {
        'best_f_inf_ts': float(fts[bidx]),
        'best_iter': int(qiter[bidx]),
        'best_ginf': float(ginf[bidx]),
        'best_fperp_inf_ts': float(fperp[bidx]),
        'final_f_inf_ts': float(fts[-1]),
        'final_fperp_inf_ts': float(fperp[-1]),
        'wall_s': wall,
        'final_iter': int(qiter[-1]) + 1,
        'converged_at': converged_at,
    }


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    base = yaml.safe_load(open(BASELINE_CFG))
    results = []
    for lr in LRS:
        for seed in SEEDS:
            tag = f'pvre_only_lr{lr:.0e}_s{seed}'
            out_dir = os.path.join(OUT_BASE, tag)
            cfg_path = os.path.join(out_dir, 'config.yaml')
            os.makedirs(out_dir, exist_ok=True)
            write_pvre_cfg(base, lr, cfg_path)
            if run_one(cfg_path, seed, out_dir):
                r = analyze(out_dir)
                r.update({'lr': lr, 'seed': seed})
                results.append(r)
                with open(os.path.join(OUT_BASE, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)

    # Aggregate by lr.
    print('\n=== mb_pvre_only aggregate (3 seeds per lr) ===', flush=True)
    print(f'{"lr":>6s} {"F_TS_mean":>11s} {"F_TS_std":>10s} '
          f'{"end_F_TS_mean":>14s} {"best_iter_mean":>15s} {"wall_mean":>10s}')
    by_lr = {}
    for r in results:
        by_lr.setdefault(r['lr'], []).append(r)
    for lr, rs in sorted(by_lr.items()):
        bts = np.array([r['best_f_inf_ts'] for r in rs])
        end = np.array([r['final_f_inf_ts'] for r in rs])
        bis = np.array([r['best_iter'] for r in rs])
        ws = np.array([r['wall_s'] for r in rs])
        print(f'{lr:>6.0e} {bts.mean():>11.4e} {bts.std():>10.4e} '
              f'{end.mean():>14.4e} {bis.mean():>15.1f} {ws.mean():>10.1f}')


if __name__ == '__main__':
    main()
