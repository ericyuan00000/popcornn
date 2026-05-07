"""Hyperparameter sweep for pvre_huber on Müller-Brown.

Holds δ=1.0 (leading from sweep_huber.py) and threshold=0 (let each
combo run to plateau without spurious early-stop), and sweeps
(lr, n_embed, depth) on a small grid. After the sweep we derive a
recommended threshold and max-steps from the winning combo's |F_⊥|_TS
plateau in its trace.json.

Usage on NERSC interactive GPU:
    srun -A m2834 -q interactive -C gpu --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python tests_ongoing/sweep_huber_mb_hp.py --seeds 0"

Output: /pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_hp/{tag}/...
        + results.json with per-combo metrics.
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
OUT_BASE = '/pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_hp_small'
RUNNER = os.path.join(REPO_ROOT, 'tests_ongoing/run_lj13_traced.py')
BASE_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown_huber.yaml')

DELTA = 1.0
N_ITER = 1000

# Müller-Brown is a 1D curve in 2D — wildly overparameterized at the
# shipped (n_embed=8, depth=4) → 610 params. Sweep small MLPs near the
# 10-param floor to see if reducing capacity removes the post-best
# overshoot we saw at (16,4) and the larger grids.
#   (n_embed=1, depth=2):  1 → 2 → 2     =   10 params
#   (n_embed=2, depth=2):  1 → 4 → 2     =   18 params
#   (n_embed=4, depth=2):  1 → 8 → 2     =   42 params
#   (n_embed=1, depth=4):  1 → 2 → 2 → 2 → 2 = 22 params
#   (n_embed=2, depth=4):  1 → 4 → 4 → 4 → 2 = 70 params
LRS = [1.0e-2, 1.0e-3, 1.0e-4]
MLPS = [(1, 2), (2, 2), (4, 2), (1, 4), (2, 4)]


def write_cfg(base, lr, n_embed, depth, dst):
    cfg = copy.deepcopy(base)
    cfg['initialization_params']['path_params']['n_embed'] = n_embed
    cfg['initialization_params']['path_params']['depth'] = depth
    cfg['initialization_params']['device'] = 'cuda'  # interactive-GPU node
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['path_integrand_kwargs'] = {
        'pvre_huber': {'delta': float(DELTA)},
    }
    leg['optimizer_params']['optimizer']['lr'] = float(lr)
    leg['optimizer_params']['threshold'] = 0.0
    leg['num_optimizer_iterations'] = int(N_ITER)
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
    if rc != 0:
        print(f'  FAILED rc={rc}', flush=True)
        return None
    return analyze(json.load(open(os.path.join(out_dir, 'trace.json'))))


def analyze(trace):
    """Return path-quality stats for the run.

    Records best |F_⊥|_TS (and its iter), final value, and the iter at
    which |F_⊥|_TS first comes within 10% of its best — that's our
    "effective convergence iter" candidate from which to derive a
    recommended max-steps and threshold.
    """
    qiter, fperp, ginf, bar = [], [], [], []
    offset = 0
    for s in trace['stages']:
        # ginf is per-iter; q_iter samples are typically every 5
        for it, f, b in zip(s['q_iter'], s['fperp_inf_ts'], s['barrier']):
            qiter.append(offset + it); fperp.append(f); bar.append(b)
            ginf.append(s['ginf'][it])
        offset += s['n_iter']
    qiter = np.array(qiter); fperp = np.array(fperp); ginf = np.array(ginf); bar = np.array(bar)

    bidx = int(np.argmin(fperp))
    best_fp = float(fperp[bidx]); best_iter = int(qiter[bidx]); best_ginf = float(ginf[bidx])

    # Effective convergence: first sample where fperp ≤ 1.1 * best_fp
    threshold_band = 1.1 * best_fp
    conv_idx_arr = np.where(fperp <= threshold_band)[0]
    conv_idx = int(conv_idx_arr[0]) if len(conv_idx_arr) > 0 else len(fperp) - 1
    conv_iter = int(qiter[conv_idx]); conv_ginf = float(ginf[conv_idx])

    wall_s = sum(s['elapsed_s'] for s in trace['stages'])
    return {
        'best_fperp': best_fp,
        'best_iter': best_iter,
        'best_ginf': best_ginf,
        'best_barrier': float(bar[bidx]),
        'conv_iter': conv_iter,
        'conv_ginf': conv_ginf,
        'final_fperp': float(fperp[-1]),
        'final_ginf': float(ginf[-1]),
        'final_barrier': float(bar[-1]),
        'wall_s': float(wall_s),
        's_per_iter': float(wall_s / max(1, qiter[-1] + 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = ap.parse_args()

    base = yaml.safe_load(open(BASE_CFG))
    os.makedirs(OUT_BASE, exist_ok=True)

    results = []
    for seed in args.seeds:
        for lr in LRS:
            for ne, d in MLPS:
                tag = f'lr{lr:.0e}_ne{ne}_d{d}_s{seed}'
                out_dir = os.path.join(OUT_BASE, tag)
                cfg_path = os.path.join(out_dir, 'config.yaml')
                os.makedirs(out_dir, exist_ok=True)
                write_cfg(base, lr, ne, d, cfg_path)
                r = run_one(cfg_path, seed, out_dir)
                if r is None:
                    continue
                r.update({'seed': seed, 'lr': lr, 'n_embed': ne, 'depth': d})
                results.append(r)
                with open(os.path.join(OUT_BASE, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)

    print_summary(results)


def print_summary(results):
    print(f'\n=== mb_hp summary (δ={DELTA}, threshold=0, n_iter={N_ITER}) ===', flush=True)
    print(f'{"lr":>6s} {"ne":>3s} {"d":>2s} {"seed":>4s} {"best_fp":>10s} {"@iter":>6s} '
          f'{"|g|@best":>10s} {"conv_it":>8s} {"|g|@conv":>10s} {"end_fp":>10s} '
          f'{"barrier":>9s} {"s/iter":>7s}')
    rs = sorted(results, key=lambda r: r['best_fperp'])
    for r in rs:
        print(f'{r["lr"]:>6.0e} {r["n_embed"]:>3d} {r["depth"]:>2d} {r["seed"]:>4d} '
              f'{r["best_fperp"]:>10.4e} {r["best_iter"]:>6d} {r["best_ginf"]:>10.4e} '
              f'{r["conv_iter"]:>8d} {r["conv_ginf"]:>10.4e} {r["final_fperp"]:>10.4e} '
              f'{r["best_barrier"]:>9.4f} {r["s_per_iter"]:>7.3f}')

    # Recommendation derived from the winner.
    if rs:
        w = rs[0]
        print(f'\nRecommended Müller-Brown Huber settings (1-seed):')
        print(f'  delta = {DELTA}')
        print(f'  lr = {w["lr"]:.0e}')
        print(f'  n_embed = {w["n_embed"]}, depth = {w["depth"]}')
        print(f'  num_optimizer_iterations ≥ {w["conv_iter"]}  (path quality within 10% of best)')
        print(f'  threshold ~ {w["conv_ginf"]:.2e}  (|g|_inf at the effective convergence iter)')


if __name__ == '__main__':
    main()
