"""Full-schedule MLP capacity sweep on LJ-13.

Unlike `plot_mlp_sweep_lj.py` (stage-1 only, tight rtol, no thresholds),
this script runs the complete shipped two-stage schedule from
`examples/configs/lj13.yaml` (pvre_squared → pvre with thresholds 1.0
and 1e-3) across a (n_embed, depth) grid. It records the production
quality metrics — final |F_perp|_TS, barrier, total iters, total wall
time — so we can choose the MLP that's actually best for the schedule
users will run, not for an artificial stage-1-only proxy.

Usage:
    python mlp_full_sweep_lj.py [--seeds N M ...]

Output: <OUT_BASE>/results.json, plus per-config subdirs with the
        per-iter trace and final XYZ.
"""
import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np
import yaml
from ase.calculators.lj import LennardJones
from ase.io import read

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CFG = os.path.join(REPO_ROOT, 'examples', 'configs', 'lj13.yaml')
OUT_BASE = '/pscratch/sd/e/ericyuan/temp/popcornn_lj13_example/mlp_full_sweep'

# Grid covers undersized → oversized region. n_params ranges 1.6k → 805k.
GRID = [(1, 2), (2, 2), (2, 4), (4, 2), (4, 4), (4, 6),
        (8, 2), (8, 4), (8, 6), (16, 4)]


def score_path(xyz):
    frames = read(xyz, index=':')
    energies, forces, positions = [], [], []
    for f in frames:
        f.calc = LennardJones()
        energies.append(f.get_potential_energy())
        forces.append(f.get_forces())
        positions.append(f.get_positions())
    e = np.array(energies)
    f = np.array(forces).reshape(len(frames), -1)
    p = np.array(positions).reshape(len(frames), -1)
    ts = int(e.argmax())
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]; t[0] = p[1] - p[0]; t[-1] = p[-1] - p[-2]
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    t_hat = t / norms
    f_perp = f - (f * t_hat).sum(axis=1, keepdims=True) * t_hat
    return {
        'barrier': float(e.max() - e[0]),
        'fperp_inf_ts': float(np.max(np.abs(f_perp[ts]))),
        'f_inf_ts': float(np.max(np.abs(f[ts]))),
    }


def run_one(n_embed, depth, seed, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg['initialization_params']['path_params']['n_embed'] = n_embed
    cfg['initialization_params']['path_params']['depth'] = depth
    cfg['initialization_params']['seed'] = seed

    tag = f'ne{n_embed}_d{depth}_s{seed}'
    out = os.path.join(OUT_BASE, tag)
    os.makedirs(out, exist_ok=True)
    tmp_yaml = os.path.join(out, 'config.yaml')
    with open(tmp_yaml, 'w') as f:
        yaml.dump(cfg, f)

    print(f'\n=== {tag} ===', flush=True)
    cmd = ['python',
           os.path.join(REPO_ROOT, 'tests_ongoing/run_lj13_traced.py'),
           '--config', tmp_yaml,
           '--out', out,
           '--monitor-every', '5']
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f'  FAILED rc={rc}'); return None

    sc = score_path(os.path.join(out, 'popcornn.xyz'))
    tr = json.load(open(os.path.join(out, 'trace.json')))
    s1 = tr['stages'][0]; s2 = tr['stages'][1]
    s1_iter = s1['converged_at'] if s1['converged_at'] is not None else s1['n_iter']
    s2_iter = s2['converged_at'] if s2['converged_at'] is not None else s2['n_iter']
    return {
        'n_embed': n_embed, 'depth': depth, 'seed': seed,
        'D': s1['n_params'],
        's1_iter': s1_iter, 's2_iter': s2_iter,
        'wall_s': sum(s['elapsed_s'] for s in tr['stages']),
        'barrier': sc['barrier'],
        'fperp_inf_ts': sc['fperp_inf_ts'],
        'f_inf_ts': sc['f_inf_ts'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--grid', type=str, default=None,
                    help='Comma-separated "ne,d;ne,d;..." overriding the default GRID.')
    args = ap.parse_args()

    grid = GRID
    if args.grid:
        grid = [tuple(int(x) for x in pair.split(','))
                for pair in args.grid.split(';')]

    base_cfg = yaml.safe_load(open(BASE_CFG))
    os.makedirs(OUT_BASE, exist_ok=True)

    results = []
    for ne, d in grid:
        for s in args.seeds:
            r = run_one(ne, d, s, base_cfg)
            if r is None:
                continue
            results.append(r)
            with open(os.path.join(OUT_BASE, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)

    print('\n=== SUMMARY (grouped by config, mean over seeds) ===')
    print(f'{"n_embed":>8} {"depth":>5} {"D":>7} {"wall(s)":>8} {"s1_it":>6} {"s2_it":>6} '
          f'{"barrier":>9} {"fperp_TS":>10}')
    by_cfg = {}
    for r in results:
        by_cfg.setdefault((r['n_embed'], r['depth']), []).append(r)
    for (ne, d), rs in sorted(by_cfg.items()):
        D = rs[0]['D']
        wall = sum(r['wall_s'] for r in rs) / len(rs)
        s1 = sum(r['s1_iter'] for r in rs) / len(rs)
        s2 = sum(r['s2_iter'] for r in rs) / len(rs)
        b = sum(r['barrier'] for r in rs) / len(rs)
        fp = sum(r['fperp_inf_ts'] for r in rs) / len(rs)
        print(f'{ne:>8} {d:>5} {D:>7} {wall:>8.1f} {s1:>6.0f} {s2:>6.0f} {b:>9.4f} {fp:>10.4e}')


if __name__ == '__main__':
    main()
