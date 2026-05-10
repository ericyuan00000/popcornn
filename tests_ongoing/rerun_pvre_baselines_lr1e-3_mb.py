"""pvre and pvre_squared baselines at lr=1e-3 on Müller-Brown.

Companion to rerun_pvre_baseline_mb.py (lr=1e-2) and
rerun_pvre_squared_baseline_mb.py (lr=1e-2). Same 18-param MLP, same
600 iters, threshold=1e-2, patience=10. Only lr changes.

Output:
  /pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_pvre_rerun_lr1e-3/
  /pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_pvre_squared_rerun_lr1e-3/
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE_ROOT = '/pscratch/sd/e/ericyuan/temp/popcornn_huber'
RUNNER = os.path.join(REPO_ROOT, 'tests_ongoing/run_lj13_traced.py')
BASELINE_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown.yaml')

INTEGRANDS = ['pvre', 'pvre_squared']
SEEDS = [0, 1, 2]
LR = 1.0e-3
N_EMBED = 2
DEPTH = 2
N_ITER = 600
THRESHOLD = 1.0e-2
PATIENCE = 10


def write_cfg(base, integrand, dst):
    cfg = copy.deepcopy(base)
    cfg['initialization_params'].setdefault('device', 'cuda')
    cfg['initialization_params']['path_params']['n_embed'] = N_EMBED
    cfg['initialization_params']['path_params']['depth'] = DEPTH
    leg0 = cfg['optimization_params'][0]
    new_leg = {
        'potential_params': leg0['potential_params'],
        'integrator_params': {
            'path_integrand_names': integrand,
            'rtol': leg0['integrator_params'].get('rtol', 1.0e-2),
        },
        'optimizer_params': {
            'optimizer': {'name': 'adam', 'lr': LR},
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
           '--monitor-every', '10']
    print(f'\n>>> {os.path.basename(out_dir)}', flush=True)
    rc = subprocess.run(cmd).returncode
    return rc == 0


def analyze(out_dir):
    tr = json.load(open(os.path.join(out_dir, 'trace.json')))
    qiter, fts, fperp, ginf, bar = [], [], [], [], []
    offset = 0
    converged_at = None
    for s in tr['stages']:
        for it, f, fp, b in zip(s['q_iter'], s['f_inf_ts'],
                                  s['fperp_inf_ts'], s['barrier']):
            qiter.append(offset + it)
            fts.append(f); fperp.append(fp); bar.append(b)
            ginf.append(s['ginf'][it])
        if s.get('converged_at') is not None and converged_at is None:
            converged_at = offset + s['converged_at']
        offset += s['n_iter']
    qiter = np.array(qiter); fts = np.array(fts); fperp = np.array(fperp)
    ginf = np.array(ginf); bar = np.array(bar)
    bidx = int(np.argmin(fts))
    return {
        'best_f_inf_ts': float(fts[bidx]),
        'best_iter': int(qiter[bidx]),
        'best_ginf': float(ginf[bidx]),
        'best_barrier': float(bar[bidx]),
        'best_fperp_inf_ts': float(fperp[bidx]),
        'final_f_inf_ts': float(fts[-1]),
        'final_fperp_inf_ts': float(fperp[-1]),
        'final_barrier': float(bar[-1]),
        'wall_s': float(sum(s['elapsed_s'] for s in tr['stages'])),
        'converged_at': converged_at,
    }


def main():
    base = yaml.safe_load(open(BASELINE_CFG))
    for integrand in INTEGRANDS:
        out_base = os.path.join(OUT_BASE_ROOT, f'mb_{integrand}_rerun_lr1e-3')
        os.makedirs(out_base, exist_ok=True)
        results = []
        for seed in SEEDS:
            tag = f'{integrand}_only_s{seed}'
            out_dir = os.path.join(out_base, tag)
            cfg_path = os.path.join(out_dir, 'config.yaml')
            os.makedirs(out_dir, exist_ok=True)
            write_cfg(base, integrand, cfg_path)
            if run_one(cfg_path, seed, out_dir):
                r = analyze(out_dir)
                r.update({'config': f'{integrand}_only', 'seed': seed,
                          'lr': LR, 'n_iter': N_ITER, 'threshold': THRESHOLD,
                          'integrand': integrand})
                results.append(r)
                with open(os.path.join(out_base, 'results.json'), 'w') as f:
                    json.dump(results, f, indent=2)

        print(f'\n=== {integrand}-only (3 seeds, single-stage, '
              f'lr={LR:.0e}, 18-param MLP, {N_ITER} iters, '
              f'threshold={THRESHOLD}) ===', flush=True)
        print(f'{"seed":>5s} {"best_F_TS":>11s} {"best_iter":>10s} '
              f'{"|g|@best":>11s} {"barrier":>9s} {"final_F_TS":>11s} '
              f'{"converged_at":>13s} {"wall_s":>8s}')
        for r in sorted(results, key=lambda x: x['seed']):
            ca = '-' if r['converged_at'] is None else str(r['converged_at'])
            print(f'{r["seed"]:>5d} {r["best_f_inf_ts"]:>11.4e} '
                  f'{r["best_iter"]:>10d} {r["best_ginf"]:>11.4e} '
                  f'{r["best_barrier"]:>9.4f} {r["final_f_inf_ts"]:>11.4e} '
                  f'{ca:>13s} {r["wall_s"]:>8.1f}')
        bts = np.array([r['best_f_inf_ts'] for r in results])
        print(f' mean {bts.mean():>11.4e}  std {bts.std():>10.4e}')


if __name__ == '__main__':
    main()
