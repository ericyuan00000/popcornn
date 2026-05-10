"""1-seed δ sweep for pvre_pseudo_huber on Müller-Brown.

Goal: find the δ at which pseudo-huber's best |F|_∞@TS lands at the
same order of magnitude as the pvre two-stage baseline (~1.0e+0).
At δ=1.0 pseudo is way too good (F_TS ≈ 9e-2); to degrade it, we
push δ up — the kernel becomes purely quadratic (pvre²-like) and
loses the linear large-residual tail that gave it the variance
robustness.

All other knobs match the pseudo_huber_winner config:
  lr=1e-2, n_embed=2, depth=2, n_iter=600, threshold=1e-2, patience=10

Per `feedback_one_seed_then_validate.md`: 1-seed bracket here, then
multi-seed validate the converged candidate in a follow-up.

Usage:
    srun -A m2834 -q interactive -C gpu --exclusive \\
         --ntasks=1 --gpus-per-task=1 \\
         bash -lc "module load conda && conda activate torchpathint && \\
                   python tests_ongoing/sweep_pseudo_huber_delta.py"

Output: /pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_pseudo_delta/
"""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = '/pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_pseudo_delta'
RUNNER = os.path.join(REPO_ROOT, 'tests_ongoing/run_lj13_traced.py')
HUBER_CFG = os.path.join(REPO_ROOT, 'examples/configs/muller_brown_huber.yaml')

DELTAS = [1.0e+1, 1.0e+2, 1.0e+3, 1.0e+4]
SEED = 0
LR = 1.0e-2
N_EMBED = 2
DEPTH = 2
N_ITER = 600
THRESHOLD = 1.0e-2
PATIENCE = 10


def write_cfg(base, delta, dst):
    cfg = copy.deepcopy(base)
    cfg['initialization_params']['path_params']['n_embed'] = N_EMBED
    cfg['initialization_params']['path_params']['depth'] = DEPTH
    cfg['initialization_params']['device'] = 'cuda'
    leg = cfg['optimization_params'][0]
    leg['integrator_params']['path_integrand_names'] = 'pvre_pseudo_huber'
    leg['integrator_params']['path_integrand_kwargs'] = {
        'pvre_pseudo_huber': {'delta': float(delta)},
    }
    leg['optimizer_params']['optimizer']['lr'] = LR
    leg['optimizer_params']['threshold'] = THRESHOLD
    leg['optimizer_params']['patience'] = PATIENCE
    leg['num_optimizer_iterations'] = N_ITER
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
    qiter, fts, fperp, ginf, bar = [], [], [], [], []
    offset = 0
    for s in tr['stages']:
        for it, f_ts, fp, b in zip(s['q_iter'], s['f_inf_ts'],
                                    s['fperp_inf_ts'], s['barrier']):
            qiter.append(offset + it)
            fts.append(f_ts); fperp.append(fp); bar.append(b)
            ginf.append(s['ginf'][it])
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
    }


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    results = []
    base = yaml.safe_load(open(HUBER_CFG))
    for delta in DELTAS:
        tag = f'd{delta:.0e}_s{SEED}'.replace('+0', '+').replace('-0', '-')
        out_dir = os.path.join(OUT_BASE, tag)
        cfg_path = os.path.join(out_dir, 'config.yaml')
        os.makedirs(out_dir, exist_ok=True)
        write_cfg(base, delta, cfg_path)
        if run_one(cfg_path, SEED, out_dir):
            r = analyze(out_dir)
            r.update({'delta': float(delta), 'seed': SEED})
            results.append(r)
            with open(os.path.join(OUT_BASE, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)

    print(f'\n=== mb_pseudo_delta sweep (seed=0, lr=1e-2, 18-param MLP, 600 iters) ===',
          flush=True)
    print(f'{"delta":>9s} {"best_F_TS":>11s} {"best_iter":>10s} {"|g|@best":>11s} '
          f'{"barrier":>9s} {"final_F_TS":>11s} {"wall_s":>8s}')
    for r in sorted(results, key=lambda x: x['delta']):
        print(f'{r["delta"]:>9.0e} {r["best_f_inf_ts"]:>11.4e} {r["best_iter"]:>10d} '
              f'{r["best_ginf"]:>11.4e} {r["best_barrier"]:>9.4f} '
              f'{r["final_f_inf_ts"]:>11.4e} {r["wall_s"]:>8.1f}')

    # Add the existing δ=1.0 datapoint (from mb_validate) for context.
    prior = '/pscratch/sd/e/ericyuan/temp/popcornn_huber/mb_validate/pseudo_huber_winner_s0/trace.json'
    if os.path.exists(prior):
        tr = json.load(open(prior))
        qiter, fts = [], []
        for s in tr['stages']:
            for it, f_ts in zip(s['q_iter'], s['f_inf_ts']):
                qiter.append(it); fts.append(f_ts)
        bidx = int(np.argmin(fts))
        print(f'{1.0:>9.0e} {fts[bidx]:>11.4e} {qiter[bidx]:>10d} '
              f'{"-":>11s} {"-":>9s} {fts[-1]:>11.4e} {"-":>8s}  (prior, mb_validate)')


if __name__ == '__main__':
    main()
